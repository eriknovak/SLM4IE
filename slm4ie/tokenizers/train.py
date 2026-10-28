"""Tokenizer training orchestration.

Plans the tokenizer x vocab-size sweep, prepares the shared inputs (one cached
training sample and, when a morphological backend is requested, the derived
morpheme lexicon), and trains the selected runs. `train_sweep` dispatches
`train_one` across runs with `run_parallel`; inputs are prepared once in the
parent so every run trains on identical text and the corpus is never re-streamed
per run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import slm4ie.tokenizers.backends  # noqa: F401  (registers backends on import)
from slm4ie.tokenizers.base import TrainContext, covering_alphabet
from slm4ie.tokenizers.config import SweepConfig
from slm4ie.tokenizers.corpus import (
    CorpusSplit,
    check_sample_side,
    corpus_fingerprint,
    count_sample_characters,
    fit_corpus_split,
    iter_sample,
    load_corpus_split,
    materialize_sample,
    save_corpus_split,
)
from slm4ie.tokenizers.morphology import (
    build_derivational_lexicon,
    build_morph_lexicon,
    load_lexicon,
    merge_lexicons,
    save_lexicon,
    split_lexemes,
    without_forms,
)
from slm4ie.tokenizers.registry import get_tokenizer
from slm4ie.utils import mlflow as ml
from slm4ie.utils.cli import stamped_log_dir
from slm4ie.utils.parallel import (
    configure_script_logging,
    cpu_default,
    resolve_workers,
    run_parallel,
)
from slm4ie.utils.versioning import corpus_digest, file_sha256

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

#: Sidecar carrying per-run training stats, read by the MLflow logger.
TRAIN_STATS_FILENAME = "train_stats.json"

#: Seconds between two reads of the process's resident memory while a run trains.
RSS_POLL_SECONDS = 1.0

#: Glob patterns identifying a trained tokenizer's defining artifacts under its
#: export dir -- the vocab/merges file (`tokenizer.json` for HuggingFace
#: backends, `spm.model` for SentencePiece) plus its `metadata.json`. Volatile
#: sidecars (timing, eval cache, MLflow link) are excluded so the digest is a
#: stable tokenizer identity across reruns.
TOKENIZER_ARTIFACT_GLOBS: tuple[str, ...] = ("tokenizer.json", "spm.model", "metadata.json")

#: Sidecar carrying the MLflow run linkage, read by the evaluation logger to
#: cross-link each eval run back to the training run that produced it.
MLFLOW_LINK_FILENAME = "mlflow_train.json"

# ===============================
# Sweep planning
# ===============================


def sweep_run_key(name: str, vocab_size: int) -> str:
    """Return the canonical run key for a tokenizer and vocab size.

    Args:
        name: Tokenizer registry key.
        vocab_size: Vocabulary size.

    Returns:
        `"<name>-<vocab_size>"`.
    """
    return f"{name}-{vocab_size}"


def parse_sweep_run_key(key: str) -> tuple[str, int]:
    """Split a run key back into its tokenizer name and vocab size.

    Args:
        key: A key produced by `sweep_run_key`.

    Returns:
        The tokenizer name and vocab size.

    Raises:
        ValueError: If `key` is not of the form `<name>-<int>`.
    """
    name, _, vocab = key.rpartition("-")
    if not name or not vocab.isdigit():
        raise ValueError(f"Malformed run key: {key!r} (expected '<name>-<vocab>').")
    return name, int(vocab)


def plan_sweep_runs(cfg: SweepConfig) -> list[str]:
    """Enumerate every tokenizer x vocab-size run key in the sweep.

    Args:
        cfg: The resolved sweep configuration.

    Returns:
        Run keys in tokenizer-major, vocab-minor order.
    """
    return [sweep_run_key(name, vocab) for name, vocab in product(cfg.tokenizers, cfg.vocab_sizes)]


def corpus_split_for(cfg: SweepConfig, max_workers: int = 1) -> CorpusSplit | None:
    """Return the sweep's corpus split, fitting and saving it on first use.

    Args:
        cfg: The resolved sweep configuration.
        max_workers: Sources ranked at once when the split is fitted.

    Returns:
        The split, or None when the config sets no
            `corpus.split.train_share`.

    Raises:
        ValueError: If a saved split was fitted with another training share or
            on a corpus whose shards have changed since.
    """
    if cfg.corpus_train_share is None:
        return None

    # reuse a saved split after checking it still fits
    if cfg.corpus_split_path.exists():
        split = load_corpus_split(cfg.corpus_split_path)
        if split.train_share != cfg.corpus_train_share:
            raise ValueError(
                f"{cfg.corpus_split_path} holds train_share {split.train_share}, "
                f"config asks for {cfg.corpus_train_share}; remove the file to refit"
            )
        if split.corpus_fingerprint != corpus_fingerprint(cfg.corpus_root):
            raise ValueError(f"{cfg.corpus_root} changed since {cfg.corpus_split_path} was fitted; remove it to refit")
        return split

    # fit and save a new split
    logger.info("Fitting the %.0f/%.0f corpus split", 100 * cfg.corpus_train_share, 100 * (1 - cfg.corpus_train_share))
    split = fit_corpus_split(cfg.corpus_root, cfg.corpus_train_share, max_workers)
    save_corpus_split(split, cfg.corpus_split_path)
    return split


# ===============================
# Input preparation
# ===============================


def prepare_inputs(cfg: SweepConfig, force: bool = False, max_workers: int = 1) -> tuple[Path, Path | None]:
    """Materialize the shared training sample and morpheme lexicon.

    Both artifacts are reused across every run. Existing artifacts are kept
    unless `force` is set. With a corpus split configured, the sample is drawn
    from its training side; with held-out lexemes configured, their forms are
    left out of the morpheme table.

    Args:
        cfg: The resolved sweep configuration.
        force: Rebuild even when the artifacts already exist.
        max_workers: Sources ranked at once if the corpus split is fitted.

    Returns:
        The training-sample path and the morpheme
            lexicon path (None when no morphological backend is requested).
    """
    # prepare the training sample
    sample_path = cfg.corpus_sample_path
    if force or not sample_path.exists():
        logger.info("Sampling training corpus -> %s", sample_path)
        split = corpus_split_for(cfg, max_workers)
        materialize_sample(cfg.corpus_root, cfg.train_budget, sample_path, split, "train" if split else None)
    else:
        if cfg.corpus_train_share is not None:
            check_sample_side(sample_path, "train")
        logger.info("Reusing cached training sample: %s", sample_path)

    # prepare the morpheme lexicon
    lexicon_path: Path | None = None
    if cfg.needs_morphology():
        lexicon_path = cfg.lexicon_path
        if force or not lexicon_path.exists():
            logger.info("Deriving morpheme lexicon from %s", cfg.sloleks_path)
            infl = build_morph_lexicon(cfg.sloleks_path, min_stem_len=cfg.min_stem_len)
            save_lexicon(infl, cfg.infl_lexicon_path)
            table = infl
            if cfg.needs_derivational():
                logger.info("Deriving derivational lexicon from %s", cfg.sloleks_relations_path)
                deriv = build_derivational_lexicon(cfg.sloleks_relations_path)
                save_lexicon(deriv, cfg.deriv_lexicon_path)
                # derivational splits win ties over the trivial single-stem inflectional entry
                table = merge_lexicons(deriv, infl)
            if cfg.heldout_lexeme_share is not None:
                heldout = split_lexemes(cfg.inflection_path, cfg.heldout_lexeme_share).heldout_forms
                logger.info("Leaving %d held-out forms out of the morpheme table", len(heldout))
                table = without_forms(table, heldout)
            save_lexicon(table, lexicon_path)
        else:
            if cfg.heldout_lexeme_share is not None:
                _check_table_excludes_heldout(cfg, lexicon_path)
            logger.info("Reusing cached morpheme lexicon: %s", lexicon_path)
    return sample_path, lexicon_path


def _check_table_excludes_heldout(cfg: SweepConfig, lexicon_path: Path) -> None:
    """Refuse a cached morpheme table that holds a held-out form.

    Args:
        cfg: The resolved sweep configuration, with held-out lexemes set.
        lexicon_path: The cached morpheme table.

    Raises:
        ValueError: If the table and the held-out forms share any form.
    """
    heldout = split_lexemes(cfg.inflection_path, cfg.heldout_lexeme_share).heldout_forms
    leaked = heldout & load_lexicon(lexicon_path).by_form.keys()
    if leaked:
        raise ValueError(f"{lexicon_path} holds {len(leaked)} held-out forms; rerun `sample --force`")


# ===============================
# Training
# ===============================


def train_one(
    key: str,
    *,
    cfg: SweepConfig,
    sample_path: Path,
    lexicon_path: Path | None,
    force: bool = False,
    alphabet: frozenset[str] | None = None,
) -> Path | None:
    """Train and persist one tokenizer x vocab-size run.

    Args:
        key: Run key (`<name>-<vocab>`).
        cfg: The resolved sweep configuration.
        sample_path: Cached training-sample path.
        lexicon_path: Morpheme lexicon path for morphological
            backends, or None.
        force: Retrain even when the artifact already exists.
        alphabet: Base alphabet of the
            character-level backends (`sample_alphabet`), or None for all.

    Returns:
        The artifact directory, or None when the run was
            skipped because its output already exists.

    Raises:
        ValueError: If a morphological run is requested without a lexicon.
    """
    name, vocab_size = parse_sweep_run_key(key)
    out_dir = cfg.output_root / key
    if (out_dir / "metadata.json").exists() and not force:
        logger.info("Skipping %s; artifact exists (use --force to retrain).", key)
        return None

    # load the morpheme lexicon for morphological backends
    lexicon = None
    if name.startswith("morph"):
        if lexicon_path is None:
            raise ValueError(f"{name} requires a morpheme lexicon but none was prepared.")
        lexicon = load_lexicon(lexicon_path)

    # train the tokenizer
    context = TrainContext(
        special_tokens=list(cfg.special_tokens),
        seed=cfg.train_budget.seed,
        lexicon=lexicon,
        alphabet=alphabet,
    )
    tokenizer = get_tokenizer(name)()
    logger.info("Training %s (vocab=%d)", name, vocab_size)
    start = time.perf_counter()
    with track_peak_rss() as memory:
        tokenizer.train(iter_sample(sample_path), vocab_size, config=context)
    train_seconds = time.perf_counter() - start

    # save the tokenizer and its stats
    tokenizer.save(out_dir)
    _write_train_stats(out_dir, key, name, vocab_size, len(tokenizer.vocab), train_seconds, cfg, memory, alphabet)
    logger.info("Saved %s -> %s (%.1fs)", key, out_dir, train_seconds)
    return out_dir


# ===============================
# Memory tracking
# ===============================


def _resident_bytes() -> int | None:
    """Return this process's resident memory, read from `/proc/self/status`.

    Returns:
        Resident set size in bytes, or None where `/proc` is absent.
    """
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


@contextmanager
def track_peak_rss() -> Iterator[dict[str, int | None]]:
    """Track the peak resident memory of this process while the block runs.

    A pool worker trains several runs in one process, so the lifetime peak
    (`ru_maxrss`) would carry over from earlier runs; polling during the block
    gives each run its own peak.

    Yields:
        `start_rss_bytes` and, once the block exits,
            `peak_rss_bytes`; both None where `/proc` is absent.
    """
    # take the starting reading
    start = _resident_bytes()
    reading: dict[str, int | None] = {"start_rss_bytes": start, "peak_rss_bytes": start}
    stop = threading.Event()

    def record() -> None:
        current = _resident_bytes()
        if current is not None and current > (reading["peak_rss_bytes"] or 0):
            reading["peak_rss_bytes"] = current

    def poll() -> None:
        while not stop.wait(RSS_POLL_SECONDS):
            record()

    # poll the peak in the background while the block runs
    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    try:
        yield reading
    finally:
        stop.set()
        poller.join()
        record()


def _write_train_stats(
    out_dir: Path,
    key: str,
    name: str,
    vocab_size: int,
    vocab_used: int,
    train_seconds: float,
    cfg: SweepConfig,
    memory: dict[str, int | None],
    alphabet: frozenset[str] | None = None,
) -> None:
    """Write the per-run training-stats sidecar for later MLflow logging.

    Training runs in a process pool, so stats are persisted next to the
    artifact and logged from the parent process after the pool drains.

    Args:
        out_dir: The run's artifact directory.
        key: Run key (`<name>-<vocab>`).
        name: Tokenizer registry name.
        vocab_size: Target vocabulary size.
        vocab_used: Actual vocabulary size after training.
        train_seconds: Wall-clock training time in seconds.
        cfg: The resolved sweep configuration.
        memory: What `track_peak_rss` measured.
        alphabet: The base alphabet the run trained
            with, or None for every character.
    """
    stats = {
        "run_key": key,
        "tokenizer": name,
        "vocab_size": vocab_size,
        "vocab_used": vocab_used,
        "train_seconds": train_seconds,
        "seed": cfg.train_budget.seed,
        "n_special_tokens": len(cfg.special_tokens),
        "max_bytes": cfg.train_budget.max_bytes,
        "max_docs": cfg.train_budget.max_docs,
        "character_coverage": cfg.character_coverage,
        "alphabet_size": None if alphabet is None else len(alphabet),
        "alphabet_sha256": None
        if alphabet is None
        else hashlib.sha256("".join(sorted(alphabet)).encode("utf-8")).hexdigest(),
        **memory,
    }
    (out_dir / TRAIN_STATS_FILENAME).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


# ===============================
# Run selection
# ===============================


def select_sweep_runs(
    cfg: SweepConfig,
    *,
    run_keys: list[str] | None = None,
    tokenizers: list[str] | None = None,
    vocab_sizes: list[int] | None = None,
) -> list[str]:
    """Resolve which run keys to execute from explicit keys or filters.

    Args:
        cfg: The resolved sweep configuration.
        run_keys: Explicit run keys; takes precedence.
        tokenizers: Restrict to these tokenizer names.
        vocab_sizes: Restrict to these vocab sizes.

    Returns:
        The selected run keys, preserving sweep order.
    """
    # explicit keys take precedence
    planned = plan_sweep_runs(cfg)
    if run_keys:
        planned_set = set(planned)
        unknown = [k for k in run_keys if k not in planned_set]
        if unknown:
            logger.warning("Run key(s) not in the configured sweep: %s", unknown)
        return list(run_keys)

    # filter by tokenizer and vocab size
    selected = planned
    if tokenizers:
        wanted = set(tokenizers)
        selected = [k for k in selected if parse_sweep_run_key(k)[0] in wanted]
    if vocab_sizes:
        wanted_vocab = set(vocab_sizes)
        selected = [k for k in selected if parse_sweep_run_key(k)[1] in wanted_vocab]
    return selected


def resolve_sweep_runs(
    cfg: SweepConfig,
    *,
    all_runs: bool,
    tokenizer: str | None,
    vocab_size: int | None,
) -> list[str]:
    """Resolve run keys from the shared one-or-all CLI selection.

    The tokenizer scripts (`train`, `analyze`, `export`) select runs the same
    way: either every run (`all_runs`) or a single `tokenizer`, optionally
    narrowed to one `vocab_size`. The two modes are mutually exclusive.

    Args:
        cfg: The resolved sweep configuration.
        all_runs: Select the whole sweep (the `--all` flag).
        tokenizer: The one tokenizer to select, or None.
        vocab_size: A single vocab size narrowing `tokenizer`,
            or None for all of its sizes.

    Returns:
        The selected run keys, in sweep order.

    Raises:
        ValueError: If the selection is missing, combined illegally, or names a
            tokenizer/vocab size that is not in the configured sweep.
    """
    if all_runs:
        if tokenizer or vocab_size is not None:
            raise ValueError("--all takes no other selector; drop --tokenizer/--vocab-size.")
        return select_sweep_runs(cfg)
    if not tokenizer:
        raise ValueError("Specify --tokenizer NAME (optionally --vocab-size N), or --all.")
    if tokenizer not in cfg.tokenizers:
        raise ValueError(f"Unknown tokenizer {tokenizer!r}. Configured: {', '.join(cfg.tokenizers)}.")
    if vocab_size is not None and vocab_size not in cfg.vocab_sizes:
        raise ValueError(f"Unknown vocab size {vocab_size}. Configured: {cfg.vocab_sizes}.")
    vocab_sizes = [vocab_size] if vocab_size is not None else None
    return select_sweep_runs(cfg, tokenizers=[tokenizer], vocab_sizes=vocab_sizes)


# ===============================
# MLflow logging
# ===============================


def _write_mlflow_link(
    out_dir: Path,
    experiment: str,
    parent_run_id: str | None,
    run_id: str | None,
    run_name: str,
) -> None:
    """Persist the MLflow training-run linkage next to the artifact.

    The evaluation logger reads this sidecar to cross-link each eval run back to
    the training run that produced the artifact.

    Args:
        out_dir: The run's artifact directory.
        experiment: MLflow experiment name.
        parent_run_id: The training sweep parent run id.
        run_id: The per-run training child run id.
        run_name: The training child run name (the run key).
    """
    link = {
        "experiment": experiment,
        "parent_run_id": parent_run_id,
        "run_id": run_id,
        "run_name": run_name,
    }
    (out_dir / MLFLOW_LINK_FILENAME).write_text(json.dumps(link, ensure_ascii=False, indent=2), encoding="utf-8")


def run_tags(cfg: SweepConfig) -> dict[str, str]:
    """Return the tags every sweep run carries beside its structural ones.

    Args:
        cfg: The resolved sweep configuration.

    Returns:
        The configured `mlflow_tags` plus `base_commit`, the
            commit the run executes from.
    """
    return {**cfg.mlflow_tags, "base_commit": ml.git_commit() or "unknown"}


def sweep_inputs(paths: dict[str, Path]) -> list[tuple[str, str, str]]:
    """Digest the shared sweep inputs that exist, for logging as dataset inputs.

    Args:
        paths: Input name (e.g. `train-sample`) to its file.

    Returns:
        `(name, digest, path)` per existing file, the
            name prefixed `sweep/` and the digest a `sha256:` content hash.
    """
    return [
        (f"sweep/{name}", f"sha256:{file_sha256(path)}", str(path)) for name, path in paths.items() if path.exists()
    ]


def log_sweep_lineage(cfg: SweepConfig, inputs: list[tuple[str, str, str]], context: str) -> None:
    """Log the split shares and the digested sweep inputs on the active run.

    Args:
        cfg: The resolved sweep configuration.
        inputs: What `sweep_inputs` returned.
        context: Lineage context of the inputs (`training` or `eval`).
    """
    ml.log_params({"corpus_train_share": cfg.corpus_train_share, "heldout_lexeme_share": cfg.heldout_lexeme_share})
    for name, digest, source in inputs:
        ml.log_dataset_input(name, digest, source, context=context)


def log_train_runs(keys: list[str], cfg: SweepConfig) -> None:
    """Log the training sweep to MLflow as a parent run with nested children.

    Each child run records the run's training parameters and timing under the
    `phase=train` tag, mirroring the evaluation logger's structure so train and
    eval runs share the experiment but stay separate. The per-run MLflow ids are
    written back to a sidecar so the evaluation logger can cross-link eval runs to
    their training run. A no-op when tracking is disabled or MLflow is absent.

    Args:
        keys: Run keys whose `train_stats.json` should be logged.
        cfg: The resolved sweep configuration.
    """
    if not cfg.mlflow_enabled:
        return
    if not ml.ensure_experiment(cfg.mlflow_experiment, tracking_uri=cfg.mlflow_tracking_uri):
        logger.warning("MLflow unavailable; skipping training logging.")
        return

    # collect the tags and shared inputs
    commit = ml.git_commit()
    tags = run_tags(cfg)
    inputs = sweep_inputs(
        {
            "corpus-split": cfg.corpus_split_path,
            "train-sample": cfg.corpus_sample_path,
            "morph-table": cfg.lexicon_path,
        }
    )

    # log one child run per trained tokenizer
    with ml.mlflow_run("sweep-train", tags={**tags, "run_type": "sweep", "phase": "train"}) as parent:
        parent_run_id = parent.info.run_id if parent is not None else None
        for key in keys:
            stats_path = cfg.output_root / key / TRAIN_STATS_FILENAME
            if not stats_path.exists():
                logger.warning("No %s for %s; skipping its training log.", TRAIN_STATS_FILENAME, key)
                continue
            stats: dict[str, Any] = json.loads(stats_path.read_text(encoding="utf-8"))
            with ml.mlflow_run(
                key,
                nested=True,
                tags={
                    **tags,
                    "model_type": stats["tokenizer"],
                    "model_version": str(stats["vocab_size"]),
                    "run_type": "sweep",
                    "phase": "train",
                    "git_commit": commit or "unknown",
                },
            ) as child:
                ml.log_params(
                    {
                        "tokenizer": stats["tokenizer"],
                        "vocab_size": stats["vocab_size"],
                        "seed": stats.get("seed"),
                        "max_bytes": stats.get("max_bytes"),
                        "max_docs": stats.get("max_docs"),
                        "n_special_tokens": stats.get("n_special_tokens"),
                        **{
                            k: stats.get(k)
                            for k in ("character_coverage", "alphabet_size", "alphabet_sha256")
                            if k in stats
                        },
                    }
                )
                log_sweep_lineage(cfg, inputs, context="training")
                ml.log_metrics(
                    {
                        "train_seconds": stats["train_seconds"],
                        "vocab_used": stats["vocab_used"],
                        **{k: stats[k] for k in ("start_rss_bytes", "peak_rss_bytes") if stats.get(k) is not None},
                    }
                )
                # a `produced` dataset input lets a pretraining run declare its tokenizer
                tokenizer_dir = cfg.output_root / key
                ml.log_dataset_input(
                    f"tokenizer/{key}",
                    corpus_digest(tokenizer_dir, globs=TOKENIZER_ARTIFACT_GLOBS),
                    str(tokenizer_dir),
                    context="produced",
                )
                child_run_id = child.info.run_id if child is not None else None
            _write_mlflow_link(cfg.output_root / key, cfg.mlflow_experiment, parent_run_id, child_run_id, key)


# ===============================
# Sweep driver
# ===============================


@dataclass
class TrainSummary:
    """Outcome of a training sweep.

    Attributes:
        trained: Run keys whose artifact was (re)trained.
        skipped: Run keys whose artifact already existed.
        failed: Run keys whose training raised.
    """

    trained: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def sample_alphabet(cfg: SweepConfig, sample_path: Path) -> frozenset[str] | None:
    """Return the characters covering `cfg.character_coverage` of the training sample.

    Args:
        cfg: The resolved sweep configuration.
        sample_path: Cached training-sample path.

    Returns:
        The covering characters, or None when the
            coverage is 1.0 and every character is kept.
    """
    if cfg.character_coverage >= 1.0:
        return None

    # count the characters and pick the covering alphabet
    counts = count_sample_characters(sample_path)
    alphabet = covering_alphabet(counts, cfg.character_coverage)
    logger.info(
        "Alphabet: %d of %d characters cover %.4f of the sample",
        len(alphabet),
        len(counts),
        cfg.character_coverage,
    )
    return alphabet


def longest_first(keys: list[str]) -> list[str]:
    """Order run keys so the slowest runs start first in the pool.

    The morphological backends train far slower than the rest, and every
    backend slows with vocabulary size, so the sweep ends soonest when they
    lead.

    Args:
        keys: Run keys (`<name>-<vocab>`).

    Returns:
        The same keys, morphological backends first, largest vocabulary first.
    """

    def cost(key: str) -> tuple[bool, int]:
        name, vocab = parse_sweep_run_key(key)
        return (not name.startswith("morph"), -vocab)

    return sorted(keys, key=cost)


def train_sweep(
    cfg: SweepConfig,
    keys: list[str],
    force: bool = False,
    max_workers: int = 0,
) -> TrainSummary:
    """Train the selected runs and log them to MLflow.

    Prepares the shared training sample (and the derived morpheme lexicon when a
    morphological backend is selected) once, then trains every run in a process
    pool.

    Args:
        cfg: The resolved sweep configuration.
        keys: Run keys (`<name>-<vocab>`) to train.
        force: Retrain runs whose artifacts already exist. The sample
            and the morpheme table are reused as they are; only
            `prepare_inputs(force=True)` (the `sample --force` step) redraws them.
        max_workers: Parallel runs. 0=auto (cpu_count // 2), 1=serial.

    Returns:
        The trained, skipped, and failed run keys.
    """
    # guard: MLflow must be reachable when tracking is on
    if cfg.mlflow_enabled and not ml.ensure_experiment(cfg.mlflow_experiment, tracking_uri=cfg.mlflow_tracking_uri):
        raise RuntimeError(f"MLflow unreachable for {cfg.mlflow_experiment}; refusing to train runs it cannot log")

    # set up workers and logging
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    workers = resolve_workers(max_workers, len(keys), cpu_default(len(keys)))
    configure_script_logging(parallel=workers > 1, console_level=logging.INFO)

    # prepare the shared inputs
    sample_path, lexicon_path = prepare_inputs(cfg)
    alphabet = sample_alphabet(cfg, sample_path)
    keys = longest_first(keys)

    def kwargs_for(_key: str) -> dict[str, Any]:
        """Return per-run kwargs for `train_one`.

        Args:
            _key: Run key (unused; kwargs are identical per run).

        Returns:
            Keyword arguments for `train_one`.
        """
        return {
            "cfg": cfg,
            "sample_path": sample_path,
            "lexicon_path": lexicon_path,
            "force": force,
            "alphabet": alphabet,
        }

    # train the runs in a process pool
    results, failures = run_parallel(
        train_one,
        keys,
        max_workers=workers,
        desc="tokenizer-train",
        pool="process",
        kwargs_for=kwargs_for,
        log_dir=stamped_log_dir("tokenizer-train"),
    )

    # summarise the outcome
    summary = TrainSummary(
        trained=[k for k, v in results.items() if v is not None],
        skipped=[k for k, v in results.items() if v is None],
        failed=[k for k, _ in failures],
    )
    logger.info(
        "Done. Trained %d, skipped %d, failed %s.",
        len(summary.trained),
        len(summary.skipped),
        summary.failed or "none",
    )

    # log the new runs, plus skipped ones trained by a run that ended before logging
    unlogged = [k for k in summary.skipped if not (cfg.output_root / k / MLFLOW_LINK_FILENAME).exists()]
    if summary.trained or unlogged:
        log_train_runs([*summary.trained, *unlogged], cfg)
    return summary
