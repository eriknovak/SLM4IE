"""Tokenizer training orchestration.

Plans the tokenizer x vocab-size sweep, prepares the shared inputs (one cached
training sample and, when a morphological backend is requested, the derived
morpheme lexicon), and trains a single run. The thin CLI wrapper in
`scripts/tokenizers/train.py` dispatches `train_one` across runs with
`run_parallel`. Inputs are prepared once in the parent so every run trains on
identical text and the corpus is never re-streamed per run.
"""

from __future__ import annotations

import json
import logging
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import slm4ie.tokenizers.backends  # noqa: F401  (registers backends on import)
from slm4ie.tokenizers.base import TrainContext
from slm4ie.tokenizers.corpus import iter_sample_cache, sample_corpus, write_sample_cache
from slm4ie.tokenizers.morphology import build_morph_lexicon, load_lexicon, save_lexicon
from slm4ie.tokenizers.registry import get_tokenizer
from slm4ie.utils import mlflow as ml
from slm4ie.utils.config import TokenizerSweepConfig

logger = logging.getLogger(__name__)

#: Sidecar carrying per-run training stats, read by the MLflow logger.
TRAIN_STATS_FILENAME = "train_stats.json"

#: Sidecar carrying the MLflow run linkage, read by the analysis logger to
#: cross-link each eval run back to the training run that produced it.
MLFLOW_LINK_FILENAME = "mlflow_train.json"


def run_key(name: str, vocab_size: int) -> str:
    """Return the canonical run key for a tokenizer and vocab size.

    Args:
        name (str): Tokenizer registry key.
        vocab_size (int): Vocabulary size.

    Returns:
        str: `"<name>-<vocab_size>"`.
    """
    return f"{name}-{vocab_size}"


def parse_run_key(key: str) -> Tuple[str, int]:
    """Split a run key back into its tokenizer name and vocab size.

    Args:
        key (str): A key produced by `run_key`.

    Returns:
        Tuple[str, int]: The tokenizer name and vocab size.

    Raises:
        ValueError: If `key` is not of the form `<name>-<int>`.
    """
    name, _, vocab = key.rpartition("-")
    if not name or not vocab.isdigit():
        raise ValueError(f"Malformed run key: {key!r} (expected '<name>-<vocab>').")
    return name, int(vocab)


def plan_runs(cfg: TokenizerSweepConfig) -> List[str]:
    """Enumerate every tokenizer x vocab-size run key in the sweep.

    Args:
        cfg (TokenizerSweepConfig): The resolved sweep configuration.

    Returns:
        List[str]: Run keys in tokenizer-major, vocab-minor order.
    """
    return [run_key(name, vocab) for name, vocab in product(cfg.tokenizers, cfg.vocab_sizes)]


def prepare_inputs(cfg: TokenizerSweepConfig, force: bool = False) -> Tuple[Path, Optional[Path]]:
    """Materialize the shared training sample and morpheme lexicon.

    Both artifacts are reused across every run. Existing artifacts are kept
    unless `force` is set.

    Args:
        cfg (TokenizerSweepConfig): The resolved sweep configuration.
        force (bool): Rebuild even when the artifacts already exist.

    Returns:
        Tuple[Path, Optional[Path]]: The training-sample path and the morpheme
            lexicon path (None when no morphological backend is requested).
    """
    sample_path = cfg.corpus_sample_path
    if force or not sample_path.exists():
        logger.info("Sampling training corpus -> %s", sample_path)
        write_sample_cache(sample_corpus(cfg.corpus_root, cfg.train_budget), sample_path)
    else:
        logger.info("Reusing cached training sample: %s", sample_path)

    lexicon_path: Optional[Path] = None
    if cfg.needs_morphology():
        lexicon_path = cfg.lexicon_path
        if force or not lexicon_path.exists():
            logger.info("Deriving morpheme lexicon from %s", cfg.sloleks_path)
            lexicon = build_morph_lexicon(cfg.sloleks_path, min_stem_len=cfg.min_stem_len)
            save_lexicon(lexicon, lexicon_path)
        else:
            logger.info("Reusing cached morpheme lexicon: %s", lexicon_path)
    return sample_path, lexicon_path


def train_one(
    key: str,
    *,
    cfg: TokenizerSweepConfig,
    sample_path: Path,
    lexicon_path: Optional[Path],
    force: bool = False,
) -> Optional[Path]:
    """Train and persist one tokenizer x vocab-size run.

    Args:
        key (str): Run key (`<name>-<vocab>`).
        cfg (TokenizerSweepConfig): The resolved sweep configuration.
        sample_path (Path): Cached training-sample path.
        lexicon_path (Optional[Path]): Morpheme lexicon path for morphological
            backends, or None.
        force (bool): Retrain even when the artifact already exists.

    Returns:
        Optional[Path]: The artifact directory, or None when the run was
            skipped because its output already exists.

    Raises:
        ValueError: If a morphological run is requested without a lexicon.
    """
    name, vocab_size = parse_run_key(key)
    out_dir = cfg.output_root / key
    if (out_dir / "metadata.json").exists() and not force:
        logger.info("Skipping %s; artifact exists (use --force to retrain).", key)
        return None

    lexicon = None
    if name.startswith("morph"):
        if lexicon_path is None:
            raise ValueError(f"{name} requires a morpheme lexicon but none was prepared.")
        lexicon = load_lexicon(lexicon_path)

    context = TrainContext(
        special_tokens=list(cfg.special_tokens),
        seed=cfg.train_budget.seed,
        lexicon=lexicon,
    )
    tokenizer = get_tokenizer(name)()
    logger.info("Training %s (vocab=%d)", name, vocab_size)
    start = time.perf_counter()
    tokenizer.train(iter_sample_cache(sample_path), vocab_size, config=context)
    train_seconds = time.perf_counter() - start
    tokenizer.save(out_dir)
    _write_train_stats(out_dir, key, name, vocab_size, len(tokenizer.vocab), train_seconds, cfg)
    logger.info("Saved %s -> %s (%.1fs)", key, out_dir, train_seconds)
    return out_dir


def _write_train_stats(
    out_dir: Path,
    key: str,
    name: str,
    vocab_size: int,
    vocab_used: int,
    train_seconds: float,
    cfg: TokenizerSweepConfig,
) -> None:
    """Write the per-run training-stats sidecar for later MLflow logging.

    Training runs in a process pool, so stats are persisted next to the
    artifact and logged from the parent process after the pool drains.

    Args:
        out_dir (Path): The run's artifact directory.
        key (str): Run key (`<name>-<vocab>`).
        name (str): Tokenizer registry name.
        vocab_size (int): Target vocabulary size.
        vocab_used (int): Actual vocabulary size after training.
        train_seconds (float): Wall-clock training time in seconds.
        cfg (TokenizerSweepConfig): The resolved sweep configuration.
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
    }
    (out_dir / TRAIN_STATS_FILENAME).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def select_runs(
    cfg: TokenizerSweepConfig,
    *,
    run_keys: Optional[List[str]] = None,
    tokenizers: Optional[List[str]] = None,
    vocab_sizes: Optional[List[int]] = None,
) -> List[str]:
    """Resolve which run keys to execute from explicit keys or filters.

    Args:
        cfg (TokenizerSweepConfig): The resolved sweep configuration.
        run_keys (Optional[List[str]]): Explicit run keys; takes precedence.
        tokenizers (Optional[List[str]]): Restrict to these tokenizer names.
        vocab_sizes (Optional[List[int]]): Restrict to these vocab sizes.

    Returns:
        List[str]: The selected run keys, preserving sweep order.
    """
    planned = plan_runs(cfg)
    if run_keys:
        planned_set = set(planned)
        unknown = [k for k in run_keys if k not in planned_set]
        if unknown:
            logger.warning("Run key(s) not in the configured sweep: %s", unknown)
        return list(run_keys)

    selected = planned
    if tokenizers:
        wanted = set(tokenizers)
        selected = [k for k in selected if parse_run_key(k)[0] in wanted]
    if vocab_sizes:
        wanted_vocab = set(vocab_sizes)
        selected = [k for k in selected if parse_run_key(k)[1] in wanted_vocab]
    return selected


def resolve_run_selection(
    cfg: TokenizerSweepConfig,
    *,
    all_runs: bool,
    tokenizer: Optional[str],
    vocab_size: Optional[int],
) -> List[str]:
    """Resolve run keys from the shared one-or-all CLI selection.

    The tokenizer scripts (`train`, `analyze`, `export`) select runs the same
    way: either every run (`all_runs`) or a single `tokenizer`, optionally
    narrowed to one `vocab_size`. The two modes are mutually exclusive.

    Args:
        cfg (TokenizerSweepConfig): The resolved sweep configuration.
        all_runs (bool): Select the whole sweep (the `--all` flag).
        tokenizer (Optional[str]): The one tokenizer to select, or None.
        vocab_size (Optional[int]): A single vocab size narrowing `tokenizer`,
            or None for all of its sizes.

    Returns:
        List[str]: The selected run keys, in sweep order.

    Raises:
        ValueError: If the selection is missing, combined illegally, or names a
            tokenizer/vocab size that is not in the configured sweep.
    """
    if all_runs:
        if tokenizer or vocab_size is not None:
            raise ValueError("--all takes no other selector; drop --tokenizer/--vocab-size.")
        return select_runs(cfg)
    if not tokenizer:
        raise ValueError("Specify --tokenizer NAME (optionally --vocab-size N), or --all.")
    if tokenizer not in cfg.tokenizers:
        raise ValueError(f"Unknown tokenizer {tokenizer!r}. Configured: {', '.join(cfg.tokenizers)}.")
    if vocab_size is not None and vocab_size not in cfg.vocab_sizes:
        raise ValueError(f"Unknown vocab size {vocab_size}. Configured: {cfg.vocab_sizes}.")
    vocab_sizes = [vocab_size] if vocab_size is not None else None
    return select_runs(cfg, tokenizers=[tokenizer], vocab_sizes=vocab_sizes)


def _write_mlflow_link(
    out_dir: Path,
    experiment: str,
    parent_run_id: Optional[str],
    run_id: Optional[str],
    run_name: str,
) -> None:
    """Persist the MLflow training-run linkage next to the artifact.

    The analysis logger reads this sidecar to cross-link each eval run back to
    the training run that produced the artifact.

    Args:
        out_dir (Path): The run's artifact directory.
        experiment (str): MLflow experiment name.
        parent_run_id (Optional[str]): The training sweep parent run id.
        run_id (Optional[str]): The per-run training child run id.
        run_name (str): The training child run name (the run key).
    """
    link = {
        "experiment": experiment,
        "parent_run_id": parent_run_id,
        "run_id": run_id,
        "run_name": run_name,
    }
    (out_dir / MLFLOW_LINK_FILENAME).write_text(json.dumps(link, ensure_ascii=False, indent=2), encoding="utf-8")


def log_training_to_mlflow(keys: List[str], cfg: TokenizerSweepConfig) -> None:
    """Log the training sweep to MLflow as a parent run with nested children.

    Each child run records the run's training parameters and timing under the
    `phase=train` tag, mirroring the analysis logger's structure so train and
    eval runs share the experiment but stay separate. The per-run MLflow ids are
    written back to a sidecar so the analysis logger can cross-link eval runs to
    their training run. A no-op when tracking is disabled or MLflow is absent.

    Args:
        keys (List[str]): Run keys whose `train_stats.json` should be logged.
        cfg (TokenizerSweepConfig): The resolved sweep configuration.
    """
    if not cfg.mlflow_enabled:
        return
    if not ml.ensure_experiment(cfg.mlflow_experiment, tracking_uri=cfg.mlflow_tracking_uri):
        logger.warning("MLflow unavailable; skipping training logging.")
        return

    commit = ml.git_commit()
    with ml.mlflow_run("sweep-train", tags={"run_type": "sweep", "phase": "train"}) as parent:
        parent_run_id = parent.info.run_id if parent is not None else None
        for key in keys:
            stats_path = cfg.output_root / key / TRAIN_STATS_FILENAME
            if not stats_path.exists():
                logger.warning("No %s for %s; skipping its training log.", TRAIN_STATS_FILENAME, key)
                continue
            stats: Dict[str, Any] = json.loads(stats_path.read_text(encoding="utf-8"))
            with ml.mlflow_run(
                key,
                nested=True,
                tags={
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
                    }
                )
                ml.log_metrics(
                    {
                        "train_seconds": stats["train_seconds"],
                        "vocab_used": stats["vocab_used"],
                    }
                )
                child_run_id = child.info.run_id if child is not None else None
            _write_mlflow_link(cfg.output_root / key, cfg.mlflow_experiment, parent_run_id, child_run_id, key)
