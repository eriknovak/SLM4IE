"""Build the final SLM4IE pretraining corpus stage-by-stage.

`scripts/data/to_pretrain.py` runs the eight-stage curation pipeline:

    0. convert        -> <output_dir>/00_convert/
    1. language       -> <output_dir>/01_language/
    2. spam           -> <output_dir>/02_spam/
    3. quality        -> <output_dir>/03_quality/
    4. repetition     -> <output_dir>/04_repetition/
    5. exact_dedup    -> <output_dir>/05_1_dedup/
    6. sentence_dedup -> <output_dir>/05_2_dedup/   (final corpus)
    7. stats          -> <output_dir>/06_statistics/

Stage 0 turns the extraction-output `<key>.jsonl` files into
datatrove-shaped `<key>/<NNNNN>.jsonl.gz` shards (formerly the job of
the standalone `to_datatrove.py` script).

Each stage writes a `.complete` sentinel into its output folder. On
rerun, a stage's sentinel is compared against a fresh hash of its
config slice; on mismatch the stage and every downstream stage are
invalidated and re-executed.

The pipeline has two modes:

* **Positional-keys mode** (e.g. ``kzb solar``): runs the four scoped stages
  (convert → language → quality → repetition) for the named datasets only and
  stops there.  Corpus stages (exact_dedup, sentence_dedup, stats) are never
  run in this mode; pass ``--all`` once all datasets are ready to run them.

* **``--all`` mode**: runs every stage for every dataset declared in
  ``extract.yaml``, skipping any per-dataset or corpus sentinels that are
  already current.  Corpus stages (exact_dedup, sentence_dedup, stats) require
  ``--all``; passing ``--stage exact_dedup`` without ``--all`` is an error.

Examples:
    # Process scoped stages for two datasets; corpus stages run later.
    uv run python scripts/data/to_pretrain.py kzb solar

    # Run (or resume) the full pipeline over every dataset.
    uv run python scripts/data/to_pretrain.py --all

    # Re-run quality and all downstream stages for every dataset.
    uv run python scripts/data/to_pretrain.py --all --force --stage quality

    # Force-rebuild just the corpus dedup stages (requires --all).
    uv run python scripts/data/to_pretrain.py --all --force --stage exact_dedup

    # Use cpu_count // 2 workers (default is serial: 1 worker).
    uv run python scripts/data/to_pretrain.py --all --max-workers 0
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, cast

import yaml

from datatrove.pipeline.dedup import SentDedupConfig

from slm4ie.data.curate import (
    ALL_STAGE_NAMES,
    STAGE_NAMES,
    cascade_from,
    cascade_invalidate,
    config_hash,
    read_sentinel,
    sentinel_is_current,
    upstream_stage,
    write_sentinel,
)
from slm4ie.data.curate.stages import CORPUS_STAGES, SCOPED_STAGES, is_scoped
from slm4ie.data.curate.overrides import effective_stage_config, validate_overrides
from slm4ie.data.curate.sentinel import (
    SENTINEL_NAME,
    cascade_invalidate_scoped,
    dataset_sentinel_is_current,
    invalidate_dataset_sentinels,
    write_dataset_sentinel,
)
from slm4ie.data.curate.convert import (
    DEFAULT_ID_FIELD,
    DEFAULT_METADATA_FIELDS,
    DEFAULT_TEXT_FIELD,
    run_convert_stage,
)
from slm4ie.data.curate.dedup import make_exact_config
from slm4ie.data.curate.pipeline import (
    CuratePaths,
    QualityConfig,
    build_language_executors,
    build_quality_executors,
    build_repetition_executors,
    build_spam_executors,
    build_exact_dedup_executors,
    build_sentence_dedup_executors,
    build_stats_executors,
    pipeline_io_counts,
)
from slm4ie.data.curate.spam import SpamAssets, SpamConfig, load_spam_assets
from slm4ie.data.io_utils import DEFAULT_MAX_SHARD_BYTES, find_project_root as _find_project_root
from slm4ie.data.stopwords import load_stopwords

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the SLM4IE pretraining corpus stage-by-stage. Each "
            "stage writes a durable artifact under <output_dir>/ and a "
            ".complete sentinel; on rerun, stale stages auto-invalidate."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("datasets", nargs="*", default=[], help="Dataset keys.")
    target.add_argument("--all", action="store_true", help="Process every dataset.")
    parser.add_argument(
        "--stage",
        choices=ALL_STAGE_NAMES,
        default="all",
        help="Stage to run. Default: all (skips finished stages).",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Override pretrain.yaml::input_dir.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override pretrain.yaml::output_dir.",
    )
    parser.add_argument(
        "--pretrain-config", type=Path, default=None,
        help="Path to pretrain.yaml (default: configs/data/pretrain.yaml).",
    )
    parser.add_argument(
        "--extract-config", type=Path, default=None,
        help="Path to extract.yaml (default: configs/data/extract.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Force re-run. With --stage X: drop X's sentinel and all "
            "downstream sentinels. Without --stage: nuke <output_dir>."
        ),
    )
    parser.add_argument(
        "--max-workers", "--tasks",
        dest="workers", type=int, default=1,
        help="Parallel workers. 1=serial (default), 0=cpu_count//2, N=N.",
    )
    args = parser.parse_args(argv)
    if args.all and args.datasets:
        parser.error("argument --all: not allowed with positional datasets")
    if not args.all and not args.datasets:
        parser.error("one of the arguments datasets --all is required")
    if args.datasets and args.stage in CORPUS_STAGES:
        parser.error(
            f"--stage {args.stage} is corpus-wide; run it with --all, "
            "not with positional dataset keys."
        )
    return args


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML file, returning an empty dict when the path does not exist.

    Args:
        path: Path to a YAML file.

    Returns:
        Parsed mapping, or `{}` when the file is missing.
    """
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _list_datasets(extract_config: Path) -> List[str]:
    """Return dataset keys declared in `extract.yaml`.

    Args:
        extract_config: Path to the extraction config.

    Returns:
        Dataset keys in declaration order.
    """
    cfg = _load_yaml(extract_config)
    return list((cfg.get("datasets") or {}).keys())


def _resolve_dirs(
    input_dir: Optional[Path], output_dir: Optional[Path], cfg: Dict[str, Any]
) -> Tuple[Path, Path]:
    """Resolve input/output dirs from overrides or pretrain.yaml.

    Args:
        input_dir: Override for `pretrain.yaml::input_dir`, or None.
        output_dir: Override for `pretrain.yaml::output_dir`, or None.
        cfg: Parsed pretrain.yaml.

    Returns:
        Tuple `(input_dir, output_dir)`.

    Raises:
        FileNotFoundError: If neither the override nor the YAML key is
            set on either side.
    """
    raw_input = input_dir if input_dir is not None else cfg.get("input_dir")
    raw_output = output_dir if output_dir is not None else cfg.get("output_dir")
    if raw_input is None or raw_output is None:
        raise FileNotFoundError(
            "Curation paths not set. Provide --input-dir/--output-dir or set "
            "pretrain.yaml::input_dir / output_dir."
        )
    return Path(raw_input), Path(raw_output)


def _load_stopwords(cfg: Dict[str, Any]) -> Tuple[Set[str], bytes]:
    """Load the stopword set and return (set, raw_bytes_for_hashing).

    Thin wrapper over `slm4ie.data.stopwords.load_stopwords`. Reads the
    language code from `cfg['stopwords']`. A missing or empty key
    disables stopwords (returns an empty set and empty bytes, after
    logging a warning). An unknown code is propagated as `ValueError`
    so a config typo fails the run.

    Args:
        cfg: Parsed pretrain.yaml.

    Returns:
        Tuple of `(stopword set, raw file bytes)`. The bytes are folded
        into the sentinel hash for stages that consume stopwords.

    Raises:
        ValueError: If `cfg['stopwords']` is set to a code that has no
            bundled list under `slm4ie/data/stopwords/`.
    """
    code = cfg.get("stopwords")
    if not code:
        logger.warning("stopwords code not configured; using empty set.")
        return set(), b""
    return load_stopwords(code)


def _load_spam_assets(cfg: Dict[str, Any]) -> SpamAssets:
    """Load the spam-filter lexicons and domain blocklist from config.

    Reads the languages and URL-blocklist toggle from `cfg['spam']`,
    then resolves the curated per-language lexicons plus the domain
    blocklist. The bundle's raw bytes are folded into the spam stage's
    sentinel hash so editing any list invalidates the stage.

    Args:
        cfg: Parsed pretrain.yaml.

    Returns:
        A `SpamAssets` bundle (empty lexicons when no languages are
        configured).

    Raises:
        ValueError: If a configured language has no curated list under
            `slm4ie/data/spam/`.
    """
    scfg = cfg.get("spam") or {}
    languages = scfg.get("languages") or []
    url_blocklist = bool(scfg.get("url_blocklist", True))
    return load_spam_assets(languages, url_blocklist=url_blocklist)


def _filter_stage_subset(stage_dir: Path, keys: List[str]) -> Path:
    """Materialize a tempdir of symlinks restricted to *keys* under *stage_dir*.

    Args:
        stage_dir: A scoped stage's output folder (e.g.
            `<output_dir>/01_language/`).
        keys: Dataset keys to expose.

    Returns:
        Path to a tempdir mirroring the requested keys via symlinks, so a
        downstream stage's reader walks only the subset's shards.

    Raises:
        FileNotFoundError: If any requested shard folder is missing or
            empty under *stage_dir*.
    """
    missing: List[str] = []
    for key in keys:
        src = stage_dir / key
        if not src.is_dir() or not any(src.glob("*.jsonl.gz")):
            missing.append(key)
    if missing:
        raise FileNotFoundError(
            f"No converted shard folder(s) under {stage_dir} for dataset(s): "
            + ", ".join(repr(k) for k in missing)
        )
    holder = Path(tempfile.mkdtemp(prefix="slm4ie-pretrain-subset-"))
    try:
        for key in keys:
            src = stage_dir / key
            holder_key = holder / key
            holder_key.mkdir()
            for shard in src.glob("*.jsonl.gz"):
                (holder_key / shard.name).symlink_to(shard.resolve())
    except BaseException:
        shutil.rmtree(holder, ignore_errors=True)
        raise
    return holder


def _has_stage_output(stage_dir: Path, key: str) -> bool:
    """Return True if *key* has shard output under *stage_dir*.

    Used to drop datasets that produced nothing upstream — declared in the
    roster but never downloaded, or fully filtered out by an earlier stage
    — before a scoped stage tries to read their (nonexistent) shards.

    Args:
        stage_dir: A stage's output folder (e.g. `<output_dir>/00_convert`).
        key: Dataset key to check.

    Returns:
        True if `<stage_dir>/<key>/` exists and holds `.jsonl.gz` shards.
    """
    folder = stage_dir / key
    return folder.is_dir() and any(folder.glob("*.jsonl.gz"))


def _starting_input_hint(paths: CuratePaths, stage: str) -> str:
    """Return an ` (input records ~N)` suffix for a stage's start log line.

    The approximate input size is read from the upstream stage's
    sentinel `records_out` field — already on disk from that stage's
    own run, so no shard scan is needed.

    Args:
        paths: Resolved curation paths.
        stage: Stage about to run.

    Returns:
        An ` (input records ~N)` suffix when the upstream stage's
        sentinel is present, or an empty string otherwise (the first
        stage, or an upstream stage that has not completed).
    """
    upstream = upstream_stage(stage)
    if upstream is None:
        return ""
    sentinel = read_sentinel(paths.stage_dir(upstream))
    if sentinel is None:
        return ""
    return f" (input records ~{sentinel.records_out})"


def _human_bytes(num: int) -> str:
    """Render a byte count as a human-readable string.

    Args:
        num: A non-negative byte count.

    Returns:
        The count scaled to the largest binary unit below 1024, with
        one decimal place for KiB and above (e.g. 1536 -> `"1.5 KiB"`)
        and no decimal for plain bytes (e.g. 512 -> `"512 B"`).
    """
    size = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if size < 1024.0:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} EiB"


def _extracted_input_summary(input_dir: Path, keys: List[str]) -> Tuple[int, int]:
    """Summarize the convert stage's input without reading record contents.

    The convert stage performs a 1:1 format conversion, so its input
    record count is redundant with the output shard count taken after
    the run. Counting input lines would mean reading every `<key>.jsonl`
    in full; instead this reports the dataset count and the summed
    on-disk size via `stat` only.

    Args:
        input_dir: Directory holding extraction output JSONLs.
        keys: Dataset keys to include.

    Returns:
        Tuple `(num_datasets_present, total_bytes)`: the count of keys
        whose `<key>.jsonl` exists under *input_dir* and the summed
        on-disk size of those files. Keys with no `<key>.jsonl` are
        excluded from both figures (the convert stage reports them as
        skipped).
    """
    present = 0
    total_bytes = 0
    for key in keys:
        path = input_dir / f"{key}.jsonl"
        if not path.exists():
            continue
        present += 1
        total_bytes += path.stat().st_size
    return present, total_bytes


def _purge_dedup_state(paths: CuratePaths, which: str) -> None:
    """Purge the dedup scratch for *which* stage.

    Args:
        paths: Resolved curation paths.
        which: Either `"exact_dedup"` or `"sentence_dedup"`.
    """
    prefix = {"exact_dedup": "exact", "sentence_dedup": "sent"}[which]
    for sub in (paths.dedup_state_dir / f"{prefix}_sigs", paths.dedup_state_dir / f"{prefix}_dups"):
        if sub.exists():
            shutil.rmtree(sub, ignore_errors=True)


@dataclass(frozen=True)
class ConvertParams:
    """Resolved convert-stage parameters for one config bucket.

    Attributes:
        text_field: Record field copied into `text`.
        id_field: Record field kept as `doc_id`.
        metadata_fields: Record fields kept under `Document.metadata`.
        include_annotations: Whether to join the annotations sidecar.
        max_shard_bytes: Compressed-byte ceiling per output shard.
    """

    text_field: str
    id_field: str
    metadata_fields: List[str]
    include_annotations: bool
    max_shard_bytes: int


@dataclass(frozen=True)
class LanguageParams:
    """Resolved language-stage parameters for one config bucket.

    Attributes:
        target_languages: ISO 639-1 codes treated as in-language.
        candidate_languages: Candidate set lingua chooses among, or None.
        mode: `filter` drops out-of-target docs; `tag` only annotates.
        minimum_relative_distance: Confidence gap lingua needs to commit.
        low_accuracy: Use lingua's lighter trigram-only model.
        max_chars: Truncate doc text to this many chars, or None.
    """

    target_languages: List[str]
    candidate_languages: Optional[List[str]]
    mode: str
    minimum_relative_distance: float
    low_accuracy: bool
    max_chars: Optional[int]


def _build_convert_params(ccfg: Dict[str, Any]) -> ConvertParams:
    """Resolve a convert-stage config slice into typed parameters.

    Args:
        ccfg: The effective `convert` config slice for one bucket.

    Returns:
        The resolved `ConvertParams`, with defaults applied.
    """
    metadata_fields_raw = ccfg.get("metadata_fields")
    return ConvertParams(
        text_field=str(ccfg.get("text_field", DEFAULT_TEXT_FIELD)),
        id_field=str(ccfg.get("id_field", DEFAULT_ID_FIELD)),
        metadata_fields=(
            [str(f) for f in metadata_fields_raw]
            if metadata_fields_raw is not None
            else list(DEFAULT_METADATA_FIELDS)
        ),
        include_annotations=bool(ccfg.get("include_annotations", False)),
        max_shard_bytes=int(ccfg.get("max_shard_bytes", DEFAULT_MAX_SHARD_BYTES)),
    )


def _build_language_params(lang_cfg: Dict[str, Any]) -> LanguageParams:
    """Resolve a language-stage config slice into typed parameters.

    Args:
        lang_cfg: The effective `language` config slice for one bucket.

    Returns:
        The resolved `LanguageParams`, with defaults applied.
    """
    return LanguageParams(
        target_languages=lang_cfg.get("targets") or ["sl"],
        candidate_languages=lang_cfg.get("candidates"),
        mode=str(lang_cfg.get("mode", "filter")),
        minimum_relative_distance=float(lang_cfg.get("minimum_relative_distance", 0.0)),
        low_accuracy=bool(lang_cfg.get("low_accuracy", False)),
        max_chars=lang_cfg.get("max_chars"),
    )


def _build_spam_config(spcfg: Dict[str, Any]) -> SpamConfig:
    """Resolve a spam-stage config slice into a `SpamConfig`.

    Args:
        spcfg: The effective `spam` config slice for one bucket.

    Returns:
        The resolved `SpamConfig`, with defaults applied.

    Raises:
        ValueError: If `model` is set; no model resolver is wired.
    """
    if spcfg.get("model"):
        raise ValueError(
            "pretrain.yaml::spam.model is set, but no model resolver is "
            "configured. Leave it null, or wire a scorer before enabling it."
        )
    return SpamConfig(
        min_adult_hits=int(spcfg.get("min_adult_hits", 2)),
        min_spam_hits=int(spcfg.get("min_spam_hits", 2)),
        keep_fraction=float(spcfg.get("keep_fraction", 0.0)),
        default_language=str(spcfg.get("default_language", "sl")),
        url_blocklist=bool(spcfg.get("url_blocklist", True)),
        use_ldnoobw=bool(spcfg.get("use_ldnoobw", True)),
        model=spcfg.get("model"),
        model_threshold=float(spcfg.get("model_threshold", 0.5)),
    )


def _build_quality_config(qcfg: Dict[str, Any]) -> QualityConfig:
    """Resolve a quality-stage config slice into a `QualityConfig`.

    Args:
        qcfg: The effective `quality` config slice for one bucket.

    Returns:
        The resolved `QualityConfig`, with defaults applied.
    """
    return QualityConfig(
        min_doc_words=int(qcfg.get("min_doc_words", 50)),
        max_doc_words=int(qcfg.get("max_doc_words", 100_000)),
        min_avg_word_length=int(qcfg.get("min_avg_word_length", 3)),
        max_avg_word_length=int(qcfg.get("max_avg_word_length", 10)),
        max_symbol_word_ratio=float(qcfg.get("max_symbol_word_ratio", 0.1)),
        max_bullet_lines_ratio=float(qcfg.get("max_bullet_lines_ratio", 0.9)),
        max_ellipsis_lines_ratio=float(qcfg.get("max_ellipsis_lines_ratio", 0.3)),
        max_non_alpha_words_ratio=float(qcfg.get("max_non_alpha_words_ratio", 0.8)),
        min_stop_words=int(qcfg.get("min_stop_words", 2)),
    )


def _stage_runner(
    stage: str,
    paths: CuratePaths,
    cfg: Dict[str, Any],
    workers: int,
    stopwords: Set[str],
    spam_assets: SpamAssets,
    dataset_keys: List[str],
    input_view: Optional[Path] = None,
    log_dir: Optional[Path] = None,
) -> Callable[[], Tuple[int, int]]:
    """Return a zero-arg callable that runs *stage*'s executor chain.

    Args:
        stage: Stage name.
        paths: Resolved curation paths.
        cfg: Parsed pretrain.yaml.
        workers: Resolved worker count.
        stopwords: Loaded stopword set (used by quality and stats).
        spam_assets: Loaded spam lexicons and domain blocklist (used by
            the spam stage).
        dataset_keys: Dataset keys to process; consumed by the convert
            stage to know which `<key>.jsonl` files to read.
        input_view: Optional symlink view of the stage's upstream output,
            restricting a scoped stage's reader to the keys being
            (re)run. Consumed by the language, quality, and repetition
            stages; ignored by convert (scoped by `dataset_keys`) and by
            the corpus stages (full-corpus read).
        log_dir: Optional directory for per-task log files (currently
            only consumed by the convert stage).

    Returns:
        A callable that runs the stage when invoked and returns its
        `(records_in, records_out)` document counts. The convert stage
        sums the per-dataset counts `run_convert_stage` returns; the
        datatrove stages read theirs from the run's `PipelineStats`
        via `pipeline_io_counts`. The stats stage reports
        `(records_in, 0)` since it emits a JSON bundle, not shards.

    Raises:
        ValueError: If *stage* is not a known stage name.
    """
    if stage == "convert":
        cparams = _build_convert_params(cfg.get("convert") or {})
        out = paths.stage_dir("convert")

        def run() -> Tuple[int, int]:
            results = run_convert_stage(
                input_dir=paths.input_folder,
                output_dir=out,
                dataset_keys=dataset_keys,
                text_field=cparams.text_field,
                id_field=cparams.id_field,
                metadata_fields=cparams.metadata_fields,
                include_annotations=cparams.include_annotations,
                max_shard_bytes=cparams.max_shard_bytes,
                workers=workers,
                log_dir=log_dir,
            )
            # Convert is a 1:1 format conversion; the count it writes
            # out is also the count it read in. Datasets with no input
            # map to None (skipped) and contribute nothing.
            total = sum(n for n in results.values() if n is not None)
            return total, total

        return run

    if stage == "language":
        # Read through the upstream subset view when only some keys are
        # being (re)run, so the reader skips shards from currently
        # untouched keys.
        lparams = _build_language_params(cfg.get("language") or {})

        def run() -> Tuple[int, int]:
            execs = build_language_executors(
                paths,
                tasks=workers,
                target_languages=lparams.target_languages,
                candidate_languages=lparams.candidate_languages,
                lang_mode=lparams.mode,
                lang_minimum_relative_distance=lparams.minimum_relative_distance,
                lang_low_accuracy=lparams.low_accuracy,
                lang_max_chars=lparams.max_chars,
                input_override=input_view,
            )
            return pipeline_io_counts(execs[-1].run())

        return run

    if stage == "spam":
        spam_config = _build_spam_config(cfg.get("spam") or {})

        def run() -> Tuple[int, int]:
            execs = build_spam_executors(
                paths,
                tasks=workers,
                spam_config=spam_config,
                adult_words=spam_assets.adult_words,
                spam_words=spam_assets.spam_words,
                domains=spam_assets.domains,
                input_override=input_view,
            )
            return pipeline_io_counts(execs[-1].run())

        return run

    if stage == "quality":
        quality_config = _build_quality_config(cfg.get("quality") or {})

        def run() -> Tuple[int, int]:
            execs = build_quality_executors(
                paths,
                tasks=workers,
                quality_config=quality_config,
                stopwords=stopwords,
                input_override=input_view,
            )
            return pipeline_io_counts(execs[-1].run())

        return run

    if stage == "repetition":
        def run() -> Tuple[int, int]:
            execs = build_repetition_executors(
                paths, tasks=workers, input_override=input_view
            )
            return pipeline_io_counts(execs[-1].run())

        return run

    if stage == "exact_dedup":
        edcfg = cfg.get("exact_dedup") or {}
        raw_precision = int(edcfg.get("precision", 64))
        if raw_precision not in (32, 64):
            raise ValueError(
                f"pretrain.yaml::exact_dedup.precision must be 32 or 64, got {raw_precision}"
            )
        raw_hash_fc = str(edcfg.get("hash_fc", "xxhash"))
        if raw_hash_fc not in ("sha1", "xxhash"):
            raise ValueError(
                f"pretrain.yaml::exact_dedup.hash_fc must be 'sha1' or 'xxhash', got {raw_hash_fc!r}"
            )
        # Both values are now narrowed by the runtime checks above; cast to
        # keep static type-checkers happy without losing the Literal contract.
        precision = cast(Literal[32, 64], raw_precision)
        hash_fc = cast(Literal["sha1", "xxhash"], raw_hash_fc)
        exact_cfg = make_exact_config(
            precision=precision,
            hash_fc=hash_fc,
            only_dedup_in_index=bool(edcfg.get("only_dedup_in_index", True)),
        )

        def run() -> Tuple[int, int]:
            try:
                execs = build_exact_dedup_executors(
                    paths,
                    tasks=workers,
                    exact_config=exact_cfg,
                )
                return pipeline_io_counts(execs[-1].run())
            finally:
                _purge_dedup_state(paths, "exact_dedup")

        return run

    if stage == "sentence_dedup":
        scfg = cfg.get("sentence_dedup") or {}
        sent_cfg = SentDedupConfig(
            n_sentences=int(scfg.get("n_sentences", 3)),
            min_doc_words=int(scfg.get("min_doc_words", 50)),
            min_num_sentences=int(scfg.get("min_num_sentences", 2)),
            split_sentences=bool(scfg.get("split_sentences", True)),
        )

        def run() -> Tuple[int, int]:
            try:
                execs = build_sentence_dedup_executors(
                    paths,
                    tasks=workers,
                    sentence_config=sent_cfg,
                )
                return pipeline_io_counts(execs[-1].run())
            finally:
                _purge_dedup_state(paths, "sentence_dedup")

        return run

    if stage == "stats":
        stcfg = cfg.get("stats") or {}

        def run() -> Tuple[int, int]:
            execs = build_stats_executors(
                paths,
                tasks=workers,
                stopwords=stopwords,
                top_k_words=int(stcfg.get("top_k_words", 5_000)),
            )
            # The stats stage emits a JSON bundle, not shards: report the
            # documents it consumed and a zero output count.
            records_in, _ = pipeline_io_counts(execs[-1].run())
            return records_in, 0

        return run

    raise ValueError(f"Unknown stage: {stage}")


def _stage_slice(stage: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the config slice that drives *stage*'s sentinel hash.

    Args:
        stage: Stage name.
        cfg: Parsed pretrain.yaml.

    Returns:
        The mapping under the stage's top-level YAML key, or `{}` if
        absent.
    """
    return dict(cfg.get(stage) or {})


def _convert_input_paths(
    input_dir: Path, key: str, include_annotations: bool
) -> List[Path]:
    """Return the source files the convert stage reads for *key*.

    Args:
        input_dir: Root of the extracted tier holding `<key>.jsonl`.
        key: Dataset key.
        include_annotations: Whether the convert stage also joins the
            `<key>.annotations.jsonl.gz` sidecar.

    Returns:
        The `<key>.jsonl` path, plus the annotations sidecar path when
        `include_annotations` is True. Paths are returned whether or not
        they exist on disk.
    """
    paths = [input_dir / f"{key}.jsonl"]
    if include_annotations:
        paths.append(input_dir / f"{key}.annotations.jsonl.gz")
    return paths


def _convert_input_fingerprint(
    input_dir: Path, key: str, include_annotations: bool
) -> str:
    """Return a cheap size+mtime fingerprint of *key*'s convert inputs.

    The fingerprint is derived from each source file's byte size and
    modification time only — never its contents — so it is computable
    with a single `os.stat` per file. Re-extracting a dataset rewrites
    the file with a new size and/or mtime, which changes the fingerprint
    and so invalidates the cached convert output. A missing file is
    encoded as `absent` so its later appearance also changes the value.

    Args:
        input_dir: Root of the extracted tier holding `<key>.jsonl`.
        key: Dataset key.
        include_annotations: Whether the convert stage also joins the
            `<key>.annotations.jsonl.gz` sidecar (folded in when True).

    Returns:
        A stable string fingerprint, e.g. `"<key>.jsonl=1234:1700000000000"`.
    """
    parts: List[str] = []
    for path in _convert_input_paths(input_dir, key, include_annotations):
        try:
            st = path.stat()
            parts.append(f"{path.name}={st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append(f"{path.name}=absent")
    return ";".join(parts)


def _convert_dataset_current(
    stage_folder: Path,
    key: str,
    expected_hash: str,
    input_dir: Path,
    include_annotations: bool,
) -> bool:
    """Return True iff *key*'s convert output is current for its input.

    Layers an input-freshness check on top of the usual config-hash
    comparison: a convert sentinel is current only when its config hash
    matches AND its recorded input fingerprint still matches the source
    file on disk. Sentinels written before fingerprints existed carry
    none; those are grandfathered by comparing the source file's mtime
    against the sentinel's completion time, so a corpus that has not been
    re-extracted is not needlessly re-converted on first upgrade.

    Args:
        stage_folder: The convert stage's output folder.
        key: Dataset key to check.
        expected_hash: Config hash recomputed from current config.
        input_dir: Root of the extracted tier holding `<key>.jsonl`.
        include_annotations: Whether convert joins the annotations sidecar.

    Returns:
        True if the cached convert output can be reused; False if the
        stage must re-run for this dataset.
    """
    sentinel = read_sentinel(stage_folder / key)
    if sentinel is None or sentinel.config_hash != expected_hash:
        return False
    current_fp = _convert_input_fingerprint(input_dir, key, include_annotations)
    if sentinel.input_fingerprint is not None:
        return sentinel.input_fingerprint == current_fp
    # Legacy sentinel (no fingerprint): treat as current only when every
    # source file predates the recorded completion time.
    return _convert_inputs_predate(
        input_dir, key, include_annotations, sentinel.completed_at
    )


def _convert_inputs_predate(
    input_dir: Path, key: str, include_annotations: bool, completed_at: str
) -> bool:
    """Return True iff every convert input predates *completed_at*.

    Used to grandfather pre-fingerprint convert sentinels: if the source
    files are older than the recorded completion timestamp, the cached
    output already reflects them and need not be rebuilt.

    Args:
        input_dir: Root of the extracted tier holding `<key>.jsonl`.
        key: Dataset key.
        include_annotations: Whether convert joins the annotations sidecar.
        completed_at: ISO-8601 timestamp from the sentinel.

    Returns:
        True if all present input files have an mtime at or before
        *completed_at*; False if any is newer or the timestamp cannot be
        parsed (fail safe: re-run).
    """
    try:
        completed = datetime.fromisoformat(completed_at)
    except ValueError:
        return False
    if completed.tzinfo is None:
        completed = completed.replace(tzinfo=timezone.utc)
    for path in _convert_input_paths(input_dir, key, include_annotations):
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mtime > completed:
            return False
    return True


def _bucket_keys_by_effective_hash(
    keys: List[str],
    stage: str,
    cfg: Dict[str, Any],
    overrides: Dict[str, Any],
    extra: bytes,
) -> Dict[str, List[str]]:
    """Group *keys* by their effective-config hash for *stage*.

    Datasets that resolve to the same effective stage config share a hash
    and can run in one executor; each distinct override forms its own
    bucket. A dataset with no override hashes identically to the plain
    global slice, so the all-defaults case stays a single bucket.

    Args:
        keys: Dataset keys to bucket, in run order.
        stage: Scoped stage name.
        cfg: Parsed pretrain.yaml.
        overrides: The `overrides:` mapping.
        extra: Stage-level extra bytes folded into the hash (stopwords /
            spam lexicon / roster); identical across keys of a stage.

    Returns:
        Mapping of effective-config hash to the keys sharing it. Bucket
        insertion order follows first appearance.
    """
    buckets: Dict[str, List[str]] = {}
    for key in keys:
        slice_ = effective_stage_config(cfg, overrides, key, stage)
        buckets.setdefault(config_hash(slice_, extra=extra), []).append(key)
    return buckets


def _dataset_keys_payload(dataset_keys: List[str]) -> bytes:
    """Return canonical bytes for the dataset key list (for hashing).

    Args:
        dataset_keys: Dataset keys this run will process. Order is
            normalized via `sorted` so positional `kzb solar` and
            `solar kzb` produce the same hash.

    Returns:
        UTF-8 JSON bytes of the sorted key list.
    """
    import json as _json

    return _json.dumps(sorted(dataset_keys), ensure_ascii=False).encode("utf-8")


def _stage_extra(
    stage: str, stopwords_bytes: bytes, spam_bytes: bytes, dataset_keys_bytes: bytes
) -> bytes:
    """Return extra bytes folded into the hash for a stage.

    Corpus stages (exact_dedup, sentence_dedup, stats) fold in the
    dataset roster so adding or removing a dataset invalidates them.
    Scoped stages (convert, language, spam, quality, repetition) exclude
    the roster so per-dataset work survives roster changes. Stopword file
    contents are folded for the stages that consume them (quality,
    stats); the spam lexicon/domain contents are folded for the spam
    stage so editing a list invalidates it.

    Args:
        stage: Stage name.
        stopwords_bytes: Raw bytes of the stopword file.
        spam_bytes: Raw bytes of the spam lexicon and domain lists.
        dataset_keys_bytes: Canonical JSON bytes of the sorted roster.

    Returns:
        Bytes to fold into the stage's sentinel hash.
    """
    roster = b"" if is_scoped(stage) else dataset_keys_bytes
    if stage == "spam":
        # Spam is scoped, so `roster` is empty; the lexicon/domain bytes
        # are what make an edited list invalidate the stage.
        return spam_bytes
    if stage in ("quality", "stats"):
        return stopwords_bytes + b"\x00" + roster if roster else stopwords_bytes
    return roster


def _resolve_requested_stages(stage: str, run_all: bool) -> Tuple[str, ...]:
    """Resolve which stages a run executes.

    A subset run (`run_all` False) with `--stage all` runs only the
    scoped stages and stops before the corpus stages. With `--all`,
    `all` means every stage. An explicit single stage is returned as-is.

    Args:
        stage: The `--stage` value (a stage name or `"all"`).
        run_all: True when `--all` was passed.

    Returns:
        The stage names to execute, in pipeline order.
    """
    if stage != "all":
        return (stage,)
    return STAGE_NAMES if run_all else SCOPED_STAGES


def _apply_force(
    output_dir: Path, *, stage: str, run_all: bool, dataset_keys: List[str]
) -> None:
    """Apply `--force` per the scoped/corpus force matrix.

    Args:
        output_dir: Curation output root.
        stage: The `--stage` value (`"all"` or a stage name).
        run_all: True when `--all` was passed.
        dataset_keys: Keys in play (the full roster under `--all`, else
            the positional subset).
    """
    paths = CuratePaths(input_folder=output_dir, output_dir=output_dir)

    # Whole-corpus reset only when --all is combined with the default
    # (all) stage. A subset `--stage all` must never wipe other datasets.
    if run_all and stage == "all":
        if output_dir.exists():
            for child in output_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        logger.warning("--force: cleared %s", output_dir)
        return

    # Forcing a corpus stage (only reachable under --all): drop the
    # corpus stage folders (data + sentinel) from that stage downstream.
    if stage in CORPUS_STAGES:
        affected = cascade_from(stage)
        for name in affected:
            shutil.rmtree(paths.stage_dir(name), ignore_errors=True)
        if set(affected) & {"exact_dedup", "sentence_dedup"}:
            shutil.rmtree(paths.dedup_state_dir, ignore_errors=True)
        logger.warning("--force --stage %s: removed %s", stage, list(affected))
        return

    # Forcing scoped work: drop the requested keys' per-dataset sentinels
    # and shard subfolders for the affected scoped stages, then drop the
    # corpus stages' sentinels (their data is rebuilt on the next --all).
    scoped_affected = (
        SCOPED_STAGES
        if stage == "all"
        else tuple(s for s in cascade_from(stage) if is_scoped(s))
    )
    for name in scoped_affected:
        folder = paths.stage_dir(name)
        invalidate_dataset_sentinels(folder, dataset_keys)
        for key in dataset_keys:
            shutil.rmtree(folder / key, ignore_errors=True)
    for name in CORPUS_STAGES:
        (paths.stage_dir(name) / SENTINEL_NAME).unlink(missing_ok=True)
    shutil.rmtree(paths.dedup_state_dir, ignore_errors=True)
    logger.warning(
        "--force: reset scoped stages %s for %s + corpus sentinels",
        list(scoped_affected), dataset_keys,
    )


def _curate(
    *,
    datasets: List[str],
    run_all: bool,
    stage: str,
    input_dir: Optional[Path],
    output_dir: Optional[Path],
    force: bool,
    workers: int,
    pretrain_config: Optional[Path] = None,
    extract_config: Optional[Path] = None,
) -> None:
    """Run the curation pipeline (argv-free entry point).

    Args:
        datasets: Positional dataset keys. Must be empty when run_all is True.
        run_all: Process every dataset from the extract config.
        stage: `--stage` value (`"all"` or a stage name).
        input_dir: Override for pretrain.yaml input_dir, or None.
        output_dir: Override for pretrain.yaml output_dir, or None.
        force: Apply the `--force` reset matrix.
        workers: Worker count (0 = auto).
        pretrain_config: Path to pretrain.yaml, or None for the default.
        extract_config: Path to extract.yaml, or None for the default.

    Raises:
        ValueError: If `datasets` is non-empty while `run_all` is True.
    """
    if run_all and datasets:
        raise ValueError("datasets must be empty when run_all is True")
    project_root = _find_project_root()
    pretrain_path = pretrain_config or (project_root / "configs" / "data" / "pretrain.yaml")
    extract_path = extract_config or (project_root / "configs" / "data" / "extract.yaml")
    cfg = _load_yaml(pretrain_path)
    # Per-dataset config overrides: validate against the full roster up
    # front so a typo or out-of-bounds section fails before any stage runs.
    overrides = cfg.get("overrides") or {}
    validate_overrides(overrides, _list_datasets(extract_path))
    input_dir, output_dir = _resolve_dirs(input_dir, output_dir, cfg)
    stopwords, stopwords_raw = _load_stopwords(cfg)
    spam_assets = _load_spam_assets(cfg)

    if run_all:
        dataset_keys = _list_datasets(extract_path)
    else:
        dataset_keys = list(datasets)
    # `workers` is a CPU budget, not an item count. The convert stage caps
    # it at the dataset count itself (run_convert_stage); the datatrove
    # stages use it as their `tasks` rank count, where shards -- far more
    # numerous than datasets -- are the unit of work. So an explicit
    # --max-workers is honored as-is; auto (0) resolves to cpu_count // 2.
    workers = workers if workers > 0 else max(1, (os.cpu_count() or 2) // 2)
    if run_all:
        logger.info("Running on all %d datasets (workers=%d)", len(dataset_keys), workers)
    else:
        logger.info(
            "Running on %d dataset(s): %s (workers=%d)",
            len(dataset_keys), ", ".join(dataset_keys), workers,
        )
    dataset_keys_bytes = _dataset_keys_payload(dataset_keys)

    paths = CuratePaths(input_folder=input_dir, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if force:
        _apply_force(output_dir, stage=stage, run_all=run_all, dataset_keys=dataset_keys)

    requested_stages = _resolve_requested_stages(stage, run_all)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    convert_log_dir = project_root / "logs" / Path(__file__).stem / stamp / "convert"

    # Keys whose per-dataset sentinel an upstream scoped stage just
    # invalidated; the next scoped stage must re-run them even if their
    # own sentinel still looks current.
    force_keys: Set[str] = set()
    for stage_name in requested_stages:
        slice_ = _stage_slice(stage_name, cfg)
        extra = _stage_extra(stage_name, stopwords_raw, spam_assets.raw_bytes, dataset_keys_bytes)
        current_hash = config_hash(slice_, extra=extra)
        stage_folder = paths.stage_dir(stage_name)

        if is_scoped(stage_name):
            # Each dataset is judged against its EFFECTIVE (override-merged)
            # config. Convert additionally reads the extracted tier, so its
            # currency also depends on a size+mtime fingerprint of the
            # source file; later scoped stages read regenerated upstream
            # output and use the plain effective-config-hash check.
            def _effective_hash(k: str) -> str:
                return config_hash(
                    effective_stage_config(cfg, overrides, k, stage_name),
                    extra=extra,
                )

            def _is_current(k: str) -> bool:
                expected = _effective_hash(k)
                if stage_name == "convert":
                    inc = bool(
                        effective_stage_config(
                            cfg, overrides, k, "convert"
                        ).get("include_annotations", False)
                    )
                    return _convert_dataset_current(
                        stage_folder, k, expected, paths.input_folder, inc
                    )
                return dataset_sentinel_is_current(stage_folder, k, expected)

            todo = [
                k for k in dataset_keys
                if k in force_keys or not _is_current(k)
            ]
            if not todo:
                logger.info("[%s] all requested datasets current; skipping.", stage_name)
                continue
            # Drop datasets with no upstream output: declared in the roster
            # but never downloaded, or fully filtered out by an earlier
            # stage. They have no shards to read, so there is nothing to
            # process and the input-view builder would raise. convert reads
            # the extraction tier directly and tolerates missing input.
            upstream = upstream_stage(stage_name)
            up_dir = paths.stage_dir(upstream) if upstream is not None else None
            if up_dir is not None:
                missing = [k for k in todo if not _has_stage_output(up_dir, k)]
                if missing:
                    logger.info(
                        "[%s] skipping %d dataset(s) with no upstream output: %s",
                        stage_name, len(missing), ", ".join(missing),
                    )
                    todo = [k for k in todo if k not in missing]
                if not todo:
                    logger.info("[%s] no datasets with upstream output; skipping.", stage_name)
                    continue
            # Invalidate downstream sentinels for the keys we are about to
            # (re)run, and force every later scoped stage to re-run them.
            # Called per scoped stage with work (not just the first): the
            # repeated drops are idempotent (unlink missing_ok), and each
            # only invalidates keys in `todo`, which are about to re-run.
            # Do NOT collapse to a single first-stage call — a later scoped
            # stage may have its own todo when an earlier one was current.
            cascade_invalidate_scoped(output_dir, stage_name, todo)
            force_keys.update(todo)

            # Bucket todo by effective config so datasets sharing a config
            # run together in one executor; each distinct override forms
            # its own bucket (one extra executor per override).
            buckets = _bucket_keys_by_effective_hash(
                todo, stage_name, cfg, overrides, extra
            )
            logger.info(
                "[%s] %d dataset(s) in %d config group(s)",
                stage_name, len(todo), len(buckets),
            )

            for bucket_hash, bucket_keys in buckets.items():
                effective = effective_stage_config(
                    cfg, overrides, bucket_keys[0], stage_name
                )
                overridden = [
                    k for k in bucket_keys
                    if (overrides.get(k) or {}).get(stage_name)
                ]
                if overridden:
                    logger.info(
                        "[%s] override group %s <- %s",
                        stage_name, overridden, effective,
                    )

                if stage_name == "convert":
                    n_datasets, input_bytes = _extracted_input_summary(
                        paths.input_folder, bucket_keys
                    )
                    logger.info(
                        "[convert] starting (%d dataset(s), %s)",
                        n_datasets, _human_bytes(input_bytes),
                    )
                else:
                    logger.info(
                        "[%s] starting%s",
                        stage_name, _starting_input_hint(paths, stage_name),
                    )

                # Build a symlink view of the upstream output restricted to
                # this bucket; convert reads extraction output directly.
                view = (
                    _filter_stage_subset(up_dir, bucket_keys)
                    if up_dir is not None
                    else None
                )
                # Hand the bucket's effective slice to the runner by swapping
                # just this stage's section in a shallow cfg copy;
                # _stage_runner reads cfg.get(stage_name).
                bucket_cfg = {**cfg, stage_name: effective}
                try:
                    runner = _stage_runner(
                        stage_name,
                        paths,
                        bucket_cfg,
                        workers,
                        stopwords,
                        spam_assets,
                        dataset_keys=bucket_keys,
                        input_view=view,
                        log_dir=convert_log_dir if stage_name == "convert" else None,
                    )
                    records_in, records_out = runner()
                finally:
                    if view is not None:
                        shutil.rmtree(view, ignore_errors=True)

                for key in bucket_keys:
                    # Only convert records an input fingerprint: it is the
                    # sole stage reading the extracted tier, so it is the
                    # only one whose currency depends on the source file.
                    fingerprint = (
                        _convert_input_fingerprint(
                            paths.input_folder,
                            key,
                            bool(effective.get("include_annotations", False)),
                        )
                        if stage_name == "convert"
                        else None
                    )
                    write_dataset_sentinel(
                        stage_folder,
                        key,
                        config_slice=effective,
                        config_hash_value=bucket_hash,
                        records_in=records_in,
                        records_out=records_out,
                        input_fingerprint=fingerprint,
                    )
                logger.info(
                    "[%s] done for %d dataset(s) (records_in=%d, records_out=%d)",
                    stage_name, len(bucket_keys), records_in, records_out,
                )
        else:
            # Corpus stage: only valid under --all (guaranteed by
            # _resolve_requested_stages and parse_args). Stage-level
            # sentinel; full-corpus read.
            if not run_all:
                logger.warning("[%s] corpus stage requires --all; skipping.", stage_name)
                continue
            if sentinel_is_current(stage_folder, current_hash):
                logger.info("[%s] sentinel current; skipping.", stage_name)
                continue
            cascade_invalidate(output_dir, stage_name)
            logger.info("[%s] starting%s", stage_name, _starting_input_hint(paths, stage_name))
            runner = _stage_runner(
                stage_name,
                paths,
                cfg,
                workers,
                stopwords,
                spam_assets,
                dataset_keys=dataset_keys,
                input_view=None,
            )
            records_in, records_out = runner()
            write_sentinel(
                stage_folder,
                config_slice=slice_,
                config_hash_value=current_hash,
                records_in=records_in,
                records_out=records_out,
            )
            logger.info(
                "[%s] done (records_in=%d, records_out=%d)",
                stage_name, records_in, records_out,
            )


def main() -> None:
    """Entry point for the to_pretrain CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    _curate(
        datasets=args.datasets,
        run_all=args.all,
        stage=args.stage,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        force=args.force,
        workers=args.workers,
        pretrain_config=args.pretrain_config,
        extract_config=args.extract_config,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
