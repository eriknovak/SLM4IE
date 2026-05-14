"""Build the final SLM4IE pretraining corpus stage-by-stage.

`scripts/data/to_pretrain.py` runs the seven-stage curation pipeline:

    0. convert        -> <output_dir>/00_convert/
    1. language       -> <output_dir>/01_language/
    2. quality        -> <output_dir>/02_quality/
    3. repetition     -> <output_dir>/03_repetition/
    4. exact_dedup    -> <output_dir>/04_1_dedup/
    5. sentence_dedup -> <output_dir>/04_2_dedup/   (final corpus)
    6. stats          -> <output_dir>/05_statistics/

Stage 0 turns the extraction-output `<key>.jsonl` files into
datatrove-shaped `<key>/<NNNNN>.jsonl.gz` shards (formerly the job of
the standalone `to_datatrove.py` script).

Each stage writes a `.complete` sentinel into its output folder. On
rerun, a stage's sentinel is compared against a fresh hash of its
config slice; on mismatch the stage and every downstream stage are
invalidated and re-executed.

Examples:
    # Run all stages, skipping any whose config slice hash is unchanged.
    uv run python scripts/data/to_pretrain.py --all

    # Run only the quality stage. Re-run downstream stats etc. on next --all.
    uv run python scripts/data/to_pretrain.py --all --stage quality

    # Force-rebuild quality + downstream (drops their sentinels).
    uv run python scripts/data/to_pretrain.py --all --force --stage quality

    # Subset of datasets — dedup operates within the given subset only.
    uv run python scripts/data/to_pretrain.py kzb solar

    # Use cpu_count // 2 workers (default is serial: 1 worker).
    uv run python scripts/data/to_pretrain.py --all --max-workers 0
"""

import argparse
import gzip
import logging
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, cast

import yaml

from datatrove.pipeline.dedup import SentDedupConfig

from slm4ie.data.curate import (
    ALL_STAGE_NAMES,
    STAGE_DIRS,
    STAGE_NAMES,
    cascade_from,
    cascade_invalidate,
    config_hash,
    sentinel_is_current,
    upstream_stage,
    write_sentinel,
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
    build_exact_dedup_executors,
    build_language_executors,
    build_quality_executors,
    build_repetition_executors,
    build_sentence_dedup_executors,
    build_stats_executors,
)
from slm4ie.data.io_utils import DEFAULT_MAX_SHARD_BYTES, find_project_root as _find_project_root
from slm4ie.data.parallel import cpu_default, resolve_workers

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
    args: argparse.Namespace, cfg: Dict[str, Any]
) -> Tuple[Path, Path]:
    """Resolve input/output dirs from CLI flags or pretrain.yaml.

    Args:
        args: Parsed CLI namespace.
        cfg: Parsed pretrain.yaml.

    Returns:
        Tuple `(input_dir, output_dir)`.

    Raises:
        FileNotFoundError: If neither the CLI flag nor the YAML key is
            set on either side.
    """
    raw_input = args.input_dir if args.input_dir is not None else cfg.get("input_dir")
    raw_output = args.output_dir if args.output_dir is not None else cfg.get("output_dir")
    if raw_input is None or raw_output is None:
        raise FileNotFoundError(
            "Curation paths not set. Provide --input-dir/--output-dir or set "
            "pretrain.yaml::input_dir / output_dir."
        )
    return Path(raw_input), Path(raw_output)


def _load_stopwords(project_root: Path, cfg: Dict[str, Any]) -> Tuple[Set[str], bytes]:
    """Load the stopword set and return (set, raw_bytes_for_hashing).

    Args:
        project_root: Project root for resolving relative stopword paths.
        cfg: Parsed pretrain.yaml.

    Returns:
        Tuple of `(stopword set, raw file bytes)`. When `stopwords:` is
        not configured or the file is missing, returns `(set(), b"")`.
    """
    rel = cfg.get("stopwords")
    if not rel:
        return set(), b""
    path = project_root / rel
    if not path.exists():
        logger.warning("stopwords file %s not found; using empty set.", path)
        return set(), b""
    raw = path.read_bytes()
    out: Set[str] = set()
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line.lower())
    return out, raw


def _filter_convert_subset(convert_dir: Path, keys: List[str]) -> Path:
    """Materialize a tempdir of symlinks restricted to *keys* under `00_convert/`.

    Args:
        convert_dir: The convert stage's output folder (typically
            `<output_dir>/00_convert/`).
        keys: Dataset keys to expose.

    Returns:
        Path to a tempdir mirroring the requested keys via symlinks.
        Downstream stages can be pointed at this folder so their
        readers walk only the subset's shards.

    Raises:
        FileNotFoundError: If any requested shard folder is missing or
            empty under *convert_dir*.
    """
    missing: List[str] = []
    for key in keys:
        src = convert_dir / key
        if not src.is_dir() or not any(src.glob("*.jsonl.gz")):
            missing.append(key)
    if missing:
        raise FileNotFoundError(
            f"No converted shard folder(s) under {convert_dir} for dataset(s): "
            + ", ".join(repr(k) for k in missing)
        )
    holder = Path(tempfile.mkdtemp(prefix="slm4ie-pretrain-subset-"))
    for key in keys:
        src = convert_dir / key
        holder_key = holder / key
        holder_key.mkdir()
        for shard in src.glob("*.jsonl.gz"):
            (holder_key / shard.name).symlink_to(shard.resolve())
    return holder


def _count_records(folder: Path) -> int:
    """Sum decompressed JSONL line counts across every shard under *folder*.

    Args:
        folder: A stage output folder containing `<dataset>/<rank>.jsonl.gz`.

    Returns:
        Total record count across all shards. Returns 0 when the folder
        does not exist.
    """
    if not folder.is_dir():
        return 0
    total = 0
    for shard in folder.glob("**/*.jsonl.gz"):
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            total += sum(1 for _ in fh)
    return total


def _count_extracted_records(input_dir: Path, keys: List[str]) -> int:
    """Sum line counts across `<input_dir>/<key>.jsonl` files.

    Args:
        input_dir: Directory holding extraction output JSONLs.
        keys: Dataset keys to include in the count.

    Returns:
        Total line count across all `<key>.jsonl` files; 0 for keys
        whose file is missing (they will be reported as skipped by the
        convert stage).
    """
    total = 0
    for key in keys:
        path = input_dir / f"{key}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            total += sum(1 for _ in fh)
    return total


def _stage_input_dir(
    paths: CuratePaths,
    stage: str,
    convert_view: Optional[Path] = None,
) -> Path:
    """Return the folder a stage reads from.

    Args:
        paths: Resolved curation paths.
        stage: Stage name.
        convert_view: Optional symlink view of `00_convert/` used in
            subset mode. When set, the language stage (and any other
            stage whose upstream is `convert`) reads through it.

    Returns:
        The previous stage's output folder, the convert subset view
        when applicable, or `paths.input_folder` when *stage* is the
        first stage in the pipeline.
    """
    prev = upstream_stage(stage)
    if prev is None:
        return paths.input_folder
    if prev == "convert" and convert_view is not None:
        return convert_view
    return paths.stage_dir(prev)


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


def _stage_runner(
    stage: str,
    paths: CuratePaths,
    cfg: Dict[str, Any],
    workers: int,
    stopwords: Set[str],
    dataset_keys: List[str],
    convert_view: Optional[Path] = None,
    log_dir: Optional[Path] = None,
) -> Callable[[], None]:
    """Return a zero-arg callable that runs *stage*'s executor chain.

    Args:
        stage: Stage name.
        paths: Resolved curation paths.
        cfg: Parsed pretrain.yaml.
        workers: Resolved worker count.
        stopwords: Loaded stopword set (used by quality and stats).
        dataset_keys: Dataset keys to process; consumed by the convert
            stage to know which `<key>.jsonl` files to read.
        convert_view: Optional symlink view of `00_convert/` for subset
            runs (used by the language stage).
        log_dir: Optional directory for per-task log files (currently
            only consumed by the convert stage).

    Returns:
        A callable that runs the stage when invoked.

    Raises:
        ValueError: If *stage* is not a known stage name.
    """
    if stage == "convert":
        ccfg = cfg.get("convert") or {}
        text_field = str(ccfg.get("text_field", DEFAULT_TEXT_FIELD))
        id_field = str(ccfg.get("id_field", DEFAULT_ID_FIELD))
        metadata_fields_raw = ccfg.get("metadata_fields")
        metadata_fields = (
            [str(f) for f in metadata_fields_raw]
            if metadata_fields_raw is not None
            else list(DEFAULT_METADATA_FIELDS)
        )
        include_annotations = bool(ccfg.get("include_annotations", False))
        max_shard_bytes = int(ccfg.get("max_shard_bytes", DEFAULT_MAX_SHARD_BYTES))
        out = paths.stage_dir("convert")

        def run() -> None:
            run_convert_stage(
                input_dir=paths.input_folder,
                output_dir=out,
                dataset_keys=dataset_keys,
                text_field=text_field,
                id_field=id_field,
                metadata_fields=metadata_fields,
                include_annotations=include_annotations,
                max_shard_bytes=max_shard_bytes,
                workers=workers,
                log_dir=log_dir,
            )

        return run

    if stage == "language":
        lang_cfg = cfg.get("language") or {}
        # Read from the convert subset view when running on a dataset
        # subset, so downstream readers don't pick up shards from
        # previously-converted but currently-excluded keys.
        override = convert_view

        def run() -> None:
            execs = build_language_executors(
                paths,
                tasks=workers,
                target_languages=lang_cfg.get("targets") or ["sl"],
                candidate_languages=lang_cfg.get("candidates"),
                lang_mode=str(lang_cfg.get("mode", "filter")),
                lang_minimum_relative_distance=float(
                    lang_cfg.get("minimum_relative_distance", 0.0)
                ),
                lang_low_accuracy=bool(lang_cfg.get("low_accuracy", False)),
                lang_max_chars=lang_cfg.get("max_chars"),
                input_override=override,
            )
            execs[-1].run()

        return run

    if stage == "quality":
        qcfg = cfg.get("quality") or {}
        quality_config = QualityConfig(
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

        def run() -> None:
            execs = build_quality_executors(
                paths,
                tasks=workers,
                quality_config=quality_config,
                stopwords=stopwords,
            )
            execs[-1].run()

        return run

    if stage == "repetition":
        def run() -> None:
            execs = build_repetition_executors(paths, tasks=workers)
            execs[-1].run()

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

        def run() -> None:
            try:
                execs = build_exact_dedup_executors(
                    paths,
                    tasks=workers,
                    exact_config=exact_cfg,
                )
                execs[-1].run()
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

        def run() -> None:
            try:
                execs = build_sentence_dedup_executors(
                    paths,
                    tasks=workers,
                    sentence_config=sent_cfg,
                )
                execs[-1].run()
            finally:
                _purge_dedup_state(paths, "sentence_dedup")

        return run

    if stage == "stats":
        stcfg = cfg.get("stats") or {}

        def run() -> None:
            execs = build_stats_executors(
                paths,
                stopwords=stopwords,
                top_k_words=int(stcfg.get("top_k_words", 5_000)),
                top_k_ngrams=int(stcfg.get("top_k_ngrams", 5_000)),
                keyword_top_k=int(stcfg.get("keyword_top_k", 200)),
                compute_keywords=bool(stcfg.get("compute_keywords", True)),
                ngram_orders=stcfg.get("ngram_orders") or (2, 3),
            )
            execs[-1].run()

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


def _stage_extra(stage: str, stopwords_bytes: bytes, dataset_keys_bytes: bytes) -> bytes:
    """Return extra bytes folded into the hash for a stage.

    The dataset key list is included for every stage so that switching
    between `--all` and a positional subset (or adding/removing a
    dataset from `extract.yaml`) invalidates the sentinels. Stopword
    file contents are included only for stages that consume them
    (`quality`, `stats`).

    Args:
        stage: Stage name.
        stopwords_bytes: Raw bytes of the stopword file.
        dataset_keys_bytes: Canonical JSON bytes of the sorted dataset
            key list.

    Returns:
        Bytes to fold into the sentinel hash.
    """
    if stage in ("quality", "stats"):
        return stopwords_bytes + b"\x00" + dataset_keys_bytes
    return dataset_keys_bytes


def main() -> None:
    """Entry point for the to_pretrain CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    workers = resolve_workers(args.workers, len(STAGE_NAMES), cpu_default(len(STAGE_NAMES)))

    project_root = _find_project_root()
    pretrain_path = args.pretrain_config or (project_root / "configs" / "data" / "pretrain.yaml")
    extract_path = args.extract_config or (project_root / "configs" / "data" / "extract.yaml")
    cfg = _load_yaml(pretrain_path)
    input_dir, output_dir = _resolve_dirs(args, cfg)
    stopwords, stopwords_raw = _load_stopwords(project_root, cfg)

    if args.all:
        dataset_keys = _list_datasets(extract_path)
        logger.info("Running on all %d datasets (workers=%d)", len(dataset_keys), workers)
    else:
        dataset_keys = list(args.datasets)
        logger.info(
            "Running on %d dataset(s): %s (workers=%d)",
            len(dataset_keys), ", ".join(dataset_keys), workers,
        )
    dataset_keys_bytes = _dataset_keys_payload(dataset_keys)

    paths = CuratePaths(input_folder=input_dir, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.force and args.stage == "all":
        if output_dir.exists():
            for child in output_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            logger.warning("--force: cleared %s", output_dir)

    if args.force and args.stage != "all":
        affected = cascade_from(args.stage)
        for name in affected:
            folder = paths.stage_dir(name)
            if folder.exists():
                shutil.rmtree(folder)
        if any(s in affected for s in ("exact_dedup", "sentence_dedup")):
            if paths.dedup_state_dir.exists():
                shutil.rmtree(paths.dedup_state_dir)
        cascade_invalidate(output_dir, args.stage)
        logger.warning(
            "--force --stage %s: removed data folders and sentinels for %s",
            args.stage, list(affected),
        )

    requested_stages = STAGE_NAMES if args.stage == "all" else (args.stage,)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    convert_log_dir = project_root / "logs" / Path(__file__).stem / stamp / "convert"

    subset_holder: Optional[Path] = None
    try:
        cascaded = False
        last_records_out: Optional[int] = None
        for stage in requested_stages:
            slice_ = _stage_slice(stage, cfg)
            extra = _stage_extra(stage, stopwords_raw, dataset_keys_bytes)
            current_hash = config_hash(slice_, extra=extra)
            stage_folder = paths.stage_dir(stage)

            if not cascaded and sentinel_is_current(stage_folder, current_hash):
                logger.info("[%s] sentinel current; skipping.", stage)
                # Skipped — next running stage must re-derive its input count
                # from disk since we did not just produce this stage's output.
                last_records_out = None
                # The convert stage is unique: when skipped under a subset
                # run we still need a symlink view of its on-disk output so
                # downstream readers don't see other keys.
                if stage == "convert" and not args.all and subset_holder is None:
                    subset_holder = _filter_convert_subset(stage_folder, dataset_keys)
                continue

            if not cascaded:
                pre_existing = [
                    s for s in cascade_from(stage)
                    if (output_dir / STAGE_DIRS[s] / ".complete").exists()
                ]
                cascade_invalidate(output_dir, stage)
                if pre_existing:
                    logger.warning(
                        "[%s] cascade-invalidating sentinels for %s", stage, pre_existing
                    )
                cascaded = True

            if last_records_out is not None:
                records_in_before = last_records_out
            elif stage == "convert":
                records_in_before = _count_extracted_records(paths.input_folder, dataset_keys)
            else:
                records_in_before = _count_records(
                    _stage_input_dir(paths, stage, convert_view=subset_holder)
                )
            logger.info("[%s] starting (input records ~%d)", stage, records_in_before)
            runner = _stage_runner(
                stage,
                paths,
                cfg,
                workers,
                stopwords,
                dataset_keys=dataset_keys,
                convert_view=subset_holder,
                log_dir=convert_log_dir if stage == "convert" else None,
            )
            runner()
            # After the convert stage runs in subset mode, materialize a
            # symlink view restricted to the requested keys; downstream
            # stages will then read through it rather than seeing every
            # dataset previously written under 00_convert/.
            if stage == "convert" and not args.all and subset_holder is None:
                subset_holder = _filter_convert_subset(stage_folder, dataset_keys)
            records_out = 0 if stage == "stats" else _count_records(stage_folder)
            last_records_out = records_out
            write_sentinel(
                stage_folder,
                config_slice=slice_,
                config_hash_value=current_hash,
                records_in=records_in_before,
                records_out=records_out,
            )
            logger.info(
                "[%s] done (records_in=%d, records_out=%d)",
                stage, records_in_before, records_out,
            )
    finally:
        if subset_holder is not None:
            shutil.rmtree(subset_holder, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
