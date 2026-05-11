"""Convert SLM4IE processed JSONL into datatrove-compatible JSONL.

Each successful per-dataset conversion writes a `.complete` sentinel
file alongside the shards. The skip-if-populated check looks for that
sentinel rather than for any `*.jsonl.gz`, so a crash mid-conversion
leaves a folder that future runs correctly recognise as incomplete
and re-convert from scratch (or that callers can list to find
truncated outputs). Folders that contain shards but no sentinel —
including any produced by older versions of this script — are
treated as incomplete and re-converted on next run.

The HuggingFace datatrove library represents documents with three
meaningful fields: text, id, and metadata (a free-form dict). Its
built-in JsonlReader automatically funnels any JSONL keys other than
text/id/media into Document.metadata.

This script reads `<key>.jsonl` from the extraction output directory
and writes a per-dataset *folder* of gzipped JSONL shards whose shape
is directly consumable by datatrove while still carrying the
dataset/domain provenance we need for source-weighted sampling.

By default the annotations sidecar (`<key>.annotations.jsonl.gz`) is
**not** read. Annotations are positionally aligned to the original
text, so they silently desync from `text` after any datatrove step
that rewrites it (line/paragraph dedup, boilerplate removal, etc.).
For task-specific fine-tuning use the dedicated converters
(`to_spans`, `to_sentiment`, `to_superglue`) which start from the
extraction directory directly.

Pass `--include-annotations` to opt in: the sidecar is then joined on
the fly and the parallel-array payload is emitted as a nested
`annotations` field on each line. The output then lands in a sibling
`datatrove_annotated/` folder (when `--output-dir` is not overridden)
to keep the cheap pretraining shards and the fat annotated shards
side by side.

Output layout — one folder per dataset, each containing one or more
shards no larger than `--max-shard-bytes` compressed:

    <output_dir>/
    └── <key>/
        ├── 00000.jsonl.gz
        ├── 00001.jsonl.gz
        └── ...

Sharding the output lets datatrove's `JsonlReader` distribute shards
round-robin across worker ranks (`get_shard` returns
`all_files[rank::world_size]`), which is the only way to parallelize a
single dataset — gzip streams are not seekable, so each file is bound
to one rank.

Output line shape per shard:

    {
        "text":        "<document text>",
        "id":          "<source>:<doc_id>",
        "dataset":     "kzb",
        "domain":      "scientific",
        "doc_id":      "s1",
        "<other md>":  "<value>",
        "annotations": { "forms": [...], ... }       # only with --include-annotations
    }

Examples:
    # one dataset (text + provenance only)
    uv run python scripts/data/to_datatrove.py kzb

    # all datasets configured in extract.yaml
    uv run python scripts/data/to_datatrove.py --all

    # opt in to annotations (writes to <processed-dir>/datatrove_annotated/)
    uv run python scripts/data/to_datatrove.py --all --include-annotations

    # larger shards (e.g. 1 GB) when fewer files per dataset are preferred
    uv run python scripts/data/to_datatrove.py --all --max-shard-bytes 1000000000
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import yaml
from tqdm import tqdm

from slm4ie.data.io_utils import (
    DEFAULT_MAX_SHARD_BYTES,
    ShardedJsonlWriter,
    find_dataset_files as _find_dataset_files,
    find_project_root as _find_project_root,
    iter_joined_records as _iter_joined_records,
    resolve_processed_dir as _resolve_processed_dir,
)
from slm4ie.data.parallel import (
    configure_script_logging,
    cpu_default,
    resolve_workers,
    run_parallel,
    workers_quiet,
)

logger = logging.getLogger(__name__)

#: Output keys that must not be shadowed by flattened metadata fields.
RESERVED_OUT_KEYS: Set[str] = {
    "text", "id", "dataset", "domain", "doc_id", "annotations",
}

#: Marker file written into a per-dataset shard folder once conversion
#: finishes cleanly. Its presence is the only signal future runs use to
#: decide whether output is complete; absence (with shards present)
#: means a prior run crashed and the folder must be re-converted.
SENTINEL_NAME: str = ".complete"


def convert_record(
    record: Dict[str, Any],
    collisions: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Convert one joined record to the datatrove output shape.

    Args:
        record (Dict[str, Any]): Input record from
            `iter_joined_records` (text plus optional annotations).
            Must carry a non-empty `uid`; the extraction pipeline
            (`scripts/data/extract.py`) always sets one, derived from
            `<source>:<doc_id>`.
        collisions (Optional[Set[str]]): Mutable set used by the
            caller to deduplicate "metadata key shadows reserved
            field" warnings across a stream. Pass None when calling
            for a single record.

    Returns:
        Dict[str, Any]: A flat dict with `text`, `id`, `dataset`,
            `domain`, optional `doc_id`, optional `annotations`,
            and any flattened metadata entries.

    Raises:
        KeyError: If `uid` is missing or empty. This indicates the
            input was not produced by `extract.py`; re-run extraction
            so each record carries a stable, source-prefixed id.
    """
    source = record["source"]
    uid = record.get("uid")
    if not uid:
        raise KeyError(
            f"Record from source {source!r} is missing 'uid'. "
            f"Re-run scripts/data/extract.py to (re)generate "
            f"<key>.jsonl with uid populated."
        )
    out: Dict[str, Any] = {
        "text": record["text"],
        "id": uid,
        "dataset": source,
        "domain": record["domain"],
    }
    if "doc_id" in record:
        out["doc_id"] = record["doc_id"]

    metadata = record.get("metadata") or {}
    for k, v in metadata.items():
        if k in RESERVED_OUT_KEYS:
            renamed = f"meta_{k}"
            if collisions is not None and k not in collisions:
                logger.warning(
                    "Metadata key %r collides with reserved output "
                    "field; renaming to %r.",
                    k, renamed,
                )
                collisions.add(k)
            out[renamed] = v
        else:
            out[k] = v

    if "annotations" in record and record["annotations"] is not None:
        out["annotations"] = record["annotations"]

    return out


def convert_stream(
    records: Iterable[Dict[str, Any]],
    writer: ShardedJsonlWriter,
) -> int:
    """Convert each input record and write it through *writer*.

    Args:
        records (Iterable[Dict[str, Any]]): Iterable of joined records
            (text plus optional annotations).
        writer (ShardedJsonlWriter): Active sharded writer that handles
            gzip compression and shard rollover.

    Returns:
        int: Number of records written.
    """
    collisions: Set[str] = set()
    count = 0
    for record in records:
        converted = convert_record(record, collisions)
        writer.write_record(converted)
        count += 1
    return count


def list_datasets_from_config(config_path: Path) -> List[str]:
    """Return the dataset keys declared in `extract.yaml`.

    Args:
        config_path (Path): Path to the extraction YAML config.

    Returns:
        List[str]: Dataset keys in declaration order.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh) or {}
    return list((cfg.get("datasets") or {}).keys())


def convert_dataset(
    key: str,
    processed_dir: Path,
    output_dir: Path,
    force: bool = False,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    include_annotations: bool = False,
) -> Optional[int]:
    """Convert a single dataset, writing `<output_dir>/<key>/<NNNNN>.jsonl.gz`.

    Args:
        key (str): Dataset key.
        processed_dir (Path): Directory containing processed input
            files (`<key>.jsonl` and optional
            `<key>.annotations.jsonl.gz`).
        output_dir (Path): Parent directory for the per-dataset shard
            folders. Created if it does not exist.
        force (bool): When True, overwrite an existing output folder.
            Defaults to False (skip and return 0).
        max_shard_bytes (int): Compressed-byte ceiling per shard.
            `ShardedJsonlWriter` rolls over to a new file when this is
            crossed. Defaults to `DEFAULT_MAX_SHARD_BYTES` (200 MB),
            sized for the curate stage's typical 16–40-way parallelism.
        include_annotations (bool): When True, join the annotations
            sidecar and emit a nested `annotations` field per record.
            Defaults to False; the sidecar is then ignored (skipped,
            no I/O).

    Returns:
        Optional[int]: Number of records written, or None when no
            input file exists for *key* (caller should treat this as
            a skip, not an error). Returns 0 when the output folder
            is already complete (sentinel present) and *force* is
            False. A folder with shards but no sentinel is treated
            as incomplete and re-converted, regardless of *force*.
    """
    pair = _find_dataset_files(processed_dir, key)
    if pair is None:
        logger.warning(
            "No processed input found for dataset %r in %s",
            key, processed_dir,
        )
        return None
    text_path, ann_path = pair
    if not include_annotations:
        ann_path = None

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_folder = output_dir / key
    sentinel = shard_folder / SENTINEL_NAME

    if shard_folder.is_dir():
        if sentinel.exists():
            if not force:
                logger.info(
                    "Skipping %r, output folder already complete: %s "
                    "(use --force to overwrite)",
                    key, shard_folder,
                )
                return 0
            # Forced rebuild of a previously-complete folder: clear
            # the sentinel before any shards drop, so a crash during
            # cleanup leaves the folder visibly incomplete.
            sentinel.unlink()
            for stale in shard_folder.glob("*.jsonl.gz"):
                stale.unlink()
        elif any(shard_folder.glob("*.jsonl.gz")):
            # Shards present without a sentinel ⇒ either a prior run
            # crashed or the folder predates the sentinel scheme. In
            # both cases the on-disk content is not trusted; re-convert.
            logger.warning(
                "Output folder %s contains shards but no %s sentinel; "
                "treating as incomplete and re-converting.",
                shard_folder, SENTINEL_NAME,
            )
            for stale in shard_folder.glob("*.jsonl.gz"):
                stale.unlink()

    logger.info(
        "Converting %s%s → %s/ (max_shard_bytes=%d, annotations=%s)",
        text_path,
        f" + {ann_path}" if ann_path else "",
        shard_folder, max_shard_bytes,
        "on" if include_annotations else "off",
    )
    records = _iter_joined_records(text_path, ann_path)
    progress = tqdm(records, desc=key, unit="doc", disable=workers_quiet())
    with ShardedJsonlWriter(shard_folder, max_shard_bytes=max_shard_bytes) as writer:
        try:
            count = convert_stream(progress, writer)
        finally:
            progress.close()
    # Stamp completion only after the writer's __exit__ has flushed
    # the final gzip footer; any earlier exception leaves the folder
    # visibly incomplete for the next run to recover from.
    sentinel.touch()
    if count == 0:
        # Almost always indicates an upstream problem (empty source,
        # over-aggressive extractor filter, schema mismatch). The
        # sentinel is still written — the run did finish cleanly with
        # zero documents — but the warning makes it visible in --all
        # runs across many datasets.
        logger.warning(
            "Dataset %r produced 0 records — input %s appears empty. "
            "Sentinel still written; the folder is considered complete.",
            key, text_path,
        )
    n_shards = len(list(shard_folder.glob("*.jsonl.gz")))
    logger.info("Wrote %d records across %d shard(s) to %s", count, n_shards, shard_folder)
    return count


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert SLM4IE processed JSONL files into datatrove-"
            "compatible <output_dir>/<key>.jsonl.gz files."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "dataset",
        nargs="?",
        help=(
            "Dataset key (e.g. 'kzb'). Mutually exclusive with --all."
        ),
    )
    target.add_argument(
        "--all",
        action="store_true",
        help="Convert every dataset declared in extract.yaml.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the processed <key>.jsonl files. "
            "Defaults to output_dir from configs/data/extract.yaml."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write <key>.jsonl.gz into. Defaults to "
            "<processed-dir>/datatrove."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to extract.yaml (default: configs/data/extract.yaml)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing <key>/ shard folders.",
    )
    parser.add_argument(
        "--include-annotations",
        action="store_true",
        help=(
            "Join the annotations sidecar and emit a nested "
            "'annotations' field per record. Off by default because "
            "annotations are positionally aligned to the original "
            "text and become stale after any datatrove step that "
            "rewrites it. When set (and --output-dir is not "
            "overridden) the default output folder switches to "
            "<processed-dir>/datatrove_annotated/ so the cheap and "
            "annotated shard sets can coexist."
        ),
    )
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        default=DEFAULT_MAX_SHARD_BYTES,
        help=(
            "Compressed-byte ceiling per output shard. The writer "
            "rolls over to <key>/00001.jsonl.gz, 00002.jsonl.gz, ... "
            "as soon as the current shard reaches this size. Default: "
            "%(default)d bytes (~200 MB), sized for 16–40-way curate "
            "parallelism."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help=(
            "Process datasets in parallel. 0=auto (cpu_count // 2), "
            "1=serial, N=N workers. Capped at the number of datasets."
        ),
    )
    return parser.parse_args(argv)


def main():
    """Run the conversion from CLI arguments."""
    args = parse_args()
    project_root = _find_project_root()

    config_path = (
        args.config
        if args.config
        else project_root / "configs" / "data" / "extract.yaml"
    )
    processed_dir = _resolve_processed_dir(config_path, args.processed_dir)
    default_output_name = (
        "datatrove_annotated" if args.include_annotations else "datatrove"
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else processed_dir / default_output_name
    )

    if args.all:
        keys = list_datasets_from_config(config_path)
    else:
        keys = [args.dataset]

    workers = resolve_workers(args.max_workers, len(keys), cpu_default(len(keys)))
    configure_script_logging(parallel=workers > 1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = project_root / "logs" / Path(__file__).stem / stamp

    if args.include_annotations and args.output_dir is not None:
        # The default sibling-folder rule (datatrove/ vs datatrove_annotated/)
        # only kicks in when --output-dir is omitted. Surface this so users
        # who pass both flags don't expect the auto-redirect.
        logger.info(
            "Writing annotated shards to user-specified --output-dir %s; "
            "the default sibling 'datatrove_annotated/' folder is bypassed.",
            output_dir,
        )

    def kwargs_for(_key: str) -> Dict[str, Any]:
        return {
            "processed_dir": processed_dir,
            "output_dir": output_dir,
            "force": args.force,
            "max_shard_bytes": args.max_shard_bytes,
            "include_annotations": args.include_annotations,
        }

    results, failures = run_parallel(
        convert_dataset,
        keys,
        max_workers=workers,
        desc="datatrove",
        pool="process",
        kwargs_for=kwargs_for,
        log_dir=log_dir,
    )

    skipped: List[str] = [k for k, v in results.items() if v is None]
    total = sum(v for v in results.values() if v is not None)

    logger.info(
        "Done. Converted %d dataset(s), %d records total. "
        "Skipped: %s. Failed: %s",
        len(results) - len(skipped),
        total,
        skipped or "none",
        [k for k, _ in failures] or "none",
    )
    if failures:
        sys.exit(2)
    if not args.all and skipped:
        sys.exit(1)


if __name__ == "__main__":
    main()
