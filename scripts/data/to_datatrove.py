"""Convert SLM4IE processed JSONL into datatrove-compatible JSONL.

The HuggingFace datatrove library represents documents with three
meaningful fields: text, id, and metadata (a free-form dict). Its
built-in JsonlReader automatically funnels any JSONL keys other than
text/id/media into Document.metadata.

This script reads `<key>.jsonl` (and the optional
`<key>.annotations.jsonl.gz` sidecar) from the extraction output
directory, joins them on the fly, and writes a per-dataset *folder*
of gzipped JSONL shards whose shape is directly consumable by
datatrove while still carrying the dataset/domain provenance and
annotation payloads we need downstream.

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
        "annotations": { "forms": [...], ... }       # if present
    }

Examples:
    # one dataset
    uv run python scripts/data/to_datatrove.py kzb

    # all datasets configured in extract.yaml
    uv run python scripts/data/to_datatrove.py --all

    # tighter shards (e.g. 250 MB) for smaller datasets / more parallelism
    uv run python scripts/data/to_datatrove.py --all --max-shard-bytes 250000000
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


def convert_record(
    record: Dict[str, Any],
    index: int,
    collisions: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Convert one joined record to the datatrove output shape.

    Args:
        record (Dict[str, Any]): Input record from
            `iter_joined_records` (text plus optional annotations).
        index (int): Zero-based record position; used to synthesize a
            fallback `id` when `uid` is missing.
        collisions (Optional[Set[str]]): Mutable set used by the
            caller to deduplicate "metadata key shadows reserved
            field" warnings across a stream. Pass None when calling
            for a single record.

    Returns:
        Dict[str, Any]: A flat dict with `text`, `id`, `dataset`,
            `domain`, optional `doc_id`, optional `annotations`,
            and any flattened metadata entries.
    """
    source = record["source"]
    out: Dict[str, Any] = {
        "text": record["text"],
        "id": record.get("uid") or f"{source}:idx-{index:014d}",
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
    for index, record in enumerate(records):
        converted = convert_record(record, index, collisions)
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
            crossed. Defaults to `DEFAULT_MAX_SHARD_BYTES` (900 MB).

    Returns:
        Optional[int]: Number of records written, or None when no
            input file exists for *key* (caller should treat this as
            a skip, not an error). Returns 0 when the output folder
            already contains shards and *force* is False.
    """
    pair = _find_dataset_files(processed_dir, key)
    if pair is None:
        logger.warning(
            "No processed input found for dataset %r in %s",
            key, processed_dir,
        )
        return None
    text_path, ann_path = pair

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_folder = output_dir / key

    if shard_folder.is_dir() and any(shard_folder.glob("*.jsonl.gz")):
        if not force:
            logger.info(
                "Skipping %r, output folder already populated: %s "
                "(use --force to overwrite)",
                key, shard_folder,
            )
            return 0
        # Force overwrite: drop the previous shards so we don't mix
        # shards from a prior run with shards from this run.
        for stale in shard_folder.glob("*.jsonl.gz"):
            stale.unlink()

    logger.info(
        "Converting %s%s → %s/ (max_shard_bytes=%d)",
        text_path,
        f" + {ann_path}" if ann_path else "",
        shard_folder, max_shard_bytes,
    )
    records = _iter_joined_records(text_path, ann_path)
    progress = tqdm(records, desc=key, unit="doc", disable=workers_quiet())
    with ShardedJsonlWriter(shard_folder, max_shard_bytes=max_shard_bytes) as writer:
        try:
            count = convert_stream(progress, writer)
        finally:
            progress.close()
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
        "--max-shard-bytes",
        type=int,
        default=DEFAULT_MAX_SHARD_BYTES,
        help=(
            "Compressed-byte ceiling per output shard. The writer "
            "rolls over to <key>/00001.jsonl.gz, 00002.jsonl.gz, ... "
            "as soon as the current shard reaches this size. Default: "
            "%(default)d bytes (~900 MB)."
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
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else processed_dir / "datatrove"
    )

    if args.all:
        keys = list_datasets_from_config(config_path)
    else:
        keys = [args.dataset]

    workers = resolve_workers(args.max_workers, len(keys), cpu_default(len(keys)))
    configure_script_logging(parallel=workers > 1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = project_root / "logs" / Path(__file__).stem / stamp

    def kwargs_for(_key: str) -> Dict[str, Any]:
        return {
            "processed_dir": processed_dir,
            "output_dir": output_dir,
            "force": args.force,
            "max_shard_bytes": args.max_shard_bytes,
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
