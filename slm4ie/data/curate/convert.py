"""Stage 0 of the curate pipeline: extract → datatrove `Document` shards.

This module owns the conversion previously implemented by the standalone
`scripts/data/to_datatrove.py` script. It reads `<key>.jsonl` (and
optionally its `<key>.annotations.jsonl.gz` sidecar) from the extraction
output directory and writes a per-dataset folder of gzipped JSONL shards
in datatrove's `Document` shape (`text`, `id`, plus arbitrary metadata
fields that datatrove's `JsonlReader` automatically funnels into
`Document.metadata`).

The extraction schema's `source` field is renamed to `dataset` here:
every downstream stage writer routes shards by `${dataset}` and
source-weighted sampling keys on it.

Layout under the stage's output folder (`<output_dir>/00_convert/`):

    00_convert/
    └── <key>/
        ├── 00000.jsonl.gz
        ├── 00001.jsonl.gz
        └── ...

Sharding lets datatrove's `JsonlReader` distribute shards round-robin
across worker ranks, which is the only way to parallelize a single
dataset since gzip streams are not seekable.

By default the annotations sidecar is NOT joined: annotations are
positionally aligned to the original text and become stale after any
downstream datatrove step that rewrites it. Set `include_annotations=True`
to opt in; the parallel-array payload is then emitted as a nested
`annotations` field on each line.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from tqdm import tqdm

from slm4ie.data.io_utils import (
    DEFAULT_MAX_SHARD_BYTES,
    ShardedJsonlWriter,
    find_dataset_files,
    iter_joined_records,
)
from slm4ie.data.parallel import (
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

#: Default field names read from `<key>.jsonl` records.
DEFAULT_TEXT_FIELD: str = "text"
DEFAULT_ID_FIELD: str = "doc_id"
#: `source` is never listed here: it is always renamed to `dataset`
#: (see `convert_record`), independent of `metadata_fields`.
DEFAULT_METADATA_FIELDS: List[str] = ["domain", "doc_id"]


def convert_record(
    record: Dict[str, Any],
    *,
    text_field: str = DEFAULT_TEXT_FIELD,
    id_field: str = DEFAULT_ID_FIELD,
    metadata_fields: Optional[List[str]] = None,
    collisions: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Convert one joined extraction record to the datatrove output shape.

    Args:
        record: Input record from `iter_joined_records` (text plus
            optional annotations). Must carry a non-empty `uid`; the
            extraction pipeline (`scripts/data/extract.py`) always sets
            one, derived from `<source>:<doc_id>`.
        text_field: Source-record field whose value becomes datatrove's
            `text`. Defaults to `"text"`.
        id_field: Source-record field whose value is preserved verbatim
            under that key in the output. The datatrove `id` itself is
            taken from `record["uid"]` (set by the extraction step) so
            it remains globally unique. Defaults to `"doc_id"`.
        metadata_fields: Source-record fields kept verbatim in the
            output (datatrove's `JsonlReader` funnels them into
            `Document.metadata`). Defaults to `["domain", "doc_id"]`.
            `source` is ignored here even if listed: it is always
            renamed to `dataset` (see below).
        collisions: Mutable set used by the caller to deduplicate
            "metadata key shadows reserved field" warnings across a
            stream. Pass None when calling for a single record.

    Returns:
        A flat dict with `text`, `id`, `dataset` (the source-record
        `source` value, renamed), optional preserved fields from
        `metadata_fields`, optional `annotations`, plus any flattened
        free-form metadata entries that came in under `record["metadata"]`.

    Raises:
        KeyError: If `uid` or *text_field* is missing or empty.
    """
    if metadata_fields is None:
        metadata_fields = DEFAULT_METADATA_FIELDS
    source = record.get("source", "<unknown>")
    uid = record.get("uid")
    if not uid:
        raise KeyError(
            f"Record from source {source!r} is missing 'uid'. "
            f"Re-run scripts/data/extract.py to (re)generate "
            f"<key>.jsonl with uid populated."
        )
    if text_field not in record:
        raise KeyError(
            f"Record from source {source!r} is missing the configured "
            f"text field {text_field!r}."
        )

    out: Dict[str, Any] = {
        "text": record[text_field],
        "id": uid,
        # The extraction schema calls the dataset key `source`; the
        # datatrove world calls it `dataset`. Every downstream stage
        # writer routes shards by `${dataset}` and source-weighted
        # sampling keys on it, so the rename happens here, once.
        "dataset": source,
    }
    # Preserve provenance fields verbatim under their original keys so
    # downstream pipeline stages (and source-weighted sampling) can read
    # them off `Document.metadata` after datatrove's reader runs. The
    # configured `id_field` is always kept (even if absent from
    # `metadata_fields`) so the source-document identifier survives.
    kept_fields = list(metadata_fields)
    if id_field not in kept_fields:
        kept_fields.append(id_field)
    for field in kept_fields:
        if field == "id":
            # `id` is reserved by datatrove; skip silently — uid already
            # carries the globally-unique identifier.
            continue
        if field == "source":
            # `source` is special-cased into `dataset` above; never kept
            # verbatim, even if a stale config still lists it.
            continue
        if field == text_field:
            continue
        if field in record:
            out[field] = record[field]

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


def _convert_stream(
    records: Iterable[Dict[str, Any]],
    writer: ShardedJsonlWriter,
    *,
    text_field: str,
    id_field: str,
    metadata_fields: List[str],
) -> int:
    """Convert each input record and write it through *writer*.

    Args:
        records: Iterable of joined records (text plus optional
            annotations).
        writer: Active sharded writer that handles gzip compression and
            shard rollover.
        text_field: See `convert_record`.
        id_field: See `convert_record`.
        metadata_fields: See `convert_record`.

    Returns:
        Number of records written.
    """
    collisions: Set[str] = set()
    count = 0
    for record in records:
        converted = convert_record(
            record,
            text_field=text_field,
            id_field=id_field,
            metadata_fields=metadata_fields,
            collisions=collisions,
        )
        writer.write_record(converted)
        count += 1
    return count


def convert_dataset(
    key: str,
    *,
    input_dir: Path,
    output_dir: Path,
    text_field: str = DEFAULT_TEXT_FIELD,
    id_field: str = DEFAULT_ID_FIELD,
    metadata_fields: Optional[List[str]] = None,
    include_annotations: bool = False,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> Optional[int]:
    """Convert a single dataset's `<key>.jsonl` into datatrove shards.

    Args:
        key: Dataset key.
        input_dir: Directory containing `<key>.jsonl` and optional
            `<key>.annotations.jsonl.gz` sidecars (the extraction
            output directory).
        output_dir: Parent folder for the per-dataset shard folder
            (`<output_dir>/<key>/`). Created if missing.
        text_field: Source-record field copied into datatrove's `text`.
        id_field: Source-record field preserved verbatim alongside
            `uid`; see `convert_record`.
        metadata_fields: Source-record fields kept verbatim in the
            output. Defaults to `["domain", "doc_id"]`; `source` is
            always renamed to `dataset`, never kept verbatim.
        include_annotations: When True, join the annotations sidecar and
            emit a nested `annotations` field per record. Off by default.
        max_shard_bytes: Compressed-byte ceiling per shard.

    Returns:
        Number of records written, or `None` when no `<key>.jsonl` is
        present under *input_dir* (caller treats this as a skip).
    """
    if metadata_fields is None:
        metadata_fields = DEFAULT_METADATA_FIELDS

    pair = find_dataset_files(input_dir, key)
    if pair is None:
        logger.warning(
            "No extraction output found for dataset %r in %s",
            key, input_dir,
        )
        return None
    text_path, ann_path = pair
    if not include_annotations:
        ann_path = None

    shard_folder = output_dir / key
    shard_folder.mkdir(parents=True, exist_ok=True)
    # Caching is owned by the curate sentinel layer; rebuilds drop the
    # whole stage folder upstream, so just clear any stale shards left
    # behind by a crash from a previous run.
    for stale in shard_folder.glob("*.jsonl.gz"):
        stale.unlink()

    logger.info(
        "[convert] %s%s -> %s/ (max_shard_bytes=%d, annotations=%s)",
        text_path,
        f" + {ann_path}" if ann_path else "",
        shard_folder, max_shard_bytes,
        "on" if include_annotations else "off",
    )
    records = iter_joined_records(text_path, ann_path)
    progress = tqdm(records, desc=key, unit="doc", disable=workers_quiet())
    with ShardedJsonlWriter(shard_folder, max_shard_bytes=max_shard_bytes) as writer:
        try:
            count = _convert_stream(
                progress,
                writer,
                text_field=text_field,
                id_field=id_field,
                metadata_fields=metadata_fields,
            )
        finally:
            progress.close()
    if count == 0:
        logger.warning(
            "Dataset %r produced 0 records; input %s appears empty.",
            key, text_path,
        )
    n_shards = len(list(shard_folder.glob("*.jsonl.gz")))
    logger.info(
        "[convert] %s: wrote %d records across %d shard(s) to %s",
        key, count, n_shards, shard_folder,
    )
    return count


def run_convert_stage(
    *,
    input_dir: Path,
    output_dir: Path,
    dataset_keys: List[str],
    text_field: str = DEFAULT_TEXT_FIELD,
    id_field: str = DEFAULT_ID_FIELD,
    metadata_fields: Optional[List[str]] = None,
    include_annotations: bool = False,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    workers: int = 1,
    log_dir: Optional[Path] = None,
) -> Dict[str, Optional[int]]:
    """Run the convert stage over *dataset_keys* with bounded parallelism.

    Args:
        input_dir: Directory holding `<key>.jsonl` files.
        output_dir: Stage output folder (`<output_dir>/00_convert/`).
            Per-key shard subfolders land directly under it.
        dataset_keys: Dataset keys to convert.
        text_field: See `convert_dataset`.
        id_field: See `convert_dataset`.
        metadata_fields: See `convert_dataset`.
        include_annotations: See `convert_dataset`.
        max_shard_bytes: See `convert_dataset`.
        workers: Effective worker count. Use `1` for serial execution,
            `0` for auto (`cpu_count // 2`), `N` for explicit width.
        log_dir: Optional directory for per-dataset log files.

    Returns:
        Mapping `{key: records_written or None}`. `None` indicates no
        input was found for that key (skip).

    Raises:
        RuntimeError: If any per-dataset conversion failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_workers = resolve_workers(
        workers, len(dataset_keys), cpu_default(len(dataset_keys))
    )

    def kwargs_for(_key: str) -> Dict[str, Any]:
        return {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "text_field": text_field,
            "id_field": id_field,
            "metadata_fields": metadata_fields,
            "include_annotations": include_annotations,
            "max_shard_bytes": max_shard_bytes,
        }

    results, failures = run_parallel(
        convert_dataset,
        dataset_keys,
        max_workers=effective_workers,
        desc="convert",
        pool="process",
        kwargs_for=kwargs_for,
        log_dir=log_dir,
    )
    if failures:
        raise RuntimeError(
            "convert stage failed for: "
            + ", ".join(f"{k} ({type(exc).__name__})" for k, exc in failures)
        )
    return results
