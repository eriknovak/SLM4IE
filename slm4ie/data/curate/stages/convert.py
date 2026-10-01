"""Stage 0 of the curate pipeline: extract → datatrove `Document` shards.

This module owns stage 0 of the curation pipeline. It reads `<key>.jsonl` (and
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tqdm import tqdm

from slm4ie.data.curate.stages import StageRun

from slm4ie.data.extract.records import find_dataset_files, iter_joined_records
from slm4ie.utils.io import (
    DEFAULT_MAX_SHARD_BYTES,
    ShardedJsonlWriter,
)
from slm4ie.utils.parallel import (
    cpu_default,
    resolve_workers,
    run_parallel,
    workers_quiet,
)
from slm4ie.utils.versioning import combine_named_digests, file_sha256

logger = logging.getLogger(__name__)

#: Output keys that must not be shadowed by flattened metadata fields.
RESERVED_OUT_KEYS: Set[str] = {
    "text",
    "id",
    "dataset",
    "domain",
    "doc_id",
    "annotations",
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
            extraction pipeline (`prepare_datasets.py extract`) always sets
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
            f"Re-run `prepare_datasets.py extract` to (re)generate "
            f"<key>.jsonl with uid populated."
        )
    if text_field not in record:
        raise KeyError(f"Record from source {source!r} is missing the configured text field {text_field!r}.")

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
                    "Metadata key %r collides with reserved output field; renaming to %r.",
                    k,
                    renamed,
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


def lift_dataset(
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
            key,
            input_dir,
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
        shard_folder,
        max_shard_bytes,
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
            key,
            text_path,
        )
    n_shards = len(list(shard_folder.glob("*.jsonl.gz")))
    logger.info(
        "[convert] %s: wrote %d records across %d shard(s) to %s",
        key,
        count,
        n_shards,
        shard_folder,
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
        text_field: See `lift_dataset`.
        id_field: See `lift_dataset`.
        metadata_fields: See `lift_dataset`.
        include_annotations: See `lift_dataset`.
        max_shard_bytes: See `lift_dataset`.
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
    effective_workers = resolve_workers(workers, len(dataset_keys), cpu_default(len(dataset_keys)))

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
        lift_dataset,
        dataset_keys,
        max_workers=effective_workers,
        desc="convert",
        pool="process",
        kwargs_for=kwargs_for,
        log_dir=log_dir,
    )
    if failures:
        raise RuntimeError(
            "convert stage failed for: " + ", ".join(f"{k} ({type(exc).__name__})" for k, exc in failures)
        )
    return results


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
            [str(f) for f in metadata_fields_raw] if metadata_fields_raw is not None else list(DEFAULT_METADATA_FIELDS)
        ),
        include_annotations=bool(ccfg.get("include_annotations", False)),
        max_shard_bytes=int(ccfg.get("max_shard_bytes", DEFAULT_MAX_SHARD_BYTES)),
    )


def _convert_input_paths(input_dir: Path, key: str, include_annotations: bool) -> List[Path]:
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


def convert_input_files(
    input_dir: Path, key: str, include_annotations: bool, previous: Optional[Dict[str, Any]] = None
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Describe *key*'s convert inputs by size and content hash.

    A file whose size and mtime match *previous* keeps its recorded hash
    instead of being reread: the mtime only decides whether to rehash, never
    whether the input changed, so a touched or copied file is not a change.

    Args:
        input_dir: Root of the extracted tier holding `<key>.jsonl`.
        key: Dataset key.
        include_annotations: Whether convert also joins the annotations sidecar.
        previous: The `input_files` recorded in the unit's last sentinel.

    Returns:
        Per file name `{size, sha256, mtime_ns}`, or `None` for an absent file.
    """
    previous = previous or {}
    files: Dict[str, Optional[Dict[str, Any]]] = {}
    for path in _convert_input_paths(input_dir, key, include_annotations):
        if not path.is_file():
            files[path.name] = None
            continue
        st = path.stat()
        known = previous.get(path.name) or {}
        if known.get("size") == st.st_size and known.get("mtime_ns") == st.st_mtime_ns and known.get("sha256"):
            sha = str(known["sha256"])
        else:
            sha = file_sha256(path)
        files[path.name] = {"size": st.st_size, "sha256": sha, "mtime_ns": st.st_mtime_ns}
    return files


def input_files_digest(files: Dict[str, Optional[Dict[str, Any]]]) -> str:
    """Return the convert input digest: each file's size and content hash, no mtime.

    Args:
        files: Output of `convert_input_files`.

    Returns:
        A `sha256:` digest over the files' names, sizes and content hashes.
    """
    return combine_named_digests({name: f"{f['size']}:{f['sha256']}" if f else None for name, f in files.items()})


def run(job: StageRun) -> Tuple[int, int]:
    """Convert the job's datasets into the job's output folder.

    Args:
        job: What to convert, and where to.

    Returns:
        `(records_in, records_out)`: convert is a 1:1 conversion, so both are
        the records written; datasets with no input contribute nothing.
    """
    params = _build_convert_params(job.config)
    results = run_convert_stage(
        input_dir=job.paths.input_folder,
        output_dir=job.output_folder,
        dataset_keys=job.dataset_keys,
        text_field=params.text_field,
        id_field=params.id_field,
        metadata_fields=params.metadata_fields,
        include_annotations=params.include_annotations,
        max_shard_bytes=params.max_shard_bytes,
        workers=job.workers,
        log_dir=job.log_dir,
    )
    total = sum(n for n in results.values() if n is not None)
    return total, total
