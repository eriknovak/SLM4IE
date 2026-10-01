"""Per-unit sentinel files for the curate pipeline.

A unit is one scoped stage × one dataset (sentinel at
`<stage_dir>/<dataset>/.complete`) or one corpus stage (sentinel at
`<stage_dir>/.complete`). The sentinel records the unit's full lineage:

* the config hash of the slice that drove it (runtime knobs excluded),
* the stage version, a hash of the stage's code (`stages.STAGE_VERSIONS`),
* the input digest — the upstream units' document digests, or for `convert`
  a content hash of the extracted source files,
* the unit's own document digest and the shard set it wrote,
* record counts, plus informational fields (completion time, git commit,
  datatrove version) that never affect currency.

A unit is current when every lineage field matches and its shards are still
on disk as recorded (`stale_reason`). Upstream reruns that reproduce the same
documents leave the recorded input digest valid, so downstream stays current.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from slm4ie.data.versioning import shard_set


#: Sentinel filename written into every completed unit folder.
SENTINEL_NAME = ".complete"

#: Stale reasons `stale_reason` reports, in the order they are checked.
NOT_BUILT = "not built"
CONFIG_CHANGED = "config changed"
LEGACY = "legacy sentinel (run `status --adopt`)"
INTEGRITY_FAILED = "integrity failure"
STAGE_VERSION_CHANGED = "stage version changed"
INPUT_CHANGED = "input changed"
OUTPUT_CHANGED = "output changed on disk"


@dataclass(frozen=True)
class Sentinel:
    """Parsed contents of a unit's `.complete` sentinel file.

    Attributes:
        completed_at: ISO-8601 UTC timestamp the unit finished.
        config_hash: SHA-256 hex digest of the unit's config slice.
        config_slice: The raw config slice the hash was computed from.
        records_in: Number of records read into the unit.
        records_out: Number of records written out (i.e. surviving).
        stage_version: Code version of the stage that built the unit; `None` marks a
            legacy sentinel written before lineage was recorded.
        input_digest: Digest of the unit's input (see module docstring).
        document_digest: Document digest of what the unit wrote, or `None`
            for a unit that writes no documents (statistics).
        shards: The unit's files, relative path to byte size.
        input_files: `convert` only: per source file `{size, sha256,
            mtime_ns}`, or `None` for an absent file. The mtime only lets a
            later run skip rehashing an untouched file.
        integrity_error: Why the unit failed its integrity check, if it did.
        info: Informational fields that never affect currency.
    """

    completed_at: str
    config_hash: str
    config_slice: Dict[str, Any]
    records_in: int
    records_out: int
    stage_version: Optional[str] = None
    input_digest: Optional[str] = None
    document_digest: Optional[str] = None
    shards: Dict[str, int] = field(default_factory=dict)
    input_files: Optional[Dict[str, Any]] = None
    integrity_error: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_legacy(self) -> bool:
        """Return True for a sentinel written before lineage was recorded."""
        return self.stage_version is None


def _write_payload(sentinel_path: Path, payload: Dict[str, Any]) -> None:
    """Write *payload* as JSON to *sentinel_path* atomically.

    Args:
        sentinel_path: Destination sentinel file.
        payload: JSON-serializable sentinel contents.
    """
    # os.replace is atomic on POSIX, so a crash never leaves a half-written sentinel.
    tmp_path = sentinel_path.with_suffix(sentinel_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, sentinel_path)


def write_sentinel(
    stage_folder: Path,
    *,
    config_slice: Dict[str, Any],
    config_hash_value: str,
    records_in: int,
    records_out: int,
    stage_version: Optional[str] = None,
    input_digest: Optional[str] = None,
    document_digest: Optional[str] = None,
    input_files: Optional[Dict[str, Any]] = None,
    integrity_error: Optional[str] = None,
    info: Optional[Dict[str, Any]] = None,
    completed_at: Optional[str] = None,
) -> Path:
    """Write the sentinel JSON file for the unit in *stage_folder*.

    The shard set is taken from *stage_folder* as it stands, so write the
    sentinel only once the unit's output is complete.

    Args:
        stage_folder: The unit's output folder.
        config_slice: The config slice the run consumed.
        config_hash_value: Pre-computed hash of *config_slice* (plus
            any extra payload like stopword file contents).
        records_in: Records read.
        records_out: Records written.
        stage_version: Code version of the stage that built the unit.
        input_digest: Digest of the unit's input.
        document_digest: Document digest of the unit's output.
        input_files: `convert` only: per source file size and hash.
        integrity_error: Integrity-check failure to record, if any.
        info: Informational fields (git commit, datatrove version).
        completed_at: Completion timestamp to keep; now when omitted.

    Returns:
        Path to the written sentinel file.
    """
    stage_folder.mkdir(parents=True, exist_ok=True)
    sentinel_path = stage_folder / SENTINEL_NAME
    payload = {
        "completed_at": completed_at or datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash_value,
        "config_slice": config_slice,
        "records_in": records_in,
        "records_out": records_out,
        "stage_version": stage_version,
        "input_digest": input_digest,
        "document_digest": document_digest,
        "shards": shard_set(stage_folder),
        "input_files": input_files,
        "integrity_error": integrity_error,
        "info": info or {},
    }
    _write_payload(sentinel_path, payload)
    return sentinel_path


def read_sentinel(stage_folder: Path) -> Optional[Sentinel]:
    """Read the sentinel JSON from *stage_folder*, or return None.

    Args:
        stage_folder: The unit's output folder.

    Returns:
        Parsed `Sentinel` instance, or `None` if the sentinel file is
        missing or malformed.
    """
    sentinel_path = stage_folder / SENTINEL_NAME
    if not sentinel_path.exists():
        return None
    try:
        data = json.loads(sentinel_path.read_text(encoding="utf-8"))
        version = data.get("stage_version")
        return Sentinel(
            completed_at=str(data.get("completed_at", "")),
            config_hash=str(data.get("config_hash", "")),
            config_slice=dict(data.get("config_slice") or {}),
            records_in=int(data.get("records_in", 0)),
            records_out=int(data.get("records_out", 0)),
            stage_version=str(version) if version is not None else None,
            input_digest=data.get("input_digest"),
            document_digest=data.get("document_digest"),
            shards={str(k): int(v) for k, v in (data.get("shards") or {}).items()},
            input_files=data.get("input_files"),
            integrity_error=data.get("integrity_error"),
            info=dict(data.get("info") or {}),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError, AttributeError):
        return None


def stale_reason(
    sentinel: Optional[Sentinel],
    folder: Path,
    *,
    expected_hash: str,
    stage_version: str,
    input_digest: Optional[str],
) -> Optional[str]:
    """Return why a unit must be rebuilt, or `None` when it is current.

    No check reads the unit's documents: the input digest is compared as
    recorded upstream, and the disk check compares file sizes, never mtimes.

    Args:
        sentinel: The unit's parsed sentinel, or `None` when missing.
        folder: The unit's output folder, for the disk check.
        expected_hash: Config hash recomputed from current config.
        stage_version: The stage's current version.
        input_digest: The unit's input digest as it stands now.

    Returns:
        One of the module's reason strings (with detail for an integrity
        failure), or `None`.
    """
    if sentinel is None:
        return NOT_BUILT
    if sentinel.config_hash != expected_hash:
        return CONFIG_CHANGED
    if sentinel.is_legacy:
        return LEGACY
    if sentinel.integrity_error:
        return f"{INTEGRITY_FAILED}: {sentinel.integrity_error}"
    if sentinel.stage_version != stage_version:
        return STAGE_VERSION_CHANGED
    if sentinel.input_digest != input_digest:
        return INPUT_CHANGED
    if shard_set(folder) != sentinel.shards:
        return OUTPUT_CHANGED
    return None


def dataset_sentinel_path(stage_folder: Path, dataset: str) -> Path:
    """Return the per-dataset sentinel path under *stage_folder*.

    Args:
        stage_folder: A scoped stage's output folder (e.g.
            `<output_dir>/02_quality`).
        dataset: Dataset key whose sentinel is addressed.

    Returns:
        Path to `<stage_folder>/<dataset>/.complete`.
    """
    return stage_folder / dataset / SENTINEL_NAME


def write_dataset_sentinel(stage_folder: Path, dataset: str, **fields: Any) -> Path:
    """Write the sentinel of *dataset*'s unit under a scoped stage.

    Args:
        stage_folder: The scoped stage's output folder.
        dataset: Dataset key the sentinel covers.
        **fields: Keyword arguments of `write_sentinel`.

    Returns:
        Path to the written sentinel file.
    """
    return write_sentinel(stage_folder / dataset, **fields)


def invalidate_dataset_sentinels(stage_folder: Path, datasets: List[str]) -> None:
    """Remove the per-dataset sentinels for *datasets* under *stage_folder*.

    Args:
        stage_folder: The scoped stage's output folder.
        datasets: Dataset keys whose sentinels should be removed. Keys
            without a sentinel are ignored.
    """
    for dataset in datasets:
        dataset_sentinel_path(stage_folder, dataset).unlink(missing_ok=True)
