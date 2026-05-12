"""Per-stage sentinel files for the curate pipeline.

Each stage writes a JSON `.complete` sentinel into its output folder
once the underlying datatrove executors finish cleanly. The sentinel
records the config slice that drove the run, a SHA-256 hash of that
slice (so future runs can detect config drift and cascade-invalidate
downstream stages), the completion timestamp, and the input/output
record counts.

Conventions:

* Per-stage sentinel: each `<output_dir>/<stage_dir>/.complete` is
  independent. A stage is "current" if and only if its sentinel exists
  AND its recorded hash matches the freshly computed slice hash.
* Cascade invalidation: when a stage is determined stale, every
  downstream stage's sentinel is removed as well (their inputs are now
  potentially different).
* Runtime-only inputs (CLI flags, worker counts, output paths) MUST NOT
  enter the hash; only output-affecting config does.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from slm4ie.data.curate.stages import STAGE_DIRS, cascade_from


#: Sentinel filename, matching the `to_datatrove.py` convention.
SENTINEL_NAME = ".complete"


@dataclass(frozen=True)
class Sentinel:
    """Parsed contents of a stage's `.complete` sentinel file.

    Attributes:
        completed_at: ISO-8601 UTC timestamp the stage finished.
        config_hash: SHA-256 hex digest of the stage's config slice.
        config_slice: The raw config slice the hash was computed from
            (useful for human inspection).
        records_in: Number of records read into the stage.
        records_out: Number of records written out (i.e. surviving).
    """

    completed_at: str
    config_hash: str
    config_slice: Dict[str, Any]
    records_in: int
    records_out: int


def config_hash(slice_: Dict[str, Any], extra: Optional[bytes] = None) -> str:
    """Compute a stable hash for a config slice.

    The slice is serialized as canonical JSON (sorted keys, no
    whitespace) before hashing. Optional `extra` bytes are appended to
    the hash input — used for hashing the *contents* of files
    referenced by the config (e.g. the stopword file), which are
    output-affecting but not part of the slice itself.

    Args:
        slice_: A JSON-serializable dict of output-affecting config.
        extra: Optional extra bytes to fold into the hash (e.g. the
            stopword file contents).

    Returns:
        Lower-case hex digest prefixed with `"sha256:"`.
    """
    h = hashlib.sha256()
    # default=str coerces non-JSON-native YAML values (datetime, date, etc.)
    # to a stable string form so the hash remains computable.
    h.update(
        json.dumps(slice_, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    )
    if extra is not None:
        # NUL-separate the slice from `extra` so byte boundaries can't collide
        # between (slice ending in null bytes) and (slice + extra=b"...").
        h.update(b"\x00")
        h.update(extra)
    return "sha256:" + h.hexdigest()


def write_sentinel(
    stage_folder: Path,
    *,
    config_slice: Dict[str, Any],
    config_hash_value: str,
    records_in: int,
    records_out: int,
) -> Path:
    """Write the sentinel JSON file for *stage_folder*.

    Args:
        stage_folder: The stage's output folder (e.g. `<output_dir>/02_quality`).
        config_slice: The config slice the run consumed.
        config_hash_value: Pre-computed hash of *config_slice* (plus
            any extra payload like stopword file contents).
        records_in: Records read.
        records_out: Records written.

    Returns:
        Path to the written sentinel file.
    """
    stage_folder.mkdir(parents=True, exist_ok=True)
    sentinel_path = stage_folder / SENTINEL_NAME
    payload = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash_value,
        "config_slice": config_slice,
        "records_in": records_in,
        "records_out": records_out,
    }
    # Atomic write: render the payload to a sibling .tmp file, then rename.
    # os.replace is atomic on POSIX so a partially-written sentinel can never
    # be observed by a concurrent reader (or a future run after a crash).
    tmp_path = sentinel_path.with_suffix(sentinel_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, sentinel_path)
    return sentinel_path


def read_sentinel(stage_folder: Path) -> Optional[Sentinel]:
    """Read the sentinel JSON from *stage_folder*, or return None.

    Args:
        stage_folder: The stage's output folder.

    Returns:
        Parsed `Sentinel` instance, or `None` if the sentinel file is
        missing or malformed.
    """
    sentinel_path = stage_folder / SENTINEL_NAME
    if not sentinel_path.exists():
        return None
    try:
        data = json.loads(sentinel_path.read_text(encoding="utf-8"))
        return Sentinel(
            completed_at=str(data.get("completed_at", "")),
            config_hash=str(data.get("config_hash", "")),
            config_slice=dict(data.get("config_slice") or {}),
            records_in=int(data.get("records_in", 0)),
            records_out=int(data.get("records_out", 0)),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def sentinel_is_current(stage_folder: Path, expected_hash: str) -> bool:
    """Return True iff *stage_folder*'s sentinel exists and matches *expected_hash*.

    Args:
        stage_folder: The stage's output folder.
        expected_hash: The hash recomputed from current config.

    Returns:
        True if the recorded hash matches; False otherwise (including
        when the sentinel is missing).
    """
    sentinel = read_sentinel(stage_folder)
    return sentinel is not None and sentinel.config_hash == expected_hash


def cascade_invalidate(output_dir: Path, stage: str) -> Tuple[str, ...]:
    """Remove the sentinel of *stage* and every downstream stage.

    Args:
        output_dir: The curation output root (which contains the
            `01_language/`, `02_quality/`, ... folders).
        stage: First stage to invalidate. `cascade_from(stage)` is used
            to compute the downstream set.

    Returns:
        Tuple of stage names considered for invalidation (i.e.
        `cascade_from(stage)`), regardless of whether each sentinel
        file existed.
    """
    affected = cascade_from(stage)
    for name in affected:
        sentinel_path = output_dir / STAGE_DIRS[name] / SENTINEL_NAME
        sentinel_path.unlink(missing_ok=True)
    return affected
