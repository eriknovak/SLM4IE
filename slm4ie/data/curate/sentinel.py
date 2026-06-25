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
from typing import Any, Dict, List, Optional, Tuple

from slm4ie.data.curate.stages import STAGE_DIRS, cascade_from, is_scoped


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
        input_fingerprint: Cheap size+mtime fingerprint of the stage's
            source input file(s), or `None` for stages that do not read
            an external input (only the `convert` stage records one).
            Lets a refreshed input invalidate the stage without hashing
            its contents.
    """

    completed_at: str
    config_hash: str
    config_slice: Dict[str, Any]
    records_in: int
    records_out: int
    input_fingerprint: Optional[str] = None


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
    input_fingerprint: Optional[str] = None,
) -> Path:
    """Write the sentinel JSON file for *stage_folder*.

    Args:
        stage_folder: The stage's output folder (e.g. `<output_dir>/02_quality`).
        config_slice: The config slice the run consumed.
        config_hash_value: Pre-computed hash of *config_slice* (plus
            any extra payload like stopword file contents).
        records_in: Records read.
        records_out: Records written.
        input_fingerprint: Optional size+mtime fingerprint of the stage's
            source input file(s). Recorded so a refreshed input can be
            detected without rehashing its contents. Omitted for stages
            that read no external input.

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
        "input_fingerprint": input_fingerprint,
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
        fingerprint = data.get("input_fingerprint")
        return Sentinel(
            completed_at=str(data.get("completed_at", "")),
            config_hash=str(data.get("config_hash", "")),
            config_slice=dict(data.get("config_slice") or {}),
            records_in=int(data.get("records_in", 0)),
            records_out=int(data.get("records_out", 0)),
            input_fingerprint=str(fingerprint) if fingerprint is not None else None,
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


def write_dataset_sentinel(
    stage_folder: Path,
    dataset: str,
    *,
    config_slice: Dict[str, Any],
    config_hash_value: str,
    records_in: int,
    records_out: int,
    input_fingerprint: Optional[str] = None,
) -> Path:
    """Write a per-dataset `.complete` sentinel for a scoped stage.

    Args:
        stage_folder: The scoped stage's output folder.
        dataset: Dataset key the sentinel covers.
        config_slice: The config slice the run consumed.
        config_hash_value: Pre-computed hash of *config_slice* (the
            dataset roster is intentionally excluded for scoped stages).
        records_in: Records read for this dataset.
        records_out: Records written for this dataset.
        input_fingerprint: Optional size+mtime fingerprint of this
            dataset's source input file(s). Only the `convert` stage
            records one; later scoped stages read regenerated upstream
            output and leave it `None`.

    Returns:
        Path to the written sentinel file.
    """
    return write_sentinel(
        stage_folder / dataset,
        config_slice=config_slice,
        config_hash_value=config_hash_value,
        records_in=records_in,
        records_out=records_out,
        input_fingerprint=input_fingerprint,
    )


def update_dataset_sentinel_counts(
    stage_folder: Path,
    dataset: str,
    *,
    records_in: int,
    records_out: int,
) -> Optional[Path]:
    """Rewrite only the record counts in an existing dataset sentinel.

    Used to backfill per-source counts onto sentinels written before the
    per-source fix (which stamped a shared bucket total into every
    bucket-mate). Every other field — `completed_at`, `config_hash`,
    `config_slice`, `input_fingerprint` — is preserved verbatim, so the
    sentinel stays current against its config and is not treated as a
    fresh run. A dataset with no existing sentinel is left untouched
    rather than fabricated.

    Args:
        stage_folder: The scoped stage's output folder.
        dataset: Dataset key whose sentinel is updated.
        records_in: True per-source records read for this dataset.
        records_out: True per-source records written for this dataset.

    Returns:
        Path to the rewritten sentinel, or `None` when no sentinel
        exists for *dataset*.
    """
    sentinel_path = dataset_sentinel_path(stage_folder, dataset)
    if not sentinel_path.exists():
        return None
    try:
        payload = json.loads(sentinel_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    payload["records_in"] = records_in
    payload["records_out"] = records_out
    tmp_path = sentinel_path.with_suffix(sentinel_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, sentinel_path)
    return sentinel_path


def dataset_sentinel_is_current(
    stage_folder: Path, dataset: str, expected_hash: str
) -> bool:
    """Return True iff *dataset*'s per-dataset sentinel matches *expected_hash*.

    Args:
        stage_folder: The scoped stage's output folder.
        dataset: Dataset key to check.
        expected_hash: The hash recomputed from current config.

    Returns:
        True if the recorded per-dataset hash matches; False otherwise
        (including when the sentinel is missing).
    """
    return sentinel_is_current(stage_folder / dataset, expected_hash)


def invalidate_dataset_sentinels(stage_folder: Path, datasets: List[str]) -> None:
    """Remove the per-dataset sentinels for *datasets* under *stage_folder*.

    Args:
        stage_folder: The scoped stage's output folder.
        datasets: Dataset keys whose sentinels should be removed. Keys
            without a sentinel are ignored.
    """
    for dataset in datasets:
        dataset_sentinel_path(stage_folder, dataset).unlink(missing_ok=True)


def cascade_invalidate_scoped(
    output_dir: Path, stage: str, keys: List[str]
) -> Tuple[str, ...]:
    """Invalidate *stage* and downstream, honoring per-dataset granularity.

    For scoped stages, only the named *keys*' per-dataset sentinels are
    removed; for corpus stages the whole stage-level sentinel is removed.
    A scoped stage's downstream set spans into the corpus stages, so a
    scoped edit also drops the corpus sentinels.

    Args:
        output_dir: The curation output root.
        stage: First stage to invalidate.
        keys: Dataset keys whose scoped-stage sentinels should drop.

    Returns:
        The stage names considered (i.e. `cascade_from(stage)`).
    """
    affected = cascade_from(stage)
    for name in affected:
        stage_folder = output_dir / STAGE_DIRS[name]
        if is_scoped(name):
            invalidate_dataset_sentinels(stage_folder, keys)
        else:
            (stage_folder / SENTINEL_NAME).unlink(missing_ok=True)
    return affected


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
