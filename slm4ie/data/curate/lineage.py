"""Lineage of every unit: sentinels, currency, the atomic swap, the lock file.

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
on disk as recorded (`stale_reason`, applied per unit by `scoped_reason` and
`corpus_reason`). Upstream reruns that reproduce the same documents leave the
recorded input digest valid, so downstream stays current.

A unit is built in its staging folder, checked (`check_unit`), and promoted by
`swap_into_place`; `recover_promotion` finishes a swap a crash interrupted.
`refresh_lock` mirrors every sentinel into the committed lock file.
"""

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from slm4ie.data.curate.config import Setup
from slm4ie.data.curate.paths import CuratePaths, has_stage_output
from slm4ie.data.curate.stages import STAGE_DIRS, STAGE_NAMES, STAGE_VERSIONS, is_scoped, upstream_stage
from slm4ie.data.curate.stages.convert import convert_input_files, input_files_digest
from slm4ie.utils.versioning import (
    ScanRequest,
    UnitScan,
    check_integrity,
    combine_named_digests,
    scan_units,
    shard_set,
    write_lock,
)

logger = logging.getLogger(__name__)


#: Sentinel filename written into every completed unit folder.
SENTINEL_NAME = ".complete"

#: Stale reasons `stale_reason` reports, in the order they are checked.
NOT_BUILT = "not built"
CONFIG_CHANGED = "config changed"
LEGACY = "legacy sentinel (run `status --adopt`)"
UNVERSIONED = "code version not recorded (run `status --adopt`)"
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

    @property
    def needs_version(self) -> bool:
        """Return True for a sentinel whose stage version predates code hashes."""
        return self.stage_version is not None and not self.stage_version.startswith("sha256:")


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
    if sentinel.needs_version:
        return UNVERSIONED
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


def stamp_stage_version(stage_folder: Path, stage_version: str) -> None:
    """Record *stage_version* in an existing sentinel, leaving every other field as it is.

    Used once, when code versions are first measured: the unit is taken to
    have been built by the code as it stands, the same claim adoption makes.

    Args:
        stage_folder: The unit's output folder.
        stage_version: The stage's current code version.
    """
    sentinel_path = stage_folder / SENTINEL_NAME
    payload = json.loads(sentinel_path.read_text(encoding="utf-8"))
    payload["stage_version"] = stage_version
    _write_payload(sentinel_path, payload)


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


def has_input(paths: CuratePaths, stage: str, key: str) -> bool:
    """Return True if the unit (*stage*, *key*) has an input to be built from.

    Args:
        paths: Resolved curation paths.
        stage: Scoped stage name.
        key: Dataset key.

    Returns:
        For `convert`, whether `<key>.jsonl` exists; otherwise whether the
        upstream unit was built, even if it kept no documents.
    """
    upstream = upstream_stage(stage)
    if upstream is None:
        return (paths.input_folder / f"{key}.jsonl").is_file()
    return read_sentinel(paths.stage_dir(upstream) / key) is not None


def swap_into_place(new: Path, final: Path) -> None:
    """Replace *final* with the finished folder *new* by renaming.

    The old output is renamed aside, the new one renamed in, then the old one
    removed; a crash in between leaves *final* missing (so it is rebuilt),
    never a mix of old and new files.

    Args:
        new: Finished folder, sentinel included.
        final: The unit's canonical output folder.
    """
    old = new.with_name(new.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        os.rename(final, old)
    os.rename(new, final)
    if old.exists():
        shutil.rmtree(old)


def recover_promotion(new: Path, final: Path) -> None:
    """Finish a swap that a crash interrupted between its two renames.

    A staged folder carries a sentinel only once it passed its check, so a
    missing *final* beside a sentinel-bearing *new* is a finished build.

    Args:
        new: The unit's staging folder.
        final: The unit's canonical output folder.
    """
    if not final.exists() and (new / SENTINEL_NAME).is_file():
        logger.warning("promoting %s, finished before an interrupted swap", new)
        swap_into_place(new, final)


#: Comment block written above the lock file's entries.
_LOCK_HEADER = (
    "# Generated by `curate_pretraining_corpus.py run`; do not edit.\n"
    "# One entry per unit: the lineage its sentinel recorded when it was built.\n"
)


def upstream_digest(paths: CuratePaths, stage: str, key: str) -> Optional[str]:
    """Return the document digest the upstream unit of (*stage*, *key*) recorded.

    Args:
        paths: Resolved curation paths.
        stage: A scoped stage after `convert`.
        key: Dataset key.

    Returns:
        The upstream sentinel's document digest, or `None` when it has none.
    """
    sentinel = read_sentinel(paths.stage_dir(cast(str, upstream_stage(stage))) / key)
    return sentinel.document_digest if sentinel is not None else None


def scoped_reason(setup: Setup, stage: str, key: str) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """Judge whether the unit (*stage*, *key*) is current.

    Args:
        setup: The loaded setup.
        stage: Scoped stage name.
        key: Dataset key.

    Returns:
        Tuple `(reason, input_digest, input_files)`: the stale reason or
        `None`, the unit's input digest as it stands now, and for `convert`
        the source-file descriptions (else `None`).
    """
    folder = setup.paths.stage_dir(stage) / key
    sentinel = read_sentinel(folder)
    files: Optional[Dict[str, Any]] = None
    digest: Optional[str] = None
    if stage == "convert":
        # A legacy sentinel is stale whatever its input, so skip hashing the source.
        if sentinel is None or not sentinel.is_legacy:
            previous = sentinel.input_files if sentinel is not None else None
            files = convert_input_files(setup.paths.input_folder, key, setup.include_annotations(key), previous)
            digest = input_files_digest(files)
    else:
        digest = upstream_digest(setup.paths, stage, key)
    reason = stale_reason(
        sentinel,
        folder,
        expected_hash=setup.expected_hash(stage, key),
        stage_version=STAGE_VERSIONS[stage],
        input_digest=digest,
    )
    return reason, digest, files


def corpus_inputs(setup: Setup, stage: str) -> List[str]:
    """Return the roster keys with upstream output that a corpus stage reads.

    Folders left upstream by keys outside the roster (e.g. benchmarks) are
    never read.

    Args:
        setup: The loaded setup.
        stage: Corpus stage name.

    Returns:
        Roster keys, in roster order.
    """
    up_dir = setup.paths.stage_dir(cast(str, upstream_stage(stage)))
    return [key for key in setup.roster if has_stage_output(up_dir, key)]


def corpus_has_input(setup: Setup, stage: str) -> bool:
    """Return True if a corpus stage's upstream has been built, even if empty.

    Args:
        setup: The loaded setup.
        stage: Corpus stage name.

    Returns:
        Whether any roster unit of the upstream scoped stage, or the upstream
        corpus stage, has a sentinel.
    """
    upstream = cast(str, upstream_stage(stage))
    if is_scoped(upstream):
        return any(has_input(setup.paths, stage, key) for key in setup.roster)
    return read_sentinel(setup.paths.stage_dir(upstream)) is not None


def corpus_input_digest(paths: CuratePaths, stage: str, input_keys: List[str]) -> Optional[str]:
    """Return a corpus stage's input digest.

    Args:
        paths: Resolved curation paths.
        stage: Corpus stage name.
        input_keys: The keys it reads (see `corpus_inputs`).

    Returns:
        For the first corpus stage, a combination of each input key's
        upstream document digest; for later ones, the upstream stage's
        document digest (`None` when it has none).
    """
    upstream = cast(str, upstream_stage(stage))
    if is_scoped(upstream):
        return combine_named_digests({key: upstream_digest(paths, stage, key) for key in input_keys})
    sentinel = read_sentinel(paths.stage_dir(upstream))
    return sentinel.document_digest if sentinel is not None else None


def corpus_reason(setup: Setup, stage: str, input_keys: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Judge whether a corpus stage is current.

    Args:
        setup: The loaded setup.
        stage: Corpus stage name.
        input_keys: The keys it reads.

    Returns:
        Tuple `(reason, input_digest)`: the stale reason or `None`, and the
        stage's input digest as it stands now.
    """
    folder = setup.paths.stage_dir(stage)
    digest = corpus_input_digest(setup.paths, stage, input_keys)
    reason = stale_reason(
        read_sentinel(folder),
        folder,
        expected_hash=setup.expected_hash(stage),
        stage_version=STAGE_VERSIONS[stage],
        input_digest=digest,
    )
    return reason, digest


def legacy_units(setup: Setup, keys: List[str], corpus: bool) -> List[str]:
    """List units whose sentinel predates lineage tracking but still matches its config.

    Such units must be adopted before a run; a legacy unit whose config has
    changed is stale anyway and is simply rebuilt.

    Args:
        setup: The loaded setup.
        keys: Dataset keys whose scoped units to check.
        corpus: Also check the corpus stages.

    Returns:
        `<stage folder>/<key>` (or `<stage folder>`) of every such unit.
    """
    found = []
    for stage in STAGE_NAMES:
        units: List[Tuple[str, Path, Optional[str]]]
        if is_scoped(stage):
            units = [(f"{STAGE_DIRS[stage]}/{key}", setup.paths.stage_dir(stage) / key, key) for key in keys]
        elif corpus:
            units = [(STAGE_DIRS[stage], setup.paths.stage_dir(stage), None)]
        else:
            continue
        for name, folder, key in units:
            sentinel = read_sentinel(folder)
            if sentinel is None or not (sentinel.is_legacy or sentinel.needs_version):
                continue
            if sentinel.config_hash == setup.expected_hash(stage, key):
                found.append(name)
    return found


def check_unit(
    in_files: List[Path], out_files: List[Path], workers: int, id_key: str = "id"
) -> Tuple[UnitScan, UnitScan, Optional[str]]:
    """Scan a unit's input and output in one pass and run the integrity check.

    Args:
        in_files: The unit's input files.
        out_files: The unit's output shards.
        workers: Processes to scan with.
        id_key: Id field of the input documents.

    Returns:
        Tuple `(input scan, output scan, integrity error or None)`.
    """
    scans = scan_units({"in": ScanRequest(in_files, id_key, digest=False), "out": ScanRequest(out_files)}, workers)
    in_scan, out_scan = scans["in"], scans["out"]
    return in_scan, out_scan, check_integrity(in_scan, out_scan)


def integrity_failure(stage: str, errors: Dict[str, str]) -> RuntimeError:
    """Return the error raised when a stage's output fails its integrity check.

    Args:
        stage: Stage name.
        errors: Failure description per dataset key.

    Returns:
        A `RuntimeError` naming every failing dataset.
    """
    detail = "; ".join(f"{key}: {error}" for key, error in errors.items())
    return RuntimeError(f"[{stage}] integrity check failed, previous output kept: {detail}")


def lock_entry(sentinel: Sentinel) -> Dict[str, Any]:
    """Return the lock-file entry for one unit's sentinel.

    Args:
        sentinel: The unit's sentinel.

    Returns:
        Its lineage and counts, without informational fields.
    """
    return {
        "config_hash": sentinel.config_hash,
        "stage_version": sentinel.stage_version,
        "input_digest": sentinel.input_digest,
        "document_digest": sentinel.document_digest,
        "records_in": sentinel.records_in,
        "records_out": sentinel.records_out,
    }


def dataset_keys_on_disk(setup: Setup) -> List[str]:
    """Return the roster plus any other dataset converted on disk (e.g. benchmarks run by name).

    Args:
        setup: The loaded setup.

    Returns:
        Roster keys in roster order, then the other converted keys sorted.
    """
    convert_dir = setup.paths.stage_dir("convert")
    on_disk = sorted(d.name for d in convert_dir.iterdir() if d.is_dir()) if convert_dir.is_dir() else []
    return setup.roster + [k for k in on_disk if k not in setup.roster and not k.startswith((".", "_"))]


def refresh_lock(setup: Setup) -> None:
    """Rewrite the lock file from every unit's sentinel on disk, if anything changed.

    The lock mirrors the sentinels, so any run — a dataset subset, a single
    stage, or `--all` — leaves it describing the corpus on disk, and a run that
    rebuilt nothing leaves the file untouched.

    Args:
        setup: The loaded setup.
    """
    entries: Dict[str, Any] = {}
    keys = dataset_keys_on_disk(setup)
    for stage in STAGE_NAMES:
        folder = setup.paths.stage_dir(stage)
        if is_scoped(stage):
            units = {key: read_sentinel(folder / key) for key in keys}
            entries[stage] = {key: lock_entry(s) for key, s in units.items() if s is not None}
        else:
            sentinel = read_sentinel(folder)
            if sentinel is not None:
                entries[stage] = lock_entry(sentinel)
    if write_lock(setup.lock_path, entries, header=_LOCK_HEADER):
        logger.info("wrote %s", setup.lock_path)


def run_info(project_root: Path) -> Dict[str, Any]:
    """Return the informational sentinel fields: git commit and datatrove version.

    Args:
        project_root: Repository root to read the commit from.

    Returns:
        Mapping with `git_commit` and `datatrove_version`, each `None` when
        it cannot be determined.
    """
    try:
        commit: Optional[str] = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.debug("git commit unavailable: %s", exc)
        commit = None
    try:
        datatrove: Optional[str] = package_version("datatrove")
    except PackageNotFoundError:
        datatrove = None
    return {"git_commit": commit, "datatrove_version": datatrove}
