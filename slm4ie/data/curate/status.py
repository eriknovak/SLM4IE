"""Report where the corpus on disk stands, and adopt corpora built before lineage.

`status` judges every unit — current, stale with the reason, or missing — from
sentinels, the extracted files' recorded hashes and the lock file, without
running a stage or reading shards. `adopt_legacy` gives sentinels written
before lineage tracking (or before code versions) their full lineage by
reading each unit once, so a corpus built earlier becomes current without a
rebuild.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from slm4ie.data.curate.config import Setup, load_setup
from slm4ie.data.curate.lineage import (
    NOT_BUILT,
    Sentinel,
    corpus_has_input,
    corpus_input_digest,
    corpus_inputs,
    corpus_reason,
    dataset_keys_on_disk,
    has_input,
    lock_entry,
    read_sentinel,
    run_info,
    scoped_reason,
    stamp_stage_version,
    upstream_digest,
    write_sentinel,
)
from slm4ie.data.curate.paths import human_bytes
from slm4ie.data.curate.stages import (
    CORPUS_STAGES,
    SCOPED_STAGES,
    STAGE_DIRS,
    STAGE_NAMES,
    STAGE_VERSIONS,
    is_scoped,
    upstream_stage,
)
from slm4ie.data.curate.stages.convert import convert_input_files, input_files_digest
from slm4ie.data.versioning import (
    ScanRequest,
    UnitScan,
    check_integrity,
    merge_digests,
    read_lock,
    scan_units,
    shard_files,
)

logger = logging.getLogger(__name__)


#: Status reason for a unit whose sentinel disagrees with the committed lock file.
LOCK_DIFFERS = "differs from lock"


#: Status reason for a built unit whose input is gone; a run leaves it as it is.
NO_INPUT = "no input; output kept"


@dataclass(frozen=True)
class UnitStatus:
    """Whether one unit is current, as `status` reports it.

    Attributes:
        stage: Stage name.
        dataset: Dataset key, or `None` for a corpus stage.
        state: `current`, `stale` (a run would rebuild it) or `missing`
            (nothing to build it from).
        reason: Why the unit is stale, else `None`.
    """

    stage: str
    dataset: Optional[str]
    state: str
    reason: Optional[str] = None


def _unit_status(
    stage: str,
    dataset: Optional[str],
    sentinel: Optional[Sentinel],
    reason: Optional[str],
    buildable: bool,
    locked: Optional[Dict[str, Any]],
    lock_exists: bool,
) -> UnitStatus:
    """Classify one unit from its sentinel, stale reason and lock entry.

    Args:
        stage: Stage name.
        dataset: Dataset key, or `None` for a corpus stage.
        sentinel: The unit's sentinel, or `None`.
        reason: The unit's stale reason, or `None` when current.
        buildable: Whether there is input to build the unit from.
        locked: The unit's lock-file entry, or `None`.
        lock_exists: Whether a lock file exists at all.

    Returns:
        The unit's `UnitStatus`.
    """
    if not buildable:
        return UnitStatus(stage, dataset, "missing", NO_INPUT if sentinel is not None else None)
    if sentinel is None:
        return UnitStatus(stage, dataset, "stale", NOT_BUILT)
    if reason is None and lock_exists and locked != lock_entry(sentinel):
        reason = LOCK_DIFFERS
    return UnitStatus(stage, dataset, "stale" if reason else "current", reason)


def status(
    *,
    input_dir: Optional[Path],
    output_dir: Optional[Path],
    pretrain_config: Path,
    extract_config: Optional[Path] = None,
    adopt: bool = False,
    workers: int = 1,
) -> List[UnitStatus]:
    """Report every unit as current, stale (with the reason) or missing.

    Read-only unless *adopt* is set: no stage runs and no data is written.
    Only `convert` may read data, to rehash an extracted file whose size or
    mtime moved since its sentinel was written.

    Args:
        input_dir: Override for the pretrain config's input_dir, or None.
        output_dir: Override for the pretrain config's output_dir, or None.
        pretrain_config: Path to the curation config.
        extract_config: Path to extract.yaml, or None for the default.
        adopt: First adopt legacy sentinels (see `adopt_legacy`); only
            sentinels are written, the next run brings the lock file up to date.
        workers: Processes used to read shards when adopting.

    Returns:
        One `UnitStatus` per roster unit, in pipeline order.
    """
    setup = load_setup(input_dir, output_dir, pretrain_config, extract_config)
    if adopt:
        adopt_legacy(setup, workers)
    lock = read_lock(setup.lock_path)
    lock_exists = setup.lock_path.is_file()
    paths = setup.paths
    results: List[UnitStatus] = []
    for stage in STAGE_NAMES:
        if is_scoped(stage):
            for key in setup.roster:
                sentinel = read_sentinel(paths.stage_dir(stage) / key)
                buildable = has_input(paths, stage, key)
                reason = scoped_reason(setup, stage, key)[0] if sentinel is not None else None
                locked = (lock.get(stage) or {}).get(key)
                results.append(_unit_status(stage, key, sentinel, reason, buildable, locked, lock_exists))
        else:
            sentinel = read_sentinel(paths.stage_dir(stage))
            reason = corpus_reason(setup, stage, corpus_inputs(setup, stage))[0] if sentinel is not None else None
            buildable = corpus_has_input(setup, stage)
            results.append(_unit_status(stage, None, sentinel, reason, buildable, lock.get(stage), lock_exists))
    return results


def _scan_all(requests: Dict[Tuple[str, str], ScanRequest], workers: int) -> Dict[Tuple[str, str], UnitScan]:
    """Scan every adoption unit with `scan_units`, logging progress by bytes read.

    Args:
        requests: What to scan per `(stage, key)` unit; the pseudo-stage
            `extracted` names a source file.
        workers: Worker processes for parsing and hashing.

    Returns:
        The `UnitScan` per unit.
    """
    files = [f for request in requests.values() for f in request.files]
    total = sum(f.stat().st_size for f in files)
    logger.info(
        "[adopt] scanning %d file(s), %s, across %d unit(s) (workers=%d)",
        len(files),
        human_bytes(total),
        len(requests),
        workers,
    )
    done = {"files": 0, "bytes": 0}

    def progress(path: Path, size: int) -> None:
        done["files"] += 1
        done["bytes"] += size
        logger.info(
            "[adopt] %d/%d files, %s/%s: %s",
            done["files"],
            len(files),
            human_bytes(done["bytes"]),
            human_bytes(total),
            path,
        )

    return scan_units(requests, workers, progress)


def adopt_legacy(setup: Setup, workers: int = 1) -> None:
    """Give legacy sentinels full lineage by reading their outputs once, without a rebuild.

    A legacy unit whose config hash still matches is read once: its document
    digest is computed and its output is checked against its input. Its
    sentinel is then rewritten with lineage, keeping its completion time. A
    unit that fails the check is recorded with the failure, so the next run
    rebuilds it (and, if its documents change, whatever sits downstream).
    Legacy units whose config no longer matches are left alone: they are
    stale either way. Units that already carry lineage but whose stage version
    predates code hashes only get the current code version recorded, without
    a read. No stage runs and no shard is written.

    Args:
        setup: The loaded setup.
        workers: Worker processes for reading shards.
    """
    paths = setup.paths

    def adoptable(sentinel: Optional[Sentinel], expected: str) -> bool:
        return sentinel is not None and sentinel.is_legacy and sentinel.config_hash == expected

    # Keys outside the roster (e.g. benchmarks run by name) are adopted too.
    keys = dataset_keys_on_disk(setup)
    stamped = 0
    folders: List[Tuple[str, Path, Optional[str]]] = [
        (s, paths.stage_dir(s) / k, k) for s in SCOPED_STAGES for k in keys
    ]
    folders += [(s, paths.stage_dir(s), None) for s in CORPUS_STAGES]
    for stage, folder, key in folders:
        sentinel = read_sentinel(folder)
        if sentinel is not None and sentinel.needs_version and sentinel.config_hash == setup.expected_hash(stage, key):
            stamp_stage_version(folder, STAGE_VERSIONS[stage])
            stamped += 1
    if stamped:
        logger.info("[adopt] recorded the current code version on %d unit(s)", stamped)
    scoped = [
        (stage, key)
        for stage in SCOPED_STAGES
        for key in keys
        if adoptable(read_sentinel(paths.stage_dir(stage) / key), setup.expected_hash(stage, key))
    ]
    corpus = [s for s in CORPUS_STAGES if adoptable(read_sentinel(paths.stage_dir(s)), setup.expected_hash(s))]
    if not scoped and not corpus:
        logger.info("[adopt] no legacy sentinels to adopt")
        return

    requests: Dict[Tuple[str, str], ScanRequest] = {}

    def need(stage: str, key: str, digest: bool) -> None:
        if stage == "extracted":
            source = paths.input_folder / f"{key}.jsonl"
            request = ScanRequest([source] if source.is_file() else [], "uid", digest, raw_sha256=True)
        else:
            request = ScanRequest(shard_files(paths.stage_dir(stage) / key), "id", digest)
        known = requests.get((stage, key))
        if known is not None and known.digest:
            request = known
        requests[(stage, key)] = request

    for stage, key in scoped:
        need(stage, key, True)
        need(upstream_stage(stage) or "extracted", key, False)
    for stage in corpus:
        if stage != "statistics":
            for key in corpus_inputs(setup, stage):
                need(stage, key, True)
                need(cast(str, upstream_stage(stage)), key, False)
    scans = _scan_all(requests, workers)
    # Reuse the hash each source file's scan took, so it is not read twice.
    input_files: Dict[str, Dict[str, Any]] = {}
    for stage, key in scoped:
        source = paths.input_folder / f"{key}.jsonl"
        if stage == "convert" and source.is_file():
            st = source.stat()
            known = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": scans[("extracted", key)].raw_sha256}
            previous = {source.name: known}
            input_files[key] = convert_input_files(paths.input_folder, key, setup.include_annotations(key), previous)
        elif stage == "convert":
            input_files[key] = convert_input_files(paths.input_folder, key, setup.include_annotations(key))
    info = {**run_info(setup.project_root), "adopted_at": datetime.now(timezone.utc).isoformat()}

    for stage, key in scoped:
        folder = paths.stage_dir(stage) / key
        legacy = cast(Sentinel, read_sentinel(folder))
        out_scan = scans[(stage, key)]
        upstream = upstream_stage(stage)
        in_scan = scans[(upstream or "extracted", key)]
        files: Optional[Dict[str, Any]] = None
        if upstream is None:
            files = input_files[key]
            input_digest: Optional[str] = input_files_digest(files)
            # Without its source file there is nothing to check the output against.
            error = check_integrity(in_scan, out_scan) if requests[("extracted", key)].files else None
        else:
            input_digest = upstream_digest(paths, stage, key)
            error = check_integrity(in_scan, out_scan)
        write_sentinel(
            folder,
            config_slice=legacy.config_slice,
            config_hash_value=legacy.config_hash,
            records_in=in_scan.records,
            records_out=out_scan.records,
            stage_version=STAGE_VERSIONS[stage],
            input_digest=input_digest,
            document_digest=out_scan.document_digest,
            input_files=files,
            integrity_error=error,
            info=info,
            completed_at=legacy.completed_at,
        )
        logger.info("[adopt] %s/%s: %s", STAGE_DIRS[stage], key, error or "ok")

    for stage in corpus:
        folder = paths.stage_dir(stage)
        legacy = cast(Sentinel, read_sentinel(folder))
        input_keys = corpus_inputs(setup, stage)
        document_digest: Optional[str] = None
        errors: Dict[str, str] = {}
        if stage != "statistics":
            upstream = cast(str, upstream_stage(stage))
            for key in input_keys:
                error = check_integrity(scans[(upstream, key)], scans[(stage, key)])
                if error:
                    errors[key] = error
            document_digest = merge_digests(cast(str, scans[(stage, key)].document_digest) for key in input_keys)
        failure = "; ".join(f"{key}: {error}" for key, error in errors.items()) or None
        write_sentinel(
            folder,
            config_slice=legacy.config_slice,
            config_hash_value=legacy.config_hash,
            records_in=legacy.records_in,
            records_out=legacy.records_out,
            stage_version=STAGE_VERSIONS[stage],
            input_digest=corpus_input_digest(paths, stage, input_keys),
            document_digest=document_digest,
            integrity_error=failure,
            info=info,
            completed_at=legacy.completed_at,
        )
        logger.info("[adopt] %s: %s", STAGE_DIRS[stage], failure or "ok")
