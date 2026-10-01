"""Run the curation pipeline stage by stage, rebuilding only stale units.

The eight stages, and the tree each writes under `<output_dir>/`:

    0. convert        -> 00_convert/
    1. language       -> 01_language/
    2. spam           -> 02_spam/
    3. quality        -> 03_quality/
    4. repetition     -> 04_repetition/
    5. exact_dedup    -> 05_exact_dedup/
    6. sentence_dedup -> 06_sentence_dedup/   (final corpus)
    7. statistics     -> 07_statistics/

`curate` is the argv-free entry point behind `curate_pretraining_corpus.py run`.
For each requested stage it asks `lineage` which units are stale, builds those
in their staging folder through the stage's module (`stages.run_stage`), checks
and promotes them, and finally brings the lock file up to date. Scoped stages
run per config bucket; corpus stages run once over the roster and resume after
a crash.
"""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from slm4ie.data.curate.config import Setup, bucket_keys_by_effective_hash, effective_stage_config, load_setup
from slm4ie.data.curate.config import stage_slice, validate_overrides
from slm4ie.data.curate.lineage import (
    SENTINEL_NAME,
    check_unit,
    corpus_has_input,
    corpus_inputs,
    corpus_reason,
    has_input,
    integrity_failure,
    invalidate_dataset_sentinels,
    legacy_units,
    read_sentinel,
    recover_promotion,
    refresh_lock,
    run_info,
    scoped_reason,
    swap_into_place,
    write_sentinel,
)
from slm4ie.data.curate.paths import CuratePaths, filter_stage_subset, has_stage_output, human_bytes, shard_layout
from slm4ie.data.curate.stages import (
    CORPUS_STAGES,
    SCOPED_STAGES,
    STAGE_NAMES,
    STAGE_VERSIONS,
    StageJob,
    cascade_from,
    is_scoped,
    run_stage,
    upstream_stage,
)
from slm4ie.data.versioning import EMPTY_DIGEST, merge_digests, shard_files

logger = logging.getLogger(__name__)


#: File in a corpus stage folder recording what an unfinished run started with.
PROGRESS_NAME: str = ".in_progress.json"


def _prepare_corpus_stage(paths: CuratePaths, stage: str, config_hash_value: str, tasks: int, inputs: str) -> bool:
    """Resume an unfinished corpus stage, or clear its staging folder for a fresh start.

    A corpus stage builds in its staging folder and is promoted only once it
    finishes and passes its integrity check, so the previous output stays in
    place until then. A run resumes only when the staging folder's progress
    file matches the current config hash, stage version, task count and
    inputs; datatrove then skips the tasks it marked complete. Otherwise the
    staging folder, the stage's logs (with the completion markers) and its
    scratch folder are removed first, so nothing from an earlier run survives.

    Args:
        paths: Resolved curation paths.
        stage: Corpus stage name.
        config_hash_value: Hash of the stage's config slice and roster.
        tasks: Task count this run uses.
        inputs: The stage's input digest and shard layout.

    Returns:
        True when resuming, False when the staging folder was cleared.
    """
    staging = paths.staging_dir(stage)
    progress_file = staging / PROGRESS_NAME
    expected = {
        "config_hash": config_hash_value,
        "stage_version": STAGE_VERSIONS[stage],
        "tasks": tasks,
        "inputs": inputs,
    }
    if progress_file.is_file() and json.loads(progress_file.read_text(encoding="utf-8")) == expected:
        return True
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(paths.logs_dir(stage), ignore_errors=True)
    shutil.rmtree(paths.scratch_dir(stage), ignore_errors=True)
    staging.mkdir(parents=True)
    progress_file.write_text(json.dumps(expected), encoding="utf-8")
    return False


def _starting_input_hint(paths: CuratePaths, stage: str) -> str:
    """Return an ` (input records ~N)` suffix for a stage's start log line.

    The approximate input size is read from the upstream stage's
    sentinel `records_out` field — already on disk from that stage's
    own run, so no shard scan is needed.

    Args:
        paths: Resolved curation paths.
        stage: Stage about to run.

    Returns:
        An ` (input records ~N)` suffix when the upstream stage's
        sentinel is present, or an empty string otherwise (the first
        stage, or an upstream stage that has not completed).
    """
    upstream = upstream_stage(stage)
    if upstream is None:
        return ""
    sentinel = read_sentinel(paths.stage_dir(upstream))
    if sentinel is None:
        return ""
    return f" (input records ~{sentinel.records_out})"


def _extracted_input_summary(input_dir: Path, keys: List[str]) -> Tuple[int, int]:
    """Summarize the convert stage's input without reading record contents.

    The convert stage performs a 1:1 format conversion, so its input
    record count is redundant with the output shard count taken after
    the run. Counting input lines would mean reading every `<key>.jsonl`
    in full; instead this reports the dataset count and the summed
    on-disk size via `stat` only.

    Args:
        input_dir: Directory holding extraction output JSONLs.
        keys: Dataset keys to include.

    Returns:
        Tuple `(num_datasets_present, total_bytes)`: the count of keys
        whose `<key>.jsonl` exists under *input_dir* and the summed
        on-disk size of those files. Keys with no `<key>.jsonl` are
        excluded from both figures (the convert stage reports them as
        skipped).
    """
    present = 0
    total_bytes = 0
    for key in keys:
        path = input_dir / f"{key}.jsonl"
        if not path.exists():
            continue
        present += 1
        total_bytes += path.stat().st_size
    return present, total_bytes


def _resolve_requested_stages(stage: str, run_all: bool) -> Tuple[str, ...]:
    """Resolve which stages a run executes.

    A subset run (`run_all` False) with `--stage all` runs only the
    scoped stages and stops before the corpus stages. With `--all`,
    `all` means every stage. An explicit single stage is returned as-is.

    Args:
        stage: The `--stage` value (a stage name or `"all"`).
        run_all: True when `--all` was passed.

    Returns:
        The stage names to execute, in pipeline order.
    """
    if stage != "all":
        return (stage,)
    return STAGE_NAMES if run_all else SCOPED_STAGES


def _apply_force(output_dir: Path, *, stage: str, run_all: bool, dataset_keys: List[str]) -> None:
    """Apply `--force` per the scoped/corpus force matrix.

    Args:
        output_dir: Curation output root.
        stage: The `--stage` value (`"all"` or a stage name).
        run_all: True when `--all` was passed.
        dataset_keys: Keys in play (the full roster under `--all`, else
            the positional subset).
    """
    paths = CuratePaths(input_folder=output_dir, output_dir=output_dir)

    # Whole-corpus reset only when --all is combined with the default
    # (all) stage. A subset `--stage all` must never wipe other datasets.
    if run_all and stage == "all":
        if output_dir.exists():
            for child in output_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        logger.warning("--force: cleared %s", output_dir)
        return

    # Forcing a corpus stage (only reachable under --all): drop the
    # corpus stage folders (data + sentinel) from that stage downstream.
    if stage in CORPUS_STAGES:
        affected = cascade_from(stage)
        for name in affected:
            shutil.rmtree(paths.stage_dir(name), ignore_errors=True)
        for name in affected:
            shutil.rmtree(paths.staging_dir(name), ignore_errors=True)
            shutil.rmtree(paths.scratch_dir(name), ignore_errors=True)
        logger.warning("--force --stage %s: removed %s", stage, list(affected))
        return

    # Forcing scoped work: drop the requested keys' per-dataset sentinels
    # and shard subfolders for the affected scoped stages, then drop the
    # corpus stages' sentinels (their data is rebuilt on the next --all).
    scoped_affected = SCOPED_STAGES if stage == "all" else tuple(s for s in cascade_from(stage) if is_scoped(s))
    for name in scoped_affected:
        folder = paths.stage_dir(name)
        invalidate_dataset_sentinels(folder, dataset_keys)
        for key in dataset_keys:
            shutil.rmtree(folder / key, ignore_errors=True)
    for name in CORPUS_STAGES:
        (paths.stage_dir(name) / SENTINEL_NAME).unlink(missing_ok=True)
    shutil.rmtree(output_dir / "_partial", ignore_errors=True)
    logger.warning(
        "--force: reset scoped stages %s for %s + corpus sentinels",
        list(scoped_affected),
        dataset_keys,
    )


def _run_scoped_bucket(
    setup: Setup,
    stage: str,
    bucket_keys: List[str],
    effective: Dict[str, Any],
    bucket_hash: str,
    inputs: Dict[str, Tuple[Optional[str], Optional[Dict[str, Any]]]],
    workers: int,
    log_dir: Optional[Path],
    info: Dict[str, Any],
) -> Tuple[int, int]:
    """Build one config bucket of a scoped stage into staging, check it, promote it.

    Every unit is checked before any is promoted, so a failure keeps every
    old output and sentinel in place.

    Args:
        setup: The loaded setup.
        stage: Scoped stage name.
        bucket_keys: Dataset keys sharing one effective config.
        effective: That effective config slice.
        bucket_hash: Its config hash.
        inputs: Per key, the input digest and (convert) source-file descriptions.
        workers: Worker count, for the stage and for scanning.
        log_dir: Per-task log folder (convert only).
        info: Informational sentinel fields.

    Returns:
        The bucket's aggregate `(records_in, records_out)` from the stage runner.

    Raises:
        RuntimeError: If any unit fails its integrity check.
    """
    paths = setup.paths
    staging = paths.staging_dir(stage)
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    upstream = upstream_stage(stage)
    up_dir = paths.stage_dir(upstream) if upstream is not None else None
    # A key whose upstream unit kept no documents is promoted empty, not skipped.
    with_docs = [k for k in bucket_keys if up_dir is None or has_stage_output(up_dir, k)]
    if not with_docs:
        counts = (0, 0)
    view = filter_stage_subset(up_dir, with_docs) if up_dir is not None and with_docs else None
    job = StageJob(
        paths=paths,
        config=effective,
        workers=workers,
        dataset_keys=with_docs,
        output_folder=staging,
        input_view=view,
        log_dir=log_dir,
        stopwords=setup.stopwords,
        spam_assets=setup.spam_assets,
    )
    try:
        if with_docs:
            counts = run_stage(stage, job)
    finally:
        if view is not None:
            shutil.rmtree(view, ignore_errors=True)

    errors: Dict[str, str] = {}
    for key in bucket_keys:
        if up_dir is None:
            in_scan, out_scan, error = check_unit(
                [paths.input_folder / f"{key}.jsonl"], shard_files(staging / key), workers, id_key="uid"
            )
        else:
            in_scan, out_scan, error = check_unit(shard_files(up_dir / key), shard_files(staging / key), workers)
        if error:
            errors[key] = error
            continue
        input_digest, input_files = inputs[key]
        write_sentinel(
            staging / key,
            config_slice=effective,
            config_hash_value=bucket_hash,
            records_in=in_scan.records,
            records_out=out_scan.records,
            stage_version=STAGE_VERSIONS[stage],
            input_digest=input_digest,
            document_digest=out_scan.document_digest,
            input_files=input_files,
            info=info,
        )
    if errors:
        shutil.rmtree(staging, ignore_errors=True)
        raise integrity_failure(stage, errors)
    for key in bucket_keys:
        swap_into_place(staging / key, paths.stage_dir(stage) / key)
    shutil.rmtree(staging, ignore_errors=True)
    return counts


def _curate_scoped(
    setup: Setup, stage: str, dataset_keys: List[str], workers: int, log_dir: Path, info: Dict[str, Any]
) -> None:
    """Rebuild every stale unit of one scoped stage.

    Args:
        setup: The loaded setup.
        stage: Scoped stage name.
        dataset_keys: Keys this run covers.
        workers: Worker count.
        log_dir: Per-task log folder for the convert stage.
        info: Informational sentinel fields.
    """
    paths = setup.paths
    todo: List[str] = []
    inputs: Dict[str, Tuple[Optional[str], Optional[Dict[str, Any]]]] = {}
    no_input: List[str] = []
    for key in dataset_keys:
        recover_promotion(paths.staging_dir(stage) / key, paths.stage_dir(stage) / key)
        # An unmounted extracted tier or an unbuilt upstream must not wipe output.
        if not has_input(paths, stage, key):
            no_input.append(key)
            continue
        reason, digest, files = scoped_reason(setup, stage, key)
        if reason:
            logger.info("[%s] %s: %s", stage, key, reason)
            todo.append(key)
            inputs[key] = (digest, files)
    if no_input:
        logger.info("[%s] skipping %d dataset(s) with no input: %s", stage, len(no_input), ", ".join(no_input))
    if not todo:
        logger.info("[%s] all requested datasets current; skipping.", stage)
        return

    # Datasets sharing an effective config run together in one executor.
    extra = setup.extra(stage)
    buckets = bucket_keys_by_effective_hash(todo, stage, setup.cfg, setup.overrides, extra)
    logger.info("[%s] %d dataset(s) in %d config group(s)", stage, len(todo), len(buckets))
    for bucket_hash, bucket_keys in buckets.items():
        effective = effective_stage_config(setup.cfg, setup.overrides, bucket_keys[0], stage)
        overridden = [k for k in bucket_keys if (setup.overrides.get(k) or {}).get(stage)]
        if overridden:
            logger.info("[%s] override group %s <- %s", stage, overridden, effective)
        if stage == "convert":
            n_datasets, input_bytes = _extracted_input_summary(paths.input_folder, bucket_keys)
            logger.info("[convert] starting (%d dataset(s), %s)", n_datasets, human_bytes(input_bytes))
        else:
            logger.info("[%s] starting%s", stage, _starting_input_hint(paths, stage))
        records_in, records_out = _run_scoped_bucket(
            setup,
            stage,
            bucket_keys,
            effective,
            bucket_hash,
            inputs,
            workers,
            log_dir if stage == "convert" else None,
            info,
        )
        logger.info(
            "[%s] done for %d dataset(s) (bucket records_in=%d, records_out=%d)",
            stage,
            len(bucket_keys),
            records_in,
            records_out,
        )


def _curate_corpus(setup: Setup, stage: str, workers: int, info: Dict[str, Any]) -> None:
    """Rebuild a corpus stage when it is stale, resuming an unfinished build.

    Args:
        setup: The loaded setup.
        stage: Corpus stage name.
        workers: Worker count.
        info: Informational sentinel fields.

    Raises:
        RuntimeError: If a dataset's output fails its integrity check.
    """
    paths = setup.paths
    recover_promotion(paths.staging_dir(stage), paths.stage_dir(stage))
    if not corpus_has_input(setup, stage):
        logger.info("[%s] upstream not built; skipping.", stage)
        return
    input_keys = corpus_inputs(setup, stage)
    reason, input_digest = corpus_reason(setup, stage, input_keys)
    if reason is None:
        logger.info("[%s] sentinel current; skipping.", stage)
        return
    logger.info("[%s] %s", stage, reason)
    expected_hash = setup.expected_hash(stage)
    if not input_keys:
        # Every dataset was filtered out upstream: the stage's output is empty.
        staging = paths.staging_dir(stage)
        shutil.rmtree(staging, ignore_errors=True)
        write_sentinel(
            staging,
            config_slice=stage_slice(stage, setup.cfg),
            config_hash_value=expected_hash,
            records_in=0,
            records_out=0,
            stage_version=STAGE_VERSIONS[stage],
            input_digest=input_digest,
            document_digest=None if stage == "statistics" else EMPTY_DIGEST,
            info=info,
        )
        swap_into_place(staging, paths.stage_dir(stage))
        return
    up_dir = paths.stage_dir(cast(str, upstream_stage(stage)))
    view = filter_stage_subset(up_dir, input_keys, holder=paths.output_dir / "_inputs" / stage)
    # One task per shard caps task memory and keeps a crashed run resumable.
    tasks, shards = shard_layout(view)
    resumed = _prepare_corpus_stage(paths, stage, expected_hash, tasks, f"{input_digest}|{shards}")
    logger.info(
        "[%s] %s%s (tasks=%d, workers=%d)",
        stage,
        "resuming" if resumed else "starting",
        _starting_input_hint(paths, stage),
        tasks,
        workers,
    )
    staging = paths.staging_dir(stage)
    job = StageJob(
        paths=paths,
        config=stage_slice(stage, setup.cfg),
        workers=workers,
        dataset_keys=input_keys,
        output_folder=staging,
        input_view=view,
        tasks=tasks,
        stopwords=setup.stopwords,
        spam_assets=setup.spam_assets,
    )
    records_in, records_out = run_stage(stage, job)

    document_digest: Optional[str] = None
    if stage != "statistics":
        errors: Dict[str, str] = {}
        digests: List[str] = []
        for key in input_keys:
            _, out_scan, error = check_unit(shard_files(up_dir / key), shard_files(staging / key), workers)
            if error:
                errors[key] = error
            digests.append(cast(str, out_scan.document_digest))
        if errors:
            # Resuming would only reproduce the failure, so start fresh next time.
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(paths.logs_dir(stage), ignore_errors=True)
            raise integrity_failure(stage, errors)
        document_digest = merge_digests(digests)
    (staging / PROGRESS_NAME).unlink(missing_ok=True)
    write_sentinel(
        staging,
        config_slice=stage_slice(stage, setup.cfg),
        config_hash_value=expected_hash,
        records_in=records_in,
        records_out=records_out,
        stage_version=STAGE_VERSIONS[stage],
        input_digest=input_digest,
        document_digest=document_digest,
        info=info,
    )
    swap_into_place(staging, paths.stage_dir(stage))
    shutil.rmtree(paths.scratch_dir(stage), ignore_errors=True)
    shutil.rmtree(view, ignore_errors=True)
    logger.info("[%s] done (records_in=%d, records_out=%d)", stage, records_in, records_out)


def curate(
    *,
    datasets: List[str],
    run_all: bool,
    stage: str,
    input_dir: Optional[Path],
    output_dir: Optional[Path],
    force: bool,
    workers: int,
    pretrain_config: Path,
    extract_config: Optional[Path] = None,
    mlflow_enabled: Optional[bool] = None,
) -> None:
    """Run the curation pipeline (argv-free entry point).

    Each requested stage rebuilds only its stale units (see `stale_reason`),
    and the lock file beside the config is brought up to date after a successful run.

    Args:
        datasets: Positional dataset keys. Must be empty when run_all is True.
        run_all: Process every dataset from the extract config.
        stage: `--stage` value (`"all"` or a stage name).
        input_dir: Override for the pretrain config's input_dir, or None.
        output_dir: Override for the pretrain config's output_dir, or None.
        force: Apply the `--force` reset matrix.
        workers: Worker count (0 = auto).
        pretrain_config: Path to the curation config. Required, so an
            experiment's own corpus variant is never assumed away.
        extract_config: Path to extract.yaml, or None for the default.
        mlflow_enabled: Tri-state override for MLflow pretrain tracking. None
            defers to the pretrain config's `mlflow.enabled`; True/False force it
            on/off. Tracking only runs after a full `--all` build.

    Raises:
        ValueError: If `datasets` is non-empty while `run_all` is True.
        RuntimeError: If a unit still carries a legacy sentinel, or a stage's
            output fails its integrity check.
    """
    if run_all and datasets:
        raise ValueError("datasets must be empty when run_all is True")
    setup = load_setup(input_dir, output_dir, pretrain_config, extract_config)
    cfg = setup.cfg
    paths = setup.paths
    output_dir = paths.output_dir
    # Validate overrides up front so a typo fails before any stage runs.
    validate_overrides(setup.overrides, setup.roster)

    dataset_keys = list(setup.roster) if run_all else list(datasets)
    # `workers` is a CPU budget, not an item count. The convert stage caps
    # it at the dataset count itself (run_convert_stage); the datatrove
    # stages use it as their `tasks` rank count, where shards -- far more
    # numerous than datasets -- are the unit of work. So an explicit
    # --max-workers is honored as-is; auto (0) resolves to cpu_count // 2.
    workers = workers if workers > 0 else max(1, (os.cpu_count() or 2) // 2)
    if run_all:
        logger.info("Running on all %d datasets (workers=%d)", len(dataset_keys), workers)
    else:
        logger.info(
            "Running on %d dataset(s): %s (workers=%d)",
            len(dataset_keys),
            ", ".join(dataset_keys),
            workers,
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if force:
        _apply_force(output_dir, stage=stage, run_all=run_all, dataset_keys=dataset_keys)

    legacy = legacy_units(setup, dataset_keys, corpus=run_all)
    if legacy:
        raise RuntimeError(
            f"{len(legacy)} unit(s) lack lineage or a code version (e.g. {legacy[0]}). "
            "Run `curate_pretraining_corpus.py status --adopt` first to adopt them without a rebuild."
        )

    info = run_info(setup.project_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    convert_log_dir = setup.project_root / "logs" / "curate" / stamp / "convert"
    for stage_name in _resolve_requested_stages(stage, run_all):
        if is_scoped(stage_name):
            _curate_scoped(setup, stage_name, dataset_keys, workers, convert_log_dir, info)
        elif run_all:
            _curate_corpus(setup, stage_name, workers, info)
        else:
            logger.warning("[%s] corpus stage requires --all; skipping.", stage_name)
    refresh_lock(setup)

    # Post-hoc MLflow tracking: only after a full build, reflecting the corpus
    # as it now exists on disk (decoupled from which stages ran this time).
    if run_all:
        mlflow_cfg = cfg.get("mlflow") or {}
        enabled = bool(mlflow_cfg.get("enabled", False)) if mlflow_enabled is None else mlflow_enabled
        if enabled:
            from slm4ie.data.curate.tracking import DEFAULT_EXPERIMENT, log_pretrain_run

            log_pretrain_run(
                output_dir,
                cfg,
                enabled=True,
                experiment=mlflow_cfg.get("experiment", DEFAULT_EXPERIMENT),
                tracking_uri=mlflow_cfg.get("tracking_uri"),
                force=force,
            )
