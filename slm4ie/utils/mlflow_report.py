"""Read-side helpers that turn tracked runs into report-ready tables.

The reporting counterpart to `slm4ie.utils.mlflow` (the write side): marimo
notebooks and other consumers call these to pull runs, metrics, and artifacts
from the MLflow store and shape them into pandas frames. Keeping the queries
here (rather than inline in a notebook) keeps the notebooks thin and lets the
shaping logic be unit-tested against a temporary store.

Functions degrade gracefully when an experiment is absent or empty, returning
empty frames / None so a report renders a clean "no runs yet" state rather than
raising.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from slm4ie.data.curate import STAGE_NAMES
from slm4ie.utils.mlflow import resolve_tracking_uri

#: Default experiment names for the data-pipeline reports, matching the write
#: side's `slm4ie/data/*` convention.
EXTRACT_EXPERIMENT = "slm4ie/data/extract"
PRETRAIN_EXPERIMENT = "slm4ie/data/pretrain"
TASKS_EXPERIMENT = "slm4ie/data/tasks"


def _client(tracking_uri: Optional[str]):
    """Return an MlflowClient bound to the resolved tracking URI.

    Args:
        tracking_uri: URI override; resolved when None.

    Returns:
        A configured `mlflow.MlflowClient`.
    """
    import mlflow

    uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    return mlflow.MlflowClient()


def _experiment_id(client, experiment: str) -> Optional[str]:
    """Return an experiment's id, or None when it does not exist.

    Args:
        client: An `mlflow.MlflowClient`.
        experiment: Experiment name.

    Returns:
        The experiment id, or None.
    """
    found = client.get_experiment_by_name(experiment)
    return found.experiment_id if found is not None else None


def experiment_run_table(experiment: str, *, tracking_uri: Optional[str] = None) -> pd.DataFrame:
    """Return one row per run in an experiment, with params and metrics.

    Args:
        experiment: Experiment name.
        tracking_uri: URI override; resolved when None.

    Returns:
        A DataFrame with `run_id`, `run_name`, `start_time`, each param, and
        each metric as columns. Empty when the experiment is absent or has no
        runs.
    """
    client = _client(tracking_uri)
    experiment_id = _experiment_id(client, experiment)
    if experiment_id is None:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for run in client.search_runs([experiment_id], max_results=1000):
        row: Dict[str, Any] = {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName"),
            "start_time": run.info.start_time,
        }
        row.update(run.data.params)
        row.update(run.data.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def latest_run(
    experiment: str,
    *,
    filter_string: str = "",
    tracking_uri: Optional[str] = None,
):
    """Return the most recent active run in an experiment, or None.

    Args:
        experiment: Experiment name.
        filter_string: Optional MLflow search filter (e.g. a tag match).
        tracking_uri: URI override; resolved when None.

    Returns:
        The latest matching `mlflow.entities.Run`, or None when none matches.
    """
    client = _client(tracking_uri)
    experiment_id = _experiment_id(client, experiment)
    if experiment_id is None:
        return None
    runs = client.search_runs(
        [experiment_id],
        filter_string=filter_string,
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    return runs[0] if runs else None


def fetch_run_json(run_id: str, artifact_path: str, *, tracking_uri: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Download and parse a JSON artifact from a run.

    Args:
        run_id: Run identifier.
        artifact_path: Artifact path within the run (e.g. `report.json`).
        tracking_uri: URI override; resolved when None.

    Returns:
        The parsed JSON object, or None when the artifact is absent.
    """
    client = _client(tracking_uri)
    try:
        local = client.download_artifacts(run_id, artifact_path)
    except (OSError, ValueError):
        return None
    path = Path(local)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def latest_report_json(
    experiment: str,
    *,
    run_name: str = "sweep-eval",
    artifact: str = "report.json",
    tracking_uri: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch the report JSON artifact from an experiment's latest matching run.

    Used by the tokenizer report to read the aggregated `report.json` from the
    MLflow store instead of the local artifact tree.

    Args:
        experiment: Experiment name.
        run_name: Run name to match (the sweep's eval parent run).
        artifact: Artifact path to download.
        tracking_uri: URI override; resolved when None.

    Returns:
        The parsed report, or None when no matching run/artifact exists.
    """
    run = latest_run(experiment, filter_string=f"tags.`mlflow.runName` = '{run_name}'", tracking_uri=tracking_uri)
    if run is None:
        return None
    return fetch_run_json(run.info.run_id, artifact, tracking_uri=tracking_uri)


def pretrain_funnel(*, experiment: str = PRETRAIN_EXPERIMENT, tracking_uri: Optional[str] = None) -> pd.DataFrame:
    """Return the per-stage survival funnel from the latest pretrain build.

    Reads the step-indexed `docs_remaining` metric history and labels each step
    with its stage name.

    Args:
        experiment: Pretrain experiment name.
        tracking_uri: URI override; resolved when None.

    Returns:
        A DataFrame with `stage_index`, `stage`, and `docs_remaining`, ordered
        by stage. Empty when no pretrain run exists.
    """
    run = latest_run(experiment, tracking_uri=tracking_uri)
    if run is None:
        return pd.DataFrame(columns=["stage_index", "stage", "docs_remaining"])
    client = _client(tracking_uri)
    history = client.get_metric_history(run.info.run_id, "docs_remaining")
    rows = [
        {
            "stage_index": point.step,
            "stage": STAGE_NAMES[point.step] if 0 <= point.step < len(STAGE_NAMES) else str(point.step),
            "docs_remaining": point.value,
        }
        for point in sorted(history, key=lambda p: p.step)
    ]
    return pd.DataFrame(rows, columns=["stage_index", "stage", "docs_remaining"])


def mixture_shares(*, experiment: str = PRETRAIN_EXPERIMENT, tracking_uri: Optional[str] = None) -> pd.DataFrame:
    """Return per-dataset word counts and shares from the latest pretrain build.

    Args:
        experiment: Pretrain experiment name.
        tracking_uri: URI override; resolved when None.

    Returns:
        A DataFrame with `dataset`, `words`, and `word_share`, sorted by share
        descending. Empty when no pretrain run exists.
    """
    run = latest_run(experiment, tracking_uri=tracking_uri)
    if run is None:
        return pd.DataFrame(columns=["dataset", "words", "word_share"])
    metrics = run.data.metrics
    datasets = sorted({key.split("__", 1)[1] for key in metrics if key.startswith("share_of_total_words__")})
    rows = [
        {
            "dataset": dataset,
            "words": metrics.get(f"final_words__{dataset}"),
            "word_share": metrics.get(f"share_of_total_words__{dataset}"),
        }
        for dataset in datasets
    ]
    frame = pd.DataFrame(rows, columns=["dataset", "words", "word_share"])
    return frame.sort_values("word_share", ascending=False, ignore_index=True) if not frame.empty else frame


def _pretrain_dataset_names(run) -> set:
    """Return the set of source-dataset names present in a pretrain run.

    Args:
        run: A pretrain `mlflow.entities.Run`.

    Returns:
        Dataset names inferred from the run's per-dataset metric keys.
    """
    names = set()
    for key in run.data.metrics:
        for prefix in ("docs_remaining__", "final_words__", "share_of_total_words__"):
            if key.startswith(prefix):
                names.add(key.split("__", 1)[1])
    return names


def contamination_risk(
    *,
    pretrain_experiment: str = PRETRAIN_EXPERIMENT,
    tasks_experiment: str = TASKS_EXPERIMENT,
    tracking_uri: Optional[str] = None,
) -> pd.DataFrame:
    """Flag task datasets whose sources also feed the pretraining corpus.

    Cross-references each task dataset's `source_keys` against the datasets
    present in the latest pretrain build. Any intersection is a pretrain-eval
    leakage risk (TODO section 4) -- most damaging for `held_out` task splits.

    Args:
        pretrain_experiment: Pretrain experiment name.
        tasks_experiment: Tasks experiment name.
        tracking_uri: URI override; resolved when None.

    Returns:
        A DataFrame with `task`, `dataset`, `role`, and `shared_sources`
        (comma-joined), one row per task dataset with a non-empty overlap.
        Empty when either experiment is missing or nothing overlaps.
    """
    pretrain = latest_run(pretrain_experiment, tracking_uri=tracking_uri)
    if pretrain is None:
        return pd.DataFrame(columns=["task", "dataset", "role", "shared_sources"])
    pretrain_names = _pretrain_dataset_names(pretrain)

    client = _client(tracking_uri)
    tasks_id = _experiment_id(client, tasks_experiment)
    if tasks_id is None:
        return pd.DataFrame(columns=["task", "dataset", "role", "shared_sources"])

    seen = set()
    rows: List[Dict[str, Any]] = []
    for run in client.search_runs([tasks_id], max_results=1000, order_by=["attributes.start_time DESC"]):
        tags = run.data.tags
        key = (tags.get("task"), tags.get("dataset"))
        if key in seen:
            continue
        seen.add(key)
        try:
            source_keys = set(json.loads(run.data.params.get("source_keys", "[]")))
        except json.JSONDecodeError:
            source_keys = set()
        shared = sorted(source_keys & pretrain_names)
        if shared:
            rows.append(
                {
                    "task": tags.get("task"),
                    "dataset": tags.get("dataset"),
                    "role": tags.get("role"),
                    "shared_sources": ", ".join(shared),
                }
            )
    return pd.DataFrame(rows, columns=["task", "dataset", "role", "shared_sources"])
