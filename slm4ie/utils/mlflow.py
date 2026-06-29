"""MLflow experiment-tracking helpers shared across the project.

Thin wrappers around MLflow that degrade to no-ops when tracking is disabled or
the `mlflow` package is unavailable, so any pipeline (tokenizer sweep, data
builds, future model training/eval) can call them unconditionally. Conventions
follow the project's MLflow guidance: the tracking URI is read from
`MLFLOW_TRACKING_URI` (overridable via config), experiments are created with an
explicit local `artifact_location`, and run-specific tagging/parent-child
structure is applied by the caller.

Beyond the basic run/param/metric/artifact wrappers, this module provides the
primitives the data-pipeline tracking relies on: step-indexed metrics for
stage funnels (`log_metrics(..., step=...)`), digest-keyed run upsert
(`find_run_by_tag` + `delete_run`), and dataset lineage (`log_dataset_input`).
"""

from __future__ import annotations

import logging
import os
import subprocess
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

#: Last-resort tracking URI when neither config nor `MLFLOW_TRACKING_URI`
#: specifies one. A local SQLite store so tracking never assumes a running
#: server; the actual deployment points at the service via the env var (or a
#: config `tracking_uri`).
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

#: Local artifact root used when creating new experiments, so artifact logging
#: does not depend on a server-internal (container) default path.
DEFAULT_ARTIFACT_LOCATION = "./mlruns/artifacts"


def _import_mlflow() -> Optional[Any]:
    """Import `mlflow` lazily, returning None when unavailable.

    Returns:
        Optional[Any]: The `mlflow` module, or None if it is not installed.
    """
    try:
        import mlflow

        return mlflow
    except ImportError:
        logger.warning("mlflow is not installed; tracking is disabled.")
        return None


def resolve_tracking_uri(configured: Optional[str]) -> str:
    """Resolve the tracking URI from config, environment, or the default.

    Args:
        configured (Optional[str]): URI from the YAML config, if any.

    Returns:
        str: The configured URI, else `MLFLOW_TRACKING_URI`, else the local
            SQLite fallback (`DEFAULT_TRACKING_URI`).
    """
    if configured:
        return configured
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def git_commit() -> Optional[str]:
    """Return the current git commit hash for reproducibility tagging.

    Returns:
        Optional[str]: The HEAD commit hash, or None when unavailable.
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _is_remote_store(uri: str) -> bool:
    """Return whether a tracking URI points at a remote HTTP(S) server.

    Args:
        uri (str): A resolved MLflow tracking URI.

    Returns:
        bool: True for `http://` / `https://` servers, else False (local file
            or database stores).
    """
    return uri.startswith(("http://", "https://"))


def ensure_experiment(
    name: str,
    *,
    tracking_uri: Optional[str] = None,
    artifact_location: str = DEFAULT_ARTIFACT_LOCATION,
) -> bool:
    """Set the active experiment, creating it when absent.

    For a local file or database store the experiment is pinned to a local
    `artifact_location` so logging does not depend on a server-internal default
    path. For a remote HTTP(S) tracking server the artifact root is left unset
    so the server decides it; a client-relative path would otherwise resolve on
    the server and may be unwritable.

    Args:
        name (str): Experiment name (e.g. `slm4ie/tokenization/slovenian`).
        tracking_uri (Optional[str]): URI override; resolved when None.
        artifact_location (str): Local artifact root for a new experiment on a
            local store; ignored for remote servers.

    Returns:
        bool: True when MLflow is available and the experiment is set, else
            False (tracking effectively disabled).
    """
    mlflow = _import_mlflow()
    if mlflow is None:
        return False

    uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(name) is None:
        location = None if _is_remote_store(uri) else artifact_location
        client.create_experiment(name, artifact_location=location)
    mlflow.set_experiment(name)
    return True


@contextmanager
def mlflow_run(
    run_name: str,
    *,
    enabled: bool = True,
    nested: bool = False,
    tags: Optional[Dict[str, Any]] = None,
) -> Iterator[Optional[Any]]:
    """Open an MLflow run, or a no-op context when tracking is off.

    Args:
        run_name (str): Display name for the run.
        enabled (bool): When False, yields None without touching MLflow.
        nested (bool): True for a child run within an active parent.
        tags (Optional[Dict[str, Any]]): Tags applied at run start.

    Yields:
        Optional[Any]: The active run object, or None when disabled.
    """
    mlflow = _import_mlflow() if enabled else None
    if mlflow is None:
        yield None
        return

    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def log_params(params: Dict[str, Any], *, enabled: bool = True) -> None:
    """Log run parameters when an MLflow run is active.

    Args:
        params (Dict[str, Any]): Parameter key/value pairs.
        enabled (bool): When False, does nothing.
    """
    mlflow = _import_mlflow() if enabled else None
    if mlflow is not None and mlflow.active_run() is not None:
        mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], *, step: Optional[int] = None, enabled: bool = True) -> None:
    """Log run metrics when an MLflow run is active.

    Args:
        metrics (Dict[str, float]): Metric key/value pairs.
        step (Optional[int]): Step index for the values. Logging the same
            metric name at successive steps produces a series MLflow charts
            over the step axis, used for per-stage pipeline funnels.
        enabled (bool): When False, does nothing.
    """
    mlflow = _import_mlflow() if enabled else None
    if mlflow is not None and mlflow.active_run() is not None:
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if numeric:
            mlflow.log_metrics(numeric, step=step)


def log_artifact(path: Path, *, enabled: bool = True) -> None:
    """Log a local file as a run artifact when a run is active.

    Args:
        path (Path): File to upload.
        enabled (bool): When False, does nothing.
    """
    mlflow = _import_mlflow() if enabled else None
    if mlflow is not None and mlflow.active_run() is not None:
        mlflow.log_artifact(str(path))


def set_tags(tags: Dict[str, Any], *, enabled: bool = True) -> None:
    """Set tags on the active run when one exists.

    Args:
        tags (Dict[str, Any]): Tag key/value pairs.
        enabled (bool): When False, does nothing.
    """
    mlflow = _import_mlflow() if enabled else None
    if mlflow is not None and mlflow.active_run() is not None:
        mlflow.set_tags(tags)


def find_run_by_tag(
    experiment: str,
    tag_key: str,
    tag_value: str,
    *,
    tracking_uri: Optional[str] = None,
    enabled: bool = True,
) -> Optional[str]:
    """Return the most recent active run in an experiment matching a tag.

    Used to make logging idempotent: a data build is keyed by its corpus
    digest tag, so the logger can detect an existing run for the same build and
    skip (or, with force, replace) it rather than create a duplicate.

    Args:
        experiment (str): Experiment name to search within.
        tag_key (str): Tag key to match (e.g. `corpus_digest`).
        tag_value (str): Tag value to match.
        tracking_uri (Optional[str]): URI override; resolved when None.
        enabled (bool): When False, returns None without touching MLflow.

    Returns:
        Optional[str]: The run id of the latest matching active run, or None
            when none matches, the experiment is absent, or tracking is off.
    """
    mlflow = _import_mlflow() if enabled else None
    if mlflow is None:
        return None

    mlflow.set_tracking_uri(resolve_tracking_uri(tracking_uri))
    client = mlflow.MlflowClient()
    experiment_obj = client.get_experiment_by_name(experiment)
    if experiment_obj is None:
        return None

    runs = client.search_runs(
        [experiment_obj.experiment_id],
        filter_string=f"tags.`{tag_key}` = '{tag_value}'",
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    return runs[0].info.run_id if runs else None


def delete_run(run_id: str, *, tracking_uri: Optional[str] = None, enabled: bool = True) -> None:
    """Delete a run by id, so a digest-keyed upsert can replace it.

    Args:
        run_id (str): Identifier of the run to delete.
        tracking_uri (Optional[str]): URI override; resolved when None.
        enabled (bool): When False, does nothing.
    """
    mlflow = _import_mlflow() if enabled else None
    if mlflow is None:
        return

    mlflow.set_tracking_uri(resolve_tracking_uri(tracking_uri))
    mlflow.MlflowClient().delete_run(run_id)


def log_dataset_input(
    name: str,
    digest: str,
    source: str,
    *,
    context: str = "produced",
    enabled: bool = True,
) -> None:
    """Record a dataset as an input of the active run for lineage.

    Declares a filesystem-backed dataset (identified by `name` + `digest`) on
    the active run. A producer logs the corpus it just built with
    `context="produced"`; a consumer (tokenizer sweep, future training/eval)
    logs the same name + digest with a consumption context so the UI links the
    consumer back to the build that produced the corpus.

    Args:
        name (str): Logical dataset name (e.g. `pretrain/05_2_dedup`).
        digest (str): Content digest identifying this build (see
            `slm4ie.data.curate.corpus_digest`).
        source (str): Filesystem path or URI the dataset lives at.
        context (str): Lineage context label (e.g. `produced`, `training`,
            `eval`).
        enabled (bool): When False, does nothing.
    """
    mlflow = _import_mlflow() if enabled else None
    if mlflow is None or mlflow.active_run() is None:
        return

    from mlflow.data.dataset_source_registry import resolve_dataset_source
    from mlflow.data.meta_dataset import MetaDataset

    # resolve_dataset_source warns when a path is interpretable as more than one
    # local source kind; the resolved LocalArtifactDatasetSource is correct, so
    # silence the otherwise-noisy warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dataset_source = resolve_dataset_source(source)
    dataset = MetaDataset(dataset_source, name=name, digest=digest)
    mlflow.log_input(dataset, context=context)
