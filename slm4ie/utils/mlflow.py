"""MLflow experiment-tracking helpers for the tokenizer sweep.

Thin wrappers around MLflow that degrade to no-ops when tracking is disabled or
the `mlflow` package is unavailable, so the training and analysis scripts can
call them unconditionally. Conventions follow the project's MLflow guidance:
the tracking URI is read from `MLFLOW_TRACKING_URI` (overridable via config),
experiments are created with an explicit local `artifact_location`, and
sweep-specific tagging/parent-child structure is applied by the caller.
"""

from __future__ import annotations

import logging
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

#: Fallback tracking URI when neither config nor environment specify one.
DEFAULT_TRACKING_URI = "http://localhost:5555"

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
        str: The configured URI, else `MLFLOW_TRACKING_URI`, else the default.
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


def log_metrics(metrics: Dict[str, float], *, enabled: bool = True) -> None:
    """Log run metrics when an MLflow run is active.

    Args:
        metrics (Dict[str, float]): Metric key/value pairs.
        enabled (bool): When False, does nothing.
    """
    mlflow = _import_mlflow() if enabled else None
    if mlflow is not None and mlflow.active_run() is not None:
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if numeric:
            mlflow.log_metrics(numeric)


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
