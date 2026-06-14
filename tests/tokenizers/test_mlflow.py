"""Tests for slm4ie/utils/mlflow.py no-op-safe helpers."""

import os
from pathlib import Path

from slm4ie.utils import mlflow as mlflow_utils


class TestResolveTrackingUri:
    """Tests for tracking-URI resolution precedence."""

    def test_configured_wins(self, monkeypatch):
        """An explicit configured URI takes precedence over the environment."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env:5555")
        assert mlflow_utils.resolve_tracking_uri("http://cfg:1") == "http://cfg:1"

    def test_env_used_when_unconfigured(self, monkeypatch):
        """The environment URI is used when none is configured."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env:5555")
        assert mlflow_utils.resolve_tracking_uri(None) == "http://env:5555"

    def test_default_when_absent(self, monkeypatch):
        """The default URI is used when neither config nor env is set."""
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        assert mlflow_utils.resolve_tracking_uri(None) == mlflow_utils.DEFAULT_TRACKING_URI


class TestDisabledIsNoOp:
    """When disabled, helpers must not touch MLflow or raise."""

    def test_run_yields_none(self):
        """A disabled run context yields None."""
        with mlflow_utils.mlflow_run("any", enabled=False) as run:
            assert run is None

    def test_logging_helpers_safe_when_disabled(self, tmp_path: Path):
        """Logging helpers are no-ops when disabled, raising nothing."""
        with mlflow_utils.mlflow_run("any", enabled=False):
            mlflow_utils.log_params({"a": 1}, enabled=False)
            mlflow_utils.log_metrics({"m": 0.5}, enabled=False)
            mlflow_utils.set_tags({"t": "x"}, enabled=False)
            artifact = tmp_path / "f.txt"
            artifact.write_text("x", encoding="utf-8")
            mlflow_utils.log_artifact(artifact, enabled=False)

    def test_git_commit_returns_str_or_none(self):
        """git_commit returns a string in a repo or None outside one."""
        result = mlflow_utils.git_commit()
        assert result is None or isinstance(result, str)


def test_default_artifact_location_is_local():
    """The default artifact location is a local relative path."""
    assert not os.path.isabs(mlflow_utils.DEFAULT_ARTIFACT_LOCATION)
