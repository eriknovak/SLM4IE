"""Tests for slm4ie/utils/mlflow.py no-op-safe helpers."""

import os
from pathlib import Path

import pytest

from slm4ie.utils import mlflow as mlflow_utils


class TestIsRemoteStore:
    """Tests for distinguishing remote tracking servers from local stores."""

    def test_http_uris_are_remote(self):
        """HTTP(S) tracking URIs are treated as remote servers."""
        assert mlflow_utils._is_remote_store("http://localhost:5555")
        assert mlflow_utils._is_remote_store("https://mlflow.example:443")

    def test_local_uris_are_not_remote(self):
        """File and database stores are treated as local."""
        assert not mlflow_utils._is_remote_store("file:///tmp/mlruns")
        assert not mlflow_utils._is_remote_store("sqlite:///mlflow.db")
        assert not mlflow_utils._is_remote_store("./mlruns")


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


class TestUpsertAndLineage:
    """Enabled-path tests against a temporary local SQLite store."""

    @pytest.fixture
    def store(self, tmp_path: Path, monkeypatch):
        """Point tracking at a throwaway SQLite store for the test."""
        pytest.importorskip("mlflow")
        uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
        return uri

    def test_find_run_by_tag_absent_experiment(self, store):
        """A missing experiment yields no match rather than raising."""
        assert mlflow_utils.find_run_by_tag("nope/x", "corpus_digest", "abc") is None

    def test_find_and_delete_round_trip(self, store):
        """A tagged run is findable by tag, then gone after deletion."""
        experiment = "slm4ie/data/test"
        mlflow_utils.ensure_experiment(experiment)
        with mlflow_utils.mlflow_run("build"):
            mlflow_utils.set_tags({"corpus_digest": "sha256:abc"})

        run_id = mlflow_utils.find_run_by_tag(experiment, "corpus_digest", "sha256:abc")
        assert run_id is not None
        assert mlflow_utils.find_run_by_tag(experiment, "corpus_digest", "other") is None

        mlflow_utils.delete_run(run_id)
        assert mlflow_utils.find_run_by_tag(experiment, "corpus_digest", "sha256:abc") is None

    def test_log_dataset_input_records_lineage(self, store, tmp_path: Path):
        """A logged dataset input round-trips with its name and digest."""
        import mlflow

        mlflow_utils.ensure_experiment("slm4ie/data/test")
        with mlflow_utils.mlflow_run("build") as run:
            mlflow_utils.log_dataset_input(
                "pretrain/05_2_dedup",
                "sha256:abc",
                str(tmp_path),
                context="produced",
            )
            run_id = run.info.run_id

        logged = mlflow.get_run(run_id).inputs.dataset_inputs
        assert [(d.dataset.name, d.dataset.digest) for d in logged] == [("pretrain/05_2_dedup", "sha256:abc")]

    def test_log_dataset_input_truncates_long_digest(self, store, tmp_path: Path):
        """A full sha256 digest is truncated to MLflow's 36-char cap."""
        import mlflow

        long_digest = "sha256:" + "a" * 64
        mlflow_utils.ensure_experiment("slm4ie/data/test")
        with mlflow_utils.mlflow_run("build") as run:
            mlflow_utils.log_dataset_input("corpus", long_digest, str(tmp_path))
            run_id = run.info.run_id

        logged = mlflow.get_run(run_id).inputs.dataset_inputs[0].dataset
        assert logged.digest == long_digest[: mlflow_utils.MAX_DATASET_DIGEST_LEN]
        assert len(logged.digest) <= mlflow_utils.MAX_DATASET_DIGEST_LEN

    def test_log_metrics_step_series(self, store):
        """Step-indexed metrics accumulate a multi-point series."""
        import mlflow

        mlflow_utils.ensure_experiment("slm4ie/data/test")
        with mlflow_utils.mlflow_run("build") as run:
            for step in range(3):
                mlflow_utils.log_metrics({"docs_remaining": 100 - step}, step=step)
            run_id = run.info.run_id

        history = mlflow.MlflowClient().get_metric_history(run_id, "docs_remaining")
        assert [point.step for point in history] == [0, 1, 2]
