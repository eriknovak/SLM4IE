"""Tests for slm4ie.finetune.tracking."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
mlflow = pytest.importorskip("mlflow")

from slm4ie.finetune.config import PlannedRun  # noqa: E402
from slm4ie.finetune.tracking import track_finetune  # noqa: E402

# ===============================
# Test Data
# ===============================

EXPERIMENT = "slm4ie/test-finetune"
LABELS = ["entailment", "contradiction"]

# ===============================
# Tests
# ===============================


def test_track_finetune_logs_one_run_with_tags_settings_and_test_score(
    tiny_encoder, roots, settings, fixture_entry, tmp_path, monkeypatch
):
    """A tracked run lands in the store once, tagged, with its learning rate, seed and test score."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    # the artifact root `./mlruns` resolves against the working directory
    monkeypatch.chdir(tmp_path)
    planned = PlannedRun(encoder=tiny_encoder, entry=fixture_entry("nli", LABELS), settings=settings)

    track_finetune(planned, roots, EXPERIMENT, {"tier": "scratch"}, device="cpu")

    client = mlflow.MlflowClient()
    [run] = client.search_runs([client.get_experiment_by_name(EXPERIMENT).experiment_id])
    assert {"tier": "scratch", "language": "sl"}.items() <= run.data.tags.items()
    assert {"branch", "base_commit"} <= run.data.tags.keys()
    assert (run.data.params["learning_rate"], run.data.params["seed"]) == ("0.001", "0")
    assert 0.0 <= run.data.metrics["test_macro_f1"] <= 1.0
