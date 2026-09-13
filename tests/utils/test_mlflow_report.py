"""Tests for slm4ie/utils/mlflow_report.py read-side helpers."""

import json
from pathlib import Path

import pytest

pytest.importorskip("mlflow")
pytest.importorskip("pandas")

import mlflow  # noqa: E402

from slm4ie.utils import mlflow_report as mr  # noqa: E402


@pytest.fixture
def store(tmp_path: Path, monkeypatch):
    """Point tracking at a throwaway SQLite store and seed data-pipeline runs."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

    mlflow.set_experiment(mr.PRETRAIN_EXPERIMENT)
    with mlflow.start_run(run_name="pretrain-build"):
        for step, remaining in enumerate([100, 90, 85, 80, 78, 70, 65, 65]):
            mlflow.log_metric("docs_remaining", remaining, step=step)
        mlflow.log_metrics(
            {
                "final_words__ssj500k": 6000,
                "share_of_total_words__ssj500k": 0.6,
                "final_words__fineweb2": 4000,
                "share_of_total_words__fineweb2": 0.4,
            }
        )

    mlflow.set_experiment(mr.TASKS_EXPERIMENT)
    with mlflow.start_run(run_name="ner/ssj500k"):
        mlflow.set_tags({"task": "ner", "dataset": "ssj500k", "role": "finetune_and_eval"})
        mlflow.log_param("source_keys", json.dumps(["ssj500k"]))
    with mlflow.start_run(run_name="sentiment/demo"):
        mlflow.set_tags({"task": "sentiment", "dataset": "demo", "role": "held_out"})
        mlflow.log_param("source_keys", json.dumps(["sentinews"]))


class TestFunnel:
    """Tests for the pretrain survival funnel."""

    def test_funnel_ordered_and_labelled(self, store):
        """The funnel is step-ordered and stage-labelled from docs_remaining."""
        funnel = mr.pretrain_funnel()
        assert list(funnel["stage_index"]) == list(range(8))
        assert funnel.iloc[0]["stage"] == "convert"
        assert funnel.iloc[-1]["docs_remaining"] == 65

    def test_missing_experiment_is_empty(self, tmp_path, monkeypatch):
        """An absent experiment yields an empty funnel, not an error."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'empty.db'}")
        assert mr.pretrain_funnel(experiment="nope/x").empty


class TestMixture:
    """Tests for per-dataset mixture shares."""

    def test_shares_sorted_desc(self, store):
        """Shares are returned per dataset, sorted by share descending."""
        shares = mr.mixture_shares()
        assert list(shares["dataset"]) == ["ssj500k", "fineweb2"]
        assert shares.iloc[0]["word_share"] == pytest.approx(0.6)


class TestContamination:
    """Tests for the pretrain-eval contamination cross-reference."""

    def test_overlap_flagged(self, store):
        """A task whose source also feeds pretrain is flagged; others are not."""
        risk = mr.contamination_risk()
        assert list(risk["dataset"]) == ["ssj500k"]  # sentiment/demo (sentinews) does not overlap
        assert risk.iloc[0]["shared_sources"] == "ssj500k"
        assert risk.iloc[0]["role"] == "finetune_and_eval"


class TestReportArtifact:
    """Tests for downloading a JSON report artifact."""

    def test_latest_report_json_round_trip(self, tmp_path, monkeypatch):
        """The latest matching run's JSON artifact is downloaded and parsed."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'r.db'}")
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'r.db'}")
        mlflow.set_experiment("slm4ie/tokenization/test")
        report = tmp_path / "report.json"
        report.write_text(json.dumps({"ok": 1}), encoding="utf-8")
        with mlflow.start_run(run_name="sweep-eval"):
            mlflow.log_artifact(str(report))

        payload = mr.latest_report_json("slm4ie/tokenization/test")
        assert payload == {"ok": 1}
        assert mr.latest_report_json("slm4ie/tokenization/test", run_name="absent") is None
