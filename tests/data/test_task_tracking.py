"""Tests for slm4ie/data/task_tracking.py."""

import gzip
import json
from pathlib import Path

import pytest

from slm4ie.data import task_tracking as tt
from slm4ie.data.tasks import TaskEntry, TasksConfig, TaskSource, TasksRoots


def _entry(task: str = "sentiment", dataset: str = "demo", splits=None, labels=None) -> TaskEntry:
    """Build a minimal TaskEntry for tests."""
    return TaskEntry(
        task=task,
        dataset=dataset,
        role="finetune_and_eval",
        source=TaskSource(kind="extracted", keys=["demo"]),
        splits=splits or {"test": "test.jsonl.gz"},
        labels=labels,
        suite=None,
        language="sl",
        license="cc",
        converter="to_sentiment",
    )


def _roots(tmp_path: Path) -> TasksRoots:
    """Build TasksRoots pointed at a temp tasks tree."""
    return TasksRoots(extracted=tmp_path / "e", raw=tmp_path / "r", tasks=tmp_path / "tasks")


def _write_split(path: Path, records: list) -> None:
    """Write records to a gzipped JSONL split file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in records).encode("utf-8")))


class TestRecordLabels:
    """Tests for cross-family label extraction."""

    def test_top_level_label(self):
        """A scalar top-level label is extracted."""
        assert tt._record_labels({"label": "positive"}) == ["positive"]

    def test_span_labels(self):
        """Span labels are extracted from dict and triple forms."""
        assert tt._record_labels({"spans": [{"label": "PER"}, [0, 2, "LOC"]]}) == ["PER", "LOC"]

    def test_no_labels(self):
        """A record without labels yields nothing."""
        assert tt._record_labels({"text": "x"}) == []


class TestProfile:
    """Tests for the on-disk dataset profile."""

    def test_counts_and_label_distribution(self, tmp_path: Path):
        """Per-split rows and combined label counts are computed."""
        out = tmp_path / "tasks" / "sentiment" / "demo"
        _write_split(out / "test.jsonl.gz", [{"label": "positive"}, {"label": "negative"}, {"label": "positive"}])
        profile = tt.profile_task_dataset(out, {"test": "test.jsonl.gz", "train": "train.jsonl.gz"})
        assert profile["split_rows"] == {"test": 3}  # train absent on disk
        assert profile["total_rows"] == 3
        assert profile["label_distribution"] == {"positive": 2, "negative": 1}


class TestLogTaskDataset:
    """Enabled-path logging against a temporary SQLite store."""

    @pytest.fixture
    def store(self, tmp_path: Path, monkeypatch):
        """Point tracking at a throwaway SQLite store."""
        pytest.importorskip("mlflow")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    def test_disabled_and_missing_return_none(self, tmp_path: Path):
        """Disabled tracking and an absent dataset are no-ops."""
        entry, roots = _entry(), _roots(tmp_path)
        assert tt.log_task_dataset("sentiment/demo", entry, roots, enabled=False) is None
        assert tt.log_task_dataset("sentiment/demo", entry, roots, enabled=True) is None

    def test_logs_metrics_labels_and_lineage(self, store, tmp_path: Path):
        """A run records split rows, label metrics, lineage, and artifact."""
        import mlflow

        entry, roots = _entry(labels=["negative", "neutral", "positive"]), _roots(tmp_path)
        out = roots.tasks / "sentiment" / "demo"
        _write_split(out / "test.jsonl.gz", [{"label": "positive"}, {"label": "positive"}])
        experiment = "slm4ie/data/test-tasks"

        digest = tt.log_task_dataset("sentiment/demo", entry, roots, experiment=experiment)
        assert digest and digest.startswith("sha256:")

        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name(experiment)
        run = client.search_runs([exp.experiment_id])[0]
        assert run.data.metrics["rows__test"] == 2
        assert run.data.metrics["label__positive"] == 2
        assert run.data.tags["task"] == "sentiment"
        assert [d.dataset.name for d in run.inputs.dataset_inputs] == ["tasks/sentiment/demo"]
        artifacts = [f.path for f in client.list_artifacts(run.info.run_id)]
        assert "label_distribution.json" in artifacts

    def test_upsert_skips_then_force_replaces(self, store, tmp_path: Path):
        """Re-logging the same digest skips unless force replaces the run."""
        import mlflow

        entry, roots = _entry(), _roots(tmp_path)
        out = roots.tasks / "sentiment" / "demo"
        _write_split(out / "test.jsonl.gz", [{"label": "positive"}])
        experiment = "slm4ie/data/test-tasks"

        tt.log_task_dataset("sentiment/demo", entry, roots, experiment=experiment)
        tt.log_task_dataset("sentiment/demo", entry, roots, experiment=experiment)
        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name(experiment)
        assert len(client.search_runs([exp.experiment_id])) == 1

        tt.log_task_dataset("sentiment/demo", entry, roots, experiment=experiment, force=True)
        assert len(client.search_runs([exp.experiment_id])) == 1


class TestLogTaskRuns:
    """Tests for the per-converter batch logging entry point."""

    def test_disabled_by_config_is_noop(self, tmp_path: Path):
        """With config + override both off, nothing is logged."""
        entry, roots = _entry(), _roots(tmp_path)
        config = TasksConfig(roots=roots, converter_defaults={}, entries=[entry], mlflow={"enabled": False})
        # Should simply return without raising even though no store is configured.
        tt.log_task_runs(config, {"sentiment/demo": entry}, ["sentiment/demo"], mlflow_enabled=None)
