"""Tests for slm4ie/data/extract_tracking.py."""

import gzip
import json
from pathlib import Path

import pytest

from slm4ie.data import extract_tracking as et


def _write_jsonl(path: Path, records: list) -> None:
    """Write records as a plain JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def _write_sidecar(path: Path, n: int) -> None:
    """Write a gzipped annotations sidecar with n trivial rows."""
    payload = "".join(f'{{"doc_id": "{i}"}}\n' for i in range(n)).encode("utf-8")
    path.write_bytes(gzip.compress(payload))


def _good_record(i: int) -> dict:
    """Return a fully-populated extracted record."""
    return {"text": f"doc {i}", "source": "x", "domain": "web", "doc_id": str(i), "metadata": {"a": 1}}


class TestProfileSource:
    """Tests for single-source profiling."""

    def test_counts_rows_empty_and_missing(self, tmp_path: Path):
        """Rows, empty text, and missing required fields are counted."""
        text = tmp_path / "k.jsonl"
        _write_jsonl(
            text,
            [
                _good_record(0),
                {"text": "   ", "source": "x", "domain": "web", "doc_id": "1", "metadata": {}},
                {"text": "ok", "source": "x", "domain": "web", "metadata": {}},  # missing doc_id
            ],
        )
        profile = et.profile_source("k", text, None, "web")
        assert profile.rows == 3
        assert profile.empty_text == 1
        assert profile.missing_fields["doc_id"] == 1
        assert profile.sidecar_aligned is None

    def test_sidecar_alignment(self, tmp_path: Path):
        """A sidecar row count equal to text rows is reported aligned."""
        text = tmp_path / "k.jsonl"
        _write_jsonl(text, [_good_record(i) for i in range(3)])
        sidecar = tmp_path / "k.annotations.jsonl.gz"
        _write_sidecar(sidecar, 3)
        aligned = et.profile_source("k", text, sidecar, "web")
        assert aligned.has_annotations and aligned.sidecar_aligned is True

        _write_sidecar(sidecar, 2)
        misaligned = et.profile_source("k", text, sidecar, "web")
        assert misaligned.sidecar_aligned is False


class TestBuildProfile:
    """Tests for the corpus-wide profile."""

    def test_present_and_absent_sources(self, tmp_path: Path):
        """Configured sources without output are marked absent."""
        _write_jsonl(tmp_path / "a.jsonl", [_good_record(0), _good_record(1)])
        datasets = {"a": {"domain": "web"}, "b": {"domain": "wiki"}}
        profile = et.build_extraction_profile(tmp_path, datasets)
        assert profile["by_source"]["a"]["present"] is True
        assert profile["by_source"]["b"]["present"] is False
        assert profile["totals"]["rows"] == 2
        assert profile["totals"]["sources_present"] == 1


class TestLogExtractionRun:
    """Enabled-path logging against a temporary SQLite store."""

    @pytest.fixture
    def extracted(self, tmp_path: Path) -> Path:
        """Build a small extracted/ tree with one source."""
        out = tmp_path / "extracted"
        _write_jsonl(out / "a.jsonl", [_good_record(i) for i in range(4)])
        return out

    @pytest.fixture
    def store(self, tmp_path: Path, monkeypatch):
        """Point tracking at a throwaway SQLite store."""
        pytest.importorskip("mlflow")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    def test_disabled_returns_none(self, extracted: Path):
        """Disabled tracking is a no-op."""
        assert et.log_extraction_run(extracted, {"a": {"domain": "web"}}, enabled=False) is None

    def test_logs_metrics_artifact_and_lineage(self, store, extracted: Path, tmp_path: Path):
        """A run records per-source metrics, profile artifact, and lineage."""
        import mlflow

        experiment = "slm4ie/data/test-extract"
        digest = et.log_extraction_run(
            extracted,
            {"a": {"domain": "web"}},
            experiment=experiment,
            artifact_dir=tmp_path / "art",
        )
        assert digest and digest.startswith("sha256:")

        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name(experiment)
        run = client.search_runs([exp.experiment_id])[0]
        assert run.data.metrics["rows__a"] == 4
        assert run.data.metrics["rows_total"] == 4
        assert run.data.tags["corpus_digest"] == digest
        assert [d.dataset.name for d in run.inputs.dataset_inputs] == ["extracted"]
        artifacts = [f.path for f in client.list_artifacts(run.info.run_id)]
        assert "profile.json" in artifacts

    def test_upsert_skips_then_force_replaces(self, store, extracted: Path, tmp_path: Path):
        """Re-logging the same digest skips unless force replaces the run."""
        import mlflow

        experiment = "slm4ie/data/test-extract"
        datasets = {"a": {"domain": "web"}}
        et.log_extraction_run(extracted, datasets, experiment=experiment, artifact_dir=tmp_path / "a1")
        et.log_extraction_run(extracted, datasets, experiment=experiment, artifact_dir=tmp_path / "a2")

        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name(experiment)
        assert len(client.search_runs([exp.experiment_id])) == 1

        et.log_extraction_run(extracted, datasets, experiment=experiment, force=True, artifact_dir=tmp_path / "a3")
        assert len(client.search_runs([exp.experiment_id])) == 1
