"""Tests for slm4ie/data/curate/tracking.py pretrain-build tracking."""

import gzip
import json
from pathlib import Path

import pytest

from slm4ie.data.curate import tracking
from slm4ie.data.curate.sentinel import write_dataset_sentinel, write_sentinel
from slm4ie.data.curate.stages import SCOPED_STAGES, STAGE_DIRS, STAGE_NAMES

DATASETS = ("alpha", "beta")


def _build_tree(root: Path) -> None:
    """Build a fake pretrain output tree with sentinels, shards, and stats.

    Scoped stages get per-dataset sentinels with a steadily shrinking survival
    count; corpus stages get a single sentinel; the final corpus gets a real
    gzipped shard so the digest is non-empty, and stats gets an aggregate.json.
    """
    for index, stage in enumerate(STAGE_NAMES):
        folder = root / STAGE_DIRS[stage]
        folder.mkdir(parents=True, exist_ok=True)
        if stage in SCOPED_STAGES:
            for d in DATASETS:
                write_dataset_sentinel(
                    folder,
                    d,
                    config_slice={"s": stage},
                    config_hash_value="h",
                    records_in=100 - index,
                    records_out=100 - index - 1,
                )
        else:
            write_sentinel(
                folder,
                config_slice={"s": stage},
                config_hash_value="h",
                records_in=200 - index,
                records_out=200 - index - 1,
            )

    # Final corpus shard so corpus_digest has content.
    shard = root / STAGE_DIRS["sentence_dedup"] / "alpha" / "000.jsonl.gz"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(gzip.compress(b'{"text": "x"}\n'))

    # Aggregate stats.
    stats_dir = root / STAGE_DIRS["stats"]
    (stats_dir / "per_dataset").mkdir(parents=True, exist_ok=True)
    aggregate = {
        "total_docs": 150,
        "total_words": 9000,
        "by_dataset": {
            "alpha": {"doc_count": 100, "word_count": 6000, "share_of_total_words": 0.667},
            "beta": {"doc_count": 50, "word_count": 3000, "share_of_total_words": 0.333},
        },
    }
    (stats_dir / "aggregate.json").write_text(json.dumps(aggregate), encoding="utf-8")
    (stats_dir / "per_dataset" / "alpha.json").write_text("{}", encoding="utf-8")


class TestFunnel:
    """Tests for reading the per-stage funnel from sentinels."""

    def test_scoped_and_corpus_counts(self, tmp_path: Path):
        """Scoped stages aggregate per-dataset; corpus stages read one sentinel."""
        _build_tree(tmp_path)
        funnel = tracking.build_pretrain_funnel(tmp_path)
        assert [e["stage"] for e in funnel] == list(STAGE_NAMES)

        convert = funnel[0]
        assert convert["scoped"] is True
        assert set(convert["by_dataset"]) == set(DATASETS)
        assert convert["records_out"] == 2 * (100 - 0 - 1)  # summed across datasets

        exact = next(e for e in funnel if e["stage"] == "exact_dedup")
        assert exact["scoped"] is False
        assert exact["by_dataset"] == {}
        assert exact["records_out"] == 200 - 5 - 1

    def test_missing_stage_has_null_counts(self, tmp_path: Path):
        """A stage with no sentinel reports null counts."""
        (tmp_path / STAGE_DIRS["convert"]).mkdir(parents=True)
        funnel = tracking.build_pretrain_funnel(tmp_path)
        assert all(e["records_out"] is None for e in funnel)


class TestFlattenConfig:
    """Tests for config flattening into params."""

    def test_nested_and_list(self):
        """Nested dicts get dotted keys; lists are serialized."""
        flat = tracking.flatten_config({"a": {"b": 1}, "c": [1, 2], "d": "x"})
        assert flat["a.b"] == "1"
        assert flat["c"] == "[1, 2]"
        assert flat["d"] == "x"


class TestLogPretrainRun:
    """Enabled-path logging against a temporary SQLite store."""

    @pytest.fixture
    def store(self, tmp_path: Path, monkeypatch):
        """Point tracking at a throwaway SQLite store."""
        pytest.importorskip("mlflow")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    def test_disabled_and_missing_corpus_return_none(self, tmp_path: Path):
        """Disabled tracking and an absent final corpus are no-ops."""
        assert tracking.log_pretrain_run(tmp_path, {}, enabled=False) is None
        assert tracking.log_pretrain_run(tmp_path, {}, enabled=True) is None

    def test_logs_funnel_scalars_and_lineage(self, store, tmp_path: Path):
        """A run records the step funnel, final scalars, artifacts, and lineage."""
        import mlflow

        out = tmp_path / "pretrain"
        _build_tree(out)
        experiment = "slm4ie/data/test-pretrain"
        digest = tracking.log_pretrain_run(out, {"language": {"k": 1}}, experiment=experiment)
        assert digest and digest.startswith("sha256:")

        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name(experiment)
        run = client.search_runs([exp.experiment_id])[0]

        history = client.get_metric_history(run.info.run_id, "docs_remaining")
        assert sorted(p.step for p in history) == list(range(len(STAGE_NAMES)))
        assert run.data.metrics["final_words"] == 9000
        assert run.data.metrics["share_of_total_words__alpha"] == pytest.approx(0.667)
        assert run.data.params["language.k"] == "1"
        assert [d.dataset.name for d in run.inputs.dataset_inputs] == ["pretrain/05_2_dedup"]
        artifacts = [f.path for f in client.list_artifacts(run.info.run_id)]
        assert "funnel.json" in artifacts and "aggregate.json" in artifacts

    def test_upsert_skips_then_force_replaces(self, store, tmp_path: Path):
        """Re-logging the same digest skips unless force replaces the run."""
        import mlflow

        out = tmp_path / "pretrain"
        _build_tree(out)
        experiment = "slm4ie/data/test-pretrain"
        tracking.log_pretrain_run(out, {}, experiment=experiment)
        tracking.log_pretrain_run(out, {}, experiment=experiment)

        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name(experiment)
        assert len(client.search_runs([exp.experiment_id])) == 1

        tracking.log_pretrain_run(out, {}, experiment=experiment, force=True)
        assert len(client.search_runs([exp.experiment_id])) == 1
