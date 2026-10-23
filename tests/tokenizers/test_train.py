"""Tests for slm4ie/tokenizers/train.py orchestration."""

import dataclasses
import gzip
import json
from pathlib import Path

import pytest

from slm4ie.tokenizers.corpus import SampleBudget
from slm4ie.tokenizers.registry import get_tokenizer
from slm4ie.tokenizers.train import (
    MLFLOW_LINK_FILENAME,
    TRAIN_STATS_FILENAME,
    log_train_runs,
    longest_first,
    parse_sweep_run_key,
    plan_sweep_runs,
    prepare_inputs,
    resolve_sweep_runs,
    sweep_run_key,
    select_sweep_runs,
    train_one,
    train_sweep,
)
from slm4ie.tokenizers.config import SweepConfig

_CORPUS_LINES = [
    "hiša stoji ob cesti",
    "hiše so velike in lepe",
    "mačka spi v hiši",
    "velika hiša ob veliki cesti",
] * 40


def _make_config(tmp_path: Path, tokenizers, vocab_sizes) -> SweepConfig:
    """Build a tiny sweep config over a fake corpus under `tmp_path`.

    Args:
        tmp_path (Path): Temp directory root.
        tokenizers (list): Tokenizer names for the sweep.
        vocab_sizes (list): Vocab sizes for the sweep.

    Returns:
        SweepConfig: A config pointing at the fake corpus.
    """
    corpus_root = tmp_path / "dd"
    sub = corpus_root / "macocu_sl"
    sub.mkdir(parents=True)
    with gzip.open(sub / "00000.jsonl.gz", "wt", encoding="utf-8") as handle:
        for i, text in enumerate(_CORPUS_LINES):
            handle.write(json.dumps({"text": text, "id": str(i), "metadata": {"dataset": "macocu_sl"}}) + "\n")

    return SweepConfig(
        corpus_root=corpus_root,
        corpus_datasets=[],
        train_budget=SampleBudget(max_docs=200, seed=1),
        tokenizers=tokenizers,
        vocab_sizes=vocab_sizes,
        special_tokens=["<unk>"],
        sloleks_path=tmp_path / "sloleks.jsonl.gz",
        min_stem_len=2,
        output_root=tmp_path / "out",
        report_dir=tmp_path / "out" / "_reports",
        eval_budget=SampleBudget(max_docs=50, seed=9),
        renyi_alpha=2.5,
        mlflow_experiment="tokenizer/test",
        mlflow_enabled=False,
    )


class TestRunKeys:
    """Tests for run-key helpers and planning."""

    def test_run_key_round_trip(self):
        """sweep_run_key and parse_sweep_run_key are inverses."""
        assert parse_sweep_run_key(sweep_run_key("morphbpe", 16000)) == ("morphbpe", 16000)

    def test_plan_runs_cartesian(self, tmp_path: Path):
        """plan_sweep_runs is the cartesian product in tokenizer-major order."""
        cfg = _make_config(tmp_path, ["bpe", "wordpiece"], [16000, 32000])
        assert plan_sweep_runs(cfg) == ["bpe-16000", "bpe-32000", "wordpiece-16000", "wordpiece-32000"]

    def test_select_runs_filters(self, tmp_path: Path):
        """select_sweep_runs narrows the sweep by tokenizer and vocab filters."""
        cfg = _make_config(tmp_path, ["bpe", "wordpiece"], [16000, 32000])
        assert select_sweep_runs(cfg, tokenizers=["bpe"]) == ["bpe-16000", "bpe-32000"]
        assert select_sweep_runs(cfg, vocab_sizes=[32000]) == ["bpe-32000", "wordpiece-32000"]


class TestTrainOne:
    """End-to-end training of a single library run."""

    def test_trains_and_skips(self, tmp_path: Path):
        """train_one produces an artifact, then skips unless forced."""
        cfg = _make_config(tmp_path, ["bpe"], [90])
        sample_path, lexicon_path = prepare_inputs(cfg)
        assert sample_path.exists()
        assert lexicon_path is None  # no morphological backend requested

        out = train_one("bpe-90", cfg=cfg, sample_path=sample_path, lexicon_path=None)
        assert out is not None
        assert (out / "metadata.json").exists()

        # Second call skips because the artifact exists.
        assert train_one("bpe-90", cfg=cfg, sample_path=sample_path, lexicon_path=None) is None
        # Forcing retrains.
        assert train_one("bpe-90", cfg=cfg, sample_path=sample_path, lexicon_path=None, force=True) is not None

    def test_artifact_loads_and_encodes(self, tmp_path: Path):
        """The saved artifact can be loaded and used to encode."""
        cfg = _make_config(tmp_path, ["bpe"], [90])
        sample_path, _ = prepare_inputs(cfg)
        out = train_one("bpe-90", cfg=cfg, sample_path=sample_path, lexicon_path=None)

        loaded = get_tokenizer("bpe").load(out)
        assert loaded.encode("hiša") != []

    def test_writes_train_stats_sidecar(self, tmp_path: Path):
        """train_one writes a train_stats.json sidecar with timing + sizes."""
        cfg = _make_config(tmp_path, ["bpe"], [90])
        sample_path, _ = prepare_inputs(cfg)
        out = train_one("bpe-90", cfg=cfg, sample_path=sample_path, lexicon_path=None)

        stats = json.loads((out / TRAIN_STATS_FILENAME).read_text(encoding="utf-8"))
        assert stats["run_key"] == "bpe-90"
        assert stats["tokenizer"] == "bpe"
        assert stats["vocab_size"] == 90
        assert stats["vocab_used"] > 0
        assert stats["train_seconds"] >= 0.0
        assert stats["seed"] == cfg.train_budget.seed


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="reads memory from /proc")
def test_train_stats_carry_peak_memory(tmp_path: Path):
    """The stats sidecar records the run's peak resident memory, at least its starting memory."""
    cfg = _make_config(tmp_path, ["bpe"], [90])
    sample_path, _ = prepare_inputs(cfg)
    out = train_one("bpe-90", cfg=cfg, sample_path=sample_path, lexicon_path=None)

    stats = json.loads((out / TRAIN_STATS_FILENAME).read_text(encoding="utf-8"))
    assert stats["peak_rss_bytes"] >= stats["start_rss_bytes"] > 0


def test_resumed_sweep_logs_runs_trained_before_it(tmp_path: Path, monkeypatch):
    """A run trained earlier but never logged to MLflow is logged when a later sweep skips it."""
    pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)
    cfg = _enable_mlflow(_make_config(tmp_path, ["bpe"], [90]), tmp_path)
    sample_path, _ = prepare_inputs(cfg)
    train_one("bpe-90", cfg=cfg, sample_path=sample_path, lexicon_path=None)

    summary = train_sweep(cfg, ["bpe-90"], max_workers=1)

    assert summary.skipped == ["bpe-90"]
    assert (cfg.output_root / "bpe-90" / MLFLOW_LINK_FILENAME).exists()


def test_morphological_and_large_runs_queue_first():
    """The slowest runs lead the pool: morphological backends, then larger vocabularies."""
    keys = ["bpe-16000", "morphbpe-16000", "bpe-128000", "morphpiece-128000"]
    assert longest_first(keys) == ["morphpiece-128000", "morphbpe-16000", "bpe-128000", "bpe-16000"]


def test_unreachable_mlflow_refuses_to_train(tmp_path: Path, monkeypatch):
    """With tracking on and no server, the sweep stops before training anything."""
    from slm4ie.tokenizers import train as train_module

    cfg = dataclasses.replace(_make_config(tmp_path, ["bpe"], [90]), mlflow_enabled=True)
    monkeypatch.setattr(train_module.ml, "ensure_experiment", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="MLflow unreachable"):
        train_sweep(cfg, ["bpe-90"], max_workers=1)
    assert not (cfg.output_root / "bpe-90").exists()


def test_forced_training_keeps_the_sample(tmp_path: Path):
    """Retraining with force reuses the drawn sample instead of redrawing it."""
    cfg = _make_config(tmp_path, ["bpe"], [90])
    sample_path, _ = prepare_inputs(cfg)
    with gzip.open(sample_path, "wt", encoding="utf-8") as handle:
        handle.write("hiša stoji\nmačka spi\n")
    before = sample_path.read_bytes()

    summary = train_sweep(cfg, ["bpe-90"], force=True, max_workers=1)

    assert summary.trained == ["bpe-90"]
    assert sample_path.read_bytes() == before


def _enable_mlflow(cfg: SweepConfig, tmp_path: Path) -> SweepConfig:
    """Return a copy of `cfg` with MLflow enabled against a local file store.

    Args:
        tmp_path (Path): Temp directory for the file-based tracking store.
        cfg (SweepConfig): The base config to copy.

    Returns:
        SweepConfig: A config logging to a tmp `file://` MLflow store.
    """
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    return dataclasses.replace(cfg, mlflow_enabled=True, mlflow_tracking_uri=uri)


class TestLogTrainingToMlflow:
    """Tests for the training-side MLflow logger."""

    def test_noop_when_disabled(self, tmp_path: Path):
        """Disabled tracking writes no linkage sidecar and never raises."""
        cfg = _make_config(tmp_path, ["bpe"], [90])
        sample_path, _ = prepare_inputs(cfg)
        train_one("bpe-90", cfg=cfg, sample_path=sample_path, lexicon_path=None)

        log_train_runs(["bpe-90"], cfg)  # mlflow_enabled is False
        assert not (cfg.output_root / "bpe-90" / MLFLOW_LINK_FILENAME).exists()

    def test_writes_link_sidecar_when_enabled(self, tmp_path: Path, monkeypatch):
        """Enabled logging records a run id in the linkage sidecar."""
        pytest.importorskip("mlflow")
        monkeypatch.chdir(tmp_path)  # keep relative artifact dirs inside tmp
        cfg = _enable_mlflow(_make_config(tmp_path, ["bpe"], [90]), tmp_path)
        sample_path, _ = prepare_inputs(cfg)
        train_one("bpe-90", cfg=cfg, sample_path=sample_path, lexicon_path=None)

        log_train_runs(["bpe-90"], cfg)
        link = json.loads((cfg.output_root / "bpe-90" / MLFLOW_LINK_FILENAME).read_text(encoding="utf-8"))
        assert link["run_name"] == "bpe-90"
        assert link["run_id"]
        assert link["experiment"] == cfg.mlflow_experiment

    def test_logs_tokenizer_as_produced_input(self, tmp_path: Path, monkeypatch):
        """Each train child run declares its tokenizer as a produced dataset input."""
        pytest.importorskip("mlflow")
        import mlflow

        monkeypatch.chdir(tmp_path)
        cfg = _enable_mlflow(_make_config(tmp_path, ["bpe"], [90, 120]), tmp_path)
        sample_path, _ = prepare_inputs(cfg)
        for key in ("bpe-90", "bpe-120"):
            train_one(key, cfg=cfg, sample_path=sample_path, lexicon_path=None)
        log_train_runs(["bpe-90", "bpe-120"], cfg)

        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(cfg.mlflow_experiment)
        children = {
            run.data.tags.get("mlflow.runName"): run
            for run in client.search_runs([experiment.experiment_id])
            if run.data.tags.get("mlflow.parentRunId")
        }
        produced = {
            key: [d.dataset for d in run.inputs.dataset_inputs if d.dataset.name.startswith("tokenizer/")]
            for key, run in children.items()
        }
        assert [d.name for d in produced["bpe-90"]] == ["tokenizer/bpe-90"]
        # Distinct tokenizers hash to distinct digests (identity for lineage).
        digest_90 = produced["bpe-90"][0].digest
        digest_120 = produced["bpe-120"][0].digest
        assert digest_90 and digest_120 and digest_90 != digest_120

    def test_child_carries_labflow_tags_and_sample_digest(self, tmp_path: Path, monkeypatch):
        """A train child run carries the configured tags, base_commit and the training sample's content digest."""
        pytest.importorskip("mlflow")
        import mlflow

        from slm4ie.utils.mlflow import MAX_DATASET_DIGEST_LEN
        from slm4ie.utils.versioning import file_sha256

        monkeypatch.chdir(tmp_path)
        cfg = _enable_mlflow(_make_config(tmp_path, ["bpe"], [90]), tmp_path)
        cfg = dataclasses.replace(cfg, mlflow_tags={"tier": "scratch", "type": "metric-validity"})
        sample_path, _ = prepare_inputs(cfg)
        train_one("bpe-90", cfg=cfg, sample_path=sample_path, lexicon_path=None)
        log_train_runs(["bpe-90"], cfg)

        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(cfg.mlflow_experiment)
        child = next(r for r in client.search_runs([experiment.experiment_id]) if r.data.tags.get("mlflow.parentRunId"))
        assert child.data.tags["tier"] == "scratch"
        assert child.data.tags["type"] == "metric-validity"
        assert "base_commit" in child.data.tags
        digests = {d.dataset.name: d.dataset.digest for d in child.inputs.dataset_inputs}
        assert digests["sweep/train-sample"] == f"sha256:{file_sha256(sample_path)}"[:MAX_DATASET_DIGEST_LEN]


class TestResolveRunSelection:
    """Tests for the shared one-or-all run selection used by all three CLIs."""

    def test_all_selects_whole_sweep(self, tmp_path: Path):
        """all_runs selects every tokenizer x vocab-size run."""
        cfg = _make_config(tmp_path, ["bpe", "wordpiece"], [16000, 32000])
        keys = resolve_sweep_runs(cfg, all_runs=True, tokenizer=None, vocab_size=None)
        assert keys == ["bpe-16000", "bpe-32000", "wordpiece-16000", "wordpiece-32000"]

    def test_tokenizer_expands_all_vocab(self, tmp_path: Path):
        """A single tokenizer expands across all vocab sizes."""
        cfg = _make_config(tmp_path, ["bpe", "wordpiece"], [16000, 32000])
        assert resolve_sweep_runs(cfg, all_runs=False, tokenizer="bpe", vocab_size=None) == [
            "bpe-16000",
            "bpe-32000",
        ]

    def test_tokenizer_and_vocab_selects_one(self, tmp_path: Path):
        """A tokenizer narrowed by vocab size selects exactly one run."""
        cfg = _make_config(tmp_path, ["bpe", "wordpiece"], [16000, 32000])
        assert resolve_sweep_runs(cfg, all_runs=False, tokenizer="bpe", vocab_size=32000) == ["bpe-32000"]

    def test_all_with_tokenizer_raises(self, tmp_path: Path):
        """Combining all_runs with a selector is rejected."""
        cfg = _make_config(tmp_path, ["bpe"], [16000])
        with pytest.raises(ValueError):
            resolve_sweep_runs(cfg, all_runs=True, tokenizer="bpe", vocab_size=None)

    def test_no_selector_raises(self, tmp_path: Path):
        """Neither all_runs nor a tokenizer is rejected."""
        cfg = _make_config(tmp_path, ["bpe"], [16000])
        with pytest.raises(ValueError):
            resolve_sweep_runs(cfg, all_runs=False, tokenizer=None, vocab_size=None)

    def test_unknown_tokenizer_raises(self, tmp_path: Path):
        """An unconfigured tokenizer name is rejected."""
        cfg = _make_config(tmp_path, ["bpe"], [16000])
        with pytest.raises(ValueError):
            resolve_sweep_runs(cfg, all_runs=False, tokenizer="nonsense", vocab_size=None)

    def test_unknown_vocab_size_raises(self, tmp_path: Path):
        """An unconfigured vocab size is rejected."""
        cfg = _make_config(tmp_path, ["bpe"], [16000])
        with pytest.raises(ValueError):
            resolve_sweep_runs(cfg, all_runs=False, tokenizer="bpe", vocab_size=99999)
