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
    log_training_to_mlflow,
    parse_run_key,
    plan_runs,
    prepare_inputs,
    resolve_run_selection,
    run_key,
    select_runs,
    train_one,
)
from slm4ie.utils.config import TokenizerSweepConfig

_CORPUS_LINES = [
    "hiša stoji ob cesti",
    "hiše so velike in lepe",
    "mačka spi v hiši",
    "velika hiša ob veliki cesti",
] * 40


def _make_config(tmp_path: Path, tokenizers, vocab_sizes) -> TokenizerSweepConfig:
    """Build a tiny sweep config over a fake corpus under `tmp_path`.

    Args:
        tmp_path (Path): Temp directory root.
        tokenizers (list): Tokenizer names for the sweep.
        vocab_sizes (list): Vocab sizes for the sweep.

    Returns:
        TokenizerSweepConfig: A config pointing at the fake corpus.
    """
    corpus_root = tmp_path / "dd"
    sub = corpus_root / "macocu_sl"
    sub.mkdir(parents=True)
    with gzip.open(sub / "00000.jsonl.gz", "wt", encoding="utf-8") as handle:
        for i, text in enumerate(_CORPUS_LINES):
            handle.write(json.dumps({"text": text, "id": str(i), "metadata": {"dataset": "macocu_sl"}}) + "\n")

    return TokenizerSweepConfig(
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
        """run_key and parse_run_key are inverses."""
        assert parse_run_key(run_key("morphbpe", 16000)) == ("morphbpe", 16000)

    def test_plan_runs_cartesian(self, tmp_path: Path):
        """plan_runs is the cartesian product in tokenizer-major order."""
        cfg = _make_config(tmp_path, ["bpe", "wordpiece"], [16000, 32000])
        assert plan_runs(cfg) == ["bpe-16000", "bpe-32000", "wordpiece-16000", "wordpiece-32000"]

    def test_select_runs_filters(self, tmp_path: Path):
        """select_runs narrows the sweep by tokenizer and vocab filters."""
        cfg = _make_config(tmp_path, ["bpe", "wordpiece"], [16000, 32000])
        assert select_runs(cfg, tokenizers=["bpe"]) == ["bpe-16000", "bpe-32000"]
        assert select_runs(cfg, vocab_sizes=[32000]) == ["bpe-32000", "wordpiece-32000"]


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


def _enable_mlflow(cfg: TokenizerSweepConfig, tmp_path: Path) -> TokenizerSweepConfig:
    """Return a copy of `cfg` with MLflow enabled against a local file store.

    Args:
        tmp_path (Path): Temp directory for the file-based tracking store.
        cfg (TokenizerSweepConfig): The base config to copy.

    Returns:
        TokenizerSweepConfig: A config logging to a tmp `file://` MLflow store.
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

        log_training_to_mlflow(["bpe-90"], cfg)  # mlflow_enabled is False
        assert not (cfg.output_root / "bpe-90" / MLFLOW_LINK_FILENAME).exists()

    def test_writes_link_sidecar_when_enabled(self, tmp_path: Path, monkeypatch):
        """Enabled logging records a run id in the linkage sidecar."""
        pytest.importorskip("mlflow")
        monkeypatch.chdir(tmp_path)  # keep relative artifact dirs inside tmp
        cfg = _enable_mlflow(_make_config(tmp_path, ["bpe"], [90]), tmp_path)
        sample_path, _ = prepare_inputs(cfg)
        train_one("bpe-90", cfg=cfg, sample_path=sample_path, lexicon_path=None)

        log_training_to_mlflow(["bpe-90"], cfg)
        link = json.loads((cfg.output_root / "bpe-90" / MLFLOW_LINK_FILENAME).read_text(encoding="utf-8"))
        assert link["run_name"] == "bpe-90"
        assert link["run_id"]
        assert link["experiment"] == cfg.mlflow_experiment


class TestResolveRunSelection:
    """Tests for the shared one-or-all run selection used by all three CLIs."""

    def test_all_selects_whole_sweep(self, tmp_path: Path):
        """all_runs selects every tokenizer x vocab-size run."""
        cfg = _make_config(tmp_path, ["bpe", "wordpiece"], [16000, 32000])
        keys = resolve_run_selection(cfg, all_runs=True, tokenizer=None, vocab_size=None)
        assert keys == ["bpe-16000", "bpe-32000", "wordpiece-16000", "wordpiece-32000"]

    def test_tokenizer_expands_all_vocab(self, tmp_path: Path):
        """A single tokenizer expands across all vocab sizes."""
        cfg = _make_config(tmp_path, ["bpe", "wordpiece"], [16000, 32000])
        assert resolve_run_selection(cfg, all_runs=False, tokenizer="bpe", vocab_size=None) == [
            "bpe-16000",
            "bpe-32000",
        ]

    def test_tokenizer_and_vocab_selects_one(self, tmp_path: Path):
        """A tokenizer narrowed by vocab size selects exactly one run."""
        cfg = _make_config(tmp_path, ["bpe", "wordpiece"], [16000, 32000])
        assert resolve_run_selection(cfg, all_runs=False, tokenizer="bpe", vocab_size=32000) == ["bpe-32000"]

    def test_all_with_tokenizer_raises(self, tmp_path: Path):
        """Combining all_runs with a selector is rejected."""
        cfg = _make_config(tmp_path, ["bpe"], [16000])
        with pytest.raises(ValueError):
            resolve_run_selection(cfg, all_runs=True, tokenizer="bpe", vocab_size=None)

    def test_no_selector_raises(self, tmp_path: Path):
        """Neither all_runs nor a tokenizer is rejected."""
        cfg = _make_config(tmp_path, ["bpe"], [16000])
        with pytest.raises(ValueError):
            resolve_run_selection(cfg, all_runs=False, tokenizer=None, vocab_size=None)

    def test_unknown_tokenizer_raises(self, tmp_path: Path):
        """An unconfigured tokenizer name is rejected."""
        cfg = _make_config(tmp_path, ["bpe"], [16000])
        with pytest.raises(ValueError):
            resolve_run_selection(cfg, all_runs=False, tokenizer="nonsense", vocab_size=None)

    def test_unknown_vocab_size_raises(self, tmp_path: Path):
        """An unconfigured vocab size is rejected."""
        cfg = _make_config(tmp_path, ["bpe"], [16000])
        with pytest.raises(ValueError):
            resolve_run_selection(cfg, all_runs=False, tokenizer="bpe", vocab_size=99999)


class TestScriptFlagsAligned:
    """All three tokenizer CLIs expose the same one-or-all selection flags."""

    def test_parse_args_expose_selection_flags(self):
        """train/analyze/export parse --tokenizer/--vocab-size/--all alike."""
        from scripts.tokenizers.analyze import parse_args as analyze_args
        from scripts.tokenizers.export import parse_args as export_args
        from scripts.tokenizers.train import parse_args as train_args

        for parse_args in (train_args, analyze_args, export_args):
            ns = parse_args(["--tokenizer", "bpe", "--vocab-size", "16000"])
            assert ns.tokenizer == "bpe"
            assert ns.vocab_size == 16000
            assert ns.all is False
            assert parse_args(["--all"]).all is True
