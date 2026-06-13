"""Tests for slm4ie/tokenizers/train.py orchestration."""

import gzip
import json
from pathlib import Path

from slm4ie.tokenizers.corpus import SampleBudget
from slm4ie.tokenizers.registry import get_tokenizer
from slm4ie.tokenizers.train import (
    parse_run_key,
    plan_runs,
    prepare_inputs,
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
