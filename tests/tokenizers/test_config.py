"""Tests for slm4ie/utils/config.py (tokenizer-sweep config loading)."""

from pathlib import Path
from textwrap import dedent

import pytest

from slm4ie.utils.config import load_tokenizer_config

_BASE_CONFIG = dedent(
    """\
    corpus:
      root: /data/pretrain/05_2_dedup
      datasets: [macocu_sl]
      budget:
        max_bytes: 1000
        seed: 5
        source_weights:
          macocu_sl: 2.0
    tokenizers: [bpe, morphbpe]
    vocab_sizes: [16000, 32000]
    special_tokens: ["<unk>"]
    morphology:
      sloleks_path: /data/tokenization/sloleks.jsonl.gz
      min_stem_len: 3
    output:
      root: /data/tokenizers
    eval:
      corpus_budget: { max_bytes: 200, seed: 99 }
      renyi_alpha: 2.5
    mlflow:
      experiment: tokenizer/slovenian
      enabled: true
    """
)


def _write(path: Path, text: str) -> Path:
    """Write `text` to `path` and return it.

    Args:
        path (Path): Destination file.
        text (str): File contents.

    Returns:
        Path: `path`, for chaining.
    """
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadTokenizerConfig:
    """Tests for load_tokenizer_config."""

    def test_parses_full_config(self, tmp_path: Path):
        """A complete config parses into the expected dataclass."""
        cfg = load_tokenizer_config(_write(tmp_path / "tokenizers.yaml", _BASE_CONFIG))
        assert cfg.corpus_root == Path("/data/pretrain/05_2_dedup")
        assert cfg.tokenizers == ["bpe", "morphbpe"]
        assert cfg.vocab_sizes == [16000, 32000]
        assert cfg.train_budget.max_bytes == 1000
        assert cfg.train_budget.source_weights == {"macocu_sl": 2.0}
        assert cfg.min_stem_len == 3
        assert cfg.report_dir == Path("/data/tokenizers/_reports")
        assert cfg.eval_budget.seed == 99
        assert cfg.mlflow_enabled is True

    def test_needs_morphology(self, tmp_path: Path):
        """needs_morphology is True when a morph backend is requested."""
        cfg = load_tokenizer_config(_write(tmp_path / "tokenizers.yaml", _BASE_CONFIG))
        assert cfg.needs_morphology() is True

    def test_local_overlay_overrides(self, tmp_path: Path):
        """A sibling .local.yaml deep-merges over the base config."""
        _write(tmp_path / "tokenizers.yaml", _BASE_CONFIG)
        _write(
            tmp_path / "tokenizers.local.yaml",
            "mlflow:\n  tracking_uri: http://example:5555\n",
        )
        cfg = load_tokenizer_config(tmp_path / "tokenizers.yaml")
        assert cfg.mlflow_tracking_uri == "http://example:5555"
        # Base values survive the merge.
        assert cfg.mlflow_experiment == "tokenizer/slovenian"

    def test_missing_required_field_raises(self, tmp_path: Path):
        """Omitting a required field raises ValueError listing it."""
        bad = "tokenizers: [bpe]\nvocab_sizes: [16000]\n"
        with pytest.raises(ValueError, match="corpus.root"):
            load_tokenizer_config(_write(tmp_path / "bad.yaml", bad))

    def test_missing_file_raises(self, tmp_path: Path):
        """A non-existent config path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_tokenizer_config(tmp_path / "nope.yaml")
