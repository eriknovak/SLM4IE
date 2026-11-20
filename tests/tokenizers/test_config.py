"""Tests for slm4ie/tokenizers/config.py (tokenizer-sweep config loading)."""

from pathlib import Path
from textwrap import dedent

import pytest

from slm4ie.tokenizers.config import load_sweep_config

_BASE_CONFIG = dedent(
    """\
    corpus:
      root: /data/pretrain/06_sentence_dedup
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
    """Tests for load_sweep_config."""

    def test_parses_full_config(self, tmp_path: Path):
        """A complete config parses into the expected dataclass."""
        cfg = load_sweep_config(_write(tmp_path / "tokenizers.yaml", _BASE_CONFIG))
        assert cfg.corpus_root == Path("/data/pretrain/06_sentence_dedup")
        assert cfg.tokenizers == ["bpe", "morphbpe"]
        assert cfg.vocab_sizes == [16000, 32000]
        assert cfg.train_budget.max_bytes == 1000
        assert cfg.train_budget.source_weights == {"macocu_sl": 2.0}
        assert cfg.min_stem_len == 3
        assert cfg.report_dir == Path("/data/tokenizers/_reports")
        assert cfg.eval_budget.seed == 99
        assert cfg.mlflow_enabled is True

    def test_default_experiment_name(self, tmp_path: Path):
        """An omitted mlflow block defaults to the slm4ie/<workstream>/<stage> name."""
        no_mlflow = _BASE_CONFIG.split("mlflow:")[0]
        cfg = load_sweep_config(_write(tmp_path / "tokenizers.yaml", no_mlflow))
        assert cfg.mlflow_experiment == "slm4ie/tokenizers/sweep"
        assert cfg.mlflow_enabled is False

    def test_needs_morphology(self, tmp_path: Path):
        """needs_morphology is True when a morph backend is requested."""
        cfg = load_sweep_config(_write(tmp_path / "tokenizers.yaml", _BASE_CONFIG))
        assert cfg.needs_morphology() is True

    def test_local_overlay_overrides(self, tmp_path: Path):
        """A sibling .local.yaml deep-merges over the base config."""
        _write(tmp_path / "tokenizers.yaml", _BASE_CONFIG)
        _write(
            tmp_path / "tokenizers.local.yaml",
            "mlflow:\n  tracking_uri: http://example:5555\n",
        )
        cfg = load_sweep_config(tmp_path / "tokenizers.yaml")
        assert cfg.mlflow_tracking_uri == "http://example:5555"
        # Base values survive the merge.
        assert cfg.mlflow_experiment == "tokenizer/slovenian"

    def test_missing_required_field_raises(self, tmp_path: Path):
        """Omitting a required field raises ValueError listing it."""
        bad = "tokenizers: [bpe]\nvocab_sizes: [16000]\n"
        with pytest.raises(ValueError, match="corpus.root"):
            load_sweep_config(_write(tmp_path / "bad.yaml", bad))

    def test_missing_file_raises(self, tmp_path: Path):
        """A non-existent config path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_sweep_config(tmp_path / "nope.yaml")


def test_split_keys_default_to_none(tmp_path: Path):
    """Without split keys the sweep samples the whole corpus and holds no lexemes out."""
    cfg = load_sweep_config(_write(tmp_path / "s.yaml", _BASE_CONFIG))
    assert (cfg.corpus_train_share, cfg.heldout_lexeme_share, cfg.inflection_path) == (None, None, None)


def test_split_keys_load(tmp_path: Path):
    """The corpus split share and the held-out lexeme settings reach the config."""
    text = _BASE_CONFIG.replace("  datasets: [macocu_sl]\n", "  datasets: [macocu_sl]\n  split: { train_share: 0.8 }\n")
    text = text.replace(
        "  min_stem_len: 3\n",
        "  min_stem_len: 3\n  inflection_path: /data/tokenization/sloleks_inflection.jsonl.gz\n"
        "  heldout_lexeme_share: 0.2\n",
    )
    cfg = load_sweep_config(_write(tmp_path / "s.yaml", text))
    assert cfg.corpus_train_share == 0.8
    assert cfg.heldout_lexeme_share == 0.2
    assert cfg.corpus_split_path == Path("/data/tokenizers/corpus_split.json")


def test_heldout_share_without_inflection_path_rejected(tmp_path: Path):
    """Holding lexemes out needs the stem-ending split they come from."""
    text = _BASE_CONFIG.replace("  min_stem_len: 3\n", "  min_stem_len: 3\n  heldout_lexeme_share: 0.2\n")
    with pytest.raises(ValueError, match="inflection_path"):
        load_sweep_config(_write(tmp_path / "s.yaml", text))


def test_mlflow_tags_load_as_strings(tmp_path: Path):
    """The dial's run tags reach the config, every value as a string."""
    text = _BASE_CONFIG.replace("  enabled: true\n", "  enabled: true\n  tags: { type: metric-validity, version: 2 }\n")
    cfg = load_sweep_config(_write(tmp_path / "s.yaml", text))
    assert cfg.mlflow_tags == {"type": "metric-validity", "version": "2"}


def test_mlflow_tags_default_to_empty(tmp_path: Path):
    """A config without run tags tags nothing beyond the structural tags."""
    assert load_sweep_config(_write(tmp_path / "s.yaml", _BASE_CONFIG)).mlflow_tags == {}


def test_character_coverage_defaults_to_every_character(tmp_path: Path):
    """A config without the key keeps every character."""
    assert load_sweep_config(_write(tmp_path / "s.yaml", _BASE_CONFIG)).character_coverage == 1.0


def test_character_coverage_loads(tmp_path: Path):
    """The dial's coverage reaches the config as a float."""
    text = _BASE_CONFIG + "character_coverage: 0.9995\n"
    assert load_sweep_config(_write(tmp_path / "s.yaml", text)).character_coverage == 0.9995


@pytest.mark.parametrize("value", ["0", "1.5", "null"])
def test_character_coverage_outside_the_unit_interval_rejected(tmp_path: Path, value: str):
    """A coverage that names no alphabet fails at load, naming the key and the value."""
    text = _BASE_CONFIG + f"character_coverage: {value}\n"
    with pytest.raises(ValueError, match="character_coverage"):
        load_sweep_config(_write(tmp_path / "s.yaml", text))


_EXPORT_SECTION = dedent(
    """\
    special_tokens: ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
    export:
      parity_lines: 50
      targets:
        modernbert:
          roles: {pad_token: "<pad>", cls_token: "<s>", sep_token: "</s>"}
          single: "<s> $A </s>"
          pair: "<s> $A </s> $B:1 </s>:1"
          model_max_length: 8192
    """
)


def _with_export(section: str) -> str:
    """Return the base config with its special tokens replaced by `section`.

    Args:
        section: YAML holding `special_tokens` and an `export` mapping.

    Returns:
        The combined config text.
    """
    return _BASE_CONFIG.replace('special_tokens: ["<unk>"]\n', "") + section


def test_export_section_absent_gives_no_targets_and_default_parity(tmp_path: Path):
    """A config without an export section has no targets and the default parity count."""
    cfg = load_sweep_config(_write(tmp_path / "s.yaml", _BASE_CONFIG))
    assert cfg.export_targets == {}
    assert cfg.export_parity_lines == 1000


def test_export_target_loads(tmp_path: Path):
    """An export target loads with its roles, templates and maximum length."""
    cfg = load_sweep_config(_write(tmp_path / "s.yaml", _with_export(_EXPORT_SECTION)))
    target = cfg.export_targets["modernbert"]
    assert target.roles == {"pad_token": "<pad>", "cls_token": "<s>", "sep_token": "</s>"}
    assert target.single == "<s> $A </s>"
    assert target.model_max_length == 8192
    assert cfg.export_parity_lines == 50


def test_export_target_with_token_outside_special_tokens_rejected(tmp_path: Path):
    """A role mapped to a token the sweep does not reserve is rejected."""
    text = _with_export(_EXPORT_SECTION.replace('cls_token: "<s>"', 'cls_token: "[CLS]"'))
    with pytest.raises(ValueError, match=r"\[CLS\]"):
        load_sweep_config(_write(tmp_path / "s.yaml", text))


def test_export_template_with_token_outside_special_tokens_rejected(tmp_path: Path):
    """A template naming a token the sweep does not reserve is rejected."""
    text = _with_export(_EXPORT_SECTION.replace('single: "<s> $A </s>"', 'single: "[CLS] $A </s>"'))
    with pytest.raises(ValueError, match=r"\[CLS\]"):
        load_sweep_config(_write(tmp_path / "s.yaml", text))


def test_export_target_with_unknown_role_rejected(tmp_path: Path):
    """A role name HuggingFace does not know is rejected."""
    text = _with_export(_EXPORT_SECTION.replace("cls_token:", "start_token:"))
    with pytest.raises(ValueError, match="start_token"):
        load_sweep_config(_write(tmp_path / "s.yaml", text))


def test_export_target_without_template_rejected(tmp_path: Path):
    """A target missing its pair template is rejected."""
    text = _with_export(_EXPORT_SECTION.replace('      pair: "<s> $A </s> $B:1 </s>:1"\n', ""))
    with pytest.raises(ValueError, match="pair"):
        load_sweep_config(_write(tmp_path / "s.yaml", text))
