"""Tests for slm4ie/data/pretokenize/config.py (the pretokenize registry loader)."""

from pathlib import Path
from textwrap import dedent

import pytest

from slm4ie.data.pretokenize.config import load_pretokenize_config
from slm4ie.utils.parallel import WorkerSettings

_CONFIG = dedent(
    """\
    roots:
      corpus: /data/pretrain
      tokenizers: /data/tokenizers
      output: /data/pretokenized
    defaults:
      seed: 7
    entries:
      sentence-dedup/bpe-32000-modernbert-1024:
        corpus: 06_sentence_dedup
        tokenizer: bpe-32000/hf-modernbert
        max_seq_len: 1024
    """
)


def _write(tmp_path: Path, text: str) -> Path:
    """Write a registry file and return its path.

    Args:
        tmp_path: Test directory.
        text: The YAML text.

    Returns:
        The written file.
    """
    path = tmp_path / "pretokenize.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_entry_resolves_paths_from_roots_and_key(tmp_path: Path):
    """An entry's corpus, tokenizer and output paths join the roots with its fields and key."""
    entry = load_pretokenize_config(_write(tmp_path, _CONFIG)).entries["sentence-dedup/bpe-32000-modernbert-1024"]
    assert entry.corpus_dir == Path("/data/pretrain/06_sentence_dedup")
    assert entry.tokenizer_dir == Path("/data/tokenizers/bpe-32000/hf-modernbert")
    assert entry.output_dir == Path("/data/pretokenized/sentence-dedup/bpe-32000-modernbert-1024")


def test_entry_takes_defaults_with_registry_overrides(tmp_path: Path):
    """Unset settings come from the built-in defaults, overridden by the registry's `defaults`."""
    entry = load_pretokenize_config(_write(tmp_path, _CONFIG)).entries["sentence-dedup/bpe-32000-modernbert-1024"]
    assert entry.formats == ("mds", "parquet")
    assert (entry.min_chunk_tokens, entry.validation_share, entry.min_validation_docs) == (8, 0.001, 10_000)
    assert entry.seed == 7
    assert entry.shard_size_bytes == 64 * 2**20


def test_entry_setting_overrides_defaults(tmp_path: Path):
    """A setting on the entry wins over the defaults."""
    text = _CONFIG + "    formats: [mds-streaming]\n    min_chunk_tokens: 16\n"
    entry = load_pretokenize_config(_write(tmp_path, text)).entries["sentence-dedup/bpe-32000-modernbert-1024"]
    assert (entry.formats, entry.min_chunk_tokens) == (("mds-streaming",), 16)


def test_entry_without_tokenizer_rejected(tmp_path: Path):
    """An entry missing its tokenizer fails, naming the field."""
    text = _CONFIG.replace("    tokenizer: bpe-32000/hf-modernbert\n", "")
    with pytest.raises(ValueError, match="tokenizer"):
        load_pretokenize_config(_write(tmp_path, text))


def test_key_not_naming_tokenizer_and_length_rejected(tmp_path: Path):
    """A key whose name differs from the tokenizer and length fails, naming the expected name."""
    text = _CONFIG.replace("    max_seq_len: 1024", "    max_seq_len: 512")
    with pytest.raises(ValueError, match="bpe-32000-modernbert-512"):
        load_pretokenize_config(_write(tmp_path, text))


def test_misspelled_entry_field_rejected(tmp_path: Path):
    """A misspelled entry field fails, naming it, instead of falling back to the default."""
    text = _CONFIG + "    fromats: [mds-streaming]\n"
    with pytest.raises(ValueError, match="fromats"):
        load_pretokenize_config(_write(tmp_path, text))


def test_unknown_format_rejected(tmp_path: Path):
    """A format no writer is registered for fails at load, naming it."""
    text = _CONFIG + "    formats: [mds, arrow]\n"
    with pytest.raises(ValueError, match="arrow"):
        load_pretokenize_config(_write(tmp_path, text))


def test_malformed_key_rejected(tmp_path: Path):
    """A key that is not `<corpus>/<tokenizer>-<length>` in lowercase fails."""
    text = _CONFIG.replace("sentence-dedup/bpe-32000-modernbert-1024:", "BPE-1024:")
    with pytest.raises(ValueError, match="key"):
        load_pretokenize_config(_write(tmp_path, text))


@pytest.mark.parametrize("share", ["0", "1.0", "-0.1"])
def test_validation_share_outside_the_unit_interval_rejected(tmp_path: Path, share: str):
    """A validation share that is not strictly between 0 and 1 fails, naming the value."""
    text = _CONFIG + f"    validation_share: {share}\n"
    with pytest.raises(ValueError, match="validation_share"):
        load_pretokenize_config(_write(tmp_path, text))


def test_min_chunk_longer_than_max_rejected(tmp_path: Path):
    """A minimum chunk length above the maximum sequence length fails."""
    text = _CONFIG + "    min_chunk_tokens: 2048\n"
    with pytest.raises(ValueError, match="min_chunk_tokens"):
        load_pretokenize_config(_write(tmp_path, text))


def test_missing_root_rejected(tmp_path: Path):
    """A registry without its output root fails, naming the root."""
    text = _CONFIG.replace("  output: /data/pretokenized\n", "")
    with pytest.raises(ValueError, match="output"):
        load_pretokenize_config(_write(tmp_path, text))


def test_committed_registry_loads():
    """The committed registry loads and declares the ModernSloBERT entry."""
    config = load_pretokenize_config(Path(__file__).parents[3] / "configs" / "data" / "pretokenize.yaml")
    assert "sentence-dedup/bpe-32000-modernbert-1024" in config.entries


def test_workers_absent_leave_memory_and_item_limit_unset(tmp_path: Path):
    """An entry with no workers section declares no memory and no item limit."""
    entry = load_pretokenize_config(_write(tmp_path, _CONFIG)).entries["sentence-dedup/bpe-32000-modernbert-1024"]
    assert entry.workers == WorkerSettings(memory_per_worker_gb=None, max_items_per_worker=None)


def test_entry_workers_override_the_defaults(tmp_path: Path):
    """An entry's workers section replaces the one in the defaults."""
    text = _CONFIG.replace("  seed: 7\n", "  workers: {memory_per_worker_gb: 4}\n")
    text += "    workers: {memory_per_worker_gb: 6.5, max_items_per_worker: 20}\n"
    entry = load_pretokenize_config(_write(tmp_path, text)).entries["sentence-dedup/bpe-32000-modernbert-1024"]
    assert entry.workers == WorkerSettings(memory_per_worker_gb=6.5, max_items_per_worker=20)


@pytest.mark.parametrize(
    "workers", ["{memory_per_worker_gb: 0}", "{memory_per_worker_gb: -2}", "{memory_per_worker_gb: big}"]
)
def test_non_positive_memory_per_worker_rejected(tmp_path: Path, workers: str):
    """A memory per worker that is not a positive number is rejected, naming the value."""
    text = _CONFIG + f"    workers: {workers}\n"
    with pytest.raises(ValueError, match="memory_per_worker_gb"):
        load_pretokenize_config(_write(tmp_path, text))


@pytest.mark.parametrize("workers", ["{max_items_per_worker: 0}", "{max_items_per_worker: 2.5}"])
def test_non_positive_item_limit_rejected(tmp_path: Path, workers: str):
    """An item limit that is not a positive integer is rejected, naming the value."""
    text = _CONFIG + f"    workers: {workers}\n"
    with pytest.raises(ValueError, match="max_items_per_worker"):
        load_pretokenize_config(_write(tmp_path, text))


def test_committed_registry_declares_four_gb_per_worker():
    """The committed registry declares the measured 4 GB per worker for its entries."""
    config = load_pretokenize_config(Path(__file__).parents[3] / "configs" / "data" / "pretokenize.yaml")
    assert {entry.workers.memory_per_worker_gb for entry in config.entries.values()} == {4.0}
