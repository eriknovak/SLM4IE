"""Tests for slm4ie/tokenizers/hf_export.py (HuggingFace export + offsets)."""

from pathlib import Path

import pytest

import slm4ie.tokenizers.backends  # noqa: F401  (registers backends)
from slm4ie.tokenizers.base import TrainContext
from slm4ie.tokenizers.hf_export import load_pretrained, save_pretrained_dir, to_hf
from slm4ie.tokenizers.morphology import MorphLexicon, _add_to_lexicon, segment_form
from slm4ie.tokenizers.registry import get_tokenizer

_LEXICON_ENTRIES = [
    ("hiša", "hiša", "Ncfsn"),
    ("hiše", "hiša", "Ncfsg"),
    ("hiši", "hiša", "Ncfsd"),
    ("hišami", "hiša", "Ncfpi"),
]

_CORPUS = [
    "hiša ob cesti",
    "hiše so velike in lepe",
    "mačka spi v hiši",
    "hišami gradijo mesto",
] * 60

_SPECIAL = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
_TEXT = "hiše ob cesti"
_FAST_BACKENDS = ["bpe", "charbpe", "wordpiece", "unigram", "morphbpe"]


def _lexicon() -> MorphLexicon:
    """Build the fixture morpheme lexicon.

    Returns:
        MorphLexicon: A lexicon over the fixture entries.
    """
    lexicon = MorphLexicon()
    for form, lemma, msd in _LEXICON_ENTRIES:
        seg = segment_form(form, lemma, msd)
        if seg is not None and seg.is_reliable:
            _add_to_lexicon(lexicon, seg)
    return lexicon


def _train_artifact(name: str, tmp_path: Path) -> Path:
    """Train a backend and save it under `tmp_path`.

    Args:
        name (str): Registry key.
        tmp_path (Path): Destination directory.

    Returns:
        Path: The artifact directory.
    """
    tokenizer = get_tokenizer(name)()
    tokenizer.train(_CORPUS, 160, config=TrainContext(special_tokens=_SPECIAL, lexicon=_lexicon()))
    tokenizer.save(tmp_path)
    return tmp_path


@pytest.mark.parametrize("name", _FAST_BACKENDS)
class TestFastExport:
    """Export of the five fast backends to PreTrainedTokenizerFast."""

    def test_offsets_present_and_aligned(self, name: str, tmp_path: Path):
        """The exported tokenizer yields offset mappings into the text."""
        hf = to_hf(_train_artifact(name, tmp_path))
        out = hf(_TEXT, return_offsets_mapping=True)
        assert out["input_ids"]
        offsets = out["offset_mapping"]
        assert len(offsets) == len(out["input_ids"])
        # Every span lies within the text.
        assert all(0 <= s <= e <= len(_TEXT) for s, e in offsets)

    def test_decode_runs(self, name: str, tmp_path: Path):
        """Decoding the ids returns the (non-empty) text content."""
        hf = to_hf(_train_artifact(name, tmp_path))
        ids = hf(_TEXT)["input_ids"]
        decoded = hf.decode(ids, skip_special_tokens=True)
        # All schemes recover the word characters (spacing may differ for
        # character-level BPE, which has no continuation marker).
        assert "hiš" in decoded and "cesti" in decoded


class TestAutoTokenizer:
    """A fast export round-trips through AutoTokenizer.from_pretrained."""

    def test_save_and_reload(self, tmp_path: Path):
        """save_pretrained_dir produces an AutoTokenizer-loadable directory."""
        from transformers import AutoTokenizer

        artifact = _train_artifact("bpe", tmp_path)
        save_pretrained_dir(artifact)
        reloaded = AutoTokenizer.from_pretrained(str(artifact))
        assert reloaded(_TEXT)["input_ids"] == to_hf(artifact)(_TEXT)["input_ids"]


class TestMorphPieceExport:
    """MorphPiece exports as a custom slow tokenizer with offsets."""

    def test_encode_with_offsets(self, tmp_path: Path):
        """encode_with_offsets returns aligned ids and spans."""
        hf = load_pretrained(_train_artifact("morphpiece", tmp_path))
        encoded = hf.encode_with_offsets(_TEXT)
        assert encoded["input_ids"]
        assert len(encoded["input_ids"]) == len(encoded["offset_mapping"])
        assert all(0 <= s <= e <= len(_TEXT) for s, e in encoded["offset_mapping"])

    def test_decode_recovers_known_word(self, tmp_path: Path):
        """Decoding recovers the known word's surface form."""
        hf = load_pretrained(_train_artifact("morphpiece", tmp_path))
        ids = hf.encode_with_offsets(_TEXT)["input_ids"]
        assert "hiše" in hf.decode(ids, skip_special_tokens=True)
