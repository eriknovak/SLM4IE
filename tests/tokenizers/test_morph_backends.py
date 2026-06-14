"""Tests for the morphological backends (MorphBPE, MorphPiece).

MorphBPE constrains merges at training and tokenizes with standard inference;
MorphPiece uses a two-path (morpheme table else BPE) inference. Reconstruction
and boundary checks use offsets so they hold across schemes.
"""

from pathlib import Path
from typing import List, Tuple

import pytest

import slm4ie.tokenizers.backends  # noqa: F401  (registers backends)
from slm4ie.tokenizers.base import TrainContext
from slm4ie.tokenizers.morphology import MorphLexicon, _add_to_lexicon, segment_form
from slm4ie.tokenizers.registry import get_tokenizer

_LEXICON_ENTRIES = [
    ("hiša", "hiša", "Ncfsn"),
    ("hiše", "hiša", "Ncfsg"),
    ("hiši", "hiša", "Ncfsd"),
    ("hišami", "hiša", "Ncfpi"),
    ("pes", "pes", "Ncmsn"),
    ("psa", "pes", "Ncmsg"),
    ("psu", "pes", "Ncmsd"),
]

_CORPUS = [
    "hiša ob cesti",
    "hiše so lepe",
    "psu in psa",
    "hišami gradijo mesto",
    "pes laja na psa",
] * 60

_MORPH_BACKENDS = ["morphbpe", "morphpiece"]


def _build_lexicon() -> MorphLexicon:
    """Build a small reliable morpheme lexicon for the tests.

    Returns:
        MorphLexicon: A lexicon over the fixture entries.
    """
    lexicon = MorphLexicon()
    for form, lemma, msd in _LEXICON_ENTRIES:
        seg = segment_form(form, lemma, msd)
        if seg is not None and seg.is_reliable:
            _add_to_lexicon(lexicon, seg)
    return lexicon


def _train(name: str, vocab_size: int = 80):
    """Train a morphological backend on the fixture corpus.

    Args:
        name (str): Registry key.
        vocab_size (int): Target vocabulary size.

    Returns:
        BaseTokenizer: The trained tokenizer.
    """
    tokenizer = get_tokenizer(name)()
    context = TrainContext(special_tokens=["<unk>"], lexicon=_build_lexicon())
    tokenizer.train(iter(_CORPUS), vocab_size, config=context)
    return tokenizer


def _covers_word(offsets: List[Tuple[str, int, int]], word: str) -> bool:
    """Return True when token offsets tile every character of `word`.

    Args:
        offsets (List[Tuple[str, int, int]]): `(piece, start, end)` spans.
        word (str): The encoded word.

    Returns:
        bool: True if the spans cover `range(len(word))`.
    """
    covered = set()
    for _piece, start, end in offsets:
        covered.update(range(start, end))
    return covered == set(range(len(word)))


def _internal_boundaries(offsets: List[Tuple[str, int, int]]) -> set:
    """Return the internal boundary offsets implied by token spans.

    Args:
        offsets (List[Tuple[str, int, int]]): `(piece, start, end)` spans.

    Returns:
        set: Distinct token start offsets excluding 0.
    """
    return {start for _piece, start, _end in offsets if start > 0}


@pytest.mark.parametrize("name", _MORPH_BACKENDS)
class TestMorphBackends:
    """Behavioral tests shared by the morphological backends."""

    def test_requires_lexicon(self, name: str):
        """Training without a lexicon raises ValueError."""
        with pytest.raises(ValueError, match="lexicon"):
            get_tokenizer(name)().train(iter(_CORPUS), 80, config=TrainContext())

    def test_offsets_tile_word(self, name: str):
        """Token offsets cover every character of a known word."""
        tokenizer = _train(name)
        assert _covers_word(tokenizer.encode_offsets("hišami"), "hišami")

    def test_respects_morpheme_boundary(self, name: str):
        """The gold morpheme boundary of a known form is a token boundary."""
        tokenizer = _train(name)
        # 'hišami' aligns to stem 'hiša' + suffix 'mi' -> boundary at offset 4.
        assert 4 in _internal_boundaries(tokenizer.encode_offsets("hišami"))

    def test_oov_word_encodes(self, name: str):
        """An unknown word (in-vocab characters) still encodes and tiles."""
        tokenizer = _train(name)
        offsets = tokenizer.encode_offsets("gradijo")
        assert offsets
        assert _covers_word(offsets, "gradijo")

    def test_save_load_round_trip(self, name: str, tmp_path: Path):
        """Save then load reproduces encoding."""
        tokenizer = _train(name)
        tokenizer.save(tmp_path)
        loaded = get_tokenizer(name).load(tmp_path)
        assert loaded.encode("hišami gradijo") == tokenizer.encode("hišami gradijo")
