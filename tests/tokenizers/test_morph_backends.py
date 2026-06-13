"""Tests for the morphological backends (MorphBPE, MorphPiece)."""

from pathlib import Path
from typing import List

import pytest

import slm4ie.tokenizers.backends  # noqa: F401  (registers backends)
from slm4ie.tokenizers.base import TrainContext, clean_piece
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


def _piece_boundaries(pieces: List[str]) -> set:
    """Return cumulative character-offset boundaries of cleaned pieces.

    Args:
        pieces (List[str]): Encoded pieces of a single word.

    Returns:
        set: Internal boundary offsets (excluding 0 and the total length).
    """
    offsets = set()
    cursor = 0
    for piece in pieces[:-1]:
        cursor += len(clean_piece(piece))
        offsets.add(cursor)
    return offsets


@pytest.mark.parametrize("name", _MORPH_BACKENDS)
class TestMorphBackends:
    """Behavioral tests shared by the morphological backends."""

    def test_requires_lexicon(self, name: str):
        """Training without a lexicon raises ValueError."""
        with pytest.raises(ValueError, match="lexicon"):
            get_tokenizer(name)().train(iter(_CORPUS), 80, config=TrainContext())

    def test_pieces_reconstruct_word(self, name: str):
        """Cleaned pieces of a known word concatenate back to the word."""
        tokenizer = _train(name)
        cleaned = "".join(clean_piece(p) for p in tokenizer.encode("hišami"))
        assert cleaned == "hišami"

    def test_respects_morpheme_boundary(self, name: str):
        """No piece spans the gold morpheme boundary of a known form."""
        tokenizer = _train(name)
        # 'hišami' aligns to stem 'hiša' + suffix 'mi' -> boundary at offset 4.
        boundaries = _piece_boundaries(tokenizer.encode("hišami"))
        assert 4 in boundaries

    def test_oov_word_falls_back(self, name: str):
        """An unknown word is still encoded and reconstructs."""
        tokenizer = _train(name)
        pieces = tokenizer.encode("računalnik")
        assert pieces != []
        assert "".join(clean_piece(p) for p in pieces) == "računalnik"

    def test_save_load_round_trip(self, name: str, tmp_path: Path):
        """save then load reproduces encoding and vocabulary."""
        tokenizer = _train(name)
        tokenizer.save(tmp_path)
        loaded = get_tokenizer(name).load(tmp_path)
        assert loaded.encode("hišami psu cesti") == tokenizer.encode("hišami psu cesti")
        assert loaded.vocab == tokenizer.vocab
