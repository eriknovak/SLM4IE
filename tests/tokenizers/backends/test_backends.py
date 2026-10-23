"""Tests for the library tokenizer backends (BPE, charBPE, WordPiece, Unigram).

These train on a tiny corpus with a tiny vocabulary so they stay fast in CI;
the real 16k/32k/64k sweeps run only outside the test suite. Reconstruction is
checked via offset tiling so the assertion holds for byte-level and
character-level schemes alike.
"""

from pathlib import Path
from typing import List, Tuple

import pytest

import slm4ie.tokenizers.backends  # noqa: F401  (registers backends)
from slm4ie.tokenizers.base import TrainContext
from slm4ie.tokenizers.registry import get_tokenizer

#: Small repeated Slovene corpus, enough for a tiny vocabulary to converge.
_CORPUS: List[str] = [
    "hiša stoji ob cesti",
    "hiše so velike in lepe",
    "pes laja na psa",
    "mačka spi v hiši",
    "junak gre v boj",
    "velika hiša ob veliki cesti",
] * 40

_SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]

#: Library backends that train without a morpheme lexicon.
_BACKENDS = ["bpe", "charbpe", "wordpiece", "unigram"]


def _train(name: str, vocab_size: int = 160):
    """Train a registered backend on the tiny corpus.

    Args:
        name (str): Registry key.
        vocab_size (int): Target vocabulary size.

    Returns:
        BaseTokenizer: The trained tokenizer.
    """
    tokenizer = get_tokenizer(name)()
    tokenizer.train(iter(_CORPUS), vocab_size, config=TrainContext(special_tokens=_SPECIAL_TOKENS))
    return tokenizer


def _covers_word(offsets: List[Tuple[str, int, int]], word: str) -> bool:
    """Return True when the token offsets exactly tile `word`.

    Args:
        offsets (List[Tuple[str, int, int]]): `(piece, start, end)` spans.
        word (str): The encoded word.

    Returns:
        bool: True if the union of spans covers every character of `word`.
    """
    covered = set()
    for _piece, start, end in offsets:
        covered.update(range(start, end))
    return covered == set(range(len(word)))


class TestBackendsRegistered:
    """The backends register themselves on import."""

    def test_all_registered(self):
        """Every expected backend resolves from the registry."""
        for name in _BACKENDS:
            assert get_tokenizer(name).name == name


@pytest.mark.parametrize("name", _BACKENDS)
class TestBackendTraining:
    """Training, encoding, offsets, and persistence per backend."""

    def test_encode_non_empty(self, name: str):
        """A trained tokenizer encodes text into pieces."""
        tokenizer = _train(name)
        assert tokenizer.encode("hiša stoji") != []
        assert len(tokenizer.vocab) > 0

    def test_offsets_tile_word(self, name: str):
        """Token offsets cover every character of an encoded word."""
        tokenizer = _train(name)
        offsets = tokenizer.encode_offsets("hiše")
        assert _covers_word(offsets, "hiše")

    def test_save_load_round_trip(self, name: str, tmp_path: Path):
        """Save then load reproduces encoding and vocabulary."""
        tokenizer = _train(name)
        tokenizer.save(tmp_path)

        loaded = get_tokenizer(name).load(tmp_path)
        assert loaded.encode("hiše velike") == tokenizer.encode("hiše velike")
        assert loaded.vocab == tokenizer.vocab
        assert loaded.vocab_size == tokenizer.vocab_size


class TestUnigramLongLines:
    """SentencePiece drops lines over its length limit, so Unigram cuts documents first."""

    def test_chunks_stay_under_the_limit_and_keep_every_word(self):
        """A long line splits into pieces under the byte limit, in order, with no word lost."""
        from slm4ie.tokenizers.backends.sp_unigram import chunk_line

        line = " ".join(["mačka"] * 50)
        pieces = list(chunk_line(line, max_bytes=40))
        assert all(len(piece.encode("utf-8")) <= 40 for piece in pieces)
        assert " ".join(pieces) == line

    def test_trains_on_documents_longer_than_the_limit(self):
        """A corpus of only over-long lines still trains, since no line is dropped whole."""
        long_document = " ".join(_CORPUS)
        assert len(long_document.encode("utf-8")) > 4192
        tokenizer = get_tokenizer("unigram")()
        tokenizer.train(iter([long_document] * 3), 160, config=TrainContext(special_tokens=_SPECIAL_TOKENS))
        assert len(tokenizer.vocab) > 2 * len(_SPECIAL_TOKENS)
        assert "<unk>" not in tokenizer.encode("hiša stoji")
