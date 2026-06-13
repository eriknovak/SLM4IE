"""Tests for the library tokenizer backends (BPE, WordPiece, Unigram).

These train on a tiny corpus with a tiny vocabulary so they stay fast in CI;
the real 16k/32k/64k sweeps run only outside the test suite.
"""

from pathlib import Path
from typing import List

import pytest

import slm4ie.tokenizers.backends  # noqa: F401  (registers backends)
from slm4ie.tokenizers.base import TrainContext, clean_piece
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

_BACKENDS = ["bpe", "wordpiece", "unigram"]


def _train(name: str, vocab_size: int = 120):
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


class TestBackendsRegistered:
    """The backends register themselves on import."""

    def test_all_registered(self):
        """Every expected backend resolves from the registry."""
        for name in _BACKENDS:
            assert get_tokenizer(name).name == name


@pytest.mark.parametrize("name", _BACKENDS)
class TestBackendTraining:
    """Training, encoding, and persistence per backend."""

    def test_encode_non_empty(self, name: str):
        """A trained tokenizer encodes text into pieces."""
        tokenizer = _train(name)
        assert tokenizer.encode("hiša stoji") != []
        assert len(tokenizer.vocab) > 0

    def test_pieces_reconstruct_word(self, name: str):
        """Cleaned pieces of a known word concatenate back to the word."""
        tokenizer = _train(name)
        cleaned = "".join(clean_piece(p) for p in tokenizer.encode("hiše"))
        assert cleaned == "hiše"

    def test_save_load_round_trip(self, name: str, tmp_path: Path):
        """save then load reproduces encoding and vocabulary."""
        tokenizer = _train(name)
        tokenizer.save(tmp_path)

        loaded = get_tokenizer(name).load(tmp_path)
        assert loaded.encode("hiše velike") == tokenizer.encode("hiše velike")
        assert loaded.vocab == tokenizer.vocab
        assert loaded.vocab_size == tokenizer.vocab_size
