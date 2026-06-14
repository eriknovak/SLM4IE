"""Tests for slm4ie/tokenizers/base.py."""

from pathlib import Path
from typing import Dict, Iterable, List

from slm4ie.tokenizers.base import (
    BaseTokenizer,
    TokenizerSpec,
    TrainContext,
    clean_piece,
)


class _CharTokenizer(BaseTokenizer):
    """Minimal character tokenizer used to exercise BaseTokenizer plumbing."""

    name = "char"

    def __init__(self) -> None:
        """Initialize with an empty character vocabulary."""
        super().__init__()
        self._vocab: Dict[str, int] = {}

    def train(self, corpus: Iterable[str], vocab_size: int, *, config: TrainContext) -> None:
        """Build a vocabulary from the unique characters in `corpus`."""
        chars = sorted({ch for text in corpus for ch in text})
        tokens = list(config.special_tokens) + chars
        self._vocab = {tok: i for i, tok in enumerate(tokens[:vocab_size])}
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[str]:
        """Return the characters of `text`."""
        return list(text)

    @property
    def vocab(self) -> Dict[str, int]:
        """Return the character vocabulary."""
        return dict(self._vocab)

    def _save_model(self, out_dir: Path) -> None:
        """Write the vocabulary as newline-separated tokens."""
        (out_dir / "vocab.txt").write_text("\n".join(self._vocab), encoding="utf-8")

    @classmethod
    def _load_model(cls, out_dir: Path) -> "_CharTokenizer":
        """Reconstruct the vocabulary from `vocab.txt`."""
        tok = cls()
        tokens = (out_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
        tok._vocab = {t: i for i, t in enumerate(tokens)}
        return tok


class TestCleanPiece:
    """Unit tests for clean_piece marker stripping."""

    def test_strips_bytelevel_marker(self):
        """The ByteLevel `Ġ` space marker is removed."""
        assert clean_piece("Ġhiša") == "hiša"

    def test_strips_sentencepiece_marker(self):
        """The SentencePiece `▁` space marker is removed."""
        assert clean_piece("▁pes") == "pes"

    def test_strips_wordpiece_continuation(self):
        """A leading WordPiece `##` is removed."""
        assert clean_piece("##ami") == "ami"

    def test_plain_piece_unchanged(self):
        """A piece without markers passes through unchanged."""
        assert clean_piece("hiš") == "hiš"


class TestBaseTokenizer:
    """Tests for the shared BaseTokenizer scaffolding."""

    def test_satisfies_tokenizer_spec(self):
        """A concrete subclass satisfies the runtime-checkable protocol."""
        assert isinstance(_CharTokenizer(), TokenizerSpec)

    def test_encode_ids_uses_vocab(self):
        """encode_ids maps known pieces and falls back to the unk id."""
        tok = _CharTokenizer()
        tok.train(["ab"], vocab_size=10, config=TrainContext(special_tokens=["<unk>"]))
        ids = tok.encode_ids("abc")
        vocab = tok.vocab
        # 'a' and 'b' are known; 'c' is unknown -> mapped to <unk>.
        assert ids[:2] == [vocab["a"], vocab["b"]]
        assert ids[2] == vocab["<unk>"]

    def test_save_load_round_trip(self, tmp_path: Path):
        """Save then load reproduces the vocabulary and metadata."""
        tok = _CharTokenizer()
        tok.train(["hiša"], vocab_size=20, config=TrainContext(special_tokens=["<unk>"]))
        tok.save(tmp_path)
        assert (tmp_path / "metadata.json").exists()

        loaded = _CharTokenizer.load(tmp_path)
        assert loaded.vocab == tok.vocab
        assert loaded.vocab_size == 20
        assert loaded.encode("hiša") == tok.encode("hiša")
