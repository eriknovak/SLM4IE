"""Character-level WordPiece backend (HuggingFace `tokenizers`).

A WordPiece tokenizer trained over Unicode characters (NFC-normalized, no
lowercasing). Non-initial subwords carry the `##` continuation prefix, which
`clean_piece` strips for the metric harness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from slm4ie.tokenizers.base import BaseTokenizer, TrainContext
from slm4ie.tokenizers.registry import register_tokenizer


@register_tokenizer("wordpiece")
class WordPieceTokenizer(BaseTokenizer):
    """WordPiece tokenizer backed by HuggingFace `tokenizers`.

    Attributes:
        name (str): Registry key, `wordpiece`.
    """

    name = "wordpiece"

    def __init__(self, tokenizer: Any = None) -> None:
        """Wrap an optional pre-built `tokenizers.Tokenizer`.

        Args:
            tokenizer (Any): A trained `tokenizers.Tokenizer`, or None for an
                untrained instance.
        """
        super().__init__()
        self._tokenizer = tokenizer

    def train(self, corpus: Iterable[str], vocab_size: int, *, config: TrainContext) -> None:
        """Train a character-level WordPiece model over `corpus`.

        Args:
            corpus (Iterable[str]): Training sentences.
            vocab_size (int): Target vocabulary size.
            config (TrainContext): Shared training settings (special tokens).
        """
        from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

        tokenizer = Tokenizer(models.WordPiece(unk_token=self.unk_token))
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=list(config.special_tokens),
            continuing_subword_prefix="##",
            show_progress=False,
        )
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        self._tokenizer = tokenizer
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[str]:
        """Return the WordPiece pieces for `text`.

        Args:
            text (str): Input text.

        Returns:
            List[str]: Vocabulary pieces, continuation pieces prefixed `##`.
        """
        return self._tokenizer.encode(text).tokens

    @property
    def vocab(self) -> Dict[str, int]:
        """Return the token-to-id vocabulary.

        Returns:
            Dict[str, int]: Token string to integer id.
        """
        return self._tokenizer.get_vocab()

    def _save_model(self, out_dir: Path) -> None:
        """Write the tokenizer to `tokenizer.json`.

        Args:
            out_dir (Path): Destination directory.
        """
        self._tokenizer.save(str(out_dir / "tokenizer.json"))

    @classmethod
    def _load_model(cls, out_dir: Path) -> "WordPieceTokenizer":
        """Load a tokenizer from `tokenizer.json`.

        Args:
            out_dir (Path): Directory holding the saved model.

        Returns:
            WordPieceTokenizer: The reconstructed tokenizer.
        """
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(out_dir / "tokenizer.json")))
