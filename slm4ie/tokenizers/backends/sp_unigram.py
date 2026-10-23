"""Unigram language-model backend (SentencePiece).

Trains a SentencePiece Unigram model directly from the in-memory corpus
iterator. Pieces carry the SentencePiece `▁` word-boundary marker, which
`clean_piece` strips for the metric harness.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from slm4ie.tokenizers.base import BaseTokenizer, TrainContext, keep_alphabet
from slm4ie.tokenizers.registry import register_tokenizer

#: SentencePiece reserves these control pieces itself; they are filtered out of
#: user-defined symbols to avoid duplicate-symbol training errors.
_SP_RESERVED = {"<unk>", "<s>", "</s>"}

#: SentencePiece's default `max_sentence_length`, in bytes: the trainer drops
#: any longer input line, and the sample holds one whole document per line.
MAX_SENTENCE_BYTES = 4192


def chunk_line(line: str, max_bytes: int = MAX_SENTENCE_BYTES) -> Iterator[str]:
    """Cut a line at whitespace into pieces of at most `max_bytes` UTF-8 bytes.

    A single word longer than `max_bytes` stays one piece, which the trainer
    then drops.

    Args:
        line (str): One input line, such as a whole document.
        max_bytes (int): Largest piece, in UTF-8 bytes.

    Yields:
        str: The line's words, rejoined by single spaces, piece by piece.
    """
    words: List[str] = []
    size = 0
    for word in line.split():
        width = len(word.encode("utf-8"))
        if words and size + 1 + width > max_bytes:
            yield " ".join(words)
            words, size = [], 0
        size += width + (1 if words else 0)
        words.append(word)
    if words:
        yield " ".join(words)


@register_tokenizer("unigram")
class UnigramTokenizer(BaseTokenizer):
    """Unigram tokenizer backed by SentencePiece.

    Attributes:
        name (str): Registry key, `unigram`.
    """

    name = "unigram"

    def __init__(self, model_bytes: bytes = b"") -> None:
        """Wrap an optional serialized SentencePiece model.

        Args:
            model_bytes (bytes): A serialized `.model` proto, or empty for an
                untrained instance.
        """
        super().__init__()
        self._model_bytes = model_bytes
        self._processor: Any = None
        if model_bytes:
            self._load_processor()

    def _load_processor(self) -> None:
        """Instantiate the SentencePiece processor from `_model_bytes`."""
        import sentencepiece as spm

        self._processor = spm.SentencePieceProcessor(model_proto=self._model_bytes)

    def train(self, corpus: Iterable[str], vocab_size: int, *, config: TrainContext) -> None:
        """Train a Unigram SentencePiece model over `corpus`.

        Each line is cut into pieces the trainer accepts (`chunk_line`), so long
        documents train like short ones. Characters outside `config.alphabet`
        are dropped before training and encode as the unknown token. The 64-bit suffix array keeps a
        multi-gigabyte sample within the trainer's index range.

        Args:
            corpus (Iterable[str]): Training sentences.
            vocab_size (int): Target vocabulary size.
            config (TrainContext): Shared training settings (special tokens,
                base alphabet).
        """
        import sentencepiece as spm

        user_symbols = [t for t in config.special_tokens if t not in _SP_RESERVED]
        pieces = (piece for line in corpus for piece in chunk_line(line))
        if config.alphabet is not None:
            # SentencePiece's own coverage counts spaces and NFKC forms, so it
            # would pick a different alphabet from the other backends.
            pieces = (keep_alphabet(piece, config.alphabet) for piece in pieces)
        model_writer = io.BytesIO()
        spm.SentencePieceTrainer.train(
            sentence_iterator=pieces,
            model_writer=model_writer,
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=1.0,
            unk_piece=self.unk_token,
            user_defined_symbols=user_symbols,
            hard_vocab_limit=False,
            max_sentence_length=MAX_SENTENCE_BYTES,
            train_extremely_large_corpus=True,
        )
        self._model_bytes = model_writer.getvalue()
        self._load_processor()
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[str]:
        """Return the Unigram pieces for `text`.

        Args:
            text (str): Input text.

        Returns:
            List[str]: Vocabulary pieces, word-initial pieces prefixed `▁`.
        """
        return self._processor.encode(text, out_type=str)

    @property
    def vocab(self) -> Dict[str, int]:
        """Return the token-to-id vocabulary.

        Returns:
            Dict[str, int]: Token string to integer id.
        """
        return {self._processor.id_to_piece(i): i for i in range(self._processor.get_piece_size())}

    def _save_model(self, out_dir: Path) -> None:
        """Write the serialized model proto to `spm.model`.

        Args:
            out_dir (Path): Destination directory.
        """
        (out_dir / "spm.model").write_bytes(self._model_bytes)

    @classmethod
    def _load_model(cls, out_dir: Path) -> "UnigramTokenizer":
        """Load a model from `spm.model`.

        Args:
            out_dir (Path): Directory holding the saved model.

        Returns:
            UnigramTokenizer: The reconstructed tokenizer.
        """
        return cls((out_dir / "spm.model").read_bytes())
