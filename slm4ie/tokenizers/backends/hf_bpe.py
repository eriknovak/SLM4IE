"""Byte-level Byte-Pair Encoding backend (HuggingFace `tokenizers`).

The canonical modern BPE, as used by GPT-2 / RoBERTa / ModernBERT: byte-level
pre-tokenization with the `Ġ` space marker, so whitespace is encoded into tokens
and encode/decode round-trips. Pieces live in byte-mapped space; the metric
harness relies on the fast tokenizer's offset mapping rather than the piece
strings, so non-ASCII characters are handled correctly.
"""

from __future__ import annotations

from typing import Iterable

from slm4ie.tokenizers.backends._hf_base import HfBackend
from slm4ie.tokenizers.base import TrainContext
from slm4ie.tokenizers.registry import register_tokenizer


@register_tokenizer("bpe")
class BpeTokenizer(HfBackend):
    """Byte-level BPE tokenizer backed by HuggingFace `tokenizers`.

    Attributes:
        name (str): Registry key, `bpe`.
    """

    name = "bpe"

    def train(self, corpus: Iterable[str], vocab_size: int, *, config: TrainContext) -> None:
        """Train a byte-level BPE model over `corpus`.

        Args:
            corpus (Iterable[str]): Training sentences.
            vocab_size (int): Target vocabulary size.
            config (TrainContext): Shared training settings (special tokens).
        """
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True, trim_offsets=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=list(config.special_tokens),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        self._tokenizer = tokenizer
        self.vocab_size = vocab_size
