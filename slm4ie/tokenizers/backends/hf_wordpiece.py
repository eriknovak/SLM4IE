"""WordPiece backend (HuggingFace `tokenizers`), BERT-style.

WordPiece as used by BERT: character-level over Unicode (NFC, no lowercasing),
non-initial subwords carry the `##` continuation prefix, and there is no
byte-level or space-in-token scheme (faithful to the original work). The metric
harness uses the fast tokenizer's offset mapping.
"""

from __future__ import annotations

from typing import Iterable

from slm4ie.tokenizers.backends._hf_base import HfBackend
from slm4ie.tokenizers.base import TrainContext
from slm4ie.tokenizers.registry import register_tokenizer


@register_tokenizer("wordpiece")
class WordPieceTokenizer(HfBackend):
    """WordPiece tokenizer backed by HuggingFace `tokenizers`.

    Attributes:
        name (str): Registry key, `wordpiece`.
    """

    name = "wordpiece"

    def train(self, corpus: Iterable[str], vocab_size: int, *, config: TrainContext) -> None:
        """Train a character-level WordPiece model over `corpus`.

        Args:
            corpus (Iterable[str]): Training sentences.
            vocab_size (int): Target vocabulary size.
            config (TrainContext): Shared training settings (special tokens).
        """
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

        tokenizer = Tokenizer(models.WordPiece(unk_token=self.unk_token))
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.WordPiece(prefix="##")
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=list(config.special_tokens),
            continuing_subword_prefix="##",
            show_progress=False,
        )
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        self._tokenizer = tokenizer
        self.vocab_size = vocab_size
