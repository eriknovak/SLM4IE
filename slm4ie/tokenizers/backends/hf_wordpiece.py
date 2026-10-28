"""WordPiece backend (HuggingFace `tokenizers`), BERT-style.

WordPiece as used by BERT: character-level over Unicode (NFC, no lowercasing),
non-initial subwords carry the `##` continuation prefix, and there is no
byte-level or space-in-token scheme (faithful to the original work). The metric
harness uses the fast tokenizer's offset mapping.
"""

from __future__ import annotations

from collections.abc import Iterable

from slm4ie.tokenizers.backends._hf_base import HfBackend, alphabet_options
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
            corpus: Training sentences.
            vocab_size: Target vocabulary size.
            config: Shared training settings (special tokens,
                base alphabet).
        """
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

        # build the WordPiece pipeline
        tokenizer = Tokenizer(models.WordPiece(unk_token=self.unk_token))
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        # train the model
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=list(config.special_tokens),
            continuing_subword_prefix="##",
            show_progress=False,
            **alphabet_options(config),
        )
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        self._tokenizer = tokenizer
        self.vocab_size = vocab_size
