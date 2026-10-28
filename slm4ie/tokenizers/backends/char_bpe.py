"""Character-level Byte-Pair Encoding backend (HuggingFace `tokenizers`).

A BPE tokenizer trained over Unicode characters (NFC-normalized, no lowercasing).
This is the character-level baseline that MorphBPE extends, so pairing `charbpe`
with `morphbpe` at the same vocab size isolates the effect of the morpheme
constraint (a clean ablation), separate from the byte-level `bpe` baseline.
"""

from __future__ import annotations

from collections.abc import Iterable

from slm4ie.tokenizers.backends._hf_base import HfBackend, alphabet_options
from slm4ie.tokenizers.base import TrainContext
from slm4ie.tokenizers.registry import register_tokenizer

# ===============================
# Tokenizer class
# ===============================


@register_tokenizer("charbpe")
class CharBpeTokenizer(HfBackend):
    """Character-level BPE tokenizer backed by HuggingFace `tokenizers`.

    Attributes:
        name (str): Registry key, `charbpe`.
    """

    name = "charbpe"

    def train(self, corpus: Iterable[str], vocab_size: int, *, config: TrainContext) -> None:
        """Train a character-level BPE model over `corpus`.

        Args:
            corpus: Training sentences.
            vocab_size: Target vocabulary size.
            config: Shared training settings (special tokens,
                base alphabet).
        """
        from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

        # build the tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token=self.unk_token))
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=list(config.special_tokens),
            show_progress=False,
            **alphabet_options(config),
        )

        # train and store the tokenizer
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        self._tokenizer = tokenizer
        self.vocab_size = vocab_size
