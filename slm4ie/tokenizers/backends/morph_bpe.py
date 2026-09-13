"""Morpheme-constrained Byte-Pair Encoding (MorphBPE, arXiv 2502.00894).

Faithful to the paper: a character-level BPE whose merge operations are
constrained during **training** to never cross a morpheme boundary, while
**inference is standard BPE** ("fully compatible with existing LLM pipelines").
Concretely, merges are learned over per-morpheme chunks (out-of-band, via
`bpe_core`) and then loaded into an ordinary `tokenizers` BPE model, so the saved
artifact is a plain `tokenizer.json` that tokenizes whole words with no morpheme
table at inference. It differs from `charbpe` only in which merges it learns.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from slm4ie.tokenizers.backends._hf_base import HfBackend, build_char_bpe_from_merges
from slm4ie.tokenizers.base import TrainContext, split_words
from slm4ie.tokenizers.bpe_core import learn_bpe
from slm4ie.tokenizers.registry import register_tokenizer


@register_tokenizer("morphbpe")
class MorphBpeTokenizer(HfBackend):
    """MorphBPE: morpheme-constrained training, standard BPE inference.

    Attributes:
        name (str): Registry key, `morphbpe`.
    """

    name = "morphbpe"

    def train(self, corpus: Iterable[str], vocab_size: int, *, config: TrainContext) -> None:
        """Learn morpheme-constrained merges, then build a standard BPE.

        Pair counts are taken over morpheme chunks for known words and over the
        whole word for out-of-vocabulary words, so no merge crosses a known
        morpheme boundary. The learned merges are loaded into a `tokenizers`
        BPE model for ordinary inference.

        Args:
            corpus (Iterable[str]): Training sentences.
            vocab_size (int): Target vocabulary size.
            config (TrainContext): Shared settings; `lexicon` is required.

        Raises:
            ValueError: If no morpheme lexicon is provided.
        """
        if config.lexicon is None:
            raise ValueError("MorphBPE requires a morpheme lexicon in TrainContext.")
        lexicon = config.lexicon

        word_freqs: Counter = Counter()
        for line in corpus:
            word_freqs.update(split_words(line))

        chunk_freqs: Counter = Counter()
        for word, freq in word_freqs.items():
            segmentation = lexicon.by_form.get(word)
            chunks = segmentation.morphemes if segmentation is not None else [word]
            for chunk in chunks:
                chunk_freqs[chunk] += freq

        budget = max(1, vocab_size - len(config.special_tokens))
        tokens, merges = learn_bpe(dict(chunk_freqs), budget)
        self._tokenizer = build_char_bpe_from_merges(
            tokens,
            merges,
            special_tokens=list(config.special_tokens),
            unk_token=self.unk_token,
        )
        self.vocab_size = vocab_size
