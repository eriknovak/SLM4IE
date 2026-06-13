"""Morpheme-aware two-path tokenizer (from scratch, MorphPiece-style).

Following the MorphPiece design (Jabbar, 2023), the vocabulary is split between
a morphological path and a statistical path. The most frequent Sloleks-derived
morphemes are kept as whole tokens (the morpheme vocabulary); the remaining
budget trains a BPE model over the uncovered morphemes and the out-of-vocabulary
words (the statistical path). At encode time a known word is segmented into its
morphemes, each emitted whole when it is in the morpheme vocabulary and BPE-split
otherwise, while an unknown word goes entirely through the BPE path.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from slm4ie.tokenizers.base import BaseTokenizer, TrainContext, split_words
from slm4ie.tokenizers.bpe_core import encode_bpe, learn_bpe, merge_ranks
from slm4ie.tokenizers.registry import register_tokenizer

#: Fraction of the (non-special) vocabulary budget reserved for whole morphemes.
_MORPHEME_BUDGET_FRACTION = 0.5


@register_tokenizer("morphpiece")
class MorphPieceTokenizer(BaseTokenizer):
    """Two-path morphological tokenizer (morpheme vocabulary + BPE fallback).

    Attributes:
        name (str): Registry key, `morphpiece`.
    """

    name = "morphpiece"

    def __init__(self) -> None:
        """Initialize an untrained MorphPiece tokenizer."""
        super().__init__()
        self._vocab: Dict[str, int] = {}
        self._ranks: Dict = {}
        self._table: Dict[str, List[str]] = {}
        self._morpheme_vocab: set = set()
        self._special_tokens: List[str] = []

    def train(self, corpus: Iterable[str], vocab_size: int, *, config: TrainContext) -> None:
        """Train the morpheme vocabulary and the BPE fallback over `corpus`.

        Args:
            corpus (Iterable[str]): Training sentences.
            vocab_size (int): Target vocabulary size.
            config (TrainContext): Shared settings; `lexicon` is required.

        Raises:
            ValueError: If no morpheme lexicon is provided.
        """
        if config.lexicon is None:
            raise ValueError("MorphPiece requires a morpheme lexicon in TrainContext.")
        lexicon = config.lexicon
        self._special_tokens = list(config.special_tokens)
        self._table = {form: list(seg.morphemes) for form, seg in lexicon.by_form.items() if len(seg.morphemes) > 1}

        word_freqs: Counter = Counter()
        for line in corpus:
            word_freqs.update(split_words(line))

        morpheme_freqs: Counter = Counter()
        for word, freq in word_freqs.items():
            seg = lexicon.by_form.get(word)
            if seg is not None:
                for morpheme in seg.morphemes:
                    morpheme_freqs[morpheme] += freq

        n_special = len(self._special_tokens)
        morph_budget = max(0, int((vocab_size - n_special) * _MORPHEME_BUDGET_FRACTION))
        self._morpheme_vocab = {m for m, _ in morpheme_freqs.most_common(morph_budget)}

        chunk_freqs: Counter = Counter()
        for word, freq in word_freqs.items():
            seg = lexicon.by_form.get(word)
            if seg is None:
                chunk_freqs[word] += freq
            else:
                for morpheme in seg.morphemes:
                    if morpheme not in self._morpheme_vocab:
                        chunk_freqs[morpheme] += freq

        bpe_budget = max(0, vocab_size - n_special - len(self._morpheme_vocab))
        tokens, merges = learn_bpe(dict(chunk_freqs), bpe_budget)
        self._ranks = merge_ranks(merges)
        self._build_vocab(tokens)
        self.vocab_size = vocab_size

    def _build_vocab(self, bpe_tokens: List[str]) -> None:
        """Assign ids to specials, morpheme tokens, then BPE tokens.

        Args:
            bpe_tokens (List[str]): Ordered BPE tokens (chars + merges).
        """
        ordered: List[str] = []
        seen = set()
        for token in [*self._special_tokens, *sorted(self._morpheme_vocab), *bpe_tokens]:
            if token not in seen:
                ordered.append(token)
                seen.add(token)
        self._vocab = {token: i for i, token in enumerate(ordered)}

    def _segment_word(self, word: str) -> Optional[List[str]]:
        """Return the morphemes for a known word, or None when unknown.

        Args:
            word (str): A surface word form.

        Returns:
            Optional[List[str]]: Morphemes for a multi-morpheme form, `[word]`
                for a single morpheme kept in the morpheme vocabulary, or None
                for an out-of-vocabulary word.
        """
        if word in self._table:
            return self._table[word]
        if word in self._morpheme_vocab:
            return [word]
        return None

    def encode(self, text: str) -> List[str]:
        """Encode `text` via the morphological and statistical paths.

        Args:
            text (str): Input text.

        Returns:
            List[str]: Whole-morpheme tokens where available, BPE pieces
                otherwise.
        """
        pieces: List[str] = []
        for word in split_words(text):
            morphemes = self._segment_word(word)
            if morphemes is None:
                pieces.extend(encode_bpe(word, self._ranks))
                continue
            for morpheme in morphemes:
                if morpheme in self._morpheme_vocab:
                    pieces.append(morpheme)
                else:
                    pieces.extend(encode_bpe(morpheme, self._ranks))
        return pieces

    @property
    def vocab(self) -> Dict[str, int]:
        """Return the token-to-id vocabulary.

        Returns:
            Dict[str, int]: Token string to integer id.
        """
        return dict(self._vocab)

    def _save_model(self, out_dir: Path) -> None:
        """Persist the vocab, merges, morpheme vocab, and boundary table.

        Args:
            out_dir (Path): Destination directory.
        """
        model = {
            "type": self.name,
            "special_tokens": self._special_tokens,
            "unk_token": self.unk_token,
            "vocab": list(self._vocab),
            "morpheme_vocab": sorted(self._morpheme_vocab),
            "merges": [list(pair) for pair, _ in sorted(self._ranks.items(), key=lambda kv: kv[1])],
        }
        (out_dir / "model.json").write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
        with gzip.open(out_dir / "morpheme_table.jsonl.gz", "wt", encoding="utf-8") as handle:
            for form, morphemes in self._table.items():
                handle.write(json.dumps({"form": form, "morphemes": morphemes}, ensure_ascii=False) + "\n")

    @classmethod
    def _load_model(cls, out_dir: Path) -> "MorphPieceTokenizer":
        """Reconstruct a MorphPiece tokenizer from saved files.

        Args:
            out_dir (Path): Directory holding the saved model.

        Returns:
            MorphPieceTokenizer: The reconstructed tokenizer.
        """
        model = json.loads((out_dir / "model.json").read_text(encoding="utf-8"))
        tokenizer = cls()
        tokenizer._special_tokens = model.get("special_tokens", [])
        tokenizer.unk_token = model.get("unk_token", tokenizer.unk_token)
        tokenizer._vocab = {token: i for i, token in enumerate(model["vocab"])}
        tokenizer._morpheme_vocab = set(model.get("morpheme_vocab", []))
        tokenizer._ranks = {tuple(pair): rank for rank, pair in enumerate(model["merges"])}
        table_path = out_dir / "morpheme_table.jsonl.gz"
        if table_path.exists():
            with gzip.open(table_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        payload = json.loads(line)
                        tokenizer._table[payload["form"]] = payload["morphemes"]
        return tokenizer
