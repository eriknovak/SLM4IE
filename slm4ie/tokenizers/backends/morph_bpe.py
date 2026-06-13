"""Morpheme-gated Byte-Pair Encoding (from scratch).

MorphBPE pre-segments each known word into its Sloleks-derived morphemes and
learns BPE merges only within morphemes, so a merge never crosses a morpheme
boundary of a known word. Words absent from the lexicon are treated as a single
chunk and undergo ordinary BPE, which is the out-of-vocabulary fallback. The
result is a standard BPE merge table plus a morpheme boundary table replayed at
encode time.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from slm4ie.tokenizers.base import BaseTokenizer, TrainContext, split_words
from slm4ie.tokenizers.bpe_core import encode_bpe, learn_bpe, merge_ranks
from slm4ie.tokenizers.registry import register_tokenizer


@register_tokenizer("morphbpe")
class MorphBpeTokenizer(BaseTokenizer):
    """Morpheme-bounded BPE tokenizer.

    Attributes:
        name (str): Registry key, `morphbpe`.
    """

    name = "morphbpe"

    def __init__(self) -> None:
        """Initialize an untrained MorphBPE tokenizer."""
        super().__init__()
        self._vocab: Dict[str, int] = {}
        self._ranks: Dict = {}
        self._table: Dict[str, List[str]] = {}
        self._special_tokens: List[str] = []

    def _pre_segment(self, word: str) -> List[str]:
        """Return the morpheme chunks for `word`, or the whole word.

        Args:
            word (str): A surface word form.

        Returns:
            List[str]: Gold morpheme chunks for known words, else `[word]`.
        """
        return self._table.get(word, [word])

    def train(self, corpus: Iterable[str], vocab_size: int, *, config: TrainContext) -> None:
        """Train morpheme-bounded BPE merges over `corpus`.

        Args:
            corpus (Iterable[str]): Training sentences.
            vocab_size (int): Target vocabulary size.
            config (TrainContext): Shared settings; `lexicon` is required.

        Raises:
            ValueError: If no morpheme lexicon is provided.
        """
        if config.lexicon is None:
            raise ValueError("MorphBPE requires a morpheme lexicon in TrainContext.")
        self._special_tokens = list(config.special_tokens)
        self._table = {
            form: list(seg.morphemes) for form, seg in config.lexicon.by_form.items() if len(seg.morphemes) > 1
        }

        word_freqs: Counter = Counter()
        for line in corpus:
            word_freqs.update(split_words(line))

        chunk_freqs: Counter = Counter()
        for word, freq in word_freqs.items():
            for chunk in self._pre_segment(word):
                chunk_freqs[chunk] += freq

        budget = max(0, vocab_size - len(self._special_tokens))
        tokens, merges = learn_bpe(dict(chunk_freqs), budget)
        self._ranks = merge_ranks(merges)
        self._build_vocab(tokens)
        self.vocab_size = vocab_size

    def _build_vocab(self, tokens: List[str]) -> None:
        """Assign ids to special tokens followed by learned tokens.

        Args:
            tokens (List[str]): Ordered learned tokens (chars + merges).
        """
        ordered: List[str] = []
        seen = set()
        for token in [*self._special_tokens, *tokens]:
            if token not in seen:
                ordered.append(token)
                seen.add(token)
        self._vocab = {token: i for i, token in enumerate(ordered)}

    def encode(self, text: str) -> List[str]:
        """Encode `text`, splitting known words at morpheme boundaries first.

        Args:
            text (str): Input text.

        Returns:
            List[str]: BPE pieces, never crossing a known morpheme boundary.
        """
        pieces: List[str] = []
        for word in split_words(text):
            for chunk in self._pre_segment(word):
                pieces.extend(encode_bpe(chunk, self._ranks))
        return pieces

    @property
    def vocab(self) -> Dict[str, int]:
        """Return the token-to-id vocabulary.

        Returns:
            Dict[str, int]: Token string to integer id.
        """
        return dict(self._vocab)

    def _save_model(self, out_dir: Path) -> None:
        """Persist the merge table, vocab, and morpheme boundary table.

        Args:
            out_dir (Path): Destination directory.
        """
        model = {
            "type": self.name,
            "special_tokens": self._special_tokens,
            "unk_token": self.unk_token,
            "vocab": list(self._vocab),
            "merges": [list(pair) for pair, _ in sorted(self._ranks.items(), key=lambda kv: kv[1])],
        }
        (out_dir / "model.json").write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
        with gzip.open(out_dir / "morpheme_table.jsonl.gz", "wt", encoding="utf-8") as handle:
            for form, morphemes in self._table.items():
                handle.write(json.dumps({"form": form, "morphemes": morphemes}, ensure_ascii=False) + "\n")

    @classmethod
    def _load_model(cls, out_dir: Path) -> "MorphBpeTokenizer":
        """Reconstruct a MorphBPE tokenizer from saved files.

        Args:
            out_dir (Path): Directory holding the saved model.

        Returns:
            MorphBpeTokenizer: The reconstructed tokenizer.
        """
        model = json.loads((out_dir / "model.json").read_text(encoding="utf-8"))
        tokenizer = cls()
        tokenizer._special_tokens = model.get("special_tokens", [])
        tokenizer.unk_token = model.get("unk_token", tokenizer.unk_token)
        tokenizer._vocab = {token: i for i, token in enumerate(model["vocab"])}
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
