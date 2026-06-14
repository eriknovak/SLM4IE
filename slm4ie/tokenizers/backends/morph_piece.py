"""Morpheme-aware two-path tokenizer (MorphPiece, arXiv 2307.07262).

Following Jabbar (2023): a word found in the MorphTable is emitted as its
morpheme tokens (the morphological path); any other word is tokenized by a
GPT-2-style **byte-level BPE** (the statistical path). The byte-level BPE also
supplies pre-tokenization, offsets, and word grouping, so this backend layers the
MorphTable swap on top of a HuggingFace `tokenizers` byte-level BPE. The
MorphTable is Sloleks-derived (substituting the paper's MorphyNet for English).

Inference is genuinely non-standard (a per-word table lookup), so this is the one
backend exported as a custom slow `PreTrainedTokenizer` rather than a fast one.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from slm4ie.tokenizers.base import BaseTokenizer, TrainContext, split_words
from slm4ie.tokenizers.registry import register_tokenizer

#: Fraction of the (non-special) vocabulary budget reserved for whole morphemes.
_MORPHEME_BUDGET_FRACTION = 0.5


@register_tokenizer("morphpiece")
class MorphPieceTokenizer(BaseTokenizer):
    """Two-path morphological tokenizer over a byte-level BPE statistical path.

    Attributes:
        name (str): Registry key, `morphpiece`.
    """

    name = "morphpiece"

    def __init__(self, bpe: Any = None) -> None:
        """Wrap an optional byte-level BPE `tokenizers.Tokenizer`.

        Args:
            bpe (Any): A trained byte-level BPE tokenizer, or None.
        """
        super().__init__()
        self._bpe = bpe
        self._table: Dict[str, List[str]] = {}
        self._vocab: Dict[str, int] = {}
        self._special_tokens: List[str] = []

    def train(self, corpus: Iterable[str], vocab_size: int, *, config: TrainContext) -> None:
        """Train the byte-level BPE path and select the morpheme vocabulary.

        Args:
            corpus (Iterable[str]): Training sentences.
            vocab_size (int): Target total vocabulary size.
            config (TrainContext): Shared settings; `lexicon` is required.

        Raises:
            ValueError: If no morpheme lexicon is provided.
        """
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        if config.lexicon is None:
            raise ValueError("MorphPiece requires a morpheme lexicon in TrainContext.")
        lexicon = config.lexicon
        self._special_tokens = list(config.special_tokens)
        sample = list(corpus)

        # Rank candidate morphemes by corpus frequency of the words using them.
        word_freqs: Counter = Counter()
        for line in sample:
            word_freqs.update(split_words(line))
        morpheme_freqs: Counter = Counter()
        for word, freq in word_freqs.items():
            seg = lexicon.by_form.get(word)
            if seg is not None:
                for morpheme in seg.morphemes:
                    morpheme_freqs[morpheme] += freq

        n_special = len(self._special_tokens)
        morph_budget = max(0, int((vocab_size - n_special) * _MORPHEME_BUDGET_FRACTION))
        kept = {m for m, _ in morpheme_freqs.most_common(morph_budget)}
        # Only keep forms whose every morpheme is a kept token; others fall to BPE.
        self._table = {
            form: list(seg.morphemes)
            for form, seg in lexicon.by_form.items()
            if len(seg.morphemes) > 1 and all(m in kept for m in seg.morphemes)
        }

        bpe_budget = max(n_special + 256, vocab_size - len(kept))
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True, trim_offsets=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=bpe_budget,
            special_tokens=self._special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator(sample, trainer=trainer)
        self._bpe = tokenizer
        self._build_vocab(sorted(kept))
        self.vocab_size = vocab_size

    def _build_vocab(self, morpheme_tokens: List[str]) -> None:
        """Combine the BPE vocabulary with the morpheme tokens.

        Args:
            morpheme_tokens (List[str]): Kept morpheme token strings.
        """
        vocab = dict(self._bpe.get_vocab())
        next_id = (max(vocab.values()) + 1) if vocab else 0
        for morpheme in morpheme_tokens:
            if morpheme not in vocab:
                vocab[morpheme] = next_id
                next_id += 1
        self._vocab = vocab

    def _compose(self, text: str) -> List[Tuple[str, int, int]]:
        """Tokenize `text` via the two paths, returning pieces with spans.

        Args:
            text (str): Input text.

        Returns:
            List[Tuple[str, int, int]]: `(piece, char_start, char_end)` where
                known words yield morpheme pieces and others yield byte-level
                BPE pieces.
        """
        encoding = self._bpe.encode(text)
        tokens, offsets, word_ids = encoding.tokens, encoding.offsets, encoding.word_ids
        result: List[Tuple[str, int, int]] = []
        index = 0
        count = len(tokens)
        while index < count:
            word_id = word_ids[index]
            if word_id is None:
                result.append((tokens[index], offsets[index][0], offsets[index][1]))
                index += 1
                continue
            stop = index
            while stop < count and word_ids[stop] == word_id:
                stop += 1
            word_start, word_end = offsets[index][0], offsets[stop - 1][1]
            surface = text[word_start:word_end]
            morphemes = self._table.get(surface)
            if morphemes is not None:
                cursor = word_start
                for morpheme in morphemes:
                    result.append((morpheme, cursor, cursor + len(morpheme)))
                    cursor += len(morpheme)
            else:
                for position in range(index, stop):
                    result.append((tokens[position], offsets[position][0], offsets[position][1]))
            index = stop
        return result

    def encode(self, text: str) -> List[str]:
        """Return the two-path pieces for `text`.

        Args:
            text (str): Input text.

        Returns:
            List[str]: Morpheme pieces for known words, byte-level BPE pieces
                otherwise.
        """
        return [piece for piece, _start, _end in self._compose(text)]

    def encode_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """Return each piece with its character span in `text`.

        Args:
            text (str): Input text.

        Returns:
            List[Tuple[str, int, int]]: `(piece, char_start, char_end)` spans.
        """
        return self._compose(text)

    @property
    def vocab(self) -> Dict[str, int]:
        """Return the combined token-to-id vocabulary.

        Returns:
            Dict[str, int]: Token string to integer id.
        """
        return dict(self._vocab)

    @property
    def morpheme_tokens(self) -> set:
        """Return the vocabulary tokens that are whole morphemes.

        These are the plain-surface tokens added on top of the byte-level BPE
        vocabulary; the decoder uses them to tell a morpheme piece (surface)
        from a byte-level BPE piece (byte-mapped).

        Returns:
            set: Morpheme token strings.
        """
        return set(self._vocab) - set(self._bpe.get_vocab())

    def _save_model(self, out_dir: Path) -> None:
        """Persist the BPE tokenizer, the MorphTable, and the combined vocab.

        Args:
            out_dir (Path): Destination directory.
        """
        self._bpe.save(str(out_dir / "tokenizer.json"))
        meta = {"special_tokens": self._special_tokens, "vocab": self._vocab}
        (out_dir / "morphpiece.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
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
        from tokenizers import Tokenizer

        tokenizer = cls(Tokenizer.from_file(str(out_dir / "tokenizer.json")))
        meta = json.loads((out_dir / "morphpiece.json").read_text(encoding="utf-8"))
        tokenizer._special_tokens = meta.get("special_tokens", [])
        tokenizer._vocab = {token: int(i) for token, i in meta["vocab"].items()}
        with gzip.open(out_dir / "morpheme_table.jsonl.gz", "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    payload = json.loads(line)
                    tokenizer._table[payload["form"]] = payload["morphemes"]
        return tokenizer
