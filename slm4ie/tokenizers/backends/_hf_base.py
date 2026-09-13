"""Shared base for backends wrapping a HuggingFace `tokenizers.Tokenizer`.

`bpe`, `charbpe`, `wordpiece`, and `morphbpe` all persist a standard
`tokenizer.json` and run inference through the same fast `tokenizers` runtime;
only their training differs. This base supplies the common encode / offset /
vocab / save / load behavior so each backend implements just `train`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from slm4ie.tokenizers.base import BaseTokenizer


class HfBackend(BaseTokenizer):
    """Base for tokenizers backed by a `tokenizers.Tokenizer` saved as JSON.

    Attributes:
        name (str): Registry key (set by subclasses).
    """

    def __init__(self, tokenizer: Any = None) -> None:
        """Wrap an optional pre-built `tokenizers.Tokenizer`.

        Args:
            tokenizer (Any): A trained `tokenizers.Tokenizer`, or None.
        """
        super().__init__()
        self._tokenizer = tokenizer

    def encode(self, text: str) -> List[str]:
        """Return the pieces for `text`.

        Args:
            text (str): Input text.

        Returns:
            List[str]: Vocabulary pieces.
        """
        return self._tokenizer.encode(text).tokens

    def encode_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """Return each piece with its character span via the fast tokenizer.

        Args:
            text (str): Input text.

        Returns:
            List[Tuple[str, int, int]]: `(piece, char_start, char_end)` spans
                into `text`, accurate for byte-level and character-level alike.
        """
        encoding = self._tokenizer.encode(text)
        return [(token, start, end) for token, (start, end) in zip(encoding.tokens, encoding.offsets)]

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
    def _load_model(cls, out_dir: Path) -> "HfBackend":
        """Load a tokenizer from `tokenizer.json`.

        Args:
            out_dir (Path): Directory holding the saved model.

        Returns:
            HfBackend: The reconstructed tokenizer.
        """
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(out_dir / "tokenizer.json")))


def build_char_bpe_from_merges(
    tokens: List[str],
    merges: List[Tuple[str, str]],
    *,
    special_tokens: List[str],
    unk_token: str,
) -> Any:
    """Assemble a character-level HuggingFace BPE tokenizer from learned merges.

    Used by MorphBPE: merges are learned out-of-band (constrained to morpheme
    boundaries) and then loaded into a standard `tokenizers` BPE model, so
    inference is ordinary character-level BPE with a real `tokenizer.json`.

    Args:
        tokens (List[str]): Ordered vocabulary tokens (characters then merges).
        merges (List[Tuple[str, str]]): Ordered learned merge pairs.
        special_tokens (List[str]): Reserved tokens to add to the vocabulary.
        unk_token (str): Unknown-token string.

    Returns:
        Any: A configured `tokenizers.Tokenizer`.
    """
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

    vocab: Dict[str, int] = {}
    for token in [*special_tokens, *tokens]:
        if token not in vocab:
            vocab[token] = len(vocab)

    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=list(merges), unk_token=unk_token))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer
