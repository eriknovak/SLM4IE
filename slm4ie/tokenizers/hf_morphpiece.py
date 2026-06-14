"""A HuggingFace slow tokenizer wrapping the MorphPiece backend.

MorphPiece cannot be a serializable fast tokenizer (its inference is a per-word
table lookup), so it is exported as a `PreTrainedTokenizer` that delegates
tokenization to the backend. It supports the standard slow-tokenizer API for LM
training plus `encode_with_offsets`, which returns the token-to-character spans
needed to map encoder predictions back to the source text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer

from slm4ie.tokenizers.backends.morph_piece import MorphPieceTokenizer

#: Maps the project's special tokens to the HuggingFace tokenizer roles.
SPECIAL_TOKEN_ROLES = {
    "<pad>": "pad_token",
    "<unk>": "unk_token",
    "<s>": "bos_token",
    "</s>": "eos_token",
    "<mask>": "mask_token",
}


class MorphPieceHFTokenizer(PreTrainedTokenizer):
    """Slow HuggingFace tokenizer backed by a trained MorphPiece model.

    Attributes:
        backend (MorphPieceTokenizer): The trained two-path backend.
    """

    def __init__(self, backend: MorphPieceTokenizer, **kwargs: Any) -> None:
        """Wrap a trained MorphPiece backend.

        Args:
            backend (MorphPieceTokenizer): The trained backend.
            **kwargs (Any): Forwarded to `PreTrainedTokenizer`.
        """
        from tokenizers import decoders

        self.backend = backend
        self._id_to_token = {index: token for token, index in backend.vocab.items()}
        self._morpheme_tokens = backend.morpheme_tokens
        self._byte_decoder = decoders.ByteLevel()
        roles = {role: token for token, role in SPECIAL_TOKEN_ROLES.items() if token in backend.vocab}
        roles.update(kwargs)
        super().__init__(**roles)

    @property
    def vocab_size(self) -> int:
        """Return the number of tokens in the backend vocabulary.

        Returns:
            int: Vocabulary size.
        """
        return len(self.backend.vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Return the token-to-id vocabulary.

        Returns:
            Dict[str, int]: Token string to integer id.
        """
        return dict(self.backend.vocab)

    def _tokenize(self, text: str, **kwargs: Any) -> List[str]:
        """Tokenize `text` via the MorphPiece backend.

        Args:
            text (str): Input text.
            **kwargs (Any): Ignored.

        Returns:
            List[str]: Token strings.
        """
        return self.backend.encode(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Map a token string to its id, falling back to the unk id.

        Args:
            token (str): A token string.

        Returns:
            int: The vocabulary id.
        """
        vocab = self.backend.vocab
        return vocab.get(token, vocab.get(self.unk_token, 0))

    def _convert_id_to_token(self, index: int) -> str:
        """Map an id back to its token string.

        Args:
            index (int): A vocabulary id.

        Returns:
            str: The token string, or the unk token if unknown.
        """
        return self._id_to_token.get(index, self.unk_token or "<unk>")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Best-effort detokenization of mixed morpheme and byte-level tokens.

        Byte-level pieces are decoded with the byte-level decoder; each run of
        morpheme pieces is treated as one word and prefixed with a space.
        Adjacent table-known words cannot be separated from the flat token
        stream, so exact reconstruction of such boundaries is not guaranteed;
        use `encode_with_offsets` for precise span mapping.

        Args:
            tokens (List[str]): Token strings.

        Returns:
            str: The reconstructed text.
        """
        parts: List[str] = []
        byte_buffer: List[str] = []
        previous_morpheme = False

        def flush() -> None:
            if byte_buffer:
                parts.append(self._byte_decoder.decode(byte_buffer))
                byte_buffer.clear()

        for token in tokens:
            if token in self._morpheme_tokens:
                if not previous_morpheme:
                    flush()
                    parts.append(" ")
                parts.append(token)
                previous_morpheme = True
            else:
                byte_buffer.append(token)
                previous_morpheme = False
        flush()
        return "".join(parts).strip()

    def encode_with_offsets(self, text: str) -> Dict[str, List[Any]]:
        """Return input ids and character offsets for `text`.

        This is the offset path for the morphological tokenizer: the returned
        `offset_mapping` gives each token's `(char_start, char_end)` span in the
        original text, for mapping encoder outputs back to source spans.

        Args:
            text (str): Input text.

        Returns:
            Dict[str, List[Any]]: `input_ids` and `offset_mapping` lists.
        """
        spans = self.backend.encode_offsets(text)
        return {
            "input_ids": [self._convert_token_to_id(token) for token, _start, _end in spans],
            "offset_mapping": [(start, end) for _token, start, end in spans],
        }

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, ...]:
        """Persist the backend artifacts into `save_directory`.

        Args:
            save_directory (str): Destination directory.
            filename_prefix (Optional[str]): Unused; kept for API compatibility.

        Returns:
            Tuple[str, ...]: The paths written.
        """
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.backend.save(directory)
        return (str(directory / "morphpiece.json"),)
