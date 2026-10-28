"""A HuggingFace slow tokenizer wrapping the MorphPiece backend.

MorphPiece cannot be a serializable fast tokenizer (its inference is a per-word
table lookup), so it is exported as a `PreTrainedTokenizer` that delegates
tokenization to the backend. It supports the standard slow-tokenizer API for LM
training plus `encode_with_offsets`, which returns the token-to-character spans
needed to map encoder predictions back to the source text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizer

from slm4ie.tokenizers.backends.morph_piece import MorphPieceTokenizer

# ===============================
# Defaults
# ===============================

#: Maps the project's special tokens to the HuggingFace tokenizer roles.
SPECIAL_TOKEN_ROLES = {
    "<pad>": "pad_token",
    "<unk>": "unk_token",
    "<s>": "bos_token",
    "</s>": "eos_token",
    "<mask>": "mask_token",
}

# ===============================
# Tokenizer class
# ===============================


class MorphPieceHFTokenizer(PreTrainedTokenizer):
    """Slow HuggingFace tokenizer backed by a trained MorphPiece model.

    Attributes:
        backend (MorphPieceTokenizer): The trained two-path backend.
    """

    def __init__(self, backend: MorphPieceTokenizer, **kwargs: Any) -> None:
        """Wrap a trained MorphPiece backend.

        Args:
            backend: The trained backend.
            **kwargs: Forwarded to `PreTrainedTokenizer`.
        """
        from tokenizers import decoders

        # store the backend and its lookups
        self.backend = backend
        self._id_to_token = {index: token for token, index in backend.vocab.items()}
        self._morpheme_tokens = backend.morpheme_tokens
        self._byte_decoder = decoders.ByteLevel()

        # map the special tokens to their roles
        roles = {role: token for token, role in SPECIAL_TOKEN_ROLES.items() if token in backend.vocab}
        roles.update(kwargs)
        super().__init__(**roles)

    @property
    def vocab_size(self) -> int:
        """Return the number of tokens in the backend vocabulary.

        Returns:
            Vocabulary size.
        """
        return len(self.backend.vocab)

    def get_vocab(self) -> dict[str, int]:
        """Return the token-to-id vocabulary.

        Returns:
            Token string to integer id.
        """
        return dict(self.backend.vocab)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Best-effort detokenization of mixed morpheme and byte-level tokens.

        Byte-level pieces are decoded with the byte-level decoder; each run of
        morpheme pieces is treated as one word and prefixed with a space.
        Adjacent table-known words cannot be separated from the flat token
        stream, so exact reconstruction of such boundaries is not guaranteed;
        use `encode_with_offsets` for precise span mapping.

        Args:
            tokens: Token strings.

        Returns:
            The reconstructed text.
        """
        # group byte-level and morpheme pieces
        parts: list[str] = []
        byte_buffer: list[str] = []
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

    def encode_with_offsets(self, text: str) -> dict[str, list[Any]]:
        """Return input ids and character offsets for `text`.

        This is the offset path for the morphological tokenizer: the returned
        `offset_mapping` gives each token's `(char_start, char_end)` span in the
        original text, for mapping encoder outputs back to source spans.

        Args:
            text: Input text.

        Returns:
            `input_ids` and `offset_mapping` lists.
        """
        spans = self.backend.encode_offsets(text)
        return {
            "input_ids": [self._convert_token_to_id(token) for token, _start, _end in spans],
            "offset_mapping": [(start, end) for _token, start, end in spans],
        }

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str, ...]:
        """Persist the backend artifacts into `save_directory`.

        Args:
            save_directory: Destination directory.
            filename_prefix: Unused; kept for API compatibility.

        Returns:
            The paths written.
        """
        # save the backend
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.backend.save(directory)
        return (str(directory / "morphpiece.json"),)

    # ===============================
    # Private methods
    # ===============================

    def _tokenize(self, text: str, **kwargs: Any) -> list[str]:
        """Tokenize `text` via the MorphPiece backend.

        Args:
            text: Input text.
            **kwargs: Ignored.

        Returns:
            Token strings.
        """
        return self.backend.encode(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Map a token string to its id, falling back to the unk id.

        Args:
            token: A token string.

        Returns:
            The vocabulary id.
        """
        vocab = self.backend.vocab
        return vocab.get(token, vocab.get(self.unk_token, 0))

    def _convert_id_to_token(self, index: int) -> str:
        """Map an id back to its token string.

        Args:
            index: A vocabulary id.

        Returns:
            The token string, or the unk token if unknown.
        """
        return self._id_to_token.get(index, self.unk_token or "<unk>")
