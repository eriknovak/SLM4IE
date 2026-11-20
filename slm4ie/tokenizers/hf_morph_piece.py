"""A HuggingFace slow tokenizer for MorphPiece, loadable through `AutoTokenizer`.

MorphPiece cannot be a serializable fast tokenizer: its inference is a per-word
table lookup that the `tokenizers` library has no step for. It is exported as a
`PreTrainedTokenizer` whose code travels with the export: `save_pretrained` on a
tokenizer registered for `AutoTokenizer` copies this file into the directory, and
`AutoTokenizer.from_pretrained(<dir>, trust_remote_code=True)` loads it there.
This module therefore imports nothing from this repository.

The two-path segmentation (`compose_pieces`) lives here and the training backend
calls it, so both encode text the same way. Beyond the standard slow-tokenizer
API, `encode_with_offsets` returns the token-to-character spans needed to map
encoder predictions back to the source text.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, NamedTuple

from transformers import PreTrainedTokenizer

# ===============================
# Constants
# ===============================

# the HuggingFace role of each of the project's special tokens
SPECIAL_TOKEN_ROLES = {
    "<pad>": "pad_token",
    "<unk>": "unk_token",
    "<s>": "bos_token",
    "</s>": "eos_token",
    "<mask>": "mask_token",
}

# exported files; the BPE path is not `tokenizer.json`, which AutoTokenizer reads as fast
VOCAB_FILES_NAMES = {
    "vocab_file": "morphpiece.json",
    "bpe_file": "morphpiece_bpe.json",
    "table_file": "morpheme_table.jsonl.gz",
}

# the same files as the training backend saves them in a sweep run directory
ARTIFACT_FILES = {
    "vocab_file": "morphpiece.json",
    "bpe_file": "tokenizer.json",
    "table_file": "morpheme_table.jsonl.gz",
}

# ===============================
# Helper functions
# ===============================


def compose_pieces(bpe: Any, table: dict[str, list[str]], text: str) -> list[tuple[str, int, int]]:
    """Tokenize `text` via MorphPiece's two paths, returning pieces with spans.

    A word whose surface form is in the morpheme table becomes its morpheme
    pieces; any other word keeps its byte-level BPE pieces. The BPE path also
    supplies the pre-tokenization that decides what a word is.

    Args:
        bpe: The byte-level BPE `tokenizers.Tokenizer`.
        table: Word surface form to its morpheme pieces.
        text: Input text.

    Returns:
        `(piece, char_start, char_end)` per token, with spans into `text`.
    """
    # encode with the BPE path
    encoding = bpe.encode(text)
    tokens, offsets, word_ids = encoding.tokens, encoding.offsets, encoding.word_ids
    result: list[tuple[str, int, int]] = []

    # swap in the morphemes of each word found in the table
    index = 0
    while index < len(tokens):
        word_id = word_ids[index]
        if word_id is None:
            result.append((tokens[index], offsets[index][0], offsets[index][1]))
            index += 1
            continue
        stop = index
        while stop < len(tokens) and word_ids[stop] == word_id:
            stop += 1
        word_start, word_end = offsets[index][0], offsets[stop - 1][1]
        morphemes = table.get(text[word_start:word_end])
        if morphemes is not None:
            cursor = word_start
            for morpheme in morphemes:
                result.append((morpheme, cursor, cursor + len(morpheme)))
                cursor += len(morpheme)
        else:
            result.extend((tokens[position], *offsets[position]) for position in range(index, stop))
        index = stop
    return result


def read_morpheme_table(table_file: Path) -> dict[str, list[str]]:
    """Read a gzipped JSONL morpheme table of `{"form", "morphemes"}` records.

    Args:
        table_file: The `morpheme_table.jsonl.gz` file.

    Returns:
        Word surface form to its morpheme pieces.
    """
    table: dict[str, list[str]] = {}
    with gzip.open(table_file, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                table[payload["form"]] = payload["morphemes"]
    return table


class TemplatePosition(NamedTuple):
    """One position of a sequence after a template wrapped it.

    Attributes:
        id: The token id.
        type_id: The token type id the template gives the position.
        is_special: 1 for a special token the template added, 0 for a sequence token.
    """

    id: int
    type_id: int
    is_special: int


def _expand_template(
    template: str, ids_a: list[int], ids_b: list[int], token_ids: dict[str, int]
) -> list[TemplatePosition]:
    """Expand a `TemplateProcessing`-style template over one or two sequences.

    A piece starting with `$` is a sequence slot (`$B` or `$1` the second
    sequence, any other the first); every other piece is a special token. Each
    piece may end in `:<type id>`.

    Args:
        template: Pieces such as `<s> $A </s> $B:1 </s>:1`.
        ids_a: Ids of the first sequence.
        ids_b: Ids of the second sequence.
        token_ids: Special token string to its id.

    Returns:
        One position per token.
    """
    positions: list[TemplatePosition] = []
    for piece in template.split():
        name, _, type_id = piece.partition(":")
        type_number = int(type_id) if type_id else 0
        if name.startswith("$"):
            sequence = ids_b if name in ("$B", "$1") else ids_a
            positions.extend(TemplatePosition(i, type_number, 0) for i in sequence)
        else:
            positions.append(TemplatePosition(token_ids[name], type_number, 1))
    return positions


# ===============================
# Tokenizer class
# ===============================


class MorphPieceHFTokenizer(PreTrainedTokenizer):
    """Slow HuggingFace tokenizer for a trained MorphPiece model.

    Attributes:
        vocab_files_names (dict[str, str]): The files `save_pretrained` writes.
        model_input_names (list[str]): The inputs a model takes from it.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        bpe_file: str,
        table_file: str,
        single_template: str | None = None,
        pair_template: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Load a MorphPiece model from its three files.

        Args:
            vocab_file: JSON holding the combined `vocab` and the `special_tokens`.
            bpe_file: The byte-level BPE path, a `tokenizers` JSON file.
            table_file: The gzipped JSONL morpheme table.
            single_template: Template that wraps one sequence, such as
                `<s> $A </s>`; None adds no special tokens.
            pair_template: Template that wraps a sequence pair, such as
                `<s> $A </s> $B:1 </s>:1`; None joins the pair as is.
            **kwargs: Special-token roles and other `PreTrainedTokenizer`
                settings. Roles not given default to `SPECIAL_TOKEN_ROLES`.
        """
        from tokenizers import Tokenizer, decoders

        # load the model before the base class asks for the vocabulary
        meta = json.loads(Path(vocab_file).read_text(encoding="utf-8"))
        self._vocab = {token: int(index) for token, index in meta["vocab"].items()}
        self._id_to_token = {index: token for token, index in self._vocab.items()}
        self._bpe = Tokenizer.from_file(str(bpe_file))
        self._table = read_morpheme_table(Path(table_file))
        self._morpheme_tokens = {morpheme for morphemes in self._table.values() for morpheme in morphemes}
        self._byte_decoder = decoders.ByteLevel()
        self._files = {"vocab_file": vocab_file, "bpe_file": bpe_file, "table_file": table_file}
        self.single_template = single_template
        self.pair_template = pair_template

        # default the roles of the project's special tokens
        roles = {role: token for token, role in SPECIAL_TOKEN_ROLES.items() if token in self._vocab}
        super().__init__(single_template=single_template, pair_template=pair_template, **{**roles, **kwargs})

    @classmethod
    def from_artifact(cls, artifact_dir: Path, **kwargs: Any) -> MorphPieceHFTokenizer:
        """Load the tokenizer from a sweep run directory the training backend saved.

        Args:
            artifact_dir: Directory holding the backend's MorphPiece files.
            **kwargs: Forwarded to the constructor (templates, roles, length).

        Returns:
            The tokenizer.
        """
        files = {key: str(Path(artifact_dir) / name) for key, name in ARTIFACT_FILES.items()}
        return cls(**files, **kwargs)

    @property
    def vocab_size(self) -> int:
        """Return the number of tokens in the vocabulary.

        Returns:
            Vocabulary size.
        """
        return len(self._vocab)

    def get_vocab(self) -> dict[str, int]:
        """Return the token-to-id vocabulary.

        Returns:
            Token string to integer id.
        """
        return dict(self._vocab)

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """Wrap one sequence or a pair in the special tokens the templates name.

        Args:
            token_ids_0: Ids of the first sequence.
            token_ids_1: Ids of the second sequence, if any.

        Returns:
            The ids with special tokens added.
        """
        return [position.id for position in self._expand(token_ids_0, token_ids_1)]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """Return the type id of each position, as the templates set it.

        Args:
            token_ids_0: Ids of the first sequence.
            token_ids_1: Ids of the second sequence, if any.

        Returns:
            One type id per position of `build_inputs_with_special_tokens`.
        """
        return [position.type_id for position in self._expand(token_ids_0, token_ids_1)]

    def get_special_tokens_mask(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None, already_has_special_tokens: bool = False
    ) -> list[int]:
        """Return 1 for each special-token position and 0 for each sequence position.

        Args:
            token_ids_0: Ids of the first sequence.
            token_ids_1: Ids of the second sequence, if any.
            already_has_special_tokens: Whether the ids already hold them.

        Returns:
            The mask, one value per position.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens=True)
        return [position.is_special for position in self._expand(token_ids_0, token_ids_1)]

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """Return how many special tokens the template adds.

        Args:
            pair: Whether the input is a sequence pair.

        Returns:
            The number of special tokens.
        """
        return sum(position.is_special for position in self._expand([], [] if pair else None))

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
        parts: list[str] = []
        byte_buffer: list[str] = []
        previous_morpheme = False

        def flush() -> None:
            if byte_buffer:
                parts.append(self._byte_decoder.decode(byte_buffer))
                byte_buffer.clear()

        # decode byte-level runs, space-separate morpheme runs
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
        """Return input ids and character offsets for `text`, without special tokens.

        The returned `offset_mapping` gives each token's `(char_start, char_end)`
        span in the original text, for mapping encoder outputs back to source
        spans; slow tokenizers have no `return_offsets_mapping`.

        Args:
            text: Input text.

        Returns:
            `input_ids` and `offset_mapping` lists.
        """
        spans = compose_pieces(self._bpe, self._table, text)
        return {
            "input_ids": [self._convert_token_to_id(token) for token, _start, _end in spans],
            "offset_mapping": [(start, end) for _token, start, end in spans],
        }

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str, ...]:
        """Write the vocabulary, the BPE path and the morpheme table into `save_directory`.

        Args:
            save_directory: Destination directory.
            filename_prefix: Optional prefix for the file names.

        Returns:
            The paths written.
        """
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        written: list[str] = []

        # copy each model file under its export name
        for key, name in VOCAB_FILES_NAMES.items():
            destination = directory / f"{prefix}{name}"
            source = Path(self._files[key])
            if source.resolve() != destination.resolve():
                destination.write_bytes(source.read_bytes())
            written.append(str(destination))
        return tuple(written)

    # ===============================
    # Private methods
    # ===============================

    def _expand(self, ids_a: list[int], ids_b: list[int] | None) -> list[TemplatePosition]:
        """Expand a sequence or pair through the template that applies.

        Args:
            ids_a: Ids of the first sequence.
            ids_b: Ids of the second sequence, or None for a single sequence.

        Returns:
            One position per token.
        """
        template = self.single_template if ids_b is None else self.pair_template
        if template is None:
            first = [TemplatePosition(i, 0, 0) for i in ids_a]
            return first + [TemplatePosition(i, 1, 0) for i in ids_b or []]
        return _expand_template(template, ids_a, ids_b or [], self._vocab)

    def _tokenize(self, text: str, **kwargs: Any) -> list[str]:
        """Tokenize `text` via the two MorphPiece paths.

        Args:
            text: Input text.
            **kwargs: Ignored.

        Returns:
            Token strings.
        """
        return [piece for piece, _start, _end in compose_pieces(self._bpe, self._table, text)]

    def _convert_token_to_id(self, token: str) -> int:
        """Map a token string to its id, falling back to the unk id.

        Args:
            token: A token string.

        Returns:
            The vocabulary id.
        """
        return self._vocab.get(token, self._vocab.get(self.unk_token, 0))

    def _convert_id_to_token(self, index: int) -> str:
        """Map an id back to its token string.

        Args:
            index: A vocabulary id.

        Returns:
            The token string, or the unk token if unknown.
        """
        return self._id_to_token.get(index, self.unk_token or "<unk>")
