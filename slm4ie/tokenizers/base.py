"""Common tokenizer interface shared by every backend.

Defines the contract all tokenizers in this package implement so they plug
uniformly into the registry, the training sweep, and the metric harness. It
also provides `TrainContext` (training-time settings handed to each backend)
and `clean_piece`, which normalizes backend-specific subword markers so the
metrics compare plain surface substrings regardless of the tokenizer family.
"""

from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from slm4ie.tokenizers.morphology import MorphLexicon

#: ByteLevel space marker (`Ġ`) and SentencePiece space marker (`▁`). Both
#: encode a leading word boundary and are stripped by `clean_piece`.
_SPACE_MARKERS = ("Ġ", "▁")

#: WordPiece continuation prefix marking a non-initial subword.
_CONTINUATION_PREFIX = "##"

#: Splits text into word runs (Unicode letters/digits, covering č/š/ž) and
#: individual non-space symbols. Shared by the morphological backends and the
#: metric harness so every component agrees on what counts as a "word".
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def split_words(text: str) -> List[str]:
    """Split `text` into word and punctuation tokens.

    Args:
        text (str): Input text.

    Returns:
        List[str]: Word runs and individual non-space symbols, in order.
    """
    return _WORD_RE.findall(text)


@dataclass
class TrainContext:
    """Training-time settings passed to every backend's `train`.

    Attributes:
        special_tokens (List[str]): Reserved tokens added to the vocabulary
            (e.g. `<pad>`, `<unk>`).
        seed (int): Seed for any stochastic step, kept for reproducibility.
        lexicon (Optional[MorphLexicon]): Sloleks-derived morpheme lexicon.
            Required by the morphological backends; None for the others.
        extra (Dict[str, Any]): Backend-specific overrides, rarely needed.
    """

    special_tokens: List[str] = field(default_factory=list)
    seed: int = 0
    lexicon: Optional["MorphLexicon"] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def clean_piece(piece: str) -> str:
    """Strip backend-specific subword markers from a token piece.

    Removes a leading WordPiece `##` continuation prefix and the ByteLevel
    (`Ġ`) / SentencePiece (`▁`) space markers, leaving the plain surface
    substring the piece covers.

    Args:
        piece (str): A raw vocabulary token as emitted by a backend.

    Returns:
        str: The piece with continuation and space markers removed.
    """
    if piece.startswith(_CONTINUATION_PREFIX):
        piece = piece[len(_CONTINUATION_PREFIX) :]
    for marker in _SPACE_MARKERS:
        piece = piece.replace(marker, "")
    return piece


@runtime_checkable
class TokenizerSpec(Protocol):
    """Structural contract implemented by every tokenizer backend.

    Attributes:
        name (str): Registry key for the backend.
    """

    name: str

    def train(self, corpus: Iterable[str], vocab_size: int, *, config: TrainContext) -> None:
        """Train the tokenizer to `vocab_size` over `corpus`."""
        ...

    def encode(self, text: str) -> List[str]:
        """Return the raw vocabulary pieces for `text`."""
        ...

    def encode_ids(self, text: str) -> List[int]:
        """Return the vocabulary ids for `text`."""
        ...

    def encode_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """Return each piece with its `(start, end)` char span in `text`."""
        ...

    def save(self, out_dir: Path) -> None:
        """Persist the tokenizer under `out_dir`."""
        ...

    @classmethod
    def load(cls, out_dir: Path) -> "TokenizerSpec":
        """Load a tokenizer previously saved under `out_dir`."""
        ...

    @property
    def vocab(self) -> Dict[str, int]:
        """Return the token-to-id vocabulary."""
        ...


class BaseTokenizer(abc.ABC):
    """Abstract base supplying shared save/load and id-mapping scaffolding.

    Concrete backends implement `train`, `encode`, `vocab`, `_save_model`,
    and `_load_model`. This base provides the `metadata.json` sidecar, the
    `save`/`load` wiring, and a default `encode_ids` derived from `vocab`.

    Attributes:
        name (str): Registry key for the backend (overridden by subclasses).
        vocab_size (int): Target vocabulary size the tokenizer was trained for.
        unk_token (str): Token used for out-of-vocabulary pieces in
            `encode_ids`.
    """

    name: str = "base"

    def __init__(self) -> None:
        """Initialize an untrained tokenizer."""
        self.vocab_size: int = 0
        self.unk_token: str = "<unk>"

    @abc.abstractmethod
    def train(self, corpus: Iterable[str], vocab_size: int, *, config: TrainContext) -> None:
        """Train the tokenizer to `vocab_size` over the text in `corpus`.

        Args:
            corpus (Iterable[str]): Training sentences/documents.
            vocab_size (int): Target vocabulary size.
            config (TrainContext): Shared training-time settings.
        """

    @abc.abstractmethod
    def encode(self, text: str) -> List[str]:
        """Segment `text` into raw vocabulary pieces.

        Args:
            text (str): Input text.

        Returns:
            List[str]: Vocabulary pieces, markers included.
        """

    @property
    @abc.abstractmethod
    def vocab(self) -> Dict[str, int]:
        """Return the token-to-id vocabulary.

        Returns:
            Dict[str, int]: Mapping from token string to integer id.
        """

    @abc.abstractmethod
    def _save_model(self, out_dir: Path) -> None:
        """Write the backend-specific model files into `out_dir`.

        Args:
            out_dir (Path): Destination directory (already created).
        """

    @classmethod
    @abc.abstractmethod
    def _load_model(cls, out_dir: Path) -> "BaseTokenizer":
        """Reconstruct a tokenizer from files written by `_save_model`.

        Args:
            out_dir (Path): Directory holding the saved model.

        Returns:
            BaseTokenizer: The reconstructed tokenizer.
        """

    def encode_ids(self, text: str) -> List[int]:
        """Map the pieces of `text` to vocabulary ids.

        Unknown pieces are mapped to the id of `unk_token` when present and
        skipped otherwise.

        Args:
            text (str): Input text.

        Returns:
            List[int]: Vocabulary ids for the encoded pieces.
        """
        vocab = self.vocab
        unk_id = vocab.get(self.unk_token)
        ids: List[int] = []
        for piece in self.encode(text):
            if piece in vocab:
                ids.append(vocab[piece])
            elif unk_id is not None:
                ids.append(unk_id)
        return ids

    def encode_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """Return each encoded piece with its character span in `text`.

        The default aligns cleaned pieces to `text` left to right, which is
        correct for character-level and metaspace schemes. Byte-level backends
        (whose pieces are not surface substrings) override this with the
        tokenizer's own offset mapping.

        Args:
            text (str): Input text.

        Returns:
            List[Tuple[str, int, int]]: `(piece, char_start, char_end)` per
                token, with spans into the original `text`.
        """
        spans: List[Tuple[str, int, int]] = []
        cursor = 0
        for piece in self.encode(text):
            surface = clean_piece(piece)
            index = text.find(surface, cursor) if surface else cursor
            if index < 0:
                index = cursor
            end = index + len(surface)
            spans.append((piece, index, end))
            cursor = end
        return spans

    def metadata(self) -> Dict[str, Any]:
        """Return the JSON-serializable metadata sidecar payload.

        Returns:
            Dict[str, Any]: Identity fields persisted alongside the model.
        """
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "actual_vocab_size": len(self.vocab),
            "unk_token": self.unk_token,
        }

    def save(self, out_dir: Path) -> None:
        """Persist the model and its metadata sidecar under `out_dir`.

        Args:
            out_dir (Path): Destination directory, created if missing.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._save_model(out_dir)
        (out_dir / "metadata.json").write_text(
            json.dumps(self.metadata(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, out_dir: Path) -> "BaseTokenizer":
        """Load a tokenizer previously saved under `out_dir`.

        Args:
            out_dir (Path): Directory holding the saved model and metadata.

        Returns:
            BaseTokenizer: The reconstructed tokenizer with metadata applied.
        """
        out_dir = Path(out_dir)
        tokenizer = cls._load_model(out_dir)
        meta_path = out_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            tokenizer.vocab_size = meta.get("vocab_size", tokenizer.vocab_size)
            tokenizer.unk_token = meta.get("unk_token", tokenizer.unk_token)
        return tokenizer
