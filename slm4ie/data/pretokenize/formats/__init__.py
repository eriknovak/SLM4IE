"""Output formats of the pretokenized corpus, one registered writer per layout.

Each training project reads the corpus in its own layout, so the layout is a
pluggable `OutputFormat` chosen per registry entry. The conversion writes one
part per input shard and split, then the format merges the parts of a split
in shard order. Every format stores the same `Sample`s in the same order, and
reads them back the same way, which the `check` subcommand relies on.

Register a new layout with `@register_format("<name>")` in a module imported
below; the registry's `formats` field then accepts the name.
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, NamedTuple

import numpy as np

# ===============================
# Constants
# ===============================

# token ids up to this vocabulary size fit unsigned 16-bit integers
UINT16_VOCAB_LIMIT = 1 << 16

# ===============================
# Samples and writers
# ===============================


@dataclass(frozen=True)
class Sample:
    """One training sample: a wrapped chunk of one document.

    Attributes:
        input_ids: Token ids, special tokens included.
        id: `<document id>#<chunk index>`.
    """

    input_ids: tuple[int, ...]
    id: str

    @property
    def len(self) -> int:
        """Return the number of token ids.

        Returns:
            The sample length.
        """
        return len(self.input_ids)


class StoredRow(NamedTuple):
    """One sample as a format stores it, read back for checking.

    Attributes:
        input_ids: Token ids, as a numpy array of the stored integer type.
        len: The stored `len` column, which a loader may read instead of the ids.
        id: `<document id>#<chunk index>`.
    """

    input_ids: np.ndarray
    len: int
    id: str


def token_dtype(vocab_size: int) -> type[np.unsignedinteger]:
    """Return the smallest unsigned integer type that holds every token id.

    Args:
        vocab_size: Size of the tokenizer's vocabulary.

    Returns:
        `np.uint16` up to 65,536 tokens, else `np.uint32`.
    """
    return np.uint16 if vocab_size <= UINT16_VOCAB_LIMIT else np.uint32


class PartWriter(abc.ABC):
    """Writes the samples of one input shard and split, in order."""

    @abc.abstractmethod
    def write(self, sample: Sample) -> None:
        """Append one sample.

        Args:
            sample: The sample.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Flush and close the part; it is complete afterwards."""


class OutputFormat(abc.ABC):
    """A layout of the pretokenized corpus.

    Attributes:
        name: Registry name, also the format's folder under the output.
    """

    name: ClassVar[str]

    @abc.abstractmethod
    def part_path(self, split_dir: Path, part: str) -> Path:
        """Return where one part of a split is written.

        Args:
            split_dir: The split folder, `<output>/<format>/<split>`.
            part: The part name; parts sort in input-shard order.

        Returns:
            The part's file or folder.
        """

    @abc.abstractmethod
    def open_part(self, path: Path, vocab_size: int, shard_size_bytes: int) -> PartWriter:
        """Open a writer for one part.

        Args:
            path: The part path, as `part_path` returns it.
            vocab_size: Size of the tokenizer's vocabulary, which sets the id type.
            shard_size_bytes: Target size of one stored shard.

        Returns:
            The writer.
        """

    @abc.abstractmethod
    def merge_parts(self, split_dir: Path, parts: list[str]) -> None:
        """Join the parts of a split into what the format's readers load.

        Args:
            split_dir: The split folder.
            parts: Part names, in input-shard order.
        """

    @abc.abstractmethod
    def iter_rows(self, split_dir: Path) -> Iterator[StoredRow]:
        """Read back every stored row of a merged split, in order, without per-token Python work.

        Args:
            split_dir: The split folder.

        Yields:
            The rows, token ids as numpy arrays.
        """

    def read_split(self, split_dir: Path) -> Iterator[Sample]:
        """Read back every sample of a merged split, in order.

        Args:
            split_dir: The split folder.

        Yields:
            The samples.
        """
        for row in self.iter_rows(split_dir):
            yield Sample(input_ids=tuple(row.input_ids.tolist()), id=row.id)


# ===============================
# Registry
# ===============================

FORMATS: dict[str, OutputFormat] = {}


def register_format(name: str) -> Callable[[type[OutputFormat]], type[OutputFormat]]:
    """Register an output format class under `name`.

    Args:
        name: The registry name.

    Returns:
        A class decorator that registers one instance and returns the class.
    """

    def decorator(cls: type[OutputFormat]) -> type[OutputFormat]:
        cls.name = name
        FORMATS[name] = cls()
        return cls

    return decorator


def get_format(name: str) -> OutputFormat:
    """Return the registered format called `name`.

    Args:
        name: The registry name.

    Returns:
        The format.

    Raises:
        KeyError: If no format has that name, listing the registered ones.
    """
    if name not in FORMATS:
        raise KeyError(f"Unknown output format {name!r}; registered: {sorted(FORMATS)}")
    return FORMATS[name]


# register the built-in formats; imported last because they import this module's classes
from slm4ie.data.pretokenize.formats import mds, parquet  # noqa: E402, F401  (imported for registration)
