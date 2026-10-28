"""Base extractor ABCs and extractor registry for the SLM4IE pipeline."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from slm4ie.data.schema import Document

# ===============================
# Registry
# ===============================

_REGISTRY: dict[str, type["BaseExtractor"]] = {}


# ===============================
# Extractor classes
# ===============================


class BaseExtractor(ABC):
    """Abstract base class for dataset extractors.

    Subclasses must implement extract() to yield Document instances
    from a given input directory.
    """

    @abstractmethod
    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Document]:
        """Extract documents from input_dir and yield them.

        Args:
            input_dir: Directory containing the raw source data.
            source: Dataset key (e.g. "ssj500k").
            domain: Text domain (e.g. "web").
            metadata: Optional parsed
                `metadata:` config block from extract.yaml. When given,
                extractors that support it merge per-document fields
                from an external TSV into `Document.metadata`.
                Extractors that ignore the kwarg simply pass through.

        Yields:
            Extracted documents in unified schema format.
        """


class FileBasedExtractor(BaseExtractor):
    """Base class for extractors that read a directory of files.

    Splits extraction into file enumeration (`iter_input_files`) and
    per-file-list parsing (`extract_files`) so an orchestrator can
    shard the file list across worker processes. The default
    `extract` parses every discovered file, preserving the
    single-pass `BaseExtractor` API.
    """

    @abstractmethod
    def iter_input_files(self, input_dir: Path) -> list[Path]:
        """Return the sorted input files this extractor will read.

        Args:
            input_dir: Directory containing the raw source data.

        Returns:
            Input files in deterministic (sorted) order.
                The full list is materialized (not a generator) so the
                orchestrator can take `len(files)` and split it into
                shards; for very large corpora this is O(n_files) memory.
        """

    @abstractmethod
    def extract_files(
        self,
        files: list[Path],
        source: str,
        domain: str,
        input_dir: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Document]:
        """Yield Documents parsed from an explicit list of files.

        Args:
            files: Files to parse, in order.
            source: Dataset key assigned to every Document.
            domain: Domain label assigned to every Document.
            input_dir: Dataset root, used to resolve a metadata
                table that lives alongside the input files.
            metadata: Optional `metadata:`
                config block; consumed by extractors that support it.

        Yields:
            Extracted documents in unified schema format.
        """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Document]:
        """Yield Documents from every file under input_dir.

        Args:
            input_dir: Directory containing the raw source data.
            source: Dataset key assigned to every Document.
            domain: Domain label assigned to every Document.
            metadata: Optional `metadata:`
                config block; consumed by extractors that support it.

        Yields:
            Extracted documents in unified schema format.
        """
        yield from self.extract_files(
            self.iter_input_files(input_dir),
            source,
            domain,
            input_dir,
            metadata,
        )


# ===============================
# Registry functions
# ===============================


def register_extractor(
    name: str,
    cls: type[BaseExtractor],
) -> None:
    """Register an extractor class under the given name.

    Args:
        name: Registry key for the extractor.
        cls: Extractor class to register.
    """
    _REGISTRY[name] = cls


def get_extractor(name: str) -> BaseExtractor:
    """Return an instantiated extractor for the given name.

    Args:
        name: Registry key of the desired extractor.

    Returns:
        A new instance of the registered extractor.

    Raises:
        KeyError: If name is not found in the registry.
    """
    if name not in _REGISTRY:
        raise KeyError(f"No extractor registered under '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]()
