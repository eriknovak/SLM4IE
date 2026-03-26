"""Base extractor ABC and registry for the SLM4IE pipeline."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, Type

from slm4ie.data.schema import Document

_REGISTRY: Dict[str, Type["BaseExtractor"]] = {}


class BaseExtractor(ABC):
    """Abstract base class for dataset extractors.

    Subclasses must implement :meth:`extract` to yield
    :class:`~slm4ie.data.schema.Document` instances from a given
    input directory.
    """

    @abstractmethod
    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Extracts documents from *input_dir* and yields them.

        Args:
            input_dir (Path): Directory containing the raw source data.
            source (str): Dataset key (e.g. ``"ssj500k"``).
            domain (str): Text domain (e.g. ``"web"``).

        Yields:
            Document: Extracted documents in unified schema format.
        """


def register_extractor(
    name: str,
    cls: Type[BaseExtractor],
) -> None:
    """Registers an extractor class under the given name.

    Args:
        name (str): Registry key for the extractor.
        cls (Type[BaseExtractor]): Extractor class to register.
    """
    _REGISTRY[name] = cls


def get_extractor(name: str) -> BaseExtractor:
    """Returns an instantiated extractor for the given name.

    Args:
        name (str): Registry key of the desired extractor.

    Returns:
        BaseExtractor: A new instance of the registered extractor.

    Raises:
        KeyError: If *name* is not found in the registry.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"No extractor registered under '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]()
