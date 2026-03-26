"""Tests for BaseExtractor ABC and extractor registry."""

from pathlib import Path
from typing import Iterator

import pytest

from slm4ie.data.extractors import (
    BaseExtractor,
    get_extractor,
    register_extractor,
)
from slm4ie.data.schema import Document


class _DummyExtractor(BaseExtractor):
    """Concrete extractor that yields a single fixed Document."""

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Yields one Document with the given source and domain.

        Args:
            input_dir (Path): Input directory (unused).
            source (str): Source identifier.
            domain (str): Domain identifier.

        Yields:
            Document: A single document.
        """
        yield Document(text="hello", source=source, domain=domain)


class TestBaseExtractor:
    """Tests for BaseExtractor abstract base class."""

    def test_cannot_instantiate_abc(self):
        """BaseExtractor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseExtractor()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        """Concrete subclass can be instantiated and yields Documents."""
        extractor = _DummyExtractor()
        results = list(
            extractor.extract(
                input_dir=Path("/tmp"),
                source="test_src",
                domain="test_domain",
            )
        )
        assert len(results) == 1
        doc = results[0]
        assert isinstance(doc, Document)
        assert doc.text == "hello"
        assert doc.source == "test_src"
        assert doc.domain == "test_domain"


class TestRegistry:
    """Tests for register_extractor and get_extractor."""

    def test_register_and_get(self):
        """Registered extractor can be retrieved and instantiated."""
        register_extractor("dummy", _DummyExtractor)
        extractor = get_extractor("dummy")
        assert isinstance(extractor, BaseExtractor)
        assert isinstance(extractor, _DummyExtractor)

    def test_get_unknown_raises(self):
        """get_extractor raises KeyError with name in message."""
        name = "nonexistent_extractor_xyz"
        with pytest.raises(KeyError, match=name):
            get_extractor(name)
