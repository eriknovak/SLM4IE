"""Tests for BaseExtractor, FileBasedExtractor and the extractor registry."""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pytest

from slm4ie.data.extract.extractors import (
    BaseExtractor,
    FileBasedExtractor,
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


class _DummyFileExtractor(FileBasedExtractor):
    """One Document per .txt file, text = file contents."""

    def iter_input_files(self, input_dir: Path) -> List[Path]:
        """Return sorted .txt files under input_dir."""
        return sorted(input_dir.rglob("*.txt"))

    def extract_files(
        self,
        files: List[Path],
        source: str,
        domain: str,
        input_dir: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Document]:
        """Yield one Document per file with its text as content."""
        del input_dir, metadata
        for filepath in files:
            yield Document(
                text=filepath.read_text(encoding="utf-8"),
                source=source,
                domain=domain,
                doc_id=filepath.stem,
            )


def test_extract_delegates_to_iter_and_parse(tmp_path: Path) -> None:
    """Default extract() == extract_files(iter_input_files(...))."""
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")

    ext = _DummyFileExtractor()
    docs = list(ext.extract(tmp_path, "dummy", "web"))

    assert [d.doc_id for d in docs] == ["a", "b"]
    assert [d.text for d in docs] == ["alpha", "beta"]


def test_extract_files_subset_is_ordered(tmp_path: Path) -> None:
    """extract_files over an explicit subset preserves the given order."""
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    ext = _DummyFileExtractor()
    subset = [tmp_path / "c.txt", tmp_path / "a.txt"]
    docs = list(ext.extract_files(subset, "dummy", "web", tmp_path))

    assert [d.doc_id for d in docs] == ["c", "a"]
