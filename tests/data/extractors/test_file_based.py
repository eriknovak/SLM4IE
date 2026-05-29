"""Tests for the FileBasedExtractor enumeration/parse split."""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from slm4ie.data.extractors import FileBasedExtractor
from slm4ie.data.schema import Document


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
