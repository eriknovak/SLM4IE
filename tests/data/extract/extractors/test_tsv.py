"""Tests for TsvExtractor."""

from pathlib import Path

from slm4ie.data.extract.extractors import get_extractor
from slm4ie.data.extract.extractors.tsv import TsvExtractor  # noqa: F401  (registers "tsv")
from slm4ie.data.schema import Document


def _extract(tmp_path: Path, content: str, metadata: dict | None = None) -> list[Document]:
    """Write `content` to news/a.tsv under tmp_path and extract it.

    Args:
        tmp_path: Temporary dataset directory.
        content: Raw bytes-as-text of the file, newlines included.
        metadata: Optional `metadata:` block forwarded to the extractor.

    Returns:
        The extracted documents.
    """
    (tmp_path / "news").mkdir()
    (tmp_path / "news" / "a.tsv").write_bytes(content.encode("utf-8"))
    return list(get_extractor("tsv").extract(tmp_path, source="s", domain="news", metadata=metadata))


class TestTsvExtractor:
    """Tests for TsvExtractor."""

    def test_reads_text_native_id_and_whitelisted_metadata(self, tmp_path: Path) -> None:
        """Configured columns map to text, id and metadata; other columns are dropped."""
        content = "nid\tcontent\ttitle\tother\n7\tDober dan.\tNaslov\tx\n"
        cfg = {"text_field": "content", "id_field": "nid", "metadata_fields": ["title"]}
        (doc,) = _extract(tmp_path, content, cfg)
        assert (doc.text, doc.doc_id, doc.metadata) == ("Dober dan.", "7", {"title": "Naslov"})

    def test_positional_id_without_id_column(self, tmp_path: Path) -> None:
        """A missing id column gives `<unit>:<ordinal>`, counting skipped rows."""
        content = "text\tn\n\t0\nPrvi.\t1\nDrugi.\t2\n"
        docs = _extract(tmp_path, content, {"text_field": "text", "id_field": "nid"})
        assert [d.doc_id for d in docs] == ["news/a:00000001", "news/a:00000002"]

    def test_skips_empty_text_rows(self, tmp_path: Path) -> None:
        """A row whose text cell is empty yields no document."""
        content = "text\tn\nA.\t1\n\t2\nB.\t3\n"
        assert [d.text for d in _extract(tmp_path, content)] == ["A.", "B."]

    def test_stray_quote_is_kept_literally(self, tmp_path: Path) -> None:
        """A stray quote stays in its cell and does not swallow the next row."""
        docs = _extract(tmp_path, 'text\tn\nsay "hi\t1\nnext\t2\n')
        assert [d.text for d in docs] == ['say "hi', "next"]

    def test_crlf_does_not_leak_into_cells(self, tmp_path: Path) -> None:
        """A CRLF line end leaves no carriage return in the last cell."""
        docs = _extract(tmp_path, "text\tn\r\nDober dan.\t1\r\n")
        assert [(d.text, d.metadata["n"]) for d in docs] == [("Dober dan.", "1")]

    def test_reads_field_longer_than_csv_default_limit(self, tmp_path: Path) -> None:
        """A 200 KB field is read whole."""
        (doc,) = _extract(tmp_path, "text\n" + "a" * 200_000 + "\n")
        assert len(doc.text) == 200_000

    def test_extracts_txt_file(self, tmp_path: Path) -> None:
        """A tab-separated file ending in .txt is discovered and read."""
        (tmp_path / "SentiNews.txt").write_text("text\tn\nDober dan.\t1\n", encoding="utf-8")
        docs = list(get_extractor("tsv").extract(tmp_path, source="s", domain="news"))
        assert [d.text for d in docs] == ["Dober dan."]
