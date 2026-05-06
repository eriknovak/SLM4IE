"""Tests for the MaCoCu XML extractor."""

from pathlib import Path

import pytest

from slm4ie.data.extractors.macocu import MacocuExtractor

_SAMPLE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE corpus SYSTEM "MaCoCu-monolingual.dtd">
<corpus id="MaCoCu-sl-2.0">
<doc id="macocu.sl.1" title="Page One" url="https://example.com/1" \
crawl_date="2022-07-01" lm_score="0.95">
<p id="macocu.sl.1.1" lang="sl">Dober dan.</p>
<p id="macocu.sl.1.2" lang="sl">Kako ste?</p>
</doc>
<doc id="macocu.sl.2" title="Page Two" url="https://example.com/2">
<p id="macocu.sl.2.1" lang="sl">Hvala, dobro.</p>
</doc>
<doc id="macocu.sl.3">
<p></p>
</doc>
</corpus>
"""


@pytest.fixture()
def extractor() -> MacocuExtractor:
    """Return a MacocuExtractor instance."""
    return MacocuExtractor()


@pytest.fixture()
def tmp_xml(tmp_path: Path) -> Path:
    """Write a sample MaCoCu XML to *tmp_path* and return the dir."""
    (tmp_path / "sample.xml").write_text(_SAMPLE_XML, encoding="utf-8")
    return tmp_path


class TestMacocuExtractor:
    """Tests for MacocuExtractor."""

    def test_extracts_one_doc_per_doc_element(
        self, extractor: MacocuExtractor, tmp_xml: Path
    ) -> None:
        """One Document per non-empty <doc>; empty ones are skipped."""
        docs = list(extractor.extract(tmp_xml, "macocu", "web"))
        assert len(docs) == 2

    def test_joins_paragraphs_with_newline(
        self, extractor: MacocuExtractor, tmp_xml: Path
    ) -> None:
        """Multiple <p> children are joined with a newline."""
        docs = list(extractor.extract(tmp_xml, "macocu", "web"))
        assert docs[0].text == "Dober dan.\nKako ste?"

    def test_doc_id_from_doc_attribute(
        self, extractor: MacocuExtractor, tmp_xml: Path
    ) -> None:
        """``doc_id`` is read from the <doc> id attribute."""
        docs = list(extractor.extract(tmp_xml, "macocu", "web"))
        assert docs[0].doc_id == "macocu.sl.1"
        assert docs[1].doc_id == "macocu.sl.2"

    def test_metadata_includes_doc_attributes(
        self, extractor: MacocuExtractor, tmp_xml: Path
    ) -> None:
        """Selected <doc> attributes are surfaced as metadata."""
        docs = list(extractor.extract(tmp_xml, "macocu", "web"))
        meta = docs[0].metadata
        assert meta["title"] == "Page One"
        assert meta["url"] == "https://example.com/1"
        assert meta["lm_score"] == "0.95"

    def test_source_and_domain(
        self, extractor: MacocuExtractor, tmp_xml: Path
    ) -> None:
        """``source`` and ``domain`` are passed through to every Document."""
        docs = list(extractor.extract(tmp_xml, "macocu", "web"))
        for doc in docs:
            assert doc.source == "macocu"
            assert doc.domain == "web"

    def test_no_annotations(
        self, extractor: MacocuExtractor, tmp_xml: Path
    ) -> None:
        """MaCoCu Documents carry plain text only."""
        docs = list(extractor.extract(tmp_xml, "macocu", "web"))
        for doc in docs:
            assert doc.annotations is None

    def test_skips_invalid_xml(
        self, extractor: MacocuExtractor, tmp_path: Path
    ) -> None:
        """Files with parse errors are skipped with a warning."""
        (tmp_path / "bad.xml").write_text("<unclosed>", encoding="utf-8")
        docs = list(extractor.extract(tmp_path, "macocu", "web"))
        assert docs == []

    def test_recursive_directory_scan(
        self, extractor: MacocuExtractor, tmp_path: Path
    ) -> None:
        """XML files in nested subdirectories are also scanned."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "nested.xml").write_text(_SAMPLE_XML, encoding="utf-8")
        docs = list(extractor.extract(tmp_path, "macocu", "web"))
        assert len(docs) == 2
