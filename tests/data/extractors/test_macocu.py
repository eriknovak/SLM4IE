"""Tests for the MaCoCu XML extractor."""

from pathlib import Path

import pytest

from slm4ie.data.extractors.macocu import MacocuExtractor

_SAMPLE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE cesDoc SYSTEM "MaCoCu-monolingual.dtd">
<cesDoc version="4">
  <cesHeader/>
  <text>
    <group>
      <tu id="1" score="0.95">
        <tuv lang="sl">
          <p>Dober dan.</p>
          <p>Kako ste?</p>
        </tuv>
      </tu>
      <tu id="2" score="0.42">
        <tuv lang="sl">
          <p>Hvala, dobro.</p>
        </tuv>
      </tu>
      <tu id="3" score="0.80">
        <tuv lang="sl">
          <p>Nasvidenje.</p>
        </tuv>
      </tu>
    </group>
  </text>
</cesDoc>
"""


@pytest.fixture()
def extractor() -> MacocuExtractor:
    """Return a MacocuExtractor instance."""
    return MacocuExtractor()


@pytest.fixture()
def tmp_xml(tmp_path: Path) -> Path:
    """Write sample MaCoCu XML to a temp dir and return it."""
    (tmp_path / "sample.xml").write_text(_SAMPLE_XML, encoding="utf-8")
    return tmp_path


class TestMacocuExtractor:
    """Tests for MacocuExtractor."""

    def test_extracts_all_tu_elements(
        self, extractor: MacocuExtractor, tmp_xml: Path
    ) -> None:
        """One Document is produced per <tu> element."""
        docs = list(extractor.extract(tmp_xml, "macocu", "web"))
        assert len(docs) == 3

    def test_extracts_text_content(
        self, extractor: MacocuExtractor, tmp_xml: Path
    ) -> None:
        """Text is joined from all <p> elements within a <tu>."""
        docs = list(extractor.extract(tmp_xml, "macocu", "web"))
        assert docs[0].text == "Dober dan. Kako ste?"

    def test_preserves_score_in_metadata(
        self, extractor: MacocuExtractor, tmp_xml: Path
    ) -> None:
        """metadata['score'] holds the score attribute from <tu>."""
        docs = list(extractor.extract(tmp_xml, "macocu", "web"))
        assert docs[0].metadata["score"] == "0.95"

    def test_doc_id_from_tu_id(
        self, extractor: MacocuExtractor, tmp_xml: Path
    ) -> None:
        """doc_id is set from the id attribute of <tu>."""
        docs = list(extractor.extract(tmp_xml, "macocu", "web"))
        assert docs[0].doc_id == "1"

    def test_source_and_domain(
        self, extractor: MacocuExtractor, tmp_xml: Path
    ) -> None:
        """source and domain are passed through to every Document."""
        docs = list(extractor.extract(tmp_xml, "macocu", "web"))
        for doc in docs:
            assert doc.source == "macocu"
            assert doc.domain == "web"

    def test_no_annotations(
        self, extractor: MacocuExtractor, tmp_xml: Path
    ) -> None:
        """Documents have no annotations (plain text only)."""
        docs = list(extractor.extract(tmp_xml, "macocu", "web"))
        for doc in docs:
            assert doc.annotations is None

    def test_skips_invalid_xml(
        self, extractor: MacocuExtractor, tmp_path: Path
    ) -> None:
        """ParseError files are skipped with a warning."""
        (tmp_path / "bad.xml").write_text(
            "<unclosed>", encoding="utf-8"
        )
        docs = list(extractor.extract(tmp_path, "macocu", "web"))
        assert docs == []

    def test_skips_empty_tu_text(
        self, extractor: MacocuExtractor, tmp_path: Path
    ) -> None:
        """<tu> elements with no text content are skipped."""
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<cesDoc version="4">
  <cesHeader/>
  <text>
    <group>
      <tu id="1" score="0.5">
        <tuv lang="sl">
          <p></p>
        </tuv>
      </tu>
    </group>
  </text>
</cesDoc>
"""
        (tmp_path / "empty.xml").write_text(xml, encoding="utf-8")
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
        assert len(docs) > 0
