"""Tests for the TEI XML extractor."""

from pathlib import Path

import pytest

from slm4ie.data.extractors.tei import TeiExtractor, _parse_msd

_ANNOTATED_TEI = """\
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <u xml:id="u1" who="#speaker1">
        <seg xml:id="seg1">
          <s xml:id="s1">
            <w lemma="predsednik" msd="UPosTag=NOUN|Case=Nom">Predsednik</w>
            <w lemma="biti" msd="UPosTag=AUX">je</w>
            <w lemma="odpreti" msd="UPosTag=VERB">odprl</w>
            <w lemma="seja" msd="UPosTag=NOUN|Case=Acc">sejo</w>
            <pc msd="UPosTag=PUNCT">.</pc>
          </s>
        </seg>
      </u>
    </body>
  </text>
</TEI>
"""

_PLAIN_TEI = """\
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <p>Dober dan. Kako ste?</p>
      <p>Hvala, dobro.</p>
    </body>
  </text>
</TEI>
"""


@pytest.fixture()
def extractor() -> TeiExtractor:
    """Return a TeiExtractor instance."""
    return TeiExtractor()


@pytest.fixture()
def tmp_annotated(tmp_path: Path) -> Path:
    """Write annotated TEI to a temp dir and return it."""
    (tmp_path / "annotated.xml").write_text(_ANNOTATED_TEI, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def tmp_plain(tmp_path: Path) -> Path:
    """Write plain TEI to a temp dir and return it."""
    (tmp_path / "plain.xml").write_text(_PLAIN_TEI, encoding="utf-8")
    return tmp_path


class TestParseMsd:
    """Unit tests for _parse_msd helper."""

    def test_upos_extracted(self) -> None:
        """UPosTag part is returned as upos."""
        upos, _ = _parse_msd("UPosTag=NOUN|Case=Nom")
        assert upos == "NOUN"

    def test_feats_exclude_upostag(self) -> None:
        """Remaining parts after UPosTag are joined as feats."""
        _, feats = _parse_msd("UPosTag=NOUN|Case=Nom|Gender=Masc")
        assert feats == "Case=Nom|Gender=Masc"

    def test_no_extra_feats(self) -> None:
        """When only UPosTag present, feats is None."""
        _, feats = _parse_msd("UPosTag=AUX")
        assert feats is None

    def test_none_input(self) -> None:
        """None input yields (None, None)."""
        assert _parse_msd(None) == (None, None)

    def test_empty_string(self) -> None:
        """Empty string yields (None, None)."""
        assert _parse_msd("") == (None, None)


class TestTeiExtractor:
    """Integration tests for TeiExtractor."""

    def test_extracts_text_from_w_elements(
        self, extractor: TeiExtractor, tmp_annotated: Path
    ) -> None:
        """Joined token forms become document text."""
        docs = list(extractor.extract(tmp_annotated, "test", "parl"))
        assert len(docs) == 1
        assert docs[0].text == "Predsednik je odprl sejo ."

    def test_extracts_lemma_and_msd(
        self, extractor: TeiExtractor, tmp_annotated: Path
    ) -> None:
        """Lemma and UPOS are parsed from annotated tokens."""
        docs = list(extractor.extract(tmp_annotated, "test", "parl"))
        tokens = docs[0].annotations.tokens
        assert tokens[0].lemma == "predsednik"
        assert tokens[0].upos == "NOUN"
        assert tokens[0].feats == "Case=Nom"
        # Token with no extra feats
        assert tokens[1].upos == "AUX"
        assert tokens[1].feats is None


    def test_extracts_plain_text_paragraphs(
        self, extractor: TeiExtractor, tmp_plain: Path
    ) -> None:
        """Plain TEI yields one Document per non-empty <p>."""
        docs = list(extractor.extract(tmp_plain, "test", "web"))
        assert len(docs) == 2
        texts = [d.text for d in docs]
        assert "Dober dan. Kako ste?" in texts
        assert "Hvala, dobro." in texts
        for doc in docs:
            assert doc.annotations is None

    def test_doc_id_from_xml_id(
        self, extractor: TeiExtractor, tmp_annotated: Path
    ) -> None:
        """doc_id is set from xml:id on <s> element."""
        docs = list(extractor.extract(tmp_annotated, "test", "parl"))
        assert docs[0].doc_id is not None
        assert docs[0].doc_id == "s1"

    def test_processes_multiple_xml_files(
        self, extractor: TeiExtractor, tmp_path: Path
    ) -> None:
        """All XML files in the directory are processed."""
        (tmp_path / "a.xml").write_text(_PLAIN_TEI, encoding="utf-8")
        (tmp_path / "b.xml").write_text(_PLAIN_TEI, encoding="utf-8")
        docs = list(extractor.extract(tmp_path, "test", "web"))
        assert len(docs) == 4

    def test_handles_nested_dirs(
        self, extractor: TeiExtractor, tmp_path: Path
    ) -> None:
        """XML files in subdirectories are found via rglob."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "nested.xml").write_text(_PLAIN_TEI, encoding="utf-8")
        docs = list(extractor.extract(tmp_path, "test", "web"))
        assert len(docs) == 2
