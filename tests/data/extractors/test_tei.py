"""Tests for the TEI XML extractor."""

from pathlib import Path

import pytest

from slm4ie.data.extractors.tei import (
    TeiExtractor,
    _mte_to_upos,
    _parse_ana,
    _parse_msd,
)

# parlamint_si / siParl style — annotated TEI wrapping each speech in
# a <u> element. Two utterances by different speakers, each carrying
# multiple <s> sentences.
_UTTERANCE_TEI = """\
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <u xml:id="u1" who="#speakerA" ana="#regular">
        <s xml:id="u1.s1">
          <w lemma="predsednik" msd="UPosTag=NOUN|Case=Nom">Predsednik</w>
          <w lemma="biti" msd="UPosTag=AUX">je</w>
          <w lemma="odpreti" msd="UPosTag=VERB">odprl</w>
          <w lemma="seja" msd="UPosTag=NOUN|Case=Acc">sejo</w>
          <pc msd="UPosTag=PUNCT">.</pc>
        </s>
        <s xml:id="u1.s2">
          <w lemma="hvala" msd="UPosTag=NOUN">Hvala</w>
          <pc msd="UPosTag=PUNCT">.</pc>
        </s>
      </u>
      <u xml:id="u2" who="#speakerB">
        <s xml:id="u2.s1">
          <w lemma="dober" msd="UPosTag=ADJ">Dober</w>
          <w lemma="dan" msd="UPosTag=NOUN">dan</w>
          <pc msd="UPosTag=PUNCT">.</pc>
        </s>
      </u>
    </body>
  </text>
</TEI>
"""

# kas style — annotated TEI with no <u> elements (scientific monographs,
# not parliamentary). One Document per file should fall out of the
# fallback path.
_KAS_TEI = """\
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <div type="chapter">
        <p xml:id="p1">
          <s xml:id="kas.s1">
            <w lemma="gospodarstvo" ana="mte:Ncnsn">Gospodarstvo</w>
            <w lemma="in" ana="mte:Cc">in</w>
            <w lemma="javen" ana="mte:Agpfsn">javna</w>
            <w lemma="uprava" ana="mte:Ncfsn">uprava</w>
            <pc ana="mte:Z" join="right">.</pc>
          </s>
          <s xml:id="kas.s2">
            <w lemma="razvoj" ana="mte:Ncmsn">Razvoj</w>
            <w lemma="sektor" ana="mte:Ncmsg">sektorja</w>
            <pc ana="mte:Z" join="right">.</pc>
          </s>
        </p>
      </div>
    </body>
  </text>
</TEI>
"""

# Plain TEI — no <w> elements. Per-<p> behavior is unchanged.
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
def tmp_utterance(tmp_path: Path) -> Path:
    """Write parliament-style annotated TEI to a temp dir and return it."""
    (tmp_path / "session.xml").write_text(_UTTERANCE_TEI, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def tmp_plain(tmp_path: Path) -> Path:
    """Write plain TEI to a temp dir and return it."""
    (tmp_path / "plain.xml").write_text(_PLAIN_TEI, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def tmp_kas(tmp_path: Path) -> Path:
    """Write KAS-style annotated TEI (no <u>) to a temp dir and return it."""
    (tmp_path / "kas.xml").write_text(_KAS_TEI, encoding="utf-8")
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


class TestMteToUpos:
    """Unit tests for _mte_to_upos helper (MTE compact → UPOS)."""

    def test_common_noun(self) -> None:
        """Nc* → NOUN."""
        assert _mte_to_upos("Ncnsn") == "NOUN"

    def test_proper_noun(self) -> None:
        """Np* → PROPN."""
        assert _mte_to_upos("Npmsn") == "PROPN"

    def test_main_verb(self) -> None:
        """Vm* → VERB."""
        assert _mte_to_upos("Vmpr3s-n") == "VERB"

    def test_auxiliary_verb(self) -> None:
        """Va* → AUX."""
        assert _mte_to_upos("Va-r3s-n") == "AUX"

    def test_coordinating_conjunction(self) -> None:
        """Cc → CCONJ."""
        assert _mte_to_upos("Cc") == "CCONJ"

    def test_subordinating_conjunction(self) -> None:
        """Cs → SCONJ."""
        assert _mte_to_upos("Cs") == "SCONJ"

    def test_punctuation(self) -> None:
        """Z → PUNCT."""
        assert _mte_to_upos("Z") == "PUNCT"

    def test_adposition(self) -> None:
        """S* → ADP."""
        assert _mte_to_upos("Sl") == "ADP"

    def test_unknown_category(self) -> None:
        """Unknown first char yields None."""
        assert _mte_to_upos("?") is None

    def test_empty(self) -> None:
        """Empty string yields None."""
        assert _mte_to_upos("") is None


class TestParseAna:
    """Unit tests for _parse_ana helper."""

    def test_mte_descriptor(self) -> None:
        """ana="mte:Ncnsn" → (NOUN, MTE=Ncnsn)."""
        upos, feats = _parse_ana("mte:Ncnsn")
        assert upos == "NOUN"
        assert feats == "MTE=Ncnsn"

    def test_punctuation_pc(self) -> None:
        """ana="mte:Z" → (PUNCT, MTE=Z)."""
        upos, feats = _parse_ana("mte:Z")
        assert upos == "PUNCT"
        assert feats == "MTE=Z"

    def test_multivalue_picks_mte(self) -> None:
        """Space-separated values: pick the mte: one."""
        upos, feats = _parse_ana("foo:bar mte:Cc")
        assert upos == "CCONJ"
        assert feats == "MTE=Cc"

    def test_no_mte(self) -> None:
        """Without an mte: value, returns (None, None)."""
        assert _parse_ana("foo:bar") == (None, None)

    def test_none(self) -> None:
        """None input yields (None, None)."""
        assert _parse_ana(None) == (None, None)


class TestTeiUtteranceExtraction:
    """Integration tests for TEI with <u> utterance wrappers."""

    def test_one_document_per_utterance(
        self, extractor: TeiExtractor, tmp_utterance: Path
    ) -> None:
        """Each <u> element becomes its own Document."""
        docs = list(extractor.extract(tmp_utterance, "parlamint_si", "parliamentary"))
        assert len(docs) == 2

    def test_utterance_text_joins_sentences(
        self, extractor: TeiExtractor, tmp_utterance: Path
    ) -> None:
        """Sentences inside a <u> are joined with newlines for sentence-dedup."""
        docs = list(extractor.extract(tmp_utterance, "parlamint_si", "parliamentary"))
        assert docs[0].text == "Predsednik je odprl sejo .\nHvala ."
        assert docs[1].text == "Dober dan ."

    def test_doc_id_from_utterance_xml_id(
        self, extractor: TeiExtractor, tmp_utterance: Path
    ) -> None:
        """doc_id comes from <u>'s xml:id, not from individual <s>."""
        docs = list(extractor.extract(tmp_utterance, "parlamint_si", "parliamentary"))
        assert docs[0].doc_id == "u1"
        assert docs[1].doc_id == "u2"

    def test_utterance_metadata_carries_speaker(
        self, extractor: TeiExtractor, tmp_utterance: Path
    ) -> None:
        """`who` and `ana` attributes on <u> flow into Document.metadata."""
        docs = list(extractor.extract(tmp_utterance, "parlamint_si", "parliamentary"))
        assert docs[0].metadata.get("who") == "#speakerA"
        assert docs[0].metadata.get("ana") == "#regular"
        assert docs[1].metadata.get("who") == "#speakerB"
        # No `ana` on u2 — metadata key simply absent rather than None.
        assert "ana" not in docs[1].metadata

    def test_utterance_annotations_flat_with_sentence_spans(
        self, extractor: TeiExtractor, tmp_utterance: Path
    ) -> None:
        """Tokens concatenate across sentences; sentences carries the spans."""
        docs = list(extractor.extract(tmp_utterance, "parlamint_si", "parliamentary"))
        ann = docs[0].annotations
        assert ann is not None
        # u1: 5 tokens (s1) + 2 tokens (s2) = 7 flat tokens.
        assert len(ann.tokens) == 7
        assert [t.form for t in ann.tokens] == [
            "Predsednik", "je", "odprl", "sejo", ".", "Hvala", ".",
        ]
        assert ann.sentences == [[0, 4], [5, 6]]


class TestTeiPerFileFallback:
    """Annotated TEI without <u> (KAS-style) collapses to per-file Document."""

    def test_one_document_per_file(
        self, extractor: TeiExtractor, tmp_kas: Path
    ) -> None:
        """No <u> in the file → exactly one Document per file."""
        docs = list(extractor.extract(tmp_kas, "kas", "academic"))
        assert len(docs) == 1

    def test_doc_id_from_file_stem(
        self, extractor: TeiExtractor, tmp_kas: Path
    ) -> None:
        """doc_id falls back to the source file's stem when no <u>."""
        docs = list(extractor.extract(tmp_kas, "kas", "academic"))
        assert docs[0].doc_id == "kas"

    def test_aggregated_text_and_annotations(
        self, extractor: TeiExtractor, tmp_kas: Path
    ) -> None:
        """All <s> tokens land in one Document with multi-sentence spans."""
        docs = list(extractor.extract(tmp_kas, "kas", "academic"))
        assert docs[0].text == "Gospodarstvo in javna uprava .\nRazvoj sektorja ."
        ann = docs[0].annotations
        assert ann is not None
        assert [t.form for t in ann.tokens] == [
            "Gospodarstvo", "in", "javna", "uprava", ".",
            "Razvoj", "sektorja", ".",
        ]
        assert [t.upos for t in ann.tokens] == [
            "NOUN", "CCONJ", "ADJ", "NOUN", "PUNCT",
            "NOUN", "NOUN", "PUNCT",
        ]
        assert ann.sentences == [[0, 4], [5, 7]]


class TestTeiPlainParagraphs:
    """Plain TEI (no <w>) keeps the original per-<p> Document behavior."""

    def test_one_document_per_paragraph(
        self, extractor: TeiExtractor, tmp_plain: Path
    ) -> None:
        """Two <p> elements → two Documents, no annotations."""
        docs = list(extractor.extract(tmp_plain, "test", "web"))
        assert len(docs) == 2
        texts = [d.text for d in docs]
        assert "Dober dan. Kako ste?" in texts
        assert "Hvala, dobro." in texts
        for doc in docs:
            assert doc.annotations is None

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


class TestTeiMetadata:
    """End-to-end metadata injection from an external TSV."""

    def test_metadata_attached_via_filename_stem(
        self, extractor: TeiExtractor, tmp_path: Path
    ) -> None:
        """KAS-style: lookup by the TEI file's stem (e.g. `kas-10000`)."""
        (tmp_path / "kas-10000.xml").write_text(_KAS_TEI, encoding="utf-8")
        tsv = tmp_path / "meta.tsv"
        tsv.write_text(
            "id\tcerif\tdoctype\n"
            "kas-10000\tP000\tDiplomsko delo\n",
            encoding="utf-8",
        )

        docs = list(
            extractor.extract(
                tmp_path,
                source="kas",
                domain="academic",
                metadata={
                    "path": "meta.tsv",
                    "key_column": "id",
                    "fields": {"cerif": "cerif", "doctype": "doctype"},
                },
            )
        )

        # KAS-style file aggregates to a single per-file Document; the
        # TSV row keyed by the filename stem populates its metadata.
        assert len(docs) == 1
        assert docs[0].metadata == {"cerif": "P000", "doctype": "Diplomsko delo"}

    def test_metadata_attached_to_plain_paragraphs(
        self, extractor: TeiExtractor, tmp_path: Path
    ) -> None:
        """Plain TEI (no <w>) also receives the per-file metadata."""
        (tmp_path / "doc-1.xml").write_text(_PLAIN_TEI, encoding="utf-8")
        tsv = tmp_path / "meta.tsv"
        tsv.write_text("id\tnote\ndoc-1\thello\n", encoding="utf-8")

        docs = list(
            extractor.extract(
                tmp_path,
                source="test",
                domain="web",
                metadata={
                    "path": "meta.tsv",
                    "key_column": "id",
                    "fields": {"note": "note"},
                },
            )
        )

        assert len(docs) == 2
        for doc in docs:
            assert doc.metadata == {"note": "hello"}

    def test_metadata_layers_under_utterance_attributes(
        self, extractor: TeiExtractor, tmp_path: Path
    ) -> None:
        """TSV per-file fields appear alongside <u>'s `who`/`ana`."""
        (tmp_path / "session-7.xml").write_text(_UTTERANCE_TEI, encoding="utf-8")
        tsv = tmp_path / "meta.tsv"
        tsv.write_text("id\tsitting\nsession-7\t2023-11-15\n", encoding="utf-8")

        docs = list(
            extractor.extract(
                tmp_path,
                source="parlamint_si",
                domain="parliamentary",
                metadata={
                    "path": "meta.tsv",
                    "key_column": "id",
                    "fields": {"sitting": "sitting"},
                },
            )
        )

        assert len(docs) == 2
        # Per-file `sitting` is on every utterance; per-utterance `who`
        # and `ana` come from the <u> element.
        assert docs[0].metadata == {
            "sitting": "2023-11-15",
            "who": "#speakerA",
            "ana": "#regular",
        }
        assert docs[1].metadata == {
            "sitting": "2023-11-15",
            "who": "#speakerB",
        }

    def test_no_metadata_kwarg_keeps_empty_dict(
        self, extractor: TeiExtractor, tmp_kas: Path
    ) -> None:
        """Without the kwarg, per-file metadata stays empty.

        Uses the KAS-style fixture (no `<u>` and no `<p>` paragraph
        metadata) so the resulting Document has nothing in `metadata`
        at all when no TSV is configured.
        """
        docs = list(extractor.extract(tmp_kas, "kas", "academic"))
        assert docs[0].metadata == {}
