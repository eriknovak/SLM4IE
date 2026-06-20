"""Tests for slm4ie/data/sloleks.py and scripts/data/to_tokenization.py."""

import gzip
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from textwrap import dedent

import pytest

from slm4ie.data import sloleks

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "scripts" / "data"),
)
import to_tokenization as to_tokenizer_eval  # noqa: E402


#: A minimal sample mirroring the real Sloleks 3.1 `<lexicon>` schema: lemma at
#: head/headword/lemma, forms under body/wordFormList/wordForm with the JOS
#: `<msd>` and orthographic forms at formRepresentations/orthographyList/
#: orthography/form. Accentuation and pronunciation `<form>` siblings are
#: included deliberately so the parser is proven to ignore them.
SAMPLE_LEXICON = dedent(
    """\
    <?xml version="1.0" encoding="UTF-8"?>
    <lexicon>
      <entry>
        <head>
          <headword><lemma>hiša</lemma></headword>
          <lexicalUnit sloleksId="LE_hisa" sloleksKey="S_hiša" type="single">
            <lexeme>hiša</lexeme>
          </lexicalUnit>
          <grammar><category>noun</category></grammar>
        </head>
        <body>
          <wordFormList>
            <wordForm>
              <msd language="sl" system="JOS">Ncfsn</msd>
              <formRepresentations>
                <orthographyList>
                  <orthography morphologyPatterns="S1"><form>hiša</form></orthography>
                </orthographyList>
                <accentuationList type="dynamic">
                  <accentuation><form>híša</form></accentuation>
                </accentuationList>
                <pronunciationList>
                  <pronunciation><form script="IPA">ˈxiːʃa</form></pronunciation>
                </pronunciationList>
              </formRepresentations>
            </wordForm>
            <wordForm>
              <msd language="sl" system="JOS">Ncfsg</msd>
              <formRepresentations>
                <orthographyList>
                  <orthography morphologyPatterns="S1"><form>hiše</form></orthography>
                </orthographyList>
              </formRepresentations>
            </wordForm>
          </wordFormList>
        </body>
      </entry>
      <entry>
        <head>
          <headword><lemma>biti</lemma></headword>
          <lexicalUnit sloleksId="LE_biti" sloleksKey="G_biti" type="single">
            <lexeme>biti</lexeme>
          </lexicalUnit>
          <grammar><category>verb</category></grammar>
        </head>
        <body>
          <wordFormList>
            <wordForm>
              <msd language="sl" system="JOS">Ggnn</msd>
              <formRepresentations>
                <orthographyList>
                  <orthography><form>biti</form></orthography>
                </orthographyList>
              </formRepresentations>
            </wordForm>
            <wordForm>
              <msd language="sl" system="JOS">Gp-spdm</msd>
              <formRepresentations>
                <orthographyList>
                  <orthography><form>sem</form></orthography>
                </orthographyList>
              </formRepresentations>
            </wordForm>
          </wordFormList>
        </body>
      </entry>
      <entry>
        <head>
          <headword><lemma>prazen</lemma></headword>
          <grammar><category>adjective</category></grammar>
        </head>
        <body>
          <wordFormList/>
        </body>
      </entry>
    </lexicon>
    """
)


def _write_sample(path: Path) -> Path:
    """Write SAMPLE_LEXICON to *path* and return it.

    Args:
        path: Destination file path; parent directories are created.

    Returns:
        Path: The path that was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SAMPLE_LEXICON, encoding="utf-8")
    return path


class TestLocalName:
    """Unit tests for sloleks._local_name."""

    def test_strips_clark_notation(self):
        """Clark-notation prefixes are stripped."""
        assert sloleks._local_name("{http://example/ns}entry") == "entry"

    def test_passthrough_when_unprefixed(self):
        """Tags without a namespace pass through unchanged."""
        assert sloleks._local_name("entry") == "entry"


class TestWordFormMsd:
    """Unit tests for sloleks._wordform_msd."""

    def test_reads_direct_msd_child(self):
        """The JOS `<msd>` directly under `<wordForm>` is returned."""
        wf = ET.fromstring(
            '<wordForm><msd language="sl" system="JOS">Ncfsn</msd>'
            "<formRepresentations/></wordForm>"
        )
        assert sloleks._wordform_msd(wf) == "Ncfsn"

    def test_missing_msd_returns_none(self):
        """A word form without an `<msd>` yields None."""
        wf = ET.fromstring("<wordForm><formRepresentations/></wordForm>")
        assert sloleks._wordform_msd(wf) is None


class TestWordFormOrthForms:
    """Unit tests for sloleks._wordform_orth_forms."""

    def test_only_orthographic_forms(self):
        """Accentuation and pronunciation forms are excluded."""
        wf = ET.fromstring(
            "<wordForm><formRepresentations>"
            "<orthographyList><orthography><form>hiša</form></orthography></orthographyList>"
            "<accentuationList><accentuation><form>híša</form></accentuation></accentuationList>"
            '<pronunciationList><pronunciation><form script="IPA">x</form></pronunciation></pronunciationList>'
            "</formRepresentations></wordForm>"
        )
        assert sloleks._wordform_orth_forms(wf) == ["hiša"]


class TestEntryToRecord:
    """Tests for the per-entry conversion."""

    def test_lemma_and_inflected_forms(self):
        """The lemma form and all inflected forms appear in the record."""
        path = Path("/tmp/_sloleks_unit_test.xml")
        try:
            _write_sample(path)
            records = list(sloleks.iter_sloleks_entries(path))
        finally:
            if path.exists():
                path.unlink()

        assert len(records) == 2
        first = records[0]
        assert first["entry_id"] == "LE_hisa"
        assert first["lemma"] == "hiša"
        assert first["lemma_msd"] == "Ncfsn"
        forms = {f["form"]: f["msd"] for f in first["forms"]}
        assert forms == {"hiša": "Ncfsn", "hiše": "Ncfsg"}

    def test_lemma_msd_from_matching_form(self):
        """lemma_msd is taken from the word form whose orth equals the lemma."""
        path = Path("/tmp/_sloleks_unit_test.xml")
        try:
            _write_sample(path)
            records = list(sloleks.iter_sloleks_entries(path))
        finally:
            if path.exists():
                path.unlink()

        biti = next(r for r in records if r["lemma"] == "biti")
        assert biti["lemma_msd"] == "Ggnn"
        forms = {f["form"]: f["msd"] for f in biti["forms"]}
        assert forms == {"biti": "Ggnn", "sem": "Gp-spdm"}

    def test_entry_without_forms_is_skipped(self):
        """Entries that yield no orthographic forms produce no record."""
        path = Path("/tmp/_sloleks_unit_test.xml")
        try:
            _write_sample(path)
            records = list(sloleks.iter_sloleks_entries(path))
        finally:
            if path.exists():
                path.unlink()
        assert all(r["lemma"] != "prazen" for r in records)


class TestIterSloleksDir:
    """Tests for the directory-level walker."""

    def test_skips_mezzanine_and_concatenates(self, tmp_path: Path):
        """All non-mezzanine *.xml files are read and entries concatenated."""
        _write_sample(tmp_path / "sloleks_3.1_001.xml")
        _write_sample(tmp_path / "sloleks_3.1_002.xml")
        # Mezzanine should be skipped.
        (tmp_path / "sloleks_3.1_mezzanine.xml").write_text(
            "<lexicon/>",
            encoding="utf-8",
        )
        records = list(sloleks.iter_sloleks_dir(tmp_path))
        # Two valid entries per file, two files -> 4 records.
        assert len(records) == 4
        lemmas = [r["lemma"] for r in records]
        assert lemmas == ["hiša", "biti", "hiša", "biti"]


class TestToTokenizerEvalConverter:
    """Integration tests for scripts/data/to_tokenization.py."""

    def test_convert_sloleks_writes_tagged_records(self, tmp_path: Path):
        """The converter produces JSONL with dataset/task tags."""
        raw_dir = tmp_path / "raw" / "sloleks"
        _write_sample(raw_dir / "sloleks_3.1_001.xml")
        out_dir = tmp_path / "out"

        count = to_tokenizer_eval.convert_dataset(
            "sloleks",
            raw_dir,
            out_dir,
        )

        assert count == 2
        out_path = out_dir / "sloleks.jsonl.gz"
        assert out_path.exists()

        with gzip.open(out_path, "rt", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]

        assert len(records) == 2
        for record in records:
            assert record["dataset"] == "sloleks"
            assert record["task"] == "TOKENIZER"
            assert "lemma" in record
            assert "forms" in record

    def test_convert_skips_when_output_exists(self, tmp_path: Path):
        """Existing outputs are skipped unless `force` is True."""
        raw_dir = tmp_path / "raw" / "sloleks"
        _write_sample(raw_dir / "sloleks_3.1_001.xml")
        out_dir = tmp_path / "out"

        first = to_tokenizer_eval.convert_dataset("sloleks", raw_dir, out_dir)
        second = to_tokenizer_eval.convert_dataset("sloleks", raw_dir, out_dir)

        assert first == 2
        assert second == 0  # skipped

        third = to_tokenizer_eval.convert_dataset(
            "sloleks", raw_dir, out_dir, force=True,
        )
        assert third == 2

    def test_convert_zero_records_raises(self, tmp_path: Path):
        """A conversion that yields no records fails loudly instead of exit 0."""
        raw_dir = tmp_path / "raw" / "sloleks"
        # Valid <lexicon> XML but with no usable entries.
        (raw_dir).mkdir(parents=True, exist_ok=True)
        (raw_dir / "sloleks_3.1_001.xml").write_text(
            "<lexicon><entry><head><headword><lemma>x</lemma></headword></head>"
            "<body><wordFormList/></body></entry></lexicon>",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            to_tokenizer_eval.convert_dataset("sloleks", raw_dir, tmp_path / "out")

    def test_unknown_dataset_returns_none(self, tmp_path: Path):
        """An unregistered dataset key yields None and skips silently."""
        result = to_tokenizer_eval.convert_dataset(
            "does_not_exist",
            tmp_path,
            tmp_path / "out",
        )
        assert result is None

    def test_missing_xml_raises(self, tmp_path: Path):
        """The Sloleks reader fails fast when no XML files are found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            list(to_tokenizer_eval._read_sloleks(empty_dir))


class TestLoadTokenizationConfig:
    """Tests for the tokenization config loader."""

    def test_parses_minimum_required_fields(self, tmp_path: Path):
        """A well-formed tokenization config parses into the expected dict."""
        config = tmp_path / "tokenization.yaml"
        config.write_text(
            dedent(
                """\
                input_dir: /tmp/raw
                output_dir: /tmp/out
                datasets:
                  - sloleks
                """
            ),
            encoding="utf-8",
        )
        cfg = to_tokenizer_eval.load_tokenization_config(config)
        assert cfg["input_dir"] == Path("/tmp/raw")
        assert cfg["output_dir"] == Path("/tmp/out")
        assert cfg["datasets"] == ["sloleks"]
