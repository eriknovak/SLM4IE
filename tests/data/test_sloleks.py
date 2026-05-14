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


SAMPLE_TEI = dedent(
    """\
    <?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text>
        <body>
          <entry xml:id="hisa-001">
            <form type="lemma">
              <orth>hiša</orth>
            </form>
            <gramGrp>
              <gram type="msd">Ncfsn</gram>
            </gramGrp>
            <form type="inflectedForm" xml:id="hisa-001-2">
              <orth>hiše</orth>
              <gramGrp>
                <gram type="msd">Ncfsg</gram>
              </gramGrp>
            </form>
            <form type="inflectedForm" xml:id="hisa-001-3">
              <orth>hiši</orth>
              <gramGrp>
                <gram type="msd">Ncfsd</gram>
              </gramGrp>
            </form>
          </entry>
          <entry xml:id="biti-001">
            <form type="lemma">
              <orth>biti</orth>
              <msd>Va-n</msd>
            </form>
            <form type="inflectedForm" xml:id="biti-001-2" feats="Va-r1s-n">
              <orth>sem</orth>
            </form>
          </entry>
          <entry xml:id="empty-001">
            <gramGrp>
              <gram type="msd">Xx</gram>
            </gramGrp>
          </entry>
        </body>
      </text>
    </TEI>
    """
)


def _write_sample(path: Path) -> Path:
    """Write SAMPLE_TEI to *path* and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SAMPLE_TEI, encoding="utf-8")
    return path


class TestLocalName:
    """Unit tests for sloleks._local_name."""

    def test_strips_clark_notation(self):
        """Clark-notation prefixes are stripped."""
        assert sloleks._local_name("{http://example/ns}entry") == "entry"

    def test_passthrough_when_unprefixed(self):
        """Tags without a namespace pass through unchanged."""
        assert sloleks._local_name("entry") == "entry"


class TestFindMsd:
    """Unit tests for sloleks._find_msd."""

    def test_msd_via_gram_type(self):
        """`<gram type="msd">` text is recognized."""
        elem = ET.fromstring(
            '<form xmlns="http://www.tei-c.org/ns/1.0">'
            '<gram type="msd">Ncfsn</gram></form>'
        )
        assert sloleks._find_msd(elem) == "Ncfsn"

    def test_msd_via_dedicated_element(self):
        """A direct `<msd>` element is recognized."""
        elem = ET.fromstring(
            '<form xmlns="http://www.tei-c.org/ns/1.0">'
            "<msd>Va-n</msd></form>"
        )
        assert sloleks._find_msd(elem) == "Va-n"

    def test_msd_via_feats_attribute(self):
        """`feats` attribute is used as a last resort."""
        elem = ET.fromstring('<form feats="Ncfpn"></form>')
        assert sloleks._find_msd(elem) == "Ncfpn"

    def test_no_msd_returns_none(self):
        """Missing MSD yields None."""
        elem = ET.fromstring("<form><orth>x</orth></form>")
        assert sloleks._find_msd(elem) is None


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
        assert first["entry_id"] == "hisa-001"
        assert first["lemma"] == "hiša"
        assert first["lemma_msd"] == "Ncfsn"
        forms = {f["form"]: f["msd"] for f in first["forms"]}
        assert forms == {
            "hiša": "Ncfsn",
            "hiše": "Ncfsg",
            "hiši": "Ncfsd",
        }

    def test_msd_inside_lemma_form(self):
        """A lemma MSD declared inside the `<form type=lemma>` is picked up."""
        path = Path("/tmp/_sloleks_unit_test.xml")
        try:
            _write_sample(path)
            records = list(sloleks.iter_sloleks_entries(path))
        finally:
            if path.exists():
                path.unlink()

        biti = next(r for r in records if r["lemma"] == "biti")
        assert biti["lemma_msd"] == "Va-n"
        forms = {f["form"]: f["msd"] for f in biti["forms"]}
        assert forms == {"biti": "Va-n", "sem": "Va-r1s-n"}

    def test_entry_without_lemma_is_skipped(self):
        """Entries that lack a lemma orthography produce no record."""
        path = Path("/tmp/_sloleks_unit_test.xml")
        try:
            _write_sample(path)
            records = list(sloleks.iter_sloleks_entries(path))
        finally:
            if path.exists():
                path.unlink()
        assert all(r["entry_id"] != "empty-001" for r in records)


class TestIterSloleksDir:
    """Tests for the directory-level walker."""

    def test_skips_mezzanine_and_concatenates(self, tmp_path: Path):
        """All non-mezzanine *.xml files are read and entries concatenated."""
        _write_sample(tmp_path / "sloleks_3.1_001.xml")
        _write_sample(tmp_path / "sloleks_3.1_002.xml")
        # Mezzanine should be skipped.
        (tmp_path / "sloleks_3.1_mezzanine.xml").write_text(
            "<TEI xmlns='http://www.tei-c.org/ns/1.0'/>",
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
