"""Tests for slm4ie/data/tokenization/run.py and the lexicon readers it dispatches to."""

import gzip
import json
import zipfile
from pathlib import Path

import pytest

from slm4ie.data.tokenization import lexicons, run as tokenization
from tests.data.tokenization.lexicons.test_sloleks import SAMPLE_LEXICON, _write_sample


class TestToTokenizerEvalConverter:
    """Integration tests for slm4ie/data/tokenization/run.py."""

    def test_convert_sloleks_writes_tagged_records(self, tmp_path: Path):
        """The converter produces JSONL with dataset/task tags."""
        raw_dir = tmp_path / "raw" / "sloleks"
        _write_sample(raw_dir / "sloleks_3.1_001.xml")
        out_dir = tmp_path / "out"

        count = tokenization.convert_dataset(
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

        first = tokenization.convert_dataset("sloleks", raw_dir, out_dir)
        second = tokenization.convert_dataset("sloleks", raw_dir, out_dir)

        assert first == 2
        assert second == 0  # skipped

        third = tokenization.convert_dataset(
            "sloleks",
            raw_dir,
            out_dir,
            force=True,
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
            tokenization.convert_dataset("sloleks", raw_dir, tmp_path / "out")

    def test_unknown_dataset_returns_none(self, tmp_path: Path):
        """An unregistered dataset key yields None and skips silently."""
        result = tokenization.convert_dataset(
            "does_not_exist",
            tmp_path,
            tmp_path / "out",
        )
        assert result is None

    def test_missing_xml_raises(self, tmp_path: Path):
        """The Sloleks lexicon fails fast when no XML files are found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            list(lexicons._read_sloleks(empty_dir))

    def test_read_sloleks_auto_unzips(self, tmp_path: Path):
        """The Sloleks lexicon unpacks a zip in raw_dir when no XML is present."""
        raw_dir = tmp_path / "raw" / "sloleks"
        raw_dir.mkdir(parents=True)
        with zipfile.ZipFile(raw_dir / "Sloleks.zip", "w") as zf:
            zf.writestr("sloleks_3.1_001.xml", SAMPLE_LEXICON)

        records = list(lexicons._read_sloleks(raw_dir))

        assert len(records) == 2
        assert all(r["dataset"] == "sloleks" for r in records)

    def test_read_sloleks_relations_auto_unzips(self, tmp_path: Path):
        """The relations lexicon unpacks a zip in raw_dir when no TSV is present."""
        raw_dir = tmp_path / "raw" / "sloleks_relations"
        raw_dir.mkdir(parents=True)
        rows = [
            "original\trelated\torig_dec\trel_dec\tmte_o\tmte_r\toid\trid\tov\trule\tpat\tadq",
            "pisati\tpisatelj\tpis_ati\tpis_at_elj\tG\tSom\t1\t2\tpis\tG.Som.5\tx\t2",
        ]
        with zipfile.ZipFile(raw_dir / "relations.zip", "w") as zf:
            zf.writestr("nssss_sloleks_word_relations_1.1.tsv", "\n".join(rows) + "\n")

        records = list(lexicons._read_sloleks_relations(raw_dir))

        assert len(records) == 1
        assert records[0]["lemma"] == "pisatelj"
        assert records[0]["morphemes"] == ["pis", "at", "elj"]
        assert records[0]["verified"] is True
        assert records[0]["dataset"] == "sloleks_relations"
