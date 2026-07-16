"""Tests for the COLESLAW extractor."""

import json
import logging
from pathlib import Path
from typing import List

import pytest

from slm4ie.data.extractors.coleslaw import ColeslawExtractor


def _write_jsonl(path: Path, records: List[dict]) -> None:
    """Write *records* as one JSON object per line to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")


@pytest.fixture()
def extractor() -> ColeslawExtractor:
    """Return a ColeslawExtractor instance."""
    return ColeslawExtractor()


class TestColeslawExtractor:
    """Tests for ColeslawExtractor."""

    def test_pisrs_uses_text_field(
        self, extractor: ColeslawExtractor, tmp_path: Path
    ) -> None:
        """Records with a populated 'text' field flow through directly."""
        _write_jsonl(
            tmp_path / "PISRS" / "register.jsonl",
            [{"id": 1, "text": "Zakon o nečem."}],
        )
        docs = list(extractor.extract(tmp_path, "coleslaw", "legal"))
        assert len(docs) == 1
        assert docs[0].text == "Zakon o nečem."
        assert docs[0].metadata["subcorpus"] == "PISRS"

    def test_usrs_uses_full_text(
        self, extractor: ColeslawExtractor, tmp_path: Path
    ) -> None:
        """Records expose 'fullText' instead of 'text' (USRS)."""
        _write_jsonl(
            tmp_path / "USRS" / "usrs.jsonl",
            [{"id": "Up-1", "fullText": "Sklep ustavnega sodišča."}],
        )
        docs = list(extractor.extract(tmp_path, "coleslaw", "legal"))
        assert len(docs) == 1
        assert docs[0].text == "Sklep ustavnega sodišča."
        assert docs[0].metadata["subcorpus"] == "USRS"

    def test_sodna_courts_concatenates_sections(
        self, extractor: ColeslawExtractor, tmp_path: Path
    ) -> None:
        """sp_courts records combine jedro/izrek/obrazlozitev in order."""
        _write_jsonl(
            tmp_path / "SodnaPraksa" / "sp_courts.jsonl",
            [
                {
                    "id": "c1",
                    "jedro": "Bistvo zadeve.",
                    "izrek": "Razveljavi se.",
                    "obrazlozitev": "Obrazložitev sledi.",
                }
            ],
        )
        docs = list(extractor.extract(tmp_path, "coleslaw", "legal"))
        assert (
            docs[0].text
            == "Bistvo zadeve.\n\nRazveljavi se.\n\nObrazložitev sledi."
        )

    def test_sodna_claims_concatenates_structured_fields(
        self, extractor: ColeslawExtractor, tmp_path: Path
    ) -> None:
        """sp_claims fields are joined when 'text' / 'fullText' absent."""
        _write_jsonl(
            tmp_path / "SodnaPraksa" / "sp_claims.jsonl",
            [
                {
                    "id": "750",
                    "skodni_dogodek": "Prometna nesreča.",
                    "poskodba": "Zvin vratu.",
                    "telesne_bolecine": "Tri tedne.",
                }
            ],
        )
        docs = list(extractor.extract(tmp_path, "coleslaw", "legal"))
        assert (
            docs[0].text
            == "Prometna nesreča.\n\nZvin vratu.\n\nTri tedne."
        )

    def test_doc_id_falls_back_to_id_field(
        self, extractor: ColeslawExtractor, tmp_path: Path
    ) -> None:
        """The 'id' field becomes doc_id when 'doc_id' is absent."""
        _write_jsonl(
            tmp_path / "PISRS" / "f.jsonl",
            [{"id": 42, "text": "x"}],
        )
        docs = list(extractor.extract(tmp_path, "coleslaw", "legal"))
        assert docs[0].doc_id == "42"

    def test_skips_records_without_any_text(
        self, extractor: ColeslawExtractor, tmp_path: Path
    ) -> None:
        """Records lacking every recognised text field are skipped."""
        _write_jsonl(
            tmp_path / "PISRS" / "f.jsonl",
            [{"id": 1, "metadata": {"foo": "bar"}}],
        )
        docs = list(extractor.extract(tmp_path, "coleslaw", "legal"))
        assert docs == []

    @pytest.mark.parametrize(
        ("record", "expected_text"),
        [
            ({"id": 1, "text": "Samo besedilo."}, "Samo besedilo."),
            ({"id": 2, "fullText": "Samo fullText."}, "Samo fullText."),
            (
                {"id": 3, "jedro": "Bistvo.", "izrek": "Izrek."},
                "Bistvo.\n\nIzrek.",
            ),
            (
                {"id": 4, "skodni_dogodek": "Nesreča.", "poskodba": "Zvin."},
                "Nesreča.\n\nZvin.",
            ),
        ],
        ids=["text", "fullText", "sp_courts", "sp_claims"],
    )
    def test_single_source_emits_no_warning(
        self,
        extractor: ColeslawExtractor,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        record: dict,
        expected_text: str,
    ) -> None:
        """Records with exactly one text source extract silently."""
        _write_jsonl(tmp_path / "PISRS" / "f.jsonl", [record])
        with caplog.at_level(logging.WARNING, logger="slm4ie.data.extractors.coleslaw"):
            docs = list(extractor.extract(tmp_path, "coleslaw", "legal"))
        assert len(docs) == 1
        assert docs[0].text == expected_text
        assert caplog.records == []

    def test_text_and_full_text_warns_and_uses_text(
        self,
        extractor: ColeslawExtractor,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A record with both 'text' and 'fullText' keeps 'text' and warns."""
        _write_jsonl(
            tmp_path / "PISRS" / "f.jsonl",
            [{"id": 1, "text": "Izbrano.", "fullText": "Zasenčeno."}],
        )
        with caplog.at_level(logging.WARNING, logger="slm4ie.data.extractors.coleslaw"):
            docs = list(extractor.extract(tmp_path, "coleslaw", "legal"))
        assert docs[0].text == "Izbrano."
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "'text'" in message
        assert "'fullText'" in message
        assert "using text" in message
        assert "PISRS" in message

    def test_full_text_and_sp_courts_warns_and_uses_full_text(
        self,
        extractor: ColeslawExtractor,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A record with 'fullText' and sp_courts fields keeps 'fullText' and warns."""
        _write_jsonl(
            tmp_path / "SodnaPraksa" / "f.jsonl",
            [{"id": "c9", "fullText": "Izbrano.", "jedro": "Zasenčeno."}],
        )
        with caplog.at_level(logging.WARNING, logger="slm4ie.data.extractors.coleslaw"):
            docs = list(extractor.extract(tmp_path, "coleslaw", "legal"))
        assert docs[0].text == "Izbrano."
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "'fullText'" in message
        assert "'sp_courts'" in message
        assert "using fullText" in message

    def test_no_text_sources_skips_without_warning(
        self,
        extractor: ColeslawExtractor,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Records with no text source at all are skipped silently."""
        _write_jsonl(
            tmp_path / "PISRS" / "f.jsonl",
            [{"id": 1, "title": "Brez besedila"}],
        )
        with caplog.at_level(logging.WARNING, logger="slm4ie.data.extractors.coleslaw"):
            docs = list(extractor.extract(tmp_path, "coleslaw", "legal"))
        assert docs == []
        assert caplog.records == []

    def test_invalid_lines_logged_and_skipped(
        self, extractor: ColeslawExtractor, tmp_path: Path
    ) -> None:
        """Malformed JSON lines are skipped, valid ones still extracted."""
        path = tmp_path / "PISRS" / "f.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "{not valid\n" + json.dumps({"id": 1, "text": "ok"}) + "\n",
            encoding="utf-8",
        )
        docs = list(extractor.extract(tmp_path, "coleslaw", "legal"))
        assert len(docs) == 1
        assert docs[0].text == "ok"
