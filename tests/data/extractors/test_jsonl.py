"""Tests for JsonlExtractor."""

import json
from pathlib import Path
from typing import Dict, List

from slm4ie.data.extractors.jsonl import JsonlExtractor
from slm4ie.data.schema import Document


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: List[Dict]) -> None:
    """Write list of dicts as JSONL to the given path.

    Args:
        path (Path): File path to write.
        records (List[Dict]): Records to serialize as JSONL.
    """
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract(tmp_path: Path, records: List[Dict]) -> List[Document]:
    """Write records to a .jsonl file and extract documents.

    Args:
        tmp_path (Path): Temporary directory.
        records (List[Dict]): Records to write and extract.

    Returns:
        List[Document]: Extracted documents.
    """
    _write_jsonl(tmp_path / "test.jsonl", records)
    extractor = JsonlExtractor()
    return list(extractor.extract(tmp_path, source="web", domain="web"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJsonlExtractor:
    """Tests for JsonlExtractor."""

    def test_extracts_text_field(self, tmp_path: Path) -> None:
        """Text field is populated from each JSONL record."""
        records = [
            {"text": "Dober dan.", "doc_id": "d1"},
            {"text": "Kako si?", "doc_id": "d2"},
        ]
        docs = _extract(tmp_path, records)
        assert len(docs) == 2
        assert docs[0].text == "Dober dan."
        assert docs[1].text == "Kako si?"

    def test_preserves_annotations(self, tmp_path: Path) -> None:
        """Tokens parsed from paragraphs have correct form, lemma, upos, feats."""
        record = {
            "text": "Dober dan.",
            "doc_id": "d1",
            "paragraphs": [
                {
                    "sentences": [
                        {
                            "tokens": [
                                {
                                    "form": "Dober",
                                    "lemma": "dober",
                                    "upos": "ADJ",
                                    "feats": "Case=Nom",
                                },
                                {
                                    "form": "dan",
                                    "lemma": "dan",
                                    "upos": "NOUN",
                                    "feats": "Case=Nom",
                                },
                                {
                                    "form": ".",
                                    "lemma": ".",
                                    "upos": "PUNCT",
                                    "feats": None,
                                },
                            ]
                        }
                    ]
                }
            ],
        }
        docs = _extract(tmp_path, [record])
        assert len(docs) == 1
        ann = docs[0].annotations
        assert ann is not None
        tokens = ann.tokens
        assert len(tokens) == 3
        assert tokens[0].form == "Dober"
        assert tokens[0].lemma == "dober"
        assert tokens[0].upos == "ADJ"
        assert tokens[0].feats == "Case=Nom"
        assert tokens[1].form == "dan"
        assert tokens[1].upos == "NOUN"
        assert tokens[2].form == "."
        assert tokens[2].upos == "PUNCT"

    def test_text_only_no_annotations(self, tmp_path: Path) -> None:
        """Records without paragraphs key produce annotations=None."""
        records = [{"text": "Samo besedilo."}]
        docs = _extract(tmp_path, records)
        assert len(docs) == 1
        assert docs[0].annotations is None

    def test_skips_empty_text(self, tmp_path: Path) -> None:
        """Records with empty or missing text are skipped."""
        records = [
            {"text": ""},
            {"text": "Veljavno besedilo.", "doc_id": "d1"},
            {"doc_id": "d2"},
        ]
        docs = _extract(tmp_path, records)
        assert len(docs) == 1
        assert docs[0].text == "Veljavno besedilo."

    def test_preserves_metadata_fields(self, tmp_path: Path) -> None:
        """Non-reserved fields are stored in metadata."""
        records = [
            {
                "text": "Primer.",
                "doc_id": "d1",
                "url": "http://example.com",
                "lang": "sl",
            }
        ]
        docs = _extract(tmp_path, records)
        assert len(docs) == 1
        meta = docs[0].metadata
        assert meta["url"] == "http://example.com"
        assert meta["lang"] == "sl"
        assert "text" not in meta
        assert "doc_id" not in meta
        assert "paragraphs" not in meta

    def test_registered_as_jsonl(self) -> None:
        """JsonlExtractor is registered under the 'jsonl' key."""
        from slm4ie.data.extractors import get_extractor

        extractor = get_extractor("jsonl")
        assert isinstance(extractor, JsonlExtractor)
