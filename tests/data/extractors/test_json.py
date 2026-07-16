"""Tests for the JSON-array extractor."""

import json
from pathlib import Path

import pytest

from slm4ie.data.extractors.json import JsonExtractor


@pytest.fixture()
def extractor() -> JsonExtractor:
    """Return a JsonExtractor instance."""
    return JsonExtractor()


def _write_json(path: Path, payload) -> None:
    """Serialize *payload* as JSON to *path*."""
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestJsonExtractor:
    """Tests for JsonExtractor."""

    def test_extracts_records_from_array(
        self, extractor: JsonExtractor, tmp_path: Path
    ) -> None:
        """One Document is produced per array element with text."""
        _write_json(
            tmp_path / "data.json",
            [
                {"text_id": "a", "text": "First record."},
                {"text_id": "b", "text": "Second record."},
            ],
        )
        docs = list(extractor.extract(tmp_path, "src", "med"))
        assert [d.text for d in docs] == [
            "First record.",
            "Second record.",
        ]

    def test_extra_fields_become_metadata(
        self, extractor: JsonExtractor, tmp_path: Path
    ) -> None:
        """Non-text, non-doc_id fields are surfaced as metadata."""
        _write_json(
            tmp_path / "data.json",
            [{"text": "hi", "source_file": "x.txt", "category": "A"}],
        )
        docs = list(extractor.extract(tmp_path, "src", "med"))
        assert docs[0].metadata == {
            "source_file": "x.txt",
            "category": "A",
        }

    def test_none_valued_fields_dropped_from_metadata(
        self, extractor: JsonExtractor, tmp_path: Path
    ) -> None:
        """Fields with a null value never land in metadata."""
        _write_json(
            tmp_path / "data.json",
            [{"text": "hi", "kept": 1, "dropped": None}],
        )
        docs = list(extractor.extract(tmp_path, "src", "med"))
        assert docs[0].metadata == {"kept": 1}

    def test_text_and_id_field_overrides(
        self, extractor: JsonExtractor, tmp_path: Path
    ) -> None:
        """`text_field`/`id_field` from the metadata block are honored."""
        _write_json(
            tmp_path / "data.json",
            [{"uri": "n1", "body": "News body.", "title": "T"}],
        )
        docs = list(
            extractor.extract(
                tmp_path,
                "src",
                "news",
                metadata={"text_field": "body", "id_field": "uri"},
            )
        )
        assert len(docs) == 1
        assert docs[0].text == "News body."
        assert docs[0].doc_id == "n1"
        assert docs[0].metadata == {"title": "T"}

    def test_metadata_fields_whitelist(
        self, extractor: JsonExtractor, tmp_path: Path
    ) -> None:
        """Only whitelisted fields are kept as metadata."""
        _write_json(
            tmp_path / "data.json",
            [
                {
                    "doc_id": "a",
                    "text": "hi",
                    "title": "T",
                    "internal": "drop me",
                    "missing_ok": None,
                }
            ],
        )
        docs = list(
            extractor.extract(
                tmp_path,
                "src",
                "med",
                metadata={"metadata_fields": ["title", "absent", "missing_ok"]},
            )
        )
        assert docs[0].metadata == {"title": "T"}

    def test_empty_text_skipped_with_text_field_override(
        self, extractor: JsonExtractor, tmp_path: Path
    ) -> None:
        """Empty/missing text is still skipped under a custom text field."""
        _write_json(
            tmp_path / "data.json",
            [{"body": ""}, {"text": "wrong field"}, {"body": "ok"}],
        )
        docs = list(
            extractor.extract(tmp_path, "src", "med", metadata={"text_field": "body"})
        )
        assert [d.text for d in docs] == ["ok"]

    def test_single_object_with_overrides(
        self, extractor: JsonExtractor, tmp_path: Path
    ) -> None:
        """A bare top-level object also honors the field mapping."""
        _write_json(tmp_path / "data.json", {"body": "solo", "uri": "u1"})
        docs = list(
            extractor.extract(
                tmp_path,
                "src",
                "med",
                metadata={"text_field": "body", "id_field": "uri"},
            )
        )
        assert len(docs) == 1
        assert docs[0].text == "solo"
        assert docs[0].doc_id == "u1"

    def test_skips_records_without_text(
        self, extractor: JsonExtractor, tmp_path: Path
    ) -> None:
        """Records with empty or missing text are skipped."""
        _write_json(
            tmp_path / "data.json",
            [
                {"text": ""},
                {"foo": "bar"},
                {"text": "ok"},
            ],
        )
        docs = list(extractor.extract(tmp_path, "src", "med"))
        assert len(docs) == 1
        assert docs[0].text == "ok"

    def test_top_level_object_is_treated_as_single_record(
        self, extractor: JsonExtractor, tmp_path: Path
    ) -> None:
        """A bare JSON object becomes one Document."""
        _write_json(tmp_path / "data.json", {"text": "just one"})
        docs = list(extractor.extract(tmp_path, "src", "med"))
        assert len(docs) == 1
        assert docs[0].text == "just one"

    def test_invalid_json_is_skipped(
        self, extractor: JsonExtractor, tmp_path: Path
    ) -> None:
        """Files with bad JSON are skipped with a warning."""
        (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
        docs = list(extractor.extract(tmp_path, "src", "med"))
        assert docs == []

    def test_recursive_scan(
        self, extractor: JsonExtractor, tmp_path: Path
    ) -> None:
        """JSON files in nested subdirectories are discovered."""
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_json(sub / "nested.json", [{"text": "inside"}])
        docs = list(extractor.extract(tmp_path, "src", "med"))
        assert len(docs) == 1
