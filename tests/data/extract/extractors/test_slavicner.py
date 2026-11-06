"""Tests for SlavicNERExtractor."""

from pathlib import Path

import pytest

from slm4ie.data.extract.extractors.slavicner import SlavicNERExtractor

# ===============================
# Test Data
# ===============================

_TEXT = "Ryanair leti iz Manchestra v Zadar.\n"
_MENTIONS = "0\t7\tRyanair\tRyanair\tORG\tORG-Ryanair\n16\t26\tManchestra\tManchester\tloc\tGPE-Manchester\n"


def _write_document(root: Path, topic: str, language: str, name: str, text: str, mentions: str | None) -> None:
    """Write one SlavicNER text file and, when given, its mention file.

    Args:
        root: Corpus root holding `raw/` and `ner/`.
        topic: Topic folder name.
        language: Language folder name.
        name: File stem.
        text: Document text.
        mentions: Mention file content, or None to leave it out.
    """
    raw_dir = root / "raw" / topic / language
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{name}.txt").write_text(text, encoding="utf-8")
    if mentions is not None:
        ner_dir = root / "ner" / topic / language
        ner_dir.mkdir(parents=True, exist_ok=True)
        (ner_dir / f"{name}.tsv").write_text(mentions, encoding="utf-8")


# ===============================
# SlavicNER extractor
# ===============================


def test_mentions_become_character_spans_with_upper_case_labels(tmp_path: Path) -> None:
    """Each mention line becomes a `[begin, end, LABEL]` span over the unchanged text."""
    _write_document(tmp_path / "SlavicNER-main", "ryanair", "sl", "sl-1", _TEXT, _MENTIONS)

    (doc,) = SlavicNERExtractor().extract(tmp_path, "slavicner", "news")

    assert doc.text == _TEXT
    assert doc.annotations.spans == [[0, 7, "ORG"], [16, 26, "LOC"]]


def test_doc_id_carries_the_topic_so_repeated_names_stay_unique(tmp_path: Path) -> None:
    """Two topics holding the same file name give two distinct ids."""
    root = tmp_path / "SlavicNER-main"
    _write_document(root, "brexit", "sl", "sl-1", _TEXT, _MENTIONS)
    _write_document(root, "ryanair", "sl", "sl-1", "Ryanair leti v Zadar.\n", "0\t7\tRyanair\tRyanair\tORG\tx\n")

    docs = list(SlavicNERExtractor().extract(tmp_path, "slavicner", "news"))

    assert [d.doc_id for d in docs] == ["brexit/sl/sl-1", "ryanair/sl/sl-1"]
    assert docs[0].metadata == {"topic": "brexit", "language": "sl"}


def test_document_without_mentions_has_empty_span_list(tmp_path: Path) -> None:
    """An empty mention file yields an annotated document with no spans."""
    _write_document(tmp_path / "SlavicNER-main", "ryanair", "sl", "sl-1", _TEXT, "")

    (doc,) = SlavicNERExtractor().extract(tmp_path, "slavicner", "news")

    assert doc.annotations.spans == []


def test_rejects_missing_mention_file(tmp_path: Path) -> None:
    """A text file with no mention file beside it raises."""
    _write_document(tmp_path / "SlavicNER-main", "ryanair", "sl", "sl-1", _TEXT, None)

    with pytest.raises(FileNotFoundError, match="sl-1.tsv"):
        list(SlavicNERExtractor().extract(tmp_path, "slavicner", "news"))


def test_rejects_offsets_that_miss_the_surface_form(tmp_path: Path) -> None:
    """Offsets that do not cover the mention's form raise with the line number."""
    _write_document(tmp_path / "SlavicNER-main", "ryanair", "sl", "sl-1", _TEXT, "1\t8\tRyanair\tRyanair\tORG\tx\n")

    with pytest.raises(ValueError, match=r"sl-1.tsv:1"):
        list(SlavicNERExtractor().extract(tmp_path, "slavicner", "news"))


def test_repeated_text_is_kept_once_under_its_first_path(tmp_path: Path) -> None:
    """An article stored under two topics yields one document, from the first path in sorted order."""
    root = tmp_path / "SlavicNER-main"
    _write_document(root, "covid-19", "sl", "sl-9", _TEXT, _MENTIONS)
    _write_document(root, "brexit", "sl", "sl-3", _TEXT, _MENTIONS)

    docs = list(SlavicNERExtractor().extract(tmp_path, "slavicner", "news"))

    assert [d.doc_id for d in docs] == ["brexit/sl/sl-3"]
