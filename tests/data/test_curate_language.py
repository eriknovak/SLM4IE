"""Tests for slm4ie.data.curate.language.LinguaLanguageFilter."""

import importlib.metadata  # noqa: F401  (datatrove workaround)
import importlib.util  # noqa: F401  (datatrove workaround)
from typing import Any, Dict, List

import pytest

pytest.importorskip("datatrove")
pytest.importorskip("lingua")

from datatrove.data import Document  # noqa: E402

from slm4ie.data.curate.language import LinguaLanguageFilter  # noqa: E402


def _doc(text: str, doc_id: str = "x") -> Document:
    """Build a Document with empty metadata for one of the test strings."""
    return Document(text=text, id=doc_id, metadata={})


def _consume(filt: LinguaLanguageFilter, docs: List[Document]) -> List[Document]:
    """Run *docs* through *filt* and return everything yielded downstream."""
    out: List[Document] = []
    for d in filt.run(iter(docs)):
        out.append(d)
    return out


SLOVENIAN = "Slovenščina je uradni jezik Republike Slovenije in eden izmed uradnih jezikov Evropske unije."
ENGLISH = "The quick brown fox jumps over the lazy dog and then runs back home."
GERMAN = "Das Wetter in Berlin ist heute schön und die Sonne scheint den ganzen Tag."


class TestLinguaLanguageFilter:
    """Behavior of the lingua-py-backed datatrove pipeline step."""

    def test_tag_mode_keeps_all_and_labels_each_doc(self) -> None:
        """Tag mode: every doc passes through with language metadata set."""
        filt = LinguaLanguageFilter(target="sl", mode="tag", threshold=0.99)
        docs = [_doc(SLOVENIAN, "sl"), _doc(ENGLISH, "en"), _doc(GERMAN, "de")]
        kept = _consume(filt, docs)

        assert [d.id for d in kept] == ["sl", "en", "de"]
        for d in kept:
            assert "language" in d.metadata
            assert "language_score" in d.metadata
            assert 0.0 <= d.metadata["language_score"] <= 1.0

    def test_tag_mode_predicts_correct_language(self) -> None:
        """Lingua picks the right language for clean Slovenian / English / German."""
        filt = LinguaLanguageFilter(target="sl", mode="tag")
        kept = _consume(
            filt,
            [_doc(SLOVENIAN, "sl"), _doc(ENGLISH, "en"), _doc(GERMAN, "de")],
        )
        by_id = {d.id: d.metadata["language"] for d in kept}
        assert by_id["sl"] == "sl"
        assert by_id["en"] == "en"
        assert by_id["de"] == "de"

    def test_filter_mode_drops_below_threshold_into_exclusion_writer(self) -> None:
        """Filter mode routes non-Slovenian docs to the exclusion writer."""
        excluded: List[Dict[str, Any]] = []

        class _CapturingWriter:
            """Minimal stand-in for a datatrove DiskWriter."""

            def __enter__(self) -> "_CapturingWriter":
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: D401, ANN001
                """Swallow nothing."""
                return False

            def write(self, doc: Document, rank: int = 0) -> None:
                """Append the dropped document to the captured list."""
                excluded.append({"id": doc.id, "language": doc.metadata.get("language")})

        filt = LinguaLanguageFilter(
            target="sl",
            mode="filter",
            threshold=0.5,
            exclusion_writer=_CapturingWriter(),
        )
        kept = _consume(
            filt,
            [_doc(SLOVENIAN, "sl"), _doc(ENGLISH, "en"), _doc(GERMAN, "de")],
        )

        kept_ids = [d.id for d in kept]
        excluded_ids = [e["id"] for e in excluded]
        assert "sl" in kept_ids
        assert set(excluded_ids) >= {"en", "de"}

    def test_invalid_mode_raises(self) -> None:
        """The constructor refuses any mode other than 'tag' or 'filter'."""
        with pytest.raises(ValueError):
            LinguaLanguageFilter(mode="bogus")
