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
        filt = LinguaLanguageFilter(targets=["sl"], mode="tag")
        docs = [_doc(SLOVENIAN, "sl"), _doc(ENGLISH, "en"), _doc(GERMAN, "de")]
        kept = _consume(filt, docs)

        assert [d.id for d in kept] == ["sl", "en", "de"]
        for d in kept:
            assert "language" in d.metadata
            language = d.metadata["language"]
            assert language is None or isinstance(language, str)

    def test_tag_mode_predicts_correct_language(self) -> None:
        """Lingua picks the right language for clean Slovenian / English / German."""
        filt = LinguaLanguageFilter(targets=["sl"], mode="tag")
        kept = _consume(
            filt,
            [_doc(SLOVENIAN, "sl"), _doc(ENGLISH, "en"), _doc(GERMAN, "de")],
        )
        by_id = {d.id: d.metadata["language"] for d in kept}
        assert by_id["sl"] == "sl"
        assert by_id["en"] == "en"
        assert by_id["de"] == "de"

    def test_filter_mode_keeps_target_languages_only(self) -> None:
        """Filter mode routes non-target docs to the exclusion writer."""
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
            targets=["sl"],
            mode="filter",
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

    def test_filter_mode_with_multiple_targets(self) -> None:
        """A multi-language target set keeps every covered language."""
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
            targets=["sl", "en"],
            mode="filter",
            exclusion_writer=_CapturingWriter(),
        )
        kept = _consume(
            filt,
            [_doc(SLOVENIAN, "sl"), _doc(ENGLISH, "en"), _doc(GERMAN, "de")],
        )

        kept_ids = {d.id for d in kept}
        excluded_ids = {e["id"] for e in excluded}
        assert {"sl", "en"}.issubset(kept_ids)
        assert "de" in excluded_ids

    def test_invalid_mode_raises(self) -> None:
        """The constructor refuses any mode other than 'tag' or 'filter'."""
        with pytest.raises(ValueError):
            LinguaLanguageFilter(mode="bogus")

    def test_invalid_minimum_relative_distance_raises(self) -> None:
        """The constructor enforces minimum_relative_distance in [0, 1)."""
        with pytest.raises(ValueError):
            LinguaLanguageFilter(minimum_relative_distance=1.0)
        with pytest.raises(ValueError):
            LinguaLanguageFilter(minimum_relative_distance=-0.1)

    def test_empty_targets_raises(self) -> None:
        """An empty targets list is rejected at construction."""
        with pytest.raises(ValueError):
            LinguaLanguageFilter(targets=[])

    def test_invalid_max_chars_raises(self) -> None:
        """Non-positive max_chars values are rejected."""
        with pytest.raises(ValueError):
            LinguaLanguageFilter(max_chars=0)
        with pytest.raises(ValueError):
            LinguaLanguageFilter(max_chars=-1)

    def test_minimum_relative_distance_reaches_builder(self) -> None:
        """When > 0, the value is chained onto LanguageDetectorBuilder."""
        captured: Dict[str, Any] = {}

        class _FakeBuilder:
            """Stand-in for lingua's LanguageDetectorBuilder that records calls."""

            def __init__(self) -> None:
                """Mark the builder as having had no distance set yet."""
                captured["distance_set"] = False
                captured["low_accuracy_set"] = False

            def with_preloaded_language_models(self) -> "_FakeBuilder":
                """Return self; preloading is irrelevant to this assertion."""
                return self

            def with_low_accuracy_mode(self) -> "_FakeBuilder":
                """Record that the low-accuracy toggle was flipped."""
                captured["low_accuracy_set"] = True
                return self

            def with_minimum_relative_distance(self, value: float) -> "_FakeBuilder":
                """Capture the value lingua would receive."""
                captured["distance_set"] = True
                captured["value"] = value
                return self

            def build(self) -> object:
                """Return a stub detector; nothing else is exercised here."""
                return object()

        filt = LinguaLanguageFilter(
            targets=["sl"], mode="tag", minimum_relative_distance=0.15
        )

        import lingua

        original_builder_cls = lingua.LanguageDetectorBuilder

        class _BuilderFactory:
            """Bridge the lingua factory call back to `_FakeBuilder`."""

            @staticmethod
            def from_languages(*_languages: object) -> _FakeBuilder:
                """Ignore the candidate set and hand back our recorder."""
                return _FakeBuilder()

        lingua.LanguageDetectorBuilder = _BuilderFactory  # type: ignore[assignment]
        try:
            filt._ensure_detector()
        finally:
            lingua.LanguageDetectorBuilder = original_builder_cls  # type: ignore[assignment]

        assert captured["distance_set"] is True
        assert captured["value"] == pytest.approx(0.15)
        assert captured["low_accuracy_set"] is False

    def test_minimum_relative_distance_zero_skips_builder_call(self) -> None:
        """Default 0.0 must NOT call with_minimum_relative_distance."""
        captured: Dict[str, Any] = {"distance_set": False}

        class _FakeBuilder:
            """Records whether the distance setter was ever called."""

            def with_preloaded_language_models(self) -> "_FakeBuilder":
                """Return self; preloading is irrelevant here."""
                return self

            def with_low_accuracy_mode(self) -> "_FakeBuilder":
                """Should not be called when low_accuracy defaults to False."""
                captured["low_accuracy_set"] = True
                return self

            def with_minimum_relative_distance(self, value: float) -> "_FakeBuilder":
                """Setting the distance at 0.0 would be a regression."""
                captured["distance_set"] = True
                captured["value"] = value
                return self

            def build(self) -> object:
                """Return a stub detector."""
                return object()

        filt = LinguaLanguageFilter(targets=["sl"], mode="tag")  # default 0.0

        import lingua

        original_builder_cls = lingua.LanguageDetectorBuilder

        class _BuilderFactory:
            """Bridge the lingua factory call back to `_FakeBuilder`."""

            @staticmethod
            def from_languages(*_languages: object) -> _FakeBuilder:
                """Ignore the candidate set and hand back our recorder."""
                return _FakeBuilder()

        lingua.LanguageDetectorBuilder = _BuilderFactory  # type: ignore[assignment]
        try:
            filt._ensure_detector()
        finally:
            lingua.LanguageDetectorBuilder = original_builder_cls  # type: ignore[assignment]

        assert captured["distance_set"] is False

    def test_low_accuracy_reaches_builder(self) -> None:
        """When low_accuracy=True, with_low_accuracy_mode() is chained."""
        captured: Dict[str, Any] = {"low_accuracy_set": False}

        class _FakeBuilder:
            """Stand-in builder that records the low-accuracy toggle."""

            def with_preloaded_language_models(self) -> "_FakeBuilder":
                """Return self; preloading is irrelevant here."""
                return self

            def with_low_accuracy_mode(self) -> "_FakeBuilder":
                """Record the toggle and return self for chaining."""
                captured["low_accuracy_set"] = True
                return self

            def with_minimum_relative_distance(self, _value: float) -> "_FakeBuilder":
                """Return self; distance is unused here."""
                return self

            def build(self) -> object:
                """Return a stub detector."""
                return object()

        filt = LinguaLanguageFilter(targets=["sl"], mode="tag", low_accuracy=True)

        import lingua

        original_builder_cls = lingua.LanguageDetectorBuilder

        class _BuilderFactory:
            """Bridge the lingua factory call back to `_FakeBuilder`."""

            @staticmethod
            def from_languages(*_languages: object) -> _FakeBuilder:
                """Ignore the candidate set and hand back our recorder."""
                return _FakeBuilder()

        lingua.LanguageDetectorBuilder = _BuilderFactory  # type: ignore[assignment]
        try:
            filt._ensure_detector()
        finally:
            lingua.LanguageDetectorBuilder = original_builder_cls  # type: ignore[assignment]

        assert captured["low_accuracy_set"] is True

    def test_max_chars_truncates_text_before_detection(self) -> None:
        """When max_chars is set, the detector receives the truncated text."""
        captured: Dict[str, Any] = {}

        class _FakeDetector:
            """Records the text passed to detect_language_of."""

            def detect_language_of(self, text: str) -> None:
                """Capture the text and refuse to commit to a prediction."""
                captured["text"] = text
                return None

        filt = LinguaLanguageFilter(targets=["sl"], mode="tag", max_chars=10)
        filt._detector = _FakeDetector()  # bypass _ensure_detector

        long_text = SLOVENIAN * 5  # well above 10 chars
        _consume(filt, [_doc(long_text, "long")])

        assert captured["text"] == long_text[:10]

    def test_no_truncation_when_max_chars_is_none(self) -> None:
        """When max_chars is None, the full doc text reaches the detector."""
        captured: Dict[str, Any] = {}

        class _FakeDetector:
            """Records the text passed to detect_language_of."""

            def detect_language_of(self, text: str) -> None:
                """Capture the text and refuse to commit to a prediction."""
                captured["text"] = text
                return None

        filt = LinguaLanguageFilter(targets=["sl"], mode="tag")  # max_chars default None
        filt._detector = _FakeDetector()

        _consume(filt, [_doc(SLOVENIAN, "sl")])

        assert captured["text"] == SLOVENIAN

    def test_targets_are_added_to_candidates(self) -> None:
        """A target language not in the candidate set is auto-included."""
        filt = LinguaLanguageFilter(targets=["sl", "fr"], candidates=["sl", "en"])
        assert "fr" in filt.candidates
        assert "sl" in filt.candidates
        assert "en" in filt.candidates
