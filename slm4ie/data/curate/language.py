"""Lingua-py language detection wrapped as a datatrove pipeline step.

The user explicitly chose `lingua-language-detector` over datatrove's
bundled fastText / GlotLID `LanguageFilter`. This module implements
`LinguaLanguageFilter` so the same `LocalPipelineExecutor` can drive
either detector — only the class name in the pipeline differs.

The detector is built lazily once per worker (`LanguageDetectorBuilder`
is expensive to construct) and the candidate set is restricted to South
Slavic plus common European languages, which sharply improves both
accuracy and throughput on Slovenian text.
"""

# Datatrove probes installed dependencies via importlib.metadata at class
# definition time. Both submodules need to be imported explicitly under
# Python 3.13; otherwise `check_required_dependencies` raises a
# spurious ImportError. Keep these imports above the datatrove imports.
import importlib.metadata  # noqa: F401
import importlib.util  # noqa: F401
from typing import List, Optional

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers.disk_base import DiskWriter

#: Default ISO 639-1 codes for the candidate language set used by the
#: detector. Covers Slovenian and its likely confounders.
DEFAULT_CANDIDATE_LANGUAGES: List[str] = [
    "sl", "hr", "sr", "bs", "mk", "en", "de", "it", "hu",
]


class LinguaLanguageFilter(PipelineStep):
    """Tag (and optionally drop) documents based on lingua-py detection.

    Each document gets `metadata.language` (ISO 639-1 code) and
    `metadata.language_score` (lingua confidence in the *target*
    language, in `[0, 1]`). When `mode="filter"` and the score is
    below `threshold`, the document is routed to
    `exclusion_writer` instead of being yielded downstream.

    Attributes:
        target: ISO 639-1 code for the language we want to keep.
        candidates: ISO 639-1 codes that lingua should consider. A
            small candidate set is much faster and more accurate than
            the full lingua language list.
        threshold: Minimum confidence in `target` to pass the filter
            in `"filter"` mode.
        mode: Either `"tag"` (keep all docs, only annotate) or
            `"filter"` (drop below-threshold docs).
        exclusion_writer: Optional `DiskWriter` that receives docs
            dropped in `"filter"` mode.
    """

    type = "🌍 - LANGUAGE"
    name = "🦜 Lingua-py"
    _requires_dependencies = [("lingua", "lingua-language-detector")]

    def __init__(
        self,
        target: str = "sl",
        candidates: Optional[List[str]] = None,
        threshold: float = 0.5,
        mode: str = "tag",
        exclusion_writer: Optional[DiskWriter] = None,
    ) -> None:
        """Build a filter; the lingua detector itself is lazily constructed.

        Args:
            target: ISO 639-1 code for the language we want to keep.
            candidates: ISO 639-1 codes that lingua should consider.
                Defaults to `DEFAULT_CANDIDATE_LANGUAGES` (Slovenian
                + likely confounders).
            threshold: Minimum confidence in `target` to pass the
                filter in `"filter"` mode. Ignored in `"tag"` mode.
            mode: Either `"tag"` or `"filter"`.
            exclusion_writer: Optional sink for dropped documents.

        Raises:
            ValueError: If `mode` is not `"tag"` or `"filter"`.
        """
        super().__init__()
        if mode not in {"tag", "filter"}:
            raise ValueError(f"mode must be 'tag' or 'filter', got {mode!r}")
        self.target = target
        self.candidates = list(candidates) if candidates is not None else list(DEFAULT_CANDIDATE_LANGUAGES)
        if self.target not in self.candidates:
            self.candidates.append(self.target)
        self.threshold = threshold
        self.mode = mode
        self.exclusion_writer = exclusion_writer
        self._detector = None
        self._target_lang = None

    def _ensure_detector(self) -> None:
        """Build the lingua detector on first use; cached on the instance."""
        if self._detector is not None:
            return

        from lingua import Language, LanguageDetectorBuilder

        code_to_language = {lang.iso_code_639_1.name.lower(): lang for lang in Language.all()}
        try:
            languages = [code_to_language[code] for code in self.candidates]
        except KeyError as exc:
            raise ValueError(f"Unknown lingua language code: {exc.args[0]!r}") from exc

        self._target_lang = code_to_language[self.target]
        self._detector = (
            LanguageDetectorBuilder.from_languages(*languages).with_preloaded_language_models().build()
        )

    def _classify(self, doc: Document) -> tuple[Optional[str], float]:
        """Run lingua on `doc.text` and return (predicted_code, target_score).

        Args:
            doc: The document to classify.

        Returns:
            A tuple `(predicted_iso_639_1_or_None, target_confidence)`.
            `predicted` is `None` when lingua refuses to commit
            (e.g. extremely short input).
        """
        self._ensure_detector()
        text = doc.text
        predicted = self._detector.detect_language_of(text)
        score = self._detector.compute_language_confidence(text, self._target_lang)
        code = predicted.iso_code_639_1.name.lower() if predicted is not None else None
        return code, float(score)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Tag each document with detected language metadata.

        Args:
            data: Input document stream.
            rank: Worker rank. Unused but required by the
                `PipelineStep` contract.
            world_size: Total number of workers. Unused.

        Yields:
            Documents with `metadata.language` and
            `metadata.language_score` set. In `"filter"` mode,
            below-threshold documents are routed to
            `exclusion_writer` instead of being yielded.
        """
        with self.exclusion_writer if self.exclusion_writer is not None else _NullContext():
            for doc in data:
                with self.track_time():
                    code, score = self._classify(doc)
                    doc.metadata["language"] = code
                    doc.metadata["language_score"] = score

                    self.stat_update("docs")
                    if code == self.target:
                        self.stat_update(f"language_{self.target}")
                    elif code is not None:
                        self.stat_update(f"language_{code}")
                    else:
                        self.stat_update("language_unknown")

                    keep = self.mode == "tag" or score >= self.threshold

                if keep:
                    yield doc
                else:
                    self.stat_update("dropped")
                    if self.exclusion_writer is not None:
                        self.exclusion_writer.write(doc, rank=rank)


class _NullContext:
    """Trivial context manager used when no exclusion writer is configured."""

    def __enter__(self) -> "_NullContext":
        """Return self; the value is unused."""
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Swallow nothing; let exceptions propagate normally."""
        return False
