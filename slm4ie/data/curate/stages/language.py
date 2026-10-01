"""Lingua-py language detection wrapped as a datatrove pipeline step.

The user explicitly chose `lingua-language-detector` over datatrove's
bundled fastText / GlotLID `LanguageFilter`. This module implements
`LinguaLanguageFilter` so the same `LocalPipelineExecutor` can drive
either detector — only the class name in the pipeline differs.

The detector is built lazily once per worker (`LanguageDetectorBuilder`
is expensive to construct) and the candidate set is restricted to South
Slavic plus common European languages, which sharply improves both
accuracy and throughput on Slovenian text. Long inputs can be truncated
via `max_chars` and the trigram-only `low_accuracy` mode is exposed as
a knob — both tradeoffs trade a sliver of accuracy for an order-of-
magnitude throughput win on web-scale corpora.
"""

# Datatrove probes installed dependencies via importlib.metadata at class
# definition time. Both submodules need to be imported explicitly under
# Python 3.13; otherwise `check_required_dependencies` raises a
# spurious ImportError. Keep these imports above the datatrove imports.
import importlib.metadata  # noqa: F401
import importlib.util  # noqa: F401
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from datatrove.data import Document, DocumentsPipeline
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers.disk_base import DiskWriter

from slm4ie.data.curate.paths import CuratePaths
from slm4ie.data.curate.stages import StageRun
from slm4ie.data.curate.stages.common import jsonl_reader, jsonl_writer, pipeline_io_counts

#: Default ISO 639-1 codes for the candidate language set used by the
#: detector. Covers Slovenian and its likely confounders.
DEFAULT_CANDIDATE_LANGUAGES: List[str] = [
    "sl",
    "hr",
    "sr",
    "bs",
    "mk",
    "en",
    "de",
    "it",
    "hu",
]

#: Default target language set used when no `targets` argument is given.
DEFAULT_TARGET_LANGUAGES: List[str] = ["sl"]


class LinguaLanguageFilter(PipelineStep):
    """Tag (and optionally drop) documents based on lingua-py detection.

    Each document gets `metadata.language` set to lingua's predicted
    ISO 639-1 code (or `None` when lingua refuses to commit). When
    `mode="filter"`, documents whose predicted language is not in the
    `targets` set are routed to `exclusion_writer` instead of being
    yielded downstream.

    Attributes:
        targets: ISO 639-1 codes considered "in-language". Documents
            whose predicted language is in this set pass the filter;
            others are dropped (in `"filter"` mode).
        candidates: ISO 639-1 codes that lingua should consider. A
            small candidate set is much faster and more accurate than
            the full lingua language list. All target codes are
            automatically added to the candidate set.
        minimum_relative_distance: Required gap between the top
            language's confidence and the runner-up's. `0.0` disables
            the check (lingua's default). Useful for high-precision
            South Slavic disambiguation, where a raw threshold alone
            tolerates substantial cross-language mass because the
            confidence scores are softmax-normalized over the
            candidate set.
        mode: Either `"tag"` (keep all docs, only annotate) or
            `"filter"` (drop out-of-target docs).
        exclusion_writer: Optional `DiskWriter` that receives docs
            dropped in `"filter"` mode.
        low_accuracy: Use lingua's trigram-only model instead of the
            default 1-5-gram model. ~5-10x faster with negligible
            accuracy loss on long European-language web text.
        max_chars: Truncate `doc.text` to this many characters before
            running lingua. The first ~2 KB is plenty of signal for a
            small candidate set; saves work proportional to doc length
            on giant web docs. `None` disables truncation.
    """

    type = "🌍 - LANGUAGE"
    name = "🦜 Lingua-py"
    _requires_dependencies = [("lingua", "lingua-language-detector")]

    def __init__(
        self,
        targets: Optional[Sequence[str]] = None,
        candidates: Optional[List[str]] = None,
        mode: str = "tag",
        exclusion_writer: Optional[DiskWriter] = None,
        minimum_relative_distance: float = 0.0,
        low_accuracy: bool = False,
        max_chars: Optional[int] = None,
    ) -> None:
        """Build a filter; the lingua detector itself is lazily constructed.

        Args:
            targets: ISO 639-1 codes considered "in-language". Defaults
                to `DEFAULT_TARGET_LANGUAGES` (`["sl"]`). Must be
                non-empty.
            candidates: ISO 639-1 codes that lingua should consider.
                Defaults to `DEFAULT_CANDIDATE_LANGUAGES` (Slovenian
                + likely confounders). Target codes are added to this
                set automatically.
            mode: Either `"tag"` or `"filter"`.
            exclusion_writer: Optional sink for dropped documents.
            minimum_relative_distance: Required confidence gap between
                the top language and the runner-up before lingua will
                commit to a prediction. `0.0` disables the check
                (lingua's default). Recommend `~0.1` for Slovenian
                vs South Slavic neighbours; higher values trade recall
                for precision.
            low_accuracy: Use lingua's trigram-only model. Much faster
                and accurate enough for long web-text inputs.
            max_chars: Truncate `doc.text` to this many chars before
                detection. Pass `None` to disable.

        Raises:
            ValueError: If `mode` is not `"tag"` or `"filter"`, if
                `targets` is empty, if `minimum_relative_distance` is
                outside `[0, 1)`, or if `max_chars` is not `None` and
                not a positive int.
        """
        super().__init__()
        if mode not in {"tag", "filter"}:
            raise ValueError(f"mode must be 'tag' or 'filter', got {mode!r}")
        if not 0.0 <= minimum_relative_distance < 1.0:
            raise ValueError(f"minimum_relative_distance must be in [0, 1), got {minimum_relative_distance!r}")
        if max_chars is not None and (not isinstance(max_chars, int) or max_chars <= 0):
            raise ValueError(f"max_chars must be a positive int or None, got {max_chars!r}")

        target_list = list(targets) if targets is not None else list(DEFAULT_TARGET_LANGUAGES)
        if not target_list:
            raise ValueError("targets must be a non-empty sequence of ISO 639-1 codes")

        self.targets: List[str] = [code.lower() for code in target_list]
        self._target_codes: Set[str] = set(self.targets)

        self.candidates = list(candidates) if candidates is not None else list(DEFAULT_CANDIDATE_LANGUAGES)
        for code in self.targets:
            if code not in self.candidates:
                self.candidates.append(code)

        self.minimum_relative_distance = minimum_relative_distance
        self.mode = mode
        self.exclusion_writer = exclusion_writer
        self.low_accuracy = low_accuracy
        self.max_chars = max_chars
        self._detector = None

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

        builder = LanguageDetectorBuilder.from_languages(*languages).with_preloaded_language_models()
        if self.low_accuracy:
            builder = builder.with_low_accuracy_mode()
        if self.minimum_relative_distance > 0.0:
            builder = builder.with_minimum_relative_distance(self.minimum_relative_distance)
        self._detector = builder.build()

    def _classify(self, doc: Document) -> Optional[str]:
        """Return lingua's predicted ISO 639-1 code for `doc.text`, or None.

        The doc is truncated to `self.max_chars` (when set) before
        detection; the first ~2 KB is sufficient signal for a small
        candidate set and avoids paying lingua's per-char cost on very
        long web docs. `None` means lingua refused to commit (extremely
        short input or below `minimum_relative_distance`).

        Args:
            doc: The document to classify.

        Returns:
            ISO 639-1 code (lowercased) for the predicted language, or
            `None` if lingua did not commit to a prediction.
        """
        self._ensure_detector()
        text = doc.text if self.max_chars is None else doc.text[: self.max_chars]
        predicted = self._detector.detect_language_of(text)
        return predicted.iso_code_639_1.name.lower() if predicted is not None else None

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Tag each document with its detected language.

        Args:
            data: Input document stream.
            rank: Worker rank. Unused but required by the
                `PipelineStep` contract.
            world_size: Total number of workers. Unused.

        Yields:
            Documents with `metadata.language` set to the predicted
            ISO 639-1 code (or `None`). In `"filter"` mode, documents
            whose predicted language is not in `targets` are routed to
            `exclusion_writer` instead of being yielded.
        """
        with self.exclusion_writer if self.exclusion_writer is not None else _NullContext():
            for doc in data:
                with self.track_time():
                    code = self._classify(doc)
                    doc.metadata["language"] = code

                    self.stat_update("docs")
                    self.stat_update(f"language_{code}" if code else "language_unknown")

                    keep = self.mode == "tag" or (code is not None and code in self._target_codes)

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


@dataclass(frozen=True)
class LanguageParams:
    """Resolved language-stage parameters for one config bucket.

    Attributes:
        target_languages: ISO 639-1 codes treated as in-language.
        candidate_languages: Candidate set lingua chooses among, or None.
        mode: `filter` drops out-of-target docs; `tag` only annotates.
        minimum_relative_distance: Confidence gap lingua needs to commit.
        low_accuracy: Use lingua's lighter trigram-only model.
        max_chars: Truncate doc text to this many chars, or None.
    """

    target_languages: List[str]
    candidate_languages: Optional[List[str]]
    mode: str
    minimum_relative_distance: float
    low_accuracy: bool
    max_chars: Optional[int]


def _build_language_params(lang_cfg: Dict[str, Any]) -> LanguageParams:
    """Resolve a language-stage config slice into typed parameters.

    Args:
        lang_cfg: The effective `language` config slice for one bucket.

    Returns:
        The resolved `LanguageParams`, with defaults applied.
    """
    return LanguageParams(
        target_languages=lang_cfg.get("targets") or ["sl"],
        candidate_languages=lang_cfg.get("candidates"),
        mode=str(lang_cfg.get("mode", "filter")),
        minimum_relative_distance=float(lang_cfg.get("minimum_relative_distance", 0.0)),
        low_accuracy=bool(lang_cfg.get("low_accuracy", False)),
        max_chars=lang_cfg.get("max_chars"),
    )


def build_language_executors(
    paths: CuratePaths,
    *,
    tasks: int = 1,
    target_languages: Sequence[str] = ("sl",),
    candidate_languages: Optional[List[str]] = None,
    lang_mode: str = "filter",
    lang_minimum_relative_distance: float = 0.0,
    lang_low_accuracy: bool = False,
    lang_max_chars: Optional[int] = None,
    input_override: Optional[Path] = None,
    output_override: Optional[Path] = None,
) -> List[LocalPipelineExecutor]:
    """Build the language stage: read 00_convert/ → lingua filter → write 01_language/.

    Args:
        paths: Resolved input/output locations.
        tasks: Parallel worker count for this stage.
        target_languages: ISO 639-1 codes considered "in-language".
        candidate_languages: ISO 639-1 candidate set for lingua.
        lang_mode: `"tag"` keeps every doc; `"filter"` drops
            out-of-target docs.
        lang_minimum_relative_distance: Required confidence gap before
            lingua commits. `0.0` disables.
        lang_low_accuracy: Use lingua's trigram-only model.
        lang_max_chars: Truncate doc text to this many chars before
            detection. `None` disables truncation.
        input_override: Optional folder to read from instead of the
            convert stage's output folder. Used by the run loop to
            restrict the language stage to a symlinked subset of
            `<output_dir>/00_convert/` when the user requests a subset
            of dataset keys.
        output_override: Optional folder to write to instead of the
            stage's output folder (the run loop's staging folder).

    Returns:
        A list with one `LocalPipelineExecutor`.
    """
    out = output_override if output_override is not None else paths.stage_dir("language")
    in_ = input_override if input_override is not None else paths.stage_dir("convert")
    executor = LocalPipelineExecutor(
        pipeline=[
            jsonl_reader(in_),
            LinguaLanguageFilter(
                targets=list(target_languages),
                candidates=candidate_languages,
                mode=lang_mode,
                minimum_relative_distance=lang_minimum_relative_distance,
                low_accuracy=lang_low_accuracy,
                max_chars=lang_max_chars,
            ),
            jsonl_writer(out),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("language")),
        skip_completed=False,
    )
    return [executor]


def run(job: StageRun) -> Tuple[int, int]:
    """Run the language stage over the job's input view.

    Args:
        job: What to filter, and where to write it.

    Returns:
        `(records_in, records_out)` from the run's datatrove stats.
    """
    params = _build_language_params(job.config)
    execs = build_language_executors(
        job.paths,
        tasks=job.workers,
        target_languages=params.target_languages,
        candidate_languages=params.candidate_languages,
        lang_mode=params.mode,
        lang_minimum_relative_distance=params.minimum_relative_distance,
        lang_low_accuracy=params.low_accuracy,
        lang_max_chars=params.max_chars,
        input_override=job.input_view,
        output_override=job.output_folder,
    )
    return pipeline_io_counts(execs[-1].run())
