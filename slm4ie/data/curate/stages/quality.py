"""Stage 3: datatrove's Gopher quality heuristics, configured from `quality:`.

Drops documents whose length, word lengths, symbol/bullet/ellipsis ratios or
stopword count fall outside the configured bounds (defaults from the Gopher
paper).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.filters import GopherQualityFilter
from datatrove.utils.typeshelper import Languages

from slm4ie.data.curate.paths import CuratePaths
from slm4ie.data.curate.stages import StageJob
from slm4ie.data.curate.stages.common import jsonl_reader, jsonl_writer, pipeline_io_counts


@dataclass
class QualityConfig:
    """Knobs for the Gopher quality heuristic filter.

    Mirrors `GopherQualityFilter.__init__` defaults so the CLI can
    override individual values from `curate.yaml` without listing
    every parameter.

    Attributes:
        min_doc_words: Minimum word count; shorter docs are dropped.
        max_doc_words: Maximum word count; longer docs are dropped.
        min_avg_word_length: Minimum average word length in chars.
        max_avg_word_length: Maximum average word length in chars.
        max_symbol_word_ratio: Max fraction of word-like tokens that
            are pure symbols.
        max_bullet_lines_ratio: Max fraction of lines starting with a
            bullet glyph.
        max_ellipsis_lines_ratio: Max fraction of lines ending in an
            ellipsis.
        max_non_alpha_words_ratio: *Minimum* fraction of words that
            must contain at least one alphabetic character (datatrove
            keeps the legacy Gopher name).
        min_stop_words: Minimum number of stopword tokens that must
            appear in the document.
    """

    min_doc_words: int = 50
    max_doc_words: int = 100_000
    min_avg_word_length: int = 3
    max_avg_word_length: int = 10
    max_symbol_word_ratio: float = 0.1
    max_bullet_lines_ratio: float = 0.9
    max_ellipsis_lines_ratio: float = 0.3
    max_non_alpha_words_ratio: float = 0.8
    min_stop_words: int = 2


def _build_quality_config(qcfg: Dict[str, Any]) -> QualityConfig:
    """Resolve a quality-stage config slice into a `QualityConfig`.

    Args:
        qcfg: The effective `quality` config slice for one bucket.

    Returns:
        The resolved `QualityConfig`, with defaults applied.
    """
    return QualityConfig(
        min_doc_words=int(qcfg.get("min_doc_words", 50)),
        max_doc_words=int(qcfg.get("max_doc_words", 100_000)),
        min_avg_word_length=int(qcfg.get("min_avg_word_length", 3)),
        max_avg_word_length=int(qcfg.get("max_avg_word_length", 10)),
        max_symbol_word_ratio=float(qcfg.get("max_symbol_word_ratio", 0.1)),
        max_bullet_lines_ratio=float(qcfg.get("max_bullet_lines_ratio", 0.9)),
        max_ellipsis_lines_ratio=float(qcfg.get("max_ellipsis_lines_ratio", 0.3)),
        max_non_alpha_words_ratio=float(qcfg.get("max_non_alpha_words_ratio", 0.8)),
        min_stop_words=int(qcfg.get("min_stop_words", 2)),
    )


def build_quality_executors(
    paths: CuratePaths,
    *,
    tasks: int = 1,
    quality_config: Optional[QualityConfig] = None,
    language: str = Languages.slovenian,
    stopwords: Optional[Set[str]] = None,
    input_override: Optional[Path] = None,
    output_override: Optional[Path] = None,
) -> List[LocalPipelineExecutor]:
    """Build the quality stage: read 02_spam/ → Gopher quality → write 03_quality/.

    Args:
        paths: Resolved input/output locations.
        tasks: Parallel worker count.
        quality_config: `GopherQualityFilter` knob bundle; defaults to
            Gopher paper values.
        language: ISO-3 code for the word/sentence tokenizer.
        stopwords: Stopword set used by `GopherQualityFilter`.
        input_override: Optional folder to read from instead of the
            language stage's output, used to restrict the stage to a
            symlinked subset of datasets.
        output_override: Optional folder to write to instead of the
            stage's output folder (the driver's staging folder).

    Returns:
        A list with one `LocalPipelineExecutor`.
    """
    cfg = quality_config or QualityConfig()
    in_ = input_override if input_override is not None else paths.stage_dir("language")
    out = output_override if output_override is not None else paths.stage_dir("quality")
    executor = LocalPipelineExecutor(
        pipeline=[
            jsonl_reader(in_),
            GopherQualityFilter(
                language=language,
                stop_words=sorted(stopwords) if stopwords else None,
                min_doc_words=cfg.min_doc_words,
                max_doc_words=cfg.max_doc_words,
                min_avg_word_length=cfg.min_avg_word_length,
                max_avg_word_length=cfg.max_avg_word_length,
                max_symbol_word_ratio=cfg.max_symbol_word_ratio,
                max_bullet_lines_ratio=cfg.max_bullet_lines_ratio,
                max_ellipsis_lines_ratio=cfg.max_ellipsis_lines_ratio,
                max_non_alpha_words_ratio=cfg.max_non_alpha_words_ratio,
                min_stop_words=cfg.min_stop_words,
            ),
            jsonl_writer(out),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("quality")),
        skip_completed=False,
    )
    return [executor]


def run(job: StageJob) -> Tuple[int, int]:
    """Run the quality stage over the job's input view.

    Args:
        job: What to filter, and where to write it; `job.stopwords` feeds
            the stopword floor.

    Returns:
        `(records_in, records_out)` from the run's datatrove stats.
    """
    execs = build_quality_executors(
        job.paths,
        tasks=job.workers,
        quality_config=_build_quality_config(job.config),
        stopwords=job.stopwords,
        input_override=job.input_view,
        output_override=job.output_folder,
    )
    return pipeline_io_counts(execs[-1].run())
