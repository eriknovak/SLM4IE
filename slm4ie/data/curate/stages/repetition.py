"""Stage 4: datatrove's Gopher repetition heuristics.

Drops documents dominated by duplicate paragraphs or lines, or saturated with
repeated n-grams, with the thresholds configured from `repetition:`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.filters import GopherRepetitionFilter
from datatrove.utils.typeshelper import Languages

from slm4ie.data.curate.paths import CuratePaths
from slm4ie.data.curate.stages import StageRun
from slm4ie.data.curate.stages.common import jsonl_reader, jsonl_writer, pipeline_io_counts

#: An n-gram threshold: n-gram size and the fraction of characters it may cover.
NGramThreshold = Tuple[int, float]


@dataclass
class RepetitionConfig:
    """Knobs for the Gopher repetition filter.

    Mirrors `GopherRepetitionFilter.__init__` defaults, so a config that
    sets none of them filters exactly as datatrove does out of the box.

    Attributes:
        dup_line_frac: Max fraction of lines that duplicate another line.
        dup_para_frac: Max fraction of paragraphs that duplicate another.
        dup_line_char_frac: Max fraction of characters in duplicate lines.
        dup_para_char_frac: Max fraction of characters in duplicate
            paragraphs.
        top_n_grams: Per n-gram size, the max fraction of characters the
            most frequent n-gram may cover.
        dup_n_grams: Per n-gram size, the max fraction of characters
            covered by duplicated n-grams.
    """

    dup_line_frac: float = 0.3
    dup_para_frac: float = 0.3
    dup_line_char_frac: float = 0.2
    dup_para_char_frac: float = 0.2
    top_n_grams: Tuple[NGramThreshold, ...] = ((2, 0.2), (3, 0.18), (4, 0.16))
    dup_n_grams: Tuple[NGramThreshold, ...] = ((5, 0.15), (6, 0.14), (7, 0.13), (8, 0.12), (9, 0.11), (10, 0.1))


def _ngram_thresholds(pairs: Sequence[Sequence[Any]]) -> Tuple[NGramThreshold, ...]:
    """Convert YAML `[n, fraction]` pairs into the tuples datatrove expects.

    Args:
        pairs: Sequence of `[n, fraction]` pairs.

    Returns:
        The pairs as a tuple of `(int, float)` tuples.
    """
    return tuple((int(n), float(frac)) for n, frac in pairs)


def _build_repetition_config(rcfg: Dict[str, Any]) -> RepetitionConfig:
    """Resolve a repetition-stage config slice into a `RepetitionConfig`.

    Args:
        rcfg: The effective `repetition` config slice for one bucket.

    Returns:
        The resolved `RepetitionConfig`, with defaults applied.
    """
    defaults = RepetitionConfig()
    return RepetitionConfig(
        dup_line_frac=float(rcfg.get("dup_line_frac", defaults.dup_line_frac)),
        dup_para_frac=float(rcfg.get("dup_para_frac", defaults.dup_para_frac)),
        dup_line_char_frac=float(rcfg.get("dup_line_char_frac", defaults.dup_line_char_frac)),
        dup_para_char_frac=float(rcfg.get("dup_para_char_frac", defaults.dup_para_char_frac)),
        top_n_grams=_ngram_thresholds(rcfg.get("top_n_grams", defaults.top_n_grams)),
        dup_n_grams=_ngram_thresholds(rcfg.get("dup_n_grams", defaults.dup_n_grams)),
    )


def build_repetition_executors(
    paths: CuratePaths,
    *,
    tasks: int = 1,
    repetition_config: Optional[RepetitionConfig] = None,
    language: str = Languages.slovenian,
    input_override: Optional[Path] = None,
    output_override: Optional[Path] = None,
) -> List[LocalPipelineExecutor]:
    """Build the repetition stage: read 03_quality/ → Gopher repetition → write 04_repetition/.

    Args:
        paths: Resolved input/output locations.
        tasks: Parallel worker count.
        repetition_config: `GopherRepetitionFilter` knob bundle; defaults
            to datatrove's values.
        language: ISO-3 code for the word/sentence tokenizer the
            repetition filter uses.
        input_override: Optional folder to read from instead of the
            quality stage's output, used to restrict the stage to a
            symlinked subset of datasets.
        output_override: Optional folder to write to instead of the
            stage's output folder (the run loop's staging folder).

    Returns:
        A list with one `LocalPipelineExecutor`.
    """
    cfg = repetition_config or RepetitionConfig()
    in_ = input_override if input_override is not None else paths.stage_dir("quality")
    out = output_override if output_override is not None else paths.stage_dir("repetition")
    executor = LocalPipelineExecutor(
        pipeline=[
            jsonl_reader(in_),
            GopherRepetitionFilter(
                dup_line_frac=cfg.dup_line_frac,
                dup_para_frac=cfg.dup_para_frac,
                dup_line_char_frac=cfg.dup_line_char_frac,
                dup_para_char_frac=cfg.dup_para_char_frac,
                top_n_grams=cfg.top_n_grams,
                dup_n_grams=cfg.dup_n_grams,
                language=language,
            ),
            jsonl_writer(out),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("repetition")),
        skip_completed=False,
    )
    return [executor]


def run(job: StageRun) -> Tuple[int, int]:
    """Run the repetition stage over the job's input view.

    Args:
        job: What to filter, and where to write it.

    Returns:
        `(records_in, records_out)` from the run's datatrove stats.
    """
    execs = build_repetition_executors(
        job.paths,
        tasks=job.workers,
        repetition_config=_build_repetition_config(job.config),
        input_override=job.input_view,
        output_override=job.output_folder,
    )
    return pipeline_io_counts(execs[-1].run())
