"""Stage 4: datatrove's Gopher repetition heuristics.

Drops documents dominated by duplicate paragraphs or lines, or saturated with
repeated n-grams. The stage exposes no config knobs today.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.filters import GopherRepetitionFilter
from datatrove.utils.typeshelper import Languages

from slm4ie.data.curate.paths import CuratePaths
from slm4ie.data.curate.stages import StageJob
from slm4ie.data.curate.stages.common import jsonl_reader, jsonl_writer, pipeline_io_counts


def build_repetition_executors(
    paths: CuratePaths,
    *,
    tasks: int = 1,
    language: str = Languages.slovenian,
    input_override: Optional[Path] = None,
    output_override: Optional[Path] = None,
) -> List[LocalPipelineExecutor]:
    """Build the repetition stage: read 03_quality/ → Gopher repetition → write 04_repetition/.

    Args:
        paths: Resolved input/output locations.
        tasks: Parallel worker count.
        language: ISO-3 code for the word/sentence tokenizer the
            repetition filter uses.
        input_override: Optional folder to read from instead of the
            quality stage's output, used to restrict the stage to a
            symlinked subset of datasets.
        output_override: Optional folder to write to instead of the
            stage's output folder (the driver's staging folder).

    Returns:
        A list with one `LocalPipelineExecutor`.
    """
    in_ = input_override if input_override is not None else paths.stage_dir("quality")
    out = output_override if output_override is not None else paths.stage_dir("repetition")
    executor = LocalPipelineExecutor(
        pipeline=[
            jsonl_reader(in_),
            GopherRepetitionFilter(language=language),
            jsonl_writer(out),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("repetition")),
        skip_completed=False,
    )
    return [executor]


def run(job: StageJob) -> Tuple[int, int]:
    """Run the repetition stage over the job's input view.

    Args:
        job: What to filter, and where to write it.

    Returns:
        `(records_in, records_out)` from the run's datatrove stats.
    """
    execs = build_repetition_executors(
        job.paths, tasks=job.workers, input_override=job.input_view, output_override=job.output_folder
    )
    return pipeline_io_counts(execs[-1].run())
