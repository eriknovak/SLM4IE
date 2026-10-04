"""Datatrove plumbing shared by the stage modules: shard reader, writer, counts.

I/O layout — every reader walks `<input_folder>/<dataset>/<part>.jsonl.gz`
recursively, and every writer emits `<output_folder>/<dataset>/<rank>.jsonl.gz`,
matching the convert stage's per-dataset shard layout, so dataset provenance
survives every stage.

Within a stage, the corpus builders (exact dedup, sentence dedup, statistics)
set `skip_completed=True`, so a rerun after a crash skips the tasks datatrove
already marked complete; the run loop clears the stage's staging folder before
a fresh start so stale markers never apply. The scoped builders keep
`skip_completed=False`: their buckets share one logging folder, so a marker
from one bucket would skip another's task. Builders are pure factories — they
do not check, write, or honor sentinels.

A stage's version hashes only its own module, so editing this one reruns no
stage; keep anything that changes what a stage writes in that stage's module.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.utils.stats import PipelineStats

from slm4ie.data.curate.paths import executor_task_stats


def jsonl_writer(stage_folder: Path) -> JsonlWriter:
    """Return a JsonlWriter that emits `<stage_folder>/<dataset>/<rank>.jsonl.gz`.

    Args:
        stage_folder: Folder to write the shards into. datatrove's
            writer creates it on first write, so callers do not need
            to mkdir beforehand.

    Returns:
        A `JsonlWriter` whose output filename template routes each
        dataset's shards into its own subfolder per rank.
    """
    return JsonlWriter(
        output_folder=str(stage_folder),
        output_filename="${dataset}/${rank}.jsonl.gz",
    )


class _TaskZeroProgressReader(JsonlReader):
    """JsonlReader that draws its file progress bar in task 0 only.

    Each task runs in its own worker, so a bar per task would interleave on the
    console; task 0's share of the files stands in for the whole stage.
    """

    def run(self, data: Optional[DocumentsPipeline] = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Read this task's shard, showing the file progress bar when `rank` is 0.

        Args:
            data: Documents from earlier pipeline steps, passed through first.
            rank: Index of the current task.
            world_size: Total number of tasks.

        Returns:
            The documents of this task's shard.
        """
        self.file_progress = rank == 0
        return super().run(data, rank, world_size)


def jsonl_reader(folder: Path) -> JsonlReader:
    """Return a JsonlReader that walks `<folder>/**/*.jsonl.gz` recursively.

    Task 0 draws a file progress bar, so a run shows how far its stage has got.
    """
    return _TaskZeroProgressReader(
        str(folder),
        glob_pattern="**/*.jsonl.gz",
        shuffle_files=False,
        recursive=True,
    )


def pipeline_io_counts(stats: PipelineStats) -> Tuple[int, int]:
    """Extract input/output document counts from a datatrove run's stats.

    `LocalPipelineExecutor.run()` returns a `PipelineStats` whose
    `stats` list holds one stats block per pipeline step, in order. For
    the `[reader, ..., writer]` pipelines built in this module the
    reader records every document it yields under the `documents`
    metric and the writer records every document it persists under
    `total`. This reads those two figures so callers can report stage
    throughput without re-scanning the on-disk shards.

    Args:
        stats: Merged stats returned by a `LocalPipelineExecutor.run()`
            call (the executor also writes them to `stats.json`).

    Returns:
        Tuple `(records_in, records_out)`: documents read by the first
        pipeline step and documents written by the last. Both are `0`
        when *stats* carries no per-step blocks. `records_out` is `0`
        for pipelines whose final step is not a writer (it records no
        `total` metric); such callers must supply their own value.
    """
    if not stats.stats:
        return 0, 0
    records_in = int(stats.stats[0]["documents"].total)
    records_out = int(stats.stats[-1]["total"].total)
    return records_in, records_out


def stage_io_counts(logging_dir: Path) -> Tuple[int, int]:
    """Sum input/output document counts over every task a stage has finished.

    datatrove writes each finished task's stats to
    `<logging_dir>/stats/<rank>.json`. A resumed run returns stats for the
    tasks it ran this time only (or `None` when all were already done), so
    stage totals are read back from these per-task files instead, over the
    ranks the executor declares (`executor_task_stats`).

    Args:
        logging_dir: The `logging_dir` of the stage's last executor.

    Returns:
        Tuple `(records_in, records_out)` as defined by `pipeline_io_counts`,
        over all finished tasks; `(0, 0)` when none has finished.
    """
    merged = PipelineStats()
    for stats_file in executor_task_stats(logging_dir):
        merged = merged + PipelineStats.from_json(json.loads(stats_file.read_text(encoding="utf-8")))
    return pipeline_io_counts(merged)
