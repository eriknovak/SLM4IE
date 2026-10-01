"""Stages 5 and 6: corpus-wide exact and sentence deduplication.

Both stages chain three datatrove executors: signature → find duplicates →
filter and write, passing signatures and duplicate lists through the stage's
scratch folder. Besides the two builders and their runners, this module holds
`doc_text` (the content getter for whole-document exact dedup), two
`ExactDedupConfig` factories (`make_exact_config` for the form driven by
`curate.yaml::exact_dedup`, `default_exact_config` as a zero-arg alias), and
`CompactSentenceDedupSignature`, a memory-lean drop-in for datatrove's
sentence-dedup signature step. Both stages share this module, so editing it
reruns both.
"""

from pathlib import Path
from typing import List, Literal, Optional, Tuple, cast

import numpy as np
from datatrove.data import Document, DocumentsPipeline
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.dedup import (
    ExactDedupConfig,
    ExactDedupFilter,
    ExactDedupSignature,
    ExactFindDedups,
    SentDedupConfig,
    SentenceDedupFilter,
    SentenceDedupSignature,
    SentenceFindDedups,
)
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages, StatHints

from slm4ie.data.curate.paths import CuratePaths
from slm4ie.data.curate.stages import StageJob
from slm4ie.data.curate.stages.common import jsonl_reader, jsonl_writer, stage_io_counts

#: Signatures buffered as Python tuples before packing into a numpy chunk.
SIGNATURE_FLUSH_EVERY: int = 1_000_000


def doc_text(doc: Document) -> str:
    """Return the text payload to hash for whole-document exact dedup.

    Args:
        doc: A datatrove Document.

    Returns:
        The document's text. `ExactDedupConfig` requires the getter
        to return `bytes` or `str`; we hash text as-is so two docs
        with byte-identical bodies are treated as duplicates regardless
        of their metadata.
    """
    return doc.text


def make_exact_config(
    *,
    precision: Literal[32, 64] = 64,
    hash_fc: Literal["sha1", "xxhash"] = "xxhash",
    only_dedup_in_index: bool = True,
) -> ExactDedupConfig:
    """Build an `ExactDedupConfig` parameterized for the curate pipeline.

    Wraps datatrove's `ExactDedupConfig` so the CLI can pass through the
    output-affecting knobs declared under `exact_dedup:` in `curate.yaml`.
    The content getter is always `doc_text` — exact dedup operates on
    document text, never on metadata.

    Args:
        precision: Hash width in bits. Choose `32` only for very small
            corpora (collision risk grows past ~10M docs); `64` is the
            collision-safe default up to ~10B docs.
        hash_fc: Hash function. `"xxhash"` is faster; `"sha1"` is
            cryptographically strong but unnecessary for dedup.
        only_dedup_in_index: When True, only deduplicate within the
            current run's index (datatrove default). Set to False when
            extending an existing dedup index across runs.

    Returns:
        A configured `ExactDedupConfig` with `doc_text` as the content
        getter.
    """
    return ExactDedupConfig(
        content_getter=doc_text,
        hash_config=HashConfig(precision=precision, hash_fc=hash_fc),
        only_dedup_in_index=only_dedup_in_index,
    )


def default_exact_config() -> ExactDedupConfig:
    """Build the default `ExactDedupConfig` used by the curation pipeline.

    Thin alias for `make_exact_config()` — kept for back-compat with
    test fixtures and the existing pipeline call sites. New code should
    call `make_exact_config(...)` directly so any non-default knob is
    explicit at the call site.

    Returns:
        An `ExactDedupConfig` whose `content_getter` is `doc_text` and
        whose hashing knobs match `make_exact_config`'s defaults (64-bit
        xxhash, `only_dedup_in_index=True`).
    """
    return make_exact_config()


class CompactSentenceDedupSignature(SentenceDedupSignature):
    """Sentence-dedup signature step that buffers hashes as packed numpy chunks.

    datatrove's `SentenceDedupSignature.run` keeps every `(hash, doc, sent)`
    triple of a task in one Python list until the task ends, at roughly 150
    bytes per sentence window; a multi-gigabyte shard then needs several
    gigabytes of RAM per worker. This subclass packs the triples into the
    same 14-byte structured dtype every `SIGNATURE_FLUSH_EVERY` entries, so a
    task holds about a tenth of the memory. Hashing, sorting and the files
    written are unchanged, so the find and filter steps read them as usual.
    """

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> None:
        """Hash every sentence window of the task's documents and save the signatures.

        Args:
            data: Documents of this task.
            rank: Task rank; names the signature files.
            world_size: Total number of tasks (unused, part of the step API).
        """
        dtype = np.dtype([("hash", self.config.hash_config.np_descr), ("doc", "<u4"), ("sent", "<u2")])
        chunks = []
        buffer = []
        for doc_idx, doc in enumerate(data):
            with self.stats.time_stats:
                self.stat_update(StatHints.total)
                buffer.extend(self.get_hashes(doc, doc_idx))
                if len(buffer) >= SIGNATURE_FLUSH_EVERY:
                    chunks.append(np.array(buffer, dtype=dtype))
                    buffer = []
        chunks.append(np.array(buffer, dtype=dtype))
        signatures = np.concatenate(chunks)
        del chunks, buffer
        self.save_hashes(rank, signatures)


def build_exact_dedup_executors(
    paths: CuratePaths,
    *,
    tasks: int = 1,
    workers: Optional[int] = None,
    finder_workers: int = 1,
    exact_config: Optional[ExactDedupConfig] = None,
    input_override: Optional[Path] = None,
    output_override: Optional[Path] = None,
) -> List[LocalPipelineExecutor]:
    """Build the exact-dedup stage: sig → find → filter+write 05_exact_dedup/.

    Three executors chained via `depends`:
        1. (parallel) read 04_repetition/ → ExactDedupSignature → <scratch>/sigs/
        2. (single)   ExactFindDedups(<scratch>/sigs/) → <scratch>/dups/
        3. (parallel) read 04_repetition/ → ExactDedupFilter → write 05_exact_dedup/

    Args:
        paths: Resolved input/output locations.
        tasks: Task count for executors 1 and 3; each task reads its own
            slice of the input shards, so more tasks means less memory per task.
        workers: Tasks run at once; defaults to `tasks`.
        finder_workers: Worker count for the single-worker find
            executor 2 (and the `finder_workers` argument of the sig
            executor 1).
        exact_config: Optional `ExactDedupConfig`; defaults to one whose
            `content_getter` hashes `doc.text`.
        input_override: Optional folder to read from instead of the
            repetition stage's output, used to restrict the stage to the
            roster's datasets through a symlinked view.
        output_override: Optional folder to write to instead of the
            stage's output folder (the driver's staging folder).

    Returns:
        Three chained `LocalPipelineExecutor`s.
    """
    cfg = exact_config or default_exact_config()
    workers = workers or tasks
    in_ = input_override if input_override is not None else paths.stage_dir("repetition")
    out = output_override if output_override is not None else paths.stage_dir("exact_dedup")
    sigs = paths.scratch_dir("exact_dedup") / "sigs"
    dups = paths.scratch_dir("exact_dedup") / "dups"

    sig = LocalPipelineExecutor(
        pipeline=[
            jsonl_reader(in_),
            ExactDedupSignature(output_folder=str(sigs), config=cfg, finder_workers=finder_workers),
        ],
        tasks=tasks,
        workers=workers,
        logging_dir=str(paths.logs_dir("exact_dedup") / "1_sig"),
        skip_completed=True,
    )
    find = LocalPipelineExecutor(
        pipeline=[ExactFindDedups(data_folder=str(sigs), output_folder=str(dups), config=cfg)],
        tasks=finder_workers,
        workers=finder_workers,
        logging_dir=str(paths.logs_dir("exact_dedup") / "2_find"),
        depends=sig,
        skip_completed=True,
    )
    filt = LocalPipelineExecutor(
        pipeline=[
            jsonl_reader(in_),
            ExactDedupFilter(data_folder=str(dups), config=cfg),
            jsonl_writer(out),
        ],
        tasks=tasks,
        workers=workers,
        logging_dir=str(paths.logs_dir("exact_dedup") / "3_filter"),
        depends=find,
        skip_completed=True,
    )
    return [sig, find, filt]


def build_sentence_dedup_executors(
    paths: CuratePaths,
    *,
    tasks: int = 1,
    workers: Optional[int] = None,
    finder_workers: int = 1,
    sentence_config: Optional[SentDedupConfig] = None,
    language: str = Languages.slovenian,
    input_override: Optional[Path] = None,
    output_override: Optional[Path] = None,
) -> List[LocalPipelineExecutor]:
    """Build the sentence-dedup stage: sig → find → filter+write 06_sentence_dedup/.

    Three executors chained via `depends`, mirroring the exact stage:
        1. (parallel) read 05_exact_dedup/ → CompactSentenceDedupSignature → <scratch>/sigs/
        2. (single)   SentenceFindDedups(<scratch>/sigs/) → <scratch>/dups/
        3. (parallel) read 05_exact_dedup/ → SentenceDedupFilter → write 06_sentence_dedup/

    Args:
        paths: Resolved input/output locations.
        tasks: Task count for executors 1 and 3; each task reads its own
            slice of the input shards, so more tasks means less memory per task.
        workers: Tasks run at once; defaults to `tasks`.
        finder_workers: Worker count for the find executor.
        sentence_config: Optional `SentDedupConfig`.
        language: ISO-3 code for the sentence tokenizer.
        input_override: Optional folder to read from instead of the
            exact-dedup stage's output, used to restrict the stage to the
            roster's datasets through a symlinked view.
        output_override: Optional folder to write to instead of the
            stage's output folder (the driver's staging folder).

    Returns:
        Three chained `LocalPipelineExecutor`s.
    """
    cfg = sentence_config or SentDedupConfig()
    workers = workers or tasks
    in_ = input_override if input_override is not None else paths.stage_dir("exact_dedup")
    out = output_override if output_override is not None else paths.stage_dir("sentence_dedup")
    sigs = paths.scratch_dir("sentence_dedup") / "sigs"
    dups = paths.scratch_dir("sentence_dedup") / "dups"

    sig = LocalPipelineExecutor(
        pipeline=[
            jsonl_reader(in_),
            CompactSentenceDedupSignature(
                output_folder=str(sigs),
                config=cfg,
                finder_workers=finder_workers,
                language=language,
            ),
        ],
        tasks=tasks,
        workers=workers,
        logging_dir=str(paths.logs_dir("sentence_dedup") / "1_sig"),
        skip_completed=True,
    )
    find = LocalPipelineExecutor(
        pipeline=[SentenceFindDedups(data_folder=str(sigs), output_folder=str(dups), config=cfg)],
        tasks=finder_workers,
        workers=finder_workers,
        logging_dir=str(paths.logs_dir("sentence_dedup") / "2_find"),
        depends=sig,
        skip_completed=True,
    )
    filt = LocalPipelineExecutor(
        pipeline=[
            jsonl_reader(in_),
            SentenceDedupFilter(data_folder=str(dups), config=cfg, language=language),
            jsonl_writer(out),
        ],
        tasks=tasks,
        workers=workers,
        logging_dir=str(paths.logs_dir("sentence_dedup") / "3_filter"),
        depends=find,
        skip_completed=True,
    )
    return [sig, find, filt]


def run_exact_dedup(job: StageJob) -> Tuple[int, int]:
    """Run whole-document exact dedup over the job's input view.

    Args:
        job: What to deduplicate, and where to write it.

    Returns:
        `(records_in, records_out)` summed over every finished filter task.

    Raises:
        ValueError: If `precision` or `hash_fc` is not a supported value.
    """
    raw_precision = int(job.config.get("precision", 64))
    if raw_precision not in (32, 64):
        raise ValueError(f"the pretrain config's exact_dedup.precision must be 32 or 64, got {raw_precision}")
    raw_hash_fc = str(job.config.get("hash_fc", "xxhash"))
    if raw_hash_fc not in ("sha1", "xxhash"):
        raise ValueError(f"the pretrain config's exact_dedup.hash_fc must be 'sha1' or 'xxhash', got {raw_hash_fc!r}")
    exact_config = make_exact_config(
        precision=cast(Literal[32, 64], raw_precision),
        hash_fc=cast(Literal["sha1", "xxhash"], raw_hash_fc),
        only_dedup_in_index=bool(job.config.get("only_dedup_in_index", True)),
    )
    execs = build_exact_dedup_executors(
        job.paths,
        tasks=job.tasks or job.workers,
        workers=job.workers,
        exact_config=exact_config,
        input_override=job.input_view,
        output_override=job.output_folder,
    )
    execs[-1].run()
    return stage_io_counts(job.paths.logs_dir("exact_dedup") / "3_filter")


def run_sentence_dedup(job: StageJob) -> Tuple[int, int]:
    """Run N-sentence window dedup over the job's input view.

    Args:
        job: What to deduplicate, and where to write it.

    Returns:
        `(records_in, records_out)` summed over every finished filter task.
    """
    sentence_config = SentDedupConfig(
        n_sentences=int(job.config.get("n_sentences", 3)),
        min_doc_words=int(job.config.get("min_doc_words", 50)),
        min_num_sentences=int(job.config.get("min_num_sentences", 2)),
        split_sentences=bool(job.config.get("split_sentences", True)),
    )
    execs = build_sentence_dedup_executors(
        job.paths,
        tasks=job.tasks or job.workers,
        workers=job.workers,
        sentence_config=sentence_config,
        input_override=job.input_view,
        output_override=job.output_folder,
    )
    execs[-1].run()
    return stage_io_counts(job.paths.logs_dir("sentence_dedup") / "3_filter")
