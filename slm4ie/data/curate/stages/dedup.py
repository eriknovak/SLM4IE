"""Stages 5 and 6: corpus-wide exact and sentence deduplication.

Both stages chain three datatrove executors: signature → find duplicates →
filter and write, passing signatures and duplicate lists through the stage's
scratch folder. Besides the two builders and their runners, this module holds
`doc_text` (the content getter for whole-document exact dedup), two
`ExactDedupConfig` factories (`make_exact_config` for the form driven by
`curate.yaml::exact_dedup`, `default_exact_config` as a zero-arg alias),
`CompactSentenceDedupSignature`, a memory-lean drop-in for datatrove's
sentence-dedup signature step, and `KeptFractionSentenceDedupFilter`, the
filter step with the kept-fraction floor. Both stages share this module, so
editing it reruns both.
"""

import contextlib
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Literal, Optional, Tuple, cast

import numpy as np
from datatrove.data import Document, DocumentsPipeline
from datatrove.executor import LocalPipelineExecutor
from datatrove.io import DataFolderLike
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
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.logging import logger
from datatrove.utils.text import split_into_sentences
from datatrove.utils.typeshelper import ExtensionHelperSD, Languages, StatHints
from tqdm import tqdm

from slm4ie.data.curate.paths import CuratePaths
from slm4ie.data.curate.stages import StageRun
from slm4ie.data.curate.stages.common import jsonl_reader, jsonl_writer, stage_io_counts

#: Signatures buffered as Python tuples before packing into a numpy chunk.
SIGNATURE_FLUSH_EVERY: int = 1_000_000

#: Last sentence index datatrove's 16-bit `sent` field can hold.
MAX_SENTENCE_INDEX: int = np.iinfo(np.uint16).max


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
    Windows starting past `MAX_SENTENCE_INDEX` overflow datatrove's 16-bit
    sentence field, so they are skipped and counted in the step's stats as
    `windows_past_sentence_limit`; the rest of such a document is never
    deduplicated.
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
                hashes = self.get_hashes(doc, doc_idx)
                kept = [(hash_, doc_i, sent) for hash_, doc_i, sent in hashes if sent <= MAX_SENTENCE_INDEX]
                if len(kept) < len(hashes):
                    self.stat_update("windows_past_sentence_limit", value=len(hashes) - len(kept))
                buffer.extend(kept)
                if len(buffer) >= SIGNATURE_FLUSH_EVERY:
                    chunks.append(np.array(buffer, dtype=dtype))
                    buffer = []
        chunks.append(np.array(buffer, dtype=dtype))
        signatures = np.concatenate(chunks)
        del chunks, buffer
        self.save_hashes(rank, signatures)


#: Kept-fraction floor when `sentence_dedup.min_kept_fraction` is unset: a
#: document that lost over half its words to windows seen elsewhere is a later
#: copy of a duplicated page (issue #40).
DEFAULT_MIN_KEPT_FRACTION: float = 0.5

#: Stat names of the filter step, one per reason a changed document is dropped.
DROP_BELOW_KEPT_FRACTION: str = "dropped_below_kept_fraction"
DROP_BELOW_MIN_DOC_WORDS: str = "dropped_below_min_doc_words"
DROP_BELOW_MIN_NUM_SENTENCES: str = "dropped_below_min_num_sentences"
DROP_EMPTY: str = "dropped_empty"


class KeptFractionSentenceDedupFilter(SentenceDedupFilter):
    """Sentence-dedup filter step with a floor on the fraction of text that survives.

    datatrove's `SentenceDedupFilter` keeps a document the find step changed
    as long as the surviving text clears `min_doc_words` and
    `min_num_sentences`. A later copy of a page that several sources crawled
    then survives as a fragment of the few sentences that did not match
    elsewhere. This subclass adds a third floor: a changed document whose
    surviving word count is below `min_kept_fraction` of its original word
    count is dropped. Both counts use the step's word tokenizer. An untouched
    document is kept regardless, as in datatrove, and `min_kept_fraction=0`
    disables the floor.

    datatrove keeps the keep decision inline in `run`, so `run` and the index
    loading in `_load_duplicates` are copied from datatrove 0.9.0 with the
    decision moved into `drop_reason`; `remove_dup_sentences` and
    `read_duplicates` are inherited. Every dropped
    document is counted under the stat of the first floor it failed
    (`DROP_BELOW_KEPT_FRACTION`, `DROP_BELOW_MIN_DOC_WORDS`,
    `DROP_BELOW_MIN_NUM_SENTENCES`, or `DROP_EMPTY` when nothing survived),
    so the per-task stats files say why the stage dropped what it dropped.
    """

    def __init__(
        self,
        data_folder: DataFolderLike,
        config: Optional[SentDedupConfig] = None,
        exclusion_writer: Optional[DiskWriter] = None,
        language: str = Languages.english,
        min_kept_fraction: float = DEFAULT_MIN_KEPT_FRACTION,
    ) -> None:
        """Build the step.

        Args:
            data_folder: Folder holding the find step's duplicate files.
            config: Window size and the two datatrove floors.
            exclusion_writer: Optional writer for the dropped and the trimmed originals.
            language: ISO-3 code for the word and sentence tokenizer.
            min_kept_fraction: Lowest fraction of a changed document's words that may
                survive; `0` disables the floor.

        Raises:
            ValueError: If `min_kept_fraction` is outside `[0, 1]`.
        """
        super().__init__(data_folder, config=config, exclusion_writer=exclusion_writer, language=language)
        if not 0 <= min_kept_fraction <= 1:
            raise ValueError(f"min_kept_fraction must be in [0, 1], got {min_kept_fraction!r}")
        self.min_kept_fraction = min_kept_fraction

    def drop_reason(self, original: str, filtered: str) -> Optional[str]:
        """Name the first floor a changed document fails, or None when it is kept.

        Args:
            original: The document's text before duplicate removal.
            filtered: The text left after duplicate removal.

        Returns:
            One of the `DROP_*` stat names, or None.
        """
        if not filtered:
            return DROP_EMPTY
        kept_words = len(self.tokenizer.word_tokenize(filtered))
        if self.min_kept_fraction > 0:
            original_words = len(self.tokenizer.word_tokenize(original))
            if kept_words < self.min_kept_fraction * original_words:
                return DROP_BELOW_KEPT_FRACTION
        if 0 < self.config.min_doc_words and kept_words < self.config.min_doc_words:
            return DROP_BELOW_MIN_DOC_WORDS
        if 0 < self.config.min_num_sentences:
            if len(split_into_sentences(filtered, self.language)) < self.config.min_num_sentences:
                return DROP_BELOW_MIN_NUM_SENTENCES
        return None

    def _load_duplicates(self, rank: int) -> Tuple[np.ndarray, np.ndarray]:
        """Read this rank's duplicate windows from every find-step folder.

        Args:
            rank: Task rank; names the duplicate files.

        Returns:
            The sorted `(doc, sent)` pairs and the index where each document's pairs start.
        """
        folders = self.data_folder.list_files(include_directories=True, recursive=False)
        files = [
            f
            for f in [f"{folder}/{rank:05d}{ExtensionHelperSD.stage_2_duplicates}" for folder in folders]
            if self.data_folder.exists(f)
        ]
        logger.info(f"Loading duplicate indexes from {len(files)} results files.")
        all_dups = np.array([], dtype=[("doc", "<u4"), ("sent", "<u2")])
        if files:
            with ThreadPoolExecutor() as pool:
                all_dups = np.concatenate(
                    list(tqdm(pool.map(self.read_duplicates, self.data_folder.open_files(files)), total=len(files))),
                    axis=0,
                )
            all_dups.sort()
        _, doc_starts = np.unique(all_dups["doc"], return_index=True)
        logger.info("Loaded duplicate indexes.")
        return all_dups, doc_starts

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Remove the duplicate windows of each document and drop what falls below a floor.

        Args:
            data: Documents of this task, in the order the signature step saw them.
            rank: Task rank; names the duplicate files.
            world_size: Total number of tasks (unused, part of the step API).

        Yields:
            The kept documents, trimmed where a window was removed.
        """
        all_dups, doc_starts = self._load_duplicates(rank)
        dups_doc_i = 0
        with self.exclusion_writer if self.exclusion_writer else contextlib.nullcontext() as writer:
            for doc_idx, doc in enumerate(data):
                self.stat_update(StatHints.total)
                with self.stats.time_stats:
                    if dups_doc_i >= len(doc_starts) or all_dups["doc"][doc_starts[dups_doc_i]] > doc_idx:
                        filtered_text, original_formatted = doc.text, None
                    else:
                        left = doc_starts[dups_doc_i]
                        right = doc_starts[dups_doc_i + 1] if dups_doc_i + 1 < len(doc_starts) else None
                        filtered_text, original_formatted = self.remove_dup_sentences(doc, all_dups["sent"][left:right])
                        dups_doc_i += 1
                    unchanged = filtered_text == doc.text
                    reason = None if unchanged and filtered_text else self.drop_reason(doc.text, filtered_text)
                if reason is None:
                    self.update_doc_stats(doc)
                    if not unchanged and writer:
                        writer.write(dataclasses.replace(doc, text=original_formatted), rank=rank)
                    doc.text = filtered_text
                    yield doc
                else:
                    self.stat_update(reason)
                    if writer:
                        doc.text = original_formatted
                        writer.write(doc, rank=rank)


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
            stage's output folder (the run loop's staging folder).

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
    min_kept_fraction: float = DEFAULT_MIN_KEPT_FRACTION,
    language: str = Languages.slovenian,
    input_override: Optional[Path] = None,
    output_override: Optional[Path] = None,
) -> List[LocalPipelineExecutor]:
    """Build the sentence-dedup stage: sig → find → filter+write 06_sentence_dedup/.

    Three executors chained via `depends`, mirroring the exact stage:
        1. (parallel) read 05_exact_dedup/ → CompactSentenceDedupSignature → <scratch>/sigs/
        2. (single)   SentenceFindDedups(<scratch>/sigs/) → <scratch>/dups/
        3. (parallel) read 05_exact_dedup/ → KeptFractionSentenceDedupFilter → write 06_sentence_dedup/

    Args:
        paths: Resolved input/output locations.
        tasks: Task count for executors 1 and 3; each task reads its own
            slice of the input shards, so more tasks means less memory per task.
        workers: Tasks run at once; defaults to `tasks`.
        finder_workers: Worker count for the find executor.
        sentence_config: Optional `SentDedupConfig`.
        min_kept_fraction: Lowest fraction of a changed document's words that
            may survive the filter step; `0` disables the floor.
        language: ISO-3 code for the sentence tokenizer.
        input_override: Optional folder to read from instead of the
            exact-dedup stage's output, used to restrict the stage to the
            roster's datasets through a symlinked view.
        output_override: Optional folder to write to instead of the
            stage's output folder (the run loop's staging folder).

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
            KeptFractionSentenceDedupFilter(
                data_folder=str(dups),
                config=cfg,
                language=language,
                min_kept_fraction=min_kept_fraction,
            ),
            jsonl_writer(out),
        ],
        tasks=tasks,
        workers=workers,
        logging_dir=str(paths.logs_dir("sentence_dedup") / "3_filter"),
        depends=find,
        skip_completed=True,
    )
    return [sig, find, filt]


def run_exact_dedup(job: StageRun) -> Tuple[int, int]:
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


def run_sentence_dedup(job: StageRun) -> Tuple[int, int]:
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
        min_kept_fraction=float(job.config.get("min_kept_fraction", DEFAULT_MIN_KEPT_FRACTION)),
        input_override=job.input_view,
        output_override=job.output_folder,
    )
    execs[-1].run()
    return stage_io_counts(job.paths.logs_dir("sentence_dedup") / "3_filter")
