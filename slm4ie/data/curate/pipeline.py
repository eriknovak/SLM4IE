"""Builders for the merged six-executor datatrove curation pipeline.

The full ladder is:

    1. lang + exact-sig         -> tempdir/lang_tagged    (parallel)
    2. exact-find               (single worker)
    3. exact-filter + sent-sig  -> tempdir/after_exact    (parallel)
    4. sent-find                (single worker)
    5. sent-filter + write      -> final/                 (parallel)
    6. corpus-stats             -> final/statistics       (single)

Stages 1, 3 and 5 scale with the `tasks` argument; stages 2 and 4
scale with `finder_workers`; stage 6 is single-process by design
because `CorpusStats` keeps global counters on its instance.

Each builder function returns a `LocalPipelineExecutor` chained to the
previous one via `depends=`. The CLI calls `build_curate_executors`
inside a `TemporaryDirectory` (or, in debug mode, a persistent
`final/_dedup/`) and then runs the last executor; datatrove walks the
`depends` chain and runs the rest in order.

I/O layout — every reader walks per-dataset folders recursively
(`<root>/<dataset>/<part>.jsonl.gz`) and every writer emits one file
per rank per dataset (`<root>/<dataset>/<rank>.jsonl.gz`). This lets
the upstream `to_datatrove.py` shard each dataset across many small
files, so `tasks > 1` actually parallelizes the heavy datasets that
used to be bound to a single rank.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

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
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.utils.typeshelper import Languages

from slm4ie.data.curate.dedup import default_exact_config
from slm4ie.data.curate.language import LinguaLanguageFilter
from slm4ie.data.curate.stats import CorpusStats

logger = logging.getLogger(__name__)


@dataclass
class CuratePaths:
    """Resolved filesystem locations for one curation run.

    Attributes:
        input_folder: Where the upstream `<key>.jsonl.gz` shards live
            (`<output_dir>/datatrove` in normal use).
        final_folder: Where the deduplicated training corpus is
            written (`<output_dir>/final`).
        statistics_folder: Where `aggregate.json` and
            `per_dataset/<key>.json` are written
            (`<output_dir>/final/statistics`).
        scratch_folder: Root for all dedup intermediate state. A
            `TemporaryDirectory` path in normal use, or
            `<output_dir>/final/_dedup` when debug mode is on.
        debug: Whether scratch state is preserved after the run and
            whether dropped duplicates are routed to inspectable JSONL
            shards.
    """

    input_folder: Path
    final_folder: Path
    statistics_folder: Path
    scratch_folder: Path
    debug: bool = False


def build_curate_executors(
    paths: CuratePaths,
    *,
    tasks: int = 1,
    finder_workers: int = 1,
    sentence_config: Optional[SentDedupConfig] = None,
    exact_config: Optional[ExactDedupConfig] = None,
    language: str = Languages.slovenian,
    target_language_iso2: str = "sl",
    candidate_languages: Optional[List[str]] = None,
    lang_threshold: float = 0.5,
    lang_mode: str = "tag",
    stopwords: Optional[Set[str]] = None,
    top_k_words: int = 5_000,
    top_k_ngrams: int = 5_000,
    keyword_top_k: int = 200,
    compute_keywords: bool = True,
    logging_dir: Optional[Path] = None,
) -> List[LocalPipelineExecutor]:
    """Build the chained six-executor curation pipeline.

    Args:
        paths: Resolved input/output/scratch locations.
        tasks: Parallel tasks for the per-shard executors (1, 3, 5).
            `finder_workers` controls the dedicated find stages
            (executors 2 and 4) separately. The corpus-stats stage
            (executor 6) runs single-process because `CorpusStats`
            keeps global counters on its instance.
        finder_workers: Number of finder tasks for the dedup find
            stages. Must match the corresponding signature stage's
            `finder_workers` argument; both are wired here.
        sentence_config: Optional sentence-dedup config override.
        exact_config: Optional exact-dedup config override; defaults
            to one whose `content_getter` hashes `doc.text`.
        language: ISO-3 code for the sentence/word tokenizer
            (`Languages.slovenian` by default).
        target_language_iso2: ISO 639-1 code for the language to
            verify with lingua-py.
        candidate_languages: ISO 639-1 candidate set for lingua.
        lang_threshold: Threshold passed to the lingua filter when
            `lang_mode="filter"`.
        lang_mode: `"tag"` keeps every document and only adds metadata;
            `"filter"` drops sub-threshold documents.
        stopwords: Stopword set passed to `CorpusStats`.
        top_k_words: Word-frequency table size.
        top_k_ngrams: Per-order n-gram table size.
        keyword_top_k: TF-IDF keywords per bucket.
        compute_keywords: Disable to skip the classla pass.
        logging_dir: Datatrove logging root. Stage subdirectories are
            created under this path.

    Returns:
        The six executors in execution order. The CLI typically runs
        only the last one; datatrove follows `depends` to invoke
        upstream stages first.
    """
    sent_cfg = sentence_config or SentDedupConfig()
    exact_cfg = exact_config or default_exact_config()

    paths.final_folder.mkdir(parents=True, exist_ok=True)
    paths.statistics_folder.mkdir(parents=True, exist_ok=True)
    paths.scratch_folder.mkdir(parents=True, exist_ok=True)

    lang_tagged = paths.scratch_folder / "lang_tagged"
    after_exact = paths.scratch_folder / "after_exact"
    exact_sigs = paths.scratch_folder / "exact_sigs"
    exact_dups = paths.scratch_folder / "exact_dups"
    sent_sigs = paths.scratch_folder / "sent_sigs"
    sent_dups = paths.scratch_folder / "sent_dups"
    base_logs = (logging_dir or paths.scratch_folder / "_logs")

    exact_dropped = paths.scratch_folder / "exact_dropped" if paths.debug else None
    sentence_dropped = paths.scratch_folder / "sentence_dropped" if paths.debug else None

    # Executor 1: read raw → lingua tag → write tagged shards → exact-sig.
    # Order matters: `ExactDedupSignature` is a sink that consumes the
    # stream without yielding, so it must be the last block. Putting the
    # `JsonlWriter` before it lets the sig step still see every doc.
    exec1 = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                str(paths.input_folder),
                glob_pattern="**/*.jsonl.gz",
                shuffle_files=False,
                recursive=True,
            ),
            LinguaLanguageFilter(
                target=target_language_iso2,
                candidates=candidate_languages,
                threshold=lang_threshold,
                mode=lang_mode,
            ),
            JsonlWriter(
                output_folder=str(lang_tagged),
                output_filename="${dataset}/${rank}.jsonl.gz",
            ),
            ExactDedupSignature(
                output_folder=str(exact_sigs),
                config=exact_cfg,
                finder_workers=finder_workers,
            ),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(base_logs / "01_lang_exact_sig"),
        skip_completed=False,
    )

    # Executor 2: exact-find. Single worker (or `finder_workers`).
    exec2 = LocalPipelineExecutor(
        pipeline=[
            ExactFindDedups(
                data_folder=str(exact_sigs),
                output_folder=str(exact_dups),
                config=exact_cfg,
            ),
        ],
        tasks=finder_workers,
        workers=finder_workers,
        logging_dir=str(base_logs / "02_exact_find"),
        depends=exec1,
        skip_completed=False,
    )

    # Executor 3: read tagged shards → exact-filter → write filtered →
    # sentence-sig (sink, must come last for the same reason as exec 1).
    exec3 = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                str(lang_tagged),
                glob_pattern="**/*.jsonl.gz",
                shuffle_files=False,
                recursive=True,
            ),
            ExactDedupFilter(
                data_folder=str(exact_dups),
                config=exact_cfg,
                exclusion_writer=(
                    JsonlWriter(output_folder=str(exact_dropped)) if exact_dropped else None
                ),
            ),
            JsonlWriter(
                output_folder=str(after_exact),
                output_filename="${dataset}/${rank}.jsonl.gz",
            ),
            SentenceDedupSignature(
                output_folder=str(sent_sigs),
                config=sent_cfg,
                finder_workers=finder_workers,
                language=language,
            ),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(base_logs / "03_exact_filter_sent_sig"),
        depends=exec2,
        skip_completed=False,
    )

    # Executor 4: sentence-find. Single worker (or `finder_workers`).
    exec4 = LocalPipelineExecutor(
        pipeline=[
            SentenceFindDedups(
                data_folder=str(sent_sigs),
                output_folder=str(sent_dups),
                config=sent_cfg,
            ),
        ],
        tasks=finder_workers,
        workers=finder_workers,
        logging_dir=str(base_logs / "04_sent_find"),
        depends=exec3,
        skip_completed=False,
    )

    # Executor 5: sentence-filter → write final corpus. Parallel-safe:
    # input shards are sharded round-robin across ranks; multiple ranks
    # may now touch the same dataset (one dataset spans many shards), so
    # the writer routes each rank's per-dataset output into a distinct
    # file via `${dataset}/${rank}.jsonl.gz`.
    exec5 = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                str(after_exact),
                glob_pattern="**/*.jsonl.gz",
                shuffle_files=False,
                recursive=True,
            ),
            SentenceDedupFilter(
                data_folder=str(sent_dups),
                config=sent_cfg,
                language=language,
                exclusion_writer=(
                    JsonlWriter(output_folder=str(sentence_dropped)) if sentence_dropped else None
                ),
            ),
            JsonlWriter(
                output_folder=str(paths.final_folder),
                output_filename="${dataset}/${rank}.jsonl.gz",
            ),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(base_logs / "05_sent_filter_write"),
        depends=exec4,
        skip_completed=False,
    )

    # Executor 6: read final corpus → CorpusStats. Single-process
    # because CorpusStats keeps global counters on its instance.
    exec6 = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                str(paths.final_folder),
                glob_pattern="**/*.jsonl.gz",
                shuffle_files=False,
                recursive=True,
            ),
            CorpusStats(
                output_path=paths.statistics_folder / "aggregate.json",
                per_dataset_dir=paths.statistics_folder / "per_dataset",
                language=language,
                stopwords=stopwords,
                top_k_words=top_k_words,
                top_k_ngrams=top_k_ngrams,
                keyword_top_k=keyword_top_k,
                compute_keywords=compute_keywords,
            ),
        ],
        tasks=1,
        workers=1,
        logging_dir=str(base_logs / "06_corpus_stats"),
        depends=exec5,
        skip_completed=False,
    )

    return [exec1, exec2, exec3, exec4, exec5, exec6]
