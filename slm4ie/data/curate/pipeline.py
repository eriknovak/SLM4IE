r"""Builders for the merged six-executor datatrove curation pipeline.

The full ladder is:

    1. lang(filter) + quality + repetition + exact-sig  -> tempdir/lang_tagged    (parallel)
    2. exact-find                                       (single worker)
    3. exact-filter + sent-sig                          -> tempdir/after_exact    (parallel)
    4. sent-find                                        (single worker)
    5. sent-filter + write                              -> final/                 (parallel)
    6. corpus-stats                                     -> final/statistics       (single)

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
from typing import List, Optional, Sequence, Set

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
from datatrove.pipeline.filters import GopherQualityFilter, GopherRepetitionFilter
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
            whether dropped documents are routed to inspectable JSONL
            shards (one folder per drop stage).
    """

    input_folder: Path
    final_folder: Path
    statistics_folder: Path
    scratch_folder: Path
    debug: bool = False


@dataclass
class QualityConfig:
    """Knobs for the Gopher quality heuristic filter.

    Mirrors `GopherQualityFilter.__init__` defaults so the CLI can
    override individual values from `curate.yaml` without listing
    every parameter explicitly.

    Attributes:
        min_doc_words: Minimum word count; shorter docs are dropped.
        max_doc_words: Maximum word count; longer docs are dropped.
        min_avg_word_length: Minimum average word length in chars.
        max_avg_word_length: Maximum average word length in chars.
        max_symbol_word_ratio: Max fraction of word-like tokens that
            are pure symbols (e.g. `#`, `…`).
        max_bullet_lines_ratio: Max fraction of lines starting with a
            bullet glyph.
        max_ellipsis_lines_ratio: Max fraction of lines ending in an
            ellipsis.
        max_non_alpha_words_ratio: Despite its name, this is the
            *minimum* fraction of words that must contain at least one
            alphabetic character (datatrove keeps the legacy name from
            the upstream Gopher paper). Documents below this fraction
            are dropped.
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


def build_curate_executors(
    paths: CuratePaths,
    *,
    tasks: int = 1,
    finder_workers: int = 1,
    sentence_config: Optional[SentDedupConfig] = None,
    exact_config: Optional[ExactDedupConfig] = None,
    quality_config: Optional[QualityConfig] = None,
    language: str = Languages.slovenian,
    target_languages: Sequence[str] = ("sl",),
    candidate_languages: Optional[List[str]] = None,
    lang_mode: str = "tag",
    lang_minimum_relative_distance: float = 0.0,
    lang_low_accuracy: bool = False,
    lang_max_chars: Optional[int] = None,
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
            (executors 2, 4) separately. The corpus-stats stage
            (executor 6) runs single-process because `CorpusStats`
            keeps global counters on its instance.
        finder_workers: Number of finder tasks for the dedup find
            stages. Must match the corresponding signature stage's
            `finder_workers` argument; both are wired here.
        sentence_config: Optional sentence-dedup config override.
        exact_config: Optional exact-dedup config override; defaults
            to one whose `content_getter` hashes `doc.text`.
        quality_config: Optional `GopherQualityFilter` knob bundle;
            defaults to Gopher paper values. The Slovenian stopword
            set is passed to the filter via `stopwords`.
        language: ISO-3 code for the sentence/word tokenizer
            (`Languages.slovenian` by default). Also drives Gopher
            filter tokenization.
        target_languages: ISO 639-1 codes considered "in-language" by
            the lingua filter. In `lang_mode="filter"`, only documents
            whose detected language is in this set are kept. Defaults
            to `("sl",)` for backwards compatibility.
        candidate_languages: ISO 639-1 candidate set for lingua.
        lang_mode: `"tag"` keeps every document and only adds metadata;
            `"filter"` drops out-of-target documents.
        lang_minimum_relative_distance: Required confidence gap between
            the top language and the runner-up before lingua will
            commit to a prediction. `0.0` disables the check. Useful
            for high-precision South Slavic disambiguation; `0.1` is
            a reasonable starting point.
        lang_low_accuracy: Use lingua's trigram-only model. ~5-10x
            faster than the default; recommended for full corpus runs.
        lang_max_chars: Truncate doc text to this many chars before
            language detection. ~2 KB of signal is plenty for a small
            candidate set; `None` disables truncation.
        stopwords: Stopword set used by both `CorpusStats` and
            `GopherQualityFilter`. Pass the Slovenian list — Gopher's
            built-in defaults are English-only and would reject most
            Slovenian documents on `min_stop_words`.
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
    quality_cfg = quality_config or QualityConfig()
    stopwords = stopwords if stopwords is not None else set()

    paths.final_folder.mkdir(parents=True, exist_ok=True)
    paths.statistics_folder.mkdir(parents=True, exist_ok=True)
    paths.scratch_folder.mkdir(parents=True, exist_ok=True)

    lang_tagged = paths.scratch_folder / "lang_tagged"
    after_exact = paths.scratch_folder / "after_exact"
    exact_sigs = paths.scratch_folder / "exact_sigs"
    exact_dups = paths.scratch_folder / "exact_dups"
    sent_sigs = paths.scratch_folder / "sent_sigs"
    sent_dups = paths.scratch_folder / "sent_dups"
    base_logs = logging_dir or paths.scratch_folder / "_logs"

    lang_dropped = paths.scratch_folder / "lang_dropped" if paths.debug else None
    quality_dropped = paths.scratch_folder / "quality_dropped" if paths.debug else None
    repetition_dropped = paths.scratch_folder / "repetition_dropped" if paths.debug else None
    exact_dropped = paths.scratch_folder / "exact_dropped" if paths.debug else None
    sentence_dropped = paths.scratch_folder / "sentence_dropped" if paths.debug else None

    # Executor 1: read raw → lingua filter → Gopher quality → Gopher
    # repetition → write tagged shards → exact-sig. Dropping
    # low-quality docs *before* the exact-sig sink saves hashing work.
    # Order matters: `ExactDedupSignature` is a sink that consumes the
    # stream without yielding, so it must be the last block. The
    # `JsonlWriter` between the filters and the sink lets the sig step
    # still see every surviving doc.
    exec1 = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                str(paths.input_folder),
                glob_pattern="**/*.jsonl.gz",
                shuffle_files=False,
                recursive=True,
            ),
            LinguaLanguageFilter(
                targets=list(target_languages),
                candidates=candidate_languages,
                mode=lang_mode,
                minimum_relative_distance=lang_minimum_relative_distance,
                low_accuracy=lang_low_accuracy,
                max_chars=lang_max_chars,
                exclusion_writer=(
                    JsonlWriter(output_folder=str(lang_dropped)) if lang_dropped else None
                ),
            ),
            GopherQualityFilter(
                language=language,
                stop_words=sorted(stopwords) if stopwords else None,
                min_doc_words=quality_cfg.min_doc_words,
                max_doc_words=quality_cfg.max_doc_words,
                min_avg_word_length=quality_cfg.min_avg_word_length,
                max_avg_word_length=quality_cfg.max_avg_word_length,
                max_symbol_word_ratio=quality_cfg.max_symbol_word_ratio,
                max_bullet_lines_ratio=quality_cfg.max_bullet_lines_ratio,
                max_ellipsis_lines_ratio=quality_cfg.max_ellipsis_lines_ratio,
                max_non_alpha_words_ratio=quality_cfg.max_non_alpha_words_ratio,
                min_stop_words=quality_cfg.min_stop_words,
                exclusion_writer=(
                    JsonlWriter(output_folder=str(quality_dropped)) if quality_dropped else None
                ),
            ),
            GopherRepetitionFilter(
                language=language,
                exclusion_writer=(
                    JsonlWriter(output_folder=str(repetition_dropped)) if repetition_dropped else None
                ),
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
        logging_dir=str(base_logs / "01_lang_quality_exact_sig"),
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
    # may touch the same dataset (one dataset spans many shards), so
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
