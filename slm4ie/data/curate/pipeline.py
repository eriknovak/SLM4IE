"""Per-stage executor builders for the curate pipeline.

Each builder function returns the `LocalPipelineExecutor`(s) needed to
run one user-facing stage. Builders are independent — a caller can run
just the language stage, just the quality stage, etc. Cross-stage
ordering (and downstream invalidation when one stage's output is stale)
is owned by the CLI runner, not by the builders.

I/O layout — every reader walks `<input_folder>/<dataset>/<part>.jsonl.gz`
recursively, and every writer emits `<output_folder>/<dataset>/<rank>.jsonl.gz`,
matching the upstream `to_datatrove.py` per-dataset shard layout. This
preserves dataset provenance through every stage.

Every builder sets `skip_completed=False` on every executor: per-stage
skip is owned by the sentinel system in `slm4ie.data.curate.sentinel`,
not by datatrove's built-in completion tracking. Builders are pure
factories — they do not check, write, or honor sentinels.
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
from slm4ie.data.curate.stages import STAGE_DIRS
from slm4ie.data.curate.stats import CorpusStats

logger = logging.getLogger(__name__)


@dataclass
class CuratePaths:
    """Filesystem locations for one curation run.

    Attributes:
        input_folder: Upstream `<key>/<part>.jsonl.gz` shards
            (typically the output of `to_datatrove.py`).
        output_dir: Curation output root. Stage folders
            (`01_language/`, `02_quality/`, ...) live directly under
            this path, alongside `_dedup_state/` and `_logs/`.
    """

    input_folder: Path
    output_dir: Path

    def stage_dir(self, stage: str) -> Path:
        """Return the folder under `output_dir` that holds *stage*'s output.

        Args:
            stage: Stage name (see `slm4ie.data.curate.stages.STAGE_NAMES`).

        Returns:
            Absolute path to the stage's output folder.

        Raises:
            KeyError: If *stage* is not a known stage name.
        """
        return self.output_dir / STAGE_DIRS[stage]

    @property
    def dedup_state_dir(self) -> Path:
        """Return the folder holding dedup sig/dup scratch between dedup sub-stages.

        Returns:
            `<output_dir>/_dedup_state`. The CLI runner purges its
            contents after each dedup sub-stage's sentinel lands.
        """
        return self.output_dir / "_dedup_state"

    def logs_dir(self, stage: str) -> Path:
        """Return the per-stage logging directory.

        Args:
            stage: Stage name (one of `STAGE_NAMES`).

        Returns:
            `<output_dir>/_logs/<stage>` — datatrove's `logging_dir`
            for that stage's executor chain.
        """
        return self.output_dir / "_logs" / stage


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


def _writer(stage_folder: Path) -> JsonlWriter:
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


def _reader(folder: Path) -> JsonlReader:
    """Return a JsonlReader that walks `<folder>/**/*.jsonl.gz` recursively."""
    return JsonlReader(
        str(folder),
        glob_pattern="**/*.jsonl.gz",
        shuffle_files=False,
        recursive=True,
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
) -> List[LocalPipelineExecutor]:
    """Build the language stage: read input → lingua filter → write 01_language/.

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

    Returns:
        A list with one `LocalPipelineExecutor`.
    """
    out = paths.stage_dir("language")
    executor = LocalPipelineExecutor(
        pipeline=[
            _reader(paths.input_folder),
            LinguaLanguageFilter(
                targets=list(target_languages),
                candidates=candidate_languages,
                mode=lang_mode,
                minimum_relative_distance=lang_minimum_relative_distance,
                low_accuracy=lang_low_accuracy,
                max_chars=lang_max_chars,
            ),
            _writer(out),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("language")),
        skip_completed=False,
    )
    return [executor]


def build_quality_executors(
    paths: CuratePaths,
    *,
    tasks: int = 1,
    quality_config: Optional[QualityConfig] = None,
    language: str = Languages.slovenian,
    stopwords: Optional[Set[str]] = None,
) -> List[LocalPipelineExecutor]:
    """Build the quality stage: read 01_language/ → Gopher quality → write 02_quality/.

    Args:
        paths: Resolved input/output locations.
        tasks: Parallel worker count.
        quality_config: `GopherQualityFilter` knob bundle; defaults to
            Gopher paper values.
        language: ISO-3 code for the word/sentence tokenizer.
        stopwords: Stopword set used by `GopherQualityFilter`.

    Returns:
        A list with one `LocalPipelineExecutor`.
    """
    cfg = quality_config or QualityConfig()
    in_ = paths.stage_dir("language")
    out = paths.stage_dir("quality")
    executor = LocalPipelineExecutor(
        pipeline=[
            _reader(in_),
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
            _writer(out),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("quality")),
        skip_completed=False,
    )
    return [executor]


def build_repetition_executors(
    paths: CuratePaths,
    *,
    tasks: int = 1,
    language: str = Languages.slovenian,
) -> List[LocalPipelineExecutor]:
    """Build the repetition stage: read 02_quality/ → Gopher repetition → write 03_repetition/.

    Args:
        paths: Resolved input/output locations.
        tasks: Parallel worker count.
        language: ISO-3 code for the word/sentence tokenizer the
            repetition filter uses.

    Returns:
        A list with one `LocalPipelineExecutor`.
    """
    in_ = paths.stage_dir("quality")
    out = paths.stage_dir("repetition")
    executor = LocalPipelineExecutor(
        pipeline=[
            _reader(in_),
            GopherRepetitionFilter(language=language),
            _writer(out),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("repetition")),
        skip_completed=False,
    )
    return [executor]


def build_exact_dedup_executors(
    paths: CuratePaths,
    *,
    tasks: int = 1,
    finder_workers: int = 1,
    exact_config: Optional[ExactDedupConfig] = None,
) -> List[LocalPipelineExecutor]:
    """Build the exact-dedup stage: sig → find → filter+write 04_1_dedup/.

    Three executors chained via `depends`:
        1. (parallel) read 03_repetition/ → ExactDedupSignature → exact_sigs/
        2. (single)   ExactFindDedups(exact_sigs/) → exact_dups/
        3. (parallel) read 03_repetition/ → ExactDedupFilter → write 04_1_dedup/

    Args:
        paths: Resolved input/output locations.
        tasks: Parallel worker count for executors 1 and 3.
        finder_workers: Worker count for the single-worker find
            executor 2 (and the `finder_workers` argument of the sig
            executor 1).
        exact_config: Optional `ExactDedupConfig`; defaults to one whose
            `content_getter` hashes `doc.text`.

    Returns:
        Three chained `LocalPipelineExecutor`s.
    """
    cfg = exact_config or default_exact_config()
    in_ = paths.stage_dir("repetition")
    out = paths.stage_dir("exact_dedup")
    sigs = paths.dedup_state_dir / "exact_sigs"
    dups = paths.dedup_state_dir / "exact_dups"

    sig = LocalPipelineExecutor(
        pipeline=[
            _reader(in_),
            ExactDedupSignature(
                output_folder=str(sigs), config=cfg, finder_workers=finder_workers
            ),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("exact_dedup") / "1_sig"),
        skip_completed=False,
    )
    find = LocalPipelineExecutor(
        pipeline=[ExactFindDedups(data_folder=str(sigs), output_folder=str(dups), config=cfg)],
        tasks=finder_workers,
        workers=finder_workers,
        logging_dir=str(paths.logs_dir("exact_dedup") / "2_find"),
        depends=sig,
        skip_completed=False,
    )
    filt = LocalPipelineExecutor(
        pipeline=[
            _reader(in_),
            ExactDedupFilter(data_folder=str(dups), config=cfg),
            _writer(out),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("exact_dedup") / "3_filter"),
        depends=find,
        skip_completed=False,
    )
    return [sig, find, filt]


def build_sentence_dedup_executors(
    paths: CuratePaths,
    *,
    tasks: int = 1,
    finder_workers: int = 1,
    sentence_config: Optional[SentDedupConfig] = None,
    language: str = Languages.slovenian,
) -> List[LocalPipelineExecutor]:
    """Build the sentence-dedup stage: sig → find → filter+write 04_2_dedup/.

    Three executors chained via `depends`, mirroring the exact stage:
        1. (parallel) read 04_1_dedup/ → SentenceDedupSignature → sent_sigs/
        2. (single)   SentenceFindDedups(sent_sigs/) → sent_dups/
        3. (parallel) read 04_1_dedup/ → SentenceDedupFilter → write 04_2_dedup/

    Args:
        paths: Resolved input/output locations.
        tasks: Parallel worker count for executors 1 and 3.
        finder_workers: Worker count for the find executor.
        sentence_config: Optional `SentDedupConfig`.
        language: ISO-3 code for the sentence tokenizer.

    Returns:
        Three chained `LocalPipelineExecutor`s.
    """
    cfg = sentence_config or SentDedupConfig()
    in_ = paths.stage_dir("exact_dedup")
    out = paths.stage_dir("sentence_dedup")
    sigs = paths.dedup_state_dir / "sent_sigs"
    dups = paths.dedup_state_dir / "sent_dups"

    sig = LocalPipelineExecutor(
        pipeline=[
            _reader(in_),
            SentenceDedupSignature(
                output_folder=str(sigs),
                config=cfg,
                finder_workers=finder_workers,
                language=language,
            ),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("sentence_dedup") / "1_sig"),
        skip_completed=False,
    )
    find = LocalPipelineExecutor(
        pipeline=[SentenceFindDedups(data_folder=str(sigs), output_folder=str(dups), config=cfg)],
        tasks=finder_workers,
        workers=finder_workers,
        logging_dir=str(paths.logs_dir("sentence_dedup") / "2_find"),
        depends=sig,
        skip_completed=False,
    )
    filt = LocalPipelineExecutor(
        pipeline=[
            _reader(in_),
            SentenceDedupFilter(data_folder=str(dups), config=cfg, language=language),
            _writer(out),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("sentence_dedup") / "3_filter"),
        depends=find,
        skip_completed=False,
    )
    return [sig, find, filt]


def build_curate_executors(*args: object, **kwargs: object) -> List[LocalPipelineExecutor]:
    """Removed: replaced by per-stage builder functions.

    This name is kept only so that the old `scripts/data/curate.py` can
    be imported during test collection without a hard `ImportError`. Any
    call will raise `NotImplementedError` immediately; the CLI is
    rewritten in Task 6.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Returns:
        Never returns — always raises.

    Raises:
        NotImplementedError: Always. Use the per-stage builders instead.
    """
    # TODO(task-6): remove this shim once scripts/data/curate.py no longer
    # imports build_curate_executors. Kept only so test collection succeeds
    # while the CLI is being rewritten.
    raise NotImplementedError(
        "build_curate_executors has been replaced by per-stage builders "
        "(build_language_executors, build_quality_executors, etc.). "
        "The CLI is rewritten in Task 6."
    )


def build_stats_executors(
    paths: CuratePaths,
    *,
    language: str = Languages.slovenian,
    stopwords: Optional[Set[str]] = None,
    top_k_words: int = 5_000,
    top_k_ngrams: int = 5_000,
    keyword_top_k: int = 200,
    compute_keywords: bool = True,
    ngram_orders: Sequence[int] = (2, 3),
) -> List[LocalPipelineExecutor]:
    """Build the stats stage: read 04_2_dedup/ → CorpusStats → 05_statistics/.

    Single-process by design: `CorpusStats` keeps every counter on the
    instance, so worker fan-out is not supported.

    Args:
        paths: Resolved input/output locations.
        language: ISO-3 code for the tokenizer.
        stopwords: Stopword set used by `CorpusStats`.
        top_k_words: Word-frequency table size.
        top_k_ngrams: Per-order n-gram table size.
        keyword_top_k: TF-IDF keywords per bucket.
        compute_keywords: Disable to skip the classla pass.
        ngram_orders: N-gram orders to compute.

    Returns:
        A list with one single-process `LocalPipelineExecutor`.
    """
    in_ = paths.stage_dir("sentence_dedup")
    out = paths.stage_dir("stats")
    out.mkdir(parents=True, exist_ok=True)

    executor = LocalPipelineExecutor(
        pipeline=[
            _reader(in_),
            CorpusStats(
                output_path=out / "aggregate.json",
                per_dataset_dir=out / "per_dataset",
                language=language,
                stopwords=stopwords or set(),
                top_k_words=top_k_words,
                top_k_ngrams=top_k_ngrams,
                keyword_top_k=keyword_top_k,
                compute_keywords=compute_keywords,
                ngram_orders=ngram_orders,
            ),
        ],
        tasks=1,
        workers=1,
        logging_dir=str(paths.logs_dir("stats")),
        skip_completed=False,
    )
    return [executor]
