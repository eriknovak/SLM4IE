"""Corpus-level statistics block for the SLM4IE Slovenian corpus.

datatrove's bundled stats blocks (`DocStats` / `WordStats` /
`SentenceStats`) are scoped to fixed groupings — `summary`,
`histogram`, `fqdn`, `suffix` — that are useful for web crawls
but cannot be coerced to bucket by `domain` or `dataset`. We
therefore implement a `CorpusStats` `PipelineStep` that maintains
the corpus counters in memory, plus a `CorpusStatsReduce` step that
merges per-rank partials into a single JSON bundle.

The map step (`CorpusStats`) runs sharded across N workers; each
rank writes a partial pickle to `partials_dir`. The reduce step
(`CorpusStatsReduce`) runs single-process, sums all counters, and
derives the top-K word-frequency table on the merged totals. For
single-process use (e.g. direct unit tests) `CorpusStats` also
accepts an `output_path` and writes the aggregate JSON itself when
`partials_dir` is not given.

The bundle reports corpus totals, per-domain and per-dataset
distributions, and a global top-K word-frequency table. Words are
tokenized with datatrove's bundled Slovenian `SpaCyTokenizer` (via
`load_word_tokenizer(Languages.slovenian)`) — the same tokenizer
that `SentenceDedupSignature` saw, so counts stay consistent. The
only sizeable counter is `word_freq`; it is word-scale (bounded by
the corpus vocabulary), so a rank's memory stays modest regardless
of corpus size.
"""

import json
import logging
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import Languages
from datatrove.utils.word_tokenizers import load_word_tokenizer

logger = logging.getLogger(__name__)

#: Words shorter than this are dropped from the word-frequency table.
#: Keeps the table free of single-letter punctuation noise.
_MIN_WORD_LEN = 2

#: Pattern that identifies "real" word tokens (letters / digits, must
#: contain at least one letter). datatrove's tokenizer returns
#: punctuation as separate tokens; we filter those out before counting.
_WORD_RE = re.compile(r"^(?=.*[^\W\d_])[\w'’-]+$", flags=re.UNICODE)


def _is_word(token: str) -> bool:
    """Return True if *token* should count toward word-frequency stats.

    Args:
        token: A token produced by datatrove's word tokenizer.

    Returns:
        True if the token has length >= `_MIN_WORD_LEN` and contains
        at least one alphabetic character.
    """
    return len(token) >= _MIN_WORD_LEN and bool(_WORD_RE.match(token))


def _build_bundle(
    *,
    top_k_words: int,
    total_docs: int,
    total_words: int,
    per_domain_docs: Counter,
    per_domain_words: Counter,
    per_dataset_docs: Counter,
    per_dataset_words: Counter,
    word_freq: Counter,
) -> Dict[str, Any]:
    """Assemble the final stats dict in the canonical aggregate.json shape.

    Args:
        top_k_words: Word-frequency table size.
        total_docs: Total documents seen.
        total_words: Total words seen.
        per_domain_docs: Document counter keyed by domain.
        per_domain_words: Word counter keyed by domain.
        per_dataset_docs: Document counter keyed by dataset.
        per_dataset_words: Word counter keyed by dataset.
        word_freq: Global word-frequency counter.

    Returns:
        Dict matching the on-disk `aggregate.json` schema.
    """
    def _distribution(docs: Counter, words: Counter) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for key in sorted(docs):
            d = docs[key]
            w = words[key]
            out[key] = {
                "doc_count": int(d),
                "word_count": int(w),
                "avg_doc_words": (w / d) if d else 0.0,
                "share_of_total_words": (w / total_words) if total_words else 0.0,
            }
        return out

    return {
        "total_docs": total_docs,
        "total_words": total_words,
        "by_domain": _distribution(per_domain_docs, per_domain_words),
        "by_dataset": _distribution(per_dataset_docs, per_dataset_words),
        f"word_freq_top_{top_k_words}": [
            [tok, int(count)] for tok, count in word_freq.most_common(top_k_words)
        ],
    }


def _write_bundle(
    bundle: Dict[str, Any],
    *,
    output_path: Path,
    per_dataset_dir: Optional[Path],
    dataset_to_domain: Dict[str, str],
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """Write the JSON bundle to disk.

    Args:
        bundle: The aggregate stats dict produced by `_build_bundle`.
        output_path: Destination JSON file. Parent directories are created.
        per_dataset_dir: Optional folder for per-dataset JSON breakdowns.
        dataset_to_domain: Mapping from dataset key to domain label.
        rank: Worker rank; appended to filename when `world_size > 1`.
        world_size: Total number of workers.
    """
    path = output_path
    if world_size > 1:
        path = path.with_name(f"{path.stem}.{rank:05d}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(bundle, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote aggregate stats bundle to %s", path)

    if per_dataset_dir is None or world_size > 1:
        return
    per_dataset_dir.mkdir(parents=True, exist_ok=True)
    by_dataset = bundle.get("by_dataset", {})
    for dataset_key, dataset_stats in by_dataset.items():
        slim = {
            "dataset": dataset_key,
            "domain": dataset_to_domain.get(dataset_key),
            "doc_count": dataset_stats["doc_count"],
            "word_count": dataset_stats["word_count"],
            "avg_doc_words": dataset_stats["avg_doc_words"],
            "share_of_total_words": dataset_stats["share_of_total_words"],
        }
        with (per_dataset_dir / f"{dataset_key}.json").open("w", encoding="utf-8") as fh:
            json.dump(slim, fh, ensure_ascii=False, indent=2)
    logger.info(
        "Wrote %d per-dataset stats files under %s",
        len(by_dataset), per_dataset_dir,
    )


class CorpusStats(PipelineStep):
    """Per-shard corpus statistics accumulator.

    Maintains the corpus counters on the instance, then either:

    - dumps a per-rank pickle of raw counters to `partials_dir` (when
      set), so a downstream `CorpusStatsReduce` can merge across
      workers; or
    - computes the final top-K aggregate and writes it to
      `output_path` directly (when `partials_dir` is `None`, used for
      single-process direct invocation, e.g. unit tests).

    Attributes:
        partials_dir: When set, run() writes
            `partials_dir/partial.<rank>.pkl` instead of an aggregate.
        output_path: Destination JSON file when running standalone.
        per_dataset_dir: Optional folder for per-dataset breakdowns
            (standalone mode only).
        language: ISO-3 code for the word tokenizer.
        stopwords: Tokens excluded from the word-frequency table.
        top_k_words: Number of entries kept in the word-frequency
            table.
    """

    type = "📊 - STATS"
    name = "📈 Corpus stats"

    def __init__(
        self,
        output_path: Optional[Path] = None,
        per_dataset_dir: Optional[Path] = None,
        language: str = Languages.slovenian,
        stopwords: Optional[Set[str]] = None,
        top_k_words: int = 5_000,
        partials_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the aggregator.

        Args:
            output_path: Aggregate JSON destination for standalone use.
                Ignored when `partials_dir` is given.
            per_dataset_dir: Optional per-dataset breakdown folder
                (standalone use only).
            language: ISO-3 code for the tokenizer
                (`Languages.slovenian` by default).
            stopwords: Optional stopword set; tokens in this set are
                excluded from the word-frequency table.
            top_k_words: Word-frequency table size.
            partials_dir: When set, write a per-rank pickle of raw
                counters here and skip aggregate-bundle computation.
                A downstream `CorpusStatsReduce` then merges them.
        """
        super().__init__()
        self.output_path = Path(output_path) if output_path is not None else None
        self.per_dataset_dir = Path(per_dataset_dir) if per_dataset_dir is not None else None
        self.language = language
        self.stopwords = set(stopwords) if stopwords is not None else set()
        self.top_k_words = top_k_words
        self.partials_dir = Path(partials_dir) if partials_dir is not None else None
        self._tokenizer = None

    def _ensure_tokenizer(self):
        """Build the datatrove tokenizer on first use."""
        if self._tokenizer is None:
            self._tokenizer = load_word_tokenizer(self.language)
        return self._tokenizer

    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize and filter to lowercase content words.

        Args:
            text: Raw document text.

        Returns:
            List of lowercased word tokens, with stopwords and pure
            punctuation removed.
        """
        tokens = self._ensure_tokenizer().word_tokenize(text)
        out: List[str] = []
        for tok in tokens:
            if not _is_word(tok):
                continue
            low = tok.lower()
            if low in self.stopwords:
                continue
            out.append(low)
        return out

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Walk the document stream and emit a partial pickle or JSON bundle.

        Args:
            data: Input document stream (post-dedup).
            rank: Worker rank.
            world_size: Number of parallel workers.

        Yields:
            Documents are passed through unchanged so further pipeline
            steps can reuse the stream.
        """
        total_docs = 0
        total_words = 0
        per_domain_docs: Counter[str] = Counter()
        per_domain_words: Counter[str] = Counter()
        per_dataset_docs: Counter[str] = Counter()
        per_dataset_words: Counter[str] = Counter()
        word_freq: Counter[str] = Counter()
        # Captures the (dataset → domain) edge that aggregate counters
        # lose. Used to label per-dataset bundles. Datasets that mix
        # multiple domains in a single shard collapse to "_mixed".
        dataset_to_domain: Dict[str, str] = {}

        for doc in data:
            with self.track_time():
                domain = doc.metadata.get("domain", "_unknown")
                dataset = doc.metadata.get("dataset", "_unknown")

                if dataset in dataset_to_domain and dataset_to_domain[dataset] != domain:
                    dataset_to_domain[dataset] = "_mixed"
                else:
                    dataset_to_domain.setdefault(dataset, domain)

                words = self._tokenize_words(doc.text)
                n = len(words)

                total_docs += 1
                total_words += n
                per_domain_docs[domain] += 1
                per_domain_words[domain] += n
                per_dataset_docs[dataset] += 1
                per_dataset_words[dataset] += n
                word_freq.update(words)

            yield doc

        if self.partials_dir is not None:
            self._write_partial(
                rank=rank,
                total_docs=total_docs,
                total_words=total_words,
                per_domain_docs=per_domain_docs,
                per_domain_words=per_domain_words,
                per_dataset_docs=per_dataset_docs,
                per_dataset_words=per_dataset_words,
                word_freq=word_freq,
                dataset_to_domain=dataset_to_domain,
            )
            return

        if self.output_path is None:
            raise ValueError("CorpusStats requires either partials_dir or output_path.")

        bundle = _build_bundle(
            top_k_words=self.top_k_words,
            total_docs=total_docs,
            total_words=total_words,
            per_domain_docs=per_domain_docs,
            per_domain_words=per_domain_words,
            per_dataset_docs=per_dataset_docs,
            per_dataset_words=per_dataset_words,
            word_freq=word_freq,
        )
        _write_bundle(
            bundle,
            output_path=self.output_path,
            per_dataset_dir=self.per_dataset_dir,
            dataset_to_domain=dataset_to_domain,
            rank=rank,
            world_size=world_size,
        )

    def _write_partial(
        self,
        *,
        rank: int,
        total_docs: int,
        total_words: int,
        per_domain_docs: Counter,
        per_domain_words: Counter,
        per_dataset_docs: Counter,
        per_dataset_words: Counter,
        word_freq: Counter,
        dataset_to_domain: Dict[str, str],
    ) -> None:
        """Pickle this rank's raw counters into `partials_dir`.

        Args:
            rank: Worker rank, used to name the partial file.
            total_docs: Local doc count.
            total_words: Local word count.
            per_domain_docs: Local per-domain doc counter.
            per_domain_words: Local per-domain word counter.
            per_dataset_docs: Local per-dataset doc counter.
            per_dataset_words: Local per-dataset word counter.
            word_freq: Local word-frequency counter.
            dataset_to_domain: Local dataset→domain mapping.
        """
        assert self.partials_dir is not None  # for type-checkers
        self.partials_dir.mkdir(parents=True, exist_ok=True)
        path = self.partials_dir / f"partial.{rank:05d}.pkl"
        payload = {
            "total_docs": total_docs,
            "total_words": total_words,
            "per_domain_docs": per_domain_docs,
            "per_domain_words": per_domain_words,
            "per_dataset_docs": per_dataset_docs,
            "per_dataset_words": per_dataset_words,
            "word_freq": word_freq,
            "dataset_to_domain": dataset_to_domain,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Wrote stats partial (rank=%d, docs=%d) to %s", rank, total_docs, path)


class CorpusStatsReduce(PipelineStep):
    """Reduce step: merge per-rank partial pickles into the final bundle.

    Reads every `partial.*.pkl` under `partials_dir`, sums all
    counters, derives the top-K word-frequency table on the merged
    totals, then writes `output_path` (and, if given, per-dataset
    breakdowns under `per_dataset_dir`). Partials are loaded one at a
    time, so peak memory is the merged accumulators plus a single
    partial. After a successful write the partials are deleted so the
    stats stage folder only carries the canonical aggregate output.

    Attributes:
        partials_dir: Folder of `partial.*.pkl` written by `CorpusStats`.
        output_path: Aggregate JSON destination.
        per_dataset_dir: Optional folder for per-dataset breakdowns.
        top_k_words: Word-frequency table size.
        cleanup: When True, remove partials after a successful reduce.
    """

    type = "📊 - STATS"
    name = "📈 Corpus stats reduce"

    def __init__(
        self,
        partials_dir: Path,
        output_path: Path,
        per_dataset_dir: Optional[Path] = None,
        top_k_words: int = 5_000,
        cleanup: bool = True,
    ) -> None:
        """Initialize the reduce step.

        Args:
            partials_dir: Where the map step wrote `partial.*.pkl`.
            output_path: Aggregate JSON destination.
            per_dataset_dir: Optional folder for per-dataset breakdowns.
            top_k_words: Word-frequency table size.
            cleanup: When True (default), remove `partial.*.pkl` from
                `partials_dir` after writing the aggregate.
        """
        super().__init__()
        self.partials_dir = Path(partials_dir)
        self.output_path = Path(output_path)
        self.per_dataset_dir = Path(per_dataset_dir) if per_dataset_dir is not None else None
        self.top_k_words = top_k_words
        self.cleanup = cleanup

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Merge partials and emit the final JSON bundle.

        Args:
            data: Unused; kept for `PipelineStep` compatibility. Any
                upstream documents are passed through (typically the
                reduce step is the only step in its executor and `data`
                is `None`).
            rank: Worker rank (must be 0; reduce is single-process).
            world_size: Number of parallel workers (must be 1).

        Yields:
            Forwarded upstream documents, if any. The step's effect is
            on disk — it writes the aggregate JSON.
        """
        if data is not None:
            yield from data

        partials = sorted(self.partials_dir.glob("partial.*.pkl"))
        if not partials:
            raise FileNotFoundError(
                f"No partial.*.pkl files found under {self.partials_dir}; "
                "did the map step run?"
            )

        with self.track_time():
            merged = self._merge_partials(partials)

            bundle = _build_bundle(
                top_k_words=self.top_k_words,
                total_docs=merged["total_docs"],
                total_words=merged["total_words"],
                per_domain_docs=merged["per_domain_docs"],
                per_domain_words=merged["per_domain_words"],
                per_dataset_docs=merged["per_dataset_docs"],
                per_dataset_words=merged["per_dataset_words"],
                word_freq=merged["word_freq"],
            )
            _write_bundle(
                bundle,
                output_path=self.output_path,
                per_dataset_dir=self.per_dataset_dir,
                dataset_to_domain=merged["dataset_to_domain"],
                rank=0,
                world_size=1,
            )

        if self.cleanup:
            for p in partials:
                p.unlink()
            # Drop the partials dir if empty; harmless if not.
            try:
                self.partials_dir.rmdir()
            except OSError:
                pass

    def _merge_partials(self, partials: Sequence[Path]) -> Dict[str, Any]:
        """Load and sum all partial pickles into one combined dict.

        Partials are loaded and merged one at a time so peak memory is
        the running accumulators plus a single partial.

        Args:
            partials: Sorted list of partial pickle paths.

        Returns:
            Dict with the same keys as a single partial payload but
            with all counters summed and `dataset_to_domain` resolved.
        """
        total_docs = 0
        total_words = 0
        per_domain_docs: Counter[str] = Counter()
        per_domain_words: Counter[str] = Counter()
        per_dataset_docs: Counter[str] = Counter()
        per_dataset_words: Counter[str] = Counter()
        word_freq: Counter[str] = Counter()
        dataset_to_domain: Dict[str, str] = {}

        for path in partials:
            with path.open("rb") as fh:
                payload = pickle.load(fh)
            total_docs += int(payload.get("total_docs", 0))
            total_words += int(payload.get("total_words", 0))
            per_domain_docs.update(payload.get("per_domain_docs", {}))
            per_domain_words.update(payload.get("per_domain_words", {}))
            per_dataset_docs.update(payload.get("per_dataset_docs", {}))
            per_dataset_words.update(payload.get("per_dataset_words", {}))
            word_freq.update(payload.get("word_freq", {}))
            for dataset, domain in payload.get("dataset_to_domain", {}).items():
                if dataset in dataset_to_domain and dataset_to_domain[dataset] != domain:
                    dataset_to_domain[dataset] = "_mixed"
                else:
                    dataset_to_domain.setdefault(dataset, domain)

        return {
            "total_docs": total_docs,
            "total_words": total_words,
            "per_domain_docs": per_domain_docs,
            "per_domain_words": per_domain_words,
            "per_dataset_docs": per_dataset_docs,
            "per_dataset_words": per_dataset_words,
            "word_freq": word_freq,
            "dataset_to_domain": dataset_to_domain,
        }
