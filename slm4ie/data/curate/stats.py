"""Corpus-level statistics block for the SLM4IE Slovenian corpus.

datatrove's bundled stats blocks (`DocStats` / `WordStats` /
`SentenceStats`) are scoped to fixed groupings — `summary`,
`histogram`, `fqdn`, `suffix` — that are useful for web crawls
but cannot be coerced to bucket by `domain` or `dataset`. We
therefore implement a single `CorpusStats` `PipelineStep` that
maintains every counter we care about in memory, runs through the full
deduplicated stream once, and emits a JSON bundle.

The block uses datatrove's bundled Slovenian `SpaCyTokenizer` (via
`load_word_tokenizer(Languages.slovenian)`) for word and sentence
boundaries — same tokenizer that `SentenceDedupSignature` saw, so
counts stay consistent. Keyword/TF-IDF lemmatization additionally
uses `classla` (Slovenian-specific stanza fork) which is loaded
lazily on first call and cached on the instance.
"""

import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import Languages
from datatrove.utils.word_tokenizers import load_word_tokenizer

logger = logging.getLogger(__name__)

#: Words shorter than this are dropped from word-frequency, n-gram, and
#: keyword tables. Keeps tables free of single-letter punctuation noise.
_MIN_WORD_LEN = 2

#: Pattern that identifies "real" word tokens (letters / digits, must
#: contain at least one letter). datatrove's tokenizer returns
#: punctuation as separate tokens; we filter those out before counting.
_WORD_RE = re.compile(r"^(?=.*[^\W\d_])[\w'’-]+$", flags=re.UNICODE)


def _is_word(token: str) -> bool:
    """Return True if *token* should count toward word/n-gram stats.

    Args:
        token: A token produced by datatrove's word tokenizer.

    Returns:
        True if the token has length >= `_MIN_WORD_LEN` and contains
        at least one alphabetic character.
    """
    return len(token) >= _MIN_WORD_LEN and bool(_WORD_RE.match(token))


class CorpusStats(PipelineStep):
    """Single-pass corpus statistics aggregator.

    Maintains every counter on the instance, then serializes them to a
    single JSON file when the input stream is exhausted. Per-shard
    aggregation is not needed at our corpus size; if the pipeline is
    sharded across workers, each rank produces its own JSON file and
    the runner merges them.

    Attributes:
        output_path: Destination JSON file. Parent directories are
            created on `run`.
        language: ISO-3 code for the word/sentence tokenizer.
        stopwords: Tokens excluded from word-frequency, n-gram, and
            keyword tables.
        top_k_words: Number of entries kept in the word-frequency
            table.
        top_k_ngrams: Number of entries kept in each of the bigram /
            trigram tables.
        keyword_top_k: Number of TF-IDF keywords kept per bucket.
        compute_keywords: When False, skip the (slow) classla
            lemmatization stage entirely.
        ngram_orders: Which n-gram orders to compute (e.g. `[2, 3]`).
    """

    type = "📊 - STATS"
    name = "📈 Corpus stats"

    def __init__(
        self,
        output_path: Path,
        per_dataset_dir: Optional[Path] = None,
        language: str = Languages.slovenian,
        stopwords: Optional[Set[str]] = None,
        top_k_words: int = 5_000,
        top_k_ngrams: int = 5_000,
        keyword_top_k: int = 200,
        compute_keywords: bool = True,
        ngram_orders: Iterable[int] = (2, 3),
    ) -> None:
        """Initialize the aggregator with output settings.

        Args:
            output_path: Where to write the corpus-wide aggregate JSON
                (typically `final/statistics/aggregate.json`).
            per_dataset_dir: Optional folder for per-dataset breakdowns.
                When given, one `<dataset>.json` is written under it
                with the same shape as the aggregate but scoped to
                docs whose `metadata.dataset` matches that key.
            language: ISO-3 code for the tokenizer
                (`Languages.slovenian` by default).
            stopwords: Optional stopword set; tokens in this set are
                excluded from word-frequency, n-gram, and keyword
                tables.
            top_k_words: Word-frequency table size.
            top_k_ngrams: Per-order n-gram table size.
            keyword_top_k: TF-IDF keywords kept per bucket.
            compute_keywords: When False, the classla lemmatization
                pass is skipped — useful for fast smoke tests.
            ngram_orders: N-gram orders to compute. Order 1 is always
                produced as the word-frequency table; pass
                `(2, 3)` for the standard bigram + trigram bundle.
        """
        super().__init__()
        self.output_path = Path(output_path)
        self.per_dataset_dir = Path(per_dataset_dir) if per_dataset_dir is not None else None
        self.language = language
        self.stopwords = set(stopwords) if stopwords is not None else set()
        self.top_k_words = top_k_words
        self.top_k_ngrams = top_k_ngrams
        self.keyword_top_k = keyword_top_k
        self.compute_keywords = compute_keywords
        self.ngram_orders = tuple(ngram_orders)
        self._tokenizer = None
        self._classla = None

    def _ensure_tokenizer(self):
        """Build the datatrove tokenizer on first use."""
        if self._tokenizer is None:
            self._tokenizer = load_word_tokenizer(self.language)
        return self._tokenizer

    def _ensure_classla(self):
        """Build a classla pipeline (tokenize+pos+lemma) on first use.

        Returns:
            A ready-to-call `classla.Pipeline` for Slovenian, or
            `None` if classla is not available (in which case the
            keyword stage logs a warning and is skipped).
        """
        if self._classla is not None:
            return self._classla
        try:
            import classla
        except ImportError:
            logger.warning("classla not installed; skipping lemmatized keyword stats.")
            self.compute_keywords = False
            return None
        try:
            self._classla = classla.Pipeline(
                lang="sl",
                processors="tokenize,pos,lemma",
                use_gpu=False,
                logging_level="WARN",
            )
        except Exception as exc:  # noqa: BLE001 — classla raises a variety of types
            logger.warning("classla model load failed (%s); skipping keyword stats.", exc)
            self.compute_keywords = False
            return None
        return self._classla

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
        """Walk the document stream and emit a JSON stats bundle.

        Args:
            data: Input document stream (post-dedup).
            rank: Worker rank; included in the output filename when
                `world_size > 1`.
            world_size: Number of parallel workers.

        Yields:
            Documents are passed through unchanged so further pipeline
            steps can reuse the stream. The block has no effect on
            document content.
        """
        total_docs = 0
        total_words = 0
        per_domain_docs: Counter[str] = Counter()
        per_domain_words: Counter[str] = Counter()
        per_dataset_docs: Counter[str] = Counter()
        per_dataset_words: Counter[str] = Counter()
        word_freq: Counter[str] = Counter()
        ngram_freqs: Dict[int, Counter[Tuple[str, ...]]] = {n: Counter() for n in self.ngram_orders}
        # Captures the (dataset → domain) edge that aggregate counters
        # lose. Used to label per-dataset bundles. Datasets that mix
        # multiple domains in a single shard collapse to "_mixed".
        dataset_to_domain: Dict[str, str] = {}

        # TF-IDF accumulators: term -> bucket -> count, plus per-bucket totals.
        keyword_buckets: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)

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

                for order, counter in ngram_freqs.items():
                    if n >= order:
                        counter.update(tuple(words[i : i + order]) for i in range(n - order + 1))

                if self.compute_keywords:
                    pipeline = self._ensure_classla()
                    if pipeline is not None:
                        try:
                            lemmas = self._lemmatize(pipeline, doc.text)
                        except Exception as exc:  # noqa: BLE001 — classla is opaque here
                            logger.debug("classla failed on doc %s: %s", doc.id, exc)
                        else:
                            keyword_buckets[(domain, dataset)].update(lemmas)

            yield doc

        bundle = self._build_bundle(
            total_docs=total_docs,
            total_words=total_words,
            per_domain_docs=per_domain_docs,
            per_domain_words=per_domain_words,
            per_dataset_docs=per_dataset_docs,
            per_dataset_words=per_dataset_words,
            word_freq=word_freq,
            ngram_freqs=ngram_freqs,
            keyword_buckets=keyword_buckets,
        )
        self._write_bundle(
            bundle,
            rank=rank,
            world_size=world_size,
            dataset_to_domain=dataset_to_domain,
        )

    def _lemmatize(self, pipeline, text: str) -> List[str]:
        """Run *text* through classla and return content-word lemmas.

        Args:
            pipeline: A `classla.Pipeline` instance.
            text: Document text.

        Returns:
            Lowercased lemmas with stopwords and punctuation removed.
            Closed-class POS tags (PUNCT, SYM, NUM, X) are dropped so
            TF-IDF focuses on content words.
        """
        skip_upos = {"PUNCT", "SYM", "NUM", "X", "PART", "DET", "ADP", "CCONJ", "SCONJ", "PRON", "AUX"}
        out: List[str] = []
        doc = pipeline(text)
        for sentence in doc.sentences:
            for word in sentence.words:
                upos = getattr(word, "upos", None)
                if upos in skip_upos:
                    continue
                lemma = (word.lemma or word.text or "").lower().strip()
                if len(lemma) < _MIN_WORD_LEN:
                    continue
                if lemma in self.stopwords:
                    continue
                out.append(lemma)
        return out

    def _build_bundle(
        self,
        *,
        total_docs: int,
        total_words: int,
        per_domain_docs: Counter,
        per_domain_words: Counter,
        per_dataset_docs: Counter,
        per_dataset_words: Counter,
        word_freq: Counter,
        ngram_freqs: Dict[int, Counter],
        keyword_buckets: Dict[Tuple[str, str], Counter],
    ) -> Dict[str, Any]:
        """Assemble the final stats dict in the canonical aggregate.json shape."""
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

        ngrams_out: Dict[str, List[List[Any]]] = {}
        for order, counter in ngram_freqs.items():
            label = {2: "bigram", 3: "trigram"}.get(order, f"{order}gram")
            ngrams_out[f"{label}_top_{self.top_k_ngrams}"] = [
                [" ".join(gram), int(count)] for gram, count in counter.most_common(self.top_k_ngrams)
            ]

        keywords: Dict[str, Dict[str, List[List[Any]]]] = {"by_domain": {}, "by_dataset": {}}
        if keyword_buckets:
            keywords = self._tfidf(keyword_buckets)

        return {
            "total_docs": total_docs,
            "total_words": total_words,
            "by_domain": _distribution(per_domain_docs, per_domain_words),
            "by_dataset": _distribution(per_dataset_docs, per_dataset_words),
            f"word_freq_top_{self.top_k_words}": [
                [tok, int(count)] for tok, count in word_freq.most_common(self.top_k_words)
            ],
            **ngrams_out,
            "keywords": keywords,
        }

    def _tfidf(self, buckets: Dict[Tuple[str, str], Counter]) -> Dict[str, Dict[str, List[List[Any]]]]:
        """Compute top-K TF-IDF keywords per domain and per dataset.

        We fold the (domain, dataset) buckets into two separate views:
        one grouping by domain (summing across datasets) and one
        grouping by dataset. TF-IDF then treats each view as its own
        "corpus" of buckets-as-documents.

        Args:
            buckets: Lemma counters keyed by `(domain, dataset)`.

        Returns:
            Mapping `{"by_domain": {...}, "by_dataset": {...}}`,
            each value a mapping from group key to a list of
            `[lemma, score]` pairs sorted by descending TF-IDF.
        """
        domain_buckets: Dict[str, Counter] = defaultdict(Counter)
        dataset_buckets: Dict[str, Counter] = defaultdict(Counter)
        for (domain, dataset), counter in buckets.items():
            domain_buckets[domain].update(counter)
            dataset_buckets[dataset].update(counter)

        return {
            "by_domain": self._tfidf_for(domain_buckets),
            "by_dataset": self._tfidf_for(dataset_buckets),
        }

    def _tfidf_for(self, buckets: Dict[str, Counter]) -> Dict[str, List[List[Any]]]:
        """Compute top-K TF-IDF keywords for one bucket grouping.

        Args:
            buckets: Mapping bucket name to a lemma counter.

        Returns:
            Mapping bucket name to a list of `[lemma, score]` pairs
            (top `keyword_top_k` entries, descending score).
        """
        if not buckets:
            return {}
        n_buckets = len(buckets)
        document_freq: Counter[str] = Counter()
        for counter in buckets.values():
            for lemma in counter:
                document_freq[lemma] += 1

        result: Dict[str, List[List[Any]]] = {}
        for bucket_name, counter in buckets.items():
            total = sum(counter.values()) or 1
            scored: List[Tuple[str, float]] = []
            for lemma, count in counter.items():
                tf = count / total
                idf = math.log((1 + n_buckets) / (1 + document_freq[lemma])) + 1.0
                scored.append((lemma, tf * idf))
            scored.sort(key=lambda item: item[1], reverse=True)
            result[bucket_name] = [[lemma, round(score, 6)] for lemma, score in scored[: self.keyword_top_k]]
        return result

    def _write_bundle(
        self,
        bundle: Dict[str, Any],
        *,
        rank: int,
        world_size: int,
        dataset_to_domain: Optional[Dict[str, str]] = None,
    ) -> None:
        """Write the JSON bundle to disk under `output_path`.

        When `per_dataset_dir` was configured, also writes one
        per-dataset JSON next to the aggregate. Each per-dataset bundle
        carries `doc_count` / `word_count` for that dataset, the
        domain it belongs to, and its share of the corpus word count.

        Args:
            bundle: The aggregate stats dict produced by `_build_bundle`.
            rank: Worker rank; appended to the filename when
                `world_size > 1` to avoid clobbering across shards.
            world_size: Total number of workers.
            dataset_to_domain: Mapping from dataset key to domain label,
                captured during `run`.
        """
        path = self.output_path
        if world_size > 1:
            path = path.with_name(f"{path.stem}.{rank:05d}{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(bundle, fh, ensure_ascii=False, indent=2)
        logger.info("Wrote aggregate stats bundle to %s", path)

        if self.per_dataset_dir is None or world_size > 1:
            return
        self.per_dataset_dir.mkdir(parents=True, exist_ok=True)
        by_dataset = bundle.get("by_dataset", {})
        domain_lookup = dataset_to_domain or {}
        for dataset_key, dataset_stats in by_dataset.items():
            slim = {
                "dataset": dataset_key,
                "domain": domain_lookup.get(dataset_key),
                "doc_count": dataset_stats["doc_count"],
                "word_count": dataset_stats["word_count"],
                "avg_doc_words": dataset_stats["avg_doc_words"],
                "share_of_total_words": dataset_stats["share_of_total_words"],
            }
            with (self.per_dataset_dir / f"{dataset_key}.json").open("w", encoding="utf-8") as fh:
                json.dump(slim, fh, ensure_ascii=False, indent=2)
        logger.info(
            "Wrote %d per-dataset stats files under %s",
            len(by_dataset), self.per_dataset_dir,
        )
