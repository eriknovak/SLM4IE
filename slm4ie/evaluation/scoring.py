"""Mention alignment, span F1 scoring and document bootstrap intervals for zero-shot NER.

A model answers with mention strings and labels. Each mention is first aligned
to a character span of the text, in reading order; a mention with no free
occurrence left in the text stays unaligned and counts as a false positive.

A document is then scored by comparing its predicted spans with its gold spans,
both in the task schema's `{"start", "end", "label"}` shape. Three match rules
are counted side by side:

- strict: a predicted span is correct when its start, end and label all equal
  a gold span's;
- relaxed: a predicted span is correct when it contains a gold span of the
  same label, starting at or before it and ending at or after it;
- partial overlap: a predicted span is correct when it shares at least one
  character with a gold span of the same label.

Under the relaxed and partial-overlap rules each predicted and each gold span
is matched at most once, and the matching pairs as many spans as possible.

Per-document counts pool into micro F1. Intervals redraw whole documents with
replacement, since spans in one document are not independent; the paired
variant redraws the same documents for two systems and keeps the difference.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ===============================
# Defaults
# ===============================

N_RESAMPLES = 10_000
SEED = 0
CONFIDENCE = 0.95
RESAMPLE_CHUNK = 1_000

# ===============================
# Result types
# ===============================


@dataclass(frozen=True)
class Counts:
    """Span counts behind one F1 score.

    Attributes:
        true_positives: Predicted spans that match a gold span.
        predicted: Predicted spans.
        gold: Gold spans.
    """

    true_positives: int = 0
    predicted: int = 0
    gold: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        """Pool two counts field by field."""
        return Counts(
            self.true_positives + other.true_positives,
            self.predicted + other.predicted,
            self.gold + other.gold,
        )

    @property
    def precision(self) -> float:
        """Share of predicted spans that are correct; 0 when nothing is predicted."""
        return self.true_positives / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        """Share of gold spans that are found; 0 when there is no gold span."""
        return self.true_positives / self.gold if self.gold else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall, `2 tp / (predicted + gold)`; 0 when both are 0."""
        return _f1(self.true_positives, self.predicted, self.gold)


@dataclass(frozen=True)
class DocumentScore:
    """Counts for one document under the three match rules.

    Attributes:
        strict: Counts under exact start, end and label.
        relaxed: Counts under a predicted span containing a gold span of the same label.
        partial: Counts under one shared character and the same label.
        by_label: Strict counts per label seen in the prediction or the gold.
    """

    strict: Counts
    relaxed: Counts
    partial: Counts
    by_label: dict[str, Counts] = field(default_factory=dict)


@dataclass(frozen=True)
class Interval:
    """A point estimate with its bootstrap interval.

    Attributes:
        estimate: The statistic on the full set of documents.
        low: Lower percentile bound over the resamples.
        high: Upper percentile bound over the resamples.
    """

    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class Alignment:
    """Returned mentions turned into character spans.

    Attributes:
        spans: Aligned spans, each `{"start": int, "end": int, "label": str}`, in answer order.
        unaligned: Mentions whose exact text has no free occurrence left in the text.
    """

    spans: list[dict[str, Any]]
    unaligned: list[dict[str, Any]]


# ===============================
# Alignment
# ===============================


def align_mentions(text: str, mentions: Iterable[Mapping[str, Any]]) -> Alignment:
    """Map returned mention strings to character spans in reading order.

    Each mention takes the first free occurrence of its exact text that
    starts at or after the end of the previous match, so a repeated mention
    moves on to the next occurrence. When none is left there, it falls back to
    the first free occurrence from the start of the text, so an answer out of
    reading order still aligns. An occurrence is free when it overlaps no
    span already taken.

    Args:
        text: The text the model read.
        mentions: Returned mentions, each `{"text": str, "label": str}`, in answer order.

    Returns:
        The aligned spans and the mentions left unaligned.
    """
    spans, unaligned, taken = [], [], []
    cursor = 0
    for item in mentions:
        surface = str(item["text"])
        start = _find_free(text, surface, cursor, taken)
        if start == -1:
            start = _find_free(text, surface, 0, taken)
        if start == -1:
            unaligned.append(dict(item))
            continue

        cursor = start + len(surface)
        taken.append((start, cursor))
        spans.append({"start": start, "end": cursor, "label": str(item["label"])})
    return Alignment(spans=spans, unaligned=unaligned)


# ===============================
# Scoring
# ===============================


def score_document(
    predicted: Iterable[Mapping[str, Any]],
    gold: Iterable[Mapping[str, Any]],
    unaligned: Iterable[Mapping[str, Any]] = (),
) -> DocumentScore:
    """Count strict, relaxed, partial-overlap and per-label span matches for one document.

    A span repeated with the same start, end and label counts once. Each
    unaligned mention adds one predicted span that matches nothing.

    Args:
        predicted: Predicted spans, each `{"start": int, "end": int, "label": str}`
            with an exclusive end.
        gold: Gold spans in the same shape.
        unaligned: Returned mentions with no span, each carrying a `label`.

    Returns:
        The document's counts under the three match rules and per label.
    """
    predicted_set = {_as_tuple(s) for s in predicted}
    gold_set = {_as_tuple(s) for s in gold}
    unaligned_labels = [str(m["label"]) for m in unaligned]
    hits = predicted_set & gold_set

    # strict counts per label
    by_label = {}
    for label in sorted({s[2] for s in predicted_set | gold_set} | set(unaligned_labels)):
        by_label[label] = Counts(
            sum(1 for s in hits if s[2] == label),
            sum(1 for s in predicted_set if s[2] == label) + unaligned_labels.count(label),
            sum(1 for s in gold_set if s[2] == label),
        )

    n_predicted, n_gold = len(predicted_set) + len(unaligned_labels), len(gold_set)
    return DocumentScore(
        strict=Counts(len(hits), n_predicted, n_gold),
        relaxed=Counts(_max_matching(predicted_set, gold_set, _contains), n_predicted, n_gold),
        partial=Counts(_max_matching(predicted_set, gold_set, _overlaps), n_predicted, n_gold),
        by_label=by_label,
    )


def macro_f1(scores: Iterable[DocumentScore]) -> float:
    """Average strict F1 over labels, each label's counts pooled over documents.

    Args:
        scores: One score per document of the test set.

    Returns:
        The mean of the per-label F1 over every label seen; 0 when none is.
    """
    pooled = defaultdict(Counts)
    for score in scores:
        for label, counts in score.by_label.items():
            pooled[label] += counts
    return sum(c.f1 for c in pooled.values()) / len(pooled) if pooled else 0.0


# ===============================
# Bootstrap intervals
# ===============================


def f1_interval(
    counts: Sequence[Counts],
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    confidence: float = CONFIDENCE,
) -> Interval:
    """Bootstrap the micro F1 of one system by redrawing whole documents.

    Args:
        counts: One count per document.
        n_resamples: Number of redrawn test sets.
        seed: Seed of the random generator.
        confidence: Coverage of the percentile interval.

    Returns:
        The pooled F1 with its percentile interval.

    Raises:
        ValueError: If `counts` is empty.
    """
    if not counts:
        raise ValueError("cannot bootstrap a test set with no documents")

    matrix = _as_matrix(counts)
    values = _f1_rows(_resample_sums(matrix, n_resamples, seed))
    return _interval(_f1_rows(matrix.sum(axis=0, keepdims=True))[0], values, confidence)


def paired_difference_interval(
    counts_a: Mapping[str, Counts],
    counts_b: Mapping[str, Counts],
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    confidence: float = CONFIDENCE,
) -> Interval:
    """Bootstrap the micro F1 of system A minus system B on the same documents.

    Each resample redraws one set of documents and scores both systems on it,
    so the interval holds only the variance in how the systems differ.

    Args:
        counts_a: Counts of system A, keyed by document id.
        counts_b: Counts of system B, keyed by the same document ids.
        n_resamples: Number of redrawn test sets.
        seed: Seed of the random generator.
        confidence: Coverage of the percentile interval.

    Returns:
        F1 of A minus F1 of B with its percentile interval.

    Raises:
        ValueError: If the two systems were scored on different documents, or on none.
    """
    if counts_a.keys() != counts_b.keys():
        unmatched = sorted(counts_a.keys() ^ counts_b.keys())
        raise ValueError(f"the systems were scored on different documents; unmatched ids: {unmatched[:5]}")
    if not counts_a:
        raise ValueError("cannot bootstrap a test set with no documents")

    # stack both systems so one redraw serves both
    ids = sorted(counts_a)
    matrix = np.hstack([_as_matrix([counts_a[i] for i in ids]), _as_matrix([counts_b[i] for i in ids])])
    sums = _resample_sums(matrix, n_resamples, seed)
    values = _f1_rows(sums[:, :3]) - _f1_rows(sums[:, 3:])
    total = matrix.sum(axis=0, keepdims=True)
    return _interval((_f1_rows(total[:, :3]) - _f1_rows(total[:, 3:]))[0], values, confidence)


# ===============================
# Helpers
# ===============================


def _find_free(text: str, surface: str, origin: int, taken: list[tuple[int, int]]) -> int:
    """Return the first start at or after `origin` where `surface` overlaps no taken span, or -1."""
    if not surface:
        return -1
    start = text.find(surface, origin)
    while start != -1 and any(start < end and begin < start + len(surface) for begin, end in taken):
        start = text.find(surface, start + 1)
    return start


def _as_tuple(span: Mapping[str, Any]) -> tuple[int, int, str]:
    """Turn a span dict into a hashable `(start, end, label)` tuple."""
    return (int(span["start"]), int(span["end"]), str(span["label"]))


def _contains(predicted: tuple[int, int, str], gold: tuple[int, int, str]) -> bool:
    """Tell whether the predicted span covers the whole gold span with the same label."""
    return predicted[2] == gold[2] and predicted[0] <= gold[0] and gold[1] <= predicted[1]


def _overlaps(predicted: tuple[int, int, str], gold: tuple[int, int, str]) -> bool:
    """Tell whether the two spans share a character and a label."""
    return predicted[2] == gold[2] and predicted[0] < gold[1] and gold[0] < predicted[1]


def _max_matching(
    predicted: set[tuple[int, int, str]],
    gold: set[tuple[int, int, str]],
    pairs: Callable[[tuple[int, int, str], tuple[int, int, str]], bool],
) -> int:
    """Count the most one-to-one pairs of predicted and gold spans that `pairs` accepts.

    Kuhn's augmenting-path algorithm: each predicted span takes a free gold
    span it pairs with, or moves an earlier prediction to another gold span.
    """
    candidates = {p: [g for g in gold if pairs(p, g)] for p in sorted(predicted)}
    owner = {}

    def augment(p: tuple[int, int, str], seen: set[tuple[int, int, str]]) -> bool:
        for g in candidates[p]:
            if g in seen:
                continue
            seen.add(g)
            if g not in owner or augment(owner[g], seen):
                owner[g] = p
                return True
        return False

    return sum(augment(p, set()) for p in candidates)


def _f1(true_positives: float, predicted: float, gold: float) -> float:
    """Return `2 tp / (predicted + gold)`, or 0 when the denominator is 0."""
    denominator = predicted + gold
    return 2 * true_positives / denominator if denominator else 0.0


def _as_matrix(counts: Sequence[Counts]) -> np.ndarray:
    """Stack counts into an integer matrix of rows `(tp, predicted, gold)`."""
    return np.array([(c.true_positives, c.predicted, c.gold) for c in counts], dtype=np.int64)


def _f1_rows(sums: np.ndarray) -> np.ndarray:
    """Compute F1 for every row `(tp, predicted, gold)` of a summed-count matrix."""
    denominator = sums[:, 1] + sums[:, 2]
    return np.divide(2 * sums[:, 0], denominator, out=np.zeros(len(sums)), where=denominator > 0)


def _resample_sums(matrix: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """Sum the count columns over documents redrawn with replacement, once per resample.

    A redraw of n documents is a multinomial count of how often each document
    is drawn; resamples are drawn in chunks to bound memory.
    """
    rng = np.random.default_rng(seed)
    n_documents = len(matrix)
    probabilities = np.full(n_documents, 1 / n_documents)
    chunks = []
    for first in range(0, n_resamples, RESAMPLE_CHUNK):
        size = min(RESAMPLE_CHUNK, n_resamples - first)
        weights = rng.multinomial(n_documents, probabilities, size=size)
        chunks.append(weights @ matrix)
    return np.vstack(chunks)


def _interval(estimate: float, values: np.ndarray, confidence: float) -> Interval:
    """Wrap an estimate with the central percentile interval of its resampled values."""
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return Interval(float(estimate), float(low), float(high))
