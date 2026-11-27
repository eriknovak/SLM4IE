"""Scores for the task registry: spans, tags, labels, choices, relations, answer strings and keyphrases.

Each task family has one public scorer, found by `scorer_for(family)`, that
takes the examples of one split and one prediction per example, aligned by
position, and returns named scores in [0, 1]. Any approach (a fine-tuned head,
a zero-shot backend) is scored the same way once it answers in the family's
prediction shape: a span list (`score_spans`), a tag list (`score_tags`), a
label (`score_labels`), a choice index (`score_choices`), a relation list
(`score_relations`), an answer string (`score_extractive_qa`) or a ranked
keyphrase list (`score_keyphrases`).

Answer strings and keyphrases are compared after one normalisation,
`normalise_text`, as the SQuAD 2.0 script does it but without its English article removal: lowercase,
drop every Unicode punctuation and symbol character, collapse whitespace.

Token accuracy scores part-of-speech tagging, pooled over every word of every
sentence. Accuracy and per-class macro F1 score the classification families.
The rest of this module aligns and scores NER mentions, as follows.

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

import functools
import inspect
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np

# ===============================
# Defaults
# ===============================

N_RESAMPLES = 10_000
SEED = 0
CONFIDENCE = 0.95
RESAMPLE_CHUNK = 1_000

# ranks cut off for keyphrase F1@k, beside F1@M over every prediction
KEYPHRASE_CUTOFFS = (5, 10)

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
# Token and classification scoring
# ===============================


def token_accuracy(predicted: Sequence[Sequence[Any]], gold: Sequence[Sequence[Any]]) -> float:
    """Share of tokens whose predicted tag equals the gold tag, pooled over sentences (POS).

    Args:
        predicted: One tag sequence per sentence.
        gold: Gold tag sequences, one per sentence, each as long as its prediction.

    Returns:
        The pooled token accuracy; 0 when there are no tokens.

    Raises:
        ValueError: If the sentence counts differ, or a sentence's tag counts differ.
    """
    if len(predicted) != len(gold):
        raise ValueError(f"{len(predicted)} predicted sentences but {len(gold)} gold sentences")

    correct = total = 0
    for index, (tags, gold_tags) in enumerate(zip(predicted, gold)):
        if len(tags) != len(gold_tags):
            raise ValueError(f"sentence {index} has {len(tags)} predicted tags but {len(gold_tags)} gold tags")
        correct += sum(a == b for a, b in zip(tags, gold_tags))
        total += len(gold_tags)
    return correct / total if total else 0.0


def accuracy(predicted: Sequence[Any], gold: Sequence[Any]) -> float:
    """Share of examples whose predicted label equals the gold label.

    Args:
        predicted: One label per example.
        gold: Gold labels, one per example.

    Returns:
        The accuracy; 0 when there are no examples.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    _check_aligned(gold, predicted)
    return sum(a == b for a, b in zip(predicted, gold)) / len(gold) if gold else 0.0


def classification_macro_f1(predicted: Sequence[Any], gold: Sequence[Any]) -> float:
    """Average per-class F1 over every class in the gold labels or the predictions.

    A class that is gold but never predicted scores 0.

    Args:
        predicted: One label per example.
        gold: Gold labels, one per example.

    Returns:
        The mean per-class F1; 0 when there are no examples.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    _check_aligned(gold, predicted)
    classes = set(predicted) | set(gold)
    scores = [
        _f1(
            sum(a == b == label for a, b in zip(predicted, gold)),
            sum(a == label for a in predicted),
            sum(b == label for b in gold),
        )
        for label in classes
    ]
    return sum(scores) / len(scores) if scores else 0.0


# ===============================
# Text normalisation
# ===============================


def normalise_text(text: str) -> str:
    """Normalise an answer string or keyphrase the way the QA and keyphrase scorers compare them.

    Args:
        text: The string.

    Returns:
        The string lowercased, without Unicode punctuation and symbol
        characters, its whitespace collapsed to single spaces.
    """
    kept = "".join(char for char in text.lower() if unicodedata.category(char)[0] not in "PS")
    return " ".join(kept.split())


# ===============================
# Family scorers
# ===============================

Scorer = Callable[..., dict[str, float]]


def score_spans(examples: Sequence[Mapping[str, Any]], predictions: Sequence[Any]) -> dict[str, float]:
    """Score one span list per example against its gold spans (`ner`).

    Args:
        examples: NER examples, each with an `id` and gold `spans`.
        predictions: One list of `{"start": int, "end": int, "label": str}` spans per example.

    Returns:
        `span_f1`, `span_relaxed_f1` and `span_partial_f1`, the micro F1 pooled
        over examples under the strict, relaxed and partial-overlap match
        rules, and `span_macro_f1`, the strict F1 averaged over labels.

    Raises:
        ValueError: If the counts differ, or a prediction is missing or not a span list.
    """
    _check_aligned(examples, predictions)
    scores = []
    for example, spans in zip(examples, predictions):
        if not _is_list(spans) or not all(_is_span(item) for item in spans):
            raise ValueError(f"example {example['id']!r} has prediction {spans!r}, not a list of spans")
        scores.append(score_document(spans, example["spans"]))

    return {
        "span_f1": sum((score.strict for score in scores), Counts()).f1,
        "span_relaxed_f1": sum((score.relaxed for score in scores), Counts()).f1,
        "span_partial_f1": sum((score.partial for score in scores), Counts()).f1,
        "span_macro_f1": macro_f1(scores),
    }


def score_tags(examples: Sequence[Mapping[str, Any]], predictions: Sequence[Any]) -> dict[str, float]:
    """Score one tag list per example, one tag per word (`pos`).

    A word tagged `None`, one the model gave no tag, is never right.

    Args:
        examples: POS examples, each with an `id` and gold `tags`.
        predictions: One list of tags per example, as long as its `tags`.

    Returns:
        `token_accuracy`, pooled over every word of every example.

    Raises:
        ValueError: If the counts differ, or a prediction is missing or not one tag per word.
    """
    _check_aligned(examples, predictions)
    for example, tags in zip(examples, predictions):
        if not _is_list(tags) or len(tags) != len(example["tags"]):
            raise ValueError(
                f"example {example['id']!r} has prediction {tags!r}, not a list of {len(example['tags'])} tags"
            )
    return {"token_accuracy": token_accuracy(predictions, [example["tags"] for example in examples])}


def score_labels(examples: Sequence[Mapping[str, Any]], predictions: Sequence[Any]) -> dict[str, float]:
    """Score one label per example (the classification families).

    Args:
        examples: Examples, each with an `id` and a gold `label`.
        predictions: One label per example, a string, boolean or integer.

    Returns:
        `accuracy` and `macro_f1` over the labels.

    Raises:
        ValueError: If the counts differ, or a prediction is missing or not a single label.
    """
    _check_aligned(examples, predictions)
    for example, label in zip(examples, predictions):
        if not isinstance(label, str | Integral):
            raise ValueError(f"example {example['id']!r} has prediction {label!r}, not a single label")
    gold = [example["label"] for example in examples]
    return {"accuracy": accuracy(predictions, gold), "macro_f1": classification_macro_f1(predictions, gold)}


def score_choices(examples: Sequence[Mapping[str, Any]], predictions: Sequence[Any]) -> dict[str, float]:
    """Score one choice index per example (`commonsense`, `reading_mc`).

    An example's choices are its `choices` list, or else its `choice1`,
    `choice2`, ... fields.

    Args:
        examples: Examples, each with an `id`, its choices and the gold choice index as `label`.
        predictions: One choice index per example.

    Returns:
        `accuracy` and `macro_f1` over the choice indices.

    Raises:
        ValueError: If the counts differ, or a prediction is missing or not an index among its choices.
    """
    _check_aligned(examples, predictions)
    predicted = []
    for example, index in zip(examples, predictions):
        n_choices = (
            len(example["choices"]) if "choices" in example else sum(key.startswith("choice") for key in example)
        )
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= int(index) < n_choices:
            raise ValueError(f"example {example['id']!r} has prediction {index!r}, not one of {n_choices} choices")
        predicted.append(int(index))
    gold = [int(example["label"]) for example in examples]
    return {"accuracy": accuracy(predicted, gold), "macro_f1": classification_macro_f1(predicted, gold)}


def score_relations(
    examples: Sequence[Mapping[str, Any]], predictions: Sequence[Any], *, gold_entities: bool
) -> dict[str, float]:
    """Score one relation list per example against its gold relations (`relation`).

    A predicted relation is strictly right when its head span, tail span (each
    start, end and entity label) and relation label all equal a gold
    relation's. A relation repeated in one prediction counts once.

    Args:
        examples: Relation examples, each with an `id`, gold `entities` and gold `relations`.
        predictions: One list of `{"head": span, "tail": span, "label": str}`
            relations per example, each span `{"start": int, "end": int, "label": str}`.
        gold_entities: True when the approach labels pairs of the gold mentions,
            so every predicted span must be a gold mention of its example;
            False when it finds the mentions itself (end to end).

    Returns:
        `relation_f1`, the strict micro F1 pooled over examples, and
        `relation_macro_f1`, the strict F1 averaged over relation labels. End to
        end adds `relation_boundary_f1`, which ignores the entity labels.

    Raises:
        ValueError: If the counts differ, a prediction is missing or not a
            relation list, or, with gold entities, a predicted span is no gold
            mention of its example.
    """
    _check_aligned(examples, predictions)
    strict, boundary, by_label = Counts(), Counts(), defaultdict(Counts)
    for example, relations in zip(examples, predictions):
        # check the prediction's shape and, with gold entities, its spans
        if not _is_list(relations) or not all(_is_relation(item) for item in relations):
            raise ValueError(f"example {example['id']!r} has prediction {relations!r}, not a list of relations")
        mentions = {mention["id"]: _as_tuple(mention) for mention in example["entities"]}
        if gold_entities:
            _check_gold_arguments(example["id"], relations, set(mentions.values()))

        # count strict, boundary and per-label matches
        predicted = {(_as_tuple(item["head"]), _as_tuple(item["tail"]), item["label"]) for item in relations}
        gold = {(mentions[link["head"]], mentions[link["tail"]], link["label"]) for link in example["relations"]}
        strict += _set_counts(predicted, gold)
        boundary += _set_counts({_unlabelled(r) for r in predicted}, {_unlabelled(r) for r in gold})
        for label in {r[2] for r in predicted | gold}:
            by_label[label] += _set_counts({r for r in predicted if r[2] == label}, {r for r in gold if r[2] == label})

    scores = {
        "relation_f1": strict.f1,
        "relation_macro_f1": _mean([counts.f1 for counts in by_label.values()]),
    }
    if not gold_entities:
        scores["relation_boundary_f1"] = boundary.f1
    return scores


def score_extractive_qa(examples: Sequence[Mapping[str, Any]], predictions: Sequence[Any]) -> dict[str, float]:
    """Score one answer string per example as SQuAD 2.0 does (`extractive_qa`).

    Each answer scores exact match and token-overlap F1 against its
    best-matching gold answer, after normalisation. An unanswerable question,
    one with no gold answer, scores 1 only for the empty answer. A kind of
    question absent from the split scores 0 on its split keys, as `accuracy`
    does on no examples, so every key is always present and in [0, 1].

    Args:
        examples: Extractive QA examples, each with an `id` and gold `answers`.
        predictions: One answer string per example; the empty string means no answer.

    Returns:
        `exact_match` and `f1` averaged over examples, and the same two scores
        over the answerable questions (`has_answer_exact_match`,
        `has_answer_f1`) and the unanswerable ones (`no_answer_exact_match`,
        `no_answer_f1`).

    Raises:
        ValueError: If the counts differ, or a prediction is missing or not a string.
    """
    _check_aligned(examples, predictions)
    exact, overlap, answerable = [], [], []
    for example, answer in zip(examples, predictions):
        if not isinstance(answer, str):
            raise ValueError(f"example {example['id']!r} has prediction {answer!r}, not an answer string")
        # gold answers with no word left after normalisation are dropped, as in SQuAD 2.0
        golds = [text for text in (normalise_text(gold["text"]) for gold in example["answers"]) if text] or [""]
        predicted = normalise_text(answer)
        exact.append(max(float(predicted == gold) for gold in golds))
        overlap.append(max(_token_f1(predicted, gold) for gold in golds))
        answerable.append(bool(example["answers"]))

    scores = {"exact_match": _mean(exact), "f1": _mean(overlap)}
    for kind, is_answerable in (("has_answer", True), ("no_answer", False)):
        scores[f"{kind}_exact_match"] = _mean([s for s, a in zip(exact, answerable) if a == is_answerable])
        scores[f"{kind}_f1"] = _mean([s for s, a in zip(overlap, answerable) if a == is_answerable])
    return scores


def score_keyphrases(examples: Sequence[Mapping[str, Any]], predictions: Sequence[Any]) -> dict[str, float]:
    """Score one ranked keyphrase list per example (`keyphrase`).

    Predicted and gold keyphrases are normalised, and a keyphrase repeated
    after normalisation counts once, at its first rank; one with no word left
    is dropped. F1@k takes the top k predictions and counts missing ranks as
    wrong (Chan et al., 2019); F1@M takes every prediction. Each is averaged
    over examples.

    Args:
        examples: Keyphrase examples, each with an `id`, its `text` and gold `keyphrases`.
        predictions: One list of keyphrase strings per example, best first.

    Returns:
        `f1_at_5`, `f1_at_10` and `f1_at_m`, and `gold_in_text`, the share of
        gold keyphrases, pooled over examples, that occur as whole words of
        the normalised text: the most a tagging approach can find.

    Raises:
        ValueError: If the counts differ, or a prediction is missing or not a list of strings.
    """
    _check_aligned(examples, predictions)
    by_cutoff = defaultdict(list)
    in_text = n_gold = 0
    for example, phrases in zip(examples, predictions):
        if not _is_list(phrases) or not all(isinstance(phrase, str) for phrase in phrases):
            raise ValueError(f"example {example['id']!r} has prediction {phrases!r}, not a list of keyphrases")
        ranked = _unique_normalised(phrases)
        gold = set(_unique_normalised(example["keyphrases"]))

        # score the top k and every prediction
        for cutoff in KEYPHRASE_CUTOFFS:
            by_cutoff[f"f1_at_{cutoff}"].append(_f1(len(gold.intersection(ranked[:cutoff])), cutoff, len(gold)))
        by_cutoff["f1_at_m"].append(_f1(len(gold.intersection(ranked)), len(ranked), len(gold)))

        # count gold keyphrases reachable in the text
        text = f" {normalise_text(example['text'])} "
        in_text += sum(f" {phrase} " in text for phrase in gold)
        n_gold += len(gold)

    scores = {key: _mean(values) for key, values in by_cutoff.items()}
    return scores | {"gold_in_text": in_text / n_gold if n_gold else 0.0}


# the scorer of each task family, by the shape its predictions take
FAMILY_SCORERS: dict[str, Scorer] = {
    "ner": score_spans,
    "pos": score_tags,
    **dict.fromkeys(("sentiment", "nli", "qa", "coref", "wsd", "entity_sentiment", "topic"), score_labels),
    **dict.fromkeys(("commonsense", "reading_mc"), score_choices),
    "relation": score_relations,
    "extractive_qa": score_extractive_qa,
    "keyphrase": score_keyphrases,
}


def scorer_for(family: str, **options: Any) -> Scorer:
    """Return the scorer of a task family.

    Args:
        family: The task family, the first part of a registry key.
        **options: Keyword arguments bound into the scorer, for a scorer that
            takes a setting beside the examples and predictions, such as
            `gold_entities` for `relation`.

    Returns:
        A function `(examples, predictions) -> dict[str, float]`; the family's
        scorer itself when no option is given.

    Raises:
        KeyError: If the family has no scorer.
        ValueError: If the family's scorer needs a setting that `options` does not give.
    """
    if family not in FAMILY_SCORERS:
        raise KeyError(f"task family '{family}' has no scorer; families with one: {sorted(FAMILY_SCORERS)}")
    scorer = FAMILY_SCORERS[family]

    # a keyword-only parameter with no default is a setting the caller must give
    required = [
        name
        for name, parameter in inspect.signature(scorer).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default is inspect.Parameter.empty
    ]
    missing = [name for name in required if name not in options]
    if missing:
        raise ValueError(f"the scorer of task family '{family}' needs the option(s) {missing}; got {sorted(options)}")
    return functools.partial(scorer, **options) if options else scorer


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


def _check_aligned(examples: Sequence[Any], predictions: Sequence[Any]) -> None:
    """Raise when there is not exactly one prediction per example."""
    if len(predictions) != len(examples):
        raise ValueError(f"{len(predictions)} predictions for {len(examples)} examples")


def _is_list(value: Any) -> bool:
    """Tell whether a prediction is a sequence of items, not a string or a mapping."""
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _is_span(value: Any) -> bool:
    """Tell whether a value is a `{"start": int, "end": int, "label": str}` span."""
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("start"), Integral)
        and isinstance(value.get("end"), Integral)
        and isinstance(value.get("label"), str)
    )


def _is_relation(value: Any) -> bool:
    """Tell whether a value is a `{"head": span, "tail": span, "label": str}` relation."""
    return (
        isinstance(value, Mapping)
        and _is_span(value.get("head"))
        and _is_span(value.get("tail"))
        and isinstance(value.get("label"), str)
    )


def _check_gold_arguments(
    example_id: str, relations: Sequence[Mapping[str, Any]], mentions: set[tuple[int, int, str]]
) -> None:
    """Raise when a predicted relation argument is no gold mention of its example."""
    for item in relations:
        for argument in (item["head"], item["tail"]):
            if _as_tuple(argument) not in mentions:
                raise ValueError(
                    f"example {example_id!r} has predicted span {argument!r}, not one of its gold mentions"
                )


def _unlabelled(relation: tuple[tuple[int, int, str], tuple[int, int, str], str]) -> tuple[int, int, int, int, str]:
    """Drop the entity labels of a relation tuple, keeping both spans' offsets and the relation label."""
    (head_start, head_end, _), (tail_start, tail_end, _), label = relation
    return (head_start, head_end, tail_start, tail_end, label)


def _set_counts(predicted: set[Any], gold: set[Any]) -> Counts:
    """Count exact matches between a predicted and a gold set."""
    return Counts(len(predicted & gold), len(predicted), len(gold))


def _unique_normalised(phrases: Iterable[str]) -> list[str]:
    """Normalise phrases, keep the first of each, and drop those with no word left."""
    return [phrase for phrase in dict.fromkeys(normalise_text(p) for p in phrases) if phrase]


def _token_f1(predicted: str, gold: str) -> float:
    """Token-overlap F1 of two normalised strings; 1 when both are empty, 0 when only one is."""
    predicted_tokens, gold_tokens = predicted.split(), gold.split()
    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)
    common = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
    return _f1(common, len(predicted_tokens), len(gold_tokens))


def _mean(values: Sequence[float]) -> float:
    """Average a list of scores; 0 when it is empty."""
    return sum(values) / len(values) if values else 0.0


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
