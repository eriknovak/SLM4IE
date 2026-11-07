"""Tests for slm4ie.evaluation.scoring."""

import pytest

from slm4ie.evaluation.scoring import (
    Alignment,
    Counts,
    align_mentions,
    f1_interval,
    macro_f1,
    paired_difference_interval,
    score_document,
)

# ===============================
# Test Data
# ===============================


def span(start: int, end: int, label: str) -> dict:
    """Build one span dict in the task schema's shape."""
    return {"start": start, "end": end, "label": label}


def mention(text: str, label: str) -> dict:
    """Build one returned mention in the answer schema's shape."""
    return {"text": text, "label": label}


GOLD = [span(0, 5, "PER"), span(10, 20, "LOC")]
TEXT = "Janez Novak je iz Kranja. Novak živi v Kranju, Janez pa ne."

# ===============================
# Counts
# ===============================


def test_counts_f1_is_twice_true_positives_over_predicted_plus_gold():
    """F1, precision and recall follow from the three counts."""
    counts = Counts(true_positives=3, predicted=4, gold=6)
    assert counts.f1 == pytest.approx(0.6)
    assert counts.precision == pytest.approx(0.75)
    assert counts.recall == pytest.approx(0.5)


def test_counts_with_nothing_predicted_or_gold_score_zero():
    """Empty counts score 0 instead of dividing by zero."""
    assert Counts().f1 == 0.0
    assert Counts().precision == 0.0
    assert Counts().recall == 0.0


def test_counts_add_field_by_field():
    """Adding counts pools each field, so `sum` works over documents."""
    assert Counts(1, 2, 3) + Counts(4, 5, 6) == Counts(5, 7, 9)
    assert sum([Counts(1, 1, 1), Counts(1, 2, 3)], Counts()) == Counts(2, 3, 4)


# ===============================
# align_mentions
# ===============================


def test_mentions_in_reading_order_map_to_their_occurrences():
    """Each mention takes the occurrence of its text where reading reached."""
    alignment = align_mentions(TEXT, [mention("Janez Novak", "PER"), mention("Kranja", "LOC")])
    assert alignment == Alignment(spans=[span(0, 11, "PER"), span(18, 24, "LOC")], unaligned=[])


def test_repeated_mention_takes_the_next_occurrence():
    """A mention text returned twice maps to its first and second occurrence."""
    alignment = align_mentions(TEXT, [mention("Novak", "PER"), mention("Novak", "PER")])
    assert alignment.spans == [span(6, 11, "PER"), span(26, 31, "PER")]


def test_surname_after_the_full_name_takes_its_next_occurrence():
    """A surname returned after the full name skips the copy inside the full name."""
    alignment = align_mentions(TEXT, [mention("Janez Novak", "PER"), mention("Novak", "PER")])
    assert alignment.spans == [span(0, 11, "PER"), span(26, 31, "PER")]


def test_mention_absent_from_the_text_is_unaligned():
    """A mention the text does not hold has no span."""
    alignment = align_mentions(TEXT, [mention("Ljubljana", "LOC")])
    assert alignment == Alignment(spans=[], unaligned=[mention("Ljubljana", "LOC")])


def test_mention_before_the_previous_match_falls_back_to_its_first_free_copy():
    """An answer out of reading order still aligns, from the start of the text."""
    alignment = align_mentions(TEXT, [mention("Kranju", "LOC"), mention("Kranja", "LOC")])
    assert alignment == Alignment(spans=[span(39, 45, "LOC"), span(18, 24, "LOC")], unaligned=[])


def test_fallback_skips_a_copy_inside_a_taken_span():
    """Searching back, a surname skips the copy inside the full name already taken."""
    mentions = [mention("Janez Novak", "PER"), mention("Kranju", "LOC"), mention("Novak", "PER")]
    assert align_mentions(TEXT, mentions).spans == [span(0, 11, "PER"), span(39, 45, "LOC"), span(26, 31, "PER")]


def test_answer_grouped_by_label_aligns_every_mention():
    """A perfect answer listed by label instead of reading order loses nothing."""
    mentions = [
        mention("Kranja", "LOC"),
        mention("Kranju", "LOC"),
        mention("Janez Novak", "PER"),
        mention("Novak", "PER"),
        mention("Janez", "PER"),
    ]
    alignment = align_mentions(TEXT, mentions)
    assert alignment.unaligned == []
    assert sorted((s["start"], s["end"]) for s in alignment.spans) == [(0, 11), (18, 24), (26, 31), (39, 45), (47, 52)]


def test_mention_returned_more_often_than_it_occurs_is_unaligned():
    """A third copy of a text that occurs twice finds no free occurrence."""
    copies = [mention("Janez", "PER")] * 3
    assert align_mentions(TEXT, copies).unaligned == [mention("Janez", "PER")]


def test_empty_mention_text_is_unaligned():
    """An empty mention string never matches."""
    assert align_mentions(TEXT, [mention("", "PER")]).unaligned == [mention("", "PER")]


def test_alignment_is_case_sensitive():
    """A mention differing from the text in case is not copied text."""
    assert align_mentions(TEXT, [mention("janez novak", "PER")]).spans == []


# ===============================
# score_document
# ===============================


def test_exact_spans_count_as_strict_and_partial_hits():
    """Spans equal to the gold hit under strict and partial overlap."""
    score = score_document(GOLD, GOLD)
    assert score.strict == Counts(2, 2, 2)
    assert score.partial == Counts(2, 2, 2)


def test_boundary_error_misses_strict_but_hits_partial():
    """A span one character short misses strict but hits partial overlap."""
    score = score_document([span(0, 4, "PER")], [span(0, 5, "PER")])
    assert score.strict == Counts(0, 1, 1)
    assert score.partial == Counts(1, 1, 1)


def test_overlap_with_another_label_is_no_partial_hit():
    """Overlap with a different label is no partial hit."""
    score = score_document([span(0, 5, "ORG")], [span(0, 5, "PER")])
    assert score.partial == Counts(0, 1, 1)


def test_touching_spans_share_no_character():
    """Spans that only touch at an end share no character."""
    score = score_document([span(5, 9, "PER")], [span(0, 5, "PER")])
    assert score.partial == Counts(0, 1, 1)


def test_one_prediction_over_two_gold_spans_hits_only_one():
    """One prediction overlapping two gold spans pairs with one of them."""
    score = score_document([span(0, 20, "PER")], [span(0, 5, "PER"), span(10, 20, "PER")])
    assert score.partial == Counts(1, 1, 2)


def test_partial_matching_pairs_every_prediction_it_can():
    """Partial overlap pairs as many spans as possible, not the first overlap."""
    # a greedy match on the first overlap would pair the wide prediction with
    # the first gold span and leave the narrow one unmatched
    predicted = [span(0, 12, "PER"), span(0, 3, "PER")]
    gold = [span(0, 2, "PER"), span(10, 12, "PER")]
    assert score_document(predicted, gold).partial == Counts(2, 2, 2)


def test_exact_span_is_a_relaxed_hit():
    """Spans equal to the gold hit under the relaxed rule."""
    assert score_document(GOLD, GOLD).relaxed == Counts(2, 2, 2)


def test_prediction_containing_the_gold_span_is_a_relaxed_hit():
    """A prediction containing the gold span misses strict but hits relaxed."""
    score = score_document([span(0, 9, "PER")], [span(2, 5, "PER")])
    assert score.strict == Counts(0, 1, 1)
    assert score.relaxed == Counts(1, 1, 1)


def test_prediction_inside_the_gold_span_is_no_relaxed_hit():
    """A prediction inside the gold span misses relaxed but hits partial."""
    score = score_document([span(1, 4, "PER")], [span(0, 5, "PER")])
    assert score.relaxed == Counts(0, 1, 1)
    assert score.partial == Counts(1, 1, 1)


def test_containing_prediction_with_another_label_is_no_relaxed_hit():
    """Containment with a different label is no relaxed hit."""
    score = score_document([span(0, 9, "ORG")], [span(2, 5, "PER")])
    assert score.relaxed == Counts(0, 1, 1)


def test_one_prediction_containing_two_gold_spans_hits_only_one():
    """One prediction containing two gold spans pairs with one of them."""
    score = score_document([span(0, 20, "PER")], [span(0, 5, "PER"), span(10, 20, "PER")])
    assert score.relaxed == Counts(1, 1, 2)


def test_relaxed_matching_pairs_every_prediction_it_can():
    """The relaxed rule pairs as many spans as possible."""
    # the wide prediction contains both gold spans, the narrow one only the
    # first, so the wide one must take the second
    predicted = [span(0, 20, "PER"), span(0, 6, "PER")]
    gold = [span(0, 5, "PER"), span(10, 20, "PER")]
    assert score_document(predicted, gold).relaxed == Counts(2, 2, 2)


def test_unaligned_mention_counts_as_a_false_positive_under_every_rule():
    """A mention with no span adds a predicted span that matches nothing, in its label too."""
    score = score_document([span(0, 5, "PER")], [span(0, 5, "PER")], unaligned=[mention("Ljubljana", "LOC")])
    assert (score.strict, score.relaxed, score.partial) == (Counts(1, 2, 1),) * 3
    assert score.by_label["LOC"] == Counts(0, 1, 0)


def test_repeated_prediction_counts_once():
    """A prediction repeated in one document counts once."""
    score = score_document([span(0, 5, "PER"), span(0, 5, "PER")], [span(0, 5, "PER")])
    assert score.strict == Counts(1, 1, 1)


def test_empty_prediction_counts_every_gold_span_as_missed():
    """An empty answer misses every gold span and scores F1 0."""
    score = score_document([], GOLD)
    assert score.strict == Counts(0, 0, 2)
    assert score.strict.f1 == 0.0


def test_strict_counts_are_kept_per_label():
    """Strict counts are kept for every label in the prediction or the gold."""
    predicted = [span(0, 5, "PER"), span(30, 35, "ORG")]
    score = score_document(predicted, GOLD)
    assert score.by_label == {
        "PER": Counts(1, 1, 1),
        "LOC": Counts(0, 0, 1),
        "ORG": Counts(0, 1, 0),
    }


# ===============================
# macro_f1
# ===============================


def test_macro_f1_weights_every_label_equally():
    """Macro F1 gives a rare label the same weight as a frequent one."""
    # PER: 9 of 9 right; LOC: 0 of 1 right -> macro 0.5, micro 0.9
    per_spans = [span(i * 10, i * 10 + 5, "PER") for i in range(9)]
    scores = [score_document(per_spans, per_spans), score_document([], [span(0, 5, "LOC")])]
    assert macro_f1(scores) == pytest.approx(0.5)


def test_macro_f1_of_no_spans_is_zero():
    """A test set with no spans has macro F1 0."""
    assert macro_f1([score_document([], [])]) == 0.0


# ===============================
# f1_interval
# ===============================


def test_f1_interval_estimate_is_the_pooled_f1():
    """The interval's estimate is the micro F1 pooled over documents."""
    counts = [Counts(1, 2, 2), Counts(3, 3, 4), Counts(0, 1, 1)]
    interval = f1_interval(counts, n_resamples=500, seed=1)
    assert interval.estimate == pytest.approx(sum(counts, Counts()).f1)
    assert interval.low <= interval.estimate <= interval.high


def test_f1_interval_is_reproducible_under_one_seed():
    """The same seed gives the same interval."""
    counts = [Counts(i % 3, 2, 3) for i in range(40)]
    assert f1_interval(counts, n_resamples=300, seed=7) == f1_interval(counts, n_resamples=300, seed=7)


def test_f1_interval_of_identical_documents_has_no_width():
    """Identical documents give an interval of zero width."""
    interval = f1_interval([Counts(1, 2, 2)] * 10, n_resamples=200, seed=0)
    assert interval.low == pytest.approx(0.5)
    assert interval.high == pytest.approx(0.5)


def test_f1_interval_rejects_an_empty_test_set():
    """A test set with no documents raises."""
    with pytest.raises(ValueError, match="no documents"):
        f1_interval([], n_resamples=10, seed=0)


# ===============================
# paired_difference_interval
# ===============================


def test_identical_systems_differ_by_zero():
    """Two identical systems differ by exactly 0 on every draw."""
    counts = {f"d{i}": Counts(i % 2, 1, 1) for i in range(20)}
    interval = paired_difference_interval(counts, counts, n_resamples=300, seed=0)
    assert (interval.estimate, interval.low, interval.high) == (0.0, 0.0, 0.0)


def test_system_better_on_every_document_has_interval_above_zero():
    """A system better on every document has its interval above 0."""
    better = {f"d{i}": Counts(2, 2, 2) for i in range(30)}
    worse = {f"d{i}": Counts(i % 2, 2, 2) for i in range(30)}
    interval = paired_difference_interval(better, worse, n_resamples=500, seed=0)
    assert interval.estimate == pytest.approx(0.75)
    assert interval.low > 0.0


def test_paired_difference_follows_document_ids_not_order():
    """Documents are paired by id, whatever order the mappings hold."""
    first = {"a": Counts(1, 1, 1), "b": Counts(0, 1, 1)}
    second = {"b": Counts(0, 1, 1), "a": Counts(1, 1, 1)}
    interval = paired_difference_interval(first, second, n_resamples=100, seed=0)
    assert (interval.low, interval.high) == (0.0, 0.0)


def test_paired_difference_rejects_unmatched_documents():
    """Systems scored on different documents raise, naming an id."""
    with pytest.raises(ValueError, match="'c'"):
        paired_difference_interval(
            {"a": Counts(), "c": Counts()}, {"a": Counts(), "b": Counts()}, n_resamples=10, seed=0
        )
