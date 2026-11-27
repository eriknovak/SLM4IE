"""Tests for slm4ie.evaluation.scoring."""

import functools

import pytest

from slm4ie.evaluation.scoring import (
    Alignment,
    Counts,
    accuracy,
    align_mentions,
    classification_macro_f1,
    f1_interval,
    macro_f1,
    paired_difference_interval,
    score_choices,
    score_document,
    score_extractive_qa,
    score_keyphrases,
    score_labels,
    score_relations,
    score_spans,
    score_tags,
    scorer_for,
    token_accuracy,
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


def test_token_accuracy_pools_tokens_over_sentences():
    """Three of the five tokens are right across two sentences."""
    predicted = [["NOUN", "VERB"], ["ADJ", "NOUN", "ADP"]]
    gold = [["NOUN", "VERB"], ["ADJ", "VERB", "NOUN"]]
    assert token_accuracy(predicted, gold) == pytest.approx(3 / 5)


def test_token_accuracy_rejects_length_mismatch():
    """A sentence with a different tag count raises."""
    with pytest.raises(ValueError, match="sentence 0"):
        token_accuracy([["NOUN"]], [["NOUN", "VERB"]])


def test_accuracy_counts_equal_labels():
    """Two of three labels match."""
    assert accuracy(["pos", "neg", "pos"], ["pos", "pos", "pos"]) == pytest.approx(2 / 3)


def test_classification_macro_f1_averages_classes_equally():
    """Class a has F1 0.8 and class b has F1 2/3."""
    predicted = ["a", "a", "b", "a"]
    gold = ["a", "a", "b", "b"]
    assert classification_macro_f1(predicted, gold) == pytest.approx((0.8 + 2 / 3) / 2)


def test_classification_macro_f1_scores_a_never_predicted_class_zero():
    """A gold class the model never predicts drags the average down."""
    assert classification_macro_f1(["a", "a"], ["a", "b"]) == pytest.approx((2 / 3 + 0) / 2)


# ===============================
# Family scorers
# ===============================


def test_score_spans_reports_the_four_span_scores():
    """One exact span, one boundary error and one missed span across two examples."""
    examples = [
        {"id": "d1", "text": "Janez Novak je iz Kranja.", "spans": [span(0, 11, "PER"), span(18, 24, "LOC")]},
        {"id": "d2", "text": "Ana", "spans": [span(0, 3, "PER")]},
    ]
    predictions = [[span(0, 11, "PER"), span(18, 25, "LOC")], []]
    # strict hits PER only; the wider LOC span is a relaxed and a partial hit
    assert score_spans(examples, predictions) == pytest.approx(
        {
            "span_f1": 2 * 1 / 5,
            "span_relaxed_f1": 2 * 2 / 5,
            "span_partial_f1": 2 * 2 / 5,
            "span_macro_f1": (2 * 1 / 3 + 0) / 2,
        }
    )


def test_score_tags_pools_words_and_counts_an_untagged_word_wrong():
    """Three of the five words are right; a word with no tag is wrong."""
    examples = [
        {"id": "s1", "words": ["a", "b"], "tags": ["NOUN", "VERB"]},
        {"id": "s2", "words": ["c", "d", "e"], "tags": ["ADJ", "VERB", "NOUN"]},
    ]
    predictions = [["NOUN", "VERB"], ["ADJ", None, "ADP"]]
    assert score_tags(examples, predictions) == pytest.approx({"token_accuracy": 3 / 5})


def test_score_labels_of_perfect_predictions_score_one():
    """Every label right scores 1 on both scores."""
    examples = [{"id": str(i), "label": label} for i, label in enumerate(["a", "a", "b"])]
    assert score_labels(examples, ["a", "a", "b"]) == pytest.approx({"accuracy": 1.0, "macro_f1": 1.0})


def test_score_labels_reports_accuracy_and_macro_f1():
    """Class a has F1 0.8 and class b has F1 2/3."""
    examples = [{"id": str(i), "label": label} for i, label in enumerate(["a", "a", "b", "b"])]
    assert score_labels(examples, ["a", "a", "b", "a"]) == pytest.approx(
        {"accuracy": 3 / 4, "macro_f1": (0.8 + 2 / 3) / 2}
    )


CHOICE_EXAMPLES = [
    {"id": "c1", "choice1": "x", "choice2": "y", "label": 0},
    {"id": "c2", "choice1": "x", "choice2": "y", "label": 1},
    {"id": "r1", "choices": ["p", "q", "r", "s"], "label": 3},
]


def test_score_choices_of_perfect_predictions_score_one():
    """Every choice right scores 1 on both scores."""
    assert score_choices(CHOICE_EXAMPLES, [0, 1, 3]) == pytest.approx({"accuracy": 1.0, "macro_f1": 1.0})


def test_score_choices_reads_two_choice_and_list_choice_examples():
    """COPA's two named choices and a choice list both take an index; two of three are right."""
    # class 0 F1 2/3, class 1 F1 0, class 3 F1 1
    assert score_choices(CHOICE_EXAMPLES, [0, 0, 3]) == pytest.approx(
        {"accuracy": 2 / 3, "macro_f1": (2 / 3 + 0 + 1) / 3}
    )


# ===============================
# Relation scorer
# ===============================

RELATION_TEXT = "Ana Novak dela pri NLB v Ljubljani."
ANA, NLB, LJUBLJANA = span(0, 9, "PER"), span(19, 22, "ORG"), span(25, 34, "LOC")
RELATION_EXAMPLE = {
    "id": "r1",
    "text": RELATION_TEXT,
    "entities": [{**ANA, "id": "m1"}, {**NLB, "id": "m2"}, {**LJUBLJANA, "id": "m3"}],
    "relations": [
        {"head": "m1", "tail": "m2", "label": "works_for"},
        {"head": "m2", "tail": "m3", "label": "located_in"},
    ],
}


def relation(head: dict, tail: dict, label: str) -> dict:
    """Build one predicted relation in the relation family's prediction shape."""
    return {"head": head, "tail": tail, "label": label}


PERFECT_RELATIONS = [relation(ANA, NLB, "works_for"), relation(NLB, LJUBLJANA, "located_in")]
# one right relation and one wrong pair labelled works_for
PARTIAL_RELATIONS = [relation(ANA, NLB, "works_for"), relation(ANA, LJUBLJANA, "works_for")]


@pytest.mark.parametrize(
    ("predictions", "expected"),
    [
        (PERFECT_RELATIONS, {"relation_f1": 1.0, "relation_macro_f1": 1.0}),
        ([], {"relation_f1": 0.0, "relation_macro_f1": 0.0}),
        # works_for has 1 hit of 2 predicted and 1 gold, located_in is missed
        (PARTIAL_RELATIONS, {"relation_f1": 2 * 1 / 4, "relation_macro_f1": (2 * 1 / 3 + 0) / 2}),
    ],
    ids=["perfect", "empty", "partial"],
)
def test_score_relations_with_gold_entities(predictions, expected):
    """Pairs of gold mentions score strict micro F1 and macro F1 over relation labels."""
    assert score_relations([RELATION_EXAMPLE], [predictions], gold_entities=True) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("predictions", "expected"),
    [
        (PERFECT_RELATIONS, {"relation_f1": 1.0, "relation_macro_f1": 1.0, "relation_boundary_f1": 1.0}),
        ([], {"relation_f1": 0.0, "relation_macro_f1": 0.0, "relation_boundary_f1": 0.0}),
        # a head span one character short misses under every rule
        (
            [relation(span(0, 8, "PER"), NLB, "works_for"), relation(NLB, LJUBLJANA, "located_in")],
            {"relation_f1": 2 * 1 / 4, "relation_macro_f1": (0 + 1) / 2, "relation_boundary_f1": 2 * 1 / 4},
        ),
    ],
    ids=["perfect", "empty", "partial"],
)
def test_score_relations_end_to_end(predictions, expected):
    """Found mentions score strict micro F1, macro F1 and boundary F1."""
    assert score_relations([RELATION_EXAMPLE], [predictions], gold_entities=False) == pytest.approx(expected)


def test_score_relations_boundary_f1_ignores_entity_labels():
    """A relation with the right spans but a wrong entity label is a boundary hit only."""
    predictions = [[relation(span(0, 9, "ORG"), NLB, "works_for"), relation(NLB, LJUBLJANA, "located_in")]]
    scores = score_relations([RELATION_EXAMPLE], predictions, gold_entities=False)
    assert scores["relation_f1"] == pytest.approx(2 * 1 / 4)
    assert scores["relation_boundary_f1"] == pytest.approx(1.0)


def test_score_relations_with_gold_entities_refuses_a_non_gold_span():
    """With gold entities, an argument that is no gold mention raises with the example id."""
    with pytest.raises(ValueError, match="'r1'"):
        score_relations([RELATION_EXAMPLE], [[relation(span(0, 3, "PER"), NLB, "works_for")]], gold_entities=True)


# ===============================
# Extractive QA scorer
# ===============================


def test_score_extractive_qa_scores_answerable_and_unanswerable_questions():
    """A partly right answer and a correct abstention, scored overall and by kind."""
    examples = [
        {"id": "q1", "context": "", "question": "", "answers": [{"text": "glavno mesto Slovenije", "start": 0}]},
        {"id": "q2", "context": "", "question": "", "answers": []},
    ]
    # both predicted words are among the three gold words: precision 1, recall 2/3, F1 0.8
    expected = {
        "exact_match": 1 / 2,
        "f1": (0.8 + 1) / 2,
        "has_answer_exact_match": 0.0,
        "has_answer_f1": 0.8,
        "no_answer_exact_match": 1.0,
        "no_answer_f1": 1.0,
    }
    assert score_extractive_qa(examples, ["mesto Slovenije", ""]) == pytest.approx(expected)


def test_score_extractive_qa_takes_the_best_gold_answer_and_keeps_articles():
    """Case, Slovene quotes and spacing are normalised away; an English article is not."""
    examples = [
        {
            "id": "q1",
            "context": "",
            "question": "",
            "answers": [{"text": "x", "start": 0}, {"text": "Ljubljana", "start": 0}],
        },
        {"id": "q2", "context": "", "question": "", "answers": [{"text": "Ljubljana", "start": 0}]},
    ]
    scores = score_extractive_qa(examples, ["  „LJUBLJANA“! ", "the Ljubljana"])
    assert scores["exact_match"] == pytest.approx(1 / 2)
    assert scores["f1"] == pytest.approx((1 + 2 / 3) / 2)


def test_score_extractive_qa_scores_no_answer_to_an_answerable_question_zero():
    """The empty answer to a question with a gold answer misses on every score."""
    examples = [{"id": "q1", "context": "", "question": "", "answers": [{"text": "Bled", "start": 0}]}]
    expected = {
        "exact_match": 0.0,
        "f1": 0.0,
        "has_answer_exact_match": 0.0,
        "has_answer_f1": 0.0,
        "no_answer_exact_match": 0.0,
        "no_answer_f1": 0.0,
    }
    assert score_extractive_qa(examples, [""]) == pytest.approx(expected)


def test_score_extractive_qa_scores_a_kind_with_no_questions_zero():
    """A split with no unanswerable question reports 0 for that kind, not a gap."""
    examples = [{"id": "q1", "context": "", "question": "", "answers": [{"text": "Bled", "start": 0}]}]
    scores = score_extractive_qa(examples, ["Bled"])
    assert scores["has_answer_exact_match"] == 1.0
    assert scores["no_answer_exact_match"] == scores["no_answer_f1"] == 0.0


# ===============================
# Keyphrase scorer
# ===============================


def test_score_keyphrases_of_perfect_predictions_score_one_at_m():
    """Every gold keyphrase predicted scores F1@M 1, while F1@5 and F1@10 count the empty ranks as wrong."""
    examples = [{"id": "k1", "text": "Volitve in vlada.", "keyphrases": ["volitve", "vlada"]}]
    # 2 hits of 2 gold: F1@5 = 4 / (5 + 2), F1@10 = 4 / (10 + 2)
    assert score_keyphrases(examples, [["Volitve", "vlada"]]) == pytest.approx(
        {"f1_at_5": 4 / 7, "f1_at_10": 4 / 12, "f1_at_m": 1.0, "gold_in_text": 1.0}
    )


def test_score_keyphrases_reports_f1_at_5_10_and_m_after_normalising():
    """Duplicates after normalisation count once; F1@5 and F1@10 treat missing ranks as wrong."""
    examples = [{"id": "k1", "text": "", "keyphrases": ["Volitve", "vlada", "parlament", "koalicija"]}]
    ranked = ["volitve", "VOLITVE!", "šport", "Vlada", "vreme", "kultura", "parlament"]
    # deduplicated: volitve, šport, vlada, vreme, kultura, parlament; 2 hits in the top 5, 3 in all 6
    assert score_keyphrases(examples, [ranked]) == pytest.approx(
        {"f1_at_5": 2 * 2 / 9, "f1_at_10": 2 * 3 / 14, "f1_at_m": 2 * 3 / 10, "gold_in_text": 0.0}
    )


def test_score_keyphrases_of_empty_predictions_scores_zero_and_the_share_in_the_text():
    """Empty predictions score 0; a gold keyphrase counts as in the text as whole words, not as part of a word."""
    examples = [
        {"id": "k1", "text": "Vlada je sprejela proračun.", "keyphrases": ["vlada", "proračun", "volitve"]},
        {"id": "k2", "text": "Vladimir je odstopil.", "keyphrases": ["vlad"]},
    ]
    assert score_keyphrases(examples, [[], []]) == pytest.approx(
        {"f1_at_5": 0.0, "f1_at_10": 0.0, "f1_at_m": 0.0, "gold_in_text": 2 / 4}
    )


# ===============================
# Shape and length errors
# ===============================


@pytest.mark.parametrize(
    ("scorer", "example", "prediction"),
    [
        (score_spans, {"id": "bad", "text": "Ana", "spans": []}, None),
        (score_spans, {"id": "bad", "text": "Ana", "spans": []}, [{"start": 0, "label": "PER"}]),
        (score_tags, {"id": "bad", "words": ["a", "b"], "tags": ["X", "Y"]}, ["X"]),
        (score_labels, {"id": "bad", "label": "a"}, None),
        (score_labels, {"id": "bad", "label": "a"}, ["a"]),
        (score_choices, {"id": "bad", "choice1": "x", "choice2": "y", "label": 0}, 2),
        (score_choices, {"id": "bad", "choices": ["x", "y"], "label": 0}, "0"),
        (functools.partial(score_relations, gold_entities=False), {**RELATION_EXAMPLE, "id": "bad"}, None),
        (functools.partial(score_relations, gold_entities=False), {**RELATION_EXAMPLE, "id": "bad"}, [{"head": ANA}]),
        (score_extractive_qa, {"id": "bad", "context": "", "question": "", "answers": []}, None),
        (score_keyphrases, {"id": "bad", "text": "", "keyphrases": []}, "vlada"),
    ],
)
def test_missing_or_misshapen_prediction_raises_naming_the_example(scorer, example, prediction):
    """A missing or wrongly shaped prediction raises with the example id."""
    with pytest.raises(ValueError, match="'bad'"):
        scorer([example], [prediction])


def test_prediction_count_unlike_example_count_raises():
    """Predictions align by position, so a missing tail is an error."""
    with pytest.raises(ValueError, match="1 predictions for 2 examples"):
        score_labels([{"id": "a", "label": "x"}, {"id": "b", "label": "x"}], ["x"])


@pytest.mark.parametrize(
    ("family", "scorer"),
    [
        ("ner", score_spans),
        ("pos", score_tags),
        ("sentiment", score_labels),
        ("nli", score_labels),
        ("qa", score_labels),
        ("coref", score_labels),
        ("wsd", score_labels),
        ("entity_sentiment", score_labels),
        ("topic", score_labels),
        ("commonsense", score_choices),
        ("reading_mc", score_choices),
        ("extractive_qa", score_extractive_qa),
        ("keyphrase", score_keyphrases),
    ],
)
def test_scorer_for_returns_the_family_scorer(family, scorer):
    """Each family maps to the scorer of its prediction shape."""
    assert scorer_for(family) is scorer


def test_scorer_for_binds_the_relation_setting():
    """The relation scorer with gold entities bound takes only examples and predictions."""
    scorer = scorer_for("relation", gold_entities=True)
    assert scorer([RELATION_EXAMPLE], [PERFECT_RELATIONS]) == pytest.approx(
        {"relation_f1": 1.0, "relation_macro_f1": 1.0}
    )


def test_scorer_for_relation_without_its_setting_raises_naming_it():
    """The relation scorer needs `gold_entities`, so asking for it without one raises with the family and option."""
    with pytest.raises(ValueError, match="'relation'.*gold_entities"):
        scorer_for("relation")


def test_scorer_for_unknown_family_raises_naming_it():
    """A family with no scorer raises with its name."""
    with pytest.raises(KeyError, match="'dependency'"):
        scorer_for("dependency")
