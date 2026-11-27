"""Tests for slm4ie.silver.passages."""

import pytest

from slm4ie.silver.passages import candidate_windows, draw_passages, prose_paragraphs

# ===============================
# Test Data
# ===============================

SENTENCE = "Vzorec smo analizirali z metodo, ki jo opisujemo v nadaljevanju tega poglavja in v prilogi."
PARAGRAPH = " ".join([SENTENCE] * 4)
CONTENTS_LINE = "3.2 Metode dela in opis vzorca raziskave ter potek zbiranja podatkov ... 17"
TABLE_LINE = "12,4 13,1 9,8 11,0 7,7 8,2 10,5 14,9 3,3 2,1 6,6 5,4 1,2 0,9."


def document(doc_id, paragraphs, **metadata):
    """Build a corpus record whose text holds one paragraph per line."""
    return {"id": doc_id, "text": "\n".join(paragraphs), "metadata": metadata}


# ===============================
# prose_paragraphs
# ===============================


def test_prose_paragraphs_drops_contents_and_table_lines():
    """A contents line and a line of numbers are no body prose."""
    assert prose_paragraphs("\n".join([CONTENTS_LINE, PARAGRAPH, TABLE_LINE])) == [PARAGRAPH]


def test_prose_paragraphs_joins_wrapped_lines():
    """Wrapped lines of one paragraph are joined when blank lines separate paragraphs."""
    wrapped = PARAGRAPH.replace(" v nadaljevanju", " \nv nadaljevanju")

    assert prose_paragraphs(wrapped + "\n \nKratka vrstica.", wrapped=True) == [PARAGRAPH]


# ===============================
# candidate_windows
# ===============================


def test_candidate_windows_stay_inside_the_length_range():
    """Every window holds between the minimum and the maximum number of words."""
    windows = candidate_windows([PARAGRAPH] * 5, min_words=100, max_words=300)

    assert windows
    assert all(100 <= len(text.split()) <= 300 for _, text in windows)


def test_candidate_windows_require_a_cue_word():
    """A window without any cue word is left out."""
    assert candidate_windows([PARAGRAPH] * 3, cue_words=["kromatograf"]) == []


# ===============================
# draw_passages
# ===============================


def test_draw_passages_takes_one_passage_per_document_and_group():
    """No two passages share a document or a non-empty group."""
    documents = [document(f"d{i}", [PARAGRAPH] * 6, medicine=f"m{i % 3}") for i in range(9)]

    passages = draw_passages(documents, count=3, seed=1, group_of=lambda record: record["metadata"]["medicine"])

    assert len({passage.doc_id for passage in passages}) == 3
    assert len({passage.group for passage in passages}) == 3


def test_draw_passages_is_repeatable_for_a_seed():
    """The same seed draws the same passages."""
    documents = [document(f"d{i}", [PARAGRAPH] * 6) for i in range(8)]

    assert draw_passages(documents, count=4, seed=7) == draw_passages(documents, count=4, seed=7)


def test_draw_passages_rejects_too_few_documents():
    """Fewer usable documents than passages wanted is an error naming both counts."""
    with pytest.raises(ValueError, match="drew 1 passages from 1 kept documents, 2 wanted"):
        draw_passages([document("d0", [PARAGRAPH] * 6)], count=2, seed=1)
