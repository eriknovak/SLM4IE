"""Tests for slm4ie.finetune.heads.token."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import AutoTokenizer  # noqa: E402

from slm4ie.finetune.heads import get_head  # noqa: E402

# ===============================
# Test Data
# ===============================

MAX_LENGTH = 16

# ===============================
# Tests
# ===============================


def test_keyphrase_overlap_between_equal_length_phrases_is_tagged_by_text_order(tiny_encoder):
    """Of two overlapping keyphrases of equal length, the one first by text keeps its occurrence."""
    tokenizer = AutoTokenizer.from_pretrained(tiny_encoder)
    text = "ana bor cev"
    example = {"id": "kp:0", "text": text, "keyphrases": ["bor cev", "ana bor"]}
    head = get_head("keyphrase")

    (feature,) = head.encode([example], tokenizer, None, MAX_LENGTH)

    # the tag of the first token of each word
    names = head.output_names(None)
    first_tags = {}
    for offset, tag in zip(feature["offsets"], feature["labels"]):
        if offset is not None and offset[1] > offset[0]:
            first_tags.setdefault(text.rfind(" ", 0, offset[0] + 1) + 1, names[tag])
    assert [first_tags[text.index(word)] for word in ("ana", "bor", "cev")] == ["B-KEY", "I-KEY", "O"]
