"""Tests for slm4ie.finetune.heads."""

import functools
import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import AutoTokenizer  # noqa: E402

from slm4ie.evaluation.scoring import scorer_for  # noqa: E402
from slm4ie.finetune.heads import available_heads, get_head  # noqa: E402
from tests.finetune.conftest import FIXTURES  # noqa: E402

# ===============================
# Test Data
# ===============================

# one family per prediction shape: spans, tags, labels, choices, relations, answer strings
# and ranked keyphrases; a label-free family has no labels
SHAPES = {
    "ner": ["PER", "LOC", "ORG"],
    "pos": ["ADJ", "ADP", "NOUN", "PROPN", "PUNCT", "VERB"],
    "nli": ["entailment", "contradiction"],
    "commonsense": [0, 1],
    "relation": ["works_at", "located_in"],
    "extractive_qa": None,
    "keyphrase": None,
}
# the setting a family's scorer takes beside the examples and predictions
SCORER_OPTIONS = {"relation": {"gold_entities": True}}
MAX_LENGTH = 16

# ===============================
# Helpers
# ===============================


def _same_scorer(first, second) -> bool:
    """Tell whether two scorers are one function with the same bound settings."""
    if isinstance(first, functools.partial) and isinstance(second, functools.partial):
        return (first.func, first.args, first.keywords) == (second.func, second.args, second.keywords)
    return first is second


# ===============================
# Tests
# ===============================


@pytest.mark.parametrize("family", sorted(SHAPES))
def test_score_equals_family_scorer_on_decoded_predictions(family, tiny_encoder):
    """A head's score on random logits equals its family's scorer applied to the head's decoded predictions."""
    labels = SHAPES[family]
    head = get_head(family)
    path = FIXTURES / "tasks" / family / "tiny" / "test.jsonl"
    examples = [json.loads(line) for line in path.read_text().splitlines()]
    features = head.encode(examples, AutoTokenizer.from_pretrained(tiny_encoder), labels, MAX_LENGTH)

    # one row per token for the taggers and extractive QA, one value per output or choice otherwise
    rng = np.random.default_rng(0)
    if family in ("ner", "pos", "keyphrase"):
        width = len(head.output_names(labels))
        logits = [rng.normal(size=(len(feature["input_ids"]), width)) for feature in features]
    elif family == "extractive_qa":
        logits = [rng.normal(size=(len(feature["input_ids"]), 2)) for feature in features]
    elif family == "commonsense":
        logits = [rng.normal(size=len(feature["input_ids"])) for feature in features]
    else:
        logits = [rng.normal(size=len(head.output_names(labels))) for _ in features]

    decoded = head.decode(examples, features, logits, labels)

    assert head.score(examples, features, logits, labels) == scorer_for(family, **SCORER_OPTIONS.get(family, {}))(
        examples, decoded
    )


@pytest.mark.parametrize("family", available_heads())
def test_head_scorer_is_the_family_scorer(family):
    """Every registered head scores with the scorer `scorer_for` returns for its family."""
    assert _same_scorer(get_head(family).scorer, scorer_for(family, **SCORER_OPTIONS.get(family, {})))
