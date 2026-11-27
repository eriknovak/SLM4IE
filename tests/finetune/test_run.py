"""Tests for slm4ie.finetune.run."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from slm4ie.finetune.run import finetune  # noqa: E402

# ===============================
# Test Data
# ===============================

# each head kind's scores, checkpoint score first, and its labels
HEAD_KINDS = {
    "ner": (["span_f1", "span_relaxed_f1", "span_partial_f1"], ["PER", "LOC", "ORG"]),
    "pos": (["token_accuracy"], ["ADJ", "ADP", "NOUN", "PROPN", "PUNCT", "VERB"]),
    "nli": (["macro_f1"], ["entailment", "contradiction"]),
    "commonsense": (["accuracy"], [0, 1]),
    "reading_mc": (["accuracy", "macro_f1"], [0, 1, 2, 3]),
    "relation": (["relation_f1", "relation_macro_f1"], ["works_at", "located_in"]),
    "entity_sentiment": (["macro_f1", "accuracy"], ["negative", "neutral", "positive"]),
    "topic": (["macro_f1", "accuracy"], ["sport", "politika", "gospodarstvo"]),
    "extractive_qa": (["f1", "exact_match", "has_answer_f1", "no_answer_f1"], None),
    "keyphrase": (["f1_at_m", "f1_at_5", "f1_at_10", "gold_in_text"], None),
}

# ===============================
# Tests
# ===============================


@pytest.mark.parametrize("task", sorted(HEAD_KINDS))
def test_finetune_returns_test_score_in_range_and_costs(task, tiny_encoder, roots, settings, fixture_entry):
    """Fine-tuning a tiny encoder on each head kind's fixture returns its test scores in [0, 1] and the costs."""
    score_names, labels = HEAD_KINDS[task]

    result = finetune(fixture_entry(task, labels), tiny_encoder, settings, roots, device="cpu")

    assert all(0.0 <= result.scores[name] <= 1.0 for name in score_names)
    assert result.parameters > 0
    assert result.train_seconds > 0
    assert result.train_peak_gpu_memory_bytes is None and result.inference_peak_gpu_memory_bytes is None


def test_finetune_refuses_entry_without_train_split(tiny_encoder, roots, settings, fixture_entry):
    """A held-out entry has no train split, so fine-tuning refuses it, naming the entry and its role."""
    held_out = fixture_entry("nli", ["entailment", "contradiction"], role="held_out")

    with pytest.raises(ValueError, match=r"nli/tiny.*held_out"):
        finetune(held_out, tiny_encoder, settings, roots, device="cpu")


def test_finetune_refuses_labelled_entry_without_labels(tiny_encoder, roots, settings, fixture_entry):
    """A labelled family's entry that declares no labels is refused, the error naming the entry."""
    with pytest.raises(ValueError, match=r"nli/tiny.*no labels"):
        finetune(fixture_entry("nli", None), tiny_encoder, settings, roots, device="cpu")


def test_finetune_falls_back_to_tokenizer_json_when_mapped_class_fails(
    tiny_encoder, roots, settings, fixture_entry, monkeypatch, caplog
):
    """An encoder whose mapped tokenizer class raises, as SloBERTa's does, still trains from its tokenizer.json."""
    score_names, labels = HEAD_KINDS["nli"]

    def broken_from_pretrained(*args, **kwargs):
        raise ValueError("too many values to unpack (expected 2)")

    monkeypatch.setattr("slm4ie.finetune.run.AutoTokenizer.from_pretrained", broken_from_pretrained)
    result = finetune(fixture_entry("nli", labels), tiny_encoder, settings, roots, device="cpu")

    assert 0.0 <= result.scores[score_names[0]] <= 1.0
    assert "loading its tokenizer.json directly" in caplog.text
