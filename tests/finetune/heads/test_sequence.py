"""Tests for slm4ie.finetune.heads.sequence."""

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


def test_entity_sentiment_keeps_a_target_mentioned_past_max_length_marked(tiny_encoder):
    """A target mentioned only after the first `max_length` tokens still reaches the model inside its markers."""
    tokenizer = AutoTokenizer.from_pretrained(tiny_encoder)
    text = (
        "Vreme je bilo lepo ves dan in vsi so bili zadovoljni z delom ter s plačilom, "
        "nato pa je Krka objavila poročilo."
    )
    start = text.index("Krka")
    example = {
        "id": "long:0",
        "text": text,
        "mentions": [{"start": start, "end": start + len("Krka")}],
        "entity_label": "ORG",
        "label": "neutral",
    }

    (feature,) = get_head("entity_sentiment").encode(
        [example], tokenizer, ["negative", "neutral", "positive"], MAX_LENGTH
    )

    assert "[ krka ]" in tokenizer.decode(feature["input_ids"])
