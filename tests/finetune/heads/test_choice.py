"""Tests for slm4ie.finetune.heads.choice."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from slm4ie.finetune.heads import get_head  # noqa: E402
from tests.finetune.conftest import FIXTURES  # noqa: E402

# ===============================
# Test Data
# ===============================

MAX_LENGTH = 16

# ===============================
# Tests
# ===============================


def test_multiple_choice_loss_of_an_example_does_not_depend_on_its_batch(tiny_encoder):
    """A 2-choice example's loss alone equals its share of a batch padded to a 4-choice example's choices."""
    head = get_head("reading_mc")
    path = FIXTURES / "tasks" / "reading_mc" / "tiny" / "train.jsonl"
    examples = {ex["id"]: ex for ex in map(json.loads, path.read_text().splitlines())}
    tokenizer = AutoTokenizer.from_pretrained(tiny_encoder)
    two, four = head.encode([examples["tiny:2"], examples["tiny:1"]], tokenizer, [0, 1, 2, 3], MAX_LENGTH)
    network = head.load_model(tiny_encoder, [0, 1, 2, 3]).eval()

    with torch.no_grad():
        alone = [head.loss(network, head.collate([feature], tokenizer.pad_token_id)) for feature in (two, four)]
        batched = head.loss(network, head.collate([two, four], tokenizer.pad_token_id))

    assert batched.item() == pytest.approx((alone[0].item() + alone[1].item()) / 2, abs=1e-5)
