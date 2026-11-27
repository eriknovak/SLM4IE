"""Fixtures shared by the fine-tuning tests: a tiny encoder, the fixture entries and their settings."""

import json
from collections.abc import Callable
from pathlib import Path

from typing import TYPE_CHECKING

import pytest

from slm4ie.data.tasks.config import EntrySource, TaskEntry, TasksRoots

# torch and transformers load inside the fixtures, so a conftest load without
# the evaluation group leaves the test modules' own skips in charge
if TYPE_CHECKING:
    from slm4ie.finetune.config import FinetuneSettings

# ===============================
# Test Data
# ===============================

FIXTURES = Path(__file__).parent / "fixtures"
SPLITS = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# ===============================
# Fixtures
# ===============================


@pytest.fixture(scope="session")
def settings() -> "FinetuneSettings":
    """Return settings small enough to train the tiny encoder in seconds."""
    from slm4ie.finetune.config import FinetuneSettings

    return FinetuneSettings(
        learning_rate=1e-3,
        seed=0,
        epochs=2,
        batch_size=2,
        eval_batch_size=4,
        gradient_accumulation=2,
        max_length=16,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        precision="bf16",
    )


@pytest.fixture(scope="session")
def fixture_entry() -> Callable[..., TaskEntry]:
    """Return a builder of the registry entry of a fixture under `fixtures/tasks/<task>/tiny/`."""

    def build(task: str, labels: list, role: str = "finetune_only") -> TaskEntry:
        return TaskEntry(
            task=task,
            dataset="tiny",
            role=role,
            source=EntrySource(kind="raw", keys=["tiny"]),
            splits=SPLITS,
            labels=labels,
            suite=None,
            language="sl",
            license="cc-by-4.0",
            converter="tiny",
        )

    return build


@pytest.fixture(scope="session")
def roots(tmp_path_factory) -> TasksRoots:
    """Point the task tier at the committed fixtures."""
    tmp = tmp_path_factory.mktemp("roots")
    return TasksRoots(extracted=tmp, raw=tmp, tasks=FIXTURES / "tasks")


@pytest.fixture(scope="session")
def tiny_encoder(tmp_path_factory) -> str:
    """Save a randomly initialised two-layer BERT and a WordPiece tokenizer trained on the fixtures."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    texts = [line for path in sorted((FIXTURES / "tasks").rglob("*.jsonl")) for line in path.read_text().splitlines()]
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.train_from_iterator(
        [" ".join(str(v) for v in json.loads(t).values()) for t in texts],
        trainers.WordPieceTrainer(vocab_size=300, special_tokens=SPECIAL_TOKENS),
    )
    cls, sep = tokenizer.token_to_id("[CLS]"), tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls), ("[SEP]", sep)],
    )

    # save the tokenizer and the model side by side
    path = tmp_path_factory.mktemp("tiny-encoder")
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    ).save_pretrained(path)
    config = BertConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    BertModel(config).save_pretrained(path)
    return str(path)
