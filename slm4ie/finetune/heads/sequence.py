"""Sequence and sequence-pair classification heads, one registration per family.

Each family reads its example into one text, or a text pair, through its own
field function; the head is otherwise the same. Scores are accuracy and macro
F1 over the entry's labels, and macro F1 picks the checkpoint, since several
of these entries are unbalanced across their labels.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from transformers import AutoModelForSequenceClassification

from slm4ie.evaluation.scoring import accuracy, classification_macro_f1
from slm4ie.finetune.heads.base import MODEL_INPUTS, Feature, TaskHead, label_index, register_head

# ===============================
# Head
# ===============================

Fields = Callable[[Mapping[str, Any]], tuple[str, str | None]]


class SequenceHead(TaskHead):
    """Classifies one text or a text pair into one of the entry's labels.

    Args:
        fields: Reads an example into its first text and its second text, or `None` for a single text.
    """

    model_class = AutoModelForSequenceClassification
    primary_metric = "macro_f1"

    def __init__(self, fields: Fields) -> None:
        """Keep the family's field function."""
        self.fields = fields

    def encode(
        self, examples: Sequence[Mapping[str, Any]], tokenizer: Any, labels: Sequence[Any], max_length: int
    ) -> list[Feature]:
        """Tokenize each example's text or pair, truncated to `max_length` tokens.

        Args:
            examples: Examples in the family's task schema, each with a `label`.
            tokenizer: A fast HuggingFace tokenizer.
            labels: The entry's label values.
            max_length: Most tokens per input; the longer text of a pair is cut first.

        Returns:
            One feature per example.

        Raises:
            ValueError: If an example's label is not among `labels`.
        """
        if not examples:
            return []

        texts = [self.fields(example) for example in examples]
        pairs = None if texts[0][1] is None else [second for _, second in texts]
        encoded = tokenizer([first for first, _ in texts], pairs, truncation=True, max_length=max_length)
        return [
            {
                **{key: encoded[key][index] for key in MODEL_INPUTS if key in encoded},
                "labels": label_index(labels, example["label"], example["id"]),
            }
            for index, example in enumerate(examples)
        ]

    def score(
        self,
        examples: Sequence[Mapping[str, Any]],
        features: Sequence[Feature],
        logits: Sequence[np.ndarray],
        labels: Sequence[Any],
    ) -> dict[str, float]:
        """Score the most likely label of each example.

        Args:
            examples: The examples.
            features: One feature per example.
            logits: One array of per-label logits per example.
            labels: The entry's label values.

        Returns:
            `accuracy` and `macro_f1` over the entry's labels.
        """
        predicted = [labels[int(row.argmax())] for row in logits]
        gold = [example["label"] for example in examples]
        return {"accuracy": accuracy(predicted, gold), "macro_f1": classification_macro_f1(predicted, gold)}


# ===============================
# Registration
# ===============================

register_head("sentiment", SequenceHead(lambda ex: (ex["text"], None)))
register_head("nli", SequenceHead(lambda ex: (ex["premise"], ex["hypothesis"])))
# the question comes first, so truncation cuts the passage
register_head("qa", SequenceHead(lambda ex: (ex["question"], ex["passage"])))
register_head("coref", SequenceHead(lambda ex: (ex["text"], f"{ex['span1']['text']}, {ex['span2']['text']}")))
register_head("wsd", SequenceHead(lambda ex: (f"{ex['word']}: {ex['sentence1']}", ex["sentence2"])))
