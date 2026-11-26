"""The multiple-choice head for `commonsense` (COPA).

Each choice is paired with the premise, marked with whether the question asks
for its cause or its effect, and the model scores every pair; the best-scored
choice is the answer. Accuracy picks the checkpoint, as COPA's two answers are
balanced.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForMultipleChoice

from slm4ie.evaluation.scoring import accuracy, classification_macro_f1
from slm4ie.finetune.heads.base import MODEL_INPUTS, Feature, TaskHead, register_head

# ===============================
# Head
# ===============================


class MultipleChoiceHead(TaskHead):
    """Picks the most plausible of an example's choices."""

    model_class = AutoModelForMultipleChoice
    primary_metric = "accuracy"

    def load_model(self, encoder: str, labels: Sequence[Any]) -> torch.nn.Module:
        """Build a multiple-choice model, which scores each choice with one output.

        Args:
            encoder: The encoder, by HuggingFace name or local path.
            labels: The entry's label values, the choice indices; unused.

        Returns:
            The model, on the CPU.
        """
        return self.model_class.from_pretrained(encoder)

    def encode(
        self, examples: Sequence[Mapping[str, Any]], tokenizer: Any, labels: Sequence[Any], max_length: int
    ) -> list[Feature]:
        """Pair each choice with the question and premise.

        Args:
            examples: COPA examples, each `{"id", "premise", "choice1", "choice2", "question", "label"}`.
            tokenizer: A fast HuggingFace tokenizer.
            labels: The entry's label values, the choice indices.
            max_length: Most tokens per pair.

        Returns:
            One feature per example, its inputs holding one row per choice.

        Raises:
            ValueError: If an example's label is not the index of one of its choices.
        """
        features = []
        for example in examples:
            choices = [example["choice1"], example["choice2"]]
            if example["label"] not in range(len(choices)):
                raise ValueError(f"example {example['id']!r} has label {example['label']!r}, not a choice index")
            first = f"{example['question']}: {example['premise']}"
            encoded = tokenizer([first] * len(choices), choices, truncation=True, max_length=max_length)
            features.append(
                {**{key: encoded[key] for key in MODEL_INPUTS if key in encoded}, "labels": example["label"]}
            )
        return features

    def score(
        self,
        examples: Sequence[Mapping[str, Any]],
        features: Sequence[Feature],
        logits: Sequence[np.ndarray],
        labels: Sequence[Any],
    ) -> dict[str, float]:
        """Score the best-scored choice of each example.

        Args:
            examples: COPA examples.
            features: One feature per example.
            logits: One array of per-choice logits per example.
            labels: The entry's label values, the choice indices.

        Returns:
            `accuracy` and `macro_f1` over the choice indices.
        """
        predicted = [int(row.argmax()) for row in logits]
        gold = [int(example["label"]) for example in examples]
        return {"accuracy": accuracy(predicted, gold), "macro_f1": classification_macro_f1(predicted, gold)}


# ===============================
# Registration
# ===============================

register_head("commonsense", MultipleChoiceHead())
