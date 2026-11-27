"""The multiple-choice head, one registration per family.

Each family reads its example into one text pair per choice through its own
pair function; the head is otherwise the same, and an example may have any
number of choices. The model scores every pair; the best-scored choice is the
answer, scored by `score_choices`. Accuracy picks the checkpoint, as it is
the score multiple-choice benchmarks report.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForMultipleChoice

from slm4ie.evaluation.scoring import score_choices
from slm4ie.finetune.heads.base import MODEL_INPUTS, Feature, TaskHead, model_inputs, register_head

# ===============================
# Head
# ===============================

Pairs = Callable[[Mapping[str, Any]], list[tuple[str, str]]]


class MultipleChoiceHead(TaskHead):
    """Picks the most plausible of an example's choices.

    Args:
        pairs: Reads an example into one text pair per choice, in choice order.
    """

    model_class = AutoModelForMultipleChoice
    primary_metric = "accuracy"
    scorer = staticmethod(score_choices)

    def __init__(self, pairs: Pairs) -> None:
        """Keep the family's pair function."""
        self.pairs = pairs

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
        """Tokenize each choice's text pair, truncated to `max_length` tokens.

        Args:
            examples: Examples in the family's task schema, each with a `label` that indexes its choices.
            tokenizer: A fast HuggingFace tokenizer.
            labels: The entry's label values, the choice indices.
            max_length: Most tokens per pair; the longer text of a pair is cut first.

        Returns:
            One feature per example, its inputs holding one row per choice.

        Raises:
            ValueError: If an example's label is not the index of one of its choices.
        """
        features = []
        for example in examples:
            pairs = self.pairs(example)
            if example["label"] not in range(len(pairs)):
                raise ValueError(f"example {example['id']!r} has label {example['label']!r}, not a choice index")
            encoded = tokenizer(
                [first for first, _ in pairs], [second for _, second in pairs], truncation=True, max_length=max_length
            )
            features.append({**model_inputs(encoded), "labels": example["label"]})
        return features

    def decode(
        self,
        examples: Sequence[Mapping[str, Any]],
        features: Sequence[Feature],
        logits: Sequence[np.ndarray],
        labels: Sequence[Any],
    ) -> list[int]:
        """Pick the best-scored of each example's own choices.

        Args:
            examples: The examples.
            features: One feature per example.
            logits: One array of per-choice logits per example, padded to its batch's most choices.
            labels: The entry's label values, the choice indices.

        Returns:
            One choice index per example.
        """
        return [int(row[: len(feature["input_ids"])].argmax()) for feature, row in zip(features, logits, strict=True)]

    def collate(self, features: Sequence[Feature], pad_id: int) -> dict[str, torch.Tensor]:
        """Pad a batch to its most choices with empty rows, then pad every row into tensors.

        An empty row becomes padding tokens under a zero attention mask. The
        batch's `choice_mask` marks the real choices for `loss`; it is not a
        model input, and `decode` ignores a padded choice by its own count.

        Args:
            features: The batch.
            pad_id: The tokenizer's padding token id.

        Returns:
            The model inputs, the labels and the choice mask as tensors.
        """
        most = max(len(feature["input_ids"]) for feature in features)
        padded = [
            {
                **feature,
                **{key: feature[key] + [[]] * (most - len(feature[key])) for key in MODEL_INPUTS if key in feature},
            }
            for feature in features
        ]
        batch = super().collate(padded, pad_id)
        batch["choice_mask"] = torch.tensor(
            [[index < len(feature["input_ids"]) for index in range(most)] for feature in features]
        )
        return batch

    def loss(self, network: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the cross-entropy over each example's real choices, its padded choices masked out.

        Args:
            network: The model `load_model` built.
            batch: A batch from `collate`, its `labels` and `choice_mask` included.

        Returns:
            The batch's mean loss, a scalar tensor.
        """
        logits = network(**{key: value for key, value in batch.items() if key in MODEL_INPUTS}).logits
        return torch.nn.functional.cross_entropy(
            logits.masked_fill(~batch["choice_mask"], float("-inf")), batch["labels"]
        )


# ===============================
# Registration
# ===============================

register_head(
    "commonsense",
    MultipleChoiceHead(lambda ex: [(f"{ex['question']}: {ex['premise']}", c) for c in (ex["choice1"], ex["choice2"])]),
)
# truncation cuts the longer text of a pair first, which is the passage
register_head(
    "reading_mc", MultipleChoiceHead(lambda ex: [(ex["passage"], f"{ex['question']} {c}") for c in ex["choices"]])
)
