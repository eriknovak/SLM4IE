"""The task-head interface and its registry.

A head turns one task family's examples into model inputs, builds the matching
HuggingFace model class on an encoder, and decodes the model's logits into the
family's prediction shape. Its scores are the family's scorer from
`slm4ie.evaluation.scoring` applied to those predictions, so a fine-tuned
head is scored exactly as any other approach's predictions are. One head
serves one or more families; each family is a registration, so a new family
is a new `register_head` call rather than a new script.
"""

import abc
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
import torch

from slm4ie.evaluation.scoring import Scorer

# ===============================
# Constants
# ===============================

# the label value the HuggingFace losses ignore
IGNORE_INDEX = -100
# the feature keys the model reads; the rest are the head's own bookkeeping
MODEL_INPUTS = ("input_ids", "attention_mask", "token_type_ids")

Feature = dict[str, Any]

# ===============================
# Head interface
# ===============================


class TaskHead(abc.ABC):
    """Encodes one task family's examples, builds its model and decodes its logits into predictions.

    Attributes:
        model_class: The HuggingFace auto class whose `from_pretrained` builds the model.
        primary_metric: The score that picks the best checkpoint on val, one of the keys `score` returns.
        scorer: The family's scorer, `(examples, predictions) -> scores`, held as a `staticmethod`.
    """

    model_class: ClassVar[Any]
    primary_metric: ClassVar[str]
    scorer: ClassVar[Scorer]

    def output_names(self, labels: Sequence[Any]) -> list[str]:
        """Return the name of each model output, in output order.

        Args:
            labels: The entry's label values.

        Returns:
            One name per output; the label values as strings by default.
        """
        return [str(label) for label in labels]

    def load_model(self, encoder: str, labels: Sequence[Any]) -> torch.nn.Module:
        """Build the head's model class on an encoder, with a fresh output layer.

        Args:
            encoder: The encoder, by HuggingFace name or local path.
            labels: The entry's label values.

        Returns:
            The model, on the CPU.
        """
        names = self.output_names(labels)
        return self.model_class.from_pretrained(
            encoder,
            num_labels=len(names),
            id2label=dict(enumerate(names)),
            label2id={name: index for index, name in enumerate(names)},
        )

    @abc.abstractmethod
    def encode(
        self, examples: Sequence[Mapping[str, Any]], tokenizer: Any, labels: Sequence[Any], max_length: int
    ) -> list[Feature]:
        """Turn examples into features, one or more per example.

        Args:
            examples: The examples of one split, in the family's task schema.
            tokenizer: A fast HuggingFace tokenizer.
            labels: The entry's label values.
            max_length: Most tokens per input.

        Returns:
            Features carrying the model inputs, a `labels` value and the head's bookkeeping.

        Raises:
            ValueError: If an example's label is not among `labels`.
        """

    @abc.abstractmethod
    def decode(
        self,
        examples: Sequence[Mapping[str, Any]],
        features: Sequence[Feature],
        logits: Sequence[np.ndarray],
        labels: Sequence[Any],
    ) -> list[Any]:
        """Turn the model's logits into one prediction per example, in the family's prediction shape.

        Args:
            examples: The examples the features were encoded from.
            features: The features, in `encode` order.
            logits: The model's logits, one array per feature, padded along the token axis.
            labels: The entry's label values.

        Returns:
            One prediction per example, in example order, as the family's scorer reads it.
        """

    def score(
        self,
        examples: Sequence[Mapping[str, Any]],
        features: Sequence[Feature],
        logits: Sequence[np.ndarray],
        labels: Sequence[Any],
    ) -> dict[str, float]:
        """Decode the model's logits and score the predictions with the family's scorer.

        Args:
            examples: The examples the features were encoded from.
            features: The features, in `encode` order.
            logits: The model's logits, one array per feature, padded along the token axis.
            labels: The entry's label values.

        Returns:
            Scores by name, `primary_metric` among them.
        """
        return self.scorer(examples, self.decode(examples, features, logits, labels))

    def loss(self, network: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the training loss of one collated batch, the model's own loss by default.

        Args:
            network: The model `load_model` built.
            batch: A batch from `collate`, its `labels` included.

        Returns:
            The batch's mean loss, a scalar tensor.
        """
        return network(**batch).loss

    def forward_logits(self, network: torch.nn.Module, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the model's logits for a batch of model inputs, the model's own `logits` by default.

        Args:
            network: The model `load_model` built.
            inputs: The model inputs of a batch from `collate`, without its labels.

        Returns:
            The logits, one row per feature, as `decode` reads them.
        """
        return network(**inputs).logits

    def collate(self, features: Sequence[Feature], pad_id: int) -> dict[str, torch.Tensor]:
        """Pad a batch of features into tensors the model reads, `labels` included.

        Args:
            features: The batch.
            pad_id: The tokenizer's padding token id.

        Returns:
            The model inputs and the labels as tensors.
        """
        batch = {}
        for key in MODEL_INPUTS:
            if key in features[0]:
                batch[key] = torch.tensor(_pad_nested([f[key] for f in features], pad_id if key == "input_ids" else 0))
        labels = [f["labels"] for f in features]
        batch["labels"] = torch.tensor(_pad_nested(labels, IGNORE_INDEX) if isinstance(labels[0], list) else labels)
        return batch


# ===============================
# Registry
# ===============================

_HEADS: dict[str, TaskHead] = {}


def register_head(family: str, head: TaskHead) -> None:
    """Register the head that fine-tunes one task family.

    Args:
        family: The task family, the first segment of an entry key.
        head: The head.

    Raises:
        ValueError: If the family already has a head.
    """
    if family in _HEADS:
        raise ValueError(f"task family '{family}' already has a head: {type(_HEADS[family]).__name__}")
    _HEADS[family] = head


def get_head(family: str) -> TaskHead:
    """Return the head registered for a task family.

    Args:
        family: The task family, the first segment of an entry key.

    Returns:
        The head.

    Raises:
        KeyError: If no head is registered for the family.
    """
    if family not in _HEADS:
        raise KeyError(f"no task head for family '{family}'; registered: {available_heads()}")
    return _HEADS[family]


def available_heads() -> list[str]:
    """Return the task families with a registered head, sorted."""
    return sorted(_HEADS)


def label_index(labels: Sequence[Any], value: Any, example_id: Any) -> int:
    """Return the position of an example's label among the entry's labels.

    Args:
        labels: The entry's label values.
        value: The example's label.
        example_id: The example's id, for the error message.

    Returns:
        The label's index.

    Raises:
        ValueError: If the label is not among `labels`.
    """
    if value not in labels:
        raise ValueError(f"example {example_id!r} has label {value!r}, not among the entry's labels {list(labels)}")
    return list(labels).index(value)


def model_inputs(encoded: Mapping[str, Any], row: int | None = None) -> Feature:
    """Pick the model inputs out of a tokenizer's output.

    Args:
        encoded: The tokenizer's output.
        row: The row to take from a batched output, or `None` to take every value whole.

    Returns:
        The values of the `MODEL_INPUTS` keys the output has.
    """
    return {key: encoded[key] if row is None else encoded[key][row] for key in MODEL_INPUTS if key in encoded}


# ===============================
# Helpers
# ===============================


def _pad_nested(rows: list[Any], fill: int) -> list[Any]:
    """Pad every innermost list of a nested list to the longest one's length."""

    def innermost(value: list[Any]) -> list[list[Any]]:
        return [value] if not value or not isinstance(value[0], list) else [x for v in value for x in innermost(v)]

    def pad(value: list[Any], width: int) -> list[Any]:
        if value and isinstance(value[0], list):
            return [pad(v, width) for v in value]
        return value + [fill] * (width - len(value))

    width = max(len(row) for row in innermost(rows))
    return pad(rows, width)
