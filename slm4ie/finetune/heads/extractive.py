"""The extractive question-answering head: start and end prediction over a context (`extractive_qa`).

Each question is paired with its context, and a context too long for one
input is cut into overlapping windows (the tokenizer's stride), each mapped
back to the context's characters through its token offsets. The model scores
every token as an answer start and an answer end. An example's answer is the
best-scored valid span over its windows, or the empty string, meaning no
answer, when the score of the first token beats it (the SQuAD 2.0 rule).
Answers are scored by `score_extractive_qa`; its F1 picks the checkpoint.

The family is label-free: the head reads no label set.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering

from slm4ie.evaluation.scoring import score_extractive_qa
from slm4ie.finetune.heads.base import Feature, TaskHead, model_inputs, register_head

# ===============================
# Constants
# ===============================

# the token whose start and end scores stand for no answer
NULL_POSITION = 0
# the most tokens an answer may span, as the HuggingFace QA examples set it
MAX_ANSWER_TOKENS = 30
# windows overlap by this share of `max_length`, as the HuggingFace examples' 128 of 384
STRIDE_SHARE = 0.25
# a question takes at most this share of a window's tokens, the rest is context
QUESTION_SHARE = 0.5
# the sequence id of the context in a question-context pair
CONTEXT_SEQUENCE = 1

# ===============================
# Head
# ===============================


class ExtractiveQaHead(TaskHead):
    """Predicts the start and end token of each question's answer in its context."""

    model_class = AutoModelForQuestionAnswering
    primary_metric = "f1"
    scorer = staticmethod(score_extractive_qa)

    def load_model(self, encoder: str, labels: Sequence[Any] | None) -> torch.nn.Module:
        """Build a question-answering model, which scores each token as a start and an end.

        Args:
            encoder: The encoder, by HuggingFace name or local path.
            labels: The entry's label values; unused, the family is label-free.

        Returns:
            The model, on the CPU.
        """
        return self.model_class.from_pretrained(encoder)

    def encode(
        self, examples: Sequence[Mapping[str, Any]], tokenizer: Any, labels: Sequence[Any] | None, max_length: int
    ) -> list[Feature]:
        """Pair each question with its context, cut into overlapping windows of `max_length` tokens.

        A question longer than `QUESTION_SHARE` of the window is cut to it. A
        window's target is the first gold answer's first and last token, or the
        first token when the question has no answer or the answer is not
        wholly inside the window.

        Args:
            examples: Extractive QA examples, each `{"id", "context", "question", "answers"}`.
            tokenizer: A fast HuggingFace tokenizer.
            labels: The entry's label values; unused.
            max_length: Most tokens per window, question included.

        Returns:
            One feature per window; `example` names its example, `offsets`
            holds each context token's character range (`None` elsewhere) and
            `labels` the target start and end token.
        """
        special = tokenizer.num_special_tokens_to_add(pair=True)
        features = []
        for index, example in enumerate(examples):
            # cut a long question, then keep the stride below the context room it leaves
            question, question_tokens = _cut(tokenizer, example["question"], math.floor(max_length * QUESTION_SHARE))
            room = max_length - special - question_tokens
            encoded = tokenizer(
                question,
                example["context"],
                truncation="only_second",
                max_length=max_length,
                stride=min(math.floor(max_length * STRIDE_SHARE), room // 2),
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
            )

            # map each window's context tokens to characters and find its target
            answer = example["answers"][0] if example["answers"] else None
            for window in range(len(encoded["input_ids"])):
                offsets = [
                    tuple(offset) if sequence == CONTEXT_SEQUENCE else None
                    for offset, sequence in zip(encoded["offset_mapping"][window], encoded.sequence_ids(window))
                ]
                features.append(
                    {
                        **model_inputs(encoded, window),
                        "example": index,
                        "offsets": offsets,
                        "labels": _answer_positions(offsets, answer),
                    }
                )
        return features

    def decode(
        self,
        examples: Sequence[Mapping[str, Any]],
        features: Sequence[Feature],
        logits: Sequence[np.ndarray],
        labels: Sequence[Any] | None,
    ) -> list[str]:
        """Pick each question's best valid span over its windows, or no answer.

        A valid span starts and ends on context tokens, ends at or after its
        start and spans at most `MAX_ANSWER_TOKENS` tokens. The no-answer score
        is the lowest first-token score over the example's windows.

        Args:
            examples: Extractive QA examples.
            features: The windows, in `encode` order.
            logits: One array per window, a start and an end logit per token.
            labels: The entry's label values; unused.

        Returns:
            One answer string per example, cut from its context; the empty
            string when no answer beats the no-answer score.
        """
        best = [(-math.inf, "")] * len(examples)
        null = [math.inf] * len(examples)
        for feature, row in zip(features, logits, strict=True):
            # score every valid span of the window
            index, offsets = feature["example"], feature["offsets"]
            length = len(feature["input_ids"])
            start, end = row[:length, 0], row[:length, 1]
            null[index] = min(null[index], float(start[NULL_POSITION] + end[NULL_POSITION]))
            in_context = np.array([offset is not None for offset in offsets])
            gap = np.arange(length)[None, :] - np.arange(length)[:, None]
            valid = (gap >= 0) & (gap < MAX_ANSWER_TOKENS) & in_context[:, None] & in_context[None, :]
            if not valid.any():
                continue

            # keep the window's best span when it beats the example's best so far
            scores = np.where(valid, start[:, None] + end[None, :], -np.inf)
            first, last = np.unravel_index(scores.argmax(), scores.shape)
            if scores[first, last] > best[index][0]:
                text = examples[index]["context"][offsets[first][0] : offsets[last][1]]
                best[index] = (float(scores[first, last]), text.strip())
        return [text if score >= null[index] else "" for index, (score, text) in enumerate(best)]

    def forward_logits(self, network: torch.nn.Module, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the start and end logits of every token, stacked on a last axis of two.

        Args:
            network: The model `load_model` built.
            inputs: The model inputs of a batch.

        Returns:
            A tensor of shape `(batch, tokens, 2)`.
        """
        output = network(**inputs)
        return torch.stack([output.start_logits, output.end_logits], dim=-1)

    def collate(self, features: Sequence[Feature], pad_id: int) -> dict[str, torch.Tensor]:
        """Pad a batch into tensors, the targets as the model's start and end positions.

        Args:
            features: The batch.
            pad_id: The tokenizer's padding token id.

        Returns:
            The model inputs with `start_positions` and `end_positions`.
        """
        batch = super().collate(features, pad_id)
        positions = batch.pop("labels")
        batch["start_positions"], batch["end_positions"] = positions[:, 0], positions[:, 1]
        return batch


# ===============================
# Helpers
# ===============================


def _cut(tokenizer: Any, text: str, most: int) -> tuple[str, int]:
    """Cut a text after its first `most` tokens; return it and its token count."""
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    if len(offsets) <= most:
        return text, len(offsets)
    return text[: offsets[most - 1][1]], most


def _answer_positions(offsets: list[tuple[int, int] | None], answer: Mapping[str, Any] | None) -> list[int]:
    """Return the first and last token of an answer wholly inside a window, else the null position twice."""
    if answer is None:
        return [NULL_POSITION, NULL_POSITION]
    start, end = answer["start"], answer["start"] + len(answer["text"])
    inside = [position for position, o in enumerate(offsets) if o is not None and o[0] < end and o[1] > start]
    if not inside or offsets[inside[0]][0] > start or offsets[inside[-1]][1] < end:
        return [NULL_POSITION, NULL_POSITION]
    return [inside[0], inside[-1]]


# ===============================
# Registration
# ===============================

register_head("extractive_qa", ExtractiveQaHead())
