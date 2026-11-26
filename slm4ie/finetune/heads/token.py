"""Token-classification heads: span tagging for `ner` and word tagging for `pos`.

Both heads work on character offsets. A NER example's text and a POS example's
words joined by spaces are tokenized with their offsets and cut into windows of
at most `max_length` tokens, so a long document is tagged whole. Each token is
then matched to the gold span or word its characters fall in.

NER tags every token in BIO form (B-X opens a span of label X, I-X continues
it, O is outside), and predicted tags turn back into character spans scored by
strict span F1. POS tags only the first token of each word, and the tag
predicted there is the word's tag, scored by token accuracy.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from transformers import AutoModelForTokenClassification

from slm4ie.evaluation.scoring import Counts, macro_f1, score_document, token_accuracy
from slm4ie.finetune.heads.base import IGNORE_INDEX, MODEL_INPUTS, Feature, TaskHead, label_index, register_head

# ===============================
# Constants
# ===============================

OUTSIDE = "O"
NO_OWNER = -1

# ===============================
# Heads
# ===============================


class SpanTaggingHead(TaskHead):
    """Tags NER spans in BIO form and scores them by strict span F1."""

    model_class = AutoModelForTokenClassification
    primary_metric = "span_f1"

    def output_names(self, labels: Sequence[Any]) -> list[str]:
        """Return O, then B- and I- of every entity label.

        Args:
            labels: The entry's entity labels.

        Returns:
            The BIO tag names.
        """
        return [OUTSIDE] + [f"{prefix}-{label}" for label in labels for prefix in ("B", "I")]

    def encode(
        self, examples: Sequence[Mapping[str, Any]], tokenizer: Any, labels: Sequence[Any], max_length: int
    ) -> list[Feature]:
        """Window each document and tag every token in BIO form.

        Args:
            examples: NER examples, each `{"id", "text", "spans"}`.
            tokenizer: A fast HuggingFace tokenizer.
            labels: The entry's entity labels.
            max_length: Most tokens per window.

        Returns:
            One feature per window.

        Raises:
            ValueError: If a gold span's label is not among `labels`.
        """
        # map each document's characters to the gold span they fall in
        tag_ids = {name: index for index, name in enumerate(self.output_names(labels))}
        owners = []
        for example in examples:
            for span in example["spans"]:
                label_index(labels, span["label"], example["id"])
            owners.append(_owners(len(example["text"]), [(s["start"], s["end"]) for s in example["spans"]]))

        # tag every window's tokens; a span cut at a window edge carries on
        windows = _tokenize_windows(tokenizer, [example["text"] for example in examples], max_length)
        previous = defaultdict(lambda: NO_OWNER)
        features = []
        for window in windows:
            index, tags = window["example"], []
            for offset in window["offsets"]:
                if offset is None:
                    tags.append(IGNORE_INDEX)
                    continue
                owner = _token_owner(owners[index], offset)
                if owner == NO_OWNER:
                    tags.append(tag_ids[OUTSIDE])
                else:
                    prefix = "I" if owner == previous[index] else "B"
                    tags.append(tag_ids[f"{prefix}-{examples[index]['spans'][owner]['label']}"])
                previous[index] = owner
            features.append({**window, "labels": tags})
        return features

    def score(
        self,
        examples: Sequence[Mapping[str, Any]],
        features: Sequence[Feature],
        logits: Sequence[np.ndarray],
        labels: Sequence[Any],
    ) -> dict[str, float]:
        """Turn predicted BIO tags into spans and score them by strict span F1.

        An I-X tag after a token outside a span of label X opens a new span. A
        span open at the end of a window continues into the next window of the
        same document.

        Args:
            examples: NER examples.
            features: The windows, in `encode` order.
            logits: One array of per-token logits per window.
            labels: The entry's entity labels.

        Returns:
            `span_f1`, `span_relaxed_f1` and `span_partial_f1`, the micro F1
            pooled over documents under the strict, relaxed and partial-overlap
            match rules, and `span_macro_f1`, the strict F1 averaged over labels.
        """
        # decode each window's tags into character spans of its document
        names = self.output_names(labels)
        predicted, current = defaultdict(list), {}
        for feature, row in zip(features, logits):
            index = feature["example"]
            for tag, offset in zip(row[: len(feature["input_ids"])].argmax(-1), feature["offsets"]):
                name = names[int(tag)]
                if offset is None:
                    continue
                if name == OUTSIDE:
                    current[index] = None
                    continue
                prefix, label = name.split("-", 1)
                span = current.get(index)
                if prefix == "I" and span is not None and span["label"] == label:
                    span["end"] = offset[1]
                    continue
                start, text = offset[0], examples[index]["text"]
                # some tokenizers count the space before a word into its token
                while start < offset[1] and text[start].isspace():
                    start += 1
                current[index] = {"start": start, "end": offset[1], "label": label}
                predicted[index].append(current[index])

        # score every document, windowless ones included
        scores = [score_document(predicted[index], example["spans"]) for index, example in enumerate(examples)]
        strict = sum((score.strict for score in scores), Counts())
        relaxed = sum((score.relaxed for score in scores), Counts())
        partial = sum((score.partial for score in scores), Counts())
        return {
            "span_f1": strict.f1,
            "span_relaxed_f1": relaxed.f1,
            "span_partial_f1": partial.f1,
            "span_macro_f1": macro_f1(scores),
        }


class WordTaggingHead(TaskHead):
    """Tags each word on its first token and scores the tags by token accuracy."""

    model_class = AutoModelForTokenClassification
    primary_metric = "token_accuracy"

    def encode(
        self, examples: Sequence[Mapping[str, Any]], tokenizer: Any, labels: Sequence[Any], max_length: int
    ) -> list[Feature]:
        """Join each sentence's words, window them and tag each word's first token.

        Args:
            examples: POS examples, each `{"id", "words", "tags"}`.
            tokenizer: A fast HuggingFace tokenizer.
            labels: The entry's tags.
            max_length: Most tokens per window.

        Returns:
            One feature per window; `first_tokens` pairs each tagged token's
            position with its word's index.

        Raises:
            ValueError: If a gold tag is not among `labels`.
        """
        # join the words and map each sentence's characters to its words
        texts, owners, tag_ids = [], [], []
        for example in examples:
            ranges, start = [], 0
            for word in example["words"]:
                ranges.append((start, start + len(word)))
                start += len(word) + 1
            texts.append(" ".join(example["words"]))
            owners.append(_owners(len(texts[-1]), ranges))
            tag_ids.append([label_index(labels, tag, example["id"]) for tag in example["tags"]])

        # tag the first token of each word, once across windows
        tagged = defaultdict(set)
        features = []
        for window in _tokenize_windows(tokenizer, texts, max_length):
            index, tags, first_tokens = window["example"], [], []
            for position, offset in enumerate(window["offsets"]):
                owner = NO_OWNER if offset is None else _token_owner(owners[index], offset)
                if owner == NO_OWNER or owner in tagged[index]:
                    tags.append(IGNORE_INDEX)
                    continue
                tagged[index].add(owner)
                tags.append(tag_ids[index][owner])
                first_tokens.append((position, owner))
            features.append({**window, "labels": tags, "first_tokens": first_tokens})
        return features

    def score(
        self,
        examples: Sequence[Mapping[str, Any]],
        features: Sequence[Feature],
        logits: Sequence[np.ndarray],
        labels: Sequence[Any],
    ) -> dict[str, float]:
        """Read each word's tag off its first token and score by token accuracy.

        A word with no token of its own is never right.

        Args:
            examples: POS examples.
            features: The windows, in `encode` order.
            logits: One array of per-token logits per window.
            labels: The entry's tags.

        Returns:
            `token_accuracy`, pooled over every word of every sentence.
        """
        predicted = [[None] * len(example["words"]) for example in examples]
        for feature, row in zip(features, logits):
            for position, word in feature["first_tokens"]:
                predicted[feature["example"]][word] = labels[int(row[position].argmax())]
        return {"token_accuracy": token_accuracy(predicted, [example["tags"] for example in examples])}


# ===============================
# Helpers
# ===============================


def _tokenize_windows(tokenizer: Any, texts: list[str], max_length: int) -> list[Feature]:
    """Tokenize texts with offsets into windows of at most `max_length` tokens.

    Returns one feature per window: its model inputs, `example` (the index of
    its text) and `offsets` (each token's character range, `None` for special
    tokens).
    """
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    windows = []
    for index in range(len(encoded["input_ids"])):
        sequence_ids = encoded.sequence_ids(index)
        offsets = [None if seq is None else tuple(o) for o, seq in zip(encoded["offset_mapping"][index], sequence_ids)]
        window = {key: encoded[key][index] for key in MODEL_INPUTS if key in encoded}
        window.update(example=encoded["overflow_to_sample_mapping"][index], offsets=offsets)
        windows.append(window)
    return windows


def _owners(length: int, ranges: list[tuple[int, int]]) -> list[int]:
    """Map each character of a text to the index of the range it falls in, or `NO_OWNER`."""
    owners = [NO_OWNER] * length
    for index, (start, end) in enumerate(ranges):
        owners[start:end] = [index] * len(owners[start:end])
    return owners


def _token_owner(owners: list[int], offset: tuple[int, int]) -> int:
    """Return the range of the first owned character of a token, or `NO_OWNER`."""
    return next((owner for owner in owners[offset[0] : offset[1]] if owner != NO_OWNER), NO_OWNER)


# ===============================
# Registration
# ===============================

register_head("ner", SpanTaggingHead())
register_head("pos", WordTaggingHead())
