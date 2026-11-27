"""Token-classification heads: span tagging for `ner` and `keyphrase`, word tagging for `pos`.

Every head works on character offsets. A NER or keyphrase example's text and
a POS example's words joined by spaces are tokenized with their offsets and
cut into windows of at most `max_length` tokens, so a long document is tagged
whole. Each token is then matched to the gold span or word its characters
fall in.

NER tags every token in BIO form (B-X opens a span of label X, I-X continues
it, O is outside), and predicted tags decode back into character spans scored
by `score_spans`. POS tags only the first token of each word, and the tag
predicted there decodes to the word's tag, scored by `score_tags`.

Keyphrase tagging is NER tagging with one tag type over every occurrence of a
gold keyphrase; its decoded phrases are ranked by mean tag probability and
scored by `score_keyphrases`.
"""

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from transformers import AutoModelForTokenClassification

from slm4ie.evaluation.scoring import normalise_text, score_keyphrases, score_spans, score_tags
from slm4ie.finetune.heads.base import IGNORE_INDEX, Feature, TaskHead, label_index, model_inputs, register_head

# ===============================
# Constants
# ===============================

OUTSIDE = "O"
NO_OWNER = -1
# the one tag type of the keyphrase head
KEYPHRASE_LABEL = "KEY"

# ===============================
# Heads
# ===============================


class SpanTaggingHead(TaskHead):
    """Tags NER spans in BIO form and scores them by strict span F1."""

    model_class = AutoModelForTokenClassification
    primary_metric = "span_f1"
    scorer = staticmethod(score_spans)

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

    def decode(
        self,
        examples: Sequence[Mapping[str, Any]],
        features: Sequence[Feature],
        logits: Sequence[np.ndarray],
        labels: Sequence[Any],
    ) -> list[list[dict[str, Any]]]:
        """Turn predicted BIO tags into character spans of each document.

        An I-X tag after a token outside a span of label X opens a new span. A
        span open at the end of a window continues into the next window of the
        same document.

        Args:
            examples: NER examples.
            features: The windows, in `encode` order.
            logits: One array of per-token logits per window.
            labels: The entry's entity labels.

        Returns:
            One list of `{"start", "end", "label"}` spans per document, empty
            for a document with no window or no predicted span.
        """
        decoded = _decode_bio(examples, features, logits, self.output_names(labels))
        return [[span for span, _ in spans] for spans in decoded]


class KeyphraseHead(SpanTaggingHead):
    """Tags keyphrase occurrences in BIO form with one tag type and ranks the phrases it finds.

    The family is label-free: the head tags one fixed type and reads no label set.
    """

    primary_metric = "f1_at_m"
    scorer = staticmethod(score_keyphrases)

    def output_names(self, labels: Sequence[Any] | None) -> list[str]:
        """Return O and the B- and I- tags of the one keyphrase type.

        Args:
            labels: The entry's label values; unused.

        Returns:
            The BIO tag names.
        """
        return super().output_names([KEYPHRASE_LABEL])

    def encode(
        self, examples: Sequence[Mapping[str, Any]], tokenizer: Any, labels: Sequence[Any] | None, max_length: int
    ) -> list[Feature]:
        """Tag every occurrence of each gold keyphrase in its text, then window and tag as NER does.

        An occurrence is a case-insensitive match on whole words; where two
        overlap, the longer keyphrase keeps its occurrence.

        Args:
            examples: Keyphrase examples, each `{"id", "text", "keyphrases"}`.
            tokenizer: A fast HuggingFace tokenizer.
            labels: The entry's label values; unused.
            max_length: Most tokens per window.

        Returns:
            One feature per window.
        """
        tagged = [
            {
                "id": example["id"],
                "text": example["text"],
                "spans": _occurrences(example["text"], example["keyphrases"]),
            }
            for example in examples
        ]
        return super().encode(tagged, tokenizer, [KEYPHRASE_LABEL], max_length)

    def decode(
        self,
        examples: Sequence[Mapping[str, Any]],
        features: Sequence[Feature],
        logits: Sequence[np.ndarray],
        labels: Sequence[Any] | None,
    ) -> list[list[str]]:
        """Rank each document's tagged phrases by their mean tag probability.

        A phrase's score is the mean probability of the tag predicted at each
        of its tokens. Phrases equal after the scorer's `normalise_text`
        merge, keeping the best-scored surface form and score; a phrase with
        no word left after it is dropped.

        Args:
            examples: Keyphrase examples.
            features: The windows, in `encode` order.
            logits: One array of per-token logits per window.
            labels: The entry's label values; unused.

        Returns:
            One list of keyphrases per document, best first.
        """
        ranked = []
        for example, spans in zip(examples, _decode_bio(examples, features, logits, self.output_names(labels))):
            scores, surfaces = {}, {}
            for span, probabilities in spans:
                phrase = example["text"][span["start"] : span["end"]]
                key, score = normalise_text(phrase), float(np.mean(probabilities))
                if key and score > scores.get(key, -math.inf):
                    scores[key], surfaces[key] = score, phrase
            ranked.append([surfaces[key] for key in sorted(scores, key=scores.get, reverse=True)])
        return ranked


class WordTaggingHead(TaskHead):
    """Tags each word on its first token and scores the tags by token accuracy."""

    model_class = AutoModelForTokenClassification
    primary_metric = "token_accuracy"
    scorer = staticmethod(score_tags)

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

    def decode(
        self,
        examples: Sequence[Mapping[str, Any]],
        features: Sequence[Feature],
        logits: Sequence[np.ndarray],
        labels: Sequence[Any],
    ) -> list[list[Any]]:
        """Read each word's tag off its first token.

        Args:
            examples: POS examples.
            features: The windows, in `encode` order.
            logits: One array of per-token logits per window.
            labels: The entry's tags.

        Returns:
            One tag per word of each sentence; `None` for a word with no token
            of its own, which the scorer counts as wrong.
        """
        predicted = [[None] * len(example["words"]) for example in examples]
        for feature, row in zip(features, logits):
            for position, word in feature["first_tokens"]:
                predicted[feature["example"]][word] = labels[int(row[position].argmax())]
        return predicted


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
        window = model_inputs(encoded, index)
        window.update(example=encoded["overflow_to_sample_mapping"][index], offsets=offsets)
        windows.append(window)
    return windows


def _decode_bio(
    examples: Sequence[Mapping[str, Any]],
    features: Sequence[Feature],
    logits: Sequence[np.ndarray],
    names: list[str],
) -> list[list[tuple[dict[str, Any], list[float]]]]:
    """Decode predicted BIO tags into character spans, each with its tokens' tag probabilities.

    Returns one list per document of `({"start", "end", "label"}, probabilities)`
    pairs, where `probabilities` holds the softmax probability of the tag
    predicted at each token of the span.
    """
    predicted, current = defaultdict(list), {}
    for feature, row in zip(features, logits):
        index, tokens = feature["example"], row[: len(feature["input_ids"])]
        probabilities = np.exp(tokens - tokens.max(-1, keepdims=True))
        probabilities = (probabilities / probabilities.sum(-1, keepdims=True)).max(-1)
        for tag, probability, offset in zip(tokens.argmax(-1), probabilities, feature["offsets"]):
            name = names[int(tag)]
            if offset is None:
                continue
            if name == OUTSIDE:
                current[index] = None
                continue
            prefix, label = name.split("-", 1)
            span = current.get(index)
            if prefix == "I" and span is not None and span[0]["label"] == label:
                span[0]["end"] = offset[1]
                span[1].append(float(probability))
                continue
            start, text = offset[0], examples[index]["text"]
            # some tokenizers count the space before a word into its token
            while start < offset[1] and text[start].isspace():
                start += 1
            current[index] = ({"start": start, "end": offset[1], "label": label}, [float(probability)])
            predicted[index].append(current[index])

    # one span list per document, windowless ones included
    return [predicted[index] for index in range(len(examples))]


def _occurrences(text: str, phrases: Sequence[str]) -> list[dict[str, Any]]:
    """Find the non-overlapping whole-word occurrences of phrases in a text, longer phrases first."""
    taken = []
    for phrase in sorted(
        {phrase.strip() for phrase in phrases if phrase.strip()}, key=lambda phrase: (-len(phrase), phrase)
    ):
        for match in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, flags=re.IGNORECASE):
            if all(match.end() <= start or end <= match.start() for start, end in taken):
                taken.append(match.span())
    return [{"start": start, "end": end, "label": KEYPHRASE_LABEL} for start, end in sorted(taken)]


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
register_head("keyphrase", KeyphraseHead())
