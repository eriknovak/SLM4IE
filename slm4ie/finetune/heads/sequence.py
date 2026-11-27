"""Sequence-classification heads: one label per example, and one relation per mention pair.

`SequenceHead` reads each family's example into one text, or a text pair,
through its own field function; the head is otherwise the same. It decodes
each example to its most likely label and scores with `score_labels`,
accuracy and macro F1 over the labels; macro F1 picks the checkpoint, since
several of these entries are unbalanced across their labels.
`EntitySentimentHead` marks the target's mentions in the text and crops a long
text around the first one.

`RelationHead` classifies every ordered pair of an example's gold mentions,
the two marked in the text, into a relation label or no relation. It decodes
to relations between the gold mentions, scored by `score_relations` with gold
entities.

Mentions are marked with plain-text brackets, not new special tokens, so the
encoder's vocabulary and embeddings stay unchanged.
"""

import functools
import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from transformers import AutoModelForSequenceClassification

from slm4ie.evaluation.scoring import score_labels, score_relations
from slm4ie.finetune.heads.base import Feature, TaskHead, label_index, model_inputs, register_head

# ===============================
# Constants
# ===============================

# the open and close markers of an entity-sentiment target and of a relation's arguments
TARGET_MARKERS = ("[", "]")
HEAD_MARKERS = ("[", "]")
TAIL_MARKERS = ("{", "}")
# the extra output of the relation head, after the relation labels
NO_RELATION = "no_relation"
# the fields of a gold mention that make a predicted relation argument
SPAN_KEYS = ("start", "end", "label")

# ===============================
# Heads
# ===============================

Fields = Callable[[Mapping[str, Any]], tuple[str, str | None]]


class SequenceHead(TaskHead):
    """Classifies one text or a text pair into one of the entry's labels.

    Args:
        fields: Reads an example into its first text and its second text, or `None` for a single text.
    """

    model_class = AutoModelForSequenceClassification
    primary_metric = "macro_f1"
    scorer = staticmethod(score_labels)

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

        texts = [self.texts(example, tokenizer, max_length) for example in examples]
        pairs = None if texts[0][1] is None else [second for _, second in texts]
        encoded = tokenizer([first for first, _ in texts], pairs, truncation=True, max_length=max_length)
        return [
            {
                **model_inputs(encoded, index),
                "labels": label_index(labels, example["label"], example["id"]),
            }
            for index, example in enumerate(examples)
        ]

    def texts(self, example: Mapping[str, Any], tokenizer: Any, max_length: int) -> tuple[str, str | None]:
        """Read an example into the text or text pair the head tokenizes, through the family's field function.

        Args:
            example: An example in the family's task schema.
            tokenizer: A fast HuggingFace tokenizer.
            max_length: Most tokens per input.

        Returns:
            The first text and the second text, or `None` for a single text.
        """
        return self.fields(example)

    def decode(
        self,
        examples: Sequence[Mapping[str, Any]],
        features: Sequence[Feature],
        logits: Sequence[np.ndarray],
        labels: Sequence[Any],
    ) -> list[Any]:
        """Pick the most likely label of each example.

        Args:
            examples: The examples.
            features: One feature per example.
            logits: One array of per-label logits per example.
            labels: The entry's label values.

        Returns:
            One label value per example.
        """
        return [labels[int(row.argmax())] for row in logits]


class EntitySentimentHead(SequenceHead):
    """Classifies the sentiment toward a target entity whose mentions are marked in the text.

    A text longer than `max_length` is cropped to a window around the target's
    first mention, so the model always sees a marked mention.
    """

    def __init__(self) -> None:
        """Read the example's text alone, which `texts` then marks and crops."""
        super().__init__(lambda ex: (ex["text"], None))

    def texts(self, example: Mapping[str, Any], tokenizer: Any, max_length: int) -> tuple[str, str | None]:
        """Mark every mention of the target in the example's text and crop it around the first mention.

        Args:
            example: An entity-sentiment example, `{"id", "text", "mentions", "entity_label", "label"}`.
            tokenizer: A fast HuggingFace tokenizer.
            max_length: Most tokens per input.

        Returns:
            The marked, cropped text and `None`.
        """
        mentions = example["mentions"]
        text, _ = self.fields(example)
        text = _mark(text, [(m["start"], m["end"], *TARGET_MARKERS) for m in mentions])
        if not mentions:
            return text, None

        # the first mention's range in the marked text, its own markers included
        opener, closer = TARGET_MARKERS
        first = min(mentions, key=lambda m: (m["start"], m["end"]))
        shift = len(opener) * sum(m["start"] < first["end"] for m in mentions) + len(closer) * sum(
            m["end"] <= first["end"] for m in mentions
        )
        return _crop(tokenizer, text, first["start"], first["end"] + shift, max_length), None


class RelationHead(TaskHead):
    """Classifies each ordered pair of an example's gold mentions into a relation label or no relation."""

    model_class = AutoModelForSequenceClassification
    primary_metric = "relation_f1"
    scorer = staticmethod(functools.partial(score_relations, gold_entities=True))

    def output_names(self, labels: Sequence[Any]) -> list[str]:
        """Return the relation labels, then the no-relation output.

        Args:
            labels: The entry's relation labels.

        Returns:
            One name per output.
        """
        return [str(label) for label in labels] + [NO_RELATION]

    def encode(
        self, examples: Sequence[Mapping[str, Any]], tokenizer: Any, labels: Sequence[Any], max_length: int
    ) -> list[Feature]:
        """Mark each ordered mention pair in its text, crop the text around the pair and tokenize it.

        A pair with a gold relation targets its relation label, the first one
        when the pair has several; every other pair targets no relation.

        Args:
            examples: Relation examples, each `{"id", "text", "entities", "relations"}`.
            tokenizer: A fast HuggingFace tokenizer.
            labels: The entry's relation labels.
            max_length: Most tokens per input.

        Returns:
            One feature per ordered pair of distinct mentions; `example` and
            `pair` name the example and the two mentions' positions in it.

        Raises:
            ValueError: If a gold relation's label is not among `labels`.
        """
        # mark and crop one text per ordered mention pair
        texts, rows = [], []
        marker_length = sum(len(marker) for marker in HEAD_MARKERS + TAIL_MARKERS)
        for index, example in enumerate(examples):
            gold = {}
            for link in example["relations"]:
                gold.setdefault((link["head"], link["tail"]), label_index(labels, link["label"], example["id"]))
            mentions = example["entities"]
            for pair in itertools.permutations(range(len(mentions)), 2):
                head, tail = (mentions[position] for position in pair)
                text = _mark(
                    example["text"],
                    [(head["start"], head["end"], *HEAD_MARKERS), (tail["start"], tail["end"], *TAIL_MARKERS)],
                )
                start, end = min(head["start"], tail["start"]), max(head["end"], tail["end"]) + marker_length
                texts.append(_crop(tokenizer, text, start, end, max_length))
                rows.append({"example": index, "pair": pair, "labels": gold.get((head["id"], tail["id"]), len(labels))})
        if not texts:
            return []

        # tokenize every pair's text at once
        encoded = tokenizer(texts, truncation=True, max_length=max_length)
        return [{**model_inputs(encoded, position), **row} for position, row in enumerate(rows)]

    def decode(
        self,
        examples: Sequence[Mapping[str, Any]],
        features: Sequence[Feature],
        logits: Sequence[np.ndarray],
        labels: Sequence[Any],
    ) -> list[list[dict[str, Any]]]:
        """Turn each pair's most likely output into a relation between its gold mentions.

        Args:
            examples: Relation examples.
            features: One feature per mention pair, in `encode` order.
            logits: One array of per-output logits per pair.
            labels: The entry's relation labels.

        Returns:
            One list of `{"head", "tail", "label"}` relations per example, each
            argument the gold mention's span with its gold entity label.
        """
        predicted = [[] for _ in examples]
        for feature, row in zip(features, logits, strict=True):
            output = int(row.argmax())
            if output == len(labels):
                continue
            mentions = examples[feature["example"]]["entities"]
            head, tail = ({key: mentions[position][key] for key in SPAN_KEYS} for position in feature["pair"])
            predicted[feature["example"]].append({"head": head, "tail": tail, "label": labels[output]})
        return predicted


# ===============================
# Helpers
# ===============================


def _mark(text: str, marks: Sequence[tuple[int, int, str, str]]) -> str:
    """Insert each mark's open marker at its start and its close marker at its end.

    At one position a close marker goes before an open one, so adjacent spans
    stay apart.
    """
    insertions = sorted(
        [(start, 1, opener) for start, _, opener, _ in marks] + [(end, 0, closer) for _, end, _, closer in marks]
    )
    pieces, cursor = [], 0
    for position, _, marker in insertions:
        pieces += [text[cursor:position], marker]
        cursor = position
    return "".join(pieces) + text[cursor:]


def _crop(tokenizer: Any, text: str, start: int, end: int, max_length: int) -> str:
    """Cut a text to a window of tokens that fits `max_length` and holds the characters `start` to `end`.

    The window is centred on those characters' tokens; when they alone exceed
    the window, it starts at their first token. A text that fits is returned
    whole.
    """
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if len(offsets) <= budget:
        return text

    # centre the window on the marked tokens, clamped to the text
    inside = [position for position, (first, last) in enumerate(offsets) if first < end and last > start]
    first, last = inside[0], inside[-1]
    spare = budget - (last - first + 1)
    low = first if spare < 0 else max(0, min(first - spare // 2, len(offsets) - budget))
    return text[offsets[low][0] : offsets[low + budget - 1][1]]


# ===============================
# Registration
# ===============================

register_head("sentiment", SequenceHead(lambda ex: (ex["text"], None)))
register_head("nli", SequenceHead(lambda ex: (ex["premise"], ex["hypothesis"])))
# the question comes first, so truncation cuts the passage
register_head("qa", SequenceHead(lambda ex: (ex["question"], ex["passage"])))
register_head("coref", SequenceHead(lambda ex: (ex["text"], f"{ex['span1']['text']}, {ex['span2']['text']}")))
register_head("wsd", SequenceHead(lambda ex: (f"{ex['word']}: {ex['sentence1']}", ex["sentence2"])))
register_head("entity_sentiment", EntitySentimentHead())
register_head("topic", SequenceHead(lambda ex: (ex["text"], None)))
register_head("relation", RelationHead())
