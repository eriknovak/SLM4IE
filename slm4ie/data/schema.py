"""Unified output schema for the SLM4IE dataset extraction pipeline."""

import dataclasses
import json
from typing import Any, TypedDict

# ===============================
# Annotation classes
# ===============================


@dataclasses.dataclass
class Token:
    """Represents a single annotated token.

    Attributes:
        form (str): Surface form of the token.
        lemma (str | None): Lemma of the token.
        upos (str | None): Universal POS tag.
        feats (str | None): Morphological features.
        space_after (bool): Whether a space follows this token in the
            original text. False for e.g. a token directly followed by
            punctuation. Defaults to True.
    """

    form: str
    lemma: str | None = None
    upos: str | None = None
    feats: str | None = None
    space_after: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Returns a dict representation, excluding None fields.

        Returns:
            Token fields with non-None values only.
        """
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


def render_sentence(tokens: list["Token"]) -> str:
    """Reconstruct sentence text from tokens honoring `space_after`.

    Consecutive tokens are separated by a single space unless the left
    token's `space_after` is False. No trailing space is emitted after
    the final token regardless of its `space_after` value.

    Args:
        tokens: Sentence tokens in reading order.

    Returns:
        The reconstructed sentence text.
    """
    # join the token forms
    parts: list[str] = []
    last = len(tokens) - 1
    for i, token in enumerate(tokens):
        parts.append(token.form)
        if token.space_after and i < last:
            parts.append(" ")
    return "".join(parts)


@dataclasses.dataclass
class Annotations:
    """Token-level annotations for a document, with sentence boundaries.

    Tokens are the document's flat token sequence in reading order;
    sentence structure is recovered from `sentences`, which carries
    one inclusive `[start, end]` token-index pair per sentence.
    A two-sentence document with 5 + 7 tokens has
    `sentences == [[0, 4], [5, 11]]`.

    Attributes:
        tokens (list[Token]): Flat token sequence across all
            sentences in the document.
        sentences (list[list[int]]): One inclusive `[start, end]`
            token-index pair per sentence, in document order.
        spans (list[list[Any]] | None): Entity spans as
            `[start, end, label]` triples, where `start` and `end`
            are character offsets into the document text (`end`
            exclusive) and `label` is upper-case. An empty list is an
            annotated document with no entities; None means the source
            carries no `NER=` tags at all.
    """

    tokens: list[Token]
    sentences: list[list[int]]
    spans: list[list[Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Returns a dict representation with tokens serialized.

        Returns:
            Annotations as a plain dict; `spans` is
                included only when the document carries entity spans.
        """
        data: dict[str, Any] = {
            "tokens": [t.to_dict() for t in self.tokens],
            "sentences": self.sentences,
        }
        if self.spans is not None:
            data["spans"] = self.spans
        return data


# ===============================
# Document class
# ===============================


@dataclasses.dataclass
class Document:
    """A single document in the unified SLM4IE corpus.

    Attributes:
        text (str): Raw text of the document.
        source (str): Dataset key (e.g. "ssj500k").
        domain (str): Text domain (e.g. "web", "parliamentary").
        doc_id (str | None): Document identifier, unique within
            `source` and stable across re-extraction. Either the
            source's native id, when that is unique, or a positional
            id (see `slm4ie.data.extract.extractors.assembly.positional_doc_id`).
        native_id (str | None): The identifier the raw source
            itself carries for this document, verbatim. Kept for
            tracing a document back to its origin; not required to be
            unique. None when the source has no per-document id.
        metadata (dict): Arbitrary metadata.
        annotations (Annotations | None): Token/sentence
            annotations, if available.
    """

    text: str
    source: str
    domain: str
    doc_id: str | None = None
    native_id: str | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    annotations: Annotations | None = None

    @property
    def uid(self) -> str | None:
        """Globally unique example identifier across datasets.

        Combines `source` and `doc_id` so that documents from
        different corpora can never collide even if they reuse the
        same internal `doc_id`. Returns None when `doc_id` is
        absent (no stable per-document key from the source).

        Returns:
            `"{source}:{doc_id}"` or None.
        """
        if self.doc_id is None:
            return None
        return f"{self.source}:{self.doc_id}"

    def to_jsonl_line(self) -> str:
        """Serializes the document to a single JSON line (text only).

        Annotations are excluded — use to_annotation_line() for those.
        None fields and empty metadata are excluded from the output.
        Uses ensure_ascii=False to preserve Unicode characters.

        Returns:
            A single JSON line with no trailing newline.
        """
        data: dict[str, Any] = {
            "text": self.text,
            "source": self.source,
            "domain": self.domain,
        }
        if self.doc_id is not None:
            data["doc_id"] = self.doc_id
            data["uid"] = self.uid
        if self.native_id is not None:
            data["native_id"] = self.native_id
        if self.metadata:
            data["metadata"] = self.metadata
        return json.dumps(data, ensure_ascii=False)

    def to_annotation_line(self) -> str | None:
        """Serializes annotations as compact parallel arrays.

        Returns None if the document has no annotations. Output
        format uses parallel arrays (forms, lemmas, upos, feats,
        space_after) instead of one dict per token to reduce
        storage size. Entity spans are written under `spans` only
        when the source carries them (see `Annotations.spans`).

        Returns:
            A single JSON line, or None.
        """
        if self.annotations is None:
            return None

        # identify the document
        tokens = self.annotations.tokens
        data: dict[str, Any] = {}
        if self.doc_id is not None:
            data["doc_id"] = self.doc_id
            data["uid"] = self.uid

        # write the parallel arrays
        data["forms"] = [t.form for t in tokens]
        data["lemmas"] = [t.lemma for t in tokens]
        data["upos"] = [t.upos for t in tokens]
        data["feats"] = [t.feats for t in tokens]
        data["space_after"] = [t.space_after for t in tokens]
        data["sentences"] = self.annotations.sentences
        if self.annotations.spans is not None:
            data["spans"] = self.annotations.spans
        return json.dumps(data, ensure_ascii=False)


# ===============================
# Task example schemas
# ===============================


class NerExample(TypedDict):
    """A single NER example in GLiNER-compatible shape.

    Attributes:
        id: Document identifier.
        text: Raw text.
        spans: List of entity spans; each is
            `{"start": int, "end": int, "label": str}`.
    """

    id: str
    text: str
    spans: list[dict[str, Any]]


class SentimentExample(TypedDict):
    """A single sentiment-classification example.

    Attributes:
        id: Document identifier.
        text: Raw text.
        label: One of the dataset's declared `labels`.
    """

    id: str
    text: str
    label: str


class NliExample(TypedDict):
    """A single natural-language-inference example.

    Attributes:
        id: Example identifier.
        premise: Premise text.
        hypothesis: Hypothesis text.
        label: Entailment label (e.g. `"entailment"`,
            `"neutral"`, `"contradiction"`, or
            `"not_entailment"` depending on the task).
    """

    id: str
    premise: str
    hypothesis: str
    label: str


class QaBooleanExample(TypedDict):
    """A single boolean QA example (e.g. BoolQ, MultiRC).

    Attributes:
        id: Example identifier.
        passage: Passage text.
        question: Question text.
        label: Boolean answer.
    """

    id: str
    passage: str
    question: str
    label: bool


class CorefExample(TypedDict):
    """A single coreference / span-pair example (e.g. WSC).

    Attributes:
        id: Example identifier.
        text: Raw text containing both spans.
        span1: Span dict of the form
            `{"start": int, "end": int, "text": str}`.
        span2: Second span dict with the same shape as `span1`.
        label: True if the two spans corefer.
    """

    id: str
    text: str
    span1: dict[str, Any]
    span2: dict[str, Any]
    label: bool


class WsdExample(TypedDict):
    """A single word-sense disambiguation example (e.g. WiC).

    Attributes:
        id: Example identifier.
        sentence1: First sentence containing the target word.
        sentence2: Second sentence containing the target word.
        word: Target word whose sense is being compared.
        label: True if the word has the same sense in both
            sentences.
    """

    id: str
    sentence1: str
    sentence2: str
    word: str
    label: bool


class CommonsenseCopaExample(TypedDict):
    """A single COPA-style commonsense reasoning example.

    Attributes:
        id: Example identifier.
        premise: Premise text.
        choice1: First candidate continuation.
        choice2: Second candidate continuation.
        question: Either `"cause"` or `"effect"`.
        label: Index of the correct choice (`0` or `1`).
    """

    id: str
    premise: str
    choice1: str
    choice2: str
    question: str
    label: int
