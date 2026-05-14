"""Unified output schema for the SLM4IE dataset extraction pipeline."""

import dataclasses
import json
from typing import Any, Dict, List, Optional, TypedDict


@dataclasses.dataclass
class Token:
    """Represents a single annotated token.

    Attributes:
        form (str): Surface form of the token.
        lemma (Optional[str]): Lemma of the token.
        upos (Optional[str]): Universal POS tag.
        feats (Optional[str]): Morphological features.
    """

    form: str
    lemma: Optional[str] = None
    upos: Optional[str] = None
    feats: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dict representation, excluding None fields.

        Returns:
            Dict[str, Any]: Token fields with non-None values only.
        """
        return {
            k: v
            for k, v in dataclasses.asdict(self).items()
            if v is not None
        }


@dataclasses.dataclass
class Annotations:
    """Sentence-level annotations for a document.

    Attributes:
        tokens (List[Token]): Annotated tokens.
        sentences (List[List[int]]): Sentence boundaries as
            [start, end] index pairs (inclusive).
    """

    tokens: List[Token]
    sentences: List[List[int]]

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dict representation with tokens serialized.

        Returns:
            Dict[str, Any]: Annotations as a plain dict.
        """
        return {
            "tokens": [t.to_dict() for t in self.tokens],
            "sentences": self.sentences,
        }


@dataclasses.dataclass
class Document:
    """A single document in the unified SLM4IE corpus.

    Attributes:
        text (str): Raw text of the document.
        source (str): Dataset key (e.g. "ssj500k").
        domain (str): Text domain (e.g. "web", "parliamentary").
        doc_id (Optional[str]): Optional document identifier.
        metadata (Dict): Arbitrary metadata.
        annotations (Optional[Annotations]): Token/sentence
            annotations, if available.
    """

    text: str
    source: str
    domain: str
    doc_id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    annotations: Optional[Annotations] = None

    @property
    def uid(self) -> Optional[str]:
        """Globally unique example identifier across datasets.

        Combines `source` and `doc_id` so that documents from
        different corpora can never collide even if they reuse the
        same internal `doc_id`. Returns None when `doc_id` is
        absent (no stable per-document key from the source).

        Returns:
            Optional[str]: `"{source}:{doc_id}"` or None.
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
            str: A single JSON line with no trailing newline.
        """
        data: Dict[str, Any] = {
            "text": self.text,
            "source": self.source,
            "domain": self.domain,
        }
        if self.doc_id is not None:
            data["doc_id"] = self.doc_id
            data["uid"] = self.uid
        if self.metadata:
            data["metadata"] = self.metadata
        return json.dumps(data, ensure_ascii=False)

    def to_annotation_line(self) -> Optional[str]:
        """Serializes annotations as compact parallel arrays.

        Returns None if the document has no annotations. Output
        format uses parallel arrays (forms, lemmas, upos, feats)
        instead of one dict per token to reduce storage size.

        Returns:
            Optional[str]: A single JSON line, or None.
        """
        if self.annotations is None:
            return None

        tokens = self.annotations.tokens
        data: Dict[str, Any] = {}
        if self.doc_id is not None:
            data["doc_id"] = self.doc_id
            data["uid"] = self.uid
        data["forms"] = [t.form for t in tokens]
        data["lemmas"] = [t.lemma for t in tokens]
        data["upos"] = [t.upos for t in tokens]
        data["feats"] = [t.feats for t in tokens]
        data["sentences"] = self.annotations.sentences
        return json.dumps(data, ensure_ascii=False)


class NerExample(TypedDict):
    """A single NER example in GLiNER-compatible shape.

    Attributes:
        id: Document identifier.
        text: Raw text.
        spans: List of entity spans; each is
            ``{"start": int, "end": int, "label": str}``.
    """

    id: str
    text: str
    spans: List[Dict[str, Any]]


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
        label: Entailment label (e.g. ``"entailment"``,
            ``"neutral"``, ``"contradiction"``, or
            ``"not_entailment"`` depending on the task).
    """

    id: str
    premise: str
    hypothesis: str
    label: str


class QaExtractiveExample(TypedDict):
    """A single extractive question-answering example.

    Attributes:
        id: Example identifier.
        context: Passage text.
        question: Question text.
        answers: List of answer dicts of the form
            ``{"text": str, "start": int}``.
    """

    id: str
    context: str
    question: str
    answers: List[Dict[str, Any]]


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
            ``{"start": int, "end": int, "text": str}``.
        span2: Second span dict with the same shape as ``span1``.
        label: True if the two spans corefer.
    """

    id: str
    text: str
    span1: Dict[str, Any]
    span2: Dict[str, Any]
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
        question: Either ``"cause"`` or ``"effect"``.
        label: Index of the correct choice (``0`` or ``1``).
    """

    id: str
    premise: str
    choice1: str
    choice2: str
    question: str
    label: int
