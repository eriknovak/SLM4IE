"""Unified output schema for the SLM4IE dataset extraction pipeline."""

import dataclasses
import json
from typing import Any, Dict, List, Optional


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
        data["forms"] = [t.form for t in tokens]
        data["lemmas"] = [t.lemma for t in tokens]
        data["upos"] = [t.upos for t in tokens]
        data["feats"] = [t.feats for t in tokens]
        data["sentences"] = self.annotations.sentences
        return json.dumps(data, ensure_ascii=False)
