"""JSONL format extractor for CLASSLA-web annotated data."""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from slm4ie.data.extractors import BaseExtractor, register_extractor
from slm4ie.data.schema import Annotations, Document, Token

logger = logging.getLogger(__name__)

_RESERVED_FIELDS = {"text", "paragraphs", "doc_id"}


def _parse_tokens_from_paragraphs(
    paragraphs: List[Dict],
) -> Optional[Annotations]:
    """Parse nested paragraphs/sentences/tokens into Annotations.

    Flattens all paragraphs → sentences → tokens into a single list,
    tracking sentence boundaries as [start, end] index pairs.

    Args:
        paragraphs (List[Dict]): List of paragraph dicts, each containing
            a ``sentences`` key with sentence dicts, each containing a
            ``tokens`` key with token dicts.

    Returns:
        Optional[Annotations]: Parsed annotations, or None if no tokens
            were found.
    """
    tokens: List[Token] = []
    sentences: List[List[int]] = []

    for paragraph in paragraphs:
        for sentence in paragraph.get("sentences", []):
            raw_tokens = sentence.get("tokens", [])
            if not raw_tokens:
                continue
            start = len(tokens)
            for tok in raw_tokens:
                tokens.append(
                    Token(
                        form=tok["form"],
                        lemma=tok.get("lemma"),
                        upos=tok.get("upos"),
                        feats=tok.get("feats"),
                    )
                )
            end = len(tokens) - 1
            sentences.append([start, end])

    if not tokens:
        return None

    return Annotations(tokens=tokens, sentences=sentences)


class JsonlExtractor(BaseExtractor):
    """Extracts Documents from CLASSLA-web JSONL files.

    One :class:`~slm4ie.data.schema.Document` is produced per JSONL line
    with a non-empty ``text`` field. Discovers all ``*.jsonl`` files in
    the given directory (sorted).
    """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Yield Documents from all JSONL files in *input_dir*.

        Args:
            input_dir (Path): Directory containing ``.jsonl`` files.
            source (str): Dataset key assigned to every Document.
            domain (str): Domain label assigned to every Document.

        Yields:
            Document: One document per valid JSONL line.
        """
        files: List[Path] = sorted(input_dir.glob("*.jsonl"))

        for filepath in files:
            yield from self._parse_file(filepath, source, domain)

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Parse a single JSONL file and yield Documents.

        Args:
            filepath (Path): Path to the JSONL file.
            source (str): Dataset key.
            domain (str): Domain label.

        Yields:
            Document: One document per valid line.
        """
        with filepath.open(encoding="utf-8") as fh:
            for lineno, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Invalid JSON on line %d of %s: %s",
                        lineno,
                        filepath,
                        exc,
                    )
                    continue

                text = record.get("text", "")
                if not text:
                    continue

                doc_id: Optional[str] = record.get("doc_id")

                paragraphs = record.get("paragraphs")
                annotations: Optional[Annotations] = None
                if paragraphs is not None:
                    annotations = _parse_tokens_from_paragraphs(paragraphs)

                metadata = {
                    k: v
                    for k, v in record.items()
                    if k not in _RESERVED_FIELDS
                }

                yield Document(
                    text=text,
                    source=source,
                    domain=domain,
                    doc_id=doc_id,
                    metadata=metadata,
                    annotations=annotations,
                )


register_extractor("jsonl", JsonlExtractor)
