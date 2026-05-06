"""JSON-array extractor for the SLM4IE pipeline.

Reads files containing a single JSON array of records, each with a
text field (e.g. PoVeJMo-VeMo-Med). One Document is emitted per
record with non-empty text. A single top-level object is also
accepted and treated as a one-record array.

Example:
    Raw input (data.json):

        [
          {
            "doc_id": "vemo.1",
            "text": "Bolnik je prišel z bolečinami.",
            "specialty": "interna",
            "year": 2023
          },
          {
            "doc_id": "vemo.2",
            "text": "Drugi opis primera."
          }
        ]

    Schema mapping:
        text:        record["text"] (records with empty/missing
                     text are skipped).
        source:      provided by caller.
        domain:      provided by caller.
        doc_id:      record["doc_id"] if present.
        metadata:    every other non-None field of the record.
        annotations: not produced.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from slm4ie.data.extractors import BaseExtractor, register_extractor
from slm4ie.data.schema import Document

logger = logging.getLogger(__name__)

_RESERVED_FIELDS = {"text", "doc_id"}


class JsonExtractor(BaseExtractor):
    """Extracts Documents from JSON files containing a top-level array.

    Recursively discovers .json files under the given directory.
    Each file is expected to be a JSON array of objects; non-array
    top-level structures are skipped with a warning. One Document is
    produced per array element with a non-empty text field.
    """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Yield Documents from all JSON array files under input_dir.

        Args:
            input_dir (Path): Directory containing .json files
                (searched recursively).
            source (str): Dataset key assigned to every Document.
            domain (str): Domain label assigned to every Document.

        Yields:
            Document: One document per record with non-empty text.
        """
        files = sorted(p for p in input_dir.rglob("*.json") if p.is_file())

        for filepath in files:
            yield from self._parse_file(filepath, source, domain)

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Parse one JSON array file and yield Documents.

        Args:
            filepath (Path): Path to the JSON file.
            source (str): Dataset key.
            domain (str): Domain label.

        Yields:
            Document: One document per valid record.
        """
        try:
            with filepath.open(encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON in %s: %s", filepath, exc)
            return

        records: List[Any]
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = [payload]
        else:
            logger.warning(
                "Skipping %s — top-level JSON is %s, expected array/object",
                filepath,
                type(payload).__name__,
            )
            return

        for record in records:
            if not isinstance(record, dict):
                continue

            text = record.get("text") or ""
            if not text:
                continue

            doc_id: Optional[str] = record.get("doc_id")
            metadata: Dict[str, Any] = {
                k: v
                for k, v in record.items()
                if k not in _RESERVED_FIELDS and v is not None
            }

            yield Document(
                text=text,
                source=source,
                domain=domain,
                doc_id=doc_id,
                metadata=metadata,
            )


register_extractor("json", JsonExtractor)
