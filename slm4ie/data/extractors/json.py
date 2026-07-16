"""JSON-array extractor for the SLM4IE pipeline.

Reads files containing a single JSON array of records, each with a
text field (e.g. PoVeJMo-VeMo-Med). One Document is emitted per
record with non-empty text. A single top-level object is also
accepted and treated as a one-record array.

The text and id source fields are configurable through the
`metadata:` block in extract.yaml, matching the `jsonl` extractor,
so feeds that name their text or id field differently (e.g. `body`
and `uri`) can be read without a bespoke extractor:

    metadata:
      text_field: body          # default: "text"
      id_field: uri             # default: "doc_id"
      metadata_fields:          # default: every non-reserved field
        - url
        - title

When `metadata_fields` is omitted, every record field except the
text field, the id field, and the structural fields `paragraphs` and
`conll` is kept under `Document.metadata`. When given, only those
listed keys present on the record are kept. In both cases fields
whose value is None are dropped.

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
        text:        record[text_field] (records with empty/missing
                     text are skipped).
        source:      provided by caller.
        domain:      provided by caller.
        doc_id:      record[id_field] if present.
        metadata:    the configured metadata_fields, or every other
                     field except the text field, the id field, and
                     the structural fields; None values are dropped.
        annotations: not produced.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from slm4ie.data.extractors import BaseExtractor, register_extractor
from slm4ie.data.schema import Document

logger = logging.getLogger(__name__)

DEFAULT_TEXT_FIELD = "text"
DEFAULT_ID_FIELD = "doc_id"

# Structural fields never copied into Document.metadata, on top of the
# (configurable) text and id fields. Kept identical to the jsonl
# extractor so the two stay consistent.
_STRUCTURAL_FIELDS = {"paragraphs", "conll"}


class JsonExtractor(BaseExtractor):
    """Extracts Documents from JSON files containing a top-level array.

    Recursively discovers .json files under the given directory.
    Each file is expected to be a JSON array of objects; non-array
    top-level structures are skipped with a warning. One Document is
    produced per array element with a non-empty text field. The text
    field, id field, and the set of record fields kept as metadata
    are configurable through the `metadata:` config block (see the
    module docstring).
    """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Document]:
        """Yield Documents from all JSON array files under input_dir.

        Args:
            input_dir (Path): Directory containing .json files
                (searched recursively).
            source (str): Dataset key assigned to every Document.
            domain (str): Domain label assigned to every Document.
            metadata (Optional[Dict[str, Any]]): Optional `metadata:`
                config block. Recognized keys: `text_field` (record
                field copied into `text`, default `text`), `id_field`
                (record field kept as `doc_id`, default `doc_id`), and
                `metadata_fields` (explicit whitelist of record fields
                to keep as metadata; when omitted, every non-structural
                field is kept).

        Yields:
            Document: One document per record with non-empty text.
        """
        cfg = metadata or {}
        text_field = str(cfg.get("text_field", DEFAULT_TEXT_FIELD))
        id_field = str(cfg.get("id_field", DEFAULT_ID_FIELD))
        metadata_fields_raw = cfg.get("metadata_fields")
        metadata_fields: Optional[List[str]] = (
            [str(f) for f in metadata_fields_raw]
            if metadata_fields_raw is not None
            else None
        )
        files = sorted(p for p in input_dir.rglob("*.json") if p.is_file())

        for filepath in files:
            yield from self._parse_file(
                filepath, source, domain, text_field, id_field, metadata_fields
            )

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
        text_field: str,
        id_field: str,
        metadata_fields: Optional[List[str]],
    ) -> Iterator[Document]:
        """Parse one JSON array file and yield Documents.

        Args:
            filepath (Path): Path to the JSON file.
            source (str): Dataset key.
            domain (str): Domain label.
            text_field (str): Record field copied into `text`.
            id_field (str): Record field kept as `doc_id`.
            metadata_fields (Optional[List[str]]): Explicit whitelist of
                record fields to keep as metadata, or None to keep every
                field except the text field, the id field, and the
                structural fields.

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

        excluded = {text_field, id_field} | _STRUCTURAL_FIELDS
        for record in records:
            if not isinstance(record, dict):
                continue

            text = record.get(text_field) or ""
            if not text:
                continue

            doc_id: Optional[str] = record.get(id_field)

            if metadata_fields is not None:
                doc_metadata: Dict[str, Any] = {
                    k: record[k]
                    for k in metadata_fields
                    if k in record and record[k] is not None
                }
            else:
                doc_metadata = {
                    k: v
                    for k, v in record.items()
                    if k not in excluded and v is not None
                }

            yield Document(
                text=text,
                source=source,
                domain=domain,
                doc_id=doc_id,
                metadata=doc_metadata,
            )


register_extractor("json", JsonExtractor)
