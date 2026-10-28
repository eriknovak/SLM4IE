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
        doc_id:      record[id_field] if present (str-coerced).
        metadata:    the configured metadata_fields, or every other
                     field except the text field, the id field, and
                     the structural fields; None values are dropped.
        annotations: not produced.
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from slm4ie.data.extract.extractors import FileBasedExtractor, register_extractor
from slm4ie.data.extract.extractors.assembly import probe_doc_id, project_metadata
from slm4ie.data.schema import Document

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

DEFAULT_TEXT_FIELD = "text"
DEFAULT_ID_FIELD = "doc_id"

# structural fields never copied into Document.metadata; same set as the jsonl extractor
_STRUCTURAL_FIELDS = {"paragraphs", "conll"}

# ===============================
# Extractor class
# ===============================


class JsonExtractor(FileBasedExtractor):
    """Extracts Documents from JSON files containing a top-level array.

    Recursively discovers .json files under the given directory; the
    `include:` knob in extract.yaml narrows them, as for every
    file-based extractor. Each file is expected to be a JSON array of
    objects; non-array top-level structures are skipped with a warning.
    One Document is produced per array element with a non-empty text
    field. The text field, id field, and the set of record fields kept
    as metadata are configurable through the `metadata:` config block
    (see the module docstring).
    """

    def iter_input_files(self, input_dir: Path) -> list[Path]:
        """Return sorted .json files under input_dir.

        Args:
            input_dir: Directory searched recursively.

        Returns:
            Sorted .json file paths.
        """
        return sorted(p for p in input_dir.rglob("*.json") if p.is_file())

    def extract_files(
        self,
        files: list[Path],
        source: str,
        domain: str,
        input_dir: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Document]:
        """Yield Documents from the given JSON array files.

        Args:
            files: .json files to parse, in order.
            source: Dataset key assigned to every Document.
            domain: Domain label assigned to every Document.
            input_dir: Dataset root; unused, every record is
                self-contained.
            metadata: Optional `metadata:`
                config block. Recognized keys: `text_field` (record
                field copied into `text`, default `text`), `id_field`
                (record field kept as `doc_id`, default `doc_id`), and
                `metadata_fields` (explicit whitelist of record fields
                to keep as metadata; when omitted, every non-structural
                field is kept).

        Yields:
            One document per record with non-empty text.
        """
        del input_dir

        # read the field settings
        cfg = metadata or {}
        text_field = str(cfg.get("text_field", DEFAULT_TEXT_FIELD))
        id_field = str(cfg.get("id_field", DEFAULT_ID_FIELD))
        metadata_fields_raw = cfg.get("metadata_fields")
        metadata_fields: list[str] | None = (
            [str(f) for f in metadata_fields_raw] if metadata_fields_raw is not None else None
        )

        # parse each file
        for filepath in files:
            yield from self._parse_file(filepath, source, domain, text_field, id_field, metadata_fields)

    # =================================
    # Private methods
    # =================================

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
        text_field: str,
        id_field: str,
        metadata_fields: list[str] | None,
    ) -> Iterator[Document]:
        """Parse one JSON array file and yield Documents.

        Args:
            filepath: Path to the JSON file.
            source: Dataset key.
            domain: Domain label.
            text_field: Record field copied into `text`.
            id_field: Record field kept as `doc_id`.
            metadata_fields: Explicit whitelist of
                record fields to keep as metadata, or None to keep every
                field except the text field, the id field, and the
                structural fields.

        Yields:
            One document per valid record.
        """
        # load the payload
        try:
            with filepath.open(encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON in %s: %s", filepath, exc)
            return

        # normalize to a list of records
        records: list[Any]
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

        # emit one document per valid record
        excluded = {text_field, id_field} | _STRUCTURAL_FIELDS
        for record in records:
            if not isinstance(record, dict):
                continue

            text = record.get(text_field) or ""
            if not text:
                continue

            doc_id = probe_doc_id(record, [id_field])

            if metadata_fields is not None:
                doc_metadata = project_metadata(record, whitelist=metadata_fields)
            else:
                doc_metadata = project_metadata(record, exclude=excluded)

            yield Document(
                text=text,
                source=source,
                domain=domain,
                doc_id=doc_id,
                native_id=doc_id,
                metadata=doc_metadata,
            )


# ===============================
# Registration
# ===============================

register_extractor("json", JsonExtractor)
