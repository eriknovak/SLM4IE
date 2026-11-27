"""TSV format extractor for tab-separated corpora with a header row.

Reads .tsv and .txt files whose first row names the columns. One Document is
produced per data row with a non-empty text column. Fields are split on
tabs only: quote characters carry no meaning (`csv.QUOTE_NONE`), so a
stray `"` in a text never swallows the following rows. The csv reader strips
the line terminator itself, so CRLF line endings are read like LF.

The column names are configurable through the `metadata:` block in
extract.yaml, like the jsonl extractor:

    metadata:
      text_field: content        # default: "text"
      id_field: nid              # default: "doc_id"
      metadata_fields:           # default: every other column
        - title
        - date

Schema mapping:
    text:      row[text_field] (rows with empty/missing text are skipped).
    doc_id:    row[id_field] if present and non-empty, else the
               positional id `<unit>:<ordinal>` (ordinal counts every
               data row, skipped ones included).
    metadata:  the configured metadata_fields, or every column except the
               text and id columns; empty cells are dropped.
"""

import csv
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from slm4ie.data.extract.extractors import FileBasedExtractor, register_extractor
from slm4ie.data.extract.extractors.assembly import positional_doc_id, probe_doc_id, project_metadata, relative_unit
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
DELIMITER = "\t"

# ===============================
# Helper functions
# ===============================


def _blank_to_none(value: Any) -> Any:
    """Map an empty cell to None so `project_metadata` drops it.

    Args:
        value: A cell value.

    Returns:
        None for an empty string, else the value unchanged.
    """
    return None if value == "" else value


# ===============================
# Extractor class
# ===============================


class TsvExtractor(FileBasedExtractor):
    """Extracts Documents from tab-separated files with a header row."""

    def iter_input_files(self, input_dir: Path) -> list[Path]:
        """Return the sorted .tsv and .txt files under input_dir.

        Args:
            input_dir: Directory searched recursively.

        Returns:
            The .tsv and .txt files, deduplicated, in sorted order.
        """
        return sorted({*input_dir.rglob("*.tsv"), *input_dir.rglob("*.txt")})

    def extract_files(
        self,
        files: list[Path],
        source: str,
        domain: str,
        input_dir: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Document]:
        """Yield Documents parsed from the given tab-separated files.

        Args:
            files: Files to parse, in order.
            source: Dataset key assigned to every Document.
            domain: Domain label assigned to every Document.
            input_dir: Dataset root, the base of the positional id's unit.
            metadata: Optional `metadata:` block with `text_field`,
                `id_field` and `metadata_fields` (see the module docstring).

        Yields:
            Document: One document per row with non-empty text.
        """
        # read the column settings
        cfg = metadata or {}
        text_field = str(cfg.get("text_field", DEFAULT_TEXT_FIELD))
        id_field = str(cfg.get("id_field", DEFAULT_ID_FIELD))
        raw_fields = cfg.get("metadata_fields")
        metadata_fields = [str(f) for f in raw_fields] if raw_fields is not None else None

        # lift the field size limit: news articles exceed the default 128 KiB
        csv.field_size_limit(sys.maxsize)

        # parse every file
        for filepath in files:
            unit = relative_unit(filepath, input_dir)
            with filepath.open(encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh, delimiter=DELIMITER, quoting=csv.QUOTE_NONE)
                for ordinal, row in enumerate(reader):
                    # skip rows without text; cells beyond the header have no column
                    row.pop(None, None)
                    text = row.get(text_field)
                    if not text:
                        continue

                    # build the document parts
                    native_id = probe_doc_id(row, [id_field])
                    doc_id = native_id if native_id is not None else positional_doc_id(unit, ordinal)

                    if metadata_fields is not None:
                        doc_metadata = project_metadata(row, whitelist=metadata_fields, value_transform=_blank_to_none)
                    else:
                        doc_metadata = project_metadata(
                            row, exclude={text_field, id_field}, value_transform=_blank_to_none
                        )

                    # emit the document
                    yield Document(
                        text=text,
                        source=source,
                        domain=domain,
                        doc_id=doc_id,
                        native_id=native_id,
                        metadata=doc_metadata,
                    )


register_extractor("tsv", TsvExtractor)
