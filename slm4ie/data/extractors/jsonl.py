"""JSONL format extractor for line-delimited JSON corpora.

Reads .jsonl files where each line is a JSON object with a text
field and optional nested paragraphs -> sentences -> tokens
annotations (CLASSLA-web style).

The text and id source fields are configurable through the
`metadata:` block in extract.yaml, so feeds that name their text or
id field differently (e.g. a news dump using `body` and `uri`) can
be read without a bespoke extractor:

    metadata:
      text_field: body          # default: "text"
      id_field: uri             # default: "doc_id"
      metadata_fields:          # default: every non-reserved field
        - url
        - title
        - dateTime
        - source

When `metadata_fields` is omitted, every record field except the
text field, the id field, `paragraphs`, and `conll` is kept under
`Document.metadata` (the original CLASSLA-web behavior). When given,
only those listed keys present on the record are kept.

Example:
    One line of a .jsonl file (formatted for readability):

        {
          "doc_id": "d1",
          "text": "Dober dan.",
          "url": "https://example.com",
          "paragraphs": [
            {
              "sentences": [
                {
                  "tokens": [
                    {"form": "Dober", "lemma": "dober",
                     "upos": "ADJ",  "feats": "Case=Nom"},
                    {"form": "dan",   "lemma": "dan",
                     "upos": "NOUN", "feats": "Case=Nom"},
                    {"form": ".",     "lemma": ".",
                     "upos": "PUNCT", "feats": null}
                  ]
                }
              ]
            }
          ]
        }

    Schema mapping:
        text:        record[text_field] (records with empty/missing
                     text are skipped).
        source:      provided by caller.
        domain:      provided by caller.
        doc_id:      record[id_field] if present.
        metadata:    the configured metadata_fields, or every other
                     field except the text field, the id field,
                     paragraphs, and conll.
        annotations:
            tokens:    flattened across all paragraphs/sentences,
                       reading form, lemma, upos, feats from each
                       token dict.
            sentences: [start, end] index pairs marking each
                       sentence's span in the flattened token list.
            Absent if the record has no paragraphs field or no
            tokens.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from slm4ie.data.extractors import BaseExtractor, register_extractor
from slm4ie.data.schema import Annotations, Document, Token

logger = logging.getLogger(__name__)

DEFAULT_TEXT_FIELD = "text"
DEFAULT_ID_FIELD = "doc_id"

# Structural fields never copied into Document.metadata, on top of the
# (configurable) text and id fields.
_STRUCTURAL_FIELDS = {"paragraphs", "conll"}


def _parse_tokens_from_paragraphs(
    paragraphs: List[Dict],
) -> Optional[Annotations]:
    """Parse nested paragraphs/sentences/tokens into Annotations.

    Flattens all paragraphs, sentences, and tokens into a single
    list, tracking sentence boundaries as [start, end] index pairs.

    Args:
        paragraphs (List[Dict]): List of paragraph dicts, each
            containing a sentences key with sentence dicts, each
            containing a tokens key with token dicts.

    Returns:
        Optional[Annotations]: Parsed annotations, or None if no
            tokens were found.
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
    """Extracts Documents from line-delimited JSON files.

    One Document is produced per JSONL line with a non-empty text
    field. Recursively discovers all .jsonl files under the given
    directory (sorted). The text field, id field, and the set of
    record fields kept as metadata are configurable through the
    `metadata:` config block (see the module docstring).
    """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Document]:
        """Yield Documents from all JSONL files under input_dir.

        Args:
            input_dir (Path): Directory containing .jsonl files
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
            Document: One document per valid JSONL line.
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
        files: List[Path] = sorted(input_dir.rglob("*.jsonl"))

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
        """Parse a single JSONL file and yield Documents.

        Args:
            filepath (Path): Path to the JSONL file.
            source (str): Dataset key.
            domain (str): Domain label.
            text_field (str): Record field copied into `text`.
            id_field (str): Record field kept as `doc_id`.
            metadata_fields (Optional[List[str]]): Explicit whitelist of
                record fields to keep as metadata, or None to keep every
                field except the text field, the id field, and the
                structural fields.

        Yields:
            Document: One document per valid line.
        """
        excluded = {text_field, id_field} | _STRUCTURAL_FIELDS
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

                text = record.get(text_field, "")
                if not text:
                    continue

                doc_id: Optional[str] = record.get(id_field)

                paragraphs = record.get("paragraphs")
                annotations: Optional[Annotations] = None
                if paragraphs is not None:
                    annotations = _parse_tokens_from_paragraphs(paragraphs)

                if metadata_fields is not None:
                    doc_metadata = {
                        k: record[k] for k in metadata_fields if k in record
                    }
                else:
                    doc_metadata = {
                        k: v for k, v in record.items() if k not in excluded
                    }

                yield Document(
                    text=text,
                    source=source,
                    domain=domain,
                    doc_id=doc_id,
                    metadata=doc_metadata,
                    annotations=annotations,
                )


register_extractor("jsonl", JsonlExtractor)
