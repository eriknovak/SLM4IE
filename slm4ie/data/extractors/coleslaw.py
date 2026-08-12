"""COLESLAW corpus extractor for the SLM4IE pipeline.

COLESLAW v1.0 ships four heterogeneous JSONL subcorpora with
distinct schemas. PISRS and UradniList records expose a text field.
USRS (Constitutional Court) records expose fullText. SodnaPraksa
sp_courts records carry separate jedro, izrek, and obrazlozitev
sections. SodnaPraksa sp_claims records have no single text body, so
their relevant prose fields are concatenated in a fixed reading
order.

This extractor walks the directory recursively and selects the
appropriate text-building strategy per record. Records that expose
more than one candidate text source (e.g. both `text` and
`fullText`) are logged with a warning naming the shadowed sources
and the chosen one; the first source in precedence order still wins.

Example:
    PISRS / UradniList (one JSON object per line):

        {"id": 1, "text": "Zakon o nečem.", "title": "Zakon"}

    USRS (Constitutional Court):

        {"id": "Up-1", "fullText": "Sklep ustavnega sodišča."}

    SodnaPraksa/sp_courts.jsonl:

        {"id": "c1", "jedro": "Bistvo.", "izrek": "Razveljavi se.",
         "obrazlozitev": "Obrazložitev sledi."}

    SodnaPraksa/sp_claims.jsonl:

        {"id": "750", "skodni_dogodek": "Prometna nesreča.",
         "poskodba": "Zvin vratu.", "telesne_bolecine": "Tri tedne."}

    Schema mapping:
        text:        first non-empty of: text -> fullText -> joined
                     jedro/izrek/obrazlozitev -> joined sp_claims
                     fields (in fixed reading order).
        source:      provided by caller.
        domain:      provided by caller.
        doc_id:      doc_id if present, else id (coerced to str),
                     else None.
        metadata:    subcorpus (parent directory name: PISRS,
                     UradniList, SodnaPraksa, USRS) plus all other
                     record fields except the ones consumed for
                     text and doc_id.
        annotations: not produced.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from slm4ie.data.extractors import BaseExtractor, register_extractor
from slm4ie.data.extractors.assembly import probe_doc_id, project_metadata
from slm4ie.data.schema import Document

logger = logging.getLogger(__name__)

_RESERVED_FIELDS = {"text", "fullText", "doc_id"}

# Order matters: jedro (essence) → izrek (operative part) → obrazlozitev
# (reasoning) mirrors the structure of Slovenian court decisions.
_SP_COURTS_FIELDS = ("jedro", "izrek", "obrazlozitev")

# sp_claims (personal-injury case summaries) lacks a unified text body.
# The relevant prose fields are concatenated in a logical reading order.
_SP_CLAIMS_FIELDS = (
    "skodni_dogodek",
    "poskodba",
    "telesne_bolecine",
    "strah",
    "zmanjsanje_zivljenjske_aktivnosti",
    "dodatne_informacije",
)


def _join_string_fields(record: Dict[str, Any], keys: tuple) -> str:
    """Concatenate non-empty string fields from a record.

    Args:
        record (Dict[str, Any]): Source record.
        keys (tuple): Field names to read in order.

    Returns:
        str: Fields joined with blank lines, or "" if none present.
    """
    parts: List[str] = []
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n\n".join(parts)


def _candidate_texts(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """List every present text source of a COLESLAW record.

    Precedence order: `text` -> `fullText` -> sp_courts fields ->
    sp_claims fields. The first entry is the one selection uses, so
    any further entries are shadowed sources worth warning about.

    Args:
        record (Dict[str, Any]): A parsed JSONL record.

    Returns:
        List[Tuple[str, str]]: `(source_name, text)` pairs for every
            non-empty source, in precedence order. Empty if the
            record has no usable text.
    """
    candidates: List[Tuple[str, str]] = []

    text = record.get("text")
    if isinstance(text, str) and text.strip():
        candidates.append(("text", text))

    full_text = record.get("fullText")
    if isinstance(full_text, str) and full_text.strip():
        candidates.append(("fullText", full_text))

    courts_text = _join_string_fields(record, _SP_COURTS_FIELDS)
    if courts_text:
        candidates.append(("sp_courts", courts_text))

    claims_text = _join_string_fields(record, _SP_CLAIMS_FIELDS)
    if claims_text:
        candidates.append(("sp_claims", claims_text))

    return candidates


class ColeslawExtractor(BaseExtractor):
    """Extracts Documents from the COLESLAW v1.0 corpus.

    Recursively discovers .jsonl files under input_dir. The name of
    the immediate parent directory (PISRS, UradniList, SodnaPraksa,
    USRS) is recorded in metadata under the subcorpus key.
    """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Document]:
        """Yield Documents from all COLESLAW JSONL files under input_dir.

        Args:
            input_dir (Path): Directory containing the COLESLAW
                subcorpora (searched recursively).
            source (str): Dataset key assigned to every Document.
            domain (str): Domain label assigned to every Document.
            metadata (Optional[Dict[str, Any]]): Ignored; this
                extractor sources metadata from each record itself.

        Yields:
            Document: One document per record with non-empty text.
        """
        del metadata
        files = sorted(p for p in input_dir.rglob("*.jsonl") if p.is_file())

        for filepath in files:
            subcorpus = filepath.parent.name
            yield from self._parse_file(
                filepath, source, domain, subcorpus
            )

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
        subcorpus: str,
    ) -> Iterator[Document]:
        """Parse one COLESLAW JSONL file and yield Documents.

        Args:
            filepath (Path): Path to the JSONL file.
            source (str): Dataset key.
            domain (str): Domain label.
            subcorpus (str): Name of the parent directory
                (e.g. "PISRS").

        Yields:
            Document: One document per valid line with usable text.
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

                if not isinstance(record, dict):
                    continue

                candidates = _candidate_texts(record)
                if not candidates:
                    continue
                chosen_source, text = candidates[0]
                doc_id = probe_doc_id(record, ["doc_id", "id"])
                if len(candidates) > 1:
                    logger.warning(
                        "Record %s in subcorpus %s has multiple text sources %s; using %s",
                        doc_id,
                        subcorpus,
                        [name for name, _ in candidates],
                        chosen_source,
                    )

                exclude = _RESERVED_FIELDS | set(_SP_COURTS_FIELDS) | set(_SP_CLAIMS_FIELDS)
                metadata = {
                    "subcorpus": subcorpus,
                    **project_metadata(record, exclude=exclude),
                }

                yield Document(
                    text=text,
                    source=source,
                    domain=domain,
                    doc_id=doc_id,
                    metadata=metadata,
                )


register_extractor("coleslaw", ColeslawExtractor)
