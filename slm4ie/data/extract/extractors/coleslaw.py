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
        doc_id:      positional, `<subcorpus>/<file stem>:<line>`
                     (see Document ids below).
        native_id:   the record's `doc_id` or `id` (coerced to str)
                     when present, else None.
        metadata:    subcorpus (parent directory name: PISRS,
                     UradniList, SodnaPraksa, USRS) plus all other
                     record fields except the ones consumed for
                     text.
        annotations: not produced.

Document ids:
    The corpus's own `id` is not a document key: it restarts per file,
    `sp_claims` and `sp_courts` share ids for the same case, and in
    `ul-uredbeni` it is the item number inside a gazette issue (148k
    rows carry 6.4k distinct ids). So `doc_id` is the positional id of
    `assembly.positional_doc_id`: the file's subcorpus directory name and
    stem plus the record's 0-based line index, e.g.
    `UradniList/ul-uredbeni:00001234`. The raw `id` is kept as
    `native_id`.
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from slm4ie.data.extract.extractors import FileBasedExtractor, register_extractor
from slm4ie.data.extract.extractors.assembly import positional_doc_id, probe_doc_id, project_metadata
from slm4ie.data.schema import Document

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

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


# ===============================
# Helper functions
# ===============================


def _join_string_fields(record: dict[str, Any], keys: tuple) -> str:
    """Concatenate non-empty string fields from a record.

    Args:
        record: Source record.
        keys: Field names to read in order.

    Returns:
        Fields joined with blank lines, or "" if none present.
    """
    parts: list[str] = []
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n\n".join(parts)


def _candidate_texts(record: dict[str, Any]) -> list[tuple[str, str]]:
    """List every present text source of a COLESLAW record.

    Precedence order: `text` -> `fullText` -> sp_courts fields ->
    sp_claims fields. The first entry is the one selection uses, so
    any further entries are shadowed sources worth warning about.

    Args:
        record: A parsed JSONL record.

    Returns:
        `(source_name, text)` pairs for every
            non-empty source, in precedence order. Empty if the
            record has no usable text.
    """
    candidates: list[tuple[str, str]] = []

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


# ===============================
# COLESLAW extractor
# ===============================


class ColeslawExtractor(FileBasedExtractor):
    """Extracts Documents from the COLESLAW v1.0 corpus.

    Recursively discovers .jsonl files under input_dir, so the catalogue's
    `include:` can narrow them. The name of the immediate parent directory
    (PISRS, UradniList, SodnaPraksa, USRS) is recorded in metadata under
    the subcorpus key.
    """

    def iter_input_files(self, input_dir: Path) -> list[Path]:
        """Return sorted .jsonl files under input_dir.

        Args:
            input_dir: Directory containing the COLESLAW
                subcorpora (searched recursively).

        Returns:
            Sorted .jsonl file paths.
        """
        return sorted(p for p in input_dir.rglob("*.jsonl") if p.is_file())

    def extract_files(
        self,
        files: list[Path],
        source: str,
        domain: str,
        input_dir: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Document]:
        """Yield Documents from the given COLESLAW JSONL files.

        Args:
            files: JSONL files to parse, in order.
            source: Dataset key assigned to every Document.
            domain: Domain label assigned to every Document.
            input_dir: Dataset root; unused, as ids key off each
                file's subcorpus directory and stem.
            metadata: Ignored; this
                extractor sources metadata from each record itself.

        Yields:
            One document per record with non-empty text.
        """
        del input_dir, metadata

        # key each file by subcorpus dir + stem, as "COLESLAW 1.0/" has a space
        for filepath in files:
            yield from self._parse_file(filepath, source, domain, f"{filepath.parent.name}/{filepath.stem}")

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
        unit: str,
    ) -> Iterator[Document]:
        """Parse one COLESLAW JSONL file and yield Documents.

        Args:
            filepath: Path to the JSONL file.
            source: Dataset key.
            domain: Domain label.
            unit: The file's subcorpus directory name and stem
                (e.g. "PISRS/register-predpisov"); the id prefix, and
                its first segment is the subcorpus.

        Yields:
            One document per valid line with usable text.
        """
        subcorpus = filepath.parent.name

        # read the file line by line
        with filepath.open(encoding="utf-8") as fh:
            for line_idx, raw_line in enumerate(fh):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Invalid JSON on line %d of %s: %s",
                        line_idx + 1,
                        filepath,
                        exc,
                    )
                    continue

                if not isinstance(record, dict):
                    continue

                # choose the text source, warning about shadowed ones
                candidates = _candidate_texts(record)
                if not candidates:
                    continue
                chosen_source, text = candidates[0]
                doc_id = positional_doc_id(unit, line_idx)
                if len(candidates) > 1:
                    logger.warning(
                        "Record %s has multiple text sources %s; using %s",
                        doc_id,
                        [name for name, _ in candidates],
                        chosen_source,
                    )

                # build the metadata from the unconsumed fields
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
                    native_id=probe_doc_id(record, ["doc_id", "id"]),
                    metadata=metadata,
                )


register_extractor("coleslaw", ColeslawExtractor)
