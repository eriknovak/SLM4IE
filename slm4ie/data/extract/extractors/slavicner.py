"""SlavicNER extractor for the SLM4IE pipeline.

Reads the SlavicNER corpus (Piskorski et al., LREC-COLING 2024), where each
document is a plain-text file under `raw/` paired with a tab-separated
mention file under `ner/` at the same relative path:

    <root>/raw/<topic>/<lang>/<name>.txt     document text
    <root>/ner/<topic>/<lang>/<name>.tsv     one mention per line

A mention line has six columns and no header: begin offset, end offset
(exclusive), surface form, lemma, category (PER, ORG, LOC, EVT, PRO) and
cross-lingual entity id. The offsets index the text file's characters as
read, so the text is kept exactly as stored.

Example:
    `raw/ryanair/sl/sl-12.txt` holding `Let iz pekla ... Manchestra ...`
    with the mention line `51	61	Manchestra	Manchester	LOC	GPE-Manchester`
    yields one `Document`.

    Schema mapping:
        text:        the text file's content, unchanged.
        source:      provided by caller.
        domain:      provided by caller.
        doc_id:      `<topic>/<lang>/<name>`, e.g. `ryanair/sl/sl-12`;
                     file names repeat across topics, so the topic is part
                     of the id.
        metadata:    `topic` and `language` from the path.
        annotations:
            tokens:    empty; the corpus carries no tokenisation.
            sentences: empty.
            spans:     `[begin, end, category]` per mention line.

Some articles are stored under two or three topics, with ids that differ and
annotations that differ slightly. Only the first copy in sorted path order is
kept, so a hash split cannot place one article on both sides.

Pick the language with `include:` in extract.yaml, e.g. `*/raw/*/sl/*.txt`.
"""

import csv
import hashlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from slm4ie.data.extract.extractors import FileBasedExtractor, register_extractor
from slm4ie.data.schema import Annotations, Document

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

# begin, end, form, lemma, category, entity id
_MENTION_COLUMNS = 6


# ===============================
# SlavicNER extractor
# ===============================


class SlavicNERExtractor(FileBasedExtractor):
    """Extracts Documents with entity spans from the SlavicNER corpus.

    Discovers every `raw/<topic>/<lang>/*.txt` file under the input
    directory, drops later copies of an identical text, and pairs each
    file with the mention file at the same path under `ner/`. Each
    document carries its mentions as character spans.
    """

    def iter_input_files(self, input_dir: Path) -> list[Path]:
        """Return the sorted document text files under input_dir.

        Args:
            input_dir: Directory searched recursively for `raw/` trees.

        Returns:
            Sorted `raw/<topic>/<lang>/*.txt` paths, each text once.
        """
        # keep the first file of each distinct text; listing runs once, before sharding
        files: list[Path] = []
        seen: set[str] = set()
        dropped = 0
        for path in sorted(p for p in input_dir.rglob("raw/*/*/*.txt") if p.is_file()):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                dropped += 1
                continue
            seen.add(digest)
            files.append(path)

        if dropped:
            logger.info("Dropped %d SlavicNER files repeating an earlier file's text", dropped)
        return files

    def extract_files(
        self,
        files: list[Path],
        source: str,
        domain: str,
        input_dir: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Document]:
        """Yield one Document per text file, with its mention spans.

        Args:
            files: Text files under a `raw/` tree, in order.
            source: Dataset key assigned to every Document.
            domain: Domain label assigned to every Document.
            input_dir: Dataset root; unused, as ids come from the `raw/` path.
            metadata: Ignored.

        Yields:
            One document per text file.

        Raises:
            FileNotFoundError: If a text file has no mention file beside it.
            ValueError: If a mention line is malformed or its offsets do not
                match its surface form in the text.
        """
        del input_dir, metadata

        for text_path in files:
            topic, language, name = text_path.parts[-3], text_path.parts[-2], text_path.stem
            text = text_path.read_text(encoding="utf-8")
            spans = _read_mentions(_mention_path(text_path), text)
            yield Document(
                text=text,
                source=source,
                domain=domain,
                doc_id=f"{topic}/{language}/{name}",
                metadata={"topic": topic, "language": language},
                annotations=Annotations(tokens=[], sentences=[], spans=spans),
            )


# ===============================
# Helper functions
# ===============================


def _mention_path(text_path: Path) -> Path:
    """Return the mention file that belongs to a document text file.

    Args:
        text_path: `<root>/raw/<topic>/<lang>/<name>.txt`.

    Returns:
        `<root>/ner/<topic>/<lang>/<name>.tsv`.
    """
    root = text_path.parents[3]
    return root / "ner" / text_path.parts[-3] / text_path.parts[-2] / f"{text_path.stem}.tsv"


def _read_mentions(path: Path, text: str) -> list[list[Any]]:
    """Read a mention file into `[begin, end, category]` spans.

    Args:
        path: Tab-separated mention file.
        text: The document text the offsets index.

    Returns:
        Spans in file order.

    Raises:
        FileNotFoundError: If the mention file is missing.
        ValueError: If a line has the wrong column count or its offsets do
            not cover its surface form.
    """
    if not path.exists():
        raise FileNotFoundError(f"SlavicNER mention file {path} is missing")

    # read and check each mention line
    spans: list[list[Any]] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for line_number, row in enumerate(csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE), start=1):
            if len(row) != _MENTION_COLUMNS:
                raise ValueError(f"{path}:{line_number} has {len(row)} columns, expected {_MENTION_COLUMNS}")
            begin, end, form, category = int(row[0]), int(row[1]), row[2], row[4].upper()
            if text[begin:end] != form:
                raise ValueError(f"{path}:{line_number} offsets {begin}-{end} give {text[begin:end]!r}, not {form!r}")
            spans.append([begin, end, category])
    return spans


register_extractor("slavicner", SlavicNERExtractor)
