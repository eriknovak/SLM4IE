"""Part-of-speech (`pos`) converter backend.

Turns the UPOS tags of extracted annotated datasets into `PosExample` rows, one
per sentence:

    {"id": "<uid>:s<n>", "words": ["Ana", "bere", "."], "tags": ["PROPN", "VERB", "PUNCT"]}

Sources are read from the extraction tree (`<roots.extracted>/<key>.jsonl`
joined with its annotations sidecar). The words and tags of a sentence are the
`forms` and `upos` arrays over its token range. A sentence whose word and tag
counts differ, a token with no tag included, is refused, and the refusals are
counted and logged per entry. The split policy is HASH on the document id, so
every sentence of one document lands in the same split; an entry's `split_on`
assigns the split from document metadata instead.
"""

import logging
from collections.abc import Iterator
from typing import Any

from slm4ie.data.schema import PosExample
from slm4ie.data.tasks.config import TaskEntry, TasksRoots
from slm4ie.data.tasks.run import (
    ConvertContext,
    SplitPolicy,
    TaskConverter,
    iter_extracted_records,
    register_converter,
    split_key_of,
    synthesize_id,
)

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# POS converter
# ===============================


@register_converter("pos")
class PosConverter(TaskConverter):
    """Converter for the part-of-speech task family (one example per sentence)."""

    name = "pos"
    split_policy = SplitPolicy.HASH
    description = (
        "Convert the UPOS tags of extracted annotated datasets into per-sentence task JSONL under "
        "<roots.tasks>/<task>/<dataset>/<split>.jsonl.gz."
    )

    def iter_examples(
        self,
        entry: TaskEntry,
        roots: TasksRoots,
        ctx: ConvertContext,
        splits: list[str],
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield `(split_key, PosExample)` pairs, one per accepted sentence.

        Args:
            entry: POS task entry.
            roots: Filesystem roots.
            ctx: Per-run options (unused by this family).
            splits: Target split names (unused; the run loop hashes each document id).

        Yields:
            `(document_id, example)` pairs the run loop hashes into splits, or
            `(split_name, example)` pairs when the entry sets `split_on`.

        Raises:
            ValueError: If the entry source kind is not `extracted`.
            FileNotFoundError: If a source `<key>.jsonl` is missing.
        """
        if entry.source.kind != "extracted":
            raise ValueError(f"pos only supports source.kind='extracted'; got {entry.source.kind!r} for {entry.key}.")

        # split every annotated document into sentences
        refused = 0
        for index, record in enumerate(iter_extracted_records(entry, roots)):
            annotations = record.get("annotations")
            if not annotations:
                continue
            doc_id = synthesize_id(record, record.get("source", entry.dataset), index)
            split_key = split_key_of(entry, record.get("metadata") or {}, doc_id)
            forms, upos = annotations["forms"], annotations["upos"]
            for number, (start, end) in enumerate(annotations["sentences"]):
                words = forms[start : end + 1]
                tags = [tag for tag in upos[start : end + 1] if tag]
                if len(words) != len(tags):
                    refused += 1
                    continue
                yield split_key, dict(PosExample(id=f"{doc_id}:s{number}", words=words, tags=tags))

        # report the refused sentences
        if refused:
            logger.warning("%s: refused %d sentence(s) whose word and tag counts differ", entry.key, refused)
