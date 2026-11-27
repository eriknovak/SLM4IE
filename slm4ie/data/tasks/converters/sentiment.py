"""Sentiment (`sentiment`) converter backend.

Turns SA datasets into `SentimentExample` rows:

    {"id": "<source>:<doc_id>", "text": "<...>", "label": "negative"}

Only `kind: extracted` sources are read, from `<roots.extracted>/<key>.jsonl`
(SentiNews, ParlaSent); any other source kind is refused. A record whose
label maps to no canonical class is dropped, and the drops are counted and
logged per entry.

The split policy is HASH: the run loop buckets each record by a deterministic hash
of its stable id. A single-split entry (e.g. a held-out `test`-only dataset)
therefore lands entirely in that one split without any special-casing here.
"""

import logging
from collections.abc import Iterator
from typing import Any

from slm4ie.data.schema import SentimentExample
from slm4ie.data.tasks.config import TaskEntry, TasksRoots
from slm4ie.data.tasks.run import (
    ConvertContext,
    SplitPolicy,
    TaskConverter,
    iter_extracted_records,
    label_allow_set,
    register_converter,
    synthesize_id,
)

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

#: Map common label spellings to canonical 3-class labels.
_LABEL_NORMALIZATION: dict[str, str] = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "neg": "negative",
    "pos": "positive",
    "neu": "neutral",
}


# ===============================
# Helper functions
# ===============================


def normalize_label(raw_label: Any | None, allow: set | None) -> str | None:
    """Map a raw label string to the canonical form, if allowed.

    Args:
        raw_label: Label as it appears in the source. May be None.
        allow: Optional allow-list of canonical label names.

    Returns:
        The canonical label, or None when the value is empty, not recognized, or
        filtered out by *allow*.
    """
    if raw_label is None:
        return None
    cleaned = str(raw_label).strip().lower()
    if not cleaned:
        return None
    canonical = _LABEL_NORMALIZATION.get(cleaned)
    if canonical is None:
        return None
    if allow is not None and canonical not in allow:
        return None
    return canonical


def _iter_extracted(
    entry: TaskEntry,
    roots: TasksRoots,
    allow: set | None,
) -> Iterator[tuple[str, SentimentExample]]:
    """Yield `(stable_id, example)` pairs from extracted sources.

    Args:
        entry: Sentiment task entry with `source.kind == 'extracted'`.
        roots: Filesystem roots.
        allow: Optional allow-list of canonical labels.

    Yields:
        `(stable_id, SentimentExample)` pairs.

    Raises:
        FileNotFoundError: If a source JSONL is missing.
    """
    # convert the extracted records
    index = 0
    dropped = 0
    for record in iter_extracted_records(entry, roots):
        raw_label = (
            record.get("label")
            or record.get("sentiment")
            or (record.get("metadata") or {}).get("sentiment")
            or (record.get("metadata") or {}).get("label")
        )
        label = normalize_label(raw_label, allow)
        if label is None:
            dropped += 1
            continue
        example_id = synthesize_id(record, entry.dataset, index)
        example: SentimentExample = SentimentExample(
            id=example_id,
            text=record.get("text", ""),
            label=label,
        )
        index += 1
        yield example_id, example

    # report the dropped records
    if dropped:
        logger.warning("%s: dropped %d record(s) whose label maps to no canonical class", entry.key, dropped)


# ===============================
# Converter class
# ===============================


@register_converter("sentiment")
class SentimentConverter(TaskConverter):
    """Converter for the sentiment-analysis task family."""

    name = "sentiment"
    split_policy = SplitPolicy.HASH
    description = "Convert SA datasets into per-split task JSONL under <roots.tasks>/<task>/<dataset>/<split>.jsonl.gz."

    def iter_examples(
        self,
        entry: TaskEntry,
        roots: TasksRoots,
        ctx: ConvertContext,
        splits: list[str],
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield `(stable_id, SentimentExample)` pairs for one entry.

        Args:
            entry: Sentiment task entry.
            roots: Filesystem roots.
            ctx: Per-run options (unused by this family).
            splits: Target split names (unused; the run loop hashes each id).

        Yields:
            `(stable_id, example)` pairs for the HASH split policy.

        Raises:
            ValueError: If the entry source kind is not `extracted`.
        """
        allow = label_allow_set(entry)

        if entry.source.kind != "extracted":
            raise ValueError(
                f"Sentiment entry {entry.task}/{entry.dataset} has source kind {entry.source.kind!r}; "
                "only 'extracted' is supported."
            )

        for split_key, example in _iter_extracted(entry, roots, allow):
            yield split_key, dict(example)
