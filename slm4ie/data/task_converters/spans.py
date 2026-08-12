"""NER (`to_spans`) converter backend.

Turns extracted NER datasets into GLiNER-style `NerExample` rows. Each output
line is:

    {"id": "<uid>", "text": "<document text>",
     "spans": [{"start": int, "end": int, "label": "<TAG>"}]}

Sources are read from the extraction tree (`<roots.extracted>/<key>.jsonl` plus
the optional annotations sidecar) and joined on the fly. Annotation spans whose
label is not in the entry's `labels` allow-list are dropped, with one warning
per dataset. The split policy is HASH: the driver buckets each document by a
deterministic hash of its stable id.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

from slm4ie.data.schema import NerExample
from slm4ie.data.task_converter import (
    ConvertContext,
    SplitPolicy,
    TaskConverter,
    iter_extracted_records,
    label_allow_set,
    register_converter,
    synthesize_id,
)
from slm4ie.data.tasks import TaskEntry, TasksRoots

logger = logging.getLogger(__name__)


def normalize_spans(raw_spans: Any) -> List[Tuple[int, int, str]]:
    """Normalize spans from list-of-lists or list-of-dicts to tuples.

    Args:
        raw_spans: Spans as either `[s, e, label]` triples or
            `{"start": s, "end": e, "label": label}` dicts.

    Returns:
        Normalized `(start, end, label)` tuples.

    Raises:
        ValueError: If a span entry is malformed.
    """
    normalized: List[Tuple[int, int, str]] = []
    if not raw_spans:
        return normalized
    for span in raw_spans:
        if isinstance(span, dict):
            normalized.append((int(span["start"]), int(span["end"]), str(span["label"])))
        elif isinstance(span, (list, tuple)) and len(span) == 3:
            start, end, label = span
            normalized.append((int(start), int(end), str(label)))
        else:
            raise ValueError(f"Unrecognized span shape: {span!r}")
    return normalized


def _record_to_example(
    record: Dict[str, Any],
    index: int,
    label_allow: Optional[set],
    dropped_labels: set,
) -> Optional[NerExample]:
    """Convert one joined record to a `NerExample`, or None.

    Args:
        record: Joined record (text + annotations).
        index: Zero-based record position.
        label_allow: Optional set of accepted labels. When provided, spans whose
            label is not in the set are dropped (and the label is recorded in
            *dropped_labels* for a one-shot warning later).
        dropped_labels: Mutable accumulator collecting filtered-out label names.

    Returns:
        A `NerExample` when the record carries a `spans` field, else None.
    """
    annotations = record.get("annotations") or {}
    raw_spans = annotations.get("spans")
    if raw_spans is None:
        return None

    spans = normalize_spans(raw_spans)
    if label_allow is not None:
        filtered: List[Tuple[int, int, str]] = []
        for start, end, label in spans:
            if label in label_allow:
                filtered.append((start, end, label))
            else:
                dropped_labels.add(label)
        spans = filtered

    return NerExample(
        id=synthesize_id(record, record.get("source", "unknown"), index),
        text=record.get("text", ""),
        spans=[{"start": start, "end": end, "label": label} for start, end, label in spans],
    )


@register_converter("to_spans")
class SpansConverter(TaskConverter):
    """Converter for the NER task family (GLiNER-style spans)."""

    name = "to_spans"
    split_policy = SplitPolicy.HASH
    description = (
        "Convert extracted NER datasets into GLiNER-style task JSONL under "
        "<roots.tasks>/<task>/<dataset>/<split>.jsonl.gz."
    )

    def iter_examples(
        self,
        entry: TaskEntry,
        roots: TasksRoots,
        ctx: ConvertContext,
        splits: List[str],
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Yield `(stable_id, NerExample)` pairs for every joined record.

        Args:
            entry: NER task entry.
            roots: Filesystem roots.
            ctx: Per-run options (unused by this family).
            splits: Target split names (unused; the driver hashes each id).

        Yields:
            `(stable_id, example)` pairs for the HASH split policy.

        Raises:
            ValueError: If the entry source kind is not `extracted`.
            FileNotFoundError: If a source `<key>.jsonl` is missing.
        """
        del ctx, splits
        if entry.source.kind != "extracted":
            raise ValueError(
                f"to_spans only supports source.kind='extracted'; got "
                f"{entry.source.kind!r} for {entry.task}/{entry.dataset}."
            )

        label_allow = label_allow_set(entry)
        dropped_labels: set = set()

        for index, record in enumerate(iter_extracted_records(entry, roots)):
            example = _record_to_example(record, index, label_allow, dropped_labels)
            if example is None:
                continue
            yield example["id"], dict(example)

        if dropped_labels:
            logger.warning(
                "Entry %s/%s dropped spans with %d label(s) outside allow-list (%s): %s",
                entry.task,
                entry.dataset,
                len(dropped_labels),
                sorted(label_allow) if label_allow else "<empty>",
                sorted(dropped_labels),
            )
