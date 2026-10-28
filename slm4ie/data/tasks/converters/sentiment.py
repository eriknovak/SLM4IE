"""Sentiment (`sentiment`) converter backend.

Turns SA datasets into `SentimentExample` rows:

    {"id": "<source>:<doc_id>", "text": "<...>", "label": "negative"}

Two source kinds are supported:

* `kind: extracted` -- reads `<roots.extracted>/<key>.jsonl` (SentiNews).
* `kind: raw` -- reads SentiNews-format TSV files from `<roots.raw>/<key>/`
  (the held-out Twitter dataset).

The split policy is HASH: the run loop buckets each record by a deterministic hash
of its stable id. A single-split entry (e.g. a held-out `test`-only dataset)
therefore lands entirely in that one split without any special-casing here.
"""

import csv
from collections.abc import Iterator
from pathlib import Path
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
    for record in iter_extracted_records(entry, roots):
        raw_label = (
            record.get("label")
            or record.get("sentiment")
            or (record.get("metadata") or {}).get("sentiment")
            or (record.get("metadata") or {}).get("label")
        )
        label = normalize_label(raw_label, allow)
        if label is None:
            continue
        example_id = synthesize_id(record, entry.dataset, index)
        example: SentimentExample = SentimentExample(
            id=example_id,
            text=record.get("text", ""),
            label=label,
        )
        index += 1
        yield example_id, example


def _iter_raw_sentinews_format(
    source_dir: Path,
    dataset: str,
    allow: set | None,
) -> Iterator[tuple[str, SentimentExample]]:
    """Yield `(stable_id, example)` from SentiNews-format TSV files.

    The held-out Twitter sentiment dataset ships in the same
    `SentiNews_<level>-level.txt` layout, so a single TSV reader covers both.

    Args:
        source_dir: Directory containing `SentiNews_*-level.*` files.
        dataset: Dataset name used to synthesize record ids.
        allow: Optional allow-list of canonical labels.

    Yields:
        `(stable_id, SentimentExample)` pairs.

    Raises:
        FileNotFoundError: If no SentiNews-format files are present.
    """
    # collect the input files
    candidates: list[Path] = []
    for pattern in ("SentiNews_*-level.*", "*.tsv", "*.txt"):
        candidates.extend(sorted(source_dir.glob(pattern)))
    seen: set = set()
    files: list[Path] = []
    for path in candidates:
        if path.is_file() and path not in seen:
            files.append(path)
            seen.add(path)
    if not files:
        raise FileNotFoundError(f"No sentiment input files found in {source_dir}.")

    # read the rows of each file
    text_keys = ("content", "text", "sentence", "paragraph", "tweet")
    index = 0
    for path in files:
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                text = next((row[k] for k in text_keys if row.get(k)), None)
                raw_label = row.get("sentiment") or row.get("label") or row.get("polarity")
                if text is None:
                    continue
                label = normalize_label(raw_label, allow)
                if label is None:
                    continue
                example_id = synthesize_id(
                    {"id": row.get("nid") or row.get("doc_id") or row.get("id")},
                    dataset,
                    index,
                )
                example: SentimentExample = SentimentExample(id=example_id, text=text, label=label)
                index += 1
                yield example_id, example


def _iter_raw(
    entry: TaskEntry,
    roots: TasksRoots,
    allow: set | None,
) -> Iterator[tuple[str, SentimentExample]]:
    """Yield `(stable_id, example)` pairs from raw sentiment sources.

    Args:
        entry: Sentiment task entry with `source.kind == 'raw'`.
        roots: Filesystem roots.
        allow: Optional allow-list of canonical labels.

    Yields:
        `(stable_id, SentimentExample)` pairs.

    Raises:
        FileNotFoundError: If a source directory is missing.
    """
    for key in entry.source.keys:
        source_dir = roots.raw / key
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Raw source for {entry.task}/{entry.dataset}: {source_dir} is not a directory.")
        yield from _iter_raw_sentinews_format(source_dir, entry.dataset, allow)


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
            ValueError: If the entry source kind is unknown.
        """
        allow = label_allow_set(entry)

        if entry.source.kind == "extracted":
            record_iter = _iter_extracted(entry, roots, allow)
        elif entry.source.kind == "raw":
            record_iter = _iter_raw(entry, roots, allow)
        else:
            raise ValueError(f"Unknown source kind {entry.source.kind!r} for {entry.task}/{entry.dataset}.")

        for split_key, example in record_iter:
            yield split_key, dict(example)
