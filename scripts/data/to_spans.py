"""Convert extracted SLM4IE NER datasets into GLiNER-style task JSONL.

Driven by ``configs/data/tasks.yaml``. Processes every entry whose
resolved converter is ``to_spans`` (i.e. NER task family) and emits
one gzipped JSONL per declared split under
``<roots.tasks>/<task>/<dataset>/<split>.jsonl.gz``.

Each output line is a ``NerExample``:

    {
        "id": "<source>:<doc_id>",
        "text": "<document text>",
        "spans": [{"start": <char_start>, "end": <char_end>, "label": "<TAG>"}]
    }

Sources are read from the extraction tree
(``<roots.extracted>/<key>.jsonl`` + the optional annotations
sidecar). Annotation spans whose label is not in the entry's
``labels:`` allow-list are dropped (one warning per dataset). Splits
are derived from a deterministic hash of each document's ``uid`` /
``doc_id`` (80/10/10 buckets), redirected to the splits declared by
the entry.

Examples:
    Convert every NER entry declared in tasks.yaml:

        uv run python scripts/data/to_spans.py --all

    Convert just one entry:

        uv run python scripts/data/to_spans.py ner/ssj500k
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from slm4ie.data.io_utils import find_project_root, iter_joined_records
from slm4ie.data.parallel import (
    configure_script_logging,
    cpu_default,
    resolve_workers,
    run_parallel,
)
from slm4ie.data.schema import NerExample
from slm4ie.data.tasks import (
    TaskEntry,
    TasksRoots,
    filter_for_converter,
    load_tasks,
    resolve_output_dir,
)
from slm4ie.data.task_writer import (
    all_outputs_exist,
    hash_split,
    outputs_for_splits,
    write_jsonl_splits,
)

logger = logging.getLogger(__name__)

#: Converter module name, used to filter `tasks.yaml`.
CONVERTER_NAME: str = "to_spans"


def _normalize_spans(
    raw_spans: Any,
) -> List[Tuple[int, int, str]]:
    """Normalize spans from list-of-lists or list-of-dicts to tuples.

    Args:
        raw_spans: Spans as either ``[s, e, label]`` triples or
            ``{"start": s, "end": e, "label": label}`` dicts.

    Returns:
        Normalized ``(start, end, label)`` tuples.

    Raises:
        ValueError: If a span entry is malformed.
    """
    normalized: List[Tuple[int, int, str]] = []
    if not raw_spans:
        return normalized
    for span in raw_spans:
        if isinstance(span, dict):
            normalized.append((
                int(span["start"]),
                int(span["end"]),
                str(span["label"]),
            ))
        elif isinstance(span, (list, tuple)) and len(span) == 3:
            start, end, label = span
            normalized.append((int(start), int(end), str(label)))
        else:
            raise ValueError(f"Unrecognized span shape: {span!r}")
    return normalized


def _record_id(record: Dict[str, Any], index: int) -> str:
    """Return a stable id for *record* with a deterministic fallback.

    Args:
        record: Joined record.
        index: Zero-based record position, used to synthesize a
            fallback id when ``uid`` and ``doc_id`` are both missing.

    Returns:
        ``uid``, or ``"<source>:<doc_id>"``, or
        ``"<source>:idx-<14-digit-index>"``.
    """
    source = record.get("source", "unknown")
    uid = record.get("uid")
    if uid:
        return str(uid)
    doc_id = record.get("doc_id")
    if doc_id is not None:
        return f"{source}:{doc_id}"
    return f"{source}:idx-{index:014d}"


def _record_to_example(
    record: Dict[str, Any],
    index: int,
    label_allow: Optional[set],
    dropped_labels: set,
) -> Optional[NerExample]:
    """Convert one joined record to a ``NerExample``, or ``None``.

    Args:
        record: Joined record (text + annotations).
        index: Zero-based record position.
        label_allow: Optional set of accepted labels. When provided,
            spans whose label is not in the set are dropped (and the
            label is added to *dropped_labels* for a one-shot warning
            later).
        dropped_labels: Mutable accumulator collecting the names of
            labels filtered out of this dataset.

    Returns:
        A ``NerExample`` when the record carries a ``spans`` field,
        otherwise ``None``.
    """
    annotations = record.get("annotations") or {}
    raw_spans = annotations.get("spans")
    if raw_spans is None:
        return None

    spans = _normalize_spans(raw_spans)
    if label_allow is not None:
        filtered: List[Tuple[int, int, str]] = []
        for start, end, label in spans:
            if label in label_allow:
                filtered.append((start, end, label))
            else:
                dropped_labels.add(label)
        spans = filtered

    return NerExample(
        id=_record_id(record, index),
        text=record.get("text", ""),
        spans=[
            {"start": start, "end": end, "label": label}
            for start, end, label in spans
        ],
    )


def _iter_examples_for_entry(
    entry: TaskEntry,
    roots: TasksRoots,
) -> Iterator[Tuple[str, NerExample]]:
    """Yield ``(split, NerExample)`` pairs for every joined input record.

    Args:
        entry: NER task entry from the registry.
        roots: Filesystem roots.

    Yields:
        ``(split_name, example)`` pairs ready for
        `write_jsonl_splits`.

    Raises:
        FileNotFoundError: If a source ``<key>.jsonl`` is missing.
    """
    if entry.source.kind != "extracted":
        raise ValueError(
            f"to_spans only supports source.kind='extracted'; got "
            f"{entry.source.kind!r} for {entry.task}/{entry.dataset}."
        )

    label_allow: Optional[set] = (
        set(str(lbl) for lbl in entry.labels)
        if entry.labels is not None
        else None
    )
    dropped_labels: set = set()
    splits_keys = list(entry.splits.keys())
    index = 0

    for key in entry.source.keys:
        text_path = roots.extracted / f"{key}.jsonl"
        ann_path = roots.extracted / f"{key}.annotations.jsonl.gz"
        if not text_path.exists():
            raise FileNotFoundError(
                f"Source for {entry.task}/{entry.dataset}: "
                f"{text_path} does not exist."
            )
        ann_arg: Optional[Path] = ann_path if ann_path.exists() else None
        for record in iter_joined_records(text_path, ann_arg):
            example = _record_to_example(
                record, index, label_allow, dropped_labels
            )
            index += 1
            if example is None:
                continue
            split_key = (
                str(record.get("uid"))
                or str(record.get("doc_id"))
                or str(index)
            )
            split = hash_split(split_key, splits_keys)
            yield split, example

    if dropped_labels:
        logger.warning(
            "Entry %s/%s dropped spans with %d label(s) outside "
            "allow-list (%s): %s",
            entry.task,
            entry.dataset,
            len(dropped_labels),
            sorted(label_allow) if label_allow else "<empty>",
            sorted(dropped_labels),
        )


def convert_entry(
    key: str,
    entry: TaskEntry,
    roots: TasksRoots,
    force: bool = False,
) -> Optional[Dict[str, int]]:
    """Convert one NER entry, writing one file per declared split.

    Args:
        key: Entry key ``"<task>/<dataset>"`` (used for logging).
        entry: Parsed task entry.
        roots: Filesystem roots.
        force: When True, re-derive even if every split already
            exists.

    Returns:
        Mapping ``{split: written_count}``, or ``None`` when the
        outputs already existed and ``force`` is False.
    """
    output_dir = resolve_output_dir(entry, roots)
    outputs = outputs_for_splits(output_dir, entry.splits)

    if not force and all_outputs_exist(outputs):
        logger.info(
            "Skipping %s: every split already exists at %s "
            "(use --force to overwrite).",
            key, output_dir,
        )
        return None

    logger.info("Converting %s -> %s", key, output_dir)
    counts = write_jsonl_splits(
        _iter_examples_for_entry(entry, roots),
        outputs,
    )
    total = sum(counts.values())
    logger.info(
        "Wrote %d records across splits %s for %s",
        total, counts, key,
    )
    return counts


def _resolve_keys(
    entries: List[TaskEntry],
    requested: List[str],
    use_all: bool,
) -> List[str]:
    """Resolve which entry keys to process from CLI selection.

    Args:
        entries: Entries already filtered to this converter.
        requested: Positional ``<task>/<dataset>`` keys from the CLI.
        use_all: Whether ``--all`` was passed.

    Returns:
        List of entry keys, preserving registry order.

    Raises:
        SystemExit: If a requested key is unknown to this converter.
    """
    known = {f"{e.task}/{e.dataset}": e for e in entries}
    if use_all:
        return list(known.keys())
    unknown = [k for k in requested if k not in known]
    if unknown:
        logger.error(
            "Unknown entries for converter %s: %s. Known: %s",
            CONVERTER_NAME, unknown, sorted(known.keys()),
        )
        sys.exit(1)
    return list(requested)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert extracted NER datasets into GLiNER-style task "
            "JSONL under <roots.tasks>/<task>/<dataset>/<split>.jsonl.gz."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "entries",
        nargs="*",
        default=[],
        help=(
            "Entry keys to process, e.g. 'ner/ssj500k ner/suk'. "
            "Mutually exclusive with --all."
        ),
    )
    target.add_argument(
        "--all",
        action="store_true",
        help="Process every NER entry declared in tasks.yaml.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to tasks.yaml (default: configs/data/tasks.yaml)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-derive outputs even when every split already exists.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help=(
            "Process entries in parallel. 0=auto (cpu_count // 2), "
            "1=serial, N=N workers. Capped at the number of entries."
        ),
    )
    args = parser.parse_args(argv)
    if args.all and args.entries:
        parser.error("argument --all: not allowed with positional entries")
    if not args.all and not args.entries:
        parser.error("one of the arguments entries --all is required")
    return args


def main() -> None:
    """Run the NER conversion from CLI arguments."""
    args = parse_args()
    project_root = find_project_root()
    config_path = (
        args.config
        if args.config is not None
        else project_root / "configs" / "data" / "tasks.yaml"
    )

    tasks_config = load_tasks(config_path)
    entries = filter_for_converter(tasks_config, CONVERTER_NAME)
    by_key = {f"{e.task}/{e.dataset}": e for e in entries}

    keys = _resolve_keys(entries, args.entries, args.all)
    if not keys:
        logger.warning(
            "No entries to process for converter %s.", CONVERTER_NAME,
        )
        return

    workers = resolve_workers(
        args.max_workers, len(keys), cpu_default(len(keys)),
    )
    configure_script_logging(parallel=workers > 1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = project_root / "logs" / Path(__file__).stem / stamp

    roots = tasks_config.roots

    def kwargs_for(key: str) -> Dict[str, Any]:
        return {
            "entry": by_key[key],
            "roots": roots,
            "force": args.force,
        }

    results, failures = run_parallel(
        convert_entry,
        keys,
        max_workers=workers,
        desc=CONVERTER_NAME,
        pool="process",
        kwargs_for=kwargs_for,
        log_dir=log_dir,
    )

    skipped = [k for k, v in results.items() if v is None]
    total = sum(
        sum(v.values()) for v in results.values() if v is not None
    )
    logger.info(
        "Done. Processed %d entr(ies); %d skipped; %d records written. "
        "Failed: %s",
        len(results) - len(skipped),
        len(skipped),
        total,
        [k for k, _ in failures] or "none",
    )
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
