"""Convert SA datasets into task-shaped per-split JSONL.

Driven by ``configs/data/tasks.yaml``. Processes every entry whose
resolved converter is ``to_sentiment`` and emits one gzipped JSONL
per declared split under
``<roots.tasks>/<task>/<dataset>/<split>.jsonl.gz``.

Each output line is a ``SentimentExample``:

    {"id": "<source>:<doc_id>", "text": "<...>", "label": "negative"}

Two source kinds are supported:

* ``kind: extracted`` -- reads from ``<roots.extracted>/<key>.jsonl``
  (used by the SentiNews entry). Splits are derived from a
  deterministic 80/10/10 hash bucket over each record's ``uid`` /
  ``doc_id``.
* ``kind: raw`` -- reads SentiNews-format text files directly from
  ``<roots.raw>/<key>/`` (used by the held-out Twitter dataset).
  Every record lands in the single declared split (typically ``test``).
"""

import argparse
import csv
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
from slm4ie.data.schema import SentimentExample
from slm4ie.data.tasks import (
    TaskEntry,
    TasksRoots,
    filter_for_converter,
    load_tasks,
    resolve_output_dir,
)
from slm4ie.data.task_tracking import log_task_runs
from slm4ie.data.task_writer import (
    all_outputs_exist,
    hash_split,
    outputs_for_splits,
    write_jsonl_splits,
)

logger = logging.getLogger(__name__)

#: Converter module name, used to filter `tasks.yaml`.
CONVERTER_NAME: str = "to_sentiment"

#: Map common label spellings to canonical 3-class labels.
_LABEL_NORMALIZATION: Dict[str, str] = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "neg": "negative",
    "pos": "positive",
    "neu": "neutral",
}


def _normalize_label(raw_label: Optional[Any], allow: Optional[set]) -> Optional[str]:
    """Map a raw label string to the canonical form, if allowed.

    Args:
        raw_label: Label as it appears in the source. May be `None`.
        allow: Optional allow-list of canonical label names.

    Returns:
        The canonical label, or `None` when the value is empty, not
        recognized, or filtered out by *allow*.
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
    allow: Optional[set],
) -> Iterator[Tuple[str, SentimentExample, str]]:
    """Yield ``(split_key, example, label)`` tuples from extracted sources.

    The ``split_key`` is a stable string used by `hash_split` to assign
    the record to one of the entry's declared splits.

    Args:
        entry: Sentiment task entry with ``source.kind == 'extracted'``.
        roots: Filesystem roots.
        allow: Optional allow-list of canonical labels.

    Yields:
        ``(split_key, SentimentExample, label)`` triples. The ``label``
        is repeated for the caller's convenience but is identical to
        ``example["label"]``.

    Raises:
        FileNotFoundError: If a source JSONL is missing.
    """
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
            raw_label = (
                record.get("label")
                or record.get("sentiment")
                or (record.get("metadata") or {}).get("sentiment")
                or (record.get("metadata") or {}).get("label")
            )
            label = _normalize_label(raw_label, allow)
            if label is None:
                continue
            source = record.get("source", entry.dataset)
            uid = record.get("uid")
            doc_id = record.get("doc_id")
            if uid:
                example_id = str(uid)
                split_key = str(uid)
            elif doc_id is not None:
                example_id = f"{source}:{doc_id}"
                split_key = example_id
            else:
                example_id = f"{source}:idx-{index:014d}"
                split_key = example_id
            example: SentimentExample = SentimentExample(
                id=example_id,
                text=record.get("text", ""),
                label=label,
            )
            index += 1
            yield split_key, example, label


def _iter_raw_sentinews_format(
    source_dir: Path,
    dataset: str,
    allow: Optional[set],
) -> Iterator[Tuple[str, SentimentExample, str]]:
    """Yield records from SentiNews-format TSV files inside *source_dir*.

    The held-out Twitter sentiment dataset ships in the same
    ``SentiNews_<level>-level.txt`` layout, so a single TSV reader
    covers both. The caller decides how to assign splits.

    Args:
        source_dir: Directory containing ``SentiNews_*-level.*`` files.
        dataset: Dataset name used to synthesize record ids.
        allow: Optional allow-list of canonical labels.

    Yields:
        ``(split_key, SentimentExample, label)`` triples.

    Raises:
        FileNotFoundError: If no SentiNews-format files are present.
    """
    candidates: List[Path] = []
    for pattern in ("SentiNews_*-level.*", "*.tsv", "*.txt"):
        candidates.extend(sorted(source_dir.glob(pattern)))
    seen: set = set()
    files: List[Path] = []
    for path in candidates:
        if path.is_file() and path not in seen:
            files.append(path)
            seen.add(path)
    if not files:
        raise FileNotFoundError(
            f"No sentiment input files found in {source_dir}."
        )

    text_keys = ("content", "text", "sentence", "paragraph", "tweet")
    index = 0
    for path in files:
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                text = next(
                    (row[k] for k in text_keys if row.get(k)),
                    None,
                )
                raw_label = (
                    row.get("sentiment")
                    or row.get("label")
                    or row.get("polarity")
                )
                if text is None:
                    continue
                label = _normalize_label(raw_label, allow)
                if label is None:
                    continue
                nid = (
                    row.get("nid")
                    or row.get("doc_id")
                    or row.get("id")
                    or f"idx-{index:08d}"
                )
                example_id = f"{dataset}:{nid}"
                example: SentimentExample = SentimentExample(
                    id=example_id,
                    text=text,
                    label=label,
                )
                index += 1
                yield example_id, example, label


def _iter_raw(
    entry: TaskEntry,
    roots: TasksRoots,
    allow: Optional[set],
) -> Iterator[Tuple[str, SentimentExample, str]]:
    """Yield records from raw sentiment sources.

    Args:
        entry: Sentiment task entry with ``source.kind == 'raw'``.
        roots: Filesystem roots.
        allow: Optional allow-list of canonical labels.

    Yields:
        ``(split_key, SentimentExample, label)`` triples.

    Raises:
        FileNotFoundError: If a source directory is missing.
    """
    for key in entry.source.keys:
        source_dir = roots.raw / key
        if not source_dir.is_dir():
            raise FileNotFoundError(
                f"Raw source for {entry.task}/{entry.dataset}: "
                f"{source_dir} is not a directory."
            )
        yield from _iter_raw_sentinews_format(
            source_dir, entry.dataset, allow,
        )


def _iter_examples_for_entry(
    entry: TaskEntry,
    roots: TasksRoots,
) -> Iterator[Tuple[str, SentimentExample]]:
    """Yield ``(split, example)`` pairs ready to write.

    Args:
        entry: Sentiment task entry.
        roots: Filesystem roots.

    Yields:
        ``(split_name, SentimentExample)`` pairs.

    Raises:
        ValueError: If the entry source kind is unknown.
    """
    allow: Optional[set] = (
        set(str(lbl) for lbl in entry.labels)
        if entry.labels is not None
        else None
    )
    splits_keys = list(entry.splits.keys())
    only_split: Optional[str] = (
        splits_keys[0] if len(splits_keys) == 1 else None
    )

    if entry.source.kind == "extracted":
        record_iter = _iter_extracted(entry, roots, allow)
    elif entry.source.kind == "raw":
        record_iter = _iter_raw(entry, roots, allow)
    else:
        raise ValueError(
            f"Unknown source kind {entry.source.kind!r} for "
            f"{entry.task}/{entry.dataset}."
        )

    for split_key, example, _label in record_iter:
        split = (
            only_split
            if only_split is not None
            else hash_split(split_key, splits_keys)
        )
        yield split, example


def convert_entry(
    key: str,
    entry: TaskEntry,
    roots: TasksRoots,
    force: bool = False,
) -> Optional[Dict[str, int]]:
    """Convert one sentiment entry, writing one file per declared split.

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
            "Convert SA datasets into per-split task JSONL under "
            "<roots.tasks>/<task>/<dataset>/<split>.jsonl.gz."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "entries",
        nargs="*",
        default=[],
        help=(
            "Entry keys to process, e.g. 'sentiment/sentinews'. "
            "Mutually exclusive with --all."
        ),
    )
    target.add_argument(
        "--all",
        action="store_true",
        help="Process every sentiment entry declared in tasks.yaml.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to tasks.yaml (default: configs/data/tasks.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-derive outputs even when every split already exists.",
    )
    parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable/disable MLflow per-dataset tracking, overriding "
            "tasks.yaml::mlflow.enabled. Default: defer to config."
        ),
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
    """Run the sentiment conversion from CLI arguments."""
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
    processed = [k for k in keys if k not in {f for f, _ in failures}]
    log_task_runs(tasks_config, by_key, processed, mlflow_enabled=args.mlflow, force=args.force)
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
