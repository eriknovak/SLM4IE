"""Convert SuperGLUE-SL subtasks into task-shaped per-split JSONL.

Driven by ``configs/data/tasks.yaml``. Processes every entry whose
resolved converter is ``to_superglue`` (the NLI / QA / coref / WSD /
COPA families) and emits one gzipped JSONL per declared split under
``<roots.tasks>/<task>/<dataset>/<split>.jsonl.gz``.

Each output line follows the appropriate ``TypedDict`` from
``slm4ie.data.schema``:

* ``nli/cb`` / ``nli/rte``  -> ``NliExample``
* ``qa/boolq``              -> ``QaBooleanExample``
* ``qa/multirc``            -> ``QaBooleanExample`` (one row per
  passage/question/answer triple)
* ``coref/wsc``             -> ``CorefExample``
* ``wsd/wic``               -> ``WsdExample``
* ``commonsense/copa``      -> ``CommonsenseCopaExample``

Source files are the per-subtask ``train.jsonl`` / ``val.jsonl`` /
``test.jsonl`` shipped inside the SuperGLUE-SL distribution. The
distribution is expected to be already extracted under
``<roots.raw>/superglue_sl/`` (e.g. ``SuperGLUE-HumanT/<Task>/``);
the ``--variant`` flag controls which translated variant is read.
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from slm4ie.data.io_utils import find_project_root
from slm4ie.data.parallel import (
    configure_script_logging,
    cpu_default,
    resolve_workers,
    run_parallel,
)
from slm4ie.data.schema import (
    CommonsenseCopaExample,
    CorefExample,
    NliExample,
    QaBooleanExample,
    WsdExample,
)
from slm4ie.data.tasks import (
    TaskEntry,
    TasksRoots,
    filter_for_converter,
    load_tasks,
    resolve_output_dir,
)
from slm4ie.data.task_writer import (
    all_outputs_exist,
    find_first_existing,
    iter_jsonl,
    outputs_for_splits,
    write_jsonl_splits,
)

logger = logging.getLogger(__name__)

#: Converter module name, used to filter `tasks.yaml`.
CONVERTER_NAME: str = "to_superglue"

#: Map each ``<task>/<dataset>`` to its SuperGLUE subtask directory.
SUBTASK_DIRS: Dict[str, str] = {
    "nli/cb": "CB",
    "nli/rte": "RTE",
    "qa/boolq": "BoolQ",
    "qa/multirc": "MultiRC",
    "coref/wsc": "WSC",
    "wsd/wic": "WiC",
    "commonsense/copa": "COPA",
}

#: Variant subdirectory candidates for each ``--variant`` value.
VARIANT_DIRS: Dict[str, Tuple[str, ...]] = {
    "humant": ("SuperGLUE-HumanT", "HumanT"),
    "googlemt": ("SuperGLUE-GoogleMT", "GoogleMT"),
}

#: Source filenames to try (per split) inside each subtask directory.
_SPLIT_FILENAMES: Dict[str, Tuple[str, ...]] = {
    "train": ("train.jsonl", "train.json"),
    "val": ("val.jsonl", "val.json"),
    "test": ("test.jsonl", "test.json"),
}


def _find_variant_root(raw_dir: Path, variant: str) -> Path:
    """Return the directory holding subtask subdirectories.

    Args:
        raw_dir: Root of the SuperGLUE-SL raw bundle
            (``<roots.raw>/superglue_sl``).
        variant: One of ``humant`` / ``googlemt``.

    Returns:
        Directory that contains per-subtask folders.

    Raises:
        FileNotFoundError: If no candidate matches on disk.
    """
    candidates = VARIANT_DIRS.get(variant, ())
    for name in candidates:
        path = raw_dir / name
        if path.is_dir():
            return path
    if (raw_dir / "BoolQ").is_dir():
        return raw_dir
    raise FileNotFoundError(
        f"Could not find SuperGLUE {variant!r} variant directory in "
        f"{raw_dir}. Tried: {candidates}."
    )


def _find_subtask_dir(variant_root: Path, subtask: str) -> Optional[Path]:
    """Return the subtask subdirectory matching *subtask* case-insensitively.

    Args:
        variant_root: Variant root.
        subtask: Canonical subtask name (e.g. ``BoolQ``).

    Returns:
        The matching directory, or ``None`` when absent.
    """
    if not variant_root.is_dir():
        return None
    for child in variant_root.iterdir():
        if child.is_dir() and child.name.lower() == subtask.lower():
            return child
    return None


def _source_path_for_split(
    subtask_dir: Path,
    split: str,
) -> Optional[Path]:
    """Return the source JSONL path for *split* inside *subtask_dir*.

    Args:
        subtask_dir: Per-subtask directory.
        split: Output split name (``train`` / ``val`` / ``test``).

    Returns:
        First existing candidate path, or ``None`` when absent.
    """
    names = _SPLIT_FILENAMES.get(split, ())
    return find_first_existing([subtask_dir / name for name in names])


def _coerce_bool(value: Any) -> Optional[bool]:
    """Coerce common boolean spellings to ``bool``.

    Args:
        value: A value pulled from a record.

    Returns:
        ``True``/``False`` for recognized inputs, ``None`` otherwise.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "t"}:
            return True
        if lowered in {"false", "0", "no", "n", "f"}:
            return False
    return None


def _stable_id(record: Dict[str, Any], dataset: str, index: int) -> str:
    """Synthesize a stable id for a SuperGLUE record.

    Args:
        record: Source record.
        dataset: ``entry.dataset`` (used as prefix when ``idx`` is
            available without one).
        index: Zero-based record position; used as fallback.

    Returns:
        A string id that is unique within the originating split.
    """
    raw = record.get("id") or record.get("idx")
    if isinstance(raw, (str, int)):
        text = str(raw)
        if ":" in text:
            return text
        return f"{dataset}:{text}"
    if isinstance(raw, dict):
        parts = [str(v) for v in raw.values() if v is not None]
        if parts:
            return f"{dataset}:{'-'.join(parts)}"
    return f"{dataset}:idx-{index:08d}"


def _convert_nli(
    record: Dict[str, Any],
    dataset: str,
    index: int,
    allow: Optional[set],
) -> Optional[Tuple[Optional[str], NliExample]]:
    """Convert one CB/RTE record to ``(split_hint, NliExample)``.

    Args:
        record: Source record.
        dataset: Entry dataset name.
        index: Zero-based record position.
        allow: Optional label allow-list.

    Returns:
        ``(None, example)`` because NLI records are kept in the split
        they came from. Returns ``None`` when required fields are
        missing.
    """
    premise = record.get("premise")
    hypothesis = record.get("hypothesis")
    label = record.get("label")
    if premise is None or hypothesis is None:
        return None
    label_str = str(label) if label is not None else ""
    if allow is not None and label_str and label_str not in allow:
        return None
    example = NliExample(
        id=_stable_id(record, dataset, index),
        premise=str(premise),
        hypothesis=str(hypothesis),
        label=label_str,
    )
    return None, example


def _convert_boolq(
    record: Dict[str, Any],
    dataset: str,
    index: int,
    allow: Optional[set],
) -> Optional[Tuple[Optional[str], QaBooleanExample]]:
    """Convert one BoolQ record to ``QaBooleanExample``.

    Args:
        record: Source record.
        dataset: Entry dataset name.
        index: Zero-based record position.
        allow: Unused (boolean labels are not filtered).

    Returns:
        ``(None, example)``, or ``None`` when required fields are
        missing.
    """
    del allow
    passage = record.get("passage") or record.get("paragraph")
    question = record.get("question")
    if passage is None or question is None:
        return None
    label = _coerce_bool(record.get("label"))
    if label is None:
        label = False
    example = QaBooleanExample(
        id=_stable_id(record, dataset, index),
        passage=str(passage),
        question=str(question),
        label=label,
    )
    return None, example


def _convert_wsc(
    record: Dict[str, Any],
    dataset: str,
    index: int,
    allow: Optional[set],
) -> Optional[Tuple[Optional[str], CorefExample]]:
    """Convert one WSC record to ``CorefExample``.

    Args:
        record: Source record with ``target`` span1/span2 fields.
        dataset: Entry dataset name.
        index: Zero-based record position.
        allow: Unused.

    Returns:
        ``(None, example)``, or ``None`` when required fields are
        missing.
    """
    del allow
    text = record.get("text")
    target = record.get("target") or {}
    span1_text = target.get("span1_text") or record.get("span1_text")
    span2_text = target.get("span2_text") or record.get("span2_text")
    span1_index = target.get("span1_index", record.get("span1_index"))
    span2_index = target.get("span2_index", record.get("span2_index"))
    if text is None or span1_text is None or span2_text is None:
        return None

    span1: Dict[str, Any] = {"text": str(span1_text)}
    span2: Dict[str, Any] = {"text": str(span2_text)}
    if span1_index is not None:
        span1["start"] = int(span1_index)
        span1["end"] = int(span1_index) + len(str(span1_text).split())
    if span2_index is not None:
        span2["start"] = int(span2_index)
        span2["end"] = int(span2_index) + len(str(span2_text).split())

    label = _coerce_bool(record.get("label"))
    if label is None:
        label = False
    example = CorefExample(
        id=_stable_id(record, dataset, index),
        text=str(text),
        span1=span1,
        span2=span2,
        label=label,
    )
    return None, example


def _convert_wic(
    record: Dict[str, Any],
    dataset: str,
    index: int,
    allow: Optional[set],
) -> Optional[Tuple[Optional[str], WsdExample]]:
    """Convert one WiC record to ``WsdExample``.

    Args:
        record: Source record.
        dataset: Entry dataset name.
        index: Zero-based record position.
        allow: Unused.

    Returns:
        ``(None, example)``, or ``None`` when required fields are
        missing.
    """
    del allow
    sentence1 = record.get("sentence1")
    sentence2 = record.get("sentence2")
    word = record.get("word")
    if sentence1 is None or sentence2 is None or word is None:
        return None
    label = _coerce_bool(record.get("label"))
    if label is None:
        label = False
    example = WsdExample(
        id=_stable_id(record, dataset, index),
        sentence1=str(sentence1),
        sentence2=str(sentence2),
        word=str(word),
        label=label,
    )
    return None, example


def _convert_copa(
    record: Dict[str, Any],
    dataset: str,
    index: int,
    allow: Optional[set],
) -> Optional[Tuple[Optional[str], CommonsenseCopaExample]]:
    """Convert one COPA record to ``CommonsenseCopaExample``.

    Args:
        record: Source record.
        dataset: Entry dataset name.
        index: Zero-based record position.
        allow: Unused (the registry's COPA labels are ``[0, 1]``).

    Returns:
        ``(None, example)``, or ``None`` when required fields are
        missing.
    """
    del allow
    premise = record.get("premise")
    choice1 = record.get("choice1")
    choice2 = record.get("choice2")
    question = record.get("question", "cause")
    if premise is None or choice1 is None or choice2 is None:
        return None
    raw_label = record.get("label")
    try:
        label_int = int(raw_label) if raw_label is not None else 0
    except (TypeError, ValueError):
        label_int = 0
    example = CommonsenseCopaExample(
        id=_stable_id(record, dataset, index),
        premise=str(premise),
        choice1=str(choice1),
        choice2=str(choice2),
        question=str(question),
        label=label_int,
    )
    return None, example


def _iter_multirc(
    record: Dict[str, Any],
    dataset: str,
    base_index: int,
) -> Iterator[QaBooleanExample]:
    """Flatten one MultiRC passage into per-answer ``QaBooleanExample`` rows.

    Args:
        record: Native MultiRC record with nested
            ``passage.questions[].answers[]`` structure.
        dataset: Entry dataset name.
        base_index: Index of the parent record, used to synthesize
            stable ids when explicit ``idx`` fields are absent.

    Yields:
        One ``QaBooleanExample`` per ``(passage, question, answer)``
        triple.
    """
    passage = record.get("passage") or {}
    paragraph_text = passage.get("text", "")
    passage_idx = record.get("idx", base_index)
    for question in passage.get("questions") or []:
        q_text = question.get("question", "")
        q_idx = question.get("idx")
        for answer in question.get("answers") or []:
            ans_idx = answer.get("idx")
            parts = [str(passage_idx)]
            if q_idx is not None:
                parts.append(f"q{q_idx}")
            if ans_idx is not None:
                parts.append(f"a{ans_idx}")
            example_id = f"{dataset}:{'-'.join(parts)}"
            label = _coerce_bool(answer.get("label"))
            if label is None:
                label = False
            combined_question = f"{q_text}\n{answer.get('text', '')}".strip()
            yield QaBooleanExample(
                id=example_id,
                passage=str(paragraph_text),
                question=combined_question,
                label=label,
            )


#: Per-task-family converter callable. Returns ``(split_override,
#: example)`` -- when ``split_override`` is ``None``, the example
#: stays in its source split.
_RECORD_CONVERTERS: Dict[
    str,
    Callable[
        [Dict[str, Any], str, int, Optional[set]],
        Optional[Tuple[Optional[str], Any]],
    ],
] = {
    "nli/cb": _convert_nli,
    "nli/rte": _convert_nli,
    "qa/boolq": _convert_boolq,
    "coref/wsc": _convert_wsc,
    "wsd/wic": _convert_wic,
    "commonsense/copa": _convert_copa,
}


def _iter_examples_for_entry(
    entry: TaskEntry,
    roots: TasksRoots,
    variant: str,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield ``(split, example)`` pairs for one SuperGLUE entry.

    Args:
        entry: Task entry assigned to ``to_superglue``.
        roots: Filesystem roots.
        variant: SuperGLUE-SL variant (``humant`` / ``googlemt``).

    Yields:
        ``(split_name, example_dict)`` pairs.

    Raises:
        FileNotFoundError: If the SuperGLUE-SL bundle is missing.
        ValueError: If the entry has no registered subtask mapping.
    """
    key = f"{entry.task}/{entry.dataset}"
    subtask = SUBTASK_DIRS.get(key)
    if subtask is None:
        raise ValueError(
            f"No SuperGLUE subtask mapping registered for {key!r}."
        )

    if entry.source.kind != "raw":
        raise ValueError(
            f"to_superglue only supports source.kind='raw'; got "
            f"{entry.source.kind!r} for {key}."
        )

    bundle_dirs = [roots.raw / src_key for src_key in entry.source.keys]
    bundle = bundle_dirs[0]
    variant_root = _find_variant_root(bundle, variant)
    subtask_dir = _find_subtask_dir(variant_root, subtask)
    if subtask_dir is None:
        raise FileNotFoundError(
            f"Subtask directory {subtask!r} not found under "
            f"{variant_root}."
        )

    allow: Optional[set] = (
        set(str(lbl) for lbl in entry.labels)
        if entry.labels is not None
        else None
    )

    is_multirc = key == "qa/multirc"
    converter = _RECORD_CONVERTERS.get(key) if not is_multirc else None

    for split in entry.splits:
        src_path = _source_path_for_split(subtask_dir, split)
        if src_path is None:
            logger.warning(
                "No source file for split %r of %s in %s; skipping split.",
                split, key, subtask_dir,
            )
            continue
        for index, record in enumerate(iter_jsonl(src_path)):
            if is_multirc:
                for example in _iter_multirc(record, entry.dataset, index):
                    yield split, dict(example)
                continue
            assert converter is not None
            converted = converter(record, entry.dataset, index, allow)
            if converted is None:
                continue
            split_override, example = converted
            yield split_override or split, dict(example)


def convert_entry(
    key: str,
    entry: TaskEntry,
    roots: TasksRoots,
    variant: str,
    force: bool = False,
) -> Optional[Dict[str, int]]:
    """Convert one SuperGLUE entry, writing one file per declared split.

    Args:
        key: Entry key ``"<task>/<dataset>"`` (used for logging).
        entry: Parsed task entry.
        roots: Filesystem roots.
        variant: SuperGLUE-SL variant (``humant`` / ``googlemt``).
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
        _iter_examples_for_entry(entry, roots, variant),
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
            "Convert SuperGLUE-SL subtasks into per-split task JSONL "
            "under <roots.tasks>/<task>/<dataset>/<split>.jsonl.gz."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "entries",
        nargs="*",
        default=[],
        help=(
            "Entry keys to process, e.g. 'nli/cb qa/boolq'. "
            "Mutually exclusive with --all."
        ),
    )
    target.add_argument(
        "--all",
        action="store_true",
        help="Process every SuperGLUE entry declared in tasks.yaml.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to tasks.yaml (default: configs/data/tasks.yaml).",
    )
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_DIRS.keys()),
        default="humant",
        help="SuperGLUE-SL variant to read (default: humant).",
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
    """Run the SuperGLUE conversion from CLI arguments."""
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
            "variant": args.variant,
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
