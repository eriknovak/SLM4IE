"""Convert raw SuperGLUE-SL downloads into per-task evaluation JSONL.

The Slovene SuperGLUE distribution ships as a zip containing 8 task
subdirectories (BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC), each
with its own ``train.jsonl`` / ``val.jsonl`` / ``test.jsonl`` files
following the original SuperGLUE schemas. This script reads those raw
JSONL files and emits gzipped copies under
``<output_dir>/superglue_sl/<task>/<split>.jsonl.gz`` so they are easy
to consume from training pipelines while remaining structurally
faithful to the SloBENCH submission format.

Records are passed through largely unchanged. MultiRC, which has
deeply nested questions/answers, is flattened to one row per answer
(``--flatten`` style is the default for MultiRC; pass
``--no-flatten-multirc`` to disable).

Examples:
    Convert every available task and split from the HumanT variant:

        uv run python scripts/data/to_superglue.py --variant humant

    Convert only CB and RTE:

        uv run python scripts/data/to_superglue.py --tasks CB RTE
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
)

from tqdm import tqdm

from slm4ie.data.download import DatasetConfig, load_config
from slm4ie.data.io_utils import find_project_root, open_output

logger = logging.getLogger(__name__)

#: The eight SuperGLUE subtasks, in the canonical order.
SUPERGLUE_TASKS: Tuple[str, ...] = (
    "BoolQ",
    "CB",
    "COPA",
    "MultiRC",
    "ReCoRD",
    "RTE",
    "WiC",
    "WSC",
)

#: Splits to look for inside each task subdirectory. Test splits often
#: lack labels but are still emitted so the user can run inference.
SUPERGLUE_SPLITS: Tuple[str, ...] = ("train", "val", "test")

#: Variant suffixes used in the published distribution.
VARIANT_DIRS: Dict[str, Tuple[str, ...]] = {
    "humant": ("SuperGLUE-HumanT", "HumanT"),
    "googlemt": ("SuperGLUE-GoogleMT", "GoogleMT"),
}


def _find_variant_root(raw_dir: Path, variant: str) -> Path:
    """Return the directory holding task subdirectories for *variant*.

    Args:
        raw_dir (Path): The dataset raw-download directory.
        variant (str): One of ``humant`` / ``googlemt``.

    Returns:
        Path: The variant root directory containing per-task subdirs.

    Raises:
        FileNotFoundError: If no matching variant directory is found.
    """
    candidates = VARIANT_DIRS.get(variant, ())
    for name in candidates:
        path = raw_dir / name
        if path.is_dir():
            return path
    if (raw_dir / "BoolQ").is_dir():
        return raw_dir
    raise FileNotFoundError(
        f"Could not find a SuperGLUE {variant!r} variant directory in "
        f"{raw_dir}. Tried: {candidates}. Did you extract the zip?"
    )


def _find_task_dir(variant_root: Path, task: str) -> Optional[Path]:
    """Return the task subdirectory matching *task*, case-insensitively.

    Args:
        variant_root (Path): Variant root directory.
        task (str): Canonical task name (e.g. ``BoolQ``).

    Returns:
        Optional[Path]: The task subdirectory, or None when not found.
    """
    for child in variant_root.iterdir():
        if child.is_dir() and child.name.lower() == task.lower():
            return child
    return None


def _find_split_file(task_dir: Path, split: str) -> Optional[Path]:
    """Locate the JSONL file for *split* inside *task_dir*.

    Args:
        task_dir (Path): Task subdirectory.
        split (str): One of ``train``/``val``/``test``.

    Returns:
        Optional[Path]: Path to the source JSONL file, or None when
            the split is not present.
    """
    for ext in (".jsonl", ".json"):
        path = task_dir / f"{split}{ext}"
        if path.exists():
            return path
    return None


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file.

    Args:
        path (Path): Path to the JSONL file.

    Yields:
        Dict[str, Any]: Parsed records.
    """
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _passthrough(
    records: Iterable[Dict[str, Any]],
) -> Iterator[Dict[str, Any]]:
    """Yield records unchanged.

    Args:
        records (Iterable[Dict[str, Any]]): Source records.

    Yields:
        Dict[str, Any]: The same records.
    """
    yield from records


def _flatten_multirc(
    records: Iterable[Dict[str, Any]],
) -> Iterator[Dict[str, Any]]:
    """Flatten MultiRC records to one row per answer candidate.

    The native MultiRC schema nests questions and answers inside each
    passage. This helper emits one record per ``(passage, question,
    answer)`` triple, which is more convenient for classification.

    Args:
        records (Iterable[Dict[str, Any]]): Native MultiRC records
            with the nested ``passage.questions[].answers[]`` shape.

    Yields:
        Dict[str, Any]: Flat records with keys ``idx``, ``paragraph``,
            ``question``, ``answer``, and ``label`` (when available).
    """
    for record in records:
        passage = record.get("passage") or {}
        paragraph_text = passage.get("text", "")
        passage_idx = record.get("idx")
        for question in passage.get("questions", []) or []:
            q_text = question.get("question", "")
            q_idx = question.get("idx")
            for answer in question.get("answers", []) or []:
                yield {
                    "idx": {
                        "passage": passage_idx,
                        "question": q_idx,
                        "answer": answer.get("idx"),
                    },
                    "paragraph": paragraph_text,
                    "question": q_text,
                    "answer": answer.get("text", ""),
                    "label": answer.get("label"),
                }


#: Per-task transform registry. Each callable converts an iterable of
#: source records into the iterable to write to disk.
_TASK_TRANSFORMS: Dict[
    str,
    Callable[
        [Iterable[Dict[str, Any]]],
        Iterator[Dict[str, Any]],
    ],
] = {
    "BoolQ": _passthrough,
    "CB": _passthrough,
    "COPA": _passthrough,
    "MultiRC": _flatten_multirc,
    "ReCoRD": _passthrough,
    "RTE": _passthrough,
    "WiC": _passthrough,
    "WSC": _passthrough,
}


def write_records(
    records: Iterable[Dict[str, Any]],
    out_stream: IO[str],
) -> int:
    """Write *records* as JSONL lines and return the count.

    Args:
        records (Iterable[Dict[str, Any]]): Records to serialize.
        out_stream (IO[str]): Writable text stream.

    Returns:
        int: Number of records written.
    """
    count = 0
    for record in records:
        out_stream.write(json.dumps(record, ensure_ascii=False))
        out_stream.write("\n")
        count += 1
    return count


def convert_split(
    src_path: Path,
    out_path: Path,
    task: str,
    flatten_multirc: bool = True,
    force: bool = False,
) -> Optional[int]:
    """Convert one (task, split) pair to ``<out_path>``.

    Args:
        src_path (Path): Source JSONL path.
        out_path (Path): Target ``.jsonl.gz`` path. Parent directories
            are created if missing.
        task (str): Canonical task name (used for transform dispatch).
        flatten_multirc (bool): When True, apply the MultiRC flattener
            even though the registry has it as the default.
        force (bool): When True, overwrite an existing output file.

    Returns:
        Optional[int]: Number of records written, or 0 when the output
            already exists and *force* is False.
    """
    if out_path.exists() and not force:
        logger.info(
            "Skipping %s, output already exists: %s "
            "(use --force to overwrite)",
            task, out_path,
        )
        return 0

    transform = _TASK_TRANSFORMS[task]
    if task == "MultiRC" and not flatten_multirc:
        transform = _passthrough

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Converting %s → %s", src_path, out_path)
    records = transform(_read_jsonl(src_path))
    progress = tqdm(records, desc=f"{task}/{src_path.stem}", unit="rec")
    with open_output(out_path) as out_stream:
        try:
            count = write_records(progress, out_stream)
        finally:
            progress.close()
    logger.info("Wrote %d records to %s", count, out_path)
    return count


def convert_dataset(
    raw_dir: Path,
    output_dir: Path,
    variant: str,
    tasks: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
    flatten_multirc: bool = True,
    force: bool = False,
) -> Dict[Tuple[str, str], int]:
    """Convert SuperGLUE-SL into per-task per-split JSONL files.

    Args:
        raw_dir (Path): Directory containing the extracted SuperGLUE
            distribution (must contain a ``SuperGLUE-{HumanT,GoogleMT}``
            subdirectory).
        output_dir (Path): Base directory for outputs. Each (task,
            split) lands at ``<output_dir>/<task>/<split>.jsonl.gz``.
        variant (str): One of ``humant`` / ``googlemt``.
        tasks (Optional[List[str]]): Subset of tasks to process
            (default: all eight).
        splits (Optional[List[str]]): Subset of splits to process
            (default: ``train``/``val``/``test``).
        flatten_multirc (bool): Whether to flatten MultiRC.
        force (bool): When True, overwrite existing outputs.

    Returns:
        Dict[Tuple[str, str], int]: Map from (task, split) to record
            count for splits that were successfully written.
    """
    variant_root = _find_variant_root(raw_dir, variant)
    selected_tasks = list(tasks) if tasks else list(SUPERGLUE_TASKS)
    selected_splits = list(splits) if splits else list(SUPERGLUE_SPLITS)

    written: Dict[Tuple[str, str], int] = {}
    for task in selected_tasks:
        task_dir = _find_task_dir(variant_root, task)
        if task_dir is None:
            logger.warning(
                "Task %r not found under %s; skipping.",
                task, variant_root,
            )
            continue

        for split in selected_splits:
            src = _find_split_file(task_dir, split)
            if src is None:
                logger.info(
                    "No %s split for task %r; skipping.",
                    split, task,
                )
                continue
            out = output_dir / task / f"{split}.jsonl.gz"
            count = convert_split(
                src, out, task,
                flatten_multirc=flatten_multirc,
                force=force,
            )
            if count is not None:
                written[(task, split)] = count
    return written


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw SuperGLUE-SL downloads into per-task "
            "evaluation JSONL under "
            "<output_dir>/<task>/<split>.jsonl.gz."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to download.yaml (default: "
            "configs/data/download.yaml). Used to locate the raw "
            "download directory when --raw-dir is not given."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="superglue_sl",
        help=(
            "Dataset key in download.yaml (default: 'superglue_sl')."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Directory holding the extracted SuperGLUE-SL "
            "distribution. Defaults to <download output_dir>/<dataset>."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Base directory for converted output. Defaults to "
            "<raw-dir>/eval/superglue_sl/<variant>."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_DIRS.keys()),
        default="humant",
        help="SuperGLUE-SL variant to use (default: humant).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=SUPERGLUE_TASKS,
        default=None,
        help="Subset of tasks to materialize (default: all 8).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SUPERGLUE_SPLITS,
        default=None,
        help="Subset of splits to materialize (default: all).",
    )
    parser.add_argument(
        "--no-flatten-multirc",
        action="store_true",
        help=(
            "Keep MultiRC records in their native nested shape "
            "instead of flattening to one row per answer."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing per-split output files.",
    )
    return parser.parse_args(argv)


def _resolve_raw_dir(
    config_path: Path,
    override: Optional[Path],
    dataset_key: str,
) -> Path:
    """Return the raw input directory for the SuperGLUE-SL dataset.

    Args:
        config_path (Path): Path to download.yaml.
        override (Optional[Path]): Explicit ``--raw-dir`` value.
        dataset_key (str): Dataset key in the download config.

    Returns:
        Path: Directory expected to hold the extracted SuperGLUE
            distribution.
    """
    if override is not None:
        return override
    base, datasets = load_config(config_path)
    cfg: Optional[DatasetConfig] = datasets.get(dataset_key)
    subdir = cfg.output_dir if cfg and cfg.output_dir else dataset_key
    return Path(base) / subdir


def main():
    """Run the SuperGLUE conversion from CLI arguments."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    project_root = find_project_root()
    config_path = (
        args.config
        if args.config
        else project_root / "configs" / "data" / "download.yaml"
    )

    raw_dir = _resolve_raw_dir(config_path, args.raw_dir, args.dataset)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else raw_dir / "eval" / "superglue_sl" / args.variant
    )

    written = convert_dataset(
        raw_dir,
        output_dir,
        variant=args.variant,
        tasks=args.tasks,
        splits=args.splits,
        flatten_multirc=not args.no_flatten_multirc,
        force=args.force,
    )

    if not written:
        logger.error(
            "No splits were written. Check --raw-dir and --variant."
        )
        sys.exit(1)

    total = sum(written.values())
    logger.info(
        "Done. Wrote %d records across %d (task, split) pairs.",
        total, len(written),
    )


if __name__ == "__main__":
    main()
