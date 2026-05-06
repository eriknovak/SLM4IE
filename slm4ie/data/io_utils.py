"""Shared I/O helpers for SLM4IE data scripts.

These utilities are factored out of the individual scripts so multiple
entry points (to_datatrove, to_spans, ...) can reuse them without
duplicating logic.
"""

import gzip
import itertools
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Dict, Iterator, Optional, Tuple

import yaml


def find_project_root() -> Path:
    """Find the project root by locating ``pyproject.toml``.

    Walks up the directory tree starting from this file until a
    directory containing ``pyproject.toml`` is found.

    Returns:
        Path: The project root directory.

    Raises:
        FileNotFoundError: If no ``pyproject.toml`` is found in any
            parent directory.
    """
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        "Could not find project root (no pyproject.toml)"
    )


def resolve_processed_dir(
    config_path: Path,
    override: Optional[Path],
) -> Path:
    """Determine the directory holding the processed JSONL files.

    Args:
        config_path (Path): Path to the extraction YAML config.
        override (Optional[Path]): Explicit ``--processed-dir`` value;
            takes precedence over the config when not None.

    Returns:
        Path: Directory containing ``<key>.jsonl`` files.

    Raises:
        FileNotFoundError: If ``override`` is None and ``config_path``
            does not exist.
    """
    if override is not None:
        return override

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Pass --processed-dir to skip config lookup."
        )

    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)
    return Path(cfg.get("output_dir", "data/processed"))


@contextmanager
def open_output(path: Optional[Path]) -> Iterator[IO[str]]:
    """Yield a writable text stream for *path*.

    Selects gzip vs. plain text based on the file suffix. ``None``
    means write to ``sys.stdout``; in that case the underlying stream
    is **not** closed when the context exits.

    Args:
        path (Optional[Path]): Output path, or None for stdout.

    Yields:
        IO[str]: A writable text stream.
    """
    if path is None:
        yield sys.stdout
        return

    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            yield fh
    else:
        with path.open("w", encoding="utf-8") as fh:
            yield fh


def open_text_stream(path: Path) -> IO[str]:
    """Open a text file for reading, transparently handling gzip.

    The caller is responsible for closing the returned stream.

    Args:
        path (Path): Path to a ``.jsonl`` or ``.jsonl.gz`` file.

    Returns:
        IO[str]: A readable text stream.
    """
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def find_dataset_files(
    processed_dir: Path,
    key: str,
) -> Optional[Tuple[Path, Optional[Path]]]:
    """Locate the text JSONL and optional annotations sidecar for a dataset.

    Args:
        processed_dir (Path): Directory containing processed files.
        key (str): Dataset key.

    Returns:
        Optional[Tuple[Path, Optional[Path]]]: A tuple of
            ``(text_path, annotations_path)`` where ``annotations_path``
            is None when the dataset has no annotations sidecar.
            Returns None when no ``<key>.jsonl`` exists.
    """
    text_path = processed_dir / f"{key}.jsonl"
    if not text_path.exists():
        return None
    ann_path = processed_dir / f"{key}.annotations.jsonl.gz"
    return (text_path, ann_path if ann_path.exists() else None)


def iter_joined_records(
    text_path: Path,
    annotations_path: Optional[Path] = None,
) -> Iterator[Dict[str, Any]]:
    """Iterate records from *text_path*, attaching annotations when present.

    Both files are assumed to share document order. When the text and
    annotation records both carry a ``doc_id`` (or ``uid``) and they
    disagree, this is treated as a hard error: a mismatch indicates
    the files were produced by different runs.

    Args:
        text_path (Path): Path to ``<key>.jsonl``.
        annotations_path (Optional[Path]): Path to the annotations
            file, or None if the dataset has no annotations.

    Yields:
        Dict[str, Any]: A text record with, when available, an
            ``annotations`` field carrying the parallel-array payload.

    Raises:
        ValueError: If a ``doc_id``/``uid`` mismatch is detected or
            the two streams differ in length.
    """
    if annotations_path is None:
        with text_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    with text_path.open(encoding="utf-8") as text_fh, open_text_stream(
        annotations_path
    ) as ann_fh:
        for lineno, (text_line, ann_line) in enumerate(
            itertools.zip_longest(text_fh, ann_fh), start=1
        ):
            if text_line is None or ann_line is None:
                missing = "text" if text_line is None else "annotations"
                raise ValueError(
                    f"Line counts differ at line {lineno}: "
                    f"{missing} stream exhausted first"
                )

            text_record = json.loads(text_line)
            ann_record = json.loads(ann_line)

            text_id = text_record.get("doc_id")
            ann_id = ann_record.get("doc_id")
            if text_id is not None and ann_id is not None and text_id != ann_id:
                raise ValueError(
                    f"doc_id mismatch at line {lineno}: "
                    f"text={text_id!r} annotations={ann_id!r}"
                )

            text_uid = text_record.get("uid")
            ann_uid = ann_record.get("uid")
            if text_uid is not None and ann_uid is not None and text_uid != ann_uid:
                raise ValueError(
                    f"uid mismatch at line {lineno}: "
                    f"text={text_uid!r} annotations={ann_uid!r}"
                )

            ann_record.pop("doc_id", None)
            ann_record.pop("uid", None)
            text_record["annotations"] = ann_record
            yield text_record
