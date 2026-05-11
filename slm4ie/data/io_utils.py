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
from types import TracebackType
from typing import IO, Any, Dict, Iterator, Optional, Tuple, Type

import yaml

#: Default ceiling on the compressed size of one shard produced by
#: `ShardedJsonlWriter` (in bytes). 200 MB is sized for the curate
#: stage's typical 16–40-way parallelism: a 5 GB compressed dataset
#: yields ~25 shards (saturates 16 ranks), an 8 GB dataset ~40 shards
#: (saturates 40 ranks), while a 50 GB dataset still produces only
#: ~250 shards — trivial filesystem-metadata cost on any modern FS.
#: Per-shard fixed overhead in datatrove (open + JsonlReader setup) is
#: microseconds, so the per-document work dominates regardless. Override
#: with `--max-shard-bytes` when a specific dataset wants different
#: granularity.
DEFAULT_MAX_SHARD_BYTES: int = 200_000_000


def find_project_root() -> Path:
    """Find the project root by locating `pyproject.toml`.

    Walks up the directory tree starting from this file until a
    directory containing `pyproject.toml` is found.

    Returns:
        Path: The project root directory.

    Raises:
        FileNotFoundError: If no `pyproject.toml` is found in any
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
        override (Optional[Path]): Explicit `--processed-dir` value;
            takes precedence over the config when not None.

    Returns:
        Path: Directory containing `<key>.jsonl` files.

    Raises:
        FileNotFoundError: If `override` is None and `config_path`
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

    Selects gzip vs. plain text based on the file suffix. `None`
    means write to `sys.stdout`; in that case the underlying stream
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


class ShardedJsonlWriter:
    """Streaming writer that rolls over gzip shards by compressed size.

    Files land at `<folder>/<NNNNN>.jsonl.gz` (zero-padded 5-digit shard
    index, starting at `00000`). Rollover is triggered after writing a
    record when the current file's compressed size meets or exceeds
    `max_shard_bytes`. The folder is created on enter and each shard is
    closed cleanly so that gzip footers are flushed before the next
    shard opens.

    The compressed size is read from the underlying binary file's
    `.tell()` after each record, which gives the position in the
    *compressed* stream — this lets us cap shards by their final
    on-disk size without reopening or stat-ing them.

    Attributes:
        folder: Destination directory for the shard files. Created on
            enter; not removed on exit.
        max_shard_bytes: Compressed-byte ceiling that triggers rollover.
        shard_index: Zero-based index of the *currently open* shard.
            After exit, equals the index of the last written shard.
    """

    folder: Path
    max_shard_bytes: int
    shard_index: int

    def __init__(
        self,
        folder: Path,
        max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    ) -> None:
        """Configure the writer.

        Args:
            folder: Destination directory; created on `__enter__`.
            max_shard_bytes: Compressed-byte ceiling per shard. Must be
                positive.

        Raises:
            ValueError: If `max_shard_bytes` is not positive.
        """
        if max_shard_bytes <= 0:
            raise ValueError(
                f"max_shard_bytes must be positive, got {max_shard_bytes}"
            )
        self.folder = folder
        self.max_shard_bytes = max_shard_bytes
        self.shard_index = 0
        self._gz: Optional[gzip.GzipFile] = None
        self._raw: Optional[IO[bytes]] = None

    def __enter__(self) -> "ShardedJsonlWriter":
        """Create the folder and open the first shard.

        Returns:
            The writer itself, for use in `with` blocks.
        """
        self.folder.mkdir(parents=True, exist_ok=True)
        self._open_current()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        """Close the active shard, flushing the gzip footer.

        Args:
            exc_type: Exception type if the `with` block raised.
            exc: Exception value if the `with` block raised.
            tb: Traceback if the `with` block raised.
        """
        self._close_current()

    def write_record(self, record: Dict[str, Any]) -> None:
        """Write *record* as a JSON line and roll over if needed.

        Args:
            record: JSON-serializable payload. Encoded with
                `ensure_ascii=False` to preserve non-ASCII text.
        """
        self.write_line(json.dumps(record, ensure_ascii=False))

    def write_line(self, line: str) -> None:
        """Write a single JSON line (no trailing newline expected) and roll over if needed.

        Args:
            line: A serialized JSON object without a trailing newline.
                The newline is appended by this method.
        """
        assert self._gz is not None and self._raw is not None
        self._gz.write(line.encode("utf-8"))
        self._gz.write(b"\n")
        # `flush()` emits zlib's buffered output so the underlying file
        # position reflects current compressed size. Without it, the
        # zlib buffer (~32 KB) would mask the true size and rollover
        # would overshoot by a buffer's worth.
        self._gz.flush()
        if self._raw.tell() >= self.max_shard_bytes:
            self._close_current()
            self.shard_index += 1
            self._open_current()

    def _open_current(self) -> None:
        """Open the shard at `self.shard_index` for binary gzip writing."""
        path = self.folder / f"{self.shard_index:05d}.jsonl.gz"
        self._raw = path.open("wb")
        self._gz = gzip.GzipFile(filename=str(path), mode="wb", fileobj=self._raw)

    def _close_current(self) -> None:
        """Close the active gzip stream and underlying file."""
        if self._gz is not None:
            self._gz.close()
            self._gz = None
        if self._raw is not None:
            self._raw.close()
            self._raw = None


def open_text_stream(path: Path) -> IO[str]:
    """Open a text file for reading, transparently handling gzip.

    The caller is responsible for closing the returned stream.

    Args:
        path (Path): Path to a `.jsonl` or `.jsonl.gz` file.

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
            `(text_path, annotations_path)` where `annotations_path`
            is None when the dataset has no annotations sidecar.
            Returns None when no `<key>.jsonl` exists.
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
    annotation records both carry a `doc_id` (or `uid`) and they
    disagree, this is treated as a hard error: a mismatch indicates
    the files were produced by different runs.

    Args:
        text_path (Path): Path to `<key>.jsonl`.
        annotations_path (Optional[Path]): Path to the annotations
            file, or None if the dataset has no annotations.

    Yields:
        Dict[str, Any]: A text record with, when available, an
            `annotations` field carrying the parallel-array payload.

    Raises:
        ValueError: If a `doc_id`/`uid` mismatch is detected or
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
            if ann_record:
                text_record["annotations"] = ann_record
            yield text_record
