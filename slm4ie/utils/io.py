"""File-system plumbing shared by every pipeline.

Project-root resolution, gzip-aware text streams and the sharded JSONL
writer. Nothing here knows the data tiers; readers of a specific tier live
next to the step that writes it (for example `slm4ie.data.extract.records`).
"""

import gzip
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Dict, Iterator, Optional, Type, Union


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
    raise FileNotFoundError("Could not find project root (no pyproject.toml)")


def resolve_project_path(
    value: Union[str, Path],
    root: Optional[Path] = None,
) -> Path:
    """Anchor a config path against the project root.

    Absolute paths pass through unchanged; relative paths are joined onto
    the project root so resolution no longer depends on the current working
    directory. Use this when turning a relative path read from a config file
    into a filesystem path.

    Args:
        value: Path string or Path, typically read from a YAML config.
        root: Project root to anchor relative values against. Defaults to
            the result of `find_project_root`.

    Returns:
        An absolute Path.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    return (root or find_project_root()) / path


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
            raise ValueError(f"max_shard_bytes must be positive, got {max_shard_bytes}")
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
