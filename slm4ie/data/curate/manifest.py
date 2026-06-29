"""Content digests over a corpus directory's output shards.

Gives each build of an extraction, pretraining, or task-conversion output a
stable identity: the digest changes if and only if the set of output shards
changes, and is reproduced exactly by a rerun that lands identical shards.
Downstream MLflow tracking upserts runs by this digest (one run per distinct
build) and declares it as dataset lineage so a consumer can point back to the
exact corpus it read.

The digest hashes a manifest of `(relative_path, size_bytes)` per shard --
pure `stat`, no byte reads -- so it stays cheap even over the multi-gigabyte
deduplicated corpus. Pass `with_rows=True` for a content-robust variant that
also folds in per-shard row counts (this decompresses every shard and is far
slower). Only the standard library is imported here so the digest can be
computed in contexts that do not install the heavier `curate` dependencies.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import List, Tuple

#: Glob patterns matched, relative to the corpus root, when building a manifest.
#: Covers plain and gzipped JSONL shards, the only output shapes the pipelines
#: emit. The two patterns are disjoint (a `.jsonl.gz` file never matches
#: `*.jsonl`), so no shard is counted twice.
DEFAULT_SHARD_GLOBS: Tuple[str, ...] = ("*.jsonl", "*.jsonl.gz")

#: Row-count value recorded for a shard when `with_rows` is False, i.e. the
#: count was deliberately not computed. Distinguishes "not counted" from a
#: genuine zero-row shard.
ROWS_NOT_COUNTED: int = -1


def _count_rows(path: Path) -> int:
    """Count newline-delimited records in a plain or gzipped JSONL shard.

    Counts newline bytes over the decompressed stream in megabyte chunks,
    skipping per-line decoding. A trailing record without a final newline is
    still counted.

    Args:
        path: Path to a `.jsonl` or `.jsonl.gz` file.

    Returns:
        The number of rows (documents) in the shard.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    last = b"\n"
    with opener(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            count += chunk.count(b"\n")
            last = chunk[-1:]
    if last not in (b"\n", b""):
        count += 1
    return count


def shard_manifest(
    root: Path,
    *,
    globs: Tuple[str, ...] = DEFAULT_SHARD_GLOBS,
    with_rows: bool = False,
) -> List[Tuple[str, int, int]]:
    """Build a sorted manifest of the output shards under a corpus root.

    Walks `root` recursively for files matching `globs` and records, per shard,
    its root-relative POSIX path and byte size (from `stat`). Row counts are
    included only when `with_rows` is True; otherwise each shard's count is
    `ROWS_NOT_COUNTED`. The result is sorted by relative path so it is stable
    across filesystem ordering.

    Args:
        root: Corpus directory to scan (e.g. `pretrain/05_2_dedup`).
        globs: Shard filename patterns to match relative to `root`.
        with_rows: When True, decompress each shard to count its rows; when
            False, leave row counts unset (much cheaper).

    Returns:
        A list of `(relative_path, size_bytes, row_count)` tuples sorted by
        relative path. `row_count` is `ROWS_NOT_COUNTED` when `with_rows` is
        False.

    Raises:
        FileNotFoundError: If `root` does not exist.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"corpus root does not exist: {root}")

    shards = {}
    for pattern in globs:
        for path in root.rglob(pattern):
            if path.is_file():
                shards[path.relative_to(root).as_posix()] = path

    manifest: List[Tuple[str, int, int]] = []
    for rel in sorted(shards):
        path = shards[rel]
        rows = _count_rows(path) if with_rows else ROWS_NOT_COUNTED
        manifest.append((rel, path.stat().st_size, rows))
    return manifest


def corpus_digest(
    root: Path,
    *,
    globs: Tuple[str, ...] = DEFAULT_SHARD_GLOBS,
    with_rows: bool = False,
) -> str:
    """Compute a stable content digest for a corpus directory.

    Hashes the shard manifest (see `shard_manifest`) as canonical
    tab-separated lines so the digest changes if and only if a shard's path,
    size, or -- when `with_rows` is True -- row count changes. Stable across
    reruns that reproduce identical shards. An existing root with no matching
    shards yields a well-defined empty-manifest digest.

    Args:
        root: Corpus directory to digest.
        globs: Shard filename patterns to match relative to `root`.
        with_rows: When True, fold per-shard row counts into the digest for a
            content-robust (but slower) identity; when False, use sizes only.

    Returns:
        A `"sha256:"`-prefixed lower-case hex digest, matching the project's
        sentinel hashing convention.

    Raises:
        FileNotFoundError: If `root` does not exist.
    """
    manifest = shard_manifest(root, globs=globs, with_rows=with_rows)
    h = hashlib.sha256()
    for rel, size, rows in manifest:
        h.update(f"{rel}\t{size}\t{rows}\n".encode("utf-8"))
    return f"sha256:{h.hexdigest()}"
