"""Content versioning for pipeline outputs: document digests, integrity, lock files.

A pipeline stage's output is versioned by what it holds, not by how it was laid
out on disk. The pieces here are pipeline-agnostic so any route that writes
JSONL documents can adopt them; the curation runner is the first consumer.

* Document digest: an order-independent hash over a set of documents — the sum
  modulo 2^256 of each document's SHA-256 over its canonical JSON. It ignores
  shard names, worker counts, compression and timestamps, so it changes only
  when the documents do, and digests of disjoint parts merge by addition.
* Integrity check: every id appears in a stage's output no more often than in
  its input, and the output holds no more records than the input. Stray shards
  from an earlier run break the first rule.
* Shard set: the files a unit wrote, by relative path and byte size. A mismatch
  means the output changed behind the pipeline's back; mtimes are ignored so
  copying or touching files is not a change.
* Lock file: a committed YAML snapshot of every unit's hashes and digests.
* Config hash and corpus digest: the cheap build identities every route's
  tracking uses — a hash of the config slice that produced an output, and a
  hash of its shard manifest (path + size, no reads).

Only the standard library and PyYAML are imported at module level, so every
route can use this module without curation's dependencies; NumPy and orjson
are imported inside the scanning functions that need them.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from array import array
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import yaml

if TYPE_CHECKING:
    import numpy as np

K = TypeVar("K", bound=Hashable)

#: Metadata keys that record where a document was read from, not what it is.
VOLATILE_METADATA_KEYS = frozenset({"file_path"})

#: Prefix naming the digest scheme, so a future scheme cannot be mistaken for it.
_DIGEST_PREFIX = "sum256:"

_MODULUS = 1 << 256

#: Digest of an empty document set.
EMPTY_DIGEST = _DIGEST_PREFIX + "0" * 64


@dataclass(frozen=True)
class UnitScan:
    """What one read pass over a unit's documents found.

    Attributes:
        records: Number of documents read.
        document_digest: Document digest of the documents, or `None` when the
            scan was asked for ids only.
        id_hashes: Sorted 64-bit hashes of every document id, one per
            document (repeats kept), for the integrity check.
        raw_sha256: SHA-256 of the files' bytes in the order read, when asked
            for; lets one read serve both the id scan and a file hash.
    """

    records: int
    document_digest: Optional[str]
    id_hashes: np.ndarray
    raw_sha256: Optional[str] = None


def _format_digest(total: int) -> str:
    """Render a digest sum as its prefixed hex string."""
    return f"{_DIGEST_PREFIX}{total % _MODULUS:064x}"


def _parse_digest(digest: str) -> int:
    """Return the integer sum behind a digest string.

    Raises:
        ValueError: If *digest* is not a document digest.
    """
    if not digest.startswith(_DIGEST_PREFIX):
        raise ValueError(f"not a document digest: {digest!r}")
    return int(digest[len(_DIGEST_PREFIX) :], 16)


def _document_hash(record: Dict[str, Any]) -> int:
    """Return the SHA-256 of *record*'s canonical JSON as an integer.

    Args:
        record: One parsed document; volatile metadata keys are dropped.

    Returns:
        The 256-bit hash as an integer.
    """
    import orjson

    metadata = record.get("metadata")
    if isinstance(metadata, dict) and VOLATILE_METADATA_KEYS & metadata.keys():
        record = {**record, "metadata": {k: v for k, v in metadata.items() if k not in VOLATILE_METADATA_KEYS}}
    canonical = orjson.dumps(record, option=orjson.OPT_SORT_KEYS)
    return int.from_bytes(hashlib.sha256(canonical).digest(), "big")


def _id_hash(value: Any) -> int:
    """Return a 64-bit hash of a document id."""
    return int.from_bytes(hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest(), "big")


def merge_scans(scans: Sequence[UnitScan]) -> UnitScan:
    """Merge scans of disjoint file sets into the scan of all of them.

    Args:
        scans: Scans to merge; all must have been taken with the same `digest`
            setting.

    Returns:
        The combined `UnitScan`.
    """
    import numpy as np

    digests = [s.document_digest for s in scans]
    merged = None if any(d is None for d in digests) else merge_digests(d for d in digests if d is not None)
    ids = np.sort(np.concatenate([s.id_hashes for s in scans])) if scans else np.empty(0, dtype=np.uint64)
    return UnitScan(records=sum(s.records for s in scans), document_digest=merged, id_hashes=ids)


@dataclass(frozen=True)
class ScanRequest:
    """What to read for one unit and what to compute from it.

    Attributes:
        files: Plain or gzipped JSONL files.
        id_key: Top-level field holding each document's id.
        digest: Compute the document digest.
        raw_sha256: Also hash the raw bytes (meaningful for one plain file).
    """

    files: Sequence[Path]
    id_key: str = "id"
    digest: bool = True
    raw_sha256: bool = False


#: Bytes per sequential read, and the most a worker is handed of a plain file.
_READ_BYTES = 64 << 20

#: Upper bound on bytes read ahead of the workers, to cap page-cache use.
_MAX_IN_FLIGHT_BYTES = 8 << 30


def _plan_ranges(path: Path, buffer: bytearray, sha: Optional[Any]) -> Iterator[Tuple[int, int]]:
    """Read *path* once, sequentially, and yield the byte ranges workers should scan.

    The read goes into one reused *buffer*, so it costs no allocation or page
    faults; its purpose is to pull the file into the page cache in disk order,
    from where the workers reread their ranges without seeking. A plain file
    is cut after the last newline of each block, so every range holds whole
    lines; a gzipped file cannot be split and is one range.

    Args:
        path: File to read.
        buffer: Reusable read buffer; its length is the block size.
        sha: A hash object to feed the raw bytes to, or `None`.

    Yields:
        `(start, end)` byte offsets in file order, covering the file.
    """
    view = memoryview(buffer)
    gz = path.suffix == ".gz"
    start = offset = 0
    with open(path, "rb", buffering=0) as fh:
        # A larger kernel readahead keeps the disk head on this file.
        os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
        while n := fh.readinto(view):
            if sha is not None:
                sha.update(view[:n])
            cut = 0 if gz else buffer.rfind(b"\n", 0, n) + 1
            if cut:
                yield start, offset + cut
                start = offset + cut
            offset += n
    if offset > start:
        yield start, offset


def _scan_stream(stream: Iterable[bytes], id_key: str, digest: bool) -> UnitScan:
    """Scan the JSONL lines of *stream*.

    Args:
        stream: Lines of JSONL.
        id_key: Top-level field holding each document's id.
        digest: Compute the document digest.

    Returns:
        The `UnitScan` of the lines.
    """
    import numpy as np
    import orjson

    total = 0
    records = 0
    ids = array("Q")
    for line in stream:
        if not line.strip():
            continue
        record = orjson.loads(line)
        records += 1
        ids.append(_id_hash(record.get(id_key)))
        if digest:
            total += _document_hash(record)
    id_hashes = np.sort(np.frombuffer(ids, dtype=np.uint64)) if records else np.empty(0, dtype=np.uint64)
    return UnitScan(records=records, document_digest=_format_digest(total) if digest else None, id_hashes=id_hashes)


def _scan_range(path: Path, start: int, end: int, id_key: str, digest: bool) -> UnitScan:
    """Scan one range planned by `_plan_ranges`; runs in a worker.

    Args:
        path: The file.
        start: First byte of the range.
        end: One past its last byte.
        id_key: Top-level field holding each document's id.
        digest: Compute the document digest.

    Returns:
        The `UnitScan` of the range.
    """
    with open(path, "rb") as fh:
        if path.suffix == ".gz":
            with gzip.GzipFile(fileobj=fh) as stream:
                return _scan_stream(stream, id_key, digest)
        fh.seek(start)
        return _scan_stream(io.BytesIO(fh.read(end - start)), id_key, digest)


def scan_units(
    requests: Mapping[K, ScanRequest], workers: int = 1, progress: Optional[Callable[[Path, int], None]] = None
) -> Dict[K, UnitScan]:
    """Scan many units with one sequential reader feeding a pool of workers.

    On a spinning disk, parallel readers make the head seek between files and
    throughput collapses. One reader streams each file in disk order into the
    page cache and hands byte ranges, not bytes, to *workers* processes that
    reread them from memory and do the parsing and hashing.

    Args:
        requests: What to scan per unit key.
        workers: Worker processes for parsing and hashing; 1 runs in-process.
        progress: Called with each file and its size once it has been read.

    Returns:
        The `UnitScan` per unit key; `raw_sha256` is set where requested.
    """
    parts: Dict[K, List[UnitScan]] = {key: [] for key in requests}
    raw: Dict[K, str] = {}
    buffer = bytearray(_READ_BYTES)
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    pending: Dict[Future, Tuple[K, int]] = {}
    in_flight = 0

    def collect(done: Iterable[Future]) -> None:
        nonlocal in_flight
        for future in done:
            key, size = pending.pop(future)
            parts[key].append(future.result())
            in_flight -= size

    try:
        for key, request in requests.items():
            for path in request.files:
                sha = hashlib.sha256() if request.raw_sha256 else None
                for start, end in _plan_ranges(path, buffer, sha):
                    if pool is None:
                        parts[key].append(_scan_range(path, start, end, request.id_key, request.digest))
                        continue
                    while pending and (len(pending) >= 2 * workers or in_flight >= _MAX_IN_FLIGHT_BYTES):
                        collect(wait(pending, return_when=FIRST_COMPLETED).done)
                    future = pool.submit(_scan_range, path, start, end, request.id_key, request.digest)
                    pending[future] = (key, end - start)
                    in_flight += end - start
                if sha is not None:
                    raw[key] = sha.hexdigest()
                if progress is not None:
                    progress(path, path.stat().st_size)
        collect(wait(pending).done)
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)

    scans: Dict[K, UnitScan] = {}
    for key, request in requests.items():
        merged = merge_scans(parts[key])
        document_digest = (merged.document_digest or EMPTY_DIGEST) if request.digest else None
        scans[key] = UnitScan(merged.records, document_digest, merged.id_hashes, raw.get(key))
    return scans


def scan_documents(
    files: Iterable[Path], *, id_key: str = "id", digest: bool = True, raw_sha256: bool = False
) -> UnitScan:
    """Read JSONL documents once, in-process, collecting ids and optionally the digest.

    Args:
        files: Plain or gzipped (`.gz`) JSONL files, in any order.
        id_key: Top-level field holding each document's id.
        digest: Compute the document digest; skip it when only the ids are
            needed (e.g. for a stage's input side).
        raw_sha256: Also hash the raw bytes read (meaningful for one plain file).

    Returns:
        The `UnitScan` of every document across *files*.
    """
    return scan_units({0: ScanRequest(list(files), id_key, digest, raw_sha256)})[0]


def scan_files(files: Sequence[Path], *, id_key: str = "id", digest: bool = True, workers: int = 1) -> UnitScan:
    """Scan *files* as one unit; see `scan_units`.

    Args:
        files: Plain or gzipped JSONL files.
        id_key: Top-level field holding each document's id.
        digest: Compute the document digest.
        workers: Worker processes for parsing and hashing.

    Returns:
        The `UnitScan` of every document across *files*.
    """
    return scan_units({0: ScanRequest(files, id_key, digest)}, workers)[0]


def shard_files(folder: Path) -> List[Path]:
    """Return the gzipped JSONL shards under *folder*, sorted.

    Args:
        folder: A unit's output folder.

    Returns:
        Every `*.jsonl.gz` below *folder*; empty when it does not exist.
    """
    return sorted(folder.rglob("*.jsonl.gz")) if folder.is_dir() else []


def check_integrity(inp: UnitScan, out: UnitScan) -> Optional[str]:
    """Check that a stage's output could have come from its input.

    Ids need not be unique: a source may repeat an id, so each id may appear
    in the output at most as often as in the input.

    Args:
        inp: Scan of the stage's input.
        out: Scan of the stage's output.

    Returns:
        A description of every violation, or `None` when the output passes.
    """
    import numpy as np

    problems = []
    if out.records > inp.records:
        problems.append(f"records_out {out.records} > records_in {inp.records}")
    out_ids, out_counts = np.unique(out.id_hashes, return_counts=True)
    in_ids, in_counts = np.unique(inp.id_hashes, return_counts=True)
    pos = np.searchsorted(in_ids, out_ids)
    found = pos < len(in_ids)
    found[found] = in_ids[pos[found]] == out_ids[found]
    allowed = np.zeros(len(out_ids), dtype=np.int64)
    allowed[found] = in_counts[pos[found]]
    excess = int(np.count_nonzero(out_counts > allowed))
    if excess:
        problems.append(f"{excess} id(s) appear more often in the output than in the input")
    return "; ".join(problems) or None


def merge_digests(digests: Iterable[str]) -> str:
    """Merge document digests of disjoint document sets into their union's digest.

    Args:
        digests: Document digests to merge.

    Returns:
        The digest of all their documents together.
    """
    return _format_digest(sum(_parse_digest(d) for d in digests))


def combine_named_digests(digests: Mapping[str, Optional[str]]) -> str:
    """Hash a key-to-digest mapping, keeping which key owns which digest.

    Args:
        digests: Digest per key (e.g. per dataset); `None` for a key with none.

    Returns:
        A `sha256:` hex digest over the sorted mapping.
    """
    import orjson

    payload = orjson.dumps(dict(digests), option=orjson.OPT_SORT_KEYS)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def shard_set(folder: Path) -> Dict[str, int]:
    """Return every visible file under *folder* with its byte size.

    Args:
        folder: A unit's output folder. Files and folders whose name starts
            with `.` (sentinels, progress files) are skipped.

    Returns:
        Mapping of POSIX path relative to *folder* to size in bytes, sorted by
        path; empty when *folder* does not exist.
    """
    if not folder.is_dir():
        return {}
    files = {}
    for path in folder.rglob("*"):
        rel = path.relative_to(folder)
        if path.is_file() and not any(part.startswith(".") for part in rel.parts):
            files[rel.as_posix()] = path.stat().st_size
    return dict(sorted(files.items()))


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of *path*'s bytes.

    Args:
        path: File to hash.

    Returns:
        Lower-case hex digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_lock(path: Path) -> Dict[str, Any]:
    """Read a lock file, or return `{}` when it does not exist.

    Args:
        path: Lock file path.

    Returns:
        The parsed lock entries.
    """
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_lock(path: Path, entries: Dict[str, Any], header: str = "") -> bool:
    """Write a lock file atomically, leaving it untouched when nothing changed.

    Args:
        path: Lock file path.
        entries: Lock entries; written in the given key order.
        header: Optional comment block written above the entries.

    Returns:
        True if the file was written, False if it already held these entries.
    """
    text = header + yaml.safe_dump(entries, sort_keys=False, allow_unicode=True)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return True


def config_hash(slice_: Dict[str, Any], extra: Optional[bytes] = None) -> str:
    """Compute a stable hash for a config slice.

    The slice is serialized as canonical JSON (sorted keys, no
    whitespace) before hashing. Optional `extra` bytes are appended to
    the hash input — used for hashing the *contents* of files
    referenced by the config (e.g. the stopword file), which are
    output-affecting but not part of the slice itself.

    Args:
        slice_: A JSON-serializable dict of output-affecting config.
        extra: Optional extra bytes to fold into the hash (e.g. the
            stopword file contents).

    Returns:
        Lower-case hex digest prefixed with `"sha256:"`.
    """
    h = hashlib.sha256()
    # default=str coerces non-JSON-native YAML values (datetime, date, etc.)
    # to a stable string form so the hash remains computable.
    h.update(json.dumps(slice_, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    if extra is not None:
        # NUL-separate the slice from `extra` so byte boundaries can't collide
        # between (slice ending in null bytes) and (slice + extra=b"...").
        h.update(b"\x00")
        h.update(extra)
    return "sha256:" + h.hexdigest()


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
        root: Corpus directory to scan (e.g. `pretrain/06_sentence_dedup`).
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
