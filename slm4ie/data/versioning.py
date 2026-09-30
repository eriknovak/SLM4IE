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
"""

import gzip
import hashlib
from array import array
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import orjson
import yaml

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
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and VOLATILE_METADATA_KEYS & metadata.keys():
        record = {**record, "metadata": {k: v for k, v in metadata.items() if k not in VOLATILE_METADATA_KEYS}}
    canonical = orjson.dumps(record, option=orjson.OPT_SORT_KEYS)
    return int.from_bytes(hashlib.sha256(canonical).digest(), "big")


def _id_hash(value: Any) -> int:
    """Return a 64-bit hash of a document id."""
    return int.from_bytes(hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest(), "big")


def scan_documents(
    files: Iterable[Path], *, id_key: str = "id", digest: bool = True, raw_sha256: bool = False
) -> UnitScan:
    """Read JSONL documents once, collecting ids and optionally the digest.

    Args:
        files: Plain or gzipped (`.gz`) JSONL files, in any order.
        id_key: Top-level field holding each document's id.
        digest: Compute the document digest; skip it when only the ids are
            needed (e.g. for a stage's input side).
        raw_sha256: Also hash the raw bytes read (meaningful for one plain file).

    Returns:
        The `UnitScan` of every document across *files*.
    """
    raw = hashlib.sha256() if raw_sha256 else None
    total = 0
    records = 0
    ids = array("Q")
    for path in files:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as fh:
            for line in fh:
                if raw is not None:
                    raw.update(line)
                if not line.strip():
                    continue
                record = orjson.loads(line)
                records += 1
                ids.append(_id_hash(record.get(id_key)))
                if digest:
                    total += _document_hash(record)
    id_hashes = np.sort(np.frombuffer(ids, dtype=np.uint64)) if records else np.empty(0, dtype=np.uint64)
    return UnitScan(
        records=records,
        document_digest=_format_digest(total) if digest else None,
        id_hashes=id_hashes,
        raw_sha256=raw.hexdigest() if raw is not None else None,
    )


def merge_scans(scans: Sequence[UnitScan]) -> UnitScan:
    """Merge scans of disjoint file sets into the scan of all of them.

    Args:
        scans: Scans to merge; all must have been taken with the same `digest`
            setting.

    Returns:
        The combined `UnitScan`.
    """
    digests = [s.document_digest for s in scans]
    merged = None if any(d is None for d in digests) else merge_digests(d for d in digests if d is not None)
    ids = np.sort(np.concatenate([s.id_hashes for s in scans])) if scans else np.empty(0, dtype=np.uint64)
    return UnitScan(records=sum(s.records for s in scans), document_digest=merged, id_hashes=ids)


def scan_files(files: Sequence[Path], *, id_key: str = "id", digest: bool = True, workers: int = 1) -> UnitScan:
    """Scan *files* like `scan_documents`, one file per worker process.

    Args:
        files: Plain or gzipped JSONL files.
        id_key: Top-level field holding each document's id.
        digest: Compute the document digest.
        workers: Processes to spread the files over; 1 scans in-process.

    Returns:
        The `UnitScan` of every document across *files*.
    """
    if workers <= 1 or len(files) <= 1:
        return scan_documents(files, id_key=id_key, digest=digest)
    with ProcessPoolExecutor(max_workers=min(workers, len(files))) as pool:
        scans: List[UnitScan] = list(pool.map(_scan_one, files, [id_key] * len(files), [digest] * len(files)))
    return merge_scans(scans)


def _scan_one(path: Path, id_key: str, digest: bool) -> UnitScan:
    """Scan a single file; a picklable entry point for `scan_files`."""
    return scan_documents([path], id_key=id_key, digest=digest)


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


def write_lock(path: Path, entries: Dict[str, Any], header: str = "") -> None:
    """Write a lock file atomically.

    Args:
        path: Lock file path.
        entries: Lock entries; written in the given key order.
        header: Optional comment block written above the entries.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(header + yaml.safe_dump(entries, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tmp.replace(path)
