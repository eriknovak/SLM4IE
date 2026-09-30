"""Tests for the document digest, integrity check, shard set and lock file."""

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("orjson")
pytest.importorskip("numpy")

from slm4ie.data.versioning import (  # noqa: E402
    EMPTY_DIGEST,
    check_integrity,
    combine_named_digests,
    file_sha256,
    merge_digests,
    read_lock,
    scan_documents,
    scan_files,
    shard_files,
    shard_set,
    write_lock,
)


def _doc(i: int, **metadata: Any) -> Dict[str, Any]:
    """Return a datatrove-shaped document with id `d<i>`."""
    return {"text": f"besedilo {i}", "id": f"d{i}", "metadata": {"dataset": "x", **metadata}}


def _write_shard(path: Path, docs: List[Dict[str, Any]]) -> None:
    """Write *docs* as one gzipped JSONL shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")


def test_digest_invariant_to_shard_split_and_order(tmp_path: Path) -> None:
    """The same documents split across shards differently digest the same."""
    docs = [_doc(i) for i in range(10)]
    _write_shard(tmp_path / "a" / "00000.jsonl.gz", docs)
    _write_shard(tmp_path / "b" / "00003.jsonl.gz", list(reversed(docs[:4])))
    _write_shard(tmp_path / "b" / "00001.jsonl.gz", docs[4:])
    a = scan_documents(sorted((tmp_path / "a").glob("*.jsonl.gz")))
    b = scan_documents(sorted((tmp_path / "b").glob("*.jsonl.gz")))
    assert a.document_digest == b.document_digest
    assert a.records == b.records == 10


def test_digest_ignores_volatile_file_path(tmp_path: Path) -> None:
    """The reader-stamped `file_path` metadata does not enter the digest."""
    _write_shard(tmp_path / "a.jsonl.gz", [_doc(1, file_path="/tmp/x/1.jsonl.gz")])
    _write_shard(tmp_path / "b.jsonl.gz", [_doc(1, file_path="/tmp/y/9.jsonl.gz")])
    assert (
        scan_documents([tmp_path / "a.jsonl.gz"]).document_digest
        == scan_documents([tmp_path / "b.jsonl.gz"]).document_digest
    )


def test_digest_changes_with_content_and_multiplicity(tmp_path: Path) -> None:
    """Changed text or a duplicated document changes the digest."""
    _write_shard(tmp_path / "a.jsonl.gz", [_doc(1), _doc(2)])
    _write_shard(tmp_path / "b.jsonl.gz", [_doc(1), {**_doc(2), "text": "drugo"}])
    _write_shard(tmp_path / "c.jsonl.gz", [_doc(1), _doc(2), _doc(2)])
    digests = {scan_documents([tmp_path / f"{n}.jsonl.gz"]).document_digest for n in "abc"}
    assert len(digests) == 3


def test_empty_scan_has_empty_digest() -> None:
    """No documents digest to the empty digest."""
    scan = scan_documents([])
    assert scan.records == 0
    assert scan.document_digest == EMPTY_DIGEST


def test_merge_digests_equals_digest_of_union(tmp_path: Path) -> None:
    """Merging per-part digests equals digesting all parts together."""
    _write_shard(tmp_path / "a.jsonl.gz", [_doc(1), _doc(2)])
    _write_shard(tmp_path / "b.jsonl.gz", [_doc(3)])
    a = scan_documents([tmp_path / "a.jsonl.gz"]).document_digest
    b = scan_documents([tmp_path / "b.jsonl.gz"]).document_digest
    union = scan_documents([tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"]).document_digest
    assert merge_digests([a, b]) == union


def test_combine_named_digests_attributes_keys() -> None:
    """Swapping which key owns which digest changes the combination."""
    assert combine_named_digests({"a": "x", "b": "y"}) != combine_named_digests({"a": "y", "b": "x"})
    assert combine_named_digests({"a": "x", "b": "y"}) == combine_named_digests({"b": "y", "a": "x"})


def test_integrity_passes_for_filtered_subset(tmp_path: Path) -> None:
    """An output that keeps a subset of the input passes, repeated input ids included."""
    _write_shard(tmp_path / "in.jsonl.gz", [_doc(1), _doc(1), _doc(2), _doc(3)])
    _write_shard(tmp_path / "out.jsonl.gz", [_doc(1), _doc(1), _doc(3)])
    inp = scan_documents([tmp_path / "in.jsonl.gz"], digest=False)
    out = scan_documents([tmp_path / "out.jsonl.gz"])
    assert check_integrity(inp, out) is None


def test_integrity_flags_duplicated_id(tmp_path: Path) -> None:
    """An id written more often than it was read fails the check."""
    _write_shard(tmp_path / "in.jsonl.gz", [_doc(1), _doc(2), _doc(3)])
    _write_shard(tmp_path / "out.jsonl.gz", [_doc(1), _doc(2)])
    _write_shard(tmp_path / "stale.jsonl.gz", [_doc(1)])
    inp = scan_documents([tmp_path / "in.jsonl.gz"], digest=False)
    out = scan_documents([tmp_path / "out.jsonl.gz", tmp_path / "stale.jsonl.gz"])
    error = check_integrity(inp, out)
    assert error is not None and "1 id" in error


def test_integrity_flags_unknown_id_and_growth(tmp_path: Path) -> None:
    """An id absent from the input, or more records out than in, fails."""
    _write_shard(tmp_path / "in.jsonl.gz", [_doc(1)])
    _write_shard(tmp_path / "out.jsonl.gz", [_doc(1), _doc(2)])
    inp = scan_documents([tmp_path / "in.jsonl.gz"], digest=False)
    out = scan_documents([tmp_path / "out.jsonl.gz"])
    error = check_integrity(inp, out)
    assert error is not None and "records_out" in error


def test_scan_reads_plain_jsonl_with_custom_id_key(tmp_path: Path) -> None:
    """Extraction-tier JSONL is scanned by its own id field."""
    path = tmp_path / "k.jsonl"
    path.write_text("\n".join(json.dumps({"uid": f"k:{i}", "text": "t"}) for i in range(3)) + "\n")
    scan = scan_documents([path], id_key="uid", digest=False)
    assert scan.records == 3
    assert len(scan.id_hashes) == 3


def test_shard_set_records_size_not_mtime(tmp_path: Path) -> None:
    """The shard set lists visible files by relative path and size only."""
    _write_shard(tmp_path / "k" / "00000.jsonl.gz", [_doc(1)])
    (tmp_path / "k" / ".complete").write_text("{}")
    shards = shard_set(tmp_path)
    assert list(shards) == ["k/00000.jsonl.gz"]
    assert shards["k/00000.jsonl.gz"] == (tmp_path / "k" / "00000.jsonl.gz").stat().st_size


def test_file_sha256_hashes_contents(tmp_path: Path) -> None:
    """Equal bytes hash equal regardless of file name."""
    (tmp_path / "a").write_bytes(b"abc")
    (tmp_path / "b").write_bytes(b"abc")
    assert file_sha256(tmp_path / "a") == file_sha256(tmp_path / "b")


def test_lock_roundtrip(tmp_path: Path) -> None:
    """A written lock file reads back unchanged."""
    entries = {"convert": {"kzb": {"config_hash": "sha256:a", "stage_version": 1}}}
    path = tmp_path / "curate.lock.yaml"
    write_lock(path, entries)
    assert read_lock(path) == entries
    assert read_lock(tmp_path / "missing.yaml") == {}


def test_parallel_scan_matches_serial(tmp_path: Path) -> None:
    """Scanning shards across processes gives the serial scan's result."""
    for n in range(3):
        _write_shard(tmp_path / f"{n:05d}.jsonl.gz", [_doc(n * 10 + i) for i in range(4)])
    files = shard_files(tmp_path)
    serial = scan_files(files)
    parallel = scan_files(files, workers=3)
    assert parallel.document_digest == serial.document_digest
    assert parallel.records == serial.records == 12
    assert (parallel.id_hashes == serial.id_hashes).all()


def test_scan_can_hash_raw_bytes_in_the_same_pass(tmp_path: Path) -> None:
    """The raw-bytes hash of a scanned file equals its file hash, blank lines included."""
    path = tmp_path / "k.jsonl"
    path.write_text(json.dumps({"uid": "k:1"}) + "\n\n" + json.dumps({"uid": "k:2"}))
    scan = scan_documents([path], id_key="uid", digest=False, raw_sha256=True)
    assert scan.raw_sha256 == file_sha256(path)
    assert scan.records == 2


@pytest.mark.parametrize("workers", [1, 3])
def test_scan_units_matches_per_file_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int) -> None:
    """Chunked sequential reading gives the same scans as reading each file whole."""
    import slm4ie.data.versioning as versioning

    monkeypatch.setattr(versioning, "_READ_BYTES", 97)
    plain = tmp_path / "k.jsonl"
    plain.write_text("".join(json.dumps({"uid": f"k:{i}", "text": "x" * i}) + "\n" for i in range(40)))
    for n in range(2):
        _write_shard(tmp_path / "s" / f"{n:05d}.jsonl.gz", [_doc(n * 10 + i) for i in range(5)])
    requests = {
        "plain": versioning.ScanRequest([plain], id_key="uid", digest=False, raw_sha256=True),
        "shards": versioning.ScanRequest(shard_files(tmp_path / "s")),
        "empty": versioning.ScanRequest([]),
    }
    seen = []
    scans = versioning.scan_units(requests, workers, progress=lambda path, size: seen.append(path.name))
    assert scans["plain"].records == 40
    assert scans["plain"].raw_sha256 == file_sha256(plain)
    assert scans["plain"].document_digest is None
    expected = scan_documents(shard_files(tmp_path / "s"))
    assert scans["shards"].document_digest == expected.document_digest
    assert (scans["shards"].id_hashes == expected.id_hashes).all()
    assert scans["empty"].records == 0 and scans["empty"].document_digest == EMPTY_DIGEST
    assert seen == ["k.jsonl", "00000.jsonl.gz", "00001.jsonl.gz"]
