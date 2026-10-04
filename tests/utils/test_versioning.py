"""Tests for slm4ie/utils/versioning.py: digests, integrity, shard sets, lock file, config hash."""

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("orjson")
pytest.importorskip("numpy")

from slm4ie.utils import versioning as manifest  # noqa: E402
from slm4ie.utils.versioning import (  # noqa: E402
    EMPTY_DIGEST,
    check_integrity,
    combine_named_digests,
    config_hash,
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
    import slm4ie.utils.versioning as versioning

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


def test_scan_units_keeps_lines_longer_than_a_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A line spanning several read blocks is scanned once, whole."""
    import slm4ie.utils.versioning as versioning

    monkeypatch.setattr(versioning, "_READ_BYTES", 16)
    path = tmp_path / "k.jsonl"
    path.write_text(json.dumps({"uid": "k:long", "text": "y" * 200}) + "\n" + json.dumps({"uid": "k:short"}))
    scan = versioning.scan_units({"k": versioning.ScanRequest([path], id_key="uid", raw_sha256=True)}, 2)["k"]
    assert scan.records == 2
    assert scan.raw_sha256 == file_sha256(path)


def _write_rows_shard(path: Path, rows: int) -> None:
    """Write a gzipped JSONL shard with `rows` trivial records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f'{{"i": {i}}}\n' for i in range(rows)).encode("utf-8")
    path.write_bytes(gzip.compress(payload))


class TestShardManifest:
    """Tests for building the per-shard manifest."""

    def test_sorted_relative_posix_paths(self, tmp_path: Path):
        """Shards are listed by sorted root-relative POSIX path."""
        _write_rows_shard(tmp_path / "b" / "000.jsonl.gz", 1)
        _write_rows_shard(tmp_path / "a" / "000.jsonl.gz", 1)
        rels = [rel for rel, _, _ in manifest.shard_listing(tmp_path)]
        assert rels == ["a/000.jsonl.gz", "b/000.jsonl.gz"]

    def test_rows_unset_by_default(self, tmp_path: Path):
        """Row counts are left unset unless explicitly requested."""
        _write_rows_shard(tmp_path / "000.jsonl.gz", 3)
        (_rel, size, rows) = manifest.shard_listing(tmp_path)[0]
        assert size > 0
        assert rows == manifest.ROWS_NOT_COUNTED

    def test_rows_counted_when_requested(self, tmp_path: Path):
        """with_rows decompresses each shard and counts its records."""
        _write_rows_shard(tmp_path / "000.jsonl.gz", 3)
        (_, _, rows) = manifest.shard_listing(tmp_path, with_rows=True)[0]
        assert rows == 3

    def test_missing_root_raises(self, tmp_path: Path):
        """A non-existent root is an explicit error."""
        with pytest.raises(FileNotFoundError):
            manifest.shard_listing(tmp_path / "nope")


class TestCorpusDigest:
    """Tests for the content digest over a corpus directory."""

    def test_prefixed_hex(self, tmp_path: Path):
        """The digest carries the project's sha256 prefix."""
        _write_rows_shard(tmp_path / "000.jsonl.gz", 1)
        assert manifest.corpus_digest(tmp_path).startswith("sha256:")

    def test_stable_across_calls(self, tmp_path: Path):
        """Identical shards yield an identical digest on repeat calls."""
        _write_rows_shard(tmp_path / "000.jsonl.gz", 2)
        assert manifest.corpus_digest(tmp_path) == manifest.corpus_digest(tmp_path)

    def test_changes_when_a_shard_changes(self, tmp_path: Path):
        """Rewriting a shard with different content changes the digest."""
        shard = tmp_path / "000.jsonl.gz"
        _write_rows_shard(shard, 2)
        before = manifest.corpus_digest(tmp_path)
        _write_rows_shard(shard, 5)
        assert manifest.corpus_digest(tmp_path) != before

    def test_changes_when_a_shard_is_added(self, tmp_path: Path):
        """Adding a shard changes the digest."""
        _write_rows_shard(tmp_path / "000.jsonl.gz", 1)
        before = manifest.corpus_digest(tmp_path)
        _write_rows_shard(tmp_path / "001.jsonl.gz", 1)
        assert manifest.corpus_digest(tmp_path) != before

    def test_empty_root_is_well_defined(self, tmp_path: Path):
        """An existing root with no shards still digests deterministically."""
        digest = manifest.corpus_digest(tmp_path)
        assert digest.startswith("sha256:")
        assert digest == manifest.corpus_digest(tmp_path)

    def test_with_rows_distinguishes_same_size_different_rows(self, tmp_path: Path):
        """with_rows separates builds a size-only digest could collide."""
        shard = tmp_path / "000.jsonl.gz"
        _write_rows_shard(shard, 2)
        size_only = manifest.corpus_digest(tmp_path)
        with_rows = manifest.corpus_digest(tmp_path, with_rows=True)
        assert size_only != with_rows


def test_config_hash_is_deterministic() -> None:
    """The hash is stable across calls with identical input."""
    cfg = {"min_doc_words": 50, "max_doc_words": 100000}
    assert config_hash(cfg) == config_hash(cfg)


def test_config_hash_differs_on_value_change() -> None:
    """Changing any value changes the hash."""
    a = {"min_doc_words": 50}
    b = {"min_doc_words": 100}
    assert config_hash(a) != config_hash(b)


def test_config_hash_ignores_key_order() -> None:
    """Two dicts with the same keys/values in different insertion order hash equal."""
    a = {"a": 1, "b": 2}
    b = {"b": 2, "a": 1}
    assert config_hash(a) == config_hash(b)


def test_config_hash_includes_extra_payload() -> None:
    """Optional extra payload (e.g. stopword file contents) affects the hash."""
    a = config_hash({"min_doc_words": 50})
    b = config_hash({"min_doc_words": 50}, extra=b"different bytes")
    assert a != b


def test_config_hash_handles_yaml_datetime_values(tmp_path: Path) -> None:
    """config_hash does not raise on YAML-style non-JSON values like datetime."""
    from datetime import datetime, timezone

    a = config_hash({"created_at": datetime(2024, 1, 1, tzinfo=timezone.utc)})
    b = config_hash({"created_at": datetime(2024, 1, 1, tzinfo=timezone.utc)})
    assert a == b


def test_config_hash_handles_non_ascii_values() -> None:
    """Non-ASCII characters in the slice influence the hash predictably."""
    a = config_hash({"stopwords_path": "stopwords_sl.txt"})
    b = config_hash({"stopwords_path": "stopwords_žirovski.txt"})
    assert a != b
