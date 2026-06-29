"""Tests for slm4ie/data/curate/manifest.py corpus digests."""

import gzip
from pathlib import Path

import pytest

from slm4ie.data.curate import manifest


def _write_shard(path: Path, rows: int) -> None:
    """Write a gzipped JSONL shard with `rows` trivial records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f'{{"i": {i}}}\n' for i in range(rows)).encode("utf-8")
    path.write_bytes(gzip.compress(payload))


class TestShardManifest:
    """Tests for building the per-shard manifest."""

    def test_sorted_relative_posix_paths(self, tmp_path: Path):
        """Shards are listed by sorted root-relative POSIX path."""
        _write_shard(tmp_path / "b" / "000.jsonl.gz", 1)
        _write_shard(tmp_path / "a" / "000.jsonl.gz", 1)
        rels = [rel for rel, _, _ in manifest.shard_manifest(tmp_path)]
        assert rels == ["a/000.jsonl.gz", "b/000.jsonl.gz"]

    def test_rows_unset_by_default(self, tmp_path: Path):
        """Row counts are left unset unless explicitly requested."""
        _write_shard(tmp_path / "000.jsonl.gz", 3)
        (_rel, size, rows) = manifest.shard_manifest(tmp_path)[0]
        assert size > 0
        assert rows == manifest.ROWS_NOT_COUNTED

    def test_rows_counted_when_requested(self, tmp_path: Path):
        """with_rows decompresses each shard and counts its records."""
        _write_shard(tmp_path / "000.jsonl.gz", 3)
        (_, _, rows) = manifest.shard_manifest(tmp_path, with_rows=True)[0]
        assert rows == 3

    def test_missing_root_raises(self, tmp_path: Path):
        """A non-existent root is an explicit error."""
        with pytest.raises(FileNotFoundError):
            manifest.shard_manifest(tmp_path / "nope")


class TestCorpusDigest:
    """Tests for the content digest over a corpus directory."""

    def test_prefixed_hex(self, tmp_path: Path):
        """The digest carries the project's sha256 prefix."""
        _write_shard(tmp_path / "000.jsonl.gz", 1)
        assert manifest.corpus_digest(tmp_path).startswith("sha256:")

    def test_stable_across_calls(self, tmp_path: Path):
        """Identical shards yield an identical digest on repeat calls."""
        _write_shard(tmp_path / "000.jsonl.gz", 2)
        assert manifest.corpus_digest(tmp_path) == manifest.corpus_digest(tmp_path)

    def test_changes_when_a_shard_changes(self, tmp_path: Path):
        """Rewriting a shard with different content changes the digest."""
        shard = tmp_path / "000.jsonl.gz"
        _write_shard(shard, 2)
        before = manifest.corpus_digest(tmp_path)
        _write_shard(shard, 5)
        assert manifest.corpus_digest(tmp_path) != before

    def test_changes_when_a_shard_is_added(self, tmp_path: Path):
        """Adding a shard changes the digest."""
        _write_shard(tmp_path / "000.jsonl.gz", 1)
        before = manifest.corpus_digest(tmp_path)
        _write_shard(tmp_path / "001.jsonl.gz", 1)
        assert manifest.corpus_digest(tmp_path) != before

    def test_empty_root_is_well_defined(self, tmp_path: Path):
        """An existing root with no shards still digests deterministically."""
        digest = manifest.corpus_digest(tmp_path)
        assert digest.startswith("sha256:")
        assert digest == manifest.corpus_digest(tmp_path)

    def test_with_rows_distinguishes_same_size_different_rows(self, tmp_path: Path):
        """with_rows separates builds a size-only digest could collide."""
        shard = tmp_path / "000.jsonl.gz"
        _write_shard(shard, 2)
        size_only = manifest.corpus_digest(tmp_path)
        with_rows = manifest.corpus_digest(tmp_path, with_rows=True)
        assert size_only != with_rows
