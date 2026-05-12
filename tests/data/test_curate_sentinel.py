"""Tests for the per-stage sentinel I/O + config-hash module."""

import json
from pathlib import Path

import pytest

from slm4ie.data.curate.sentinel import (
    Sentinel,
    cascade_invalidate,
    config_hash,
    read_sentinel,
    sentinel_is_current,
    write_sentinel,
)


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


def test_config_hash_includes_extra_payload(tmp_path: Path) -> None:
    """Optional extra payload (e.g. stopword file contents) affects the hash."""
    a = config_hash({"min_doc_words": 50})
    b = config_hash({"min_doc_words": 50}, extra=b"different bytes")
    assert a != b


def test_write_then_read_sentinel_roundtrip(tmp_path: Path) -> None:
    """Writing and reading a sentinel returns the same structured data."""
    folder = tmp_path / "02_quality"
    folder.mkdir()
    write_sentinel(
        folder,
        config_slice={"min_doc_words": 50},
        config_hash_value="sha256:abc",
        records_in=100,
        records_out=80,
    )
    sentinel = read_sentinel(folder)
    assert sentinel is not None
    assert sentinel.config_hash == "sha256:abc"
    assert sentinel.config_slice == {"min_doc_words": 50}
    assert sentinel.records_in == 100
    assert sentinel.records_out == 80
    assert sentinel.completed_at


def test_read_sentinel_returns_none_when_missing(tmp_path: Path) -> None:
    """Missing sentinel returns None instead of raising."""
    assert read_sentinel(tmp_path / "01_language") is None


def test_sentinel_is_current_matches_hash(tmp_path: Path) -> None:
    """sentinel_is_current returns True iff the recorded hash equals the new hash."""
    folder = tmp_path / "02_quality"
    folder.mkdir()
    write_sentinel(
        folder,
        config_slice={"min_doc_words": 50},
        config_hash_value="sha256:abc",
        records_in=1,
        records_out=1,
    )
    assert sentinel_is_current(folder, "sha256:abc") is True
    assert sentinel_is_current(folder, "sha256:different") is False


def test_sentinel_is_current_false_when_missing(tmp_path: Path) -> None:
    """A missing sentinel is never current."""
    assert sentinel_is_current(tmp_path / "missing", "sha256:abc") is False


def test_cascade_invalidate_removes_sentinels(tmp_path: Path) -> None:
    """cascade_invalidate removes sentinels for stage + every downstream stage."""
    for name in ("02_quality", "03_repetition", "04_1_dedup", "04_2_dedup", "05_statistics"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / ".complete").write_text("{}")
    (tmp_path / "01_language").mkdir()
    (tmp_path / "01_language" / ".complete").write_text("{}")

    removed = cascade_invalidate(tmp_path, "quality")
    assert "quality" in removed
    assert "stats" in removed
    # language was before quality — must NOT be invalidated.
    assert (tmp_path / "01_language" / ".complete").exists()
    # quality + downstream sentinels gone.
    for name in ("02_quality", "03_repetition", "04_1_dedup", "04_2_dedup", "05_statistics"):
        assert not (tmp_path / name / ".complete").exists()


def test_cascade_invalidate_handles_missing_sentinels_silently(tmp_path: Path) -> None:
    """It's fine to invalidate when sentinels don't exist; result lists requested stages."""
    removed = cascade_invalidate(tmp_path, "exact_dedup")
    assert removed == ("exact_dedup", "sentence_dedup", "stats")


def test_sentinel_filename_is_complete(tmp_path: Path) -> None:
    """The sentinel file is named .complete, matching to_datatrove.py convention."""
    folder = tmp_path / "02_quality"
    folder.mkdir()
    write_sentinel(
        folder,
        config_slice={},
        config_hash_value="sha256:x",
        records_in=0,
        records_out=0,
    )
    assert (folder / ".complete").exists()
