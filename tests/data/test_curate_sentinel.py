"""Tests for the per-stage sentinel I/O + config-hash module."""

import os
from pathlib import Path

import pytest

from slm4ie.data.versioning import config_hash
from slm4ie.data.curate.lineage import (
    CONFIG_CHANGED,
    INPUT_CHANGED,
    INTEGRITY_FAILED,
    LEGACY,
    NOT_BUILT,
    OUTPUT_CHANGED,
    STAGE_VERSION_CHANGED,
    read_sentinel,
    stale_reason,
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


def test_config_hash_includes_extra_payload() -> None:
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


def test_sentinel_lineage_roundtrips(tmp_path: Path) -> None:
    """Lineage fields and the shard set survive a write/read roundtrip."""
    folder = tmp_path / "00_convert" / "news"
    folder.mkdir(parents=True)
    (folder / "00000.jsonl.gz").write_bytes(b"abc")
    write_sentinel(
        folder,
        config_slice={},
        config_hash_value="sha256:abc",
        records_in=1,
        records_out=1,
        stage_version="v2",
        input_digest="sha256:in",
        document_digest="sum256:doc",
        input_files={"news.jsonl": {"size": 3, "sha256": "x", "mtime_ns": 1}},
        info={"git_commit": "c"},
    )
    sentinel = read_sentinel(folder)
    assert sentinel is not None
    assert sentinel.stage_version == "v2"
    assert sentinel.input_digest == "sha256:in"
    assert sentinel.document_digest == "sum256:doc"
    assert sentinel.shards == {"00000.jsonl.gz": 3}
    assert sentinel.input_files == {"news.jsonl": {"size": 3, "sha256": "x", "mtime_ns": 1}}
    assert sentinel.info == {"git_commit": "c"}
    assert not sentinel.is_legacy


def test_sentinel_without_version_is_legacy(tmp_path: Path) -> None:
    """A sentinel written before lineage tracking reads back as legacy."""
    folder = tmp_path / "02_quality"
    folder.mkdir()
    (folder / ".complete").write_text('{"config_hash": "sha256:abc", "records_in": 1, "records_out": 1}')
    sentinel = read_sentinel(folder)
    assert sentinel is not None and sentinel.is_legacy


def _current_unit(tmp_path: Path) -> Path:
    """Write a unit with one shard and a full-lineage sentinel; return its folder."""
    folder = tmp_path / "03_quality" / "a"
    folder.mkdir(parents=True)
    (folder / "00000.jsonl.gz").write_bytes(b"abc")
    write_sentinel(
        folder,
        config_slice={},
        config_hash_value="h",
        records_in=1,
        records_out=1,
        stage_version="v1",
        input_digest="in",
        document_digest="doc",
    )
    return folder


def _reason(folder: Path, **overrides: object) -> object:
    """Return `stale_reason` for *folder* with the matching lineage, overridden."""
    kwargs = {"expected_hash": "h", "stage_version": "v1", "input_digest": "in", **overrides}
    return stale_reason(read_sentinel(folder), folder, **kwargs)  # type: ignore[arg-type]


def test_stale_reason_current_when_lineage_matches(tmp_path: Path) -> None:
    """A unit whose lineage and shards match is current."""
    assert _reason(_current_unit(tmp_path)) is None


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"expected_hash": "h2"}, CONFIG_CHANGED),
        ({"stage_version": "v2"}, STAGE_VERSION_CHANGED),
        ({"input_digest": "in2"}, INPUT_CHANGED),
    ],
)
def test_stale_reason_names_the_changed_field(tmp_path: Path, override: dict, reason: str) -> None:
    """Each lineage mismatch is reported under its own reason."""
    assert _reason(_current_unit(tmp_path), **override) == reason


def test_stale_reason_missing_sentinel(tmp_path: Path) -> None:
    """A unit without a sentinel has not been built."""
    assert _reason(tmp_path / "nothing") == NOT_BUILT


def test_stale_reason_ignores_mtime_but_not_size(tmp_path: Path) -> None:
    """Touching a shard keeps the unit current; changing its size or adding one does not."""
    folder = _current_unit(tmp_path)
    shard = folder / "00000.jsonl.gz"
    os.utime(shard, ns=(1, 1))
    assert _reason(folder) is None
    (folder / "00001.jsonl.gz").write_bytes(b"stale")
    assert _reason(folder) == OUTPUT_CHANGED
    (folder / "00001.jsonl.gz").unlink()
    shard.write_bytes(b"abcd")
    assert _reason(folder) == OUTPUT_CHANGED


def test_stale_reason_legacy_and_integrity(tmp_path: Path) -> None:
    """Legacy sentinels and recorded integrity failures are stale."""
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / ".complete").write_text('{"config_hash": "h"}')
    assert _reason(legacy) == LEGACY
    assert _reason(legacy, expected_hash="h2") == CONFIG_CHANGED
    failed = tmp_path / "failed"
    write_sentinel(
        failed,
        config_slice={},
        config_hash_value="h",
        records_in=1,
        records_out=2,
        stage_version="v1",
        input_digest="in",
        integrity_error="boom",
    )
    assert str(_reason(failed)).startswith(INTEGRITY_FAILED)


def test_sentinel_filename_is_complete(tmp_path: Path) -> None:
    """The sentinel file is named .complete, matching the curate stage convention."""
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


def test_read_sentinel_returns_none_on_malformed_json(tmp_path: Path) -> None:
    """A corrupt sentinel file returns None instead of raising."""
    folder = tmp_path / "02_quality"
    folder.mkdir()
    (folder / ".complete").write_text("not json at all {[")
    assert read_sentinel(folder) is None


def test_read_sentinel_returns_none_on_bad_field_types(tmp_path: Path) -> None:
    """A sentinel with non-coercible numeric fields returns None."""
    import json as _json

    folder = tmp_path / "02_quality"
    folder.mkdir()
    (folder / ".complete").write_text(
        _json.dumps(
            {
                "completed_at": "2026-05-12T00:00:00Z",
                "config_hash": "sha256:abc",
                "config_slice": {},
                "records_in": "not_a_number",
                "records_out": 0,
            }
        )
    )
    assert read_sentinel(folder) is None


def test_write_sentinel_does_not_leave_tmp_artifact(tmp_path: Path) -> None:
    """The atomic write replaces the final path and leaves no .tmp sibling."""
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
    assert not (folder / ".complete.tmp").exists()


def test_invalidate_dataset_sentinels(tmp_path: Path) -> None:
    """Invalidating removes only the named datasets' sentinels."""
    from slm4ie.data.curate.lineage import (
        dataset_sentinel_path,
        invalidate_dataset_sentinels,
        write_dataset_sentinel,
    )

    stage_dir = tmp_path / "01_language"
    for key in ("a", "b"):
        write_dataset_sentinel(
            stage_dir,
            key,
            config_slice={},
            config_hash_value="h",
            records_in=1,
            records_out=1,
        )
    invalidate_dataset_sentinels(stage_dir, ["a"])
    assert not dataset_sentinel_path(stage_dir, "a").exists()
    assert dataset_sentinel_path(stage_dir, "b").exists()
