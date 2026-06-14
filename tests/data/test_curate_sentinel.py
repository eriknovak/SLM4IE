"""Tests for the per-stage sentinel I/O + config-hash module."""

from pathlib import Path

from slm4ie.data.curate.sentinel import (
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


def test_sentinel_input_fingerprint_roundtrips(tmp_path: Path) -> None:
    """A recorded input fingerprint survives a write/read roundtrip."""
    folder = tmp_path / "00_convert"
    folder.mkdir()
    write_sentinel(
        folder,
        config_slice={},
        config_hash_value="sha256:abc",
        records_in=1,
        records_out=1,
        input_fingerprint="news.jsonl=10:20",
    )
    sentinel = read_sentinel(folder)
    assert sentinel is not None
    assert sentinel.input_fingerprint == "news.jsonl=10:20"


def test_sentinel_input_fingerprint_defaults_to_none(tmp_path: Path) -> None:
    """Sentinels written without a fingerprint read back as None (legacy-safe)."""
    folder = tmp_path / "02_quality"
    folder.mkdir()
    write_sentinel(
        folder,
        config_slice={},
        config_hash_value="sha256:abc",
        records_in=1,
        records_out=1,
    )
    sentinel = read_sentinel(folder)
    assert sentinel is not None
    assert sentinel.input_fingerprint is None


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
    for name in ("03_quality", "04_repetition", "05_1_dedup", "05_2_dedup", "06_statistics"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / ".complete").write_text("{}")
    for name in ("00_convert", "01_language", "02_spam"):
        (tmp_path / name).mkdir()
        (tmp_path / name / ".complete").write_text("{}")

    removed = cascade_invalidate(tmp_path, "quality")
    assert "quality" in removed
    assert "stats" in removed
    # convert, language, and spam were before quality — must NOT be invalidated.
    assert (tmp_path / "00_convert" / ".complete").exists()
    assert (tmp_path / "01_language" / ".complete").exists()
    assert (tmp_path / "02_spam" / ".complete").exists()
    # quality + downstream sentinels gone.
    for name in ("03_quality", "04_repetition", "05_1_dedup", "05_2_dedup", "06_statistics"):
        assert not (tmp_path / name / ".complete").exists()


def test_cascade_invalidate_handles_missing_sentinels_silently(tmp_path: Path) -> None:
    """It's fine to invalidate when sentinels don't exist; result lists requested stages."""
    removed = cascade_invalidate(tmp_path, "exact_dedup")
    assert removed == ("exact_dedup", "sentence_dedup", "stats")


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
        _json.dumps({
            "completed_at": "2026-05-12T00:00:00Z",
            "config_hash": "sha256:abc",
            "config_slice": {},
            "records_in": "not_a_number",
            "records_out": 0,
        })
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


def test_dataset_sentinel_roundtrip(tmp_path: Path) -> None:
    """A per-dataset sentinel is current only when its hash matches."""
    from slm4ie.data.curate.sentinel import (
        dataset_sentinel_is_current,
        dataset_sentinel_path,
        write_dataset_sentinel,
    )

    stage_dir = tmp_path / "02_quality"
    write_dataset_sentinel(
        stage_dir,
        "gigafida",
        config_slice={"min_doc_words": 20},
        config_hash_value="abc123",
        records_in=10,
        records_out=8,
    )
    assert dataset_sentinel_path(stage_dir, "gigafida").exists()
    assert dataset_sentinel_is_current(stage_dir, "gigafida", "abc123") is True
    assert dataset_sentinel_is_current(stage_dir, "gigafida", "different") is False
    # A different dataset under the same stage is independent.
    assert dataset_sentinel_is_current(stage_dir, "kas", "abc123") is False


def test_invalidate_dataset_sentinels(tmp_path: Path) -> None:
    """Invalidating removes only the named datasets' sentinels."""
    from slm4ie.data.curate.sentinel import (
        dataset_sentinel_is_current,
        invalidate_dataset_sentinels,
        write_dataset_sentinel,
    )

    stage_dir = tmp_path / "01_language"
    for key in ("a", "b"):
        write_dataset_sentinel(
            stage_dir, key, config_slice={}, config_hash_value="h",
            records_in=1, records_out=1,
        )
    invalidate_dataset_sentinels(stage_dir, ["a"])
    assert dataset_sentinel_is_current(stage_dir, "a", "h") is False
    assert dataset_sentinel_is_current(stage_dir, "b", "h") is True


def test_cascade_invalidate_scoped_mixes_per_dataset_and_corpus(tmp_path: Path) -> None:
    """Scoped stages drop per-dataset sentinels; corpus stages drop stage sentinels."""
    from slm4ie.data.curate.sentinel import (
        cascade_invalidate_scoped,
        dataset_sentinel_is_current,
        sentinel_is_current,
        write_dataset_sentinel,
        write_sentinel,
    )

    out = tmp_path
    # Scoped stage quality has per-dataset sentinels for a, b.
    q = out / "03_quality"
    for key in ("a", "b"):
        write_dataset_sentinel(q, key, config_slice={}, config_hash_value="h",
                               records_in=1, records_out=1)
    # Corpus stage exact_dedup has a stage-level sentinel.
    d = out / "05_1_dedup"
    write_sentinel(d, config_slice={}, config_hash_value="h",
                   records_in=1, records_out=1)

    # Invalidate from quality, only for dataset 'a'.
    cascade_invalidate_scoped(out, "quality", ["a"])

    # quality/a dropped, quality/b kept (only requested keys invalidated).
    assert dataset_sentinel_is_current(q, "a", "h") is False
    assert dataset_sentinel_is_current(q, "b", "h") is True
    # Downstream corpus stage dropped wholesale (roster/inputs changed).
    assert sentinel_is_current(d, "h") is False
