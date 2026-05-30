"""Tests for scripts/data/to_pretrain.py helpers."""

from pathlib import Path

import pytest

from scripts.data.to_pretrain import _filter_stage_subset


def test_filter_stage_subset_links_requested_keys(tmp_path: Path) -> None:
    """_filter_stage_subset mirrors only the requested keys via symlinks."""
    stage = tmp_path / "01_language"
    for key in ("a", "b"):
        (stage / key).mkdir(parents=True)
        (stage / key / "000.jsonl.gz").write_bytes(b"x")
    view = _filter_stage_subset(stage, ["a"])
    try:
        assert (view / "a" / "000.jsonl.gz").is_symlink()
        assert not (view / "b").exists()
    finally:
        import shutil

        shutil.rmtree(view, ignore_errors=True)


def test_filter_stage_subset_missing_key_raises(tmp_path: Path) -> None:
    """_filter_stage_subset raises when a key has no shards."""
    stage = tmp_path / "01_language"
    (stage / "a").mkdir(parents=True)
    (stage / "a" / "000.jsonl.gz").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        _filter_stage_subset(stage, ["a", "missing"])
