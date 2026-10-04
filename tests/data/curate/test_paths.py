"""Tests for the curation path helpers (`slm4ie.data.curate.paths`)."""

from dataclasses import replace
from pathlib import Path

import pytest

from slm4ie.data.curate import paths as curate_paths
from slm4ie.data.curate.paths import (
    CuratePaths,
    bucket_log_scope,
    filter_stage_subset,
    read_bucket_index,
    record_bucket,
)


def _paths(tmp_path: Path) -> CuratePaths:
    """Build a CuratePaths anchored under *tmp_path* for structural tests."""
    return CuratePaths(
        input_folder=tmp_path / "datatrove",
        output_dir=tmp_path / "curated",
    )


def test_filter_stage_subset_links_requested_keys(tmp_path: Path) -> None:
    """filter_stage_subset mirrors only the requested keys via symlinks."""
    stage = tmp_path / "01_language"
    for key in ("a", "b"):
        (stage / key).mkdir(parents=True)
        (stage / key / "000.jsonl.gz").write_bytes(b"x")
    view = filter_stage_subset(stage, ["a"])
    try:
        assert (view / "a" / "000.jsonl.gz").is_symlink()
        assert not (view / "b").exists()
    finally:
        import shutil

        shutil.rmtree(view, ignore_errors=True)


def test_filter_stage_subset_missing_key_raises(tmp_path: Path) -> None:
    """filter_stage_subset raises when a key has no shards."""
    stage = tmp_path / "01_language"
    (stage / "a").mkdir(parents=True)
    (stage / "a" / "000.jsonl.gz").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        filter_stage_subset(stage, ["a", "missing"])


class TestStageSubsetFiltering:
    """`filter_stage_subset` mirrors the requested keys into a scratch view."""

    def test_filter_stage_subset_mirrors_multiple_datasets(self, tmp_path: Path) -> None:
        """`filter_stage_subset` builds symlinks for every requested key."""
        convert_dir = tmp_path / "00_convert"
        for key in ("kzb", "solar"):
            shard_dir = convert_dir / key
            shard_dir.mkdir(parents=True)
            (shard_dir / "00000.jsonl.gz").write_bytes(b"\x1f\x8b")

        holder = curate_paths.filter_stage_subset(convert_dir, ["kzb", "solar"])
        try:
            assert (holder / "kzb" / "00000.jsonl.gz").is_symlink()
            assert (holder / "solar" / "00000.jsonl.gz").is_symlink()
        finally:
            import shutil

            shutil.rmtree(holder, ignore_errors=True)

    def test_filter_stage_subset_lists_all_missing_keys(self, tmp_path: Path) -> None:
        """Missing shard folders are reported together in one error."""
        convert_dir = tmp_path / "00_convert"
        (convert_dir / "kzb").mkdir(parents=True)
        (convert_dir / "kzb" / "00000.jsonl.gz").write_bytes(b"\x1f\x8b")

        with pytest.raises(FileNotFoundError) as excinfo:
            curate_paths.filter_stage_subset(convert_dir, ["kzb", "missing1", "missing2"])
        msg = str(excinfo.value)
        assert "missing1" in msg
        assert "missing2" in msg
        assert "'kzb'" not in msg


def test_logs_dir_is_scoped_by_config_bucket(tmp_path: Path) -> None:
    """Scoped paths put each bucket's executor under its own folder; unscoped paths keep the stage folder."""
    paths = _paths(tmp_path)
    assert paths.logs_dir("quality") == paths.output_dir / "_logs" / "quality"
    scoped = replace(paths, log_scope="abc123")
    assert scoped.logs_dir("quality") == paths.output_dir / "_logs" / "quality" / "abc123"
    assert scoped.stage_dir("quality") == paths.stage_dir("quality")


def test_bucket_log_scope_names_config_and_dataset_set(tmp_path: Path) -> None:
    """The folder changes with the config hash or the dataset set and ignores key order."""
    same = bucket_log_scope("sha256:abc", ["d2", "d1"])
    assert same == bucket_log_scope("sha256:abc", ["d1", "d2"])
    assert same.startswith("abc-")
    assert same != bucket_log_scope("sha256:abc", ["d1"])
    assert same != bucket_log_scope("sha256:abd", ["d1", "d2"])


def test_bucket_index_lists_each_dataset_under_its_last_executor(tmp_path: Path) -> None:
    """A dataset moves to the executor that ran it last; an executor left with nothing is dropped."""
    (tmp_path / "f1").mkdir()
    record_bucket(tmp_path, "f1", "sha256:h1", ["d1", "d2"])
    record_bucket(tmp_path, "f2", "sha256:h2", ["d3"])
    record_bucket(tmp_path, "f3", "sha256:h1", ["d1"])
    assert read_bucket_index(tmp_path) == {
        "f1": {"config_hash": "sha256:h1", "datasets": ["d2"]},
        "f2": {"config_hash": "sha256:h2", "datasets": ["d3"]},
        "f3": {"config_hash": "sha256:h1", "datasets": ["d1"]},
    }
    record_bucket(tmp_path, "f4", "sha256:h3", ["d2"])
    assert "f1" not in read_bucket_index(tmp_path) and not (tmp_path / "f1").exists()
    assert read_bucket_index(tmp_path / "missing") == {}
