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


def test_stage_extra_folds_roster_only_for_corpus_stages() -> None:
    """Scoped stages exclude the roster; corpus stages include it."""
    from scripts.data.to_pretrain import _stage_extra

    roster = b'["a","b"]'
    sw = b"stopwords"
    # Scoped: roster must NOT appear.
    assert _stage_extra("language", sw, roster) == b""
    assert _stage_extra("quality", sw, roster) == sw  # stopwords only, no roster
    # Corpus: roster present.
    assert roster in _stage_extra("exact_dedup", sw, roster)
    assert roster in _stage_extra("stats", sw, roster)
    assert sw in _stage_extra("stats", sw, roster)  # stats also folds stopwords


def test_corpus_stage_with_positional_keys_errors() -> None:
    """--stage exact_dedup with positional keys is rejected."""
    import pytest

    from scripts.data.to_pretrain import parse_args

    with pytest.raises(SystemExit):
        parse_args(["gigafida", "--stage", "exact_dedup"])


def test_corpus_stage_with_all_is_ok() -> None:
    """--all --stage stats parses fine."""
    from scripts.data.to_pretrain import parse_args

    args = parse_args(["--all", "--stage", "stats"])
    assert args.all is True
    assert args.stage == "stats"


def test_scoped_stage_with_positional_keys_ok() -> None:
    """--stage quality with positional keys is allowed (scoped stage)."""
    from scripts.data.to_pretrain import parse_args

    args = parse_args(["gigafida", "--stage", "quality"])
    assert args.datasets == ["gigafida"]
    assert args.stage == "quality"
