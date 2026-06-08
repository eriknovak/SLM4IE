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
    sp = b"spamlex"
    # Scoped: roster must NOT appear.
    assert _stage_extra("language", sw, sp, roster) == b""
    assert _stage_extra("quality", sw, sp, roster) == sw  # stopwords only, no roster
    # Spam folds its lexicon/domain bytes (and never the roster — it is scoped).
    assert _stage_extra("spam", sw, sp, roster) == sp
    # Corpus: roster present.
    assert roster in _stage_extra("exact_dedup", sw, sp, roster)
    assert roster in _stage_extra("stats", sw, sp, roster)
    assert sw in _stage_extra("stats", sw, sp, roster)  # stats also folds stopwords


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


def test_resolve_requested_stages() -> None:
    """Subset 'all' = scoped stages; --all 'all' = every stage."""
    from scripts.data.to_pretrain import _resolve_requested_stages
    from slm4ie.data.curate.stages import SCOPED_STAGES, STAGE_NAMES

    assert _resolve_requested_stages(stage="all", run_all=False) == SCOPED_STAGES
    assert _resolve_requested_stages(stage="all", run_all=True) == STAGE_NAMES
    assert _resolve_requested_stages(stage="quality", run_all=False) == ("quality",)
    assert _resolve_requested_stages(stage="exact_dedup", run_all=True) == ("exact_dedup",)


def test_force_subset_stage_drops_only_requested_keys(tmp_path: Path) -> None:
    """--force gigafida --stage quality drops gigafida's quality sentinel, keeps others."""
    from slm4ie.data.curate.sentinel import (
        dataset_sentinel_is_current,
        write_dataset_sentinel,
    )
    from scripts.data.to_pretrain import _apply_force

    out = tmp_path / "pretrain"
    q = out / "03_quality"
    for key in ("gigafida", "kas"):
        write_dataset_sentinel(q, key, config_slice={}, config_hash_value="h",
                               records_in=1, records_out=1)
    _apply_force(out, stage="quality", run_all=False, dataset_keys=["gigafida"])
    assert dataset_sentinel_is_current(q, "gigafida", "h") is False
    assert dataset_sentinel_is_current(q, "kas", "h") is True


def test_force_corpus_stage_removes_corpus_folders(tmp_path: Path) -> None:
    """--force --all --stage exact_dedup removes dedup data + sentinel and dedup state."""
    from slm4ie.data.curate.sentinel import write_sentinel
    from scripts.data.to_pretrain import _apply_force

    out = tmp_path / "pretrain"
    dedup = out / "05_1_dedup"
    write_sentinel(dedup, config_slice={}, config_hash_value="h",
                   records_in=1, records_out=1)
    (dedup / "alfa").mkdir(parents=True)
    (dedup / "alfa" / "000.jsonl.gz").write_bytes(b"x")
    state = out / "_dedup_state"
    state.mkdir(parents=True)
    _apply_force(out, stage="exact_dedup", run_all=True, dataset_keys=["alfa"])
    assert not dedup.exists()
    assert not state.exists()


def test_force_all_stage_all_nukes_output(tmp_path: Path) -> None:
    """--force --all (default stage all) clears the whole output dir."""
    from scripts.data.to_pretrain import _apply_force

    out = tmp_path / "pretrain"
    (out / "00_convert" / "alfa").mkdir(parents=True)
    (out / "00_convert" / "alfa" / "000.jsonl.gz").write_bytes(b"x")
    _apply_force(out, stage="all", run_all=True, dataset_keys=["alfa"])
    assert list(out.iterdir()) == []
