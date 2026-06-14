"""Tests for scripts/data/to_pretrain.py helpers."""

import os
from pathlib import Path

import pytest

from scripts.data.to_pretrain import (
    _convert_dataset_current,
    _convert_input_fingerprint,
    _filter_stage_subset,
)
from slm4ie.data.curate.sentinel import write_dataset_sentinel


def _write_extracted(input_dir: Path, key: str, text: str = "x") -> Path:
    """Write a minimal extracted `<key>.jsonl` and return its path."""
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / f"{key}.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def test_convert_input_fingerprint_changes_on_size(tmp_path: Path) -> None:
    """Growing the source file changes its fingerprint."""
    _write_extracted(tmp_path, "news", "short")
    before = _convert_input_fingerprint(tmp_path, "news", False)
    _write_extracted(tmp_path, "news", "a much longer body of text")
    after = _convert_input_fingerprint(tmp_path, "news", False)
    assert before != after


def test_convert_input_fingerprint_changes_on_mtime(tmp_path: Path) -> None:
    """Rewriting with the same size but a newer mtime changes the fingerprint."""
    path = _write_extracted(tmp_path, "news", "abcde")
    before = _convert_input_fingerprint(tmp_path, "news", False)
    os.utime(path, ns=(2_000_000_000_000_000_000, 2_000_000_000_000_000_000))
    after = _convert_input_fingerprint(tmp_path, "news", False)
    assert before != after


def test_convert_input_fingerprint_marks_absent(tmp_path: Path) -> None:
    """A missing source file is encoded distinctly, not raised."""
    fp = _convert_input_fingerprint(tmp_path, "missing", False)
    assert "absent" in fp


def test_convert_input_fingerprint_folds_annotations(tmp_path: Path) -> None:
    """With annotations on, the sidecar participates in the fingerprint."""
    _write_extracted(tmp_path, "news", "abc")
    without = _convert_input_fingerprint(tmp_path, "news", False)
    (tmp_path / "news.annotations.jsonl.gz").write_bytes(b"gz")
    with_ann = _convert_input_fingerprint(tmp_path, "news", True)
    assert without != with_ann


def test_convert_dataset_current_true_when_unchanged(tmp_path: Path) -> None:
    """A fresh sentinel with a matching fingerprint is current."""
    stage = tmp_path / "00_convert"
    input_dir = tmp_path / "extracted"
    _write_extracted(input_dir, "news")
    fp = _convert_input_fingerprint(input_dir, "news", False)
    write_dataset_sentinel(
        stage, "news", config_slice={}, config_hash_value="h",
        records_in=1, records_out=1, input_fingerprint=fp,
    )
    assert _convert_dataset_current(stage, "news", "h", input_dir, False) is True


def test_convert_dataset_current_false_when_input_changed(tmp_path: Path) -> None:
    """A changed source file makes a fingerprinted sentinel stale."""
    stage = tmp_path / "00_convert"
    input_dir = tmp_path / "extracted"
    path = _write_extracted(input_dir, "news", "small")
    fp = _convert_input_fingerprint(input_dir, "news", False)
    write_dataset_sentinel(
        stage, "news", config_slice={}, config_hash_value="h",
        records_in=1, records_out=1, input_fingerprint=fp,
    )
    path.write_text("a substantially larger body", encoding="utf-8")
    assert _convert_dataset_current(stage, "news", "h", input_dir, False) is False


def test_convert_dataset_current_false_on_config_change(tmp_path: Path) -> None:
    """A config-hash mismatch is stale regardless of the fingerprint."""
    stage = tmp_path / "00_convert"
    input_dir = tmp_path / "extracted"
    _write_extracted(input_dir, "news")
    fp = _convert_input_fingerprint(input_dir, "news", False)
    write_dataset_sentinel(
        stage, "news", config_slice={}, config_hash_value="old",
        records_in=1, records_out=1, input_fingerprint=fp,
    )
    assert _convert_dataset_current(stage, "news", "new", input_dir, False) is False


def test_convert_dataset_current_grandfathers_legacy_old_input(tmp_path: Path) -> None:
    """A legacy sentinel (no fingerprint) is current when input predates it."""
    stage = tmp_path / "00_convert"
    input_dir = tmp_path / "extracted"
    path = _write_extracted(input_dir, "news")
    # Source file far in the past; sentinel completed_at is "now".
    os.utime(path, ns=(1_000_000_000_000_000_000, 1_000_000_000_000_000_000))
    write_dataset_sentinel(
        stage, "news", config_slice={}, config_hash_value="h",
        records_in=1, records_out=1,  # no input_fingerprint -> legacy
    )
    assert _convert_dataset_current(stage, "news", "h", input_dir, False) is True


def test_convert_dataset_current_legacy_stale_when_input_newer(tmp_path: Path) -> None:
    """A legacy sentinel is stale when the source file is newer than completion."""
    import json

    from slm4ie.data.curate.sentinel import SENTINEL_NAME

    stage = tmp_path / "00_convert"
    input_dir = tmp_path / "extracted"
    path = _write_extracted(input_dir, "news")
    # Hand-write a legacy sentinel completed in the distant past.
    (stage / "news").mkdir(parents=True)
    (stage / "news" / SENTINEL_NAME).write_text(
        json.dumps({
            "completed_at": "2000-01-01T00:00:00+00:00",
            "config_hash": "h",
            "config_slice": {},
            "records_in": 1,
            "records_out": 1,
        }),
        encoding="utf-8",
    )
    # Source file is "now" — newer than the recorded completion.
    os.utime(path, ns=(4_000_000_000_000_000_000, 4_000_000_000_000_000_000))
    assert _convert_dataset_current(stage, "news", "h", input_dir, False) is False


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
