"""Tests for scripts/data/to_pretrain.py helpers."""

import os
from pathlib import Path

import pytest

from scripts.data.to_pretrain import (
    _build_convert_params,
    _build_language_params,
    _build_quality_config,
    _build_spam_config,
    _convert_dataset_current,
    _convert_input_fingerprint,
    _filter_stage_subset,
)
from slm4ie.data.curate.overrides import STAGE_KNOBS, effective_stage_config
from slm4ie.data.curate.sentinel import write_dataset_sentinel


def _write_extracted(input_dir: Path, key: str, text: str = "x") -> Path:
    """Write a minimal extracted `<key>.jsonl` and return its path."""
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / f"{key}.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Override pass-through: every overridable knob of every scoped stage must
# flow override -> effective_stage_config -> stage builder -> config object.
# ---------------------------------------------------------------------------

#: Distinct non-default value per quality knob (field name == knob name).
_QUALITY_OVERRIDES = {
    "min_doc_words": 7,
    "max_doc_words": 12345,
    "min_avg_word_length": 1,
    "max_avg_word_length": 99,
    "max_symbol_word_ratio": 0.42,
    "max_bullet_lines_ratio": 0.55,
    "max_ellipsis_lines_ratio": 0.9,
    "max_non_alpha_words_ratio": 0.33,
    "min_stop_words": 4,
}

#: Distinct non-default value per spam knob except `model` (field == knob).
_SPAM_OVERRIDES = {
    "min_adult_hits": 5,
    "min_spam_hits": 6,
    "keep_fraction": 0.25,
    "default_language": "de",
    "url_blocklist": False,
    "use_ldnoobw": False,
    "model_threshold": 0.77,
}

#: Distinct non-default value per convert knob (field name == knob name).
_CONVERT_OVERRIDES = {
    "text_field": "body",
    "id_field": "uri",
    "metadata_fields": ["url", "title"],
    "include_annotations": True,
    "max_shard_bytes": 4242,
}

#: language knob -> (override value, LanguageParams field) — knob names
#: `targets`/`candidates` map to renamed fields.
_LANGUAGE_OVERRIDES = {
    "targets": (["de", "fr"], "target_languages"),
    "candidates": (["sl", "hr"], "candidate_languages"),
    "mode": ("tag", "mode"),
    "minimum_relative_distance": (0.25, "minimum_relative_distance"),
    "low_accuracy": (True, "low_accuracy"),
    "max_chars": (2000, "max_chars"),
}


def test_quality_overrides_cover_every_knob_and_pass_through() -> None:
    """Each quality knob, when overridden, reaches the QualityConfig."""
    assert set(_QUALITY_OVERRIDES) == STAGE_KNOBS["quality"]
    for knob, value in _QUALITY_OVERRIDES.items():
        eff = effective_stage_config({"quality": {}}, {"d": {"quality": {knob: value}}}, "d", "quality")
        qc = _build_quality_config(eff)
        assert getattr(qc, knob) == value, f"quality knob {knob!r} did not pass through"


def test_spam_overrides_cover_every_knob_and_pass_through() -> None:
    """Each spam knob (bar `model`), when overridden, reaches the SpamConfig."""
    # `model` is exercised separately because setting it raises.
    assert set(_SPAM_OVERRIDES) | {"model"} == STAGE_KNOBS["spam"]
    for knob, value in _SPAM_OVERRIDES.items():
        eff = effective_stage_config({"spam": {}}, {"d": {"spam": {knob: value}}}, "d", "spam")
        sc = _build_spam_config(eff)
        assert getattr(sc, knob) == value, f"spam knob {knob!r} did not pass through"


def test_spam_model_override_raises() -> None:
    """Overriding spam.model is rejected (no resolver is wired)."""
    eff = effective_stage_config({"spam": {}}, {"d": {"spam": {"model": "x"}}}, "d", "spam")
    with pytest.raises(ValueError, match="model"):
        _build_spam_config(eff)


def test_convert_overrides_cover_every_knob_and_pass_through() -> None:
    """Each convert knob, when overridden, reaches the ConvertParams."""
    assert set(_CONVERT_OVERRIDES) == STAGE_KNOBS["convert"]
    for knob, value in _CONVERT_OVERRIDES.items():
        eff = effective_stage_config({"convert": {}}, {"d": {"convert": {knob: value}}}, "d", "convert")
        cp = _build_convert_params(eff)
        assert getattr(cp, knob) == value, f"convert knob {knob!r} did not pass through"


def test_language_overrides_cover_every_knob_and_pass_through() -> None:
    """Each language knob, when overridden, reaches the LanguageParams."""
    assert set(_LANGUAGE_OVERRIDES) == STAGE_KNOBS["language"]
    for knob, (value, field) in _LANGUAGE_OVERRIDES.items():
        eff = effective_stage_config(
            {"language": {}}, {"d": {"language": {knob: value}}}, "d", "language"
        )
        lp = _build_language_params(eff)
        assert getattr(lp, field) == value, f"language knob {knob!r} did not pass through"


def test_repetition_has_no_overridable_knobs() -> None:
    """Repetition exposes no knobs today; documents the gap explicitly."""
    assert STAGE_KNOBS["repetition"] == frozenset()


def test_bucket_keys_by_effective_hash_groups_shared_configs() -> None:
    """Datasets sharing an effective config land in one bucket; overrides split out."""
    from scripts.data.to_pretrain import _bucket_keys_by_effective_hash
    from slm4ie.data.curate import config_hash

    cfg = {"quality": {"min_doc_words": 20, "max_ellipsis_lines_ratio": 0.3}}
    overrides = {"news": {"quality": {"max_ellipsis_lines_ratio": 0.9}}}
    extra = b""
    buckets = _bucket_keys_by_effective_hash(
        ["a", "b", "news"], "quality", cfg, overrides, extra
    )
    groups = sorted(sorted(v) for v in buckets.values())
    assert groups == [["a", "b"], ["news"]]
    # The default bucket's hash equals the plain global-slice hash (rollout-safe).
    default_hash = config_hash(
        {"min_doc_words": 20, "max_ellipsis_lines_ratio": 0.3}, extra=extra
    )
    assert default_hash in buckets
    assert sorted(buckets[default_hash]) == ["a", "b"]


def test_bucket_keys_two_datasets_sharing_one_override_group_together() -> None:
    """Two datasets with the SAME override share a single bucket."""
    from scripts.data.to_pretrain import _bucket_keys_by_effective_hash

    cfg = {"quality": {"min_doc_words": 20}}
    overrides = {
        "x": {"quality": {"min_doc_words": 5}},
        "y": {"quality": {"min_doc_words": 5}},
    }
    buckets = _bucket_keys_by_effective_hash(["x", "y"], "quality", cfg, overrides, b"")
    assert len(buckets) == 1
    assert sorted(next(iter(buckets.values()))) == ["x", "y"]


def test_curate_rejects_bad_override(tmp_path: Path) -> None:
    """_curate fails fast (before any stage) on an invalid override block."""
    import yaml

    from scripts.data import to_pretrain
    from slm4ie.data.curate.overrides import OverrideConfigError

    cfgs = tmp_path / "configs" / "data"
    cfgs.mkdir(parents=True)
    (cfgs / "extract.yaml").write_text(
        yaml.safe_dump({"datasets": {"news": {"extractor": "jsonl", "domain": "news"}}})
    )
    (cfgs / "pretrain.yaml").write_text(
        yaml.safe_dump(
            {
                "input_dir": str(tmp_path / "in"),
                "output_dir": str(tmp_path / "out"),
                "quality": {"min_doc_words": 20},
                "overrides": {"news": {"quality": {"max_elipsis_lines_ratio": 0.9}}},
            }
        )
    )
    with pytest.raises(OverrideConfigError, match="unknown knob"):
        to_pretrain._curate(
            datasets=["news"],
            run_all=False,
            stage="all",
            input_dir=None,
            output_dir=None,
            force=False,
            workers=1,
            pretrain_config=cfgs / "pretrain.yaml",
            extract_config=cfgs / "extract.yaml",
        )


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
