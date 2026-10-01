"""Tests for the curation runner helpers (slm4ie/data/curate/runner.py)."""

import os
from pathlib import Path

import pytest

pytest.importorskip("datatrove")

from slm4ie.data.curate.runner import (  # noqa: E402
    _build_convert_params,
    _build_language_params,
    _build_quality_config,
    _build_spam_config,
    _convert_input_files,
    _input_files_digest,
    _filter_stage_subset,
    _list_datasets,
)
from slm4ie.data.curate.overrides import STAGE_KNOBS, effective_stage_config


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
        eff = effective_stage_config({"language": {}}, {"d": {"language": {knob: value}}}, "d", "language")
        lp = _build_language_params(eff)
        assert getattr(lp, field) == value, f"language knob {knob!r} did not pass through"


def test_repetition_has_no_overridable_knobs() -> None:
    """Repetition exposes no knobs today; documents the gap explicitly."""
    assert STAGE_KNOBS["repetition"] == frozenset()


def test_bucket_keys_by_effective_hash_groups_shared_configs() -> None:
    """Datasets sharing an effective config land in one bucket; overrides split out."""
    from slm4ie.data.curate.runner import _bucket_keys_by_effective_hash
    from slm4ie.data.curate import config_hash

    cfg = {"quality": {"min_doc_words": 20, "max_ellipsis_lines_ratio": 0.3}}
    overrides = {"news": {"quality": {"max_ellipsis_lines_ratio": 0.9}}}
    extra = b""
    buckets = _bucket_keys_by_effective_hash(["a", "b", "news"], "quality", cfg, overrides, extra)
    groups = sorted(sorted(v) for v in buckets.values())
    assert groups == [["a", "b"], ["news"]]
    # The default bucket's hash equals the plain global-slice hash (rollout-safe).
    default_hash = config_hash({"min_doc_words": 20, "max_ellipsis_lines_ratio": 0.3}, extra=extra)
    assert default_hash in buckets
    assert sorted(buckets[default_hash]) == ["a", "b"]


def test_bucket_keys_two_datasets_sharing_one_override_group_together() -> None:
    """Two datasets with the SAME override share a single bucket."""
    from slm4ie.data.curate.runner import _bucket_keys_by_effective_hash

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

    from slm4ie.data.curate.overrides import OverrideConfigError
    from slm4ie.data.curate.runner import curate

    cfgs = tmp_path / "configs" / "data"
    cfgs.mkdir(parents=True)
    (cfgs / "extract.yaml").write_text(yaml.safe_dump({"datasets": {"news": {"extractor": "jsonl", "domain": "news"}}}))
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
        curate(
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


def _digest(input_dir: Path, key: str, annotations: bool = False, previous: object = None) -> str:
    """Return the convert input digest of *key* under *input_dir*."""
    return _input_files_digest(_convert_input_files(input_dir, key, annotations, previous))  # type: ignore[arg-type]


def test_convert_input_digest_changes_with_content(tmp_path: Path) -> None:
    """Rewriting the source with other bytes of the same size changes the digest."""
    _write_extracted(tmp_path, "news", "abcde")
    before = _digest(tmp_path, "news")
    _write_extracted(tmp_path, "news", "vwxyz")
    assert _digest(tmp_path, "news") != before


def test_convert_input_digest_ignores_mtime(tmp_path: Path) -> None:
    """Touching the source, or rewriting identical bytes, keeps the digest."""
    path = _write_extracted(tmp_path, "news", "abcde")
    before = _digest(tmp_path, "news")
    os.utime(path, ns=(2_000_000_000_000_000_000, 2_000_000_000_000_000_000))
    assert _digest(tmp_path, "news") == before
    _write_extracted(tmp_path, "news", "abcde")
    assert _digest(tmp_path, "news") == before


def test_convert_input_files_reuse_hash_when_size_and_mtime_match(tmp_path: Path) -> None:
    """An untouched file keeps its recorded hash instead of being reread."""
    _write_extracted(tmp_path, "news", "abcde")
    files = _convert_input_files(tmp_path, "news", False)
    recorded = {"news.jsonl": {**files["news.jsonl"], "sha256": "recorded"}}  # type: ignore[dict-item]
    assert _convert_input_files(tmp_path, "news", False, recorded)["news.jsonl"]["sha256"] == "recorded"  # type: ignore[index]


def test_convert_input_files_mark_absent(tmp_path: Path) -> None:
    """A missing source file is recorded as absent, not raised."""
    assert _convert_input_files(tmp_path, "missing", False) == {"missing.jsonl": None}


def test_convert_input_digest_folds_annotations(tmp_path: Path) -> None:
    """With annotations on, the sidecar participates in the digest."""
    _write_extracted(tmp_path, "news", "abc")
    without = _digest(tmp_path, "news")
    (tmp_path / "news.annotations.jsonl.gz").write_bytes(b"gz")
    assert _digest(tmp_path, "news", annotations=True) != without


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
    from slm4ie.data.curate.runner import _stage_extra

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
    assert roster in _stage_extra("statistics", sw, sp, roster)
    assert sw in _stage_extra("statistics", sw, sp, roster)  # statistics also folds stopwords


def test_resolve_requested_stages() -> None:
    """Subset 'all' = scoped stages; --all 'all' = every stage."""
    from slm4ie.data.curate.runner import _resolve_requested_stages
    from slm4ie.data.curate.stages import SCOPED_STAGES, STAGE_NAMES

    assert _resolve_requested_stages(stage="all", run_all=False) == SCOPED_STAGES
    assert _resolve_requested_stages(stage="all", run_all=True) == STAGE_NAMES
    assert _resolve_requested_stages(stage="quality", run_all=False) == ("quality",)
    assert _resolve_requested_stages(stage="exact_dedup", run_all=True) == ("exact_dedup",)


def test_force_subset_stage_drops_only_requested_keys(tmp_path: Path) -> None:
    """--force gigafida --stage quality drops gigafida's quality sentinel, keeps others."""
    from slm4ie.data.curate.sentinel import dataset_sentinel_path, write_dataset_sentinel
    from slm4ie.data.curate.runner import _apply_force

    out = tmp_path / "pretrain"
    q = out / "03_quality"
    for key in ("gigafida", "kas"):
        write_dataset_sentinel(q, key, config_slice={}, config_hash_value="h", records_in=1, records_out=1)
    _apply_force(out, stage="quality", run_all=False, dataset_keys=["gigafida"])
    assert not dataset_sentinel_path(q, "gigafida").exists()
    assert dataset_sentinel_path(q, "kas").exists()


def test_force_corpus_stage_removes_corpus_folders(tmp_path: Path) -> None:
    """--force --all --stage exact_dedup removes dedup data + sentinel and dedup state."""
    from slm4ie.data.curate.sentinel import write_sentinel
    from slm4ie.data.curate.runner import _apply_force

    out = tmp_path / "pretrain"
    dedup = out / "05_exact_dedup"
    write_sentinel(dedup, config_slice={}, config_hash_value="h", records_in=1, records_out=1)
    (dedup / "alfa").mkdir(parents=True)
    (dedup / "alfa" / "000.jsonl.gz").write_bytes(b"x")
    state = out / "_partial" / "05_exact_dedup.scratch"
    state.mkdir(parents=True)
    _apply_force(out, stage="exact_dedup", run_all=True, dataset_keys=["alfa"])
    assert not dedup.exists()
    assert not state.exists()


def test_force_all_stage_all_nukes_output(tmp_path: Path) -> None:
    """--force --all (default stage all) clears the whole output dir."""
    from slm4ie.data.curate.runner import _apply_force

    out = tmp_path / "pretrain"
    (out / "00_convert" / "alfa").mkdir(parents=True)
    (out / "00_convert" / "alfa" / "000.jsonl.gz").write_bytes(b"x")
    _apply_force(out, stage="all", run_all=True, dataset_keys=["alfa"])
    assert list(out.iterdir()) == []


def test_list_datasets_skips_benchmark_role(tmp_path: Path) -> None:
    """`--all` resolves to pretraining sources only; benchmarks stay out of the corpus."""
    cfg = tmp_path / "extract.yaml"
    cfg.write_text(
        "datasets:\n"
        "  web:\n    extractor: jsonl\n    domain: web\n"
        "  gated_web:\n    extractor: jsonl\n    domain: web\n    access: gated\n"
        "  gold:\n    extractor: conllu\n    domain: mixed\n    role: benchmark\n",
        encoding="utf-8",
    )
    assert _list_datasets(cfg) == ["web", "gated_web"]
