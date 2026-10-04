"""Tests for the curation config: overrides, stage knobs, buckets and the dataset roster."""

from dataclasses import fields
from pathlib import Path

import pytest

pytest.importorskip("datatrove")

from slm4ie.data.curate.config import (
    STAGE_KNOBS,
    OverrideConfigError,
    _list_datasets,
    effective_stage_config,
    load_curate_config,
    validate_overrides,
    validate_spam_knobs,
)
from slm4ie.data.curate.stages.convert import _build_convert_params  # noqa: E402
from slm4ie.data.curate.stages.language import _build_language_params  # noqa: E402
from slm4ie.data.curate.stages.quality import QualityConfig, _build_quality_config  # noqa: E402
from slm4ie.data.curate.stages.spam import SpamConfig, _build_spam_config  # noqa: E402


def test_quality_knobs_match_dataclass() -> None:
    """The quality knob whitelist stays in lockstep with QualityConfig."""
    assert STAGE_KNOBS["quality"] == {f.name for f in fields(QualityConfig)}


def test_spam_knobs_match_dataclass() -> None:
    """The spam knob whitelist stays in lockstep with SpamConfig."""
    assert STAGE_KNOBS["spam"] == {f.name for f in fields(SpamConfig)}


def test_effective_config_deep_merges_over_global() -> None:
    """A dataset override patches only the named knobs; others inherit."""
    cfg = {"quality": {"min_doc_words": 20, "max_ellipsis_lines_ratio": 0.3}}
    overrides = {"slovenian_news": {"quality": {"max_ellipsis_lines_ratio": 0.9}}}
    eff = effective_stage_config(cfg, overrides, "slovenian_news", "quality")
    assert eff == {"min_doc_words": 20, "max_ellipsis_lines_ratio": 0.9}


def test_effective_config_no_override_returns_global_copy() -> None:
    """A dataset with no override yields a value equal to the global slice."""
    cfg = {"quality": {"min_doc_words": 20}}
    eff = effective_stage_config(cfg, {}, "kas", "quality")
    assert eff == {"min_doc_words": 20}
    # Must be a copy, not the same object (no mutation of cfg downstream).
    eff["min_doc_words"] = 999
    assert cfg["quality"]["min_doc_words"] == 20


def test_effective_config_missing_stage_returns_override_only() -> None:
    """When the global stage section is absent, the override stands alone."""
    eff = effective_stage_config({}, {"a": {"language": {"mode": "tag"}}}, "a", "language")
    assert eff == {"mode": "tag"}


def test_validate_accepts_empty_overrides() -> None:
    """Absent/empty overrides validate trivially."""
    validate_overrides({}, ["a", "b"])
    validate_overrides(None, ["a", "b"])


def test_validate_rejects_unknown_dataset() -> None:
    """A dataset key not in the roster is a hard error."""
    with pytest.raises(OverrideConfigError, match="unknown dataset"):
        validate_overrides({"typo_news": {"quality": {"min_doc_words": 5}}}, ["slovenian_news"])


def test_validate_rejects_corpus_stage() -> None:
    """Overriding a corpus stage is a hard error."""
    with pytest.raises(OverrideConfigError, match="scoped stages"):
        validate_overrides({"a": {"exact_dedup": {"precision": 32}}}, ["a"])


def test_validate_rejects_global_key() -> None:
    """Overriding a global key (e.g. stopwords) is a hard error."""
    with pytest.raises(OverrideConfigError, match="scoped stages"):
        validate_overrides({"a": {"stopwords": "sl"}}, ["a"])


def test_validate_rejects_unknown_knob() -> None:
    """A typo'd knob inside a valid stage is a hard error."""
    with pytest.raises(OverrideConfigError, match="unknown knob"):
        validate_overrides({"a": {"quality": {"max_elipsis_lines_ratio": 0.9}}}, ["a"])


def test_validate_accepts_valid_block() -> None:
    """A well-formed override block passes."""
    validate_overrides(
        {
            "slovenian_news": {
                "quality": {"max_ellipsis_lines_ratio": 0.9},
                "language": {"mode": "tag"},
            }
        },
        ["slovenian_news", "kas"],
    )


def test_validate_rejects_non_mapping_section() -> None:
    """A dataset whose value is not a stage->knobs mapping is rejected."""
    with pytest.raises(OverrideConfigError, match="mapping"):
        validate_overrides({"a": ["quality"]}, ["a"])


def test_validate_rejects_non_mapping_knobs() -> None:
    """A stage whose value is not a knob mapping is rejected."""
    with pytest.raises(OverrideConfigError, match="mapping"):
        validate_overrides({"a": {"quality": [1, 2, 3]}}, ["a"])


@pytest.mark.parametrize(
    "knobs",
    [
        {"min_adult_hits": 0},
        {"min_spam_hits": -1},
        {"min_spam_hits": 1.5},
        {"min_adult_hits": True},
        {"keep_fraction": 1.5},
        {"keep_fraction": -0.1},
        {"keep_fraction": "half"},
    ],
)
def test_validate_rejects_out_of_bounds_spam_knobs(knobs) -> None:
    """Spam thresholds below 1 and fractions outside [0, 1] fail in an override."""
    with pytest.raises(OverrideConfigError, match="overrides.a.spam"):
        validate_overrides({"a": {"spam": knobs}}, ["a"])


def test_validate_spam_knobs_names_the_global_slice() -> None:
    """The global slice is checked with its own path in the message."""
    with pytest.raises(OverrideConfigError, match="spam.min_adult_hits"):
        validate_spam_knobs({"min_adult_hits": 0}, "spam")


def test_validate_accepts_large_spam_threshold() -> None:
    """A huge threshold stays legal: it neutralises the signal."""
    validate_overrides({"a": {"spam": {"min_spam_hits": 999999, "keep_fraction": 1}}}, ["a"])


def test_load_curate_config_rejects_bad_global_spam(tmp_path) -> None:
    """The loader bounds-checks the global spam slice."""
    config = tmp_path / "curate.yaml"
    config.write_text("spam:\n  min_adult_hits: 0\n")
    with pytest.raises(OverrideConfigError, match="spam.min_adult_hits"):
        load_curate_config(tmp_path, tmp_path, config, None)


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
    from slm4ie.data.curate.config import bucket_keys_by_effective_hash
    from slm4ie.utils.versioning import config_hash

    cfg = {"quality": {"min_doc_words": 20, "max_ellipsis_lines_ratio": 0.3}}
    overrides = {"news": {"quality": {"max_ellipsis_lines_ratio": 0.9}}}
    extra = b""
    buckets = bucket_keys_by_effective_hash(["a", "b", "news"], "quality", cfg, overrides, extra)
    groups = sorted(sorted(v) for v in buckets.values())
    assert groups == [["a", "b"], ["news"]]
    # The default bucket's hash equals the plain global-slice hash (rollout-safe).
    default_hash = config_hash({"min_doc_words": 20, "max_ellipsis_lines_ratio": 0.3}, extra=extra)
    assert default_hash in buckets
    assert sorted(buckets[default_hash]) == ["a", "b"]


def test_bucket_keys_two_datasets_sharing_one_override_group_together() -> None:
    """Two datasets with the SAME override share a single bucket."""
    from slm4ie.data.curate.config import bucket_keys_by_effective_hash

    cfg = {"quality": {"min_doc_words": 20}}
    overrides = {
        "x": {"quality": {"min_doc_words": 5}},
        "y": {"quality": {"min_doc_words": 5}},
    }
    buckets = bucket_keys_by_effective_hash(["x", "y"], "quality", cfg, overrides, b"")
    assert len(buckets) == 1
    assert sorted(next(iter(buckets.values()))) == ["x", "y"]


def test_stage_extra_folds_roster_only_for_corpus_stages() -> None:
    """Scoped stages exclude the roster; corpus stages include it."""
    from slm4ie.data.curate.config import _stage_extra

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
