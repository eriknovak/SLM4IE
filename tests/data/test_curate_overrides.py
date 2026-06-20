"""Tests for the per-dataset pretrain override resolver/validator."""

from dataclasses import fields

import pytest

pytest.importorskip("datatrove")

from slm4ie.data.curate.overrides import (  # noqa: E402
    STAGE_KNOBS,
    OverrideConfigError,
    effective_stage_config,
    validate_overrides,
)
from slm4ie.data.curate.pipeline import QualityConfig  # noqa: E402
from slm4ie.data.curate.spam import SpamConfig  # noqa: E402


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
        validate_overrides(
            {"a": {"quality": {"max_elipsis_lines_ratio": 0.9}}}, ["a"]
        )


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
