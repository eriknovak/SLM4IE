"""Tests for the stage name/folder mapping in slm4ie.data.curate.stages."""

from slm4ie.data.curate.stages import (
    ALL_STAGE_NAMES,
    STAGE_DIRS,
    STAGE_NAMES,
    cascade_from,
    config_slice_keys,
    final_corpus_dir,
    stats_dir,
)


def test_stage_names_are_in_pipeline_order() -> None:
    """STAGE_NAMES lists the seven stages in execution order."""
    assert STAGE_NAMES == (
        "convert",
        "language",
        "quality",
        "repetition",
        "exact_dedup",
        "sentence_dedup",
        "stats",
    )


def test_all_stage_names_includes_sentinel() -> None:
    """ALL_STAGE_NAMES is STAGE_NAMES plus the 'all' sentinel."""
    assert ALL_STAGE_NAMES == STAGE_NAMES + ("all",)


def test_stage_dirs_use_numeric_prefix() -> None:
    """Each stage maps to its numbered folder name."""
    assert STAGE_DIRS == {
        "convert": "00_convert",
        "language": "01_language",
        "quality": "02_quality",
        "repetition": "03_repetition",
        "exact_dedup": "04_1_dedup",
        "sentence_dedup": "04_2_dedup",
        "stats": "05_statistics",
    }


def test_final_corpus_dir_is_sentence_dedup() -> None:
    """The final pretraining corpus lives under 04_2_dedup/."""
    assert final_corpus_dir() == "04_2_dedup"


def test_stats_dir_matches_mapping() -> None:
    """Stats lives under 05_statistics/."""
    assert stats_dir() == "05_statistics"


def test_config_slice_keys_per_stage() -> None:
    """Each stage advertises the top-level YAML key(s) that govern it."""
    assert config_slice_keys("convert") == ("convert",)
    assert config_slice_keys("language") == ("language",)
    assert config_slice_keys("quality") == ("quality",)
    assert config_slice_keys("repetition") == ("repetition",)
    assert config_slice_keys("exact_dedup") == ("exact_dedup",)
    assert config_slice_keys("sentence_dedup") == ("sentence_dedup",)
    assert config_slice_keys("stats") == ("stats",)


def test_cascade_from_returns_stage_and_successors() -> None:
    """cascade_from yields the stage and every downstream stage in order."""
    assert cascade_from("convert") == STAGE_NAMES
    assert cascade_from("language") == STAGE_NAMES[1:]
    assert cascade_from("quality") == STAGE_NAMES[2:]
    assert cascade_from("exact_dedup") == STAGE_NAMES[4:]
    assert cascade_from("stats") == ("stats",)


def test_cascade_from_rejects_unknown_stage() -> None:
    """An unknown stage name raises KeyError."""
    import pytest

    with pytest.raises(KeyError):
        cascade_from("not_a_stage")


def test_stage_keysets_are_consistent() -> None:
    """STAGE_NAMES, STAGE_DIRS, and the slice-key map cover the same stages."""
    assert set(STAGE_DIRS) == set(STAGE_NAMES)
    for name in STAGE_NAMES:
        # Indirect probe of the private slice-key map via the public accessor:
        # the call must not raise and must return a non-empty tuple.
        assert config_slice_keys(name)


def test_config_slice_keys_rejects_unknown_stage() -> None:
    """An unknown stage name raises KeyError."""
    import pytest

    with pytest.raises(KeyError):
        config_slice_keys("not_a_stage")


def test_upstream_stage_for_first_stage_is_none() -> None:
    """The first stage (convert) has no upstream stage."""
    from slm4ie.data.curate.stages import upstream_stage

    assert upstream_stage("convert") is None


def test_upstream_stage_returns_predecessor() -> None:
    """upstream_stage returns the preceding stage in execution order."""
    from slm4ie.data.curate.stages import upstream_stage

    assert upstream_stage("language") == "convert"
    assert upstream_stage("quality") == "language"
    assert upstream_stage("repetition") == "quality"
    assert upstream_stage("exact_dedup") == "repetition"
    assert upstream_stage("sentence_dedup") == "exact_dedup"
    assert upstream_stage("stats") == "sentence_dedup"


def test_upstream_stage_rejects_unknown_stage() -> None:
    """An unknown stage name raises KeyError."""
    import pytest

    from slm4ie.data.curate.stages import upstream_stage

    with pytest.raises(KeyError):
        upstream_stage("not_a_stage")
