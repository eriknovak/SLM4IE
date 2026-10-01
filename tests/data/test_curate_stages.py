"""Tests for the stage name/folder mapping in slm4ie.data.curate.stages."""

from pathlib import Path

from slm4ie.data.curate.stages import (
    ALL_STAGE_NAMES,
    CORPUS_STAGES,
    SCOPED_STAGES,
    STAGE_DIRS,
    STAGE_NAMES,
    cascade_from,
    config_slice_keys,
    final_corpus_dir,
    is_scoped,
    statistics_dir,
)


def test_stage_names_are_in_pipeline_order() -> None:
    """STAGE_NAMES lists the eight stages in execution order."""
    assert STAGE_NAMES == (
        "convert",
        "language",
        "spam",
        "quality",
        "repetition",
        "exact_dedup",
        "sentence_dedup",
        "statistics",
    )


def test_all_stage_names_includes_sentinel() -> None:
    """ALL_STAGE_NAMES is STAGE_NAMES plus the 'all' sentinel."""
    assert ALL_STAGE_NAMES == STAGE_NAMES + ("all",)


def test_stage_dirs_use_numeric_prefix() -> None:
    """Each stage maps to its numbered folder name."""
    assert STAGE_DIRS == {
        "convert": "00_convert",
        "language": "01_language",
        "spam": "02_spam",
        "quality": "03_quality",
        "repetition": "04_repetition",
        "exact_dedup": "05_exact_dedup",
        "sentence_dedup": "06_sentence_dedup",
        "statistics": "07_statistics",
    }


def test_final_corpus_dir_is_sentence_dedup() -> None:
    """The final pretraining corpus lives under 06_sentence_dedup/."""
    assert final_corpus_dir() == "06_sentence_dedup"


def test_statistics_dir_matches_mapping() -> None:
    """Stats lives under 07_statistics/."""
    assert statistics_dir() == "07_statistics"


def test_config_slice_keys_per_stage() -> None:
    """Each stage advertises the top-level YAML key(s) that govern it."""
    assert config_slice_keys("convert") == ("convert",)
    assert config_slice_keys("language") == ("language",)
    assert config_slice_keys("spam") == ("spam",)
    assert config_slice_keys("quality") == ("quality",)
    assert config_slice_keys("repetition") == ("repetition",)
    assert config_slice_keys("exact_dedup") == ("exact_dedup",)
    assert config_slice_keys("sentence_dedup") == ("sentence_dedup",)
    assert config_slice_keys("statistics") == ("statistics",)


def test_cascade_from_returns_stage_and_successors() -> None:
    """cascade_from yields the stage and every downstream stage in order."""
    assert cascade_from("convert") == STAGE_NAMES
    assert cascade_from("language") == STAGE_NAMES[1:]
    assert cascade_from("spam") == STAGE_NAMES[2:]
    assert cascade_from("quality") == STAGE_NAMES[3:]
    assert cascade_from("exact_dedup") == STAGE_NAMES[5:]
    assert cascade_from("statistics") == ("statistics",)


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
    assert upstream_stage("spam") == "language"
    assert upstream_stage("quality") == "spam"
    assert upstream_stage("repetition") == "quality"
    assert upstream_stage("exact_dedup") == "repetition"
    assert upstream_stage("sentence_dedup") == "exact_dedup"
    assert upstream_stage("statistics") == "sentence_dedup"


def test_upstream_stage_rejects_unknown_stage() -> None:
    """An unknown stage name raises KeyError."""
    import pytest

    from slm4ie.data.curate.stages import upstream_stage

    with pytest.raises(KeyError):
        upstream_stage("not_a_stage")


def test_scoped_and_corpus_partition_stage_names() -> None:
    """SCOPED_STAGES + CORPUS_STAGES is exactly STAGE_NAMES, in order."""
    assert SCOPED_STAGES + CORPUS_STAGES == STAGE_NAMES
    assert set(SCOPED_STAGES).isdisjoint(CORPUS_STAGES)


def test_scoped_membership() -> None:
    """convert/language/spam/quality/repetition are scoped; dedup/statistics are not."""
    assert SCOPED_STAGES == ("convert", "language", "spam", "quality", "repetition")
    assert CORPUS_STAGES == ("exact_dedup", "sentence_dedup", "statistics")


def test_is_scoped() -> None:
    """is_scoped returns True only for scoped stages."""
    assert is_scoped("quality") is True
    assert is_scoped("exact_dedup") is False


def _copy_package(tmp_path: Path) -> Path:
    """Copy the curate package's sources into *tmp_path* and return the copy."""
    import shutil

    import slm4ie.data.curate as curate_pkg

    target = tmp_path / "curate"
    shutil.copytree(Path(curate_pkg.__file__).parent, target, ignore=shutil.ignore_patterns("__pycache__"))
    return target


def test_stage_versions_are_code_hashes() -> None:
    """Every stage carries a version derived from its code."""
    from slm4ie.data.curate.stages import STAGE_NAMES, STAGE_VERSIONS

    assert set(STAGE_VERSIONS) == set(STAGE_NAMES)
    assert all(v.startswith("sha256:") for v in STAGE_VERSIONS.values())


def test_editing_a_stage_module_changes_only_its_version(tmp_path: Path) -> None:
    """A comment added to the language module bumps the language stage alone."""
    from slm4ie.data.curate.stages import STAGE_NAMES, code_version

    package = _copy_package(tmp_path)
    before = {stage: code_version(stage, package) for stage in STAGE_NAMES}
    with (package / "language.py").open("a", encoding="utf-8") as fh:
        fh.write("\n# a comment\n")
    after = {stage: code_version(stage, package) for stage in STAGE_NAMES}
    assert [s for s in STAGE_NAMES if before[s] != after[s]] == ["language"]


def test_editing_a_builder_changes_only_its_stage(tmp_path: Path) -> None:
    """Editing the quality builder in the shared pipeline module bumps quality alone."""
    from slm4ie.data.curate.stages import STAGE_NAMES, code_version

    package = _copy_package(tmp_path)
    before = {stage: code_version(stage, package) for stage in STAGE_NAMES}
    source = (package / "pipeline.py").read_text(encoding="utf-8")
    (package / "pipeline.py").write_text(
        source.replace(
            '    in_ = input_override if input_override is not None else paths.stage_dir("language")\n    out = output_override if output_override is not None else paths.stage_dir("quality")',
            '    # tweak\n    in_ = input_override if input_override is not None else paths.stage_dir("language")\n    out = output_override if output_override is not None else paths.stage_dir("quality")',
        ),
        encoding="utf-8",
    )
    after = {stage: code_version(stage, package) for stage in STAGE_NAMES}
    assert [s for s in STAGE_NAMES if before[s] != after[s]] == ["quality"]
