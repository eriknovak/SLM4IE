"""Tests for the repetition stage builder (`slm4ie.data.curate.stages.repetition`)."""

import importlib.metadata  # noqa: F401  (datatrove workaround)
import importlib.util  # noqa: F401  (datatrove workaround)
from pathlib import Path

import pytest

pytest.importorskip("datatrove")

from datatrove.pipeline.filters import (  # noqa: E402
    GopherQualityFilter,
    GopherRepetitionFilter,
)

from slm4ie.data.curate.paths import CuratePaths  # noqa: E402
from slm4ie.data.curate.stages.repetition import RepetitionConfig, build_repetition_executors  # noqa: E402


def _paths(tmp_path: Path) -> CuratePaths:
    """Build a CuratePaths anchored under *tmp_path* for structural tests."""
    return CuratePaths(
        input_folder=tmp_path / "datatrove",
        output_dir=tmp_path / "curated",
    )


class TestRepetitionStage:
    """The repetition stage reads 02_quality/ and writes 03_repetition/."""

    def test_returns_one_executor(self, tmp_path: Path) -> None:
        """The repetition stage runs as a single executor."""
        execs = build_repetition_executors(_paths(tmp_path))
        assert len(execs) == 1

    def test_pipeline_contains_repetition_filter(self, tmp_path: Path) -> None:
        """The pipeline runs GopherRepetitionFilter but NOT the quality filter."""
        execs = build_repetition_executors(_paths(tmp_path))
        types_ = [type(s) for s in execs[0].pipeline]
        assert GopherRepetitionFilter in types_
        assert GopherQualityFilter not in types_


def test_repetition_executor_honors_input_override(tmp_path: Path) -> None:
    """build_repetition_executors reads from input_override when provided."""
    from slm4ie.data.curate.paths import CuratePaths
    from slm4ie.data.curate.stages.repetition import build_repetition_executors

    paths = CuratePaths(input_folder=tmp_path / "in", output_dir=tmp_path / "out")
    override = tmp_path / "view"
    override.mkdir()
    execs = build_repetition_executors(paths, tasks=1, input_override=override)
    reader = execs[0].pipeline[0]
    assert str(override) in reader.data_folder.path


def test_repetition_executor_passes_its_config_to_the_filter(tmp_path: Path) -> None:
    """Every RepetitionConfig field reaches GopherRepetitionFilter unchanged."""
    rc = RepetitionConfig(dup_line_frac=0.5, top_n_grams=((2, 0.3),), dup_n_grams=((5, 0.2),))
    execs = build_repetition_executors(_paths(tmp_path), repetition_config=rc)
    flt = next(s for s in execs[0].pipeline if isinstance(s, GopherRepetitionFilter))
    assert flt.dup_line_frac == 0.5
    assert flt.top_n_grams == ((2, 0.3),)
    assert flt.dup_n_grams == ((5, 0.2),)
