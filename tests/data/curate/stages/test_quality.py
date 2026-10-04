"""Tests for the quality stage builder (`slm4ie.data.curate.stages.quality`)."""

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
from slm4ie.data.curate.stages.quality import (  # noqa: E402
    QualityConfig,
    build_quality_executors,
)


def _paths(tmp_path: Path) -> CuratePaths:
    """Build a CuratePaths anchored under *tmp_path* for structural tests."""
    return CuratePaths(
        input_folder=tmp_path / "datatrove",
        output_dir=tmp_path / "curated",
    )


class TestQualityStage:
    """The quality stage reads 02_spam/ and writes 03_quality/."""

    def test_returns_one_executor(self, tmp_path: Path) -> None:
        """The quality stage runs as a single executor."""
        execs = build_quality_executors(_paths(tmp_path))
        assert len(execs) == 1

    def test_pipeline_contains_gopher_quality(self, tmp_path: Path) -> None:
        """The pipeline runs GopherQualityFilter but NOT the repetition filter."""
        execs = build_quality_executors(_paths(tmp_path))
        types_ = [type(s) for s in execs[0].pipeline]
        assert GopherQualityFilter in types_
        assert GopherRepetitionFilter not in types_

    def test_quality_config_threaded(self, tmp_path: Path) -> None:
        """QualityConfig overrides reach the underlying GopherQualityFilter."""
        cfg = QualityConfig(min_doc_words=10, max_doc_words=200, min_stop_words=0)
        execs = build_quality_executors(_paths(tmp_path), quality_config=cfg)
        quality = next(s for s in execs[0].pipeline if isinstance(s, GopherQualityFilter))
        assert quality.min_doc_words == 10
        assert quality.max_doc_words == 200
        assert quality.min_stop_words == 0

    def test_stopwords_become_gopher_stop_words(self, tmp_path: Path) -> None:
        """Stopwords are wired into GopherQualityFilter."""
        execs = build_quality_executors(_paths(tmp_path), stopwords={"in", "je", "na"})
        quality = next(s for s in execs[0].pipeline if isinstance(s, GopherQualityFilter))
        assert {"in", "je", "na"}.issubset(quality.stop_words)


def test_quality_executor_honors_input_override(tmp_path: Path) -> None:
    """build_quality_executors reads from input_override when provided."""
    from slm4ie.data.curate.paths import CuratePaths
    from slm4ie.data.curate.stages.quality import build_quality_executors

    paths = CuratePaths(input_folder=tmp_path / "in", output_dir=tmp_path / "out")
    override = tmp_path / "view"
    override.mkdir()
    execs = build_quality_executors(paths, tasks=1, input_override=override)
    reader = execs[0].pipeline[0]
    assert str(override) in reader.data_folder.path
