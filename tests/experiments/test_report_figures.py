"""The project theme: datachart starts in it, and it is colour-blind safe on the report plate."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("datachart_themes.slm4ie")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import report_figures  # noqa: E402  — path set above so the shared helper resolves


def test_session_starts_in_the_project_theme() -> None:
    """The installed default theme is slm4ie, so every chart draws in its palette."""
    from datachart.config import config
    from datachart_themes.slm4ie import PALETTE

    assert config.theme == report_figures.THEME
    assert config["color_general_multiple"] == PALETTE


def test_series_palette_passes_the_gate() -> None:
    """Every pair of series stays apart for colour-blind readers on the report plate.

    The soft fills sit under 3:1 against the plate by design; the theme's ink
    outline on every filled mark carries that contrast instead.
    """
    from datachart.themes import score_palette
    from datachart_themes.slm4ie import PALETTE

    score = score_palette(PALETTE, "#fbfcfd")
    assert score.verdict == "pass", str(score)


def test_save_figure_takes_a_figure_or_a_builder(tmp_path: Path) -> None:
    """Both call shapes write a date-free SVG in the report's font stack."""
    from datachart.charts import BarChart

    def build():
        return BarChart([{"label": "a", "y": 1}, {"label": "b", "y": 2}])

    for name, figure in (("built.svg", build()), ("builder.svg", build)):
        report_figures.save_figure(figure, tmp_path / name)
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert "<dc:date>" not in text
        assert "Instrument Sans" in text


def test_save_figure_writes_the_same_bytes_twice(tmp_path: Path) -> None:
    """Two saves of one figure are identical, so a rerun leaves an unchanged file untouched in git."""
    from datachart.charts import LineChart

    def build():
        return LineChart([{"x": 1, "y": 1}, {"x": 2, "y": 3}])

    report_figures.save_figure(build, tmp_path / "first.svg")
    report_figures.save_figure(build, tmp_path / "second.svg")
    assert (tmp_path / "first.svg").read_bytes() == (tmp_path / "second.svg").read_bytes()
