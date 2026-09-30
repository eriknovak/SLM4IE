"""The project theme: registered on import, active, and colour-blind safe."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("datachart")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import report_figures  # noqa: E402  — path set above so the shared helper resolves


def test_theme_is_active_after_import() -> None:
    """Importing the module makes the slm4ie theme the one every chart draws in."""
    from datachart.config import config
    from datachart.config.configuration import THEMES

    assert report_figures.THEME in THEMES
    assert config["color_general_multiple"] == report_figures.SERIES
    assert config["plot_heatmap_cmap"] == report_figures.SEQUENTIAL
    assert config["plot_heatmap_cmap_diverging"] == report_figures.DIVERGING


def test_series_palette_passes_the_gate() -> None:
    """Every pair of series stays apart for colour-blind readers and reaches 3:1 on the plate."""
    from datachart.themes import score_palette

    score = score_palette(report_figures.SERIES, "#fbfcfd")
    assert score.verdict == "pass", str(score)
    assert score.low_contrast == 0, str(score)


def test_save_figure_takes_a_figure_or_a_builder(tmp_path: Path) -> None:
    """Both call shapes write a dated-free SVG in the report's font stack."""
    from datachart.charts import BarChart

    def build():
        return BarChart([{"label": "a", "y": 1}, {"label": "b", "y": 2}])

    for name, figure in (("built.svg", build()), ("builder.svg", build)):
        report_figures.save_figure(figure, tmp_path / name)
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert "<dc:date>" not in text
        assert "Instrument Sans" in text
