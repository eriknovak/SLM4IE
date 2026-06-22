"""Tests for static and interactive chart export."""

import pytest

pytest.importorskip("plotly")

import plotly.graph_objects as go  # noqa: E402  (after importorskip)
import plotly.io as pio  # noqa: E402

from slm4ie.viz import export  # noqa: E402


def test_interactive_config_sets_format_and_scale():
    """The config carries the requested download format and scale."""
    cfg = export.interactive_config("svg", scale=2.0)
    assert cfg["toImageButtonOptions"] == {"format": "svg", "scale": 2.0}
    assert cfg["displaylogo"] is False


def test_enable_interactive_download_merges_renderer_config():
    """Enabling download writes the modebar options onto the default renderer."""
    name = pio.renderers.default
    if not name or name not in pio.renderers:
        name = "browser"
    export.enable_interactive_download("svg", scale=3.0)
    assert pio.renderers[name].config["toImageButtonOptions"]["format"] == "svg"


def test_save_figure_writes_files(tmp_path):
    """save_figure produces non-empty PNG and SVG files."""
    pytest.importorskip("kaleido")
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
    png = export.save_figure(fig, tmp_path / "chart.png", scale=2.0, width=400, height=300)
    svg = export.save_figure(fig, tmp_path / "chart.svg", width=400, height=300)
    assert png.exists() and png.stat().st_size > 0
    assert svg.exists() and svg.stat().st_size > 0
