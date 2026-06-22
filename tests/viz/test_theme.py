"""Tests for the SLM4IE plotly template."""

import pytest

pytest.importorskip("plotly")

import plotly.io as pio  # noqa: E402  (after importorskip)

from slm4ie.viz import palette, theme  # noqa: E402


def test_build_template_carries_palette_and_background():
    """The template uses the categorical colorway on a white background."""
    template = theme.build_template()
    assert list(template.layout.colorway) == list(palette.CATEGORICAL)
    assert template.layout.plot_bgcolor == "white"
    assert template.layout.paper_bgcolor == "white"


def test_build_template_sets_trace_weights():
    """Scatter and bar traces get the standard line weight and bar outline."""
    template = theme.build_template()
    assert template.data.scatter[0].line.width == 2.5
    assert template.data.bar[0].marker.line.color == palette.ACCENT["emphasis"]


def test_register_theme_registers_and_defaults():
    """Registering the theme adds it by name and makes it the default."""
    name = theme.register_theme()
    assert name == theme.TEMPLATE_NAME
    assert theme.TEMPLATE_NAME in pio.templates
    assert pio.templates.default == theme.TEMPLATE_NAME


def test_register_theme_can_skip_default():
    """Passing set_default=False registers without changing the default."""
    pio.templates.default = "plotly"
    theme.register_theme(set_default=False)
    assert theme.TEMPLATE_NAME in pio.templates
    assert pio.templates.default == "plotly"
