"""Plotly template carrying the SLM4IE chart look.

Registers a named plotly template (`slm4ie`) that encodes the project's
publication-ready style: white background, a light gray grid with an emphasized
zero baseline, serif bold centered titles and serif axis labels, sans tick and
hover text, moderate line and marker weights, and thin dark bar outlines.

Call `register_theme()` once (e.g. at the top of a notebook) to register the
template and make it the plotly default; every figure created afterwards inherits
the look.
"""

from typing import Dict

import plotly.graph_objects as go
import plotly.io as pio

from slm4ie.viz.palette import ACCENT, CATEGORICAL

TEMPLATE_NAME = "slm4ie"

_SERIF = "Georgia, 'Times New Roman', Times, serif"
_SANS = "'Helvetica Neue', Helvetica, Arial, sans-serif"
_INK = "#1A1A1A"
_GRID = "#E6E6E6"
_ZERO = "#9AA0A6"
_AXIS_LINE = "#CCCCCC"


def _axis() -> Dict[str, object]:
    """Build the shared axis styling for the template.

    Returns:
        Axis layout options: light gray grid, emphasized zero baseline, outside
        ticks in the sans face, and serif axis titles.
    """
    return dict(
        showgrid=True,
        gridcolor=_GRID,
        gridwidth=1,
        zeroline=True,
        zerolinecolor=_ZERO,
        zerolinewidth=1.5,
        linecolor=_AXIS_LINE,
        ticks="outside",
        tickfont=dict(family=_SANS, size=11, color=_INK),
        title=dict(font=dict(family=_SERIF, size=14, color=_INK)),
    )


def build_template() -> go.layout.Template:
    """Construct the SLM4IE plotly template.

    Returns:
        A plotly template with the project's fonts, background, grid, color
        sequence, and default trace weights.
    """
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=_SANS, size=12, color=_INK),
        title=dict(font=dict(family=_SERIF, size=18, color=_INK), x=0.5, xanchor="center"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=list(CATEGORICAL),
        xaxis=_axis(),
        yaxis=_axis(),
        hoverlabel=dict(font=dict(family=_SANS, size=12)),
    )
    template.data.scatter = [go.Scatter(line=dict(width=2.5), marker=dict(size=7))]
    template.data.bar = [go.Bar(marker=dict(line=dict(color=ACCENT["emphasis"], width=1)))]
    return template


def register_theme(set_default: bool = True) -> str:
    """Register the SLM4IE template with plotly.

    Args:
        set_default: Also make the template the plotly default so new figures
            inherit it without an explicit `template=` argument.

    Returns:
        The registered template name.
    """
    pio.templates[TEMPLATE_NAME] = build_template()
    if set_default:
        pio.templates.default = TEMPLATE_NAME
    return TEMPLATE_NAME
