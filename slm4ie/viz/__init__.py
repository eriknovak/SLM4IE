"""Shared chart styling for SLM4IE.

Exposes the project's plotly theme and color system so notebooks (and any future
publication-export scripts) render a single coherent, colorblind-safe look. Call
`register_theme()` once, then color categorical series with `categorical_colors`
and continuous or signed data with `SEQUENTIAL` / `DIVERGING`.

The `charts` subpackage is the reserved home for builders of chart types plotly
does not provide natively (bump plots, Pareto-front scatters, waffle charts);
they are added there as concrete charts need them.
"""

from slm4ie.viz.export import (
    enable_interactive_download,
    interactive_config,
    save_figure,
)
from slm4ie.viz.palette import (
    ACCENT,
    CATEGORICAL,
    DIVERGING,
    RAMP,
    SEQUENTIAL,
    categorical_colors,
)
from slm4ie.viz.theme import TEMPLATE_NAME, build_template, register_theme

__all__ = [
    "ACCENT",
    "CATEGORICAL",
    "DIVERGING",
    "RAMP",
    "SEQUENTIAL",
    "TEMPLATE_NAME",
    "build_template",
    "categorical_colors",
    "enable_interactive_download",
    "interactive_config",
    "register_theme",
    "save_figure",
]
