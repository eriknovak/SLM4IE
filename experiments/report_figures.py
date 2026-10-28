"""The one way an analysis writes a figure, drawn in the project's theme.

The `slm4ie` theme is a module of the datachart-themes package, and this
project names it as datachart's session default: `pyproject.toml` lists it in
the `datachart.default_theme` entry-point group, so datachart starts in it on
import and every chart an `analysis.py` builds draws in the project colours
without a `set_theme` call. Importing this module checks that it did, and
`save_figure` then writes the figure as a transparent SVG for the report's
light plate. `build_report.py` sets every figure on that plate in both of its
themes, so no dark variant is drawn and nothing is ever inverted.
"""

import re
from collections.abc import Callable
from pathlib import Path

import matplotlib
from datachart.config import config
from datachart.utils import save_figure as save_datachart_figure

# ===============================
# Defaults
# ===============================

THEME = "slm4ie"

# the report's font stack, written into the SVG because datachart drops fonts not installed here
FONT_STACK = "'Instrument Sans', 'Helvetica', 'Arial', sans-serif"
FONT_FAMILY = re.compile(r"font-family: [^;\"]+")
# drop the save time matplotlib stamps into the SVG, so reruns leave the file unchanged
SVG_DATE = re.compile(r"\s*<dc:date>[^<]*</dc:date>")


# ===============================
# Check theme
# ===============================

# fail unless datachart started in the project theme
if config.theme != THEME:
    raise RuntimeError(
        f"datachart started in the {config.theme!r} theme, not {THEME!r}: "
        "install the analysis group (`uv sync --group analysis`) so datachart-themes is present"
    )

# keep text as text so the report's font face resolves at render
matplotlib.rcParams["svg.fonttype"] = "none"
# fixed salt keeps clip-path and marker ids stable across reruns
matplotlib.rcParams["svg.hashsalt"] = THEME

# ===============================
# Main function
# ===============================


def save_figure(figure: object | Callable[[], object], path: str | Path) -> None:
    """Writes one figure, transparent, drawn for the report's light plate.

    Args:
        figure: The datachart figure, or a zero-argument callable that builds
            it; either draws in the project theme, which datachart starts in.
        path: Destination of the figure.

    Raises:
        ValueError: If `path` has no suffix.
    """
    target = Path(path)
    if not target.suffix:
        raise ValueError(f"figure path needs a suffix: {target}")

    # build the figure if given a builder, then save it
    if callable(figure):
        figure = figure()
    save_datachart_figure(figure, str(target), transparent=True)

    # pin the font stack and drop the save time in SVGs
    if target.suffix.lower() == ".svg":
        text = target.read_text(encoding="utf-8")
        text = SVG_DATE.sub("", FONT_FAMILY.sub(f"font-family: {FONT_STACK}", text))
        target.write_text(text, encoding="utf-8")
