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
from pathlib import Path
from typing import Callable, Union

import matplotlib
from datachart.config import config
from datachart.utils import save_figure as save_datachart_figure

THEME = "slm4ie"

# The report renders figure text in its own face. datachart drops font names
# that are not installed on the machine drawing the figure, so the stack is
# written into the saved SVG instead, where the reader's browser resolves it.
FONT_STACK = "'Instrument Sans', 'Helvetica', 'Arial', sans-serif"
FONT_FAMILY = re.compile(r"font-family: [^;\"]+")
# matplotlib stamps the save time into the SVG; without it a rerun on unchanged
# tables leaves the committed file untouched
SVG_DATE = re.compile(r"\s*<dc:date>[^<]*</dc:date>")

if config.theme != THEME:
    raise RuntimeError(
        f"datachart started in the {config.theme!r} theme, not {THEME!r}: "
        "install the analysis group (`uv sync --group analysis`) so datachart-themes is present"
    )
# text stays text rather than glyph outlines: smaller SVGs, selectable labels,
# and the report's font face resolves at render
matplotlib.rcParams["svg.fonttype"] = "none"
# a fixed salt keeps clip-path and marker ids stable, so a rerun on unchanged
# tables leaves the committed file untouched
matplotlib.rcParams["svg.hashsalt"] = THEME


def save_figure(figure: Union[object, Callable[[], object]], path: Union[str, Path]) -> None:
    """Writes one figure, transparent, drawn for the report's light plate.

    Args:
        figure (Union[object, Callable[[], object]]): The datachart figure, or a
            zero-argument callable that builds it; either draws in the project
            theme, which datachart starts in.
        path (Union[str, Path]): Destination of the figure.

    Raises:
        ValueError: If `path` has no suffix.
    """
    target = Path(path)
    if not target.suffix:
        raise ValueError(f"figure path needs a suffix: {target}")
    if callable(figure):
        figure = figure()
    save_datachart_figure(figure, str(target), transparent=True)
    if target.suffix.lower() == ".svg":
        text = target.read_text(encoding="utf-8")
        text = SVG_DATE.sub("", FONT_FAMILY.sub(f"font-family: {FONT_STACK}", text))
        target.write_text(text, encoding="utf-8")
