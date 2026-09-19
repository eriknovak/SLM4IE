"""Report figure styling: every figure is drawn twice, light and dark.

`build_report.py` shows `<stem>.svg` to a light-theme reader and
`<stem>.dark.svg` to a dark-theme one, so a figure is never inverted or set on
a plate to survive the other theme. Both variants are the same chart with the
same series in the same order; only the ink and the step of each hue change.

An `analysis.py` imports `save_figure` from here and hands it a callable that
builds the chart, because datachart bakes the theme in at build time.
"""

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Generator, List, Union

import matplotlib
from datachart.config import config
from datachart.themes import INK_THEME
from datachart.utils import save_figure as save_datachart_figure

# Text stays text rather than glyph outlines: smaller SVGs, selectable labels,
# and the report's own font face resolves at render time.
matplotlib.rcParams["svg.fonttype"] = "none"

LIGHT = "report-light"
DARK = "report-dark"

# Categorical slots in fixed order: the same eight hues, each stepped for its
# own surface, validated as a set against both report surfaces.
SERIES_LIGHT: List[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK: List[str] = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

# Ink and furniture, matching the report's own text and rule colors per theme.
INK: Dict[str, Dict[str, str]] = {
    LIGHT: {"fg": "#1f252f", "muted": "#5a6472", "grid": "#c8ced7", "surface": "#e8ebf0"},
    DARK: {"fg": "#e6e8ee", "muted": "#8b93a3", "grid": "#2a2f38", "surface": "#191c22"},
}

# The report renders figure text in its own face. datachart drops font names
# that are not installed on the machine drawing the figure, so the stack is
# written into the saved SVG instead, where the reader's browser resolves it.
FONT_STACK = "'Instrument Sans', 'Helvetica', 'Arial', sans-serif"
FONT_FAMILY = re.compile(r"font-family: [^;\"]+")


def _theme(mode: str) -> Dict:
    """Returns the datachart theme for one report mode, built off the ink theme.

    Args:
        mode (str): Either `LIGHT` or `DARK`.

    Returns:
        Dict: The theme attributes to register.
    """
    ink = INK[mode]
    return dict(
        INK_THEME,
        color_general_multiple=SERIES_LIGHT if mode == LIGHT else SERIES_DARK,
        font_general_size=12,
        font_general_color=ink["fg"],
        font_title_color=ink["fg"],
        font_subtitle_color=ink["fg"],
        font_xlabel_color=ink["muted"],
        font_ylabel_color=ink["muted"],
        font_xlabel_size=11,
        font_ylabel_size=11,
        axes_ticks_label_size=10,
        # the frame keeps the legend readable where it overlaps the data; its
        # colours come from rcParams below, since datachart passes neither
        plot_legend_frameon=True,
        plot_legend_font_size=10,
        plot_legend_label_color=ink["fg"],
        plot_grid_color=ink["grid"],
        plot_bar_edge_color=ink["surface"],
        plot_hist_edge_color=ink["surface"],
        plot_scatter_edge_color=ink["surface"],
        plot_vline_color=ink["muted"],
        plot_hline_color=ink["muted"],
        plot_bar_error_color=ink["muted"],
        plot_bar_value_color=ink["fg"],
        plot_text_color=ink["fg"],
        plot_text_box_facecolor="none",
        plot_text_box_edgecolor=ink["grid"],
        plot_text_arrow_color=ink["muted"],
        plot_heatmap_font_color=ink["fg"],
        plot_heatmap_frame_color=ink["muted"],
        plot_heatmap_edge_color=ink["surface"],
    )


def _rc(mode: str) -> Dict[str, object]:
    """Returns what matplotlib owns rather than the datachart theme.

    The legend frame is one of these: datachart passes matplotlib only `frameon`,
    `loc`, the two font sizes, `alignment`, `shadow` and `labelcolor`, so its face
    and edge come from here. `text.color` covers the legend's own "Legend" title,
    the one string datachart does not set a fill on.

    Args:
        mode (str): Either `LIGHT` or `DARK`.

    Returns:
        Dict[str, object]: rcParams to apply while the chart is built.
    """
    ink = INK[mode]
    return {
        "axes.edgecolor": ink["muted"],
        "xtick.color": ink["muted"],
        "ytick.color": ink["muted"],
        "text.color": ink["fg"],
        "legend.facecolor": ink["surface"],
        "legend.edgecolor": ink["grid"],
        "legend.framealpha": 0.92,
    }


@contextmanager
def _applied(mode: str) -> Generator[None, None, None]:
    """Applies one report theme for the duration of a chart build.

    Args:
        mode (str): Either `LIGHT` or `DARK`.

    Yields:
        None: With the theme and rcParams in force.
    """
    config.register_theme(mode, _theme(mode))
    config.set_theme(mode)
    before = {key: matplotlib.rcParams[key] for key in _rc(mode)}
    matplotlib.rcParams.update(_rc(mode))
    try:
        yield
    finally:
        matplotlib.rcParams.update(before)


def save_figure(build: Callable[[], object], path: Union[str, Path]) -> None:
    """Builds and writes a figure twice, `<stem>.svg` and `<stem>.dark.svg`.

    Both are transparent: the report page, not the figure, supplies the
    background. `build` is a callable rather than a figure because datachart
    reads the theme when the chart is constructed, so each variant needs its
    own build.

    Args:
        build (Callable[[], object]): Zero-argument callable returning the figure.
        path (Union[str, Path]): Destination of the light variant; the dark one
            is written beside it as `<stem>.dark<suffix>`.

    Raises:
        ValueError: If `path` has no suffix to derive the dark variant from.
    """
    target = Path(path)
    if not target.suffix:
        raise ValueError(f"figure path needs a suffix: {target}")
    for mode, out in ((LIGHT, target), (DARK, target.with_name(f"{target.stem}.dark{target.suffix}"))):
        with _applied(mode):
            save_datachart_figure(build(), str(out), transparent=True)
        if out.suffix.lower() == ".svg":
            out.write_text(
                FONT_FAMILY.sub(f"font-family: {FONT_STACK}", out.read_text(encoding="utf-8")), encoding="utf-8"
            )
