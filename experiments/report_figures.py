"""The project's chart theme and the one way an analysis writes a figure.

Importing this module registers the `slm4ie` datachart theme and makes it the
active one, so every chart an `analysis.py` builds afterwards draws in the
project colours; `save_figure` then writes it as a transparent SVG for the
report's light plate. `build_report.py` sets every figure on that plate in
both of its themes, so no dark variant is drawn and nothing is ever inverted.

The series palette takes its hues from the SLM4IE site (the banner's blue and
teal, the publication families' rust and purple) with an amber accent and the
page ink; the accents were moved in lightness and chroma, hue held, until every
pair passes datachart's colour-blindness gate. The value ramps are the banner
gradient made monotonic in lightness, and site rust against site blue.
"""

import re
from pathlib import Path
from typing import Callable, Dict, List, Union

import matplotlib
from datachart.config import config
from datachart.constants import TRAIT
from datachart.themes import INK_THEME, derive_theme
from datachart.utils import save_figure as save_datachart_figure

THEME = "slm4ie"

# Categorical slots in fixed order. Every pair passes `score_palette` on the
# report's plate: OKLab dE >= 8 under deuteranopia and protanopia, >= 15 for
# normal vision, and every colour reaches 3:1.
SERIES: List[str] = [
    "#1b4dad",  # site blue
    "#ac4b2d",  # site rust
    "#15a2a3",  # site teal
    "#8760ae",  # site purple, lifted clear of the blue for protan readers
    "#be8928",  # amber accent
    "#2b2f3a",  # page ink
]

# Pale aqua through the banner's teal and blue to its navy; lightness falls monotonically.
SEQUENTIAL: List[str] = ["#e6f4f4", "#5cc8c8", "#15a3a3", "#1b4dad", "#0d2a63"]

# Site rust against site blue through a neutral, for centred heatmaps.
DIVERGING: List[str] = ["#7a3418", "#b0542a", "#eec9b4", "#f2f2f2", "#a9c2ea", "#1b4dad", "#0d2a63"]

# Ink and furniture, matching the report's own text and rule colours on the light plate.
INK: Dict[str, str] = {
    "fg": "#1f252f",
    "muted": "#5a6472",
    "grid": "#c8ced7",
    "surface": "#e8ebf0",
}

# The report renders figure text in its own face. datachart drops font names
# that are not installed on the machine drawing the figure, so the stack is
# written into the saved SVG instead, where the reader's browser resolves it.
FONT_STACK = "'Instrument Sans', 'Helvetica', 'Arial', sans-serif"
FONT_FAMILY = re.compile(r"font-family: [^;\"]+")
# matplotlib stamps the save time into the SVG; without it a rerun on unchanged
# tables leaves the committed file untouched
SVG_DATE = re.compile(r"\s*<dc:date>[^<]*</dc:date>")


def _theme() -> Dict:
    """Returns the project theme: the ink theme's furniture with the site palette.

    Returns:
        Dict: The theme attributes to register.
    """
    return derive_theme(
        INK_THEME,
        lead=SERIES,
        # line style and marker cycles carry series identity beside colour
        traits=[TRAIT.PATTERNED],
        color_general_singular=SEQUENTIAL,
        plot_heatmap_cmap=SEQUENTIAL,
        plot_heatmap_cmap_diverging=DIVERGING,
        color_parallel_hue_continuous=SEQUENTIAL[1:],
        plot_dumbbell_start_color=SEQUENTIAL[1],
        plot_dumbbell_end_color=SEQUENTIAL[3],
        font_general_size=12,
        font_general_color=INK["fg"],
        font_title_color=INK["fg"],
        font_subtitle_color=INK["fg"],
        font_xlabel_color=INK["muted"],
        font_ylabel_color=INK["muted"],
        font_xlabel_size=11,
        font_ylabel_size=11,
        axes_ticks_label_size=10,
        # the frame keeps the legend readable where it overlaps the data; its
        # colours come from rcParams below, since datachart passes neither
        plot_legend_frameon=True,
        plot_legend_font_size=10,
        plot_legend_label_color=INK["fg"],
        plot_grid_color=INK["grid"],
        # a hairline in ink on every bar; a highlighted bar doubles it to a clear outline
        plot_bar_edge_color=INK["fg"],
        plot_bar_edge_width=0.75,
        plot_hist_edge_color=INK["surface"],
        plot_scatter_edge_color=INK["surface"],
        plot_vline_color=INK["muted"],
        plot_hline_color=INK["muted"],
        plot_bar_error_color=INK["muted"],
        plot_value_color=INK["fg"],
        plot_text_color=INK["fg"],
        plot_text_box_facecolor="none",
        plot_text_box_edgecolor=INK["grid"],
        plot_text_arrow_color=INK["muted"],
        plot_heatmap_font_color=INK["fg"],
        plot_heatmap_frame_color=INK["muted"],
        plot_heatmap_edge_color=INK["surface"],
    )


# What matplotlib owns rather than the datachart theme. The legend frame is one
# of these: datachart passes matplotlib only `frameon`, `loc`, the two font
# sizes, `alignment`, `shadow` and `labelcolor`, so its face and edge come from
# here. `text.color` covers the legend's own "Legend" title, the one string
# datachart does not set a fill on. Text stays text rather than glyph outlines:
# smaller SVGs, selectable labels, and the report's font face resolves at render.
RC: Dict[str, object] = {
    "svg.fonttype": "none",
    "axes.edgecolor": INK["muted"],
    "xtick.color": INK["muted"],
    "ytick.color": INK["muted"],
    "text.color": INK["fg"],
    "legend.facecolor": INK["surface"],
    "legend.edgecolor": INK["grid"],
    "legend.framealpha": 0.92,
}

config.register_theme(THEME, _theme())
config.set_theme(THEME)
matplotlib.rcParams.update(RC)


def save_figure(figure: Union[object, Callable[[], object]], path: Union[str, Path]) -> None:
    """Writes one figure, transparent, drawn for the report's light plate.

    Args:
        figure (Union[object, Callable[[], object]]): The datachart figure, or a
            zero-argument callable that builds it; either draws in the project
            theme, which is active from the moment this module is imported.
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
