"""Report figures and tables: each checked against its rules before it is written.

The `slm4ie` theme is a module of the datachart-themes package, and this
project names it as datachart's session default: `pyproject.toml` lists it in
the `datachart.default_theme` entry-point group, so datachart starts in it on
import and every chart an `analysis.py` builds draws in the project colours
without a `set_theme` call. Importing this module checks that it did.

`save_figure` measures the drawn layout, refuses to write a figure that breaks
a `FIG-nnn` rule of labflow's `figures.md`, and writes a passing figure as a
transparent SVG stamped with the check version, the rules run and the claim.
Text sizes are checked against the project theme's own sizes, not the skill's
font table. `build_report.py` sets every figure on a light plate in both of its
themes, so no dark variant is drawn and nothing is ever inverted.

`save_table` writes a CSV the same way: it refuses a table that breaks a
`TAB-nnn` rule of labflow's `tables.md`. `check_table` runs the same rules on a
table already written. Both live in `table_rules.py` beside this file, which
uses the standard library only, and are re-exported here; the two files travel
together. `half_width` turns an asymmetric interval into the symmetric error
bar datachart draws.

Every figure takes its size from here, one constant per place on the record
page, passed as the chart's `figsize`; never a literal or a datachart
`FIG_SIZE`. The page picks the column from the shape, so a 6.3-inch figure
under 1.8:1 (`FIG_SIZE.FULL_MEDIUM`, `FULL_TALL`) lands in the 374 px half
column and the theme's 8-point text renders under the 9-point floor (FIG-005).

    | Constant      | Size (in)  | Column, and how it gets there        | 8 pt renders at |
    | ------------- | ---------- | ------------------------------------ | --------------- |
    | `ONE_COLUMN`  | 6.3 x 3.4  | one-column: 1.8:1 or wider, default  | 10.4 pt         |
    | `HALF_SQUARE` | 3.0 x 3.0  | two-column: narrower than 1.8:1      | 10.4 pt         |
    | `HALF_TALL`   | 3.0 x 7.2  | two-column: a long list of bars      | 10.4 pt         |
    | `FULL_WIDTH`  | 8.9 x 4.9  | full-width: `width="full-width"`     | 9.0 pt          |
"""

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from html import unescape
from itertools import combinations, pairwise
from math import cos, hypot, radians, sin
from pathlib import Path
from typing import NamedTuple
from xml.sax.saxutils import escape

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datachart.config import config
from datachart.utils import save_figure as save_datachart_figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from table_rules import (  # noqa: F401  — re-exported, so an analysis.py imports save_table from here
    CLAUSE,
    RESULT_STEM,
    TABLE_RULES,
    TABLES_FOLDER,
    Breach,
    TableCheckError,
    check_table,
    save_table,
)

# ===============================
# Defaults
# ===============================

THEME = "slm4ie"

# the report's font stack, written into the SVG because datachart drops fonts not installed here
FONT_STACK = "'Instrument Sans', 'Helvetica', 'Arial', sans-serif"
FONT_FAMILY = re.compile(r"font-family: [^;\"]+")
# drop the save time matplotlib stamps into the SVG, so reruns leave the file unchanged
SVG_DATE = re.compile(r"\s*<dc:date>[^<]*</dc:date>")

# the datachart style keys that size each text element; FIG-004 reads their
# values from the project theme in force
FONT_SIZES: dict[str, tuple[str, ...]] = {
    "title": ("font_title_size",),
    "panel title": ("font_subtitle_size",),
    "axis label": ("font_xlabel_size", "font_ylabel_size", "plot_parallel_dim_label_size"),
    "tick label": ("axes_ticks_label_size", "plot_parallel_tick_label_size"),
    "legend text": ("plot_legend_font_size",),
    "legend title": ("plot_legend_title_size",),
    "value label": (
        "plot_value_fontsize",
        "plot_heatmap_font_size",
        "plot_calendar_heatmap_font_size",
        "plot_contour_label_font_size",
    ),
    "annotation": ("plot_text_size",),
    "colorbar label": ("font_ylabel_size",),
}

# a text on the axes is a value label or an annotation; either size is its table size
SIZE_ALTERNATIVES = {"annotation": ("annotation", "value label")}
# datachart sizes a treemap tile's label to fit the tile, so FIG-004 skips it
TILE_LABEL = "tile label"
TREEMAP_TILE_GID = "tile:"

# drawn width in CSS pixels of each place a figure sits on a record page, measured
# in a 1440 px wide window; figures.md has the table and how it was measured
COLUMN_WIDTHS_PX: dict[str, int] = {
    "one-column": 785,
    "two-column": 374,
    "full-width": 963,
}
# a figure whose width over height is under this takes half the row, as the record
# page sets it (labflow_report/record.py NARROW_FIGURE; a test keeps the two equal)
NARROW_FIGURE = 1.8
CSS_PX_PER_INCH = 96
# the figure sizes that pass the floor, one per place; the module docstring has the table
ONE_COLUMN = (6.3, 3.4)
HALF_SQUARE = (3.0, 3.0)
HALF_TALL = (3.0, 7.2)
FULL_WIDTH = (8.9, 4.9)
RENDERED_FLOOR_PT = 9.0
MIN_TICK_GAP_PT = 2.0
TITLE_WORDS = (2, 5)

# the stamp is read by the report builder; raise the version when a rule changes
CHECK_VERSION = 5
STAMP_ID = "labflow-figure-check"
STAMP = re.compile(rf'<metadata id="{STAMP_ID}">(.*?)</metadata>', re.DOTALL)
SVG_OPEN = re.compile(r"<svg\b[^>]*>")

RULES: dict[str, str] = {
    "FIG-001": "text against text",
    "FIG-002": "legend placement",
    "FIG-003": "clipping",
    "FIG-004": "font size",
    "FIG-005": "rendered size floor",
    "FIG-006": "crowded ticks",
    "FIG-007": "axis labels",
    "FIG-008": "axis units",
    "FIG-009": "title length",
    "FIG-010": "series names",
    "FIG-011": "claim arm emphasised",
    "FIG-012": "claim threshold line",
    "FIG-013": "bars from zero",
    "FIG-014": "file name",
    "FIG-015": "data table",
    "FIG-016": "dataset figure claim",
}

UNIT = re.compile(r"\(.+\)|\[.+\]")
# a coefficient or a fraction is its own unit while its ticks stay within -1 to 1
UNITLESS = re.compile(
    r"\b(rho|tau(-b)?|kappa|f1|auc|correlation|coefficient|share|fraction|proportion)\b", re.IGNORECASE
)
UNITLESS_SPAN = 1.0
DEFAULT_SERIES = re.compile(
    r"^(_?child\s*\d+|_?line\s*\d+|series\s*\d*|data\s*\d*|set\s*\d+|y\d*|none|nan|legend)$",
    re.IGNORECASE,
)
COLORBAR_AXES = "<colorbar>"
OVERLAP_PX = 0.5
EDGE_PX = 1.0
SALIENCE_EPS = 1e-6

# a figure is `<stem>-<quantity>-by-<dimension>` (RESULT_STEM) in figures/
FIGURES_FOLDER = "figures"
DATASET_PREFIX = "dataset-"

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
# Main functions
# ===============================


@dataclass(frozen=True)
class Claim:
    """The claim a figure shows, as the record names it.

    Examples:
        >>> Claim(clause="C1", quantity="F1", arms=("ours",), threshold=0.75)

    Attributes:
        clause (str): The clause id the figure decides, `H<n>`, `C<n>` or `Q<n>`.
        quantity (str): The quantity the clause reads, as the record names it.
        arms (tuple[str, ...]): The arms or series the clause names, as the figure labels them.
        threshold (float | None): The clause's bar or baseline, in the quantity's units.
    """

    clause: str
    quantity: str
    arms: tuple[str, ...] = ()
    threshold: float | None = None

    def __post_init__(self) -> None:
        """Checks the fields and freezes the arms as a tuple."""
        if not isinstance(self.clause, str) or not CLAUSE.match(self.clause):
            raise ValueError(f"claim clause must look like H1, C2 or Q3, got {self.clause!r}")
        if not isinstance(self.quantity, str) or not self.quantity.strip():
            raise ValueError(f"claim quantity must name the quantity, got {self.quantity!r}")
        arms = tuple(self.arms)
        if not all(isinstance(arm, str) and arm.strip() for arm in arms):
            raise ValueError(f"claim arms must be non-empty names, got {arms!r}")
        object.__setattr__(self, "arms", arms)
        if self.threshold is not None:
            object.__setattr__(self, "threshold", float(self.threshold))


class FigureCheckError(ValueError):
    """Raised when a figure breaks a rule; the figure is not written.

    Attributes:
        path (Path): The file that was not written.
        breaches (tuple[Breach, ...]): The rules broken, in rule order.
    """

    def __init__(self, path: Path, breaches: list[Breach]) -> None:
        """Keeps the path and the breaches and lists them in the message."""
        self.path = path
        self.breaches = tuple(breaches)
        lines = "\n".join(f"  {breach}" for breach in self.breaches)
        super().__init__(f"figure {path} not written, {len(self.breaches)} breaches (figures.md):\n{lines}")


def save_figure(
    figure: object | Callable[[], object],
    path: str | Path,
    *,
    claim: Claim | None = None,
    width: str = "one-column",
) -> None:
    """Checks and writes one figure, transparent, drawn for the report's light plate.

    The drawn layout is checked against the `FIG-nnn` rules of figures.md
    before anything is written. A passing SVG carries the report's font stack,
    no save time, and a stamp with the check version, the rules run and the
    claim.

    Examples:
        >>> claim = Claim("C1", "F1", arms=("ours",), threshold=0.75)
        >>> save_figure(lambda: BarChart(...), "figures/f1-by-arm.svg", claim=claim)

    Args:
        figure: The datachart figure, or a zero-argument callable that builds
            it; either draws in the project theme, which datachart starts in.
        path: Destination of the figure.
        claim: The claim the figure shows; `None` for a figure no finding rests on.
        width: Where the report places a figure 1.8:1 or wider: `one-column`
            or `full-width`. A narrower figure is checked at `two-column`
            whatever this says, and a wider one passed `two-column` at
            `one-column`, as the record page places them.

    Raises:
        ValueError: If `path` has no suffix or `width` is not a report column.
        TypeError: If `claim` is not a `Claim`.
        FigureCheckError: If the figure breaks a rule; nothing is written.
    """
    target = Path(path)
    if not target.suffix:
        raise ValueError(f"figure path needs a suffix: {target}")
    if width not in COLUMN_WIDTHS_PX:
        raise ValueError(f"figure width must be one of {', '.join(COLUMN_WIDTHS_PX)}, got {width!r}")
    if claim is not None and not isinstance(claim, Claim):
        raise TypeError(f"claim must be a Claim, got {type(claim).__name__}")

    # build the figure if given a builder, then check and save it
    built = callable(figure)
    if built:
        figure = figure()
    try:
        column = _column(figure, width)
        rules, breaches = _check(figure, target, claim, column)
        if breaches:
            raise FigureCheckError(target, breaches)
        save_datachart_figure(figure, str(target), transparent=True)
    finally:
        # a figure handed in stays the caller's to close
        if built:
            plt.close(figure)

    # pin the font stack, drop the save time and stamp the svg
    if target.suffix.lower() == ".svg":
        svg = target.read_text(encoding="utf-8")
        svg = SVG_DATE.sub("", FONT_FAMILY.sub(f"font-family: {FONT_STACK}", svg))
        target.write_text(_stamped(svg, rules, claim, column), encoding="utf-8")


def read_stamp(path: str | Path) -> dict | None:
    """Reads the check stamp `save_figure` wrote into an SVG.

    Args:
        path: The SVG file.

    Returns:
        The stamp, with `check_version`, `rules`, `claim` and `width`, or `None`
        when the file carries none.
    """
    match = STAMP.search(Path(path).read_text(encoding="utf-8"))
    return json.loads(unescape(match.group(1))) if match else None


def half_width(value: float, low: float, high: float) -> float:
    """Returns the wider side of an interval: datachart draws symmetric error bars only.

    Examples:
        >>> half_width(0.5, 0.25, 1.0)
        0.5

    Args:
        value: The point estimate the bar shows.
        low: The lower end of the interval.
        high: The upper end of the interval.

    Returns:
        The distance to whichever end lies further from `value`.
    """
    return float(max(value - low, high - value))


# ===============================
# Check helpers
# ===============================


def _column(figure, width: str) -> str:
    """The report column the record page sets the figure in, read from its shape.

    Args:
        figure: The matplotlib figure.
        width (str): The column the caller passed.

    Returns:
        str: `two-column` for a figure under NARROW_FIGURE; else `full-width`
        when passed, `one-column` otherwise.
    """
    if figure.get_figwidth() / figure.get_figheight() < NARROW_FIGURE:
        return "two-column"
    return "full-width" if width == "full-width" else "one-column"


def _stamped(svg: str, rules: list[str], claim: Claim | None, width: str) -> str:
    """Returns the SVG with the check stamp as its first child.

    Args:
        svg (str): The written SVG.
        rules (list[str]): The rule ids the check ran.
        claim (Claim | None): The claim the figure shows.
        width (str): The report column the figure was checked for.

    Returns:
        str: The stamped SVG.
    """
    stamp = {
        "check_version": CHECK_VERSION,
        "rules": rules,
        "claim": asdict(claim) if claim else None,
        "width": width,
    }
    block = f'<metadata id="{STAMP_ID}">{escape(json.dumps(stamp, ensure_ascii=False))}</metadata>'
    return SVG_OPEN.sub(lambda match: match.group(0) + block, svg, count=1)


class _Item(NamedTuple):
    """One measured text or legend box: what it is, its words and where it sits."""

    kind: str
    label: str
    size: float | None
    group: tuple[int, str] | None
    box: Bbox
    polygon: list[tuple[float, float]]


class _Mark(NamedTuple):
    """One data mark: its kind, its artist and the series or category names it carries."""

    kind: str
    artist: object
    names: frozenset


class _Layout(NamedTuple):
    """The measured figure every rule reads."""

    figure: object
    renderer: object
    axes: list[object]
    items: list[_Item]
    legends: list[object]
    width: str


def _check(figure, target: Path, claim: Claim | None, width: str) -> tuple[list[str], list[Breach]]:
    """Measures the drawn figure and runs every rule that applies.

    Args:
        figure: The matplotlib figure.
        target (Path): Where the figure is to be written.
        claim (Claim | None): The claim the figure shows.
        width (str): The report column the figure sits in.

    Returns:
        tuple[list[str], list[Breach]]: The rule ids run and the breaches found.
    """
    layout = _measure(figure, width)
    checks = [
        ("FIG-001", _text_overlaps),
        ("FIG-002", _legend_covers),
        ("FIG-003", _clipped),
        ("FIG-004", _off_size),
        ("FIG-005", _under_floor),
        ("FIG-006", _crowded_ticks),
        ("FIG-007", _unlabelled_axes),
        ("FIG-008", _unitless_axes),
        ("FIG-009", _title_length),
        ("FIG-010", _default_names),
        ("FIG-013", _bars_off_zero),
        ("FIG-014", lambda _: _figure_misnamed(target)),
        ("FIG-016", lambda _: _dataset_claim(target, claim)),
    ]
    if claim is not None:
        checks.append(("FIG-015", lambda _: _data_table_missing(target)))
    if claim is not None and claim.arms:
        checks.append(("FIG-011", lambda measured: _arms_not_emphasised(measured, claim.arms)))
    if claim is not None and claim.threshold is not None:
        checks.append(("FIG-012", lambda measured: _threshold_missing(measured, claim.threshold)))
    checks.sort()
    breaches = [Breach(rule, message) for rule, check in checks for message in check(layout)]
    return [rule for rule, _ in checks], breaches


def _measure(figure, width: str) -> _Layout:
    """Draws the figure once, layout engine included, and collects what the rules read."""
    # draw on an agg canvas so every extent is measured
    canvas = figure.canvas if hasattr(figure.canvas, "get_renderer") else FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()

    axes = [ax for ax in figure.axes if ax.get_visible() and ax.axison and ax.get_label() != COLORBAR_AXES]
    legends = [ax.get_legend() for ax in figure.axes if ax.get_legend() is not None] + list(figure.legends)
    items = _texts(figure, renderer, legends) + [
        _Item("legend", "legend", None, ("legend", id(legend)), box, _corners(box))
        for legend in legends
        if legend.get_visible()
        for box in [legend.get_window_extent(renderer)]
    ]
    return _Layout(figure, renderer, axes, items, legends, width)


def _texts(figure, renderer, legends: list) -> list[_Item]:
    """Every visible, non-empty text of the figure, classed by the font table's elements."""
    found: list[_Item] = []

    def add(kind: str, text: Text | None, group: tuple | None = None) -> None:
        if text is None or not text.get_visible() or not text.get_text().strip():
            return
        box = text.get_window_extent(renderer)
        found.append(
            _Item(
                kind,
                text.get_text(),
                text.get_fontsize(),
                group,
                box,
                _polygon(text, box, renderer),
            )
        )

    # figure-level title, shared axis labels and notes
    sup = {"_suptitle": "title", "_supxlabel": "axis label", "_supylabel": "axis label"}
    for name, kind in sup.items():
        add(kind, getattr(figure, name, None))
    sup_ids = {id(getattr(figure, name, None)) for name in sup}
    for text in figure.texts:
        if id(text) not in sup_ids:
            add("annotation", text)

    # per axes: panel title, axis labels, tick labels and texts on the data
    for index, ax in enumerate(figure.axes):
        if not ax.get_visible():
            continue
        label_kind = "colorbar label" if ax.get_label() == COLORBAR_AXES else "axis label"
        add("panel title", ax.title)
        add(label_kind, ax.xaxis.label)
        add(label_kind, ax.yaxis.label)
        if ax.axison:
            for key, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
                for label in _tick_labels(axis):
                    add("tick label", label, (index, key))
                add("tick label", axis.get_offset_text())
        is_treemap = any(str(patch.get_gid() or "").startswith(TREEMAP_TILE_GID) for patch in ax.patches)
        for text in ax.texts:
            add(TILE_LABEL if is_treemap else "annotation", text)

    # legend entries and titles
    for legend in legends:
        if legend.get_visible():
            for text in legend.get_texts():
                add("legend text", text, ("legend", id(legend)))
            add("legend title", legend.get_title(), ("legend", id(legend)))
    return found


def _tick_labels(axis) -> list[Text]:
    """The tick labels an axis draws: visible, non-empty, inside the view."""
    low, high = sorted(axis.get_view_interval())
    slack = (high - low) * 1e-9
    ticks = axis.get_major_ticks(len(axis.get_majorticklocs())) + axis.get_minor_ticks(len(axis.get_minorticklocs()))
    return [
        label
        for tick in ticks
        if low - slack <= tick.get_loc() <= high + slack
        for label in (tick.label1, tick.label2)
        if label.get_visible() and label.get_text().strip()
    ]


def _polygon(text: Text, box: Bbox, renderer) -> list[tuple[float, float]]:
    """The rotated rectangle a text covers, in display pixels."""
    angle = text.get_rotation()
    if abs(angle % 90) < 1e-6:
        return _corners(box)

    # the unrotated size, turned about the centre the rotated box shares
    text.set_rotation(0)
    flat = text.get_window_extent(renderer)
    text.set_rotation(angle)
    half_w, half_h = flat.width / 2, flat.height / 2
    cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    c, s = cos(radians(angle)), sin(radians(angle))
    return [
        (cx + dx * c - dy * s, cy + dx * s + dy * c)
        for dx, dy in (
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        )
    ]


def _corners(box: Bbox) -> list[tuple[float, float]]:
    return [(box.x0, box.y0), (box.x1, box.y0), (box.x1, box.y1), (box.x0, box.y1)]


def _overlap(p: list[tuple[float, float]], q: list[tuple[float, float]], margin: float) -> bool:
    """Whether two convex polygons overlap by more than `margin` pixels on every axis.

    A negative margin asks whether they come closer than that distance.
    """
    for polygon in (p, q):
        for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
            norm = hypot(x2 - x1, y2 - y1)
            if norm == 0:
                continue
            nx, ny = (y1 - y2) / norm, (x2 - x1) / norm
            on_p = [x * nx + y * ny for x, y in p]
            on_q = [x * nx + y * ny for x, y in q]
            if min(max(on_p), max(on_q)) - max(min(on_p), min(on_q)) <= margin:
                return False
    return True


def _short(text: str) -> str:
    text = " ".join(text.split())
    return repr(text if len(text) <= 30 else text[:29] + "…")


def _panel(layout: _Layout, ax) -> str:
    """How a message names one axes: the chart, or the panel by title or position."""
    if len(layout.axes) == 1:
        return "the chart"
    title = ax.get_title().strip()
    return f"panel {title!r}" if title else f"panel {layout.figure.axes.index(ax) + 1}"


def _axis_label(layout: _Layout, ax, key: str) -> str:
    """The label an axis carries: its own, else the figure's shared one."""
    own = (ax.get_xlabel() if key == "x" else ax.get_ylabel()).strip()
    shared = getattr(layout.figure, f"_sup{key}label", None)
    return own or (shared.get_text().strip() if shared is not None else "")


def _is_number(text: str) -> bool:
    plain = text.strip().replace("−", "-").replace(",", "").rstrip("%")
    if plain.startswith("$"):
        # mathtext, as on log axes
        return True
    try:
        float(plain)
    except ValueError:
        return False
    return True


def _text_overlaps(layout: _Layout) -> list[str]:
    messages = []
    for a, b in combinations(layout.items, 2):
        if a.group is not None and a.group == b.group:
            continue
        if not a.box.overlaps(b.box) or not _overlap(a.polygon, b.polygon, OVERLAP_PX):
            continue
        messages.append(f"{a.kind} {_short(a.label)} overlaps {b.kind} {_short(b.label)}")
    return messages


def _legend_covers(layout: _Layout) -> list[str]:
    messages = []
    for legend in layout.legends:
        box = legend.get_window_extent(layout.renderer)
        for ax in layout.axes:
            if any(_covers(box, mark.artist, layout.renderer) for mark in _marks(ax)):
                messages.append(
                    f"the legend covers data in {_panel(layout, ax)}; move it to an empty corner or outside the axes"
                )
    return messages


def _covers(box: Bbox, artist, renderer) -> bool:
    """Whether a legend box lies over a data mark."""
    if isinstance(artist, Line2D):
        path = artist.get_transform().transform_path(artist.get_path())
        if artist.get_linestyle() in ("None", "", " "):
            return any(box.contains(x, y) for x, y in path.vertices)
        return path.intersects_bbox(box, filled=False)
    if isinstance(artist, PathCollection):
        points = artist.get_offset_transform().transform(artist.get_offsets())
        return any(box.contains(x, y) for x, y in points)
    common = Bbox.intersection(box, artist.get_window_extent(renderer))
    return common is not None and common.width > OVERLAP_PX and common.height > OVERLAP_PX


def _clipped(layout: _Layout) -> list[str]:
    edge = layout.figure.bbox
    messages = []
    for item in layout.items:
        sides = [
            side
            for side, past in (
                ("left", item.box.x0 < edge.x0 - EDGE_PX),
                ("right", item.box.x1 > edge.x1 + EDGE_PX),
                ("bottom", item.box.y0 < edge.y0 - EDGE_PX),
                ("top", item.box.y1 > edge.y1 + EDGE_PX),
            )
            if past
        ]
        if sides:
            messages.append(f"{item.kind} {_short(item.label)} runs past the {' and '.join(sides)} edge of the figure")
    return messages


def _off_size(layout: _Layout) -> list[str]:
    found: dict[tuple[str, float], list[str]] = {}
    for item in layout.items:
        if item.size is None or item.kind == TILE_LABEL:
            continue
        if all(abs(item.size - points) > 0.05 for points in _theme_sizes(item.kind)):
            found.setdefault((item.kind, item.size), []).append(item.label)
    return [
        f"{kind} at {size:g} pt, theme size {' or '.join(f'{points:g}' for points in _theme_sizes(kind))} pt, "
        f"e.g. {_short(labels[0])}; drop the size override, the theme sets {', '.join(FONT_SIZES[kind])}"
        for (kind, size), labels in found.items()
    ]


def _theme_sizes(kind: str) -> list[float]:
    """The sizes in points the project theme gives a text element, or an alternative it may be."""
    keys = [key for alternative in SIZE_ALTERNATIVES.get(kind, (kind,)) for key in FONT_SIZES[alternative]]
    return sorted({float(config[key]) for key in keys if config[key] is not None})


def _under_floor(layout: _Layout) -> list[str]:
    column = COLUMN_WIDTHS_PX[layout.width]
    figure_width, figure_height = layout.figure.get_size_inches()
    scale = column / (figure_width * CSS_PX_PER_INCH)

    # the way out depends on the column the shape put the figure in
    wider = {
        "two-column": f"or draw it {NARROW_FIGURE:g}:1 or wider for the whole row; "
        "use figsize=HALF_SQUARE or HALF_TALL, or ONE_COLUMN",
        "one-column": 'or pass width="full-width" for a section\'s own fold; '
        'use figsize=ONE_COLUMN, or FULL_WIDTH with width="full-width"',
        "full-width": "or split it by attribute; use figsize=FULL_WIDTH",
    }[layout.width]

    messages = []
    for kind, size in sorted({(item.kind, item.size) for item in layout.items if item.size is not None}):
        if size * scale < RENDERED_FLOOR_PT:
            widest = column * size / (RENDERED_FLOOR_PT * CSS_PX_PER_INCH)
            messages.append(
                f"{kind} renders at {size * scale:.1f} pt in a {layout.width} figure "
                f"{figure_width:g} x {figure_height:g} in, under the {RENDERED_FLOOR_PT:g} pt floor; "
                f"make it at most {widest:.1f} in wide {wider} from create_assets"
            )
    return messages


def _crowded_ticks(layout: _Layout) -> list[str]:
    gap = MIN_TICK_GAP_PT * layout.figure.dpi / 72
    groups: dict[tuple[int, str], list[_Item]] = {}
    for item in layout.items:
        if item.kind == "tick label" and item.group is not None:
            groups.setdefault(item.group, []).append(item)
    messages = []
    for (index, key), labels in groups.items():
        along = 0 if key == "x" else 1
        labels.sort(key=lambda item: sum(point[along] for point in item.polygon))
        crowded = [(a, b) for a, b in pairwise(labels) if _overlap(a.polygon, b.polygon, -gap)]
        if crowded:
            a, b = crowded[0]
            panel = _panel(layout, layout.figure.axes[index])
            messages.append(
                f"{key} tick labels of {panel} crowd ({_short(a.label)} and {_short(b.label)} sit under "
                f"{MIN_TICK_GAP_PT:g} pt apart); use fewer ticks, horizontal bars or split the figure"
            )
    return messages


def _unlabelled_axes(layout: _Layout) -> list[str]:
    return [
        f"{_panel(layout, ax)} has no {key}-axis label"
        for ax in layout.axes
        for key in ("x", "y")
        if not _axis_label(layout, ax, key)
    ]


def _unitless_axes(layout: _Layout) -> list[str]:
    messages = []
    for ax in layout.axes:
        for key, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            label = _axis_label(layout, ax, key)
            ticks = [text.get_text() for text in _tick_labels(axis)]
            if (
                label
                and ticks
                and all(_is_number(tick) for tick in ticks)
                and not UNIT.search(label)
                and not _names_unitless_quantity(label, ticks)
            ):
                messages.append(
                    f"{key}-axis label {label!r} of {_panel(layout, ax)} has no unit; add one in parentheses, "
                    "or name the coefficient or fraction it shows (rho, tau-b, kappa, F1, share)"
                )
    return messages


def _names_unitless_quantity(label: str, ticks: list[str]) -> bool:
    """Whether the label names a coefficient or fraction whose ticks stay within -1 to 1."""
    if not UNITLESS.search(label):
        return False
    plain = [tick.strip().replace("\u2212", "-").replace(",", "") for tick in ticks]
    if any(tick.startswith("$") or tick.endswith("%") for tick in plain):
        return False
    return all(abs(float(tick)) <= UNITLESS_SPAN for tick in plain)


def _title_length(layout: _Layout) -> list[str]:
    suptitle = getattr(layout.figure, "_suptitle", None)
    title = suptitle.get_text().strip() if suptitle is not None else ""
    if not title and len(layout.axes) == 1:
        title = layout.axes[0].get_title().strip()
    if not title:
        return ["the figure has no title"]
    words = len(title.split())
    low, high = TITLE_WORDS
    if low <= words <= high:
        return []
    return [f"title {title!r} has {words} word{'' if words == 1 else 's'}; use {low} to {high}"]


def _default_names(layout: _Layout) -> list[str]:
    return [
        f"the legend shows the default name {_short(item.label)}; name the series"
        for item in layout.items
        if item.kind in ("legend text", "legend title") and DEFAULT_SERIES.match(item.label.strip())
    ]


def _bars_off_zero(layout: _Layout) -> list[str]:
    messages = []
    for ax in layout.axes:
        bars = [container for container in ax.containers if isinstance(container, BarContainer)]
        if not bars:
            continue
        axis = ax.xaxis if getattr(bars[0], "orientation", "vertical") == "horizontal" else ax.yaxis
        low, high = sorted(axis.get_view_interval())
        if axis.get_scale() == "linear" and not low <= 0 <= high:
            messages.append(
                f"the bars of {_panel(layout, ax)} do not start at zero: the value axis runs {low:g} to {high:g}"
            )
    return messages


def _arms_not_emphasised(layout: _Layout, arms: tuple[str, ...]) -> list[str]:
    messages = []
    for arm in arms:
        found = False
        for ax in layout.axes:
            marks = _marks(ax)
            ours = [mark for mark in marks if arm in mark.names]
            if not ours:
                continue
            found = True
            rest = [mark.artist for mark in marks if mark.kind == ours[0].kind and arm not in mark.names]
            if not _stands_out([mark.artist for mark in ours], rest):
                messages.append(
                    f"arm {arm!r} does not stand out in {_panel(layout, ax)}; highlight it or mute the rest (datachart emphasis)"
                )
        if not found:
            messages.append(f"arm {arm!r} is not a series or category of the figure")
    return messages


def _marks(ax) -> list[_Mark]:
    """The data marks of one axes: bars, boxes, drawn lines and scatter points."""
    marks = []
    for container in ax.containers:
        if isinstance(container, BarContainer):
            series = {container.get_label()} - {None}
            for patch in container.patches:
                x, y = (
                    patch.get_x() + patch.get_width() / 2,
                    patch.get_y() + patch.get_height() / 2,
                )
                marks.append(_Mark("bar", patch, frozenset(series | _category(ax, x, y))))
    for patch in ax.patches:
        if isinstance(patch, PathPatch):
            extent = patch.get_path().get_extents()
            marks.append(
                _Mark(
                    "box",
                    patch,
                    frozenset(_category(ax, *extent.get_points().mean(axis=0))),
                )
            )
    for line in ax.lines:
        xy = line.get_xydata()
        drawn = line.get_linestyle() not in ("None", "", " ") and len(xy) >= 2
        if drawn and not (np.allclose(xy[:, 0], xy[0, 0]) or np.allclose(xy[:, 1], xy[0, 1])):
            marks.append(_Mark("line", line, frozenset({line.get_label()})))
    for collection in ax.collections:
        if isinstance(collection, PathCollection):
            marks.append(_Mark("points", collection, frozenset({collection.get_label()})))
    return marks


def _category(ax, x: float, y: float) -> set:
    """The category tick label nearest a mark's centre, on whichever axis holds categories."""
    for axis, at in ((ax.xaxis, x), (ax.yaxis, y)):
        labels = [
            (tick.get_loc(), tick.label1.get_text()) for tick in axis.get_major_ticks(len(axis.get_majorticklocs()))
        ]
        named = [(loc, text) for loc, text in labels if text.strip() and not _is_number(text)]
        near = [text for loc, text in named if abs(loc - at) <= 0.5]
        if near:
            return {near[0]}
    return set()


def _stands_out(ours: list[object], rest: list[object]) -> bool:
    """Whether the arm's marks are at least as opaque and as thick as every other mark, and beat each on one."""
    alpha = min(_salience(artist)[0] for artist in ours)
    width = min(_salience(artist)[1] for artist in ours)
    for artist in rest:
        other_alpha, other_width = _salience(artist)
        if alpha < other_alpha - SALIENCE_EPS or width < other_width - SALIENCE_EPS:
            return False
        if alpha <= other_alpha + SALIENCE_EPS and width <= other_width + SALIENCE_EPS:
            return False
    return True


def _salience(artist) -> tuple[float, float]:
    """A mark's opacity and stroke width, the two things datachart emphasis changes."""
    if isinstance(artist, PathCollection):
        faces = artist.get_facecolors()
        alpha = artist.get_alpha() if artist.get_alpha() is not None else (faces[0][3] if len(faces) else 1.0)
        widths = artist.get_linewidths()
        return float(alpha), float(max(widths) if len(widths) else 0.0)
    if isinstance(artist, Line2D):
        alpha = (
            artist.get_alpha() if artist.get_alpha() is not None else matplotlib.colors.to_rgba(artist.get_color())[3]
        )
        return float(alpha), float(artist.get_linewidth())
    alpha = artist.get_alpha() if artist.get_alpha() is not None else artist.get_facecolor()[3]
    return float(alpha), float(artist.get_linewidth())


def _threshold_missing(layout: _Layout, threshold: float) -> list[str]:
    for ax in layout.axes:
        segments = [line.get_xydata() for line in ax.lines]
        segments += [
            np.asarray(segment)
            for collection in ax.collections
            if isinstance(collection, LineCollection)
            for segment in collection.get_segments()
        ]
        for xy in segments:
            if len(xy) >= 2 and (np.allclose(xy[:, 1], threshold) or np.allclose(xy[:, 0], threshold)):
                return []
    return [f"no reference line at the claim's threshold {threshold:g}; draw it with hlines or vlines"]


def _figure_misnamed(target: Path) -> list[str]:
    messages = []
    if target.parent.name != FIGURES_FOLDER:
        messages.append(f"the figure sits in {target.parent.name or '.'}/, not {FIGURES_FOLDER}/")
    if not RESULT_STEM.match(target.stem):
        messages.append(f"name {target.name!r} is not <stem>-<quantity>-by-<dimension> in lowercase hyphenated words")
    return messages


def _data_table_missing(target: Path) -> list[str]:
    table = target.parent.parent / TABLES_FOLDER / f"{target.stem}.csv"
    if table.exists():
        return []
    return [f"the figure shows a claim but its data table {TABLES_FOLDER}/{table.name} is missing; write it first"]


def _dataset_claim(target: Path, claim: Claim | None) -> list[str]:
    if claim is None or not target.stem.startswith(DATASET_PREFIX):
        return []
    return [f"dataset figure {target.name!r} shows claim {claim.clause}; a dataset figure carries no claim"]
