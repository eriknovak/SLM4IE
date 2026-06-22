"""Color system for SLM4IE charts.

Defines the project's cool, academic, publication-ready palette and the three
color scales every chart draws from:

- A categorical palette for nominal dimensions (tokenizers, metrics, waffle
  slices). The colors are contrast-interleaved so adjacent categories get
  maximally different hues, which keeps overlapping lines and small markers
  legible.
- A sequential ramp (light cream to dark navy) for ordinal or continuous
  magnitude (e.g. vocabulary size).
- A diverging scale (navy = good, cream = neutral, amber = bad) for signed
  comparisons. The blue/orange axis is colorblind-safe; red/green is avoided.

Plus a small set of neutral accents for emphasis outlines and guide lines.

All values are derived from the author's reference figures. The exact hex codes
were tuned for legibility on a white background and may be adjusted; downstream
code should reference the names here rather than hardcoding hex strings.
"""

from typing import Dict, List, Sequence

# Cool ramp faithful to the reference figures, ordered light to dark. Used as
# the source for the sequential scale and available for large filled areas
# (bars, waffle cells) where the palest tones still read.
RAMP: List[str] = [
    "#EAEAD0",  # cream
    "#A9D6B8",  # mint
    "#4F9E8F",  # teal
    "#A3C4E0",  # sky
    "#3E6DA8",  # blue
    "#1F3864",  # navy
]

# Categorical palette for nominal dimensions. Same cool mood as RAMP, but
# contrast-interleaved (so adjacent categories differ strongly in hue and
# lightness) and with the palest tones deepened so thin lines and small markers
# stay visible on white.
CATEGORICAL: List[str] = [
    "#1F3864",  # navy
    "#7FC0A0",  # green (mint, deepened)
    "#3E6DA8",  # blue
    "#8FA67E",  # sage (cream, deepened to read on white)
    "#A3C4E0",  # sky
    "#4F9E8F",  # teal
]

# Diverging endpoints (colorblind-safe blue/orange axis).
_GOOD = "#1F3864"  # navy
_NEUTRAL = "#F2F2E8"  # near-cream
_BAD = "#D9883E"  # muted amber

# Neutral accents for emphasis and guides.
ACCENT: Dict[str, str] = {
    "emphasis": "#21314F",  # dark outline for highlighting (e.g. morpheme spans)
    "neutral": "#5A6473",  # gray for frontier lines, error bars, guides
}


def _even_stops(colors: Sequence[str]) -> List[List[object]]:
    """Spread colors evenly over the [0, 1] colorscale domain.

    Args:
        colors: Hex colors ordered from the low end of the scale to the high end.

    Returns:
        A plotly colorscale: a list of `[position, color]` pairs.
    """
    last = len(colors) - 1
    return [[i / last, color] for i, color in enumerate(colors)]


# Sequential scale for magnitude (cream to navy).
SEQUENTIAL: List[List[object]] = _even_stops(RAMP)

# Diverging scale for signed comparisons (amber = bad, cream = neutral,
# navy = good). Apply with `zmid=0` so the neutral midpoint lands on zero.
DIVERGING: List[List[object]] = [[0.0, _BAD], [0.5, _NEUTRAL], [1.0, _GOOD]]


def categorical_colors(keys: Sequence[str]) -> Dict[str, str]:
    """Assign a stable categorical color to each key.

    Colors are taken from `CATEGORICAL` in order, cycling if there are more keys
    than colors. The assignment is deterministic in the order of `keys`, so
    building the map once from the full list of categories pins each category to
    one color across every chart.

    Args:
        keys: Category names in a stable order (e.g. the configured tokenizers).

    Returns:
        Mapping from each key to its hex color.
    """
    palette = CATEGORICAL
    return {key: palette[i % len(palette)] for i, key in enumerate(keys)}
