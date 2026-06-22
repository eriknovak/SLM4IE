"""Export SLM4IE charts to static image files.

Two entry points:

- `save_figure` writes a figure to PNG, SVG, or PDF at a fixed size and scale,
  via plotly's `kaleido` engine. Use it for reproducible publication figures.
- `enable_interactive_download` configures the plotly renderer so the chart
  modebar's download button produces SVG (or another format) at high resolution.
  Call it once next to `register_theme()` so every figure shown in a notebook
  exports cleanly with a single click.

The `kaleido` engine is an optional dependency (the `notebook` extra). It is
imported lazily so importing this module never requires it.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import plotly.graph_objects as go
import plotly.io as pio


def save_figure(
    fig: go.Figure,
    path: Union[str, Path],
    scale: float = 3.0,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Path:
    """Write a figure to a static image file.

    The output format is inferred from the file extension (`.png`, `.svg`, or
    `.pdf`). `scale` only affects raster formats (PNG); vector formats (SVG, PDF)
    ignore it.

    Args:
        fig: The plotly figure to export.
        path: Destination file path; its suffix selects the format.
        scale: Resolution multiplier for raster output.
        width: Output width in pixels; defaults to the figure's layout width.
        height: Output height in pixels; defaults to the figure's layout height.

    Returns:
        The path that was written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(out, scale=scale, width=width, height=height)
    return out


def interactive_config(image_format: str = "svg", scale: float = 3.0) -> Dict[str, Any]:
    """Build a plotly config that sets the modebar download format.

    Args:
        image_format: Format the camera button exports (`svg`, `png`, `jpeg`, or
            `webp`).
        scale: Resolution multiplier for raster downloads.

    Returns:
        A plotly config dict suitable for `mo.ui.plotly(fig, config=...)` or for
        merging into a renderer's config.
    """
    return {
        "toImageButtonOptions": {"format": image_format, "scale": scale},
        "displaylogo": False,
    }


def enable_interactive_download(image_format: str = "svg", scale: float = 3.0) -> None:
    """Make the chart modebar download button export the given format.

    Merges `interactive_config` into the default plotly renderer's config, which
    marimo reads when rendering a bare figure. Has no effect if no usable
    renderer is available.

    Args:
        image_format: Format the camera button exports (`svg`, `png`, `jpeg`, or
            `webp`).
        scale: Resolution multiplier for raster downloads.
    """
    name = pio.renderers.default
    if not name or name not in pio.renderers:
        name = "browser"
    if name not in pio.renderers:
        return
    renderer = pio.renderers[name]
    existing = getattr(renderer, "config", None) or {}
    renderer.config = {**existing, **interactive_config(image_format, scale)}
