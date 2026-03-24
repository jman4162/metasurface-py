"""Publication-quality plot styling."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
from cycler import cycler

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Figure size presets for common publication targets
_STYLE_PRESETS: dict[str, tuple[float, float]] = {
    "ieee": (3.5, 2.8),
    "ieee_double": (7.0, 4.0),
    "nature": (3.5, 2.8),  # 89mm ≈ 3.5in
    "poster": (8.0, 6.0),
}

_FONT_SCALE: dict[str, float] = {
    "ieee": 1.0,
    "ieee_double": 1.0,
    "nature": 1.0,
    "poster": 1.6,
}


def set_publication_style(
    target: Literal["ieee", "ieee_double", "nature", "poster"] = "ieee",
) -> None:
    """Apply publication-quality matplotlib rcParams.

    Args:
        target: Publication target controlling figure size and font scaling.
            "ieee" — 3.5in single-column (default).
            "ieee_double" — 7in double-column.
            "nature" — 89mm single-column.
            "poster" — large format with scaled-up fonts.
    """
    figsize = _STYLE_PRESETS[target]
    scale = _FONT_SCALE[target]
    plt.rcParams.update(
        {
            "font.size": 10 * scale,
            "axes.labelsize": 11 * scale,
            "axes.titlesize": 11 * scale,
            "xtick.labelsize": 9 * scale,
            "ytick.labelsize": 9 * scale,
            "legend.fontsize": 9 * scale,
            "figure.figsize": figsize,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
            "axes.grid": True,
            "axes.prop_cycle": cycler(color=COLORBLIND_PALETTE),
        }
    )


def save_figure(
    fig: Figure,
    path: str | Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> list[Path]:
    """Save a figure in multiple formats for publication.

    Args:
        fig: Matplotlib figure to save.
        path: Base path without extension (e.g., "figures/pattern").
        formats: File formats to save. Defaults to PDF (vector) + PNG (raster).

    Returns:
        List of saved file paths.
    """
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        saved.append(out)
    return saved


# Wong colorblind-safe palette (8 colors)
COLORBLIND_PALETTE: list[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]
