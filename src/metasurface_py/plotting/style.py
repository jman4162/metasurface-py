"""Publication-quality plot styling."""

from __future__ import annotations

import matplotlib.pyplot as plt
from cycler import cycler


def set_publication_style() -> None:
    """Apply publication-quality matplotlib rcParams.

    Sets font sizes, linewidths, and other parameters suitable for
    journal figures.
    """
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.figsize": (3.5, 2.8),
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
