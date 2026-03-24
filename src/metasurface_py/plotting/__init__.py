"""Plotting utilities for patterns, geometry, and results."""

from metasurface_py.plotting.convergence import plot_convergence
from metasurface_py.plotting.geometry import (
    plot_element_amplitude_phase,
    plot_lattice,
    plot_state_map,
)
from metasurface_py.plotting.patterns import (
    plot_pattern_2d,
    plot_pattern_3d,
    plot_pattern_comparison,
    plot_pattern_polar,
    plot_pattern_uv,
)
from metasurface_py.plotting.style import (
    COLORBLIND_PALETTE,
    save_figure,
    set_publication_style,
)
from metasurface_py.plotting.sweeps import (
    plot_gain_vs_frequency,
    plot_gain_vs_scan_angle,
)

__all__ = [
    "COLORBLIND_PALETTE",
    "plot_convergence",
    "plot_element_amplitude_phase",
    "plot_gain_vs_frequency",
    "plot_gain_vs_scan_angle",
    "plot_lattice",
    "plot_pattern_2d",
    "plot_pattern_3d",
    "plot_pattern_comparison",
    "plot_pattern_polar",
    "plot_pattern_uv",
    "plot_state_map",
    "save_figure",
    "set_publication_style",
]
