"""Parameter sweep plotting functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_gain_vs_scan_angle(
    data: dict[str, list[tuple[float, float]]],
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Plot peak gain vs steering angle for multiple configurations.

    Args:
        data: Dict mapping label to list of (angle_deg, gain_dbi) tuples.
        ax: Matplotlib axes. Created if None.
        **kwargs: Passed to ax.plot().

    Returns:
        The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots()

    for label, points in data.items():
        angles = [p[0] for p in points]
        gains = [p[1] for p in points]
        ax.plot(angles, gains, marker="o", label=label, **kwargs)

    ax.set_xlabel("Scan Angle [deg]")
    ax.set_ylabel("Peak Gain [dBi]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_gain_vs_frequency(
    data: dict[str, list[tuple[float, float]]],
    ax: Axes | None = None,
    freq_unit: str = "GHz",
    **kwargs: Any,
) -> Axes:
    """Plot gain vs frequency for multiple configurations.

    Args:
        data: Dict mapping label to list of (freq, gain_dbi) tuples.
            Frequency values should match freq_unit.
        ax: Matplotlib axes. Created if None.
        freq_unit: Label for frequency axis (default "GHz").
        **kwargs: Passed to ax.plot().

    Returns:
        The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots()

    for label, points in data.items():
        freqs = [p[0] for p in points]
        gains = [p[1] for p in points]
        ax.plot(freqs, gains, marker="o", label=label, **kwargs)

    ax.set_xlabel(f"Frequency [{freq_unit}]")
    ax.set_ylabel("Peak Gain [dBi]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
