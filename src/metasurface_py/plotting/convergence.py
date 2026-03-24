"""Optimization convergence plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_convergence(
    data: dict[str, npt.NDArray[np.floating[Any]]],
    ax: Axes | None = None,
    ylabel: str = "Objective Value",
    **kwargs: Any,
) -> Axes:
    """Plot optimization convergence curves.

    Args:
        data: Dict mapping label to 1D array of objective values
            per iteration.
        ax: Matplotlib axes. Created if None.
        ylabel: Label for y-axis.
        **kwargs: Passed to ax.plot().

    Returns:
        The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots()

    for label, values in data.items():
        iterations = np.arange(1, len(values) + 1)
        ax.plot(iterations, values, label=label, **kwargs)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
