"""Pareto front visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from metasurface_py.optimize.multiobjective import ParetoResult


def plot_pareto_front(
    result: ParetoResult,
    ax: Axes | None = None,
    negate_axes: tuple[bool, bool] = (True, True),
    **kwargs: Any,
) -> Axes:
    """Plot Pareto front from a ParetoResult.

    By default, negates both axes since objectives are typically
    formulated as minimization (negative gain, negative SLL).

    Args:
        result: ParetoResult from pareto_sweep.
        ax: Matplotlib axes. Created if None.
        negate_axes: Whether to negate (x, y) for display.
            Default (True, True) converts min objectives to max.
        **kwargs: Passed to ax.scatter().

    Returns:
        The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots()

    x = result.objective_values[:, 0]
    y = result.objective_values[:, 1]

    if negate_axes[0]:
        x = -x
    if negate_axes[1]:
        y = -y

    marker = kwargs.pop("marker", "o")
    s = kwargs.pop("s", 50)
    ax.scatter(x, y, marker=marker, s=s, **kwargs)

    # Connect points with line
    sort_idx = x.argsort()
    ax.plot(x[sort_idx], y[sort_idx], alpha=0.5, linestyle="--")

    xlabel = result.obj_a_name
    ylabel = result.obj_b_name
    if negate_axes[0]:
        xlabel = f"-({xlabel})"
    if negate_axes[1]:
        ylabel = f"-({ylabel})"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Pareto Front")
    ax.grid(True, alpha=0.3)
    return ax
