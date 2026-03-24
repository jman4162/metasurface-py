"""Optional plotting functions requiring seaborn or plotly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_quantization_heatmap(
    data: npt.NDArray[np.floating[Any]],
    row_labels: list[str],
    col_labels: list[str],
    ax: Axes | None = None,
    xlabel: str = "Number of Bits",
    ylabel: str = "Array Size",
    value_label: str = "Gain Loss [dB]",
    **kwargs: Any,
) -> Axes:
    """Plot a heatmap using seaborn (optional dependency).

    Useful for parametric studies such as gain loss vs (bits, array size).

    Args:
        data: 2D array of values, shape (n_rows, n_cols).
        row_labels: Labels for rows.
        col_labels: Labels for columns.
        ax: Matplotlib axes. Created if None.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        value_label: Label for colorbar.
        **kwargs: Passed to seaborn.heatmap().

    Returns:
        The matplotlib Axes.

    Raises:
        ImportError: If seaborn is not installed.
    """
    try:
        import seaborn as sns
    except ImportError as e:
        raise ImportError(
            "seaborn is required for plot_quantization_heatmap. "
            "Install with: pip install seaborn"
        ) from e

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    annot = kwargs.pop("annot", True)
    fmt = kwargs.pop("fmt", ".2f")
    cmap = kwargs.pop("cmap", "YlOrRd")
    sns.heatmap(
        data,
        ax=ax,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        cbar_kws={"label": value_label},
        **kwargs,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_pattern_3d_interactive(
    pattern: Any,
    db: bool = True,
    db_range: float = 30.0,
    title: str = "3D Radiation Pattern",
) -> Any:
    """Plot interactive 3D radiation pattern using plotly.

    Args:
        pattern: Far-field pattern xr.DataArray with dims (theta, phi).
        db: If True, use dB scale.
        db_range: Dynamic range in dB.
        title: Figure title.

    Returns:
        Plotly Figure object (can be saved as HTML).

    Raises:
        ImportError: If plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "plotly is required for plot_pattern_3d_interactive. "
            "Install with: pip install plotly"
        ) from e

    theta = pattern.coords["theta"].values
    phi = pattern.coords["phi"].values
    theta_g, phi_g = np.meshgrid(theta, phi, indexing="ij")

    data = np.abs(pattern.values)
    if np.max(data) > 0:
        data = data / np.max(data)

    if db:
        with np.errstate(divide="ignore"):
            r = 20.0 * np.log10(np.maximum(data, 1e-10))
        r = np.maximum(r, -db_range)
        r = (r + db_range) / db_range
        colorbar_title = "Pattern [dB]"
    else:
        r = data
        colorbar_title = "Pattern"

    x = r * np.sin(theta_g) * np.cos(phi_g)
    y = r * np.sin(theta_g) * np.sin(phi_g)
    z = r * np.cos(theta_g)

    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=r,
                colorscale="Viridis",
                colorbar={"title": colorbar_title},
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
        },
    )
    return fig
