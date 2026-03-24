"""Geometry and state visualization functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from metasurface_py.geometry.lattice import SupportsLattice
    from metasurface_py.surfaces.state import SurfaceState


def plot_lattice(
    lattice: SupportsLattice,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Plot element positions.

    Args:
        lattice: Lattice object.
        ax: Matplotlib axes. Created if None.
        **kwargs: Passed to ax.scatter().

    Returns:
        The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots()

    pos = lattice.positions
    marker = kwargs.pop("marker", "s")
    s = kwargs.pop("s", 20)
    ax.scatter(pos[:, 0] * 1e3, pos[:, 1] * 1e3, marker=marker, s=s, **kwargs)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    return ax


def plot_state_map(
    state: SurfaceState,
    nx: int,
    ny: int,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Axes, Any]:
    """Plot phase state as a 2D color map on the element grid.

    Args:
        state: Surface state.
        nx: Number of elements along x.
        ny: Number of elements along y.
        ax: Matplotlib axes. Created if None.
        **kwargs: Passed to ax.pcolormesh().

    Returns:
        Tuple of (Axes, QuadMesh).
    """
    if ax is None:
        _, ax = plt.subplots()

    phase_map = np.rad2deg(state.values.reshape(nx, ny))
    cmap = kwargs.pop("cmap", "twilight")
    mesh = ax.pcolormesh(phase_map, cmap=cmap, shading="auto", **kwargs)
    ax.set_xlabel("Element x")
    ax.set_ylabel("Element y")
    ax.set_aspect("equal")
    plt.colorbar(mesh, ax=ax, label="Phase [deg]")
    return ax, mesh


def plot_element_amplitude_phase(
    response: npt.NDArray[np.complexfloating[Any, Any]],
    nx: int,
    ny: int,
    fig: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Plot amplitude and phase maps side by side.

    Args:
        response: Complex element responses, shape (N,).
        nx: Number of elements along x.
        ny: Number of elements along y.
        fig: Matplotlib figure. Created if None.
        **kwargs: Passed to pcolormesh.

    Returns:
        The matplotlib Figure.
    """
    if fig is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    else:
        axes = fig.subplots(1, 2)
        ax1, ax2 = axes[0], axes[1]

    amp = np.abs(response).reshape(nx, ny)
    phase = np.rad2deg(np.angle(response)).reshape(nx, ny)

    m1 = ax1.pcolormesh(amp, cmap="viridis", shading="auto", **kwargs)
    ax1.set_xlabel("Element x")
    ax1.set_ylabel("Element y")
    ax1.set_aspect("equal")
    ax1.set_title("Amplitude")
    plt.colorbar(m1, ax=ax1, label="Magnitude")

    m2 = ax2.pcolormesh(
        phase,
        cmap="twilight",
        shading="auto",
        **kwargs,
    )
    ax2.set_xlabel("Element x")
    ax2.set_ylabel("Element y")
    ax2.set_aspect("equal")
    ax2.set_title("Phase")
    plt.colorbar(m2, ax=ax2, label="Phase [deg]")

    fig.tight_layout()
    return fig
