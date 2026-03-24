"""Geometry and state visualization functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

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
