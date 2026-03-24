"""Far-field pattern plotting functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import xarray as xr
    from matplotlib.axes import Axes


def plot_pattern_2d(
    pattern: xr.DataArray,
    cut_phi: float = 0.0,
    ax: Axes | None = None,
    db: bool = True,
    normalize: bool = True,
    **kwargs: Any,
) -> Axes:
    """Plot a 2D pattern cut (E-plane or H-plane).

    Args:
        pattern: Far-field pattern with dims (theta, phi).
        cut_phi: Phi value for the cut [rad].
        ax: Matplotlib axes. Created if None.
        db: If True, plot in dB scale.
        normalize: If True, normalize to peak.
        **kwargs: Passed to ax.plot().

    Returns:
        The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots()

    phi = pattern.coords["phi"].values
    theta = pattern.coords["theta"].values
    phi_idx = int(np.argmin(np.abs(phi - cut_phi)))

    cut_data = np.abs(pattern.values[:, phi_idx])

    if normalize and np.max(cut_data) > 0:
        cut_data = cut_data / np.max(cut_data)

    if db:
        with np.errstate(divide="ignore"):
            cut_data = 20.0 * np.log10(np.maximum(cut_data, 1e-10))
        ylabel = "Normalized Pattern [dB]" if normalize else "Pattern [dB]"
    else:
        ylabel = "Normalized Pattern" if normalize else "Pattern"

    theta_deg = np.rad2deg(theta)
    label = kwargs.pop("label", f"$\\phi = {np.rad2deg(cut_phi):.0f}°$")
    ax.plot(theta_deg, cut_data, label=label, **kwargs)
    ax.set_xlabel("$\\theta$ [deg]")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_pattern_polar(
    pattern: xr.DataArray,
    cut_phi: float = 0.0,
    ax: Axes | None = None,
    db: bool = True,
    db_range: float = 40.0,
    **kwargs: Any,
) -> Axes:
    """Plot a polar pattern cut.

    Args:
        pattern: Far-field pattern with dims (theta, phi).
        cut_phi: Phi value for the cut [rad].
        ax: Polar matplotlib axes. Created if None.
        db: If True, plot in dB scale.
        db_range: Dynamic range in dB for floor clipping.
        **kwargs: Passed to ax.plot().

    Returns:
        The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    phi = pattern.coords["phi"].values
    theta = pattern.coords["theta"].values
    phi_idx = int(np.argmin(np.abs(phi - cut_phi)))

    cut_data = np.abs(pattern.values[:, phi_idx])

    if np.max(cut_data) > 0:
        cut_data = cut_data / np.max(cut_data)

    if db:
        with np.errstate(divide="ignore"):
            cut_data = 20.0 * np.log10(np.maximum(cut_data, 1e-10))
        cut_data = np.maximum(cut_data, -db_range)
        # Shift to [0, db_range]
        cut_data = cut_data + db_range

    ax.plot(theta, cut_data, **kwargs)
    ax.set_theta_zero_location("N")  # type: ignore[attr-defined]
    ax.set_theta_direction(-1)  # type: ignore[attr-defined]
    return ax


def plot_pattern_uv(
    pattern: xr.DataArray,
    ax: Axes | None = None,
    db: bool = True,
    db_range: float = 40.0,
    **kwargs: Any,
) -> tuple[Axes, Any]:
    """Plot pattern in u-v space as a filled contour.

    Args:
        pattern: Far-field pattern with dims (theta, phi).
        ax: Matplotlib axes. Created if None.
        db: If True, plot in dB scale.
        db_range: Dynamic range for color scaling.
        **kwargs: Passed to ax.pcolormesh().

    Returns:
        Tuple of (Axes, QuadMesh).
    """
    if ax is None:
        _, ax = plt.subplots()

    theta = pattern.coords["theta"].values
    phi = pattern.coords["phi"].values
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    u = np.sin(theta_grid) * np.cos(phi_grid)
    v = np.sin(theta_grid) * np.sin(phi_grid)
    data = np.abs(pattern.values)

    if np.max(data) > 0:
        data = data / np.max(data)

    if db:
        with np.errstate(divide="ignore"):
            data = 20.0 * np.log10(np.maximum(data, 1e-10))
        data = np.maximum(data, -db_range)
        label = "Pattern [dB]"
    else:
        label = "Pattern"

    cmap = kwargs.pop("cmap", "viridis")
    mesh = ax.pcolormesh(u, v, data, cmap=cmap, shading="auto", **kwargs)
    ax.set_xlabel("u = sin($\\theta$)cos($\\phi$)")
    ax.set_ylabel("v = sin($\\theta$)sin($\\phi$)")
    ax.set_aspect("equal")
    plt.colorbar(mesh, ax=ax, label=label)
    return ax, mesh
