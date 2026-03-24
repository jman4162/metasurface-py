"""Export utilities for sharing data with external tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

    from metasurface_py.surfaces.state import SurfaceState


def export_state_csv(
    state: SurfaceState,
    path: str | Path,
    nx: int | None = None,
    ny: int | None = None,
) -> Path:
    """Export phase state to CSV for MATLAB/external tools.

    Args:
        state: Surface state to export.
        path: Output file path (.csv).
        nx: Number of elements along x (for 2D reshaping).
        ny: Number of elements along y (for 2D reshaping).

    Returns:
        Path to saved file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    values = np.rad2deg(state.values)
    if nx is not None and ny is not None:
        values = values.reshape(nx, ny)
        header = "Phase map [degrees], rows=x, cols=y"
    else:
        header = "Phase values [degrees], one per element"

    np.savetxt(p, values, delimiter=",", header=header, fmt="%.6f")
    return p


def export_pattern_csv(
    pattern: xr.DataArray,
    path: str | Path,
) -> Path:
    """Export far-field pattern to CSV.

    Exports theta [deg], phi [deg], magnitude, phase [deg]
    for each observation angle.

    Args:
        pattern: Far-field pattern with dims (theta, phi).
        path: Output file path (.csv).

    Returns:
        Path to saved file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    theta = pattern.coords["theta"].values
    phi = pattern.coords["phi"].values
    theta_g, phi_g = np.meshgrid(theta, phi, indexing="ij")

    data = pattern.values
    rows = np.column_stack(
        [
            np.rad2deg(theta_g.ravel()),
            np.rad2deg(phi_g.ravel()),
            np.abs(data.ravel()),
            np.rad2deg(np.angle(data.ravel())),
        ]
    )

    header = "theta_deg,phi_deg,magnitude,phase_deg"
    np.savetxt(p, rows, delimiter=",", header=header, fmt="%.6f")
    return p
