"""Model comparison and validation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from metasurface_py.core.conventions import k0
from metasurface_py.em.array_factor import array_factor

if TYPE_CHECKING:
    from metasurface_py.core.types import AngleGrid
    from metasurface_py.surfaces.metasurface import Metasurface
    from metasurface_py.surfaces.state import SurfaceState


def compare_models(
    surface_a: Metasurface,
    state_a: SurfaceState,
    surface_b: Metasurface,
    state_b: SurfaceState,
    freq: float,
    angles: AngleGrid,
) -> xr.Dataset:
    """Compare two models over a frequency/angle grid.

    Useful for comparing reduced-order predictions against
    full-wave solver outputs or measured data.

    Args:
        surface_a: First model surface.
        state_a: First model state.
        surface_b: Second model surface.
        state_b: Second model state.
        freq: Frequency [Hz].
        angles: Observation angle grid.

    Returns:
        xr.Dataset with magnitude_error_db, phase_error_deg,
        and aggregate metrics as attributes.
    """
    kw = k0(freq)

    weights_a = surface_a.cell.response(state_a.values, freq)
    af_a = array_factor(
        surface_a.positions,
        weights_a,
        kw,
        angles.theta,
        angles.phi,
    )

    weights_b = surface_b.cell.response(state_b.values, freq)
    af_b = array_factor(
        surface_b.positions,
        weights_b,
        kw,
        angles.theta,
        angles.phi,
    )

    mag_a = np.abs(af_a)
    mag_b = np.abs(af_b)
    phase_a = np.angle(af_a)
    phase_b = np.angle(af_b)

    # Magnitude error in dB
    with np.errstate(divide="ignore", invalid="ignore"):
        mag_a_safe = np.maximum(mag_a, 1e-30)
        mag_b_safe = np.maximum(mag_b, 1e-30)
        mag_error_db = 20.0 * np.log10(mag_a_safe / mag_b_safe)

    # Phase error in degrees
    phase_diff = np.angle(np.exp(1j * (phase_a - phase_b)))
    phase_error_deg = np.rad2deg(phase_diff)

    # Aggregate metrics
    rms_mag_error = float(
        np.sqrt(np.mean(mag_error_db[np.isfinite(mag_error_db)] ** 2))
    )
    rms_phase_error = float(np.sqrt(np.mean(phase_error_deg**2)))

    return xr.Dataset(
        {
            "magnitude_error_db": (
                ["theta", "phi"],
                mag_error_db,
            ),
            "phase_error_deg": (
                ["theta", "phi"],
                phase_error_deg,
            ),
        },
        coords={
            "theta": angles.theta,
            "phi": angles.phi,
        },
        attrs={
            "rms_magnitude_error_db": rms_mag_error,
            "rms_phase_error_deg": rms_phase_error,
            "freq_hz": freq,
        },
    )
