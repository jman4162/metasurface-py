"""Radar cross-section and detection models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from metasurface_py.core.conventions import SPEED_OF_LIGHT, k0
from metasurface_py.core.xarray_utils import make_pattern_dataset
from metasurface_py.em.array_factor import array_factor

if TYPE_CHECKING:
    import xarray as xr

    from metasurface_py.core.types import AngleGrid, Position3D
    from metasurface_py.surfaces.metasurface import Metasurface
    from metasurface_py.surfaces.state import SurfaceState


def monostatic_rcs(
    surface: Metasurface,
    state: SurfaceState,
    freq: float,
    angles: AngleGrid,
) -> xr.DataArray:
    """Compute monostatic RCS pattern.

    For each observation direction, the RCS is proportional to
    |AF(theta, phi)|^2 where the illumination and observation
    directions are the same.

    Args:
        surface: Metasurface object.
        state: Surface configuration.
        freq: Frequency [Hz].
        angles: Observation angles.

    Returns:
        RCS pattern as xr.DataArray [m^2], dims (theta, phi).
    """
    kw = k0(freq)
    lam = SPEED_OF_LIGHT / freq
    weights = surface.cell.response(state.values, freq)

    # For monostatic: incident = observation direction
    # RCS = 4*pi * |AF|^2 * (dx*dy)^2 / lambda^2
    af = array_factor(
        surface.positions,
        weights,
        kw,
        angles.theta,
        angles.phi,
    )
    # Normalize by element area (approximate)
    rcs = 4.0 * np.pi * np.abs(af) ** 2 * lam**2 / (4.0 * np.pi) ** 2

    return make_pattern_dataset(
        rcs,
        theta=angles.theta,
        phi=angles.phi,
        name="rcs",
        attrs={"unit": "m^2", "freq_hz": freq, "type": "monostatic"},
    )


def bistatic_rcs(
    surface: Metasurface,
    state: SurfaceState,
    freq: float,
    theta_inc: float,
    phi_inc: float,
    angles_obs: AngleGrid,
) -> xr.DataArray:
    """Compute bistatic RCS pattern.

    Args:
        surface: Metasurface object.
        state: Surface configuration.
        freq: Frequency [Hz].
        theta_inc: Incident polar angle [rad].
        phi_inc: Incident azimuthal angle [rad].
        angles_obs: Observation angles.

    Returns:
        Bistatic RCS as xr.DataArray [m^2].
    """
    kw = k0(freq)
    lam = SPEED_OF_LIGHT / freq
    weights = surface.cell.response(
        state.values,
        freq,
        theta_inc,
        phi_inc,
    )

    af = array_factor(
        surface.positions,
        weights,
        kw,
        angles_obs.theta,
        angles_obs.phi,
    )
    rcs = 4.0 * np.pi * np.abs(af) ** 2 * lam**2 / (4.0 * np.pi) ** 2

    return make_pattern_dataset(
        rcs,
        theta=angles_obs.theta,
        phi=angles_obs.phi,
        name="rcs_bistatic",
        attrs={
            "unit": "m^2",
            "freq_hz": freq,
            "type": "bistatic",
            "theta_inc": theta_inc,
            "phi_inc": phi_inc,
        },
    )


def detection_snr(
    surface: Metasurface,
    state: SurfaceState,
    target_pos: Position3D,
    freq: float,
    tx_power: float = 1.0,
    noise_power: float = 1e-12,
    target_rcs: float = 1.0,
) -> float:
    """Compute radar detection SNR for a point target.

    Uses simplified radar equation:
    SNR = Pt * G^2 * lambda^2 * sigma / ((4*pi)^3 * R^4 * Pn)

    Args:
        surface: Metasurface acting as radar aperture.
        state: Surface configuration.
        target_pos: Target position.
        freq: Frequency [Hz].
        tx_power: Transmit power [W].
        noise_power: Noise power [W].
        target_rcs: Target radar cross-section [m^2].

    Returns:
        Detection SNR (linear).
    """
    lam = SPEED_OF_LIGHT / freq
    kw = k0(freq)
    positions = surface.positions

    # Distance to target from array center
    target = np.array(
        [target_pos.x, target_pos.y, target_pos.z],
        dtype=np.float64,
    )
    center = positions.mean(axis=0)
    r = float(np.linalg.norm(target - center))

    if r < 1e-10:
        return 0.0

    # Array gain in target direction
    weights = surface.cell.response(state.values, freq)
    # Direction to target
    diff = target - center
    diff_norm = diff / np.linalg.norm(diff)
    k_target = kw * diff_norm

    # Steering vector for target direction
    phase = positions @ k_target
    sv = np.exp(1j * phase)
    gain = float(np.abs(np.sum(weights * sv)) ** 2)

    snr = (
        tx_power
        * gain**2
        * lam**2
        * target_rcs
        / ((4 * np.pi) ** 3 * r**4 * noise_power)
    )
    return float(snr)
