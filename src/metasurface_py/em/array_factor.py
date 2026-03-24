"""Array factor computation and far-field pattern analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from metasurface_py.core.conventions import k0
from metasurface_py.core.math_utils import direction_cosines
from metasurface_py.core.xarray_utils import make_pattern_dataset

if TYPE_CHECKING:
    import xarray as xr

    from metasurface_py.core.types import AngleGrid
    from metasurface_py.surfaces.metasurface import Metasurface
    from metasurface_py.surfaces.state import SurfaceState


def array_factor(
    positions: npt.NDArray[np.floating[Any]],
    weights: npt.NDArray[np.complexfloating[Any, Any]],
    wavenumber: float,
    theta: npt.NDArray[np.floating[Any]],
    phi: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.complexfloating[Any, Any]]:
    """Compute array factor over a grid of observation angles.

    AF(theta, phi) = sum_n w_n * exp(j * k * r_hat(theta,phi) . r_n)

    Args:
        positions: Element positions, shape (N, 3) [meters].
        weights: Complex element weights, shape (N,).
        wavenumber: Free-space wavenumber k0 [rad/m].
        theta: Observation polar angles, shape (n_theta,) [rad].
        phi: Observation azimuthal angles, shape (n_phi,) [rad].

    Returns:
        Complex array factor, shape (n_theta, n_phi).
    """
    # Direction cosines for all (theta, phi) combinations
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    u, v, w = direction_cosines(theta_grid, phi_grid)

    # r_hat . r_n for all observation angles and all elements
    # r_hat shape: (n_theta, n_phi, 3)
    r_hat = np.stack([u, v, w], axis=-1)  # (n_theta, n_phi, 3)

    # positions: (N, 3), r_hat: (n_theta, n_phi, 3)
    # phase = k * r_hat . r_n -> shape (n_theta, n_phi, N)
    phase = wavenumber * np.einsum("ijk,nk->ijn", r_hat, positions)

    # AF = sum_n w_n * exp(j * phase_n)
    af = np.einsum("ijn,n->ij", np.exp(1j * phase), weights)
    return af  # type: ignore[no-any-return]


def far_field_pattern(
    surface: Metasurface,
    state: SurfaceState,
    freq: float,
    angles: AngleGrid,
    theta_inc: float = 0.0,
    phi_inc: float = 0.0,
) -> xr.DataArray:
    """Compute the far-field radiation pattern of a metasurface.

    Combines element response (from the cell model) with the array factor.

    Args:
        surface: The metasurface object.
        state: Current surface state.
        freq: Frequency [Hz].
        angles: Observation angles.
        theta_inc: Incident polar angle [rad].
        phi_inc: Incident azimuthal angle [rad].

    Returns:
        Complex far-field pattern as labeled xr.DataArray with dims (theta, phi).
        Attributes include unit="complex_field".
    """
    weights = surface.response(state, freq, theta_inc, phi_inc)
    kw = k0(freq)
    af = array_factor(surface.positions, weights, kw, angles.theta, angles.phi)
    return make_pattern_dataset(
        af,
        theta=angles.theta,
        phi=angles.phi,
        name="far_field",
        attrs={"unit": "complex_field", "freq_hz": freq},
    )


def directivity(
    pattern: xr.DataArray,
) -> xr.DataArray:
    """Compute directivity from a far-field pattern.

    D(theta, phi) = 4*pi * |AF|^2 / integral(|AF|^2 * sin(theta) dtheta dphi)

    Args:
        pattern: Complex far-field pattern with dims (theta, phi).

    Returns:
        Directivity pattern (linear) with same dims.
    """
    theta = pattern.coords["theta"].values
    phi = pattern.coords["phi"].values

    power = np.abs(pattern.values) ** 2

    # Numerical integration using trapezoidal rule
    dtheta = np.gradient(theta) if len(theta) > 1 else np.array([np.pi])
    dphi = np.gradient(phi) if len(phi) > 1 else np.array([2 * np.pi])

    sin_theta = np.sin(theta)
    # Integrate: sum |AF|^2 * sin(theta) * dtheta * dphi
    integrand = power * sin_theta[:, np.newaxis]
    total_power = float(np.sum(integrand * dtheta[:, np.newaxis] * dphi[np.newaxis, :]))

    if total_power < 1e-30:
        d = np.zeros_like(power)
    else:
        d = 4.0 * np.pi * power / total_power

    return make_pattern_dataset(
        d,
        theta=theta,
        phi=phi,
        name="directivity",
        attrs={"unit": "linear"},
    )


def peak_gain_db(pattern: xr.DataArray) -> float:
    """Peak directivity/gain in dBi from a far-field pattern.

    Args:
        pattern: Complex far-field pattern.

    Returns:
        Peak directivity in dBi.
    """
    d = directivity(pattern)
    peak = float(np.max(d.values))
    if peak <= 0:
        return -np.inf
    return float(10.0 * np.log10(peak))


def sidelobe_level(
    pattern: xr.DataArray,
    main_beam_theta: float,
    main_beam_phi: float,
    exclusion_radius_rad: float = 0.1,
) -> float:
    """Compute peak sidelobe level relative to main beam [dB].

    Args:
        pattern: Complex far-field pattern.
        main_beam_theta: Main beam polar angle [rad].
        main_beam_phi: Main beam azimuthal angle [rad].
        exclusion_radius_rad: Angular radius around main beam to exclude [rad].

    Returns:
        Peak sidelobe level in dB (negative value = sidelobes below main beam).
    """
    theta = pattern.coords["theta"].values
    phi = pattern.coords["phi"].values
    power = np.abs(pattern.values) ** 2

    main_beam_power = float(np.max(power))
    if main_beam_power <= 0:
        return 0.0

    # Create angular distance mask
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    # Approximate angular distance
    ang_dist = np.sqrt(
        (theta_grid - main_beam_theta) ** 2
        + (np.sin(main_beam_theta) * (phi_grid - main_beam_phi)) ** 2
    )

    sidelobe_mask = ang_dist > exclusion_radius_rad
    if not np.any(sidelobe_mask):
        return -np.inf

    peak_sidelobe = float(np.max(power[sidelobe_mask]))
    if peak_sidelobe <= 0:
        return -np.inf

    return float(10.0 * np.log10(peak_sidelobe / main_beam_power))


def half_power_beamwidth(
    pattern: xr.DataArray,
    cut_phi: float = 0.0,
) -> float:
    """Compute half-power beamwidth (HPBW) for a given phi cut [rad].

    Args:
        pattern: Complex far-field pattern.
        cut_phi: Phi value for the cut plane [rad].

    Returns:
        HPBW in radians. Returns NaN if cannot be determined.
    """
    phi = pattern.coords["phi"].values
    theta = pattern.coords["theta"].values

    # Find nearest phi index
    phi_idx = int(np.argmin(np.abs(phi - cut_phi)))
    cut = np.abs(pattern.values[:, phi_idx]) ** 2

    peak_val = np.max(cut)
    if peak_val <= 0:
        return float("nan")

    half_power = peak_val / 2.0
    peak_idx = int(np.argmax(cut))
    above = cut >= half_power

    if not np.any(above):
        return float("nan")

    # Find contiguous region around the peak
    # Search left from peak
    left = peak_idx
    while left > 0 and above[left - 1]:
        left -= 1
    # Search right from peak
    right = peak_idx
    while right < len(above) - 1 and above[right + 1]:
        right += 1

    return float(theta[right] - theta[left])
