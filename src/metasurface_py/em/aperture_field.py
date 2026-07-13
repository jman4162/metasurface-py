"""Adiabatic-Floquet-mode aperture fields and the vector radiation integral.

Implements the reduced-order radiation model of modulated metasurface
antennas (Faenzi et al., Sci. Rep. 9:10178, 2019, Eq. 3): a cylindrical
TM surface wave launched at the aperture center acquires a slowly varying
complex wavenumber k0_loc(rho) = beta_sw + beta_delta(rho) - j*alpha(rho)
from the local modulated-reactance dispersion problem; the -1 Floquet
harmonic of the modulated sheet converts the surface-wave current into a
radiating tangential aperture field, which is integrated over the circular
aperture (equivalence principle over the ground plane) to obtain the
far-field E_theta / E_phi components.

Convention: exp(+j*omega*t); outgoing cylindrical waves use the Hankel
function of the second kind H_1^(2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.special import hankel2

from metasurface_py.core.conventions import ETA_0, k0
from metasurface_py.core.math_utils import direction_cosines
from metasurface_py.core.types import DIM_PHI, DIM_THETA
from metasurface_py.core.xarray_utils import make_pattern_dataset
from metasurface_py.em.array_factor import directivity

if TYPE_CHECKING:
    from metasurface_py.core.types import AngleGrid

ComplexArray = npt.NDArray[np.complexfloating[Any, Any]]
RealArray = npt.NDArray[np.floating[Any]]

# Maximum elements of the (angles x sources) phase matrix held in memory
# at once inside radiate_aperture; blocks of observation angles are
# processed sequentially to stay under this.
_MAX_BLOCK_ELEMENTS = 20_000_000


def afm_surface_current(
    rho: RealArray,
    k_local: ComplexArray,
) -> ComplexArray:
    """0-mode surface-wave current J_rho(rho) of the adiabatic Floquet model.

    J(rho) = H_1^(2)( int_0^rho k_local(rho') drho' ), the cylindrical
    outgoing-wave form of Faenzi et al. (2019) Eq. (3) for the n=0 mode,
    with the local complex wavenumber accumulated adiabatically along the
    radius. The imaginary part of k_local (leakage alpha) produces the
    radial amplitude decay.

    Args:
        rho: Radial samples, increasing, shape (n,) [m].
        k_local: Local complex wavenumber beta(rho) - j*alpha(rho), shape (n,).

    Returns:
        Complex current profile, shape (n,) (unnormalized).
    """
    rho_arr = np.asarray(rho, dtype=np.float64)
    k_arr = np.asarray(k_local, dtype=np.complex128)
    # Accumulated phase from the center; the first cell integrates k from 0.
    psi = np.empty_like(k_arr)
    psi[0] = k_arr[0] * rho_arr[0]
    if rho_arr.shape[0] > 1:
        psi[1:] = psi[0] + np.cumsum(0.5 * (k_arr[1:] + k_arr[:-1]) * np.diff(rho_arr))
    result: ComplexArray = hankel2(1, psi)
    return result


def modulated_aperture_field(
    rho: RealArray,
    phi: RealArray,
    j0_current: ComplexArray,
    x_avg: float,
    big_k: float,
    m_rho_rho: RealArray,
    m_rho_phi: RealArray,
    spiral_turns: int,
    phase_offset: float = 0.0,
    delta_rho_phi: float = -np.pi / 2.0,
    extra_phase: RealArray | None = None,
) -> tuple[ComplexArray, ComplexArray]:
    """Radiating -1-harmonic aperture field of the modulated reactance.

    The e^{+j(K s + Phi)} half of the modulation cos(K s + Phi) multiplies
    the 0-mode current to produce the fast (radiating) harmonic:
    E_rho = j*x_avg*(m_rr/2)*exp(j*(K*rho + Phi_rr(phi)))*J(rho), with
    Phi_rr = spiral_turns*phi + phase_offset, and analogously E_phi from
    the rho-phi tensor entry with Phi_rp = Phi_rr + delta_rho_phi.

    With spiral_turns = -1 and delta_rho_phi = -pi/2 the aperture field is
    (rho_hat - j*phi_hat)*A(rho)*e^{-j*phi} = (x_hat - j*y_hat)*A(rho),
    i.e. uniform RHCP for a broadside design.

    Args:
        rho: Radial samples, shape (n_rho,) [m].
        phi: Azimuthal samples, shape (n_phi,) [rad].
        j0_current: 0-mode current J(rho), shape (n_rho,).
        x_avg: Average transparent reactance [ohm].
        big_k: Modulation wavenumber K = 2*pi/period [rad/m].
        m_rho_rho: Modulation index profile for the rho-rho entry, shape (n_rho,).
        m_rho_phi: Modulation index profile for the rho-phi entry, shape (n_rho,);
            zeros for a scalar (isotropic) modulation.
        spiral_turns: Azimuthal spiral index n in Phi = n*phi (+-1 gives CP).
        phase_offset: Constant modulation phase [rad].
        delta_rho_phi: Phase of the rho-phi entry relative to rho-rho [rad].
        extra_phase: Optional additional modulation phase on the
            (rho, phi) grid, shape (n_rho, n_phi) [rad]. Used for tilted
            pencil beams, where the modulation phase carries a
            -k0*sin(theta0)*rho*cos(phi - phi0) term.

    Returns:
        (e_rho, e_phi): complex aperture fields, each shape (n_rho, n_phi).
    """
    rho_arr = np.asarray(rho, dtype=np.float64)
    phi_arr = np.asarray(phi, dtype=np.float64)
    radial = (
        1j
        * x_avg
        * 0.5
        * np.exp(1j * big_k * rho_arr)
        * np.asarray(j0_current, dtype=np.complex128)
    )
    spiral = np.exp(1j * (spiral_turns * phi_arr + phase_offset))
    e_rho = (m_rho_rho * radial)[:, np.newaxis] * spiral[np.newaxis, :]
    e_phi = (m_rho_phi * radial)[:, np.newaxis] * (spiral * np.exp(1j * delta_rho_phi))[
        np.newaxis, :
    ]
    if extra_phase is not None:
        twist = np.exp(1j * np.asarray(extra_phase, dtype=np.float64))
        e_rho = e_rho * twist
        e_phi = e_phi * twist
    return e_rho, e_phi


def radiate_aperture(
    positions: RealArray,
    e_x: ComplexArray,
    e_y: ComplexArray,
    cell_areas: RealArray,
    freq: float,
    angles: AngleGrid,
) -> xr.Dataset:
    """Far field of a tangential aperture field over a ground plane.

    Vector radiation integral (Balanis, Antenna Theory 4th ed., Ch. 12.5,
    exp(+j*omega*t) form): with f_{x,y}(theta, phi) =
    sum_n E_{x,y,n} * dA_n * exp(+j*k0*r_hat . r_n),

        E_theta = (j*k0/(2*pi)) * (f_x*cos(phi) + f_y*sin(phi))
        E_phi   = (j*k0/(2*pi)) * cos(theta) * (f_y*cos(phi) - f_x*sin(phi))

    The returned fields are r*E [V]: multiply by exp(-j*k0*r)/r for the
    field at distance r. Radiated power follows from
    :func:`radiated_power`, and D = 4*pi*|rE|^2 / (2*eta0*P).

    Args:
        positions: Aperture sample positions, shape (N, 3) [m].
        e_x: Aperture E_x at the samples, shape (N,) [V/m].
        e_y: Aperture E_y at the samples, shape (N,) [V/m].
        cell_areas: Quadrature areas, shape (N,) [m^2].
        freq: Frequency [Hz].
        angles: Observation angles (theta in [0, pi/2] for the half-space).

    Returns:
        xr.Dataset with complex data_vars ``E_theta``, ``E_phi`` [V]
        over dims (theta, phi); attrs carry freq and normalization.
    """
    kw = k0(freq)
    theta = np.asarray(angles.theta, dtype=np.float64)
    phi = np.asarray(angles.phi, dtype=np.float64)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    u, v, w = direction_cosines(theta_grid, phi_grid)
    r_hat = np.stack([u, v, w], axis=-1).reshape(-1, 3)

    wx = np.asarray(e_x, dtype=np.complex128) * cell_areas
    wy = np.asarray(e_y, dtype=np.complex128) * cell_areas

    n_angles = r_hat.shape[0]
    n_src = positions.shape[0]
    block = max(1, _MAX_BLOCK_ELEMENTS // max(n_src, 1))
    fx = np.empty(n_angles, dtype=np.complex128)
    fy = np.empty(n_angles, dtype=np.complex128)
    for start in range(0, n_angles, block):
        stop = min(start + block, n_angles)
        phase = np.exp(1j * kw * (r_hat[start:stop] @ positions.T))
        fx[start:stop] = phase @ wx
        fy[start:stop] = phase @ wy
    fx_grid = fx.reshape(theta_grid.shape)
    fy_grid = fy.reshape(theta_grid.shape)

    scale = 1j * kw / (2.0 * np.pi)
    cos_p = np.cos(phi_grid)
    sin_p = np.sin(phi_grid)
    e_theta = scale * (fx_grid * cos_p + fy_grid * sin_p)
    e_phi = scale * np.cos(theta_grid) * (fy_grid * cos_p - fx_grid * sin_p)

    attrs = {"freq_hz": freq, "normalization": "r*E [V]; field = value*exp(-j*k0*r)/r"}
    ds = xr.Dataset(
        {
            "E_theta": make_pattern_dataset(
                e_theta, theta=theta, phi=phi, name="E_theta", attrs=attrs
            ),
            "E_phi": make_pattern_dataset(
                e_phi, theta=theta, phi=phi, name="E_phi", attrs=attrs
            ),
        },
        attrs=attrs,
    )
    return ds


def total_field(pattern: xr.Dataset) -> xr.DataArray:
    """Total field magnitude sqrt(|E_theta|^2 + |E_phi|^2) as a DataArray.

    Suitable input for :func:`metasurface_py.em.array_factor.directivity`
    and the other scalar pattern metrics.
    """
    total = (abs(pattern["E_theta"]) ** 2 + abs(pattern["E_phi"]) ** 2) ** 0.5
    total.name = "total_field"
    total.attrs.update(pattern.attrs)
    return total


def pattern_directivity(pattern: xr.Dataset) -> xr.DataArray:
    """Directivity (linear) of a two-component E_theta/E_phi pattern."""
    return directivity(total_field(pattern))


def radiated_power(pattern: xr.Dataset) -> float:
    """Total radiated power [W] from an r*E pattern Dataset.

    P = (1/(2*eta0)) * integral(|E_theta|^2 + |E_phi|^2) sin(theta) dtheta dphi.
    """
    theta = pattern.coords[DIM_THETA].values
    phi = pattern.coords[DIM_PHI].values
    power = (
        np.abs(pattern["E_theta"].values) ** 2 + np.abs(pattern["E_phi"].values) ** 2
    )
    dtheta = np.gradient(theta) if theta.size > 1 else np.array([np.pi])
    dphi = np.gradient(phi) if phi.size > 1 else np.array([2 * np.pi])
    integrand = power * np.sin(theta)[:, np.newaxis]
    total = float(np.sum(integrand * dtheta[:, np.newaxis] * dphi[np.newaxis, :]))
    return total / (2.0 * ETA_0)


def aperture_flux_power(
    e_x: ComplexArray,
    e_y: ComplexArray,
    cell_areas: RealArray,
) -> float:
    """Power carried by a tangential aperture field [W].

    P = (1/(2*eta0)) * sum(|E_x|^2 + |E_y|^2) * dA. For electrically large
    apertures this matches the integrated radiated power of
    :func:`radiate_aperture` to within the quadrature error.
    """
    density = np.abs(e_x) ** 2 + np.abs(e_y) ** 2
    return float(np.sum(density * cell_areas) / (2.0 * ETA_0))
