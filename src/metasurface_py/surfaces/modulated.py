"""Surface-wave-fed modulated metasurface antennas.

Reduced-order model of the antennas in Faenzi et al., Sci. Rep. 9:10178
(2019): a circular aperture on a grounded dielectric slab, fed at the
center by a monopole that launches a cylindrical TM surface wave; a
sinusoidally modulated sheet reactance converts the surface wave into a
radiating leaky wave.

This object intentionally does NOT reuse :class:`Metasurface`: there is
no incident plane wave, no discrete unit-cell state, and the "state" is
a set of continuous modulation profiles. Unlike the RIS classes, the
substrate permittivity and thickness here are computational inputs (they
set the surface-wave dispersion), not provenance metadata.

Feed model: the monopole launcher is not modeled; ``launch_efficiency``
scales the radiated power and is an explicit user knob.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import numpy.typing as npt
import xarray as xr

from metasurface_py.core.conventions import k0, wavelength
from metasurface_py.em.aperture_field import (
    afm_surface_current,
    modulated_aperture_field,
    radiate_aperture,
)
from metasurface_py.em.leakywave import (
    SurfaceWaveSolution,
    dispersion_map,
    dispersion_modulated_reactance,
    gain_bandwidth_product,
    relative_bandwidth,
    solve_sw_transparent,
    sw_group_velocity,
)
from metasurface_py.geometry.aperture import CircularAperture

if TYPE_CHECKING:
    from metasurface_py.core.types import AngleGrid, SubstrateInfo

RealArray = npt.NDArray[np.floating[Any]]
ProfileLike = float | Callable[[RealArray], RealArray]


def _evaluate_profile(profile: ProfileLike, rho: RealArray) -> RealArray:
    if callable(profile):
        return np.asarray(profile(rho), dtype=np.float64)
    return np.full(rho.shape, float(profile), dtype=np.float64)


def _leaky_matched_period(x_op: float, freq: float, m_seed: float) -> float:
    """Modulation period matched to the leaky wave: K = beta_lw.

    The -1 harmonic of the modulated current is then radially in phase
    (broadside contribution); beam tilt is applied separately through a
    linear term in the modulation phase. beta_lw depends weakly on the
    period through the dispersion problem, so a short fixed-point
    iteration is used.
    """
    from metasurface_py.em.leakywave import sw_wavenumber_tm

    big_k = sw_wavenumber_tm(x_op, freq)
    for _ in range(3):
        mode = dispersion_modulated_reactance(x_op, m_seed, 2.0 * math.pi / big_k, freq)
        big_k = mode.beta
    return 2.0 * math.pi / big_k


def _profile_seed(m: ProfileLike) -> float:
    if callable(m):
        return float(np.max(m(np.linspace(1e-3, 1.0, 32))))
    return float(m)


@dataclass(frozen=True)
class SinusoidalModulation:
    """Scalar (isotropic) sinusoidal reactance modulation.

    X(rho, phi) = x_avg * (1 + m(rho) * cos(psi)), with modulation phase
    psi = K*rho + spiral_turns*phi + phase_offset
          - k0*sin(tilt_theta)*rho*cos(phi - tilt_phi),
    K = 2*pi/period. A pure radial period (tilt_theta = 0) with
    K = beta_lw - k0*sin(theta0) produces a conical beam at theta0; a
    pencil beam at (theta0, phi0) requires K = beta_lw plus the linear
    tilt term (this is the slowly varying s(rho) of Faenzi et al. Eq. 2).

    Args:
        m: Modulation index, constant or radial profile m(rho).
        period: Modulation period [m].
        spiral_turns: Azimuthal spiral index n in Phi = n*phi.
        phase_offset: Constant modulation phase [rad].
        tilt_theta: Pencil-beam polar tilt angle [rad].
        tilt_phi: Pencil-beam azimuth [rad].
    """

    m: ProfileLike
    period: float
    spiral_turns: int = 0
    phase_offset: float = 0.0
    tilt_theta: float = 0.0
    tilt_phi: float = 0.0

    @classmethod
    def for_broadside(
        cls,
        x_op: float,
        freq: float,
        m: ProfileLike,
        spiral_turns: int = 0,
        phase_offset: float = 0.0,
    ) -> Self:
        """Period matched to the leaky wave (K = beta_lw) for a broadside beam."""
        return cls(
            m=m,
            period=_leaky_matched_period(x_op, freq, _profile_seed(m)),
            spiral_turns=spiral_turns,
            phase_offset=phase_offset,
        )

    @classmethod
    def for_tilted_beam(
        cls,
        x_op: float,
        freq: float,
        m: ProfileLike,
        theta0: float,
        phi0: float = 0.0,
        spiral_turns: int = 0,
        phase_offset: float = 0.0,
    ) -> Self:
        """Pencil beam tilted to (theta0, phi0) [rad]."""
        return cls(
            m=m,
            period=_leaky_matched_period(x_op, freq, _profile_seed(m)),
            spiral_turns=spiral_turns,
            phase_offset=phase_offset,
            tilt_theta=theta0,
            tilt_phi=phi0,
        )


@dataclass(frozen=True)
class TensorModulation:
    """Anisotropic sinusoidal reactance modulation (circular polarization).

    Tensor entries (Faenzi et al. 2019, Eq. 1-2):
    X_rr = x_avg*(1 + m_rho_rho*cos(K*rho + Phi_rr)),
    X_rp = x_avg*m_rho_phi*cos(K*rho + Phi_rr + delta_rho_phi), with
    Phi_rr = spiral_turns*phi + phase_offset. With spiral_turns = -1 and
    delta_rho_phi = -pi/2 the radiating -1 harmonic is uniform RHCP at
    broadside; spiral_turns = +1, delta_rho_phi = +pi/2 gives LHCP.

    The rho-rho entry governs the TM leaky-wave dispersion; the rho-phi
    entry adds the orthogonal aperture-field component.
    """

    m_rho_rho: ProfileLike
    m_rho_phi: ProfileLike
    period: float
    spiral_turns: int = -1
    phase_offset: float = 0.0
    delta_rho_phi: float = -math.pi / 2.0
    tilt_theta: float = 0.0
    tilt_phi: float = 0.0

    @classmethod
    def circular_broadside(
        cls,
        x_op: float,
        freq: float,
        m: ProfileLike,
        hand: str = "rhcp",
    ) -> Self:
        """Broadside circularly polarized design (spiral modulation)."""
        if hand not in ("rhcp", "lhcp"):
            raise ValueError(f"hand must be 'rhcp' or 'lhcp', got {hand!r}")
        sign = -1 if hand == "rhcp" else +1
        return cls(
            m_rho_rho=m,
            m_rho_phi=m,
            period=_leaky_matched_period(x_op, freq, _profile_seed(m)),
            spiral_turns=sign,
            delta_rho_phi=sign * math.pi / 2.0,
        )

    @classmethod
    def circular_tilted(
        cls,
        x_op: float,
        freq: float,
        m: ProfileLike,
        theta0: float,
        phi0: float = 0.0,
        hand: str = "rhcp",
    ) -> Self:
        """Circularly polarized pencil beam tilted to (theta0, phi0) [rad]."""
        if hand not in ("rhcp", "lhcp"):
            raise ValueError(f"hand must be 'rhcp' or 'lhcp', got {hand!r}")
        sign = -1 if hand == "rhcp" else +1
        return cls(
            m_rho_rho=m,
            m_rho_phi=m,
            period=_leaky_matched_period(x_op, freq, _profile_seed(m)),
            spiral_turns=sign,
            delta_rho_phi=sign * math.pi / 2.0,
            tilt_theta=theta0,
            tilt_phi=phi0,
        )


ModulationLike = SinusoidalModulation | TensorModulation


@dataclass(frozen=True)
class BandwidthGainResult:
    """Closed-form bandwidth and gain-bandwidth estimates.

    Attributes:
        freq: Frequency [Hz].
        radius_lambda: Aperture radius in free-space wavelengths.
        v_g_over_c: Normalized surface-wave group velocity.
        gb_uniform: Gain-bandwidth product, uniform modulation (linear).
        gb_tapered: Gain-bandwidth product, optimal tapering (linear).
        bandwidth: Maximum relative bandwidth delta_f/f0 (uniform).
    """

    freq: float
    radius_lambda: float
    v_g_over_c: float
    gb_uniform: float
    gb_tapered: float
    bandwidth: float


@dataclass(frozen=True)
class ModulatedMetasurfaceAntenna:
    """Circular surface-wave-fed modulated metasurface antenna.

    Args:
        radius: Aperture radius [m].
        x_transparent: Average transparent sheet reactance [ohm]
            (capacitive: negative).
        substrate: Grounded slab; ``eps_r`` and ``thickness_mm`` are used
            in the dispersion computation.
        modulation: Scalar or tensor sinusoidal modulation.
        launch_efficiency: Fraction of feed power launched into the
            surface wave (the monopole itself is not modeled).
        n_rho: Radial aperture samples.
        n_phi: Azimuthal aperture samples.
    """

    radius: float
    x_transparent: float
    substrate: SubstrateInfo
    modulation: ModulationLike
    launch_efficiency: float = 1.0
    n_rho: int = 512
    n_phi: int = 256

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        if self.x_transparent >= 0:
            raise ValueError(
                "x_transparent must be negative (capacitive sheet), got "
                f"{self.x_transparent}"
            )
        if not 0.0 < self.launch_efficiency <= 1.0:
            raise ValueError(
                f"launch_efficiency must be in (0, 1], got {self.launch_efficiency}"
            )

    @property
    def aperture(self) -> CircularAperture:
        """Polar sampling grid over the aperture."""
        return CircularAperture(radius=self.radius, n_rho=self.n_rho, n_phi=self.n_phi)

    def surface_wave(self, freq: float) -> SurfaceWaveSolution:
        """Self-consistent TM surface-wave solution (beta_sw, x_op, x_slab)."""
        return solve_sw_transparent(
            self.x_transparent,
            freq,
            self.substrate.eps_r,
            self.substrate.thickness_mm * 1e-3,
        )

    def _m_profiles(self, rho: RealArray) -> tuple[RealArray, RealArray]:
        mod = self.modulation
        if isinstance(mod, TensorModulation):
            return (
                _evaluate_profile(mod.m_rho_rho, rho),
                _evaluate_profile(mod.m_rho_phi, rho),
            )
        return _evaluate_profile(mod.m, rho), np.zeros_like(rho)

    def dispersion(
        self,
        freq: float,
        m_values: npt.NDArray[np.floating[Any]] | None = None,
    ) -> xr.Dataset:
        """Leaky-wave dispersion versus modulation index at ``freq``.

        Defaults to a sweep from 0 to the maximum modulation index of
        this antenna's profile (Fig. 10-style map).
        """
        sw = self.surface_wave(freq)
        if m_values is None:
            rho = self.aperture.rho
            m_rr, _ = self._m_profiles(rho)
            m_top = max(float(np.max(m_rr)), 0.05)
            m_values = np.linspace(0.0, m_top, 33)
        return dispersion_map(sw.x_op, m_values, self.modulation.period, freq)

    def aperture_field(self, freq: float) -> xr.Dataset:
        """AFM aperture field (radiating -1 harmonic) at ``freq``.

        Returns:
            xr.Dataset with dims (rho, phi_ap): complex ``e_rho``,
            ``e_phi`` and the 0-mode current ``j0_rho``; attrs carry
            beta_sw, x_op, K and freq.
        """
        ap = self.aperture
        sw = self.surface_wave(freq)
        rho = ap.rho
        m_rr, m_rp = self._m_profiles(rho)
        mod = self.modulation

        # Local complex wavenumber from the dispersion problem, interpolated
        # over the (unique) modulation-index values of the radial profile.
        m_grid = np.linspace(0.0, max(float(np.max(m_rr)), 1e-6), 25)
        disp = dispersion_map(sw.x_op, m_grid, mod.period, freq)
        kfree = k0(freq)
        beta_loc = np.interp(m_rr, m_grid, disp["beta_over_k0"].values) * kfree
        alpha_loc = np.interp(m_rr, m_grid, disp["alpha_over_k0"].values) * kfree
        k_local = beta_loc - 1j * alpha_loc

        j0 = afm_surface_current(rho, k_local.astype(np.complex128))
        j0 = j0 * math.sqrt(self.launch_efficiency)
        extra_phase: RealArray | None = None
        if mod.tilt_theta != 0.0:
            extra_phase = (
                -kfree
                * math.sin(mod.tilt_theta)
                * rho[:, np.newaxis]
                * np.cos(ap.phi[np.newaxis, :] - mod.tilt_phi)
            )
        e_rho, e_phi = modulated_aperture_field(
            rho,
            ap.phi,
            j0,
            x_avg=self.x_transparent,
            big_k=2.0 * math.pi / mod.period,
            m_rho_rho=m_rr,
            m_rho_phi=m_rp,
            spiral_turns=mod.spiral_turns,
            phase_offset=mod.phase_offset,
            delta_rho_phi=(
                mod.delta_rho_phi
                if isinstance(mod, TensorModulation)
                else -math.pi / 2.0
            ),
            extra_phase=extra_phase,
        )
        return xr.Dataset(
            {
                "e_rho": (("rho", "phi_ap"), e_rho),
                "e_phi": (("rho", "phi_ap"), e_phi),
                "j0_rho": (("rho",), j0),
                "alpha": (("rho",), alpha_loc),
                "beta": (("rho",), beta_loc),
            },
            coords={
                "rho": ("rho", rho, {"unit": "m"}),
                "phi_ap": ("phi_ap", ap.phi, {"unit": "rad"}),
            },
            attrs={
                "freq_hz": freq,
                "beta_sw": sw.beta,
                "x_op_ohm": sw.x_op,
                "x_transparent_ohm": self.x_transparent,
                "modulation_K": 2.0 * math.pi / mod.period,
                "launch_efficiency": self.launch_efficiency,
            },
        )

    def far_field(self, freq: float, angles: AngleGrid) -> xr.Dataset:
        """Far-field E_theta / E_phi pattern (r*E [V]) at ``freq``."""
        ap = self.aperture
        fields = self.aperture_field(freq)
        e_rho = fields["e_rho"].values
        e_phi = fields["e_phi"].values
        phi_ap = ap.phi[np.newaxis, :]
        cos_p = np.cos(phi_ap)
        sin_p = np.sin(phi_ap)
        e_x = (e_rho * cos_p - e_phi * sin_p).ravel()
        e_y = (e_rho * sin_p + e_phi * cos_p).ravel()
        return radiate_aperture(
            ap.positions,
            e_x,
            e_y,
            ap.cell_areas.ravel(),
            freq,
            angles,
        )

    def bandwidth_gain(self, freq: float) -> BandwidthGainResult:
        """Closed-form gain-bandwidth estimates (Faenzi et al. 2019)."""
        v_g = sw_group_velocity(
            self.x_transparent,
            freq,
            self.substrate.eps_r,
            self.substrate.thickness_mm * 1e-3,
        )
        a_lambda = self.radius / wavelength(freq)
        return BandwidthGainResult(
            freq=freq,
            radius_lambda=a_lambda,
            v_g_over_c=v_g,
            gb_uniform=gain_bandwidth_product(v_g, a_lambda, uniform=True),
            gb_tapered=gain_bandwidth_product(v_g, a_lambda, uniform=False),
            bandwidth=relative_bandwidth(v_g, a_lambda),
        )


def dual_band_reactance_map(
    x_avg: float,
    modulations: Sequence[ModulationLike],
    centers: Sequence[tuple[float, float]],
    extent: float,
    n: int = 256,
) -> xr.Dataset:
    """Superposed spiral-modulation reactance map (Faenzi et al. Eqs. 4-6).

    Synthesis-only: evaluates the rho-rho reactance entry of a
    superposition of sinusoidal modulations, each referred to its own
    feed center, over a Cartesian grid. Reproduces the interference
    layout of the dual-band antenna (paper Fig. 9 inset).

    Args:
        x_avg: Average transparent reactance [ohm].
        modulations: One modulation per band.
        centers: Feed center (x, y) of each modulation [m].
        extent: Half-size of the square evaluation window [m].
        n: Grid points per axis.

    Returns:
        xr.Dataset with dims (y, x) and data_var ``x_rr`` [ohm].
    """
    if len(modulations) != len(centers):
        raise ValueError("modulations and centers must have the same length")
    axis = np.linspace(-extent, extent, n)
    xg, yg = np.meshgrid(axis, axis)
    total = np.zeros_like(xg)
    for mod, (cx, cy) in zip(modulations, centers, strict=True):
        rr = np.hypot(xg - cx, yg - cy)
        pp = np.arctan2(yg - cy, xg - cx)
        big_k = 2.0 * math.pi / mod.period
        m_rr = mod.m_rho_rho if isinstance(mod, TensorModulation) else mod.m
        m_vals = _evaluate_profile(m_rr, rr.ravel()).reshape(rr.shape)
        total += m_vals * np.cos(big_k * rr + mod.spiral_turns * pp + mod.phase_offset)
    x_rr = x_avg * (1.0 + total)
    return xr.Dataset(
        {"x_rr": (("y", "x"), x_rr)},
        coords={
            "x": ("x", axis, {"unit": "m"}),
            "y": ("y", axis, {"unit": "m"}),
        },
        attrs={"x_avg_ohm": x_avg, "num_modulations": len(modulations)},
    )
