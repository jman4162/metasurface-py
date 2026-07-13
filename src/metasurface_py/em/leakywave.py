"""Surface-wave and leaky-wave dispersion on modulated impedance sheets.

Reduced-order physics for surface-wave-fed modulated metasurface (MTS)
antennas: a grounded dielectric slab carries a printed-patch cladding
modeled as a "transparent" capacitive sheet reactance; the combination
supports a TM surface wave. Sinusoidally modulating the reactance turns
the surface wave into a leaky wave whose -1 Floquet harmonic radiates.

Two equivalent boundary-condition descriptions are used (Faenzi et al.,
Sci. Rep. 9:10178, 2019; Minatti et al., IEEE TAP 64(9):3896, 2016):

- transparent sheet reactance ``x_sheet`` [ohm]: the printed cladding
  alone, in parallel with the grounded-slab input reactance ``x_slab``.
  Capacitive, so ``x_sheet < 0``.
- opaque (impenetrable) reactance ``x_op`` [ohm]: the parallel
  combination seen from above, replacing sheet + slab + ground by one
  boundary condition. Inductive (``x_op > 0``) for a TM surface wave,
  with ``beta_sw/k0 = sqrt(1 + (x_op/eta0)^2)``.

The modulated-reactance dispersion problem X(x) = x_op*(1 + m*cos(2*pi*x/d))
is solved by transverse resonance with a truncated Floquet expansion
(Oliner & Hessel, IRE Trans. AP-7, 1959). The complex root
k_x = beta - j*alpha is found by Newton iteration with continuation in m.

Convention: exp(+j*omega*t); waves propagate as exp(-j*k_x*x) and the
leaky mode has alpha > 0 (decay along propagation).
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr

from metasurface_py.core.conventions import EPS_0, ETA_0, freq_to_omega, k0


@dataclass(frozen=True)
class SurfaceWaveSolution:
    """Self-consistent TM surface-wave solution on sheet + grounded slab.

    Attributes:
        beta: Surface-wave phase constant [rad/m].
        x_op: Equivalent opaque (impenetrable) reactance [ohm].
        x_slab: Grounded-slab input reactance at beta [ohm].
        x_sheet: Transparent sheet reactance used [ohm].
        freq: Frequency [Hz].
    """

    beta: float
    x_op: float
    x_slab: float
    x_sheet: float
    freq: float


@dataclass(frozen=True)
class LeakyWaveMode:
    """Complex leaky-wave mode of a sinusoidally modulated reactance.

    Attributes:
        beta: Phase constant of the fundamental (0-indexed) harmonic [rad/m].
        alpha: Attenuation (leakage) constant [Np/m], alpha >= 0.
        beta_sw: Unmodulated surface-wave phase constant [rad/m].
        converged: True if the Newton iteration and mode-count check passed.
        n_modes: Floquet harmonics per side used in the truncation.
    """

    beta: float
    alpha: float
    beta_sw: float
    converged: bool
    n_modes: int

    @property
    def k_lw(self) -> complex:
        """Complex leaky wavenumber beta - j*alpha [rad/m]."""
        return complex(self.beta, -self.alpha)

    @property
    def beta_delta(self) -> float:
        """Modulation-induced shift beta - beta_sw [rad/m]."""
        return self.beta - self.beta_sw


def grounded_slab_reactance_tm(
    freq: float, eps_r: float, thickness: float, beta: float
) -> float:
    """TM input reactance of a grounded dielectric slab at the top face.

    X_slab = (k_z1 / (omega * eps0 * eps_r)) * tan(k_z1 * h), with
    k_z1 = sqrt(eps_r*k0^2 - beta^2). Positive (inductive) for thin slabs
    below the first thickness resonance.

    Args:
        freq: Frequency [Hz].
        eps_r: Slab relative permittivity.
        thickness: Slab thickness [m].
        beta: Transverse propagation constant along the surface [rad/m].

    Returns:
        Input reactance [ohm] (sign carries inductive/capacitive).
    """
    if eps_r < 1.0:
        raise ValueError(f"eps_r must be >= 1, got {eps_r}")
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")
    omega = freq_to_omega(freq)
    kz1_sq = eps_r * k0(freq) ** 2 - beta**2
    kz1 = cmath.sqrt(complex(kz1_sq, 0.0))
    # For kz1_sq < 0 this reduces to -|kz1|*tanh(|kz1|*h)/(omega*eps0*eps_r).
    x_slab = (kz1 / (omega * EPS_0 * eps_r)) * cmath.tan(kz1 * thickness)
    return float(x_slab.real)


def transparent_to_opaque(x_sheet: float, x_slab: float) -> float:
    """Opaque reactance from transparent sheet + slab (parallel combination).

    j*X_op = (j*x_sheet * j*x_slab) / (j*x_sheet + j*x_slab). A capacitive
    sheet (x_sheet < 0) in parallel with the inductive slab yields the
    inductive opaque reactance that supports the TM surface wave.
    """
    denom = x_sheet + x_slab
    if abs(denom) < 1e-12:
        raise ValueError("x_sheet + x_slab is zero: parallel resonance")
    return x_sheet * x_slab / denom


def opaque_to_transparent(x_op: float, x_slab: float) -> float:
    """Inverse of :func:`transparent_to_opaque`."""
    denom = x_slab - x_op
    if abs(denom) < 1e-12:
        raise ValueError("x_slab equals x_op: transparent reactance diverges")
    return x_op * x_slab / denom


def sw_wavenumber_tm(x_op: float, freq: float) -> float:
    """TM surface-wave phase constant on an impenetrable reactance boundary.

    beta_sw / k0 = sqrt(1 + (x_op / eta0)^2), valid for inductive x_op > 0.

    Args:
        x_op: Opaque (impenetrable) surface reactance [ohm], must be > 0.
        freq: Frequency [Hz].

    Returns:
        beta_sw [rad/m].
    """
    if x_op <= 0:
        raise ValueError(
            f"TM surface wave requires inductive opaque reactance, got {x_op} ohm"
        )
    return k0(freq) * math.sqrt(1.0 + (x_op / ETA_0) ** 2)


def solve_sw_transparent(
    x_sheet: float,
    freq: float,
    eps_r: float,
    thickness: float,
) -> SurfaceWaveSolution:
    """Self-consistent TM surface wave for a transparent sheet on a grounded slab.

    Solves the transverse-resonance condition
    X_op(beta) = sqrt(beta^2 - k0^2) / (omega * eps0), where X_op(beta) is
    the parallel combination of the sheet with the slab reactance at beta.

    Args:
        x_sheet: Transparent sheet reactance [ohm] (capacitive: < 0).
        freq: Frequency [Hz].
        eps_r: Slab relative permittivity.
        thickness: Slab thickness [m].

    Returns:
        SurfaceWaveSolution with beta, x_op, x_slab.

    Raises:
        ValueError: If no TM surface-wave root exists in (k0, sqrt(eps_r)*k0).
    """
    from scipy.optimize import brentq

    kfree = k0(freq)
    omega = freq_to_omega(freq)

    def residual(beta: float) -> float:
        x_slab = grounded_slab_reactance_tm(freq, eps_r, thickness, beta)
        denom = x_sheet + x_slab
        if abs(denom) < 1e-9:
            return math.inf
        x_op = x_sheet * x_slab / denom
        return x_op - math.sqrt(beta**2 - kfree**2) / (omega * EPS_0)

    lo = kfree * (1.0 + 1e-9)
    hi = kfree * math.sqrt(eps_r) * (1.0 - 1e-9)
    betas = np.linspace(lo, hi, 512)
    vals = np.array([residual(float(b)) for b in betas])
    finite = np.isfinite(vals)
    sign_change = np.where(
        finite[:-1] & finite[1:] & (np.sign(vals[:-1]) != np.sign(vals[1:]))
    )[0]
    if sign_change.size == 0:
        raise ValueError(
            "No TM surface-wave root found; check x_sheet sign (capacitive < 0) "
            "and slab parameters"
        )
    i = int(sign_change[0])
    beta_root = float(brentq(residual, float(betas[i]), float(betas[i + 1])))
    x_slab = grounded_slab_reactance_tm(freq, eps_r, thickness, beta_root)
    x_op = transparent_to_opaque(x_sheet, x_slab)
    return SurfaceWaveSolution(
        beta=beta_root, x_op=x_op, x_slab=x_slab, x_sheet=x_sheet, freq=freq
    )


def _harmonic_impedance(kxn: complex, kfree: float, omega: float) -> complex:
    """Normalized TM free-space impedance of one Floquet harmonic.

    Evanescent harmonics (|Re k_xn| >= k0) take the proper branch
    k_zn = -j*sqrt(k_xn^2 - k0^2); harmonics inside the visible region
    take the outgoing branch k_zn = +sqrt(k0^2 - k_xn^2), which is the
    improper (leaky) choice for the radiating -1 harmonic.
    """
    if abs(kxn.real) < kfree:
        kzn = np.sqrt(complex(kfree**2) - kxn * kxn)
    else:
        kzn = -1j * np.sqrt(kxn * kxn - complex(kfree**2))
    return complex(kzn / (omega * EPS_0 * ETA_0))


def _transverse_resonance(
    kx: complex,
    x_op: float,
    m: float,
    big_k: float,
    kfree: float,
    omega: float,
    n_modes: int,
) -> complex:
    """Transverse-resonance residual in continued-fraction (Oliner-Hessel) form.

    The tridiagonal Floquet system is reduced onto the fundamental
    harmonic: F(kx) = Z_0 + j*x_op - T_plus - T_minus, where each T is a
    continued fraction over the harmonics on one side,
    T = c^2 / (Z_n + j*x_op - c^2 / (...)), c = j*x_op*m/2. All
    impedances are normalized by eta0. F is analytic in kx away from the
    branch points, so complex Newton converges quadratically.
    """
    xn = 1j * x_op / ETA_0
    c = 1j * x_op * m / (2.0 * ETA_0)
    c2 = c * c

    def side(sign: int) -> complex:
        t: complex = 0.0 + 0.0j
        for n in range(n_modes, 0, -1):
            z = _harmonic_impedance(kx + sign * n * big_k, kfree, omega) + xn
            t = c2 / (z - t)
        return t

    z0 = _harmonic_impedance(kx, kfree, omega) + xn
    return z0 - side(+1) - side(-1)


def _newton_root(
    seed: complex,
    x_op: float,
    m: float,
    big_k: float,
    kfree: float,
    omega: float,
    n_modes: int,
) -> tuple[complex, bool]:
    """Complex Newton iteration on the Floquet determinant."""
    kx = seed
    step = 1e-7 * kfree
    for _ in range(60):
        f0 = _transverse_resonance(kx, x_op, m, big_k, kfree, omega, n_modes)
        fp = _transverse_resonance(kx + step, x_op, m, big_k, kfree, omega, n_modes)
        fm = _transverse_resonance(kx - step, x_op, m, big_k, kfree, omega, n_modes)
        deriv = (fp - fm) / (2.0 * step)
        if deriv == 0:
            return kx, False
        delta = f0 / deriv
        kx = kx - delta
        if abs(delta) < 1e-12 * kfree:
            return kx, True
    return kx, False


def dispersion_modulated_reactance(
    x_op: float,
    m: float,
    period: float,
    freq: float,
    n_modes: int = 7,
) -> LeakyWaveMode:
    """Complex leaky-wave mode of X(x) = x_op*(1 + m*cos(2*pi*x/period)).

    Solves the truncated Floquet transverse-resonance determinant by
    complex Newton iteration with continuation in the modulation index
    (steps of <= 0.05 starting from the analytic m=0 surface wave).

    Args:
        x_op: Average opaque reactance [ohm], > 0.
        m: Modulation index (0 <= m < 1).
        period: Modulation period d [m]; K = 2*pi/d.
        freq: Frequency [Hz].
        n_modes: Floquet harmonics per side in the truncation.

    Returns:
        LeakyWaveMode with beta, alpha and convergence status. ``converged``
        also requires the root to move < 1e-5*k0 when n_modes is increased
        by 2.

    References:
        Oliner & Hessel, IRE Trans. Antennas Propag. 7, 201-208 (1959).
        Minatti et al., IEEE Trans. Antennas Propag. 64(9), 3896-3906 (2016).
    """
    if not 0.0 <= m < 1.0:
        raise ValueError(f"modulation index must be in [0, 1), got {m}")
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    beta_sw = sw_wavenumber_tm(x_op, freq)
    if m == 0.0:
        return LeakyWaveMode(
            beta=beta_sw, alpha=0.0, beta_sw=beta_sw, converged=True, n_modes=n_modes
        )

    kfree = k0(freq)
    omega = freq_to_omega(freq)
    big_k = 2.0 * math.pi / period

    n_steps = max(1, math.ceil(m / 0.05))
    kx = complex(beta_sw, 0.0)
    ok = True
    for i in range(1, n_steps + 1):
        m_i = m * i / n_steps
        seed = kx if kx.imag < 0 else complex(kx.real, -1e-4 * kfree)
        kx, step_ok = _newton_root(seed, x_op, m_i, big_k, kfree, omega, n_modes)
        ok = ok and step_ok

    kx_check, check_ok = _newton_root(kx, x_op, m, big_k, kfree, omega, n_modes + 2)
    ok = ok and check_ok and abs(kx_check - kx) < 1e-5 * kfree

    alpha = -kx.imag
    converged = ok and alpha >= -1e-9 * kfree
    return LeakyWaveMode(
        beta=float(kx.real),
        alpha=float(max(alpha, 0.0)),
        beta_sw=beta_sw,
        converged=converged,
        n_modes=n_modes,
    )


def dispersion_map(
    x_op: float,
    m_values: npt.NDArray[np.floating[Any]],
    period: float,
    freq: float,
    n_modes: int = 7,
) -> xr.Dataset:
    """Leaky-wave beta and alpha versus modulation index.

    Reproduces Fig. 10-style maps from Faenzi et al. (2019): the
    propagation and attenuation constants of the leaky wave as functions
    of the modulation index.

    Args:
        x_op: Average opaque reactance [ohm].
        m_values: Modulation indices, shape (n_m,).
        period: Modulation period [m].
        freq: Frequency [Hz].
        n_modes: Floquet truncation per side.

    Returns:
        xr.Dataset with dim ``m`` and data_vars ``beta_over_k0``,
        ``alpha_over_k0``, ``converged``; attrs carry x_op, period, freq.
    """
    kfree = k0(freq)
    m_arr = np.asarray(m_values, dtype=np.float64)
    beta = np.empty_like(m_arr)
    alpha = np.empty_like(m_arr)
    conv = np.empty(m_arr.shape, dtype=bool)
    for i, m in enumerate(m_arr):
        mode = dispersion_modulated_reactance(x_op, float(m), period, freq, n_modes)
        beta[i] = mode.beta / kfree
        alpha[i] = mode.alpha / kfree
        conv[i] = mode.converged
    return xr.Dataset(
        {
            "beta_over_k0": ("m", beta),
            "alpha_over_k0": ("m", alpha),
            "converged": ("m", conv),
        },
        coords={"m": ("m", m_arr)},
        attrs={
            "x_op_ohm": x_op,
            "period_m": period,
            "freq_hz": freq,
            "n_modes": n_modes,
        },
    )


def alpha_for_taper(
    rho: npt.NDArray[np.floating[Any]],
    amplitude: npt.NDArray[np.floating[Any]],
    efficiency: float = 0.9,
) -> npt.NDArray[np.floating[Any]]:
    """Radial leakage profile alpha(rho) for a target aperture amplitude.

    Cylindrical leaky-wave illumination synthesis:
    2*alpha(rho) = rho*|A|^2 / ((1/eta_s)*int_0^a rho'|A|^2 drho'
                                 - int_0^rho rho'|A|^2 drho')
    where eta_s is the fraction of surface-wave power radiated over the
    aperture (Minatti et al., IEEE TAP 64(9), 3907-3919, 2016).

    Args:
        rho: Radial samples, increasing, shape (n,) [m].
        amplitude: Target aperture amplitude |A(rho)|, shape (n,).
        efficiency: Fraction of SW power radiated (0 < eta_s < 1).

    Returns:
        alpha(rho) [Np/m], shape (n,).
    """
    if not 0.0 < efficiency < 1.0:
        raise ValueError(f"efficiency must be in (0, 1), got {efficiency}")
    rho_arr = np.asarray(rho, dtype=np.float64)
    a2 = rho_arr * np.asarray(amplitude, dtype=np.float64) ** 2
    cumulative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (a2[1:] + a2[:-1]) * np.diff(rho_arr))]
    )
    total = cumulative[-1]
    if total <= 0:
        raise ValueError("amplitude profile has zero power")
    denom = total / efficiency - cumulative
    return 0.5 * a2 / denom  # type: ignore[no-any-return]


def sw_group_velocity(
    x_sheet: float,
    freq: float,
    eps_r: float,
    thickness: float,
    rel_step: float = 1e-3,
) -> float:
    """Normalized surface-wave group velocity v_g/c.

    Central finite difference of the surface-wave dispersion beta(omega).
    The transparent sheet is treated as a quasi-static capacitance, so
    x_sheet scales as f_ref/f at the stencil points (Faenzi et al. 2019,
    wideband section).

    Args:
        x_sheet: Transparent sheet reactance at ``freq`` [ohm].
        freq: Frequency [Hz].
        eps_r: Slab relative permittivity.
        thickness: Slab thickness [m].
        rel_step: Relative frequency step for the stencil.

    Returns:
        v_g / c (dimensionless, in (0, 1]).
    """
    from metasurface_py.core.conventions import SPEED_OF_LIGHT

    f_lo = freq * (1.0 - rel_step)
    f_hi = freq * (1.0 + rel_step)
    beta_lo = solve_sw_transparent(x_sheet * freq / f_lo, f_lo, eps_r, thickness).beta
    beta_hi = solve_sw_transparent(x_sheet * freq / f_hi, f_hi, eps_r, thickness).beta
    d_omega = 2.0 * math.pi * (f_hi - f_lo)
    v_g = d_omega / (beta_hi - beta_lo)
    return float(v_g / SPEED_OF_LIGHT)


def gain_bandwidth_product(
    v_g_over_c: float,
    radius_lambda: float,
    uniform: bool = True,
) -> float:
    """Gain-bandwidth product estimate for modulated MTS antennas.

    GB ~= 22*(v_g/c)*a_lambda for uniform amplitude modulation and
    GB ~= 47*(v_g/c)*a_lambda/(a_lambda + 2) for optimal amplitude
    tapering, with G linear and B = delta_f/f0 (Faenzi et al. 2019,
    citing Minatti et al., IEEE TAP 65(6), 2836-2842, 2017).

    Args:
        v_g_over_c: Normalized SW group velocity.
        radius_lambda: Aperture radius in free-space wavelengths.
        uniform: True for uniform modulation, False for optimal tapering.

    Returns:
        Gain-bandwidth product (linear gain times relative bandwidth).
    """
    if uniform:
        return 22.0 * v_g_over_c * radius_lambda
    return 47.0 * v_g_over_c * radius_lambda / (radius_lambda + 2.0)


def relative_bandwidth(v_g_over_c: float, radius_lambda: float) -> float:
    """Maximum relative bandwidth B ~= 0.95*(v_g/c)/a_lambda (uniform modulation)."""
    return 0.95 * v_g_over_c / radius_lambda
