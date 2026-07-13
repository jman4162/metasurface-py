"""Tests for surface-wave and leaky-wave dispersion on modulated reactances."""

from __future__ import annotations

import numpy as np
import pytest

from metasurface_py.core.conventions import ETA_0, k0
from metasurface_py.em.leakywave import (
    alpha_for_taper,
    dispersion_map,
    dispersion_modulated_reactance,
    gain_bandwidth_product,
    grounded_slab_reactance_tm,
    opaque_to_transparent,
    relative_bandwidth,
    solve_sw_transparent,
    sw_group_velocity,
    sw_wavenumber_tm,
    transparent_to_opaque,
)

# Faenzi et al. 2019 dual-band antenna: RO3010, eps_r = 10.2, h = 0.635 mm.
EPS_R = 10.2
H = 0.635e-3


class TestSurfaceWave:
    def test_paper_reactance_pair_f1(self) -> None:
        """Faenzi 2019: transparent -1058 ohm <-> opaque 0.6*eta0 at 26.25 GHz.

        This test locks the sign conventions of the sheet/slab/opaque
        reactance chain against published values.
        """
        sol = solve_sw_transparent(-1058.0, 26.25e9, EPS_R, H)
        assert sol.x_op == pytest.approx(0.6 * ETA_0, rel=5e-3)

    def test_paper_reactance_pair_f2(self) -> None:
        """Faenzi 2019: transparent -796 ohm <-> opaque 1.1*eta0 at 32.05 GHz."""
        sol = solve_sw_transparent(-796.0, 32.05e9, EPS_R, H)
        assert sol.x_op == pytest.approx(1.1 * ETA_0, rel=5e-3)

    def test_sw_wavenumber_formula(self) -> None:
        freq = 20e9
        x_op = 0.8 * ETA_0
        beta = sw_wavenumber_tm(x_op, freq)
        assert beta / k0(freq) == pytest.approx(np.sqrt(1 + 0.8**2))

    def test_sw_wavenumber_rejects_capacitive(self) -> None:
        with pytest.raises(ValueError, match="inductive"):
            sw_wavenumber_tm(-100.0, 20e9)

    def test_transparent_opaque_roundtrip(self) -> None:
        x_slab = 190.0
        x_sheet = -900.0
        x_op = transparent_to_opaque(x_sheet, x_slab)
        assert opaque_to_transparent(x_op, x_slab) == pytest.approx(x_sheet)

    def test_transverse_resonance_consistency(self) -> None:
        """The solved beta satisfies beta = k0*sqrt(1 + (x_op/eta0)^2)."""
        sol = solve_sw_transparent(-1058.0, 26.25e9, EPS_R, H)
        assert sol.beta == pytest.approx(sw_wavenumber_tm(sol.x_op, sol.freq), rel=1e-9)

    def test_slab_reactance_inductive_when_thin(self) -> None:
        freq = 26.25e9
        beta = 1.1 * k0(freq)
        assert grounded_slab_reactance_tm(freq, EPS_R, H, beta) > 0


class TestModulatedDispersion:
    @pytest.fixture
    def setup(self) -> tuple[float, float, float]:
        freq = 26.25e9
        sol = solve_sw_transparent(-1058.0, freq, EPS_R, H)
        period = 2 * np.pi / sol.beta
        return sol.x_op, period, freq

    def test_zero_modulation_limit(self, setup: tuple[float, float, float]) -> None:
        x_op, period, freq = setup
        mode = dispersion_modulated_reactance(x_op, 0.0, period, freq)
        assert mode.alpha == 0.0
        assert mode.beta == pytest.approx(sw_wavenumber_tm(x_op, freq))
        assert mode.converged

    def test_small_m_quadratic_scaling(self, setup: tuple[float, float, float]) -> None:
        """Oliner-Hessel perturbation limit: alpha and beta_delta scale as m^2."""
        x_op, period, freq = setup
        m1 = dispersion_modulated_reactance(x_op, 0.04, period, freq)
        m2 = dispersion_modulated_reactance(x_op, 0.08, period, freq)
        assert m2.alpha / m1.alpha == pytest.approx(4.0, rel=0.05)
        assert m2.beta_delta / m1.beta_delta == pytest.approx(4.0, rel=0.05)

    def test_converges_at_large_m(self, setup: tuple[float, float, float]) -> None:
        x_op, period, freq = setup
        mode = dispersion_modulated_reactance(x_op, 0.4, period, freq)
        assert mode.converged
        assert mode.alpha > 0
        assert mode.beta > mode.beta_sw  # inductive modulation slows the wave

    def test_alpha_positive_and_monotonic(
        self, setup: tuple[float, float, float]
    ) -> None:
        x_op, period, freq = setup
        ds = dispersion_map(x_op, np.linspace(0.0, 0.4, 9), period, freq)
        alpha = ds["alpha_over_k0"].values
        assert np.all(alpha >= 0)
        assert np.all(np.diff(alpha) > -1e-12)
        assert bool(ds["converged"].all())

    def test_invalid_args(self, setup: tuple[float, float, float]) -> None:
        x_op, period, freq = setup
        with pytest.raises(ValueError, match="modulation index"):
            dispersion_modulated_reactance(x_op, 1.5, period, freq)
        with pytest.raises(ValueError, match="period"):
            dispersion_modulated_reactance(x_op, 0.2, -1.0, freq)


class TestTaperSynthesis:
    def test_energy_accounting(self) -> None:
        """1 - exp(-2*int(alpha)) equals the requested radiation efficiency."""
        rho = np.linspace(1e-4, 0.1, 4001)
        amplitude = np.exp(-((rho - 0.04) ** 2) / (2 * 0.02**2))
        eta = 0.7
        alpha = alpha_for_taper(rho, amplitude, efficiency=eta)
        attenuation = np.exp(-2.0 * np.trapezoid(alpha, rho))
        assert 1.0 - attenuation == pytest.approx(eta, rel=1e-3)

    def test_uniform_amplitude_alpha_grows(self) -> None:
        rho = np.linspace(1e-4, 0.1, 512)
        alpha = alpha_for_taper(rho, np.ones_like(rho), efficiency=0.8)
        assert np.all(alpha > 0)
        assert alpha[-1] > alpha[0]

    def test_invalid_efficiency(self) -> None:
        rho = np.linspace(1e-4, 0.1, 16)
        with pytest.raises(ValueError, match="efficiency"):
            alpha_for_taper(rho, np.ones_like(rho), efficiency=1.5)


class TestBandwidthFormulas:
    def test_group_velocity_physical(self) -> None:
        v = sw_group_velocity(-1058.0, 26.25e9, EPS_R, H)
        assert 0.0 < v < 1.0

    def test_gain_bandwidth_values(self) -> None:
        assert gain_bandwidth_product(0.5, 10.0, uniform=True) == pytest.approx(110.0)
        assert gain_bandwidth_product(0.5, 10.0, uniform=False) == pytest.approx(
            47.0 * 0.5 * 10.0 / 12.0
        )
        assert relative_bandwidth(0.5, 10.0) == pytest.approx(0.0475)
