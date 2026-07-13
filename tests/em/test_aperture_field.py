"""Tests for AFM aperture currents and the vector radiation integral."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import j1

from metasurface_py.core.conventions import wavelength
from metasurface_py.core.types import AngleGrid
from metasurface_py.em.aperture_field import (
    afm_surface_current,
    aperture_flux_power,
    pattern_directivity,
    radiate_aperture,
    radiated_power,
    total_field,
)
from metasurface_py.geometry.aperture import CircularAperture

FREQ = 30e9
LAM = wavelength(FREQ)


def _uniform_aperture_pattern(
    radius: float, n_rho: int = 96, n_phi: int = 96
) -> tuple[CircularAperture, AngleGrid, object]:
    ap = CircularAperture(radius=radius, n_rho=n_rho, n_phi=n_phi)
    n = ap.positions.shape[0]
    e_x = np.ones(n, dtype=complex)
    e_y = np.zeros(n, dtype=complex)
    angles = AngleGrid.from_degrees(
        theta=np.linspace(0, 90, 181), phi=np.linspace(0, 355, 72)
    )
    ff = radiate_aperture(ap.positions, e_x, e_y, ap.cell_areas.ravel(), FREQ, angles)
    return ap, angles, ff


class TestUniformCircularAperture:
    """Analytic oracle: uniform aperture radiates the Airy pattern 2*J1(x)/x."""

    def test_pattern_matches_airy(self) -> None:
        radius = 3 * LAM
        _, angles, ff = _uniform_aperture_pattern(radius)
        e_theta = ff["E_theta"].values[:, 0]  # phi = 0 cut, x-polarized -> E_theta
        x = 2 * np.pi * radius / LAM * np.sin(angles.theta)
        airy = np.where(x == 0, 1.0, 2 * j1(x) / np.maximum(x, 1e-12))
        measured = np.abs(e_theta) / np.abs(e_theta[0])
        mask = angles.theta < np.deg2rad(30)
        np.testing.assert_allclose(measured[mask], np.abs(airy[mask]), atol=2e-4)

    def test_peak_directivity_near_ideal(self) -> None:
        """D approaches 4*pi*A/lambda^2; ~2% of spectrum is evanescent at a=3lam."""
        radius = 3 * LAM
        _, _, ff = _uniform_aperture_pattern(radius)
        d = pattern_directivity(ff)
        peak_db = 10 * np.log10(float(d.max()))
        ideal_db = 10 * np.log10(4 * np.pi * np.pi * radius**2 / LAM**2)
        assert peak_db == pytest.approx(ideal_db, abs=0.15)

    def test_power_conservation(self) -> None:
        """Radiated power matches aperture Poynting flux (large-aperture limit)."""
        radius = 3 * LAM
        ap, _, ff = _uniform_aperture_pattern(radius)
        n = ap.positions.shape[0]
        p_ap = aperture_flux_power(
            np.ones(n, dtype=complex),
            np.zeros(n, dtype=complex),
            ap.cell_areas.ravel(),
        )
        assert radiated_power(ff) / p_ap == pytest.approx(1.0, abs=0.05)

    def test_output_dataset_structure(self) -> None:
        _, _, ff = _uniform_aperture_pattern(2 * LAM, n_rho=32, n_phi=32)
        assert set(ff.data_vars) == {"E_theta", "E_phi"}
        assert ff["E_theta"].dims == ("theta", "phi")
        assert ff.attrs["freq_hz"] == FREQ
        tf = total_field(ff)
        assert tf.dims == ("theta", "phi")
        assert np.all(tf.values >= 0)


class TestAfmSurfaceCurrent:
    def test_leakage_decay(self) -> None:
        """rho*|J|^2 tracks exp(-2*int(alpha)) in the asymptotic region."""
        beta = 600.0
        alpha = 5.0
        rho = np.linspace(1e-3, 0.5, 2000)
        k_local = np.full(rho.shape, beta - 1j * alpha, dtype=complex)
        j = afm_surface_current(rho, k_local)
        power_flow = rho * np.abs(j) ** 2
        expected = np.exp(-2 * alpha * rho)
        # Compare in the asymptotic region rho >> 1/beta, normalized mid-range.
        mask = rho > 0.1
        ratio = power_flow[mask] / expected[mask]
        np.testing.assert_allclose(ratio / ratio.mean(), 1.0, rtol=1e-2)

    def test_lossless_current_conserves_cylindrical_power(self) -> None:
        beta = 600.0
        rho = np.linspace(1e-3, 0.5, 2000)
        k_local = np.full(rho.shape, beta + 0j, dtype=complex)
        j = afm_surface_current(rho, k_local)
        power_flow = rho * np.abs(j) ** 2
        mask = rho > 0.1
        np.testing.assert_allclose(
            power_flow[mask] / power_flow[mask].mean(), 1.0, rtol=1e-2
        )
