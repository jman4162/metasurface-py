"""Tests for circular-polarization decomposition and axial ratio."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from metasurface_py.core.polarization import axial_ratio_db, circular_components


def _pattern(values: np.ndarray) -> xr.DataArray:  # type: ignore[type-arg]
    theta = np.linspace(0.0, np.pi / 2, values.shape[0])
    phi = np.linspace(0.0, 2 * np.pi, values.shape[1], endpoint=False)
    return xr.DataArray(
        values, dims=["theta", "phi"], coords={"theta": theta, "phi": phi}
    )


class TestCircularComponents:
    def test_pure_rhcp(self) -> None:
        """E_phi = -j*E_theta is RHCP under exp(+j*omega*t)."""
        e_theta = _pattern(np.ones((4, 8), dtype=complex))
        e_phi = _pattern(-1j * np.ones((4, 8), dtype=complex))
        cp = circular_components(e_theta, e_phi)
        assert float(np.abs(cp["e_lhcp"]).max()) < 1e-14
        # Power preserved: |E_R|^2 = |E_theta|^2 + |E_phi|^2 = 2
        np.testing.assert_allclose(np.abs(cp["e_rhcp"].values) ** 2, 2.0)

    def test_pure_lhcp(self) -> None:
        e_theta = _pattern(np.ones((4, 8), dtype=complex))
        e_phi = _pattern(1j * np.ones((4, 8), dtype=complex))
        cp = circular_components(e_theta, e_phi)
        assert float(np.abs(cp["e_rhcp"]).max()) < 1e-14

    def test_power_conservation(self) -> None:
        rng = np.random.default_rng(7)
        e_theta = _pattern(
            rng.standard_normal((5, 6)) + 1j * rng.standard_normal((5, 6))
        )
        e_phi = _pattern(rng.standard_normal((5, 6)) + 1j * rng.standard_normal((5, 6)))
        cp = circular_components(e_theta, e_phi)
        total_cp = np.abs(cp["e_rhcp"].values) ** 2 + np.abs(cp["e_lhcp"].values) ** 2
        total_lin = np.abs(e_theta.values) ** 2 + np.abs(e_phi.values) ** 2
        np.testing.assert_allclose(total_cp, total_lin, rtol=1e-12)


class TestAxialRatio:
    def test_pure_cp_is_zero_db(self) -> None:
        e_theta = _pattern(np.ones((3, 4), dtype=complex))
        e_phi = _pattern(-1j * np.ones((3, 4), dtype=complex))
        ar = axial_ratio_db(e_theta, e_phi)
        np.testing.assert_allclose(ar.values, 0.0, atol=1e-10)

    def test_linear_clips_at_floor(self) -> None:
        e_theta = _pattern(np.ones((3, 4), dtype=complex))
        e_phi = _pattern(np.zeros((3, 4), dtype=complex))
        ar = axial_ratio_db(e_theta, e_phi, floor_db=60.0)
        np.testing.assert_allclose(ar.values, 60.0)

    def test_elliptical_value(self) -> None:
        """|E_R| = 3, |E_L| = 1 gives AR = (3+1)/(3-1) = 2 -> 6.02 dB."""
        # E_theta = (E_R + E_L)/sqrt(2), E_phi = -j(E_R - E_L)/sqrt(2)
        e_r, e_l = 3.0, 1.0
        e_theta = _pattern(np.full((2, 2), (e_r + e_l) / np.sqrt(2), dtype=complex))
        e_phi = _pattern(np.full((2, 2), -1j * (e_r - e_l) / np.sqrt(2), dtype=complex))
        ar = axial_ratio_db(e_theta, e_phi)
        np.testing.assert_allclose(ar.values, 20 * np.log10(2.0), rtol=1e-10)


class TestGeometryAperture:
    def test_cell_areas_sum_to_disk(self) -> None:
        from metasurface_py.geometry.aperture import CircularAperture

        ap = CircularAperture(radius=0.05, n_rho=32, n_phi=16)
        assert ap.cell_areas.sum() == pytest.approx(np.pi * 0.05**2, rel=1e-12)

    def test_positions_shape_and_radius(self) -> None:
        from metasurface_py.geometry.aperture import CircularAperture

        ap = CircularAperture(radius=0.1, n_rho=8, n_phi=8)
        assert ap.positions.shape == (64, 3)
        r = np.hypot(ap.positions[:, 0], ap.positions[:, 1])
        assert float(r.max()) < 0.1
        assert np.all(ap.positions[:, 2] == 0.0)

    def test_invalid_args_raise(self) -> None:
        from metasurface_py.geometry.aperture import CircularAperture

        with pytest.raises(ValueError, match="radius"):
            CircularAperture(radius=-1.0)
        with pytest.raises(ValueError, match="n_rho"):
            CircularAperture(radius=1.0, n_rho=1)
