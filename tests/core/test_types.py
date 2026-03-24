"""Tests for core types module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metasurface_py.core.types import (
    AngleGrid,
    FrequencyGrid,
    Position3D,
    SphericalPosition,
)


class TestFrequencyGrid:
    def test_from_ghz(self) -> None:
        fg = FrequencyGrid.from_ghz([1.0, 2.0, 5.0])
        np.testing.assert_allclose(fg.values, [1e9, 2e9, 5e9])

    def test_wavelengths(self) -> None:
        fg = FrequencyGrid(values=np.array([1.0]))  # 1 Hz -> lambda = c
        assert fg.wavelengths[0] == pytest.approx(299_792_458.0)

    def test_k0(self) -> None:
        freq = 10e9
        fg = FrequencyGrid(values=np.array([freq]))
        expected_k = 2 * math.pi * freq / 299_792_458.0
        assert fg.k0[0] == pytest.approx(expected_k)

    def test_num_freqs(self) -> None:
        fg = FrequencyGrid(values=np.array([1e9, 2e9, 3e9]))
        assert fg.num_freqs == 3


class TestAngleGrid:
    def test_from_degrees(self) -> None:
        ag = AngleGrid.from_degrees(theta=[0.0, 45.0, 90.0], phi=[0.0, 180.0])
        np.testing.assert_allclose(ag.theta, np.radians([0, 45, 90]))
        np.testing.assert_allclose(ag.phi, np.radians([0, 180]))

    def test_theta_deg_property(self) -> None:
        ag = AngleGrid(theta=np.array([0.0, math.pi / 4]), phi=np.array([0.0]))
        np.testing.assert_allclose(ag.theta_deg, [0.0, 45.0])

    def test_phi_deg_property(self) -> None:
        ag = AngleGrid(theta=np.array([0.0]), phi=np.array([0.0, math.pi]))
        np.testing.assert_allclose(ag.phi_deg, [0.0, 180.0])


class TestPosition3D:
    def test_to_array(self) -> None:
        p = Position3D(1.0, 2.0, 3.0)
        np.testing.assert_array_equal(p.to_array(), [1.0, 2.0, 3.0])

    def test_distance_to(self) -> None:
        p1 = Position3D(0.0, 0.0, 0.0)
        p2 = Position3D(3.0, 4.0, 0.0)
        assert p1.distance_to(p2) == pytest.approx(5.0)

    def test_distance_to_self(self) -> None:
        p = Position3D(1.0, 2.0, 3.0)
        assert p.distance_to(p) == pytest.approx(0.0)

    def test_direction_to(self) -> None:
        p1 = Position3D(0.0, 0.0, 0.0)
        p2 = Position3D(1.0, 0.0, 0.0)
        d = p1.direction_to(p2)
        np.testing.assert_allclose(d, [1.0, 0.0, 0.0])

    def test_direction_to_coincident_raises(self) -> None:
        p = Position3D(1.0, 2.0, 3.0)
        with pytest.raises(ValueError, match="coincident"):
            p.direction_to(p)


class TestSphericalPosition:
    def test_to_cartesian_zenith(self) -> None:
        sp = SphericalPosition(r=1.0, theta=0.0, phi=0.0)
        cart = sp.to_cartesian()
        assert cart.x == pytest.approx(0.0, abs=1e-15)
        assert cart.y == pytest.approx(0.0, abs=1e-15)
        assert cart.z == pytest.approx(1.0)

    def test_to_cartesian_horizon(self) -> None:
        sp = SphericalPosition(r=2.0, theta=math.pi / 2, phi=0.0)
        cart = sp.to_cartesian()
        assert cart.x == pytest.approx(2.0)
        assert cart.y == pytest.approx(0.0, abs=1e-15)
        assert cart.z == pytest.approx(0.0, abs=1e-15)

    def test_from_degrees(self) -> None:
        sp = SphericalPosition.from_degrees(r=1.0, theta_deg=90.0, phi_deg=90.0)
        assert sp.theta == pytest.approx(math.pi / 2)
        assert sp.phi == pytest.approx(math.pi / 2)
