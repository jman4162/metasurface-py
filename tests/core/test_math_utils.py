"""Tests for core math utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metasurface_py.core.math_utils import (
    cartesian_to_spherical,
    db10,
    db20,
    direction_cosines,
    from_db10,
    from_db20,
    normalize_phase,
    spherical_to_cartesian,
    steering_vector,
)


class TestDbConversions:
    def test_db20_scalar(self) -> None:
        assert db20(1.0) == pytest.approx(0.0)
        assert db20(10.0) == pytest.approx(20.0)
        assert db20(0.1) == pytest.approx(-20.0)

    def test_db20_roundtrip(self) -> None:
        val = 3.7
        assert from_db20(db20(val)) == pytest.approx(val)

    def test_db10_scalar(self) -> None:
        assert db10(1.0) == pytest.approx(0.0)
        assert db10(100.0) == pytest.approx(20.0)

    def test_db10_roundtrip(self) -> None:
        val = 42.0
        assert from_db10(db10(val)) == pytest.approx(val)

    def test_db20_array(self) -> None:
        arr = np.array([1.0, 10.0, 100.0])
        result = db20(arr)
        np.testing.assert_allclose(result, [0.0, 20.0, 40.0])


class TestNormalizePhase:
    def test_already_in_range(self) -> None:
        assert normalize_phase(0.5) == pytest.approx(0.5)

    def test_wrap_positive(self) -> None:
        assert normalize_phase(3 * math.pi) == pytest.approx(math.pi, abs=1e-10)

    def test_wrap_negative(self) -> None:
        # -3*pi wraps to -pi (equivalent to +pi, but angle() returns -pi)
        result = normalize_phase(-3 * math.pi)
        assert abs(result) == pytest.approx(math.pi, abs=1e-10)

    def test_array(self) -> None:
        phases = np.array([0.0, 2 * math.pi, -2 * math.pi, 5 * math.pi])
        result = normalize_phase(phases)
        expected = np.array([0.0, 0.0, 0.0, math.pi])
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestCoordinateTransforms:
    def test_spherical_to_cartesian_zenith(self) -> None:
        x, y, z = spherical_to_cartesian(1.0, 0.0, 0.0)
        np.testing.assert_allclose([x, y, z], [0.0, 0.0, 1.0], atol=1e-15)

    def test_spherical_to_cartesian_x_axis(self) -> None:
        x, y, z = spherical_to_cartesian(1.0, math.pi / 2, 0.0)
        np.testing.assert_allclose([x, y, z], [1.0, 0.0, 0.0], atol=1e-15)

    def test_spherical_cartesian_roundtrip(self) -> None:
        r0, th0, ph0 = 3.5, 1.2, 0.7
        x, y, z = spherical_to_cartesian(r0, th0, ph0)
        r, th, ph = cartesian_to_spherical(x, y, z)
        np.testing.assert_allclose([r, th, ph], [r0, th0, ph0], atol=1e-12)

    def test_cartesian_to_spherical_origin(self) -> None:
        r, _theta, _phi = cartesian_to_spherical(0.0, 0.0, 0.0)
        assert r == pytest.approx(0.0, abs=1e-15)


class TestDirectionCosines:
    def test_broadside(self) -> None:
        u, v, w = direction_cosines(0.0, 0.0)
        np.testing.assert_allclose([u, v, w], [0.0, 0.0, 1.0], atol=1e-15)

    def test_horizon_phi0(self) -> None:
        u, v, w = direction_cosines(math.pi / 2, 0.0)
        np.testing.assert_allclose([u, v, w], [1.0, 0.0, 0.0], atol=1e-15)


class TestSteeringVector:
    def test_broadside_uniform(self) -> None:
        """Broadside steering should give all-ones vector."""
        positions = np.array([[0, 0, 0], [0.5, 0, 0], [1.0, 0, 0]], dtype=np.float64)
        k_vec = np.array([0, 0, 0], dtype=np.float64)  # broadside = zero k_vec
        sv = steering_vector(positions, k_vec)
        np.testing.assert_allclose(sv, [1.0, 1.0, 1.0])

    def test_shape(self) -> None:
        n = 10
        positions = np.random.default_rng(42).random((n, 3))
        k_vec = np.array([1.0, 0.0, 0.0])
        sv = steering_vector(positions, k_vec)
        assert sv.shape == (n,)

    def test_unit_magnitude(self) -> None:
        """Steering vector entries should have unit magnitude."""
        positions = np.random.default_rng(42).random((5, 3))
        k_vec = np.array([2.0, 1.0, 0.5])
        sv = steering_vector(positions, k_vec)
        np.testing.assert_allclose(np.abs(sv), 1.0)
