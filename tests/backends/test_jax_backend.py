"""Tests for JAX backend.

Tests skip gracefully if JAX is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from metasurface_py.backends.jax_backend import (  # noqa: E402
    array_factor_jax,
    max_gain_objective_jax,
)
from metasurface_py.core.conventions import k0  # noqa: E402
from metasurface_py.em.array_factor import array_factor  # noqa: E402


class TestJAXArrayFactor:
    def test_matches_numpy(self) -> None:
        """JAX array factor should match NumPy within 1e-6."""
        positions = np.array(
            [[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0]],
            dtype=np.float64,
        )
        weights = np.array(
            [1.0 + 0j, 1.0 + 0j, 1.0 + 0j],
            dtype=np.complex128,
        )
        kw = 200.0
        theta = np.linspace(0.1, 1.5, 10)
        phi = np.array([0.0, 1.0])

        np_result = array_factor(
            positions,
            weights,
            kw,
            theta,
            phi,
        )
        jax_result = array_factor_jax(
            positions,
            weights,
            kw,
            theta,
            phi,
        )

        np.testing.assert_allclose(
            np.array(jax_result),
            np_result,
            atol=1e-6,
        )

    def test_shape(self) -> None:
        positions = np.random.default_rng(0).random((5, 3))
        weights = np.ones(5, dtype=np.complex128)
        theta = np.linspace(0.1, 1.5, 20)
        phi = np.array([0.0])
        result = array_factor_jax(
            positions,
            weights,
            100.0,
            theta,
            phi,
        )
        assert np.array(result).shape == (20, 1)


class TestJAXGrad:
    def test_gradient_is_finite(self) -> None:
        """jax.grad should produce finite gradients."""
        positions = np.array(
            [[0, 0, 0], [0.015, 0, 0], [0, 0.015, 0]],
            dtype=np.float64,
        )
        kw = k0(10e9)
        theta = np.linspace(0.1, np.pi - 0.1, 30)
        phi = np.linspace(0, 2 * np.pi - 0.1, 12)

        def obj(state: jax.Array) -> jax.Array:
            return max_gain_objective_jax(
                state,
                positions,
                kw,
                0.5,
                0.0,
                theta,
                phi,
            )

        state = jax.numpy.array([0.0, 1.0, 2.0])
        grad = jax.grad(obj)(state)
        assert np.all(np.isfinite(np.array(grad)))
        assert np.any(np.array(grad) != 0)
