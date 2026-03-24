"""Tests for surface constraints."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metasurface_py.surfaces.constraints import (
    add_manufacturing_noise,
    apply_group_constraint,
    apply_mask,
    phase_quantize,
)


class TestPhaseQuantize:
    def test_1bit(self) -> None:
        state = np.array([0.1, 3.0])
        result = phase_quantize(state, num_bits=1)
        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert abs(result[1]) == pytest.approx(math.pi, abs=1e-10)

    def test_2bit(self) -> None:
        state = np.array([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
        result = phase_quantize(state, num_bits=2)
        np.testing.assert_allclose(np.cos(result), np.cos(state), atol=1e-10)


class TestApplyMask:
    def test_zeros_inactive(self) -> None:
        state = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([True, False, True, False])
        result = apply_mask(state, mask)
        np.testing.assert_allclose(result, [1.0, 0.0, 3.0, 0.0])


class TestApplyGroupConstraint:
    def test_groups_share_phase(self) -> None:
        state = np.array([0.0, 0.2, math.pi, math.pi + 0.2])
        groups = np.array([0, 0, 1, 1])
        result = apply_group_constraint(state, groups)
        assert result[0] == pytest.approx(result[1])
        assert result[2] == pytest.approx(result[3])


class TestManufacturingNoise:
    def test_shape_preserved(self) -> None:
        state = np.zeros(100)
        rng = np.random.default_rng(42)
        result = add_manufacturing_noise(state, std_dev=0.1, rng=rng)
        assert result.shape == (100,)

    def test_noise_is_nonzero(self) -> None:
        state = np.zeros(100)
        rng = np.random.default_rng(42)
        result = add_manufacturing_noise(state, std_dev=0.1, rng=rng)
        assert np.std(result) > 0.05

    def test_zero_noise(self) -> None:
        state = np.ones(10)
        rng = np.random.default_rng(42)
        result = add_manufacturing_noise(state, std_dev=0.0, rng=rng)
        np.testing.assert_allclose(result, state)
