"""Tests for element state spaces and quantization."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metasurface_py.elements.states import (
    ContinuousPhaseSpace,
    CustomCodebook,
    DiscretePhaseSpace,
    quantize,
    random_state,
)


class TestDiscretePhaseSpace:
    def test_1bit_has_2_states(self) -> None:
        space = DiscretePhaseSpace(num_bits=1)
        assert space.codebook is not None
        assert len(space.codebook) == 2

    def test_2bit_has_4_states(self) -> None:
        space = DiscretePhaseSpace(num_bits=2)
        assert space.codebook is not None
        assert len(space.codebook) == 4

    def test_codebook_on_unit_circle(self) -> None:
        space = DiscretePhaseSpace(num_bits=3)
        assert space.codebook is not None
        np.testing.assert_allclose(np.abs(space.codebook), 1.0)

    def test_uniform_spacing(self) -> None:
        space = DiscretePhaseSpace(num_bits=2)
        assert space.codebook is not None
        phases = np.angle(space.codebook)
        # Check spacing via phasor differences (avoids wrapping issues)
        angular_diffs = np.abs(np.diff(np.exp(1j * phases)))
        # For pi/2 spacing, |exp(j*pi/2) - exp(j*0)| = sqrt(2)
        np.testing.assert_allclose(angular_diffs, math.sqrt(2), atol=1e-10)


class TestContinuousPhaseSpace:
    def test_default_bounds(self) -> None:
        space = ContinuousPhaseSpace()
        assert space.bounds == (0.0, 2 * math.pi)

    def test_custom_bounds(self) -> None:
        space = ContinuousPhaseSpace(bounds=(-math.pi, math.pi))
        assert space.bounds == (-math.pi, math.pi)

    def test_kind(self) -> None:
        space = ContinuousPhaseSpace()
        assert space.kind == "continuous"


class TestCustomCodebook:
    def test_from_values(self) -> None:
        vals = np.array([1.0, -1.0, 1j, -1j], dtype=np.complex128)
        space = CustomCodebook(vals)
        assert space.codebook is not None
        assert len(space.codebook) == 4
        assert space.kind == "discrete"


class TestQuantize:
    def test_exact_match(self) -> None:
        space = DiscretePhaseSpace(num_bits=2)
        assert space.codebook is not None
        phases = np.angle(space.codebook)  # exactly on codebook
        result = quantize(phases, space.codebook)
        np.testing.assert_allclose(result, phases, atol=1e-10)

    def test_nearest_neighbor(self) -> None:
        space = DiscretePhaseSpace(num_bits=1)  # 0, pi
        assert space.codebook is not None
        # Phase close to 0 should map to 0, close to pi should map to pi
        phases = np.array([0.1, 2.9])
        result = quantize(phases, space.codebook)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-10)
        assert abs(result[1]) == pytest.approx(math.pi, abs=1e-10)

    def test_output_shape(self) -> None:
        space = DiscretePhaseSpace(num_bits=2)
        assert space.codebook is not None
        phases = np.random.default_rng(0).uniform(0, 2 * math.pi, 100)
        result = quantize(phases, space.codebook)
        assert result.shape == (100,)


class TestRandomState:
    def test_continuous_shape(self) -> None:
        space = ContinuousPhaseSpace()
        state = random_state(space, 50, rng=np.random.default_rng(42))
        assert state.shape == (50,)

    def test_continuous_in_bounds(self) -> None:
        space = ContinuousPhaseSpace(bounds=(0.0, 2 * math.pi))
        state = random_state(space, 1000, rng=np.random.default_rng(42))
        assert np.all(state >= 0.0)
        assert np.all(state <= 2 * math.pi)

    def test_discrete_shape(self) -> None:
        space = DiscretePhaseSpace(num_bits=2)
        state = random_state(space, 50, rng=np.random.default_rng(42))
        assert state.shape == (50,)

    def test_discrete_values_in_codebook(self) -> None:
        space = DiscretePhaseSpace(num_bits=2)
        assert space.codebook is not None
        state = random_state(space, 100, rng=np.random.default_rng(42))
        codebook_phases = np.angle(space.codebook)
        for val in state:
            assert any(abs(val - cp) < 1e-10 for cp in codebook_phases)
