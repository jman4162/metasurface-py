"""Tests for optimization module."""

from __future__ import annotations

import numpy as np
import pytest

from metasurface_py.core.types import AngleGrid
from metasurface_py.elements import DiscretePhaseSpace, PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.em import steering_phase
from metasurface_py.geometry import RectangularLattice
from metasurface_py.optimize import (
    MaxGainObjective,
    OptimizationResult,
    optimize_continuous,
    refine_discrete,
    relax_then_quantize,
)
from metasurface_py.surfaces import Metasurface


@pytest.fixture()
def small_surface() -> tuple[Metasurface, float, AngleGrid]:
    """8x8 surface at 10 GHz for fast tests."""
    freq = 10e9
    lam = 3e8 / freq
    lattice = RectangularLattice(
        nx=8,
        ny=8,
        dx=lam / 2,
        dy=lam / 2,
    )
    cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
    surface = Metasurface(lattice=lattice, cell=cell)
    angles = AngleGrid(
        theta=np.linspace(0.01, np.pi - 0.01, 90),
        phi=np.linspace(0, 2 * np.pi - 0.01, 36),
    )
    return surface, freq, angles


@pytest.fixture()
def small_discrete_surface() -> tuple[Metasurface, float, AngleGrid]:
    """8x8 discrete surface at 10 GHz."""
    freq = 10e9
    lam = 3e8 / freq
    lattice = RectangularLattice(
        nx=8,
        ny=8,
        dx=lam / 2,
        dy=lam / 2,
    )
    cell = PhaseOnlyCell(
        state_space=DiscretePhaseSpace(num_bits=2),
    )
    surface = Metasurface(lattice=lattice, cell=cell)
    angles = AngleGrid(
        theta=np.linspace(0.01, np.pi - 0.01, 90),
        phi=np.linspace(0, 2 * np.pi - 0.01, 36),
    )
    return surface, freq, angles


class TestMaxGainObjective:
    def test_steering_phase_is_near_optimal(
        self,
        small_surface: tuple[Metasurface, float, AngleGrid],
    ) -> None:
        """Analytical steering phase should give near-optimal gain."""
        surface, freq, angles = small_surface
        target_theta = np.radians(20.0)
        obj = MaxGainObjective(target_theta, 0.0, angles)

        # Analytical steering
        phase = steering_phase(
            surface.lattice,
            theta_steer=target_theta,
            phi_steer=0.0,
            freq=freq,
        )
        analytical_val = obj(phase, surface, freq)

        # Random phase
        rng = np.random.default_rng(42)
        random_phase = rng.uniform(0, 2 * np.pi, 64)
        random_val = obj(random_phase, surface, freq)

        # Analytical should be better (more negative = higher gain)
        assert analytical_val < random_val

    def test_returns_negative_gain(
        self,
        small_surface: tuple[Metasurface, float, AngleGrid],
    ) -> None:
        surface, freq, angles = small_surface
        obj = MaxGainObjective(np.radians(0.0), 0.0, angles)
        val = obj(np.zeros(64), surface, freq)
        assert val < 0  # gain should be positive -> objective negative


class TestOptimizeContinuous:
    def test_improves_over_random(
        self,
        small_surface: tuple[Metasurface, float, AngleGrid],
    ) -> None:
        """Optimization should improve over random initial state."""
        surface, freq, angles = small_surface
        target_theta = np.radians(20.0)
        obj = MaxGainObjective(target_theta, 0.0, angles)

        result = optimize_continuous(
            obj,
            surface,
            freq,
            angles,
            method="L-BFGS-B",
            maxiter=50,
            seed=42,
        )

        assert isinstance(result, OptimizationResult)
        assert result.runtime_seconds > 0
        assert len(result.convergence_history) > 0
        # Final should be better than initial
        assert result.objective_value <= result.convergence_history[0]

    def test_result_has_correct_state_size(
        self,
        small_surface: tuple[Metasurface, float, AngleGrid],
    ) -> None:
        surface, freq, angles = small_surface
        obj = MaxGainObjective(np.radians(10.0), 0.0, angles)
        result = optimize_continuous(
            obj,
            surface,
            freq,
            angles,
            maxiter=10,
            seed=0,
        )
        assert result.state.num_elements == 64


class TestRefineDiscrete:
    def test_improves_or_maintains(
        self,
        small_discrete_surface: tuple[Metasurface, float, AngleGrid],
    ) -> None:
        """Discrete refinement should not degrade the objective."""
        surface, freq, angles = small_discrete_surface
        target_theta = np.radians(20.0)
        obj = MaxGainObjective(target_theta, 0.0, angles)

        # Start from steering phase quantized
        phase = steering_phase(
            surface.lattice,
            theta_steer=target_theta,
            phi_steer=0.0,
            freq=freq,
        )
        state = surface.set_state(phase).quantize()
        initial_val = obj(state.values, surface, freq)

        result = refine_discrete(
            obj,
            surface,
            state,
            freq,
            angles,
            max_sweeps=1,
        )

        assert result.objective_value <= initial_val + 1e-10


class TestRelaxThenQuantize:
    def test_full_pipeline(
        self,
        small_discrete_surface: tuple[Metasurface, float, AngleGrid],
    ) -> None:
        """Full pipeline should produce a valid discrete state."""
        surface, freq, angles = small_discrete_surface
        target_theta = np.radians(20.0)
        obj = MaxGainObjective(target_theta, 0.0, angles)

        result = relax_then_quantize(
            obj,
            surface,
            freq,
            angles,
            continuous_method="L-BFGS-B",
            refine=True,
            maxiter=30,
            seed=42,
        )

        assert isinstance(result, OptimizationResult)
        assert result.state_continuous is not None
        assert result.runtime_seconds > 0
        assert len(result.convergence_history) > 0

    def test_outperforms_random(
        self,
        small_discrete_surface: tuple[Metasurface, float, AngleGrid],
    ) -> None:
        """Optimized discrete state should beat random."""
        surface, freq, angles = small_discrete_surface
        target_theta = np.radians(20.0)
        obj = MaxGainObjective(target_theta, 0.0, angles)

        result = relax_then_quantize(
            obj,
            surface,
            freq,
            angles,
            maxiter=30,
            seed=42,
        )

        # Compare against random
        rng = np.random.default_rng(99)
        random_vals = []
        for _ in range(10):
            rp = rng.uniform(0, 2 * np.pi, 64)
            random_vals.append(obj(rp, surface, freq))

        best_random = min(random_vals)
        assert result.objective_value <= best_random
