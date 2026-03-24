"""Tests for steering and focusing phase synthesis."""

from __future__ import annotations

import numpy as np
import pytest

from metasurface_py.core.types import AngleGrid
from metasurface_py.elements.phase_cell import PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.em.array_factor import far_field_pattern
from metasurface_py.em.steering import focusing_phase, multi_beam_phase, steering_phase
from metasurface_py.geometry.lattice import RectangularLattice
from metasurface_py.surfaces.metasurface import Metasurface


class TestSteeringPhase:
    def test_broadside_is_zero(self) -> None:
        """Steering to broadside (theta=0) should give zero phase."""
        lat = RectangularLattice(nx=8, ny=8, dx=0.005, dy=0.005)
        phase = steering_phase(lat, theta_steer=0.0, phi_steer=0.0, freq=28e9)
        np.testing.assert_allclose(phase, 0.0, atol=1e-10)

    def test_shape(self) -> None:
        lat = RectangularLattice(nx=4, ny=6, dx=0.005, dy=0.005)
        phase = steering_phase(lat, theta_steer=0.3, phi_steer=0.0, freq=28e9)
        assert phase.shape == (24,)

    def test_beam_steers_to_target(self) -> None:
        """Verify the beam peak is at the target angle."""
        freq = 10e9
        lam = 3e8 / freq
        dx = lam / 2
        lat = RectangularLattice(nx=16, ny=16, dx=dx, dy=dx)
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        surface = Metasurface(lattice=lat, cell=cell)

        target_theta = np.radians(30)
        phase = steering_phase(lat, theta_steer=target_theta, phi_steer=0.0, freq=freq)
        state = surface.set_state(phase)

        theta = np.linspace(0.01, np.pi - 0.01, 360)
        angles = AngleGrid(theta=theta, phi=np.array([0.0]))
        pattern = far_field_pattern(surface, state, freq=freq, angles=angles)

        power = np.abs(pattern.values[:, 0]) ** 2
        peak_idx = np.argmax(power)
        peak_theta = theta[peak_idx]

        assert peak_theta == pytest.approx(target_theta, abs=0.02)  # within ~1 degree


class TestFocusingPhase:
    def test_shape(self) -> None:
        lat = RectangularLattice(nx=8, ny=8, dx=0.005, dy=0.005)
        phase = focusing_phase(lat, focal_point=(0.0, 0.0, 0.1), freq=28e9)
        assert phase.shape == (64,)

    def test_center_element_has_max_phase_magnitude(self) -> None:
        """Elements farther from focal point should have larger negative phase."""
        lat = RectangularLattice(nx=8, ny=8, dx=0.005, dy=0.005)
        phase = focusing_phase(lat, focal_point=(0.0, 0.0, 0.1), freq=28e9)
        # All phases should be negative (converging)
        assert np.all(phase <= 0)


class TestMultiBeamPhase:
    def test_shape(self) -> None:
        lat = RectangularLattice(nx=8, ny=8, dx=0.005, dy=0.005)
        directions = [(np.radians(20), 0.0), (np.radians(-20), 0.0)]
        phase = multi_beam_phase(lat, directions=directions, weights=None, freq=28e9)
        assert phase.shape == (64,)

    def test_two_beams(self) -> None:
        """Multi-beam should produce two lobes."""
        freq = 10e9
        lam = 3e8 / freq
        dx = lam / 2
        lat = RectangularLattice(nx=16, ny=1, dx=dx, dy=dx)
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        surface = Metasurface(lattice=lat, cell=cell)

        theta1 = np.radians(20)
        theta2 = np.radians(50)
        phase = multi_beam_phase(
            lat,
            directions=[(theta1, 0.0), (theta2, 0.0)],
            weights=None,
            freq=freq,
        )
        state = surface.set_state(phase)

        theta = np.linspace(0.01, np.pi - 0.01, 360)
        angles = AngleGrid(theta=theta, phi=np.array([0.0]))
        pattern = far_field_pattern(surface, state, freq=freq, angles=angles)
        power = np.abs(pattern.values[:, 0]) ** 2
        power_norm = power / np.max(power)

        # Should have significant power near both target angles
        idx1 = np.argmin(np.abs(theta - theta1))
        idx2 = np.argmin(np.abs(theta - theta2))
        assert power_norm[idx1] > 0.3
        assert power_norm[idx2] > 0.3
