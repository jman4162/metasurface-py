"""Tests for PhaseOnlyCell."""

from __future__ import annotations

import math

import numpy as np

from metasurface_py.elements.phase_cell import PhaseOnlyCell
from metasurface_py.elements.protocols import UnitCellModel
from metasurface_py.elements.states import ContinuousPhaseSpace, DiscretePhaseSpace


class TestPhaseOnlyCell:
    def test_implements_protocol(self) -> None:
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        assert isinstance(cell, UnitCellModel)

    def test_continuous_num_states_none(self) -> None:
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        assert cell.num_states is None

    def test_discrete_num_states(self) -> None:
        cell = PhaseOnlyCell(state_space=DiscretePhaseSpace(num_bits=2))
        assert cell.num_states == 4

    def test_response_unit_amplitude(self) -> None:
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        phases = np.array([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
        resp = cell.response(phases, freq=28e9)
        np.testing.assert_allclose(np.abs(resp), 1.0)

    def test_response_correct_phase(self) -> None:
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        phases = np.array([0.0, math.pi / 4, math.pi / 2])
        resp = cell.response(phases, freq=28e9)
        np.testing.assert_allclose(np.angle(resp), phases, atol=1e-10)

    def test_lossy_amplitude(self) -> None:
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace(), amplitude=0.7)
        phases = np.array([0.0, math.pi])
        resp = cell.response(phases, freq=28e9)
        np.testing.assert_allclose(np.abs(resp), 0.7)

    def test_freq_independence(self) -> None:
        """PhaseOnlyCell should give same response at any frequency."""
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        phases = np.array([0.5, 1.0, 1.5])
        r1 = cell.response(phases, freq=1e9)
        r2 = cell.response(phases, freq=100e9)
        np.testing.assert_allclose(r1, r2)
