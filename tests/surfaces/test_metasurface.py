"""Tests for Metasurface and SurfaceState."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metasurface_py.elements.phase_cell import PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace, DiscretePhaseSpace
from metasurface_py.geometry.lattice import RectangularLattice
from metasurface_py.surfaces.metasurface import Metasurface
from metasurface_py.surfaces.state import SurfaceState


class TestMetasurface:
    def test_num_elements(self) -> None:
        lat = RectangularLattice(nx=8, ny=8, dx=0.005, dy=0.005)
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        ms = Metasurface(lattice=lat, cell=cell, mode="reflect")
        assert ms.num_elements == 64

    def test_set_state(self) -> None:
        lat = RectangularLattice(nx=4, ny=4, dx=0.01, dy=0.01)
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        ms = Metasurface(lattice=lat, cell=cell)
        phase = np.zeros(16)
        state = ms.set_state(phase)
        assert state.num_elements == 16

    def test_set_state_wrong_size_raises(self) -> None:
        lat = RectangularLattice(nx=4, ny=4, dx=0.01, dy=0.01)
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        ms = Metasurface(lattice=lat, cell=cell)
        with pytest.raises(ValueError, match="Expected 16"):
            ms.set_state(np.zeros(10))

    def test_response(self) -> None:
        lat = RectangularLattice(nx=4, ny=4, dx=0.01, dy=0.01)
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        ms = Metasurface(lattice=lat, cell=cell)
        state = ms.set_state(np.zeros(16))
        resp = ms.response(state, freq=28e9)
        np.testing.assert_allclose(np.abs(resp), 1.0)

    def test_2d_phase_input(self) -> None:
        lat = RectangularLattice(nx=4, ny=4, dx=0.01, dy=0.01)
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        ms = Metasurface(lattice=lat, cell=cell)
        phase_2d = np.zeros((4, 4))
        state = ms.set_state(phase_2d)
        assert state.num_elements == 16


class TestSurfaceState:
    def test_quantize(self) -> None:
        space = DiscretePhaseSpace(num_bits=1)
        values = np.array([0.1, 3.0, 0.5, 2.5])
        state = SurfaceState(values=values, space=space)
        assert space.codebook is not None
        quantized = state.quantize(space.codebook)
        # All values should be either 0 or pi
        for v in quantized.values:
            assert v == pytest.approx(0.0, abs=1e-10) or abs(v) == pytest.approx(
                math.pi, abs=1e-10
            )

    def test_with_defects(self) -> None:
        space = ContinuousPhaseSpace()
        values = np.array([1.0, 2.0, 3.0, 4.0])
        state = SurfaceState(values=values, space=space)
        defect_mask = np.array([True, False, True, False])
        result = state.with_defects(defect_mask)
        assert result.values[0] == pytest.approx(1.0)
        assert result.values[1] == pytest.approx(0.0)
        assert result.values[2] == pytest.approx(3.0)
        assert result.values[3] == pytest.approx(0.0)

    def test_apply_grouping(self) -> None:
        space = ContinuousPhaseSpace()
        values = np.array([0.0, 0.1, math.pi, math.pi + 0.1])
        state = SurfaceState(values=values, space=space)
        groups = np.array([0, 0, 1, 1])
        result = state.apply_grouping(groups)
        # Group 0: mean of 0.0 and 0.1
        assert result.values[0] == pytest.approx(result.values[1])
        # Group 1: mean of pi and pi+0.1
        assert result.values[2] == pytest.approx(result.values[3])
