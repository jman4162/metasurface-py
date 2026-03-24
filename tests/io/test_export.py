"""Tests for io export utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from metasurface_py.core.types import AngleGrid
from metasurface_py.elements import PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.em import far_field_pattern
from metasurface_py.geometry import RectangularLattice
from metasurface_py.io import export_pattern_csv, export_state_csv
from metasurface_py.surfaces import Metasurface

if TYPE_CHECKING:
    from pathlib import Path


class TestExportState:
    def test_export_1d(self, tmp_path: Path) -> None:
        from metasurface_py.surfaces.state import SurfaceState

        state = SurfaceState(
            values=np.array([0.0, 1.0, 2.0, 3.0]),
            space=ContinuousPhaseSpace(),
        )
        out = export_state_csv(state, tmp_path / "state.csv")
        assert out.exists()
        data = np.loadtxt(out, delimiter=",")
        assert len(data) == 4

    def test_export_2d(self, tmp_path: Path) -> None:
        from metasurface_py.surfaces.state import SurfaceState

        state = SurfaceState(
            values=np.zeros(16),
            space=ContinuousPhaseSpace(),
        )
        out = export_state_csv(
            state,
            tmp_path / "state2d.csv",
            nx=4,
            ny=4,
        )
        assert out.exists()
        data = np.loadtxt(out, delimiter=",")
        assert data.shape == (4, 4)


class TestExportPattern:
    def test_export(self, tmp_path: Path) -> None:
        lattice = RectangularLattice(
            nx=4,
            ny=4,
            dx=0.01,
            dy=0.01,
        )
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        surface = Metasurface(lattice=lattice, cell=cell)
        state = surface.set_state(np.zeros(16))
        angles = AngleGrid(
            theta=np.linspace(0.1, 1.5, 10),
            phi=np.array([0.0, 1.0]),
        )
        pattern = far_field_pattern(
            surface,
            state,
            freq=10e9,
            angles=angles,
        )
        out = export_pattern_csv(
            pattern,
            tmp_path / "pattern.csv",
        )
        assert out.exists()
        data = np.loadtxt(out, delimiter=",")
        assert data.shape == (20, 4)  # 10 theta * 2 phi, 4 columns
