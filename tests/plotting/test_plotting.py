"""Smoke tests for plotting module.

Tests that each plot function runs without error and returns the
expected type. Uses Agg backend to avoid display requirements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from metasurface_py.core.types import AngleGrid
from metasurface_py.elements import PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.em import far_field_pattern
from metasurface_py.geometry import RectangularLattice
from metasurface_py.plotting import (
    plot_convergence,
    plot_element_amplitude_phase,
    plot_gain_vs_frequency,
    plot_gain_vs_scan_angle,
    plot_lattice,
    plot_pattern_2d,
    plot_pattern_3d,
    plot_pattern_comparison,
    plot_pattern_polar,
    plot_pattern_uv,
    plot_state_map,
    save_figure,
    set_publication_style,
)
from metasurface_py.surfaces import Metasurface

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def small_pattern() -> Any:
    """Create a small pattern for testing."""
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
    return far_field_pattern(
        surface,
        state,
        freq=10e9,
        angles=angles,
    )


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    """Close all figures after each test."""
    yield
    plt.close("all")


class TestPatternPlots:
    def test_plot_pattern_2d(self, small_pattern: Any) -> None:
        ax = plot_pattern_2d(small_pattern, cut_phi=0.0)
        assert ax is not None

    def test_plot_pattern_polar(
        self,
        small_pattern: Any,
    ) -> None:
        ax = plot_pattern_polar(small_pattern, cut_phi=0.0)
        assert ax is not None

    def test_plot_pattern_uv(self, small_pattern: Any) -> None:
        ax, mesh = plot_pattern_uv(small_pattern)
        assert ax is not None
        assert mesh is not None

    def test_plot_pattern_comparison(
        self,
        small_pattern: Any,
    ) -> None:
        ax = plot_pattern_comparison(
            [(small_pattern, "A"), (small_pattern, "B")],
            cut_phi=0.0,
        )
        assert ax is not None

    def test_plot_pattern_3d(self, small_pattern: Any) -> None:
        # Need full theta/phi coverage for 3D
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
            theta=np.linspace(0.1, np.pi - 0.1, 20),
            phi=np.linspace(0, 2 * np.pi - 0.1, 20),
        )
        pattern = far_field_pattern(
            surface,
            state,
            freq=10e9,
            angles=angles,
        )
        fig, ax = plot_pattern_3d(pattern)
        assert fig is not None
        assert ax is not None


class TestGeometryPlots:
    def test_plot_lattice(self) -> None:
        lattice = RectangularLattice(
            nx=4,
            ny=4,
            dx=0.01,
            dy=0.01,
        )
        ax = plot_lattice(lattice)
        assert ax is not None

    def test_plot_state_map(self) -> None:
        from metasurface_py.surfaces.state import SurfaceState

        state = SurfaceState(
            values=np.linspace(0, 2 * np.pi, 16),
            space=ContinuousPhaseSpace(),
        )
        ax, _mesh = plot_state_map(state, nx=4, ny=4)
        assert ax is not None

    def test_plot_element_amplitude_phase(self) -> None:
        response = np.exp(1j * np.linspace(0, 2 * np.pi, 16))
        fig = plot_element_amplitude_phase(response, nx=4, ny=4)
        assert fig is not None


class TestSweepPlots:
    def test_plot_gain_vs_scan_angle(self) -> None:
        data = {"A": [(0, 10.0), (30, 8.0), (60, 5.0)]}
        ax = plot_gain_vs_scan_angle(data)
        assert ax is not None

    def test_plot_gain_vs_frequency(self) -> None:
        data = {"A": [(9.0, 10.0), (10.0, 12.0), (11.0, 10.5)]}
        ax = plot_gain_vs_frequency(data)
        assert ax is not None


class TestConvergence:
    def test_plot_convergence(self) -> None:
        data = {"Run 1": np.array([10.0, 8.0, 6.0, 5.5])}
        ax = plot_convergence(data)
        assert ax is not None


class TestStyle:
    def test_set_publication_style(self) -> None:
        set_publication_style()
        assert plt.rcParams["font.size"] == 10.0

    def test_set_publication_style_poster(self) -> None:
        set_publication_style(target="poster")
        assert plt.rcParams["font.size"] > 10.0

    def test_save_figure(self, tmp_path: Path) -> None:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        paths = save_figure(fig, tmp_path / "test_fig")
        assert len(paths) == 2
        assert (tmp_path / "test_fig.pdf").exists()
        assert (tmp_path / "test_fig.png").exists()
