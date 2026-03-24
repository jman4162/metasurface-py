"""Tests for sensing module."""

from __future__ import annotations

import numpy as np
import pytest

from metasurface_py.core.types import AngleGrid, Position3D
from metasurface_py.elements import PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.geometry import RectangularLattice
from metasurface_py.sensing import (
    crlb_position,
    detection_snr,
    fisher_information_matrix,
    monostatic_rcs,
)
from metasurface_py.surfaces import Metasurface


@pytest.fixture()
def radar_surface() -> tuple[Metasurface, float]:
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
    return surface, freq


class TestMonostaticRCS:
    def test_shape(
        self,
        radar_surface: tuple[Metasurface, float],
    ) -> None:
        surface, freq = radar_surface
        state = surface.set_state(np.zeros(64))
        angles = AngleGrid(
            theta=np.linspace(0.1, 1.5, 20),
            phi=np.array([0.0]),
        )
        rcs = monostatic_rcs(surface, state, freq, angles)
        assert rcs.dims == ("theta", "phi")
        assert rcs.shape == (20, 1)

    def test_positive_values(
        self,
        radar_surface: tuple[Metasurface, float],
    ) -> None:
        surface, freq = radar_surface
        state = surface.set_state(np.zeros(64))
        angles = AngleGrid(
            theta=np.linspace(0.1, 1.5, 20),
            phi=np.array([0.0]),
        )
        rcs = monostatic_rcs(surface, state, freq, angles)
        assert np.all(rcs.values >= 0)


class TestDetectionSNR:
    def test_positive_snr(
        self,
        radar_surface: tuple[Metasurface, float],
    ) -> None:
        surface, freq = radar_surface
        state = surface.set_state(np.zeros(64))
        target = Position3D(x=0.0, y=0.0, z=100.0)
        snr = detection_snr(surface, state, target, freq)
        assert snr > 0

    def test_snr_decreases_with_distance(
        self,
        radar_surface: tuple[Metasurface, float],
    ) -> None:
        surface, freq = radar_surface
        state = surface.set_state(np.zeros(64))
        snr_near = detection_snr(
            surface,
            state,
            Position3D(z=50.0),
            freq,
        )
        snr_far = detection_snr(
            surface,
            state,
            Position3D(z=200.0),
            freq,
        )
        assert snr_near > snr_far


class TestFisherInformation:
    def test_fim_shape(
        self,
        radar_surface: tuple[Metasurface, float],
    ) -> None:
        surface, freq = radar_surface
        state = surface.set_state(np.zeros(64))
        target = Position3D(x=0.0, y=0.0, z=50.0)
        fim = fisher_information_matrix(
            surface,
            state,
            target,
            freq,
        )
        assert fim.shape == (3, 3)

    def test_fim_positive_semidefinite(
        self,
        radar_surface: tuple[Metasurface, float],
    ) -> None:
        surface, freq = radar_surface
        state = surface.set_state(np.zeros(64))
        target = Position3D(x=10.0, y=10.0, z=50.0)
        fim = fisher_information_matrix(
            surface,
            state,
            target,
            freq,
            snr=100.0,
        )
        eigenvalues = np.linalg.eigvalsh(fim)
        assert np.all(eigenvalues >= -1e-10)

    def test_crlb_decreases_with_snr(
        self,
        radar_surface: tuple[Metasurface, float],
    ) -> None:
        surface, freq = radar_surface
        state = surface.set_state(np.zeros(64))
        target = Position3D(x=10.0, y=10.0, z=50.0)
        crlb_low = crlb_position(
            surface,
            state,
            target,
            freq,
            snr=10.0,
        )
        crlb_high = crlb_position(
            surface,
            state,
            target,
            freq,
            snr=100.0,
        )
        # Higher SNR should give lower CRLB
        assert np.all(crlb_high < crlb_low)
