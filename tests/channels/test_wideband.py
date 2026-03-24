"""Tests for wideband/OFDM channel model."""

from __future__ import annotations

import numpy as np
import pytest

from metasurface_py.channels import RISLink, WidebandRISLink
from metasurface_py.core.types import FrequencyGrid, Position3D
from metasurface_py.elements import PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.geometry import RectangularLattice
from metasurface_py.surfaces import Metasurface


class TestWidebandRISLink:
    @pytest.fixture()
    def wb_link(self) -> WidebandRISLink:
        freq_center = 10e9
        lam = 3e8 / freq_center
        lattice = RectangularLattice(
            nx=8,
            ny=8,
            dx=lam / 2,
            dy=lam / 2,
        )
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        surface = Metasurface(lattice=lattice, cell=cell)
        freqs = FrequencyGrid(
            values=np.linspace(9.5e9, 10.5e9, 11),
        )
        return WidebandRISLink(
            surface=surface,
            tx=Position3D(0, 0, 20),
            rx=Position3D(10, 0, 1.5),
            frequencies=freqs,
            include_direct=False,
        )

    def test_channel_shape(
        self,
        wb_link: WidebandRISLink,
    ) -> None:
        state = wb_link.surface.set_state(np.zeros(64))
        h = wb_link.channel_vs_frequency(state)
        assert h.shape == (11,)

    def test_power_shape(
        self,
        wb_link: WidebandRISLink,
    ) -> None:
        state = wb_link.surface.set_state(np.zeros(64))
        p = wb_link.received_power_vs_frequency(state)
        assert p.shape == (11,)
        assert np.all(p >= 0)

    def test_capacity_positive(
        self,
        wb_link: WidebandRISLink,
    ) -> None:
        state = wb_link.surface.set_state(np.zeros(64))
        cap = wb_link.ofdm_capacity(state, snr_linear=100)
        assert cap > 0

    def test_single_freq_matches_siso(self) -> None:
        """Single-subcarrier wideband should match SISO."""
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
        tx = Position3D(0, 0, 20)
        rx = Position3D(10, 0, 1.5)

        siso = RISLink(
            surface=surface,
            tx=tx,
            rx=rx,
            freq=freq,
            include_direct=False,
        )
        siso_state = siso.optimal_state_continuous()
        siso_power = siso.received_power(siso_state)

        wb = WidebandRISLink(
            surface=surface,
            tx=tx,
            rx=rx,
            frequencies=FrequencyGrid(values=np.array([freq])),
            include_direct=False,
        )
        wb_power = wb.received_power_vs_frequency(siso_state)

        assert wb_power[0] == pytest.approx(siso_power, rel=1e-6)
