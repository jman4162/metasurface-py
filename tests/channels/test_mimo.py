"""Tests for MIMO channel model."""

from __future__ import annotations

import numpy as np
import pytest

from metasurface_py.channels import MIMORISLink, RISLink, UniformLinearArray
from metasurface_py.core.types import Position3D
from metasurface_py.elements import PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.geometry import RectangularLattice
from metasurface_py.surfaces import Metasurface


class TestUniformLinearArray:
    def test_positions_shape(self) -> None:
        ula = UniformLinearArray(
            num_antennas=4,
            spacing=0.01,
            center=Position3D(0, 0, 10),
        )
        assert ula.positions.shape == (4, 3)

    def test_centered(self) -> None:
        ula = UniformLinearArray(
            num_antennas=4,
            spacing=1.0,
            center=Position3D(0, 0, 0),
        )
        center = ula.positions.mean(axis=0)
        np.testing.assert_allclose(center, [0, 0, 0], atol=1e-10)


class TestMIMORISLink:
    @pytest.fixture()
    def mimo_link(self) -> MIMORISLink:
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
        tx_ula = UniformLinearArray(
            num_antennas=2,
            spacing=lam / 2,
            center=Position3D(0, 0, 20),
        )
        rx_ula = UniformLinearArray(
            num_antennas=2,
            spacing=lam / 2,
            center=Position3D(10, 0, 1.5),
        )
        return MIMORISLink(
            surface=surface,
            tx_positions=tx_ula.positions,
            rx_positions=rx_ula.positions,
            freq=freq,
            include_direct=False,
        )

    def test_channel_matrix_shape(
        self,
        mimo_link: MIMORISLink,
    ) -> None:
        state = mimo_link.optimal_state_continuous()
        h = mimo_link.channel_matrix(state)
        assert h.shape == (2, 2)

    def test_capacity_positive(
        self,
        mimo_link: MIMORISLink,
    ) -> None:
        state = mimo_link.optimal_state_continuous()
        cap = mimo_link.capacity(state, snr_linear=100)
        assert cap > 0

    def test_siso_consistency(self) -> None:
        """1x1 MIMO should approximate SISO RISLink."""
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

        # SISO
        siso = RISLink(
            surface=surface,
            tx=tx,
            rx=rx,
            freq=freq,
            include_direct=False,
        )
        siso_state = siso.optimal_state_continuous()
        siso_power = siso.received_power(siso_state)

        # 1x1 MIMO
        mimo = MIMORISLink(
            surface=surface,
            tx_positions=np.array([[tx.x, tx.y, tx.z]]),
            rx_positions=np.array([[rx.x, rx.y, rx.z]]),
            freq=freq,
            include_direct=False,
        )
        mimo_state = mimo.optimal_state_continuous()
        h = mimo.channel_matrix(mimo_state)
        mimo_power = float(np.abs(h[0, 0]) ** 2)

        # Should be similar (not exact due to different phase optimization)
        ratio = mimo_power / max(siso_power, 1e-30)
        assert 0.5 < ratio < 2.0
