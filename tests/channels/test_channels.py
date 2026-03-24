"""Tests for channels module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metasurface_py.channels import (
    RISLink,
    free_space_path_loss,
    free_space_path_loss_db,
)
from metasurface_py.core.types import Position3D
from metasurface_py.elements import PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.geometry import RectangularLattice
from metasurface_py.surfaces import Metasurface


class TestFreeSpacePathLoss:
    def test_friis_formula(self) -> None:
        """FSPL should match Friis equation."""
        from metasurface_py.core.conventions import SPEED_OF_LIGHT

        d = 100.0
        freq = 28e9
        lam = SPEED_OF_LIGHT / freq
        expected = (4.0 * math.pi * d / lam) ** 2
        assert free_space_path_loss(d, freq) == pytest.approx(expected)

    def test_db_conversion(self) -> None:
        d = 100.0
        freq = 28e9
        linear = free_space_path_loss(d, freq)
        db = free_space_path_loss_db(d, freq)
        assert db == pytest.approx(10 * math.log10(linear))

    def test_inverse_square_law(self) -> None:
        """Doubling distance should add ~6 dB."""
        freq = 10e9
        pl1 = free_space_path_loss_db(100.0, freq)
        pl2 = free_space_path_loss_db(200.0, freq)
        assert (pl2 - pl1) == pytest.approx(6.02, abs=0.1)

    def test_negative_distance_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            free_space_path_loss(-1.0, 10e9)


class TestRISLink:
    @pytest.fixture()
    def simple_link(self) -> RISLink:
        """16x16 RIS at origin, TX and RX at short distances."""
        freq = 10e9
        lam = 3e8 / freq
        lattice = RectangularLattice(
            nx=16,
            ny=16,
            dx=lam / 2,
            dy=lam / 2,
        )
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        surface = Metasurface(lattice=lattice, cell=cell)
        tx = Position3D(x=0.0, y=0.0, z=10.0)
        rx = Position3D(x=5.0, y=0.0, z=1.5)
        return RISLink(
            surface=surface,
            tx=tx,
            rx=rx,
            freq=freq,
            include_direct=False,
        )

    def test_optimal_beats_random(
        self,
        simple_link: RISLink,
    ) -> None:
        """Optimal continuous phases should beat random."""
        opt_state = simple_link.optimal_state_continuous()
        opt_power = simple_link.received_power(opt_state)

        rng = np.random.default_rng(42)
        random_powers = []
        for _ in range(20):
            from metasurface_py.surfaces.state import SurfaceState

            n_elem = simple_link.surface.num_elements
            rp = rng.uniform(0, 2 * np.pi, n_elem)
            rs = SurfaceState(
                values=rp,
                space=ContinuousPhaseSpace(),
            )
            random_powers.append(simple_link.received_power(rs))

        assert opt_power > max(random_powers)

    def test_snr_reasonable(self, simple_link: RISLink) -> None:
        """SNR should be a reasonable value."""
        opt_state = simple_link.optimal_state_continuous()
        snr = simple_link.snr_db(
            opt_state,
            tx_power_dbm=30.0,
            noise_dbm=-90.0,
        )
        # Should be positive with these parameters
        assert snr > 0

    def test_link_budget_returns_result(
        self,
        simple_link: RISLink,
    ) -> None:
        opt_state = simple_link.optimal_state_continuous()
        result = simple_link.link_budget(opt_state)
        assert result.path_loss_direct_db > 0
        assert result.path_loss_ris_db > 0
        assert result.snr_db > 0
        # rx_power_dbm should be finite
        assert result.rx_power_dbm > -200

    def test_n_squared_scaling(self) -> None:
        """RIS power gain should scale roughly as N^2."""
        freq = 10e9
        lam = 3e8 / freq
        tx = Position3D(x=0.0, y=0.0, z=100.0)
        rx = Position3D(x=20.0, y=0.0, z=1.5)

        powers = []
        sizes = [4, 8, 16]
        for n in sizes:
            lattice = RectangularLattice(
                nx=n,
                ny=n,
                dx=lam / 2,
                dy=lam / 2,
            )
            cell = PhaseOnlyCell(
                state_space=ContinuousPhaseSpace(),
            )
            surface = Metasurface(lattice=lattice, cell=cell)
            link = RISLink(
                surface=surface,
                tx=tx,
                rx=rx,
                freq=freq,
                include_direct=False,
            )
            opt = link.optimal_state_continuous()
            powers.append(link.received_power(opt))

        # Power ratio between N=16 and N=4 should be ~(16^2/4^2)^2 = 256
        # (N^2 elements, each contributing coherently -> N^4 power scaling)
        # Actually for optimal RIS: power ~ (N * lambda/(4*pi*d))^2 so ~ N^2
        # But the per-element channel also depends on N through element area
        # In practice, check that larger arrays give significantly more power
        assert powers[2] > powers[1] > powers[0]
        # Ratio should be roughly (16/4)^2 = 16x between sizes 4 and 16
        ratio = powers[2] / powers[0]
        expected = (sizes[2] ** 2 / sizes[0] ** 2) ** 2
        # Allow generous tolerance due to geometry effects
        assert ratio == pytest.approx(expected, rel=0.5)
