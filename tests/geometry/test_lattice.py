"""Tests for lattice geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metasurface_py.geometry.lattice import (
    HexagonalLattice,
    RectangularLattice,
    SupportsLattice,
)


class TestRectangularLattice:
    def test_num_elements(self) -> None:
        lat = RectangularLattice(nx=8, ny=8, dx=0.01, dy=0.01)
        assert lat.num_elements == 64

    def test_positions_shape(self) -> None:
        lat = RectangularLattice(nx=4, ny=6, dx=0.005, dy=0.005)
        assert lat.positions.shape == (24, 3)

    def test_positions_centered(self) -> None:
        lat = RectangularLattice(nx=3, ny=3, dx=1.0, dy=1.0)
        pos = lat.positions
        np.testing.assert_allclose(pos.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-15)

    def test_positions_z_zero(self) -> None:
        lat = RectangularLattice(nx=4, ny=4, dx=0.01, dy=0.01)
        np.testing.assert_allclose(lat.positions[:, 2], 0.0)

    def test_spacing(self) -> None:
        dx, dy = 0.005, 0.007
        lat = RectangularLattice(nx=2, ny=1, dx=dx, dy=dy)
        pos = lat.positions
        dist = np.linalg.norm(pos[1] - pos[0])
        assert dist == pytest.approx(dx)

    def test_mask_reduces_elements(self) -> None:
        mask = np.ones((4, 4), dtype=bool)
        mask[0, 0] = False
        mask[3, 3] = False
        lat = RectangularLattice(nx=4, ny=4, dx=0.01, dy=0.01, element_mask=mask)
        assert lat.num_elements == 14
        assert lat.positions.shape == (14, 3)

    def test_from_wavelength(self) -> None:
        freq = 28e9
        lat = RectangularLattice.from_wavelength(
            nx=8,
            ny=8,
            spacing_fraction=0.5,
            freq=freq,
        )
        expected_d = 0.5 * 299_792_458.0 / freq
        assert lat.dx == pytest.approx(expected_d)
        assert lat.dy == pytest.approx(expected_d)

    def test_extent(self) -> None:
        lat = RectangularLattice(nx=10, ny=20, dx=0.01, dy=0.02)
        assert lat.extent == pytest.approx((0.09, 0.38))

    def test_area(self) -> None:
        lat = RectangularLattice(nx=10, ny=10, dx=0.01, dy=0.01)
        # area = nx * dx * ny * dy = 10 * 0.01 * 10 * 0.01 = 0.01
        assert lat.area == pytest.approx(0.01)

    def test_custom_origin(self) -> None:
        origin = np.array([1.0, 2.0, 3.0])
        lat = RectangularLattice(nx=2, ny=2, dx=0.01, dy=0.01, origin=origin)
        center = lat.positions.mean(axis=0)
        np.testing.assert_allclose(center, origin, atol=1e-15)

    def test_implements_protocol(self) -> None:
        lat = RectangularLattice(nx=2, ny=2, dx=0.01, dy=0.01)
        assert isinstance(lat, SupportsLattice)


class TestHexagonalLattice:
    def test_num_elements(self) -> None:
        lat = HexagonalLattice(nx=5, ny=4, dx=0.01)
        assert lat.num_elements == 20

    def test_positions_shape(self) -> None:
        lat = HexagonalLattice(nx=5, ny=4, dx=0.01)
        assert lat.positions.shape == (20, 3)

    def test_positions_centered(self) -> None:
        lat = HexagonalLattice(nx=5, ny=5, dx=0.01)
        pos = lat.positions
        np.testing.assert_allclose(pos.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-15)

    def test_dy_hex_packing(self) -> None:
        dx = 0.01
        lat = HexagonalLattice(nx=4, ny=4, dx=dx)
        expected_dy = dx * math.sqrt(3) / 2
        assert lat.dy == pytest.approx(expected_dy)

    def test_row_offset(self) -> None:
        """Odd rows should be offset by dx/2 relative to even rows."""
        lat = HexagonalLattice(nx=4, ny=2, dx=1.0)
        pos = lat.positions
        # Row 0 has 4 elements, row 1 has 4 elements
        row0 = pos[:4]
        row1 = pos[4:]
        # The x-positions of row 1 should be offset by 0.5 from row 0
        x_diff = row1[0, 0] - row0[0, 0]
        assert x_diff == pytest.approx(0.5)

    def test_mask(self) -> None:
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 1] = False
        lat = HexagonalLattice(nx=3, ny=3, dx=0.01, element_mask=mask)
        assert lat.num_elements == 8

    def test_implements_protocol(self) -> None:
        lat = HexagonalLattice(nx=3, ny=3, dx=0.01)
        assert isinstance(lat, SupportsLattice)
