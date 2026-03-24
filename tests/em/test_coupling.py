"""Tests for mutual coupling model."""

from __future__ import annotations

import numpy as np

from metasurface_py.em.coupling import apply_coupling, mutual_impedance_approx
from metasurface_py.geometry import RectangularLattice


class TestMutualCoupling:
    def test_diagonal_is_identity(self) -> None:
        freq = 10e9
        lam = 3e8 / freq
        lattice = RectangularLattice(
            nx=4,
            ny=4,
            dx=lam / 2,
            dy=lam / 2,
        )
        coupling = mutual_impedance_approx(lattice, freq)
        np.testing.assert_allclose(
            np.diag(coupling),
            1.0 + 0j,
            atol=1e-10,
        )

    def test_symmetric(self) -> None:
        freq = 10e9
        lam = 3e8 / freq
        lattice = RectangularLattice(
            nx=4,
            ny=4,
            dx=lam / 2,
            dy=lam / 2,
        )
        coupling = mutual_impedance_approx(lattice, freq)
        np.testing.assert_allclose(
            coupling,
            coupling.T,
            atol=1e-10,
        )

    def test_shape(self) -> None:
        lattice = RectangularLattice(
            nx=3,
            ny=3,
            dx=0.01,
            dy=0.01,
        )
        coupling = mutual_impedance_approx(lattice, 10e9)
        assert coupling.shape == (9, 9)

    def test_apply_coupling(self) -> None:
        n = 4
        response = np.ones(n, dtype=np.complex128)
        coupling = np.eye(n, dtype=np.complex128)
        coupled = apply_coupling(response, coupling)
        np.testing.assert_allclose(coupled, response)
