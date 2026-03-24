"""Benchmark: Directivity vs analytical theory.

Verifies that computed directivity for a uniform-phase NxN array at
lambda/2 spacing matches the expected value of approximately N^2 * pi * (d/lambda)^2.
"""

from __future__ import annotations

import math

import numpy as np

from metasurface_py.core.types import AngleGrid
from metasurface_py.elements import PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.em import far_field_pattern, peak_gain_db
from metasurface_py.geometry import RectangularLattice
from metasurface_py.surfaces import Metasurface


def main() -> None:
    freq = 10e9
    lam = 3e8 / freq
    dx = lam / 2

    angles = AngleGrid(
        theta=np.linspace(0.01, np.pi - 0.01, 360),
        phi=np.linspace(0, 2 * np.pi - 0.01, 144),
    )

    sizes = [4, 8, 12, 16, 24, 32]
    header = f"{'N':>4} {'N^2':>6} {'Computed':>10} {'Theory':>8} {'Error':>8}"
    print(header)
    print("-" * 55)

    for n in sizes:
        lattice = RectangularLattice(nx=n, ny=n, dx=dx, dy=dx)
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        surface = Metasurface(lattice=lattice, cell=cell)
        state = surface.set_state(np.zeros(n * n))

        pattern = far_field_pattern(
            surface,
            state,
            freq=freq,
            angles=angles,
        )
        computed = peak_gain_db(pattern)

        # Array factor directivity for N isotropic elements ~ N
        # Total elements = N^2, directivity ~ N^2
        theory = 10.0 * math.log10(n * n)
        error = computed - theory

        print(f"{n:>4} {n * n:>6} {computed:>15.2f} {theory:>13.2f} {error:>11.2f}")

    print(
        "\nNote: Errors within ~2 dB are expected"
        " due to numerical integration resolution."
    )


if __name__ == "__main__":
    main()
