"""Dual-beam synthesis.

Demonstrates multi-beam pattern synthesis using a 24x24 metasurface
at 28 GHz with two beams at (20, 0) and (20, 180) degrees.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from metasurface_py.core.types import AngleGrid
from metasurface_py.elements import PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.em import far_field_pattern, multi_beam_phase
from metasurface_py.geometry import RectangularLattice
from metasurface_py.plotting import (
    plot_pattern_3d,
    plot_pattern_polar,
    set_publication_style,
)
from metasurface_py.surfaces import Metasurface


def main() -> None:
    set_publication_style()

    freq = 28e9
    lattice = RectangularLattice.from_wavelength(
        nx=24,
        ny=24,
        spacing_fraction=0.5,
        freq=freq,
    )

    directions = [
        (np.radians(20.0), 0.0),
        (np.radians(20.0), np.pi),
    ]
    phase = multi_beam_phase(
        lattice,
        directions=directions,
        weights=None,
        freq=freq,
    )

    cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
    surface = Metasurface(lattice=lattice, cell=cell)
    state = surface.set_state(phase)

    # Polar plot
    angles_1d = AngleGrid(
        theta=np.linspace(0.01, np.pi - 0.01, 360),
        phi=np.array([0.0]),
    )
    pattern_1d = far_field_pattern(
        surface,
        state,
        freq=freq,
        angles=angles_1d,
    )

    _fig1, ax1 = plt.subplots(subplot_kw={"projection": "polar"})
    plot_pattern_polar(pattern_1d, cut_phi=0.0, ax=ax1)
    ax1.set_title("Dual-beam — Polar Pattern", pad=15)

    # 3D pattern
    angles_3d = AngleGrid(
        theta=np.linspace(0.01, np.pi - 0.01, 90),
        phi=np.linspace(0, 2 * np.pi - 0.01, 72),
    )
    pattern_3d = far_field_pattern(
        surface,
        state,
        freq=freq,
        angles=angles_3d,
    )

    plot_pattern_3d(pattern_3d)

    plt.show()


if __name__ == "__main__":
    main()
