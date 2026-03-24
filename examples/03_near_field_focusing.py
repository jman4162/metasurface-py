"""Near-field focusing phase profile.

Demonstrates a 20x20 metasurface at 10 GHz with a focusing phase
profile for a focal point 0.5 m above the surface center.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from metasurface_py.core.types import AngleGrid
from metasurface_py.elements import PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.em import far_field_pattern, focusing_phase
from metasurface_py.geometry import RectangularLattice
from metasurface_py.plotting import (
    plot_pattern_uv,
    plot_state_map,
    set_publication_style,
)
from metasurface_py.surfaces import Metasurface


def main() -> None:
    set_publication_style(target="ieee_double")

    freq = 10e9
    lattice = RectangularLattice.from_wavelength(
        nx=20,
        ny=20,
        spacing_fraction=0.5,
        freq=freq,
    )

    focal_point = (0.0, 0.0, 0.5)  # 0.5 m above center
    phase = focusing_phase(lattice, focal_point=focal_point, freq=freq)

    cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
    surface = Metasurface(lattice=lattice, cell=cell)
    state = surface.set_state(phase)

    # Phase profile — shows concentric rings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    plot_state_map(state, nx=20, ny=20, ax=ax1)
    ax1.set_title("Focusing Phase Profile")

    # Far-field pattern in u-v space
    angles = AngleGrid(
        theta=np.linspace(0.01, np.pi / 2, 90),
        phi=np.linspace(0, 2 * np.pi - 0.01, 72),
    )
    pattern = far_field_pattern(surface, state, freq=freq, angles=angles)
    plot_pattern_uv(pattern, ax=ax2)
    ax2.set_title("Far-field Pattern (u-v)")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
