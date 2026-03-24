"""Beam steering with quantization comparison.

Demonstrates a 16x16 metasurface at 28 GHz steered to 30 degrees.
Compares continuous phase, 2-bit, and 1-bit quantized patterns.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from metasurface_py.core.types import AngleGrid
from metasurface_py.elements import DiscretePhaseSpace, PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.em import far_field_pattern, steering_phase
from metasurface_py.geometry import RectangularLattice
from metasurface_py.plotting import (
    plot_pattern_comparison,
    plot_state_map,
    set_publication_style,
)
from metasurface_py.surfaces import Metasurface


def main() -> None:
    set_publication_style()

    freq = 28e9
    lattice = RectangularLattice.from_wavelength(
        nx=16,
        ny=16,
        spacing_fraction=0.5,
        freq=freq,
    )

    theta_steer = np.radians(30.0)
    phase = steering_phase(
        lattice,
        theta_steer=theta_steer,
        phi_steer=0.0,
        freq=freq,
    )

    angles = AngleGrid(
        theta=np.linspace(0.01, np.pi - 0.01, 360),
        phi=np.array([0.0]),
    )

    # Continuous phase
    cell_cont = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
    surface_cont = Metasurface(lattice=lattice, cell=cell_cont)
    state_cont = surface_cont.set_state(phase)
    pattern_cont = far_field_pattern(
        surface_cont,
        state_cont,
        freq=freq,
        angles=angles,
    )

    # 2-bit quantized
    cell_2bit = PhaseOnlyCell(state_space=DiscretePhaseSpace(num_bits=2))
    surface_2bit = Metasurface(lattice=lattice, cell=cell_2bit)
    state_2bit = surface_2bit.set_state(phase).quantize()
    pattern_2bit = far_field_pattern(
        surface_2bit,
        state_2bit,
        freq=freq,
        angles=angles,
    )

    # 1-bit quantized
    cell_1bit = PhaseOnlyCell(state_space=DiscretePhaseSpace(num_bits=1))
    surface_1bit = Metasurface(lattice=lattice, cell=cell_1bit)
    state_1bit = surface_1bit.set_state(phase).quantize()
    pattern_1bit = far_field_pattern(
        surface_1bit,
        state_1bit,
        freq=freq,
        angles=angles,
    )

    # Pattern comparison
    _fig1, ax1 = plt.subplots()
    plot_pattern_comparison(
        [
            (pattern_cont, "Continuous"),
            (pattern_2bit, "2-bit"),
            (pattern_1bit, "1-bit"),
        ],
        cut_phi=0.0,
        ax=ax1,
    )
    ax1.set_title("Beam Steering to 30° — Quantization Comparison")
    ax1.set_xlim([0, 90])
    ax1.set_ylim([-40, 0])

    # Phase state map (2-bit)
    _fig2, ax2 = plt.subplots()
    plot_state_map(state_2bit, nx=16, ny=16, ax=ax2)
    ax2.set_title("2-bit Phase Distribution")

    plt.show()


if __name__ == "__main__":
    main()
