"""Gain vs scan angle sweep.

Demonstrates scan performance of a 32x32 RIS at 28 GHz.
Compares continuous, 2-bit, and 1-bit quantization across scan angles.
"""

from __future__ import annotations

import numpy as np

from metasurface_py.core.types import AngleGrid
from metasurface_py.elements import DiscretePhaseSpace, PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.em import far_field_pattern, peak_gain_db, steering_phase
from metasurface_py.geometry import RectangularLattice
from metasurface_py.plotting import plot_gain_vs_scan_angle, set_publication_style
from metasurface_py.surfaces import Metasurface


def main() -> None:
    set_publication_style()

    freq = 28e9
    lattice = RectangularLattice.from_wavelength(
        nx=32,
        ny=32,
        spacing_fraction=0.5,
        freq=freq,
    )

    scan_angles_deg = np.arange(0, 65, 5, dtype=float)
    angles = AngleGrid(
        theta=np.linspace(0.01, np.pi - 0.01, 360),
        phi=np.linspace(0, 2 * np.pi - 0.01, 72),
    )

    configs = {
        "Continuous": ContinuousPhaseSpace(),
        "2-bit": DiscretePhaseSpace(num_bits=2),
        "1-bit": DiscretePhaseSpace(num_bits=1),
    }

    results: dict[str, list[tuple[float, float]]] = {k: [] for k in configs}

    for scan_deg in scan_angles_deg:
        phase = steering_phase(
            lattice,
            theta_steer=np.radians(scan_deg),
            phi_steer=0.0,
            freq=freq,
        )
        for label, space in configs.items():
            cell = PhaseOnlyCell(state_space=space)
            surface = Metasurface(lattice=lattice, cell=cell)
            state = surface.set_state(phase)
            if space.kind == "discrete":
                state = state.quantize()
            pattern = far_field_pattern(
                surface,
                state,
                freq=freq,
                angles=angles,
            )
            gain = peak_gain_db(pattern)
            results[label].append((scan_deg, gain))

    plot_gain_vs_scan_angle(results)

    import matplotlib.pyplot as plt

    plt.title("32x32 RIS — Gain vs Scan Angle at 28 GHz")
    plt.show()


if __name__ == "__main__":
    main()
