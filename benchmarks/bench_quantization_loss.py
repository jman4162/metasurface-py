"""Benchmark: Quantization loss vs theoretical values.

Verifies that phase quantization loss converges to theoretical values:
- 1-bit: ~3.92 dB
- 2-bit: ~0.91 dB
- 3-bit: ~0.22 dB
"""

from __future__ import annotations

import numpy as np

from metasurface_py.core.types import AngleGrid
from metasurface_py.elements import DiscretePhaseSpace, PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.em import far_field_pattern, peak_gain_db, steering_phase
from metasurface_py.geometry import RectangularLattice
from metasurface_py.surfaces import Metasurface


def main() -> None:
    freq = 10e9
    lam = 3e8 / freq
    dx = lam / 2
    theta_steer = np.radians(20.0)

    angles = AngleGrid(
        theta=np.linspace(0.01, np.pi - 0.01, 360),
        phi=np.linspace(0, 2 * np.pi - 0.01, 72),
    )

    # Theoretical quantization loss: sinc^2(pi / 2^B)
    theoretical = {
        1: 3.92,
        2: 0.91,
        3: 0.22,
    }

    sizes = [8, 16, 32]
    bits_list = [1, 2, 3]

    header = f"{'N':>4} {'Bits':>5} {'Cont':>8} {'Quant':>8} {'Loss':>8} {'Theory':>8}"
    print(header)
    print("-" * 60)

    for n in sizes:
        lattice = RectangularLattice(nx=n, ny=n, dx=dx, dy=dx)
        phase = steering_phase(
            lattice,
            theta_steer=theta_steer,
            phi_steer=0.0,
            freq=freq,
        )

        # Continuous reference
        cell_cont = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        surface_cont = Metasurface(lattice=lattice, cell=cell_cont)
        state_cont = surface_cont.set_state(phase)
        pattern_cont = far_field_pattern(
            surface_cont,
            state_cont,
            freq=freq,
            angles=angles,
        )
        gain_cont = peak_gain_db(pattern_cont)

        for bits in bits_list:
            cell_q = PhaseOnlyCell(
                state_space=DiscretePhaseSpace(num_bits=bits),
            )
            surface_q = Metasurface(lattice=lattice, cell=cell_q)
            state_q = surface_q.set_state(phase).quantize()
            pattern_q = far_field_pattern(
                surface_q,
                state_q,
                freq=freq,
                angles=angles,
            )
            gain_q = peak_gain_db(pattern_q)
            loss = gain_cont - gain_q
            theory = theoretical[bits]

            print(
                f"{n:>4} {bits:>5} {gain_cont:>11.2f} {gain_q:>12.2f} "
                f"{loss:>10.2f} {theory:>12.2f}"
            )

    print("\nNote: Loss should converge to theory for large arrays.")


if __name__ == "__main__":
    main()
