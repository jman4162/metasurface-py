"""Benchmark: RIS SNR scaling vs N^2 theory.

Verifies that optimal RIS SNR gain scales approximately as N^2
(number of elements squared) for the free-space SISO case.
"""

from __future__ import annotations

import math

from metasurface_py.channels import RISLink
from metasurface_py.core.types import Position3D
from metasurface_py.elements import PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.geometry import RectangularLattice
from metasurface_py.surfaces import Metasurface


def main() -> None:
    freq = 10e9
    lam = 3e8 / freq
    dx = lam / 2

    tx = Position3D(x=0.0, y=0.0, z=50.0)
    rx = Position3D(x=20.0, y=0.0, z=1.5)

    tx_power_dbm = 20.0
    noise_dbm = -90.0

    sizes = [4, 8, 12, 16, 20, 24]
    snrs: list[float] = []

    for n in sizes:
        lattice = RectangularLattice(nx=n, ny=n, dx=dx, dy=dx)
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        surface = Metasurface(lattice=lattice, cell=cell)
        link = RISLink(
            surface=surface,
            tx=tx,
            rx=rx,
            freq=freq,
            include_direct=False,
        )
        opt_state = link.optimal_state_continuous()
        snr = link.snr_db(opt_state, tx_power_dbm, noise_dbm)
        snrs.append(snr)

    # N^2 scaling means SNR should increase by 20*log10(N2/N1) dB
    # when going from N1^2 to N2^2 elements
    print(f"{'N':>4} {'N^2':>6} {'SNR [dB]':>10} {'Predicted':>10} {'Error':>8}")
    print("-" * 42)

    ref_snr = snrs[0]
    ref_n = sizes[0]
    for n, snr in zip(sizes, snrs, strict=False):
        # Predicted: ref_snr + 20*log10((n/ref_n)^2)
        predicted = ref_snr + 40.0 * math.log10(n / ref_n)
        error = snr - predicted
        print(f"{n:>4} {n * n:>6} {snr:>10.2f} {predicted:>10.2f} {error:>8.2f}")

    print("\nNote: Errors within ~1 dB indicate correct N^2 scaling.")


if __name__ == "__main__":
    main()
