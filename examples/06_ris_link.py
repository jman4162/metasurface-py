"""RIS-assisted communication link.

Demonstrates a narrowband SISO RIS-assisted link at 28 GHz.
Compares received power with optimal continuous, 2-bit quantized,
and no-RIS configurations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from metasurface_py.channels import RISLink, free_space_path_loss_db
from metasurface_py.core.types import Position3D
from metasurface_py.elements import DiscretePhaseSpace, PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.geometry import RectangularLattice
from metasurface_py.plotting import set_publication_style
from metasurface_py.surfaces import Metasurface


def main() -> None:
    set_publication_style()

    freq = 28e9
    lam = 3e8 / freq

    tx = Position3D(x=0.0, y=0.0, z=20.0)
    rx = Position3D(x=10.0, y=0.0, z=1.5)

    sizes = [8, 12, 16, 20, 24]
    snr_optimal: list[float] = []
    snr_2bit: list[float] = []
    snr_no_ris: list[float] = []

    tx_power_dbm = 20.0
    noise_dbm = -90.0

    # Direct link SNR (constant across sizes)
    d_direct = tx.distance_to(rx)
    pl_direct = free_space_path_loss_db(d_direct, freq)
    snr_direct = tx_power_dbm - pl_direct - noise_dbm

    for n in sizes:
        lattice = RectangularLattice(
            nx=n,
            ny=n,
            dx=lam / 2,
            dy=lam / 2,
        )

        # Optimal continuous
        cell_cont = PhaseOnlyCell(
            state_space=ContinuousPhaseSpace(),
        )
        surface_cont = Metasurface(
            lattice=lattice,
            cell=cell_cont,
        )
        link_cont = RISLink(
            surface=surface_cont,
            tx=tx,
            rx=rx,
            freq=freq,
            include_direct=True,
        )
        opt_state = link_cont.optimal_state_continuous()
        snr_optimal.append(
            link_cont.snr_db(opt_state, tx_power_dbm, noise_dbm),
        )

        # 2-bit quantized
        cell_2bit = PhaseOnlyCell(
            state_space=DiscretePhaseSpace(num_bits=2),
        )
        surface_2bit = Metasurface(
            lattice=lattice,
            cell=cell_2bit,
        )
        link_2bit = RISLink(
            surface=surface_2bit,
            tx=tx,
            rx=rx,
            freq=freq,
            include_direct=True,
        )
        state_2bit = link_2bit.optimal_state_continuous().quantize(
            cell_2bit.state_space.codebook,
        )
        snr_2bit.append(
            link_2bit.snr_db(state_2bit, tx_power_dbm, noise_dbm),
        )

        snr_no_ris.append(snr_direct)

    # Bar chart
    [n * n for n in sizes]
    labels = [f"{n}x{n}\n({n * n})" for n in sizes]
    x = np.arange(len(sizes))
    width = 0.25

    _fig, ax = plt.subplots()
    ax.bar(x - width, snr_optimal, width, label="Optimal (cont.)")
    ax.bar(x, snr_2bit, width, label="2-bit quantized")
    ax.bar(x + width, snr_no_ris, width, label="No RIS")
    ax.set_xlabel("RIS Size (elements)")
    ax.set_ylabel("SNR [dB]")
    ax.set_title("RIS-Assisted Link — SNR vs Array Size")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
