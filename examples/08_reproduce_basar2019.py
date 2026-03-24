"""Reproduce key results from Basar et al. 2019.

Reference: E. Basar et al., "Wireless Communications Through
Reconfigurable Intelligent Surfaces," IEEE Access, 2019.

Reproduces the fundamental RIS performance analysis:
- SNR scaling with number of elements (N^2 law)
- Path loss comparison: RIS-assisted vs direct link
- Effect of RIS placement on coverage

Note: Uses free-space LoS model. The paper also considers Rayleigh
fading; our deterministic model captures the mean/expected behavior.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from metasurface_py.channels import RISLink, free_space_path_loss_db
from metasurface_py.core.types import Position3D
from metasurface_py.elements import PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.geometry import RectangularLattice
from metasurface_py.plotting import set_publication_style
from metasurface_py.surfaces import Metasurface


def main() -> None:
    set_publication_style(target="ieee_double")

    freq = 5.8e9  # 5.8 GHz
    lam = 3e8 / freq
    tx_power_dbm = 20.0
    noise_dbm = -90.0

    tx = Position3D(x=0.0, y=0.0, z=10.0)
    ris_origin = np.array([30.0, 0.0, 10.0])
    rx = Position3D(x=50.0, y=0.0, z=1.5)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # ── Panel 1: SNR vs N (N^2 scaling law) ───────────────────

    sizes = [4, 8, 12, 16, 20, 24, 32]
    snr_opt: list[float] = []
    n_elements: list[int] = []

    for n in sizes:
        lattice = RectangularLattice(
            nx=n,
            ny=n,
            dx=lam / 2,
            dy=lam / 2,
            origin=ris_origin,
        )
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        surface = Metasurface(lattice=lattice, cell=cell)
        link = RISLink(
            surface=surface,
            tx=tx,
            rx=rx,
            freq=freq,
            include_direct=False,
        )
        opt = link.optimal_state_continuous()
        snr_opt.append(
            link.snr_db(opt, tx_power_dbm, noise_dbm),
        )
        n_elements.append(n * n)

    # Theoretical N^2 line (normalized to first point)
    n_arr = np.array(n_elements, dtype=float)
    theory = snr_opt[0] + 20 * np.log10(n_arr / n_arr[0])

    axes[0].plot(n_elements, snr_opt, "o-", label="Simulated")
    axes[0].plot(
        n_elements,
        theory,
        "--",
        color="gray",
        label="$N^2$ scaling (theory)",
    )
    axes[0].set_xlabel("Number of RIS Elements")
    axes[0].set_ylabel("SNR [dB]")
    axes[0].set_title("$N^2$ Scaling Law")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Panel 2: Path loss comparison ─────────────────────────

    d_range = np.linspace(10, 80, 30)
    pl_direct: list[float] = []
    pl_ris: list[float] = []

    n_ris = 16
    lattice_fixed = RectangularLattice(
        nx=n_ris,
        ny=n_ris,
        dx=lam / 2,
        dy=lam / 2,
        origin=ris_origin,
    )
    cell_fixed = PhaseOnlyCell(
        state_space=ContinuousPhaseSpace(),
    )
    surface_fixed = Metasurface(
        lattice=lattice_fixed,
        cell=cell_fixed,
    )

    for d in d_range:
        rx_d = Position3D(x=float(d), y=0.0, z=1.5)

        # Direct path loss
        pl_direct.append(
            free_space_path_loss_db(
                tx.distance_to(rx_d),
                freq,
            ),
        )

        # RIS effective path loss
        link = RISLink(
            surface=surface_fixed,
            tx=tx,
            rx=rx_d,
            freq=freq,
            include_direct=False,
        )
        opt = link.optimal_state_continuous()
        rx_power = link.received_power(opt)
        if rx_power > 0:
            pl_ris.append(-10 * math.log10(rx_power))
        else:
            pl_ris.append(200.0)

    axes[1].plot(d_range, pl_direct, "--", label="Direct (no RIS)")
    axes[1].plot(
        d_range,
        pl_ris,
        "-",
        label=f"Via RIS ({n_ris}x{n_ris})",
    )
    axes[1].set_xlabel("TX-RX Distance [m]")
    axes[1].set_ylabel("Path Loss [dB]")
    axes[1].set_title("Path Loss Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_yaxis()

    # ── Panel 3: Coverage map (SNR vs RX position) ────────────

    rx_x = np.linspace(5, 80, 40)
    snr_with_ris: list[float] = []
    snr_without_ris: list[float] = []

    for x in rx_x:
        rx_pos = Position3D(x=float(x), y=0.0, z=1.5)

        # With RIS
        link = RISLink(
            surface=surface_fixed,
            tx=tx,
            rx=rx_pos,
            freq=freq,
            include_direct=True,
        )
        opt = link.optimal_state_continuous()
        snr_with_ris.append(
            link.snr_db(opt, tx_power_dbm, noise_dbm),
        )

        # Without RIS
        pl = free_space_path_loss_db(
            tx.distance_to(rx_pos),
            freq,
        )
        snr_without_ris.append(
            tx_power_dbm - pl - noise_dbm,
        )

    axes[2].plot(rx_x, snr_with_ris, "-", label="With RIS")
    axes[2].plot(
        rx_x,
        snr_without_ris,
        "--",
        label="Without RIS",
    )
    axes[2].axvline(
        x=30,
        color="gray",
        alpha=0.3,
        linestyle=":",
        label="RIS location",
    )
    axes[2].set_xlabel("RX Position [m]")
    axes[2].set_ylabel("SNR [dB]")
    axes[2].set_title("Coverage Extension")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        "Reproduction: Basar et al., IEEE Access 2019",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
