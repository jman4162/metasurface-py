"""Reproduce key results from Wu & Zhang 2019.

Reference: Q. Wu and R. Zhang, "Intelligent Reflecting Surface Enhanced
Wireless Network: Joint Active and Passive Beamforming Design,"
IEEE TWC, 2019. (arxiv: 1809.01423)

Reproduces:
- Fig. 5: Receive SNR vs number of reflecting elements N
  Shows N^2 scaling of RIS-assisted link.
- Fig. 4: Receive SNR vs AP-user distance
  Shows coverage extension via RIS.

Note: Our model uses deterministic free-space LoS channels (no fading),
so absolute SNR values differ from the paper's Rayleigh fading model.
The qualitative trends (N^2 scaling, coverage extension) match.
"""

from __future__ import annotations

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

    freq = 2.4e9  # 2.4 GHz (typical indoor)
    lam = 3e8 / freq
    tx_power_dbm = 5.0  # 5 dBm (from paper: p = 5 dBm)
    noise_dbm = -80.0  # -80 dBm (from paper: sigma^2 = -80 dBm)

    # Geometry from paper Fig. 2:
    # AP at origin, IRS at d0=51m away, user in between
    d0 = 51.0  # AP-IRS horizontal distance
    dv = 2.0  # vertical separation

    ap = Position3D(x=0.0, y=0.0, z=dv)
    ris_pos = Position3D(x=d0, y=0.0, z=dv)

    # ── Fig 5 reproduction: SNR vs N ──────────────────────────

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    d_values = [15.0, 43.0]  # user distances (from paper)
    n_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for d_user in d_values:
        user = Position3D(x=d_user, y=0.0, z=0.0)
        snr_ris: list[float] = []
        snr_no_ris: list[float] = []

        for n_total in n_values:
            # Square-ish array: ny=10 fixed, nx varies (per paper)
            ny = 10
            nx = max(1, n_total // ny)
            lattice = RectangularLattice(
                nx=nx,
                ny=ny,
                dx=lam / 2,
                dy=lam / 2,
                origin=np.array(
                    [ris_pos.x, ris_pos.y, ris_pos.z],
                ),
            )
            cell = PhaseOnlyCell(
                state_space=ContinuousPhaseSpace(),
            )
            surface = Metasurface(lattice=lattice, cell=cell)

            # With RIS (optimal phases)
            link = RISLink(
                surface=surface,
                tx=ap,
                rx=user,
                freq=freq,
                include_direct=True,
            )
            opt = link.optimal_state_continuous()
            snr = link.snr_db(opt, tx_power_dbm, noise_dbm)
            snr_ris.append(snr)

            # Without RIS (direct link only)
            pl = free_space_path_loss_db(
                ap.distance_to(user),
                freq,
            )
            snr_direct = tx_power_dbm - pl - noise_dbm
            snr_no_ris.append(snr_direct)

        ax1.plot(
            n_values,
            snr_ris,
            "o-",
            label=f"With RIS (d={d_user:.0f}m)",
        )
        ax1.plot(
            n_values,
            snr_no_ris,
            "x--",
            label=f"Without RIS (d={d_user:.0f}m)",
        )

    ax1.set_xlabel("Number of Reflecting Elements, N")
    ax1.set_ylabel("Receive SNR [dB]")
    ax1.set_title(
        "SNR vs N (cf. Wu & Zhang 2019, Fig. 5)",
    )
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Fig 4 reproduction: SNR vs distance ───────────────────

    n_fixed = 50  # N=50 elements
    ny = 10
    nx = n_fixed // ny
    d_range = np.linspace(5, 50, 20)

    lattice_fixed = RectangularLattice(
        nx=nx,
        ny=ny,
        dx=lam / 2,
        dy=lam / 2,
        origin=np.array(
            [ris_pos.x, ris_pos.y, ris_pos.z],
        ),
    )
    cell_fixed = PhaseOnlyCell(
        state_space=ContinuousPhaseSpace(),
    )
    surface_fixed = Metasurface(
        lattice=lattice_fixed,
        cell=cell_fixed,
    )

    snr_vs_d_ris: list[float] = []
    snr_vs_d_direct: list[float] = []

    for d in d_range:
        user = Position3D(x=float(d), y=0.0, z=0.0)
        link = RISLink(
            surface=surface_fixed,
            tx=ap,
            rx=user,
            freq=freq,
            include_direct=True,
        )
        opt = link.optimal_state_continuous()
        snr_vs_d_ris.append(
            link.snr_db(opt, tx_power_dbm, noise_dbm),
        )

        pl = free_space_path_loss_db(
            ap.distance_to(user),
            freq,
        )
        snr_vs_d_direct.append(
            tx_power_dbm - pl - noise_dbm,
        )

    ax2.plot(d_range, snr_vs_d_ris, "o-", label="With RIS (N=50)")
    ax2.plot(
        d_range,
        snr_vs_d_direct,
        "x--",
        label="Without RIS",
    )
    ax2.set_xlabel("AP-User Horizontal Distance, d [m]")
    ax2.set_ylabel("Receive SNR [dB]")
    ax2.set_title(
        "SNR vs Distance (cf. Wu & Zhang 2019, Fig. 4)",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Reproduction: Wu & Zhang, IEEE TWC 2019",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
