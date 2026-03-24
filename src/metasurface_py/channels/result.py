"""Link budget result container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metasurface_py.surfaces.state import SurfaceState


@dataclass(frozen=True)
class LinkBudgetResult:
    """Result of an RIS-assisted link budget computation.

    Args:
        rx_power_dbm: Received power [dBm].
        snr_db: Signal-to-noise ratio [dB].
        path_loss_direct_db: Direct TX-RX path loss [dB].
        path_loss_ris_db: Effective RIS-assisted path loss [dB].
        ris_gain_db: Improvement over direct link [dB].
        state: The surface state used.
    """

    rx_power_dbm: float
    snr_db: float
    path_loss_direct_db: float
    path_loss_ris_db: float
    ris_gain_db: float
    state: SurfaceState
