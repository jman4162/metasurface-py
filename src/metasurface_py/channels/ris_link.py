"""RIS-assisted narrowband SISO link model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from metasurface_py.channels.pathloss import (
    free_space_path_loss_db,
)
from metasurface_py.channels.result import LinkBudgetResult
from metasurface_py.core.conventions import SPEED_OF_LIGHT, k0

if TYPE_CHECKING:
    from metasurface_py.core.types import Position3D
    from metasurface_py.surfaces.metasurface import Metasurface
    from metasurface_py.surfaces.state import SurfaceState


@dataclass(frozen=True)
class RISLink:
    """Narrowband SISO RIS-assisted free-space link.

    Models the cascaded channel: TX -> RIS -> RX, plus an optional
    direct TX -> RX path.

    Uses the standard model from Wu & Zhang (2019):
        h_ris = h_sr^H @ diag(Phi) @ h_ri
    where h_ri is TX-to-RIS, h_sr is RIS-to-RX, and Phi is the
    diagonal RIS response matrix.

    Args:
        surface: The metasurface (RIS).
        tx: Transmitter position.
        rx: Receiver position.
        freq: Operating frequency [Hz].
        include_direct: Whether to include the direct TX-RX path.
    """

    surface: Metasurface
    tx: Position3D
    rx: Position3D
    freq: float
    include_direct: bool = True

    def _element_channels(
        self,
    ) -> tuple[
        npt.NDArray[np.complexfloating[Any, Any]],
        npt.NDArray[np.complexfloating[Any, Any]],
    ]:
        """Compute TX-to-RIS and RIS-to-RX channel vectors.

        Returns free-space LoS channels with distance-dependent
        phase and amplitude for each element.
        """
        positions = self.surface.positions  # (N, 3)
        tx_arr = np.array(
            [self.tx.x, self.tx.y, self.tx.z],
            dtype=np.float64,
        )
        rx_arr = np.array(
            [self.rx.x, self.rx.y, self.rx.z],
            dtype=np.float64,
        )
        kw = k0(self.freq)
        lam = SPEED_OF_LIGHT / self.freq

        # TX -> each element
        d_tx = np.sqrt(
            np.sum((positions - tx_arr[np.newaxis, :]) ** 2, axis=1),
        )
        h_ri = (lam / (4.0 * np.pi * d_tx)) * np.exp(-1j * kw * d_tx)

        # Each element -> RX
        d_rx = np.sqrt(
            np.sum((positions - rx_arr[np.newaxis, :]) ** 2, axis=1),
        )
        h_sr = (lam / (4.0 * np.pi * d_rx)) * np.exp(-1j * kw * d_rx)

        return h_ri, h_sr

    def _direct_channel(self) -> complex:
        """Compute direct TX -> RX channel coefficient."""
        d = self.tx.distance_to(self.rx)
        kw = k0(self.freq)
        lam = SPEED_OF_LIGHT / self.freq
        return complex(
            (lam / (4.0 * np.pi * d)) * np.exp(-1j * kw * d),
        )

    def received_power(self, state: SurfaceState) -> float:
        """Compute received power (linear) for a given surface state.

        Assumes unit transmit power.

        Args:
            state: Current surface configuration.

        Returns:
            Received power [linear, relative to unit Tx power].
        """
        h_ri, h_sr = self._element_channels()
        phi = self.surface.cell.response(state.values, self.freq)

        # RIS channel: h_sr^H @ diag(phi) @ h_ri
        h_ris = np.sum(np.conj(h_sr) * phi * h_ri)

        if self.include_direct:
            h_d = self._direct_channel()
            h_total = h_ris + h_d
        else:
            h_total = h_ris

        return float(np.abs(h_total) ** 2)

    def snr_db(
        self,
        state: SurfaceState,
        tx_power_dbm: float = 30.0,
        noise_dbm: float = -90.0,
    ) -> float:
        """Compute SNR in dB.

        Args:
            state: Surface configuration.
            tx_power_dbm: Transmit power [dBm].
            noise_dbm: Noise power [dBm].

        Returns:
            SNR [dB].
        """
        rx_power_linear = self.received_power(state)
        tx_linear = 10.0 ** ((tx_power_dbm - 30.0) / 10.0)
        noise_linear = 10.0 ** ((noise_dbm - 30.0) / 10.0)
        rx = rx_power_linear * tx_linear
        if rx <= 0 or noise_linear <= 0:
            return -math.inf
        return 10.0 * math.log10(rx / noise_linear)

    def optimal_state_continuous(self) -> SurfaceState:
        """Compute analytically optimal continuous phase configuration.

        For free-space SISO, the optimal phases align all element
        contributions in phase at the receiver:
            phi_n = -(angle(h_ri_n) + angle(h_sr_n))

        Returns:
            SurfaceState with optimal continuous phases.
        """
        from metasurface_py.elements.states import ContinuousPhaseSpace
        from metasurface_py.surfaces.state import SurfaceState as SS

        h_ri, h_sr = self._element_channels()
        optimal_phase = -(np.angle(h_ri) + np.angle(h_sr))
        return SS(
            values=optimal_phase,
            space=ContinuousPhaseSpace(),
        )

    def link_budget(
        self,
        state: SurfaceState,
        tx_power_dbm: float = 30.0,
        noise_dbm: float = -90.0,
    ) -> LinkBudgetResult:
        """Compute full link budget.

        Args:
            state: Surface configuration.
            tx_power_dbm: Transmit power [dBm].
            noise_dbm: Noise power [dBm].

        Returns:
            LinkBudgetResult with all metrics.
        """
        rx_power = self.received_power(state)
        tx_w = 10.0 ** ((tx_power_dbm - 30.0) / 10.0)
        rx_w = rx_power * tx_w
        rx_dbm = 10.0 * math.log10(max(rx_w, 1e-30)) + 30.0

        snr = self.snr_db(state, tx_power_dbm, noise_dbm)
        d_direct = self.tx.distance_to(self.rx)
        pl_direct = free_space_path_loss_db(d_direct, self.freq)

        # Effective RIS path loss
        pl_ris = -10.0 * math.log10(rx_power) if rx_power > 0 else math.inf

        ris_gain = pl_direct - pl_ris

        return LinkBudgetResult(
            rx_power_dbm=rx_dbm,
            snr_db=snr,
            path_loss_direct_db=pl_direct,
            path_loss_ris_db=pl_ris,
            ris_gain_db=ris_gain,
            state=state,
        )
