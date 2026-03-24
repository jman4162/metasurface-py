"""Wideband/OFDM RIS-assisted link model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from metasurface_py.core.conventions import SPEED_OF_LIGHT, k0

if TYPE_CHECKING:
    from metasurface_py.core.types import FrequencyGrid, Position3D
    from metasurface_py.surfaces.metasurface import Metasurface
    from metasurface_py.surfaces.state import SurfaceState


@dataclass(frozen=True)
class WidebandRISLink:
    """Wideband/OFDM RIS-assisted SISO link.

    Evaluates the RIS-assisted channel at each subcarrier frequency.
    A single RIS phase configuration applies to all subcarriers
    (frequency-flat RIS control, which is physical reality).

    Args:
        surface: Metasurface (RIS).
        tx: Transmitter position.
        rx: Receiver position.
        frequencies: OFDM subcarrier frequencies.
        include_direct: Include direct TX-RX path.
    """

    surface: Metasurface
    tx: Position3D
    rx: Position3D
    frequencies: FrequencyGrid
    include_direct: bool = True

    def _channel_at_freq(
        self,
        state: SurfaceState,
        freq: float,
    ) -> complex:
        """Compute scalar channel at a single frequency."""
        positions = self.surface.positions
        kw = k0(freq)
        lam = SPEED_OF_LIGHT / freq

        tx_arr = np.array(
            [self.tx.x, self.tx.y, self.tx.z],
            dtype=np.float64,
        )
        rx_arr = np.array(
            [self.rx.x, self.rx.y, self.rx.z],
            dtype=np.float64,
        )

        d_tx = np.sqrt(
            np.sum((positions - tx_arr) ** 2, axis=1),
        )
        h_ri = (lam / (4 * np.pi * d_tx)) * np.exp(
            -1j * kw * d_tx,
        )

        d_rx = np.sqrt(
            np.sum((positions - rx_arr) ** 2, axis=1),
        )
        h_sr = (lam / (4 * np.pi * d_rx)) * np.exp(
            -1j * kw * d_rx,
        )

        phi = self.surface.cell.response(state.values, freq)
        h_ris = complex(np.sum(np.conj(h_sr) * phi * h_ri))

        if self.include_direct:
            d_direct = float(np.linalg.norm(rx_arr - tx_arr))
            h_d = complex(
                (lam / (4 * np.pi * d_direct)) * np.exp(-1j * kw * d_direct),
            )
            return h_ris + h_d
        return h_ris

    def channel_vs_frequency(
        self,
        state: SurfaceState,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]:
        """Channel coefficient at each subcarrier.

        Args:
            state: RIS phase configuration.

        Returns:
            Complex channel, shape (n_freq,).
        """
        return np.array(
            [self._channel_at_freq(state, float(f)) for f in self.frequencies.values],
            dtype=np.complex128,
        )

    def received_power_vs_frequency(
        self,
        state: SurfaceState,
    ) -> npt.NDArray[np.floating[Any]]:
        """Received power at each subcarrier (linear).

        Args:
            state: RIS configuration.

        Returns:
            Power per subcarrier, shape (n_freq,).
        """
        h = self.channel_vs_frequency(state)
        result: npt.NDArray[np.floating] = np.abs(h) ** 2
        return result

    def ofdm_capacity(
        self,
        state: SurfaceState,
        snr_linear: float = 100.0,
    ) -> float:
        """OFDM sum-rate capacity [bits/s/Hz].

        C = (1/K) * sum_k log2(1 + snr * |h[k]|^2)

        Args:
            state: RIS configuration.
            snr_linear: SNR per subcarrier (linear).

        Returns:
            Capacity in bits/s/Hz.
        """
        power = self.received_power_vs_frequency(state)
        n_sub = len(power)
        rates = np.log2(1.0 + snr_linear * power)
        return float(np.sum(rates) / n_sub)
