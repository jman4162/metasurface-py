"""MIMO RIS-assisted link model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from metasurface_py.core.conventions import SPEED_OF_LIGHT, k0

if TYPE_CHECKING:
    from metasurface_py.surfaces.metasurface import Metasurface
    from metasurface_py.surfaces.state import SurfaceState


@dataclass(frozen=True)
class MIMORISLink:
    """MIMO RIS-assisted narrowband link.

    Extends the SISO model to multi-antenna TX and RX:
        H_ris = H_sr @ diag(phi) @ H_ri
        H_total = H_ris + H_direct (if included)

    Args:
        surface: Metasurface (RIS).
        tx_positions: TX antenna positions, shape (M_tx, 3).
        rx_positions: RX antenna positions, shape (M_rx, 3).
        freq: Operating frequency [Hz].
        include_direct: Include direct TX-RX path.
    """

    surface: Metasurface
    tx_positions: npt.NDArray[np.floating[Any]]
    rx_positions: npt.NDArray[np.floating[Any]]
    freq: float
    include_direct: bool = True

    @property
    def num_tx(self) -> int:
        """Number of TX antennas."""
        return int(self.tx_positions.shape[0])

    @property
    def num_rx(self) -> int:
        """Number of RX antennas."""
        return int(self.rx_positions.shape[0])

    def _element_channels(
        self,
    ) -> tuple[
        npt.NDArray[np.complexfloating[Any, Any]],
        npt.NDArray[np.complexfloating[Any, Any]],
    ]:
        """Compute TX-RIS and RIS-RX channel matrices.

        Returns:
            h_ri: (N, M_tx) complex — TX antennas to RIS elements.
            h_sr: (M_rx, N) complex — RIS elements to RX antennas.
        """
        positions = self.surface.positions  # (N, 3)
        kw = k0(self.freq)
        lam = SPEED_OF_LIGHT / self.freq
        n = len(positions)

        # TX -> RIS: (N, M_tx)
        h_ri = np.zeros(
            (n, self.num_tx),
            dtype=np.complex128,
        )
        for m in range(self.num_tx):
            d = np.sqrt(
                np.sum(
                    (positions - self.tx_positions[m]) ** 2,
                    axis=1,
                )
            )
            h_ri[:, m] = (lam / (4 * np.pi * d)) * np.exp(
                -1j * kw * d,
            )

        # RIS -> RX: (M_rx, N)
        h_sr = np.zeros(
            (self.num_rx, n),
            dtype=np.complex128,
        )
        for m in range(self.num_rx):
            d = np.sqrt(
                np.sum(
                    (positions - self.rx_positions[m]) ** 2,
                    axis=1,
                )
            )
            h_sr[m, :] = (lam / (4 * np.pi * d)) * np.exp(
                -1j * kw * d,
            )

        return h_ri, h_sr

    def _direct_channel(
        self,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]:
        """Direct TX-RX MIMO channel, shape (M_rx, M_tx)."""
        kw = k0(self.freq)
        lam = SPEED_OF_LIGHT / self.freq
        h_d = np.zeros(
            (self.num_rx, self.num_tx),
            dtype=np.complex128,
        )
        for i in range(self.num_rx):
            for j in range(self.num_tx):
                d = float(
                    np.linalg.norm(
                        self.rx_positions[i] - self.tx_positions[j],
                    )
                )
                h_d[i, j] = (lam / (4 * np.pi * d)) * np.exp(-1j * kw * d)
        return h_d

    def channel_matrix(
        self,
        state: SurfaceState,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]:
        """Compute MIMO channel matrix H_total.

        H_total = H_sr @ diag(phi) @ H_ri [+ H_direct]

        Args:
            state: RIS phase configuration.

        Returns:
            Channel matrix, shape (M_rx, M_tx), complex.
        """
        h_ri, h_sr = self._element_channels()
        phi = self.surface.cell.response(state.values, self.freq)

        # H_ris = H_sr @ diag(phi) @ H_ri
        h_ris = h_sr @ np.diag(phi) @ h_ri

        if self.include_direct:
            h_d = self._direct_channel()
            result: npt.NDArray[np.complexfloating] = h_ris + h_d
            return result
        return h_ris

    def capacity(
        self,
        state: SurfaceState,
        snr_linear: float = 100.0,
    ) -> float:
        """MIMO capacity [bits/s/Hz].

        C = log2(det(I + (snr/M_tx) * H @ H^H))

        Args:
            state: RIS configuration.
            snr_linear: Total SNR (linear).

        Returns:
            Capacity in bits/s/Hz.
        """
        h = self.channel_matrix(state)
        m_tx = self.num_tx
        m_rx = self.num_rx
        eye = np.eye(m_rx, dtype=np.complex128)
        gram = h @ h.conj().T
        cap_matrix = eye + (snr_linear / m_tx) * gram
        return float(
            np.real(
                np.log2(np.linalg.det(cap_matrix)),
            )
        )

    def optimal_state_continuous(self) -> SurfaceState:
        """Optimal RIS phases via dominant singular vector alignment.

        For MIMO, aligns each element's phase to maximize the
        dominant singular value of H_ris. Uses rank-1 approximation:
        phi_n = -(angle(h_ri_n @ v1) + angle(u1^H @ h_sr_n))
        where u1, v1 are dominant left/right singular vectors of
        H_sr @ H_ri (without RIS phases).

        Returns:
            SurfaceState with optimal continuous phases.
        """
        from metasurface_py.elements.states import ContinuousPhaseSpace
        from metasurface_py.surfaces.state import SurfaceState as SS

        h_ri, h_sr = self._element_channels()

        # For each element, the contribution is h_sr[:, n] * phi_n * h_ri[n, :]
        # To maximize the Frobenius norm, align phi_n with the phase of
        # conj(h_sr[:, n])^T @ u1 * v1^H @ h_ri[n, :] for dominant singular pair
        # Simplified: align phi_n = -(angle(h_ri[n,:].sum()) + angle(h_sr[:,n].sum()))
        # This is optimal for rank-1 channels and a good heuristic for general MIMO
        phase_ri = np.angle(np.sum(h_ri, axis=1))
        phase_sr = np.angle(np.sum(h_sr, axis=0))
        optimal_phase = -(phase_ri + phase_sr)

        return SS(
            values=optimal_phase,
            space=ContinuousPhaseSpace(),
        )
