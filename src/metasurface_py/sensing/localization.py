"""Localization metrics: Fisher information and CRLB."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from metasurface_py.core.conventions import k0

if TYPE_CHECKING:
    from metasurface_py.core.types import Position3D
    from metasurface_py.surfaces.metasurface import Metasurface
    from metasurface_py.surfaces.state import SurfaceState


def fisher_information_matrix(
    surface: Metasurface,
    state: SurfaceState,
    target_pos: Position3D,
    freq: float,
    snr: float = 20.0,
) -> npt.NDArray[np.floating[Any]]:
    """Compute Fisher Information Matrix for position estimation.

    Uses a simplified far-field signal model where the FIM
    depends on the array geometry and element responses.

    Args:
        surface: Metasurface object.
        state: Surface configuration.
        target_pos: Target position to localize.
        freq: Frequency [Hz].
        snr: Signal-to-noise ratio (linear).

    Returns:
        3x3 Fisher Information Matrix.
    """
    kw = k0(freq)
    positions = surface.positions  # (N, 3)
    target = np.array(
        [target_pos.x, target_pos.y, target_pos.z],
        dtype=np.float64,
    )

    # Distances from each element to target
    diff = positions - target[np.newaxis, :]  # (N, 3)
    distances = np.sqrt(np.sum(diff**2, axis=1))  # (N,)
    distances = np.maximum(distances, 1e-10)

    # Unit vectors from elements to target
    unit_vecs = diff / distances[:, np.newaxis]  # (N, 3)

    # Element weights
    weights = surface.cell.response(state.values, freq)
    w_mag = np.abs(weights)

    # FIM: J_ij = 2*snr * sum_n |w_n|^2 * k^2 * u_ni * u_nj
    fim = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            fim[i, j] = (
                2 * snr * kw**2 * np.sum(w_mag**2 * unit_vecs[:, i] * unit_vecs[:, j])
            )

    return fim


def crlb_position(
    surface: Metasurface,
    state: SurfaceState,
    target_pos: Position3D,
    freq: float,
    snr: float = 20.0,
) -> npt.NDArray[np.floating[Any]]:
    """Compute Cramer-Rao Lower Bound on position estimation.

    CRLB = diag(J^{-1}) gives minimum variance for each
    position coordinate.

    Args:
        surface: Metasurface object.
        state: Surface configuration.
        target_pos: Target position.
        freq: Frequency [Hz].
        snr: Signal-to-noise ratio (linear).

    Returns:
        (3,) array of CRLB values [m^2] for (x, y, z).
    """
    fim = fisher_information_matrix(
        surface,
        state,
        target_pos,
        freq,
        snr,
    )
    try:
        fim_inv = np.linalg.inv(fim)
        return np.diag(fim_inv)
    except np.linalg.LinAlgError:
        return np.full(3, np.inf)
