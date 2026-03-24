"""Mutual coupling approximation models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from metasurface_py.core.conventions import k0

if TYPE_CHECKING:
    from metasurface_py.geometry.lattice import SupportsLattice


def mutual_impedance_approx(
    lattice: SupportsLattice,
    freq: float,
    method: str = "canonical_dipole",
) -> npt.NDArray[np.complexfloating[Any, Any]]:
    """Compute approximate mutual impedance matrix.

    Uses canonical minimum-scattering dipole model for inter-element
    coupling. The coupling coefficient between elements i and j is
    approximated as: Z_ij ~ exp(-j*k*d_ij) / (k*d_ij) for i != j.

    Args:
        lattice: Element positions.
        freq: Frequency [Hz].
        method: Coupling model ("canonical_dipole").

    Returns:
        N x N complex coupling matrix (diagonal = 1).
    """
    if method != "canonical_dipole":
        msg = f"Unknown coupling method: {method}"
        raise ValueError(msg)

    positions = lattice.positions  # (N, 3)
    n = len(positions)
    kw = k0(freq)

    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))  # (N, N)

    # Coupling matrix
    coupling = np.eye(n, dtype=np.complex128)
    mask = dist > 1e-10
    coupling[mask] = np.exp(-1j * kw * dist[mask]) / (kw * dist[mask])

    return coupling


def apply_coupling(
    element_response: npt.NDArray[np.complexfloating[Any, Any]],
    coupling_matrix: npt.NDArray[np.complexfloating[Any, Any]],
) -> npt.NDArray[np.complexfloating[Any, Any]]:
    """Modify element responses to account for mutual coupling.

    The coupled response is: w_coupled = C @ w_isolated
    where C is the coupling matrix.

    Args:
        element_response: Isolated element responses, shape (N,).
        coupling_matrix: N x N coupling matrix.

    Returns:
        Coupled element responses, shape (N,).
    """
    coupled: npt.NDArray[np.complexfloating[Any, Any]] = (
        coupling_matrix @ element_response
    )
    return coupled
