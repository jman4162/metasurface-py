"""Phase synthesis for beam steering and focusing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from metasurface_py.core.conventions import k0
from metasurface_py.core.math_utils import direction_cosines

if TYPE_CHECKING:
    from metasurface_py.geometry.lattice import SupportsLattice


def steering_phase(
    lattice: SupportsLattice,
    theta_steer: float,
    phi_steer: float,
    freq: float,
) -> npt.NDArray[np.floating[Any]]:
    """Compute progressive phase shift for beam steering.

    For a reflective surface with normal incidence, the required phase
    at element n is: phi_n = -k0 * (x_n*u + y_n*v)
    where (u, v) = (sin(theta)*cos(phi), sin(theta)*sin(phi)).

    The negative sign ensures constructive interference in the target direction
    under our exp(-j*omega*t) convention.

    Args:
        lattice: Element positions.
        theta_steer: Target polar angle [rad].
        phi_steer: Target azimuthal angle [rad].
        freq: Frequency [Hz].

    Returns:
        Phase values per element [rad], shape (N,).
    """
    kw = k0(freq)
    positions = lattice.positions  # (N, 3)
    u, v, w = direction_cosines(theta_steer, phi_steer)
    k_vec = kw * np.array([u, v, w], dtype=np.float64).ravel()
    # Phase to steer: negate the steering vector phase
    phase = -positions @ k_vec
    return phase


def focusing_phase(
    lattice: SupportsLattice,
    focal_point: npt.NDArray[np.floating[Any]] | tuple[float, float, float],
    freq: float,
) -> npt.NDArray[np.floating[Any]]:
    """Compute phase for near-field focusing at a point.

    Phase at element n: phi_n = -k0 * |focal_point - r_n|
    (spherical wave conjugation).

    Args:
        lattice: Element positions.
        focal_point: Target focal point (x, y, z) [meters].
        freq: Frequency [Hz].

    Returns:
        Phase values per element [rad], shape (N,).
    """
    kw = k0(freq)
    positions = lattice.positions  # (N, 3)
    fp = np.asarray(focal_point, dtype=np.float64).ravel()
    distances = np.sqrt(np.sum((positions - fp[np.newaxis, :]) ** 2, axis=1))
    return -kw * distances  # type: ignore[no-any-return]


def multi_beam_phase(
    lattice: SupportsLattice,
    directions: list[tuple[float, float]],
    weights: list[float] | None,
    freq: float,
) -> npt.NDArray[np.floating[Any]]:
    """Compute phase for multi-beam synthesis (superposition approximation).

    Combines steering vectors for multiple beam directions with optional weights.
    This produces an approximate multi-beam pattern; the result should be
    optimized for best performance.

    Args:
        directions: List of (theta, phi) steering angles [rad].
        weights: Optional amplitude weights per beam. Defaults to uniform.
        freq: Frequency [Hz].

    Returns:
        Phase values per element [rad], shape (N,).
    """
    kw = k0(freq)
    positions = lattice.positions  # (N, 3)

    if weights is None:
        weights = [1.0] * len(directions)

    # Sum complex steering vectors
    composite = np.zeros(positions.shape[0], dtype=np.complex128)
    for (theta, phi), w in zip(directions, weights, strict=False):
        u, v, ww = direction_cosines(theta, phi)
        k_vec = kw * np.array([u, v, ww], dtype=np.float64).ravel()
        composite += w * np.exp(-1j * (positions @ k_vec))

    return np.angle(composite)
