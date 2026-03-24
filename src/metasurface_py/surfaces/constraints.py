"""Hardware constraint functions for surface states."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metasurface_py.elements.states import quantize


def phase_quantize(
    state: npt.NDArray[np.floating[Any]],
    num_bits: int,
) -> npt.NDArray[np.floating[Any]]:
    """Quantize continuous phases to uniform discrete levels.

    Args:
        state: Phase values in radians, shape (N,).
        num_bits: Number of quantization bits (1, 2, 3, ...).

    Returns:
        Quantized phase values, shape (N,).
    """
    n_levels = 2**num_bits
    phases = np.linspace(0, 2 * np.pi, n_levels, endpoint=False)
    codebook = np.exp(1j * phases)
    return quantize(state, codebook)


def apply_group_constraint(
    state: npt.NDArray[np.floating[Any]],
    groups: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Enforce grouping: elements in the same group share the majority phase.

    Args:
        state: Phase values, shape (N,).
        groups: Group ID per element, shape (N,).

    Returns:
        Constrained phase values, shape (N,).
    """
    result = state.copy()
    for gid in np.unique(groups):
        members = groups == gid
        mean_phase = float(np.angle(np.mean(np.exp(1j * state[members]))))
        result[members] = mean_phase
    return result


def apply_mask(
    state: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.floating[Any]]:
    """Zero out masked (inactive) elements.

    Args:
        state: Phase values, shape (N,).
        mask: Boolean mask, True = active, shape (N,).

    Returns:
        State with inactive elements set to zero.
    """
    result = state.copy()
    result[~mask] = 0.0
    return result


def add_manufacturing_noise(
    state: npt.NDArray[np.floating[Any]],
    std_dev: float,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.floating[Any]]:
    """Add Gaussian phase noise to simulate manufacturing variability.

    Args:
        state: Phase values, shape (N,).
        std_dev: Standard deviation of phase noise [rad].
        rng: Random number generator. Uses default if None.

    Returns:
        Noisy phase values, shape (N,).
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, std_dev, size=state.shape)
    return state + noise
