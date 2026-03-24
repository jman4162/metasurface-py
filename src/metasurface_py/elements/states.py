"""State space representations and quantization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class StateSpace:
    """Base state space description.

    Args:
        kind: "continuous" or "discrete".
        num_bits: Number of bits for discrete states (None for continuous).
        codebook: Complex codebook values for discrete states.
        bounds: (lower, upper) bounds for continuous states.
    """

    kind: Literal["continuous", "discrete"]
    num_bits: int | None = None
    codebook: npt.NDArray[np.complexfloating[Any, Any]] | None = None
    bounds: tuple[float, float] | None = None


def ContinuousPhaseSpace(
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
) -> StateSpace:
    """Create a continuous phase state space.

    Args:
        bounds: (lower, upper) phase bounds in radians.
    """
    return StateSpace(kind="continuous", bounds=bounds)


def DiscretePhaseSpace(num_bits: int) -> StateSpace:
    """Create a discrete uniform-phase state space.

    Generates a uniform codebook with 2^num_bits states evenly spaced
    around the unit circle.

    Args:
        num_bits: Number of quantization bits (1 -> 2 states, 2 -> 4 states, etc.).
    """
    n_states = 2**num_bits
    phases = np.linspace(0, 2 * np.pi, n_states, endpoint=False)
    codebook = np.exp(1j * phases)
    return StateSpace(kind="discrete", num_bits=num_bits, codebook=codebook)


def CustomCodebook(
    values: npt.NDArray[np.complexfloating[Any, Any]],
) -> StateSpace:
    """Create a discrete state space from an arbitrary complex codebook.

    Args:
        values: Complex codebook entries, shape (num_states,).
    """
    values = np.asarray(values, dtype=np.complex128)
    num_bits_approx = int(np.log2(len(values)))
    return StateSpace(
        kind="discrete",
        num_bits=num_bits_approx if 2**num_bits_approx == len(values) else None,
        codebook=values,
    )


def quantize(
    continuous_state: npt.NDArray[np.floating[Any]],
    codebook: npt.NDArray[np.complexfloating[Any, Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Project continuous phase values to nearest codebook entries.

    Args:
        continuous_state: Phase values in radians, shape (N,).
        codebook: Complex codebook entries, shape (M,).

    Returns:
        Quantized phase values in radians, shape (N,).
    """
    continuous_phasors = np.exp(1j * np.asarray(continuous_state, dtype=np.float64))
    codebook = np.asarray(codebook, dtype=np.complex128)
    # Find nearest codebook entry by minimum angular distance
    # This is equivalent to maximum real part of (continuous * conj(codebook))
    inner = np.outer(continuous_phasors, np.conj(codebook))  # (N, M)
    indices = np.argmax(np.real(inner), axis=1)
    return np.angle(codebook[indices])  # type: ignore[no-any-return]


def random_state(
    space: StateSpace,
    shape: int | tuple[int, ...],
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.floating[Any]]:
    """Generate random states from a state space.

    Args:
        space: The state space to sample from.
        shape: Shape of the output array.
        rng: Random number generator. Uses default if None.

    Returns:
        Random state values (phases in radians for phase spaces).
    """
    if rng is None:
        rng = np.random.default_rng()

    if space.kind == "continuous":
        lo, hi = space.bounds or (0.0, 2 * np.pi)
        return rng.uniform(lo, hi, size=shape)
    else:
        if space.codebook is None:
            raise ValueError("Discrete state space requires a codebook")
        indices = rng.integers(0, len(space.codebook), size=shape)
        return np.angle(space.codebook[indices])  # type: ignore[no-any-return]
