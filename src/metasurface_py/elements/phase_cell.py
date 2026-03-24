"""Phase-only unit-cell model (Level 0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from metasurface_py.elements.states import StateSpace


@dataclass(frozen=True)
class PhaseOnlyCell:
    """Ideal phase-shifting element.

    Each element applies a pure phase shift with optional fixed amplitude.
    This is the simplest model (Level 0): no frequency or angle dependence.

    Args:
        state_space: Admissible state space (continuous or discrete phase).
        amplitude: Fixed amplitude for all elements (default 1.0 = lossless).
    """

    state_space: StateSpace
    amplitude: float = 1.0

    @property
    def num_states(self) -> int | None:
        """Number of discrete states, or None for continuous."""
        if (
            self.state_space.kind == "discrete"
            and self.state_space.codebook is not None
        ):
            return len(self.state_space.codebook)
        return None

    def response(
        self,
        state: npt.NDArray[np.floating[Any]],
        freq: float,
        theta_inc: float = 0.0,
        phi_inc: float = 0.0,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]:
        """Compute complex response: amplitude * exp(j * phase).

        Args:
            state: Phase values in radians, shape (N,).
            freq: Frequency [Hz] (unused in this model).
            theta_inc: Incident polar angle [rad] (unused in this model).
            phi_inc: Incident azimuthal angle [rad] (unused in this model).

        Returns:
            Complex reflection/transmission coefficients, shape (N,).
        """
        return self.amplitude * np.exp(1j * np.asarray(state, dtype=np.float64))  # type: ignore[no-any-return]
