"""Protocol defining the unit-cell model interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from metasurface_py.elements.states import StateSpace


@runtime_checkable
class UnitCellModel(Protocol):
    """Protocol for any unit-cell response model.

    A UnitCellModel maps control state + incident conditions to complex
    reflection or transmission coefficients.
    """

    @property
    def num_states(self) -> int | None:
        """Number of discrete states, or None for continuous."""
        ...

    @property
    def state_space(self) -> StateSpace:
        """The admissible state space for this cell."""
        ...

    def response(
        self,
        state: npt.NDArray[np.floating[Any]],
        freq: float,
        theta_inc: float = 0.0,
        phi_inc: float = 0.0,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]:
        """Compute complex response for given states and incident conditions.

        Args:
            state: Control state values, shape (N,) where N is number of elements.
                   For phase-only cells, values are phases in radians.
                   For discrete cells, values are codebook indices (as floats).
            freq: Frequency [Hz].
            theta_inc: Incident polar angle [rad].
            phi_inc: Incident azimuthal angle [rad].

        Returns:
            Complex response coefficients, shape (N,).
        """
        ...
