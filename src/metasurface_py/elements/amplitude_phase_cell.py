"""Amplitude-phase coupled unit-cell model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

if TYPE_CHECKING:
    from metasurface_py.elements.states import StateSpace


@dataclass(frozen=True)
class AmplitudePhaseCell:
    """Element with coupled amplitude-phase response.

    Common in varactor-based designs where the amplitude and phase
    are both functions of the control voltage/state.

    Args:
        state_space: Admissible state space.
        amplitude_vs_state: Amplitude values at codebook points.
        phase_vs_state: Phase values [rad] at codebook points.
        control_points: Control state values corresponding to
            amplitude/phase arrays. If None, uses uniform spacing
            over [0, 2*pi].
    """

    state_space: StateSpace
    amplitude_vs_state: npt.NDArray[np.floating[Any]]
    phase_vs_state: npt.NDArray[np.floating[Any]]
    control_points: npt.NDArray[np.floating[Any]] | None = None

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
        """Compute coupled amplitude-phase response.

        Interpolates amplitude and phase as functions of the
        control state.

        Args:
            state: Control state values, shape (N,).
            freq: Frequency [Hz] (unused in this model).
            theta_inc: Incident angle [rad] (unused).
            phi_inc: Incident angle [rad] (unused).

        Returns:
            Complex response, shape (N,).
        """
        if self.control_points is not None:
            cp = self.control_points
        else:
            n = len(self.amplitude_vs_state)
            cp = np.linspace(0, 2 * np.pi, n, endpoint=False)

        amp_interp = interp1d(
            cp,
            self.amplitude_vs_state,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        phase_interp = interp1d(
            cp,
            self.phase_vs_state,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        state_arr = np.asarray(state, dtype=np.float64)
        amp = amp_interp(state_arr)
        phase = phase_interp(state_arr)
        return amp * np.exp(1j * phase)  # type: ignore[no-any-return]
