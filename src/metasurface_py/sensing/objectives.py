"""Sensing-specific optimization objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from metasurface_py.sensing.radar import detection_snr

if TYPE_CHECKING:
    from metasurface_py.core.types import Position3D
    from metasurface_py.surfaces.metasurface import Metasurface


@dataclass(frozen=True)
class MaxDetectionSNRObjective:
    """Maximize radar detection SNR for a point target.

    Returns negative SNR (dB) for minimization.

    Args:
        target_pos: Target position.
        tx_power: Transmit power [W].
        noise_power: Noise power [W].
        target_rcs: Target RCS [m^2].
    """

    target_pos: Position3D
    tx_power: float = 1.0
    noise_power: float = 1e-12
    target_rcs: float = 1.0

    def __call__(
        self,
        state: npt.NDArray[np.floating[Any]],
        surface: Metasurface,
        freq: float,
        **kwargs: Any,
    ) -> float:
        """Evaluate: returns negative detection SNR in dB."""
        from metasurface_py.surfaces.state import SurfaceState

        ss = SurfaceState(
            values=state,
            space=surface.cell.state_space,
        )
        snr_linear = detection_snr(
            surface,
            ss,
            self.target_pos,
            freq,
            self.tx_power,
            self.noise_power,
            self.target_rcs,
        )
        if snr_linear <= 0:
            return 100.0
        return float(-10.0 * np.log10(snr_linear))


@dataclass(frozen=True)
class JointCommsSensingObjective:
    """Weighted combination of comms and sensing objectives.

    objective = alpha * comms_obj + (1-alpha) * sensing_obj

    Args:
        comms_objective: Communication objective callable.
        sensing_objective: Sensing objective callable.
        alpha: Weight for comms (0 to 1). Default 0.5.
    """

    comms_objective: Any
    sensing_objective: Any
    alpha: float = 0.5

    def __call__(
        self,
        state: npt.NDArray[np.floating[Any]],
        surface: Metasurface,
        freq: float,
        **kwargs: Any,
    ) -> float:
        """Evaluate weighted joint objective."""
        comms_val = self.comms_objective(state, surface, freq)
        sensing_val = self.sensing_objective(state, surface, freq)
        return float(self.alpha * comms_val + (1 - self.alpha) * sensing_val)
