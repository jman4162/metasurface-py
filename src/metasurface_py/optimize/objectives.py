"""Objective functions for metasurface optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from metasurface_py.core.conventions import k0
from metasurface_py.em.array_factor import (
    array_factor,
    directivity,
    sidelobe_level,
)

if TYPE_CHECKING:
    from metasurface_py.core.types import AngleGrid
    from metasurface_py.surfaces.metasurface import Metasurface


def _evaluate_pattern(
    state: npt.NDArray[np.floating[Any]],
    surface: Metasurface,
    freq: float,
    angles: AngleGrid,
) -> npt.NDArray[np.complexfloating[Any, Any]]:
    """Compute array factor from raw phase state."""
    weights = surface.cell.response(state, freq)
    kw = k0(freq)
    return array_factor(
        surface.positions,
        weights,
        kw,
        angles.theta,
        angles.phi,
    )


@dataclass(frozen=True)
class MaxGainObjective:
    """Maximize directivity in a target direction.

    Returns negative gain (for minimization).

    Args:
        target_theta: Target polar angle [rad].
        target_phi: Target azimuthal angle [rad].
        angles: Observation angle grid for pattern evaluation.
    """

    target_theta: float
    target_phi: float
    angles: AngleGrid

    def __call__(
        self,
        state: npt.NDArray[np.floating[Any]],
        surface: Metasurface,
        freq: float,
        **kwargs: Any,
    ) -> float:
        """Evaluate: returns negative peak gain (minimize this)."""
        from metasurface_py.core.xarray_utils import (
            make_pattern_dataset,
        )

        af = _evaluate_pattern(state, surface, freq, self.angles)
        pattern = make_pattern_dataset(
            af,
            theta=self.angles.theta,
            phi=self.angles.phi,
        )
        d = directivity(pattern)

        # Find value nearest to target direction
        theta_idx = int(np.argmin(np.abs(self.angles.theta - self.target_theta)))
        phi_idx = int(np.argmin(np.abs(self.angles.phi - self.target_phi)))
        gain_linear = float(d.values[theta_idx, phi_idx])
        if gain_linear <= 0:
            return 100.0  # penalty
        return float(-10.0 * np.log10(gain_linear))


@dataclass(frozen=True)
class MinSidelobeObjective:
    """Minimize peak sidelobe level.

    Returns positive SLL (in dB, closer to 0 is worse).

    Args:
        target_theta: Main beam polar angle [rad].
        target_phi: Main beam azimuthal angle [rad].
        angles: Observation angle grid.
        exclusion_radius: Angular exclusion around main beam [rad].
    """

    target_theta: float
    target_phi: float
    angles: AngleGrid
    exclusion_radius: float = 0.15

    def __call__(
        self,
        state: npt.NDArray[np.floating[Any]],
        surface: Metasurface,
        freq: float,
        **kwargs: Any,
    ) -> float:
        """Evaluate: returns negative SLL (minimize for lower sidelobes)."""
        from metasurface_py.core.xarray_utils import (
            make_pattern_dataset,
        )

        af = _evaluate_pattern(state, surface, freq, self.angles)
        pattern = make_pattern_dataset(
            af,
            theta=self.angles.theta,
            phi=self.angles.phi,
        )
        sll = sidelobe_level(
            pattern,
            self.target_theta,
            self.target_phi,
            self.exclusion_radius,
        )
        # SLL is already negative (dB below peak)
        # We want to minimize (make more negative), so return -SLL
        return -sll


@dataclass(frozen=True)
class WeightedGainSidelobeObjective:
    """Weighted combination of gain and sidelobe level.

    objective = alpha * (-gain_dBi) + (1-alpha) * (-SLL_dB)

    Lower is better for both terms.

    Args:
        target_theta: Target beam direction [rad].
        target_phi: Target beam azimuthal angle [rad].
        angles: Observation angle grid.
        alpha: Weight for gain term (0 to 1). Default 0.7.
        exclusion_radius: SLL exclusion radius [rad].
    """

    target_theta: float
    target_phi: float
    angles: AngleGrid
    alpha: float = 0.7
    exclusion_radius: float = 0.15

    def __call__(
        self,
        state: npt.NDArray[np.floating[Any]],
        surface: Metasurface,
        freq: float,
        **kwargs: Any,
    ) -> float:
        """Evaluate weighted objective."""
        gain_obj = MaxGainObjective(
            self.target_theta,
            self.target_phi,
            self.angles,
        )
        sll_obj = MinSidelobeObjective(
            self.target_theta,
            self.target_phi,
            self.angles,
            self.exclusion_radius,
        )
        gain_val = gain_obj(state, surface, freq)
        sll_val = sll_obj(state, surface, freq)
        return self.alpha * gain_val + (1 - self.alpha) * sll_val


@dataclass(frozen=True)
class MaxCapacityObjective:
    """Maximize MIMO capacity for an RIS-assisted link.

    Returns negative capacity (for minimization).

    Args:
        mimo_link: MIMORISLink instance.
        snr_linear: Total SNR (linear).
    """

    mimo_link: Any  # MIMORISLink — Any to avoid circular import
    snr_linear: float = 100.0

    def __call__(
        self,
        state: npt.NDArray[np.floating[Any]],
        surface: Metasurface,
        freq: float,
        **kwargs: Any,
    ) -> float:
        """Evaluate: returns negative capacity."""
        from metasurface_py.surfaces.state import SurfaceState

        ss = SurfaceState(
            values=state,
            space=surface.cell.state_space,
        )
        cap = self.mimo_link.capacity(ss, self.snr_linear)
        return float(-cap)
