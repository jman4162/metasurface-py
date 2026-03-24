"""Antenna array geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from metasurface_py.core.types import Position3D


@dataclass(frozen=True)
class UniformLinearArray:
    """Uniform linear array (ULA) of antennas.

    Args:
        num_antennas: Number of antenna elements.
        spacing: Inter-element spacing [meters].
        center: Center position of the array.
        axis: Unit vector along array axis. Default [1,0,0] (x-axis).
    """

    num_antennas: int
    spacing: float
    center: Position3D
    axis: npt.NDArray[np.floating[Any]] | None = None

    @property
    def positions(self) -> npt.NDArray[np.floating[Any]]:
        """Antenna positions, shape (M, 3) [meters]."""
        ax = (
            self.axis
            if self.axis is not None
            else np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float64,
            )
        )
        ax = ax / np.linalg.norm(ax)
        offsets = (
            np.arange(self.num_antennas, dtype=np.float64)
            - (self.num_antennas - 1) / 2.0
        ) * self.spacing
        center = np.array(
            [self.center.x, self.center.y, self.center.z],
            dtype=np.float64,
        )
        return center[np.newaxis, :] + offsets[:, np.newaxis] * ax[np.newaxis, :]
