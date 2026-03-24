"""Optimization result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from metasurface_py.surfaces.state import SurfaceState


@dataclass(frozen=True)
class OptimizationResult:
    """Result of a metasurface optimization run.

    Args:
        state: Final optimized surface state.
        state_continuous: Pre-quantization continuous state (if applicable).
        objective_value: Final objective function value.
        convergence_history: Objective value per iteration.
        runtime_seconds: Wall-clock time for optimization.
        method: Name of the optimization method used.
        config: Frozen dict of all optimization parameters.
    """

    state: SurfaceState
    state_continuous: SurfaceState | None = None
    objective_value: float = 0.0
    convergence_history: npt.NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    runtime_seconds: float = 0.0
    method: str = ""
    config: dict[str, Any] = field(default_factory=dict)
