"""Discrete refinement via coordinate descent."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from metasurface_py.optimize.result import OptimizationResult
from metasurface_py.surfaces.state import SurfaceState

if TYPE_CHECKING:
    from collections.abc import Callable

    from metasurface_py.core.types import AngleGrid
    from metasurface_py.surfaces.metasurface import Metasurface


def refine_discrete(
    objective: Callable[
        [npt.NDArray[np.floating[Any]], Metasurface, float],
        float,
    ],
    surface: Metasurface,
    state: SurfaceState,
    freq: float,
    angles: AngleGrid,
    max_sweeps: int = 3,
) -> OptimizationResult:
    """Refine a discrete state via coordinate descent.

    For each element, tries all codebook entries and keeps the best.
    Repeats for max_sweeps passes.

    Args:
        objective: Callable(state, surface, freq) -> float to minimize.
        surface: Metasurface object.
        state: Initial discrete surface state.
        freq: Frequency [Hz].
        angles: Observation angles (for reference).
        max_sweeps: Number of full sweeps over all elements.

    Returns:
        OptimizationResult with refined discrete state.
    """
    codebook = state.space.codebook
    if codebook is None:
        msg = "refine_discrete requires a discrete state space"
        raise ValueError(msg)

    codebook_phases = np.angle(codebook)
    current = state.values.copy()
    best_val = objective(current, surface, freq)
    history = [best_val]

    t0 = time.perf_counter()

    for _sweep in range(max_sweeps):
        improved = False
        for i in range(len(current)):
            original = current[i]
            best_phase = original

            for cp in codebook_phases:
                current[i] = cp
                val = objective(current, surface, freq)
                if val < best_val:
                    best_val = val
                    best_phase = cp
                    improved = True

            current[i] = best_phase
            history.append(best_val)

        if not improved:
            break

    elapsed = time.perf_counter() - t0

    final_state = SurfaceState(
        values=current.copy(),
        space=state.space,
        mask=state.mask,
    )

    return OptimizationResult(
        state=final_state,
        objective_value=best_val,
        convergence_history=np.array(history, dtype=np.float64),
        runtime_seconds=elapsed,
        method="coordinate_descent",
        config={"max_sweeps": max_sweeps, "freq": freq},
    )
