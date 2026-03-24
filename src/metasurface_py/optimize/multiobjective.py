"""Multi-objective optimization via weighted Pareto sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from metasurface_py.optimize.relax_quantize import relax_then_quantize

if TYPE_CHECKING:
    from metasurface_py.core.types import AngleGrid
    from metasurface_py.surfaces.metasurface import Metasurface
    from metasurface_py.surfaces.state import SurfaceState


@dataclass(frozen=True)
class ParetoResult:
    """Result of a Pareto sweep.

    Args:
        states: List of Pareto-optimal surface states.
        objective_values: (n_points, 2) array of objective values.
        weights: (n_points,) array of alpha weights used.
        obj_a_name: Name of first objective.
        obj_b_name: Name of second objective.
    """

    states: list[SurfaceState]
    objective_values: npt.NDArray[np.floating[Any]]
    weights: npt.NDArray[np.floating[Any]]
    obj_a_name: str = "Objective A"
    obj_b_name: str = "Objective B"


def pareto_sweep(
    objective_a: Any,
    objective_b: Any,
    surface: Metasurface,
    freq: float,
    angles: AngleGrid,
    n_points: int = 11,
    maxiter: int = 100,
    seed: int = 42,
    obj_a_name: str = "Objective A",
    obj_b_name: str = "Objective B",
) -> ParetoResult:
    """Generate Pareto front via weighted scalarization.

    Sweeps alpha from 0 to 1, optimizing:
        objective = alpha * obj_a + (1 - alpha) * obj_b

    Args:
        objective_a: First objective callable.
        objective_b: Second objective callable.
        surface: Metasurface object.
        freq: Frequency [Hz].
        angles: Observation angles.
        n_points: Number of Pareto points.
        maxiter: Max iterations per point.
        seed: Random seed.
        obj_a_name: Label for first objective.
        obj_b_name: Label for second objective.

    Returns:
        ParetoResult with states and objective values.
    """
    alphas = np.linspace(0.0, 1.0, n_points)
    states: list[SurfaceState] = []
    obj_values: list[tuple[float, float]] = []

    for alpha in alphas:

        def weighted_obj(
            state: npt.NDArray[np.floating[Any]],
            surf: Metasurface,
            f: float,
            *,
            _alpha: float = alpha,
            **kwargs: Any,
        ) -> float:
            va = objective_a(state, surf, f)
            vb = objective_b(state, surf, f)
            return float(_alpha * va + (1 - _alpha) * vb)

        result = relax_then_quantize(
            weighted_obj,
            surface,
            freq,
            angles,
            maxiter=maxiter,
            seed=seed,
            refine=True,
        )

        # Evaluate individual objectives at solution
        val_a = objective_a(
            result.state.values,
            surface,
            freq,
        )
        val_b = objective_b(
            result.state.values,
            surface,
            freq,
        )
        states.append(result.state)
        obj_values.append((val_a, val_b))

    return ParetoResult(
        states=states,
        objective_values=np.array(obj_values, dtype=np.float64),
        weights=alphas,
        obj_a_name=obj_a_name,
        obj_b_name=obj_b_name,
    )
