"""Continuous optimization wrappers around SciPy."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import differential_evolution, minimize

from metasurface_py.optimize.result import OptimizationResult
from metasurface_py.surfaces.state import SurfaceState

if TYPE_CHECKING:
    from collections.abc import Callable

    from metasurface_py.core.types import AngleGrid
    from metasurface_py.surfaces.metasurface import Metasurface


def optimize_continuous(
    objective: Callable[
        [npt.NDArray[np.floating[Any]], Metasurface, float],
        float,
    ],
    surface: Metasurface,
    freq: float,
    angles: AngleGrid,
    method: Literal["L-BFGS-B", "differential_evolution"] = "L-BFGS-B",
    x0: npt.NDArray[np.floating[Any]] | None = None,
    maxiter: int = 200,
    seed: int | None = None,
    **scipy_kwargs: Any,
) -> OptimizationResult:
    """Optimize continuous phase values using SciPy.

    Args:
        objective: Callable(state, surface, freq) -> float to minimize.
        surface: Metasurface object.
        freq: Frequency [Hz].
        angles: Observation angles (passed through for reference).
        method: "L-BFGS-B" (local) or "differential_evolution" (global).
        x0: Initial phase values [rad]. Random if None.
        maxiter: Maximum iterations.
        seed: Random seed for reproducibility.
        **scipy_kwargs: Additional kwargs for the SciPy optimizer.

    Returns:
        OptimizationResult with optimized continuous state.
    """
    n = surface.num_elements
    bounds = [(0.0, 2 * np.pi)] * n
    history: list[float] = []

    if x0 is None:
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(0, 2 * np.pi, n)

    def _obj(x: npt.NDArray[np.floating[Any]]) -> float:
        val = objective(x, surface, freq)
        history.append(val)
        return val

    t0 = time.perf_counter()

    if method == "L-BFGS-B":
        result = minimize(
            _obj,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, **scipy_kwargs},
        )
        best_x = result.x
        best_val = float(result.fun)
    elif method == "differential_evolution":
        result = differential_evolution(
            _obj,
            bounds=bounds,
            maxiter=maxiter,
            seed=seed,
            **scipy_kwargs,
        )
        best_x = result.x
        best_val = float(result.fun)
    else:
        msg = f"Unknown method: {method}"
        raise ValueError(msg)

    elapsed = time.perf_counter() - t0

    state = SurfaceState(
        values=np.asarray(best_x, dtype=np.float64),
        space=surface.cell.state_space,
    )

    return OptimizationResult(
        state=state,
        objective_value=best_val,
        convergence_history=np.array(history, dtype=np.float64),
        runtime_seconds=elapsed,
        method=method,
        config={
            "freq": freq,
            "maxiter": maxiter,
            "seed": seed,
            "method": method,
        },
    )
