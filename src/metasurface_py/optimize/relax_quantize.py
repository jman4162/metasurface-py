"""Relax-then-quantize optimization pipeline."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from metasurface_py.optimize.continuous import optimize_continuous
from metasurface_py.optimize.discrete import refine_discrete
from metasurface_py.optimize.result import OptimizationResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from metasurface_py.core.types import AngleGrid
    from metasurface_py.surfaces.metasurface import Metasurface


def relax_then_quantize(
    objective: Callable[
        [npt.NDArray[np.floating[Any]], Metasurface, float],
        float,
    ],
    surface: Metasurface,
    freq: float,
    angles: AngleGrid,
    continuous_method: Literal["L-BFGS-B", "differential_evolution"] = "L-BFGS-B",
    refine: bool = True,
    maxiter: int = 200,
    seed: int | None = None,
) -> OptimizationResult:
    """Optimize via relax-then-quantize pipeline.

    1. Optimize continuous phases
    2. Quantize to the surface's discrete codebook
    3. Optionally refine with coordinate descent

    This is the standard approach in metasurface optimization
    literature and the recommended default workflow.

    Args:
        objective: Callable(state, surface, freq) -> float to minimize.
        surface: Metasurface object.
        freq: Frequency [Hz].
        angles: Observation angle grid.
        continuous_method: Method for continuous optimization.
        refine: Whether to apply discrete refinement after quantization.
        maxiter: Max iterations for continuous optimization.
        seed: Random seed.

    Returns:
        OptimizationResult with final state and full history.
    """
    t0 = time.perf_counter()

    # Step 1: Continuous optimization
    cont_result = optimize_continuous(
        objective,
        surface,
        freq,
        angles,
        method=continuous_method,
        maxiter=maxiter,
        seed=seed,
    )
    continuous_state = cont_result.state

    # Step 2: Quantize
    codebook = surface.cell.state_space.codebook
    if codebook is None:
        msg = "relax_then_quantize requires a discrete cell state space with a codebook"
        raise ValueError(msg)

    quantized_state = continuous_state.quantize(codebook)
    quantized_val = objective(
        quantized_state.values,
        surface,
        freq,
    )

    # Step 3: Optional discrete refinement
    if refine:
        refined_result = refine_discrete(
            objective,
            surface,
            quantized_state,
            freq,
            angles,
            max_sweeps=2,
        )
        final_state = refined_result.state
        final_val = refined_result.objective_value
        refine_history = refined_result.convergence_history
    else:
        final_state = quantized_state
        final_val = quantized_val
        refine_history = np.array([quantized_val], dtype=np.float64)

    elapsed = time.perf_counter() - t0

    # Combine histories
    full_history = np.concatenate(
        [
            cont_result.convergence_history,
            np.array([quantized_val], dtype=np.float64),
            refine_history,
        ]
    )

    return OptimizationResult(
        state=final_state,
        state_continuous=continuous_state,
        objective_value=final_val,
        convergence_history=full_history,
        runtime_seconds=elapsed,
        method=f"relax_then_quantize({continuous_method})",
        config={
            "freq": freq,
            "continuous_method": continuous_method,
            "refine": refine,
            "maxiter": maxiter,
            "seed": seed,
        },
    )
