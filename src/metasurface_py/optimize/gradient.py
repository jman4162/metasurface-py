"""Gradient-based optimization using JAX autodiff."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from metasurface_py.optimize.result import OptimizationResult


def optimize_gradient(
    objective_jax: Any,
    n_elements: int,
    maxiter: int = 500,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> OptimizationResult:
    """Gradient descent optimization using JAX autodiff.

    Uses jax.grad for exact gradients instead of finite differences.
    Significantly faster than SciPy for large arrays.

    Args:
        objective_jax: JAX-compatible callable(state) -> scalar.
            Must be differentiable via jax.grad.
        n_elements: Number of phase elements.
        maxiter: Maximum iterations.
        learning_rate: Step size for gradient descent.
        seed: Random seed for initial state.

    Returns:
        OptimizationResult with optimized state.

    Raises:
        ImportError: If JAX is not installed.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as e:
        msg = (
            "JAX is required for gradient optimization. "
            "Install with: pip install metasurface-py[jax]"
        )
        raise ImportError(msg) from e

    grad_fn = jax.grad(objective_jax)

    key = jax.random.PRNGKey(seed)
    state = jax.random.uniform(
        key,
        shape=(n_elements,),
        minval=0.0,
        maxval=2 * np.pi,
    )

    history: list[float] = []
    t0 = time.perf_counter()

    for _i in range(maxiter):
        val = float(objective_jax(state))
        history.append(val)
        g = grad_fn(state)
        state = state - learning_rate * g
        # Wrap to [0, 2*pi]
        state = state % (2 * jnp.pi)

    elapsed = time.perf_counter() - t0
    final_val = float(objective_jax(state))

    from metasurface_py.elements.states import ContinuousPhaseSpace
    from metasurface_py.surfaces.state import SurfaceState

    return OptimizationResult(
        state=SurfaceState(
            values=np.asarray(state, dtype=np.float64),
            space=ContinuousPhaseSpace(),
        ),
        objective_value=final_val,
        convergence_history=np.array(history, dtype=np.float64),
        runtime_seconds=elapsed,
        method="jax_gradient_descent",
        config={
            "maxiter": maxiter,
            "learning_rate": learning_rate,
            "seed": seed,
        },
    )
