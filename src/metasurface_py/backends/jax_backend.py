"""JAX-accelerated computation kernels.

Provides JIT-compiled, differentiable versions of core functions.
Requires JAX: pip install metasurface-py[jax]

Usage:
    from metasurface_py.backends.jax_backend import (
        array_factor_jax, directivity_jax, max_gain_objective_jax
    )
"""

from __future__ import annotations

from typing import Any

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def _check_jax() -> None:
    if not HAS_JAX:
        msg = (
            "JAX is required for this backend. "
            "Install with: pip install metasurface-py[jax]"
        )
        raise ImportError(msg)


def array_factor_jax(
    positions: Any,
    weights: Any,
    wavenumber: float,
    theta: Any,
    phi: Any,
) -> Any:
    """JIT-compiled array factor computation.

    Same interface as em.array_factor but uses JAX for:
    - JIT compilation (faster repeated calls)
    - Automatic differentiation (jax.grad compatible)
    - GPU acceleration (if available)

    Args:
        positions: Element positions, shape (N, 3).
        weights: Complex element weights, shape (N,).
        wavenumber: Free-space wavenumber k0.
        theta: Observation polar angles, shape (n_theta,).
        phi: Observation azimuthal angles, shape (n_phi,).

    Returns:
        Complex array factor, shape (n_theta, n_phi).
    """
    _check_jax()
    return _array_factor_jit(
        jnp.asarray(positions),
        jnp.asarray(weights),
        wavenumber,
        jnp.asarray(theta),
        jnp.asarray(phi),
    )


def directivity_jax(
    af_values: Any,
    theta: Any,
    phi: Any,
) -> Any:
    """JAX-accelerated directivity computation.

    Args:
        af_values: Complex array factor, shape (n_theta, n_phi).
        theta: Theta values [rad], shape (n_theta,).
        phi: Phi values [rad], shape (n_phi,).

    Returns:
        Directivity pattern (linear), shape (n_theta, n_phi).
    """
    _check_jax()
    return _directivity_impl(
        jnp.asarray(af_values),
        jnp.asarray(theta),
        jnp.asarray(phi),
    )


def max_gain_objective_jax(
    state: Any,
    positions: Any,
    wavenumber: float,
    target_theta: float,
    target_phi: float,
    theta: Any,
    phi: Any,
) -> Any:
    """Differentiable max-gain objective for JAX optimization.

    Returns negative gain at target direction.
    Compatible with jax.grad for gradient-based optimization.

    Args:
        state: Phase values, shape (N,).
        positions: Element positions, shape (N, 3).
        wavenumber: k0.
        target_theta: Target polar angle [rad].
        target_phi: Target azimuthal angle [rad].
        theta: Observation angles for integration.
        phi: Observation angles for integration.

    Returns:
        Scalar: negative gain in dBi.
    """
    _check_jax()
    return _max_gain_impl(
        jnp.asarray(state),
        jnp.asarray(positions),
        wavenumber,
        target_theta,
        target_phi,
        jnp.asarray(theta),
        jnp.asarray(phi),
    )


# ── JIT-compiled implementations ──────────────────────────────

if HAS_JAX:

    @jax.jit  # type: ignore[untyped-decorator]
    def _array_factor_jit(
        positions: Any,
        weights: Any,
        wavenumber: float,
        theta: Any,
        phi: Any,
    ) -> Any:
        theta_g, phi_g = jnp.meshgrid(theta, phi, indexing="ij")
        u = jnp.sin(theta_g) * jnp.cos(phi_g)
        v = jnp.sin(theta_g) * jnp.sin(phi_g)
        w = jnp.cos(theta_g)
        r_hat = jnp.stack([u, v, w], axis=-1)
        phase = wavenumber * jnp.einsum(
            "ijk,nk->ijn",
            r_hat,
            positions,
        )
        return jnp.einsum(
            "ijn,n->ij",
            jnp.exp(1j * phase),
            weights,
        )

    def _directivity_impl(
        af_values: Any,
        theta: Any,
        phi: Any,
    ) -> Any:
        power = jnp.abs(af_values) ** 2
        sin_theta = jnp.sin(theta)
        dtheta = jnp.gradient(theta)
        dphi = jnp.gradient(phi)
        integrand = power * sin_theta[:, None]
        total = jnp.sum(
            integrand * dtheta[:, None] * dphi[None, :],
        )
        return jnp.where(
            total > 1e-30,
            4.0 * jnp.pi * power / total,
            jnp.zeros_like(power),
        )

    @jax.jit  # type: ignore[untyped-decorator]
    def _max_gain_impl(
        state: Any,
        positions: Any,
        wavenumber: float,
        target_theta: float,
        target_phi: float,
        theta: Any,
        phi: Any,
    ) -> Any:
        weights = jnp.exp(1j * state)
        af = _array_factor_jit(
            positions,
            weights,
            wavenumber,
            theta,
            phi,
        )
        d = _directivity_impl(af, theta, phi)
        t_idx = jnp.argmin(jnp.abs(theta - target_theta))
        p_idx = jnp.argmin(jnp.abs(phi - target_phi))
        gain = d[t_idx, p_idx]
        return -10.0 * jnp.log10(jnp.maximum(gain, 1e-30))
