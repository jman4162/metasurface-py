"""Shared mathematical utility functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating[Any]]
ScalarOrArray = FloatArray | float


def db20(x: ScalarOrArray) -> ScalarOrArray:
    """Convert linear amplitude to dB (20*log10)."""
    return 20.0 * np.log10(np.abs(x))


def from_db20(x_db: ScalarOrArray) -> ScalarOrArray:
    """Convert dB to linear amplitude (10^(x/20))."""
    return 10.0 ** (np.asarray(x_db) / 20.0)


def db10(x: ScalarOrArray) -> ScalarOrArray:
    """Convert linear power to dB (10*log10)."""
    return 10.0 * np.log10(np.abs(x))


def from_db10(x_db: ScalarOrArray) -> ScalarOrArray:
    """Convert dB to linear power (10^(x/10))."""
    return 10.0 ** (np.asarray(x_db) / 10.0)


def normalize_phase(phase: ScalarOrArray) -> ScalarOrArray:
    """Wrap phase to [-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(phase)))


def spherical_to_cartesian(
    r: npt.NDArray[np.floating[Any]] | float,
    theta: npt.NDArray[np.floating[Any]] | float,
    phi: npt.NDArray[np.floating[Any]] | float,
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
]:
    """Convert spherical (r, theta, phi) to Cartesian (x, y, z).

    Uses ISO convention: theta from +z, phi from +x.
    """
    r_arr = np.asarray(r, dtype=np.float64)
    theta_arr = np.asarray(theta, dtype=np.float64)
    phi_arr = np.asarray(phi, dtype=np.float64)
    x = r_arr * np.sin(theta_arr) * np.cos(phi_arr)
    y = r_arr * np.sin(theta_arr) * np.sin(phi_arr)
    z = r_arr * np.cos(theta_arr)
    return x, y, z


def cartesian_to_spherical(
    x: npt.NDArray[np.floating[Any]] | float,
    y: npt.NDArray[np.floating[Any]] | float,
    z: npt.NDArray[np.floating[Any]] | float,
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
]:
    """Convert Cartesian (x, y, z) to spherical (r, theta, phi).

    Returns ISO convention: theta from +z, phi from +x.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)
    r = np.sqrt(x_arr**2 + y_arr**2 + z_arr**2)
    theta = np.arccos(np.clip(z_arr / np.maximum(r, 1e-30), -1.0, 1.0))
    phi = np.arctan2(y_arr, x_arr)
    return r, theta, phi


def direction_cosines(
    theta: npt.NDArray[np.floating[Any]] | float,
    phi: npt.NDArray[np.floating[Any]] | float,
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
]:
    """Compute direction cosines (u, v, w) from spherical angles.

    u = sin(theta)*cos(phi), v = sin(theta)*sin(phi), w = cos(theta).
    """
    theta_arr = np.asarray(theta, dtype=np.float64)
    phi_arr = np.asarray(phi, dtype=np.float64)
    u = np.sin(theta_arr) * np.cos(phi_arr)
    v = np.sin(theta_arr) * np.sin(phi_arr)
    w = np.cos(theta_arr)
    return u, v, w


def steering_vector(
    positions: npt.NDArray[np.floating[Any]],
    k_vec: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.complexfloating[Any, Any]]:
    """Compute array manifold/steering vector.

    Args:
        positions: Element positions, shape (N, 3) [meters].
        k_vec: Wave vector, shape (3,) [rad/m]. For a plane wave arriving
               from direction (theta, phi), k_vec = k0 * [sin(theta)cos(phi),
               sin(theta)sin(phi), cos(theta)].

    Returns:
        Complex steering vector, shape (N,).
        Convention: exp(+j * k_vec . r_n) per IEEE antenna standard.
    """
    phase = positions @ k_vec  # (N,)
    return np.exp(1j * phase)
