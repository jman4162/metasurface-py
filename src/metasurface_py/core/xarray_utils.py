"""Helpers for building labeled xarray datasets from simulation outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr

from metasurface_py.core.types import DIM_FREQ, DIM_PHI, DIM_THETA


def make_pattern_dataset(
    data: npt.NDArray[np.complexfloating[Any, Any]] | npt.NDArray[np.floating[Any]],
    theta: npt.NDArray[np.floating[Any]],
    phi: npt.NDArray[np.floating[Any]],
    freq: float | npt.NDArray[np.floating[Any]] | None = None,
    name: str = "pattern",
    attrs: dict[str, Any] | None = None,
) -> xr.DataArray:
    """Create a labeled far-field pattern DataArray.

    Args:
        data: Pattern data. Shape must be (n_theta, n_phi) or (n_freq, n_theta, n_phi).
        theta: Theta values [rad].
        phi: Phi values [rad].
        freq: Frequency value(s) [Hz]. If scalar or None, the freq dimension is omitted
              for 2D data.
        name: Name for the DataArray.
        attrs: Additional attributes (e.g., {"unit": "dBi"}).

    Returns:
        Labeled xr.DataArray with dimensions [theta, phi] or [freq, theta, phi].
    """
    theta_arr = np.asarray(theta, dtype=np.float64)
    phi_arr = np.asarray(phi, dtype=np.float64)
    all_attrs: dict[str, Any] = {"coordinate_system": "spherical_iso"}
    if attrs:
        all_attrs.update(attrs)

    if data.ndim == 2:
        result = xr.DataArray(
            data,
            dims=[DIM_THETA, DIM_PHI],
            coords={
                DIM_THETA: ("theta", theta_arr, {"unit": "rad"}),
                DIM_PHI: ("phi", phi_arr, {"unit": "rad"}),
            },
            name=name,
            attrs=all_attrs,
        )
    elif data.ndim == 3:
        if freq is None:
            raise ValueError("freq must be provided for 3D pattern data")
        freq_arr = np.atleast_1d(np.asarray(freq, dtype=np.float64))
        result = xr.DataArray(
            data,
            dims=[DIM_FREQ, DIM_THETA, DIM_PHI],
            coords={
                DIM_FREQ: ("freq", freq_arr, {"unit": "Hz"}),
                DIM_THETA: ("theta", theta_arr, {"unit": "rad"}),
                DIM_PHI: ("phi", phi_arr, {"unit": "rad"}),
            },
            name=name,
            attrs=all_attrs,
        )
    else:
        raise ValueError(f"Expected 2D or 3D data, got shape {data.shape}")

    return result


def make_element_dataset(
    data: npt.NDArray[np.complexfloating[Any, Any]] | npt.NDArray[np.floating[Any]],
    nx: int,
    ny: int,
    name: str = "element_data",
    attrs: dict[str, Any] | None = None,
) -> xr.DataArray:
    """Create a labeled element-indexed DataArray.

    Args:
        data: Per-element data, shape (nx, ny) or (nx*ny,).
        nx: Number of elements in x.
        ny: Number of elements in y.
        name: Name for the DataArray.
        attrs: Additional attributes.

    Returns:
        Labeled xr.DataArray with dimensions [x_elem, y_elem].
    """
    if data.ndim == 1:
        data = data.reshape(nx, ny)

    return xr.DataArray(
        data,
        dims=["x_elem", "y_elem"],
        coords={
            "x_elem": np.arange(nx),
            "y_elem": np.arange(ny),
        },
        name=name,
        attrs=attrs or {},
    )
