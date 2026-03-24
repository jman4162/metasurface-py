"""Lookup-table-based unit-cell model (Level 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from metasurface_py.elements.states import StateSpace

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class LookupTableCell:
    """Unit-cell model based on a lookup table of measured/simulated responses.

    The table stores complex response indexed by (state, freq, theta_inc).
    Interpolation is used for intermediate values.

    This is the Level 1 model: angle- and frequency-dependent element response.

    Args:
        table: Complex response data with dims (state, freq, theta).
        state_space: The state space derived from the table.
    """

    table: xr.DataArray
    state_space: StateSpace

    @classmethod
    def from_xarray(cls, table: xr.DataArray) -> Self:
        """Create from an xarray DataArray with dims (state, freq, theta).

        The 'state' coordinate should contain phase values in radians
        or codebook indices.
        """
        states = table.coords["state"].values
        codebook = np.exp(1j * np.asarray(states, dtype=np.float64))
        space = StateSpace(
            kind="discrete",
            codebook=codebook,
            num_bits=(
                int(np.log2(len(states)))
                if len(states) & (len(states) - 1) == 0
                else None
            ),
        )
        return cls(table=table, state_space=space)

    @classmethod
    def from_hdf5(cls, path: str | Path) -> Self:
        """Load lookup table from an HDF5/NetCDF file."""
        ds = xr.open_dataarray(path)
        return cls.from_xarray(ds)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        state_col: str = "state",
        freq_col: str = "freq",
        theta_col: str = "theta",
        mag_col: str = "magnitude",
        phase_col: str = "phase_deg",
    ) -> Self:
        """Load lookup table from a CSV file.

        Expects columns for state, frequency, theta, magnitude, and phase.

        Args:
            path: Path to CSV file.
            state_col: Column name for state values.
            freq_col: Column name for frequency values [Hz].
            theta_col: Column name for incidence angle [degrees].
            mag_col: Column name for response magnitude (linear).
            phase_col: Column name for response phase [degrees].
        """
        import pandas as pd

        df = pd.read_csv(path)
        states = np.sort(df[state_col].unique())
        freqs = np.sort(df[freq_col].unique())
        thetas = np.sort(df[theta_col].unique())

        data = np.zeros((len(states), len(freqs), len(thetas)), dtype=np.complex128)
        for i, s in enumerate(states):
            for j, f in enumerate(freqs):
                for k, t in enumerate(thetas):
                    row = df[
                        (df[state_col] == s)
                        & (df[freq_col] == f)
                        & (df[theta_col] == t)
                    ]
                    if len(row) == 1:
                        mag = float(row[mag_col].iloc[0])
                        phase_deg = float(row[phase_col].iloc[0])
                        data[i, j, k] = mag * np.exp(1j * np.deg2rad(phase_deg))

        table = xr.DataArray(
            data,
            dims=["state", "freq", "theta"],
            coords={
                "state": states,
                "freq": freqs,
                "theta": np.deg2rad(thetas),
            },
            attrs={"unit": "complex_coefficient"},
        )
        return cls.from_xarray(table)

    @property
    def num_states(self) -> int | None:
        """Number of discrete states."""
        if self.state_space.codebook is not None:
            return len(self.state_space.codebook)
        return None

    def response(
        self,
        state: npt.NDArray[np.floating[Any]],
        freq: float,
        theta_inc: float = 0.0,
        phi_inc: float = 0.0,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]:
        """Compute interpolated complex response.

        Args:
            state: Phase values in radians, shape (N,).
            freq: Frequency [Hz].
            theta_inc: Incident polar angle [rad].
            phi_inc: Incident azimuthal angle [rad] (unused — table is phi-symmetric).

        Returns:
            Complex response coefficients, shape (N,).
        """
        state_coords = self.table.coords["state"].values.astype(np.float64)
        freq_coords = self.table.coords["freq"].values.astype(np.float64)
        theta_coords = self.table.coords["theta"].values.astype(np.float64)

        # Interpolate magnitude and phase separately for stability
        mag_data = np.abs(self.table.values)
        phase_data = np.angle(self.table.values)

        mag_interp = RegularGridInterpolator(
            (state_coords, freq_coords, theta_coords),
            mag_data,
            bounds_error=False,
            fill_value=None,
        )
        phase_interp = RegularGridInterpolator(
            (state_coords, freq_coords, theta_coords),
            phase_data,
            bounds_error=False,
            fill_value=None,
        )

        state_arr = np.asarray(state, dtype=np.float64)
        points = np.column_stack(
            [
                state_arr,
                np.full_like(state_arr, freq),
                np.full_like(state_arr, theta_inc),
            ]
        )

        mag = mag_interp(points)
        phase = phase_interp(points)
        return mag * np.exp(1j * phase)  # type: ignore[no-any-return]
