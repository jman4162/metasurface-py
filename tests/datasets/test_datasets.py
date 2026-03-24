"""Tests for datasets module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from metasurface_py.datasets import load_result, save_result
from metasurface_py.datasets.sweeps import frequency_sweep
from metasurface_py.experiments import ExperimentConfig

if TYPE_CHECKING:
    from pathlib import Path


class TestFrequencySweep:
    def test_frequency_sweep(self) -> None:
        config = ExperimentConfig(
            nx=4,
            ny=4,
            num_bits=2,
            freq=10e9,
            theta_points=45,
            phi_points=18,
        )
        freqs = np.array([9e9, 10e9, 11e9])
        ds = frequency_sweep(config, freqs)
        assert "peak_gain_dbi" in ds
        assert len(ds.coords["freq"]) == 3


class TestResultIO:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        import xarray as xr

        ds = xr.Dataset(
            {"gain": ("freq", [10.0, 12.0, 14.0])},
            coords={"freq": [1e9, 2e9, 3e9]},
        )
        path = tmp_path / "test_result.nc"
        save_result(ds, path)
        loaded = load_result(path)
        np.testing.assert_allclose(
            loaded["gain"].values,
            [10.0, 12.0, 14.0],
        )
