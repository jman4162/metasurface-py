"""Result serialization utilities."""

from __future__ import annotations

from pathlib import Path

import xarray as xr


def save_result(
    dataset: xr.Dataset,
    path: str | Path,
) -> Path:
    """Save an xarray Dataset to NetCDF.

    Args:
        dataset: Dataset to save (e.g., from run_sweep).
        path: Output file path (.nc).

    Returns:
        Path to saved file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(p)
    return p


def load_result(path: str | Path) -> xr.Dataset:
    """Load an xarray Dataset from NetCDF.

    Args:
        path: Path to NetCDF file.

    Returns:
        Loaded Dataset.
    """
    return xr.open_dataset(Path(path))
