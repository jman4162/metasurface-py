"""Lookup table import and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from metasurface_py.elements.lookup_cell import LookupTableCell


def import_lookup_table(
    path: str | Path,
    format: str = "auto",
    **kwargs: Any,
) -> LookupTableCell:
    """Import a lookup table from file, auto-detecting format.

    Dispatches to LookupTableCell.from_csv() or .from_hdf5()
    based on file extension.

    Args:
        path: Path to data file (.csv, .h5, .hdf5, .nc).
        format: "auto", "csv", or "hdf5".
        **kwargs: Passed to the underlying loader.

    Returns:
        LookupTableCell ready for use.
    """
    p = Path(path)
    if format == "auto":
        suffix = p.suffix.lower()
        if suffix == ".csv":
            format = "csv"
        elif suffix in {".h5", ".hdf5", ".nc", ".netcdf"}:
            format = "hdf5"
        else:
            msg = (
                f"Cannot auto-detect format for '{suffix}'. "
                "Use format='csv' or format='hdf5'."
            )
            raise ValueError(msg)

    if format == "csv":
        return LookupTableCell.from_csv(p, **kwargs)
    elif format == "hdf5":
        return LookupTableCell.from_hdf5(p)
    else:
        msg = f"Unknown format: {format}"
        raise ValueError(msg)


@dataclass
class ValidationReport:
    """Report from validating a lookup table.

    Args:
        is_passive: Whether all responses have |r| <= 1.
        max_magnitude: Maximum response magnitude.
        num_states: Number of discrete states.
        freq_range: (min_freq, max_freq) in Hz.
        theta_range: (min_theta, max_theta) in rad.
        warnings: List of warning messages.
    """

    is_passive: bool = True
    max_magnitude: float = 0.0
    num_states: int = 0
    freq_range: tuple[float, float] = (0.0, 0.0)
    theta_range: tuple[float, float] = (0.0, 0.0)
    warnings: list[str] = field(default_factory=list)


def validate_lookup_table(
    cell: LookupTableCell,
) -> ValidationReport:
    """Validate a lookup table for physical consistency.

    Checks:
    - Passivity: |response| <= 1 for all entries
    - Frequency and angle coverage
    - Number of states

    Args:
        cell: LookupTableCell to validate.

    Returns:
        ValidationReport with findings.
    """
    data = cell.table.values
    magnitudes = np.abs(data)
    max_mag = float(np.max(magnitudes))
    is_passive = bool(max_mag <= 1.0 + 1e-6)

    freq_coords = cell.table.coords["freq"].values
    theta_coords = cell.table.coords["theta"].values
    state_coords = cell.table.coords["state"].values

    warnings: list[str] = []
    if not is_passive:
        warnings.append(f"Non-passive: max |response| = {max_mag:.4f} > 1.0")
    if len(freq_coords) < 2:
        warnings.append("Only 1 frequency point — no interpolation possible")
    if len(theta_coords) < 2:
        warnings.append("Only 1 angle point — no angle interpolation possible")

    return ValidationReport(
        is_passive=is_passive,
        max_magnitude=max_mag,
        num_states=len(state_coords),
        freq_range=(float(freq_coords[0]), float(freq_coords[-1])),
        theta_range=(float(theta_coords[0]), float(theta_coords[-1])),
        warnings=warnings,
    )
