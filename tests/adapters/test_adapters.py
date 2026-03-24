"""Tests for adapters module."""

from __future__ import annotations

import numpy as np
import xarray as xr

from metasurface_py.adapters.lookup import validate_lookup_table
from metasurface_py.adapters.validation import compare_models
from metasurface_py.core.types import AngleGrid
from metasurface_py.elements import PhaseOnlyCell
from metasurface_py.elements.lookup_cell import LookupTableCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.geometry import RectangularLattice
from metasurface_py.surfaces import Metasurface


class TestValidateLookupTable:
    def _make_passive_cell(self) -> LookupTableCell:
        states = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        freqs = np.array([9e9, 10e9, 11e9])
        thetas = np.array([0.0, 0.5, 1.0])
        data = 0.9 * np.exp(
            1j * states[:, None, None] * np.ones((1, len(freqs), len(thetas)))
        )
        table = xr.DataArray(
            data,
            dims=["state", "freq", "theta"],
            coords={
                "state": states,
                "freq": freqs,
                "theta": thetas,
            },
        )
        return LookupTableCell.from_xarray(table)

    def _make_non_passive_cell(self) -> LookupTableCell:
        states = np.array([0.0, np.pi])
        freqs = np.array([10e9])
        thetas = np.array([0.0])
        data = np.array([[[1.5 + 0j]], [[1.2 + 0j]]])
        table = xr.DataArray(
            data,
            dims=["state", "freq", "theta"],
            coords={
                "state": states,
                "freq": freqs,
                "theta": thetas,
            },
        )
        return LookupTableCell.from_xarray(table)

    def test_passive_table_passes(self) -> None:
        cell = self._make_passive_cell()
        report = validate_lookup_table(cell)
        assert report.is_passive
        assert len(report.warnings) == 0

    def test_non_passive_flagged(self) -> None:
        cell = self._make_non_passive_cell()
        report = validate_lookup_table(cell)
        assert not report.is_passive
        assert any("Non-passive" in w for w in report.warnings)

    def test_report_metadata(self) -> None:
        cell = self._make_passive_cell()
        report = validate_lookup_table(cell)
        assert report.num_states == 4
        assert report.freq_range[0] == 9e9
        assert report.freq_range[1] == 11e9


class TestCompareModels:
    def test_identical_models_zero_error(self) -> None:
        freq = 10e9
        lam = 3e8 / freq
        lattice = RectangularLattice(
            nx=4,
            ny=4,
            dx=lam / 2,
            dy=lam / 2,
        )
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
        surface = Metasurface(lattice=lattice, cell=cell)
        state = surface.set_state(np.zeros(16))
        angles = AngleGrid(
            theta=np.linspace(0.1, 1.5, 20),
            phi=np.array([0.0]),
        )
        ds = compare_models(
            surface,
            state,
            surface,
            state,
            freq,
            angles,
        )
        assert "magnitude_error_db" in ds
        assert "phase_error_deg" in ds
        # Same model should give zero error
        mag_err = ds.attrs["rms_magnitude_error_db"]
        assert mag_err < 0.01
