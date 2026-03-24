"""The central Metasurface object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from metasurface_py.surfaces.state import SurfaceState

if TYPE_CHECKING:
    import xarray as xr

    from metasurface_py.core.types import AngleGrid
    from metasurface_py.elements.protocols import UnitCellModel
    from metasurface_py.geometry.lattice import SupportsLattice


@dataclass(frozen=True)
class Metasurface:
    """A finite programmable metasurface aperture.

    Combines a lattice (element positions) with a unit-cell model (element response).

    Args:
        lattice: Element placement geometry.
        cell: Unit-cell response model.
        mode: Operating mode ("reflect" or "transmit").
    """

    lattice: SupportsLattice
    cell: UnitCellModel
    mode: Literal["reflect", "transmit"] = "reflect"

    @property
    def num_elements(self) -> int:
        """Number of active elements."""
        return self.lattice.num_elements

    @property
    def positions(self) -> npt.NDArray[np.floating[Any]]:
        """Element positions, shape (N, 3)."""
        return self.lattice.positions

    def set_state(self, phase: npt.NDArray[np.floating[Any]]) -> SurfaceState:
        """Create a SurfaceState from phase values.

        Args:
            phase: Phase values in radians. Shape (N,) or (nx, ny).
                   Flattened if 2D.

        Returns:
            A SurfaceState object.
        """
        phase_flat = np.asarray(phase, dtype=np.float64).ravel()
        if phase_flat.shape[0] != self.num_elements:
            raise ValueError(
                f"Expected {self.num_elements} phase values, got {phase_flat.shape[0]}"
            )
        mask = None
        if self.lattice.mask is not None:
            mask = self.lattice.mask.ravel()
            # Only keep phases for active elements
            if phase_flat.shape[0] == mask.shape[0]:
                phase_flat = phase_flat[mask]
        return SurfaceState(
            values=phase_flat,
            space=self.cell.state_space,
            mask=mask,
        )

    def response(
        self,
        state: SurfaceState,
        freq: float,
        theta_inc: float = 0.0,
        phi_inc: float = 0.0,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]:
        """Compute per-element complex response.

        Args:
            state: Surface state.
            freq: Frequency [Hz].
            theta_inc: Incident polar angle [rad].
            phi_inc: Incident azimuthal angle [rad].

        Returns:
            Complex element responses, shape (N,).
        """
        return self.cell.response(state.values, freq, theta_inc, phi_inc)

    def far_field(
        self,
        state: SurfaceState,
        freq: float,
        angles: AngleGrid,
        theta_inc: float = 0.0,
        phi_inc: float = 0.0,
    ) -> xr.DataArray:
        """Compute far-field pattern (convenience wrapper).

        Delegates to em.array_factor.far_field_pattern.
        """
        from metasurface_py.em.array_factor import far_field_pattern

        return far_field_pattern(
            self,
            state,
            freq=freq,
            angles=angles,
            theta_inc=theta_inc,
            phi_inc=phi_inc,
        )
