"""Lattice definitions for metasurface element placement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt

from metasurface_py.core.conventions import wavelength


@runtime_checkable
class SupportsLattice(Protocol):
    """Protocol for any object providing element positions."""

    @property
    def positions(self) -> npt.NDArray[np.floating[Any]]:
        """Element positions, shape (N, 3) in meters."""
        ...

    @property
    def num_elements(self) -> int:
        """Total number of active elements."""
        ...

    @property
    def mask(self) -> npt.NDArray[np.bool_] | None:
        """Boolean mask of active elements (True = active), or None if all active."""
        ...


@dataclass(frozen=True)
class RectangularLattice:
    """Rectangular grid of elements in the xy-plane.

    Args:
        nx: Number of elements along x.
        ny: Number of elements along y.
        dx: Element spacing along x [meters].
        dy: Element spacing along y [meters].
        origin: Center of the array, shape (3,). Defaults to (0, 0, 0).
        element_mask: Boolean mask (nx, ny), True = active. None = all active.
    """

    nx: int
    ny: int
    dx: float
    dy: float
    origin: npt.NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    element_mask: npt.NDArray[np.bool_] | None = None

    @classmethod
    def from_wavelength(
        cls,
        nx: int,
        ny: int,
        spacing_fraction: float,
        freq: float,
        origin: npt.NDArray[np.floating[Any]] | None = None,
    ) -> Self:
        """Create lattice with spacing as a fraction of wavelength.

        Args:
            nx: Number of elements along x.
            ny: Number of elements along y.
            spacing_fraction: Element spacing as fraction of wavelength (e.g., 0.5).
            freq: Reference frequency [Hz].
            origin: Center of array. Defaults to (0, 0, 0).
        """
        lam = wavelength(freq)
        d = spacing_fraction * lam
        return cls(
            nx=nx,
            ny=ny,
            dx=d,
            dy=d,
            origin=origin if origin is not None else np.zeros(3, dtype=np.float64),
        )

    @property
    def positions(self) -> npt.NDArray[np.floating[Any]]:
        """Element positions, shape (N, 3) in meters, centered on origin."""
        ix = np.arange(self.nx, dtype=np.float64)
        iy = np.arange(self.ny, dtype=np.float64)
        # Center the array
        ix = (ix - (self.nx - 1) / 2.0) * self.dx
        iy = (iy - (self.ny - 1) / 2.0) * self.dy
        gx, gy = np.meshgrid(ix, iy, indexing="ij")
        pos = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(self.nx * self.ny)])
        pos = pos + self.origin
        if self.element_mask is not None:
            pos = pos[self.element_mask.ravel()]
        return pos

    @property
    def num_elements(self) -> int:
        """Number of active elements."""
        if self.element_mask is not None:
            return int(np.sum(self.element_mask))
        return self.nx * self.ny

    @property
    def mask(self) -> npt.NDArray[np.bool_] | None:
        """Boolean mask of active elements."""
        return self.element_mask

    @property
    def extent(self) -> tuple[float, float]:
        """Physical extent (width_x, width_y) of the array [meters]."""
        return ((self.nx - 1) * self.dx, (self.ny - 1) * self.dy)

    @property
    def area(self) -> float:
        """Physical area of the array [m^2]."""
        return float(self.nx * self.dx * self.ny * self.dy)


@dataclass(frozen=True)
class HexagonalLattice:
    """Hexagonal (triangular) grid of elements in the xy-plane.

    Rows are offset by half the x-spacing on alternating y-indices,
    creating a close-packed hexagonal arrangement.

    Args:
        nx: Number of elements per row.
        ny: Number of rows.
        dx: Element spacing along x [meters].
        origin: Center of the array, shape (3,). Defaults to (0, 0, 0).
        element_mask: Boolean mask (nx, ny), True = active. None = all active.
    """

    nx: int
    ny: int
    dx: float
    origin: npt.NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    element_mask: npt.NDArray[np.bool_] | None = None

    @property
    def dy(self) -> float:
        """Row spacing for hexagonal packing: dx * sqrt(3)/2."""
        return self.dx * np.sqrt(3.0) / 2.0  # type: ignore[no-any-return]

    @property
    def positions(self) -> npt.NDArray[np.floating[Any]]:
        """Element positions, shape (N, 3) in meters, centered on origin."""
        dy = self.dy
        points: list[list[float]] = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                x_offset = 0.5 * self.dx if (iy % 2 == 1) else 0.0
                x = ix * self.dx + x_offset
                y = iy * dy
                points.append([x, y, 0.0])
        pos = np.array(points, dtype=np.float64)
        # Center on origin
        centroid = pos.mean(axis=0)
        pos = pos - centroid + self.origin
        if self.element_mask is not None:
            pos = pos[self.element_mask.ravel()]
        return pos  # type: ignore[no-any-return]

    @property
    def num_elements(self) -> int:
        """Number of active elements."""
        if self.element_mask is not None:
            return int(np.sum(self.element_mask))
        return self.nx * self.ny

    @property
    def mask(self) -> npt.NDArray[np.bool_] | None:
        """Boolean mask of active elements."""
        return self.element_mask
