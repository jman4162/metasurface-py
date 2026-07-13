"""Continuous circular aperture sampling grids.

Used by surface-wave-fed (modulated metasurface) antennas, where the
radiating quantity is a continuous aperture field rather than a discrete
set of unit cells. Cell-centered polar sampling with exact ring areas
keeps quadrature error second order in the grid spacing.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class CircularAperture:
    """Cell-centered polar sampling grid over a circular aperture.

    Args:
        radius: Aperture radius [m].
        n_rho: Number of radial samples.
        n_phi: Number of azimuthal samples.
    """

    radius: float
    n_rho: int = 256
    n_phi: int = 128

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        if self.n_rho < 2 or self.n_phi < 4:
            raise ValueError("need n_rho >= 2 and n_phi >= 4")

    @cached_property
    def rho(self) -> npt.NDArray[np.floating[Any]]:
        """Radial sample points (cell centers), shape (n_rho,) [m]."""
        d_rho = self.radius / self.n_rho
        return (np.arange(self.n_rho, dtype=np.float64) + 0.5) * d_rho

    @cached_property
    def phi(self) -> npt.NDArray[np.floating[Any]]:
        """Azimuthal sample points (cell centers), shape (n_phi,) [rad]."""
        d_phi = 2.0 * np.pi / self.n_phi
        return (np.arange(self.n_phi, dtype=np.float64) + 0.5) * d_phi

    @property
    def area(self) -> float:
        """Total aperture area [m^2]."""
        return float(np.pi * self.radius**2)

    @cached_property
    def cell_areas(self) -> npt.NDArray[np.floating[Any]]:
        """Exact annular-sector cell areas, shape (n_rho, n_phi) [m^2].

        Uses 0.5*(rho_out^2 - rho_in^2)*d_phi so the cell areas sum
        exactly to pi*radius^2.
        """
        d_rho = self.radius / self.n_rho
        d_phi = 2.0 * np.pi / self.n_phi
        edges = np.arange(self.n_rho + 1, dtype=np.float64) * d_rho
        ring = 0.5 * (edges[1:] ** 2 - edges[:-1] ** 2) * d_phi
        return np.broadcast_to(ring[:, np.newaxis], (self.n_rho, self.n_phi)).copy()

    @cached_property
    def positions(self) -> npt.NDArray[np.floating[Any]]:
        """Cartesian sample positions in the z=0 plane, shape (n_rho*n_phi, 3) [m].

        Flattened with rho as the slow axis (matches ``cell_areas.ravel()``).
        """
        rho_g, phi_g = np.meshgrid(self.rho, self.phi, indexing="ij")
        x = (rho_g * np.cos(phi_g)).ravel()
        y = (rho_g * np.sin(phi_g)).ravel()
        z = np.zeros_like(x)
        return np.stack([x, y, z], axis=-1)
