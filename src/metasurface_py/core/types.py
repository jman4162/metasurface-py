"""Lightweight data types shared across the package."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import numpy.typing as npt

# Type aliases for clarity
ComplexArray = npt.NDArray[np.complexfloating[Any, Any]]
RealArray = npt.NDArray[np.floating[Any]]


@dataclass(frozen=True)
class FrequencyGrid:
    """A set of frequency points with derived quantities.

    Args:
        values: Frequency values in Hz.
    """

    values: npt.NDArray[np.floating[Any]]

    @classmethod
    def from_ghz(
        cls,
        values: npt.NDArray[np.floating[Any]] | list[float],
    ) -> Self:
        """Create from values in GHz."""
        arr = np.asarray(values, dtype=np.float64)
        return cls(values=arr * 1e9)

    @property
    def wavelengths(self) -> npt.NDArray[np.floating[Any]]:
        """Free-space wavelengths [m]."""
        from metasurface_py.core.conventions import SPEED_OF_LIGHT

        return SPEED_OF_LIGHT / self.values

    @property
    def k0(self) -> npt.NDArray[np.floating[Any]]:
        """Free-space wavenumbers [rad/m]."""
        from metasurface_py.core.conventions import SPEED_OF_LIGHT

        return 2.0 * np.pi * self.values / SPEED_OF_LIGHT

    @property
    def num_freqs(self) -> int:
        """Number of frequency points."""
        return int(self.values.shape[0])


@dataclass(frozen=True)
class AngleGrid:
    """A grid of angular observation directions.

    Angles are stored in radians (ISO spherical convention).

    Args:
        theta: Polar angles from +z axis [rad].
        phi: Azimuthal angles from +x axis [rad].
    """

    theta: npt.NDArray[np.floating[Any]]
    phi: npt.NDArray[np.floating[Any]]

    @classmethod
    def from_degrees(
        cls,
        theta: npt.NDArray[np.floating[Any]] | list[float],
        phi: npt.NDArray[np.floating[Any]] | list[float],
    ) -> Self:
        """Create from angles in degrees."""
        return cls(
            theta=np.deg2rad(np.asarray(theta, dtype=np.float64)),
            phi=np.deg2rad(np.asarray(phi, dtype=np.float64)),
        )

    @property
    def theta_deg(self) -> npt.NDArray[np.floating[Any]]:
        """Theta values in degrees."""
        return np.rad2deg(self.theta)  # type: ignore[no-any-return]

    @property
    def phi_deg(self) -> npt.NDArray[np.floating[Any]]:
        """Phi values in degrees."""
        return np.rad2deg(self.phi)  # type: ignore[no-any-return]


@dataclass(frozen=True)
class Position3D:
    """A point in 3D Cartesian space [meters]."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_array(self) -> npt.NDArray[np.floating[Any]]:
        """Return as (3,) numpy array."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def distance_to(self, other: Position3D) -> float:
        """Euclidean distance to another point."""
        return float(
            math.sqrt(
                (self.x - other.x) ** 2
                + (self.y - other.y) ** 2
                + (self.z - other.z) ** 2
            )
        )

    def direction_to(self, other: Position3D) -> npt.NDArray[np.floating[Any]]:
        """Unit vector pointing from self toward other."""
        diff = np.array(
            [other.x - self.x, other.y - self.y, other.z - self.z], dtype=np.float64
        )
        norm = np.linalg.norm(diff)
        if norm < 1e-15:
            raise ValueError("Cannot compute direction between coincident points")
        return diff / norm


@dataclass(frozen=True)
class SphericalPosition:
    """A point in spherical coordinates.

    Args:
        r: Radial distance [m].
        theta: Polar angle from +z [rad].
        phi: Azimuthal angle from +x [rad].
    """

    r: float = 1.0
    theta: float = 0.0
    phi: float = 0.0

    def to_cartesian(self) -> Position3D:
        """Convert to Cartesian coordinates."""
        x = self.r * math.sin(self.theta) * math.cos(self.phi)
        y = self.r * math.sin(self.theta) * math.sin(self.phi)
        z = self.r * math.cos(self.theta)
        return Position3D(x=x, y=y, z=z)

    @classmethod
    def from_degrees(cls, r: float, theta_deg: float, phi_deg: float) -> Self:
        """Create from angles in degrees."""
        return cls(
            r=r,
            theta=math.radians(theta_deg),
            phi=math.radians(phi_deg),
        )


# Standard xarray dimension names
DIM_FREQ: str = "freq"
DIM_THETA: str = "theta"
DIM_PHI: str = "phi"
DIM_X_ELEM: str = "x_elem"
DIM_Y_ELEM: str = "y_elem"
DIM_STATE: str = "state"
DIM_POL_TX: str = "pol_tx"
DIM_POL_RX: str = "pol_rx"
DIM_USER: str = "user"
DIM_TARGET: str = "target"
DIM_TRIAL: str = "trial"


@dataclass(frozen=True)
class SubstrateInfo:
    """Optional metadata describing the substrate (for documentation, not computation).

    In reduced-order models, substrate properties are baked into the unit-cell
    response data. This exists only for provenance tracking.
    """

    name: str = ""
    eps_r: float = 1.0
    loss_tangent: float = 0.0
    thickness_mm: float = 0.0
    notes: str = ""
    _metadata: dict[str, object] = field(default_factory=dict)
