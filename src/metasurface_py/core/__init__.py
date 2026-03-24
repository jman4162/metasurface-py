"""Core physics conventions, data types, and shared utilities."""

from metasurface_py.core.conventions import (
    EPS_0,
    ETA_0,
    MU_0,
    SPEED_OF_LIGHT,
    CoordinateSystem,
    NormalizationMode,
    PolarizationBasis,
    k0,
    wavelength,
)
from metasurface_py.core.types import AngleGrid, FrequencyGrid, Position3D

__all__ = [
    "EPS_0",
    "ETA_0",
    "MU_0",
    "SPEED_OF_LIGHT",
    "AngleGrid",
    "CoordinateSystem",
    "FrequencyGrid",
    "NormalizationMode",
    "PolarizationBasis",
    "Position3D",
    "k0",
    "wavelength",
]
