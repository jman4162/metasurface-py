"""Single source of truth for electromagnetic conventions used throughout the package.

Time-harmonic convention: exp(-j*omega*t)  (IEEE antenna standard)
This means:
  - A positive phase shift exp(+j*phi) represents a phase advance / time delay.
  - Outgoing spherical waves go as exp(-j*k*r) / r.
  - The steering vector for direction k_hat is exp(+j*k*k_hat.dot(r_n)).

Coordinate system: ISO spherical (physics convention)
  - theta: polar angle from +z axis (0 = zenith, pi/2 = horizon)
  - phi: azimuthal angle from +x axis toward +y axis

Polarization basis: theta-hat, phi-hat (Ludwig-2 / standard spherical)

References:
  - IEEE Std 145-2013 (Definitions of Terms for Antennas)
  - Balanis, "Antenna Theory", 4th ed., Ch. 2
"""

from __future__ import annotations

import math
from enum import Enum

# --- Physical constants (SI) ---

SPEED_OF_LIGHT: float = 299_792_458.0
"""Speed of light in vacuum [m/s]."""

MU_0: float = 4.0e-7 * math.pi
"""Permeability of free space [H/m]."""

EPS_0: float = 1.0 / (MU_0 * SPEED_OF_LIGHT**2)
"""Permittivity of free space [F/m]."""

ETA_0: float = MU_0 * SPEED_OF_LIGHT
"""Intrinsic impedance of free space [ohms]. Approximately 376.73 ohms."""

# --- Time-harmonic convention ---

TIME_CONVENTION: str = "exp(-j*omega*t)"
"""IEEE antenna convention. Phase advance is positive."""

PHASOR_SIGN: int = -1
"""Sign in exp(PHASOR_SIGN * j * omega * t). Value is -1 for IEEE convention."""


# --- Enums ---


class CoordinateSystem(Enum):
    """Supported coordinate systems for angular quantities."""

    SPHERICAL_ISO = "spherical_iso"
    """ISO/physics spherical: theta from +z, phi from +x toward +y."""


class PolarizationBasis(Enum):
    """Supported polarization decompositions."""

    THETA_PHI = "theta_phi"
    """Standard spherical basis vectors (Ludwig-2)."""

    LUDWIG3 = "ludwig3"
    """Ludwig-3 co/cross-pol definition."""

    RHCP_LHCP = "rhcp_lhcp"
    """Right-hand / left-hand circular polarization."""


class NormalizationMode(Enum):
    """Far-field pattern normalization modes."""

    PEAK = "peak"
    """Normalize to peak value (0 dB at maximum)."""

    DIRECTIVITY = "directivity"
    """Absolute directivity [dBi]."""

    REALIZED_GAIN = "realized_gain"
    """Realized gain including mismatch and efficiency."""


# --- Convenience functions ---


def wavelength(freq: float) -> float:
    """Free-space wavelength [m] for a given frequency [Hz]."""
    if freq <= 0:
        raise ValueError(f"Frequency must be positive, got {freq}")
    return SPEED_OF_LIGHT / freq


def k0(freq: float) -> float:
    """Free-space wavenumber [rad/m] for a given frequency [Hz]."""
    if freq <= 0:
        raise ValueError(f"Frequency must be positive, got {freq}")
    return 2.0 * math.pi * freq / SPEED_OF_LIGHT


def freq_to_omega(freq: float) -> float:
    """Angular frequency [rad/s] for a given frequency [Hz]."""
    if freq <= 0:
        raise ValueError(f"Frequency must be positive, got {freq}")
    return 2.0 * math.pi * freq
