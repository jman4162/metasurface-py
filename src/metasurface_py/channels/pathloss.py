"""Free-space path loss models."""

from __future__ import annotations

import math

from metasurface_py.core.conventions import SPEED_OF_LIGHT


def free_space_path_loss(distance: float, freq: float) -> float:
    """Friis free-space path loss (linear scale).

    FSPL = (4 * pi * d * f / c)^2

    Args:
        distance: Distance [meters].
        freq: Frequency [Hz].

    Returns:
        Path loss as a linear ratio (>= 1).
    """
    if distance <= 0:
        raise ValueError(f"Distance must be positive, got {distance}")
    if freq <= 0:
        raise ValueError(f"Frequency must be positive, got {freq}")
    lam = SPEED_OF_LIGHT / freq
    return (4.0 * math.pi * distance / lam) ** 2


def free_space_path_loss_db(distance: float, freq: float) -> float:
    """Friis free-space path loss in dB.

    Args:
        distance: Distance [meters].
        freq: Frequency [Hz].

    Returns:
        Path loss in dB (positive value).
    """
    return 10.0 * math.log10(free_space_path_loss(distance, freq))
