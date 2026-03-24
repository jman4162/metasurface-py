"""Electromagnetic modeling: array factor, far-field, steering."""

from metasurface_py.em.array_factor import (
    array_factor,
    directivity,
    far_field_pattern,
    half_power_beamwidth,
    peak_gain_db,
    sidelobe_level,
)
from metasurface_py.em.coupling import apply_coupling, mutual_impedance_approx
from metasurface_py.em.steering import focusing_phase, multi_beam_phase, steering_phase

__all__ = [
    "apply_coupling",
    "array_factor",
    "directivity",
    "far_field_pattern",
    "focusing_phase",
    "half_power_beamwidth",
    "multi_beam_phase",
    "mutual_impedance_approx",
    "peak_gain_db",
    "sidelobe_level",
    "steering_phase",
]
