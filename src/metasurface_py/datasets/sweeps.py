"""Parameter sweep execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr

from metasurface_py.em import far_field_pattern, peak_gain_db, steering_phase
from metasurface_py.experiments.config import ExperimentConfig, build_from_config
from metasurface_py.optimize import relax_then_quantize


@dataclass
class ParameterSweep:
    """Definition of a parameter sweep.

    Args:
        name: Descriptive name for the sweep.
        parameter: Config field to sweep (e.g., "freq", "num_bits").
        values: Array of parameter values to sweep over.
        base_config: Base experiment configuration.
    """

    name: str
    parameter: str
    values: npt.NDArray[np.floating[Any]]
    base_config: ExperimentConfig


def run_sweep(sweep: ParameterSweep) -> xr.Dataset:
    """Execute a parameter sweep.

    For each value of the swept parameter, builds and runs the
    experiment, collecting objective values and peak gains.

    Args:
        sweep: Parameter sweep definition.

    Returns:
        xr.Dataset with the sweep parameter as a coordinate,
        containing objective_value and runtime variables.
    """
    obj_values: list[float] = []
    runtimes: list[float] = []

    for val in sweep.values:
        config = sweep.base_config.model_copy(
            update={sweep.parameter: float(val)},
        )
        surface, objective, angles = build_from_config(config)
        result = relax_then_quantize(
            objective,
            surface,
            config.freq,
            angles,
            continuous_method=config.optimizer_method,  # type: ignore[arg-type]
            refine=config.refine,
            maxiter=config.maxiter,
            seed=config.seed,
        )
        obj_values.append(result.objective_value)
        runtimes.append(result.runtime_seconds)

    coords = {sweep.parameter: sweep.values}
    return xr.Dataset(
        {
            "objective_value": (
                [sweep.parameter],
                np.array(obj_values),
            ),
            "runtime_seconds": (
                [sweep.parameter],
                np.array(runtimes),
            ),
        },
        coords=coords,
        attrs={
            "sweep_name": sweep.name,
            "parameter": sweep.parameter,
        },
    )


def frequency_sweep(
    config: ExperimentConfig,
    freqs: npt.NDArray[np.floating[Any]],
) -> xr.Dataset:
    """Convenience: sweep over frequencies.

    For each frequency, computes far-field pattern and peak gain
    without optimization (uses analytical steering phase).

    Args:
        config: Base experiment config.
        freqs: Array of frequencies [Hz].

    Returns:
        xr.Dataset with freq coordinate and peak_gain_dbi variable.
    """
    gains: list[float] = []
    for f in freqs:
        cfg = config.model_copy(update={"freq": float(f)})
        surface, _objective, angles = build_from_config(cfg)
        phase = steering_phase(
            surface.lattice,
            theta_steer=np.radians(cfg.target_theta_deg),
            phi_steer=np.radians(cfg.target_phi_deg),
            freq=float(f),
        )
        state = surface.set_state(phase)
        if cfg.num_bits is not None:
            state = state.quantize()
        pattern = far_field_pattern(
            surface,
            state,
            freq=float(f),
            angles=angles,
        )
        gains.append(peak_gain_db(pattern))

    return xr.Dataset(
        {"peak_gain_dbi": ("freq", np.array(gains))},
        coords={"freq": freqs},
        attrs={"sweep_type": "frequency"},
    )
