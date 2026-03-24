"""Experiment configuration and serialization."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from metasurface_py.core.types import AngleGrid
from metasurface_py.elements import DiscretePhaseSpace, PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.geometry import RectangularLattice
from metasurface_py.optimize.objectives import (
    MaxGainObjective,
    MinSidelobeObjective,
    WeightedGainSidelobeObjective,
)
from metasurface_py.surfaces import Metasurface


class ExperimentConfig(BaseModel):
    """Serializable experiment configuration.

    Describes a complete experiment: geometry, cell model, optimization
    objective, and solver settings. Can be saved/loaded as TOML.
    """

    name: str = "experiment"
    # Lattice
    nx: int = 16
    ny: int = 16
    spacing_fraction: float = 0.5
    freq: float = 28e9
    mode: Literal["reflect", "transmit"] = "reflect"
    # Cell
    cell_type: str = "phase_only"
    num_bits: int | None = None
    amplitude: float = 1.0
    # Objective
    objective_type: str = "max_gain"
    target_theta_deg: float = 30.0
    target_phi_deg: float = 0.0
    alpha: float = 0.7
    # Optimizer
    optimizer_method: str = "L-BFGS-B"
    maxiter: int = 200
    refine: bool = True
    # Angles
    theta_points: int = 90
    phi_points: int = 36
    # Reproducibility
    seed: int = 42
    metadata: dict[str, Any] = {}


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Save experiment config to TOML file.

    Args:
        config: Configuration to save.
        path: Output file path (.toml).
    """
    import tomli_w

    data = config.model_dump()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        tomli_w.dump(data, f)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load experiment config from TOML file.

    Args:
        path: Path to TOML file.

    Returns:
        Loaded ExperimentConfig.
    """
    with Path(path).open("rb") as f:
        data = tomllib.load(f)
    return ExperimentConfig(**data)


def build_from_config(
    config: ExperimentConfig,
) -> tuple[Metasurface, Any, AngleGrid]:
    """Instantiate objects from an experiment config.

    Args:
        config: Experiment configuration.

    Returns:
        Tuple of (Metasurface, Objective, AngleGrid).
    """
    lattice = RectangularLattice.from_wavelength(
        nx=config.nx,
        ny=config.ny,
        spacing_fraction=config.spacing_fraction,
        freq=config.freq,
    )

    if config.num_bits is not None:
        space = DiscretePhaseSpace(num_bits=config.num_bits)
    else:
        space = ContinuousPhaseSpace()

    cell = PhaseOnlyCell(state_space=space, amplitude=config.amplitude)
    surface = Metasurface(
        lattice=lattice,
        cell=cell,
        mode=config.mode,
    )

    angles = AngleGrid(
        theta=np.linspace(0.01, np.pi - 0.01, config.theta_points),
        phi=np.linspace(0, 2 * np.pi - 0.01, config.phi_points),
    )

    target_theta = np.radians(config.target_theta_deg)
    target_phi = np.radians(config.target_phi_deg)

    objective: Any
    if config.objective_type == "max_gain":
        objective = MaxGainObjective(target_theta, target_phi, angles)
    elif config.objective_type == "min_sidelobe":
        objective = MinSidelobeObjective(
            target_theta,
            target_phi,
            angles,
        )
    elif config.objective_type == "weighted":
        objective = WeightedGainSidelobeObjective(
            target_theta,
            target_phi,
            angles,
            alpha=config.alpha,
        )
    else:
        msg = f"Unknown objective type: {config.objective_type}"
        raise ValueError(msg)

    return surface, objective, angles
