"""Experiment execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from metasurface_py.experiments.config import ExperimentConfig, build_from_config
from metasurface_py.experiments.reproducibility import capture_environment
from metasurface_py.optimize import OptimizationResult, relax_then_quantize


@dataclass(frozen=True)
class ExperimentResult:
    """Result of an experiment run.

    Args:
        config: The experiment configuration used.
        optimization: The optimization result.
        environment: Environment metadata for reproducibility.
    """

    config: ExperimentConfig
    optimization: OptimizationResult
    environment: dict[str, Any] = field(default_factory=dict)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Execute an experiment from a configuration.

    Builds the metasurface, objective, and angles from config,
    then runs relax_then_quantize optimization.

    Args:
        config: Experiment configuration.

    Returns:
        ExperimentResult with optimization result and provenance.
    """
    surface, objective, angles = build_from_config(config)
    env = capture_environment()

    opt_result = relax_then_quantize(
        objective,
        surface,
        config.freq,
        angles,
        continuous_method=config.optimizer_method,  # type: ignore[arg-type]
        refine=config.refine,
        maxiter=config.maxiter,
        seed=config.seed,
    )

    return ExperimentResult(
        config=config,
        optimization=opt_result,
        environment=env,
    )
