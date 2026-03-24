"""Optimization module for metasurface phase configuration."""

from metasurface_py.optimize.continuous import optimize_continuous
from metasurface_py.optimize.discrete import refine_discrete
from metasurface_py.optimize.multiobjective import ParetoResult, pareto_sweep
from metasurface_py.optimize.objectives import (
    MaxGainObjective,
    MinSidelobeObjective,
    WeightedGainSidelobeObjective,
)
from metasurface_py.optimize.relax_quantize import relax_then_quantize
from metasurface_py.optimize.result import OptimizationResult

__all__ = [
    "MaxGainObjective",
    "MinSidelobeObjective",
    "OptimizationResult",
    "ParetoResult",
    "WeightedGainSidelobeObjective",
    "optimize_continuous",
    "pareto_sweep",
    "refine_discrete",
    "relax_then_quantize",
]
