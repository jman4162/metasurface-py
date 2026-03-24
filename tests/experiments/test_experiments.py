"""Tests for experiments module."""

from __future__ import annotations

from metasurface_py.experiments import (
    ExperimentConfig,
    build_from_config,
    capture_environment,
    run_experiment,
    set_global_seed,
)
from metasurface_py.surfaces import Metasurface


class TestExperimentConfig:
    def test_default_config(self) -> None:
        config = ExperimentConfig()
        assert config.nx == 16
        assert config.freq == 28e9
        assert config.seed == 42

    def test_custom_config(self) -> None:
        config = ExperimentConfig(
            name="test",
            nx=8,
            ny=8,
            num_bits=2,
            freq=10e9,
        )
        assert config.num_bits == 2
        assert config.freq == 10e9

    def test_build_from_config_continuous(self) -> None:
        config = ExperimentConfig(nx=4, ny=4, num_bits=None)
        surface, _objective, _angles = build_from_config(config)
        assert isinstance(surface, Metasurface)
        assert surface.num_elements == 16

    def test_build_from_config_discrete(self) -> None:
        config = ExperimentConfig(nx=4, ny=4, num_bits=2)
        surface, _objective, _angles = build_from_config(config)
        assert surface.cell.num_states == 4


class TestReproducibility:
    def test_capture_environment(self) -> None:
        env = capture_environment()
        assert "metasurface_py_version" in env
        assert "python_version" in env
        assert "platform" in env
        assert "timestamp" in env
        assert "git_commit" in env

    def test_set_global_seed(self) -> None:
        import numpy as np

        set_global_seed(42)
        a = np.random.random()
        set_global_seed(42)
        b = np.random.random()
        assert a == b


class TestRunExperiment:
    def test_run_small_experiment(self) -> None:
        config = ExperimentConfig(
            nx=6,
            ny=6,
            num_bits=2,
            freq=10e9,
            maxiter=10,
            theta_points=45,
            phi_points=18,
            seed=0,
        )
        result = run_experiment(config)
        assert result.config == config
        assert result.optimization.state is not None
        assert result.optimization.runtime_seconds > 0
        assert "metasurface_py_version" in result.environment
