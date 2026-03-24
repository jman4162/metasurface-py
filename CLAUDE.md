# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`metasurface-py` is an open-source Python package for design, analysis, and optimization of programmable electromagnetic metasurfaces (RIS/IRS, reflectarrays, transmitarrays) for wireless communication and sensing. See SPEC.md for the full specification.

**Status:** v0.1.0 release candidate. All phases (0-4) complete: core, geometry, elements, surfaces, EM, plotting, optimization, channels, experiments, datasets, adapters, IO, sensing. 58 source files, 170+ tests, 6 examples, 3 benchmarks. CI via GitHub Actions.

## Build and Development Commands

- Python 3.11+
- `src/` layout with `pyproject.toml` (hatchling backend)
- Install: `pip install -e ".[dev]"` for development
- Linting: `ruff check src/ tests/` — must pass with zero violations
- Formatting: `ruff format src/ tests/` — all code must be ruff-formatted
- Type checking: `mypy src/` — must pass in strict mode
- Testing: `pytest tests/` — run full test suite
- Single test: `pytest tests/core/test_math_utils.py::test_db_roundtrip -v`
- Optional extras: `[jax]`, `[rf]`, `[openems]`, `[dev]`

## Code Quality Requirements

All code must pass **ruff** and **mypy** before being considered complete:
- **ruff**: All linting rules and formatting must pass. Run `ruff check --fix` to auto-fix, then `ruff format` to format.
- **mypy**: Strict mode (`mypy --strict src/`). All functions must have type annotations. Use `numpy.typing.NDArray` for array types. Protocols must use `typing.Protocol` with `runtime_checkable` where useful.

## Architecture

Three-layer architecture:

1. **Core physics/data layer** — coordinates, frequency grids, polarization, aperture indexing, units, material/state containers. Fewest dependencies, strongest API stability.
2. **Model/solver layer** — unit-cell response interpolation, aperture field synthesis, mutual coupling, scattering/channel models, external solver adapters.
3. **Optimization/application layer** — objectives, constraints, differentiable pipelines, mixed discrete/continuous optimization, experiment management.

### Package Structure (from SPEC.md §7.1)

```
src/metasurface_py/
├── core/           # canonical data models, shared math, conventions
├── geometry/       # lattice, aperture, coordinates, panels
├── materials/      # substrate, conductor, lumped/tunable states
├── elements/       # unit-cell response models and codebooks
├── surfaces/       # finite metasurface objects
├── em/             # reduced-order EM, fields, scattering, coupling
├── channels/       # comms propagation and RIS-assisted links
├── sensing/        # localization, radar, inverse scattering metrics
├── optimize/       # gradient-based, derivative-free, mixed-integer
├── datasets/       # xarray-based sweeps and serialization
├── adapters/       # Meep/openEMS/scikit-rf/measured-data interfaces
├── experiments/    # reproducible run configs and orchestration
├── plotting/       # field, pattern, geometry, convergence plots
├── io/             # import/export, config, result archives
├── cli/            # command line entry points
└── examples/       # research workflows and tutorials
```

### Key Abstractions

- **Lattice** — periodic element placement (rectangular, hexagonal)
- **UnitCellModel** — maps control state + incident conditions to response; subtypes include PhaseOnlyCell, AmplitudePhaseCell, TensorCell, LookupTableCell, EquivalentImpedanceCell, MeasuredCell
- **Metasurface** — finite aperture = lattice + cell model + per-element state
- **SurfaceState** — concrete realization of programmable controls (binary, multi-bit, continuous)
- **Scene/PropagationScenario** — sources, receivers, carriers, paths, environment
- **Objective** — scalar metric to optimize (gain, SINR, coverage, localization, etc.)
- **Experiment** — serializable bundle: geometry + materials + cell model + solver + objectives + optimizer + seeds + outputs

## Design Principles

- **Reduced-order first, full-wave compatible** — implement reduced-order EM models directly; interface with external solvers (Meep, openEMS, HFSS) rather than reimplementing them.
- **Labeled scientific data** — use xarray with explicit dimensions (freq, theta, phi, pol, element index, state, user, target, trial). Never anonymous tensors.
- **Optimization is first-class** — central to the API, not a wrapper.
- **Hardware constraints are explicit** — phase quantization, discrete codebooks, grouped control, dead elements, lossy states, limited tuning ranges, bias topology.
- **Composable models over inheritance** — prefer functional and protocol-based composition.
- **Immutable configs, rich result objects** — every solver/optimizer call returns metadata, runtime, config, and diagnostics.

## Required Dependencies

NumPy, SciPy, xarray, Matplotlib, pydantic (or attrs/dataclasses), h5py

## Optional Dependencies

JAX (differentiable/JIT), scikit-rf (RF interop), PyTorch (surrogates), plotly (interactive viz), zarr/dask (large sweeps), meep/openEMS (full-wave validation)

## Modeling Fidelity Levels

- **Level 0:** Array-factor / phase-sheet (fast, ignores coupling)
- **Level 1:** Element-response with angle/frequency dependence (default v1 mode)
- **Level 2:** Finite-aperture reduced-order interaction (mutual coupling, embedded response)
- **Level 3:** External full-wave validation adapters (spot checks, calibration)

## Visualization Requirements

All charts and data visualizations must be **academic publication quality**. This is a core requirement, not a nice-to-have. Supported visualization libraries:
- **Matplotlib** (required) — primary library for static 2D/3D plots: radiation patterns, state maps, convergence curves. Use `plotting/style.py` publication presets (font sizes, linewidths, colorblind-safe palettes).
- **Seaborn** (optional) — statistical visualizations, heatmaps, and multi-panel figures where Matplotlib alone is verbose.
- **Plotly** (optional) — interactive 3D radiation patterns, parameter sweep explorations, and HTML-exportable figures for supplementary materials.
- **Three.js** (optional) — browser-based 3D visualization of metasurface geometries, beam patterns, and field distributions for web demos and interactive documentation.

Publication quality means: proper axis labels with units, consistent font sizing, vector-format export (PDF/SVG), colorblind-safe palettes, no chartjunk, and figures that can be inserted directly into IEEE/Nature/APS journal submissions without modification.

## Conventions

Phasor sign convention, time-harmonic convention, coordinate system, polarization basis, field/power normalization, far-field normalization, and dB vs linear storage must all be defined in a single authoritative `core/conventions` module.

## Planning Session Guidelines

Every planning session must address three questions:

1. **What do you recommend and why (or why not)?** — State the recommended approach explicitly, with reasoning. If rejecting an alternative, explain why it's wrong for this project, not just that another option exists.

2. **Right-size the solution** — Do not over-engineer (no plugin frameworks, deep abstractions, or generalized machinery for single-use cases) and do not under-engineer (no shortcuts that leave researchers hitting walls when they try real problems with real constraints). The bar: would a PhD student with EM/wireless background be able to use this for a paper within a day of installing it?

3. **What are the limiting factors for adoption?** — The target users are graduate-level researchers and engineers. Identify and address the specific barriers they face:
   - **Install friction** — heavy or conflicting dependencies kill adoption in academic environments with managed clusters and mixed OS setups.
   - **Time-to-first-result** — researchers won't invest in a tool that requires hours of setup or reading before producing a useful output.
   - **Trust gap** — researchers won't publish with a tool unless they can validate its outputs against known results or their own full-wave data.
   - **Notation/convention mismatch** — EM and wireless communities use different conventions; ambiguity here causes silent errors and abandoned tools.
   - **Constraint realism** — idealized continuous-phase optimizations are publishable only in narrow contexts; the tool must handle real hardware constraints to be taken seriously.
   - **Interoperability** — researchers already have data from HFSS/CST/Meep/measurements; if they can't import it easily, the tool is dead on arrival.
