# metasurface-py

[![CI](https://github.com/jman4162/metasurface-py/actions/workflows/ci.yml/badge.svg)](https://github.com/jman4162/metasurface-py/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

Open-source Python package for design, analysis, and optimization of programmable electromagnetic metasurfaces for wireless communication and sensing.

## Installation

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
import numpy as np
from metasurface_py.geometry import RectangularLattice
from metasurface_py.elements import PhaseOnlyCell, DiscretePhaseSpace
from metasurface_py.surfaces import Metasurface
from metasurface_py.em import steering_phase, far_field_pattern
from metasurface_py.core.types import AngleGrid
from metasurface_py.plotting import plot_pattern_2d

lattice = RectangularLattice(nx=32, ny=32, dx=5.4e-3, dy=5.4e-3)
cell = PhaseOnlyCell(state_space=DiscretePhaseSpace(num_bits=2))
surface = Metasurface(lattice=lattice, cell=cell, mode="reflect")

freq = 28e9
phase = steering_phase(lattice, theta_steer=np.radians(30), phi_steer=0.0, freq=freq)
state = surface.set_state(phase).quantize(cell.state_space.codebook)

angles = AngleGrid.from_degrees(theta=np.arange(-90, 91, dtype=float), phi=np.array([0.0, 90.0]))
pattern = far_field_pattern(surface, state, freq=freq, angles=angles)
plot_pattern_2d(pattern, cut_phi=0.0)
```

## Features

- **Metasurface modeling** — Rectangular/hexagonal lattices, phase-only and lookup-table unit cells, amplitude-phase coupled elements
- **Far-field analysis** — Array factor, directivity, sidelobe level, HPBW, beam steering, focusing, multi-beam synthesis
- **Optimization** — Continuous (L-BFGS-B, DE), discrete refinement, relax-then-quantize pipeline, multi-objective Pareto sweeps
- **Hardware constraints** — Phase quantization, grouped control lines, dead elements, manufacturing noise
- **RIS channel models** — Free-space path loss, narrowband SISO RIS-assisted links, optimal phase computation
- **Sensing** — Monostatic/bistatic RCS, detection SNR, Fisher information, CRLB for localization
- **Mutual coupling** — Canonical dipole coupling approximation
- **Publication-quality plotting** — 13+ plot functions, IEEE/Nature/poster presets, colorblind-safe palettes, PDF/PNG export
- **Experiment management** — TOML configs, parameter sweeps, reproducibility metadata
- **Interoperability** — CSV/HDF5/Touchstone import, scikit-rf adapter, xarray labeled outputs

## Citation

If you use metasurface-py in your research, please cite:

```bibtex
@software{metasurface_py,
  author       = {Hodge, John},
  title        = {metasurface-py: Design, analysis, and optimization of programmable electromagnetic metasurfaces},
  year         = {2026},
  url          = {https://github.com/jman4162/metasurface-py},
  version      = {0.1.0},
  license      = {BSD-3-Clause},
}
```

## License

BSD-3-Clause
