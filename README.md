# metasurface-py

[![CI](https://github.com/jman4162/metasurface-py/actions/workflows/ci.yml/badge.svg)](https://github.com/jman4162/metasurface-py/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/metasurface-py.svg)](https://pypi.org/project/metasurface-py/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://jman4162.github.io/metasurface-py/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://metasurface-py.streamlit.app)

Open-source Python package for design, analysis, and optimization of electromagnetic metasurfaces for wireless communication and sensing: programmable reflecting surfaces (RIS/IRS), reflectarrays, and surface-wave-fed modulated metasurface (leaky-wave / holographic) antennas. Covers radiation patterns, RIS-assisted channel models, beamforming optimization under hardware constraints, and radar/localization metrics.

**Documentation:** <https://jman4162.github.io/metasurface-py/>

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
- **Modulated MTS antennas** — Surface-wave-fed leaky-wave apertures: TM surface-wave dispersion, Oliner–Hessel modulated-reactance solver, adiabatic Floquet-mode aperture fields, CP far fields (reproduces Faenzi et al., Sci. Rep. 2019)
- **Sensing** — Monostatic/bistatic RCS, detection SNR, Fisher information, CRLB for localization
- **Mutual coupling** — Canonical dipole coupling approximation
- **Publication-quality plotting** — 13+ plot functions, IEEE/Nature/poster presets, colorblind-safe palettes, PDF/PNG export
- **Experiment management** — TOML configs, parameter sweeps, reproducibility metadata
- **Interoperability** — CSV/HDF5/Touchstone import, scikit-rf adapter, xarray labeled outputs

## Tutorials

Interactive Jupyter notebooks that run in Google Colab (no install required):

| Notebook | Description | Colab |
|----------|-------------|-------|
| [01 Getting Started](notebooks/01_getting_started.ipynb) | Create a metasurface, steer a beam, compare quantization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jman4162/metasurface-py/blob/main/notebooks/01_getting_started.ipynb) |
| [02 Optimization](notebooks/02_optimization.ipynb) | Relax-then-quantize with hardware constraints | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jman4162/metasurface-py/blob/main/notebooks/02_optimization.ipynb) |
| [03 RIS Link](notebooks/03_ris_link.ipynb) | RIS-assisted communication link, N² scaling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jman4162/metasurface-py/blob/main/notebooks/03_ris_link.ipynb) |
| [04 Sensing & ISAC](notebooks/04_sensing_isac.ipynb) | Radar detection, localization CRLB, comms-sensing tradeoff | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jman4162/metasurface-py/blob/main/notebooks/04_sensing_isac.ipynb) |

## Examples

Runnable scripts in [`examples/`](examples/), including reproductions of published results:

| Script | Description |
|--------|-------------|
| [01 Beam steering](examples/01_beam_steering.py) | Steer a beam, compare 1/2/3-bit phase quantization |
| [02 Gain vs scan angle](examples/02_gain_vs_scan_angle.py) | Scan-loss sweep |
| [03 Near-field focusing](examples/03_near_field_focusing.py) | Spherical-wave focusing phase profile |
| [04 Multi-beam](examples/04_multi_beam.py) | Dual-beam synthesis |
| [05 Optimized steering](examples/05_optimize_steering.py) | Beam optimization under hardware constraints |
| [06 RIS link](examples/06_ris_link.py) | RIS-assisted communication link budget |
| [07 Wu & Zhang 2019](examples/07_reproduce_wu_zhang2019.py) | Reproduces "Intelligent Reflecting Surface Enhanced Wireless Network" (IEEE TWC 2019): N² SNR scaling, coverage extension |
| [08 Basar et al. 2019](examples/08_reproduce_basar2019.py) | Reproduces "Wireless Communications Through Reconfigurable Intelligent Surfaces" (IEEE Access 2019): scaling law, path loss, coverage |
| [09 Faenzi et al. 2019](examples/09_reproduce_faenzi2019.py) | Reproduces "Metasurface Antennas: New Models, Applications and Realizations" (Sci. Rep. 2019): leaky-wave dispersion, RHCP patterns, bandwidth ([model docs](https://jman4162.github.io/metasurface-py/modulated_mts_antennas/)) |

## Citation

If you use metasurface-py in your research, please cite:

```bibtex
@software{metasurface_py,
  author       = {Hodge, John},
  title        = {metasurface-py: Design, analysis, and optimization of programmable electromagnetic metasurfaces},
  year         = {2026},
  url          = {https://github.com/jman4162/metasurface-py},
  version      = {0.3.0},
  license      = {BSD-3-Clause},
}
```

## License

BSD-3-Clause
