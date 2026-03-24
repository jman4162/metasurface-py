# metasurface-py

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
