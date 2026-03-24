"""Optimized beam steering with hardware constraints.

Demonstrates the relax-then-quantize optimization pipeline on a 12x12
metasurface with 2-bit quantization. Compares analytical steering,
quantized steering, and optimized-then-quantized patterns.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from metasurface_py.core.types import AngleGrid
from metasurface_py.elements import DiscretePhaseSpace, PhaseOnlyCell
from metasurface_py.em import far_field_pattern, steering_phase
from metasurface_py.geometry import RectangularLattice
from metasurface_py.optimize import MaxGainObjective, relax_then_quantize
from metasurface_py.plotting import (
    plot_convergence,
    plot_pattern_comparison,
    set_publication_style,
)
from metasurface_py.surfaces import Metasurface


def main() -> None:
    set_publication_style(target="ieee_double")

    freq = 10e9
    lam = 3e8 / freq
    n = 12
    lattice = RectangularLattice(nx=n, ny=n, dx=lam / 2, dy=lam / 2)
    cell = PhaseOnlyCell(state_space=DiscretePhaseSpace(num_bits=2))
    surface = Metasurface(lattice=lattice, cell=cell)

    target_theta = np.radians(30.0)
    angles = AngleGrid(
        theta=np.linspace(0.01, np.pi - 0.01, 360),
        phi=np.array([0.0]),
    )
    angles_full = AngleGrid(
        theta=np.linspace(0.01, np.pi - 0.01, 90),
        phi=np.linspace(0, 2 * np.pi - 0.01, 36),
    )

    # Analytical steering -> quantize
    phase_analytical = steering_phase(
        lattice,
        theta_steer=target_theta,
        phi_steer=0.0,
        freq=freq,
    )
    state_quantized = surface.set_state(phase_analytical).quantize()
    pattern_quantized = far_field_pattern(
        surface,
        state_quantized,
        freq=freq,
        angles=angles,
    )

    # Optimized -> quantize -> refine
    obj = MaxGainObjective(target_theta, 0.0, angles_full)
    result = relax_then_quantize(
        obj,
        surface,
        freq,
        angles_full,
        continuous_method="L-BFGS-B",
        refine=True,
        maxiter=100,
        seed=42,
    )
    pattern_optimized = far_field_pattern(
        surface,
        result.state,
        freq=freq,
        angles=angles,
    )

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    plot_pattern_comparison(
        [
            (pattern_quantized, "Analytical + Quantize"),
            (pattern_optimized, "Optimized + Quantize"),
        ],
        cut_phi=0.0,
        ax=ax1,
    )
    ax1.set_title("Pattern Comparison (2-bit)")
    ax1.set_xlim([0, 90])
    ax1.set_ylim([-30, 0])

    plot_convergence(
        {"Relax-then-quantize": result.convergence_history},
        ax=ax2,
        ylabel="Objective (neg. gain dBi)",
    )
    ax2.set_title("Convergence")

    fig.tight_layout()
    plt.show()

    print(f"Runtime: {result.runtime_seconds:.2f}s")
    print(f"Final objective: {result.objective_value:.2f}")


if __name__ == "__main__":
    main()
