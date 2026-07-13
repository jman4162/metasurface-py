# Modulated metasurface antennas (Faenzi et al. 2019)

`metasurface_py` includes a reduced-order model of surface-wave-fed
modulated metasurface (MTS) antennas, built to reproduce the fast-model
tier of:

> M. Faenzi, G. Minatti, D. González-Ovejero, F. Caminita, E. Martini,
> C. Della Giovampaola, and S. Maci, "Metasurface Antennas: New Models,
> Applications and Realizations," *Scientific Reports* 9:10178 (2019).
> [doi:10.1038/s41598-019-46522-z](https://doi.org/10.1038/s41598-019-46522-z)

Run the reproduction with:

```bash
python examples/09_reproduce_faenzi2019.py
```

## Model

A monopole at the center of a circular aperture launches a cylindrical
TM surface wave on a grounded dielectric slab whose printed cladding is
a sinusoidally modulated sheet reactance. The model chain is:

1. **Surface wave** — self-consistent TM dispersion of the transparent
   sheet + grounded slab
   (`em.leakywave.solve_sw_transparent`), reproducing the paper's quoted
   transparent/opaque reactance pairs (−1058 Ω ↔ 0.6ζ₀ at 26.25 GHz,
   −796 Ω ↔ 1.1ζ₀ at 32.05 GHz) to 0.2%.
2. **Leaky wave** — complex wavenumber β − jα of the modulated
   impenetrable reactance X(x) = X̄(1 + m·cos(2πx/d)) from an
   Oliner–Hessel truncated-Floquet transverse resonance solved by
   complex Newton iteration (`em.leakywave.dispersion_modulated_reactance`).
3. **Aperture field** — adiabatic Floquet-mode surface current
   J ∝ H₁⁽²⁾(∫k dρ) with the local complex wavenumber; the −1 harmonic
   of the modulation converts it into the radiating tangential field
   (`em.aperture_field`). Spiral tensor modulation (Φ = ∓φ with the
   ρφ entry in quadrature) yields circular polarization.
4. **Far field** — vector radiation integral over the ground plane,
   returning E_θ/E_φ; RHCP/LHCP split and axial ratio via
   `core.polarization`.

The user-facing object is
`surfaces.ModulatedMetasurfaceAntenna` with
`SinusoidalModulation` (scalar) or `TensorModulation` (CP) profiles:

```python
import numpy as np
from metasurface_py.core import AngleGrid, SubstrateInfo, circular_components
from metasurface_py.core.conventions import wavelength
from metasurface_py.em import pattern_directivity, solve_sw_transparent
from metasurface_py.surfaces import ModulatedMetasurfaceAntenna, TensorModulation

f0 = 26.25e9
sub = SubstrateInfo(name="RO3010", eps_r=10.2, thickness_mm=0.635)
sw = solve_sw_transparent(-1058.0, f0, sub.eps_r, sub.thickness_mm * 1e-3)
mod = TensorModulation.circular_broadside(sw.x_op, f0, m=0.3, hand="rhcp")
ant = ModulatedMetasurfaceAntenna(
    radius=10 * wavelength(f0), x_transparent=-1058.0,
    substrate=sub, modulation=mod,
)
angles = AngleGrid.from_degrees(np.linspace(0, 90, 181), np.linspace(0, 355, 72))
ff = ant.far_field(f0, angles)
rhcp = circular_components(ff["E_theta"], ff["E_phi"])["e_rhcp"]
d = pattern_directivity(ff)
```

## What this reproduces quantitatively

- Leaky-wave dispersion maps β_LW(m), α_LW(m) (paper Fig. 10 style).
- Broadside and tilted CP directivity patterns: beam direction, HPBW,
  near-in sidelobe envelope, peak directivity to ~1 dB of the lossless
  bound implied by the paper's quoted efficiencies (Figs. 5a, 6).
- Boresight axial ratio for spiral tensor designs.
- Gain-bandwidth products and relative-bandwidth closed forms
  (GB ≈ 22(v_g/c)a_λ etc.) and the fixed-modulation bandwidth roll-off.
- Superposed dual-band reactance layouts (paper Eqs. 4–6,
  `surfaces.dual_band_reactance_map`).

## What this does not reproduce, and why

- **GR-MoM / FMM full-wave results** — the package is reduced-order by
  design; validate against external solvers or measurements instead.
- **Printed element synthesis** (coffee-bean / grain-of-rice pixel
  databases) — requires periodic full-wave unit-cell simulation; import
  such data through `elements.LookupTableCell` / `adapters`.
- **Absolute gain including losses** — ohmic, dielectric and feed losses
  are not modeled; `launch_efficiency` is an explicit knob, so the model
  gives lossless directivity (an upper reference for measured gain).
- **Quantitative cross-polarization floors** — set by higher-order
  Floquet modes, launcher asymmetry and fabrication; the first-order
  tensor AFM model predicts the correct order of magnitude only.
- The measured curves themselves — quoted metrics are checked in
  `examples/data/faenzi2019_reference_metrics.csv`; digitized pattern
  curves can be dropped into `examples/data/` for overlay (see the
  README there).

## Conventions

Everything follows `core.conventions`: exp(+jωt) time dependence,
outgoing cylindrical waves H⁽²⁾, leaky wavenumber k = β − jα, ISO
spherical angles, IEEE RHCP with E_R = (E_θ + jE_φ)/√2.
