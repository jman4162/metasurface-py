# SPEC.md

# `metasurface-py` (working name)
## Open-source Python package for design, analysis, and optimization of programmable electromagnetic metasurfaces for wireless communication and sensing

---

## 1. Executive summary

This document proposes the architecture and scope for an open-source Python package focused on **programmable electromagnetic metasurfaces** (also called RIS/IRS, reflectarrays, transmitarrays, and tunable impedance surfaces depending on context) for **wireless communication, integrated sensing, beam control, and inverse design**.

The recommendation is to build a package that is:

- **research-grade but not monolithic**;
- **Python-first** for rapid iteration, reproducibility, and optimization workflows;
- **physics-aware**, with support for reduced-order EM models and well-defined hooks into external full-wave solvers;
- **useful to PhD-level researchers and advanced engineers**, not just ML users;
- **modular enough** to support wireless communications, radar/sensing, and aperture engineering workflows;
- **deliberately scoped** so that v1 does not attempt to become a general-purpose commercial EM solver.

### Core recommendation

Build around a **three-layer architecture**:

1. **Core physics/data layer**
   - Metasurface geometry, lattice, unit-cell state, materials, parameter sweeps, labeled multidimensional datasets.
2. **Model/solver layer**
   - Reduced-order EM models, array-factor and aperture models, mutual-coupling approximations, scattering/channel models, sensing models, plus adapters to external solvers.
3. **Optimization/application layer**
   - Gradient-based and derivative-free optimization for beam steering, nulling, coverage shaping, localization, sensing, and joint comms-sensing objectives.

This is the right level of ambition because it fills a real tooling gap: there are strong existing tools for full-wave EM and RF network analysis, but far fewer Python-native frameworks that connect **metasurface physics**, **wireless system objectives**, and **reproducible optimization workflows**.

---

## 2. Product vision

### Vision statement

Create the default open-source Python framework for **programmable metasurface research**, where users can move from:

- unit-cell response models,
- to finite-aperture synthesis,
- to channel/sensing simulation,
- to constrained optimization,
- to validation against external full-wave tools,

without rewriting one-off MATLAB scripts for every paper.

### Primary value proposition

The package should make it easy to answer questions like:

- Given a discrete phase-codebook metasurface, what beam pattern and efficiency can I achieve over frequency and scan angle?
- How does mutual coupling or angle-dependent unit-cell response change system-level wireless gains?
- What surface configuration maximizes SINR, coverage, secrecy, localization accuracy, or radar detectability subject to hardware constraints?
- How close is a reduced-order model to full-wave validation for a given design regime?
- Which variables should be optimized: geometry, bias state, codebook, waveform, or joint TX/RX/metasurface control?

---

## 3. Design principles

1. **Reduced-order first, full-wave compatible**  
   The package should implement reduced-order models directly and interface with full-wave solvers rather than reimplementing a complete industrial-grade Maxwell solver in v1.

2. **Labeled scientific data over anonymous tensors**  
   Metasurface research naturally involves dimensions like frequency, polarization, angle, element index, code state, scan angle, and Monte Carlo trial. These should be explicit.

3. **Optimization is a first-class feature**  
   Optimization is not a thin wrapper added later. It should be central to the API.

4. **Physics constraints should be explicit**  
   Discrete states, phase quantization, amplitude-phase coupling, loss, bias topology, passivity, reciprocity, bandwidth limits, and fabrication constraints should be representable.

5. **Composable models beat giant inheritance trees**  
   Prefer functional and protocol-based composition over deeply nested object hierarchies.

6. **Reproducibility beats novelty theater**  
   Every simulation or optimization run should be easy to serialize, replay, compare, and validate.

7. **One excellent path per task**  
   Avoid five competing APIs for the same problem. Offer a small number of stable abstractions.

---

## 4. Target users

### Primary users

- PhD students in electromagnetics, antennas, RF, signal processing, and wireless communications;
- research engineers building RIS/metasurface demonstrators;
- advanced industry teams evaluating programmable surfaces for beam shaping, coverage enhancement, sensing, or localization;
- computational researchers doing inverse design or surrogate modeling.

### Secondary users

- ML engineers who need physically grounded environments for optimization;
- graduate students learning programmable aperture concepts;
- RF engineers integrating unit-cell measurements into array-level studies.

### Explicitly not the primary audience

- users who want a turnkey commercial CAD replacement;
- beginners with no EM or wireless background;
- users who only want a generic deep-learning framework.

---

## 5. Scope and non-goals

## In scope for v1

- 2D programmable metasurface apertures with configurable lattices;
- reflective and transmissive surfaces at a reduced-order modeling level;
- element-wise tunable states: phase-only, amplitude-phase, binary/multi-bit codebooks, impedance/admittance abstractions;
- frequency-dependent and angle-dependent element response models;
- finite-aperture field synthesis and far-field analysis;
- approximate mutual coupling models;
- communication-oriented propagation/channel models with metasurface interaction;
- sensing-oriented forward models and metrics;
- optimization under hardware constraints;
- adapters to external full-wave solvers and measured lookup tables;
- publication-quality parameter sweeps, ablations, and reproducible experiments.

## Out of scope for v1

- a new general-purpose 3D full-wave solver intended to compete with Meep, openEMS, CST, or HFSS;
- PCB/layout/CAD GUI authoring tools;
- full semiconductor/bias-network co-design at transistor detail;
- real-time digital twin infrastructure for deployed systems;
- photorealistic scene rendering and lidar-grade graphics pipelines;
- multi-physics thermal/mechanical solvers beyond simple hooks.

## Why these non-goals are correct

Trying to solve full-wave EM, CAD, optimization, wireless systems, and deployment in one package will produce a brittle research monolith. The highest leverage open-source contribution is the layer that is currently fragmented across scripts: **metasurface abstractions + reduced-order models + optimization + external-solver interoperability**.

---

## 6. Recommended technical positioning

This package should sit between three existing ecosystems:

1. **Scientific Python / optimization**  
   Numerical kernels, optimization, datasets, visualization, testing.

2. **RF and EM tooling**  
   Existing solvers and network-analysis packages should be leveraged, not displaced.

3. **Wireless communication and sensing simulation**  
   System-level objectives, channels, beamforming, localization, and radar metrics.

### Positioning statement

> `metasurface-py` is not a replacement for a full-wave solver. It is the open research layer that bridges tunable metasurface physics, system models, and optimization in a reproducible Python workflow.

---

## 7. Package architecture

## 7.1 High-level architecture

```text
metasurface_py/
├── core/           # canonical data models and shared math
├── geometry/       # lattice, aperture, coordinates, panels
├── materials/      # substrate, conductor, lumped/tunable states
├── elements/       # unit-cell response models and codebooks
├── surfaces/       # finite metasurface objects
├── em/             # reduced-order EM, fields, scattering, coupling
├── channels/       # comms propagation and RIS-assisted links
├── sensing/        # localization, radar, inverse scattering metrics
├── optimize/       # gradient-based, derivative-free, mixed-integer APIs
├── datasets/       # xarray-based sweeps and serialization
├── adapters/       # Meep/openEMS/scikit-rf/measured-data interfaces
├── experiments/    # reproducible run configs and orchestration
├── plotting/       # field, pattern, geometry, convergence, Pareto plots
├── io/             # import/export, config, result archives
├── cli/            # command line entry points
└── examples/       # research workflows and tutorials
```

## 7.2 Architectural layers

### Layer A: core physics/data layer

Responsibilities:

- coordinate systems;
- frequency grids;
- polarization conventions;
- aperture indexing;
- material and state parameter containers;
- canonical tensor shapes and metadata;
- units and validation.

This layer should have the fewest dependencies and the strongest API stability.

### Layer B: model/solver layer

Responsibilities:

- unit-cell response interpolation;
- equivalent surface impedance/admittance/susceptibility models;
- aperture current and field synthesis;
- mutual coupling approximations;
- reduced-order scattering and channel models;
- hooks to external EM tools.

### Layer C: optimization/application layer

Responsibilities:

- objective definitions;
- constraints;
- differentiable pipelines where possible;
- mixed discrete/continuous optimization;
- multi-objective studies;
- experiment management.

---

## 8. Canonical abstractions

The package should expose a small set of abstractions that appear repeatedly across workflows.

## 8.1 `Lattice`

Represents periodic or quasi-periodic element placement.

Fields:

- `basis_vectors`
- `shape`
- `spacing`
- `origin`
- `orientation`
- `mask`

Supports rectangular and hexagonal lattices initially.

## 8.2 `UnitCellModel`

Represents tunable element behavior.

Responsibilities:

- map control state and incident conditions to response;
- support scalar or matrix-valued response;
- expose codebook and admissible state set;
- provide interpolation across frequency and angle.

Subtypes:

- `PhaseOnlyCell`
- `AmplitudePhaseCell`
- `TensorCell`
- `LookupTableCell`
- `EquivalentImpedanceCell`
- `MeasuredCell`

## 8.3 `Metasurface`

Finite programmable aperture composed of a lattice plus cell model.

Responsibilities:

- assign state per element;
- evaluate aperture response;
- apply masks, defects, quantization, and hardware topology constraints;
- export panel state maps.

## 8.4 `SurfaceState`

A concrete realization of programmable controls.

Examples:

- binary state map;
- 2-bit phase state map;
- continuous tunable reactance map;
- grouped states constrained by bias lines.

## 8.5 `Scene` / `PropagationScenario`

Defines sources, receivers, carriers, paths, and environment abstractions.

Capabilities:

- LOS + RIS-assisted links;
- multi-user downlink/uplink abstractions;
- bistatic/monostatic sensing geometries;
- near-field and far-field source/observer placement.

## 8.6 `Objective`

Encapsulates a scalar metric to optimize.

Examples:

- peak gain at target angle;
- sidelobe suppression;
- weighted coverage;
- sum rate;
- SINR;
- localization dilution metric;
- Cramér–Rao-type surrogate objective;
- radar cross-section shaping;
- multi-band compromise objective.

## 8.7 `Experiment`

Serializable bundle describing one complete study.

Contains:

- geometry;
- materials;
- cell model;
- solver/model selection;
- objectives;
- optimizer settings;
- random seeds;
- outputs and metadata.

---

## 9. Mathematical modeling scope

The package should support a hierarchy of fidelity levels rather than one universal model.

## 9.1 Level 0: array-factor / phase-sheet models

Use when:

- rapid early-stage aperture synthesis is needed;
- coupling is weak or intentionally ignored;
- system-level optimization requires many evaluations.

Capabilities:

- steering and focusing;
- phase quantization effects;
- far-field pattern studies;
- basic communication/sensing sweeps.

Limitations:

- ignores many unit-cell physics effects;
- optimistic when strong angle/frequency/coupling dependence matters.

## 9.2 Level 1: element-response models with angle/frequency dependence

Use lookup tables or analytic models for element response:

- reflection/transmission coefficient vs frequency, angle, polarization, state;
- optional phase-amplitude coupling;
- optional passivity checks.

This should be the default v1 research mode.

## 9.3 Level 2: finite-aperture reduced-order interaction models

Add approximations for:

- mutual coupling;
- embedded element response;
- effective aperture current updates;
- local periodic approximation corrections;
- Green’s-function-based interaction kernels.

This gives a strong tradeoff between realism and runtime.

## 9.4 Level 3: external full-wave validation adapters

Use external solvers or measured data for:

- spot checks;
- calibration of surrogate models;
- paper-quality validation figures;
- extraction of lookup tables;
- selected finite-array studies.

The package should orchestrate these studies, but not own the full-wave numerics in v1.

---

## 10. Wireless communication and sensing modeling

A major differentiator should be support for **application-level metrics**, not just EM fields.

## 10.1 Communications workflows

Support:

- SISO/SIMO/MISO/MIMO abstractions where appropriate;
- narrowband first, wideband next;
- RIS-assisted channel composition;
- beamforming with metasurface state control;
- multi-user weighted objectives;
- hardware-aware rate/SINR optimization.

Representative outputs:

- effective channel matrices;
- received power maps;
- SINR and sum-rate curves;
- outage metrics;
- robustness to channel estimation error.

## 10.2 Sensing workflows

Support:

- illumination pattern shaping;
- target-dependent scattering approximations;
- localization geometry studies;
- range-angle-Doppler surrogate objectives where feasible;
- integrated sensing and communications tradeoff studies.

Representative outputs:

- ambiguity-like metrics;
- Fisher information surrogates;
- beampatterns for search vs track modes;
- clutter/interference shaping objectives.

## 10.3 Recommended modeling philosophy

Do not try to encode every wireless standard in the core package. Expose generic scene and channel abstractions, then provide examples for common research scenarios.

---

## 11. Optimization strategy

Optimization should be central, but the package should avoid pretending every metasurface problem is smoothly differentiable.

## 11.1 Optimizer families to support

### A. Deterministic gradient-based

Use for:

- continuous states;
- differentiable surrogates;
- inverse design with smooth relaxations.

Recommended methods:

- L-BFGS-B;
- SLSQP;
- projected gradient;
- augmented Lagrangian wrappers.

### B. Derivative-free local/global

Use for:

- discrete codebooks;
- noisy objectives;
- non-smooth mixed-variable design.

Recommended methods:

- differential evolution;
- CMA-ES or compatible optional dependency;
- simulated annealing for small discrete problems;
- random-restart local search.

### C. Mixed discrete-continuous workflows

Use for realistic hardware constraints.

Recommended approach:

- continuous relaxation for warm start;
- codebook projection / quantization;
- local discrete refinement.

### D. Multi-objective optimization

Support Pareto studies for:

- gain vs bandwidth;
- sum rate vs fairness;
- communication vs sensing;
- efficiency vs sidelobe level.

## 11.2 What not to over-engineer

Do **not** build a giant custom optimizer zoo in v1. Wrap mature scientific optimizers cleanly and focus original effort on:

- parameterization,
- constraints,
- differentiable objectives,
- quantization/projection logic,
- experiment reproducibility.

---

## 12. Recommended numerical stack

## 12.1 Required core dependencies

- **NumPy** for core arrays
- **SciPy** for interpolation, optimization, linear algebra, sparse tools
- **xarray** for labeled multidimensional datasets
- **Matplotlib** for baseline plotting
- **pydantic** or **attrs/dataclasses** for config validation
- **h5py** or xarray-native serialization backend

## 12.2 Recommended optional dependencies

- **JAX** for differentiable and JIT-accelerated pipelines
- **scikit-rf** for RF/network-data interoperability
- **PyTorch** only if learning-based surrogates become important enough to justify it
- **plotly** for interactive visualization
- **zarr/dask** for large parameter sweeps
- **meep** / **openEMS** adapters for external EM validation

## 12.3 Dependency philosophy

### Recommended

Keep the mandatory install lightweight enough that users can:

```bash
pip install metasurface-py
```

and immediately run reduced-order studies.

### Not recommended

Do not make full-wave solver bindings, GPU stacks, and ML frameworks hard dependencies. Those should be extras:

```bash
pip install metasurface-py[jax]
pip install metasurface-py[rf]
pip install metasurface-py[openems]
pip install metasurface-py[dev]
```

---

## 13. Data model and tensor conventions

This is one of the most important design decisions.

## 13.1 Use labeled arrays for scientific outputs

Outputs should be stored in a way that makes dimensions explicit.

Common dimensions:

- `freq`
- `theta`
- `phi`
- `pol_tx`
- `pol_rx`
- `x_elem`
- `y_elem`
- `state`
- `user`
- `target`
- `time`
- `trial`

### Why

Anonymous tensor conventions become unmanageable in metasurface work because the same study often mixes:

- element-space,
- aperture-space,
- angle-space,
- frequency-space,
- communication-user dimensions,
- sensing-target dimensions.

## 13.2 Canonical complex conventions

The spec must define clearly:

- phasor sign convention;
- time-harmonic convention;
- coordinate system;
- polarization basis;
- normalization of fields and power;
- far-field pattern normalization;
- dB vs linear storage rules.

These should live in one authoritative conventions module and docs page.

## 13.3 Recommended serialization

For v1:

- small to moderate results: xarray-backed NetCDF/HDF5;
- larger campaign outputs: optional Zarr.

This is enough without building a custom database layer.

---

## 14. External solver and measurement adapters

Interoperability is more important than solver ambition.

## 14.1 Adapter types

### A. Lookup-table import adapters

Import element responses from:

- measured data;
- HFSS/CST/openEMS/Meep generated tables;
- CSV/HDF5/Touchstone-like intermediate formats where appropriate.

### B. Orchestration adapters

Allow the package to:

- generate parameter sweeps;
- emit solver-ready configurations where possible;
- ingest results back into canonical datasets.

### C. Validation adapters

Utilities for:

- comparing reduced-order predictions to solver outputs;
- error metrics over frequency/angle/state;
- fitting surrogate models to trusted data.

## 14.2 Recommended priority order

1. measured/CSV/HDF5 lookup-table import
2. scikit-rf network-data import/export
3. openEMS adapter
4. Meep adapter
5. any deeper solver-specific automation later

This order reflects the highest research payoff for the least engineering burden.

---

## 15. API design recommendations

The API should favor a clean functional core with lightweight objects.

## 15.1 Example user workflow

```python
from metasurface_py.geometry import RectangularLattice
from metasurface_py.elements import LookupTableCell
from metasurface_py.surfaces import Metasurface
from metasurface_py.channels import RISLink
from metasurface_py.optimize import optimize_surface_state

lattice = RectangularLattice(nx=32, ny=32, dx=0.45, dy=0.45, unit="lambda0")
cell = LookupTableCell.from_hdf5("unit_cell_library.h5")
surface = Metasurface(lattice=lattice, cell=cell, mode="reflect")

problem = RISLink(
    surface=surface,
    tx=..., rx=..., freq=28e9,
    objective="rx_power",
    constraints={"bits": 2, "group_size": (2, 2)}
)

result = optimize_surface_state(problem, method="relax_then_quantize")
pattern = surface.far_field(result.state, theta=..., phi=...)
```

## 15.2 API rules

- Keep object construction explicit.
- Prefer immutable configs where practical.
- Avoid hidden globals.
- Separate model definition from experiment execution.
- Return rich result objects, not bare arrays.
- Every solver/optimizer call should return metadata, runtime, config, and diagnostics.

---

## 16. Constraints and hardware realism

This is a differentiator and should be treated seriously.

Support constraints such as:

- phase quantization;
- finite codebooks;
- grouped control lines;
- dead elements;
- lossy states;
- limited tuning ranges;
- element-wise manufacturing variability;
- sparse activation;
- switching penalties;
- time-varying reconfiguration cost;
- reciprocity/passivity constraints where applicable.

### Why this matters

A package that optimizes unconstrained continuous phase everywhere is useful for toy papers but much less useful for serious research.

---

## 17. Validation and verification strategy

A credible package needs a formal V&V plan.

## 17.1 Verification

Verify code against known identities and controlled cases:

- array-factor baselines;
- reciprocity checks where appropriate;
- energy accounting checks;
- consistency across coordinate transforms;
- convergence of numerical quadrature / interpolation;
- gradient checks for differentiable objectives.

## 17.2 Validation

Validate reduced-order models against:

- canonical analytical cases;
- published benchmark geometries;
- external full-wave spot checks;
- measured unit-cell or finite-array data when available.

## 17.3 Recommended benchmark suite

Ship a benchmark suite with a few durable cases:

1. phase-only anomalous reflection from finite aperture;
2. multi-beam synthesis with discrete states;
3. RIS-assisted SISO link improvement vs baseline;
4. near-field focusing example;
5. communication-sensing tradeoff example;
6. reduced-order vs external-solver comparison on a small finite surface.

---

## 18. Performance strategy

## 18.1 Performance philosophy

Optimize the kernels that matter, but do not prematurely migrate everything to C++/CUDA.

## 18.2 Recommended path

1. clean NumPy/SciPy implementation first;
2. profile;
3. vectorize bottlenecks;
4. optionally add JAX-backed kernels for differentiable heavy workloads;
5. add sparse/FFT accelerations where physics supports them.

## 18.3 What not to do in v1

- do not start with a custom C++ extension unless profiling proves it is necessary;
- do not require GPUs for core workflows;
- do not split the package into many micro-packages too early.

---

## 19. Experiment management and reproducibility

Each paper-quality result should be reproducible from a single serialized experiment config.

## 19.1 Recommended features

- config files in TOML or YAML;
- deterministic seed handling;
- automatic capture of package version, git commit, and environment;
- result manifests with metrics and output file paths;
- optional sweep runners for parameter studies.

## 19.2 Nice to have later

- lightweight experiment registry;
- remote execution hooks;
- cloud/HPC launch templates.

These are not v1 blockers.

---

## 20. Documentation strategy

Documentation is critical because the audience is advanced and impatient.

## 20.1 Documentation types

### A. Concept docs

Explain:

- modeling assumptions;
- fidelity levels;
- sign conventions;
- how communication and sensing abstractions map to EM quantities.

### B. API docs

Reference every public class/function cleanly.

### C. Research tutorials

Examples should look like mini papers:

- define problem;
- state assumptions;
- run simulation;
- optimize;
- plot results;
- interpret limitations.

### D. Validation notebooks

Show how reduced-order models compare to trusted references.

## 20.2 Documentation anti-patterns to avoid

- README-only documentation;
- hidden conventions buried in source code;
- demos that only work on toy random data;
- polished plots without exposing assumptions.

---

## 21. Testing strategy

## 21.1 Test categories

- unit tests for math and data handling;
- regression tests for benchmark outputs;
- property-based tests for invariants;
- gradient/finite-difference consistency tests;
- adapter tests with small static fixtures;
- slow integration tests gated in CI or nightly runs.

## 21.2 Numerical testing guidance

Use tolerances that reflect real numerical sensitivity, but do not allow vague tests that pass everything.

---

## 22. Packaging and repository structure

## 22.1 Repository layout

```text
repo/
├── src/metasurface_py/
├── tests/
├── docs/
├── examples/
├── benchmarks/
├── pyproject.toml
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── SPEC.md
```

## 22.2 Build and packaging recommendation

Use modern `pyproject.toml` packaging with optional extras and typed code.

Recommended baseline:

- Python 3.11+
- `src/` layout
- `ruff` + `pytest` + `mypy` or `pyright`
- GitHub Actions CI
- semantic versioning once public API stabilizes

---

## 23. Proposed public modules for v1

A realistic v1 public surface could be:

- `geometry`
- `elements`
- `surfaces`
- `em.aperture`
- `em.coupling`
- `channels`
- `sensing`
- `optimize`
- `datasets`
- `adapters.lookup`
- `plotting`

### Not recommended for v1 public API

- deeply nested solver internals;
- unstable low-level research kernels;
- half-finished machine-learning modules.

Keep those experimental.

---

## 24. Suggested development roadmap

## Phase 0: architecture and conventions

Deliver:

- conventions doc;
- core data classes;
- lattice and surface representations;
- xarray result containers;
- minimal plotting.

## Phase 1: reduced-order metasurface modeling

Deliver:

- phase-only and lookup-table cell models;
- far-field synthesis;
- quantized state support;
- simple communication and sensing scenarios;
- baseline optimizers.

## Phase 2: realism and validation

Deliver:

- angle/frequency-dependent response;
- coupling approximations;
- measured-data import;
- external validation adapters;
- benchmark suite.

## Phase 3: differentiable and large-scale workflows

Deliver:

- JAX backend for selected kernels;
- mixed discrete/continuous workflows;
- multi-objective optimization;
- larger sweep support.

## Phase 4: advanced research extensions

Possible directions:

- transmitarrays and multi-layer surfaces;
- dynamic time-sequence coding;
- near-field MIMO focusing;
- integrated sensing and communication pipelines;
- surrogate-model fitting and active learning.

---

## 25. Recommended choices and rationale

## Strong recommendations

### 1. Build reduced-order physics yourself; integrate full-wave tools instead of replacing them

**Why:** this is the highest-value open-source gap. It makes the package broadly useful without burying the project under solver complexity.

### 2. Make xarray-backed labeled outputs a core design choice

**Why:** metasurface research is inherently multidimensional, and mislabeled tensor axes create constant mistakes.

### 3. Treat optimization as a top-level package, not an afterthought

**Why:** most serious metasurface work is a design/optimization loop, not just a forward solve.

### 4. Support hardware constraints from the start

**Why:** otherwise the package will drift into idealized toy problems.

### 5. Keep the mandatory dependency footprint small

**Why:** install friction kills adoption, especially in academic labs.

## Moderate recommendations

### 6. Add JAX as an optional backend, not a mandatory core dependency

**Why:** it is powerful for differentiable inverse design, but many labs still want simple CPU-first NumPy/SciPy workflows.

### 7. Use dataclasses/attrs internally and reserve heavier validation for I/O boundaries

**Why:** this keeps the core lightweight and explicit.

### 8. Ship a small but polished benchmark suite early

**Why:** benchmark-driven trust matters more than a huge feature list.

## Not recommended early

### 9. Do not lead with deep RL or flashy AI modules

**Why not:** those features are easy to market but usually rest on weak physical abstractions if introduced too early.

### 10. Do not create a giant plugin framework before real users exist

**Why not:** most early plugin systems are architecture speculation. Simple adapter protocols are enough at first.

### 11. Do not overgeneralize for every metasurface domain immediately

**Why not:** microwave/wireless reflective and transmissive surfaces already provide a large, coherent first scope.

---

## 26. Risks and mitigation

## Risk 1: the package becomes a loose collection of scripts

**Mitigation:** enforce canonical abstractions and dataset conventions.

## Risk 2: the package becomes too idealized for serious researchers

**Mitigation:** include hardware constraints, validation workflows, and measured/full-wave adapters early.

## Risk 3: the package becomes too heavy to install or understand

**Mitigation:** keep v1 small, modular, and well documented.

## Risk 4: numerical assumptions become inconsistent across modules

**Mitigation:** centralize conventions and testing of invariants.

## Risk 5: performance disappoints on larger surfaces

**Mitigation:** profile first, then selectively accelerate bottlenecks with vectorization, sparse methods, or optional JAX.

---

## 27. Minimal viable v1 definition

A successful v1 should let a researcher:

1. define a finite metasurface aperture;
2. attach a tunable unit-cell response model;
3. impose discrete or continuous control constraints;
4. evaluate far-field or link-level metrics across frequency/angle;
5. optimize the surface state for a target objective;
6. serialize the experiment and reproduce the result;
7. optionally compare the reduced-order result to imported trusted data.

If v1 can do those seven things well, it is already highly publishable and genuinely useful.

---

## 28. Example v1 research workflows

### Workflow A: anomalous reflection synthesis

- import or define unit-cell reflection coefficients;
- create 40x40 reflective surface;
- optimize 2-bit phase map for target steering angle;
- quantify gain, efficiency, and sidelobes over bandwidth.

### Workflow B: RIS-assisted communication link

- define TX, RX, and surface geometry;
- simulate direct + RIS-assisted channel;
- optimize constrained states for received power or SINR;
- compare ideal continuous vs quantized control.

### Workflow C: joint comms-sensing tradeoff

- define communication user and sensing target;
- optimize weighted multi-objective metric;
- generate Pareto frontier over rate vs sensing score.

### Workflow D: reduced-order vs trusted data comparison

- import lookup table extracted from external solver;
- fit or interpolate cell model;
- compare finite-surface predictions against validation data;
- quantify error and operating regimes where the surrogate is acceptable.

---

## 29. Suggested contributor guidelines

Contributors should be encouraged to add:

- new unit-cell models;
- new scene/objective classes;
- new adapters;
- validated benchmark cases;
- tutorials tied to published or publishable workflows.

They should be discouraged from:

- adding undocumented one-off solver hacks;
- introducing competing conventions;
- merging large features without tests and benchmark cases.

---

## 30. Final recommendation

Build this package as a **research operating system for programmable metasurfaces**, not as a vanity full-wave solver and not as a pile of paper-specific scripts.

The right balance is:

- **more rigorous than a wireless-only RIS simulator**,
- **lighter and more extensible than a full EM CAD tool**,
- **more physically grounded than an ML-only optimization repo**.

That means:

- reduced-order EM in core,
- strong dataset and conventions layer,
- optimization as a first-class API,
- external solver interoperability,
- realistic hardware constraints,
- reproducible experiments.

That is ambitious enough to matter and scoped enough to ship.

---

## 31. Prior art and ecosystem notes

This proposal intentionally assumes interoperability with, rather than replacement of, mature tools in adjacent ecosystems such as:

- Scientific Python optimization and packaging tools
- xarray-based labeled scientific data workflows
- scikit-rf for RF/microwave network analysis
- Meep and openEMS for full-wave validation or data generation

The package should learn from these ecosystems’ strengths while staying focused on programmable metasurface workflows.
