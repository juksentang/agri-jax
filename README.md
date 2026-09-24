# Agri-JAX

**Differentiable, batch-parallel field-scale crop–soil process models in JAX.**

[中文](README.zh.md) · [Showcase](https://juksentang.github.io/agri-jax/)

> **Status (September 2026): early implementation.** The core runtime, file readers, reference-model runners and the potential-evapotranspiration processes are in place and tested. The Richards soil-water solver and the CERES-Maize crop module are the next two pieces. The PyPI package is a name-reserving placeholder.

## Acknowledgements

This research was enabled in part by support provided by [Calcul Québec](https://www.calculquebec.ca) and the [Digital Research Alliance of Canada](https://alliancecan.ca). All GPU work runs on the rorqual cluster operated by Calcul Québec.

## What it is

Agri-JAX writes a field-scale crop–soil model as pure JAX functions. The crop part is an independent implementation of CERES-Maize from the open-source [DSSAT-CSM](https://github.com/DSSAT/dssat-csm-os) (BSD-3). Soil water and potential evapotranspiration follow the published RZWQM2 equations: a Brooks–Corey Richards equation, Shuttleworth–Wallace evapotranspiration and tile drainage (Ahuja et al., 2000; Farahani & Ahuja, 1996; Shuttleworth & Wallace, 1985).

Because every process is a pure function, a multi-year field simulation can be run for 10⁵ parameter sets in one `vmap`, differentiated with `jax.grad`, and placed inside gradient-based calibration, Hamiltonian Monte Carlo, ensemble data assimilation or a hybrid process–ML training loop. Correctness is checked against the outputs of the DSSAT-CSM and RZWQM2 reference models.

Speed is not the goal in itself. The aim is a model with layered soil physics and management events that is as cheap to run in bulk and as easy to differentiate as the simpler differentiable crop models, while still reading the DSSAT and RZWQM parameter files that agronomists already have.

## Implementation status

| Component | Implemented | Compared against a reference |
|---|---|---|
| Core: state pytrees, `@process`, scan/vmap runtime, three-rules lint | yes | runtime vs an independent Python day loop (1e-12); gradients vs finite differences |
| RZWQM2 and DSSAT-CSM file readers and writers | yes | byte-identical round trips on 15 scenarios; DSSAT example outputs |
| Reference-model runners (RZWQM2, DSSAT-CSM) | yes | all 15 RZWQM2 scenarios and all DSSAT maize examples run |
| Brooks–Corey hydraulics | yes | reference-model field capacity and wilting point on 100+ horizons; gradient checks |
| Shuttleworth–Wallace, ASCE and Priestley–Taylor PET | yes | ASCE to 5e-6 mm/d; Shuttleworth–Wallace growing-season RMSE 0.02 / 0.17 mm/d vs RZWQM2 on one site-year |
| Organ queue, event table, AmeriFlux loader | yes | conservation properties; AmeriFlux CA-TPA data |
| Richards soil-water solver | next | — |
| CERES-Maize crop module | next | — |
| Coupled water–crop model, calibration, uncertainty quantification | after the two above | — |

New modules are added only after the existing ones pass automated, deterministic comparisons against an independent reference.

## Why

| Task | Runs needed | Needs gradients |
|---|---|---|
| Calibration and uncertainty quantification (LHS, MCMC, HMC) | 10⁴–10⁶ | yes, for HMC and variational inference |
| Regional gridded simulation (pixels × years × scenarios) | 10⁶–10⁸ | no |
| Ensemble data assimilation (EnKF, particle filters) | 10³–10⁴ members in lockstep | no |
| Hybrid process–ML models | one batch per training step | yes |

DSSAT, RZWQM2, APSIM, STICS and WOFOST are sequential Fortran or C# codes built for one field at a time. Their implementations do not natively support array-style batch execution and are not differentiable. Differentiable crop models do exist: diffWOFOST and torchcrop for crop growth, JAX-CanVeg for the canopy and land surface. Agri-JAX differs in what it couples differentiably: layered soil-water dynamics, crop growth and management events, with the DSSAT and RZWQM parameter files usable as they are. Comparisons between these tools are best made on soil layering, the flow equation, management processes, gradient handling and validation scope.

## Design

A process is a pure function `(state, params, forcing) -> state`, declared with `@process(reads=..., writes=...)`. Process authors follow three rules and never write `scan`, `vmap`, `jit` or `lax.cond`:

1. Everything read is an argument; everything changed is in the return value.
2. Branch with `jnp.where` or `jnp.select`, never with a Python `if` on state.
3. Never write a loop over layers, days or samples. The runtime owns time and batch.

The rules are enforced by an AST lint and, with `AGRI_JAX_CHECK=1`, by a runtime check of declared writes. They bind process authors; numerical kernels such as the tridiagonal solver are written directly in JAX.

A model is a state definition plus an ordered list of processes, and any process can be swapped. Soil water will come in two interchangeable forms, a DSSAT-style tipping bucket and an RZWQM-style Richards solver, so one crop module can be compared under both. Crops are a state dimension, so intercropping is a change of array shape rather than a new code path. Organs such as leaves are held in a fixed-size queue, which leaves room for crops like tobacco that are harvested leaf by leaf.

## Preliminary throughput

These numbers come from a computational skeleton, not the full model. The skeleton has the planned per-day shape: a 37-node implicit Richards step with 24 sub-steps and 3 Newton iterations, a tridiagonal solve, Shuttleworth–Wallace-shaped PET and a CERES-shaped crop step, over 3287 days. On one NVIDIA H100:

| Configuration | 10⁵ nine-year runs | Throughput |
|---|---|---|
| float64, 24 sub-steps × 3 Newton | 243 s | 411 runs/s |
| float32, 24 × 3 | 101 s | 993 runs/s |
| float64, 12 × 2 | 82 s | 1220 runs/s |

The RZWQM2 reference binary takes 24 s per run on one CPU core. The same 10⁵ runs cost 667 core-hours, roughly 10 wall-clock hours on a cluster. At this batch size the GPU is saturated, so run time follows the number of implicit solves per day. How far the scheme can be thinned while staying in agreement with the reference model, and while keeping gradients trustworthy, is the question the first paper will answer.

## Roadmap

1. **Richards solver and CERES-Maize.** Each is validated day by day against RZWQM2 and DSSAT-CSM outputs before coupling.
2. **Coupled model on one site.** AmeriFlux CA-TPA maize, 10⁵-sample timing, gradient checks at three levels: plausible outputs, correct derivatives, and derivatives fit for inference.
3. **Calibration and uncertainty.** Gradient-based and HMC calibration, Sobol sensitivity, multi-site joint calibration and parameter identifiability.
4. **Extensions.** Intercropping, nitrogen and carbon cycling, hybrid process–ML models, and a report interface for agricultural economists with Stata and R front ends.

## Development

```bash
git clone https://github.com/juksentang/agri-jax && cd agri-jax
uv sync --all-extras
uv run pytest -q tests/unit tests/integration
uv run ruff check . && uv run pyright
uv run python -m agri_jax.core.lint src --strict
```

Data-backed tests read from `--data-dir` or `AGRI_JAX_DATA`. The unit tier needs no data. The PyPI release (`pip install agri-jax`) stays a placeholder until the coupled model works.

| Directory | Contents |
|---|---|
| `src/agri_jax/core` | state, process decorator, runtime, events, organ queue, units, lint |
| `src/agri_jax/processes` | soil water, PET, crop, canopy, arbitration |
| `src/agri_jax/models` | assembled models |
| `src/agri_jax/io` | RZWQM2, DSSAT, AmeriFlux and CA-TPA readers |
| `src/agri_jax/port` | reference-model runners and comparison reports |
| `docs/showcase` | source of the showcase page |

## Licensing and provenance

Apache-2.0. CERES-Maize is implemented independently from the open-source DSSAT-CSM (BSD-3), whose attribution is retained. Soil water and PET are implemented from the published RZWQM2 equations. No RZWQM2 code is included or redistributed. RZWQM2 is used only as a reference model: its outputs are compared privately and only the resulting numbers are reported. The comparison against DSSAT-CSM is public and reproducible.

## Citation

See `CITATION.cff`. A design note and a validation report will follow as preprints.
