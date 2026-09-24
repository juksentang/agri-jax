# Agri-JAX

**Differentiable, batch-parallel field-scale crop–soil process models in JAX.**

Agri-JAX is an independent JAX implementation of the CERES-Maize crop module from the open-source [DSSAT-CSM](https://github.com/DSSAT/dssat-csm-os) (BSD-3), and of RZWQM2-style soil water and potential evapotranspiration from the published equations (Ahuja et al., 2000; Farahani & Ahuja, 1996; Shuttleworth & Wallace, 1985): Brooks–Corey Richards equation, Shuttleworth–Wallace potential evapotranspiration and tile drainage, written as pure JAX functions so that a full multi-year field simulation can be run for 10⁵ parameter sets in one `vmap`, differentiated with `jax.grad`, and embedded in gradient-based calibration, Hamiltonian Monte Carlo, ensemble data assimilation, or hybrid process–ML training loops. It is validated by comparing its outputs against the DSSAT-CSM and RZWQM2 reference models.

The goal is not speed for its own sake. It is to make a *field-scale* model, with real soil physics and management events, as cheap to run in bulk and as differentiable as the regional-scale bucket models that already exist in this space, while staying compatible with the DSSAT and RZWQM parameter files that agronomists have built up over decades.

> **Status: design stage (September 2026).** The package on PyPI is a placeholder that reserves the name and installs an empty module. The design documents below are complete; the proof of concept starts now. Nothing here has been validated against the reference models yet. Do not use it for science until the validation report exists.

## Showcase

An interactive walkthrough of the design is live at <https://juksentang.github.io/agri-jax/en/> (English) and <https://juksentang.github.io/agri-jax/> (Chinese).

## Why

| Task | Runs needed | Needs gradients |
|---|---|---|
| Calibration / uncertainty quantification (LHS, MCMC, HMC) | 10⁴–10⁶ | yes (HMC, variational) |
| Regional gridded simulation (pixels × years × scenarios) | 10⁶–10⁸ | no |
| Ensemble data assimilation (EnKF, particle filters) | 10³–10⁴ members in lockstep | no |
| Hybrid process–ML models (model inside a training loop) | one batch per step | yes |

Existing crop models (DSSAT, RZWQM2, APSIM, STICS, WOFOST) are sequential Fortran or C# codes: fast for one field, impossible to batch, and not differentiable. Differentiable modelling has already reshaped hydrology; field-scale crop–soil modelling is still a gap. The differentiable crop models that exist today (diffWOFOST, torchcrop) are regional-scale bucket models. Agri-JAX is field-scale.

## Design in three rules

A process is a pure function of `(state, params, forcing) -> state`. Authors follow three rules and never see `scan`, `vmap`, `jit`, or `lax.cond`:

1. Everything read is in the arguments; everything changed is in the return value.
2. Branch with `jnp.where` / `jnp.select`, never with Python `if` on state.
3. Never write a loop. Time and samples are handled by the runtime.

A model is a state definition plus an ordered list of processes; any process can be swapped. Soil water comes in two interchangeable flavours, `tipping_bucket` (DSSAT) and `richards` (RZWQM), so the same crop module can be compared under both. Crops are a state dimension (`n_crop`), so intercropping is a shape, not a code path.

## Preliminary throughput (skeleton, not the real model)

A computational skeleton with the same per-day shape as the planned model (37-node implicit Richards × 24 sub-steps × 3 Newton iterations, tridiagonal solve, Shuttleworth–Wallace-shaped PET, CERES-shaped crop step, 3287 days) on one NVIDIA H100:

| Precision | 10⁵ nine-year runs | Throughput |
|---|---|---|
| float64 | 243 s | 411 sims/s |
| float32 | 101 s | 993 sims/s |

The Fortran reference (RZWQM2, one CPU core) takes 24 s per run: 667 core-hours and roughly 10 wall-clock hours for the same 10⁵ runs on a cluster. At this batch size the GPU is saturated and time scales with the number of implicit solves per day: halving sub-steps and Newton iterations (12 × 2) gives 82 s for 10⁵ runs and 14 minutes for 10⁶ runs on one H100. How far the scheme can be thinned without losing agreement with the reference model is the core question of the first paper.

## Roadmap

| Phase | Deliverable |
|---|---|
| Proof of concept (4 weeks) | RZWQM water balance + Shuttleworth–Wallace PET + CERES-Maize on one site, validated day by day against RZWQM2 and DSSAT-CSM reference-model outputs; 10⁵-sample `vmap` timing; gradient check and one NUTS run |
| Framework + first models (6 months) | pip package, docs, validation report, reference-model comparison tools, Sobol and multi-site calibration, intercropping, an economist-facing report API (posterior intervals, marginal effects, elasticities, identifiability diagnostics) with Stata/R front ends |
| Applications | Multi-site joint gradient calibration and parameter identifiability; hybrid process–ML models on held-out flux-tower sites |

## Documentation

Design documentation is maintained in an internal archive and will be published with the first validated release.

| Document | Content |
|---|---|
| [README.zh.md](README.zh.md) | Original plan (Chinese) |

## Install

```bash
pip install agri-jax        # placeholder 0.0.x: reserves the name, no model code yet
```

## Licensing and provenance

Apache-2.0. CERES-Maize is implemented independently from the open-source DSSAT-CSM (BSD-3), whose attribution is retained. Soil water and PET are implemented from the published RZWQM2 equations (Ahuja et al., 2000; Farahani & Ahuja, 1996; Shuttleworth & Wallace, 1985); no RZWQM2 code is included or redistributed. RZWQM2 is used only as a reference model: the comparison against its outputs is run privately and reported as numbers only. The comparison against DSSAT-CSM is public and reproducible.

## Citation

See `CITATION.cff`. A design note and the validation report will be posted as preprints.
