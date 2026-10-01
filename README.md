# Agri-JAX

**Differentiable, batch-parallel field-scale crop–soil models in JAX** — an independent implementation from published equations, validated against DSSAT-CSM (BSD-3) and RZWQM2 outputs.

[中文](README.zh.md) · [Showcase](https://juksentang.github.io/agri-jax/)

Every process is a pure function, so a season can be run for 10⁵ parameter sets in one `vmap` and differentiated with `jax.grad`. The models read the DSSAT and RZWQM parameter files that agronomists already have.

> **Status.** The DSSAT-CSM v4.8.6 daily model (multi-layer tipping-bucket soil water, SPAM evapotranspiration, root water uptake, CERES-Maize; nitrogen off) runs whole seasons free and is compared with `dscsm048` on 65 seasons. A Richards soil-water solver is a parallel module; its coupled assembly with RZWQM2-style processes is in progress. `agrijax.calib.calibrate` fits CERES-Maize cultivar coefficients in one call and writes them back as a DSSAT `.CUL` row. The PyPI package (`agrijax`) is a name-reserving placeholder.

## Quick start

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juksentang/agri-jax/blob/main/tutorial/agrijax_tutorial.ipynb)

```bash
pip install "agrijax[plot] @ git+https://github.com/juksentang/agri-jax"   # Python >= 3.11; no DSSAT program needed
```

```python
import agrijax as aj

exp = aj.dssat.experiment("UFGA8201")              # DSSAT's example experiment, read from its public files
season = exp.run(treatment=4)                      # one season of Agri-JAX's DSSAT-CSM v4.8.6 maize model
print(season.summary)                              # silking, maturity, grain yield, tops weight
aj.plot.season(season, observed=exp.observed(4))   # daily LAI, biomass, grain and soil water, with observations
season.write_dssat_out("out")                      # PlantGro.OUT, SoilWat.OUT, Summary.OUT in DSSAT's format

scen = exp.scenarios(treatment=4, years=range(1978, 1988), sowing_shift=[-14, 0, 14])
batch = scen.run({"G2": [800.0, 900.0, 1000.0]})   # every cultivar sample on all 30 scenarios, one batched call
g = exp.gradient(treatment=4, outputs=["HWAM"], params=["G2", "G3"])   # derivatives, each with a trust label
salus = exp.run(treatment=4, soil_evaporation="salus")                 # a validated alternative process
with aj.options(precision="float32", device="gpu"):                    # opt-in; your JAX settings are restored
    quick = scen.run({"G2": [800.0, 900.0, 1000.0]})
res = aj.calibrate(exp, treatments=[4], holdout=[6])                   # cultivar calibration (joint CMA-ES)
res.write_cul("out/MZCER048.CUL")

ref = exp.reference(treatment=4)                   # optional: the original DSSAT program, fetched on first use
season.compare_summary(ref)                        # dates and yield against DSSAT
res.check_dssat()                                  # the calibrated row run in DSSAT
```

Agri-JAX builds every input from DSSAT's own files (FileX, `.SOL`, `.WTH`, `.CUL` / `.ECO` / `.SPE`), downloaded once from DSSAT's repositories at fixed versions; the DSSAT program is needed only for the comparison calls (`exp.reference`, `check_dssat`). Treatments that need a DSSAT process Agri-JAX does not implement yet are refused with the reason (see Limitations). Calls outside `aj.options(...)` switch JAX to float64 for the rest of the process, since the model is validated in float64. `.OUT` files written by Agri-JAX say so in their header. The [tutorial](tutorial/agrijax_tutorial.ipynb) walks through all of this; calibration details are in [docs/calibration.md](docs/calibration.md).

## Validation

All rows: nitrogen off, float64, compared with DSSAT-CSM v4.8.6.0 (`dscsm048`) or RZWQM2 4.6 output; measured 2026-09-23 to 2026-09-30.

| Check | Result |
|---|---|
| 65 whole seasons run free (all modules of the day simulated; 58 DSSAT maize example treatments and 7 seasons at the AmeriFlux site CA-TPA; 11 138 days) | yield of the 59 seasons with grain: worst 1.5 % (BRPI0202 t04), median 4.0e-5 relative; the 6 growth-chamber seasons without grain give 0 as DSSAT does; emergence, silking and maturity dates equal in 65 of 65; growth stage and leaf number equal every day |
| Daily series of the same runs against DSSAT output | LAI RMSE ≤ 0.0033 (example treatments) and ≤ 0.0061 (CA-TPA) m² m⁻²; biomass (CWAD) RMSE ≤ 0.34 % and ≤ 0.075 % of the season maximum; profile water RMSE ≤ 0.33 mm; end-of-day soil water (tipping bucket) in every layer ≤ 9.2e-5 and ≤ 9.4e-4 cm³ cm⁻³ from DSSAT's own values. The example treatments agree at DSSAT's print resolution; the CA-TPA seasons are 5 to 18 times above it in daily evapotranspiration and soil water |
| Daily water ledger of the same runs | residual ≤ 6.6e-14 mm on every day of every run |
| XTRACT root uptake (58 runs, 10 050 days), PETPT potential ET and soil albedo (65 runs), one day at a time on the reference's own inputs | within 1.03, 4.7 and 4.5 units of REAL\*4 rounding; no day outside the test limit |
| CERES-Maize alone, driven by the reference's soil water and transpiration (58 treatments) | daily LAI within 0.50 %, biomass 0.63 %, yield 0.030 %, stages equal every day |
| H100 float64 (layer loops unrolled, the GPU default; whole card and a 1g.10gb slice) against CPU float64, 65 seasons | yield differs by ≤ 1.4e-15 relative, stage dates equal in 65 of 65 |
| Richards solver, 96×8 grid, CA-TPA 2015–2023 against RZWQM2, driven by RZWQM2's daily infiltration, evaporation and root uptake, each year started from RZWQM2's profile | storage RMSE 0.0152–0.0480 cm per year; the 24×3 grid is within 0.05 cm in five of nine years; daily mass-balance error ≤ 6.3e-6 cm |
| Shuttleworth–Wallace and ASCE potential ET, CA-TPA 2015 (one site-year) | potential evaporation RMSE 4.0e-4 mm/d and potential transpiration RMSE 6.6e-5 mm/d against RZWQM2; ASCE against `pyet` < 1e-3 mm/d |

Bit-for-bit reproducibility holds for a fixed program shape. On the CPU the same shapes and batch composition reproduce the acceptance report byte for byte; a different batch size or composition can change the last bits. On a whole H100, unrolled and looped layer recurrences give identical stage dates, soil water, runoff, drainage and yield, but last-bit differences in other outputs (17 of 65 seasons are bit-identical in every output). The tests are `tests/integration/test_day_dssat486_free.py` and `test_day_dssat486_free_gpu.py`; they compare against tables written by an instrumented DSSAT build, which are not distributed (the native-input test `tests/integration/test_dssat_free_inputs.py` checks that inputs built from DSSAT's files equal them).

## Speed

Task: the 65 validation seasons replicated to 10⁵ seasons (171 days on average), float64, ten daily series returned to the host. Agri-JAX perturbs 7 parameters by U(0.9, 1.1) per sample; DSSAT replicas are unperturbed. Hardware: rorqual (Calcul Québec), 2 × AMD EPYC 9654 (192 cores), a whole NVIDIA H100 80GB HBM3 (Xeon Gold 6448Y host), JAX 0.10.2. "1 core" is one pinned core of such a node; "32 cores" is a 32-core allocation of such a node, not a 32-core workstation. One measurement per cell; the repeated-call time varies by up to 20 % between processes on the node. CPU rows 2026-09-28, H100 rows 2026-09-29.

- **First run** (cold): a fresh process with an empty compilation cache, from Python start to the series on the host. **Warm start**: a second fresh process reading JAX's persistent compilation cache (set `jax_persistent_cache_min_compile_time_secs = 0`, or small programs are not cached); tracing and lowering are still paid. **Repeated call**: same process, inputs on the device, median of 3, summed over the six compiled programs (one per soil-layer count and evaporation method) and averaged over the two processes.
- **DSSAT-CSM**: from staging the run directories (node-local RAM disk; a disk-backed file system would be slower for the daily files) to the last `dscsm048` invocation having written its outputs, in parallel chunks over the cores. Repeated = the run part only, directories already staged.

| Hardware | Agri-JAX first run / warm start / repeated call | DSSAT-CSM whole run with daily files (summary only) / repeated | DSSAT ÷ Agri-JAX first run / warm start / repeated call |
|---|---|---|---|
| 1 core | 99.1 s / 56.0 s / 48.9 s | 1426 s (745 s) / 1426 s | 14 / 25 / 29 |
| 32 cores | 20.8 s / 11.3 s / 2.1 s | 46.8 s (26.9 s) / 46.4 s | 2.3 / 4.1 / 22 |
| 192 cores (whole node) | 20.5 s / 10.9 s / 0.66 s | 17.4 s (17.2 s) / 16.5 s | 0.85 / 1.6 / 25 |
| whole H100, layer loops unrolled (GPU default) | 77.5 s / 14.4 s / 0.424 s | not run | |
| whole H100, `AGRI_JAX_DEPTH_UNROLL=0` | 28.7 s / 11.2 s / 1.31 s | not run | |

The two sides do different work. DSSAT writes complete daily files (303 kB of text per season, 30.3 GB at 10⁵) and runs the whole CSM, including soil temperature and organic matter. Agri-JAX runs only the modules of the validated day and returns ten daily series (LAI, CWAD, GWAD, stage, profile water, ES, EP, EOP, runoff, drainage; 1.5 GB at 10⁵). Against summary-only DSSAT the 1-core ratios roughly halve (repeated 15 instead of 29); the node ratios do not move. The ratios are not the speed-up of one numerical kernel. On the current code the first-run and warm-start times are a few seconds higher, mostly in tracing and lowering (192-core node: 26.1 s / 13.6 s; H100 unrolled: 80.5 s / 16.8 s; repeated calls unchanged; measured 2026-09-30).

| Task | Use | Measured basis |
|---|---|---|
| ≤ 10³ seasons, once, cold start | DSSAT-CSM | 10³ seasons on 1 core: DSSAT 14.3 s, Agri-JAX 55.6 s first run (8.5 s warm start). One-off break-even: 1 core about 4·10³ seasons first run, 6·10² warm start; 32 cores about 4·10⁴ and 2·10⁴; 192 cores about 6·10⁴ warm start (first run beyond the measured range) |
| 10⁵ seasons, once, few cores | Agri-JAX | 14× (first run) to 25× (warm start) on 1 core, 2.3× to 4.1× on 32 cores |
| 10⁵ seasons, once, a whole 192-core node | either | 0.85× first run, 1.6× warm start |
| Many calls at one array shape (calibration, sensitivity, ensembles) | Agri-JAX, CPU node or GPU | repeated 10⁵-season call 22–29× faster than DSSAT on 1–192 cores; 0.66 s on the node, 0.42 s on the H100. Keep parameters as arguments and pad the sample count to a few sizes: a new shape recompiles |
| One GPU run or a few gradient steps | `AGRI_JAX_DEPTH_UNROLL=0` | unrolling makes each forward call 3.1× faster but adds about 45 s to a first run (10⁵ seasons; break-even about 50 repeated calls) and 251 s of compile in reverse mode (10⁴ seasons, 315.6 s against 64.2 s; break-even about 80 gradient calls) |
| Forward runs only, large batches | float32 | H100 1.83–2.15×, CPU node 1.15–1.23×. Yield differs from float64 by 3.3e-7 (median) and by up to 1.5 % in one of the 65 validation seasons; on 10 000 perturbed samples 13 of the 9076 seasons with grain differ by more than 0.1 %, at most 1.15 %. Not measured for gradients or calibration |

Not measured, so no ratio is given: 10⁶ or more seasons, T4 or Colab GPUs, laptops. The Agri-JAX side of the benchmark is `scripts/bench/d4_jax_scaling.py`.

## Calibration

`calibrate(...)` reads the treatments and observations of a DSSAT maize experiment, fits the six cultivar coefficients (P1, P2, P5, G2, G3, PHINT) by batched joint CMA-ES (the default), picks the best candidate by its value **as written** to the `.CUL` row (five characters per number; rounding can move a date by a day), writes the row and can run `dscsm048` with it. [docs/calibration.md](docs/calibration.md) gives scope, targets and methods.

- **UFGA8201, cultivar IB0035** (treatment 4 calibrated, treatment 6 held out; 32 CPU cores, 35.3 s, 2026-09-30): objective 1.015 → 0.0100, held out 1.046 → 0.0239. DSSAT-CSM with the written row gives the same silking and maturity dates and yield within 2.9e-4 relative of Agri-JAX.
- **Round trip, 22 calibrated runs** (11 treatments, two methods, rows written into a copy of `MZCER048.CUL`, 2026-09-28): dates equal and yield within 0.1 % of Agri-JAX in 22 of 22.
- **Why joint CMA-ES**: recovering known cultivars from noise-free synthetic observations (11 problems × 8 starts × 3 seeds), 225 of 264 fits reached an objective of 1e-4, against 212 for a staged scheme with gradient steps for the growth coefficients (the study's scheme; `method="staged"` is derivative-free), in 135 s against 210 s on a 192-core node (2026-09-28). The staged scheme needs fewer model calls per fit (median 443 forward and 51 two-direction derivative calls against 1008 forward). Gradient steps (`method="adam"`) use only coefficients whose gradient passes a trust report.
- **Identifiability, UFGA8201** (2026-09-28): from dates, yield, biomass and LAI alone, G2 and G3 lie along a curved valley of the loss, and fits spread over G2 355–982 and G3 7.6–16.4. Adding grain number (H#AM) concentrates them (G2 831–984, G3 7.5–9.4, near the published 924.3 and 8.17) and cuts the held-out grain-number error (treatment 6, best fit of each of three seeds): −51…−49 % → −7…−6 %. The price is the rainfed treatment 2, whose yield error grows from −5 % to −41 %: the model under-predicts kernel number under water stress. The loss contours are conditional slices, not posteriors, and are on the showcase page.

## Composable by design

A process is a pure function `(state, params, forcing) -> state` declared with `@process(reads=..., writes=...)`. Authors follow three rules and never write `scan`, `vmap` or `jit`: everything read is an argument and everything changed is returned; branch with `jnp.where`, never a Python `if` on state; no loops over layers, days or samples. An AST lint (AJ001–AJ012, AJ020–AJ021, which include NumPy calls on traced values and in-place changes of an argument) and, with `AGRI_JAX_CHECK=1`, a runtime check of the declared writes enforce them. Every model coefficient is declared once with unit, meaning and provenance (reference model and version, file and line), and is a calibratable parameter by default.

Ports carry units, each field has one owner, and the DSSAT day is a table of entries, each running a registered process. Any entry can be replaced; the replacement is checked against the entry's declarations and a conformance gate, and `python -m agrijax.testing.conformance --key 'soil_water/*'` runs the conformance kit (metadata, lint, units, reads by perturbation, balances, eager against `jit` and `vmap`, float32, gradients against finite differences). One line swaps the soil evaporation of the day from Ritchie (`MESEV = R`) to SALUS:

```python
procs = day_processes(SLOT, soil_evaporation=SOIL_EVAPORATION_KEYS["S"])   # both from agrijax.models.day_dssat486
```

Both methods reproduce `dscsm048`. Swapping changes the model: over the 59 seasons with grain (2026-09-30) the median yield change is −0.07 %, from −27.7 % (BRPI0202 t02) to +16.1 % (SIAZ9501 t03), and the water ledger closes to ≤ 5.9e-14 mm in every swapped run.

- [docs/swapping_a_process.md](docs/swapping_a_process.md): the swap and the checks it passes.
- [docs/tutorial_new_process.md](docs/tutorial_new_process.md): your own formula, from NumPy to a registered, checked process in eight steps, without data.
- [docs/debugging.md](docs/debugging.md): one process called directly, `jax.disable_jit`, `jax.debug.print`, NaN gradients, day-by-day comparison with a reference run.

## Related tools

PCSE (WOFOST in Python) has run-time variable ownership and per-module forced-state tests, the closest precedent for our ownership and replay checks. diffWOFOST and torchcrop (both PyTorch) are differentiable crop models with tutorials and gradient calibration. AquaCrop-OSPy is a Python AquaCrop with tutorials and a comparison against stored outputs of the original. Crop2ML is a component exchange standard and complements an in-framework contract; exporting Agri-JAX processes to it would be a natural step. APSIM Next Generation has swappable modules with regression tests in continuous integration. DSSAT-CSM itself runs gridded simulations in parallel through MPI (Geosci. Model Dev. 14, 6541, 2021). JAX-CanVeg is a differentiable canopy and land-surface model in JAX with a speed comparison against its MATLAB original. TrunX is a JAX implementation of the 3-PG forest model compared with r3PG. We found no other tool that combines a differentiable, batch-parallel, multi-layer soil–crop model with management events, validated against DSSAT-CSM outputs (and RZWQM2 outputs for its soil-water and PET components), with per-coefficient provenance and author checks (survey of September 2026); please tell us if we missed one.

## Limitations

- Maize only (CERES-Maize), nitrogen off. `calibrate` refuses treatments where nitrogen changes DSSAT's yield by more than 5 %.
- Inputs are built from DSSAT's files for treatments whose options Agri-JAX implements (50 of the 65 validation runs); automatic irrigation, the CENTURY organic matter with surface residue, tillage, tile drainage and a water table are refused. `calibrate` needs a measured nitrogen effect, so it runs on the DSSAT v4.8.6 maize example treatments only.
- Tile drainage exists as a framework only. The Richards solver is not yet coupled to the crop in an RZWQM2-style model; the 24×3 grid misses 0.05 cm in four of nine CA-TPA years, and on eight other RZWQM2 scenarios the 96×8 grid is within 0.05 cm in 18 of 82 site-years; Shuttleworth–Wallace is validated on one site-year.
- Bit-for-bit reproducibility holds for a fixed program shape (see Validation). float32 on the GPU is not bit-identical to the CPU, and float32 is measured for forward runs only.
- Gradients are labelled by scope: the forward model is validated; some parameters support gradients; gradients through phenology events are experimental (with `method="adam"`, P1, P2, P5 and PHINT stay derivative-free).
- The speed numbers are for one task on one cluster, one measurement per cell.

## Development

```bash
git clone https://github.com/juksentang/agri-jax && cd agri-jax
uv sync --all-extras
uv run pytest -q tests/unit tests/integration
uv run ruff check . && uv run pyright
uv run python -m agrijax.core.lint src --strict
```

Data-backed tests read from `--data-dir` or `AGRI_JAX_DATA`; the unit tier needs no data. `AGRI_JAX_CHECK=1` adds the runtime write check. [CONTRIBUTING.md](CONTRIBUTING.md) has the rules for merging.

| Directory | Contents |
|---|---|
| `src/agrijax/core` | state, process decorator, runtime, events, organ queue, units, lint |
| `src/agrijax/processes` | soil water, PET, crop, canopy, arbitration |
| `src/agrijax/models` | assembled models (the DSSAT day, demos) |
| `src/agrijax/calib`, `sites` | calibration (`calibrate`, optimisers, gradient trust) and the inputs of the free-run day |
| `src/agrijax/io`, `port` | RZWQM2, DSSAT, AmeriFlux readers and writers; reference-model runners and comparison tools |
| `tests`, `examples`, `scripts/bench` | tests, runnable examples, benchmark scripts |
| `docs` | the guides above and the source of the showcase page |

## Licence, provenance, acknowledgements, citation

Apache-2.0. CERES-Maize and the DSSAT soil-water balance are implemented independently from the open-source DSSAT-CSM (BSD-3), whose attribution is retained in `THIRD_PARTY_NOTICES.md`. Soil water and PET follow the published RZWQM2 equations (Ahuja et al., 2000; Farahani & Ahuja, 1996; Shuttleworth & Wallace, 1985). No RZWQM2 code is included or redistributed; RZWQM2 is a reference model whose outputs are compared and only the resulting numbers are reported.

The Newton early-stopping experiment for the Richards solver (`scripts/bench/collab/`) is by Jiaqi Zhang (pull request #1). This research was enabled in part by support provided by [Calcul Québec](https://www.calculquebec.ca) and the [Digital Research Alliance of Canada](https://alliancecan.ca). All GPU work runs on the rorqual cluster operated by Calcul Québec.

To cite the software, see `CITATION.cff`, which lists the authors. A design note and a validation report will follow as preprints.
