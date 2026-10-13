# Calibrating a CERES-Maize cultivar

`agrijax.calib.calibrate` fits the six cultivar coefficients of CERES-Maize (`P1`, `P2`, `P5`, `G2`, `G3`, `PHINT` of `MZCER048.CUL`) to the observations of a DSSAT maize example experiment. It writes them as a new cultivar row and can check that row in DSSAT-CSM. You do not assemble the model, the forcing or the parameter tree yourself.

```python
import jax

jax.config.update("jax_enable_x64", True)  # the model is validated in float64
from agrijax.calib import calibrate

res = calibrate(
    "UFGA8201",  # a DSSAT v4.8.6 maize example experiment, or a path to its .MZX
    treatments=[4],  # calibrate on these
    holdout=[6],  # evaluate on this one, never fit it
    write_cul="out/MZCER048.CUL",  # must not exist yet
    dssat_check=True,  # run dscsm048 with the new row and compare
)
print(res)
res.params  # {"P1": 246.3, "P2": 1.193, ...}: the values as written to the .CUL row
res.fit  # pandas table: one row per observation, observed / calibrated / published
res.dssat_check  # per treatment: DSSAT's and Agri-JAX's yield, silking and maturity dates
```

On a CPU node, set `XLA_FLAGS=--xla_force_host_platform_device_count=<cores>` before you import JAX. The candidates of all starts are simulated together and sharded over the devices.

## Current limitation: which experiments work

The DSSAT-CSM v4.8.6 day of Agri-JAX (`agrijax.models.day_dssat486`) is run free, with nitrogen off. Soil water (tipping bucket), soil evaporation, transpiration, root water uptake and CERES-Maize are all simulated. It is an independent implementation from the published DSSAT-CSM equations, validated against DSSAT-CSM (BSD-3) outputs on the maize example treatments.

The free-run day still takes the soil and the soil-water inputs, as DSSAT holds them, from tables written by an instrumented DSSAT-CSM build. These tables exist for the DSSAT v4.8.6 maize example treatments and are **not distributed**. `calibrate` therefore works only on those treatments, and only where the tables are under the data directory (`data_dir=` or `AGRI_JAX_DATA`). Your own experiments need a native builder of these inputs from the DSSAT input files, which is not written yet.

Among the example treatments, those where nitrogen matters are refused, because the calibrated model runs with nitrogen off. `agrijax.calib.workflow.NITROGEN_STRESS` holds, for every example treatment with tables, the change of DSSAT's yield between nitrogen on and off. It was measured with `dscsm048` using `scripts/calib/nitrogen_stress_table.py`, which you can rerun after a DSSAT engine update.

- A treatment is in scope when the change is at most 2 %.
- Between 2 % and 5 % it is in scope with a warning.
- Above 5 % it is refused.
- Dates: the script also records whether DSSAT's silking (`ADAT`) or maturity (`MDAT`) date changes with nitrogen on (the `dates differ` flag, `NITROGEN_DATES_DIFFER`). A treatment with a changed date is in scope with a warning even when its yield change is below 2 %, because the date targets would be fitted with nitrogen off. The flag does not refuse a treatment by itself. Only SIAZ9501 t03 and SIAZ9601 t07 are flagged (the growth-chamber experiment GAGR0201, refused as a whole, is flagged on every treatment). SIAZ9601 t07 (+0.4 % yield) is the only flagged treatment that is in scope.

| Experiment | Cultivar(s) | In scope | Warning (2–5 %, or dates differ) | Refused (> 5 %) |
|---|---|---|---|---|
| BRPI0202 | IB0171, IB0173, IB0172, IB0174 | t02, t03, t05–t08 | t04 (−4.9 %) | t01 (−6.9 %) |
| FLSC8101 | IB0012 | t01 | – | t02 (−10.5 %) |
| IBWA8301 | IB0063 (t1–3), IB0060 (t4–6) | t03, t06 | – | t01, t02, t04, t05 (−41 to −73 %) |
| IUAF9901 | IB1052 | t02 | t04 (−3.1 %) | t01, t03 (−44, −62 %) |
| SIAZ9501 | ZA0002 | t01, t02, t07 | t04, t05, t06, t08 (2.4–4.5 %) | t03 (+10.9 %, dates also differ) |
| SIAZ9601 | ZA0002 | t01–t04, t06, t08, t09 | t05 (−2.1 %), t07 (dates differ, +0.4 %) | – |
| UFGA8201 | IB0035 | t04, t06 | – | t01, t02 (+14.1 %), t03, t05 (−31, −26 %) |

GHWA0401 (a nitrogen × phosphorus trial) and GAGR0201 (growth chamber, its seasons end before silking) are refused as a whole. Anything outside the scope is refused with the reason (`ScopeError`) before anything runs. FLSC8101 t01 is in scope, but its observed maturity falls outside the simulated days (see Targets).

Each call calibrates one cultivar. If the treatments use several cultivars, pick one with `cultivar=`. With `treatments=None`, every supported treatment of that cultivar is used, except the held-out ones. Several experiments of one cultivar can be combined:

```python
calibrate(["SIAZ9501", "SIAZ9601"], treatments=[("SIAZ9501", 1), ("SIAZ9601", 1)])
```

## What runs

For every treatment, `dscsm048` first runs once with the published cultivar, nitrogen off. That run supplies the crop parameters, the weather and the season, from planting to maturity or harvest; the days of that season carry the real forcing. The forcing is then extended 60 days past the season (last day repeated, no rain or irrigation), so that a candidate cultivar that matures later still matures. If the calibrated cultivar matures on those padded days, you get a warning, because DSSAT will differ there.

## Targets

`targets="auto"` (the default) uses the silking and maturity dates (`ADAT`, `MDAT`), yield, tops weight and grain number at maturity (`HWAM`, `CWAM`, `H#AM`) and the LAI series (`LAID`), where observed. `targets="all"` adds every other observed variable that maps to a model output (`agrijax.calib.observations.DSSAT_TARGETS`: the organ-weight series `LWAD`, `SWAD`, `GWAD`, `CWAD`, leaf number ...). You can also pass a list, e.g. `targets=("ADAT", "MDAT", "HWAM", "H#AM")`. Variables without a model output are listed in `res.unsupported` with the reason.

The organ weights are opt-in because the leaf / stem split may be defined differently. On UFGA8201 the observed `LWAD` is 20–60 % below the CERES-Maize leaf weight at the published cultivar, and `SWAD` is 30–60 % above the model's stem weight, while their sum agrees within about 10 %. CERES-Maize's leaf weight includes the leaf sheaths. The observations may count the sheaths as stem. This has not been checked against the experiments' documentation.

The objective is, per observed variable, the mean squared normalised residual, summed over the variables. Dates are in days; other variables are divided by their mean observed value (a squared normalised RMSE).

**Nothing observed is dropped silently.**

- An A-file target (a date or a value at maturity) that cannot be placed on the simulated days raises `ObservationError`. For example, FLSC8101 treatment 1 has an observed maturity of 1981210, one day after the reference season ends; leave the treatment out, or pass `targets=` without `MDAT`.
- A T-file observation dated after the season, such as an LAI sampled after harvest, is not used. You get a warning, and it is listed in `res.dropped`.
- A date observed more than once is averaged, with a warning.

## Coefficients and methods

`params="auto"` (the default) starts from the six coefficients. Any coefficient the observations carry no information on is fixed at its published value: one whose change does not move the objective over its `MINIMA`–`MAXIMA` box, or within ±25 % of the published value. The usual case is P2 on a cultivar without a photoperiod signal. `res.sensitivity` holds the scans and `res.fixed` the reasons. An explicit `params=("P1", "P5", "G2", "G3")` calibrates exactly those, with a warning for any that do not move the objective.

If G2 and G3 are calibrated but no grain number (`H#AM` or `G#AD`) is observed, you get a warning. Yield, biomass and LAI alone do not separate kernel number from kernel weight, so G2 and G3 can trade off against each other.

- `method="cma"` (default): joint CMA-ES on all free coefficients from `starts` points (the published cultivar and points within ±25 % of it), with restarts and `budget` evaluations per start (default 4000).
- `method="staged"`: first CMA-ES on the phenology coefficients against the dates and the LAI before silking, then on G2 and G3 against the other targets, then a joint refinement. Every stage is derivative-free.
- `method="adam"`: gradient steps, only on the coefficients whose gradient passes the gradient-trust report on every treatment. P1, P2, P5 and PHINT are derivative-free by default, because their event gradients are experimental. The other coefficients stay at their published values and are reported in `res.trust`.

**The result is selected on the written coefficients.** A `.CUL` row holds five characters per value (for example P1 to 0.1, G3 to 0.01 or 0.001), and rounding can move a date by a day when a threshold is close.

- The candidates are, for every start: the best point of each run that ended in a restart, the start's overall best point and its 8 best points (unrounded objective). The last run of a start does not end in a restart and adds no candidate of its own. The published cultivar itself is also a candidate.
- Each candidate is rounded as the row will print it, and the one with the lowest objective at those values is returned.
- If no written candidate beats the published cultivar, the published cultivar is returned, with a warning, and `res.improved` is false.

`res.loss` reports the objective at the written values, at the unrounded point and at the published cultivar.

## Writing the row and checking it in DSSAT

`write_cul=` writes a copy of the engine's `MZCER048.CUL` with one new row. The row id is `cul_id=`, by default the first free id among `AJ0001`, `AJ0002`, ... The source file is never modified, and an existing destination is refused before anything runs.

`dssat_check=True` runs `dscsm048` on every treatment, calibrated and held out, with the experiment pointed at the new row. It then compares yield, silking and maturity with Agri-JAX's prediction at the written values. `res.dssat_check["agree"]` is true when the dates are equal and the yields are within 0.1 %.

## Example output

UFGA8201, cultivar IB0035, treatment 4 calibrated, treatment 6 held out, default settings, on the rorqual cluster with 32 CPU cores:

```
coefficient   published   calibrated (written)
  P1              259          246.3
  P2            1.193          1.193   fixed: flat: the objective does not change within +-25% of the published value
  P5            947.1          973.2
  G2            924.3          900.3
  G3             8.17          7.892
  PHINT            43          44.81
objective: published 1.015 -> calibrated 0.01003 (unrounded 0.01003)
held out: published 1.046 -> calibrated 0.0239
targets (observations): ADAT (1), CWAM (1), H#AM (1), HWAM (1), LAID (12), MDAT (1)
.CUL row:
  AJ0001 AgriJAX IB0035       . IB0001 246.3 1.193 973.2 900.3 7.892 44.81
DSSAT check (dscsm048 with the written row): agrees
  UFGA8201_t04: HWAM DSSAT 11773 / Agri-JAX 11776.4 (rel 2.9e-04); ADAT 1982132 / 1982132; MDAT 1982185 / 1982185
  UFGA8201_t06: HWAM DSSAT 9598 / Agri-JAX 9600.0 (rel 2.1e-04); ADAT 1982132 / 1982132; MDAT 1982185 / 1982185
wall 35.3 s (compile 5.2 s)
```

## Without DSSAT: the core

`agrijax.calib.fit.fit_cultivar` runs the same calibration on any `CultivarProblem`: a batch simulator of the six coefficients, the observations and the published cultivar. It reads no files and runs no DSSAT. The unit tests use it, and `calibrate()` itself, on a small synthetic twin (`tests/unit/test_calib_workflow.py`).
