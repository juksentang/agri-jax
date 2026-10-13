# Swapping a process: the soil evaporation of the DSSAT day

Agri-JAX is an independent implementation from published equations, validated against DSSAT-CSM (BSD-3) and RZWQM2 outputs. Its DSSAT-CSM v4.8.6.0 day (`agrijax.models.day_dssat486`) is validated against DSSAT-CSM v4.8.6.0 (`dscsm048`) outputs. It is a table of entries (weather, soil water, SPAM, root water uptake, CERES-Maize, water ledger), and each entry runs a registered process. This page shows how to replace one of them, the soil evaporation, and what the framework checks when you do.

Two soil evaporation methods are registered, and both reproduce `dscsm048`:

| Key | Method | DSSAT `MESEV` | How the water leaves the soil |
|---|---|---|---|
| `soil_water/soilev@dssat-4.8.6.0:faithful` | Ritchie two-stage (`SOILEV`) | `R` | ES from the top layer |
| `soil_water/esr_soilevap@dssat-4.8.6.0:faithful` | SALUS layered (`ESR_SoilEvap`, Suleiman and Ritchie 2003) | `S` | ES_LYR layer by layer |

## The swap

```python
from agrijax.core import run
from agrijax.models.day_dssat486 import (
    SLOT, SOIL_EVAPORATION_KEYS, day_dssat486, day_outputs, day_params, day_processes,
)

key = SOIL_EVAPORATION_KEYS["S"]            # "soil_water/esr_soilevap@dssat-4.8.6.0:faithful"

# 1. the processes: one entry changes, and it is checked against the entry's interface
procs = day_processes(SLOT, soil_evaporation=key)

# 2. the params: the bucket and XTRACT follow the way this implementation removes water
params = day_params(site, crop_params, ksevap=kep, ktrans=kep, soil_evaporation=key)

# 3. the day: same entries, lags, port owners and phased writes as before; all checked
model = day_dssat486(SLOT).compile(procs, outputs=day_outputs(SLOT), check=True, exact_lags=True)

final, daily = run(model, params, forcing, state, return_final=True)
```

Pass the same key to both calls. Without the argument, `day_processes` uses Ritchie (`MESEV = R`) and `day_params` uses the site's `MESEV`; if the two disagree, the entry rejects the params (below). `examples/swap_soil_evaporation.py` runs a whole synthetic season both ways and prints the differences.

## What is checked

`day_processes` resolves the key in the process registry and checks the process in two steps. The first uses only its declarations. The second is a conformance gate: the process runs once on six synthetic profiles.

* **Units and dims.** Every field the process reads or writes must have the unit and dims of the SPAM store (`SoilEvapState`). A process whose state class declares `es` in `cm d-1` is rejected, because the store holds `mm d-1`.
* **Reads.** The process may read only what the entry provides: the day's soil water and WATBAL rates, the soil properties, the mulch step's `EOS_SOIL` and `EM`, and its own carried store. A process that reads the potential-rate port P5 directly is rejected, because the mulch entry has already turned `EOS` into `EOS_SOIL`.
* **Writes.** `ES` and `EVAP` are required, as whole fields: a sub-path such as `es_lyr.x` does not count. A layered method writes both `ES_LYR` and `SWDELTU`, never only one of them. Writes to the soil water or the soil properties are rejected, because those belong to the bucket.
* **Registry metadata.** The process needs a key in the `soil_water` slot, a supported grid, provenance and sources.
* **The gate.** The six profiles are wet, intermediate and dry, one with `EOS_SOIL = 0`, and one with `EOS_SOIL` above what the soil can supply. On each, `0 <= ES <= EOS_SOIL` and `EVAP = ES + EM` must hold. A layered method must also satisfy `ES = sum(ES_LYR)` and `ES_LYR = -SWDELTU DLAYR 10`, so the water the bucket removes is the water that is booked. A flipped `SWDELTU` sign, for example, fails here. The gate checks necessary conditions only. A process that halves `ES`, `ES_LYR` and `SWDELTU` together passes it, so validation against a reference stays in the integration tier.

Every failure raises `SoilEvaporationError` listing all the reasons:

```text
soil_water/potential_es_cm@none:demo cannot fill spam.soil_evaporation:
  potential_es_cm: field 'es' is in 'cm d-1' in the implementation, the SPAM store holds 'mm d-1'
```

**Conservation responsibility.** The evaporation processes only compute fluxes; the bucket removes the water. With Ritchie the bucket takes `ES` from layer 1. With SALUS it applies `SWDELTU` to every layer and skips its own upward flow, and XTRACT lowers each layer's available water by `ES_LYR`. `day_params(..., soil_evaporation=key)` sets this coupling (the bucket's and XTRACT's `salus_es` flags) from the implementation's declared writes. `soil_evaporation_params_problems(params, key)` checks the flags themselves on the host, including every sample of a batch. Inside `jit` the flags are traced, so the entry checks the static record that `day_params` stores alongside them, and rejects params that do not state one. Without these checks a mismatch would be silent: the water ledger would still close, but on the wrong water. A `SoilEvaporation` can only be built through its constructor, which runs these checks, and `resolve_soil_evaporation` checks any instance it is given again.

**The day does not change.** The entry's own reads and writes are the same for every implementation, so `Day.check` (allowed lags, one owner per path, phased writes) and the contract checks pass unchanged. `tests/unit/test_day_dssat486.py` checks both implementations, a conforming stand-in (no soil evaporation), and ten wrong stand-ins: wrong unit, missing dims, wrong ports, half-layered, sub-path writes, wrong slot and grid, no key, a flipped `SWDELTU`, `ES` halved while `ES_LYR` is left unchanged, and `ES` above the potential.

## What changes when you swap

`tests/integration/test_day_dssat486_swap.py` runs each reference treatment twice, once with its own method and once with the other, and writes the differences in yield, season ES and EP, profile water and maturity date to `validation/aj_dint/swap_soil_evaporation.json` in the data directory. The native run must stay within 2 % of the `dscsm048` yield. The swapped run is a different model, so only its closure is asserted: finite values and a closed daily water ledger.

## The same swap from the facade

The two methods are DSSAT's own `MESEV` option, so the facade (`import agrijax as aj`) offers them by name, on DSSAT's experiments, without any of the building blocks above:

```python
exp = aj.dssat.experiment("UFGA8201")                    # MESEV = R in its file
aj.dssat.alternatives()                                  # the swaps validated against dscsm048
salus = exp.run(treatment=4, soil_evaporation="salus")   # the same season with SALUS
ref = exp.reference(treatment=4, soil_evaporation="salus")   # dscsm048 with MESEV = S
salus.compare_summary(ref)
scen = exp.scenarios(treatment=4, years=range(1978, 1988), sowing_shift=[-14, 0, 14], soil_evaporation="salus")
res = aj.calibrate(exp, treatments=[4], holdout=[6], soil_evaporation="salus")
```

`soil_evaporation=` is accepted by `Experiment.run`, `inputs`, `scenarios`, `run_batch`, `reference` and `dssat_batch`, and by `aj.calibrate`. A swap sets `MESEV` on the `METHODS` line of a copy of the experiment file and builds everything from that copy: the season ends at the swapped model's own maturity, and the DSSAT runs (`reference`, `Scenarios.reference`, `dssat_batch`, the calibration's `dssat_check`) read the same file, so a swapped Agri-JAX run is always compared with DSSAT run on the same swap. A name that is not in `aj.dssat.alternatives()` raises a `SwapError` that lists the valid ones. The experiment file needs a `METHODS` line to carry the option, unless the swap is to DSSAT's default (Ritchie); a calibration on a swap needs the native inputs (`inputs="native"`, the facade's default). `tests/integration/test_facade_swap.py` runs the swapped seasons, scenarios, cultivar samples and a calibration against `dscsm048` on the same swap, and reproduces the numbers of the swap test above (`swap_soil_evaporation.json`) from the facade. A process of your own still goes through the building blocks above.

## Adding your own implementation

Write a `@process` on `SoilEvapState` with a key in the `soil_water` slot and its provenance. Use the `faithful` variant when it follows a reference version. Without a reference, use `ref_version = none` and a variant such as `demo`. Pass the process, or its key once it is registered, as `soil_evaporation=`. Run `soil_evaporation_problems(proc)` and `soil_evaporation_gate_problems(proc)` first to see what the entry would reject. Then write conformance cases for it like those of the built-in methods in `src/agrijax/testing/conformance/_builtin/dssat_evap.py`, and run the kit (`python -m agrijax.testing.conformance --key 'soil_water/*'`). Before you trust its numbers, compare it with its reference model in the integration tier.
