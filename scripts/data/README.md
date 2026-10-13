# Local data layout (`~/agri_jax_data`)

Nothing below is in git. The root is `$AGRI_JAX_DATA` (default `~/agri_jax_data`);
`tests/conftest.py` passes it to tests as `--data-dir`, and data-backed tests skip when a file is
missing. The path is kept short on purpose: the RZWQM2 binary truncates file names at 80
characters, so every run directory must have an absolute path shorter than 80 characters.

```
agri_jax_data/
├── narval_mirror/                 reference scenarios and tools (not distributed)
│   ├── RZWQM_sw_batch/            scenarios: CA-TPA/, CA-ER1/, ... ; all_parameters.csv; NOTES.md
│   │   └── CA-TPA/
│   │       ├── Scenario/          rzwqm.dat cntrl.dat IPNAMES.DAT MZDSSAT.RZX CA-TPA.MET/.BRK/.sno
│   │       │                      MZCER040.CUL/.ECO/.SPE DSSATWTH.WTH rzinit.dat plgen.dat expdata.dat
│   │       └── AutoAnalysis/      parameter.csv (the LHS matrix, row i = run i), AutoAnalysis.py,
│   │                              Experiment_Result_RZBatch.xlsx (observations), analysis_result.xlsx
│   ├── RZWQM_Tool/                main_ryzen5_avx512, DSSAT/ database, LHS_ana_Gen/GenerateDat.py, rz_indiv.sh
├── catpa_lhs/                     CA-TPA LHS batch results (not distributed)
│   ├── catpa_daily.npz            float32 [n_run, n_day]: sw_cm evap_cm transp_cm lai grain_kg_ha aet_cm;
│   │                              run [n_run]; time [n_day] (YYYY.DDD from January 1 of the first year)
│   ├── catpa_yield.npz            run, season (0-based), yield_kg_ha (from each OVERVIEW.OUT)
│   └── npy/                       the members of catpa_daily.npz uncompressed, made on first
│                                  load_catpa_lhs() call so the arrays can be memory-mapped
├── catpa/                         derived CA-TPA reference data (scripts/data/make_catpa_refs.py)
│   ├── events.csv                 management events of the run window: date,event,value,unit,source_line,detail
│   ├── ref_2015/                  one-year base run: LAYER.PLT CA-TPA.ana MANAGE.OUT OVERVIEW.OUT IPNAMES.DAT
│   ├── base_2015_2023/            full-period base run, same files
│   └── lhs_run0/                  full-period run of LHS row 0 (0_rzwqm.dat + outputs)
├── run/                           RZWQM / DSSAT run directories (short paths; scratch, safe to delete)
│   └── layers/                    the hand-made one-year run used to locate the per-layer output
├── port_index/                    Fortran source index (agrijax.port.fortran_index)
└── venv/, *.log                   cluster helper venv and monitor logs
```

Rebuild everything derived:

```bash
uv run --project ~/Agri_JAX python ~/Agri_JAX/scripts/data/make_catpa_refs.py --extract-npy   # ~40 s + npy
```

## Loaders (`agrijax.io.catpa`)

| function | returns |
|---|---|
| `read_layer_output(path, start=None)` | `LAYER.PLT` as an `xarray.Dataset` with dims `(time, depth)` (also `agrijax.io.rzwqm.layers`) |
| `build_events(dat, start, end)` / `load_events(path)` | the management table (`pandas.DataFrame`) |
| `read_manage_out(path)` | events that RZWQM2 reports in `MANAGE.OUT` (used to check `build_events`) |
| `load_catpa_lhs(data_dir, variables=, runs=, mmap=True)` | dict: `run`, `time`, `yyyyddd`, daily arrays, `yields` [n_run, n_season], `season_year` |
| `load_catpa_params(path, canonical=False)` | `parameter.csv` as a DataFrame indexed by `run` |

### `.ana` column → LHS array

| array | `.ana` column (1-based) | header |
|---|---|---|
| `sw_cm` | 2 | STORED SOIL WATER (CM), whole profile |
| `evap_cm` | 6 | ACTUAL EVAPORATION (CM/DAY) |
| `transp_cm` | 7 | ACTUAL TRANSPIRATION (CM/DAY) |
| `lai` | 43 | LEAF AREA INDEX |
| `grain_kg_ha` | 44 | BIOMASS OF GRAIN (KG/HA) |
| `aet_cm` | 84 | ACTUAL ET (CM), daily |

The extraction skipped the `<first year>.000` initial-state row, so day index 0 is January 1 of
the first simulated year. Run `i` used row `i` (0-based) of `parameter.csv`: `GenerateDat.py`
writes `{i}_rzwqm.dat` from row `i`. Re-running row 0 here (`catpa/lhs_run0`) reproduces the
cluster's seasonal yields to within a few kg/ha (floating-point drift between the two machines).

## Per-layer soil water: `LAYER.PLT`

RZWQM2 writes theta for every numerical node every day to `LAYER.PLT` (the "vector plot file").
It is controlled by two records of `cntrl.dat`:

* `VARIABLES TO BE PLOTTED AGAINST TIME AND DEPTH`: the number of variables (up to 8) and their
  ids (e.g. soil water content, pressure head, bulk density, NO3-N, plant water uptake). The
  loaders need soil water content, bulk density and NO3-N for the DSSAT initial conditions.
* `Output Control`: `VECTOR TABULAR  SCALAR TABULAR  VECTOR PLOT  SCALAR PLOT`. The third flag
  writes `LAYER.PLT`. The first flag would add `LAYER1.OUT ...`, which holds the same data in
  9-column pages and is not needed.

Rows are `DAY DEPTH var1 ... var8` (`FORMAT(1X,I10,3X,30G15.6E3)`). `DAY` counts simulation days
from 1 (the IPNAMES start date) and does not reset at the end of a year. Each node depth is the
bottom of its layer (the node records of `rzwqm.dat`), so the layer thickness is
`diff([0, depth])`, and `sum(theta * thickness)` reproduces `.ana` column 2 to output precision. The other candidates are not per node. `AVG6IN.OUT` holds 6-inch (15.24 cm) averages.
`SoilWat.OUT` covers the DSSAT soil layers, every 10 days, and only while a crop is growing.
The tabular `LAYER*.OUT` files are switched off (first Output Control flag = 0).

## Management

The only management source is the scenario's `rzwqm.dat`: `MZDSSAT.RZX` holds DSSAT settings and
`expdata.dat` holds observations. `build_events` resolves its plant-management, fertilizer,
pesticide, tillage, manure and irrigation blocks (including the planting-relative records) the
way `Rzman.for` does, within the run window, and `make_catpa_refs.py` checks the result against
`MANAGE.OUT` of `base_2015_2023` event for event before writing `catpa/events.csv`. The scenario
values (dates, rates, depths) are data: they live only in the local files, never in the
repository. Loaders that need them (e.g. the season dates of
`agrijax.io.ameriflux.seasons_from_events`) read `events.csv`.
