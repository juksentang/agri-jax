# Local data layout (`~/agri_jax_data`)

Nothing below is in git. The root is `$AGRI_JAX_DATA` (default `~/agri_jax_data`);
`tests/conftest.py` passes it to tests as `--data-dir`, and data-backed tests skip when a file is
missing. The path is kept short on purpose: the RZWQM2 binary truncates file names at 80
characters, so every run directory must have an absolute path shorter than 80 characters.

```
agri_jax_data/
├── narval_mirror/                 read-only mirror of narval (scripts/data_sync.sh mirror)
│   ├── RZWQM_sw_batch/            scenarios: CA-TPA/, CA-ER1/, ... ; all_parameters.csv; NOTES.md
│   │   └── CA-TPA/
│   │       ├── Scenario/          rzwqm.dat cntrl.dat IPNAMES.DAT MZDSSAT.RZX CA-TPA.MET/.BRK/.sno
│   │       │                      MZCER040.CUL/.ECO/.SPE DSSATWTH.WTH rzinit.dat plgen.dat expdata.dat
│   │       └── AutoAnalysis/      parameter.csv (100 000 x 30 LHS matrix, row i = run i), AutoAnalysis.py,
│   │                              Experiment_Result_RZBatch.xlsx (observations), analysis_result.xlsx
│   ├── RZWQM_Tool/                main_ryzen5_avx512, DSSAT/ database, LHS_ana_Gen/GenerateDat.py, rz_indiv.sh
│   └── RZWQM_Linux_Ver45/src/     RZWQM2 Fortran source (private: read it, never copy it into the repo)
├── catpa_lhs/                     CA-TPA 100k-run LHS results (scripts/data_sync.sh catpa_lhs, ~4 GB)
│   ├── catpa_daily.npz            float32 [100000, 3287]: sw_cm evap_cm transp_cm lai grain_kg_ha aet_cm;
│   │                              run [100000]; time [3287] (YYYY.DDD, 2015.001 .. 2023.365)
│   ├── catpa_yield.npz            run, season (0..6 = 2015..2021), yield_kg_ha (from each OVERVIEW.OUT)
│   └── npy/                       the members of catpa_daily.npz uncompressed (7.9 GB), made on first
│                                  load_catpa_lhs() call so the arrays can be memory-mapped
├── catpa/                         derived CA-TPA reference data (scripts/data/make_catpa_refs.py)
│   ├── events.csv                 management events 2015-2023: date,event,value,unit,source_line,detail
│   ├── ref_2015/                  one-year base run: LAYER.PLT CA-TPA.ana MANAGE.OUT OVERVIEW.OUT IPNAMES.DAT
│   ├── base_2015_2023/            full-period base run, same files
│   └── lhs_run0/                  full-period run of LHS row 0 (0_rzwqm.dat + outputs)
├── run/                           RZWQM / DSSAT run directories (short paths; scratch, safe to delete)
│   └── layers/                    the hand-made one-year run used to locate the per-layer output
├── port_index/                    Fortran source index (agri_jax.port.fortran_index)
└── venv/, *.log                   cluster helper venv and monitor logs
```

Rebuild everything derived:

```bash
bash ~/Agri_JAX/scripts/data_sync.sh catpa_lhs                                  # npz from rorqual
uv run --project ~/Agri_JAX python ~/Agri_JAX/scripts/data/make_catpa_refs.py --extract-npy   # ~40 s + npy
```

## Loaders (`agri_jax.io.catpa`)

| function | returns |
|---|---|
| `read_layer_output(path, start=None)` | `LAYER.PLT` as an `xarray.Dataset` with dims `(time, depth)` (also `agri_jax.io.rzwqm.layers`) |
| `build_events(dat, start, end)` / `load_events(path)` | the management table (`pandas.DataFrame`) |
| `read_manage_out(path)` | events that RZWQM2 reports in `MANAGE.OUT` (used to check `build_events`) |
| `load_catpa_lhs(data_dir, variables=, runs=, mmap=True)` | dict: `run`, `time`, `yyyyddd`, daily arrays, `yields` [n_run, 7], `season_year` |
| `load_catpa_params(path, canonical=False)` | `parameter.csv` as a DataFrame indexed by `run` |

### `.ana` column → LHS array

| array | `.ana` column (1-based) | header |
|---|---|---|
| `sw_cm` | 2 | STORED SOIL WATER (CM), 0-150 cm |
| `evap_cm` | 6 | ACTUAL EVAPORATION (CM/DAY) |
| `transp_cm` | 7 | ACTUAL TRANSPIRATION (CM/DAY) |
| `lai` | 43 | LEAF AREA INDEX |
| `grain_kg_ha` | 44 | BIOMASS OF GRAIN (KG/HA) |
| `aet_cm` | 84 | ACTUAL ET (CM), daily |

The extraction skipped the `2015.000` initial-state row, so day index 0 is 2015-01-01.
Run `i` used row `i` (0-based) of `parameter.csv`: `GenerateDat.py` writes `{i}_rzwqm.dat` from
row `i`. Re-running row 0 here (`catpa/lhs_run0`) reproduces the cluster's seasonal yields to
within 2 kg/ha: [10082, 7108, 6551, 9923, 9467, 10205, 11585] here against
[10082, 7108, 6551, 9924, 9467, 10207, 11585] on the cluster. The small gap is floating-point
drift between the two machines.

## Per-layer soil water: `LAYER.PLT`

RZWQM2 writes theta for every numerical node every day to `LAYER.PLT` (the "vector plot file").
The shipped CA-TPA `cntrl.dat` already switches this on, so no changes are needed:

* `VARIABLES TO BE PLOTTED AGAINST TIME AND DEPTH`: `2 43 20 9 32 33 39 36`. These are soil
  water content, pressure head, bulk density, NO3-N, pesticide 1 and 2, plant water uptake and
  N uptake. Up to 8 variables are allowed.
* `Output Control`: `VECTOR TABULAR  SCALAR TABULAR  VECTOR PLOT  SCALAR PLOT` = `0 1 1 1`.
  The third flag writes `LAYER.PLT`. Setting the first flag would add `LAYER1.OUT ...`, which holds
  the same data in 9-column pages and is not needed.

Rows are `DAY DEPTH var1 ... var8` (`FORMAT(1X,I10,3X,30G15.6E3)`). `DAY` counts simulation days
from 1 (the IPNAMES start date) and does not reset at the end of a year. CA-TPA has 37 nodes
from 1 to 150 cm, and each node depth is the bottom of its layer. So the layer thickness is
`diff([0, depth])`, and `sum(theta * thickness)` reproduces `.ana` column 2 to within 6e-5 cm
on every day of 2015. The other candidates are not per node. `AVG6IN.OUT` holds 6-inch (15.24 cm) averages.
`SoilWat.OUT` covers the 8 DSSAT soil layers, every 10 days, and only while a crop is growing.
The tabular `LAYER*.OUT` files are switched off (first Output Control flag = 0).

## Management (CA-TPA, 2015-2023)

The only source is `rzwqm.dat`: `MZDSSAT.RZX` holds DSSAT settings and `expdata.dat` holds
observations. The window has 7 maize seasons. Planting is on 04-28 in 2015-2019 and on 05-05 in
2020-2021, at 80 000 seeds/ha with 76 cm row spacing. Harvest is on a fixed date (09-20 or
09-25, and 10-25 / 10-20 in 2020 / 2021). Each season also gets:

* 180 kg/ha NO3 broadcast on the planting day,
* 2.5 kg/ha glyphosate the day before planting,
* one chisel-plow tillage on a fixed date, 15 cm deep (some years after planting).

There is no irrigation or manure, and 2022-2023 are fallow. `build_events` resolves the
planting-relative records the way `Rzman.for` does, and the result matches `MANAGE.OUT` of
`base_2015_2023` event for event.
