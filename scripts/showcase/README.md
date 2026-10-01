# Showcase figures

Scripts that draw the figures of the showcase page (`docs/showcase/fig/*.svg`, each in English `_en` and
Chinese `_zh`) from the result files of the validation and benchmark runs. Only the scripts are in the
repository; the result files (per-run reports, benchmark records) are read from a data directory
(`--data-dir`, `$AGRI_JAX_DATA`, default `~/agri_jax_data`) and are produced by the integration tests
(`--runslow`) and the benchmark scripts in `scripts/bench`, which run on the cluster.

| Script | Figures | Reads |
|---|---|---|
| `fig_validation.py` | `validation_runs` (65 runs: yield, LAI, soil-water and ledger error distributions), `validation_daily` (one season day by day against DSSAT) | `validation/aj_dint/d2_1a_free_run.json`, `validation/aj_dint/daily/<run>.npz`, a `dscsm048` reference run (`--run-reference`) |
| `fig_speed.py` | `speed_unified` (JAX and DSSAT-CSM on one timing boundary, 1 core to one H100), `speed_unroll` (the GPU layer-loop change) | `validation/aj_dbench_d42`, `validation/aj_dbench_d43` |
| `fig_calibration.py` | `calibration_g2g3` (loss contours with and without grain number) | `validation/aj_d31/ident.json` |
| `fig_swap.py` | `swap_soil_evaporation` (soil-evaporation swap: own method against the other, and the change in yield) | `validation/aj_dint/swap_soil_evaporation.json` |

```bash
python scripts/showcase/make_all.py --data-dir ~/agri_jax_data      # all of them
python scripts/showcase/fig_speed.py --data-dir ~/agri_jax_data     # one script; --lang en zh, --out DIR
```

The figures are plain SVG, 1000 x 640 units, text and marks on a panel-coloured background. Colours are CSS variables
with the palette of the page as fallback: inlined in the page they follow its light/dark switch, as an `<img>`
they follow `prefers-color-scheme`. Every number in a figure is read from the result files; the scripts check
the figures against the reports' own summaries and stop if text would leave the panel.

Requirements: `numpy`; `contourpy` >= 1.3 for the calibration figure; the daily validation figure also needs
`$AGRI_JAX_DSSAT` (a DSSAT-CSM engine directory) for its reference run.
