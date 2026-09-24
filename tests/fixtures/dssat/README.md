# DSSAT-CSM fixtures (public, BSD-3-Clause)

`MZCER048.CUL`, `MZCER048.ECO` and `MZCER048.SPE` are copied unchanged from
[dssat-csm-os](https://github.com/DSSAT/dssat-csm-os) `Data/Genotype/` at commit
`d6556cca9f926c40eddffd9adcf77e970db7511b`, under the BSD-3-Clause licence reproduced in
`LICENSE-DSSAT.txt` (Copyright (c) 2021, DSSAT Foundation). They let the unit tier exercise the
genotype readers without the private data tree.

The experiment, weather and soil examples (`UFGA8201.MZX`, `UFGA8201.WTH`, `SOIL.SOL`) come from
[dssat-csm-data](https://github.com/DSSAT/dssat-csm-data), which carries no licence file, so they
are not redistributed here; the tests that read them are in `tests/integration/`.

`dscsm048_examples_simulated.csv` holds the *simulated* columns (dates, HWAM, CWAM, ADAP, MDAP,
LAIX, PRCM, ETCM) of the 76 maize runs of the 12 `example_data/Maize` experiments, as written by
the `dscsm048` v4.8.6.0 binary (BSD-3-Clause, dssat-csm-os). It is the reviewed baseline of
`tests/integration/test_dssat_examples.py::test_record_summary_csv`. The observed values and
treatment names come from dssat-csm-data and are therefore not included.
