# DSSAT-CSM fixtures (public, BSD-3-Clause)

`MZCER048.CUL`, `MZCER048.ECO` and `MZCER048.SPE` are copied unchanged from
[dssat-csm-os](https://github.com/DSSAT/dssat-csm-os) `Data/Genotype/` at commit
`d6556cca9f926c40eddffd9adcf77e970db7511b`, under the BSD-3-Clause licence reproduced in
`LICENSE-DSSAT.txt` (Copyright (c) 2021, DSSAT Foundation). They let the unit tier exercise the
genotype readers without the private data tree.

The experiment, weather and soil examples (`UFGA8201.MZX`, `UFGA8201.WTH`, `SOIL.SOL`) come from
[dssat-csm-data](https://github.com/DSSAT/dssat-csm-data), which carries no licence file, so they
are not redistributed here; the tests that read them are in `tests/integration/`.
