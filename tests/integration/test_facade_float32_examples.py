"""Float32 against float64 on every DSSAT maize example treatment the facade runs (slow): the figures
of the float32 accuracy note in :mod:`agrijax.facade_execution` are measured here, from DSSAT's public
example data only (no instrumented-engine tables, no DSSAT run).

* **Seasons.** Every treatment of the maize example experiments (``example_data/Maize/*.MZX``) that
  the native inputs support (the refused ones are listed with the reason and counted) runs with
  ``precision="float32"`` and with the default float64 (CPU): grain yield and tops weight at the end
  of the season, relative difference (:data:`SEASON_MEDIAN_RTOL`, :data:`SEASON_MAX_RTOL`), and the
  silking and maturity dates (none may differ).
* **A perturbed batch.** UFGA8201 treatment 4 on its ten weather years 1978-1987 x :data:`SAMPLES`
  cultivar samples with ``P1``, ``P5``, ``G2``, ``G3``, ``PHINT`` drawn uniformly within +-10 % of the
  published values (seed 0): the yield difference of the seasons with a yield (median, 99th
  percentile, largest, the count above 1e-3) and the stage dates.

Each figure is printed (``-s``) so the note can quote this run; the bounds are a few times the
measured values (they guard the note, they are not the note).
"""

from __future__ import annotations

import warnings
from typing import Any

import jax
import numpy as np
import pytest

import agrijax as aj
from agrijax.port.run_fortran import DSSAT_ENGINE
from agrijax.sites.dssat_free_run import NativeInputError

pytestmark = [
    pytest.mark.slow,
    pytest.mark.allow_skip(reason="needs DSSAT's example data (AGRI_JAX_DSSAT)"),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the float64 reference runs in float64"),
]

MAIZE = DSSAT_ENGINE / "example_data" / "Maize"
CPU = {"device": "cpu"}
#: the seasons: median and largest relative yield / tops-weight difference, float32 against float64
SEASON_MEDIAN_RTOL = 1e-5
SEASON_MAX_RTOL = 5e-2
#: the perturbed batch: cultivar samples per weather year, the coefficients perturbed and the bounds
SAMPLES = 100
PERTURBED = ("P1", "P5", "G2", "G3", "PHINT")
BATCH_MEDIAN_RTOL = 1e-5
BATCH_P99_RTOL = 1e-3
BATCH_MAX_RTOL = 5e-2
YEARS = range(1978, 1988)


def _rel(a: Any, b: Any) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.abs(b), 1.0)


def _same_date(a: float, b: float) -> bool:
    return (np.isnan(a) and np.isnan(b)) or a == b


@pytest.fixture(scope="module")
def seasons() -> dict[str, Any]:
    files = sorted(MAIZE.glob("*.MZX"))
    if not files:
        pytest.skip(f"DSSAT's maize example experiments not found under {MAIZE}")
    rows: list[dict[str, Any]] = []
    refused: dict[str, str] = {}
    for fx in files:
        exp = aj.dssat.experiment(fx.stem, data_root=DSSAT_ENGINE)
        for t in exp.treatments:
            key = f"{exp.name}_t{t:02d}"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", aj.dssat.NitrogenWarning)
                try:
                    s64 = exp.run(treatment=t, **CPU)
                except NativeInputError as e:
                    refused[key] = "; ".join(e.features)
                    continue
                s32 = exp.run(treatment=t, precision="float32", **CPU)
            rows.append(
                {
                    "key": key,
                    "hwam64": s64.summary["HWAM"],
                    "yield": float(_rel(s32.summary["HWAM"], s64.summary["HWAM"])),
                    "tops": float(_rel(s32.summary["CWAM"], s64.summary["CWAM"])),
                    "dates_differ": sum(
                        not _same_date(s32.summary[k], s64.summary[k]) for k in ("ADAT", "MDAT")
                    ),
                }
            )
    return {"rows": rows, "refused": refused}


def test_float32_seasons_on_the_example_treatments(seasons):
    rows = seasons["rows"]
    y = np.array([r["yield"] for r in rows])
    b = np.array([r["tops"] for r in rows])
    worst = rows[int(np.argmax(y))]
    print(
        f"\nfloat32 vs float64, {len(rows)} example treatments run ({len(seasons['refused'])} refused: "
        f"{sorted(seasons['refused'])}); with grain {sum(r['hwam64'] > 0 for r in rows)}\n"
        f"  yield: median {np.median(y):.2e}, max {y.max():.2e} ({worst['key']}), above 1e-3: "
        f"{int((y > 1e-3).sum())}, above 1e-2: {int((y > 1e-2).sum())}\n"
        f"  tops weight: median {np.median(b):.2e}, max {b.max():.2e}\n"
        f"  silking / maturity dates differing: {sum(r['dates_differ'] for r in rows)} of {2 * len(rows)}"
    )
    for r in sorted(rows, key=lambda r: -r["yield"])[:5]:
        print(f"  {r['key']}: yield {r['yield']:.2e}, tops {r['tops']:.2e}, HWAM {r['hwam64']:.0f}")
    assert len(rows) >= 55  # the maize example treatments the facade runs (57 measured, 19 refused)
    assert sum(r["dates_differ"] for r in rows) == 0
    assert np.median(y) < SEASON_MEDIAN_RTOL and np.median(b) < SEASON_MEDIAN_RTOL
    assert y.max() < SEASON_MAX_RTOL and b.max() < SEASON_MAX_RTOL


def test_float32_on_a_perturbed_batch():
    mzx = MAIZE / "UFGA8201.MZX"
    if not mzx.is_file():
        pytest.skip(f"UFGA8201 not found under {MAIZE}")
    exp = aj.dssat.experiment("UFGA8201", data_root=DSSAT_ENGINE)
    scen = exp.scenarios(treatment=4, years=YEARS)
    rng = np.random.default_rng(0)
    pub = scen.published
    cul = {n: pub[n] * rng.uniform(0.9, 1.1, SAMPLES) for n in PERTURBED}
    t64 = scen.run(cul, **CPU).table
    t32 = scen.run(cul, precision="float32", **CPU).table
    alive = t64["HWAM"].to_numpy() > 0
    y = _rel(t32["HWAM"], t64["HWAM"])[alive]
    dates = sum(
        not _same_date(a, b)
        for k in ("ADAT", "MDAT")
        for a, b in zip(t32[k].to_numpy(float), t64[k].to_numpy(float), strict=True)
    )
    print(
        f"\nUFGA8201 t4, {len(YEARS)} years x {SAMPLES} samples ({', '.join(PERTURBED)} +-10 %): "
        f"{len(t64)} seasons, {int(alive.sum())} with a yield; float32 vs float64 yield median "
        f"{np.median(y):.2e}, p99 {np.quantile(y, 0.99):.2e}, max {y.max():.2e}, above 1e-3: "
        f"{int((y > 1e-3).sum())}; stage dates differing {dates} of {2 * len(t64)}"
    )
    assert alive.sum() > len(t64) // 2
    assert dates == 0
    assert np.median(y) < BATCH_MEDIAN_RTOL and np.quantile(y, 0.99) < BATCH_P99_RTOL
    assert y.max() < BATCH_MAX_RTOL
