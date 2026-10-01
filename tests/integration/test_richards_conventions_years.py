"""The RZWQM2 conventions (DRAIN cap, flux-mode evaporation limit) on the M1 replays.

The M1 replay of ``richards_adaptive_years`` (restart mode, the years of a site as ``vmap`` lanes)
with ``AdaptiveStepping.exact()`` and ``RichardsConfig(drain_cap=..., evaporation_limit=...)`` and the site's
``aef``, for the four variants ``none``, ``drain``, ``flux``, ``both`` (the conventions are
measured apart from the solver; ``none`` is the plain adaptive run and reproduces its pins).

Measured (rorqual, float64; ``<data>/validation/w1_c_conv/m1_years.csv``), M1 site-years of 91 (CA-TPA
9 + 82 in the 8 comparable scenarios): none 26, drain 43, flux 32, both 62 (CA-TPA 9 in every variant;
outside CA-TPA 17, 34, 23, 53). No unconverged step, exhausted budget or held node in any run; the
largest accepted-step balance is 1.4e-14 cm with the cap and 7.9e-12 cm without. Pinned here: CA-TPA,
US_OPE and US_Rockfish, every year and variant (US_OPE: 1 / 7 / 1 / 9 of 9, US_Rockfish 0 / 0 / 2 / 8
of 8).
The pins are the measured values to 6 significant digits; the replay is deterministic, so the
tolerance is 1e-5 cm as in ``test_richards_years``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from richards_adaptive_years import replay_for, year_lanes
from richards_years import M1_STORAGE_RMSE

pytestmark = [
    pytest.mark.allow_skip(reason="needs the CA-TPA 2015-2023 run and the scenario runs under the data dir"),
    pytest.mark.skipif(not jax.config.read("jax_enable_x64"), reason="M1 is a float64 comparison"),
    pytest.mark.slow,
]

PIN_ABS = 1e-5
TOL_BALANCE = 1e-10
VARIANTS: dict[str, dict[str, Any]] = {
    "none": {},
    "drain": {"drain_cap": True},
    "flux": {"evaporation_limit": "flux_peak"},
    "both": {"drain_cap": True, "evaporation_limit": "flux_peak"},
}
#: (site, year) -> variant -> storage RMSE against RZWQM2 [cm]
PINS: dict[tuple[str, int], dict[str, float]] = {
    ("CA-TPA", 2015): {"none": 0.0194332, "drain": 0.0180709, "flux": 0.0187404, "both": 0.0173398},
    ("CA-TPA", 2016): {"none": 0.0353815, "drain": 0.0350519, "flux": 0.033678, "both": 0.0333318},
    ("CA-TPA", 2017): {"none": 0.0335203, "drain": 0.0328497, "flux": 0.0335203, "both": 0.0328497},
    ("CA-TPA", 2018): {"none": 0.0449496, "drain": 0.0447485, "flux": 0.0447367, "both": 0.044535},
    ("CA-TPA", 2019): {"none": 0.0307449, "drain": 0.0303861, "flux": 0.0307449, "both": 0.0303861},
    ("CA-TPA", 2020): {"none": 0.0216602, "drain": 0.0206528, "flux": 0.019985, "both": 0.0191092},
    ("CA-TPA", 2021): {"none": 0.028671, "drain": 0.0281393, "flux": 0.028671, "both": 0.0281393},
    ("CA-TPA", 2022): {"none": 0.0121697, "drain": 0.0121027, "flux": 0.0121697, "both": 0.0121027},
    ("CA-TPA", 2023): {"none": 0.0277877, "drain": 0.0273798, "flux": 0.0264517, "both": 0.0260303},
    ("US_OPE", 2015): {"none": 0.225981, "drain": 0.0684854, "flux": 0.22743, "both": 0.040694},
    ("US_OPE", 2016): {"none": 0.0691246, "drain": 0.0483234, "flux": 0.0590192, "both": 0.0327962},
    ("US_OPE", 2017): {"none": 0.0586582, "drain": 0.026707, "flux": 0.058543, "both": 0.0187132},
    ("US_OPE", 2018): {"none": 0.384485, "drain": 0.0368589, "flux": 0.384834, "both": 0.0362736},
    ("US_OPE", 2019): {"none": 0.0827391, "drain": 0.050323, "flux": 0.0716193, "both": 0.0330082},
    ("US_OPE", 2020): {"none": 0.108194, "drain": 0.0303019, "flux": 0.107764, "both": 0.0290943},
    ("US_OPE", 2021): {"none": 0.0632958, "drain": 0.0348638, "flux": 0.0572001, "both": 0.0260302},
    ("US_OPE", 2022): {"none": 0.033998, "drain": 0.0309692, "flux": 0.0328341, "both": 0.0274034},
    ("US_OPE", 2023): {"none": 0.0955533, "drain": 0.025129, "flux": 0.0939943, "both": 0.021914},
    ("US_Rockfish", 2016): {"none": 0.455201, "drain": 0.446137, "flux": 0.0764227, "both": 0.0350125},
    ("US_Rockfish", 2017): {"none": 0.302304, "drain": 0.300056, "flux": 0.0391909, "both": 0.0309755},
    ("US_Rockfish", 2018): {"none": 0.208444, "drain": 0.150805, "flux": 0.1457, "both": 0.0416345},
    ("US_Rockfish", 2019): {"none": 0.262079, "drain": 0.256623, "flux": 0.0426281, "both": 0.029283},
    ("US_Rockfish", 2020): {"none": 0.154642, "drain": 0.103205, "flux": 0.119902, "both": 0.0430245},
    ("US_Rockfish", 2021): {"none": 0.290169, "drain": 0.282905, "flux": 0.0721889, "both": 0.0354497},
    ("US_Rockfish", 2022): {"none": 0.280346, "drain": 0.271235, "flux": 0.0606109, "both": 0.0316122},
    ("US_Rockfish", 2023): {"none": 0.375923, "drain": 0.355495, "flux": 0.111955, "both": 0.0387648},
}
SITES = ("CA-TPA", "US_OPE", "US_Rockfish")


def _run(site: str, variant: str, data_dir: Path) -> dict[str, Any]:
    import jax.numpy as jnp
    from jax import lax

    from agrijax.processes.soil_water.richards import (
        AdaptiveStepping,
        RichardsConfig,
        RichardsParams,
        SoilWater,
        richards_day,
    )

    rep = replay_for(site, data_dir)
    lanes = year_lanes(rep)
    params = RichardsParams(
        soil=rep.soil,
        grid=rep.grid,
        config=RichardsConfig(**VARIANTS[variant]),
        stepping=AdaptiveStepping.exact(),
    )
    keys = (
        "n_unconverged",
        "budget_exhausted",
        "n_active",
        "step_balance_max",
        "drain_moved",
        "balance_error",
    )

    def lane(th0: Any, sup: Any, eva: Any, upt: Any) -> Any:
        def body(w: Any, f: Any) -> Any:
            w2 = richards_day(w, params, *f, aef=rep.aef)
            return w2, {"storage": w2.storage(rep.grid), **{k: getattr(w2.flux, k) for k in keys}}

        return lax.scan(body, SoilWater.from_theta(th0, rep.soil), (sup, eva, upt))[1]

    args = tuple(jnp.asarray(x) for x in (lanes.th0, lanes.sup, lanes.eva, lanes.upt))
    out = jax.jit(jax.vmap(lane))(*args)
    m = lanes.idx >= 0
    order = np.argsort(lanes.idx[m])
    r = {k: np.asarray(v)[m][order] for k, v in out.items()}
    rmse = {int(y): float(np.sqrt(np.mean((r["storage"][rep.years == y] - rep.storage[rep.years == y]) ** 2)))
            for y in np.unique(rep.years)}  # fmt: skip
    return {"rmse": rmse, **r}


@pytest.fixture(scope="module")
def runs(data_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    todo = [(s, v) for s in SITES for v in VARIANTS]
    with ThreadPoolExecutor(len(todo)) as ex:
        res = list(ex.map(lambda sv: _run(*sv, data_dir), todo))
    return dict(zip(todo, res, strict=True))


@pytest.mark.parametrize("site", SITES)
def test_m1_by_convention_variant_matches_the_measured_pins(runs: dict, site: str) -> None:
    for variant in VARIANTS:
        r = runs[(site, variant)]
        for year, rmse in r["rmse"].items():
            assert rmse == pytest.approx(PINS[(site, year)][variant], abs=PIN_ABS), (site, year, variant)
            assert (rmse < M1_STORAGE_RMSE) == (PINS[(site, year)][variant] < M1_STORAGE_RMSE)
        assert r["n_unconverged"].sum() == 0.0 and r["budget_exhausted"].sum() == 0.0
        assert r["n_active"].sum() == 0.0
        assert r["step_balance_max"].max() <= TOL_BALANCE
        assert (r["drain_moved"].sum() > 0.0) == ("drain_cap" in VARIANTS[variant])


def test_catpa_meets_m1_in_every_variant_and_both_conventions_lower_every_year(runs: dict) -> None:
    none, both = runs[("CA-TPA", "none")]["rmse"], runs[("CA-TPA", "both")]["rmse"]
    for variant in VARIANTS:
        assert all(v < M1_STORAGE_RMSE for v in runs[("CA-TPA", variant)]["rmse"].values())
    assert all(both[y] <= none[y] for y in none)
    assert max(both.values()) == pytest.approx(0.044535, abs=PIN_ABS)  # 2018; none: 0.0449496
