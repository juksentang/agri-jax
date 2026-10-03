"""The weather sensitivity facade (``agrijax.facade_weather``) on functions with known answers, no data.

* ``HMET``'s ``TAVG`` is linear in ``(TMAX, TMIN)`` for a given day length: the float64 weights sum to
  1 and reproduce the float32 reference transcription to its rounding;
* the day verdict follows ``agrijax.calib.trust``'s levels: level 2 three-valued on the exact derivative
  at the small steps, level 3 = the straight-through derivative agrees with every user-step secant
  (disagreeing secants fail, they are never undecidable), the day's level 1 / 2 / 3; the labels and
  caveats (radiation and temperature: the stage calendar; rain and irrigation: germination);
* the right-hand derivative of the mulch interception at 0 mm of rain (1 - cover, not 1), values unchanged;
* the perturbation hook adds a day's change to every place the model reads it (and only that day);
* the single-day rows (top-k and random days; one difference scheme per day: forward at every step
  where the value would go below 0, else central) and the whole-season rows;
* the whole pipeline (``_analyse``: calendar, stages, single-day check, whole-season check, labels,
  phenology-free companion, brute force, attribution) on a toy season whose outputs are known
  functions of the weather: exact for a linear output (every check passes, the whole-season gap and
  the attribution's rerun-minus-linear are 0), a quadratic output (central differences exact; the
  forward-only rain differences fail at the user steps), with the caveats in every row;
* the climatology of weather files (day-of-year means, Feb 29, leave-one-out: the attribution's default
  climatology leaves the season's own year out), the stage helpers, the facade signatures, and the
  facade's cache: a second ``year=`` call reuses the scenario's inputs and programs.
"""

from __future__ import annotations

import datetime as dt
import inspect
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from agrijax import dssat as ajd
from agrijax import facade_weather as fw
from agrijax.forcing.dssat_weather import (
    daylength486,
    hourly_mean_temperature,
    hourly_mean_temperature_weights,
)

pytestmark = [
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="finite differences need float64"),
    pytest.mark.allow_skip(reason="finite differences are float64-only (CI float32 pass)"),
]


# ------------------------------------------------------------------------------ TAVG weights
def test_tavg_weights_sum_to_one_and_reproduce_hmet():
    doy = np.arange(1, 366, 7)
    sun = daylength486(doy, 29.6)
    rng = np.random.default_rng(1)
    tmin = rng.uniform(-5.0, 22.0, doy.size)
    tmax = tmin + rng.uniform(2.0, 15.0, doy.size)
    ref = hourly_mean_temperature(tmax, tmin, sun).astype(float)
    wx, wn = hourly_mean_temperature_weights(sun.dayl.astype(float))
    np.testing.assert_allclose(wx + wn, 1.0, atol=1e-12)
    assert np.all((wx > 0.0) & (wx < 1.0))
    # float32 reference arithmetic: a few units of the last place of ~30 degC
    np.testing.assert_allclose(wx * tmax.astype(np.float32) + wn * tmin.astype(np.float32), ref, atol=5e-5)


# ------------------------------------------------------------------------------ the verdict
def test_day_verdict_follows_the_trust_levels():
    cfg = fw.WeatherTrustConfig()
    y0 = np.array([1000.0])
    one = np.ones((1, 6))
    ad = 2.0 * one
    adx = 2.0 * one
    adx[0, 4] = 1.0  # case 4: the exact derivative is wrong at consistent small steps
    small = [2.0 * one, 2.0 * one, 2.0 * one]
    for f in small:
        f[0, 2] = f[0, 3] = 0.0  # cases 2, 3: inconsistent small steps (a quantum inside)
    small[0][0, 2] = small[0][0, 3] = 5.0
    small[1][0, 5] = small[2][0, 5] = small[0][0, 5] = 0.0  # case 5: everything zero ...
    large = [2.0 * one, 2.0 * one, 2.0 * one]
    large[2][0, 1] = 2.5  # case 1: one user-step secant 25 % off (curvature): fails, never undecidable
    large[2][0, 3] = 2.5  # case 3: small undecidable and a secant off
    ad[0, 5] = adx[0, 5] = 0.0
    for f in large:
        f[0, 5] = 0.0  # ... both zero
    v = fw.day_verdict(ad, adx, small, large, y0, np.ones(6), cfg)
    assert list(v["level2"][0]) == ["pass", "pass", "undecidable", "undecidable", "fail", "pass"]
    assert list(v["level3"][0]) == [True, False, True, False, True, True]
    assert list(v["level"][0]) == [3, 2, 3, 1, 1, 3]
    assert list(v["status"][0]) == ["pass", "fail", "pass", "fail", "fail", "pass"]
    np.testing.assert_allclose(v["rel_err_large_max"][0, 1], 0.2)
    assert fw._variable_label([3, 3]) == fw.LABEL_VALIDATED
    assert fw._variable_label([3, 2, 1]) == f"{fw.LABEL_FAILS}: 2 of 3 checked days fail"
    for var in ("TMAX", "TMIN", "SRAD"):
        assert fw.TEMPERATURE_CAVEAT in fw._with_caveat(var, fw.LABEL_VALIDATED)
    for var in ("RAIN", "IRRD"):
        lab = fw._with_caveat(var, fw.LABEL_VALIDATED)
        assert fw.WATER_CAVEAT in lab and fw.TEMPERATURE_CAVEAT not in lab


def test_mulch_interception_right_derivative_at_zero_rain():
    from agrijax.processes.soil_water.bucket import kernels as K

    def run(w, mass=2000.0, mulchwat=0.0, cover=0.3):
        return K.mulch_rate(w, mulchwat, 0.0, mass, cover, 0.0, 3.5, True)

    def tangents(**kw):
        (r, dr) = jax.jvp(lambda w: run(w, **kw), (jnp.asarray(0.0),), (jnp.asarray(1.0),))
        return r, dr

    r, dr = tangents()
    assert float(r.watavl) == 0.0 and float(r.mulwatadd) == 0.0  # values as the dry branch gives them
    np.testing.assert_allclose([float(dr.watavl), float(dr.mulwatadd)], [0.7, 0.3])
    h = 1e-6  # the right-hand difference of the forward model
    np.testing.assert_allclose(float(run(h).watavl) / h, 0.7, rtol=1e-9)
    np.testing.assert_allclose(float(run(h).mulwatadd) / h, 0.3, rtol=1e-9)
    # no mulch, or a saturated mulch (deficit <= 0): all of the rain goes on, derivative 1
    for kw in ({"mass": 0.0}, {"mulchwat": 1e4}):
        _, dr = tangents(**kw)
        np.testing.assert_allclose([float(dr.watavl), float(dr.mulwatadd)], [1.0, 0.0])
    # above 0 mm nothing changes
    _r1, dr1 = jax.jvp(lambda w: run(w), (jnp.asarray(0.5),), (jnp.asarray(1.0),))
    np.testing.assert_allclose(float(dr1.watavl), 0.7)
    for mode in ("ste", "exact"):  # a derivative of the function, the same in every gradient mode
        from agrijax.core.grad import gradient_mode

        with gradient_mode(mode):
            _, dr = tangents()
        np.testing.assert_allclose(float(dr.watavl), 0.7)


def test_stage_helpers():
    st = np.array([7, 7, 8, 9, 9, 1, 1, 1])
    assert fw._stage_runs(st) == [(7, 0, 1), (8, 2, 2), (9, 3, 4), (1, 5, 7)]
    np.testing.assert_array_equal(fw._morning_stage(np.array([7, 8, 8, 9])), [7, 7, 8, 8])
    assert set(fw.STAGE_NAMES) == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}


# ------------------------------------------------------------------------------ the hook
@dataclass(frozen=True)
class _Rec:
    def replace(self, **kw: Any) -> Any:
        return replace(self, **kw)


@dataclass(frozen=True)
class _Soil(_Rec):
    rain: Any
    irrigation: Any
    tmax: Any
    other: Any


@dataclass(frozen=True)
class _Wx(_Rec):
    srad: Any
    tmax: Any
    tmin: Any
    tavg: Any
    co2: Any


@dataclass(frozen=True)
class _Crop(_Rec):
    srad: Any
    tmax: Any
    tmin: Any
    dayl: Any


def test_hook_adds_the_day_change_everywhere_the_model_reads_it():
    s_n, t_n = 2, 4
    f = {
        "soil": _Soil(jnp.ones(s_n), jnp.zeros(s_n), jnp.full(s_n, 30.0), jnp.ones((s_n, 3))),
        "spam": {"weather": _Wx(*(jnp.full(s_n, v) for v in (20.0, 30.0, 15.0, 22.0, 400.0))), "x": 1},
        "crop": _Crop(*(jnp.full(s_n, v) for v in (20.0, 30.0, 15.0, 13.0))),
    }
    delta = np.zeros((s_n, t_n, len(fw.VARIABLES)))
    delta[:, 2] = [1.0, 2.0, 3.0, 4.0, 5.0]  # SRAD TMAX TMIN RAIN IRRD on day 2
    wts = np.zeros((3, t_n, 2))
    wts[:, :, 0], wts[:, :, 1] = 0.6, 0.4
    tid = jnp.array([0, 2])
    g = fw._hook(f, 2, tid, jnp.asarray(delta), jnp.asarray(wts))
    np.testing.assert_allclose(g["soil"].rain, 5.0)
    np.testing.assert_allclose(g["soil"].irrigation, 5.0)
    np.testing.assert_allclose(g["soil"].tmax, 32.0)
    np.testing.assert_allclose(g["soil"].other, 1.0)
    w = g["spam"]["weather"]
    np.testing.assert_allclose([w.srad[0], w.tmax[0], w.tmin[0], w.co2[0]], [21.0, 32.0, 18.0, 400.0])
    np.testing.assert_allclose(w.tavg, 22.0 + 0.6 * 2.0 + 0.4 * 3.0)
    c = g["crop"]
    np.testing.assert_allclose([c.srad[1], c.tmax[1], c.tmin[1], c.dayl[1]], [21.0, 32.0, 18.0, 13.0])
    assert g["spam"]["x"] == 1
    same = fw._hook(f, 1, tid, jnp.asarray(delta), jnp.asarray(wts))  # another day: unchanged
    np.testing.assert_array_equal(same["soil"].rain, f["soil"].rain)
    np.testing.assert_array_equal(same["spam"]["weather"].tavg, f["spam"]["weather"].tavg)


# ------------------------------------------------------------------------------ the rows
def test_single_day_and_whole_season_rows():
    cfg = fw.WeatherTrustConfig(top_k=2, random_days=1, seed=3)
    t_n, n_days = 6, 9
    g0 = np.zeros((n_days, len(fw.VARIABLES)))
    g0[:t_n, fw._IX["SRAD"]] = [0.1, 5.0, 0.2, 3.0, 0.0, 0.0]
    g0[:t_n, fw._IX["RAIN"]] = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0]
    x = np.ones((n_days, len(fw.VARIABLES)))
    x[:, fw._IX["RAIN"]] = 0.0
    x[2, fw._IX["RAIN"]] = 3.0
    x[4, fw._IX["RAIN"]] = 10.0
    days = fw._check_days(g0, ["SRAD", "RAIN"], t_n, cfg)
    tops = {(v, t) for (v, t, k) in days if k == "top"}
    assert tops == {("SRAD", 1), ("SRAD", 3), ("RAIN", 4), ("RAIN", 2)}
    assert sum(1 for (_, _, k) in days if k == "random") == 2
    deltas, keys = fw._check_rows(days, x, n_days, cfg)
    assert np.all(deltas[0] == 0.0)
    for (v, t), ent in keys.items():
        for grp in ("small", "large"):
            assert [h for (_, _, h) in ent[grp]] == list(
                (cfg.small_steps if grp == "small" else cfg.steps)[v]
            )
            for ip, im, h in ent[grp]:
                assert deltas[ip][t, fw._IX[v]] == h and np.count_nonzero(deltas[ip]) == 1
                assert (im < 0) == ent["one_sided"]
                if im >= 0:
                    assert deltas[im][t, fw._IX[v]] == -h
    # one scheme per day: 3 mm of rain is below the largest step (5 mm): forward at every step, also
    # at 1 and 2 mm; 10 mm: central at every step; SRAD 1 MJ = the largest step: central
    assert keys[("RAIN", 2)]["one_sided"] and not keys[("RAIN", 4)]["one_sided"]
    assert not keys[("SRAD", 1)]["one_sided"]
    assert len(deltas) == 1 + sum(6 if e["one_sided"] else 12 for e in keys.values())
    ws = fw._whole_season_deltas(["SRAD", "TMAX", "TMIN", "RAIN"], x, cfg)
    labs = [lab for lab, _ in ws]
    assert labs == [
        "SRAD x(1 +- 5 %)",
        "TMAX +-1 degC",
        "TMIN +-1 degC",
        "RAIN x(1 +- 10 %)",
        "TMAX and TMIN +-1 degC",
    ]
    np.testing.assert_allclose(ws[0][1][:, fw._IX["SRAD"]], 0.05)
    np.testing.assert_allclose(ws[3][1][4, fw._IX["RAIN"]], 1.0)
    np.testing.assert_allclose(ws[4][1][:, [fw._IX["TMAX"], fw._IX["TMIN"]]], 1.0)


# ------------------------------------------------------------------------------ the pipeline on a toy
T_RUN, T_SIM, MAT, SILK = 20, 26, 17, 9
_RNG = np.random.default_rng(7)
WEATHER = np.zeros((T_SIM, len(fw.VARIABLES)))
WEATHER[:, fw._IX["SRAD"]] = _RNG.uniform(10.0, 25.0, T_SIM)
WEATHER[:, fw._IX["TMIN"]] = _RNG.uniform(10.0, 18.0, T_SIM)
WEATHER[:, fw._IX["TMAX"]] = WEATHER[:, fw._IX["TMIN"]] + _RNG.uniform(5.0, 12.0, T_SIM)
WEATHER[:, fw._IX["RAIN"]] = np.where(_RNG.uniform(size=T_SIM) < 0.4, _RNG.uniform(1.0, 30.0, T_SIM), 0.0)
C = _RNG.uniform(0.5, 2.0, (T_SIM, len(fw.VARIABLES)))
C[MAT + 1 :] = 0.0  # nothing after maturity counts


class _ToyPrograms:
    """Outputs of a toy season: HWAM linear in the weather, CWAM quadratic; maturity a day earlier per
    whole degree of season warming (a 'phenology' the gradient does not see)."""

    def __init__(self) -> None:
        self.n_days = T_SIM
        self.many = SimpleNamespace(ndev=1)

    def weather_of(self, i: int) -> np.ndarray:
        return WEATHER

    def values(self, delta: np.ndarray, scn: np.ndarray, which: str = "many") -> tuple[np.ndarray, float]:
        w = WEATHER[None] + np.asarray(delta)
        lin = np.sum(C[None] * w, axis=(1, 2))
        quad = np.sum(C[None] * w**2, axis=(1, 2)) / 100.0
        warm = np.mean(np.asarray(delta)[:, :, fw._IX["TMAX"]], axis=1)
        mat = MAT - np.floor(warm)
        return np.stack([lin, quad, mat, np.full_like(mat, SILK)], axis=1), 0.0

    def gradient(self) -> tuple[np.ndarray, np.ndarray]:
        return np.stack([C, 2.0 * C * WEATHER / 100.0]), np.stack([C, 2.0 * C * WEATHER / 100.0])


class _ToyRun:
    days = np.asarray(
        [int((dt.date(1982, 3, 1) + dt.timedelta(days=t)).strftime("%Y%j")) for t in range(T_RUN)]
    )
    n_days = T_RUN
    params_crop = SimpleNamespace(yrplt=int(days[2]))

    def simulate(self, pad_days: int = 0) -> dict[str, np.ndarray]:
        n = T_RUN + pad_days
        ist = np.full(n, 10)
        ist[:2], ist[2:4], ist[4:9], ist[9:13], ist[13:MAT] = 7, 8, 1, 4, 5
        return {"istage": ist[:, None]}


@pytest.fixture(scope="module")
def toy():
    progs = _ToyPrograms()
    g, gx = progs.gradient()
    cfg = fw.WeatherTrustConfig(top_k=4, random_days=3)
    vs = ["SRAD", "TMAX", "TMIN", "RAIN"]
    ws = fw._analyse(progs, 0, _ToyRun(), "toy", ["HWAM", "CWAM"], vs, g, gx, 0.5, 0.5, cfg, True, [])  # type: ignore[arg-type]
    return ws, progs


def test_toy_calendar_and_stages(toy):
    ws, _ = toy
    assert ws.dates["days"] == MAT + 1 and ws.dates["matured"]
    assert ws.dates["planting"] == dt.date(1982, 3, 3)
    d = ws.daily
    assert len(d) == 2 * (MAT + 1) * 4
    h = d[(d["output"] == "HWAM") & (d["variable"] == "SRAD")]
    np.testing.assert_allclose(h["derivative"], C[: MAT + 1, fw._IX["SRAD"]])
    np.testing.assert_allclose(h["value"], WEATHER[: MAT + 1, fw._IX["SRAD"]])
    assert list(h["dap"])[:3] == [-2, -1, 0]
    assert list(h["stage"])[:4] == [7, 7, 7, 8]  # the morning stage
    st = ws.stages[(ws.stages["output"] == "HWAM") & (ws.stages["variable"] == "SRAD")]
    np.testing.assert_allclose(st["sum"].sum(), C[: MAT + 1, fw._IX["SRAD"]].sum())
    np.testing.assert_allclose(st["share"].sum(), 1.0)
    assert ws.values["HWAM"] == pytest.approx(float(np.sum(C * WEATHER)))
    cal = ws.calendar("HWAM")
    assert list(cal.columns) == ["SRAD", "TMAX", "TMIN", "RAIN"] and len(cal) == MAT + 1


def test_toy_checks(toy):
    ws, _ = toy
    sd = ws.checks.single_day
    lin = sd[sd["output"] == "HWAM"]
    assert set(lin["status"]) == {"pass"}
    assert set(lin["level"]) == {3} and set(lin["level2"]) == {"pass"}
    quad = sd[sd["output"] == "CWAM"]
    # central differences of a quadratic are exact; forward ones carry h / 2 of curvature: at the user
    # steps that is a level-3 failure (level 2 passes: at 1e-3..1e-5 mm the curvature is negligible)
    assert set(quad.loc[~quad["one_sided"], "level"]) == {3}
    fwd = quad[quad["one_sided"]]
    assert len(fwd) and set(fwd["level"]) == {2} and set(fwd["level2"]) == {"pass"}
    assert set(fwd["status"]) == {"fail"}
    assert ws.trust["SRAD"] == f"{fw.LABEL_VALIDATED} ({fw.TEMPERATURE_CAVEAT})"
    n_rain = int((sd["variable"] == "RAIN").sum())
    assert (
        ws.trust["RAIN"] == f"{fw.LABEL_FAILS}: {len(fwd)} of {n_rain} checked days fail ({fw.WATER_CAVEAT})"
    )
    assert ws.trust["TMAX"].startswith(fw.LABEL_VALIDATED) and fw.TEMPERATURE_CAVEAT in ws.trust["TMAX"]
    s = ws.checks.summary.set_index(["output", "variable"])
    assert int(s["days"].sum()) == len(sd)
    np.testing.assert_allclose(s[["level3", "level2", "level1"]].sum(axis=1), 1.0)
    assert s.loc[("HWAM", "RAIN"), "level3"] == 1.0
    np.testing.assert_allclose(s.loc[("CWAM", "RAIN"), "level2"], len(fwd) / s.loc[("CWAM", "RAIN"), "days"])
    whole = ws.checks.whole_season
    h = whole[whole["output"] == "HWAM"].set_index("perturbation")
    np.testing.assert_allclose(h["gap"], 0.0, atol=1e-9)
    np.testing.assert_allclose(h["curvature"], 0.0, atol=1e-9)
    assert h.loc["TMAX +-1 degC", "mdat_shift_plus"] == -1  # the toy's 'phenology'
    assert h.loc["SRAD x(1 +- 5 %)", "mdat_shift_plus"] == 0
    q = whole[whole["output"] == "CWAM"].set_index("perturbation")
    np.testing.assert_allclose(q["gap"], 0.0, atol=1e-9)  # central difference of a quadratic
    assert np.all(q["curvature"] > 0.0)
    assert ws.daily["checked"].isin(["", "pass", "fail"]).all()
    assert "temperature derivatives" in str(ws)


def test_toy_companions(toy):
    ws, _ = toy
    pf = ws.phenology_free([-1.0, 1.0])
    h = pf[pf["output"] == "HWAM"]
    np.testing.assert_allclose(h["gap"], 0.0, atol=1e-9)
    assert list(h["mdat_shift"]) == [1, -1]
    bf = ws.brute_force(["SRAD", "RAIN"])  # the same check on every day
    n_fwd = int((WEATHER[: MAT + 1, fw._IX["RAIN"]] < 5.0).sum())
    assert bf.attrs["rows"] == 1 + 12 * (MAT + 1) + 12 * (MAT + 1 - n_fwd) + 6 * n_fwd
    assert len(bf) == 2 * 2 * (MAT + 1) and set(bf.loc[bf["output"] == "HWAM", "level"]) == {3}
    sh = bf.attrs["shares"].set_index(["output", "variable"])
    np.testing.assert_allclose(sh.loc[("CWAM", "RAIN"), "level3"], 1.0 - n_fwd / (MAT + 1))
    same = ws.checks.single_day.merge(bf, on=["output", "variable", "day"], suffixes=("", "_bf"))
    assert (same["level"] == same["level_bf"]).all()  # the check and the brute force agree day by day
    plain = ws.brute_force(["SRAD", "RAIN"], check=False)
    assert plain.attrs["rows"] == 1 + 2 * (MAT + 1)
    lin = plain[plain["output"] == "HWAM"]
    np.testing.assert_allclose(lin["fd"], lin["derivative"], rtol=1e-6)
    clim = pd.DataFrame(
        {v: np.full(366, m) for v, m in (("SRAD", 18.0), ("TMAX", 27.0), ("TMIN", 14.0), ("RAIN", 3.0))},
        index=pd.RangeIndex(1, 367, name="doy"),
    )
    att = ws.attribution(clim)
    tot = att.total[att.total["output"] == "HWAM"].set_index("variable")
    np.testing.assert_allclose(tot["linear"], tot["rerun"], rtol=1e-9, atol=1e-9)  # linear output
    np.testing.assert_allclose(
        tot.loc["ALL", "linear"], tot.loc[["SRAD", "TMAX", "TMIN", "RAIN"], "linear"].sum()
    )
    a = att.daily[(att.daily["output"] == "HWAM") & (att.daily["variable"] == "SRAD")]
    np.testing.assert_allclose(a["anomaly"], WEATHER[: MAT + 1, fw._IX["SRAD"]] - 18.0)
    for frame in (att.daily, att.by_stage, att.total):
        assert frame["caveat"].str.contains(fw.LINEAR_CAVEAT, regex=False).all()
    assert (
        fw.TEMPERATURE_CAVEAT in tot.loc["TMAX", "caveat"]
        and fw.WATER_CAVEAT not in tot.loc["TMAX", "caveat"]
    )
    assert (
        fw.WATER_CAVEAT in tot.loc["RAIN", "caveat"]
        and fw.TEMPERATURE_CAVEAT not in tot.loc["RAIN", "caveat"]
    )
    assert fw.WATER_CAVEAT in tot.loc["ALL", "caveat"] and fw.TEMPERATURE_CAVEAT in tot.loc["ALL", "caveat"]
    yrdoy = ws._ctx.yrdoy[: MAT + 1]
    an = pd.DataFrame({v: np.ones(MAT + 1) for v in ("SRAD", "TMAX", "TMIN", "RAIN")}, index=yrdoy)
    att2 = ws.attribution(anomaly=an, rerun=False)
    t2 = att2.total[att2.total["output"] == "HWAM"].set_index("variable")
    np.testing.assert_allclose(t2.loc["SRAD", "linear"], C[: MAT + 1, fw._IX["SRAD"]].sum())
    with pytest.raises(ValueError, match="no weather files"):
        ws.attribution()


# ------------------------------------------------------------------------------ climatology, facade
SITE = {
    "INSI": "TEST",
    "LAT": 30.0,
    "LONG": -82.0,
    "ELEV": 10.0,
    "TAV": 20.0,
    "AMP": 10.0,
    "REFHT": 2.0,
    "WNDHT": 2.0,
}


def test_climatology_of_weather_files(tmp_path):
    from agrijax.io.dssat.wth import write_wth

    files = []
    for y, off in ((1999, 0.0), (2000, 2.0)):
        dates = pd.date_range(f"{y}-01-01", f"{y}-12-31")
        df = pd.DataFrame(
            {
                "date": dates,
                "srad": 10.0 + off,
                "tmax": 25.0 + off,
                "tmin": 12.0,
                "rain": np.where(np.arange(len(dates)) % 2 == 0, 4.0, 0.0) + off,
            }
        )
        files.append(write_wth(df, tmp_path / f"TEST{y % 100:02d}01.WTH", site=SITE))
    tab = fw.climatology(files)
    assert tab.attrs["years"] == [1999, 2000]
    assert list(tab.index) == list(range(1, 367))
    np.testing.assert_allclose(tab.loc[100, ["SRAD", "TMAX", "TMIN"]], [11.0, 26.0, 12.0])
    np.testing.assert_allclose(tab.loc[60, "SRAD"], 12.0)  # Feb 29: only 2000
    one = fw.climatology(files, years=[1999])
    np.testing.assert_allclose(one["SRAD"], 10.0)
    assert fw.station_files(tmp_path, "TEST") == sorted(files)
    with pytest.raises(ValueError, match="no weather"):
        fw.climatology(files, years=[1950])


def test_attribution_leaves_the_season_year_out(toy, tmp_path):
    from agrijax.io.dssat.wth import write_wth

    ws, _ = toy
    files = []
    for y, srad in ((1981, 10.0), (1982, 30.0), (1983, 14.0)):  # the toy season is 1982
        dates = pd.date_range(f"{y}-01-01", f"{y}-12-31")
        df = pd.DataFrame({"date": dates, "srad": srad, "tmax": 25.0, "tmin": 12.0, "rain": 1.0})
        files.append(write_wth(df, tmp_path / f"TEST{y % 100:02d}01.WTH", site=SITE))
    np.testing.assert_allclose(fw.climatology(files, exclude=[1982])["SRAD"], 12.0)
    ctx = ws._ctx
    old = ctx.weather_files
    try:
        ctx.weather_files = files
        att = ws.attribution(rerun=False)
        assert att.years == [1981, 1983]
        a = att.daily[(att.daily["output"] == "HWAM") & (att.daily["variable"] == "SRAD")]
        np.testing.assert_allclose(a["climatology"], 12.0)
        assert ws.attribution(rerun=False, leave_season_out=False).years == [1981, 1982, 1983]
    finally:
        ctx.weather_files = old


def test_facade_reuses_a_scenario_and_its_programs(monkeypatch):
    calls = {"scenarios": 0, "ws": []}
    exp = ajd.Experiment(
        "TEST", ajd.Path("/nonexistent/TEST.MZX"), ajd.Path("/nonexistent"), "", {2: "t"}, {}, "", "XX"
    )
    run = SimpleNamespace(name="run")

    def scenarios(self, trno, years=None, sowing_shift=(0,), soil_evaporation=None):
        calls["scenarios"] += 1
        return SimpleNamespace(runs=[run], table=pd.DataFrame({"year": [1985]}))

    def fake_ws(runs, names, **kw):
        calls["ws"].append((runs, names, kw["owner"], kw["key"]))
        return ["result"]

    monkeypatch.setattr(ajd.Experiment, "scenarios", scenarios)
    monkeypatch.setattr(fw, "weather_sensitivity", fake_ws)
    for _ in range(2):
        assert exp.weather_sensitivity(2, year=1985) == "result"
    assert calls["scenarios"] == 1  # the scenario's inputs are built once
    (r1, n1, o1, k1), (r2, n2, o2, k2) = calls["ws"]
    assert o1 is exp and o2 is exp and k1 == k2  # the programs are cached on the experiment by content
    assert r1[0] is run and r2[0] is run and n1 == n2 == ["TEST_t02 1985 +0 d"]
    exp.weather_sensitivity(2, year=1985, sowing_shift=7)
    assert calls["scenarios"] == 2 and calls["ws"][-1][3] != k1


def test_cache_builds_once_per_owner_and_key():
    class Owner:
        pass

    built = []
    own = Owner()
    for _ in range(2):
        fw._cached(own, ("k", 1), lambda: built.append(1) or "progs")  # type: ignore[arg-type,return-value]
    assert len(built) == 1
    fw._cached(own, ("k", 2), lambda: built.append(1) or "progs")  # type: ignore[arg-type,return-value]
    assert len(built) == 2


def test_variables_parsing_and_facade_signatures():
    assert fw._parse_variables(["srad", "RAIN"]) == ["SRAD", "RAIN"]
    for bad in ([], ["SRAD", "SRAD"], ["WIND"]):
        with pytest.raises(ValueError, match="variables"):
            fw._parse_variables(bad)
    sig = inspect.signature(ajd.Experiment.weather_sensitivity)
    assert list(sig.parameters)[:4] == ["self", "treatment", "outputs", "variables"]
    assert sig.parameters["variables"].default == ("SRAD", "TMAX", "TMIN", "RAIN")
    assert {"year", "sowing_shift", "check", "config"} <= set(sig.parameters)
    assert "variables" in inspect.signature(ajd.Scenarios.weather_sensitivity).parameters
    res = fw.WeatherSensitivities(["a", "b", "c"])
    res.years = [1979, 1982, 1982]
    assert res[0] == "a" and res[1979] == "a"
    with pytest.raises(KeyError, match="1982"):
        res[1982]
