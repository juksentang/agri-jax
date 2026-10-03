"""``exp.weather_sensitivity`` on DSSAT's UFGA8201 example, rainfed treatment 2 (slow, needs the DSSAT
example data; no DSSAT run).

* the unperturbed row is bit for bit the calibration's program (``DaySimulator`` without the hook) and
  the season of ``Experiment.run`` (the single-season program);
* the perturbation reaches the model where an independent construction puts it: a day's ``SRAD``,
  ``TMAX`` (with ``TAVG`` through ``HMET``'s weights) or ``RAIN`` changed in the treatment's inputs
  (``FreeRunInputs.series`` and the crop weather) and simulated by the single-season program gives the
  batched rerun's yield;
* on grain-filling days chosen by stage (not by the check), below the 35 degC threshold, the
  temperature derivative equals the central difference of those independent runs, and equals the
  exact-mode derivative;
* the calendar, the stage table, the labels (with their caveats) and level shares, the whole-season
  check (radiation and temperature move the stage days), the attribution's structure (leave-one-out
  climatology, the caveat on every row);
* a second ``year=`` call compiles nothing.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

from agrijax import facade_weather as fw
from agrijax.port.run_fortran import DSSAT_ENGINE

pytestmark = [
    pytest.mark.slow,
    pytest.mark.allow_skip(reason="needs the DSSAT example data (AGRI_JAX_DSSAT)"),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the DSSAT day runs in float64"),
]

TRNO = 2
OUTPUTS = ["HWAM", "CWAM"]
VARIABLES = ["SRAD", "TMAX", "TMIN", "RAIN"]
#: batched rerun against the single-season program (two programs, the same arithmetic)
PROGRAM_RTOL = 1e-9


@pytest.fixture(scope="module")
def exp():
    if not (DSSAT_ENGINE / "example_data" / "Maize" / "UFGA8201.MZX").is_file():
        pytest.skip(f"DSSAT example data not found under {DSSAT_ENGINE}")
    import agrijax as aj

    return aj.dssat.experiment("UFGA8201", data_root=DSSAT_ENGINE)


@pytest.fixture(scope="module")
def ws(exp):
    return exp.weather_sensitivity(TRNO, outputs=OUTPUTS, variables=VARIABLES)


def _perturbed(x, t: int, var: str, h: float):
    """The treatment's inputs with day ``t``'s ``var`` changed by ``h``, built independently of the hook."""
    from agrijax.forcing.dssat_weather import hourly_mean_temperature_weights

    s = {k: np.array(v, dtype=float, copy=True) for k, v in x.series.items()}
    c = x.forcing_crop
    if var == "SRAD":
        s["srad"][t] += h
        c = c.replace(srad=c.srad.at[t].add(h))
    elif var == "RAIN":
        s["rain"][t] += h
    else:
        wx, wn = hourly_mean_temperature_weights(np.asarray(c.dayl, dtype=float))
        w = wx if var == "TMAX" else wn
        s["tavg"][t] += w[t] * h
        if var == "TMAX":
            s["tmax"][t] += h
            s["tmax_w"][t] += h
            c = c.replace(tmax=c.tmax.at[t].add(h))
        else:
            s["tmin_w"][t] += h
            c = c.replace(tmin=c.tmin.at[t].add(h))
    return dataclasses.replace(x, series=s, forcing_crop=c)


def _yield(x) -> float:
    o = x.simulate()
    ist = np.asarray(o["istage"])[:, 0]
    hit = np.nonzero(ist == 10)[0]
    t = int(hit[0]) if hit.size else ist.size - 1
    return float(np.asarray(o["gwad"])[t, 0])


def test_the_unperturbed_row_is_the_calibration_program_and_the_season(exp, ws):
    from agrijax.calib.dssat_day import DaySimulator

    progs = ws._ctx.progs
    z = np.zeros((1, progs.n_days, len(fw.VARIABLES)))
    y_hook, _ = progs.values(z, np.zeros(1, np.int64))
    sim = DaySimulator(progs.runs, [list(e) for e in progs._entries], pad_days=fw.PAD_DAYS)
    y_cal = sim(progs.theta[:1], np.zeros(1, np.int64))
    assert np.array_equal(y_hook, y_cal)  # bit for bit: the hook adds zeros
    season = exp.run(TRNO)
    assert ws.values["HWAM"] == pytest.approx(season.summary["HWAM"], rel=1e-9)
    assert ws.values["CWAM"] == pytest.approx(season.summary["CWAM"], rel=1e-9)
    assert ws.dates["maturity"].strftime("%Y%j") == str(int(season.summary["MDAT"]))


def test_the_perturbation_reaches_the_model_where_an_independent_construction_puts_it(exp, ws):
    x = exp.inputs(TRNO)
    progs = ws._ctx.progs
    sd = ws.checks.single_day
    for var, h in (("SRAD", 1.0), ("TMAX", 1.0), ("TMIN", 1.0), ("RAIN", 5.0)):
        t = int(sd[(sd["variable"] == var) & (sd["selected"] == "top")]["day"].iloc[0])
        d = np.zeros((1, progs.n_days, len(fw.VARIABLES)))
        d[0, t, fw._IX[var]] = h
        y, _ = progs.values(d, np.zeros(1, np.int64))
        assert float(y[0, 0]) == pytest.approx(_yield(_perturbed(x, t, var, h)), rel=PROGRAM_RTOL), (var, t)


def test_grain_filling_temperature_derivative_equals_independent_differences(exp, ws):
    x = exp.inputs(TRNO)
    d = ws.daily[(ws.daily["output"] == "HWAM") & (ws.daily["variable"] == "TMAX")]
    fill = d[(d["stage"] == 5) & (d["value"] < 34.0) & (d["derivative"] != 0.0)].head(2)
    assert len(fill) == 2
    for _, r in fill.iterrows():
        t, h = int(r["day"]), 0.1
        fd = (_yield(_perturbed(x, t, "TMAX", h)) - _yield(_perturbed(x, t, "TMAX", -h))) / (2 * h)
        assert r["derivative"] == pytest.approx(fd, rel=1e-3), (t, r["derivative"], fd)
        assert r["derivative"] == pytest.approx(r["derivative_exact"], rel=1e-9)


def test_calendar_stages_labels_and_checks(ws):
    t_cal = ws.dates["days"]
    assert len(ws.daily) == len(OUTPUTS) * t_cal * len(VARIABLES)
    assert list(ws.trust) == VARIABLES
    for v, lab in ws.trust.items():
        assert lab.startswith((fw.LABEL_VALIDATED, fw.LABEL_FAILS))
        assert (fw.TEMPERATURE_CAVEAT in lab) == (v in fw.PHENOLOGY)
        assert (fw.WATER_CAVEAT in lab) == (v in fw.WATER)
    summ = ws.checks.summary
    np.testing.assert_allclose(summ[["level3", "level2", "level1"]].sum(axis=1), 1.0)
    for (o, v), g in ws.daily.groupby(["output", "variable"]):
        st = ws.stages[(ws.stages["output"] == o) & (ws.stages["variable"] == v)]
        assert st["sum"].sum() == pytest.approx(g["derivative"].sum(), rel=1e-9, abs=1e-12)
        assert st["days"].sum() == t_cal
    # after maturity nothing counts; before planting the soil water does (rain)
    assert set(ws.stages["stage"]) <= set(fw.STAGE_NAMES)
    whole = ws.checks.whole_season.set_index(["output", "perturbation"])
    srad = whole.loc[("HWAM", "SRAD x(1 +- 5 %)")]
    assert srad["adat_shift_plus"] < 0  # radiation moves the early stage days (growing-point temperature)
    temp = whole.loc[("HWAM", "TMAX and TMIN +-1 degC")]
    assert temp["mdat_shift_plus"] < 0 < temp["mdat_shift_minus"]  # warmer: earlier maturity
    sd = ws.checks.single_day
    assert len(sd) and set(sd["status"]) <= {"pass", "fail"} and set(sd["level"]) <= {1, 2, 3}
    att = ws.attribution()
    assert len(att.years) > 10 and 1982 not in att.years  # leave-one-out
    for frame in (att.daily, att.by_stage, att.total):
        assert frame["caveat"].str.contains(fw.LINEAR_CAVEAT, regex=False).all()
    tot = att.total[att.total["output"] == "HWAM"].set_index("variable")
    assert list(tot.index) == [*VARIABLES, "ALL"]
    assert np.isfinite(tot["linear"]).all() and np.isfinite(tot["rerun"]).all()
    pf = ws.phenology_free([-1.0, 1.0])
    assert list(pf[pf["output"] == "HWAM"]["mdat_shift"]) == [
        int(temp["mdat_shift_minus"]),
        int(temp["mdat_shift_plus"]),
    ]
    print("\n" + str(ws))


def test_a_second_year_call_compiles_nothing(exp):
    first = exp.weather_sensitivity(TRNO, outputs=["HWAM"], variables=["SRAD"], year=1985, check=False)
    compiled = first._ctx.progs.compile_s
    again = exp.weather_sensitivity(TRNO, outputs=["HWAM"], variables=["SRAD"], year=1985, check=False)
    assert again._ctx.progs is first._ctx.progs
    assert again._ctx.progs.compile_s == compiled and again.timing["compile_s"] == 0.0
    np.testing.assert_array_equal(again.daily["derivative"], first.daily["derivative"])
