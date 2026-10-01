"""The one-call cultivar calibration (:mod:`agrijax.calib.fit`, :mod:`agrijax.calib.workflow`) on a
synthetic twin: a small analytic stand-in for the crop with the same interface as the free-run day
simulator (six ``.CUL`` coefficients in, entry values out), no data and no DSSAT.

The twin: silking day ``floor(10 + P1 / 10 + PHINT / 5)``, maturity ``silking + floor(P5 / 12)``
(piecewise-constant dates, as the real model's), grain number ``4 G2 env``, yield
``grain number x G3 x grain-fill days / 400`` (G2 and G3 trade off unless the grain number is
observed), tops ``2 yield + 60 PHINT``, LAI ``4 env (1 - exp(-t PHINT / 2000))``; P2 has no effect.
The end-to-end run on a DSSAT experiment (``dscsm048``, the free-run tables) is the slow integration
test ``tests/integration/test_calib_workflow_dssat.py``.
"""

from __future__ import annotations

import json
import types
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from agrijax.calib.dssat_day import CUL_ORDER, E_DATE, E_DAY, E_FINAL, OUT_NAMES, Entry, entry_values
from agrijax.calib.fit import (
    PROBES,
    CalibrationWarning,
    CultivarProblem,
    Observed,
    Treatment,
    _Evaluator,
    _Objective,
    fit_cultivar,
)
from agrijax.calib.workflow import (
    NITROGEN_DATES_DIFFER,
    NITROGEN_STRESS,
    NITROGEN_STRESS_MAX,
    NITROGEN_STRESS_OK,
    OUT_OF_SCOPE,
    CalibrationResult,
    ObservationError,
    ScopeError,
    _nitrogen_warning,
    _observations,
    _parse_treatments,
    _scope_reason,
)
from agrijax.io.dssat.cultivar_write import cul_written
from agrijax.io.dssat.observed import read_observed
from agrijax.sites.dssat_inputs import yrdoy_range

#: the decimal convention of MZCER048.CUL (its MINIMA row, DSSAT-CSM v4.8.6.0 Genotype/MZCER048.CUL:51)
DEC = {"P1": 1, "P2": 3, "P5": 1, "G2": 1, "G3": 2, "PHINT": 2}
#: IB0035 (MZCER048.CUL): the twin's truth
TRUTH = np.asarray([259.0, 1.193, 947.1, 924.3, 8.17, 43.0])
#: the twin's "published" cultivar (the start), away from the truth
PUBLISHED = np.asarray([280.0, 1.0, 900.0, 800.0, 9.0, 46.0])
LAI_DAYS = (20, 40, 60, 80)
I_LAI, I_GWAD, I_CWAD, I_GAD = (OUT_NAMES.index(n) for n in ("lai", "gwad", "cwad", "g_ad"))


def needs_x64(fn):
    """``calibrate`` runs in float64 only (the CI float32 pass skips these)."""
    fn = pytest.mark.skipif(not jax.config.jax_enable_x64, reason="calibrate runs in float64 only")(fn)
    return pytest.mark.allow_skip(reason="calibrate is float64-only (CI float32 pass)")(fn)


def twin_values(xp, theta, kind, out, t, env):
    """Entry values of the twin (module docstring); ``xp`` is numpy or jax.numpy, arrays broadcast."""
    p1, p5, g2, g3, phint = (
        theta[..., 0:1],
        theta[..., 2:3],
        theta[..., 3:4],
        theta[..., 4:5],
        theta[..., 5:6],
    )
    silk = xp.floor(10.0 + p1 / 10.0 + phint / 5.0)
    mat = silk + xp.floor(p5 / 12.0)
    gn = 4.0 * g2 * env
    hw = gn * g3 * (mat - silk) / 400.0
    cw = 2.0 * hw + 60.0 * phint
    lai = 4.0 * env * (1.0 - xp.exp(-t * phint / 2000.0))
    date = xp.where(out == 4, silk, xp.where(out == 10, mat, 5.0))
    final = xp.where(out == I_GWAD, hw, xp.where(out == I_CWAD, cw, xp.where(out == I_GAD, gn, 0.0)))
    day = xp.where(out == I_LAI, lai, 0.0)
    return xp.where(kind == E_DATE, date, xp.where(kind == E_FINAL, final, xp.where(kind == E_DAY, day, 0.0)))


def make_problem(
    codes=("ADAT", "MDAT", "HWAM", "CWAM", "H#AM", "LAID"),
    envs=(1.0, 0.8, 0.9),
    written=None,
):
    days = np.asarray(yrdoy_range(1982056, 1982056 + 199), dtype=np.int64)
    obs_entry = {
        "ADAT": [Entry(E_DATE, 4)],
        "MDAT": [Entry(E_DATE, 10)],
        "HWAM": [Entry(E_FINAL, I_GWAD)],
        "CWAM": [Entry(E_FINAL, I_CWAD)],
        "H#AM": [Entry(E_FINAL, I_GAD)],
        "LAID": [Entry(E_DAY, I_LAI, d) for d in LAI_DAYS],
    }
    treatments, rows = [], []
    for b in range(len(envs)):
        ent = [e for _, e in PROBES]
        codes_b = []
        for c in codes:
            for e in obs_entry[c]:
                codes_b.append((c, len(ent)))
                ent.append(e)
        treatments.append(Treatment(f"TWIN0001_t{b + 1:02d}", days, tuple(ent)))
        rows.append(codes_b)
    e_n = max(len(t.entries) for t in treatments)
    tab = {k: np.zeros((len(envs), e_n)) for k in ("kind", "out", "t")}
    for b, tr in enumerate(treatments):
        for j, e in enumerate(tr.entries):
            tab["kind"][b, j], tab["out"][b, j], tab["t"][b, j] = e.kind, e.out, e.t
    env_arr = np.asarray(envs)

    def simulate(theta, tid):
        return twin_values(
            np, np.asarray(theta), tab["kind"][tid], tab["out"][tid], tab["t"][tid], env_arr[tid][:, None]
        )

    y_true = simulate(np.repeat(TRUTH[None], len(envs), axis=0), np.arange(len(envs)))
    observed = []
    for b, codes_b in enumerate(rows):
        for c, j in codes_b:
            e = treatments[b].entries[j]
            day = (
                int(days[int(y_true[b, j])])
                if e.kind == E_DATE
                else (int(days[e.t]) if e.kind == E_DAY else -1)
            )
            observed.append(Observed(b, c, j, float(y_true[b, j]), day))

    def jax_loss(obs, winv):
        k, o, t = (jnp.asarray(tab[n]) for n in ("kind", "out", "t"))
        env = jnp.asarray(env_arr)[:, None]
        ob, w = jnp.asarray(obs), jnp.asarray(winv)

        def f(theta):
            y = twin_values(jnp, jnp.asarray(theta)[None], k, o, t, env)
            return jnp.sum(((y - ob) * w) ** 2, axis=1)

        return f

    def written_cul(theta):
        return np.asarray(
            [
                [cul_written(dict(zip(CUL_ORDER, r, strict=True)), decimals=DEC)[n] for n in CUL_ORDER]
                for r in np.asarray(theta).tolist()
            ]
        )

    return CultivarProblem(
        simulate=simulate,
        treatments=treatments,
        observed=observed,
        published=PUBLISHED.copy(),
        written=written or written_cul,
        jax_loss=jax_loss,
    )


def _loss_at(problem, cal, theta):
    (loss,) = _Evaluator(problem, cal).losses(np.asarray(theta)[None], _Objective(problem, cal))
    return float(loss[0])


# ------------------------------------------------------------------ the core
def test_entry_values_host_matches_definition():
    ist = np.asarray([0, 1, 1, 2, 4, 4, 5, 10, 10])
    lai = np.arange(9.0)
    outs = {n: np.zeros(9) for n in OUT_NAMES} | {"istage": ist, "lai": lai, "gwad": lai * 10}
    ents = [
        Entry(E_DATE, 4),
        Entry(E_DATE, 10),
        Entry(E_DATE, 3),
        Entry(E_FINAL, I_GWAD),
        Entry(E_DAY, I_LAI, 5),
    ]
    # silking on index 4, maturity 7, a stage never reached -> the season length; yield at maturity
    np.testing.assert_array_equal(entry_values(outs, ents), [4.0, 7.0, 9.0, 70.0, 5.0])


def test_cma_improves_on_published_and_returns_written_coefficients():
    pb = make_problem()
    cal = range(3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CalibrationWarning)  # no G2/G3 warning: grain number observed
        with pytest.raises(CalibrationWarning, match="P2"):
            fit_cultivar(pb, cal, starts=4, seed=0, budget=300)
    with pytest.warns(CalibrationWarning, match="fixed at the published values"):
        res = fit_cultivar(pb, cal, starts=4, seed=0, budget=1500)
    # P2 moves nothing: fixed, reported, at the published value
    assert res.sensitivity["P2"]["inert"] and "P2" not in res.free and "inert" in res.fixed["P2"]
    assert res.theta_written[CUL_ORDER.index("P2")] == PUBLISHED[CUL_ORDER.index("P2")]
    assert set(res.free) == {"P1", "P5", "G2", "G3", "PHINT"}
    assert res.loss["written"] < 0.05 * res.loss["published"]
    # the returned coefficients are .CUL values, and the reported loss is the loss at them
    w = dict(zip(CUL_ORDER, res.theta_written.tolist(), strict=True))
    assert cul_written(w, decimals=DEC) == w
    assert res.loss["written"] == pytest.approx(_loss_at(pb, cal, res.theta_written), rel=1e-12)
    assert res.loss["written"] == min(p["loss_written"] for p in res.per_start)
    assert res.loss["published"] == pytest.approx(_loss_at(pb, cal, PUBLISHED), rel=1e-12)
    # with the grain number observed, G2 and G3 come back near the truth
    for n in ("G2", "G3"):
        i = CUL_ORDER.index(n)
        assert abs(res.theta_written[i] - TRUTH[i]) / TRUTH[i] < 0.05, (n, res.theta_written)
    # the fit table: one row per observation, dates as YYYYDDD
    assert len(res.fit) == len(pb.observed)
    adat = [r for r in res.fit if r["code"] == "ADAT"]
    assert all(r["kind"] == "date" and r["observed"] > 1982000 for r in adat)
    assert res.calls["candidates"] > 0 and res.method == "cma"
    assert not any("grain number" in w for w in res.warnings)


def test_selection_is_on_the_written_values_not_the_unrounded_ones():
    """A 'written' map that moves P1 by 15 degC d (two silking days): the unrounded optimum is no
    longer the best once written; the result is chosen and reported at the written values."""
    shift = np.zeros(6)
    shift[0] = -15.0
    pb = make_problem(written=lambda th: np.asarray(th) + shift)
    with pytest.warns(CalibrationWarning):
        res = fit_cultivar(pb, range(3), starts=4, seed=1, budget=600)
    assert res.loss["written"] == min(p["loss_written"] for p in res.per_start)
    assert res.loss["written"] == pytest.approx(_loss_at(pb, range(3), res.theta_written), rel=1e-12)
    np.testing.assert_allclose(res.theta_written[0], res.theta_unrounded[0] - 15.0)
    assert res.loss["written"] > res.loss["unrounded"] and res.improved


def test_grain_coefficients_without_grain_number_warn():
    pb = make_problem(codes=("ADAT", "MDAT", "HWAM", "CWAM", "LAID"))
    with pytest.warns(CalibrationWarning, match="no grain number"):
        fit_cultivar(pb, range(3), starts=2, seed=0, budget=60)


def test_explicit_params_warn_when_inert_and_fix_the_rest():
    pb = make_problem()
    with pytest.warns(CalibrationWarning, match="inert or flat"):
        res = fit_cultivar(pb, range(3), params=("P2", "G2"), starts=2, seed=0, budget=60)
    assert res.free == ("P2", "G2")
    assert res.fixed["P1"] == "not requested"
    for n in ("P1", "P5", "G3", "PHINT"):
        assert res.theta_written[CUL_ORDER.index(n)] == PUBLISHED[CUL_ORDER.index(n)]
    with pytest.raises(ValueError, match="distinct names"):
        fit_cultivar(pb, range(3), params=("P9",), starts=2, budget=60)


def test_holdout_is_evaluated_not_fitted():
    pb = make_problem()
    with pytest.warns(CalibrationWarning):
        res = fit_cultivar(pb, [0, 1], holdout=[2], starts=3, seed=0, budget=900)
    h = res.holdout
    assert h is not None and h["treatments"] == ["TWIN0001_t03"]
    assert h["loss_written"] < h["loss_published"]
    assert {r["set"] for r in h["fit"]} == {"holdout"}
    assert all(r["treatment"] != "TWIN0001_t03" for r in res.fit)
    with pytest.raises(ValueError, match="both calibrated and held out"):
        fit_cultivar(pb, [0, 1], holdout=[1], starts=2, budget=60)


def test_staged_method():
    pb = make_problem()
    with pytest.warns(CalibrationWarning):
        res = fit_cultivar(pb, range(3), method="staged", starts=3, seed=0, budget=1500)
    assert res.method == "staged"
    assert res.calls["stages"]["stage1"]["params"] == ["P1", "P5", "PHINT"]
    assert res.calls["stages"]["stage2"]["params"] == ["G2", "G3"]
    assert res.loss["written"] < 0.2 * res.loss["published"]


@needs_x64
def test_adam_only_on_trusted_gradients():
    assert jax.config.jax_enable_x64
    pb = make_problem()
    with pytest.warns(CalibrationWarning, match="adam calibrates"):
        res = fit_cultivar(pb, range(3), method="adam", starts=2, seed=0, budget=150)
    # the phenology coefficients are derivative-free by default; G2 / G3 are smooth in the twin
    assert set(res.free) == {"G2", "G3"}
    for n in ("P1", "P5", "PHINT"):
        assert "not trusted" in res.fixed[n]
        assert res.theta_written[CUL_ORDER.index(n)] == PUBLISHED[CUL_ORDER.index(n)]
    assert "plan" in res.trust
    assert res.loss["written"] < res.loss["published"]


def test_bad_arguments():
    pb = make_problem()
    with pytest.raises(ValueError, match="method"):
        fit_cultivar(pb, range(3), method="lbfgs")
    with pytest.raises(ValueError, match="no calibration treatment"):
        fit_cultivar(pb, [])


# ------------------------------------------------------------------ observations -> targets (files)
def _write_observed(d: Path) -> Path:
    x = d / "TEST8201.MZX"
    x.write_text("*EXP.DETAILS: TEST8201MZ synthetic\n")
    (d / "TEST8201.MZA").write_text(
        "*EXP. DATA (A): TEST8201MZ synthetic\n\n"
        "@TRNO   HWAM  H#AM  ADAT  MDAT  HWUM\n"
        "     1  9000  3000   132   200 0.300\n"
        "     2  8000  2800   130   185 0.290\n"
    )
    (d / "TEST8201.MZT").write_text(
        "*EXP. DATA (T): TEST8201MZ synthetic\n\n"
        "@TRNO   DATE  LAID\n"
        "     1 82100  2.00\n"
        "     1 82150  3.50\n"
        "     1 82195  1.00\n"
        "     2 82100  1.80\n"
        "     2 82100  2.20\n"
    )
    return x


def _run(trno: int):
    days = np.asarray(yrdoy_range(1982056, 1982190), dtype=np.int64)
    return types.SimpleNamespace(key=f"TEST8201_t{trno:02d}", trno=trno, days=days)


def test_observations_outside_the_season_are_loud(tmp_path):
    obs = read_observed(_write_observed(tmp_path))
    # treatment 1: MDAT (day 200) after the last simulated day (190): an error, not a silent drop
    rows, errors, dropped, uns, avg = _observations(obs, _run(1), None)
    assert [e["code"] for e in errors] == ["MDAT"]
    assert errors[0]["observed"] == 1982200 and errors[0]["last_day"] == 1982190
    # the T-file LAI of day 195 lies after the season: dropped, and listed
    assert [(d["code"], d["day"]) for d in dropped] == [("LAID", 1982195)]
    assert {r[0] for r in rows} == {"HWAM", "H#AM", "ADAT", "LAID"}
    assert "HWUM" in uns
    # treatment 2: everything inside the season
    rows2, errors2, dropped2, _, avg2 = _observations(obs, _run(2), None)
    assert not errors2 and not dropped2
    # the LAI of treatment 2 observed twice on day 100: averaged, and said so
    assert not avg and avg2 and "averaged" in avg2[0]
    lai2 = [r for r in rows2 if r[0] == "LAID"]
    assert len(lai2) == 1 and lai2[0][2] == pytest.approx(2.0)
    adat = next(r for r in rows2 if r[0] == "ADAT")
    assert adat[1] == Entry(E_DATE, 4) and adat[3] == 1982130
    assert adat[2] == 1982130 - 1982056  # a day index on the simulated days
    # leaving the code out removes the error
    _, errors3, _, _, _ = _observations(obs, _run(1), ["HWAM", "ADAT"])
    assert not errors3


def test_unreadable_observed_date_is_an_observation_error(tmp_path, monkeypatch):
    """An A-file date that cannot be placed and that ``reada_date`` cannot read is reported as
    "not a readable date" (an ``ObservationError`` naming the treatment and code), never dropped.
    With the real ``reada_date`` this record is defensive: the function returns ``None`` only for
    values the preceding positive-date test already skips, so the test substitutes it."""
    import agrijax.io.dssat.observed as observed_mod

    obs = read_observed(_write_observed(tmp_path))
    monkeypatch.setattr(observed_mod, "reada_date", lambda *a, **k: None)
    # treatment 1: MDAT 200 lies after the last simulated day (190), so it is not placed as a target
    _, errors, _, _, _ = _observations(obs, _run(1), None)
    assert errors == [{"treatment": "TEST8201_t01", "code": "MDAT", "observed": 200.0, "unreadable": True}]
    # a code left out of ``targets`` raises nothing
    _, errors2, _, _, _ = _observations(obs, _run(1), ["HWAM", "ADAT"])
    assert not errors2


@needs_x64
def test_calibrate_reports_an_unreadable_date(tmp_path, monkeypatch):
    import agrijax.io.dssat.observed as observed_mod
    from agrijax.calib import calibrate

    eng, data = _fake_setup(tmp_path, monkeypatch)
    # treatment 4's observed maturity on day of year 300, after the twin's last day (255)
    mza = eng / "example_data" / "Maize" / "UFGA8201.MZA"
    o = _twin_obs(_ENV[4])
    mza.write_text(
        "*EXP. DATA (A): UFGA8201MZ synthetic\n\n@TRNO   ADAT  MDAT   HWAM   CWAM   H#AM\n"
        f"{4:6d}{int(_T0 % 1000 + o['ADAT']):6d}{300:6d}{o['HWAM']:7.1f}{o['CWAM']:7.1f}{o['H#AM']:7.1f}\n"
    )
    monkeypatch.setattr(observed_mod, "reada_date", lambda *a, **k: None)
    with pytest.raises(
        ObservationError, match=r"UFGA8201_t04: observed MDAT 300 is not a readable date"
    ) as exc:
        calibrate("UFGA8201", treatments=[4], data_dir=data, engine=eng)
    assert "nothing is dropped silently" in str(exc.value)


# ------------------------------------------------------------------ scope and arguments
def test_scope_reasons(tmp_path):
    from agrijax.sites.dssat_free_run import missing_tables

    trts = {k: "IB0035" for k in range(1, 7)}
    # measured nitrogen effects: UFGA8201 t01 +14 % (refused), t04 -0.04 % (in scope)
    assert NITROGEN_STRESS["UFGA8201_t01"] > NITROGEN_STRESS_MAX
    assert "nitrogen matters" in str(_scope_reason("UFGA8201", 1, trts, tmp_path))
    assert "growth-chamber" in str(_scope_reason("GAGR0201", 1, trts, tmp_path))
    assert "no treatment 9" in str(_scope_reason("UFGA8201", 9, trts, tmp_path))
    assert "not measured" in str(_scope_reason("TEST8201", 1, trts, tmp_path))
    msg = _scope_reason("UFGA8201", 4, trts, tmp_path)
    assert msg is not None and "input tables" in msg and "not distributed" in msg
    for p in missing_tables("UFGA8201", 4, tmp_path):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    assert _scope_reason("UFGA8201", 4, trts, tmp_path) is None
    assert all(isinstance(v, str) and v for v in OUT_OF_SCOPE.values())
    assert all(abs(v) < 1.0 for v in NITROGEN_STRESS.values())


def test_nitrogen_dates_differ_table():
    # the flagged treatments are measured ones; SIAZ9601 t07 (+0.37 % yield) changes its dates with
    # nitrogen on, so it is in scope only with a warning
    assert NITROGEN_DATES_DIFFER <= set(NITROGEN_STRESS)
    # measured with the table (dscsm048 build486, 2026-09-30; the growth chamber GAGR0201 is out of scope)
    assert NITROGEN_DATES_DIFFER == {"SIAZ9501_t03", "SIAZ9601_t07"}
    assert NITROGEN_STRESS["SIAZ9501_t03"] > NITROGEN_STRESS_MAX  # refused on the yield anyway
    assert (
        "SIAZ9601_t07" in NITROGEN_DATES_DIFFER and abs(NITROGEN_STRESS["SIAZ9601_t07"]) < NITROGEN_STRESS_OK
    )
    assert "date" in str(_nitrogen_warning("SIAZ9601_t07"))
    # every flagged treatment that is not refused gets a warning, whatever its yield change
    for k in NITROGEN_DATES_DIFFER:
        if abs(NITROGEN_STRESS[k]) <= NITROGEN_STRESS_MAX:
            assert _nitrogen_warning(k) is not None, k


def test_nitrogen_warning_band(monkeypatch):
    import agrijax.calib.workflow as wf

    monkeypatch.setattr(wf, "NITROGEN_DATES_DIFFER", frozenset({"TWIN_t04", "TWIN_t05"}))
    for k, d in {
        "TWIN_t01": 0.0,
        "TWIN_t02": NITROGEN_STRESS_OK,
        "TWIN_t03": -NITROGEN_STRESS_OK,
        "TWIN_t04": 0.0037,
        "TWIN_t05": -0.03,
        "TWIN_t06": 0.0301,
        "TWIN_t07": -NITROGEN_STRESS_MAX,
    }.items():
        monkeypatch.setitem(wf.NITROGEN_STRESS, k, d)
    # at or below the 2 % limit and with equal dates: no warning
    for k in ("TWIN_t01", "TWIN_t02", "TWIN_t03"):
        assert _nitrogen_warning(k) is None, k
    # the warning band (above 2 %, up to and including 5 %): a warning with the change and the limits
    w6 = _nitrogen_warning("TWIN_t06")
    assert w6 is not None and w6.startswith("TWIN_t06:") and "+3.0%" in w6 and "date" not in w6
    assert "above +-2%" in w6 and "+-5%" in w6
    w7 = _nitrogen_warning("TWIN_t07")
    assert w7 is not None and "-5.0%" in w7
    # dates differ, yield within 2 %: a warning about the dates only
    w4 = _nitrogen_warning("TWIN_t04")
    assert w4 is not None and "silking (ADAT) or maturity (MDAT) date" in w4 and "yield" not in w4
    # both reasons in one message
    w5 = _nitrogen_warning("TWIN_t05")
    assert w5 is not None and "-3.0%" in w5 and "MDAT" in w5


@needs_x64
def test_calibrate_warns_in_the_band_and_refuses_above(tmp_path, monkeypatch):
    """A treatment in the 2-5 % band, or whose dates change with nitrogen on, is calibrated with a
    warning; above 5 % it is refused. ``targets=('HWUM',)`` stops the call right after the scope and
    warning steps, with a target error (no fit runs)."""
    import agrijax.calib.workflow as wf
    from agrijax.calib import calibrate

    eng, data = _fake_setup(tmp_path, monkeypatch)
    kw = dict(treatments=[4], holdout=[6], targets=("HWUM",), data_dir=data, engine=eng)
    monkeypatch.setattr(wf, "NITROGEN_DATES_DIFFER", frozenset())
    # calibration treatment in the band, held-out treatment below 2 %: one warning, no refusal
    monkeypatch.setitem(wf.NITROGEN_STRESS, "UFGA8201_t04", 0.03)
    monkeypatch.setitem(wf.NITROGEN_STRESS, "UFGA8201_t06", -0.0199)
    with (
        pytest.warns(
            CalibrationWarning, match=r"UFGA8201_t04: DSSAT yield with nitrogen on differs by \+3\.0%"
        ) as rec,
        pytest.raises(ObservationError, match="not mapped"),
    ):
        calibrate("UFGA8201", **kw)
    assert [str(w.message) for w in rec if "nitrogen" in str(w.message)] == [
        str(_nitrogen_warning("UFGA8201_t04"))
    ]
    # the held-out treatment counts too
    monkeypatch.setitem(wf.NITROGEN_STRESS, "UFGA8201_t04", 0.0)
    monkeypatch.setitem(wf.NITROGEN_STRESS, "UFGA8201_t06", -0.045)
    with (
        pytest.warns(
            CalibrationWarning, match=r"UFGA8201_t06: DSSAT yield with nitrogen on differs by -4\.5%"
        ),
        pytest.raises(ObservationError, match="not mapped"),
    ):
        calibrate("UFGA8201", **kw)
    # dates differ, yield change below 2 %: warned
    monkeypatch.setitem(wf.NITROGEN_STRESS, "UFGA8201_t06", 0.0)
    monkeypatch.setattr(wf, "NITROGEN_DATES_DIFFER", frozenset({"UFGA8201_t04"}))
    with (
        pytest.warns(CalibrationWarning, match=r"UFGA8201_t04: DSSAT's silking \(ADAT\) or maturity"),
        pytest.raises(ObservationError, match="not mapped"),
    ):
        calibrate("UFGA8201", **kw)
    # no reason to warn: no nitrogen warning
    monkeypatch.setattr(wf, "NITROGEN_DATES_DIFFER", frozenset())
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter("always")
        with pytest.raises(ObservationError, match="not mapped"):
            calibrate("UFGA8201", **kw)
    assert not [w for w in rec2 if "nitrogen" in str(w.message)]
    # above the 5 % limit: refused, flagged dates or not
    for differ in (frozenset(), frozenset({"UFGA8201_t04"})):
        monkeypatch.setattr(wf, "NITROGEN_DATES_DIFFER", differ)
        monkeypatch.setitem(wf.NITROGEN_STRESS, "UFGA8201_t04", 0.0501)
        with pytest.raises(ScopeError, match="nitrogen matters"):
            calibrate("UFGA8201", **kw)


def test_parse_treatments():
    assert _parse_treatments([2, 4], ["UFGA8201"], "t") == [("UFGA8201", 2), ("UFGA8201", 4)]
    assert _parse_treatments(6, ["UFGA8201"], "t") == [("UFGA8201", 6)]
    two = ["SIAZ9501", "SIAZ9601"]
    assert _parse_treatments(["SIAZ9501_t01", ("SIAZ9601", 1), "SIAZ9601:3"], two, "t") == [
        ("SIAZ9501", 1),
        ("SIAZ9601", 1),
        ("SIAZ9601", 3),
    ]
    assert _parse_treatments(("SIAZ9601", 2), two, "t") == [("SIAZ9601", 2)]
    with pytest.raises(ValueError, match="ambiguous"):
        _parse_treatments([1], two, "t")
    with pytest.raises(ValueError, match="not among"):
        _parse_treatments(["UFGA8201_t02"], two, "t")
    assert _parse_treatments(None, two, "t") == []


def test_cul_written_is_the_printed_row():
    ib0035 = {"P1": 259.0, "P2": 1.193, "P5": 947.1, "G2": 924.3, "G3": 8.168, "PHINT": 43.0}
    assert cul_written(ib0035, decimals=DEC) == ib0035
    w = cul_written(
        {"P1": 254.61, "P2": 1.1934, "P5": 954.53, "G2": 442.71, "G3": 14.662, "PHINT": 45.838}, decimals=DEC
    )
    assert w == {"P1": 254.6, "P2": 1.193, "P5": 954.5, "G2": 442.7, "G3": 14.66, "PHINT": 45.84}


def test_result_summary_and_json():
    pb = make_problem()
    with pytest.warns(CalibrationWarning):
        r = fit_cultivar(pb, [0, 1], holdout=[2], starts=2, seed=0, budget=60)
    res = CalibrationResult(
        cultivar="TW0001",
        treatments=["TWIN0001_t01", "TWIN0001_t02"],
        params=dict(zip(CUL_ORDER, r.theta_written.tolist(), strict=True)),
        params_unrounded=dict(zip(CUL_ORDER, r.theta_unrounded.tolist(), strict=True)),
        published=dict(zip(CUL_ORDER, PUBLISHED.tolist(), strict=True)),
        free=r.free,
        fixed=r.fixed,
        loss=r.loss,
        fit=pd.DataFrame(r.fit),
        targets=r.targets,
        scales=r.scales,
        dropped=[],
        unsupported={},
        sensitivity=r.sensitivity,
        trust=r.trust,
        predictions=r.predictions,
        reference={},
        per_start=r.per_start,
        method=r.method,
        calls=r.calls,
        wall_s=r.wall_s,
        holdout={**r.holdout, "fit": pd.DataFrame(r.holdout["fit"])} if r.holdout else None,
        warnings=r.warnings,
    )
    text = str(res)
    assert "TW0001" in text and "objective: published" in text and "ADAT" in text and "held out" in text
    json.dumps(res.to_dict(), default=float)


def test_published_outside_the_box_is_reported():
    pb = make_problem()
    pb.published = PUBLISHED.copy()
    pb.published[CUL_ORDER.index("G2")] = 1100.0  # MAXIMA 990 (like IB0172 in MZCER048.CUL)
    with pytest.warns(CalibrationWarning, match="outside the MINIMA / MAXIMA box"):
        res = fit_cultivar(pb, range(3), starts=2, seed=0, budget=60)
    assert res.theta_written[CUL_ORDER.index("G2")] <= 990.0


def test_published_is_returned_when_nothing_beats_it():
    """Published = truth: no written candidate can have a lower objective, so the published cultivar
    comes back, with a warning, instead of a worse point."""
    pb = make_problem()
    pb.published = TRUTH.copy()
    with pytest.warns(CalibrationWarning, match="the published cultivar is returned"):
        res = fit_cultivar(pb, range(3), starts=2, seed=0, budget=60)
    assert not res.improved
    np.testing.assert_array_equal(res.theta_written, TRUTH)
    assert res.loss["written"] == res.loss["published"]


# ------------------------------------------------------------------ calibrate() end to end, no DSSAT
_CUL = (
    "*MAIZE CULTIVAR COEFFICIENTS: MZCER048 MODEL (synthetic copy for a unit test)\r\n"
    "@VAR#  VRNAME.......... EXPNO   ECO#    P1    P2    P5    G2    G3 PHINT\r\n"
    "999991 MINIMA               . DFAULT   5.0 0.000 580.0 248.0  5.00 38.00\r\n"
    "999992 MAXIMA               . DFAULT 450.0 2.000 999.0 990.0 16.50 75.00\r\n"
    "IB0035 McCurdy 84aa         . IB0001 280.0 1.000 900.0 800.0  9.00 46.00\r\n"
)
_FILEX = (
    "*EXP.DETAILS: UFGA8201MZ synthetic twin\n\n"
    "*TREATMENTS                        -------------FACTOR LEVELS------------\n"
    "@N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM\n"
    " 1 1 0 0 T1                         1  1  0  1  1  1  1  0  0  0  0  0  1\n"
    " 4 1 0 0 T4                         1  1  0  1  1  2  2  0  0  0  0  0  1\n"
    " 6 1 0 0 T6                         1  1  0  1  1  3  2  0  0  0  0  0  1\n\n"
    "*CULTIVARS\n@C CR INGENO CNAME\n 1 MZ IB0035 McCurdy 84aa\n"
)
#: the twin's environment factor per treatment of the fake experiment
_ENV = {4: 1.0, 6: 0.8}
_T0 = 1982056


def _twin_obs(env: float) -> dict:
    ents = [
        Entry(E_DATE, 4),
        Entry(E_DATE, 10),
        Entry(E_FINAL, I_GWAD),
        Entry(E_FINAL, I_CWAD),
        Entry(E_FINAL, I_GAD),
    ]
    ents += [Entry(E_DAY, I_LAI, d) for d in LAI_DAYS]
    k = np.asarray([e.kind for e in ents])
    o = np.asarray([e.out for e in ents])
    t = np.asarray([e.t for e in ents])
    return dict(
        zip(
            ["ADAT", "MDAT", "HWAM", "CWAM", "H#AM", *[f"L{d}" for d in LAI_DAYS]],
            twin_values(np, TRUTH[None], k, o, t, env)[0].tolist(),
            strict=True,
        )
    )


class _TwinDay:
    """Stands in for the free-run day simulator (same interface) on the fake runs."""

    def __init__(self, runs, entries):
        e_n = max(len(e) for e in entries)
        self.tab = {n: np.zeros((len(runs), e_n)) for n in ("kind", "out", "t")}
        for b, ent in enumerate(entries):
            for j, e in enumerate(ent):
                self.tab["kind"][b, j], self.tab["out"][b, j], self.tab["t"][b, j] = e.kind, e.out, e.t
        self.env = np.asarray([_ENV[r.trno] for r in runs])

    def __call__(self, theta, tid):
        tb = self.tab
        return twin_values(
            np, np.asarray(theta), tb["kind"][tid], tb["out"][tid], tb["t"][tid], self.env[tid][:, None]
        )

    def jax_loss(self, obs, winv):
        raise NotImplementedError

    def stats(self):
        return {"seasons": 0, "program_calls": 0, "compile_s": 0.0, "n_devices": 1}


#: the input sources the fake reference runs were asked for (last :func:`_fake_setup`)
_SOURCES: list[str] = []


def _fake_setup(tmp_path, monkeypatch):
    import agrijax.calib.workflow as wf
    from agrijax.io.dssat.genotype import read_cul
    from agrijax.sites.dssat_free_run import missing_tables

    eng, data = tmp_path / "eng", tmp_path / "data"
    exe = eng / "source" / "build486" / "bin" / "dscsm048"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    (eng / "source" / "Data" / "Genotype").mkdir(parents=True)
    (eng / "source" / "Data" / "Genotype" / "MZCER048.CUL").write_bytes(_CUL.encode("latin-1"))
    maize = eng / "example_data" / "Maize"
    maize.mkdir(parents=True)
    (maize / "UFGA8201.MZX").write_text(_FILEX)
    obs = {t: _twin_obs(env) for t, env in _ENV.items()}
    (maize / "UFGA8201.MZA").write_text(
        "*EXP. DATA (A): UFGA8201MZ synthetic\n\n@TRNO   ADAT  MDAT   HWAM   CWAM   H#AM\n"
        + "".join(
            f"{t:6d}{int(_T0 % 1000 + o['ADAT']):6d}{int(_T0 % 1000 + o['MDAT']):6d}"
            f"{o['HWAM']:7.1f}{o['CWAM']:7.1f}{o['H#AM']:7.1f}\n"
            for t, o in obs.items()
        )
    )
    (maize / "UFGA8201.MZT").write_text(
        "*EXP. DATA (T): UFGA8201MZ synthetic\n\n@TRNO   DATE  LAID\n"
        + "".join(f"{t:6d} {82056 + d:5d} {o[f'L{d}']:5.3f}\n" for t, o in obs.items() for d in LAI_DAYS)
    )
    for t in _ENV:
        for p in missing_tables("UFGA8201", t, data):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"")
    days = np.asarray(yrdoy_range(_T0, _T0 + 199), dtype=np.int64)
    pub = dict(zip(CUL_ORDER, PUBLISHED.tolist(), strict=True))
    sources = _SOURCES
    sources.clear()

    def reference_inputs(filex, trno, dest, data_, engine, source="tables", root=None):
        sources.append(source)
        return types.SimpleNamespace(
            key=f"UFGA8201_t{trno:02d}",
            exp="UFGA8201",
            trno=trno,
            days=days,
            nl=4,
            mesev="R",
            published=lambda: dict(pub),
            summary={"HWAM": 1.0},
        )

    def dssat_run(filex, trno, dest, engine, cultivar, cul_file, root=None):
        row = read_cul(cul_file).loc[cultivar[1]]
        th = np.asarray([[float(row[n]) for n in CUL_ORDER]])
        ents = [Entry(E_DATE, 4), Entry(E_DATE, 10), Entry(E_FINAL, I_GWAD)]
        k, o, t = (np.asarray([getattr(e, a) for e in ents]) for a in ("kind", "out", "t"))
        a, m, hw = twin_values(np, th, k, o, t, _ENV[trno])[0]
        return {
            "row": {
                "HWAM": hw,
                "CWAM": 0.0,
                "H#AM": 0.0,
                "ADAT": float(days[int(a)]),
                "MDAT": float(days[int(m)]),
            },
            "read_new_row": True,
        }

    monkeypatch.setattr(wf, "_reference_inputs", reference_inputs)
    monkeypatch.setattr(wf, "_day_simulator", _TwinDay)
    monkeypatch.setattr(wf, "_dssat_run", dssat_run)
    return eng, data


@needs_x64
def test_calibrate_end_to_end_on_the_twin(tmp_path, monkeypatch):
    from agrijax.calib import calibrate

    eng, data = _fake_setup(tmp_path, monkeypatch)
    cul = tmp_path / "out" / "MZCER048.CUL"
    with pytest.warns(CalibrationWarning):
        res = calibrate(
            "UFGA8201",
            treatments=[4],
            holdout=[6],
            write_cul=cul,
            dssat_check=True,
            starts=3,
            budget=900,
            data_dir=data,
            engine=eng,
        )
    assert res.cultivar == "IB0035" and res.treatments == ["UFGA8201_t04"] and res.improved
    assert set(res.targets) == {"ADAT", "MDAT", "HWAM", "CWAM", "H#AM", "LAID"}
    assert res.loss["written"] < 0.1 * res.loss["published"]
    assert res.holdout is not None and res.holdout["loss_written"] < res.holdout["loss_published"]
    assert res.cul_id == "AJ0001" and res.cul_line in cul.read_text(errors="replace")
    assert res.dssat_check is not None and res.dssat_check["agree"], res.dssat_check
    assert "P2" in res.fixed and "IB0035" in str(res)
    # an existing .CUL is refused before anything runs
    with pytest.raises(FileExistsError, match="write_cul"):
        calibrate("UFGA8201", treatments=[4], write_cul=cul, data_dir=data, engine=eng)


@needs_x64
def test_calibrate_refusals_without_dssat(tmp_path, monkeypatch):
    from agrijax.calib import calibrate
    from agrijax.calib.workflow import ObservationError, ScopeError

    eng, data = _fake_setup(tmp_path, monkeypatch)
    with pytest.raises(ScopeError, match="nitrogen matters"):
        calibrate("UFGA8201", treatments=[1, 4], data_dir=data, engine=eng)
    with pytest.raises(ScopeError, match="no maize experiment file"):
        calibrate("XXXX9901", data_dir=data, engine=eng)
    with pytest.raises(ObservationError, match="not mapped"):
        calibrate("UFGA8201", treatments=[4], targets=("HWUM",), data_dir=data, engine=eng)
    with pytest.raises(ObservationError, match="not observed"):
        calibrate("UFGA8201", treatments=[4], targets=("L#SM",), data_dir=data, engine=eng)
    with pytest.raises(ValueError, match="targets"):
        calibrate("UFGA8201", treatments=[4], targets="everything", data_dir=data, engine=eng)
    with pytest.raises(FileNotFoundError, match="dscsm048"):
        calibrate("UFGA8201", treatments=[4], data_dir=data, engine=tmp_path / "none")


@needs_x64
def test_calibrate_input_sources(tmp_path, monkeypatch):
    """``inputs="auto"`` takes the tables when every treatment has them, the native inputs otherwise;
    the native scope check names the unported DSSAT process."""
    from agrijax.calib import calibrate
    from agrijax.calib.workflow import ScopeError
    from agrijax.sites.dssat_free_run import SPAM_TABLES, missing_tables

    eng, data = _fake_setup(tmp_path, monkeypatch)
    kw = {"treatments": [4], "holdout": [6], "starts": 1, "budget": 60, "data_dir": data, "engine": eng}
    with pytest.warns(CalibrationWarning):
        res = calibrate("UFGA8201", **kw)
    assert res.inputs == "tables" and set(_SOURCES) == {"tables"} and "tables inputs" in str(res)
    _SOURCES.clear()
    with pytest.warns(CalibrationWarning):
        res_n = calibrate("UFGA8201", inputs="native", **kw)
    assert res_n.inputs == "native" and set(_SOURCES) == {"native"}
    assert res_n.params == res.params  # the twin does not depend on the source
    # without the tables, "auto" is native; "tables" is refused naming the missing tables
    (data / SPAM_TABLES / "UFGA8201_t06_spam_in.npz").unlink()
    assert missing_tables("UFGA8201", 6, data)
    with pytest.warns(CalibrationWarning):
        assert calibrate("UFGA8201", **kw).inputs == "native"
    with pytest.raises(ScopeError, match="not distributed"):
        calibrate("UFGA8201", inputs="tables", **kw)
    with pytest.raises(ValueError, match="inputs"):
        calibrate("UFGA8201", inputs="dumps", **kw)
    # a treatment the native inputs do not support is refused before anything runs
    fx = eng / "example_data" / "Maize" / "UFGA8201.MZX"
    fx.write_text(
        fx.read_text().replace(
            " 6 1 0 0 T6                         1  1  0  1  1  3  2  0  0  0  0  0  1",
            " 6 1 0 0 T6                         1  1  0  1  1  3  2  0  0  0  1  0  1",
        )
    )
    with pytest.raises(ScopeError, match="WTHMOD"):
        calibrate("UFGA8201", inputs="native", **kw)


@needs_x64
def test_result_writes_the_row_and_checks_dssat_afterwards(tmp_path, monkeypatch):
    from agrijax.calib import calibrate

    eng, data = _fake_setup(tmp_path, monkeypatch)
    with pytest.warns(CalibrationWarning):
        res = calibrate(
            "UFGA8201", treatments=[4], holdout=[6], starts=2, budget=600, data_dir=data, engine=eng
        )
    assert res.cul_line is None and res.dssat_check is None and res.context is not None
    cul = tmp_path / "o" / "MZCER048.CUL"
    line = res.write_cul(cul)
    assert res.cul_id == "AJ0001" and line in cul.read_text(errors="replace") and res.cul_path == cul
    with pytest.raises(FileExistsError):
        res.write_cul(cul)
    t = res.check_dssat()
    assert list(t["treatment"]) == ["UFGA8201_t04", "UFGA8201_t06"] and t["agree"].all()
    assert res.dssat_check is not None and res.dssat_check["agree"] and "DSSAT check" in str(res)
    assert "context" not in res.to_dict()


@needs_x64
def test_native_calibration_does_not_look_for_dssat(tmp_path, monkeypatch):
    """``inputs="native"`` without ``engine=``: the DSSAT program is not looked for (the framing:
    Agri-JAX runs without DSSAT; only the DSSAT check fetches it)."""
    from agrijax.calib import calibrate
    from agrijax.port import run_fortran

    eng, data = _fake_setup(tmp_path, monkeypatch)
    asked: list[object] = []
    real = run_fortran.dscsm_paths
    monkeypatch.setattr(run_fortran, "dscsm_paths", lambda e=None: asked.append(e) or real(e))
    with pytest.warns(CalibrationWarning):
        res = calibrate(
            "UFGA8201", treatments=[4], starts=1, budget=60, data_dir=data, data=eng, inputs="native"
        )
    assert res.inputs == "native" and asked == [] and res.context["engine"] is None
