"""The soil evaporation swap of the facade (``soil_evaporation=`` of ``exp.run``, ``exp.scenarios`` and
``aj.calibrate``) against ``dscsm048`` and against the swap test of the day.

* **Against DSSAT on the same swap.** A swapped season of UFGA8201 (``MESEV = R`` in the file; swapped
  to SALUS) and of SIAZ9501 (``MESEV = S``; swapped to Ritchie) against ``dscsm048`` run with that
  option set in the experiment file (``exp.reference(..., soil_evaporation=...)``): the silking and
  maturity dates and the yield and tops weight at maturity, and the daily series to the printed
  precision of DSSAT's output. The swap matters on the rainfed treatment (yield -10 % on UFGA8201 t1) and
  leaves the irrigated ones (t4, t6) within a kg ha-1.
* **Against the swap test of the day.** ``test_day_dssat486_swap.py`` runs each reference treatment with
  its own soil evaporation and with the other one, from the reference run's tables, and reports yield,
  season ES and EP, profile water, maturity shift and the closure of the water ledger of both. The
  facade's swapped run (native inputs, no DSSAT run) reproduces the report's row of the same
  treatments, number by number.
* **Scenarios, cultivar batches and the calibration.** The weather-year x sowing-date scenarios and
  the cultivar samples on the swap against ``dscsm048`` on the same swap (``Scenarios.reference``,
  ``Experiment.dssat_batch``); a calibration on the swap, whose DSSAT check runs ``dscsm048`` with
  ``MESEV`` swapped too.
* **The module's user code.** The code block at the top of ``agrijax.facade_swap``, executed as it is
  written.
"""

from __future__ import annotations

import os
import re
import textwrap
import time
import warnings
from pathlib import Path
from typing import Any

import day_dssat486_free_harness as h
import jax
import numpy as np
import pytest
import test_day_dssat486_swap as swap_test

from agrijax import facade_swap as fs
from agrijax.models.day_dssat486 import SOIL_EVAPORATION_KEYS
from agrijax.port.run_fortran import DSSAT_ENGINE, dscsm_paths

pytestmark = [
    pytest.mark.allow_skip(reason="needs dscsm048 v4.8.6.0 (AGRI_JAX_DSSAT) and the DSSAT example data"),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the DSSAT day runs in float64"),
]

#: a swapped season against ``dscsm048`` on the same swap (HWAM and CWAM are printed to 1 kg ha-1; the
#: facade's own test measures 3e-5 for the experiment's own choice, here at most 1.3e-4 on 4 runs)
SEASON_YIELD_RTOL = 1e-3
#: the yield change of the swap against DSSAT's, kg ha-1: the two printed HWAM values, 0.5 each, and the
#: model difference (measured at most 0.35 on the three treatments)
HWAM_CHANGE_ATOL = 1.5
#: daily series against ``dscsm048``, RMSE over the season maximum (printed precision of the output files;
#: the own choice reaches 1.3e-3 on SIAZ9501, the swapped ones at most the same)
DAILY_RMSE_REL = 2e-3
#: scenarios and cultivar samples on the swap against ``dscsm048`` (as the facade's own test; measured
#: on the rainfed treatment, whose low yields sit closest to the printed precision: scenarios 1.3e-4,
#: cultivar samples 7.2e-4)
BATCH_YIELD_RTOL = 1e-3
#: the facade's numbers against the swap test's report row, relative (the native inputs equal the table
#: ones except the hourly mean TAVG, ``test_dssat_native_inputs.NATIVE_VS_TABLES_RTOL``; measured: the
#: three rows are reproduced exactly, 0 difference)
SWAP_TEST_RTOL = 1e-6
#: the treatments whose swap-test row is reproduced: UFGA8201 t1 is the report's own run (``MESEV = R``,
#: swapped to SALUS), t4 the quick start's, SIAZ9501 t1 swaps the other way (``MESEV = S`` to Ritchie)
SWAP_TEST_RUNS = [("UFGA8201", 1), ("UFGA8201", 4), ("SIAZ9501", 1)]


@pytest.fixture(scope="module")
def engine() -> Path:
    if not dscsm_paths(DSSAT_ENGINE)[0].is_file():
        pytest.skip(f"dscsm048 not found under {DSSAT_ENGINE}")
    return DSSAT_ENGINE


@pytest.fixture(scope="module")
def exp(engine: Path):
    import agrijax as aj

    return aj.dssat.experiment("UFGA8201", data_root=engine)


@pytest.fixture(scope="module")
def siaz(engine: Path):
    import agrijax as aj

    return aj.dssat.experiment("SIAZ9501", data_root=engine)


def _other(experiment: Any, treatment: int) -> str:
    """The alternative the experiment's file does not use on ``treatment``."""
    own = fs.own_mesev(experiment.filex, treatment)
    return next(a.name for a in fs.ALTERNATIVES if a.mesev != own)


# ------------------------------------------------------------------ against DSSAT on the same swap
@pytest.mark.parametrize("treatment", [1, 4, 6])
def test_swapped_season_against_dscsm048_on_the_same_swap(exp, treatment):
    assert fs.own_mesev(exp.filex, treatment) == "R"
    own = exp.run(treatment)
    swapped = exp.run(treatment, soil_evaporation="salus")
    assert swapped.inputs.mesev == "S" and own.inputs.mesev == "R"
    assert swapped.inputs.out is None  # no DSSAT run
    ref = exp.reference(treatment, soil_evaporation="salus")
    s = swapped.compare_summary(ref)
    assert s.loc["ADAT", "difference"] == 0 and s.loc["MDAT", "difference"] == 0
    assert abs(s.loc["HWAM", "difference"]) < SEASON_YIELD_RTOL, s
    assert abs(s.loc["CWAM", "difference"]) < SEASON_YIELD_RTOL, s
    d = swapped.compare_daily(ref)
    assert (d["RMSE / max"] < DAILY_RMSE_REL).all(), d
    print(
        f"UFGA8201 t{treatment} salus: HWAM {s.loc['HWAM', 'difference']:.2e}, max RMSE/max "
        f"{d['RMSE / max'].max():.2e}; yield own {own.summary['HWAM']:.1f} swapped {swapped.summary['HWAM']:.1f}"
    )
    # the swap changes the yield by what it changes in DSSAT (each HWAM is printed to 1 kg ha-1)
    r_own = exp.reference(treatment)
    change, dssat_change = (
        swapped.summary["HWAM"] - own.summary["HWAM"],
        ref.summary["HWAM"] - r_own.summary["HWAM"],
    )
    assert abs(change - dssat_change) < HWAM_CHANGE_ATOL, (change, dssat_change)
    assert np.sum(own.outputs["es"]) != np.sum(swapped.outputs["es"])  # the two methods differ by design


def test_the_swap_matters_on_the_rainfed_treatment(exp):
    own, salus = exp.run(1), exp.run(1, soil_evaporation="salus")
    assert salus.summary["HWAM"] < 0.95 * own.summary["HWAM"]  # measured -10 %
    ref_own, ref_salus = exp.reference(1), exp.reference(1, soil_evaporation="salus")
    assert ref_salus.summary["HWAM"] < 0.95 * ref_own.summary["HWAM"]  # DSSAT says the same


def test_ritchie_swap_of_a_salus_experiment_against_dscsm048(siaz):
    assert fs.own_mesev(siaz.filex, 1) == "S"
    own = siaz.run(1)
    swapped = siaz.run(1, soil_evaporation="ritchie")
    assert own.inputs.mesev == "S" and swapped.inputs.mesev == "R"
    assert siaz.inputs(1, soil_evaporation="salus") is siaz.inputs(1)  # its own: nothing is rebuilt
    ref = siaz.reference(1, soil_evaporation="ritchie")
    s = swapped.compare_summary(ref)
    assert s.loc["ADAT", "difference"] == 0 and s.loc["MDAT", "difference"] == 0
    assert (
        abs(s.loc["HWAM", "difference"]) < SEASON_YIELD_RTOL
        and abs(s.loc["CWAM", "difference"]) < SEASON_YIELD_RTOL
    )
    d = swapped.compare_daily(ref)
    assert (d["RMSE / max"] < DAILY_RMSE_REL).all(), d
    assert np.sum(own.outputs["es"]) != np.sum(swapped.outputs["es"])
    import agrijax as aj

    assert aj.dssat.alternatives(siaz, 1)["experiment's own"].to_dict() == {"ritchie": False, "salus": True}


def test_the_own_choice_is_not_a_swap(exp):
    a = exp.run(4)
    b = exp.run(4, soil_evaporation="ritchie")  # UFGA8201's own
    assert exp.inputs(4, soil_evaporation="ritchie") is exp.inputs(4)
    np.testing.assert_array_equal(a.daily["gwad"].to_numpy(), b.daily["gwad"].to_numpy())
    assert a.summary == b.summary


# ------------------------------------------------------------------ against the swap test of the day
def _compare_rows(a: dict[str, Any], b: dict[str, Any], key: str) -> float:
    """The largest relative difference of two swap-test rows (strings and integers must be equal)."""
    worst = 0.0
    assert a.keys() == b.keys()
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, (bool, str, type(None))) or isinstance(x, (int, np.integer)):
            assert x == y, (key, k, x, y)
        else:
            rel = abs(x - y) / max(abs(x), abs(y), 1.0)
            worst = max(worst, rel)
            assert rel < SWAP_TEST_RTOL, (key, k, x, y, rel)
    return worst


def test_the_swap_reproduces_the_swap_tests_numbers(exp, siaz, data_dir, tmp_path_factory, engine):
    keys = SWAP_TEST_RUNS
    for sub in (h.DSW, h.DET, h.A12):
        if not (data_dir / sub).is_dir():
            pytest.skip(f"{data_dir / sub} not found")
    miss = [k for k in keys if not (data_dir / h.A12 / f"{h.key_of(*k)}_spam.npz").is_file()]
    if miss:
        pytest.skip(f"no SPAM dump tables of {miss}")
    work = tmp_path_factory.mktemp("swap_rows")
    jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    outs = h.run_references(keys, work, data_dir, jobs)
    runs = {k: h.build(k[0], k[1], outs[h.key_of(*k)], data_dir) for k in keys}
    experiments = {"UFGA8201": exp, "SIAZ9501": siaz}
    worst = 0.0
    for k, r in runs.items():
        other = SOIL_EVAPORATION_KEYS["S" if r.mesev == "R" else "R"]
        native = h.run_group([r], "free")
        swapped = h.run_group([r], "free", soil_evaporation=other)
        report = swap_test._row(
            r, {n: v[0] for n, v in native.items()}, {n: v[0] for n, v in swapped.items()}
        )
        e = experiments[k[0]]
        choice = _other(e, k[1])
        assert SOIL_EVAPORATION_KEYS[fs.resolve(choice).mesev] == other  # the same implementation
        own_run, swapped_run = e.run(k[1]), e.run(k[1], soil_evaporation=choice)
        assert (
            swapped_run.inputs.n_days == r.n_days == own_run.inputs.n_days
        )  # the season the report runs over
        facade = swap_test._row(r, own_run.outputs, swapped_run.outputs)
        w = _compare_rows(report, facade, h.key_of(*k))
        worst = max(worst, w)
        print(
            f"{h.key_of(*k)}: {report['native']} -> {report['swapped']}: yield {report['native_yield']:.1f} -> "
            f"{report['swapped_yield']:.1f} ({report['yield_change_rel']:+.2%}), ES {report['native_es_mm']:.1f} -> "
            f"{report['swapped_es_mm']:.1f} mm, maturity shift {report['maturity_shift_days']} d; "
            f"facade vs report: largest relative difference {w:.2e}"
        )
        # the report's own assertions hold for the facade's run as well
        assert (
            facade["finite"]
            and max(facade["native_ledger_mm"], facade["swapped_ledger_mm"]) <= h.LEDGER_ATOL_MM
        )
        assert facade["swapped_es_mm"] != facade["native_es_mm"]
    print(f"largest relative difference to the swap test's numbers: {worst:.2e}")


# ------------------------------------------------------------------ scenarios, cultivar batches
def test_swapped_scenarios_and_cultivar_batches_against_dscsm048(exp):
    kw: dict[str, Any] = {"years": [1979, 1982, 1985], "sowing_shift": [-14, 0, 14]}
    plain = exp.scenarios(treatment=1, **kw)
    scen = exp.scenarios(treatment=1, soil_evaporation="salus", **kw)
    assert (
        len(scen.runs) == 9
        and all(r.mesev == "S" for r in scen.runs)
        and all(r.mesev == "R" for r in plain.runs)
    )
    assert not {f.parent for f in scen.filex} & {f.parent for f in plain.filex}
    dss = scen.reference()  # dscsm048 on the swapped files
    assert len(dss) == 9
    # the table of the swapped scenarios is the swapped model's season, as DSSAT's
    t = scen.table.merge(dss, on=["year", "sowing_shift"], suffixes=("", "_d"))
    assert (t["ADAT"] == t["ADAT_d"]).all() and (t["MDAT"] == t["MDAT_d"]).all()
    err = (np.abs(t["HWAM"] - t["HWAM_d"]) / t["HWAM_d"]).max()
    print(f"swapped scenarios against dscsm048: largest yield difference {err:.2e}")
    assert err < BATCH_YIELD_RTOL
    # it is not the experiment's own: DSSAT's own scenarios give other yields
    own_dss = plain.reference()
    assert (np.abs(dss["HWAM"] - own_dss["HWAM"]) / own_dss["HWAM"]).max() > 0.01
    rng = np.random.default_rng(0)
    pub = scen.published
    cul = {n: pub[n] * rng.uniform(0.9, 1.1, 12) for n in ("P1", "P5", "G2", "G3", "PHINT")}
    for n in cul:
        cul[n][0] = pub[n]
    first = scen.run(cul)
    assert first.timing["seasons"] == 9 * 12
    s0 = first.table[first.table["sample"] == 0].merge(dss, on=["year", "sowing_shift"], suffixes=("", "_d"))
    assert len(s0) == 9 and (np.abs(s0["HWAM"] - s0["HWAM_d"]) / s0["HWAM_d"]).max() < BATCH_YIELD_RTOL
    db = exp.dssat_batch(treatment=1, cultivar=cul, soil_evaporation="salus")  # DSSAT on the swap, 12 samples
    assert db.seasons == 12
    ours = scen.run(db.cultivar).table
    ours = ours[(ours["year"] == 1982) & (ours["sowing_shift"] == 0)].reset_index(drop=True)
    err = (np.abs(ours["HWAM"] - db.table["HWAM"]) / db.table["HWAM"]).max()
    print(f"cultivar samples on the swap against dscsm048: largest yield difference {err:.2e}")
    assert err < BATCH_YIELD_RTOL
    assert (ours["ADAT"] == db.table["ADAT"]).all() and (ours["MDAT"] == db.table["MDAT"]).all()
    # DSSAT on the experiment's own choice gives other yields for the same samples
    db_own = exp.dssat_batch(treatment=1, cultivar=cul)
    assert (np.abs(db.table["HWAM"] - db_own.table["HWAM"]) / db_own.table["HWAM"]).max() > 0.01


def test_a_swapped_season_with_a_changed_cultivar_agrees_with_dscsm048(exp):
    c = {"P1": 280.0, "G2": 650.0}
    s = exp.run(treatment=1, cultivar=c, soil_evaporation="salus")
    r = exp.reference(treatment=1, cultivar=c, soil_evaporation="salus")
    assert r.cultivar is not None and r.cultivar["P1"] == c["P1"]
    assert abs(s.summary["HWAM"] - r.summary["HWAM"]) / r.summary["HWAM"] < BATCH_YIELD_RTOL
    assert s.summary["MDAT"] == r.summary["MDAT"]


# ------------------------------------------------------------------ calibrate
def test_calibration_on_the_swap_checks_dssat_on_the_swap(exp, engine):
    import agrijax as aj

    kw: dict[str, Any] = {"treatments": [4], "holdout": [6], "starts": 2, "budget": 96, "seed": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = aj.calibrate(exp, **kw)
        swapped = aj.calibrate(exp, soil_evaporation="salus", dssat_check=True, **kw)
    assert plain.inputs == swapped.inputs == "native"
    assert swapped.notes[-1] == fs.note("salus") and fs.note("salus") not in plain.notes
    # it ran on the swapped model: the published cultivar's season in the result is the swapped one
    t6 = exp.run(6, soil_evaporation="salus").summary
    assert swapped.reference["UFGA8201_t06"]["HWAM"] == pytest.approx(t6["HWAM"], rel=1e-12)
    assert swapped.reference["UFGA8201_t06"]["HWAM"] != plain.reference["UFGA8201_t06"]["HWAM"]
    # DSSAT, run with the written coefficients and MESEV swapped, agrees with Agri-JAX
    assert swapped.dssat_check is not None and swapped.dssat_check["agree"], swapped.dssat_check
    assert plain.dssat_check is None
    with pytest.raises(fs.SwapError, match="needs the native inputs"):
        aj.calibrate(exp, soil_evaporation="salus", inputs="tables", **kw)


# ------------------------------------------------------------------ the module's user code
def _user_code() -> str:
    """The code block of the module docstring (indented four spaces, after the ``::``)."""
    doc = fs.__doc__ or ""
    m = re.search(r"::\n\n((?:    .*\n|\n)+?)\n\*\*What a swap is", doc)
    assert m is not None, "the module docstring has no code block"
    return textwrap.dedent(m.group(1))


@pytest.mark.slow
def test_the_modules_user_code_runs_as_written(engine, monkeypatch):
    import agrijax as aj

    code = _user_code()
    assert 5 <= len([ln for ln in code.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]) <= 15
    monkeypatch.setenv("AGRI_JAX_DSSAT", str(engine))
    monkeypatch.setattr(aj.dssat, "data", lambda root=None: engine)  # the machine has no network
    ns: dict[str, Any] = {}
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(code, "facade_swap.__doc__", "exec"), ns)
    print(f"the module's user code: {time.perf_counter() - t0:.0f} s")
    assert ns["salus"].inputs.mesev == "S" and ns["ritchie"].inputs.mesev == "R"
    assert ns["salus"].compare_summary(ns["ref"]).loc["MDAT", "difference"] == 0
    assert len(ns["batch"].table) == 30 * 3
    assert ns["res"].notes[-1] == fs.note("salus")
    assert ns["aj"] is aj
