"""The season readout of :mod:`agrijax.calib.dssat_day` on synthetic daily outputs (no data, no model run):
the end of the crop (the first ``MDATE`` setting under harvest at maturity, the setting that holds on the
harvest day otherwise), the run's last real day, the censored dates, the
end status, the device gather against the host readout, the earlier (``readout="padded"``) gather kept
bit for bit, and the harvest option of the free-run inputs. The soil-water and evapotranspiration
outputs under the season readout: derived on the host from the day outputs (``evap``, ``ep``,
``soil_sw``) and, on the synthetic soil run of ``test_calib_day_soil`` (float64), read by the
simulator on the run's real days only.

The day indices of a synthetic season: ``T`` simulated days (the batch group's length), the run's last
real day ``last`` (the harvest day, or the season plus its real weather extension), padding after it.
"""

from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.calib.dssat_day import (
    CROP_END,
    CUL_ORDER,
    E_DATE,
    E_DAY,
    E_FINAL,
    E_PAD,
    E_STATUS,
    END_CENSORED,
    END_FAILED,
    END_HARVESTED,
    END_MATURED,
    END_NAMES,
    ISTAGE_EMERGENCE,
    ISTAGE_MATURITY,
    ISTAGE_SILKING,
    OUT_NAMES,
    READOUTS,
    WATER_OUT_NAMES,
    DaySimulator,
    Entry,
    _first_idx,
    _gather_one,
    _gather_season,
    _last_real_day,
    entry_values,
)
from agrijax.processes.crop.ceres_maize.constants import ISTAGE_AFTER_MATURITY, MDATE_NONE
from agrijax.processes.crop.ceres_maize.constants import ISTAGE_MATURITY as ISTAGE_END_GRAIN_FILL
from agrijax.sites.dssat_free_run import FreeRunInputs, check_harvest, harvest_option

T = 40
I_GWAD, I_CWAD, I_LAI = (OUT_NAMES.index(n) for n in ("gwad", "cwad", "lai"))
ENTRIES = [
    Entry(E_DATE, ISTAGE_EMERGENCE),
    Entry(E_DATE, ISTAGE_SILKING),
    Entry(E_DATE, ISTAGE_MATURITY),
    Entry(E_DATE, CROP_END),
    Entry(E_FINAL, I_GWAD),
    Entry(E_FINAL, I_CWAD),
    Entry(E_DAY, I_LAI, 3),
    Entry(E_STATUS, 0),
    Entry(E_PAD, 0),
]
J = {"edat": 0, "adat": 1, "mdat10": 2, "end": 3, "gwad": 4, "cwad": 5, "lai3": 6, "status": 7, "pad": 8}


def season(
    *, silk: int = 10, efg_end: int | None = 20, mature: int | None = 23, fail: int | None = None
) -> dict[str, np.ndarray]:
    """Daily outputs ``[T]``: emergence on day 2, silking on ``silk``, ISTAGE 6 from ``efg_end``,
    ISTAGE 10 and ``MDATE`` from ``mature``; a failure on ``fail`` sets ISTAGE 6 and ``MDATE`` that day
    (a later ``mature`` sets ``MDATE`` again, as the failed crop's maturity does). Every output is a
    distinct exact number per day (exact in float32)."""
    ist = np.zeros(T, np.int32)
    ist[2:] = ISTAGE_EMERGENCE
    ist[silk:] = ISTAGE_SILKING
    if efg_end is not None:
        ist[efg_end:] = ISTAGE_END_GRAIN_FILL
    if fail is not None:
        ist[fail:] = ISTAGE_END_GRAIN_FILL
    if mature is not None:
        ist[mature:] = ISTAGE_AFTER_MATURITY
    mdate = np.full(T, MDATE_NONE, np.int32)
    for d in sorted(d for d in (fail, mature) if d is not None):
        mdate[d:] = 1982000 + d
    t = np.arange(T, dtype=float)
    out = {n: 1000.0 * (k + 1) + t for k, n in enumerate(OUT_NAMES)}
    return {**out, "istage": ist, "mdate": mdate}


def device(outs: dict[str, np.ndarray], last: int, open_: int, entries=ENTRIES, dtype=None) -> np.ndarray:
    """The device gather (``jit`` of :func:`_gather_season`) on one sample."""
    dt = dtype or (jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    y = jnp.asarray(np.stack([outs[n] for n in OUT_NAMES]), dtype=dt)
    ist = jnp.asarray(outs["istage"])
    md = jnp.asarray(outs["mdate"])
    k, o, t = (jnp.asarray([getattr(e, a) for e in entries], jnp.int32) for a in ("kind", "out", "t"))
    f = jax.jit(_gather_season)
    return np.asarray(f(y, ist, md, jnp.int32(last), jnp.int32(open_), k, o, t), dtype=np.float64)


def both(outs: dict[str, np.ndarray], last: int, harvest: str = "M") -> np.ndarray:
    """Host and device readouts, checked equal bit for bit (T7), returned once."""
    open_ = END_CENSORED if harvest == "M" else END_HARVESTED
    host = entry_values(outs, ENTRIES, last_day=last, harvest=harvest)
    dev = device(outs, last, open_)
    np.testing.assert_array_equal(dev, host)
    return host


def value(name: str, day: int) -> float:
    return 1000.0 * (OUT_NAMES.index(name) + 1) + day


# ------------------------------------------------------------------------ T1-T7: the readout
def test_t1_normal_maturity_reads_istage_10_not_istage_6():
    v = both(season(efg_end=20, mature=23), last=T - 1)
    assert v[J["end"]] == v[J["mdat10"]] == 23  # not the first ISTAGE 6 (day 20)
    assert v[J["gwad"]] == value("gwad", 23) and v[J["cwad"]] == value("cwad", 23)
    assert v[J["status"]] == END_MATURED and v[J["adat"]] == 10 and v[J["edat"]] == 2
    assert v[J["lai3"]] == value("lai", 3) and v[J["pad"]] == 0


@pytest.mark.parametrize("later_10", [None, 35])
def test_t2_crop_failure_ends_the_crop_on_the_failure_day(later_10):
    last = 30
    v = both(season(efg_end=None, fail=14, mature=later_10), last=last)
    assert v[J["end"]] == 14 and v[J["gwad"]] == value("gwad", 14)
    assert v[J["mdat10"]] == last + 1  # ISTAGE 10 never on a real day: censored
    assert v[J["status"]] == END_FAILED


def test_t2_a_failed_crop_harvested_at_maturity_ends_on_the_failure_day():
    # harvest at maturity: dscsm048 harvests on the day MDATE is first set (AUTHAR YREND = MDATE) and the
    # season ends there, so a later maturity of the failed crop (which sets MDATE again) is never reached
    v = both(season(efg_end=None, fail=14, mature=15), last=30, harvest="M")
    assert v[J["end"]] == 14 and v[J["gwad"]] == value("gwad", 14)
    assert v[J["status"]] == END_FAILED


def test_t2_a_failed_crop_that_matures_before_a_reported_harvest_ends_at_that_maturity():
    # reported harvest: the season runs to the harvest day and MDATE is set again at the failed crop's
    # maturity, dscsm048's MDAT is then that day; the status stays a failure (the first setting was not
    # at ISTAGE 10)
    v = both(season(efg_end=None, fail=14, mature=15), last=30, harvest="R")
    assert v[J["end"]] == v[J["mdat10"]] == 15 and v[J["gwad"]] == value("gwad", 15)
    assert v[J["status"]] == END_FAILED
    # a maturity after the harvest day is not read: the failure day is the end
    v = both(season(efg_end=None, fail=14, mature=35), last=30, harvest="R")
    assert v[J["end"]] == 14 and v[J["mdat10"]] == 31 and v[J["status"]] == END_FAILED


@pytest.mark.parametrize("mature", [None, 28, 36])
def test_t3_reported_harvest_reads_the_harvest_day(mature):
    harvest_day = 25  # maturity after it: on real extension weather (28) or padding (36), or never
    v = both(season(efg_end=22, mature=mature), last=harvest_day, harvest="R")
    assert v[J["gwad"]] == value("gwad", harvest_day) and v[J["cwad"]] == value("cwad", harvest_day)
    assert v[J["end"]] == v[J["mdat10"]] == harvest_day + 1  # censored at n_days
    assert v[J["status"]] == END_HARVESTED


def test_t3_maturity_before_a_reported_harvest_is_read_at_maturity():
    v = both(season(efg_end=18, mature=21), last=25, harvest="R")
    assert v[J["end"]] == 21 and v[J["gwad"]] == value("gwad", 21) and v[J["status"]] == END_MATURED


def test_t4_maturity_on_the_real_extension_and_after_it():
    last = 30  # season 0-24, real weather 25-30, padding after
    v = both(season(efg_end=24, mature=27), last=last)
    assert v[J["end"]] == 27 and v[J["gwad"]] == value("gwad", 27) and v[J["status"]] == END_MATURED
    v = both(season(efg_end=28, mature=33), last=last)
    assert v[J["gwad"]] == value("gwad", last) and v[J["end"]] == v[J["mdat10"]] == last + 1
    assert v[J["status"]] == END_CENSORED


@pytest.mark.parametrize("mature", [23, 33, None])
def test_t5_padded_days_are_never_read(mature):
    last = 28
    base = season(efg_end=21 if mature == 23 else 30, mature=mature)
    ref = both(base, last=last)
    junk = {k: np.array(v, copy=True) for k, v in base.items()}
    for n in OUT_NAMES:
        junk[n][last + 1 :] = np.nan
    junk["istage"][last + 1 :] = ISTAGE_AFTER_MATURITY  # a stage reached on padding changes nothing
    junk["mdate"][last + 1 :] = 1982999
    v = both(junk, last=last)
    assert np.all(np.isfinite(v))
    np.testing.assert_array_equal(v, ref)


def test_t6_each_run_is_censored_at_its_own_last_day():
    # one batch group: the same daily outputs, two runs of different length; FreeRunInputs.last_real_day
    short = types.SimpleNamespace(n_days=20, n_ext=60, harvest="M")
    long_ = types.SimpleNamespace(n_days=30, n_ext=3, harvest="M")
    harv = types.SimpleNamespace(n_days=20, n_ext=60, harvest="R")
    lr = FreeRunInputs.last_real_day
    assert lr(short, 10) == 29 and lr(long_, 10) == 32 and lr(harv, 10) == 19  # type: ignore[arg-type]
    assert lr(short, 0) == 19  # type: ignore[arg-type]
    assert _last_real_day(types.SimpleNamespace(n_days=12), 60) == 11  # a run without the method
    outs = season(efg_end=36, mature=39)
    lasts = jnp.asarray([24, 32], jnp.int32)
    y = jnp.asarray(np.stack([outs[n] for n in OUT_NAMES]))[None].repeat(2, axis=0)
    k, o, t = (jnp.asarray([[getattr(e, a) for e in ENTRIES]] * 2, jnp.int32) for a in ("kind", "out", "t"))
    ist = jnp.asarray(outs["istage"])[None].repeat(2, axis=0)
    md = jnp.asarray(outs["mdate"])[None].repeat(2, axis=0)
    opn = jnp.full(2, END_CENSORED, jnp.int32)
    v = np.asarray(jax.jit(jax.vmap(_gather_season))(y, ist, md, lasts, opn, k, o, t))
    assert v[0, J["end"]] == 25 and v[1, J["end"]] == 33
    assert v[0, J["gwad"]] == value("gwad", 24) and v[1, J["gwad"]] == value("gwad", 32)


def test_t7_host_default_last_day_is_the_last_output_day():
    outs = season(efg_end=36, mature=None)
    np.testing.assert_array_equal(entry_values(outs, ENTRIES), entry_values(outs, ENTRIES, last_day=T - 1))
    # without the MDATE output the end of the crop falls back to ISTAGE 10 (a failure is not seen)
    fail = season(efg_end=None, fail=14, mature=None)
    no_mdate = {k: v for k, v in fail.items() if k != "mdate"}
    v = entry_values(no_mdate, ENTRIES, last_day=30)
    assert v[J["end"]] == 31 and v[J["gwad"]] == value("gwad", 30) and v[J["status"]] == END_CENSORED
    # [days, 1] outputs (FreeRunInputs.simulate) read as [days]
    col = {k: np.asarray(v)[:, None] for k, v in fail.items()}
    np.testing.assert_array_equal(entry_values(col, ENTRIES, last_day=30), both(fail, last=30))


# ------------------------------------------------------------------------ T8: the padded readout
def _gather_one_reference(y, ist, etype, eout, et):
    """The gather of the readout before the season readout (a frozen copy)."""
    t_n = ist.shape[0]
    codes = jnp.asarray([1, 4, 10])
    firsts = jnp.stack([_first_idx(ist == c) for c in (1, 4, 10)])
    date = jnp.sum(jnp.where(eout[:, None] == codes[None, :], firsts[None, :], 0), axis=1).astype(y.dtype)
    t_mat = jnp.minimum(firsts[2], t_n - 1)
    t_eff = jnp.where(etype == E_FINAL, t_mat, jnp.clip(et, 0, t_n - 1))
    o = jnp.clip(eout, 0, y.shape[0] - 1)
    val = y[o, t_eff]
    return jnp.where(etype == E_DATE, date, jnp.where(etype == E_PAD, 0.0, val))


def test_t8_padded_readout_is_the_earlier_gather_bit_for_bit():
    rng = np.random.default_rng(0)
    for _ in range(20):
        mature = int(rng.integers(5, T + 8))
        outs = season(silk=int(rng.integers(3, 8)), efg_end=None, mature=mature if mature < T else None)
        y = jnp.asarray(rng.normal(size=(len(OUT_NAMES), T)))
        ents = [Entry(E_DATE, c) for c in (1, 4, 10, 3)] + [Entry(E_FINAL, i) for i in range(len(OUT_NAMES))]
        ents += [
            Entry(E_DAY, int(rng.integers(0, len(OUT_NAMES))), int(rng.integers(-3, T + 3))),
            Entry(E_PAD, 0),
        ]
        k, o, t = (jnp.asarray([getattr(e, a) for e in ents], jnp.int32) for a in ("kind", "out", "t"))
        ist = jnp.asarray(outs["istage"])
        np.testing.assert_array_equal(
            np.asarray(jax.jit(_gather_one)(y, ist, k, o, t)),
            np.asarray(jax.jit(_gather_one_reference)(y, ist, k, o, t)),
        )


def test_t8_season_equals_padded_on_a_sample_matured_on_a_real_day():
    outs = season(efg_end=20, mature=23)
    ents = [e for e in ENTRIES if e.kind != E_STATUS and not (e.kind == E_DATE and e.out == CROP_END)]
    y = jnp.asarray(np.stack([outs[n] for n in OUT_NAMES]))
    k, o, t = (jnp.asarray([getattr(e, a) for e in ents], jnp.int32) for a in ("kind", "out", "t"))
    old = np.asarray(_gather_one(y, jnp.asarray(outs["istage"]), k, o, t))
    for last in (23, 30, T - 1):
        new = device(outs, last, END_CENSORED, entries=ents, dtype=y.dtype)
        np.testing.assert_array_equal(new, old)


# ------------------------------------------------------------------------ T9: refusals
def _fake_run(n_days: int = 10) -> types.SimpleNamespace:
    return types.SimpleNamespace(n_days=n_days, n_ext=0, harvest="M", nl=4, mesev="R")


def test_t9_day_simulator_refusals_before_building_anything():
    r = _fake_run()
    with pytest.raises(ValueError, match="outside the run's real days"):
        DaySimulator([r], [[Entry(E_DAY, I_LAI, 10)]])
    with pytest.raises(ValueError, match="season readout"):
        DaySimulator([r], [[Entry(E_STATUS, 0)]], readout="padded")
    with pytest.raises(ValueError, match="season readout"):
        DaySimulator([r], [[Entry(E_DATE, CROP_END)]], readout="padded")
    with pytest.raises(ValueError, match="readout must be one of"):
        DaySimulator([r], [[Entry(E_FINAL, I_GWAD)]], readout="last")
    with pytest.raises(ValueError, match="outside the run's real days"):
        entry_values(season(), [Entry(E_DAY, I_LAI, 31)], last_day=30)
    with pytest.raises(ValueError, match="outside the"):
        entry_values(season(), ENTRIES, last_day=T)
    assert READOUTS == ("season", "padded") and END_NAMES[END_CENSORED] == "censored"


# ------------------------------------------------------------------------ T10: gradients
def test_t10_gradient_is_finite_and_reads_the_readout_day():
    cases = [
        (season(efg_end=20, mature=23), T - 1, END_CENSORED, 23),  # matured
        (season(efg_end=None, fail=14, mature=None), 30, END_CENSORED, 14),  # failed
        (season(efg_end=33, mature=36), 30, END_CENSORED, 30),  # censored
        (season(efg_end=22, mature=28), 25, END_HARVESTED, 25),  # harvested
    ]
    ys = jnp.asarray(np.stack([np.stack([c[0][n] for n in OUT_NAMES]) for c in cases]))
    real = jnp.asarray(np.arange(T) <= 30)  # NaN after the real days: nothing there may enter
    ist = jnp.asarray(np.stack([c[0]["istage"] for c in cases]))
    md = jnp.asarray(np.stack([c[0]["mdate"] for c in cases]))
    lasts = jnp.asarray([min(c[1], 30) for c in cases], jnp.int32)
    opn = jnp.asarray([c[2] for c in cases], jnp.int32)
    ents = [Entry(E_FINAL, I_GWAD), Entry(E_DATE, CROP_END), Entry(E_STATUS, 0)]
    k, o, t = (
        jnp.asarray([[getattr(e, a) for e in ents]] * len(cases), jnp.int32) for a in ("kind", "out", "t")
    )
    ramp = jnp.asarray(1.0 + np.arange(T) / T)

    def loss(a):
        y = jnp.where(real, ys * a * ramp, jnp.nan)  # both branches finite where they carry a derivative
        v = jax.vmap(_gather_season)(y, ist, md, lasts, opn, k, o, t)
        return jnp.sum(v)

    g = float(jax.grad(loss)(1.0))
    expect = sum(value("gwad", c[3]) * (1.0 + c[3] / T) for c in cases)
    assert np.isfinite(g) and g == pytest.approx(expect, rel=1e-6)


# ------------------------------------------------------------------------ T11: the harvest option
def _filex_dict(harvs: str | None, sm: int = 1) -> dict:
    mgmt = {} if harvs is None else {"HARVS": harvs}
    return {
        "TREATMENTS": [{"N": 1, "SM": sm}],
        "SIMULATION CONTROLS": {1: {"MANAGEMENT": mgmt}},
    }


def test_t11_harvest_option_and_the_reported_harvest_check():
    assert harvest_option(_filex_dict("R"), 1) == "R"
    assert harvest_option(_filex_dict("M"), 1) == "M"
    assert harvest_option(_filex_dict(None), 1) == "M"
    assert harvest_option(_filex_dict(""), 1) == "M"
    assert harvest_option(_filex_dict("a"), 1) == "A"
    assert harvest_option(_filex_dict("R", sm=0), 1) == "R"  # no level named: level 1
    days = np.asarray([1982100, 1982101, 1982102])
    check_harvest("X_t01", "R", {"HDAT": 1982102.0}, days)
    check_harvest("X_t01", "M", {"HDAT": float("nan")}, days)
    with pytest.raises(ValueError, match="HARVS = R"):
        check_harvest("X_t01", "R", {"HDAT": 1982101.0}, days)
    with pytest.raises(ValueError, match="HARVS = R"):
        check_harvest("X_t01", "R", {"HDAT": float("nan")}, days)
    other = types.SimpleNamespace(n_days=20, n_ext=60, harvest="A")
    assert FreeRunInputs.last_real_day(other, 60) == 19  # type: ignore[arg-type]
    assert FreeRunInputs.harvest == "M"  # the default of a direct construction


# ------------------------------------------------------------------------ T12-T13: water outputs
def test_t12_water_outputs_are_derived_from_the_day_outputs_on_the_host():
    """``et = evap + ep`` and ``sw<k> = soil_sw[:, k - 1]`` when the outputs carry the day outputs
    (``season_outputs``) instead of the water names; equal to the device gather of the same series."""
    last, nl = 30, 3
    t = np.arange(T, dtype=float)
    base = season(efg_end=20, mature=23)
    day = {k: v for k, v in base.items() if k not in WATER_OUT_NAMES}
    day["evap"] = 0.25 * t + 1.0
    day["ep"] = (0.5 * t + 0.125)[:, None]  # a plant axis of one, as the crop outputs carry
    day["soil_sw"] = 0.01 * t[:, None] + 0.125 * np.arange(1, nl + 1)[None, :]
    i_et, sw = OUT_NAMES.index("et"), {k: OUT_NAMES.index(f"sw{k}") for k in range(1, nl + 1)}
    ents = [
        Entry(E_DAY, i_et, 5),
        Entry(E_DAY, sw[2], 7),
        Entry(E_DAY, sw[3], last),
        Entry(E_FINAL, i_et),
        Entry(E_FINAL, sw[1]),
        Entry(E_FINAL, I_GWAD),
    ]
    host = entry_values(day, ents, last_day=last)
    expect = [
        day["evap"][5] + day["ep"][5, 0],
        day["soil_sw"][7, 1],
        day["soil_sw"][last, 2],
        day["evap"][23] + day["ep"][23, 0],  # the end of the crop: maturity on day 23
        day["soil_sw"][23, 0],
        value("gwad", 23),
    ]
    np.testing.assert_array_equal(host, expect)
    explicit = dict(base)
    explicit["et"] = day["evap"] + day["ep"][:, 0]
    for k in range(1, nl + 1):
        explicit[f"sw{k}"] = day["soil_sw"][:, k - 1]
    np.testing.assert_array_equal(entry_values(explicit, ents, last_day=last), host)
    if jax.config.jax_enable_x64:  # the series are not exact in float32
        np.testing.assert_array_equal(device(explicit, last, END_CENSORED, entries=ents), host)
    with pytest.raises(ValueError, match="outside the run's real days"):
        entry_values(day, [Entry(E_DAY, sw[1], last + 1)], last_day=last)


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the synthetic run is built in float64")
@pytest.mark.allow_skip(reason="float64 synthetic run, skipped in the float32 tier")
def test_t13_season_readout_of_water_entries_on_a_synthetic_run():
    """The simulator's season readout of ``et`` / ``sw<k>`` entries equals the host readout of the
    day's own outputs (``run_sites``, the keys ``evap``, ``ep``, ``soil_sw``); its ``E_DAY`` values are
    those of the padded readout on the real days; an ``E_DAY`` water entry after the run's last real
    day is refused under the season readout and accepted under the padded one."""
    from agrijax.core.runtime import run_sites
    from agrijax.sites.dssat_free_run import _season_model, stack_trees

    from . import test_calib_day_soil as synth

    run = synth._inputs(synth._profile())
    probe = DaySimulator([run], [[Entry(E_FINAL, I_GWAD)]])
    last, n = probe.last_days[0], probe.n_days[(run.nl, run.mesev)]
    assert last < n - 1  # the run has padded days
    i_et = OUT_NAMES.index("et")
    sw = {k: OUT_NAMES.index(f"sw{k}") for k in range(1, run.nl + 1)}
    days = range(0, last + 1, 3)
    day_ents = [Entry(E_DAY, i_et, d) for d in days]
    day_ents += [Entry(E_DAY, sw[k], d) for k in (1, run.nl) for d in days]
    ends = [Entry(E_FINAL, i_et), Entry(E_FINAL, sw[run.nl]), Entry(E_FINAL, I_GWAD)]
    ends += [Entry(E_DATE, CROP_END), Entry(E_STATUS, 0)]
    ents = day_ents + ends
    sim = DaySimulator([run], [ents])
    assert sim.water[(run.nl, run.mesev)] == ("et", "sw1", f"sw{run.nl}")
    theta = np.asarray([[run.published()[c] for c in CUL_ORDER]])
    y = sim(theta, np.zeros(1, np.int64))[0]
    p = run.params()
    o = run_sites(_season_model(run.mesev), *(stack_trees([x]) for x in (p, run.forcing(n), run.state(p))))
    o = {k: np.asarray(v)[0] for k, v in o.items()}
    assert "et" not in o and "sw1" not in o  # the host readout derives them
    host = entry_values(o, ents, last_day=last, harvest=run.harvest)
    # two compiled programs: fusion or FMA contraction may differ in the last bits
    np.testing.assert_allclose(y, host, rtol=1e-12, atol=1e-12)
    assert np.ptp(y[: len(days)]) > 0.0 and np.all(np.isfinite(y))
    padded = DaySimulator([run], [day_ents], readout="padded")
    np.testing.assert_allclose(
        padded(theta, np.zeros(1, np.int64))[0], y[: len(day_ents)], rtol=1e-13, atol=0.0
    )
    with pytest.raises(ValueError, match="outside the run's real days"):
        DaySimulator([run], [[Entry(E_DAY, sw[1], last + 1)]])
    DaySimulator([run], [[Entry(E_DAY, sw[1], last + 1)]], readout="padded")
