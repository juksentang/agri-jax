"""The DSSAT-CSM v4.8.6.0 day (``models/day_dssat486``): declaration, lag check and synthetic runs, no data.

* The day is :data:`agrijax.iface.contract.DSSAT_DAY_TABLE` in ``CSM_Main/LAND.for`` order, with the
  DSSAT day's ports (:data:`~agrijax.iface.contract.DSSAT_PORTS`), and compiles with ``check=True``
  and ``exact_lags=True``: every one of the nine allowed lags (:data:`~agrijax.iface.contract.
  DSSAT_DAY_LAGS`) is used, by the free day (MESEV R and S) and by the replay configuration.
  Removing a lag makes :meth:`Day.check` fail.
* The free day replays nothing of the soil water, the evaporation or the uptake.
* A synthetic free season (the crop conformance season on its nine layers): finite, the crop reads
  its water only from P1 (the crop forcing's ``sw``, ``eop``, ``trwup`` are NaN), the actual
  transpiration is the extraction (``EP = 10 sum(uptake)``, ``EP <= EOP``), the water ledger closes
  to rounding every day, P6 is the crop's LAI.
* With the soil water, ``EOP`` and ``TRWUP`` replayed (:func:`replay_processes` plus
  :func:`trwup_replay_entry`), the assembly is the M2 configuration: its CERES-Maize outputs equal
  the stand-alone crop model's bit for bit.
* The soil-value convention: ``real4`` gives the ``float32`` value promoted, ``input`` the input.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from agrijax.core import run
from agrijax.core.day import DayLagError
from agrijax.core.process import process
from agrijax.core.state import State, field
from agrijax.iface.contract import (
    DSSAT_DAY_LAGS,
    DSSAT_DAY_PHASES,
    DSSAT_DAY_TABLE,
    DSSAT_PORTS,
    dssat_allowed_lags,
    dssat_day_problems,
    dssat_port_spec,
)
from agrijax.iface.crop import CanopyRecord, RootRecord
from agrijax.models.day_dssat486 import (
    SLOT,
    SOIL_EVAPORATION_KEYS,
    DssatSite,
    SoilEvaporation,
    SoilEvaporationError,
    day_dssat486,
    day_outputs,
    day_params,
    day_processes,
    dssat_soil_values,
    initial_state,
    layers_in,
    layers_in_entry,
    replay_processes,
    resolve_soil_evaporation,
    soil_evaporation_gate_problems,
    soil_evaporation_params_problems,
    soil_evaporation_problems,
    trwup_replay_entry,
)
from agrijax.models.entries import noop_entry
from agrijax.processes.crop.ceres_maize import DSSAT_COEFFICIENTS, CeresMaizeState, ceres_maize_model
from agrijax.processes.crop.ceres_maize.phenology import growing_point_thermal_time
from agrijax.processes.pet.spam_dssat import SpamWeather
from agrijax.processes.soil_water.bucket import BucketForcing, BucketState, MulchForcing
from agrijax.processes.soil_water.bucket_evap import (
    SoilEvapState,
    soil_evaporation_esr,
    soil_evaporation_soilev,
)
from agrijax.processes.water_supply import RootwuState, SoilView, rootwu_estimate
from agrijax.testing.conformance._builtin import crop as crop_case
from agrijax.testing.conformance._builtin import day_dssat486 as layers_in_case

USED = {
    ("pet.priestley_taylor", "iface.canopy.maize"),
    ("spam.pse", "iface.canopy.maize"),
    ("spam.mulch_evap", "soil_water.mulch_wat"),
    ("spam.soil_evaporation", "soil_water.theta"),
    ("spam.transpiration", "iface.canopy.maize"),
    ("water_supply.maize.rootwu", "iface.root.maize"),
    ("water_supply.maize.rootwu", "iface.crop_water.maize.sw"),
    ("spam.xtract", "soil_water.theta"),
    ("spam.xtract", "iface.canopy.maize"),
}


def _compiled(mesev="R", replace=None, soil_evaporation=None):
    day = day_dssat486(SLOT)
    if soil_evaporation is None:
        procs = day_processes(SLOT, mesev=mesev, replace=replace)
    else:
        procs = day_processes(SLOT, soil_evaporation=soil_evaporation, replace=replace)
    return day, day.compile(procs, outputs=day_outputs(SLOT), check=True, exact_lags=True)


def _covered(report, day) -> set[tuple[str, str]]:
    """The allowed lags the model's lagged reads use (each read folded onto its covering lag)."""
    out = set()
    for r, p in report.used:
        lag = next(lag for lag in day.lags if lag.covers(r, p))
        out.add(lag.pair)
    return out


def test_table_is_consistent() -> None:
    assert dssat_day_problems() == []
    assert len(DSSAT_DAY_LAGS) == len(USED) == 9
    assert set(DSSAT_PORTS) == {"P1", "P4", "P5", "P6", "P7", "P9"} | {f"PD{i}" for i in range(1, 8)}
    assert dssat_port_spec("P2").path == "iface.root.{slot}"  # unchanged ports come from PORTS
    # potential rates in P5, actual ones in PD1
    assert {n for n, _ in DSSAT_PORTS["P5"].fields} == {
        "eo_priestley_taylor",
        "soil_evaporation",
        "transpiration",
    }
    assert "evaporation" in {n for n, _ in DSSAT_PORTS["PD1"].fields}
    assert all(e.status != "replay" for e in DSSAT_DAY_TABLE)


def test_day_follows_land_for() -> None:
    day, model = _compiled()
    assert day.ref == "dssat-4.8.6.0"
    assert tuple(p.name for p in day.phases) == DSSAT_DAY_PHASES
    assert day.entries == tuple(e.name(SLOT) for e in DSSAT_DAY_TABLE) == model.names
    rate = next(p for p in day.phases if p.name == "rate").entries
    integr = next(p for p in day.phases if p.name == "integr").entries
    # LAND.for RATE: SOIL (SOILDYN, then WATBAL), then SPAM (PET, PSE, MULCH_EVAP, SOILEV, TRANS,
    # ROOTWU, EP / XTRACT)
    assert rate[:2] == ("soil_water.albedo", "soil_water.rate") and rate[-1] == "spam.xtract"
    spam = [e for e in rate if e.startswith(("pet.", "spam."))]
    assert spam == [
        "pet.priestley_taylor",
        "spam.pse",
        "spam.mulch_evap",
        "spam.soil_evaporation",
        "spam.transpiration",
        "spam.xtract",
    ]
    assert integr[:2] == ("soil_water.integrate", "crops.maize.layers_in")
    assert integr[2:7] == tuple(
        f"crops.maize.{n}" for n in ("phenology", "stress", "growth", "roots", "publish")
    )
    assert integr[-1] == "crops.maize.canopy" and day.entries[-1] == "ledger.close"


@pytest.mark.parametrize("config", ["R", "S", "replay"])
def test_lag_check_uses_every_allowed_lag(config) -> None:
    replace = replay_processes(SLOT) if config == "replay" else None
    day, model = _compiled("R" if config == "replay" else config, replace)
    report = day.check(model, exact_lags=True)
    assert _covered(report, day) == USED and report.unused_pairs == ()
    assert {lag.pair for lag in day.lags} == {lag.pair for lag in dssat_allowed_lags(SLOT)}
    # the soil module's SOILDYN / RATE / INTEGR entries are one module: SW is carried, not a lag
    carried = set(day.carried_reads(model))
    assert ("soil_water.rate", "soil_water.sw") in carried
    assert ("soil_water.albedo", "soil_water.theta") in carried


@pytest.mark.parametrize("pair", sorted(USED))
def test_removing_a_lag_fails_the_check(pair) -> None:
    day, model = _compiled()
    lags = tuple(lag for lag in day.lags if lag.pair != pair)
    assert len(lags) == len(day.lags) - 1
    with pytest.raises(DayLagError):
        dataclasses.replace(day, lags=lags).check(model)


def test_free_day_replays_nothing() -> None:
    procs = day_processes(SLOT)
    assert not [k for k, v in procs.items() if v.source.startswith("replay")]
    reps = replay_processes(SLOT)
    assert all(v.source.startswith("replay") for v in reps.values())
    assert set(reps) < set(procs)
    with pytest.raises(KeyError):
        day_processes(SLOT, replace={"no.such.entry": procs["ledger.close"]})
    with pytest.raises(ValueError):
        day_processes(SLOT, mesev="X")


def test_soil_value_convention() -> None:
    x = np.array([0.026, 0.3, 15.0])
    r4 = dssat_soil_values(x, "real4")
    assert np.array_equal(r4, x.astype(np.float32).astype(np.float64))
    assert r4[0] != 0.026 and r4[2] == 15.0
    assert np.array_equal(dssat_soil_values(x, "input"), x)
    with pytest.raises(ValueError):
        dssat_soil_values(x, "float16")


# ------------------------------------------------------------------ synthetic runs
N_DAYS = 90
YRPLT = crop_case.START + 5


def _site(mesev: str = "R") -> DssatSite:
    dl = np.asarray(crop_case.DLAYR)
    return DssatSite(
        dlayr=dl,
        ds=np.cumsum(dl),
        ll=np.asarray(crop_case.LL),
        dul=np.asarray(crop_case.DUL),
        sat=np.asarray(crop_case.SAT),
        swcn=np.zeros(dl.size),
        cn=72.0,
        swcon=0.5,
        salb=0.13,
        u=6.0,
        mesev=mesev,
        meinf="S",
    )


def _free_inputs(
    mesev: str = "R",
    soil_values: str = "real4",
    cold_spell: bool = False,
    soil_evaporation: str | None = None,
):
    """The crop conformance season with a free soil: weather, rain, a mulch record; the crop
    forcing's water fields NaN. ``cold_spell`` puts :data:`COLD_DAYS` of snowfall and then
    :data:`THAW_DAYS` of a melting pack right after sowing (see :func:`_cold_spell`)."""
    dtype = jnp.float64
    p = crop_case.params(dtype, YRPLT)
    f = crop_case.weather(7, dtype, N_DAYS)
    rng = np.random.default_rng(3)
    params = day_params(
        _site(mesev),
        p,
        ksevap=0.6854839,
        ktrans=0.6854839,
        soil_values=soil_values,
        soil_evaporation=soil_evaporation,
    )
    rain = np.where(rng.uniform(size=N_DAYS) < 0.25, rng.uniform(2.0, 30.0, N_DAYS), 0.0)
    if cold_spell:
        f, rain = _cold_spell(f, rain)

    def a(x):
        return jnp.asarray(np.asarray(x, dtype=float), dtype)

    mulch = MulchForcing(
        mass=a(np.full(N_DAYS, 1500.0)),
        cover=a(np.full(N_DAYS, 0.35)),
        new_mass=a(np.zeros(N_DAYS)),
        watfac=a(np.full(N_DAYS, 3.8)),
    )
    bf = BucketForcing(rain=a(rain), tmax=f.tmax, irrigation=a(np.zeros(N_DAYS)), mulch=mulch)
    tavg = (np.asarray(f.tmax) + np.asarray(f.tmin)) / 2.0
    weather = SpamWeather(
        tavg=a(tavg), wind_run=a(np.full(N_DAYS, 150.0)), co2=f.co2, srad=f.srad, tmax=f.tmax, tmin=f.tmin
    )
    nan = lambda x: jnp.full_like(x, jnp.nan)  # noqa: E731
    fc = f.replace(sw=nan(f.sw), eop=nan(f.eop), trwup=nan(f.trwup))
    forcing = {
        "crop": fc,
        "soil": bf,
        "spam": {
            "weather": weather,
            "mulch_am": a(np.full(N_DAYS, 30.0)),
            "mulch_extfac": a(np.full(N_DAYS, 0.86)),
        },
    }
    soil = params["soil"].soil
    sw0 = soil.ll + 0.7 * (soil.dul - soil.ll)
    state = initial_state(
        bucket=BucketState.initial(sw0, dtype=dtype),
        soil_evap=SoilEvapState.initial(sw0, soil.dlayr, soil.ds, soil.dul, soil.ll, params["evap"].u),
        crop=CeresMaizeState.initial(params["crop"], 1),
        rootwu=RootwuState.initial(1, int(sw0.shape[-1]), dtype),
        salb=params["albedo"].salb,
        storage0=jnp.sum(sw0 * soil.dlayr),
    )
    return params, forcing, state


#: day indices of the synthetic cold spell (sowing is day 5): snowfall (TMAX <= 1 degC), then a
#: melting pack under warm days with sub-zero nights
COLD_DAYS = (6, 7, 8)
THAW_DAYS = tuple(range(9, 16))
COLD_TMAX, COLD_TMIN, COLD_PRECIP = 0.5, -6.0, 30.0
THAW_TMAX, THAW_TMIN = 20.0, -3.0


def _cold_spell(f, rain):
    """The weather of :func:`_free_inputs` with the cold spell: 90 mm of precipitation on the
    cold days (WATBAL keeps it as snow), then days of 20 / -3 degC, on which the pack melts by
    ``SNOMLT = TMAX`` = 20 mm d-1 (WBSUBS.for:48) and lasts four days."""
    tmax, tmin, rain = np.array(f.tmax), np.array(f.tmin), np.array(rain)
    for k in COLD_DAYS:
        tmax[k], tmin[k], rain[k] = COLD_TMAX, COLD_TMIN, COLD_PRECIP
    for k in THAW_DAYS:
        tmax[k], tmin[k], rain[k] = THAW_TMAX, THAW_TMIN, 0.0
    return f.replace(tmax=jnp.asarray(tmax), tmin=jnp.asarray(tmin)), rain


def _run(model, params, forcing, state):
    return jax.jit(lambda p, f, s: run(model, p, f, s, return_final=True))(params, forcing, state)


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the closure is checked in float64")
@pytest.mark.allow_skip(reason="float64 comparison, skipped in the float32 tier")
@pytest.mark.parametrize("mesev", ["R", "S"])
def test_free_season_closes_and_reads_only_the_ports(mesev) -> None:
    params, forcing, state = _free_inputs(mesev)
    _, model = _compiled(mesev)
    final, out = _run(model, params, forcing, state)
    for k in ("lai", "cwad", "soil_sw", "es", "ep", "eo", "eop", "msalb", "p1_trwup"):
        assert np.all(np.isfinite(np.asarray(out[k]))), k
    assert float(np.max(np.asarray(out["lai"]))) > 0.5  # the crop grew on the free soil
    ep = np.asarray(out["ep"])
    np.testing.assert_allclose(ep, 10.0 * np.asarray(out["uptake"]).sum(-1), rtol=1e-12, atol=1e-15)
    assert np.all(ep <= np.asarray(out["eop"]) + 1e-12) and ep.max() > 0.0
    # P6 is the crop's end-of-day LAI; the albedo moves with the top layer
    np.testing.assert_array_equal(np.asarray(out["p6_lai"]), np.asarray(out["lai"]))
    # P6 height is the crop's CANHT in cm (100 CANHT, MZ_GROSUB.for:1818-1831)
    canopy = final["iface"]["canopy"][SLOT]
    canht = np.asarray(final["crops"][SLOT].growth.canht)
    assert float(canht.max()) > 0.0
    np.testing.assert_allclose(np.asarray(canopy.height), 100.0 * canht, rtol=1e-15, atol=0.0)
    # P9: the crop's SNOW is the bucket's snow pack after WATBAL RATE (LAND.for:386)
    np.testing.assert_array_equal(np.asarray(out["p9_swe"]), np.asarray(out["snow"]))
    assert np.ptp(np.asarray(out["msalb"])) > 0.0
    led = final["ledger"]["water"]
    assert float(led.max_abs_residual) < 1e-11, float(led.max_abs_residual)
    assert abs(float(led.closure())) < 1e-10


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the closure is checked in float64")
@pytest.mark.allow_skip(reason="float64 comparison, skipped in the float32 tier")
def test_rootwu_reads_yesterdays_record_and_the_ledger_closes() -> None:
    """ROOTWU on day d reads the root record CERES published on day d-1 and the soil water at the
    end of day d-1 (the allowed lags of SPAM.for:281): recomputed day by day from the free day's own
    records, the P1 ``trwup`` is the same; the day-0 call sees the empty initial record."""
    params, forcing, state = _free_inputs()
    _, model = _compiled()
    final, out = _run(model, params, forcing, state)
    sw_prev = np.asarray(out["p1_sw"])[:-1]
    trwup = np.asarray(out["p1_trwup"])[1:, 0]
    assert np.all(np.isfinite(np.asarray(out["p1_trwup"]))) and trwup.max() > 0.0
    root = jax.jit(
        lambda: run(
            day_dssat486(SLOT).compile(day_processes(SLOT), outputs=lambda s, p, f: s["iface"]["root"][SLOT]),
            params,
            forcing,
            state,
        )
    )()
    rw = params["rootwu"]
    tss = jnp.zeros((1, sw_prev.shape[1]))
    want = []
    for k in range(sw_prev.shape[0]):
        one = RootRecord(*(x[k] for x in (root.rlv, root.rtdep, root.rwumx, root.pormin, root.xhlai)))
        r = rootwu_estimate(one, SoilView(rw.dlayr, rw.ll, rw.sat, jnp.asarray(sw_prev[k])), tss)
        tss = r.tss
        want.append(float(r.trwup[0]))
    np.testing.assert_allclose(trwup, want, rtol=1e-12, atol=1e-15)
    assert float(np.asarray(out["p1_trwup"])[0, 0]) == 0.0
    led = final["ledger"]["water"]
    assert float(led.max_abs_residual) < 1e-11, float(led.max_abs_residual)
    assert abs(float(led.closure())) < 1e-10


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the thermal time is checked in float64")
@pytest.mark.allow_skip(reason="float64 comparison, skipped in the float32 tier")
def test_snow_reaches_the_crop_through_p9_and_sets_the_crown_temperature() -> None:
    """A cold spell right after sowing: WATBAL keeps the precipitation of the cold days as snow,
    ``layers_in`` hands the pack to the crop (P9 ``swe``), and MZ_PHENOL's thermal time of a
    growing point below ground takes the snow branch: with ``TMIN < 0 < TMAX`` the crown
    temperatures are ``TEMPCN = 2 + TMIN (0.4 + 0.0018 (min(SNOW, 15) - 15)^2)`` and
    ``TEMPCX = TMAX``, ``DTT = (TEMPCN + TEMPCX) / 2 - TBASE`` (MZ_PHENOL.for:404-458), in place of
    the soil-temperature branch the same weather gives without snow."""
    params, forcing, state = _free_inputs(cold_spell=True)
    _, model = _compiled()
    _, out = _run(model, params, forcing, state)
    snow, swe, dtt = (np.asarray(out[k]) for k in ("snow", "p9_swe", "dttd"))
    dtt = dtt[:, 0]
    np.testing.assert_array_equal(swe, snow)
    assert np.all(swe[list(COLD_DAYS)] > 0.0) and swe[COLD_DAYS[-1]] == pytest.approx(3 * COLD_PRECIP)
    under = [k for k in THAW_DAYS if swe[k] > 0.0]
    assert len(under) >= 3 and swe[THAW_DAYS[-1]] == 0.0  # the pack melts away
    np.testing.assert_array_equal(dtt[list(COLD_DAYS)], 0.0)  # TMAX < TBASE
    cul = params["crop"].cultivar
    tbase = float(cul.tbase)
    f = forcing["crop"]
    for k in under:
        tmn, tmx = float(f.tmin[k]), float(f.tmax[k])
        snowfac = 0.4 + 0.0018 * (min(float(swe[k]), 15.0) - 15.0) ** 2
        want = (2.0 + tmn * snowfac + tmx) / 2.0 - tbase
        assert dtt[k] == pytest.approx(max(want, 0.0), rel=1e-12, abs=1e-12), k
        # the soil branch of the same day (no snow) gives another value
        bare = growing_point_thermal_time(
            f.tmax[k], f.tmin[k], f.srad[k], f.dayl[k], 0.0, cul.tbase, cul.topt, DSSAT_COEFFICIENTS.phenol
        )
        assert abs(float(bare) - dtt[k]) > 0.1, (k, float(bare), dtt[k])
    # once the pack is gone the soil branch is back
    k = next(k for k in THAW_DAYS if swe[k] == 0.0)
    bare = growing_point_thermal_time(
        f.tmax[k], f.tmin[k], f.srad[k], f.dayl[k], 0.0, cul.tbase, cul.topt, DSSAT_COEFFICIENTS.phenol
    )
    assert dtt[k] == pytest.approx(float(bare), rel=1e-12)


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="bit-identity is checked in float64")
@pytest.mark.allow_skip(reason="float64 comparison, skipped in the float32 tier")
def test_m2_configuration_is_the_standalone_crop_bit_for_bit() -> None:
    params, forcing, state = _free_inputs(soil_values="input")
    p = params["crop"]
    f = crop_case.weather(7, jnp.float64, N_DAYS)
    rng = np.random.default_rng(5)
    nl = int(p.soil.dlayr.shape[0])
    z = jnp.zeros((N_DAYS, 1))
    zd = jnp.zeros(N_DAYS)
    forcing = {
        **forcing,
        "replay": {
            "soil": {"runoff": zd, "drain": zd, "residue_water": zd, "snow": zd, "sw": f.sw, "mulch_wat": zd},
            "spam": {
                "eo": f.eop * 1.2,
                "es": jnp.asarray(rng.uniform(0.0, 0.3, N_DAYS)),
                "em": zd,
                "eop": f.eop[:, None],
                "uptake": jnp.asarray(rng.uniform(0.0, 0.05, (N_DAYS, nl))),
            },
            "canopy": CanopyRecord(lai=z + 1.0, tlai=z + 1.0, height=z),
            "trwup": f.trwup[:, None],
        },
    }
    # the replayed soil and fluxes are not a water balance: the ledger is not part of this check
    replace = {
        **replay_processes(SLOT),
        f"water_supply.{SLOT}.rootwu": trwup_replay_entry(SLOT),
        "ledger.close": noop_entry("ledger.close", why="the replayed fluxes of this test are not a balance"),
    }
    _, model = _compiled(replace=replace)
    ref = jax.jit(lambda pp, ff, s: run(ceres_maize_model(), pp, ff, s))(p, f, CeresMaizeState.initial(p, 1))
    _, out = _run(model, params, forcing, state)
    assert float(np.max(np.asarray(ref["lai"]))) > 0.5
    for k, v in ref.items():
        np.testing.assert_array_equal(np.asarray(out[k]), np.asarray(v), err_msg=k)


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="bit-identity is checked in float64")
@pytest.mark.allow_skip(reason="float64 comparison, skipped in the float32 tier")
def test_free_season_bit_identical_with_layer_loops_unrolled() -> None:
    """``depth_unroll`` (agrijax.core.execution) is a layout of the program, not a model change: the
    free season with the bucket's layer recurrences unrolled (the GPU default) and as loops (the
    CPU default) gives every output and the final state bit for bit on this backend."""
    from agrijax.core.execution import execution

    params, forcing, state = _free_inputs("R")
    _, model = _compiled("R")
    res, whiles = {}, {}
    for on in (False, True):
        with execution(depth_unroll=on):
            fn = lambda p, f, s: run(model, p, f, s, return_final=True)  # noqa: E731 (a fresh trace)
            whiles[on] = jax.jit(fn).lower(params, forcing, state).as_text().count("stablehlo.while")
            res[on] = jax.jit(lambda p, f, s: run(model, p, f, s, return_final=True))(params, forcing, state)
    assert whiles[False] > whiles[True], whiles  # two different programs
    a, b = jax.tree_util.tree_leaves_with_path(res[False]), jax.tree_util.tree_leaves(res[True])
    assert len(a) == len(b)
    for (path, x), y in zip(a, b, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=jax.tree_util.keystr(path))


def test_layers_in_adapter_writes_only_p1_sw_and_p9_swe() -> None:
    """The adapter of row D10 (its conformance case has no slot contract): it reads the soil's
    ``theta`` (P7) and snow pack (PD7), writes only P1 ``sw`` and P9 ``swe``, and its conformance
    case binds exactly the ports it uses, at the DSSAT day's paths."""
    assert layers_in.reads == ("theta", "snow")
    assert layers_in.writes == ("water_out.sw", "snow_out.swe")
    entry = layers_in_entry(SLOT)
    assert entry.reads == ("soil_water.theta", "soil_water.snow")
    assert entry.writes == (f"iface.crop_water.{SLOT}.sw", "iface.snow.swe")
    (case,) = layers_in_case.cases()
    used = {p.split(".", 1)[0] for p in (*layers_in.reads, *layers_in.writes)}
    assert set(case.ports) == used
    paths = {k: v.format(slot=SLOT) for k, v in case.ports.items()}
    assert paths == {
        "theta": dssat_port_spec("P7").global_path(SLOT),
        "snow": dssat_port_spec("PD7").global_path(SLOT),
        "water_out": dssat_port_spec("P1").global_path(SLOT),
        "snow_out": dssat_port_spec("P9").global_path(SLOT),
    }
    # the day's other records pass through: only sw and swe change, to theta and snow
    s0, _, f = case.make(np.random.default_rng(0), jnp.float64, "nominal")
    s1 = layers_in(s0, None, jax.tree.map(lambda x: x[0], f))
    np.testing.assert_array_equal(np.asarray(s1.water_out.sw), np.asarray(s0.theta))
    assert float(s1.snow_out.swe) == float(s0.snow)
    for a, b in ((s0.water_out.eop, s1.water_out.eop), (s0.water_out.trwup, s1.water_out.trwup)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    for name in ("melt", "melt_runoff", "sublimation"):
        assert float(getattr(s1.snow_out, name)) == float(getattr(s0.snow_out, name))
    np.testing.assert_array_equal(np.asarray(s1.theta), np.asarray(s0.theta))
    assert float(s1.snow) == float(s0.snow)


# ------------------------------------------------------------------------ soil evaporation swap
# The soil evaporation is chosen by registry key, checked against the
# entry's interface (units and dims of the SPAM store, reads, writes, grid, slot) before any trace.
_STANDIN = {
    "provenance": "equations_only",
    "sources": (("stand-in of the swap tests", "tests/unit/test_day_dssat486.py"),),
    "grid": "point",
    "deviates": (),
    "register": False,
}


class CmEvapState(State):
    """A stand-in store whose soil evaporation is in cm d-1 (the SPAM store holds mm d-1)."""

    eos_soil: Array = field(unit="mm d-1", dims=())
    em: Array = field(unit="mm d-1", dims=())
    es: Array = field(unit="cm d-1", dims=())
    evap: Array = field(unit="mm d-1", dims=())


@process(reads=("eos_soil",), writes=("es", "evap"), key="soil_water/es_in_cm@none:demo", **_STANDIN)
def _es_in_cm(state: CmEvapState, params: Any, forcing_t: Any) -> CmEvapState:
    """Wrong unit: ES in cm d-1. Source: test stand-in."""
    return state.replace(es=state.eos_soil / 10.0, evap=state.em + state.eos_soil)


@process(
    reads=("pet.soil_evaporation",),
    writes=("es", "evap", "sw"),
    key="soil_water/reads_p5@none:demo",
    **_STANDIN,
)
def _reads_p5(state: SoilEvapState, params: Any, forcing_t: Any) -> SoilEvapState:
    """Wrong ports: reads P5 (not bound for the entry) and writes the soil water. Source: test stand-in."""
    return state


@process(
    reads=("eos_soil",),
    writes=("es", "evap", "es_lyr"),
    key="soil_water/half_layered@none:demo",
    **_STANDIN,
)
def _half_layered(state: SoilEvapState, params: Any, forcing_t: Any) -> SoilEvapState:
    """Writes ES_LYR without SWDELTU. Source: test stand-in."""
    return state


@process(
    reads=("eos_soil",),
    writes=("es",),
    key="pet/no_evap@none:demo",
    **{**_STANDIN, "grid": "rzwqm2_nodes"},
)
def _wrong_slot(state: SoilEvapState, params: Any, forcing_t: Any) -> SoilEvapState:
    """Wrong slot and grid, no EVAP. Source: test stand-in."""
    return state


@process(reads=("eos_soil",), writes=("es", "evap"), register=False)
def _unkeyed(state: SoilEvapState, params: Any, forcing_t: Any) -> SoilEvapState:
    """No registry key. Source: test stand-in."""
    return state


@process(reads=("em",), writes=("es", "evap"), key="soil_water/no_soil_evaporation@none:demo", **_STANDIN)
def _no_soil_evaporation(state: SoilEvapState, params: Any, forcing_t: Any) -> SoilEvapState:
    """A conforming stand-in: no soil evaporation (ES = 0, EVAP = EM). Source: test stand-in."""
    return state.replace(es=jnp.zeros_like(state.es), evap=state.em)


def test_soil_evaporation_keys_name_the_validated_processes() -> None:
    assert SOIL_EVAPORATION_KEYS == {"R": soil_evaporation_soilev.key, "S": soil_evaporation_esr.key}
    for mesev, layered in (("R", False), ("S", True)):
        impl = resolve_soil_evaporation(SOIL_EVAPORATION_KEYS[mesev])
        assert impl.layer_removal is layered and impl.mesev == mesev
        assert resolve_soil_evaporation(mesev) == impl
        assert soil_evaporation_problems(impl.process) == []


@pytest.mark.parametrize("mesev", ["R", "S"])
def test_swap_by_key_is_the_mesev_day_and_passes_every_check(mesev) -> None:
    """The key gives the day the MESEV switch gave: same entries, reads, writes and sources; the
    compiled day passes Day.check (exact lags, owners, phased writes) and the contract checks."""
    key = SOIL_EVAPORATION_KEYS[mesev]
    by_key = day_processes(SLOT, soil_evaporation=key)
    by_mesev = day_processes(SLOT, mesev=mesev)
    assert list(by_key) == list(by_mesev)
    for k in by_key:
        a, b = by_key[k], by_mesev[k]
        assert (a.name, a.reads, a.writes, a.source) == (b.name, b.reads, b.writes, b.source), k
    assert key in by_key["spam.soil_evaporation"].source
    day, model = _compiled(soil_evaporation=key)
    report = day.check(model, exact_lags=True)
    assert _covered(report, day) == USED
    assert day.owner_problems(model) == [] and day.write_phase_problems(model) == []
    assert dssat_day_problems() == []


def test_the_swap_changes_one_entry_only() -> None:
    ritchie = day_processes(SLOT, soil_evaporation=SOIL_EVAPORATION_KEYS["R"])
    salus = day_processes(SLOT, soil_evaporation=SOIL_EVAPORATION_KEYS["S"])
    changed = [k for k in ritchie if ritchie[k].source != salus[k].source]
    assert changed == ["spam.soil_evaporation"]
    e_r, e_s = ritchie["spam.soil_evaporation"], salus["spam.soil_evaporation"]
    assert (e_r.reads, e_r.writes) == (e_s.reads, e_s.writes)  # the entry's ports do not move


@pytest.mark.parametrize(
    ("standin", "expected"),
    [
        (_es_in_cm, ["field 'es' is in 'cm d-1'", "holds 'mm d-1'"]),
        (_reads_p5, ["reads 'pet.soil_evaporation'", "writes 'sw'"]),
        (_half_layered, ["writes ['es_lyr'] of ('es_lyr', 'swdeltu')"]),
        (_wrong_slot, ["slot 'pet'", "grid 'rzwqm2_nodes'", "does not write ['evap']"]),
        (_unkeyed, ["no registry key"]),
    ],
)
def test_an_incompatible_implementation_is_rejected_with_its_reasons(standin, expected) -> None:
    problems = "\n".join(soil_evaporation_problems(standin))
    for text in expected:
        assert text in problems, (text, problems)
    with pytest.raises(SoilEvaporationError, match=r"cannot fill spam\.soil_evaporation"):
        day_processes(SLOT, soil_evaporation=standin)


def test_unknown_or_ambiguous_choices_are_rejected() -> None:
    with pytest.raises(SoilEvaporationError, match="no registered process"):
        day_processes(SLOT, soil_evaporation="soil_water/no_such@dssat-4.8.6.0:faithful")
    with pytest.raises(SoilEvaporationError, match="not in"):
        day_processes(SLOT, soil_evaporation="X")
    with pytest.raises(TypeError, match="not both"):
        day_processes(SLOT, mesev="R", soil_evaporation=SOIL_EVAPORATION_KEYS["S"])


def test_params_built_for_the_other_implementation_are_rejected_at_trace() -> None:
    """The bucket and XTRACT would book a SALUS evaporation from the top layer (or the reverse)
    and the ledger would still close: the static check stops it before the first step."""
    params, forcing, state = _free_inputs("R")
    _, model = _compiled(soil_evaporation=SOIL_EVAPORATION_KEYS["S"])
    with pytest.raises(SoilEvaporationError, match="day_params"):
        _run(model, params, forcing, state)


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the closure is checked in float64")
@pytest.mark.allow_skip(reason="float64 comparison, skipped in the float32 tier")
def test_swapped_season_is_the_other_mesev_bit_for_bit_and_closes() -> None:
    """A site with MESEV = R run with the SALUS key is the MESEV = S site's day bit for bit (the
    site's MESEV is only the default choice); a conforming stand-in also compiles and closes."""
    p_s, f_s, s_s = _free_inputs("S")
    _, m_s = _compiled("S")
    _, out_s = _run(m_s, p_s, f_s, s_s)
    key = SOIL_EVAPORATION_KEYS["S"]
    p_x, f_x, s_x = _free_inputs("R", soil_evaporation=key)
    _, m_x = _compiled(soil_evaporation=key)
    final, out_x = _run(m_x, p_x, f_x, s_x)
    for k in out_s:
        np.testing.assert_array_equal(np.asarray(out_x[k]), np.asarray(out_s[k]), err_msg=k)
    assert float(final["ledger"]["water"].max_abs_residual) < 1e-11
    p_r, f_r, s_r = _free_inputs("R")
    _, out_r = _run(_compiled("R")[1], p_r, f_r, s_r)
    assert float(np.sum(out_r["es"])) != float(np.sum(out_s["es"]))
    p_z, f_z, s_z = _free_inputs("R", soil_evaporation=_no_soil_evaporation)
    day, m_z = _compiled(soil_evaporation=_no_soil_evaporation)
    day.check(m_z, exact_lags=True)
    final_z, out_z = _run(m_z, p_z, f_z, s_z)
    assert float(np.max(np.abs(np.asarray(out_z["es"])))) == 0.0
    assert float(final_z["ledger"]["water"].max_abs_residual) < 1e-11


# ---- construction, sub-path writes, dims, the conformance gate, the coupling flags
_ESR = soil_evaporation_esr
_ESR_KW = {"reads": _ESR.reads, "writes": _ESR.writes, **_STANDIN, "grid": "dssat_layers"}


class NoDimsState(State):
    """A stand-in store whose ES declares no dims."""

    eos_soil: Array = field(unit="mm d-1", dims=())
    em: Array = field(unit="mm d-1", dims=())
    es: Array = field(unit="mm d-1")
    evap: Array = field(unit="mm d-1", dims=())


@process(reads=("eos_soil",), writes=("es", "evap"), key="soil_water/no_dims@none:demo", **_STANDIN)
def _no_dims(state: NoDimsState, params: Any, forcing_t: Any) -> NoDimsState:
    """ES without dims. Source: test stand-in."""
    return state


@process(
    reads=("eos_soil",),
    writes=("es", "evap", "es_lyr.x", "swdeltu.y"),
    key="soil_water/sub_paths@none:demo",
    **_STANDIN,
)
def _sub_paths(state: SoilEvapState, params: Any, forcing_t: Any) -> SoilEvapState:
    """Sub-path writes of the layer fields. Source: test stand-in."""
    return state


@process(key="soil_water/esr_flipped@none:demo", **_ESR_KW)
def _esr_flipped(state: SoilEvapState, params: Any, forcing_t: Any) -> SoilEvapState:
    """ESR with the sign of SWDELTU flipped (the soil would gain the water). Source: test stand-in."""
    new = _ESR(state, params, forcing_t)
    return new.replace(swdeltu=-new.swdeltu)


@process(key="soil_water/esr_half_es@none:demo", **_ESR_KW)
def _esr_half_es(state: SoilEvapState, params: Any, forcing_t: Any) -> SoilEvapState:
    """ESR with ES halved (ES_LYR unchanged). Source: test stand-in."""
    new = _ESR(state, params, forcing_t)
    return new.replace(es=new.es / 2.0, evap=new.es / 2.0 + new.em)


@process(key="soil_water/es_above_eos@none:demo", reads=("eos_soil", "em"), writes=("es", "evap"), **_STANDIN)
def _es_above_eos(state: SoilEvapState, params: Any, forcing_t: Any) -> SoilEvapState:
    """ES three times the potential. Source: test stand-in."""
    return state.replace(es=3.0 * state.eos_soil, evap=3.0 * state.eos_soil + state.em)


def test_the_built_in_implementations_pass_the_gate() -> None:
    assert soil_evaporation_gate_problems(soil_evaporation_soilev) == []
    assert soil_evaporation_gate_problems(soil_evaporation_esr) == []
    assert soil_evaporation_gate_problems(_no_soil_evaporation) == []


def test_a_checked_implementation_cannot_be_built_around_the_checks() -> None:
    assert SoilEvaporation(soil_evaporation_esr).layer_removal is True
    with pytest.raises(SoilEvaporationError):
        SoilEvaporation(_reads_p5)
    with pytest.raises(TypeError):
        SoilEvaporation(soil_evaporation_esr, layer_removal=False)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("standin", "expected"),
    [
        (_no_dims, ["field 'es' has dims None"]),
        (_sub_paths, ["writes 'es_lyr.x': a sub-path", "writes 'swdeltu.y': a sub-path"]),
        (_esr_flipped, ["ES_LYR = -SWDELTU DLAYR 10 fails"]),
        (_esr_half_es, ["ES = sum(ES_LYR) fails"]),
        (_es_above_eos, ["ES <= EOS_SOIL fails"]),
    ],
)
def test_wrong_layer_semantics_are_rejected(standin, expected) -> None:
    with pytest.raises(SoilEvaporationError) as err:
        resolve_soil_evaporation(standin)
    for text in expected:
        assert text in str(err.value), (text, str(err.value))
    assert not _layer_removal_of(_sub_paths)


def _layer_removal_of(proc) -> bool:
    return {"es_lyr", "swdeltu"} <= {p for p in proc.writes if "." not in p}


def test_the_coupling_flags_themselves_are_checked() -> None:
    params, _, _ = _free_inputs("R")
    salus = SOIL_EVAPORATION_KEYS["S"]
    assert soil_evaporation_params_problems(params, SOIL_EVAPORATION_KEYS["R"]) == []
    got = "\n".join(soil_evaporation_params_problems(params, salus))
    assert "layer_removal" in got and "params['soil'].salus_es" in got and "params['xtract'].salus_es" in got
    # the static record removed and one flag edited by hand: the flag is still caught on the host
    edited = {**params, "evap": params["evap"].replace(layer_removal=None)}
    edited["soil"] = edited["soil"].replace(salus_es=jnp.asarray(1.0))
    got = soil_evaporation_params_problems(edited, SOIL_EVAPORATION_KEYS["R"])
    assert len(got) == 1 and "params['soil'].salus_es" in got[0]


def test_unstated_params_are_rejected_inside_jit() -> None:
    params, forcing, state = _free_inputs("R")
    params = {**params, "evap": params["evap"].replace(layer_removal=None)}
    _, model = _compiled("R")
    with pytest.raises(SoilEvaporationError, match="cannot be checked"):
        _run(model, params, forcing, state)


class _UncheckedSoilEvaporation(SoilEvaporation):
    """A subclass that skips the checks."""

    def __post_init__(self) -> None:
        pass


def test_resolve_checks_a_given_instance_again() -> None:
    with pytest.raises(SoilEvaporationError):
        resolve_soil_evaporation(_UncheckedSoilEvaporation(_reads_p5))
    with pytest.raises(SoilEvaporationError):
        day_processes(SLOT, soil_evaporation=_UncheckedSoilEvaporation(_esr_flipped))
    bare = object.__new__(SoilEvaporation)
    with pytest.raises(SoilEvaporationError, match="without a process"):
        resolve_soil_evaporation(bare)
    object.__setattr__(bare, "process", _reads_p5)
    with pytest.raises(SoilEvaporationError):
        resolve_soil_evaporation(bare)
    ok = resolve_soil_evaporation(_UncheckedSoilEvaporation(soil_evaporation_esr))
    assert type(ok) is SoilEvaporation and ok.layer_removal is True
