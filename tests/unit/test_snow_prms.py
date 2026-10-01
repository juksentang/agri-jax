"""``agrijax.processes.snow`` (PRMS snowpack of RZWQM2 4.6): data-free checks against independent references.

The comparison with the reference model (dump tables, ``.ana`` columns 92 and 105, fresh
scenario runs) is ``tests/integration/test_snow_prms_reference.py``. Here:

* conservation of water: ``dSWE = intercepted - melt - melt runoff - sublimation`` every day of
  random winters (a batch of sites in one ``vmap``);
* the latent heat of fusion: a heat gain above the cold content melts ``(Q - PK_DEF) / 203.2`` inch
  of the pack (80 cal g-1 over one inch of water);
* closed-form days: a cold snowfall on bare ground keeps all its water but the sublimation
  ``POTET_SUBLIM x 0.011 cm`` under full cover; a thin isothermal pack on a warm day melts
  completely and splits into ``FRAC_INFIL`` and ``1 - FRAC_INFIL``; a warm day without a pack does
  not call the routine; rain on a pack is taken by the pack;
* the day-200 (north) and day-10 (south) re-initialisation flags;
* the calendar of ``CDATE`` with its aliased year;
* the ``.sno`` reader, the breakpoint precipitation, the coefficient table.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from agrijax.core.units import parse_unit
from agrijax.forcing.precipitation import daily_storm_precipitation
from agrijax.io.rzwqm.met import BrkData
from agrijax.io.rzwqm.storms import RAIN_MODIFIER_ROW, storm_depths
from agrijax.processes.snow import (
    PRMS_SNOW,
    PrmsSnowParams,
    SnowForcing,
    SnowState,
    parse_sno,
    snow_prms,
)
from agrijax.processes.snow import prms as P
from agrijax.processes.snow.coefficients import coefficient_table
from agrijax.testing.conformance._builtin.snow import LATITUDE, SNO

C = PRMS_SNOW
#: relative tolerance of the closed forms: float64 rounding, or float32 when x64 is off
RTOL = 1e-12 if jax.config.jax_enable_x64 else 1e-5
#: absolute tolerance of the daily water balance [cm]
ATOL = 1e-12 if jax.config.jax_enable_x64 else 1e-5


def _params(latitude: float = LATITUDE) -> PrmsSnowParams:
    return PrmsSnowParams.from_sno(SNO, latitude)


def _day(tmin: float, tmax: float, precip: float, srad: float = 8.0, doy: float = 40.0) -> SnowForcing:
    a = lambda x: jnp.asarray(x, dtype=jnp.result_type(float))  # noqa: E731
    return SnowForcing(tmin=a(tmin), tmax=a(tmax), srad=a(srad), precipitation=a(precip), doy=a(doy))


def _pack(swe: float, temp: float, *, den: float = 0.25, free: float = 0.0) -> SnowState:
    """A pack from an earlier run (no re-initialisation) at temperature ``temp``."""
    s = SnowState.initial(swe)
    pkwe = swe * C.inch_per_cm
    a = lambda x: jnp.asarray(x, dtype=s.swe.dtype)  # noqa: E731
    return s.replace(
        pk_def=a(-temp * pkwe * C.ice_heat_inch),
        pk_temp=a(temp),
        pk_ice=a(pkwe - free),
        freeh2o=a(free),
        pk_depth=a(pkwe / den),
        pk_den=a(den),
        pss=a(pkwe),
        pst=a(pkwe),
        albedo=a(0.7),
        iso=a(1.0),
        mso=a(1.0),
        intal=a(1.0),
        slst=a(3.0),
        sstart=a(0.0),
        started=a(1.0),
        month=a(2.0),
        cdate_day=a(8.0),
    )


def _f(x: Any) -> float:
    return float(np.asarray(x))


# ------------------------------------------------------------------------ closed-form days
def test_cold_snowfall_on_bare_ground_keeps_its_water_but_the_sublimation() -> None:
    s = snow_prms(SnowState.initial(0.0), _params(), _day(-9.0, -3.0, 1.5))
    sublim = SNO.potet_sublim * C.sublimation_potential  # full cover: the curve's last point is 1
    np.testing.assert_allclose(_f(s.intercepted), 1.5, rtol=RTOL)
    assert _f(s.cover) == 1.0
    assert _f(s.out.melt) == 0.0 and _f(s.out.melt_runoff) == 0.0
    np.testing.assert_allclose(_f(s.out.sublimation), sublim, rtol=RTOL)
    np.testing.assert_allclose(_f(s.swe), 1.5 - sublim, rtol=RTOL)
    np.testing.assert_allclose(_f(s.out.swe), 10.0 * _f(s.swe), rtol=RTOL)  # the crop's SNOW [mm]
    assert _f(s.pk_temp) < 0.0 and _f(s.pk_def) > 0.0
    assert _f(s.sstart) == 0.0 and _f(s.started) == 1.0  # the first call re-initialised


def test_warm_day_without_a_pack_does_not_call_the_routine() -> None:
    s0 = SnowState.initial(0.0)
    s = snow_prms(s0, _params(), _day(3.0, 12.0, 2.0))
    assert _f(s.intercepted) == 0.0  # the rain infiltrates: the soil keeps the storms
    for leaf0, leaf in zip(jax.tree_util.tree_leaves(s0), jax.tree_util.tree_leaves(s), strict=True):
        assert np.array_equal(np.asarray(leaf0), np.asarray(leaf))


def test_freezing_day_without_precipitation_and_pack_is_not_called() -> None:
    s0 = SnowState.initial(0.0)
    s = snow_prms(s0, _params(), _day(-10.0, -2.0, 0.0))
    assert _f(s.sstart) == 1.0 and _f(s.swe) == 0.0 and _f(s.intercepted) == 0.0


def test_thin_isothermal_pack_melts_completely_and_splits() -> None:
    s = snow_prms(_pack(0.05, 0.0), _params(), _day(10.0, 20.0, 0.0, srad=20.0))
    assert _f(s.swe) == 0.0 and _f(s.cover) == 0.0
    np.testing.assert_allclose(_f(s.out.melt) + _f(s.out.melt_runoff), 0.05, rtol=RTOL)
    np.testing.assert_allclose(_f(s.out.melt), SNO.frac_infil * 0.05, rtol=RTOL)
    assert _f(s.out.sublimation) == 0.0  # melted in the night balance, before the sublimation
    assert _f(s.pk_den) == 0.0 and _f(s.albedo) == 0.0


def test_rain_on_a_pack_is_taken_by_the_pack() -> None:
    s0 = _pack(3.0, -2.0)
    s = snow_prms(s0, _params(), _day(1.0, 5.0, 0.8))
    np.testing.assert_allclose(_f(s.intercepted), 0.8, rtol=RTOL)
    closure = _f(s.swe) - 3.0 - (0.8 - _f(s.out.melt) - _f(s.out.melt_runoff) - _f(s.out.sublimation))
    assert abs(closure) <= ATOL


@pytest.mark.parametrize(("doy", "lat", "expect"), [(200.0, 0.7, 1.0), (200.0, -0.7, 0.0), (10.0, -0.7, 1.0)])
def test_reinitialisation_flag(doy: float, lat: float, expect: float) -> None:
    s0 = SnowState.initial(0.0).replace(sstart=jnp.asarray(0.0))
    s = snow_prms(s0, _params(lat), _day(15.0, 25.0, 0.0, doy=doy))
    assert _f(s.sstart) == expect


# ------------------------------------------------------------------------ conservation
def test_water_is_conserved_every_day_of_random_winters() -> None:
    rng = np.random.default_rng(7)
    n_site, n_day = 6, 120
    tmin = rng.uniform(-15.0, 5.0, (n_site, n_day))
    tmax = tmin + rng.uniform(1.0, 12.0, (n_site, n_day))
    precip = np.where(rng.uniform(size=(n_site, n_day)) < 0.35, rng.uniform(0.05, 2.5, (n_site, n_day)), 0.0)
    doy = np.tile(np.concatenate([np.arange(300, 366), np.arange(1, 55)]).astype(float), (n_site, 1))
    f = SnowForcing(
        tmin=jnp.asarray(tmin),
        tmax=jnp.asarray(tmax),
        srad=jnp.asarray(rng.uniform(2.0, 16.0, (n_site, n_day))),
        precipitation=jnp.asarray(precip),
        doy=jnp.asarray(doy),
    )
    p = _params()
    s0 = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[SnowState.initial(0.0)] * n_site)

    def one(s: Any, ff: Any) -> Any:
        return jax.lax.scan(lambda st, ft: (snow_prms(st, p, ft),) * 2, s, ff)[1]

    traj = jax.jit(jax.vmap(one))(s0, f)
    swe = np.asarray(traj.swe)
    prev = np.concatenate([np.zeros((n_site, 1)), swe[:, :-1]], axis=1)
    inflow = np.asarray(traj.intercepted)
    out = np.asarray(traj.out.melt + traj.out.melt_runoff + traj.out.sublimation)
    np.testing.assert_allclose(swe - prev, inflow - out, rtol=0, atol=ATOL)
    assert (swe > 0).sum() > n_site * 20 and (
        np.asarray(traj.out.melt) > 0
    ).sum() > n_site  # both regimes ran
    # the storms of a day the snow routine does not take are the soil's; a taken day takes them all
    taken = inflow > 0
    np.testing.assert_allclose(inflow[taken], precip[taken], rtol=RTOL)


def test_latent_heat_of_a_partial_melt() -> None:
    """``CALIN``: a gain ``Q`` above the cold content melts ``(Q - PK_DEF) / 203.2`` inch; what the
    ice can hold (``FREEH2O_CAP x ice``) stays as free water, the rest leaves as melt."""
    p = _params()
    pkwe, deficit, q = 1.2, 5.0, 40.0
    k = {n: jnp.asarray(0.0) for n in (*P.PACK_FIELDS, "snowmelt", "snow_evap")}
    k.update(
        pkwe=jnp.asarray(pkwe), pk_ice=jnp.asarray(pkwe), pk_def=jnp.asarray(deficit), sca=jnp.asarray(1.0)
    )
    k.update(pk_den=jnp.asarray(0.3), pk_depth=jnp.asarray(pkwe / 0.3))
    r = P._calin(k, jnp.asarray(q), jnp.asarray(True), p, C)
    melted = (q - deficit) / C.latent_heat_inch
    np.testing.assert_allclose(_f(r["freeh2o"]) + _f(r["snowmelt"]), melted, rtol=RTOL)
    np.testing.assert_allclose(_f(r["freeh2o"]), SNO.freeh2o_cap * (pkwe - melted), rtol=RTOL)
    np.testing.assert_allclose(_f(r["pkwe"]) + _f(r["snowmelt"]), pkwe, rtol=RTOL)
    assert _f(r["pk_def"]) == 0.0 and _f(r["pk_temp"]) == 0.0


# ------------------------------------------------------------------------ calendar, readers
@pytest.mark.parametrize(
    ("it", "doy", "month", "dom"),
    [
        (4.0, 60.0, 2.0, 29.0),
        (4.0, 61.0, 3.0, 1.0),
        (5.0, 60.0, 3.0, 1.0),
        (5.0, 61.0, 3.0, 2.0),
        (0.0, 1.0, 1.0, 1.0),
    ],
)
def test_cdate_month_with_the_previous_day_of_month_as_year(
    it: float, doy: float, month: float, dom: float
) -> None:
    k = {"cdate_day": jnp.asarray(it), "month": jnp.asarray(7.0)}
    m, d = P._cdate_month(jnp.asarray(doy), k)
    assert (_f(m), _f(d)) == (month, dom)


def test_cdate_day_366_outside_a_leap_reading_keeps_the_month() -> None:
    k = {"cdate_day": jnp.asarray(5.0), "month": jnp.asarray(12.0)}
    m, d = P._cdate_month(jnp.asarray(366.0), k)
    assert (_f(m), _f(d)) == (12.0, 5.0)


_SNO_TEXT = """====
= a synthetic .sno file (layout of READ_SNOW)
====
0.5  0.2  0.04  0.2  31.0
0.5  0.7
0.3  0.1
=====
1  0.4  0.6
====
0.4  0.8  0.3
1 2 3 4 5 6 7 8 9 10 11 12
0 0 0 0 0 0 0 0 0 0 0 0
====
0.7  80  85  40.0  2  2
0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0
0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
"""


def test_parse_sno_reads_records_across_lines_and_picks_the_curve() -> None:
    s = parse_sno(_SNO_TEXT)
    assert (s.den_max, s.tmax_allsnow, s.albset_rnm, s.albset_sna) == (0.5, 31.0, 0.5, 0.1)
    assert s.cov_type == 1 and s.potet_sublim == 0.3
    assert s.cecn_coef == tuple(float(i) for i in range(1, 13))
    assert (s.melt_look, s.melt_force, s.hru_deplcrv, s.ndepl) == (80, 85, 2, 2)
    assert s.depletion_curve[1] == 0.1 and len(s.depletion_curve) == 11
    p = PrmsSnowParams.from_sno(s, 0.7)
    assert p.melt_look == 80 and p.depletion and _f(p.snarea_thresh) == 40.0


def test_depletion_curve_zero_is_full_cover_and_unreproduced_inputs_raise() -> None:
    s = parse_sno(_SNO_TEXT.replace("40.0  2  2", "40.0  0  2"))
    assert s.depletion_curve == (1.0,) * 11
    with pytest.raises(NotImplementedError, match="COV_TYPE"):
        PrmsSnowParams.from_sno(parse_sno(_SNO_TEXT.replace("1  0.4  0.6", "3  0.4  0.6")), 0.7)
    with pytest.raises(NotImplementedError, match="thunderstorm"):
        PrmsSnowParams.from_sno(
            parse_sno(_SNO_TEXT.replace("0 0 0 0 0 0 0 0 0 0 0 0", "0 0 0 0 0 1 0 0 0 0 0 0")), 0.7
        )


def test_daily_storm_precipitation_sums_the_storms_of_a_day() -> None:
    events = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-02-01"]),
            "depth_in": [0.3, 0.005, 0.2, 0.1],
        }
    )
    bps = pd.DataFrame(
        {"event": [0, 0, 1, 1, 2, 2, 3, 3], "cum_depth_in": [0.0, 0.3, 0.0, 0.005, 0.0, 0.2, 0.0, 0.1]}
    )
    days = pd.date_range("2020-01-01", "2020-02-01")
    brk = BrkData(calendar_code=1, events=events, breakpoints=bps)
    got = daily_storm_precipitation(storm_depths(brk), days)
    expect = np.zeros(len(days))
    expect[1] = 0.3 * 2.54  # the 0.005-inch storm is below the 0.01-inch threshold
    expect[2] = 0.2 * 2.54
    expect[-1] = 0.1 * 2.54
    np.testing.assert_allclose(got, expect, rtol=1e-15)
    mod = np.full((8, 12), 100.0)
    mod[RAIN_MODIFIER_ROW] = 50.0
    got = daily_storm_precipitation(storm_depths(brk, met_modifiers=mod), days)
    np.testing.assert_allclose(got[-1], 0.5 * 0.1 * 2.54, rtol=1e-15)
    mod[RAIN_MODIFIER_ROW, 1] = 80.0  # a month-dependent modifier is not reproduced (storm reader)
    with pytest.raises(NotImplementedError):
        storm_depths(brk, met_modifiers=mod)


def test_coefficient_table_cites_the_reference_and_the_prms_manual() -> None:
    rows = coefficient_table()
    assert len(rows) >= 50
    for r in rows:
        assert r["ref_version"] == "rzwqm2-4.6", r["path"]
        assert r["file"] and r["line"] and r["routine"] and "Leavesley" in r["paper"], r["path"]
        assert r["statement"] == "", r["path"]  # no RZWQM2 statement is reproduced
        parse_unit(r["unit"])
