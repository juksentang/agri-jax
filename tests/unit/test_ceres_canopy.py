"""The canopy record of CERES-Maize for the PET module (port P6, day-table entry 15a), no data.

* :func:`published_lai`: the green LAI before maturity and when no later harvest is planned; after
  maturity with ``HDATE > MDATE`` the linear decline ``XHLAI - XHLAI / min(30, HDATE - MDATE) *
  (YRDOY - MDATE)`` (a NumPy transcription of the same expression, bit for bit), 0 on the harvest
  date and never negative;
* :func:`stalk_height`: the NumPy transcription bit for bit, half of ``htmax`` at the stalk mass
  ``biohalf``, the grain mass rounded half away from zero (Fortran ``NINT``);
* :func:`ceres_canopy`: ``tlai == lai``, the height is the running maximum of the stalk-mass
  height (starting from the previous published height), the season's planting population and
  planned harvest date are used, the harvest reset zeroes the record so the next season starts
  from 0; without canopy parameters or the canopy port it refuses to trace;
* the canopy parameters are optional: a CERES run without them is unchanged, and the contract's
  day table registers the entry under the RZWQM2 key.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core.events import EventTable
from agrijax.core.process import lookup
from agrijax.iface.contract import day_entry
from agrijax.iface.crop import CanopyRecord
from agrijax.processes.crop.ceres_maize import (
    RZWQM2_CANOPY,
    CanopyCoefficients,
    CeresCanopyParams,
    CeresMaizeState,
    CeresSeasons,
    ceres_canopy,
    ceres_harvest,
    published_lai,
    stalk_height,
)

from .test_ceres_growth import season_forcing
from .test_ceres_phenology import make_params

F = jnp.result_type(float)
C = RZWQM2_CANOPY
#: relative tolerance of a float64 transcription against the kernel: a few ulp of the kernel's dtype
RT = 1e-15 if F == jnp.float64 else 2e-6
#: absolute tolerance of an expected 0 of x - x / span * span (exact in float64, a few ulp in float32)
AT = 0.0 if F == jnp.float64 else 1e-6


def _np_lai(xhlai, mdate, hdate, yrdoy, days_max=30.0):
    x = np.asarray(xhlai, dtype=np.float64)
    out = x.copy()
    for i in range(x.size):
        m, h, d = int(mdate[i]), int(hdate[i]), int(yrdoy[i])
        if d > m and h > m and m > 0:
            span = min(days_max, float(h - m))
            out[i] = x[i] - x[i] / span * float(d - m)
    return np.maximum(out, 0.0)


def _np_height(biomas, grnwt, ears, pltpop, htmax=244.6, biohalf=43.07):
    g = np.asarray(grnwt) * np.asarray(ears) * 10.0
    nint = np.sign(g) * np.floor(np.abs(g) + 0.5)
    stalk = np.asarray(biomas) * 10.0 - nint
    stem = stalk / 10.0 / pltpop
    alpha = -(htmax + htmax) * math.log(0.5) / biohalf
    return htmax * (1.0 - np.exp(-alpha * stem / (htmax + htmax)))


def test_lai_before_maturity_and_without_a_later_harvest_is_the_green_lai() -> None:
    x = jnp.asarray([0.0, 1.3, 2.7, 4.1], F)
    yrdoy = jnp.asarray([2020200, 2020290, 2020295, 2020299], jnp.int32)
    for mdate, hdate in ((-99, 2020299), (2020281, 2020281), (2020281, 2020270), (2020300, 2020310)):
        m = jnp.full(4, mdate, jnp.int32)
        out = published_lai(x, m, jnp.asarray(hdate, jnp.int32), yrdoy, C)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(x))


def test_lai_declines_linearly_after_maturity_to_zero_on_the_harvest_date() -> None:
    # the CA-TPA 2020 and 2021 seasons: MDATE 2020281, HDATE 2020299 (18 days); 2021284 / 2021293
    for mdate, hdate in ((2020281, 2020299), (2021284, 2021293), (2021100, 2021200)):
        days = np.arange(mdate - 3, min(hdate, mdate + 40) + 1)
        x = np.full(days.size, 0.5091)
        m = np.full(days.size, mdate)
        h = np.full(days.size, hdate)
        out = published_lai(jnp.asarray(x, F), jnp.asarray(m), jnp.asarray(hdate), jnp.asarray(days), C)
        want = _np_lai(x, m, h, days)
        np.testing.assert_allclose(np.asarray(out), want, rtol=RT, atol=AT)
        span = min(30, hdate - mdate)
        assert np.all(np.asarray(out)[days - mdate >= span] <= AT)
        run = (days >= mdate) & (days - mdate <= span)  # MDATE to the day it reaches 0
        assert np.all(np.diff(np.asarray(out)[run]) < 0.0)
        assert np.all(np.asarray(out) >= 0.0)


def test_stalk_height_is_the_regression_and_half_of_htmax_at_biohalf() -> None:
    rng = np.random.default_rng(3)
    # inputs in the kernel's dtype; the transcription rounds the grain product in that dtype too
    # (so NINT sees the same number) and evaluates the rest in float64
    biomas, grnwt, ears = (
        np.asarray(rng.uniform(lo, hi, 64), F) for lo, hi in ((0.0, 2400.0), (0.0, 250.0), (5.0, 9.0))
    )
    out = stalk_height(jnp.asarray(biomas), jnp.asarray(grnwt), jnp.asarray(ears), jnp.asarray(8.0, F), C)
    grain = np.floor(grnwt * ears * np.asarray(10.0, F) + np.asarray(0.5, F)).astype(np.float64)
    want = _np_height(biomas.astype(np.float64), grain / 10.0, np.ones(64), 8.0)
    # float32: the stalk mass is a difference of two sums of up to 2.4e4 kg ha-1 (absolute rounding)
    atol = 0.0 if F == jnp.float64 else 1e-5 * C.htmax
    np.testing.assert_allclose(np.asarray(out), want, rtol=max(RT, 4e-16 * 8), atol=atol)
    # S = biohalf [g plant-1] with no grain: stalk [kg ha-1] = biohalf * pltpop * 10
    pop = 7.5
    b = jnp.asarray([C.biohalf * pop], F)  # g m-2
    h = stalk_height(b, jnp.zeros(1, F), jnp.zeros(1, F), jnp.asarray(pop, F), C)
    np.testing.assert_allclose(np.asarray(h), C.htmax / 2.0, rtol=max(RT, 1e-12))


def test_grain_mass_is_rounded_half_away_from_zero() -> None:
    # grain 12.5 kg ha-1 -> 13 (jnp.round would give 12); biomas 10 g m-2 = 100 kg ha-1
    h = stalk_height(
        jnp.asarray([10.0], F), jnp.asarray([1.25], F), jnp.asarray([1.0], F), jnp.asarray(1.0, F), C
    )
    np.testing.assert_allclose(np.asarray(h), _np_height([10.0], [1.25], [1.0], 1.0), rtol=RT)
    stem = (100.0 - 13.0) / 10.0
    alpha = -(2 * C.htmax) * math.log(0.5) / C.biohalf
    assert float(h[0]) == pytest.approx(
        C.htmax * (1 - math.exp(-alpha * stem / (2 * C.htmax))), rel=max(RT, 1e-14)
    )


def _state(params, n_crop=2):
    s = CeresMaizeState.initial(params, n_crop, events=True)
    return s.replace(canopy_out=CanopyRecord.zeros(n_crop), season=jnp.asarray(0, jnp.int32))


def _seasons(params, pops=(7.2, 8.0)):
    y0 = int(np.asarray(params.yrplt))
    return CeresSeasons(
        yrplt=jnp.asarray([y0, y0 + 400], jnp.int32),
        pltpop=jnp.asarray(pops, F),
        sdepth=jnp.asarray([7.0, 7.0], F),
        rowspc=jnp.asarray([61.0, 61.0], F),
    )


def test_canopy_record_uses_the_season_population_and_is_a_running_maximum() -> None:
    p0 = make_params()
    p = p0.replace(
        seasons=_seasons(p0),
        canopy=CeresCanopyParams(hdate=jnp.asarray([2001250, 2002250], jnp.int32)),
    )
    f, _ = season_forcing(0)
    ft = jax.tree_util.tree_map(lambda x: x[30], f)
    s = _state(p)
    g = s.growth.replace(
        lai=jnp.asarray([2.0, 3.0], F),
        biomas=jnp.asarray([500.0, 900.0], F),
        grnwt=jnp.asarray([0.0, 20.0], F),
    )
    s = s.replace(growth=g, phen=s.phen.replace(ears=jnp.asarray([0.0, 7.0], F)))
    for season, pop in ((0, 7.2), (1, 8.0)):
        s1 = ceres_canopy(s.replace(season=jnp.asarray(season, jnp.int32)), p, ft)
        want = _np_height([500.0, 900.0], [0.0, 20.0], [0.0, 7.0], pop)
        np.testing.assert_allclose(np.asarray(s1.canopy_out.height), want, rtol=RT)
        np.testing.assert_array_equal(np.asarray(s1.canopy_out.lai), [2.0, 3.0])
        np.testing.assert_array_equal(np.asarray(s1.canopy_out.tlai), np.asarray(s1.canopy_out.lai))
    # running maximum: a lower stalk-mass height keeps the previous published height
    s2 = ceres_canopy(s.replace(growth=g.replace(biomas=jnp.asarray([100.0, 950.0], F))), p, ft)
    prev = ceres_canopy(s, p, ft).canopy_out.height
    s3 = ceres_canopy(s2.replace(canopy_out=s2.canopy_out.replace(height=prev)), p, ft)
    h_low = _np_height([100.0, 950.0], [0.0, 20.0], [0.0, 7.0], 7.2)
    np.testing.assert_allclose(np.asarray(s3.canopy_out.height), np.maximum(np.asarray(prev), h_low), rtol=RT)
    assert float(s3.canopy_out.height[0]) == float(prev[0]) and float(s3.canopy_out.height[1]) > float(
        prev[1]
    )


def test_the_season_hdate_row_sets_the_decline_and_harvest_zeroes_the_record() -> None:
    p0 = make_params()
    p = p0.replace(
        seasons=_seasons(p0),
        canopy=CeresCanopyParams(hdate=jnp.asarray([2001240, 2002250], jnp.int32)),
    )
    f, _ = season_forcing(0)
    ft = jax.tree_util.tree_map(lambda x: x[30], f)
    today = int(np.asarray(ft.yrdoy))
    s = _state(p)
    s = s.replace(
        growth=s.growth.replace(lai=jnp.asarray([1.0, 1.0], F), biomas=jnp.asarray([800.0, 800.0], F)),
        phen=s.phen.replace(mdate=jnp.full(2, today - 3, jnp.int32)),
    )
    out = ceres_canopy(s, p, ft).canopy_out
    span = min(30, 2001240 - (today - 3))
    np.testing.assert_allclose(np.asarray(out.lai), 1.0 - 1.0 / span * 3.0, rtol=RT)
    # the harvest reset after the entry: bare soil, and the next day's record starts from height 0
    s = s.replace(water_in=s.water_in.replace(trwup=jnp.ones(2, F)))
    ev = EventTable.empty(1)
    ev_t = jax.tree_util.tree_map(lambda x: x[0], ev.replace(harvest=jnp.asarray([True])))
    s_h = ceres_harvest(ceres_canopy(s, p, ft), p, ev_t)
    for leaf in jax.tree_util.tree_leaves(s_h.canopy_out):
        assert float(jnp.max(jnp.abs(leaf))) == 0.0
    nxt = ceres_canopy(s_h, p, ft).canopy_out
    assert float(jnp.max(nxt.height)) == 0.0 and float(jnp.max(nxt.lai)) == 0.0


def test_canopy_needs_its_parameters_and_its_port() -> None:
    p = make_params()
    f, _ = season_forcing(0)
    ft = jax.tree_util.tree_map(lambda x: x[30], f)
    s = _state(p)
    with pytest.raises(ValueError, match=r"params\.canopy"):
        ceres_canopy(s, p, ft)
    with pytest.raises(ValueError, match="canopy_out"):
        ceres_canopy(s.replace(canopy_out=None), p.replace(canopy=CeresCanopyParams.none()), ft)


def test_contract_entry_and_coefficients() -> None:
    e = day_entry("crops.maize.canopy", "maize")
    assert e.key == "crop/ceres_maize.canopy@rzwqm2-4.6:faithful" and e.status == "registered"
    assert lookup(e.key).name == "ceres_canopy"
    assert (C.htmax, C.biohalf, C.height_fraction, C.dry_down_days) == (244.6, 43.07, 0.5, 30.0)
    cal = CanopyCoefficients().calibratable_paths()
    assert set(cal) == {"htmax", "biohalf", "dry_down_days"}
    none = CeresCanopyParams.none(3)
    np.testing.assert_array_equal(np.asarray(none.hdate), [-99, -99, -99])
    assert int(none.hdate_of(jnp.asarray(7, jnp.int32))) == -99
