"""The DSSAT-CSM v4.8.6.0 tipping bucket (``processes/soil_water/bucket``) without data.

* The kernels against a second implementation: a plain-Python, statement-by-statement loop
  translation of ``INFIL``, ``SATFLO``, ``UP_FLOW``, ``RNOFF``, ``SNOWFALL`` and ``MULCHWATER``
  (DSSAT-CSM v4.8.6.0, BSD-3) below, on random profiles that reach every branch (saturation
  excess pushed back up, the ``24 SWCN`` caps, drainage below the wetting front, upward and
  downward unsaturated flow, padded layers).
* Conservation of every kernel and of the whole day: RATE, a SPAM replay and INTEGR with the
  water ledger closing to 1e-10 cm under ``AGRI_JAX_CHECK=1``.
* Finite gradients through the day, the module declaration against the registry and the fields.

The day-by-day comparison with the reference is ``tests/integration/test_bucket_dssat.py``.
"""

from __future__ import annotations

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core.ledger import WaterLedger, water_ledger
from agrijax.core.model import Model
from agrijax.core.process import lookup, process
from agrijax.core.runtime import run
from agrijax.iface.soil import SinkInputs
from agrijax.iface.surface import PETFluxes
from agrijax.processes.soil_water.bucket import (
    BUCKET_LEDGER_INFLOWS,
    BUCKET_LEDGER_OUTFLOWS,
    MODULE,
    WATBAL_COEFFICIENTS,
    BucketForcing,
    BucketParams,
    BucketSoil,
    BucketState,
    MulchForcing,
    bucket_integrate,
    bucket_ledger_channels,
    bucket_rate,
    bucket_storage,
)
from agrijax.processes.soil_water.bucket import kernels as K

C = WATBAL_COEFFICIENTS
#: the loop translation is compared to 1e-12: a float64 statement (float32 rounds differently and may
#: take the other side of a threshold)


# ============================================================================ loop translation
def ref_infil(dlayr, ds, dul, sat, sw, swcn, swcon, pinf, actwtd, nlayr, hits=None):
    """INFIL.for:44-147, one statement per line (``hits`` counts the branches taken)."""
    hits = {} if hits is None else hits

    def hit(k):
        hits[k] = hits.get(k, 0) + 1

    drn = [0.0] * len(sw)
    swtemp = list(sw)
    excs = 0.0
    lost = 0.0
    tmpexcs = 0.0
    for L in range(nlayr):
        hold = (sat[L] - swtemp[L]) * dlayr[L]
        if pinf > 1e-4 and pinf > hold:
            if ds[L] > actwtd and actwtd > 0.0:
                drn[L] = 0.0
                hit("below_water_table")
            else:
                drcm = (0.9 if L == 0 else 1.0) * swcon * (sat[L] - dul[L]) * dlayr[L]
                drn[L] = pinf - hold + drcm
                if swcn[L] > 0.0 and drn[L] > swcn[L] * 24.0:
                    drn[L] = swcn[L] * 24.0
                    hit("cap")
            swtemp[L] = swtemp[L] + (pinf - drn[L]) / dlayr[L]
            if swtemp[L] > sat[L]:
                tmpexcs = (swtemp[L] - sat[L]) * dlayr[L]
                swtemp[L] = sat[L]
                if L == 0 and tmpexcs > 0.0:
                    excs += tmpexcs
                    hit("excs")
                if L > 0:
                    hit("push")
                    stopped = False
                    for LK in range(L - 1, -1, -1):
                        if tmpexcs < 0.0001:
                            stopped = True
                            break
                        h = min((sat[LK] - swtemp[LK]) * dlayr[LK], tmpexcs)
                        swtemp[LK] += h / dlayr[LK]
                        drn[LK] = max(drn[LK] - tmpexcs, 0.0)
                        tmpexcs -= h
                        if LK == 0 and tmpexcs > 0.0:
                            excs += tmpexcs
                    if stopped:
                        hit("stop")
                        lost += tmpexcs
            pinf = drn[L]
        else:
            swtemp[L] = swtemp[L] + pinf / dlayr[L]
            if swtemp[L] >= dul[L] + 0.003 and ds[L] < actwtd and actwtd > 0.0:
                drcm = (0.9 if L == 0 else 1.0) * (swtemp[L] - dul[L]) * swcon * dlayr[L]
                drn[L] = drcm
                if swcn[L] > 0.0 and drn[L] > swcn[L] * 24.0:
                    drn[L] = swcn[L] * 24.0
                    drcm = drn[L]
                swtemp[L] -= drcm / dlayr[L]
                pinf = drcm
                hit("drain_below_front")
            else:
                pinf = 0.0
                drn[L] = 0.0
    swdelts = [swtemp[L] - sw[L] if L < nlayr else 0.0 for L in range(len(sw))]
    return swdelts, drn, pinf * 10.0, excs, lost


def ref_satflo(dlayr, dul, sat, sw, swcn, swcon, nlayr):
    """SATFLO.for:46-116."""
    drn = [0.0] * len(sw)
    swtemp = list(sw)
    for L in range(nlayr):
        drmx = 0.0
        if swtemp[L] >= dul[L] + 0.003:
            drmx = max(0.0, (swtemp[L] - dul[L]) * swcon * dlayr[L])
        if L == 0:
            drn[L] = drmx
        else:
            hold = (dul[L] - swtemp[L]) * dlayr[L] if swtemp[L] < dul[L] else 0.0
            drn[L] = max(drn[L - 1] + drmx - hold, 0.0)
        if swcn[L] > 0.0 and drn[L] > swcn[L] * 24.0:
            drn[L] = swcn[L] * 24.0
    for L in range(nlayr - 1, 0, -1):
        swtemp[L] = swtemp[L] + (drn[L - 1] - drn[L]) / dlayr[L]
        over = (swtemp[L] - sat[L]) * dlayr[L]
        if over > 0.0:
            if over > drn[L - 1]:
                swtemp[L] -= drn[L - 1] / dlayr[L]
                drn[L - 1] = 0.0
            else:
                swtemp[L] = sat[L]
                drn[L - 1] -= over
    swtemp[0] -= drn[0] / dlayr[0]
    swdelts = [swtemp[L] - sw[L] if L < nlayr else 0.0 for L in range(len(sw))]
    return swdelts, drn, drn[nlayr - 1] * 10.0


def ref_upflow(dlayr, dul, ll, sat, sw, sw_avail, nlayr):
    """WBSUBS.for:283-396 (UP_FLOW)."""
    n = len(sw)
    up = [0.0] * n
    swtemp = list(sw)
    sw_inf = list(sw_avail)
    avail = [max(0.0, sw_avail[L] - ll[L]) for L in range(n)]
    esw = [dul[L] - ll[L] for L in range(n)]
    ist = 0 if dlayr[0] >= 5.0 else 1
    for L in range(ist, nlayr - 1):
        M = L + 1
        swold = swtemp[L]
        t1 = max(0.0, min(swtemp[L] - ll[L], esw[L]))
        t2 = max(0.0, min(swtemp[M] - ll[M], esw[M]))
        dbar = min(
            0.88 * math.exp(35.4 * ((t1 * dlayr[L] + t2 * dlayr[M]) / (dlayr[L] + dlayr[M])) * 0.5), 100.0
        )
        grad = (t2 / esw[M] - t1 / esw[L]) * (esw[M] * dlayr[M] + esw[L] * dlayr[L]) / (dlayr[M] + dlayr[L])
        up[L] = dbar * grad / ((dlayr[L] + dlayr[M]) * 0.5)
        if up[L] > 0.0:
            if swtemp[L] <= dul[L]:
                swtemp[L] += up[L] / dlayr[L]
                sw_inf[L] += up[L] / dlayr[L]
                if swtemp[L] > dul[L] or sw_inf[L] > sat[L]:
                    fix = max(0.0, (swtemp[L] - dul[L]) * dlayr[L], (sw_inf[L] - sat[L]) * dlayr[L])
                    fix = min(up[L], fix)
                    up[L] -= fix
                    swtemp[L] = swold + up[L] / dlayr[L]
            else:
                up[L] = 0.0
            if up[L] / dlayr[M] > avail[M]:
                up[L] = avail[M] * dlayr[M]
                swtemp[L] = swold + up[L] / dlayr[L]
            swtemp[M] -= up[L] / dlayr[M]
        elif up[L] < 0.0:
            if swtemp[L] >= ll[L]:
                if abs(up[L] / dlayr[L]) > avail[L]:
                    up[L] = -avail[L] * dlayr[L]
                swtemp[L] += up[L] / dlayr[L]
                swtemp[M] -= up[L] / dlayr[M]
                sw_inf[M] -= up[L] / dlayr[M]
                if sw_inf[M] > sat[M]:
                    fix = min(abs(up[L]), (sw_inf[M] - sat[M]) * dlayr[M])
                    up[L] += fix
                    swtemp[L] = swold + up[L] / dlayr[L]
                    swtemp[M] -= fix / dlayr[M]
            else:
                up[L] = 0.0
    return [swtemp[L] - sw[L] for L in range(n)], up


def ref_rnoff(cn, ll, sat, sw, watavl, cover, mulch_on):
    """RNOFF.for:76-116 (no plastic mulch)."""
    smx = 254.0 * (100.0 / cn - 1.0)
    swabi = max(
        0.0, 0.15 * ((sat[0] - sw[0]) / (sat[0] - ll[0] * 0.5) + (sat[1] - sw[1]) / (sat[1] - ll[1] * 0.5))
    )
    iabs = max(swabi, swabi + (0.6 - swabi) * cover) if mulch_on else swabi
    pb = watavl - iabs * smx
    if watavl > 0.001 and pb > 0:
        return pb**2 / (watavl + (1.0 - iabs) * smx)
    return 0.0


def ref_snowfall(tmax, rain, snow):
    """WATBAL.for:282-288 and SNOWFALL (WBSUBS.for:44-60)."""
    if not (tmax <= 1.0 or snow > 0.0):
        return snow, rain
    if tmax > 1.0:
        melt = min(tmax + rain * 0.4, snow)
        snow, watavl = snow - melt, rain + melt
    else:
        snow, watavl = snow + rain, 0.0
    return (0.0 if snow < 0.001 else snow), watavl


def ref_mulch(watavl, mulchwat, evap_prev, mass, cover, new, watfac):
    """MULCHWAT.for:98-151."""
    msat = watfac * 1e-4 * mass
    res = 0.5 * watfac * 1e-4 * new if new > 1e-6 else 0.0
    if mass > 0.01 and watavl > 0.0:
        deficit = msat - (mulchwat + res) + min(mulchwat * 0.85, evap_prev)
        add = max(min(deficit, watavl * cover), 0.0)
        return max(watavl - add, 0.0), add, res
    return watavl, 0.0, res


# ============================================================================ random profiles
def profile(rng: np.random.Generator, n: int, nlayr: int, kind: str):
    dlayr = np.where(np.arange(n) < nlayr, rng.choice([4.0, 5.0, 10.0, 15.0, 20.0, 30.0], n), 0.0)
    ds = np.cumsum(dlayr) * (dlayr > 0)
    ll = rng.uniform(0.05, 0.15, n)
    dul = ll + rng.uniform(0.1, 0.2, n)
    sat = dul + rng.uniform(0.05, 0.15, n)
    if kind == "wet":
        sw = sat - rng.uniform(0.0, 0.02, n)
    elif kind == "dry":
        sw = ll + rng.uniform(-0.01, 0.05, n)
    else:
        sw = ll + rng.uniform(0.0, 1.1, n) * (sat - ll)
    swcn = np.where(rng.uniform(size=n) < 0.5, rng.uniform(0.05, 0.5, n), -99.0)
    act = dlayr > 0
    z = lambda x: np.where(act, x, 0.0)  # noqa: E731
    return z(dlayr), z(ds), z(ll), z(dul), z(sat), z(sw), z(swcn)


CASES = [(s, k) for s in range(12) for k in ("mid", "wet", "dry")]


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="float64 statement")
@pytest.mark.allow_skip(reason="float64 comparison, skipped in the float32 tier")
@pytest.mark.parametrize(("seed", "kind"), CASES)
def test_infil_and_satflo_match_the_loop_translation(seed: int, kind: str) -> None:
    rng = np.random.default_rng(seed)
    n, nlayr = 9, int(rng.integers(3, 9))
    dlayr, ds, _ll, dul, sat, sw, swcn = profile(rng, n, nlayr, kind)
    swcon = float(rng.uniform(0.2, 0.8))
    actwtd = 1000.0 if seed % 3 else float(rng.uniform(20.0, 80.0))
    pinf = float(rng.choice([0.00005, rng.uniform(0.2, 3.0), rng.uniform(4.0, 12.0)]))
    got = K.infil(dlayr, ds, dul, sat, sw, swcn, swcon, pinf, actwtd)
    ref = ref_infil(*(list(x) for x in (dlayr, ds, dul, sat, sw, swcn)), swcon, pinf, actwtd, nlayr)
    np.testing.assert_allclose(np.asarray(got.swdelts), ref[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(got.drn)[:nlayr], ref[1][:nlayr], rtol=0, atol=1e-12)
    for a, b in zip((got.drain, got.excs, got.lost), ref[2:], strict=True):
        assert abs(float(a) - b) <= 1e-11, (float(a), b)
    # conservation: what entered is stored, drained, rejected or dropped
    total = float(np.sum(np.asarray(got.swdelts) * dlayr) + got.drain / 10.0 + got.excs + got.lost)
    assert abs(total - pinf) <= 1e-12 * (1 + pinf)
    got = K.satflo(dlayr, dul, sat, sw, swcn, swcon)
    ref = ref_satflo(*(list(x) for x in (dlayr, dul, sat, sw, swcn)), swcon, nlayr)
    np.testing.assert_allclose(np.asarray(got.swdelts), ref[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(got.drn)[:nlayr], ref[1][:nlayr], rtol=0, atol=1e-12)
    assert abs(float(got.drain) - ref[2]) <= 1e-11
    assert abs(float(np.sum(np.asarray(got.swdelts) * dlayr) + got.drain / 10.0)) <= 1e-12


def test_infil_reaches_every_branch() -> None:
    """The profiles of the loop comparison reach every branch of INFIL: saturation excess pushed back
    up (and the push stopped below 1e-4 cm), the SWCN cap, the top layer's excess, the water table,
    drainage below the wetting front."""
    hits: dict[str, int] = {}
    for seed, kind in CASES:
        rng = np.random.default_rng(seed)
        n, nlayr = 9, int(rng.integers(3, 9))
        dlayr, ds, _ll, dul, sat, sw, swcn = profile(rng, n, nlayr, kind)
        swcon = float(rng.uniform(0.2, 0.8))
        actwtd = 1000.0 if seed % 3 else float(rng.uniform(20.0, 80.0))
        for pinf in (0.5, 3.0, 12.0):
            ref_infil(*(list(x) for x in (dlayr, ds, dul, sat, sw, swcn)), swcon, pinf, actwtd, nlayr, hits)
    want = {"push", "stop", "cap", "excs", "below_water_table", "drain_below_front"}
    assert want <= set(hits), hits


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="float64 statement")
@pytest.mark.allow_skip(reason="float64 comparison, skipped in the float32 tier")
@pytest.mark.parametrize(("seed", "kind"), CASES)
def test_upflow_matches_the_loop_translation(seed: int, kind: str) -> None:
    rng = np.random.default_rng(100 + seed)
    n, nlayr = 9, int(rng.integers(2, 9))
    dlayr, _ds, ll, dul, sat, sw, _swcn = profile(rng, n, nlayr, kind)
    dlayr[0] = rng.choice([3.0, 5.0, 10.0])
    avail = np.maximum(0.0, sw + np.where(dlayr > 0, rng.uniform(-0.02, 0.05, n), 0.0))
    got = K.up_flow(dlayr, dul, ll, sat, sw, avail)
    ref = ref_upflow(*(list(x) for x in (dlayr, dul, ll, sat, sw, avail)), nlayr)
    np.testing.assert_allclose(np.asarray(got.swdeltu)[:nlayr], ref[0][:nlayr], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(got.upflow)[:nlayr], ref[1][:nlayr], rtol=0, atol=1e-12)
    assert np.all(np.asarray(got.swdeltu)[nlayr:] == 0.0)
    assert abs(float(np.sum(np.asarray(got.swdeltu) * dlayr))) <= 1e-13


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="float64 statement")
@pytest.mark.allow_skip(reason="float64 comparison, skipped in the float32 tier")
@pytest.mark.parametrize("seed", range(40))
def test_surface_kernels_match_the_loop_translation(seed: int) -> None:
    rng = np.random.default_rng(200 + seed)
    ll = rng.uniform(0.05, 0.15, 2)
    sat = ll + rng.uniform(0.2, 0.35, 2)
    sw = ll + rng.uniform(0.0, 1.0, 2) * (sat - ll)
    cn = float(rng.uniform(60.0, 95.0))
    watavl = float(rng.choice([0.0, 0.0005, rng.uniform(1.0, 120.0)]))
    cover = float(rng.uniform(0.0, 1.0))
    for on in (True, False):
        got = float(K.rnoff(cn, ll, sat, sw, watavl, cover, on).runoff)
        assert abs(got - ref_rnoff(cn, ll, sat, sw, watavl, cover, on)) <= 1e-12 * (1 + watavl)
    tmax = float(rng.choice([rng.uniform(-10.0, 1.0), 1.0, rng.uniform(1.0, 15.0)]))
    rain, snow = float(rng.uniform(0.0, 20.0)), float(rng.choice([0.0, 0.0005, rng.uniform(0.0, 50.0)]))
    s = K.snowfall(tmax, rain, snow)
    rs, rw = ref_snowfall(tmax, rain, snow)
    assert abs(float(s.snow) - rs) <= 1e-12 and abs(float(s.watavl) - rw) <= 1e-12
    assert abs((snow + rain) - (float(s.snow) + float(s.watavl) + float(s.dropped))) <= 1e-12
    mass, new = (
        float(rng.choice([0.0, rng.uniform(100.0, 6000.0)])),
        float(rng.choice([0.0, rng.uniform(0.0, 500.0)])),
    )
    mw, ev = float(rng.uniform(0.0, 2.0)), float(rng.uniform(0.0, 0.5))
    m = K.mulch_rate(watavl, mw, ev, mass, cover, new, 3.5, True)
    r = ref_mulch(watavl, mw, ev, mass, cover, new, 3.5)
    for a, b in zip((m.watavl, m.mulwatadd, m.reswatadd), r, strict=True):
        assert abs(float(a) - b) <= 1e-12 * (1 + watavl)


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="float64 statement")
@pytest.mark.allow_skip(reason="float64 comparison, skipped in the float32 tier")
def test_integration_rounds_to_1e6_and_books_the_rounding() -> None:
    sw = jnp.asarray([0.2, 0.3, 0.00004, 0.0])
    dl = jnp.asarray([5.0, 10.0, 15.0, 0.0])
    deltas = jnp.asarray([0.01234567, -0.0000011, 0.0, 0.0])
    new, rnd = K.integrate_sw(sw, dl, dl, jnp.asarray(1.3), deltas, True)
    raw = np.asarray([0.2 - 0.13 / 5.0 + 0.01234567, 0.3 - 0.0000011, 0.00004, 0.0])
    want = np.sign(raw) * np.floor(np.abs(raw) * 1e6 + 0.5) / 1e6
    want[2] = 0.0  # below 1e-4
    np.testing.assert_allclose(np.asarray(new), want, rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(rnd)[:3], (want - raw)[:3] * np.asarray(dl)[:3], rtol=0, atol=1e-15)


# ============================================================================ the whole day
N_DAY, N_L = 12, 6


def _day_inputs(seed: int, salus: bool = False):
    rng = np.random.default_rng(300 + seed)
    dlayr, ds, ll, dul, sat, sw, swcn = profile(rng, N_L, 5, "mid")
    soil = BucketSoil(
        dlayr=jnp.asarray(dlayr), ds=jnp.asarray(ds), ll=jnp.asarray(ll), dul=jnp.asarray(dul),
        sat=jnp.asarray(sat), swcn=jnp.asarray(swcn), cn=jnp.asarray(78.0), swcon=jnp.asarray(0.5),
    )  # fmt: skip
    p = BucketParams(
        soil=soil,
        mulch_on=jnp.asarray(1.0),
        salus_es=jnp.asarray(1.0 if salus else 0.0),
        actwtd=jnp.asarray(1000.0),
        pm_fraction=jnp.asarray(0.0),
    )
    rain = np.where(rng.uniform(size=N_DAY) < 0.4, rng.uniform(0.0, 60.0, N_DAY), 0.0)
    tmax = rng.uniform(-6.0, 25.0, N_DAY)
    f = BucketForcing(
        rain=jnp.asarray(rain),
        tmax=jnp.asarray(tmax),
        irrigation=jnp.asarray(np.where(rng.uniform(size=N_DAY) < 0.2, 25.0, 0.0)),
        mulch=MulchForcing(
            mass=jnp.full(N_DAY, 1500.0),
            cover=jnp.full(N_DAY, 0.4),
            new_mass=jnp.asarray(np.where(np.arange(N_DAY) == 5, 800.0, 0.0)),
            watfac=jnp.full(N_DAY, 3.5),
        ),
    )
    uptake = np.where(np.arange(N_L) < 4, rng.uniform(0.0, 0.08, (N_DAY, N_L)), 0.0) * (dlayr > 0)
    evap = np.where(np.arange(N_L) < 2, rng.uniform(0.0, 0.05, (N_DAY, N_L)), 0.0) * (dlayr > 0)
    extra = {"uptake": jnp.asarray(uptake), "evap": jnp.asarray(evap), "es": jnp.asarray(rng.uniform(0.0, 0.3, N_DAY)),
             "em": jnp.asarray(rng.uniform(0.0, 0.02, N_DAY))}  # fmt: skip
    s0 = BucketState.initial(sw, mulch_wat=0.3).replace(pet=PETFluxes.zeros())
    return p, f, s0, extra


class DayForcing(eqx.Module):
    bucket: BucketForcing
    uptake: jax.Array
    evap: jax.Array
    es: jax.Array
    em: jax.Array


@process(reads=(), writes=("soil_water.sink_in.uptake", "soil_water.evap_layers", "iface.pet"), register=False,
         source="test replay of SPAM")  # fmt: skip
def _spam(state, params, f):
    """Replay of SPAM's outputs.

    Source: DSSAT-CSM v4.8.6.0 SPAM/SPAM.for.
    """
    b = state["soil_water"]
    b = b.replace(sink_in=b.sink_in.replace(uptake=f.uptake), evap_layers=f.evap)
    pet = state["iface"]["pet"].replace(soil_evaporation=f.es * 0.1, residue_evaporation=f.em * 0.1)
    return {**state, "soil_water": b, "iface": {**state["iface"], "pet": pet}}


def _global(proc):
    """``proc`` on the global dict state (bucket at ``soil_water``, P5 at ``iface.pet``)."""

    def fn(state, params, f):
        b = state["soil_water"].replace(pet=state["iface"]["pet"])
        out = proc(b, params, f.bucket)
        return {**state, "soil_water": out.replace(pet=None)}

    return process(
        fn, reads=("*",), writes=("soil_water",), name=proc.name, register=False, source=proc.source
    )


def _day_model():
    inflows, outflows = bucket_ledger_channels()
    led = water_ledger(
        storage=lambda s, p, f: bucket_storage(s["soil_water"], p, f.bucket),
        inflows={k: (lambda s, p, f, g=g: g(s, p, f.bucket)) for k, g in inflows.items()},
        outflows={k: (lambda s, p, f, g=g: g(s, p, f.bucket)) for k, g in outflows.items()},
        reads=("soil_water", "iface.pet"),
    )
    return Model(
        None,
        [_global(bucket_rate), _spam, _global(bucket_integrate), led],
        outputs=("ledger.water.residual",),
    )


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="float64 statement")
@pytest.mark.allow_skip(reason="float64 comparison, skipped in the float32 tier")
@pytest.mark.parametrize("salus", [False, True])
def test_day_closes_the_water_ledger(monkeypatch: pytest.MonkeyPatch, salus: bool) -> None:
    monkeypatch.setenv("AGRI_JAX_CHECK", "1")
    p, f, s0, x = _day_inputs(1, salus)
    ledger = WaterLedger.init(
        bucket_storage(s0, p), inflows=BUCKET_LEDGER_INFLOWS, outflows=BUCKET_LEDGER_OUTFLOWS
    )
    state = {"soil_water": s0.replace(pet=None), "iface": {"pet": s0.pet}, "ledger": {"water": ledger}}
    df = DayForcing(bucket=f, uptake=x["uptake"], evap=x["evap"], es=x["es"], em=x["em"])
    final, out = jax.jit(lambda s: run(_day_model(), p, df, s, return_final=True))(state)
    res = np.asarray(out["ledger.water.residual"])
    assert np.all(np.isfinite(res)) and np.max(np.abs(res)) <= 1e-10, res
    led = final["ledger"]["water"]
    assert float(led.total_in()["rain"]) > 0.0 and float(led.total_out()["drainage"]) >= 0.0
    assert abs(float(led.closure())) <= 1e-9


def test_day_gradients_are_finite() -> None:
    p, f, s0, x = _day_inputs(2)
    df = DayForcing(bucket=f, uptake=x["uptake"], evap=x["evap"], es=x["es"], em=x["em"])
    state = {"soil_water": s0.replace(pet=None), "iface": {"pet": s0.pet}}
    model = Model(
        None, [_global(bucket_rate), _spam, _global(bucket_integrate)], outputs=("soil_water.flux.drain",)
    )

    def loss(q):
        pp = dataclasses.replace(
            p, soil=p.soil.replace(swcon=q[0], cn=q[1]), coefficients=C.as_arrays().replace(upflow_dbar0=q[2])
        )
        return jnp.sum(run(model, pp, df, state)["soil_water.flux.drain"])

    g = jax.grad(loss)(jnp.asarray([0.5, 78.0, 0.88]))
    assert np.all(np.isfinite(np.asarray(g))) and float(g[0]) > 0.0, g


def test_batch_equals_single_runs() -> None:
    ins = [_day_inputs(k) for k in range(3)]
    model = Model(None, [_global(bucket_rate), _spam, _global(bucket_integrate)], outputs=("soil_water.sw",))

    def one(p, df, s):
        return run(model, p, df, s)["soil_water.sw"]

    stacked = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *[
        (p, DayForcing(bucket=f, uptake=x["uptake"], evap=x["evap"], es=x["es"], em=x["em"]),
         {"soil_water": s.replace(pet=None), "iface": {"pet": s.pet}}) for p, f, s, x in ins
    ])  # fmt: skip
    batched = jax.jit(jax.vmap(one))(*stacked)
    for k, (p, f, s, x) in enumerate(ins):
        single = jax.jit(one)(p, DayForcing(bucket=f, uptake=x["uptake"], evap=x["evap"], es=x["es"], em=x["em"]),
                              {"soil_water": s.replace(pet=None), "iface": {"pet": s.pet}})  # fmt: skip
        np.testing.assert_array_equal(np.asarray(batched[k]), np.asarray(single))


# ============================================================================ declaration
def test_module_declaration_matches_the_code() -> None:
    for entry, key in MODULE.entries:
        proc = lookup(key)
        assert proc is not None, key
        assert entry.startswith(MODULE.slot + ".")
    assert set(MODULE.ledger[0].inflows) == set(BUCKET_LEDGER_INFLOWS)
    assert set(MODULE.ledger[0].outflows) == set(BUCKET_LEDGER_OUTFLOWS)
    inflows, outflows = bucket_ledger_channels()
    assert set(inflows) == set(BUCKET_LEDGER_INFLOWS) and set(outflows) == set(BUCKET_LEDGER_OUTFLOWS)
    meta = BucketParams.field_metadata()
    meta.update({f"soil.{k}": v for k, v in BucketSoil.field_metadata().items()})
    for prm in MODULE.parameters:
        assert prm.path in meta, prm.path
        assert meta[prm.path]["unit"] == prm.unit, (prm.path, meta[prm.path]["unit"], prm.unit)
    smeta = BucketState.field_metadata()
    for name, _ in MODULE.inputs_own:
        assert name in smeta
    fmeta = BucketForcing.field_metadata()
    for names, _ in MODULE.forcing:
        for nm in names.split(", "):
            assert nm in fmeta, nm


def test_sink_record_is_the_contract_record() -> None:
    s = BucketState.initial(np.full(4, 0.2))
    assert type(s.sink_in) is SinkInputs and s.theta.shape == (4,)


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="float64 statement")
@pytest.mark.allow_skip(reason="float64 comparison, skipped in the float32 tier")
def test_integration_real4_store_survives_jit() -> None:
    """With ``real4`` the jitted INTEGR stores the REAL*4 value of the 1e-6-rounded content on the
    backend JAX runs on (the store must not be folded away, as a convert pair is on GPU)."""
    rng = np.random.default_rng(11)
    n = 256
    sw = jnp.asarray(rng.uniform(0.05, 0.45, (n, 4)))
    dl = jnp.broadcast_to(jnp.asarray([5.0, 10.0, 15.0, 20.0]), (n, 4))
    deltas = jnp.asarray(rng.uniform(-0.01, 0.01, (n, 4)))
    es = jnp.asarray(rng.uniform(0.0, 2.0, n))
    f = jax.jit(lambda *a: K.integrate_sw(*a, real4=True)[0])
    g = jax.jit(lambda *a: K.integrate_sw(*a, real4=False)[0])
    r4 = np.asarray(f(sw, dl, dl, es, deltas, True))
    r8 = np.asarray(g(sw, dl, dl, es, deltas, True))
    np.testing.assert_array_equal(r4, r8.astype(np.float32).astype(np.float64))
    assert np.mean(r4 != r8) > 0.9


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="float64 statement")
@pytest.mark.allow_skip(reason="float64 statement, skipped in the float32 tier")
def test_integration_real4_store_is_reduce_precision_in_the_hlo() -> None:
    """Structural guard, visible on CPU: the jitted INTEGR with ``real4`` carries a
    ``reduce-precision`` in its lowered and compiled HLO (a convert pair instead would pass on CPU
    but vanish on GPU, see :func:`agrijax.core.grad.real4_store`)."""
    sw = jnp.full((3, 4), 0.25)
    dl = jnp.broadcast_to(jnp.asarray([5.0, 10.0, 15.0, 20.0]), (3, 4))
    deltas = jnp.full((3, 4), 0.001)
    es = jnp.full((3,), 0.5)
    f = jax.jit(lambda *a: K.integrate_sw(*a, real4=True)[0])
    lowered = f.lower(sw, dl, dl, es, deltas, True)
    assert "reduce_precision" in lowered.as_text()
    assert "reduce-precision" in lowered.compile().as_text()
    # and without real4 there is none (the guard looks at the right thing)
    g = jax.jit(lambda *a: K.integrate_sw(*a, real4=False)[0])
    assert "reduce_precision" not in g.lower(sw, dl, dl, es, deltas, True).as_text()
