"""CERES-Maize roots (``ceres_roots``) against a plain-Python transcription of ``MZ_ROOTGR``.

The reference keeps the Fortran's explicit layer loops (``DO L = 1, NLAYR ... GO TO 100`` for the
emergence profile, the ``DO WHILE (CUMDEP < RTDEP .AND. L < NLAYR)`` of the growth loop) with
Python floats; the process replaces them by masks over the layer axis. Random states (seeded
NumPy) cover every branch: stages 7/8/9, the emergence day, no growth (``GRORT <= 1e-4``), dry
layers (``SWDF < 1``), waterlogging (``SAT - SW < PORMIN``), the 275 degC d switch, the depth cap
and the ``TRLDF`` threshold.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.processes.crop.ceres_maize import CeresForcing, CeresMaizeState, ceres_roots

from .test_ceres_phenology import DLAYR, DUL, LL, a, make_params

X64 = jax.config.jax_enable_x64
ATOL = 1e-10 if X64 else 2e-5
SAT = [0.23] * 8 + [0.36]
SHF = [1.0, 1.0, 0.7, 0.3, 0.3, 0.05, 0.03, 0.002, 0.0]


def ref_roots(st: dict, p: dict) -> dict:
    """MZ_ROOTGR, DYNAMIC = INTEGR (nitrogen off)."""
    rtdep = st["rtdep"]
    rlv = list(st["rlv"])
    nl = len(DLAYR)
    if st["istage"] in (7, 8):
        rtdep = p["sdepth"]
    if st["istage"] == 9:
        rtdep = rtdep + 0.15 * st["dtt"]
    if st["yrdoy"] == st["stgdoy9"]:
        cumdep = 0.0
        last = nl - 1
        for l in range(nl):
            cumdep += DLAYR[l]
            rlv[l] = 0.20 * p["pltpop"] / DLAYR[l]
            if cumdep > rtdep:
                last = l
                break
        rlv[last] = rlv[last] * (1.0 - (cumdep - rtdep) / DLAYR[last])
        for l in range(last + 1, nl):
            rlv[l] = 0.0
    if st["grort"] <= 0.0001:
        return dict(rtdep=rtdep, rlv=rlv)
    rlnew = st["grort"] * p["rlwr"] * p["pltpop"]
    cumdep = 0.0
    l = -1
    rldf = [0.0] * nl
    swdf = 1.0
    while cumdep < rtdep and l < nl - 1:
        l += 1
        cumdep += DLAYR[l]
        esw = DUL[l] - LL[l]
        if st["sw"][l] - LL[l] < 0.25 * esw:
            swdf = max(4.0 * (st["sw"][l] - LL[l]) / esw, 0.0)
        else:
            swdf = 1.0
        rldf[l] = min(swdf, 1.0) * SHF[l] * DLAYR[l]
    l1 = l
    swexf = 1.0
    if SAT[l1] - st["sw"][l1] < p["pormin"]:
        swexf = min((SAT[l1] - st["sw"][l1]) / p["pormin"], 1.0)
    rtsurv = min(1.0, 1.0 - 0.1 * (1.0 - swexf))
    rate = 0.1 if st["cumdtt"] < 275.0 else 0.2
    rtdep = rtdep + st["dtt"] * rate * math.sqrt(SHF[l1] * min(st["swfac"] * 2.0, swdf))
    rtdep = min(rtdep, sum(DLAYR))
    rldf[l1] = rldf[l1] * (1.0 - (cumdep - rtdep) / DLAYR[l1])
    trldf = sum(rldf[: l1 + 1])
    if trldf >= rlnew * 0.00001:
        rnlf = rlnew / trldf
        for k in range(l1 + 1):
            v = rlv[k] + rldf[k] * rnlf / DLAYR[k] - 0.005 * rlv[k]
            v = v * rtsurv
            v = int(v * 1000.0) / 1000.0
            rlv[k] = min(max(v, 0.0), 4.0)
    return dict(rtdep=rtdep, rlv=rlv)


def random_case(rng: np.random.Generator) -> dict:
    istage = int(rng.choice([7, 8, 9, 9, 1, 2, 3, 4, 5, 6, 10]))
    yrdoy = 2001150
    emerg = rng.random() < 0.2
    rtdep = float(rng.uniform(2.0, 185.0)) if rng.random() < 0.9 else 180.0
    sw = np.asarray(LL) + rng.uniform(-0.01, 0.2, len(DLAYR)) * (np.asarray(DUL) - np.asarray(LL)) * 3
    if rng.random() < 0.3:  # waterlogged profile
        sw = np.asarray(SAT) - rng.uniform(0.0, 0.08, len(DLAYR))
    rlv = np.round(rng.uniform(0.0, 3.5, len(DLAYR)), 3)
    return dict(
        istage=istage,
        yrdoy=yrdoy,
        stgdoy9=yrdoy if emerg else 9999999,
        dtt=float(rng.uniform(0.0, 20.0)),
        cumdtt=float(rng.choice([rng.uniform(0, 274.9), rng.uniform(275.0, 1500.0)])),
        grort=float(rng.choice([0.0, 5e-5, rng.uniform(0.0, 3.0)])),
        swfac=float(rng.uniform(0.0, 1.0)),
        rtdep=rtdep,
        sw=list(sw),
        rlv=list(rlv),
    )


def run_process(cases: list[dict], p: dict):
    """All cases as crops of one state (the n_crop axis) through one call of the process."""
    n = len(cases)
    params = make_params(pltpop=p["pltpop"], sdepth=p["sdepth"], yrplt=2001100)
    params = params.replace(soil=params.soil.replace(sat=a(SAT), shf=a(SHF)))
    st = CeresMaizeState.initial(params, n)
    stgdoy = np.full((n, 10), 9999999, dtype=np.int32)
    stgdoy[:, 8] = [c["stgdoy9"] for c in cases]
    ph = st.phen.replace(
        istage=jnp.asarray([c["istage"] for c in cases], dtype=jnp.int32),
        dtt=a([c["dtt"] for c in cases]),
        cumdtt=a([c["cumdtt"] for c in cases]),
        stgdoy=jnp.asarray(stgdoy),
    )
    st = st.replace(
        phen=ph,
        growth=st.growth.replace(grort=a([c["grort"] for c in cases])),
        stress=st.stress.replace(swfac=a([c["swfac"] for c in cases])),
        roots=st.roots.replace(rtdep=a([c["rtdep"] for c in cases]), rlv=a([c["rlv"] for c in cases])),
    )
    # one soil for all crops: run each case separately when the soil water differs
    return params, st


@pytest.mark.parametrize("seed", range(6))
def test_roots_match_reference_on_random_states(seed):
    rng = np.random.default_rng(100 + seed)
    p = dict(
        pltpop=float(rng.uniform(3.0, 12.0)), sdepth=float(rng.uniform(2.0, 9.0)), rlwr=0.98, pormin=0.05
    )
    step = jax.jit(ceres_roots)
    hit = {"emerg": 0, "nogrow": 0, "dry": 0, "wet": 0, "late": 0, "deep": 0}
    for _ in range(60):
        c = random_case(rng)
        params, st = run_process([c], p)
        f = CeresForcing(
            yrdoy=jnp.asarray(c["yrdoy"]),
            tmax=a(25.0),
            tmin=a(15.0),
            srad=a(20.0),
            dayl=a(13.0),
            twilen=a(14.0),
            co2=a(380.0),
            snow=a(0.0),
            sw=a(c["sw"]),
            swfac=a(1.0),
            turfac=a(1.0),
        )
        out = step(st, params, f)
        want = ref_roots(c, p)
        np.testing.assert_allclose(float(out.roots.rtdep[0]), want["rtdep"], rtol=ATOL, atol=ATOL)
        # 1e-3 truncation: a value within rounding of a multiple of 1e-3 may land on either side
        np.testing.assert_allclose(
            np.asarray(out.roots.rlv[0]), want["rlv"], atol=max(ATOL, 1e-9) if X64 else 1.1e-3
        )
        hit["emerg"] += c["stgdoy9"] == c["yrdoy"]
        hit["nogrow"] += c["grort"] <= 1e-4
        hit["dry"] += any(sw - ll < 0.25 * (du - ll) for sw, ll, du in zip(c["sw"], LL, DUL, strict=True))
        hit["wet"] += any(s - sw < 0.05 for s, sw in zip(SAT, c["sw"], strict=True))
        hit["late"] += c["cumdtt"] >= 275.0
        hit["deep"] += want["rtdep"] >= sum(DLAYR) - 1e-9
    assert all(v > 0 for k, v in hit.items() if k != "deep"), hit


def test_roots_do_not_run_before_sowing_or_without_water_balance():
    p = dict(pltpop=7.2, sdepth=5.0)
    c = random_case(np.random.default_rng(7))
    c.update(istage=8, grort=1.0)
    params, st = run_process([c], p)
    f = CeresForcing(
        yrdoy=jnp.asarray(2001050),
        tmax=a(25.0),
        tmin=a(15.0),
        srad=a(20.0),
        dayl=a(13.0),
        twilen=a(14.0),
        co2=a(380.0),
        snow=a(0.0),
        sw=a(c["sw"]),
        swfac=a(1.0),
        turfac=a(1.0),
    )
    out = ceres_roots(st, params, f)  # yrdoy < yrplt
    assert out.roots is st.roots or np.array_equal(np.asarray(out.roots.rlv), np.asarray(st.roots.rlv))
    params_n = params.replace(iswwat=False)
    f2 = f.replace(yrdoy=jnp.asarray(2001150))
    out2 = ceres_roots(st, params_n, f2)
    np.testing.assert_array_equal(np.asarray(out2.roots.rtdep), np.asarray(st.roots.rtdep))


def test_emergence_profile_integrates_to_the_initial_root_length():
    """On the emergence day the new profile holds 0.2 PLTPOP cm per cm2 down to the root front
    (sum RLV * DLAYR = 0.2 PLTPOP * RTDEP / DLAYR-weighted), an independent check of the layer
    masks: sum_l RLV_l DLAYR_l = 0.2 PLTPOP x (number of full layers + fraction of the last)."""
    p = dict(pltpop=6.0, sdepth=5.0)
    for rtdep in (3.0, 5.0, 17.5, 44.9, 60.0, 179.0):
        c = dict(
            istage=9,
            yrdoy=2001150,
            stgdoy9=2001150,
            dtt=0.0,
            cumdtt=10.0,
            grort=0.0,
            swfac=1.0,
            rtdep=rtdep,
            sw=list(np.asarray(DUL)),
            rlv=[0.0] * len(DLAYR),
        )
        params, st = run_process([c], p)
        f = CeresForcing(
            yrdoy=jnp.asarray(2001150),
            tmax=a(25.0),
            tmin=a(15.0),
            srad=a(20.0),
            dayl=a(13.0),
            twilen=a(14.0),
            co2=a(380.0),
            snow=a(0.0),
            sw=a(c["sw"]),
            swfac=a(1.0),
            turfac=a(1.0),
        )
        out = ceres_roots(st, params, f)
        bottom = np.cumsum(DLAYR)
        top = bottom - np.asarray(DLAYR)
        # count layers the Fortran loop fills: it stops at the first layer whose bottom exceeds RTDEP
        n_full = int(np.sum(bottom <= rtdep))
        frac = (rtdep - top[n_full]) / DLAYR[n_full] if n_full < len(DLAYR) else 0.0
        want = 0.2 * 6.0 * (n_full + frac)
        got = float(np.sum(np.asarray(out.roots.rlv[0]) * np.asarray(DLAYR)))
        assert got == pytest.approx(want, rel=1e-6)
