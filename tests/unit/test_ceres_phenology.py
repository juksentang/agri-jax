"""CERES-Maize phenology (``ceres_phenology``, ``thermal_time``) against a plain-Python day loop.

The reference below is a scalar transcription of ``MZ_PHENOL.for`` (DSSAT-CSM v4.8.6.0, BSD-3):
``if / elif`` on the stage, a ``for`` loop over the 24 hours of the sine interpolation and over
the soil layers for the seed layer, ``math`` functions and Python floats. It shares no code with
the vectorised process (which evaluates every stage block for every crop and selects with
``jnp.where``), so agreement on every day of synthetic seasons that visit every stage, every
thermal-time branch and every failure path is an independent check of the stage machine.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.processes.crop.ceres_maize import (
    CeresCultivar,
    CeresForcing,
    CeresMaizeParams,
    CeresMaizeState,
    CeresSoil,
    CeresSpecies,
    ceres_phenology,
    ceres_water_replay,
    thermal_time,
)
from agrijax.processes.crop.ceres_maize._util import daylength, twilight_daylength

X64 = jax.config.jax_enable_x64
RTOL = 1e-9 if X64 else 2e-5

# ------------------------------------------------------------------ parameters (MZCER048, IB0035)
CUL = dict(
    p1=259.0,
    p2=1.193,
    p5=947.1,
    g2=924.3,
    g3=8.17,
    phint=43.0,
    tbase=8.0,
    topt=34.0,
    ropt=34.0,
    p2o=12.5,
    djti=4.0,
    gdde=6.0,
    dsgft=170.0,
    rue=4.2,
    tsen=6.0,
    cday=15.0,
)
SPE: dict[str, Any] = dict(
    prftc=(6.2, 16.5, 33.0, 44.0),
    rgfil=(5.5, 16.0, 27.0, 35.0),
    parsr=0.5,
    co2x=(0, 220, 280, 330, 400, 490, 570, 750, 990, 9999),
    co2y=(0.0, 0.85, 0.95, 1.0, 1.02, 1.04, 1.05, 1.06, 1.07, 1.08),
    fslfw=0.05,
    rsgr=0.1,
    rsgrt=5.0,
    carbot=7.0,
    dsgt=21.0,
    dget=150.0,
    swcg=0.02,
    stmwte=0.2,
    rtwte=0.2,
    lfwte=0.2,
    seedrve=0.2,
    leafnoe=1.0,
    plae=1.0,
    pormin=0.05,
    rwumx=0.03,
    rlwr=0.98,
    rwuep1=1.5,
    canht_pot=1.6,
    bsgdd=250.0,
)
DLAYR = [5.0, 10.0, 15.0, 15.0, 15.0, 30.0, 30.0, 30.0, 30.0]
LL = [0.026, 0.025, 0.025, 0.025, 0.025, 0.028, 0.028, 0.029, 0.070]
DUL = [0.096, 0.086, 0.086, 0.086, 0.086, 0.090, 0.090, 0.130, 0.258]


def a(x):
    return jnp.asarray(np.asarray(x, dtype=float))


def make_params(cul=None, pltpop=7.2, sdepth=7.0, yrplt=2001100, iswwat=True) -> CeresMaizeParams:
    c = dict(CUL)
    c.update(cul or {})
    return CeresMaizeParams(
        cultivar=CeresCultivar(**{k: a(v) for k, v in c.items()}),
        species=CeresSpecies(**{k: a(v) for k, v in SPE.items()}),
        soil=CeresSoil(
            dlayr=a(DLAYR), ll=a(LL), dul=a(DUL), sat=a([0.23] * 8 + [0.36]), shf=a([1.0] * 9), slpf=a(0.92)
        ),
        pltpop=a(pltpop),
        sdepth=a(sdepth),
        rowspc=a(61.0),
        yrplt=jnp.asarray(yrplt, dtype=jnp.int32),
        iswwat=iswwat,
    )


# ------------------------------------------------------------------ the plain-Python reference
def ref_daylength(doy: float, lat: float) -> float:
    pi = 3.14159
    rad = pi / 180.0
    dec = -23.45 * math.cos(2.0 * pi * (doy + 10.0) / 365.0)
    soc = min(max(math.tan(rad * dec) * math.tan(rad * lat), -1.0), 1.0)
    return min(max(12.0 + 24.0 * math.asin(soc) / pi, 0.0), 24.0)


def ref_twilight(doy: float, lat: float) -> float:
    s1 = math.sin(lat * 0.01745)
    c1 = math.cos(lat * 0.01745)
    dec = 0.4093 * math.sin(0.0172 * (doy - 82.2))
    dlv = (-s1 * math.sin(dec) - 0.1047) / (c1 * math.cos(dec))
    dlv = min(max(dlv, -0.87), 1.0)
    return 7.639 * math.acos(dlv)


def ref_dtt(tmax, tmin, srad, dayl, snow, leafno, istage, c) -> float:
    tempcn, tempcx = tmin, tmax
    xs = min(snow, 15.0)
    if tmin < 0.0:
        tempcn = 2.0 + tmin * (0.4 + 0.0018 * (xs - 15.0) ** 2)
    if tmax < 0.0:
        tempcx = 2.0 + tmax * (0.4 + 0.0018 * (xs - 15.0) ** 2)
    dopt = c["topt"]
    if 3 < istage <= 6:
        dopt = c["ropt"]
    tbase = c["tbase"]
    if tmax < tbase:
        dtt = 0.0
    elif tmin > dopt:
        dtt = dopt - tbase
    elif leafno <= 10:
        if xs > 0.0:
            dtt = (tempcn + tempcx) / 2.0 - tbase
        else:
            acoef = 0.01061 * srad + 0.5902
            tdsoil = acoef * tmax + (1.0 - acoef) * tmin
            tnsoil = 0.36354 * tmax + 0.63646 * tmin
            if tdsoil < tbase:
                dtt = 0.0
            else:
                if tnsoil < tbase:
                    tnsoil = tbase
                if tdsoil > dopt:
                    tdsoil = dopt
                tmsoil = tdsoil * (dayl / 24.0) + tnsoil * ((24.0 - dayl) / 24.0)
                if tmsoil < tbase:
                    dtt = (tbase + tdsoil) / 2.0 - tbase
                else:
                    dtt = (tnsoil + tdsoil) / 2.0 - tbase
                dtt = min(dtt, dopt - tbase)
    elif tmin < tbase or tmax > dopt:
        dtt = 0.0
        for i in range(1, 25):
            th = (tmax + tmin) / 2.0 + (tmax - tmin) / 2.0 * math.sin(3.14 / 12.0 * i)
            th = min(max(th, tbase), dopt)
            dtt += (th - tbase) / 24.0
    else:
        dtt = (tmax + tmin) / 2.0 - tbase
    return max(dtt, 0.0)


def ref_init() -> dict:
    return dict(
        istage=7,
        sumdtt=0.0,
        cumdtt=0.0,
        dtt=0.0,
        ndas=0.0,
        xstage=0.1,
        sind=0.0,
        p3=0.0,
        p9=0.0,
        tlno=0.0,
        xnti=0.0,
        gpp=0.0,
        ears=0.0,
        idurp=0,
        l0=0,
        stgdoy=[9999999] * 10,
        mdate=-99,
        status=0,
    )


def ref_day(
    st: dict, c: dict, spe: dict, pltpop: float, sdepth: float, yrplt: int, iswwat: bool, wx: dict, grow: dict
) -> tuple[dict, float]:
    """One MZ_PHENOL INTEGR day; returns (state, pltpop)."""
    st = dict(st)
    st["stgdoy"] = list(st["stgdoy"])
    yrdoy = wx["yrdoy"]
    if not (yrdoy == yrplt or st["istage"] != 7):
        return st, pltpop
    dtt = ref_dtt(wx["tmax"], wx["tmin"], wx["srad"], wx["dayl"], wx["snow"], grow["leafno"], st["istage"], c)
    st["dtt"] = dtt
    st["sumdtt"] += dtt
    st["cumdtt"] += dtt
    s = st["istage"]
    if s == 7:
        st["stgdoy"][6] = yrdoy
        st["ndas"] = 0.0
        st["istage"] = 8
        st["sumdtt"] = 0.0
        if not iswwat:
            return st, pltpop
        cum = 0.0
        l0 = len(DLAYR) - 1
        for i, d in enumerate(DLAYR):
            cum += d
            if sdepth < cum:
                l0 = i
                break
        st["l0"] = l0
    elif s == 8:
        if iswwat:
            l0 = st["l0"]
            l1 = min(l0 + 1, len(DLAYR) - 1)
            sw = wx["sw"]
            if sw[l0] <= LL[l0]:
                swsd = (sw[l0] - LL[l0]) * 0.65 + (sw[l1] - LL[l1]) * 0.35
                st["ndas"] += 1
                if st["ndas"] >= spe["dsgt"]:
                    st["istage"] = 6
                    st["gpp"] = 1.0
                    st["mdate"] = yrdoy
                    st["status"] = 12
                    return st, 0.0
                if swsd < spe["swcg"]:
                    return st, pltpop
        st["stgdoy"][7] = yrdoy
        st["istage"] = 9
        st["cumdtt"] = 0.0
        st["sumdtt"] = 0.0
        st["p9"] = 45.0 + c["gdde"] * sdepth
    elif s == 9:
        st["ndas"] += 1
        if st["sumdtt"] < st["p9"]:
            return st, pltpop
        if st["p9"] > spe["dget"]:
            st["istage"] = 6
            st["gpp"] = 1.0
            st["mdate"] = yrdoy
            st["status"] = 13
            return st, 0.0
        st["stgdoy"][8] = yrdoy
        st["istage"] = 1
        st["sumdtt"] -= st["p9"]
        st["tlno"] = 30.0
    elif s == 1:
        st["ndas"] += 1
        st["xstage"] = st["sumdtt"] / c["p1"]
        if st["sumdtt"] < c["p1"]:
            return st, pltpop
        st["stgdoy"][0] = yrdoy
        st["istage"] = 2
        st["sind"] = 0.0
    elif s == 2:
        st["ndas"] += 1
        st["xstage"] = 1.0 + 0.5 * st["sind"]
        if wx["twilen"] > c["p2o"]:
            ratein = 1.0 / (c["djti"] + c["p2"] * (wx["twilen"] - c["p2o"]))
        else:
            ratein = 1.0 / c["djti"]
        st["sind"] += ratein
        if st["sind"] < 1.0:
            return st, pltpop
        st["stgdoy"][1] = yrdoy
        st["istage"] = 3
        st["tlno"] = st["sumdtt"] / (c["phint"] * 0.5) + 5.0
        st["p3"] = (st["tlno"] + 0.5) * c["phint"] - st["sumdtt"]
        st["xnti"] = grow["xn"]
        st["sumdtt"] = 0.0
    elif s == 3:
        st["ndas"] += 1
        st["xstage"] = 1.5 + 3.0 * st["sumdtt"] / st["p3"]
        if st["sumdtt"] < st["p3"]:
            return st, pltpop
        st["stgdoy"][2] = yrdoy
        st["istage"] = 4
        st["sumdtt"] -= st["p3"]
        st["idurp"] = 0
    elif s == 4:
        st["ndas"] += 1
        st["idurp"] += 1
        st["xstage"] = 4.5 + 5.5 * st["sumdtt"] / (c["p5"] * 0.95)
        if st["sumdtt"] < c["dsgft"]:
            return st, pltpop
        psker = grow["sump"] * 1000.0 / st["idurp"] * 3.4 / 5.0
        gpp = c["g2"] * psker / 7200.0 + 50.0
        gpp = max(min(gpp, c["g2"]), 0.0)
        ears = pltpop
        gpp = max(gpp, 51.0)
        if gpp < c["g2"] * 0.15:
            ears = pltpop * (gpp / (c["g2"] * 0.15)) ** 0.33
        elif pltpop > 12.0 and gpp < c["g2"] * 0.5:
            barfac = 0.0085 * (1.0 - gpp / c["g2"]) * pltpop**1.5
            ears = pltpop * (gpp / (c["g2"] * 0.50)) ** barfac
        st["gpp"] = gpp
        st["ears"] = max(ears, 0.0)
        st["stgdoy"][3] = yrdoy
        st["istage"] = 5
    elif s == 5:
        st["ndas"] += 1
        st["xstage"] = 4.5 + 5.5 * st["sumdtt"] / c["p5"]
        if st["sumdtt"] < c["p5"] * 0.95:
            return st, pltpop
        st["stgdoy"][4] = yrdoy
        st["istage"] = 6
    elif s == 6:
        if dtt < 2.0:
            st["sumdtt"] = c["p5"]
        if st["sumdtt"] < c["p5"]:
            return st, pltpop
        st["stgdoy"][5] = yrdoy
        st["mdate"] = yrdoy
        st["status"] = 1
        st["istage"] = 10
        st["cumdtt"] = 0.0
        st["dtt"] = 0.0
        if pltpop != 0.0 and st["gpp"] <= 0.0:
            st["gpp"] = 1.0
    return st, pltpop


# ------------------------------------------------------------------ synthetic seasons
def season(
    seed: int,
    n: int = 260,
    *,
    lat: float = 29.6,
    cold: bool = False,
    dry_days: int = 0,
    always_dry: bool = False,
    start: int = 2001090,
) -> dict:
    """Deterministic synthetic weather (NumPy RNG), soil water and growth feedback per day."""
    rng = np.random.default_rng(seed)
    days = np.arange(n)
    doy = (start % 1000 + days - 1) % 365 + 1
    yrdoy = (start // 1000) * 1000 + doy
    base = 24.0 + 8.0 * np.sin(2 * np.pi * (doy - 110) / 365.0)
    if cold:
        base = base - 17.0
    tmax = base + 5.0 + rng.normal(0.0, 3.0, n)
    tmin = base - 7.0 + rng.normal(0.0, 3.0, n)
    tmin = np.minimum(tmin, tmax - 0.5)
    srad = np.clip(18.0 + rng.normal(0.0, 5.0, n), 1.0, 30.0)
    snow = np.where(tmax < 2.0, rng.uniform(0.0, 30.0, n), 0.0) if cold else np.zeros(n)
    dayl = np.array([ref_daylength(float(d), lat) for d in doy])
    twilen = np.array([ref_twilight(float(d), lat) for d in doy])
    sw = np.tile(np.asarray(DUL) - 0.01, (n, 1))
    if always_dry:
        sw[:] = np.asarray(LL) - 0.002
    elif dry_days:
        sw[:dry_days] = np.asarray(LL) - 0.002
    # growth feedback read by the phenology (leaf number, oldest expanding leaf, stage-4 assimilation)
    cum = np.cumsum(np.maximum((tmax + tmin) / 2 - 8.0, 0.0))
    xn = 1.0 + np.maximum(cum - 120.0, 0.0) / 40.0
    leafno = np.where(days < 12, 0, np.floor(xn)).astype(int)
    sump = np.maximum(days - 60, 0) * 0.9
    return dict(
        yrdoy=yrdoy.astype(np.int32),
        doy=doy,
        tmax=tmax,
        tmin=tmin,
        srad=srad,
        snow=snow,
        dayl=dayl,
        twilen=twilen,
        sw=sw,
        xn=xn,
        leafno=leafno,
        sump=sump,
    )


def forcing_day(w: dict, t: int) -> CeresForcing:
    return CeresForcing(
        yrdoy=jnp.asarray(w["yrdoy"][t]),
        tmax=a(w["tmax"][t]),
        tmin=a(w["tmin"][t]),
        srad=a(w["srad"][t]),
        dayl=a(w["dayl"][t]),
        twilen=a(w["twilen"][t]),
        co2=a(380.0),
        snow=a(w["snow"][t]),
        sw=a(w["sw"][t]),
        eop=a(0.0),
        trwup=a(0.0),
    )


# the phenology reads the soil water of the crop's water port, which the replay fills from the forcing
_STEP = jax.jit(lambda s, p, f: ceres_phenology(ceres_water_replay(s, p, f), p, f))


def run_both(
    params: CeresMaizeParams,
    cul: dict,
    w: dict,
    sdepth: float,
    pltpop: float,
    yrplt: int,
    iswwat: bool,
    n_crop: int = 1,
):
    """Run the JAX process and the reference day by day; return both trajectories."""
    state = CeresMaizeState.initial(params, n_crop)
    ref = ref_init()
    ref_pop = pltpop
    out_jax, out_ref = [], []
    for t in range(len(w["yrdoy"])):
        grow = dict(leafno=int(w["leafno"][t]), xn=float(w["xn"][t]), sump=float(w["sump"][t]))
        g = state.growth
        g = g.replace(
            leafno=jnp.full_like(g.leafno, grow["leafno"]),
            xn=jnp.full_like(g.xn, grow["xn"]),
            sump=jnp.full_like(g.sump, grow["sump"]),
        )
        state = state.replace(growth=g)
        state = _STEP(state, params, forcing_day(w, t))
        ref, ref_pop = ref_day(
            ref,
            cul,
            SPE,
            ref_pop,
            sdepth,
            yrplt,
            iswwat,
            dict(
                yrdoy=int(w["yrdoy"][t]),
                tmax=w["tmax"][t],
                tmin=w["tmin"][t],
                srad=w["srad"][t],
                dayl=w["dayl"][t],
                twilen=w["twilen"][t],
                snow=w["snow"][t],
                sw=list(w["sw"][t]),
            ),
            grow,
        )
        out_jax.append(jax.tree_util.tree_map(np.asarray, (state.phen, state.growth.pltpop)))
        out_ref.append((dict(ref), ref_pop))
    return out_jax, out_ref


def check_equal(out_jax, out_ref, crop: int = 0):
    for t, ((ph, pop), (r, rpop)) in enumerate(zip(out_jax, out_ref, strict=True)):
        ctx = f"day {t}"
        assert int(ph.istage[crop]) == r["istage"], ctx
        assert list(ph.stgdoy[crop]) == r["stgdoy"], ctx
        assert int(ph.mdate[crop]) == r["mdate"], ctx
        assert int(ph.crop_status[crop]) == r["status"], ctx
        assert int(ph.idurp[crop]) == r["idurp"], ctx
        for k in (
            "sumdtt",
            "cumdtt",
            "dtt",
            "ndas",
            "xstage",
            "sind",
            "p3",
            "p9",
            "tlno",
            "xnti",
            "gpp",
            "ears",
        ):
            np.testing.assert_allclose(
                ph.__getattribute__(k)[crop], r[k], rtol=RTOL, atol=RTOL * 10, err_msg=f"{k} {ctx}"
            )
        np.testing.assert_allclose(pop[crop], rpop, rtol=RTOL, err_msg=ctx)


def _stages_visited(out_ref) -> set[int]:
    return {r["istage"] for r, _ in out_ref}


# ------------------------------------------------------------------ tests
def test_normal_season_matches_reference_every_day():
    w = season(1)
    p = make_params(yrplt=int(w["yrdoy"][3]))
    out_jax, out_ref = run_both(p, CUL, w, 7.0, 7.2, int(w["yrdoy"][3]), True)
    check_equal(out_jax, out_ref)
    assert _stages_visited(out_ref) >= {8, 9, 1, 2, 3, 4, 5, 6, 10}
    assert out_ref[-1][0]["status"] == 1


def test_high_latitude_photoperiod_and_high_population_barrenness():
    """TWILEN > P2O slows induction (stage 2); PLTPOP > 12 with GPP < G2/2 triggers barrenness."""
    w = season(2, lat=48.0, start=2001120)
    w["sump"] = np.minimum(w["sump"], 0.8)  # little assimilation in stage 4 -> low GPP
    cul = dict(CUL, p2=0.8)
    p = make_params(cul=cul, pltpop=14.0, yrplt=int(w["yrdoy"][0]))
    assert (w["twilen"] > CUL["p2o"]).mean() > 0.5
    out_jax, out_ref = run_both(p, dict(CUL, **cul), w, 7.0, 14.0, int(w["yrdoy"][0]), True)
    check_equal(out_jax, out_ref)
    last = out_ref[-1][0]
    assert last["ears"] < 14.0 and last["gpp"] < CUL["g2"] * 0.5


def test_very_low_kernel_number_smooth_ear_reduction():
    """GPP below 0.15 G2: EARS = PLTPOP (GPP / (0.15 G2))^0.33."""
    w = season(3)
    w["sump"] = np.zeros_like(w["sump"])
    cul = dict(CUL, g2=600.0)
    p = make_params(cul=cul, yrplt=int(w["yrdoy"][0]))
    out_jax, out_ref = run_both(p, dict(CUL, **cul), w, 7.0, 7.2, int(w["yrdoy"][0]), True)
    check_equal(out_jax, out_ref)
    assert out_ref[-1][0]["gpp"] == 51.0 and out_ref[-1][0]["ears"] < 7.2


def test_cold_season_snow_and_all_thermal_time_branches():
    w = season(4, cold=True, n=300, start=2001060)
    p = make_params(yrplt=int(w["yrdoy"][0]))
    out_jax, out_ref = run_both(p, CUL, w, 7.0, 7.2, int(w["yrdoy"][0]), True)
    check_equal(out_jax, out_ref)
    assert (w["snow"] > 0).any() and (w["tmin"] < 0).any() and (w["tmax"] < CUL["tbase"]).any()


def test_dry_seed_layer_delays_germination():
    w = season(5, dry_days=9)
    p = make_params(yrplt=int(w["yrdoy"][1]))
    out_jax, out_ref = run_both(p, CUL, w, 7.0, 7.2, int(w["yrdoy"][1]), True)
    check_equal(out_jax, out_ref)
    germ = out_ref[-1][0]["stgdoy"][7]
    assert germ == int(w["yrdoy"][9])


def test_germination_failure_after_dsgt_dry_days():
    w = season(6, always_dry=True, n=120)
    p = make_params(yrplt=int(w["yrdoy"][0]))
    out_jax, out_ref = run_both(p, CUL, w, 7.0, 7.2, int(w["yrdoy"][0]), True)
    check_equal(out_jax, out_ref)
    # the failure day: stage 6, population 0, MDATE set (as in DSSAT the season would be harvested
    # on MDATE; run on, the stage-6 block later "matures" the dead crop exactly as the Fortran does)
    fail = next(t for t, (r, _) in enumerate(out_ref) if r["status"] == 12)
    assert out_ref[fail][0]["istage"] == 6 and out_ref[fail][0]["mdate"] == int(w["yrdoy"][fail])
    assert fail == int(SPE["dsgt"])  # sowing day, then DSGT dry days in stage 8
    assert float(out_jax[-1][1][0]) == 0.0


def test_emergence_failure_deep_sowing():
    """P9 = 45 + GDDE * SDEPTH > DGET = 150 for a 20 cm sowing depth."""
    w = season(7, n=120)
    p = make_params(sdepth=20.0, yrplt=int(w["yrdoy"][0]))
    out_jax, out_ref = run_both(p, CUL, w, 20.0, 7.2, int(w["yrdoy"][0]), True)
    check_equal(out_jax, out_ref)
    assert any(r["status"] == 13 for r, _ in out_ref)


def test_without_water_balance_germinates_on_the_day_after_sowing():
    w = season(8, always_dry=True, n=200)
    p = make_params(yrplt=int(w["yrdoy"][0]), iswwat=False)
    out_jax, out_ref = run_both(p, CUL, w, 7.0, 7.2, int(w["yrdoy"][0]), False)
    check_equal(out_jax, out_ref)
    assert out_ref[-1][0]["stgdoy"][7] == int(w["yrdoy"][1])


def test_n_crop_axis_crops_follow_their_own_cultivars():
    """Three crops with different P1 / P5 / PHINT in one state each equal their own reference run."""
    w = season(9)
    cul3 = dict(p1=[200.0, 259.0, 320.0], p5=[800.0, 947.1, 1000.0], phint=[38.0, 43.0, 50.0])
    p = make_params(cul=cul3, yrplt=int(w["yrdoy"][2]))
    out_jax, _ = run_both(p, CUL, w, 7.0, 7.2, int(w["yrdoy"][2]), True, n_crop=3)
    for k in range(3):
        ck = dict(CUL, p1=cul3["p1"][k], p5=cul3["p5"][k], phint=cul3["phint"][k])
        pk = make_params(cul={kk: v[k] for kk, v in cul3.items()}, yrplt=int(w["yrdoy"][2]))
        _, out_ref = run_both(pk, ck, w, 7.0, 7.2, int(w["yrdoy"][2]), True)
        check_equal(out_jax, out_ref, crop=k)
    mat = [int(out_jax[-1][0].stgdoy[k][5]) for k in range(3)]
    assert len(set(mat)) == 3  # different cultivars mature on different days


def test_before_sowing_nothing_changes():
    w = season(10, n=20)
    p = make_params(yrplt=int(w["yrdoy"][10]))
    out_jax, out_ref = run_both(p, CUL, w, 7.0, 7.2, int(w["yrdoy"][10]), True)
    check_equal(out_jax, out_ref)
    for t in range(10):
        assert int(out_jax[t][0].istage[0]) == 7 and float(out_jax[t][0].sumdtt[0]) == 0.0


@pytest.mark.parametrize(
    ("tmax", "tmin", "leafno", "snow", "istage"),
    [
        (7.0, 2.0, 12, 0.0, 1),  # TMAX < TBASE
        (40.0, 35.0, 12, 0.0, 1),  # TMIN > DOPT
        (30.0, 15.0, 5, 0.0, 1),  # growing point below ground, soil temperature
        (30.0, 1.0, 5, 0.0, 1),  # soil temperature, TNSOIL clipped to TBASE
        (9.0, -3.0, 5, 0.0, 1),  # TDSOIL < TBASE
        (-2.0, -9.0, 5, 12.0, 9),  # snow cover, crown temperature
        (38.0, 20.0, 12, 0.0, 1),  # sine interpolation above DOPT
        (20.0, 4.0, 12, 0.0, 5),  # sine interpolation below TBASE, reproductive DOPT
        (30.0, 18.0, 12, 0.0, 3),  # daily mean
    ],
)
def test_thermal_time_branches_equal_reference(tmax, tmin, leafno, snow, istage):
    p = make_params(cul=dict(ropt=30.0))
    c = dict(CUL, ropt=30.0)
    for srad, dayl in ((5.0, 10.0), (25.0, 14.5)):
        got = float(
            thermal_time(
                a(tmax),
                a(tmin),
                a(srad),
                a(dayl),
                a(snow),
                jnp.asarray(leafno),
                jnp.asarray(istage),
                p.cultivar,
            )
        )
        want = ref_dtt(tmax, tmin, srad, dayl, snow, leafno, istage, c)
        assert got == pytest.approx(want, rel=RTOL, abs=1e-6)


def test_daylength_and_twilight_equal_reference_formulae():
    doy = np.arange(1, 366, dtype=float)
    for lat in (-45.0, 0.0, 29.63, 42.02, 66.0):
        got = np.asarray(daylength(a(doy), lat))
        tw = np.asarray(twilight_daylength(a(doy), lat))
        np.testing.assert_allclose(got, [ref_daylength(d, lat) for d in doy], rtol=RTOL, atol=1e-5)
        np.testing.assert_allclose(tw, [ref_twilight(d, lat) for d in doy], rtol=RTOL, atol=1e-5)
        if abs(lat) < 60.0:  # TWILIGHT caps DLV at -0.87 (21.9 h), DAYLEN reaches 24 h near the pole
            assert np.all(tw >= got)  # twilight lengthens the day
