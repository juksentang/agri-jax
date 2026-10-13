"""Conformance cases of CERES-Maize (``crop/ceres_maize.*``), its season boundaries and its water replay.

The inputs of a crop process are a mid-season state, so ``make`` runs a synthetic season first:
NumPy weather around a warm-season mean (the recipe of ``tests/unit/test_ceres_phenology.py``),
ample soil water with a few dry spells, the MZCER048 cultivar IB0035 of the unit tests, and the
crop on its own (``ceres_maize_model``: the water replay writes the ``water_in`` port). From the
end of day ``d`` the day's order is replayed up to the process under test, which is then run for
three days. Variants pick ``d``: ``vegetative`` (leaf expansion) and ``grain_fill``. The season is
computed once per seed and dtype.

The season start (``season_init``) starts from the end of day ``d`` with a two-row per-season
table and an event table whose flag is set on the first day (variant ``boundary``) or on no day
(``ordinary``).
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from agrijax.core.events import EventTable
from agrijax.core.model import Model
from agrijax.core.runtime import run
from agrijax.iface.crop import CanopyRecord, CropNIn
from agrijax.processes.crop.ceres_maize import (
    CROP_PROCESSES,
    CROP_PROCESSES_SMOOTHED,
    DSSAT_COEFFICIENTS,
    REPLAY_PROCESSES,
    CeresCultivar,
    CeresMaizeParams,
    CeresMaizeState,
    CeresReplayForcing,
    CeresSeasons,
    CeresSoil,
    CeresSpecies,
    SmoothingCoefficients,
    ceres_growth,
    ceres_growth_smoothed,
    ceres_phenology,
    ceres_phenology_smoothed,
    ceres_roots,
    ceres_snow_replay,
    ceres_stress,
    ceres_water_replay,
    with_soft_state,
)
from agrijax.processes.crop.ceres_maize._util import daylength, twilight_daylength

from ..case import CALIBRATABLE, ConformanceCase, GradSpec

# ------------------------------------------------------------------ MZCER048 IB0035, as the unit tests
CUL = dict(
    p1=259.0, p2=1.193, p5=947.1, g2=924.3, g3=8.17, phint=43.0, tbase=8.0, topt=34.0, ropt=34.0,
    p2o=12.5, djti=4.0, gdde=6.0, dsgft=170.0, rue=4.2, tsen=6.0, cday=15.0,
)  # fmt: skip
SPE: dict[str, Any] = dict(
    prftc=(6.2, 16.5, 33.0, 44.0), rgfil=(5.5, 16.0, 27.0, 35.0), parsr=0.5,
    co2x=(0, 220, 280, 330, 400, 490, 570, 750, 990, 9999),
    co2y=(0.0, 0.85, 0.95, 1.0, 1.02, 1.04, 1.05, 1.06, 1.07, 1.08),
    fslfw=0.05, fslfn=0.05, rsgr=0.1, rsgrt=5.0, carbot=7.0, dsgt=21.0, dget=150.0, swcg=0.02, stmwte=0.2,
    rtwte=0.2, lfwte=0.2, seedrve=0.2, leafnoe=1.0, plae=1.0, pormin=0.05, rwumx=0.03, rlwr=0.98,
    rwuep1=1.5, canht_pot=1.6, bsgdd=250.0,
)  # fmt: skip
DLAYR = (5.0, 10.0, 15.0, 15.0, 15.0, 30.0, 30.0, 30.0, 30.0)
LL = (0.026, 0.025, 0.025, 0.025, 0.025, 0.028, 0.028, 0.029, 0.070)
DUL = (0.096, 0.086, 0.086, 0.086, 0.086, 0.090, 0.090, 0.130, 0.258)
SAT = (0.23,) * 8 + (0.36,)
LATITUDE = 29.6
START = 2001090
N_SEASON = 130
N_DAYS = 3
#: the end-of-day index each variant starts from
DAY: dict[str, int] = {"vegetative": 35, "grain_fill": 85}
#: variant of the nitrogen replay growth that starts the day before the first day of effective
#: grain filling (``STGDOY(4)``), so that its first day applies the grain-number cap
GRAIN_NUMBER = "grain_number"
#: the day's order after the water replay (``MZ_CERES``); the process under test runs from the
#: state just before it
ORDER = ("phenology", "stress", "growth", "roots", "publish")
_WATER_IN = {"water_in": "iface.crop_water.{slot}"}
#: the ports each process uses (the slot-contract check requires a case to bind exactly those)
PORTS: dict[str, dict[str, str]] = {
    "phenology": {**_WATER_IN, "snow_in": "iface.snow"},
    "stress": _WATER_IN,
    "growth": {},
    "roots": _WATER_IN,
    "publish": {"root_out": "iface.root.{slot}"},
}
PORTS_N = {"n_in": "iface.crop_n.{slot}"}


def params(dtype: Any, yrplt: int) -> CeresMaizeParams:
    def a(x: Any) -> Any:
        return jnp.asarray(np.asarray(x, dtype=float), dtype)

    return CeresMaizeParams(
        cultivar=CeresCultivar(**{k: a(v) for k, v in CUL.items()}),
        species=CeresSpecies(**{k: a(v) for k, v in SPE.items()}),
        soil=CeresSoil(dlayr=a(DLAYR), ll=a(LL), dul=a(DUL), sat=a(SAT), shf=a((1.0,) * 9), slpf=a(0.92)),
        pltpop=a(7.2),
        sdepth=a(7.0),
        rowspc=a(61.0),
        yrplt=jnp.asarray(yrplt, dtype=jnp.int32),
        iswwat=True,
        coefficients=DSSAT_COEFFICIENTS.as_arrays(dtype),
    )


def weather(seed: int, dtype: Any, n: int = N_SEASON + N_DAYS) -> CeresReplayForcing:
    """Deterministic synthetic season (NumPy generator)."""
    rng = np.random.default_rng(seed)
    days = np.arange(n)
    doy = (START % 1000 + days - 1) % 365 + 1
    yrdoy = (START // 1000) * 1000 + doy
    base = 24.0 + 8.0 * np.sin(2 * np.pi * (doy - 110) / 365.0)
    tmax = base + 5.0 + rng.normal(0.0, 3.0, n)
    tmin = np.minimum(base - 7.0 + rng.normal(0.0, 3.0, n), tmax - 0.5)
    srad = np.clip(18.0 + rng.normal(0.0, 5.0, n), 1.0, 30.0)
    eop = rng.uniform(2.0, 6.0, n)
    dry = (days % 37) > 22
    trwup = np.where(dry, rng.uniform(0.2, 0.9, n) * 0.1 * eop, 2.0 * 0.1 * eop)
    sw = np.tile(np.asarray(DUL) - 0.01, (n, 1))

    def a(x: Any) -> Any:
        return jnp.asarray(np.asarray(x, dtype=float), dtype)

    d = jnp.asarray(doy, dtype)
    return CeresReplayForcing(
        yrdoy=jnp.asarray(yrdoy, jnp.int32),
        tmax=a(tmax),
        tmin=a(tmin),
        srad=a(srad),
        dayl=jnp.asarray(daylength(d, LATITUDE), dtype),
        twilen=jnp.asarray(twilight_daylength(d, LATITUDE), dtype),
        co2=a(np.full(n, 380.0)),
        snow=a(np.zeros(n)),
        sw=a(sw),
        eop=a(eop),
        trwup=a(trwup),
    )


_SEASON = Model(CeresMaizeState, [*REPLAY_PROCESSES, *CROP_PROCESSES], outputs=lambda s, p, f: s)
_RUN = jax.jit(lambda p, f, s: run(_SEASON, p, f, s))
_STEPS = {"phenology": ceres_phenology, "stress": ceres_stress, "growth": ceres_growth, "roots": ceres_roots}
#: the season and the steps of the non-faithful smoothed variant (phenology and blended growth)
_SEASON_SMOOTHED = Model(
    CeresMaizeState, [*REPLAY_PROCESSES, *CROP_PROCESSES_SMOOTHED], outputs=lambda s, p, f: s
)
_RUN_SMOOTHED = jax.jit(lambda p, f, s: run(_SEASON_SMOOTHED, p, f, s))
_STEPS_SMOOTHED = {**_STEPS, "phenology": ceres_phenology_smoothed, "growth": ceres_growth_smoothed}


@functools.lru_cache(maxsize=32)
def _season(seed: int, dtype_name: str, smoothed: bool = False) -> tuple[Any, Any, Any]:
    dtype = jnp.dtype(dtype_name)
    f = weather(seed, dtype)
    p = params(dtype, int(np.asarray(f.yrdoy)[2]))
    s0 = CeresMaizeState.initial(p, 1, dtype=dtype)
    if smoothed:
        p = p.replace(smoothing=SmoothingCoefficients().as_arrays(dtype))
        return _RUN_SMOOTHED(p, f, with_soft_state(s0)), f, p
    return _RUN(p, f, s0), f, p


@functools.lru_cache(maxsize=64)
def _pre(
    seed: int, dtype_name: str, variant: str, before: str, smoothed: bool = False
) -> tuple[Any, Any, Any]:
    traj, f, p = _season(seed, dtype_name, smoothed)
    d = _start_day(traj, f, variant)
    s = jax.tree_util.tree_map(lambda x: x[d], traj)
    ft = jax.tree_util.tree_map(lambda x: x[d + 1], f)
    s = ceres_snow_replay(ceres_water_replay(s, p, ft), p, ft)
    steps = _STEPS_SMOOTHED if smoothed else _STEPS
    for name in ORDER:
        if name == before:
            break
        s = steps[name](s, p, ft)
    days = jax.tree_util.tree_map(lambda x: x[d + 1 : d + 1 + N_DAYS], f)
    return s, p, days


def _start_day(traj: Any, f: Any, variant: str) -> int:
    """The end-of-day index a variant starts from (:data:`DAY`, or for :data:`GRAIN_NUMBER` the
    day before the first day of effective grain filling of the synthetic season)."""
    if variant != GRAIN_NUMBER:
        return DAY[variant]
    efg = np.asarray(traj.phen.stgdoy)[:, 0, 3] == np.asarray(f.yrdoy)[: len(np.asarray(traj.phen.stgdoy))]
    if not efg.any():
        raise ValueError("the synthetic season does not reach effective grain filling")
    return int(np.argmax(efg)) - 1


#: ranges of the nitrogen factors of the nitrogen replay growth case: NSTRES, AGEFAC, NDEF3 (-)
#: and NPOOL [g plant-1] (the cap NPOOL / (0.062 x 0.0095) = 170-1020 grains binds in some draws)
N_RANGES = {"nstres": (0.4, 0.9), "agefac": (0.4, 0.9), "ndef3": (0.5, 0.95), "npool": (0.1, 0.6)}


def maker(before: str, nstress: bool = False, smoothed: bool = False) -> Any:
    """``make`` of the case of the process ``before`` (the state just before it runs); with
    ``nstress`` the crop's nitrogen record holds drawn factors (:data:`N_RANGES`); with ``smoothed``
    the season runs the smoothed phenology and growth (soft clocks in ``state.soft``, array gate
    scale in ``params.smoothing``)."""

    def make(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
        seed = int(rng.integers(0, 2**31 - 1))
        s, p, f = _pre(seed, jnp.dtype(dtype).name, variant, before, smoothed)
        if nstress:
            dt = s.growth.lai.dtype
            n = {
                k: jnp.asarray(rng.uniform(lo, hi, np.shape(s.growth.lai)), dt)
                for k, (lo, hi) in N_RANGES.items()
            }
            s = s.replace(n_in=CropNIn(**n))
        return s, p, f

    return make


#: the season-boundary cases: flag on the first day (``boundary``) or on none (``ordinary``)
SEASON_VARIANTS = ("boundary", "ordinary")
#: sowing date of the second row of the per-season table, days after the first season's
SECOND_SEASON_AFTER = 400


def _events(dtype: Any, flag: str, variant: str) -> EventTable:
    """An event table of ``N_DAYS`` days with ``flag`` set on day 0 (``boundary``) or never."""
    ev = EventTable.empty(N_DAYS)
    on = jnp.asarray(np.arange(N_DAYS) == 0) if variant == "boundary" else jnp.zeros(N_DAYS, bool)
    ev = ev.replace(**{flag: on})
    return jax.tree_util.tree_map(
        lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, ev
    )


def season_maker(flag: str) -> Any:
    """``make`` of a season-boundary case: the end-of-day state of day ``grain_fill`` with a
    two-row per-season table (season 0 running), the ports filled, and the event ``flag``."""

    def make(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
        seed = int(rng.integers(0, 2**31 - 1))
        traj, _, p = _season(seed, jnp.dtype(dtype).name)
        s = jax.tree_util.tree_map(lambda x: x[DAY["grain_fill"]], traj)
        y0 = int(np.asarray(p.yrplt))
        table = CeresSeasons(
            yrplt=jnp.asarray([y0, y0 + SECOND_SEASON_AFTER], jnp.int32),
            pltpop=jnp.asarray(np.asarray(p.pltpop) * np.asarray([1.0, rng.uniform(0.8, 1.2)]), dtype),
            sdepth=jnp.asarray(np.asarray(p.sdepth) * np.asarray([1.0, rng.uniform(0.8, 1.2)]), dtype),
            rowspc=jnp.asarray(np.asarray(p.rowspc) * np.asarray([1.0, rng.uniform(0.8, 1.2)]), dtype),
        )
        p = p.replace(seasons=table)
        n_crop = s.roots.rlv.shape[0]
        u = lambda *shape: jnp.asarray(rng.uniform(0.1, 1.0, shape), dtype)  # noqa: E731
        s = s.replace(
            season=jnp.asarray(0, jnp.int32),
            canopy_out=CanopyRecord(lai=u(n_crop), tlai=u(n_crop), height=u(n_crop)),
            sowing=jnp.asarray(False),  # event-driven: the season initialisation writes the flag
        )
        return s, p, _events(dtype, flag, variant)

    return make


_NO_BALANCE = (
    "CERES-Maize conserves assimilate between organs, tested day by day against the model's own "
    "partitioning in tests/unit/test_ceres_growth.py; there is no stock with daily in- and outflows "
    "at the process boundary"
)
_VARIANTS = tuple(DAY)
_GRAD = GradSpec()
#: the smoothed variant: the hard-coded coefficients and the cultivar coefficients its gates smooth
_SMOOTHED_GRAD = GradSpec(
    wrt=(
        CALIBRATABLE,
        "cultivar.p1",
        "cultivar.p2",
        "cultivar.p5",
        "cultivar.phint",
        "cultivar.g2",
        "cultivar.g3",
    )
)
#: the season boundaries use no hard-coded coefficient; their parameters are the season table's
_SEASON_GRAD = GradSpec(wrt=("seasons.pltpop", "seasons.sdepth", "seasons.rowspc"))


def cases() -> list[ConformanceCase]:
    out = [
        ConformanceCase(
            key=f"crop/ceres_maize.{name}@dssat-4.8.6.0:faithful",
            make=maker(name),
            variants=_VARIANTS,
            n_days=N_DAYS,
            ports=PORTS[name],
            no_balance=_NO_BALANCE,
            grad=_GRAD,
            coefficient_sets=("coefficients",),
            # batched, phen.xstage differs from the per-sample run by 1 ulp (measured on a cluster CPU node)
            transforms_exact=name != "phenology",
        )
        for name in ORDER
    ]
    out.append(
        ConformanceCase(
            key="crop/ceres_maize.growth@dssat-4.8.6.0:nstress_replay",
            make=maker("growth", nstress=True),
            variants=(*_VARIANTS, GRAIN_NUMBER),
            n_days=N_DAYS,
            ports=PORTS_N,
            no_balance=_NO_BALANCE,
            grad=_GRAD,
            coefficient_sets=("coefficients",),
        )
    )
    for name in ("phenology", "growth"):
        out.append(
            ConformanceCase(
                key=f"crop/ceres_maize.{name}@dssat-4.8.6.0:alt_smoothed",
                make=maker(name, smoothed=True),
                variants=_VARIANTS,
                n_days=N_DAYS,
                ports=PORTS[name],
                no_balance=_NO_BALANCE,
                grad=_SMOOTHED_GRAD,
                coefficient_sets=("coefficients", "smoothing"),
                transforms_exact=False,
            )
        )
    out.append(
        ConformanceCase(
            key="crop/ceres_maize.season_init@dssat-4.8.6.0:faithful",
            make=season_maker("sow"),
            variants=SEASON_VARIANTS,
            n_days=N_DAYS,
            no_balance="a season initialisation selects the SEASINIT state: no stock with daily flows",
            grad=_SEASON_GRAD,
            forcing_fields=("sow",),
            coefficient_sets=(),  # the SEASINIT state uses none of the crop's hard-coded coefficients
        )
    )
    out.append(
        ConformanceCase(
            key="water_supply/forcing_replay@none:replay",
            make=maker("phenology"),
            variants=("vegetative",),
            n_days=N_DAYS,
            own="crops.{slot}",
            ports={"water_in": "iface.crop_water.{slot}"},
            no_balance="a replay of the reference run's crop water drivers: no conserved quantity",
            grad=None,
            no_grad="a replay copies the forcing: it has no parameter to differentiate",
            forcing_fields=("sw", "eop", "trwup"),
            coefficient_sets=(),  # the replay uses none of the crop's coefficients
        )
    )
    return out
