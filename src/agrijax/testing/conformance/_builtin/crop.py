"""Conformance cases of CERES-Maize (``crop/ceres_maize.*``) and its water replay.

The inputs of a crop process are a mid-season state, so ``make`` runs a synthetic season first:
NumPy weather around a warm-season mean (the recipe of ``tests/unit/test_ceres_phenology.py``),
ample soil water with a few dry spells, the MZCER048 cultivar IB0035 of the unit tests, and the
crop on its own (``ceres_maize_model``: the water replay writes the ``water_in`` port). From the
end of day ``d`` the day's order is replayed up to the process under test, which is then run for
three days. Variants pick ``d``: ``vegetative`` (leaf expansion) and ``grain_fill``. The season is
computed once per seed and dtype.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from agrijax.core.model import Model
from agrijax.core.runtime import run
from agrijax.iface.crop import CropNIn
from agrijax.processes.crop.ceres_maize import (
    CROP_PROCESSES,
    DSSAT_COEFFICIENTS,
    CeresCultivar,
    CeresForcing,
    CeresMaizeParams,
    CeresMaizeState,
    CeresSoil,
    CeresSpecies,
    ceres_growth,
    ceres_phenology,
    ceres_roots,
    ceres_stress,
    ceres_water_replay,
)
from agrijax.processes.crop.ceres_maize._util import daylength, twilight_daylength

from ..case import ConformanceCase, GradSpec

# ------------------------------------------------------------------ MZCER048 IB0035, as the unit tests
CUL = dict(
    p1=259.0, p2=1.193, p5=947.1, g2=924.3, g3=8.17, phint=43.0, tbase=8.0, topt=34.0, ropt=34.0,
    p2o=12.5, djti=4.0, gdde=6.0, dsgft=170.0, rue=4.2, tsen=6.0, cday=15.0,
)  # fmt: skip
SPE: dict[str, Any] = dict(
    prftc=(6.2, 16.5, 33.0, 44.0), rgfil=(5.5, 16.0, 27.0, 35.0), parsr=0.5,
    co2x=(0, 220, 280, 330, 400, 490, 570, 750, 990, 9999),
    co2y=(0.0, 0.85, 0.95, 1.0, 1.02, 1.04, 1.05, 1.06, 1.07, 1.08),
    fslfw=0.05, rsgr=0.1, rsgrt=5.0, carbot=7.0, dsgt=21.0, dget=150.0, swcg=0.02, stmwte=0.2,
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
#: the day's order after the water replay (``MZ_CERES``); the process under test runs from the
#: state just before it
ORDER = ("phenology", "stress", "growth", "roots", "publish")
_WATER_IN = {"water_in": "iface.crop_water.{slot}"}
#: the ports each process uses (the slot-contract check requires a case to bind exactly those)
PORTS: dict[str, dict[str, str]] = {
    "phenology": _WATER_IN,
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


def weather(seed: int, dtype: Any, n: int = N_SEASON + N_DAYS) -> CeresForcing:
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
    return CeresForcing(
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


_SEASON = Model(CeresMaizeState, [ceres_water_replay, *CROP_PROCESSES], outputs=lambda s, p, f: s)
_RUN = jax.jit(lambda p, f, s: run(_SEASON, p, f, s))
_STEPS = {"phenology": ceres_phenology, "stress": ceres_stress, "growth": ceres_growth, "roots": ceres_roots}


@functools.lru_cache(maxsize=32)
def _season(seed: int, dtype_name: str) -> tuple[Any, Any, Any]:
    dtype = jnp.dtype(dtype_name)
    f = weather(seed, dtype)
    p = params(dtype, int(np.asarray(f.yrdoy)[2]))
    traj = _RUN(p, f, CeresMaizeState.initial(p, 1, dtype=dtype))
    return traj, f, p


@functools.lru_cache(maxsize=64)
def _pre(seed: int, dtype_name: str, variant: str, before: str) -> tuple[Any, Any, Any]:
    traj, f, p = _season(seed, dtype_name)
    d = DAY[variant]
    s = jax.tree_util.tree_map(lambda x: x[d], traj)
    ft = jax.tree_util.tree_map(lambda x: x[d + 1], f)
    s = ceres_water_replay(s, p, ft)
    for name in ORDER:
        if name == before:
            break
        s = _STEPS[name](s, p, ft)
    days = jax.tree_util.tree_map(lambda x: x[d + 1 : d + 1 + N_DAYS], f)
    return s, p, days


def maker(before: str, nstress: bool = False) -> Any:
    """``make`` of the case of the process ``before`` (the state just before it runs)."""

    def make(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
        seed = int(rng.integers(0, 2**31 - 1))
        s, p, f = _pre(seed, jnp.dtype(dtype).name, variant, before)
        if nstress:
            ns = rng.uniform(0.4, 0.9, np.shape(s.growth.lai))
            s = s.replace(n_in=CropNIn(nstres=jnp.asarray(ns, s.growth.lai.dtype)))
        return s, p, f

    return make


_NO_BALANCE = (
    "CERES-Maize conserves assimilate between organs, tested day by day against the model's own "
    "partitioning in tests/unit/test_ceres_growth.py; there is no stock with daily in- and outflows "
    "at the process boundary"
)
_VARIANTS = tuple(DAY)
_GRAD = GradSpec()


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
            # batched, phen.xstage differs from the per-sample run by 1 ulp (measured on rorqual)
            transforms_exact=name != "phenology",
        )
        for name in ORDER
    ]
    out.append(
        ConformanceCase(
            key="crop/ceres_maize.growth@dssat-4.8.6.0:nstress_replay",
            make=maker("growth", nstress=True),
            variants=_VARIANTS,
            n_days=N_DAYS,
            ports=PORTS_N,
            no_balance=_NO_BALANCE,
            grad=_GRAD,
            coefficient_sets=("coefficients",),
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
