"""Conformance cases of the soil-water processes: the RZWQM2 day (``faithful``, ``replay_flux``),
the Green-Ampt event alone and the Richards redistribution alone.

Synthetic inputs on the CA-TPA node grid (37 nodes, 150 cm, five horizons; the rzwqm.dat records
also used by ``tests/unit/test_richards.py``), with the horizon parameters scaled by a few per cent
per sample and a random initial water profile, three days of NumPy-generated weather:

* ``nominal``: a storm of two breakpoint intervals on day 0 (and a small one on day 2), daytime
  evaporation demand and root uptake in the top 14 nodes;
* ``intense``: a 4-6 cm storm in half an hour on a wet profile (the event produces runoff);
* ``dry`` (gradients only): a profile near the residual water content without rain (evaporation
  limited by the soil).

The day and the Richards step run with converged numerics (``n_iter = 10`` Newton iterations,
48 sub-steps: the configuration of ``tests/unit/test_soil_water_ledger.py``, which closes the
ledger to 1e-10 cm), so the water balance is checked to the kit's float64 tolerance; the default
24 x 3 numerics report their unconverged residual in ``flux.balance_error`` instead. The processes
still read their sinks and demand from the forcing and write ``RichardsState`` (gaps G1 and the P7
path, W1): the cases bind no port. The balance adds every sink channel of the ledger
(:data:`~agrijax.processes.soil_water.day.SOIL_WATER_LEDGER_OUTFLOWS`).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from agrijax.core.state import get_path
from agrijax.processes.soil_water.coefficients import RZWQM2_GREEN_AMPT
from agrijax.processes.soil_water.day import (
    SOIL_WATER_LEDGER_OUTFLOWS,
    DayConfig,
    SoilWaterDayForcing,
    SoilWaterDayParams,
)
from agrijax.processes.soil_water.hydraulics import H_CLAMP_RZWQM, SoilHydraulicParams
from agrijax.processes.soil_water.infiltration import GreenAmptConfig, GreenAmptParams, StormForcing
from agrijax.processes.soil_water.richards import (
    RichardsConfig,
    RichardsForcing,
    RichardsGrid,
    RichardsParams,
    RichardsState,
    SoilWater,
)

from ..case import CALIBRATABLE, Balance, ConformanceCase, GradSpec, Tolerance

N_DAYS, N_BP, N_ROOT = 3, 2, 14
HOURS = 24
#: CA-TPA rzwqm.dat node records (layer bottom TLT, distance to the next node DELZ) and horizons
TLT = np.array(
    [1, 2, 4, 7, 11, 15, 19, 23, 26, 30, 34, 38, 43, 48, 53, 58, 63, 67, 70, 73, 77, 82, 86, 90, 94, 98,
     103, 108, 113, 118, 123, 128, 133, 138, 143, 147, 150], dtype=float
)  # fmt: skip
DELZ = np.array(
    [1, 1, 3, 3, 5, 3, 5, 3, 3, 5, 3, 5, 5, 5, 5, 5, 5, 3, 3, 3, 5, 5, 3, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5,
     5, 3, 0], dtype=float
)  # fmt: skip
HORIZON_BOTTOM = np.array([15.0, 30.0, 70.0, 90.0, 150.0])
#: rec1 rows (hb, lambda, eps, ksat, theta_r, theta_s) and rec2 rows (fc13, fc110, wp, hb_k, c2, n1, a1)
REC1 = np.array(
    [
        [14.6545, 0.22, 2.966, 5.41, 0.055, 0.453],
        [14.6545, 0.26, 2.966, 3.16, 0.032, 0.453],
        [14.6545, 0.36, 2.966, 3.31, 0.043, 0.453],
        [14.6545, 0.17, 2.966, 3.32, 0.048, 0.453],
        [14.6545, 0.322, 2.966, 2.59, 0.041, 0.453],
    ]
)
REC2 = np.tile([0.0, 0.0, 0.0, 14.6545, 7440.01, 0.0, 0.0], (5, 1))
AEF = 0.9
#: converged numerics (tests/unit/test_soil_water_ledger.py: the ledger closes to 1e-10 cm)
RICHARDS = RichardsConfig(n_sub=48, n_iter=10)
DAY = DayConfig(n_pre=12, n_post=36)
#: the Brooks-Corey parameters differentiated (the soil has no Coefficients set: they are data)
SOIL_WRT = tuple(f"soil.{k}" for k in ("ksat", "lambda_", "hb", "theta_r", "theta_s"))


def _cast(tree: Any, dtype: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype) if jnp.issubdtype(jnp.result_type(x), jnp.floating) else x, tree
    )


def _soil(rng: np.random.Generator) -> SoilHydraulicParams:
    """CA-TPA horizons with ksat, hb and lambda scaled by U(0.95, 1.05) per horizon."""
    rec1 = REC1.copy()
    for col in (0, 1, 3):
        rec1[:, col] *= rng.uniform(0.95, 1.05, len(rec1))
    node_horizon = np.searchsorted(HORIZON_BOTTOM, TLT, side="left")
    return SoilHydraulicParams.from_rzwqm_records(rec1, REC2, node_horizon=node_horizon)


def _grid() -> RichardsGrid:
    return RichardsGrid.from_rzwqm(TLT, DELZ)


def _water(rng: np.random.Generator, soil: Any, variant: str) -> SoilWater:
    nodes = soil.at_nodes()
    tr, ts = np.asarray(nodes.theta_r), np.asarray(nodes.theta_s)
    n = len(TLT)
    if variant == "dry":
        frac = rng.uniform(0.1, 0.2, n)
    elif variant == "intense":
        frac = rng.uniform(0.55, 0.7, n)
    else:
        frac = np.linspace(0.35, 0.6, n) + rng.uniform(-0.05, 0.05, n)
    return SoilWater.from_theta(jnp.asarray(tr + frac * (ts - tr)), soil)


def _weather(rng: np.random.Generator, variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hourly evaporation demand [cm h-1] and daily uptake per node [cm d-1]; the hourly supply is zero."""
    hours = np.arange(HOURS)
    shape = np.where((hours >= 6) & (hours < 18), np.sin(np.pi * (hours - 6 + 0.5) / 12), 0.0)
    daily_e = rng.uniform(0.3, 0.5, N_DAYS) if variant == "dry" else rng.uniform(0.05, 0.3, N_DAYS)
    evap = np.outer(daily_e, shape / shape.sum())
    uptake = np.zeros((N_DAYS, len(TLT)))
    weights = np.linspace(2.0, 0.2, N_ROOT) / np.linspace(2.0, 0.2, N_ROOT).sum()
    uptake[:, :N_ROOT] = rng.uniform(0.05, 0.25, (N_DAYS, 1)) * weights
    return np.zeros((N_DAYS, HOURS)), evap, uptake


def _storms(rng: np.random.Generator, variant: str) -> StormForcing:
    """Day 0 storm (and a small day-2 one); ``intense``: 4-6 cm in half an hour; ``dry``: none."""
    ts0 = np.full(N_DAYS, float(HOURS))
    dur = np.zeros((N_DAYS, N_BP))
    dep = np.zeros((N_DAYS, N_BP))
    if variant == "intense":
        ts0[0] = rng.uniform(2.0, 10.0)
        dur[0] = (0.25, 0.25)
        dep[0] = rng.uniform(2.0, 3.0, N_BP)
    elif variant == "nominal":
        ts0[0] = rng.uniform(2.0, 10.0)
        dur[0] = rng.uniform(0.5, 1.0, N_BP)
        dep[0] = rng.uniform(0.5, 1.2, N_BP)
        ts0[2] = rng.uniform(12.0, 18.0)
        dur[2, 0] = rng.uniform(0.3, 0.6)
        dep[2, 0] = rng.uniform(0.1, 0.3)
    return StormForcing(ts0=jnp.asarray(ts0), duration=jnp.asarray(dur), depth=jnp.asarray(dep))


def _richards_params(rng: np.random.Generator) -> RichardsParams:
    return RichardsParams(
        soil=_soil(rng),
        grid=_grid(),
        h_min=jnp.asarray(H_CLAMP_RZWQM),
        pond_max=jnp.asarray(0.0),
        config=RICHARDS,
    )


def make_day(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """Day parameters, state and three days of forcing (supply zero: the storm brings the rain)."""
    rp = _richards_params(rng)
    ga = GreenAmptParams(
        config=GreenAmptConfig.for_grid(np.diff(np.concatenate([[0.0], TLT]))),
        aef=jnp.asarray(AEF),
        coefficients=RZWQM2_GREEN_AMPT.as_arrays(dtype),
    )
    params = SoilWaterDayParams(richards=rp, infiltration=ga, config=DAY)
    state = RichardsState(soil_water=_water(rng, rp.soil, variant))
    supply, evap, uptake = _weather(rng, variant)
    forcing = SoilWaterDayForcing(
        storm=_storms(rng, variant),
        supply=jnp.asarray(supply),
        evaporation=jnp.asarray(evap),
        uptake=jnp.asarray(uptake),
    )
    return _cast(state, dtype), _cast(params, dtype), _cast(forcing, dtype)


def _storm_as_supply(storm: StormForcing) -> np.ndarray:
    """The storm's rain as an hourly surface supply [cm h-1] (the ``replay_flux`` / Richards input)."""
    ts0, dep = (np.asarray(x, np.float64) for x in (storm.ts0, storm.depth))
    supply = np.zeros((N_DAYS, HOURS))
    for d in range(N_DAYS):
        total = float(dep[d].sum())
        if total > 0.0:
            start = min(int(ts0[d]), HOURS - 1)
            supply[d, start] = total  # all of it in the storm's first hour
    return supply


def make_day_replay(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """As :func:`make_day`, with the storm's rain also given as the hourly supply."""
    s, p, f = make_day(rng, dtype, variant)
    supply = jnp.asarray(_storm_as_supply(f.storm), dtype)
    return s, p, f.replace(supply=supply)


def make_richards(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """Richards parameters, state and forcing: the storm's rain as the hourly supply."""
    rp = _richards_params(rng)
    state = RichardsState(soil_water=_water(rng, rp.soil, variant))
    _, evap, uptake = _weather(rng, variant)
    supply = _storm_as_supply(_storms(rng, variant))
    forcing = RichardsForcing(
        supply=jnp.asarray(supply), evaporation=jnp.asarray(evap), uptake=jnp.asarray(uptake)
    )
    return _cast(state, dtype), _cast(rp, dtype), _cast(forcing, dtype)


# ------------------------------------------------------------------ balances
def _tl(params: Any) -> Any:
    return params.richards.grid.tl if isinstance(params, SoilWaterDayParams) else params.grid.tl


def _storage(state: Any, params: Any) -> Any:
    w = state.soil_water
    return jnp.sum(w.theta * _tl(params)) + w.pond


def _inflow(before: Any, after: Any, params: Any, forcing_t: Any) -> Any:
    return jnp.sum(forcing_t.supply) + after.soil_water.flux.rain


def _outflow(before: Any, after: Any, params: Any, forcing_t: Any) -> Any:
    return sum(get_path(after, path) for path in SOIL_WATER_LEDGER_OUTFLOWS.values())


WATER = Balance("soil water (profile and pond)", "cm", storage=_storage, inflow=_inflow, outflow=_outflow)


def _event_inflow(before: Any, after: Any, params: Any, forcing_t: Any) -> Any:
    return after.soil_water.flux.rain


def _event_outflow(before: Any, after: Any, params: Any, forcing_t: Any) -> Any:
    f = after.soil_water.flux
    return f.event_runoff + f.seepage


EVENT_WATER = Balance(
    "soil water of the event (rain in; event runoff and seepage out)",
    "cm",
    storage=_storage,
    inflow=_event_inflow,
    outflow=_event_outflow,
)


# ------------------------------------------------------------------ losses
def _loss(final: Any) -> Any:
    """Profile water, pond and the day's surface and bottom fluxes (smooth where the day is)."""
    w = final.soil_water
    f = w.flux
    return jnp.sum(w.theta) + w.pond + f.infiltration + f.evaporation + f.drainage + f.runoff


def _event_loss(final: Any) -> Any:
    w = final.soil_water
    f = w.flux
    return jnp.sum(w.theta) + f.event_infiltration + f.event_runoff + f.seepage


def _day_wrt() -> tuple[str, ...]:
    return (CALIBRATABLE, *(f"richards.{p}" for p in SOIL_WRT))


#: open issue for W1 (measured on rorqual, W0R-C): float32 inputs with x64 enabled do not trace
_UPCAST = (
    "open issue W1/SW-1: hydraulics._prepare casts h to jnp.result_type(float), float64 under x64 "
    "(hydraulics.py _prepare; _prepare_tilled likewise), so float32 inputs with x64 enabled are "
    "upcast inside the kernels and the scan carries change dtype (Richards _iterate carry "
    "float32[37] -> float64[37]; Green-Ampt depth_scan carry float32[] -> float64[]); the float32 "
    "pass with AGRI_JAX_X64=0 runs and passes"
)
#: vmap(jit) against jit and eager against jit, float64 then float32
_TRANSFORMS_TOL = (Tolerance(1e-12, 1e-11), Tolerance(1e-4, 1e-5))
_TRANSFORMS_WHY = (
    "48 sub-steps x 10 Newton iterations: the batched and the eager programs round differently "
    "(measured float64, vmap against jit: h 34 ulp = 9.7e-13 cm, theta 5 ulp, drainage 38 ulp = "
    "3.6e-15 cm, balance_error 1.4e-14 cm; eager against jit: h 1.6e-12 cm, runoff -2.8e-17 against "
    "0 cm; float32 with x64 off, largest over both: balance_error 7.3e-6 cm, drainage 3.0e-6 cm, "
    "h 4.7e-4 cm, theta 1.2e-7); batch independence stays bit for bit"
)


#: the float32 parts under x64 only: the float64 parts of these checks run unexempted and must
#: pass, and with x64 disabled the checks run unexempted (ConformanceCase.exempt_float32_x64)
_F32_X64 = {"balance": _UPCAST, "precision": _UPCAST, "grad_finite": _UPCAST}


def cases() -> list[ConformanceCase]:
    return [
        ConformanceCase(
            key="soil_water/day@rzwqm2-4.6:faithful",
            make=make_day,
            variants=("nominal", "intense"),
            n_days=N_DAYS,
            balances=(WATER,),
            grad=GradSpec(wrt=_day_wrt(), loss=_loss, edge_variants=("dry",)),
            coefficient_sets=("infiltration.coefficients",),
            forcing_fields=("storm", "supply", "evaporation", "uptake"),
            transforms_tol=_TRANSFORMS_TOL,
            transforms_why=_TRANSFORMS_WHY,
            exempt_float32_x64=_F32_X64,
        ),
        ConformanceCase(
            key="soil_water/day@rzwqm2-4.6:replay_flux",
            make=make_day_replay,
            variants=("nominal", "intense"),
            n_days=N_DAYS,
            balances=(WATER,),
            grad=GradSpec(wrt=tuple(f"richards.{p}" for p in SOIL_WRT), loss=_loss, edge_variants=("dry",)),
            coefficient_sets=(),  # the replay does not run the event: no Green-Ampt coefficient is used
            forcing_fields=("supply", "evaporation", "uptake"),
            transforms_tol=_TRANSFORMS_TOL,
            transforms_why=_TRANSFORMS_WHY,
            exempt_float32_x64=_F32_X64,
        ),
        ConformanceCase(
            key="soil_water/infiltration_ga@rzwqm2-4.6:faithful",
            make=make_day,
            variants=("nominal", "intense"),
            n_days=N_DAYS,
            balances=(EVENT_WATER,),
            grad=GradSpec(wrt=_day_wrt(), loss=_event_loss),
            coefficient_sets=("infiltration.coefficients",),
            forcing_fields=("storm",),
            exempt_float32_x64=_F32_X64,
        ),
        ConformanceCase(
            key="soil_water/richards@rzwqm2-4.6:faithful",
            make=make_richards,
            variants=("nominal", "intense"),
            n_days=N_DAYS,
            balances=(WATER,),
            grad=GradSpec(wrt=SOIL_WRT, loss=_loss, edge_variants=("dry",)),
            coefficient_sets=(),  # the Richards step reads no Coefficients set (the soil is data)
            forcing_fields=("supply", "evaporation", "uptake"),
            transforms_tol=_TRANSFORMS_TOL,
            transforms_why=_TRANSFORMS_WHY,
            exempt_float32_x64=_F32_X64,
        ),
    ]
