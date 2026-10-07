"""Conformance cases of the soil-water processes: the Richards redistribution
(``soil_water/richards@rzwqm2-4.6:faithful``).

Synthetic inputs on a synthetic node grid (30 nodes of 5 cm, 150 cm, five horizons of synthetic
modified Brooks-Corey parameters), with the horizon parameters scaled by a few per cent per sample
and a random initial water profile, three days of NumPy-generated weather:

* ``nominal``: rain on day 0 (and a small shower on day 2), daytime evaporation demand and root
  uptake in the top 14 nodes;
* ``intense``: 4-6 cm of rain in one hour on a wet profile (the surface ponds and runs off);
* ``dry`` (gradients only): a profile near the residual water content without rain (evaporation
  limited by the soil).

The Richards step runs with converged numerics (``n_iter = 10`` Newton iterations, 48 sub-steps),
so the water balance is checked to the kit's float64 tolerance; the default 24 x 3 numerics report
their unconverged residual in ``flux.balance_error`` instead. The process reads its sinks and demand
from the forcing and writes ``RichardsState`` instead of the sink-input port and the
``soil_water.theta`` port of the contract: the case binds no port. The balance adds every sink
channel of the ledger (:data:`~agrijax.processes.soil_water.sinks.SINK_LEDGER_OUTFLOWS`).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from agrijax.core.state import get_path
from agrijax.processes.soil_water.hydraulics import H_CLAMP_RZWQM, SoilHydraulicParams
from agrijax.processes.soil_water.richards import (
    FixedStepping,
    RichardsForcing,
    RichardsGrid,
    RichardsParams,
    RichardsState,
    SoilWater,
)
from agrijax.processes.soil_water.sinks import SINK_LEDGER_OUTFLOWS

from ..case import Balance, ConformanceCase, GradSpec, Tolerance

N_DAYS, N_BP, N_ROOT = 3, 2, 14
HOURS = 24
#: synthetic node grid: 30 cells of 5 cm (layer bottoms TLT) and five horizons
N_NODE, DZ = 30, 5.0
TLT = DZ * np.arange(1, N_NODE + 1, dtype=float)
HORIZON_BOTTOM = np.array([15.0, 30.0, 70.0, 90.0, 150.0])
#: synthetic rec1 rows (hb, lambda, eps, ksat, theta_r, theta_s) and rec2 rows
#: (fc13, fc110, wp, hb_k, c2, n1, a1); not the parameters of any site
REC1 = np.array(
    [
        [15.0, 0.25, 3.0, 5.0, 0.05, 0.45],
        [15.0, 0.30, 3.0, 3.0, 0.03, 0.45],
        [15.0, 0.35, 3.0, 3.5, 0.04, 0.45],
        [15.0, 0.20, 3.0, 3.0, 0.05, 0.45],
        [15.0, 0.30, 3.0, 2.5, 0.04, 0.45],
    ]
)
REC2 = np.tile([0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0], (5, 1))
#: converged numerics (the ledger closes to 1e-10 cm)
RICHARDS = FixedStepping(n_sub=48, n_iter=10)
#: the Brooks-Corey parameters differentiated (the soil has no Coefficients set: they are data)
SOIL_WRT = tuple(f"soil.{k}" for k in ("ksat", "lambda_", "hb", "theta_r", "theta_s"))
#: ledger outflows of the Richards day: ``{name: state path of its daily total}``
OUTFLOWS: dict[str, str] = {
    "evaporation": "soil_water.flux.evaporation",
    "drainage": "soil_water.flux.drainage",
    "runoff": "soil_water.flux.runoff",
    **SINK_LEDGER_OUTFLOWS,
}


def _cast(tree: Any, dtype: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype) if jnp.issubdtype(jnp.result_type(x), jnp.floating) else x, tree
    )


def _soil(rng: np.random.Generator) -> SoilHydraulicParams:
    """The synthetic horizons with ksat, hb and lambda scaled by U(0.95, 1.05) per horizon."""
    rec1 = REC1.copy()
    for col in (0, 1, 3):
        rec1[:, col] *= rng.uniform(0.95, 1.05, len(rec1))
    node_horizon = np.searchsorted(HORIZON_BOTTOM, TLT, side="left")
    return SoilHydraulicParams.from_rzwqm_records(rec1, REC2, node_horizon=node_horizon)


def _grid() -> RichardsGrid:
    return RichardsGrid.uniform(N_NODE, DZ)


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


def _rain(rng: np.random.Generator, variant: str) -> np.ndarray:
    """The hourly surface supply [cm h-1]: rain on day 0 (and a small day-2 shower); ``intense``:
    4-6 cm in one hour; ``dry``: none."""
    supply = np.zeros((N_DAYS, HOURS))
    if variant == "intense":
        supply[0, int(rng.uniform(2.0, 10.0))] = float(rng.uniform(2.0, 3.0, N_BP).sum())
    elif variant == "nominal":
        supply[0, int(rng.uniform(2.0, 10.0))] = float(rng.uniform(0.5, 1.2, N_BP).sum())
        supply[2, int(rng.uniform(12.0, 18.0))] = float(rng.uniform(0.1, 0.3))
    return supply


def _richards_params(rng: np.random.Generator) -> RichardsParams:
    return RichardsParams(
        soil=_soil(rng),
        grid=_grid(),
        h_min=jnp.asarray(H_CLAMP_RZWQM),
        pond_max=jnp.asarray(0.0),
        stepping=RICHARDS,
    )


def make_richards(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """Richards parameters, state and forcing: the rain as the hourly supply."""
    rp = _richards_params(rng)
    state = RichardsState(soil_water=_water(rng, rp.soil, variant))
    _, evap, uptake = _weather(rng, variant)
    supply = _rain(rng, variant)
    forcing = RichardsForcing(
        supply=jnp.asarray(supply), evaporation=jnp.asarray(evap), uptake=jnp.asarray(uptake)
    )
    return _cast(state, dtype), _cast(rp, dtype), _cast(forcing, dtype)


# ------------------------------------------------------------------ balances
def _storage(state: Any, params: Any) -> Any:
    w = state.soil_water
    return jnp.sum(w.theta * params.grid.tl) + w.pond


def _inflow(before: Any, after: Any, params: Any, forcing_t: Any) -> Any:
    return jnp.sum(forcing_t.supply) + after.soil_water.flux.rain


def _outflow(before: Any, after: Any, params: Any, forcing_t: Any) -> Any:
    return sum(get_path(after, path) for path in OUTFLOWS.values())


WATER = Balance("soil water (profile and pond)", "cm", storage=_storage, inflow=_inflow, outflow=_outflow)


# ------------------------------------------------------------------ losses
def _loss(final: Any) -> Any:
    """Profile water, pond and the day's surface and bottom fluxes (smooth where the day is)."""
    w = final.soil_water
    f = w.flux
    return jnp.sum(w.theta) + w.pond + f.infiltration + f.evaporation + f.drainage + f.runoff


#: known issue (measured on the cluster): float32 inputs with x64 enabled do not trace
_UPCAST = (
    "known issue: hydraulics._prepare casts h to jnp.result_type(float), float64 under x64 "
    "(hydraulics.py _prepare), so float32 inputs with x64 enabled are "
    "upcast inside the kernels and the scan carries change dtype (Richards _iterate carry "
    "float32[n] -> float64[n]); the float32 "
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
