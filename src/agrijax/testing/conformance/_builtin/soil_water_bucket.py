"""Conformance cases of the DSSAT-CSM v4.8.6.0 tipping bucket (``processes/soil_water/bucket``):
``soil_water/tipping_bucket.rate`` and ``soil_water/tipping_bucket.integrate``.

Synthetic soil of five DSSAT layers (5, 10, 15, 20, 30 cm) plus one padded layer
(``dlayr = 0``, the way runs with fewer layers are batched), three days:

* RATE ``nominal``: a 20-40 mm rain with residue on day 0 (infiltration), a dry day 1 on a profile
  above the drained upper limit (saturated drainage), a light rain on day 2; ``wet``: a nearly
  saturated profile with the saturated conductivity capping the drainage and a 50-80 mm storm
  (excess pushed back up, runoff); ``dry`` (gradients only): a profile near the lower limit, no
  rain (upward flow only).
* INTEGR ``nominal``: a RATE flux record, root uptake in the top four layers and soil evaporation
  from layer 1 (Ritchie); ``salus``: the same with the SALUS per-layer evaporation.

The integration's balance closes with its own flows (the RATE changes, the uptake, the
evaporation, the mulch water and the truncation); the whole day with the ledger is checked in
``tests/unit/test_bucket.py``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from agrijax.iface.soil import SinkInputs
from agrijax.iface.surface import PETFluxes
from agrijax.processes.soil_water.bucket import (
    WATBAL_COEFFICIENTS,
    BucketFluxes,
    BucketForcing,
    BucketParams,
    BucketSoil,
    BucketState,
    MulchForcing,
)

from ..case import Balance, ConformanceCase, GradSpec, Tolerance

N_DAYS = 3
DLAYR = np.array([5.0, 10.0, 15.0, 20.0, 30.0, 0.0])
ACTIVE = DLAYR > 0.0
N = DLAYR.shape[0]


def _soil(rng: np.random.Generator, dtype: Any, capped: bool) -> BucketSoil:
    ll = np.where(ACTIVE, rng.uniform(0.06, 0.12, N), 0.0)
    dul = np.where(ACTIVE, ll + rng.uniform(0.12, 0.18, N), 0.0)
    sat = np.where(ACTIVE, dul + rng.uniform(0.08, 0.14, N), 0.0)
    swcn = np.where(ACTIVE, rng.uniform(0.2, 0.5, N) if capped else -99.0, 0.0)
    ds = np.cumsum(DLAYR)
    return BucketSoil(
        dlayr=jnp.asarray(DLAYR, dtype),
        ds=jnp.asarray(np.where(ACTIVE, ds, 0.0), dtype),
        ll=jnp.asarray(ll, dtype),
        dul=jnp.asarray(dul, dtype),
        sat=jnp.asarray(sat, dtype),
        swcn=jnp.asarray(swcn, dtype),
        cn=jnp.asarray(rng.uniform(70.0, 85.0), dtype),
        swcon=jnp.asarray(rng.uniform(0.3, 0.6), dtype),
    )


def _params(soil: BucketSoil, dtype: Any, salus: bool = False) -> BucketParams:
    return BucketParams(
        soil=soil,
        mulch_on=jnp.asarray(1.0, dtype),
        salus_es=jnp.asarray(1.0 if salus else 0.0, dtype),
        actwtd=jnp.asarray(1000.0, dtype),
        pm_fraction=jnp.asarray(0.0, dtype),
        coefficients=WATBAL_COEFFICIENTS.as_arrays(dtype),
    )


def _forcing(rng: np.random.Generator, dtype: Any, rain: np.ndarray) -> BucketForcing:
    t = lambda x: jnp.asarray(x, dtype)  # noqa: E731
    return BucketForcing(
        rain=t(rain),
        tmax=t(rng.uniform(18.0, 30.0, N_DAYS)),
        irrigation=t(np.zeros(N_DAYS)),
        mulch=MulchForcing(
            mass=t(rng.uniform(800.0, 2000.0, N_DAYS)),
            cover=t(rng.uniform(0.2, 0.5, N_DAYS)),
            new_mass=t(np.zeros(N_DAYS)),
            watfac=t(np.full(N_DAYS, 3.5)),
        ),
    )


def _state(sw: np.ndarray, dtype: Any, rng: np.random.Generator) -> BucketState:
    s = BucketState.initial(
        np.where(ACTIVE, sw, 0.0),
        mulch_wat=float(rng.uniform(0.1, 0.4)),
        mulch_evap_prev=float(rng.uniform(0.05, 0.2)),
        dtype=dtype,
    )
    return s


def make_rate(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """RATE inputs for ``variant`` (nominal, wet, dry)."""
    soil = _soil(rng, dtype, capped=variant == "wet")
    ll, dul, sat = (np.asarray(x, float) for x in (soil.ll, soil.dul, soil.sat))
    if variant == "wet":
        sw = sat - rng.uniform(0.005, 0.02, N)
        rain = np.array([rng.uniform(50.0, 80.0), 0.0, rng.uniform(5.0, 10.0)])
    elif variant == "dry":
        sw = ll + rng.uniform(0.01, 0.04, N)
        rain = np.zeros(N_DAYS)
    else:
        sw = dul + rng.uniform(0.01, 0.05, N)
        rain = np.array([rng.uniform(20.0, 40.0), 0.0, rng.uniform(2.0, 6.0)])
    return _state(sw, dtype, rng), _params(soil, dtype), _forcing(rng, dtype, rain)


def make_integrate(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """INTEGR inputs: a RATE flux record, uptake, soil and mulch evaporation (Ritchie or SALUS)."""
    soil = _soil(rng, dtype, capped=False)
    ll, dul = (np.asarray(x, float) for x in (soil.ll, soil.dul))
    sw = np.where(ACTIVE, ll + rng.uniform(0.3, 0.9, N) * (dul - ll), 0.0)
    salus = variant == "salus"
    s = _state(sw, dtype, rng)
    t = lambda x: jnp.asarray(np.where(ACTIVE, x, 0.0), dtype)  # noqa: E731
    z = jnp.zeros((), dtype)
    flux = BucketFluxes.zeros(N, dtype).replace(
        swdelts=t(rng.uniform(-0.005, 0.01, N)),
        swdeltu=t(np.zeros(N) if salus else rng.uniform(-0.002, 0.002, N)),
        mulch_intercept=jnp.asarray(rng.uniform(0.0, 0.3), dtype),
        residue_water=z,
        truncation=z,
    )
    uptake = np.where(np.arange(N) < 4, rng.uniform(0.0, 0.05, N), 0.0)
    evap = np.where(np.arange(N) < 3, rng.uniform(0.0, 0.03, N), 0.0) if salus else np.zeros(N)
    s = s.replace(
        flux=flux,
        sink_in=SinkInputs.zeros(N, dtype).replace(uptake=t(uptake)),
        evap_layers=t(evap),
        pet=PETFluxes.zeros(dtype).replace(
            soil_evaporation=jnp.asarray(rng.uniform(0.05, 0.2), dtype),
            residue_evaporation=jnp.asarray(rng.uniform(0.0, 0.03), dtype),
        ),
    )
    return s, _params(soil, dtype, salus=salus), _forcing(rng, dtype, np.zeros(N_DAYS))


# ------------------------------------------------------------------ the integration's balance
def _storage(s: BucketState, p: BucketParams) -> Any:
    return jnp.sum(s.sw * p.soil.dlayr, axis=-1) + (s.snow + s.mulch_wat) * 0.1


def _salus(p: BucketParams) -> Any:
    return jnp.asarray(p.salus_es, dtype=bool)


def _inflow(b: BucketState, a: BucketState, p: BucketParams, f: Any) -> Any:
    f0 = b.flux
    dy = p.soil.dlayr
    u = jnp.where(_salus(p), 0.0, jnp.sum(f0.swdeltu * dy, axis=-1))
    return jnp.sum(f0.swdelts * dy, axis=-1) + u + (f0.mulch_intercept + f0.residue_water) * 0.1


def _outflow(b: BucketState, a: BucketState, p: BucketParams, f: Any) -> Any:
    evap = jnp.where(_salus(p), jnp.sum(b.evap_layers, axis=-1), b.pet.soil_evaporation)
    return (
        jnp.sum(b.sink_in.uptake, axis=-1)
        + evap
        + b.pet.residue_evaporation
        + (a.flux.truncation - b.flux.truncation)
    )


def cases() -> list[ConformanceCase]:
    return [
        ConformanceCase(
            key="soil_water/tipping_bucket.rate@dssat-4.8.6.0:faithful",
            make=make_rate,
            variants=("nominal", "wet"),
            n_days=N_DAYS,
            own="soil_water",
            ports={},
            no_balance=(
                "WATBAL RATE computes the day's changes (SWDELTS, SWDELTU, runoff, drainage) without "
                "applying them; the stock changes at INTEGR, whose case closes the balance with these "
                "changes, and the whole day closes the water ledger in tests/unit/test_bucket.py"
            ),
            grad=GradSpec(edge_variants=("dry",)),
            forcing_fields=("rain", "tmax", "irrigation", "mulch", "soil"),
            coefficient_sets=("coefficients",),
        ),
        ConformanceCase(
            key="soil_water/tipping_bucket.integrate@dssat-4.8.6.0:faithful",
            make=make_integrate,
            variants=("nominal", "salus"),
            n_days=N_DAYS,
            own="soil_water",
            ports={"pet": "iface.pet"},
            balances=(Balance("water", "cm", _storage, _inflow, _outflow),),
            # flux.truncation is the rounded minus the unrounded water of each layer, a cancellation
            # of ~10 cm depths: float32 against float64 it differs by the float32 rounding of those
            # depths and by a 1e-6 quantum flip of a 30 cm layer (3e-5 cm; measured 4.5e-6 cm)
            f32=Tolerance(1e-4, 3e-5),
            transforms_tol=(Tolerance(1e-12, 1e-14), Tolerance(1e-5, 1e-5)),
            transforms_why=(
                "flux.truncation cancels the rounded against the unrounded water depth of each layer "
                "(~10 cm): eager and jit evaluate the unrounded depth a few ulp apart (2e-15 cm each in "
                "float64; measured 486814 ulp of the small truncation value, i.e. below 1e-14 cm), every "
                "other leaf agrees to rtol 1e-12"
            ),
            grad=None,
            no_grad=(
                "no calibratable coefficient: the integration's numbers are the 1e-6 rounding (static) "
                "and two cuts (not calibrated); its output is rounded to 1e-6 (straight-through "
                "derivative), so finite differences of the soil parameters are not a derivative check; "
                "gradients through the whole day are checked finite in tests/unit/test_bucket.py"
            ),
            forcing_fields=("soil", "dlayr_end"),
            coefficient_sets=("coefficients",),
        ),
    ]
