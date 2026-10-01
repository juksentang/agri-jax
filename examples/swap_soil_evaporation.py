"""Swap one process of the DSSAT-CSM v4.8.6.0 day: the soil evaporation, Ritchie two-stage (SOILEV)
against SALUS (ESR_SoilEvap), both chosen by registry key.

A synthetic maize season on the nine-layer sandy soil of the crop conformance case (no data needed):
the same weather, soil and cultivar run once with each implementation; the script prints what the
swap changes (yield, season soil evaporation, transpiration, profile water) and that the water
ledger closes in both. Then it plugs in a deliberately wrong stand-in (soil evaporation in cm d-1)
and prints why the day rejects it. See docs/swapping_a_process.md.

    uv run python examples/swap_soil_evaporation.py
"""

from __future__ import annotations

from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jaxtyping import Array  # noqa: E402

from agrijax.core import run  # noqa: E402
from agrijax.core.process import process  # noqa: E402
from agrijax.core.state import State, field  # noqa: E402
from agrijax.models.day_dssat486 import (  # noqa: E402
    SLOT,
    SOIL_EVAPORATION_KEYS,
    DssatSite,
    SoilEvaporationError,
    day_dssat486,
    day_outputs,
    day_params,
    day_processes,
    initial_state,
    soil_evaporation_params_problems,
)
from agrijax.processes.crop.ceres_maize import CeresMaizeState  # noqa: E402
from agrijax.processes.pet.spam_dssat import SpamWeather  # noqa: E402
from agrijax.processes.soil_water.bucket import BucketForcing, BucketState, MulchForcing  # noqa: E402
from agrijax.processes.soil_water.bucket_evap import SoilEvapState  # noqa: E402
from agrijax.processes.water_supply import RootwuState  # noqa: E402
from agrijax.testing.conformance import synthetic as case  # noqa: E402

N_DAYS = case.N_SEASON + 10
KEP = case.KEP  # CERES-Maize KEP of the DSSAT maize examples (KSEVAP = KTRANS), labelled there
#: a synthetic site (soil of the conformance kit; runoff curve number CN, drainage SWCON [d-1],
#: bare-soil albedo SALB and stage-1 evaporation limit U [mm] as in a DSSAT soil profile): test inputs
SITE = DssatSite(
    dlayr=np.asarray(case.DLAYR),
    ds=np.cumsum(case.DLAYR),
    ll=np.asarray(case.LL),
    dul=np.asarray(case.DUL),
    sat=np.asarray(case.SAT),
    swcn=np.zeros(len(case.DLAYR)),
    cn=72.0,
    swcon=0.5,
    salb=0.13,
    u=6.0,
)


def inputs(soil_evaporation: str) -> tuple[Any, Any, Any]:
    """Params (built for ``soil_evaporation``), forcing and initial state of the synthetic season."""
    f = case.ceres_weather(7, jnp.float64, N_DAYS)
    params = day_params(
        SITE,
        case.ceres_params(jnp.float64, case.START + 5),
        ksevap=KEP,
        ktrans=KEP,
        soil_evaporation=soil_evaporation,
    )
    rng = np.random.default_rng(3)
    rain = jnp.asarray(np.where(rng.uniform(size=N_DAYS) < 0.25, rng.uniform(2.0, 30.0, N_DAYS), 0.0))
    const = lambda v: jnp.full(N_DAYS, v)  # noqa: E731
    mulch = MulchForcing(mass=const(0.0), cover=const(0.0), new_mass=const(0.0), watfac=const(0.0))
    forcing = {
        "crop": f.replace(
            sw=jnp.full_like(f.sw, jnp.nan),
            eop=jnp.full_like(f.eop, jnp.nan),
            trwup=jnp.full_like(f.trwup, jnp.nan),
        ),
        "soil": BucketForcing(rain=rain, tmax=f.tmax, irrigation=const(0.0), mulch=mulch),
        "spam": {
            "weather": SpamWeather(
                tavg=(f.tmax + f.tmin) / 2.0,
                wind_run=const(150.0),
                co2=f.co2,
                srad=f.srad,
                tmax=f.tmax,
                tmin=f.tmin,
            ),
            "mulch_am": const(0.0),
            "mulch_extfac": const(0.0),
        },
    }
    soil = params["soil"].soil
    sw0 = soil.ll + 0.7 * (soil.dul - soil.ll)
    state = initial_state(
        bucket=BucketState.initial(sw0, dtype=jnp.float64),
        soil_evap=SoilEvapState.initial(sw0, soil.dlayr, soil.ds, soil.dul, soil.ll, params["evap"].u),
        crop=CeresMaizeState.initial(params["crop"], 1),
        rootwu=RootwuState.initial(1, int(sw0.shape[-1]), jnp.float64),
        salb=params["albedo"].salb,
        storage0=jnp.sum(sw0 * soil.dlayr),
    )
    return params, forcing, state


def season(soil_evaporation: str) -> dict[str, float]:
    """One season with the soil evaporation ``soil_evaporation`` (a registry key)."""
    model = day_dssat486(SLOT).compile(
        day_processes(SLOT, soil_evaporation=soil_evaporation),
        outputs=day_outputs(SLOT),
        check=True,
        exact_lags=True,
    )
    params, forcing, state = inputs(soil_evaporation)
    assert not soil_evaporation_params_problems(params, soil_evaporation)  # the coupling flags, on the host
    final, out = jax.jit(lambda p, f, s: run(model, p, f, s, return_final=True))(params, forcing, state)
    dlayr = np.asarray(params["soil"].soil.dlayr)
    return {
        "yield kg/ha": float(out["gwad"][-1, 0]),
        "ES mm": float(np.sum(out["es"])),
        "EP mm": float(np.sum(out["ep"])),
        "profile water mm (end)": float(np.sum(np.asarray(out["soil_sw"][-1]) * dlayr) * 10.0),
        "ledger max residual mm": float(final["ledger"]["water"].max_abs_residual) * 10.0,
    }


class CmEvapState(State):
    """A wrong stand-in's store: the soil evaporation in cm d-1 (the day's SPAM store holds mm d-1)."""

    eos_soil: Array = field(unit="mm d-1", dims=())
    em: Array = field(unit="mm d-1", dims=())
    es: Array = field(unit="cm d-1", dims=())
    evap: Array = field(unit="mm d-1", dims=())


@process(
    reads=("eos_soil", "em"),
    writes=("es", "evap"),
    key="soil_water/potential_es_cm@none:demo",
    provenance="equations_only",
    sources=(("ES = EOS_SOIL, in cm d-1 (deliberately wrong unit)", "examples/swap_soil_evaporation.py"),),
    grid="point",
    deviates=(),
    register=False,
)
def potential_es_cm(state: CmEvapState, params: Any, forcing_t: Any) -> CmEvapState:
    """ES at the potential rate, in cm d-1. Source: example stand-in."""
    return state.replace(es=state.eos_soil / 10.0, evap=state.em + state.eos_soil)


if __name__ == "__main__":
    ritchie, salus = SOIL_EVAPORATION_KEYS["R"], SOIL_EVAPORATION_KEYS["S"]
    a, b = season(ritchie), season(salus)
    print(f"{'':26s}{'Ritchie (SOILEV)':>18s}{'SALUS (ESR)':>14s}{'change':>10s}")
    for k in a:
        print(f"{k:26s}{a[k]:18.4g}{b[k]:14.4g}{b[k] - a[k]:10.3g}")
    try:
        day_processes(SLOT, soil_evaporation=potential_es_cm)
    except SoilEvaporationError as e:
        print(f"\nrejected before any trace:\n{e}")
