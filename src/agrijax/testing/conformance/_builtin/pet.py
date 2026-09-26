"""Conformance cases of the three PET processes (Shuttleworth-Wallace, ASCE reference, Priestley-Taylor).

Three days of synthetic summer weather on one site: a cropped surface with flat residue (nominal),
and, for gradients only, bare soil without residue (``LAI = 0``) and a cold, humid day. The
processes still read their own ``surface`` subtree, not the canopy and soil-water ports (gap G5),
so the cases bind no port. The case data live here and not in ``processes/pet`` (frozen while the
H1 hardening changes it).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agrijax.processes.pet import PET_COEFFICIENTS, PETParams
from agrijax.processes.pet.daily import DailyWeather, PETFluxes, PETSiteParams, PETState, SurfaceState

from ..case import ConformanceCase, GradSpec

N_DAYS = 3
#: CA-TPA surface layer and site (as tests/unit/test_pet_grad_finite.py)
WC13, WC15, ELEVATION, LATITUDE = 0.255198, 0.141628, 200.0, 0.745163


def make(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """Site parameters, surface state and three days of weather for ``variant``."""

    def a(x: Any) -> Any:
        return np.asarray(x, dtype)

    cold = variant == "cold"
    bare = variant == "bare"
    tmin = rng.uniform(-4.0, 0.0, N_DAYS) if cold else rng.uniform(10.0, 16.0, N_DAYS)
    tmax = tmin + rng.uniform(6.0, 12.0, N_DAYS)
    weather = DailyWeather(
        tmin=a(tmin),
        tmax=a(tmax),
        srad=a(rng.uniform(4.0, 8.0, N_DAYS) if cold else rng.uniform(15.0, 26.0, N_DAYS)),
        rh=a(rng.uniform(85.0, 95.0, N_DAYS) if cold else rng.uniform(50.0, 75.0, N_DAYS)),
        wind_run=a(rng.uniform(80.0, 250.0, N_DAYS)),
        doy=a(rng.integers(20, 40, N_DAYS) if cold else rng.integers(170, 200, N_DAYS)),
    )
    lai = 0.0 if bare else float(rng.uniform(1.5, 4.0))
    surface = SurfaceState(
        lai=a(lai),
        tlai=a(lai + (0.0 if bare else float(rng.uniform(0.1, 0.5)))),
        height_cm=a(0.0 if bare else float(rng.uniform(60.0, 200.0))),
        theta_surface=a(rng.uniform(0.15, 0.3)),
        residue_mass=a(0.0 if bare else float(rng.uniform(1000.0, 4000.0))),
        residue_age=a(rng.uniform(10.0, 60.0)),
        residue_wet=a(0.0),
    )
    z = a(0.0)
    state = PETState(surface=surface, pet=PETFluxes(z, z, z, z, z, z))
    pet = PETParams(
        albedo_dry=a(0.25),
        albedo_wet=a(0.15),
        albedo_maturity=a(0.23),
        albedo_residue=a(0.31),
        soil_resistance=a(rng.uniform(40.0, 70.0)),
        stomatal_resistance=a(rng.uniform(150.0, 250.0)),
    )
    params = PETSiteParams(
        pet=pet,
        elevation=a(ELEVATION),
        latitude=a(LATITUDE),
        wc13=a(WC13),
        wc15=a(WC15),
        wind_height=a(2.0),
        albedo_soil=a(0.13),
        trat=a(1.0),
        coefficients=PET_COEFFICIENTS.as_arrays(dtype),
    )
    return state, params, weather


_NO_BALANCE = "potential fluxes: no stock changes here; the soil-water day and the ledger close the balance"
#: recorded gaps of the PET package (frozen while H1 changes it); an exemption must keep failing
_UPCAST = (
    "gap W0-B/PET-1: float32 inputs give float64 fluxes (an implicit upcast inside the PET "
    "kernels); fixed in processes/pet after the H1 hardening merges"
)
_REF_BUILD = (
    "gap W0-B/PET-2: pet_asce_reference is faithful to asce-ewri-2005 but names no ref_build; "
    "the edit is in processes/pet/daily.py after the H1 hardening merges"
)
_WEATHER = ("tmin", "tmax", "srad", "rh", "wind_run", "doy")


def cases() -> list[ConformanceCase]:
    # the upcast needs float64 to exist: only the float32 part under x64 is exempt
    upcast = {"precision": _UPCAST}
    return [
        ConformanceCase(
            key="pet/shuttleworth_wallace@rzwqm2-4.6:faithful",
            make=make,
            n_days=N_DAYS,
            no_balance=_NO_BALANCE,
            grad=GradSpec(edge_variants=("bare", "cold")),
            coefficient_sets=("coefficients.sw",),
            forcing_fields=_WEATHER,
            exempt_float32_x64=upcast,
        ),
        ConformanceCase(
            key="pet/asce_reference@asce-ewri-2005:faithful",
            make=make,
            n_days=N_DAYS,
            no_balance=_NO_BALANCE,
            grad=GradSpec(edge_variants=("cold",)),
            coefficient_sets=("coefficients.asce",),
            forcing_fields=_WEATHER,
            exempt_float32_x64=upcast,
            exempt_checks={"registry": _REF_BUILD},
            # batched, XLA evaluates the tall-reference expression with 1-2 ulp difference
            transforms_exact=False,
        ),
        ConformanceCase(
            key="pet/priestley_taylor@dssat-4.8.6.0:faithful",
            make=make,
            n_days=N_DAYS,
            no_balance=_NO_BALANCE,
            grad=GradSpec(edge_variants=("bare", "cold")),
            coefficient_sets=("coefficients.pt",),
            forcing_fields=("tmin", "tmax", "srad"),
            exempt_float32_x64=upcast,
        ),
    ]
