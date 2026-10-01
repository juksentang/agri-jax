"""Conformance cases of the PET processes (Shuttleworth-Wallace, ASCE reference, Priestley-Taylor)
and of the ``EOP`` adapter that turns the PET port into the crop's potential transpiration.

Three days of synthetic summer weather on one site: a cropped surface with flat residue (nominal),
and, for gradients only, bare soil without residue (``LAI = 0``) and a cold, humid day. The PET
module (own subtree ``surface.pet``) reads the canopy port P6 (``iface.canopy.<slot>``) and the
node water content P7 (``soil_water.theta``) and writes P5 (``iface.pet``); each case binds exactly
the ports its process uses.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agrijax.iface.crop import CanopyRecord, CropWaterIn
from agrijax.iface.surface import DailyWeather, PETFluxes
from agrijax.processes.pet import PET_COEFFICIENTS, PETParams
from agrijax.processes.pet.daily import RESIDUE_KINDS, PETSiteParams, PETState, SurfaceResidue
from agrijax.processes.pet.eop import EOPState

from ..case import ConformanceCase, GradSpec

N_DAYS = 3
N_CROP = 1
N_NODE = 4
#: CA-TPA surface layer and site (as tests/unit/test_pet_grad_finite.py)
WC13, WC15, ELEVATION, LATITUDE = 0.255198, 0.141628, 200.0, 0.745163


def make(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """Site parameters, PET module state (ports filled) and three days of weather for ``variant``."""

    def a(x: Any) -> Any:
        return np.asarray(x, dtype)

    cold = variant == "cold"
    bare = variant == "bare"
    tmin = rng.uniform(-4.0, 0.0, N_DAYS) if cold else rng.uniform(10.0, 16.0, N_DAYS)
    tmax = tmin + rng.uniform(6.0, 12.0, N_DAYS)
    srad = rng.uniform(4.0, 8.0, N_DAYS) if cold else rng.uniform(15.0, 26.0, N_DAYS)
    weather = DailyWeather(
        tmin=a(tmin),
        tmax=a(tmax),
        srad=a(srad),
        rh=a(rng.uniform(85.0, 95.0, N_DAYS) if cold else rng.uniform(50.0, 75.0, N_DAYS)),
        wind_run=a(rng.uniform(80.0, 250.0, N_DAYS)),
        doy=a(rng.integers(20, 40, N_DAYS) if cold else rng.integers(170, 200, N_DAYS)),
        # RTH differs from RTS by up to 0.11 MJ m-2 d-1 in the reference runs
        srad_horizontal=a(srad + rng.uniform(-0.1, 0.1, N_DAYS)),
    )
    lai = np.zeros(N_CROP) if bare else rng.uniform(1.5, 4.0, N_CROP)
    canopy = CanopyRecord(
        lai=a(lai),
        tlai=a(lai + (0.0 if bare else rng.uniform(0.1, 0.5, N_CROP))),
        height=a(np.zeros(N_CROP) if bare else rng.uniform(60.0, 200.0, N_CROP)),
    )
    theta = a(rng.uniform(0.15, 0.3, N_NODE))
    z = a(0.0)
    state = PETState(canopy=canopy, theta=theta, pet=PETFluxes(z, z, z, z, z, z))
    pet = PETParams(
        albedo_dry=a(0.25),
        albedo_wet=a(0.15),
        albedo_maturity=a(0.23),
        albedo_residue=a(0.31),
        soil_resistance=a(rng.uniform(40.0, 70.0)),
        stomatal_resistance=a(rng.uniform(150.0, 250.0)),
    )
    residue = SurfaceResidue(
        mass=a(0.0 if bare else float(rng.uniform(1000.0, 4000.0))),
        age=a(rng.uniform(10.0, 60.0)),
        wet=a(0.0),
        kind=a(RESIDUE_KINDS["none"] if bare else float(rng.choice([1.0, 2.0, 3.0]))),
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
        rainfall_zone=3,
        residue_cover_factor=2.5,
        coefficients=PET_COEFFICIENTS.as_arrays(dtype),
        residue=residue,
    )
    return state, params, weather


def make_eop(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """Today's PET record (transpiration 0 for ``bare``) and a crop-water record for two crops."""

    def a(x: Any) -> Any:
        return np.asarray(x, dtype)

    z = a(0.0)
    t = a(0.0 if variant == "bare" else float(rng.uniform(0.05, 0.8)))
    pet = PETFluxes(t, a(rng.uniform(0.0, 0.3)), a(rng.uniform(0.0, 0.1)), z, z, z)
    n_crop = 2
    water = CropWaterIn(
        sw=a(rng.uniform(0.1, 0.3, 5)), eop=a(np.zeros(n_crop)), trwup=a(rng.uniform(0.0, 0.5, n_crop))
    )
    return EOPState(pet=pet, crop_water=water), None, None


_NO_BALANCE = "potential fluxes: no stock changes here; the soil-water day and the ledger close the balance"
_WEATHER = ("tmin", "tmax", "srad", "rh", "wind_run", "doy")
_PET_PORT = {"pet": "iface.pet"}
_CANOPY = {"canopy": "iface.canopy.{slot}"}


def cases() -> list[ConformanceCase]:
    return [
        ConformanceCase(
            key="pet/shuttleworth_wallace@rzwqm2-4.6:faithful",
            make=make,
            n_days=N_DAYS,
            ports={**_CANOPY, "theta": "soil_water.theta", **_PET_PORT},
            no_balance=_NO_BALANCE,
            grad=GradSpec(edge_variants=("bare", "cold")),
            coefficient_sets=("coefficients.sw",),
            forcing_fields=(*_WEATHER, "srad_horizontal"),
            # float32 (x64 off): vmap(jit) differs from jit by 3 ulp in soil_evaporation, measured
            # on a cluster CPU node; float64 is bit for bit
            transforms_exact=False,
        ),
        ConformanceCase(
            key="pet/asce_reference@asce-ewri-2005:faithful",
            make=make,
            n_days=N_DAYS,
            ports=_PET_PORT,
            no_balance=_NO_BALANCE,
            grad=GradSpec(edge_variants=("cold",)),
            coefficient_sets=("coefficients.asce",),
            forcing_fields=_WEATHER,
            # batched, XLA evaluates the tall-reference expression with 1-2 ulp difference
            transforms_exact=False,
        ),
        ConformanceCase(
            key="pet/priestley_taylor@dssat-4.8.6.0:faithful",
            make=make,
            n_days=N_DAYS,
            ports={**_CANOPY, **_PET_PORT},
            no_balance=_NO_BALANCE,
            grad=GradSpec(edge_variants=("bare", "cold")),
            coefficient_sets=("coefficients.pt",),
            forcing_fields=("tmin", "tmax", "srad"),
        ),
        ConformanceCase(
            key="crop_iface/eop_from_pet@rzwqm2-4.6:faithful",
            make=make_eop,
            n_days=N_DAYS,
            ports={**_PET_PORT, "crop_water": "iface.crop_water.{slot}"},
            no_balance=(
                "a unit conversion of the day's potential transpiration (cm d-1 to mm d-1): no stock; "
                "the soil-water day removes the water taken up and the ledger closes it"
            ),
            grad=None,
            no_grad=(
                "the process has no parameter (EOP = 10 PET); its derivative with respect to the "
                "port is exactly 10, tested in tests/unit/test_pet_ports.py"
            ),
            coefficient_sets=(),
        ),
    ]
