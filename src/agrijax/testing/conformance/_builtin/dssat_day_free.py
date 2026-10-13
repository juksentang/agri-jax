"""Conformance cases of the processes the DSSAT day's free run added:

* ``soil_water/xtract@dssat-4.8.6.0:faithful`` - ``EP`` and the root extraction ``XTRACT`` (P4);
* ``soil_water/soil_albedo@dssat-4.8.6.0:faithful`` - the daily ``MSALB`` of ``SOILDYN`` (PD2);
* ``pet/priestley_taylor@dssat-4.8.6.0:port_soil_albedo`` - ``PETPT`` on the daily ``MSALB``;
* ``crop_iface/canopy_from_ceres@dssat-4.8.6.0:faithful`` - the DSSAT day's canopy adapter (P6).

Synthetic inputs on five DSSAT layers. ``XTRACT`` variants: ``nominal`` (the uptake below the
root supply and the available water), ``limited`` (``EOP > 10 TRWUP``: ``WUF = 1``), ``dry`` (two
layers at the lower limit, one limited by ``SW_AVAIL``), ``bare`` (``XHLAI = 0``: ``EP = 0``) and
``salus`` (``MESEV = 'S'``: the per-layer evaporation lowers the supply). The albedo: ``nominal``
(mulch, ``FF`` inside ``[0, 2]``), ``bare`` (no mulch cover), ``wet`` (``FF`` above 2), ``nomulch``
(the mulch switch off). ``PETPT``: ``nominal``, ``bare`` (LAI 0: ``ALBEDO = MSALB``), ``hot``
(``TMAX > 35``) and ``cold`` (``TMAX < 5``).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from agrijax.core.state import Forcing, field
from agrijax.iface.crop import CanopyRecord, CropWaterIn
from agrijax.iface.soil import SinkInputs
from agrijax.iface.surface import EvaporationRecord, PETFluxes, SoilAlbedo
from agrijax.models.day_dssat486_adapters import CANOPY_KEY, CanopyInState, canopy_from_ceres
from agrijax.processes.pet.coefficients import DSSAT_PT
from agrijax.processes.pet.spam_dssat import PTAlbedoParams, PTAlbedoState, SpamWeather
from agrijax.processes.soil_water.bucket import BucketFluxes, BucketForcing, MulchForcing
from agrijax.processes.soil_water.bucket_evap import (
    ALBEDO_COEFFICIENTS,
    SoilAlbedoParams,
    SoilAlbedoState,
    XtractParams,
    XtractState,
)

from ..case import ConformanceCase, GradSpec

N_DAYS = 3
N_CROP = 1
DLAYR = np.array([5.0, 10.0, 15.0, 15.0, 15.0])
LL = np.array([0.08, 0.09, 0.10, 0.11, 0.12])
DUL = np.array([0.20, 0.21, 0.22, 0.23, 0.24])

_NO_BALANCE_X = (
    "XTRACT computes the day's layer extraction and publishes it in the sink record (P4); it changes no "
    "store: the tipping bucket removes the uptake in its INTEGR (SWDELTX = -uptake / DLAYR) and closes "
    "the soil water balance (tests/integration/test_day_dssat486_free.py ledger)"
)
_NO_BALANCE_A = "a surface property (albedo), not a stock"
_NO_BALANCE_PT = "a potential rate (EO), not a stock"
_DSSAT_PORTS = (
    "the DSSAT-CSM day's ports (agrijax.iface.contract.DSSAT_PORTS: PD1, PD2, P4 on the DSSAT layers) are "
    "not part of the slot contracts (SLOT_CONTRACTS), which describe the RZWQM2 day"
)


class DayForcing(Forcing):
    """A forcing that only gives the runs their time axis (the process reads no forcing)."""

    day: Array = field(unit="d", description="day index of the run", dims=("T",))


def _a(dtype: Any) -> Any:
    def a(x: Any) -> Any:
        return jnp.asarray(np.asarray(x, dtype=np.float64), dtype)

    return a


def _canopy(a: Any, lai: float) -> CanopyRecord:
    v = np.full(N_CROP, lai)
    return CanopyRecord(lai=a(v), tlai=a(v), height=a(np.full(N_CROP, 150.0)))


# ------------------------------------------------------------------------ XTRACT
def make_xtract(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    a = _a(dtype)
    n = DLAYR.size
    frac = rng.uniform(0.4, 0.8, n)
    sw = LL + frac * (DUL - LL)
    rwu = rng.uniform(0.005, 0.03, n) * (np.arange(n) < 4)
    trwup = float(rwu.sum())
    eop = 10.0 * trwup * float(rng.uniform(0.3, 0.7))
    if variant == "limited":
        eop = 10.0 * trwup * float(rng.uniform(1.3, 1.8))
    if variant == "dry":
        sw[1] = LL[1]
        sw[2] = LL[2] - 0.001
        sw[3] = LL[3] + 1e-4  # SW_AVAIL - LL limits the layer
        eop = 10.0 * trwup * float(rng.uniform(1.3, 1.8))
    lai = 0.0 if variant == "bare" else float(rng.uniform(1.5, 4.0))
    es_cm = float(rng.uniform(0.05, 0.25))
    es_lyr = np.zeros(n)
    if variant == "salus":
        w = np.array([0.6, 0.3, 0.1, 0.0, 0.0])
        es_lyr = es_cm * w
    swdelts = -rng.uniform(0.0, 0.002, n)
    swdeltu = rng.uniform(0.0, 0.001, n) * (variant != "salus")
    flux = BucketFluxes.zeros(n, dtype).replace(swdelts=a(swdelts), swdeltu=a(swdeltu))
    ev = EvaporationRecord.zeros(n, dtype).replace(
        soil_evaporation=a(es_cm), soil_evaporation_layers=a(es_lyr), evaporation=a(10.0 * es_cm)
    )
    state = XtractState(
        trwu=a(0.0),
        sw=a(sw),
        flux=flux,
        evaporation=ev,
        water=CropWaterIn(sw=a(sw), eop=a(np.full(N_CROP, eop)), trwup=a(np.full(N_CROP, trwup))),
        rwu=a(rwu[None, :]),
        canopy=_canopy(a, lai),
        sink=SinkInputs.zeros(n, dtype),
    )
    params = XtractParams(dlayr=a(DLAYR), ll=a(LL), salus_es=a(1.0 if variant == "salus" else 0.0))
    return state, params, DayForcing(day=a(np.arange(N_DAYS)))


# ------------------------------------------------------------------------ SOILDYN albedo
def make_albedo(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    a = _a(dtype)
    n = DLAYR.size
    sw = LL + rng.uniform(0.3, 0.7, n) * (DUL - LL)
    if variant == "wet":
        sw[0] = 0.03 + 2.2 * (DUL[0] - 0.03)
    cover = 0.0 if variant == "bare" else float(rng.uniform(0.3, 0.8))
    mulch = MulchForcing(
        mass=a(np.full(N_DAYS, 3000.0 * cover)),
        cover=a(np.full(N_DAYS, cover)),
        new_mass=a(np.zeros(N_DAYS)),
        watfac=a(np.full(N_DAYS, 3.8)),
    )
    forcing = BucketForcing(
        rain=a(rng.uniform(0.0, 5.0, N_DAYS)),
        tmax=a(rng.uniform(15.0, 30.0, N_DAYS)),
        irrigation=a(np.zeros(N_DAYS)),
        mulch=mulch,
    )
    state = SoilAlbedoState(sw=a(sw), albedo=SoilAlbedo.constant(0.13, dtype))
    params = SoilAlbedoParams(
        salb=a(float(rng.uniform(0.1, 0.2))),
        dul1=a(DUL[0]),
        mulch_on=a(0.0 if variant == "nomulch" else 1.0),
        coefficients=ALBEDO_COEFFICIENTS.as_arrays(dtype),
    )
    return state, params, forcing


# ------------------------------------------------------------------------ PETPT on the daily MSALB
def make_pt(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    a = _a(dtype)
    lai = 0.0 if variant == "bare" else float(rng.uniform(1.0, 4.0))
    tmax = rng.uniform(20.0, 32.0, N_DAYS)
    if variant == "hot":
        tmax = rng.uniform(36.0, 40.0, N_DAYS)
    if variant == "cold":
        tmax = rng.uniform(1.0, 4.0, N_DAYS)
    weather = SpamWeather(
        tavg=a(tmax - 5.0),
        wind_run=a(rng.uniform(60.0, 250.0, N_DAYS)),
        co2=a(rng.uniform(360.0, 420.0, N_DAYS)),
        srad=a(rng.uniform(10.0, 28.0, N_DAYS)),
        tmax=a(tmax),
        tmin=a(tmax - rng.uniform(8.0, 12.0, N_DAYS)),
    )
    state = PTAlbedoState(
        canopy=_canopy(a, lai),
        pet=PETFluxes.zeros(dtype),
        albedo=SoilAlbedo(msalb=a(float(rng.uniform(0.08, 0.2))), swalb=a(0.1)),
    )
    return state, PTAlbedoParams(coefficients=DSSAT_PT.as_arrays(dtype)), weather


# ------------------------------------------------------------------------ canopy adapter
def make_canopy(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    a = _a(dtype)
    state = CanopyInState(
        lai=a(rng.uniform(0.5, 4.0, N_CROP)),
        canht=a(rng.uniform(0.5, 1.6, N_CROP)),
        canopy_out=CanopyRecord.zeros(N_CROP, dtype),
    )
    return state, None, DayForcing(day=a(np.arange(N_DAYS)))


def cases() -> list[ConformanceCase]:
    return [
        ConformanceCase(
            key="soil_water/xtract@dssat-4.8.6.0:faithful",
            make=make_xtract,
            variants=("nominal", "limited", "dry", "bare", "salus"),
            n_days=N_DAYS,
            own="surface.xtract",
            ports={
                "sw": "soil_water.theta",
                "flux": "soil_water.flux",
                "evaporation": "iface.evaporation",
                "water": "iface.crop_water.{slot}",
                "rwu": "water_supply.{slot}.rwu",
                "canopy": "iface.canopy.{slot}",
                "sink": "soil_water.sink_in",
            },
            slot_contract=None,
            no_slot_contract=_DSSAT_PORTS,
            no_balance=_NO_BALANCE_X,
            grad=GradSpec(wrt=("ll", "dlayr"), edge_variants=("dry", "bare")),
        ),
        ConformanceCase(
            key="soil_water/soil_albedo@dssat-4.8.6.0:faithful",
            make=make_albedo,
            variants=("nominal", "bare", "wet", "nomulch"),
            n_days=N_DAYS,
            own="surface.albedo",
            ports={"sw": "soil_water.theta", "albedo": "iface.soil_albedo"},
            slot_contract=None,
            no_slot_contract=_DSSAT_PORTS,
            no_balance=_NO_BALANCE_A,
            grad=GradSpec(edge_variants=("wet",)),
            coefficient_sets=("coefficients",),
        ),
        ConformanceCase(
            key="pet/priestley_taylor@dssat-4.8.6.0:port_soil_albedo",
            make=make_pt,
            variants=("nominal", "bare", "hot", "cold"),
            n_days=N_DAYS,
            ports={"canopy": "iface.canopy.{slot}", "pet": "iface.pet", "albedo": "iface.soil_albedo"},
            slot_contract=None,
            no_slot_contract=_DSSAT_PORTS,
            no_balance=_NO_BALANCE_PT,
            grad=GradSpec(edge_variants=("bare",)),
            coefficient_sets=("coefficients",),
            forcing_fields=("srad", "tmax", "tmin"),
        ),
        ConformanceCase(
            key=CANOPY_KEY,
            make=make_canopy,
            process=canopy_from_ceres,
            n_days=N_DAYS,
            own="surface.canopy_in",
            slot_contract=None,
            no_slot_contract="an assembly adapter of the DSSAT day between the crop's own state and P6; "
            + _DSSAT_PORTS,
            ports={
                "lai": "crops.{slot}.growth.lai",
                "canht": "crops.{slot}.growth.canht",
                "canopy_out": "iface.canopy.{slot}",
            },
            no_balance="an adapter (P6 lai = tlai = LAI, height = 100 CANHT), checked in "
            "tests/unit/test_day_dssat486.py",
            exempt_checks={
                "units": "the adapter reads the crop's own end-of-day LAI and CANHT (crops.<slot>.growth), "
                "which are the crop module's state, not contract ports; its output P6 is checked"
            },
            grad=None,
            no_grad="a linear map without parameters (d lai / d LAI = 1, d height / d CANHT = 100)",
        ),
    ]
