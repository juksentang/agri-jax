"""Conformance cases of the DSSAT-CSM SPAM partition (pet slot) and the soil / mulch evaporation
(soil_water slot) of the DSSAT day:

* ``pet/spam_pse@dssat-4.8.6.0:faithful`` - ``EO -> EOS`` (``PSE``);
* ``pet/spam_trans@dssat-4.8.6.0:faithful`` - ``EO, EVAP -> EOP`` (``TRANS``, ``TRATIO``);
* ``soil_water/mulch_evap@dssat-4.8.6.0:faithful`` - ``EOS -> EM, EOS_SOIL`` (``MULCH_EVAP``);
* ``soil_water/soilev@dssat-4.8.6.0:faithful`` - Ritchie two-stage ``ES`` (``SOILEV``);
* ``soil_water/esr_soilevap@dssat-4.8.6.0:faithful`` - layered ``ES`` (``ESR_SoilEvap``).

Synthetic inputs on five DSSAT layers (thicknesses of the maize examples' top layers): a cropped
surface in mid season (nominal) and, where the process has one, the edge variants (bare soil,
old ``PSE`` form, capped ``EOP``, no mulch, stage 1 / stage 2 of ``SOILEV``, wet / intermediate /
dry profiles of ``ESR``). No balances: every one of these processes computes fluxes only and
changes no stock (the bucket and the mulch module apply them; see the process docstrings).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agrijax.iface.crop import CanopyRecord
from agrijax.iface.surface import PETFluxes
from agrijax.processes.pet.spam_dssat import (
    SPAM_COEFFICIENTS,
    SpamParams,
    SpamPSEState,
    SpamTransState,
    SpamWeather,
)
from agrijax.processes.soil_water.bucket_evap import EVAP_COEFFICIENTS, SoilEvapParams, SoilEvapState

from ..case import ConformanceCase, GradSpec

N_DAYS = 3
N_CROP = 1
DLAYR = np.array([5.0, 10.0, 15.0, 15.0, 15.0])
DS = np.cumsum(DLAYR)
LL = np.array([0.08, 0.09, 0.10, 0.11, 0.12])
DUL = np.array([0.20, 0.21, 0.22, 0.23, 0.24])
#: CERES-Maize KEP of the reference runs (KCAN 0.85 in MZCER048.SPE, MZ_PHENOL.for:338)
KEP = 0.6854839

_NO_BALANCE = (
    "potential or actual evaporation fluxes only: no stock changes here; the bucket (WATBAL) "
    "removes ES from the soil layers and closes the balance, the mulch module removes EM"
)


def _canopy(a: Any, lai: float) -> CanopyRecord:
    v = np.full(N_CROP, lai)
    return CanopyRecord(lai=a(v), tlai=a(v), height=a(np.full(N_CROP, 150.0)))


def _pet(a: Any, eo: float = 0.0, eos_cm: float = 0.0) -> PETFluxes:
    z = a(0.0)
    return PETFluxes(
        transpiration=z,
        soil_evaporation=a(eos_cm),
        residue_evaporation=z,
        reference_short=z,
        reference_tall=z,
        eo_priestley_taylor=a(eo),
    )


def _spam_params(a: Any, dtype: Any, ksevap: float = KEP) -> SpamParams:
    return SpamParams(
        ksevap=a(ksevap), ktrans=a(KEP), c4=True, coefficients=SPAM_COEFFICIENTS.as_arrays(dtype)
    )


def make_pse(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """Nominal: LAI 1.5-4 with the crop's KSEVAP; ``bare``: LAI 0; ``old``: KSEVAP = -99 (the
    form before a crop sets it) with LAI either side of the split at 1."""

    def a(x: Any) -> Any:
        return np.asarray(x, dtype)

    lai = 0.0 if variant == "bare" else float(rng.uniform(1.5, 4.0))
    if variant == "old":
        lai = float(rng.choice([rng.uniform(0.2, 0.8), rng.uniform(1.3, 3.0)]))
    state = SpamPSEState(canopy=_canopy(a, lai), pet=_pet(a, eo=float(rng.uniform(2.0, 7.0))))
    return state, _spam_params(a, dtype, -99.0 if variant == "old" else KEP), None


def make_trans(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """Nominal: LAI 1.5-4, small evaporation; ``cap``: evaporation large enough that EOP is capped
    at ``EO - EOP_reduc - EVAP``; ``bare``: LAI 0 (TRANS not called)."""

    def a(x: Any) -> Any:
        return np.asarray(x, dtype)

    lai = 0.0 if variant == "bare" else float(rng.uniform(1.5, 4.0))
    eo = float(rng.uniform(3.0, 7.0))
    evap = eo * float(rng.uniform(0.7, 0.9)) if variant == "cap" else float(rng.uniform(0.05, 0.4))
    state = SpamTransState(canopy=_canopy(a, lai), pet=_pet(a, eo=eo), evaporation=a(evap))
    weather = SpamWeather(
        tavg=a(rng.uniform(15.0, 28.0, N_DAYS)),
        wind_run=a(rng.uniform(60.0, 250.0, N_DAYS)),
        co2=a(rng.uniform(360.0, 420.0, N_DAYS)),
    )
    return state, _spam_params(a, dtype), weather


def make_soil(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """Soil-side evaporation state on five layers for ``variant``:

    ``nominal`` (mulch on, stage 2 of SOILEV, no rain, dry profile for ESR), ``nomulch`` (mass 0),
    ``stage1`` (SUMES1 below U), ``rain`` (WINF > SUMES2: stage-2 refill), ``wet`` and
    ``intermediate`` (ESR wet profiles; the top layer above / below the threshold)."""

    def a(x: Any) -> Any:
        return np.asarray(x, dtype)

    u = float(rng.uniform(6.0, 9.0))
    frac = rng.uniform(0.3, 0.6, DLAYR.size)
    sw = LL + frac * (DUL - LL)
    swdelts = np.zeros(DLAYR.size)
    if variant == "wet":
        sw = DUL + rng.uniform(0.02, 0.05, DLAYR.size)
        swdelts = -rng.uniform(0.001, 0.005, DLAYR.size)
    if variant == "intermediate":
        sw = DUL + rng.uniform(0.02, 0.05, DLAYR.size)
        sw[0] = LL[0] + 0.3 * (DUL[0] - LL[0])
    swdeltu = -rng.uniform(0.0, 0.002, DLAYR.size)
    s2 = float(rng.uniform(3.0, 8.0))
    s1, t = u, (s2 / 3.5) ** 2
    if variant == "stage1":
        s1, s2, t = float(rng.uniform(0.5, 0.5 * u)), 0.0, 0.0
    winf = float(s2 + rng.uniform(1.0, 3.0)) if variant == "rain" else 0.0
    mass = 0.0 if variant == "nomulch" else float(rng.uniform(1500.0, 4000.0))
    eos_cm = float(rng.uniform(0.15, 0.45))
    z = a(0.0)
    zl = a(np.zeros(DLAYR.size))
    state = SoilEvapState(
        sw=a(sw),
        swdelts=a(swdelts),
        swdeltu=a(swdeltu),
        winf=a(winf),
        dlayr=a(DLAYR),
        ds=a(DS),
        dul=a(DUL),
        ll=a(LL),
        mulch_mass=a(mass),
        mulch_cover=a(0.0 if mass == 0.0 else float(rng.uniform(0.4, 0.8))),
        mulch_water=a(float(rng.uniform(0.2, 1.0))),
        mulch_am=a(4.0),
        mulch_extfac=a(0.8),
        sumes1=a(s1),
        sumes2=a(s2),
        t=a(t),
        swef=a(0.9 - 0.00038 * (DLAYR[0] - 30.0) ** 2),
        eos_soil=a(eos_cm * 10.0 * float(rng.uniform(0.6, 0.9))),
        em=a(float(rng.uniform(0.0, 0.3))),
        es=z,
        es_lyr=zl,
        upflow=zl,
        evap=z,
        pet=_pet(a, eos_cm=eos_cm),
    )
    params = SoilEvapParams(
        u=a(u), pmfraction=a(0.0), mulch_active=True, coefficients=EVAP_COEFFICIENTS.as_arrays(dtype)
    )
    return state, params, None


_SOILS = {"pet": "iface.pet"}


def cases() -> list[ConformanceCase]:
    return [
        ConformanceCase(
            key="pet/spam_pse@dssat-4.8.6.0:faithful",
            make=make_pse,
            variants=("nominal", "old"),
            n_days=N_DAYS,
            ports={"canopy": "iface.canopy.{slot}", "pet": "iface.pet"},
            no_balance=_NO_BALANCE,
            grad=GradSpec(edge_variants=("bare",)),
            coefficient_sets=("coefficients.pse",),
        ),
        ConformanceCase(
            key="pet/spam_trans@dssat-4.8.6.0:faithful",
            make=make_trans,
            variants=("nominal", "cap"),
            n_days=N_DAYS,
            ports={"canopy": "iface.canopy.{slot}", "pet": "iface.pet", "evaporation": "soil_water.evap"},
            slot_contract=None,
            no_slot_contract=(
                "TRANS reads the day's actual evaporation EVAP of the soil water module (SPAM's same-day "
                "cycle PSE -> soil evaporation -> TRANS); the slot contract of 'pet' (RZWQM2 day) has no "
                "port for it (the DSSAT-day contract revision adds one)"
            ),
            exempt_checks={
                "units": "port evaporation -> soil_water.evap is not a port of the coupling contract yet "
                "(the DSSAT-day revision adds it); every other field is checked"
            },
            no_balance=_NO_BALANCE,
            grad=GradSpec(edge_variants=("bare",)),
            coefficient_sets=("coefficients.trans",),
            forcing_fields=("tavg", "wind_run", "co2"),
        ),
        ConformanceCase(
            key="soil_water/mulch_evap@dssat-4.8.6.0:faithful",
            make=make_soil,
            variants=("nominal", "nomulch"),
            n_days=N_DAYS,
            ports=_SOILS,
            no_balance=_NO_BALANCE,
            grad=GradSpec(),
            coefficient_sets=("coefficients.mulch", "coefficients.gate"),
        ),
        ConformanceCase(
            key="soil_water/soilev@dssat-4.8.6.0:faithful",
            make=make_soil,
            variants=("nominal", "stage1", "rain"),
            n_days=N_DAYS,
            ports={},
            no_balance=_NO_BALANCE,
            grad=GradSpec(),
            coefficient_sets=("coefficients.soilev", "coefficients.gate"),
        ),
        ConformanceCase(
            key="soil_water/esr_soilevap@dssat-4.8.6.0:faithful",
            make=make_soil,
            variants=("nominal", "wet", "intermediate"),
            n_days=N_DAYS,
            ports={},
            no_balance=_NO_BALANCE,
            grad=GradSpec(),
            coefficient_sets=("coefficients.esr", "coefficients.gate"),
        ),
    ]
