"""Conformance case of the PRMS snowpack ``snow/prms@rzwqm2-4.6:faithful`` (port P9).

Six winter days at CA-TPA (latitude 42.7 N) on a pack of a few centimetres that exists at the
start (the routine has run before: no re-initialisation): a cold snowfall, a cold clear day, a
thaw, rain on the pack, a second thaw and a frozen dry day. Variants for the finite-gradient
check only: ``empty`` (warm days, no pack: the routine is not called) and ``first_snow`` (no pack,
the first snowfall of the run with the routine's start values). The pack's water balance
``dSWE = intercepted - melt - melt runoff - sublimation`` closes every day.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agrijax.iface.surface import SnowOut
from agrijax.processes.snow import PRMS_SNOW, PrmsSnowParams, SnowForcing, SnowState
from agrijax.processes.snow.sno import SnoFile

from ..case import Balance, ConformanceCase, GradSpec, Tolerance

N_DAYS = 6
LATITUDE = 0.745163
#: the parameters of every reference .sno file (the Lucerne CO 1994-95 set)
SNO = SnoFile(
    den_max=0.6,
    den_init=0.1,
    freeh2o_cap=0.05,
    settle_const=0.1,
    tmax_allsnow=32.0,
    albset_rnm=0.6,
    albset_rna=0.8,
    albset_snm=0.2,
    albset_sna=0.05,
    cov_type=1,
    covden_sum=0.5,
    covden_win=0.5,
    rad_trncf=0.5,
    emis_noppt=0.757,
    potet_sublim=0.5,
    cecn_coef=(5.0,) * 12,
    tstorm_mo=(0,) * 12,
    frac_infil=0.8,
    melt_look=90,
    melt_force=90,
    snarea_thresh=50.0,
    hru_deplcrv=1,
    ndepl=1,
    snarea_curve=((0.05, 0.24, 0.40, 0.53, 0.65, 0.75, 0.82, 0.88, 0.93, 0.99, 1.0),),
)


def make(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """Parameters, the morning pack and six days of weather for ``variant``."""

    def a(x: Any) -> Any:
        return np.asarray(x, dtype)

    params = PrmsSnowParams.from_sno(SNO, LATITUDE, dtype=dtype)
    if variant == "empty":
        tmin = rng.uniform(2.0, 6.0, N_DAYS)
        tmax = tmin + rng.uniform(5.0, 10.0, N_DAYS)
        precip = np.where(rng.uniform(size=N_DAYS) < 0.5, rng.uniform(0.2, 1.5, N_DAYS), 0.0)
        state = SnowState.initial(0.0, dtype)
    elif variant == "first_snow":
        tmin = rng.uniform(-9.0, -6.0, N_DAYS)
        tmax = tmin + rng.uniform(2.0, 4.0, N_DAYS)
        precip = np.concatenate([rng.uniform(0.5, 1.5, 1), np.zeros(N_DAYS - 1)])
        state = SnowState.initial(0.0, dtype)
    else:
        # cold snowfall, cold clear, thaw, rain on the pack, thaw, frozen dry
        tmin = np.array([-8.0, -12.0, -1.0, 1.0, 0.5, -9.0]) + rng.uniform(-0.5, 0.5, N_DAYS)
        tmax = np.array([-2.0, -5.0, 7.0, 8.0, 9.0, -3.0]) + rng.uniform(-0.5, 0.5, N_DAYS)
        precip = np.array([1.2, 0.0, 0.0, 0.6, 0.0, 0.0]) * rng.uniform(0.8, 1.2, N_DAYS)
        swe = float(rng.uniform(2.0, 4.0))
        pkwe = swe * PRMS_SNOW.inch_per_cm
        den = float(rng.uniform(0.2, 0.3))
        temp = float(rng.uniform(-3.0, -1.0))
        z = a(0.0)
        state = SnowState(
            swe=a(swe),
            pk_def=a(-temp * pkwe * PRMS_SNOW.ice_heat_inch),
            pk_temp=a(temp),
            pk_ice=a(pkwe),
            freeh2o=z,
            pk_depth=a(pkwe / den),
            pk_den=a(den),
            pss=a(pkwe),
            pst=a(pkwe * 1.3),
            snsv=z,
            albedo=a(0.7),
            iasw=z,
            iso=a(1.0),
            mso=a(1.0),
            lso=z,
            lst=z,
            slst=a(4.0),
            intal=a(1.0),
            scrv=z,
            pksv=z,
            scasv=z,
            sstart=z,
            started=a(1.0),
            month=a(2.0),
            cdate_day=a(9.0),
            cover=z,
            intercepted=z,
            out=SnowOut.zeros(dtype),
        )
    forcing = SnowForcing(
        tmin=a(tmin),
        tmax=a(tmax),
        srad=a(rng.uniform(6.0, 12.0, N_DAYS)),
        precipitation=a(precip),
        doy=a(np.arange(40, 40 + N_DAYS)),
    )
    return state, params, forcing


def _storage(state: Any, params: Any) -> Any:
    return state.swe


def _inflow(before: Any, after: Any, params: Any, forcing_t: Any) -> Any:
    return after.intercepted


def _outflow(before: Any, after: Any, params: Any, forcing_t: Any) -> Any:
    return after.out.melt + after.out.melt_runoff + after.out.sublimation


def cases() -> list[ConformanceCase]:
    return [
        ConformanceCase(
            key="snow/prms@rzwqm2-4.6:faithful",
            make=make,
            n_days=N_DAYS,
            ports={"out": "iface.snow"},
            balances=(
                Balance(
                    "snow water equivalent",
                    "cm",
                    _storage,
                    _inflow,
                    _outflow,
                    tol64=Tolerance(1e-12, 1e-13),
                    tol32=Tolerance(1e-6, 1e-6),
                ),
            ),
            grad=GradSpec(edge_variants=("empty", "first_snow")),
            forcing_fields=("tmin", "tmax", "srad", "precipitation", "doy"),
        )
    ]
