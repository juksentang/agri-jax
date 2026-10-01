"""PRMS single-layer snowpack of RZWQM2 4.6 (the ``ISHAW = 0`` path), one call per day.

RZWQM2 runs the PRMS ``snocomp`` snowpack (Leavesley et al. 1983) through its interface
``SNOWCOMP_PRMS`` (``RZWQM/Snowprms.for``) once a day in ``PHYSCL`` when SHAW is off
(``Rzday.for`` 1529-1592):

* the snow branch is entered when the pack holds water or the day's mean air temperature
  ``TM = (TMIN + TMAX) / 2`` is at or below 0 degC (``Rzday.for`` 876, 1529). On such a day **all**
  of the day's breakpoint storms go to the pack routine, not to infiltration (``RFDACC``,
  1537-1541), and the routine is called when there is a pack or precipitation (1543-1549);
* precipitation is snow when the day's mean temperature is at or below 0 degC, rain otherwise
  (``Snowprms.for`` 78-109); rain on a pack is added to it (``PPT_TO_PACK``);
* the pack's heat is its cold content ``PK_DEF`` [cal cm-2] (pack temperature
  ``-PK_DEF / (1.27 PKWE)``), its free water is held up to ``FREEH2O_CAP`` times the ice, and a
  night and a day half-day energy balance (``SNOWBAL``: short wave after the albedo and the
  transmission coefficient, long wave from the air temperature, convection-condensation on rain
  days, conduction to the surface) melt it (``CALIN``) or cool it (``CALOSS``);
* the albedo decays with the days since the last snowfall (``SNALBEDO``), the snow-covered area
  follows the areal depletion curve (``SNOWCOV``), the pack settles (``SNORUN`` 362-367);
* sublimation is ``POTET_SUBLIM x ESN x cover`` from the pack, with the sublimation potential
  ``ESN`` fixed at 0.011 cm d-1 (``Rzpet.for`` 1425: the ``SNOWQE`` estimate and its cap by the
  potential soil evaporation are overwritten). Only the Shuttleworth-Wallace PET sets ``ESN``: a
  scenario with a reference-ET method (``IPET = 1, 2``) never does, ``ESN`` keeps the start value 0
  of the ``-init=zero`` build and the pack does not sublimate (:attr:`PrmsSnowParams.ipet`);
* back in ``PHYSCL`` (``Rzday.for`` 1552-1554) the day's melt ``SMELT`` splits into the melt that
  infiltrates, ``AIRR = FRAC_INFIL x SMELT`` (an event without breakpoints), and the melt runoff
  ``SNRO = SMELT - AIRR``, and the soil evaporation is scaled by ``1 - FSNC`` (the snow cover).

The pack water equivalent ``SNP`` [cm] is ``.ana`` column 92 (labelled "SNOW DEPTH") and, times
10, the ``SNOW`` [mm] of the embedded DSSAT crop (``DSSATDRV.for`` 663); ``SMELT`` is ``.ana``
column 105. The module writes the port P9 (``iface.snow``: melt, melt runoff, SWE in mm,
sublimation) and keeps the pack in its own state (``surface.snow``), including the day's snow
cover ``cover`` and the precipitation the routine took, ``intercepted``: the two quantities the
RZWQM2 soil-water day uses to drop the day's storms and to scale its evaporation. They are fields of
the snow state, not of the port P9, and the soil-water day of this implementation does not read them
yet.

Every branch of the Fortran is a ``jnp.where`` on both arms (the three process rules); divisions
use guarded denominators so that both arms stay finite. The reference is compiled with ``-save
-init=zero`` (``Makefile.mak`` 22): every local variable keeps its value between calls and starts
at zero, so the counters of ``SNALBEDO`` (``SLST``, ``INTAL``) and ``SNOWCOV`` (``SCRV``, ``PKSV``,
``SNOWCOV_AREASV``) and the month of ``CDATE`` are state here.

Written from the PRMS documentation (Leavesley, Lichty, Troutman and Saindon 1983, USGS WRI
83-4238) with the reference source read for its conventions and constants (file, line and
routine cited; no RZWQM2 statement is reproduced).

Not reproduced (outside every reference ``.sno`` file: all 16 of ``RZWQM_sw_batch`` are the same
Lucerne CO 1994-95 set with ``COV_TYPE = 1`` and no thunderstorm month):

* ``COV_TYPE > 1`` (the sublimation then depends on the day's potential transpiration and the
  winter or summer cover density; ``COV_TYPE = 3`` halves the convection coefficient);
* thunderstorm months (``TSTORM_MO = 1``: the rain-day emissivity from the ratio of the daily to
  the clear-sky radiation).

:func:`PrmsSnowParams.from_sno` raises on either. The mixed rain-snow event of ``PPT_TO_PACK``
(``PPTMIX = 1``) is never produced by ``SNOWCOMP_PRMS`` (``Snowprms.for`` 79: rain or snow per
day) and is not written here.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.coefficients import numerical_guard
from agrijax.core.ports import port
from agrijax.core.process import Deviation, Source, process
from agrijax.core.state import Forcing, Params, State, field
from agrijax.core.units import MM_PER_CM
from agrijax.iface.surface import SnowOut

from .coefficients import ALBEDO_DAYS, PRMS_SNOW, PrmsSnowCoefficients
from .sno import SnoFile

__all__ = [
    "PACK_FIELDS",
    "PrmsSnowParams",
    "SnowForcing",
    "SnowState",
    "snow_prms",
]

#: square root guard: the pack conduction ``sqrt(0.0154 PK_DEN 13751)`` at ``PK_DEN = 0``
_SQRT_FLOOR = numerical_guard(
    "snow.sqrt_floor",
    1e-30,
    "floor of the conduction square root: a finite gradient at an empty pack, representable in float32",
)

# ------------------------------------------------------------------------ CDATE calendar (Rzman.for 37-56)
#: day of year of the last day of each previous month, ``KDA`` of ``CDATE`` (Rzman.for 37)
_KDA: tuple[int, ...] = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)
#: ``CDATE`` treats the year as leap when it is divisible by this (Rzman.for 39)
_LEAP_DIVISOR = 4
#: day of year of 29 February in a leap year, and its month and day of month (Rzman.for 42-46)
_LEAP_DOY, _LEAP_MONTH, _LEAP_DOM = 60, 2, 29
#: the stage value of the melt stage and of the accumulation stage (``ISO``, ``MSO``, ``INTAL``)
_MELT, _ACCUM = 2.0, 1.0


class SnowForcing(Forcing):
    """Daily forcing of the PRMS snowpack (time axis first)."""

    tmin: Array = field(
        unit="degC", dims="T", fortran_name="TMIN", description="daily minimum air temperature"
    )
    tmax: Array = field(
        unit="degC", dims="T", fortran_name="TMAX", description="daily maximum air temperature"
    )
    srad: Array = field(
        unit="MJ m-2 d-1",
        dims="T",
        fortran_name="RTS",
        description="daily solar radiation RTS, hourly re-sum (agrijax.forcing.radiation)",
    )
    precipitation: Array = field(
        unit="cm d-1",
        dims="T",
        fortran_name="RFDNEW",
        description="the day's breakpoint precipitation, its storms summed (agrijax.forcing.precipitation)",
    )
    doy: Array = field(unit="d", dims="T", fortran_name="JDAY", description="day of year 1..366")


class PrmsSnowParams(Params):
    """The ``.sno`` parameters of the site (``READ_SNOW``, ``Snowprms.for`` 1024-1127) and the latitude.

    Build it with :meth:`from_sno`, which checks that the file stays inside what the module
    reproduces. Array fields are calibratable leaves; the day-of-year thresholds are static.
    """

    den_max: Array = field(
        unit="g cm-3", fortran_name="DEN_MAX", dims=(), description="average maximum pack density"
    )
    den_init: Array = field(
        unit="g cm-3", fortran_name="DEN_INIT", dims=(), description="initial density of new-fallen snow"
    )
    freeh2o_cap: Array = field(
        unit="-", fortran_name="FREEH2O_CAP", dims=(), description="free-water holding capacity of the pack"
    )
    settle_const: Array = field(
        unit="-", fortran_name="SETTLE_CONST", dims=(), description="pack settlement time constant"
    )
    tmax_allsnow: Array = field(
        unit="degF",
        fortran_name="TMAX_ALLSNOW",
        dims=(),
        description="precipitation all snow below this maximum",
    )
    albset_rnm: Array = field(
        unit="-", fortran_name="ALBSET_RNM", dims=(), description="albedo reset, rain fraction, melt stage"
    )
    albset_snm: Array = field(
        unit="in", fortran_name="ALBSET_SNM", dims=(), description="albedo reset, new snow, melt stage"
    )
    covden_win: Array = field(
        unit="-", fortran_name="COVDEN_WIN", dims=(), description="winter vegetation cover density"
    )
    rad_trncf: Array = field(
        unit="-", fortran_name="RAD_TRNCF", dims=(), description="solar radiation transmission coefficient"
    )
    emis_noppt: Array = field(
        unit="-",
        fortran_name="EMIS_NOPPT",
        dims=(),
        description="emissivity of air on days without precipitation",
    )
    potet_sublim: Array = field(
        unit="-", fortran_name="POTET_SUBLIM", dims=(), description="fraction of the potential ET sublimated"
    )
    cecn_coef: Array = field(
        unit="cal cm-2 degC-1",
        fortran_name="CECN_COEF",
        dims=("12",),
        description="convection-condensation energy coefficient per month",
    )
    frac_infil: Array = field(
        unit="-", fortran_name="FRAC_INFIL", dims=(), description="fraction of the snowmelt that infiltrates"
    )
    snarea_thresh: Array = field(
        unit="in",
        fortran_name="SNAREA_THRESH",
        dims=(),
        description="maximum threshold water equivalent of the snow depletion",
    )
    snarea_curve: Array = field(
        unit="-",
        fortran_name="SNAREA_CURVE",
        dims=("11",),
        description="the site's areal depletion curve (HRU_DEPLCRV), cover at FRAC = 0, 0.1, ..., 1",
    )
    latitude: Array = field(unit="rad", fortran_name="XLAT", dims=(), description="site latitude")
    coefficients: PrmsSnowCoefficients = eqx.field(default_factory=lambda: PRMS_SNOW)
    melt_look: int = field(
        unit="d",
        static=True,
        fortran_name="MELT_LOOK",
        description="day of year to look for spring melt",
        default=0,
    )
    melt_force: int = field(
        unit="d",
        static=True,
        fortran_name="MELT_FORCE",
        description="day of year that forces the spring melt stage",
        default=0,
    )
    depletion: bool = field(
        static=True,
        fortran_name="NDEPL",
        description="an areal depletion curve is read (NDEPL > 0)",
        default=True,
    )
    ipet: int = field(
        static=True,
        fortran_name="IPET",
        description=(
            "PET method of the scenario (rzwqm.dat): 0, Shuttleworth-Wallace, sets the sublimation "
            "potential ESN every day; 1 or 2, a crop-coefficient reference ET, never sets it (ESN = 0)"
        ),
        default=0,
    )

    @classmethod
    def from_sno(
        cls,
        sno: SnoFile,
        latitude_rad: float,
        *,
        ipet: int = 0,
        dtype: Any = None,
        coefficients: PrmsSnowCoefficients = PRMS_SNOW,
    ) -> PrmsSnowParams:
        """The parameters of a parsed ``.sno`` file (:func:`agrijax.processes.snow.sno.read_sno`).

        ``ipet`` is the PET method of the scenario's ``rzwqm.dat`` (item 19 of the evaporation
        record, ``Rzmain.for`` 4717-4731): the sublimation potential ``ESN`` is set only by the
        Shuttleworth-Wallace routine (``IPET = 0``, ``Rzday.for`` 1233, 1924); with a reference-ET
        method it keeps the start value 0 of the ``-init=zero`` build and the pack does not sublimate.

        Raises ``NotImplementedError`` for a cover type above 1 or a thunderstorm month (not
        reproduced, see the module docstring).
        """
        sno.check_reproduced()

        def a(x: Any) -> Array:
            return jnp.asarray(x, dtype=dtype if dtype is not None else jnp.result_type(float))

        return cls(
            den_max=a(sno.den_max),
            den_init=a(sno.den_init),
            freeh2o_cap=a(sno.freeh2o_cap),
            settle_const=a(sno.settle_const),
            tmax_allsnow=a(sno.tmax_allsnow),
            albset_rnm=a(sno.albset_rnm),
            albset_snm=a(sno.albset_snm),
            covden_win=a(sno.covden_win),
            rad_trncf=a(sno.rad_trncf),
            emis_noppt=a(sno.emis_noppt),
            potet_sublim=a(sno.potet_sublim),
            cecn_coef=a(sno.cecn_coef),
            frac_infil=a(sno.frac_infil),
            snarea_thresh=a(sno.snarea_thresh),
            snarea_curve=a(sno.depletion_curve),
            latitude=a(latitude_rad),
            coefficients=coefficients,
            melt_look=int(sno.melt_look),
            melt_force=int(sno.melt_force),
            depletion=bool(sno.ndepl > 0),
            ipet=int(ipet),
        )


def _sf(unit: str, description: str, fortran_name: str = "") -> Any:
    return field(unit=unit, dims=(), description=description, fortran_name=fortran_name)


class SnowState(State):
    """The pack of RZWQM2's PRMS routine and its saved counters (``surface.snow``), plus the port P9.

    ``swe`` is the pack water equivalent in cm (``SNOWPK``, ``SNP``); the other pack variables are
    kept in the routine's own units (inches, langleys, degC) as PRMS holds them between days.
    Flags and counters are floating 0/1/2 values.
    """

    swe: Array = _sf("cm", "pack water equivalent (SNP, .ana column 92)", "SNOWPK")
    pk_def: Array = _sf("cal cm-2", "pack cold content (heat deficit)", "PK_DEF")
    pk_temp: Array = _sf("degC", "pack temperature", "PK_TEMP")
    pk_ice: Array = _sf("in", "ice in the pack", "PK_ICE")
    freeh2o: Array = _sf("in", "free liquid water in the pack", "FREEH2O")
    pk_depth: Array = _sf("in", "pack depth", "PK_DEPTH")
    pk_den: Array = _sf("g cm-3", "pack density", "PK_DEN")
    pss: Array = _sf("in", "previous pack water equivalent plus new snow (settlement)", "PSS")
    pst: Array = _sf("in", "largest water equivalent of the season (depletion)", "PST")
    snsv: Array = _sf("in", "new snow since the albedo reset in the melt stage", "SNSV")
    albedo: Array = _sf("-", "pack albedo", "ALBEDO")
    iasw: Array = _sf("-", "flag: new snow over a partial cover (depletion curve left)", "IASW")
    iso: Array = _sf("-", "melt stage: 1 accumulation, 2 spring melt", "ISO")
    mso: Array = _sf("-", "melt-look stage: 1 before MELT_LOOK, 2 after", "MSO")
    lso: Array = _sf("d", "days of an isothermal pack since MELT_LOOK", "LSO")
    lst: Array = _sf("-", "flag: shallow new snow in the melt stage", "LST")
    slst: Array = _sf("d", "days since the last snowfall (albedo counter)", "SLST")
    intal: Array = _sf("-", "albedo table: 1 accumulation, 2 melt", "INTAL")
    scrv: Array = _sf("in", "water equivalent at which the new-snow cover starts to deplete", "SCRV")
    pksv: Array = _sf("in", "water equivalent before the last new snow", "PKSV")
    scasv: Array = _sf("-", "snow cover before the last new snow", "SNOWCOV_AREASV")
    sstart: Array = _sf("-", "flag: re-initialise the pack routine at its next call", "SSTART")
    started: Array = _sf("-", "flag: the routine has been called once (its first call sets PSS)", "README")
    month: Array = _sf("-", "month of the last call as CDATE returns it", "MO")
    cdate_day: Array = _sf("d", "day of month of the last call (CDATE's aliased year argument)", "IT")
    cover: Array = _sf("-", "today's snow-covered fraction handed to PHYSCL (0 when not called)", "FSNC")
    intercepted: Array = field(
        unit="cm d-1",
        dims=(),
        description="today's precipitation taken by the snow routine (none of the day's storms infiltrate)",
        fortran_name="RFDACC",
    )
    out: SnowOut = port(description="the day's snow outputs (P9, iface.snow)")

    @classmethod
    def initial(cls, swe_cm: Any = 0.0, dtype: Any = None) -> SnowState:
        """The state at the start of a run: the initial pack ``swe_cm`` [cm] and the reference's
        start values (``-init=zero``; ``SSTART`` and ``INTAL`` start at their DATA values)."""
        dt = dtype if dtype is not None else jnp.result_type(float)
        z = jnp.zeros((), dtype=dt)
        one = jnp.ones((), dtype=dt)
        kw = {n: z for n in PACK_FIELDS}
        kw.update(swe=jnp.asarray(swe_cm, dtype=dt), intal=one, sstart=one)
        return cls(**kw, cover=z, intercepted=z, out=SnowOut.zeros(dt))


#: the state fields the process reads (everything but its outputs)
PACK_FIELDS: tuple[str, ...] = (
    "swe",
    "pk_def",
    "pk_temp",
    "pk_ice",
    "freeh2o",
    "pk_depth",
    "pk_den",
    "pss",
    "pst",
    "snsv",
    "albedo",
    "iasw",
    "iso",
    "mso",
    "lso",
    "lst",
    "slst",
    "intal",
    "scrv",
    "pksv",
    "scasv",
    "sstart",
    "started",
    "month",
    "cdate_day",
)

#: the fields SNOINIT zeroes on a re-initialisation (Snowprms.for 160-179; ISO and MSO are set to 1)
_SNOINIT_ZERO: tuple[str, ...] = (
    "iasw",
    "lso",
    "pk_def",
    "pk_temp",
    "pk_ice",
    "freeh2o",
    "pk_depth",
    "pst",
    "pk_den",
    "albedo",
    "snsv",
    "lst",
)
#: the fields SNORUN zeroes when the day's balances empty the pack (Snowprms.for 424-436)
_EMPTY_ZERO: tuple[str, ...] = (
    "pk_depth",
    "pss",
    "snsv",
    "lst",
    "pst",
    "iasw",
    "albedo",
    "pk_den",
    "sca",
    "pk_def",
    "pk_temp",
    "pk_ice",
    "freeh2o",
)

#: the pack variables carried through the routines (PRMS units), a dict in the kernels
_K = dict[str, Array]


def _sel(mask: Array, new: _K, old: _K) -> _K:
    """``new`` where ``mask``, else ``old`` (every key of ``old``)."""
    return {k: jnp.where(mask, new[k], v) for k, v in old.items()}


def _nz(x: Array) -> Array:
    """A denominator that is never zero (1 where ``x`` is 0): both arms of a where stay finite."""
    return jnp.where(x != 0.0, x, 1.0)


def _calin(k: _K, cal: Array, mask: Array, prm: PrmsSnowParams, c: PrmsSnowCoefficients) -> _K:
    """``CALIN`` (``Snowprms.for`` 612-675): a heat gain ``cal`` [cal cm-2] warms or melts the pack.

    Source: RZWQM2 4.6 Snowprms.for CALIN; Leavesley et al. (1983) snowpack heat balance.
    """
    dif = cal - k["pk_def"]
    cold = dif < 0.0
    melt = dif > 0.0
    a_def = k["pk_def"] - cal
    a_temp = -a_def / _nz(k["pkwe"] * c.ice_heat_inch)
    pmlt = dif / c.latent_heat_inch
    apmlt = pmlt * k["sca"]
    apk_ice = k["pk_ice"] / _nz(k["sca"])
    full = melt & (pmlt > apk_ice)
    part = melt & ~(pmlt > apk_ice)
    ice_p = k["pk_ice"] - apmlt
    free_p = k["freeh2o"] + apmlt
    pwcap = prm.freeh2o_cap * ice_p
    excess = free_p - pwcap
    drain = part & (excess > 0.0)
    pkwe_p = k["pkwe"] - excess
    depth_p = pkwe_p / _nz(k["pk_den"])
    z = jnp.zeros_like(k["pkwe"])
    new = dict(k)
    new["pk_def"] = jnp.where(cold, a_def, z)
    new["pk_temp"] = jnp.where(cold, a_temp, z)
    new["snowmelt"] = k["snowmelt"] + jnp.where(full, k["pkwe"], jnp.where(drain, excess, z))
    new["pkwe"] = jnp.where(full, z, jnp.where(drain, pkwe_p, k["pkwe"]))
    new["pk_ice"] = jnp.where(full, z, jnp.where(part, ice_p, k["pk_ice"]))
    new["freeh2o"] = jnp.where(full, z, jnp.where(part, jnp.where(drain, pwcap, free_p), k["freeh2o"]))
    new["pk_depth"] = jnp.where(full, z, jnp.where(drain, depth_p, k["pk_depth"]))
    new["pss"] = jnp.where(full, z, jnp.where(drain, pkwe_p, k["pss"]))
    new.update({n: jnp.where(full, z, k[n]) for n in ("iasw", "sca", "pst", "pk_den")})  # a complete melt
    return _sel(mask, new, k)


def _caloss(k: _K, cal: Array, mask: Array, c: PrmsSnowCoefficients) -> _K:
    """``CALOSS`` (``Snowprms.for`` 572-604): a heat loss ``cal`` [cal cm-2, negative] freezes free
    water first, then adds to the cold content.

    Source: RZWQM2 4.6 Snowprms.for CALOSS; Leavesley et al. (1983) snowpack heat balance.
    """
    nofree = k["freeh2o"] <= 0.0
    dif = cal + k["freeh2o"] * c.latent_heat_inch
    all_freeze = ~nofree & (dif <= 0.0)
    part_freeze = ~nofree & (dif > 0.0)
    frozen = -cal / c.latent_heat_inch
    new = dict(k)
    pk_def = jnp.where(nofree, k["pk_def"] - cal, jnp.where(all_freeze & (dif < 0.0), -dif, k["pk_def"]))
    new["pk_def"] = pk_def
    new["pk_ice"] = jnp.where(
        all_freeze, k["pk_ice"] + k["freeh2o"], jnp.where(part_freeze, k["pk_ice"] + frozen, k["pk_ice"])
    )
    new["freeh2o"] = jnp.where(
        all_freeze, jnp.zeros_like(cal), jnp.where(part_freeze, k["freeh2o"] - frozen, k["freeh2o"])
    )
    temp = -pk_def / _nz(k["pkwe"] * c.ice_heat_inch_caloss)
    new["pk_temp"] = jnp.where(~part_freeze & (k["pkwe"] > 0.0), temp, k["pk_temp"])
    return _sel(mask, new, k)


def _ppt_to_pack(
    k: _K,
    net_rain: Array,
    net_snow: Array,
    tavgc: Array,
    tmaxf: Array,
    mask: Array,
    prm: PrmsSnowParams,
    c: PrmsSnowCoefficients,
) -> _K:
    """``PPT_TO_PACK`` without a mixed event (``Snowprms.for`` 459-565): rain on a pack first
    (warming it, freezing in the cold content, then ``CALIN``), then new snow at its temperature.

    Source: RZWQM2 4.6 Snowprms.for PPT_TO_PACK (PPTMIX = 0); Leavesley et al. (1983).
    """
    z = jnp.zeros_like(net_rain)
    train_alt = (tmaxf + prm.tmax_allsnow) * c.mean_weight - c.fahrenheit_offset
    train = jnp.maximum(z, jnp.where(tavgc <= 0.0, train_alt, tavgc))
    tsnow = jnp.minimum(z, tavgc)
    # ---- rain on an existing pack
    rain = (k["pkwe"] > 0.0) & (net_rain > 0.0)
    pkwe_r = k["pkwe"] + net_rain
    deficit = k["pk_def"] > 0.0
    caln = (c.latent_heat_fusion + train) * c.cm_per_inch
    pndz = k["pk_def"] / _nz(caln)
    eq = deficit & (net_rain == pndz)
    lt = deficit & (net_rain < pndz)
    gt = deficit & (net_rain > pndz)
    lt_def = k["pk_def"] - caln * net_rain
    lt_temp = -lt_def / _nz(pkwe_r * c.ice_heat_inch)
    base = {**k, "pkwe": pkwe_r}
    r = _sel(rain & eq, {**base, "pk_def": z, "pk_temp": z, "pk_ice": k["pk_ice"] + net_rain}, k)
    r = _sel(rain & lt, {**base, "pk_def": lt_def, "pk_temp": lt_temp, "pk_ice": k["pk_ice"] + net_rain}, r)
    r = _sel(
        rain & gt,
        {**base, "pk_def": z, "pk_temp": z, "pk_ice": k["pk_ice"] + pndz, "freeh2o": net_rain - pndz},
        r,
    )
    r = _sel(rain & ~deficit, {**base, "freeh2o": k["freeh2o"] + net_rain}, r)
    calpr = jnp.where(gt, train * (net_rain - pndz) * c.cm_per_inch, train * net_rain * c.cm_per_inch)
    r = _calin(r, calpr, rain & (gt | ~deficit), prm, c)
    # ---- new snow
    snow = net_snow > 0.0
    pkwe_s = r["pkwe"] + net_snow
    base_s = {**r, "pkwe": pkwe_s, "pk_ice": r["pk_ice"] + net_snow}
    den_s = _nz(pkwe_s * c.ice_heat_inch)
    warm_temp = -r["pk_def"] / den_s
    calps = tsnow * net_snow * c.ice_heat_inch
    cold_def = r["pk_def"] - calps
    cold_temp = -cold_def / den_s
    warm = tsnow >= 0.0
    has_free = r["freeh2o"] > 0.0
    s = _sel(snow & warm, {**base_s, "pk_temp": warm_temp}, r)
    s = _sel(snow & ~warm & ~has_free, {**base_s, "pk_def": cold_def, "pk_temp": cold_temp}, s)
    s = _sel(snow & ~warm & has_free, _caloss(base_s, calps, snow, c), s)
    return _sel(mask, s, k)


def _snowcov(
    k: _K, newsnow: Array, net_snow: Array, mask: Array, prm: PrmsSnowParams, c: PrmsSnowCoefficients
) -> _K:
    """``SNOWCOV`` (``Snowprms.for`` 951-1022): snow-covered area from the areal depletion curve.

    Source: RZWQM2 4.6 Snowprms.for SNOWCOV; Leavesley et al. (1983) snow-covered area depletion.
    """
    curve = prm.snarea_curve
    npts = curve.shape[-1]
    full_cover = curve[..., npts - 1]
    pkwe = k["pkwe"]
    pst = jnp.where(pkwe > k["pst"], pkwe, k["pst"])
    ai = jnp.where(pst >= prm.snarea_thresh, prm.snarea_thresh, pst)
    covered = pkwe >= ai
    no_new = newsnow == 0.0
    on = k["iasw"] != 0.0
    above = pkwe > k["scrv"]
    back = pkwe >= k["pksv"]
    pcty = (pkwe - k["pksv"]) / _nz(k["scrv"] - k["pksv"])
    sca_back = k["scasv"] + pcty * (full_cover - k["scasv"])
    frac = pkwe / _nz(ai)
    idx = jnp.trunc(jnp.minimum(c.curve_points * (frac + c.curve_index_offset), c.curve_last_point))
    jdx = jnp.clip(idx - 1.0, 1.0, float(npts - 1))
    dify = frac * c.curve_points - (jdx - 1.0)
    j0 = jdx.astype(jnp.int32) - 1
    cj = jnp.take(curve, j0, axis=-1)
    ci = jnp.take(curve, j0 + 1, axis=-1)
    sca_interp = cj + dify * (ci - cj)
    partial = ~covered
    old_on = partial & no_new & on
    interp = partial & no_new & ~(on & (above | back))
    first_new = partial & ~no_new & ~(k["iasw"] > 0.0)
    scrv_new = pkwe - c.new_snow_cover_fraction * net_snow
    new = dict(k)
    new["pst"] = pst
    new["sca"] = jnp.where(interp, sca_interp, jnp.where(old_on & ~above & back, sca_back, full_cover))
    new["iasw"] = jnp.where(
        covered | (old_on & ~above & ~back),
        jnp.zeros_like(pkwe),
        jnp.where(first_new, jnp.ones_like(pkwe), k["iasw"]),
    )
    new["scrv"] = jnp.where(partial & ~no_new, scrv_new, k["scrv"])
    new["pksv"] = jnp.where(first_new, pkwe - net_snow, k["pksv"])
    new["scasv"] = jnp.where(first_new, full_cover, k["scasv"])
    return _sel(mask, new, k)


def _snalbedo(
    k: _K,
    newsnow: Array,
    net_snow: Array,
    prmx: Array,
    mask: Array,
    prm: PrmsSnowParams,
    c: PrmsSnowCoefficients,
) -> _K:
    """``SNALBEDO`` (``Snowprms.for`` 682-788): albedo from the days since the last snowfall.

    The shallow-snow reset ``SLST = SALB - 3`` (line 700) takes ``SALB`` = the soil albedo ``AS``
    that ``SNOWCOMP_PRMS`` passes (line 66), a fraction, so it is always below the floor 1 of
    line 701 and the cap of 703 never applies: the reset is to 1 day. The set-back of line 745
    belongs to a mixed rain-snow event, which ``SNOWCOMP_PRMS`` never produces.

    Source: RZWQM2 4.6 Snowprms.for SNALBEDO; Leavesley et al. (1983) snow albedo decay.
    """
    z = jnp.zeros_like(net_snow)
    one = jnp.ones_like(net_snow)
    no_new = newsnow == 0.0
    reset_shallow = no_new & (k["lst"] > 0.0)
    melt_stage = k["mso"] == _MELT
    b = ~no_new & melt_stage & (prmx < prm.albset_rnm)
    snsv_b = k["snsv"] + net_snow
    keep = ~(net_snow > prm.albset_snm) & ~(snsv_b > prm.albset_snm)
    acc = ~no_new & ~melt_stage
    slst = jnp.where(reset_shallow, one, jnp.where(b | acc, z, k["slst"]))
    lst = jnp.where(reset_shallow | acc, z, jnp.where(b, jnp.where(keep, one, z), k["lst"]))
    snsv = jnp.where(reset_shallow | acc, z, jnp.where(b, jnp.where(keep, snsv_b, z), k["snsv"]))
    rows = c.albedo_table_rows
    days = jnp.trunc(slst + c.albedo_days_round)
    acum = jnp.stack([getattr(c.accumulation, n) * one for n in ALBEDO_DAYS], axis=-1)
    amlt = jnp.stack([getattr(c.melt, n) * one for n in ALBEDO_DAYS], axis=-1)
    in_acum = days <= rows
    melt_row = jnp.where(k["intal"] != _MELT, jnp.where(in_acum, days, days - c.albedo_melt_offset), days)
    melt_row = jnp.minimum(melt_row, float(rows))
    i_acum = jnp.clip(days, 1.0, float(rows)).astype(jnp.int32) - 1
    i_melt = jnp.clip(melt_row, 1.0, float(rows)).astype(jnp.int32) - 1
    old_alb = jnp.where(
        (k["intal"] != _MELT) & in_acum, jnp.take(acum, i_acum, axis=-1), jnp.take(amlt, i_melt, axis=-1)
    )
    old = days > 0.0
    new_alb = jnp.where(melt_stage, c.albedo_new_snow_melt, c.albedo_new_snow_accumulation)
    new = dict(k)
    new["slst"] = slst + one
    new["lst"] = lst
    new["snsv"] = snsv
    new["albedo"] = jnp.where(old, old_alb, new_alb)
    new["intal"] = jnp.where(old, k["intal"], jnp.where(melt_stage, _MELT, _ACCUM))
    return _sel(mask, new, k)


def _snowbal(
    k: _K,
    temp: Array,
    sw: Array,
    emis: Array,
    basin_ppt: Array,
    cst: Array,
    cec: Array,
    mask: Array,
    prm: PrmsSnowParams,
    c: PrmsSnowCoefficients,
) -> _K:
    """``SNOWBAL`` (``Snowprms.for`` 795-891): the half-day energy balance of the pack.

    Source: RZWQM2 4.6 Snowprms.for SNOWBAL (thunderstorm months not reproduced); Leavesley et
    al. (1983) snowpack energy balance.
    """
    z = jnp.zeros_like(temp)
    air = c.stefan_boltzmann_half_day * (temp + c.kelvin_offset) ** 4
    below = temp < 0.0
    ts = jnp.where(below, temp, z)
    sno = jnp.where(below, air, c.snow_longwave_melting)
    sky = (1.0 - prm.covden_win) * (emis * air - sno)
    can = prm.covden_win * (air - sno)
    cecsub = jnp.where((temp > 0.0) & (basin_ppt > 0.0), cec * temp, z)
    cal = sky + can + cecsub + sw
    gain = (ts >= 0.0) & (cal > 0.0)
    pk_temp = k["pk_temp"]
    qcond = cst * (ts - pk_temp)
    neg = ~gain & (qcond < 0.0)
    pos = ~gain & (qcond > 0.0)
    to_calin = gain | (~gain & (qcond == 0.0) & (pk_temp >= 0.0) & (cal > 0.0))
    den = _nz(k["pkwe"] * c.ice_heat_inch)
    r = _calin(k, cal, to_calin, prm, c)
    neg_def = k["pk_def"] - qcond
    r = _sel(neg & (pk_temp < 0.0), {**k, "pk_def": neg_def, "pk_temp": -neg_def / den}, r)
    r = _sel(neg & (pk_temp >= 0.0), _caloss(k, qcond, neg, c), r)
    sub = k["pk_def"] - qcond
    warm_temp = -sub / den
    pkt = -ts * k["pkwe"] * c.ice_heat_inch
    sub2 = k["pk_def"] - pkt - qcond
    cold_def = jnp.where(sub2 < 0.0, pkt, sub2 + pkt)
    cold_temp = -cold_def / den
    warm_air = ts >= 0.0
    r = _sel(
        pos & warm_air,
        {**k, "pk_def": jnp.where(sub < 0.0, z, sub), "pk_temp": jnp.where(sub < 0.0, z, warm_temp)},
        r,
    )
    r = _sel(
        pos & ~warm_air,
        {**k, "pk_def": cold_def, "pk_temp": jnp.where(sub2 < 0.0, ts, cold_temp)},
        r,
    )
    return _sel(mask, r, k)


def _snowevap(k: _K, potet: Array, mask: Array, prm: PrmsSnowParams, c: PrmsSnowCoefficients) -> _K:
    """``SNOWEVAP`` for cover types 0 and 1 (``Snowprms.for`` 899-943): sublimation from the pack.

    Source: RZWQM2 4.6 Snowprms.for SNOWEVAP; Leavesley et al. (1983) snowpack sublimation.
    """
    ez = prm.potet_sublim * potet * k["sca"]
    z = jnp.zeros_like(ez)
    some = ez > 0.0
    all_gone = some & (ez >= k["pkwe"])
    part = some & ~(ez >= k["pkwe"])
    new = dict(k)
    new["snow_evap"] = jnp.where(all_gone, k["pkwe"], jnp.where(part, ez, z))
    new["pkwe"] = jnp.where(all_gone, z, jnp.where(part, k["pkwe"] - ez, k["pkwe"]))
    new["pk_ice"] = jnp.where(all_gone, z, jnp.where(part, k["pk_ice"] - ez, k["pk_ice"]))
    new["pk_def"] = jnp.where(
        all_gone, z, jnp.where(part, k["pk_def"] + k["pk_temp"] * ez * c.ice_heat_inch, k["pk_def"])
    )
    new["freeh2o"] = jnp.where(all_gone, z, k["freeh2o"])
    new["pk_temp"] = jnp.where(all_gone, z, k["pk_temp"])
    return _sel(mask, new, k)


def _cdate_month(doy: Array, k: _K) -> tuple[Array, Array]:
    """``CALL CDATE(JDAY, IT, MO, IT)`` of ``SNOWCOMP_PRMS`` (``Snowprms.for`` 60; ``Rzman.for``
    3-59): the month and day of month of ``JDAY``, with the saved ``IT`` (the previous call's day
    of month) as the year that decides the leap-year shift. A day past the table (366 in a
    non-leap reading) leaves both unchanged.

    Source: RZWQM2 4.6 Rzman.for CDATE (calendar), Snowprms.for 60 (its call).
    """
    jday = jnp.trunc(doy)
    it = jnp.trunc(k["cdate_day"])
    leap = jnp.floor_divide(it, _LEAP_DIVISOR) * _LEAP_DIVISOR == it
    feb29 = leap & (jday == _LEAP_DOY)
    iday = jnp.where(leap & (jday > _LEAP_DOY), jday - 1.0, jday)
    kda = jnp.asarray(_KDA, dtype=doy.dtype)
    ends = kda[1:]
    month = jnp.sum(iday > ends[:-1]) + 1.0
    found = iday <= ends[-1]
    dom = iday - jnp.take(kda, month.astype(jnp.int32) - 1)
    new_month = jnp.where(feb29, _LEAP_MONTH, jnp.where(found, month, k["month"]))
    new_day = jnp.where(feb29, _LEAP_DOM, jnp.where(found, dom, k["cdate_day"]))
    return new_month.astype(doy.dtype), new_day.astype(doy.dtype)


def _snowcomp(
    k: _K,
    rfd: Array,
    f: SnowForcing,
    prm: PrmsSnowParams,
    c: PrmsSnowCoefficients,
    depletion: bool,
    esn_set: bool,
) -> _K:
    """``SNOWCOMP_PRMS`` and ``SNORUN`` (``Snowprms.for`` 3-143, 190-451) for one day's call.

    ``k`` holds the pack in PRMS units with ``pkwe`` the water equivalent [in]; ``rfd`` is the
    precipitation handed to the routine [cm]. Returns the pack after the day and ``snowmelt``,
    ``snow_evap`` [in] and ``sca`` (the returned cover; 0 when the pack routine did nothing).

    Source: RZWQM2 4.6 Snowprms.for SNOWCOMP_PRMS, SNOINIT, SNORUN; Leavesley et al. (1983).
    """
    z = jnp.zeros_like(rfd)
    one = jnp.ones_like(rfd)
    doy = jnp.asarray(f.doy, dtype=rfd.dtype)
    # ---- first call ever (README): PSS starts at the initial pack (lines 46-50)
    k = {**k, "pss": jnp.where(k["started"] > 0.0, k["pss"], k["pkwe"])}
    # ---- SNOINIT on SSTART (lines 52-57, 151-182); SNORUN's first-day stage reset (313-318)
    init = {**k, **dict.fromkeys(_SNOINIT_ZERO, z), "iso": one, "mso": one}
    k = _sel(k["sstart"] > 0.0, init, k)
    month, cdate_day = _cdate_month(doy, k)
    k = {**k, "month": month, "cdate_day": cdate_day}
    # ---- inputs in PRMS units (lines 62-109)
    swrad = f.srad * c.langley_per_mj_m2
    tavgc = (f.tmax + f.tmin) * c.mean_weight
    tmaxf = f.tmax / c.celsius_per_fahrenheit + c.fahrenheit_offset
    tminf = f.tmin / c.celsius_per_fahrenheit + c.fahrenheit_offset
    # ESN: set every day by the Shuttleworth-Wallace PET (IPET = 0), else its start value 0
    potet = jnp.where(esn_set, c.sublimation_potential, 0.0) * c.inch_per_cm * one
    wet = rfd > 0.0
    ppt = rfd * c.inch_per_cm
    basin_ppt = jnp.where(wet, ppt, z)
    is_snow = wet & (tavgc <= 0.0)
    is_rain = wet & ~(tavgc <= 0.0)
    net_snow = jnp.where(is_snow, ppt, z)
    net_rain = jnp.where(is_rain, ppt, z)
    net_ppt = basin_ppt
    prmx = jnp.where(is_rain, one, z)
    newsnow = jnp.where(is_snow, one, z)
    # ---- SNORUN (lines 298-451)
    deninv = 1.0 / prm.den_init
    setden = prm.settle_const / prm.den_max
    set1 = 1.0 / (1.0 + prm.settle_const)
    k = {**k, "snowmelt": z, "snow_evap": z, "sca": one}
    k["iso"] = jnp.where(doy == prm.melt_force, _MELT, k["iso"])
    k["mso"] = jnp.where(doy == prm.melt_look, _MELT, k["mso"])
    active = (k["pkwe"] > 0.0) | (newsnow != 0.0)
    to_pack = active & (((k["pkwe"] > 0.0) & (net_ppt > 0.0)) | (net_snow > 0.0))
    k = _ppt_to_pack(k, net_rain, net_snow, tavgc, tmaxf, to_pack, prm, c)
    pack = active & (k["pkwe"] > 0.0)
    if depletion:  # static: an areal depletion curve is read (NDEPL > 0)
        k = _snowcov(k, newsnow, net_snow, pack, prm, c)
    k = _snalbedo(k, newsnow, net_snow, prmx, pack, prm, c)
    tminc = (tminf - c.fahrenheit_offset) * c.celsius_per_fahrenheit
    tmaxc = (tmaxf - c.fahrenheit_offset) * c.celsius_per_fahrenheit
    emis = jnp.where(basin_ppt > 0.0, one, prm.emis_noppt * one)
    swn = swrad * (1.0 - k["albedo"]) * prm.rad_trncf
    i_month = jnp.clip(k["month"], 1.0, float(prm.cecn_coef.shape[-1])).astype(jnp.int32) - 1
    cec = jnp.take(prm.cecn_coef, i_month, axis=-1) * c.cec_half_day
    # settlement and density (lines 362-367)
    pss = k["pss"] + net_snow
    dpt1 = (net_snow * deninv + setden * pss + k["pk_depth"]) * set1
    pk_den = k["pkwe"] / _nz(dpt1)
    effk = c.conductivity_per_density * pk_den
    cst = pk_den * jnp.sqrt(jnp.maximum(effk * c.half_day_seconds_over_pi, _SQRT_FLOOR))
    k = _sel(pack, {**k, "pss": pss, "pk_depth": dpt1, "pk_den": pk_den}, k)
    # forced spring melt (lines 370-382)
    looking = pack & (k["iso"] == _ACCUM) & (k["mso"] == _MELT)
    iso_warm = k["pk_temp"] >= 0.0
    lso1 = k["lso"] + 1.0
    start_melt = looking & iso_warm & (lso1 <= c.forced_melt_days)
    k["lso"] = jnp.where(looking, jnp.where(iso_warm & ~(lso1 <= c.forced_melt_days), lso1, z), k["lso"])
    k["iso"] = jnp.where(start_melt, _MELT, k["iso"])
    # night and day half-day balances (lines 385-404)
    k = _snowbal(k, (tminc + tavgc) * c.mean_weight, z, emis, basin_ppt, cst, cec, pack, prm, c)
    day = pack & (k["pkwe"] > 0.0)
    k = _snowbal(k, (tmaxc + tavgc) * c.mean_weight, swn, emis, basin_ppt, cst, cec, day, prm, c)
    evap = pack & (k["pkwe"] > 0.0)
    k = _snowevap(k, potet, evap, prm, c)
    # end of day (lines 416-437)
    left = pack & (k["pkwe"] > 0.0)
    gone = pack & ~(k["pkwe"] > 0.0)
    snsv = k["snsv"] - k["snowmelt"]
    kept = {
        **k,
        "pk_depth": k["pkwe"] / _nz(k["pk_den"]),
        "pss": k["pkwe"],
        "snsv": jnp.where(k["lst"] > 0.0, jnp.where(snsv <= 0.0, z, snsv), k["snsv"]),
    }
    k = _sel(left, kept, k)
    cleared = {**k, **dict.fromkeys(_EMPTY_ZERO, z)}
    k = _sel(gone, cleared, k)
    # the routine's returned cover and melt (lines 439-448): nothing when it did not run
    k["sca"] = jnp.where(active, k["sca"], z)
    k["snowmelt"] = jnp.where(active, k["snowmelt"], z)
    k["snow_evap"] = jnp.where(active, k["snow_evap"], z)
    return k


@process(
    reads=PACK_FIELDS,
    writes=(*PACK_FIELDS, "cover", "intercepted", "out"),
    source="PRMS snowpack (Leavesley et al. 1983) as RZWQM2 4.6 SNOWCOMP_PRMS runs it (ISHAW = 0)",
    fortran_name="SNOWCOMP_PRMS",
    key="snow/prms@rzwqm2-4.6:faithful",
    provenance="reference_only_conventions",
    grid="point",
    ref_build="RZWQM2 4.6 main_ryzen5_avx512",
    sources=(
        Source(
            "daily snow branch, storms to the pack, melt split AIRR/SNRO",
            "RZWQM2 4.6 Rzday.for 869-884, 1529-1592",
        ),
        Source(
            "interface: units, rain/snow split, returned SNP, SMELT, FSNC", "RZWQM2 4.6 Snowprms.for 3-143"
        ),
        Source(
            "pack driver: settlement, forced melt, balances, sublimation",
            "Snowprms.for 190-451; Leavesley et al. (1983)",
        ),
        Source("rain and snow into the pack", "Snowprms.for 459-565 (PPT_TO_PACK); Leavesley et al. (1983)"),
        Source(
            "heat gain and loss of the pack", "Snowprms.for 572-675 (CALOSS, CALIN); Leavesley et al. (1983)"
        ),
        Source("albedo decay", "Snowprms.for 682-788 (SNALBEDO); Leavesley et al. (1983)"),
        Source("half-day energy balance", "Snowprms.for 795-891 (SNOWBAL); Leavesley et al. (1983)"),
        Source("sublimation, ESN = 0.011 cm/d", "Snowprms.for 899-943 (SNOWEVAP); Rzpet.for 1423-1425"),
        Source("areal depletion curve", "Snowprms.for 951-1022 (SNOWCOV); Leavesley et al. (1983)"),
        Source("month of the call", "Snowprms.for 60; Rzman.for 3-59 (CDATE)"),
    ),
    deviates=(
        Deviation(
            "not reproduced: cover types above 1 and thunderstorm months",
            "they need the day's potential transpiration and clear-sky radiation from the PET routine; "
            "PrmsSnowParams.from_sno raises on them",
            "all 16 .sno files of RZWQM_sw_batch have COV_TYPE = 1 and TSTORM_MO = 0 (one md5)",
        ),
        Deviation(
            "an empty-density pack (PK_DEN = 0 with water in it) divides by 1, not by 0",
            "guarded denominators keep both arms of every where finite (the reference would carry Inf); "
            "it needs a pack when the routine re-initialises (SSTART) or a zero first density",
            "no such day in the CA-TPA 2015-2023 run (tests/integration/test_snow_prms_reference.py)",
        ),
    ),
)
def snow_prms(state: SnowState, params: PrmsSnowParams, forcing_t: SnowForcing) -> SnowState:
    """One day of the RZWQM2 4.6 PRMS snowpack: pack, melt split and the P9 record.

    Reads the forcing fields ``tmin``, ``tmax``, ``srad``, ``precipitation`` and ``doy``.

    Source: RZWQM2 4.6 Rzday.for 869-884, 1529-1592 (PHYSCL) and Snowprms.for (SNOWCOMP_PRMS);
    Leavesley et al. (1983), USGS WRI 83-4238.
    """
    c = params.coefficients
    f = forcing_t
    z = jnp.zeros_like(state.swe)
    one = jnp.ones_like(state.swe)
    lat = params.latitude
    doy = jnp.asarray(f.doy, dtype=state.swe.dtype)
    reset = ((doy == c.reset_doy_north) & (lat > 0.0)) | ((doy == c.reset_doy_south) & (lat < 0.0))
    sstart = jnp.where(reset, one, state.sstart)
    tm = (f.tmin + f.tmax) * c.mean_weight
    branch = (state.swe > 0.0) | (tm <= 0.0)
    rfd = jnp.where(branch, f.precipitation, z)
    called = branch & ((state.swe > 0.0) | (rfd > 0.0))
    k = {n: getattr(state, n) for n in PACK_FIELDS if n != "swe"}
    k["sstart"] = sstart
    k["pkwe"] = state.swe * c.inch_per_cm
    new = _snowcomp(k, rfd, f, params, c, params.depletion, params.ipet == 0)
    swe = new["pkwe"] / c.inch_per_cm
    smelt = new["snowmelt"] / c.inch_per_cm
    melt = smelt * params.frac_infil
    upd: dict[str, Array] = {
        n: jnp.where(called, new[n], k[n]) for n in PACK_FIELDS if n not in ("swe", "sstart", "started")
    }
    upd["swe"] = jnp.where(called, swe, state.swe)
    upd["sstart"] = jnp.where(called, z, sstart)
    upd["started"] = jnp.where(called, one, state.started)
    sublimation = new["snow_evap"] / c.inch_per_cm
    out = SnowOut(
        melt=jnp.where(called, melt, z),
        melt_runoff=jnp.where(called, smelt - melt, z),
        swe=upd["swe"] * MM_PER_CM,
        sublimation=jnp.where(called, sublimation, z),
    )
    return state.replace(**upd, cover=jnp.where(called, new["sca"], z), intercepted=rfd, out=out)
