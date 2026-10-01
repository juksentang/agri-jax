"""The numbers the DSSAT-CSM v4.8.6.0 vertical water balance hard-codes (``WATBAL`` and callees).

Every coefficient of the tipping bucket is declared here once, with its unit, meaning and the
DSSAT-CSM v4.8.6.0 file, line and statement it comes from (DSSAT-CSM is BSD-3, so the statement
is quoted; ``tests/integration/test_bucket_coefficients_source.py`` reads each line of the
reference source and checks the quote). :data:`WATBAL_COEFFICIENTS` holds the reference values.

Calibratable by default are the response coefficients (the SCS abstraction, the upward-flow
diffusivity, the drainage of the top layer, the snow melt, the mulch interception); the switches
that only reproduce the reference's arithmetic (thresholds that skip a computation, the rounding of
the water content, the truncation of tiny stocks) are leaves kept out of the calibration vector
(``calibrate=False``), and the decimal count of the water-content rounding is static.

Source: DSSAT-CSM v4.8.6.0 ``Soil/SoilWater/{WATBAL,RNOFF,INFIL,SATFLO,WBSUBS}.for`` and
``Soil/Mulch/MULCHWAT.for`` (BSD-3, Copyright 1998-2026 DSSAT Foundation, University of Florida,
International Fertilizer Development Center); Ritchie J. T. (1998) Soil water balance and plant
water stress, in Tsuji et al. (eds) *Understanding Options for Agricultural Production*, Kluwer,
41-54; USDA-SCS (1972) National Engineering Handbook, section 4 (curve number).
"""

from __future__ import annotations

from typing import Any

from agrijax.core.coefficients import Coefficients, Provenance, coef

__all__ = ["WATBAL_COEFFICIENTS", "WatbalCoefficients"]

_REF = "dssat-4.8.6.0"
_RITCHIE = "Ritchie (1998)"
_SW = "Soil/SoilWater/"


def _c(value: float, unit: str, description: str, where: str, routine: str, statement: str, **kw: Any) -> Any:
    """A coefficient of DSSAT-CSM v4.8.6.0 ``<where>`` (``file:line``), routine ``routine``."""
    prov = Provenance.at(_REF, where, routine=routine, statement=statement, paper=kw.pop("paper", _RITCHIE))
    return coef(value, unit, description, prov, **kw)


class WatbalCoefficients(Coefficients):
    """The numbers of DSSAT ``WATBAL``, ``RNOFF``, ``INFIL``, ``SATFLO``, ``UP_FLOW``, ``SNOWFALL``
    and ``MULCHWATER`` (Ritchie's multi-layer tipping bucket with SCS runoff)."""

    # ------------------------------------------------------------------ SNOWFALL (WBSUBS.for)
    snow_tmax: float = _c(
        1.0,
        "degC",
        "maximum temperature at or below which precipitation accumulates as snow (and above which "
        "the snow pack melts)",
        _SW + "WATBAL.for:282",
        "WATBAL",
        "IF (TMAX .LE. 1.0 .OR. SNOW .GT. 0.0) THEN",
    )
    melt_rain: float = _c(
        0.4,
        "-",
        "snow melt per mm of rain on a melt day (SNOMLT = TMAX + 0.4 RAIN, degree-day of 1 mm degC-1)",
        _SW + "WBSUBS.for:48",
        "SNOWFALL",
        "SNOMLT = TMAX + RAIN*0.4",
    )
    melt_per_degc: float = _c(
        1.0,
        "mm degC-1",
        "snow melt per degree of maximum temperature (the TMAX term of SNOMLT = TMAX + 0.4 RAIN)",
        _SW + "WBSUBS.for:48",
        "SNOWFALL",
        "SNOMLT = TMAX + RAIN*0.4",
    )
    snow_min: float = _c(
        0.001,
        "mm",
        "snow pack below which the pack is set to zero",
        _SW + "WBSUBS.for:60",
        "SNOWFALL",
        "if(snow.lt.0.001) snow = 0",
        calibrate=False,
    )
    # ------------------------------------------------------------------ MULCHWATER (MULCHWAT.for)
    mulch_sat_per_mass: float = _c(
        1.0e-4,
        "ha kg-1",
        "mulch water holding capacity per unit residue mass and unit MUL_WATFAC (MULCHSAT = WATFAC "
        "1e-4 MULCHMASS, mm)",
        "Soil/Mulch/MULCHWAT.for:106",
        "MULCHWATER",
        "MULCHSAT = WATFAC * 1.E-4 * MULCHMASS  !mm H2O",
    )
    new_mulch_wet_fraction: float = _c(
        0.5,
        "-",
        "fraction of saturation of newly added residue (RESWATADD = 0.5 WATFAC 1e-4 NEWMULCH)",
        "Soil/Mulch/MULCHWAT.for:110",
        "MULCHWATER",
        "RESWATADD = 0.5 * WATFAC * 1.E-4 * MULCH % NEWMULCH",
    )
    new_mulch_min: float = _c(
        1.0e-6,
        "kg ha-1",
        "new residue mass above which it brings water into the mulch",
        "Soil/Mulch/MULCHWAT.for:109",
        "MULCHWATER",
        "IF (MULCH % NEWMULCH > 1.E-6) THEN",
        calibrate=False,
    )
    mulch_mass_min: float = _c(
        0.01,
        "kg ha-1",
        "residue mass above which the mulch intercepts rain",
        "Soil/Mulch/MULCHWAT.for:119",
        "MULCHWATER",
        "IF (MULCHMASS .GT. 0.01 .AND. WATAVL .GT. 0.0) THEN",
        calibrate=False,
    )
    mulch_evap_fraction: float = _c(
        0.85,
        "-",
        "fraction of the mulch water expected to evaporate on the day (anticipated evaporation "
        "MIN(0.85 MULCHWAT, MULCHEVAP) added to the interception deficit)",
        "Soil/Mulch/MULCHWAT.for:129",
        "MULCHWATER",
        "&                    + MIN(MULCHWAT*0.85, MULCHEVAP)",
    )
    mulch_wat_min: float = _c(
        1.0e-4,
        "mm",
        "mulch water below which it is set to zero at the integration",
        "Soil/Mulch/MULCHWAT.for:164",
        "MULCHWATER",
        "IF (MULCHWAT < 1.E-4) MULCHWAT = 0.",
        calibrate=False,
    )
    # ------------------------------------------------------------------ RNOFF (RNOFF.for)
    scs_smx_scale: float = _c(
        254.0,
        "mm",
        "soil retention scale of the curve number (SMX = 254 (100 / CN - 1), mm)",
        _SW + "RNOFF.for:78",
        "RNOFF",
        "SMX = 254.0 * (100.0/CN - 1.0)",
        paper="USDA-SCS (1972)",
    )
    scs_cn_max: float = _c(
        100.0,
        "-",
        "curve number of an impervious surface (SMX = 254 (100 / CN - 1))",
        _SW + "RNOFF.for:78",
        "RNOFF",
        "SMX = 254.0 * (100.0/CN - 1.0)",
        paper="USDA-SCS (1972)",
        calibrate=False,
    )
    swabi_scale: float = _c(
        0.15,
        "-",
        "scale of the initial-abstraction ratio from the top two layers' air-filled fractions",
        _SW + "RNOFF.for:84",
        "RNOFF",
        "SWABI = 0.15 * ((SAT(1) - SW(1)) / (SAT(1) - LL(1) * 0.5) +",
    )
    swabi_ll_fraction: float = _c(
        0.5,
        "-",
        "fraction of the lower limit in the denominator SAT - 0.5 LL of the abstraction index",
        _SW + "RNOFF.for:84",
        "RNOFF",
        "SWABI = 0.15 * ((SAT(1) - SW(1)) / (SAT(1) - LL(1) * 0.5) +",
    )
    mulch_iabs_max: float = _c(
        0.6,
        "-",
        "initial-abstraction ratio under full mulch cover",
        _SW + "RNOFF.for:41",
        "RNOFF",
        "REAL, PARAMETER :: MAXIABS = 0.6",
        fortran_name="MAXIABS",
    )
    runoff_watavl_min: float = _c(
        0.001,
        "mm",
        "available water above which runoff is computed",
        _SW + "RNOFF.for:103",
        "RNOFF",
        "IF (WATAVL .GT. 0.001) THEN",
        calibrate=False,
    )
    plastic_fraction_min: float = _c(
        1.0e-6,
        "-",
        "plastic-mulch cover above which its runoff share is applied",
        _SW + "RNOFF.for:114",
        "RNOFF",
        "IF (PMFRACTION .GT. 1.E-6) THEN",
        calibrate=False,
    )
    # ------------------------------------------------------------------ WATBAL / INFIL / SATFLO
    pinf_min: float = _c(
        0.0001,
        "cm",
        "water available for infiltration above which INFIL (not SATFLO) runs",
        _SW + "WATBAL.for:380",
        "WATBAL",
        "IF (PINF .GT. 0.0001) THEN",
        calibrate=False,
    )
    infil_pinf_min: float = _c(
        1.0e-4,
        "cm",
        "infiltration input to a layer above which the layer may saturate and drain (INFIL)",
        _SW + "INFIL.for:55",
        "INFIL",
        "IF (PINF .GT. 1.E-4 .AND. PINF .GT. HOLD) THEN",
        calibrate=False,
    )
    top_drain_factor: float = _c(
        0.9,
        "-",
        "reduction of the drainage coefficient SWCON in the top layer on infiltration days",
        _SW + "INFIL.for:65",
        "INFIL",
        "DRCM = 0.9 * SWCON * (SAT(L) - DUL(L)) * DLAYR(L)     !JTR",
    )
    dul_margin: float = _c(
        0.003,
        "cm3 cm-3",
        "excess over the drained upper limit at or above which a layer drains",
        _SW + "SATFLO.for:58",
        "SATFLO",
        "IF (SWTEMP(L) .GE. (DUL(L) + 0.003)) THEN",
        calibrate=False,
    )
    excess_min: float = _c(
        0.0001,
        "cm",
        "excess water below which INFIL stops redistributing it into the layers above (the rest is "
        "not accounted for)",
        _SW + "INFIL.for:94",
        "INFIL",
        "IF (TMPEXCS .LT. 0.0001) GOTO 760",
        calibrate=False,
    )
    # ------------------------------------------------------------------ UP_FLOW (WBSUBS.for)
    upflow_top_min_dlayr: float = _c(
        5.0,
        "cm",
        "top-layer thickness from which the upward flow starts at layer 1 (else at layer 2)",
        _SW + "WBSUBS.for:310",
        "UP_FLOW",
        "IF (DLAYR(1) .GE. 5.0) THEN",
        calibrate=False,
    )
    upflow_dbar0: float = _c(
        0.88,
        "cm2 d-1",
        "diffusivity of the unsaturated flow at zero extractable water (DBAR = 0.88 exp(35.4 THETA))",
        _SW + "WBSUBS.for:333",
        "UP_FLOW",
        "DBAR = 0.88 * EXP(35.4*((THET1*DLAYR(L) + THET2*DLAYR(M))/",
    )
    upflow_dbar_slope: float = _c(
        35.4,
        "-",
        "exponential increase of the diffusivity with the mean extractable water",
        _SW + "WBSUBS.for:333",
        "UP_FLOW",
        "DBAR = 0.88 * EXP(35.4*((THET1*DLAYR(L) + THET2*DLAYR(M))/",
    )
    upflow_mean_factor: float = _c(
        0.5,
        "-",
        "factor of the two-layer mean in the diffusivity and of the distance between layer centres",
        _SW + "WBSUBS.for:341",
        "UP_FLOW",
        "UPFLOW(L) = DBAR * GRAD / ((DLAYR(L) + DLAYR(M)) * 0.5)",
        calibrate=False,
    )
    upflow_dbar_max: float = _c(
        100.0,
        "cm2 d-1",
        "upper limit of the diffusivity",
        _SW + "WBSUBS.for:335",
        "UP_FLOW",
        "DBAR = MIN(DBAR, 100.0)",
    )
    # ------------------------------------------------------------------ integration (WATBAL.for)
    sw_decimals: int = _c(
        6,
        "-",
        "decimals to which the integrated water content is rounded (ANINT(SW 1e6) / 1e6)",
        _SW + "WATBAL.for:503",
        "WATBAL",
        "NewSW = ANINT(SW(L) * 1.e6)/ 1.e6",
        static=True,
    )
    sw_zero: float = _c(
        1.0e-4,
        "cm3 cm-3",
        "integrated water content below which it is set to zero",
        _SW + "WATBAL.for:504",
        "WATBAL",
        "IF (abs(NewSW) < 1.e-4) NewSW = 0.0",
        calibrate=False,
    )


#: the DSSAT-CSM v4.8.6.0 values of :class:`WatbalCoefficients`
WATBAL_COEFFICIENTS = WatbalCoefficients()
