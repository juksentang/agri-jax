"""The coefficients CERES-Maize hard-codes in its Fortran, as named, documented parameters.

DSSAT-CSM reads the cultivar (``*.CUL``), ecotype (``*.ECO``) and species (``MZCER048.SPE``)
coefficients from files (:class:`~.state.CeresCultivar`, :class:`~.state.CeresSpecies`); every
other number of the maize equations is a literal in ``MZ_GROSUB``, ``MZ_PHENOL`` or
``MZ_ROOTGR``. Those literals are the fields below, grouped by routine, each with its unit, its
meaning and where it stands in DSSAT-CSM v4.8.6.0: ``source`` is ``<file>:<line>`` and
``fortran`` the Fortran statement (whitespace aside) that holds the value, so that
``tests/integration/test_ceres_coefficients_source.py`` can check every default against the
reference source.

Defaults are Python floats, so a run with :data:`DSSAT_COEFFICIENTS` traces exactly the literals
the equations had before they were named (same values, same operation order). To calibrate or
differentiate a coefficient, put a :class:`CeresCoefficients` with array leaves in
``CeresMaizeParams.coefficients`` (``None``, the default, means :data:`DSSAT_COEFFICIENTS`), e.g.
``params.replace(coefficients=CeresCoefficients().as_arrays())``. Integer thresholds (day counts,
leaf numbers) are static fields: they are not pytree leaves, and changing one retraces.

Not hoisted, on purpose: unit conversions (``0.1`` mm -> cm, ``0.0001`` cm2 -> m2, ``10``
g m-2 -> kg ha-1, ``0.001`` mg -> g, ``1000`` g -> mg, ``0.01`` cm -> m), the ``1e-3``
quantisation of ``TURFAC`` and ``RLV`` (a print / storage precision, not a response), stage codes,
crop-status codes, the 24 hours of the hourly thermal-time sum, integer powers (``x**2``,
``x**3``: the base can be negative, so a real exponent would not be finite), the ``XSTAGE`` scale
(``1.5 + 3 SUMDTT / P3`` etc.: only the nitrogen module reads it) and the numerical guards
(``1e-6``, ``1e-12``) the port adds for finite gradients.

DSSAT-CSM is distributed under the BSD 3-clause licence (Copyright 1998-2026 DSSAT Foundation,
University of Florida, International Fertilizer Development Center); this is an independent
re-implementation of its published equations.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from agrijax.core.state import Params

__all__ = [
    "BSGDD",
    "CANHT_POT",
    "DSSAT_COEFFICIENTS",
    "CeresCoefficients",
    "GrosubCoefficients",
    "PhenolCoefficients",
    "RootgrCoefficients",
    "coefficient_table",
]


def _c(value: float, unit: str, description: str, source: str, fortran: str, *, static: bool = False) -> Any:
    """A coefficient field: default ``value``, its unit and meaning, and its DSSAT-CSM origin."""
    meta = {
        "unit": unit,
        "description": description,
        "fortran_name": "",
        "dims": None,
        "source": f"DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/{source}",
        "fortran": fortran,
    }
    return eqx.field(default=value, static=static, metadata=meta)


# constants MZ_GROSUB sets in SEASINIT and passes on as species parameters (CeresSpecies)
CANHT_POT = 1.6
"""Potential canopy height [m] (``CANHT_POT = 1.6``, MZ_GROSUB.for:665, SEASINIT)."""
BSGDD = 250.0
"""Thermal time before silking at which ear growth starts [degC d] (``BSGDD = 250.0``,
MZ_GROSUB.for:656, SEASINIT, J. I. Lizaso 2006)."""


class GrosubCoefficients(Params):
    """Literals of ``MZ_GROSUB`` (``DYNAMIC = INTEGR``): growth, partitioning and senescence."""

    # ---- stage-date initialisations
    swmin_frac: float = _c(
        0.85,
        "-",
        "minimum stem weight in grain fill as a fraction of stem weight at silking",
        "MZ_GROSUB.for:911",
        "SWMIN  = STMWT*0.85",
    )
    cumph_emergence: float = _c(
        0.514, "-", "cumulative phyllochrons at emergence", "MZ_GROSUB.for:990", "CUMPH   = 0.514"
    )
    # ---- excess-water stress
    tss_days: float = _c(
        2.0,
        "d",
        "saturated days after which a layer's root activity drops",
        "MZ_GROSUB.for:1094",
        "IF (TSS(L) .GT. 2.) THEN",
    )
    # ---- light interception and assimilation
    lifac_max: float = _c(
        1.5,
        "-",
        "light extinction coefficient at zero row-spacing term",
        "MZ_GROSUB.for:1120",
        "LIFAC  = 1.5 - 0.768 * ((ROWSPC * 0.01)**2 * PLTPOP)**0.1",
    )
    lifac_slope: float = _c(
        0.768,
        "-",
        "decrease of the extinction coefficient with row spacing and population",
        "MZ_GROSUB.for:1120",
        "LIFAC  = 1.5 - 0.768 * ((ROWSPC * 0.01)**2 * PLTPOP)**0.1",
    )
    lifac_exp: float = _c(
        0.1,
        "-",
        "exponent of (row spacing^2 x population) in the extinction coefficient",
        "MZ_GROSUB.for:1120",
        "LIFAC  = 1.5 - 0.768 * ((ROWSPC * 0.01)**2 * PLTPOP)**0.1",
    )
    tavgd_tmin_w: float = _c(
        0.25,
        "-",
        "weight of TMIN in the daytime temperature of photosynthesis",
        "MZ_GROSUB.for:1132",
        "TAVGD = 0.25*TMIN+0.75*TMAX",
    )
    tavgd_tmax_w: float = _c(
        0.75,
        "-",
        "weight of TMAX in the daytime temperature of photosynthesis",
        "MZ_GROSUB.for:1132",
        "TAVGD = 0.25*TMIN+0.75*TMAX",
    )
    # ---- leaf appearance
    pc_cumph: float = _c(
        5.0,
        "-",
        "phyllochrons below which the first leaves appear faster",
        "MZ_GROSUB.for:1174",
        "IF (CUMPH .LT. 5.0) THEN",
    )
    pc_base: float = _c(
        0.66, "-", "phyllochron factor of the first leaf", "MZ_GROSUB.for:1175", "PC     = 0.66+0.068*CUMPH"
    )
    pc_slope: float = _c(
        0.068,
        "-",
        "increase of the phyllochron factor per phyllochron",
        "MZ_GROSUB.for:1175",
        "PC     = 0.66+0.068*CUMPH",
    )
    # ---- leaf area and leaf weight
    plag1_xn: float = _c(
        4.0,
        "-",
        "leaf number below which stage-1 expansion is linear in XN",
        "MZ_GROSUB.for:1217",
        "IF (XN .LT. 4.0) THEN",
    )
    plag1_lin: float = _c(
        4.0,
        "cm2 plant-1",
        "stage-1 leaf expansion per phyllochron and leaf (XN < 4)",
        "MZ_GROSUB.for:1223",
        "PLAG = 4.0*XN*TI*",
    )
    plag1_quad: float = _c(
        3.0,
        "cm2 plant-1",
        "stage-1 leaf expansion per phyllochron and leaf^2",
        "MZ_GROSUB.for:1214",
        "PLAG   = 3.0*XN*XN*TI*",
    )
    plag23_quad: float = _c(
        3.5,
        "cm2 plant-1",
        "stage-2/3 leaf expansion per phyllochron and leaf^2",
        "MZ_GROSUB.for:1282",
        "PLAG   = 3.5*XN*XN*TI",
    )
    plag3_xn2_max: float = _c(
        170.0,
        "-",
        "leaf number^2 at which stage-3 expansion stops increasing",
        "MZ_GROSUB.for:1344",
        "PLAG   = 3.5*170.0*TI",
    )
    xn3_quad: float = _c(
        12.0,
        "-",
        "leaf number up to which stage-3 expansion grows with XN^2",
        "MZ_GROSUB.for:1335",
        "IF (XN .LT. 12.0) THEN",
    )
    xn3_decline: float = _c(
        2.9999,
        "-",
        "leaves before the last at which stage-3 expansion starts to decline",
        "MZ_GROSUB.for:1342",
        "ELSEIF (XN .LT. TLNO-2.9999) THEN",
    )
    plag3_decline_offset: float = _c(
        3.0,
        "-",
        "leaf offset of the declining stage-3 expansion (XN + 3 - TLNO)",
        "MZ_GROSUB.for:1349",
        "PLAG   = 170.0*3.5/((XN+3.0-TLNO)**0.5)*TI*",
    )
    plag3_decline_exp: float = _c(
        0.5,
        "-",
        "exponent of the declining stage-3 expansion",
        "MZ_GROSUB.for:1349",
        "PLAG   = 170.0*3.5/((XN+3.0-TLNO)**0.5)*TI*",
    )
    sla_juvenile: float = _c(
        250.0,
        "cm2 g-(1/1.25)",
        "leaf area per leaf weight^(1/1.25) in stage 1",
        "MZ_GROSUB.for:1229",
        "XLFWT  = (PLA/250.0)**1.25",
    )
    sla_leaf: float = _c(
        267.0,
        "cm2 g-0.8",
        "leaf area per leaf weight^0.8 (stages 1-3)",
        "MZ_GROSUB.for:1246",
        "PLA    = (LFWT+GROLF)**0.8*267.0",
    )
    lfwt_exp: float = _c(
        1.25,
        "-",
        "exponent of leaf weight on (leaf area / SLA) in stages 1-2",
        "MZ_GROSUB.for:1229",
        "XLFWT  = (PLA/250.0)**1.25",
    )
    pla_exp: float = _c(
        0.8,
        "-",
        "exponent of leaf area on leaf weight when growth is limited",
        "MZ_GROSUB.for:1246",
        "PLA    = (LFWT+GROLF)**0.8*267.0",
    )
    grolf3_coef: float = _c(
        0.00116,
        "g cm-2.5",
        "stage-3 leaf growth per expansion and leaf area^0.25",
        "MZ_GROSUB.for:1339",
        "GROLF  = 0.00116*PLAG*PLA**0.25",
    )
    grolf3_exp: float = _c(
        0.25,
        "-",
        "exponent of plant leaf area in stage-3 leaf growth",
        "MZ_GROSUB.for:1339",
        "GROLF  = 0.00116*PLAG*PLA**0.25",
    )
    # ---- partitioning
    grort1_min_frac: float = _c(
        0.25,
        "-",
        "minimum fraction of assimilate to roots in stage 1",
        "MZ_GROSUB.for:1238",
        "IF (GRORT .LE. 0.25*CARBO) THEN",
    )
    grolf_max_frac: float = _c(
        0.75,
        "-",
        "maximum fraction of assimilate to leaves in stages 1-2",
        "MZ_GROSUB.for:1245",
        "GROLF  = CARBO*0.75",
    )
    cumph3_phint: float = _c(
        2.0,
        "-",
        "phyllochrons before the end of stage 3 at which leaf appearance stops",
        "MZ_GROSUB.for:1316",
        "IF (SUMDTT > P3 - (2. * PHINT)) THEN",
    )
    groear_max: float = _c(
        0.81,
        "-",
        "maximum fraction of assimilate to the ear (logistic ear growth)",
        "MZ_GROSUB.for:1326",
        "GROEAR = (0.81/(1.0+EXP(-0.02*(CUMDTTEG-210.0))))*CARBO",
    )
    groear_slope: float = _c(
        0.02,
        "(degC d)-1",
        "slope of the logistic ear growth in ear thermal time",
        "MZ_GROSUB.for:1326",
        "GROEAR = (0.81/(1.0+EXP(-0.02*(CUMDTTEG-210.0))))*CARBO",
    )
    groear_mid: float = _c(
        210.0,
        "degC d",
        "ear thermal time at half the maximum ear fraction",
        "MZ_GROSUB.for:1326",
        "GROEAR = (0.81/(1.0+EXP(-0.02*(CUMDTTEG-210.0))))*CARBO",
    )
    grostm3_coef: float = _c(
        0.0182,
        "-",
        "stage-3 stem growth per leaf growth and (XN - XNTI)^2",
        "MZ_GROSUB.for:1340",
        "GROSTM = GROLF*0.0182*(XN-XNTI)**2",
    )
    grostm3_late_a: float = _c(
        3.0,
        "g plant-1",
        "first factor of the late stage-3 stem growth per phyllochron",
        "MZ_GROSUB.for:1354",
        "GROSTM = 3.000*3.1*TI",
    )
    grostm3_late_b: float = _c(
        3.1,
        "-",
        "second factor of the late stage-3 stem growth per phyllochron",
        "MZ_GROSUB.for:1354",
        "GROSTM = 3.000*3.1*TI",
    )
    ear_stem_split: float = _c(
        0.5,
        "-",
        "share of stem growth given to the ear when the ear demand exceeds it",
        "MZ_GROSUB.for:1363",
        "GROEAR = GROSTM * 0.5",
    )
    grort3_min_frac: float = _c(
        0.10,
        "-",
        "minimum fraction of assimilate to roots in stage 3",
        "MZ_GROSUB.for:1370",
        "IF (GRORT .LE. 0.10*CARBO .AND. TURFAC .GT. 0.0) THEN",
    )
    grf3_frac: float = _c(
        0.90,
        "-",
        "fraction of assimilate to the shoot when stage-3 roots are at their minimum",
        "MZ_GROSUB.for:1372",
        "GRF   = CARBO*0.90/(GROSTM+GROLF+GROEAR)",
    )
    grort4_frac: float = _c(
        0.08, "-", "fraction of assimilate to roots in stage 4", "MZ_GROSUB.for:1407", "GRORT = CARBO * 0.08"
    )
    stem_root_split4: float = _c(
        0.5,
        "-",
        "share of the non-ear assimilate to stem (and to roots) when short in stage 4",
        "MZ_GROSUB.for:1411",
        "GRORT=(CARBO-GROEAR)*0.5",
    )
    stem_root_split5: float = _c(
        0.50,
        "-",
        "share of the surplus assimilate to stem (and to roots) in stage 5",
        "MZ_GROSUB.for:1513",
        "STMWT  = STMWT + GROSTM*0.50",
    )
    swmin_margin: float = _c(
        1.07,
        "-",
        "stem weight / SWMIN below which leaf reserves move to the stem",
        "MZ_GROSUB.for:1517",
        "IF (STMWT .LE. SWMIN*1.07) THEN",
    )
    leaf_to_stem5: float = _c(
        0.0050,
        "d-1",
        "fraction of leaf weight moved to the stem when the stem is depleted",
        "MZ_GROSUB.for:1518",
        "STMWT  = STMWT + LFWT*0.0050",
    )
    # ---- grain filling
    pltpop_min5: float = _c(
        0.01,
        "plant m-2",
        "population at or below which stage-5 growth stops",
        "MZ_GROSUB.for:1434",
        "IF (PLTPOP .LE. 0.01) RETURN",
    )
    carbo_active: float = _c(
        0.0001,
        "g plant-1 d-1",
        "assimilation above which grain filling proceeds",
        "MZ_GROSUB.for:1436",
        "IF (ABS(CARBO) .GT. 0.0001) THEN",
    )
    grogrn_sw_base: float = _c(
        0.45,
        "-",
        "relative grain growth at full water stress",
        "MZ_GROSUB.for:1468",
        "GROGRN = RGFILL*GPP*G3*0.001*(0.45+0.55*SWFAC)",
    )
    grogrn_sw_slope: float = _c(
        0.55,
        "-",
        "increase of relative grain growth with SWFAC",
        "MZ_GROSUB.for:1468",
        "GROGRN = RGFILL*GPP*G3*0.001*(0.45+0.55*SWFAC)",
    )
    # ---- normal leaf senescence (SLAN) and leaf respiration
    slan12_tt: float = _c(
        10000.0,
        "degC d",
        "thermal time scale of normal senescence in stages 1-2",
        "MZ_GROSUB.for:1252",
        "SLAN = SUMDTT*PLA/10000.0",
    )
    slan3_div: float = _c(
        1000.0, "-", "leaf area / normal senescence in stage 3", "MZ_GROSUB.for:1384", "SLAN   = PLA/1000.0"
    )
    slan4_base: float = _c(
        0.05,
        "-",
        "senesced fraction of leaf area at the start of stage 4",
        "MZ_GROSUB.for:1418",
        "SLAN   = PLA*(0.05+SUMDTT/200.0*0.05)",
    )
    slan4_tt: float = _c(
        200.0,
        "degC d",
        "thermal time scale of senescence in stage 4",
        "MZ_GROSUB.for:1418",
        "SLAN   = PLA*(0.05+SUMDTT/200.0*0.05)",
    )
    slan4_slope: float = _c(
        0.05,
        "-",
        "senesced fraction of leaf area per slan4_tt in stage 4",
        "MZ_GROSUB.for:1418",
        "SLAN   = PLA*(0.05+SUMDTT/200.0*0.05)",
    )
    slan5_base: float = _c(
        0.1,
        "-",
        "senesced fraction of leaf area at the start of stage 5",
        "MZ_GROSUB.for:1438",
        "SLAN   = PLA*(0.1+0.60*(SUMDTT/P5)**3)",
    )
    slan5_slope: float = _c(
        0.60,
        "-",
        "senesced fraction of leaf area per (SUMDTT / P5)^3 in stage 5",
        "MZ_GROSUB.for:1438",
        "SLAN   = PLA*(0.1+0.60*(SUMDTT/P5)**3)",
    )
    sla_senes: float = _c(
        600.0,
        "cm2 g-1",
        "senesced leaf area per gram of leaf weight lost",
        "MZ_GROSUB.for:1255",
        "LFWT   = LFWT - SLAN/600.0",
    )
    # ---- stress senescence
    slfc_lai: float = _c(
        4.0,
        "m2 m-2",
        "LAI above which competition senesces leaves",
        "MZ_GROSUB.for:1620",
        "IF (LAI .GT. 4.0) THEN",
    )
    slfc_slope: float = _c(
        0.008,
        "(m2 m-2)-1",
        "competition senescence per unit LAI above slfc_lai",
        "MZ_GROSUB.for:1621",
        "SLFC = 1.0 - 0.008*(LAI-4.0)",
    )
    slft_tmin: float = _c(
        6.0,
        "degC",
        "minimum temperature at or below which cold senesces leaves",
        "MZ_GROSUB.for:1626",
        "IF (TMIN.LE.6.0) THEN",
    )
    slft_coef: float = _c(
        0.01,
        "degC-2",
        "cold senescence per squared degree below slft_tmin",
        "MZ_GROSUB.for:1627",
        "SLFT  = AMAX1 (0.0, 1.0 - 0.01 * (TMIN-6.0)**2)",
    )
    # ---- crop failure
    cold_leafno: int = _c(
        4,
        "-",
        "leaves above which a leafless crop can die of cold",
        "MZ_GROSUB.for:1653",
        "IF (LEAFNO .GT. 4 .AND. LAI .LE. 0.0 .AND. ISTAGE .LE. 4",
        static=True,
    )
    cold_days: int = _c(
        6,
        "d",
        "cold days after which a leafless crop dies",
        "MZ_GROSUB.for:1654",
        ".AND. ICOLD .GT. 6) THEN",
        static=True,
    )
    drought_swfac: float = _c(
        0.1,
        "-",
        "SWFAC at or below which a day counts as severely water stressed",
        "MZ_GROSUB.for:1690",
        "IF (SWFAC .GT. 0.1) THEN",
    )
    drought_lai: float = _c(
        0.1,
        "m2 m-2",
        "LAI at or below which drought can end the crop",
        "MZ_GROSUB.for:1696",
        "IF (LAI .LE. 0.1 .AND. ISTAGE .LT. 4",
    )
    drought_days: int = _c(
        10,
        "d",
        "severe water-stress days after which the crop fails",
        "MZ_GROSUB.for:1697",
        ".AND. NWSD .GT. 10) THEN",
        static=True,
    )
    # ---- state totals
    carbo_floor: float = _c(
        0.001,
        "g plant-1 d-1",
        "assimilation stored when the day's is not positive",
        "MZ_GROSUB.for:1762",
        "CARBO = 0.001",
    )
    root_growth_frac: float = _c(
        0.5,
        "-",
        "fraction of the root assimilate that becomes root weight",
        "MZ_GROSUB.for:1787",
        "RTWT   = RTWT  + 0.5*GRORT-0.005*RTWT",
    )
    root_senes: float = _c(
        0.005, "d-1", "daily root weight loss", "MZ_GROSUB.for:1787", "RTWT   = RTWT  + 0.5*GRORT-0.005*RTWT"
    )
    canht_pop: float = _c(
        0.4238,
        "m2 plant-1",
        "population coefficient of the LAI that gives the potential height",
        "MZ_GROSUB.for:1829",
        "CANHT = XLAI / (0.4238 * PLTPOP + 0.3424) * CANHT_POT",
    )
    canht_base: float = _c(
        0.3424,
        "m2 m-2",
        "LAI that gives the potential height at zero population",
        "MZ_GROSUB.for:1829",
        "CANHT = XLAI / (0.4238 * PLTPOP + 0.3424) * CANHT_POT",
    )


class PhenolCoefficients(Params):
    """Literals of ``MZ_PHENOL`` (``DYNAMIC = INTEGR``): thermal time and the stage machine."""

    # ---- thermal time
    snow_max: float = _c(
        15.0,
        "cm",
        "snow depth at which the crown insulation saturates",
        "MZ_PHENOL.for:405",
        "XS     = AMIN1 (XS,15.0)",
    )
    snowfac_base: float = _c(
        0.4,
        "-",
        "crown / air temperature ratio below a saturating snow cover",
        "MZ_PHENOL.for:412",
        "TEMPCN = 2.0 + TMIN*(0.4+0.0018*(XS-15.0)**2)",
    )
    snowfac_coef: float = _c(
        0.0018,
        "cm-2",
        "increase of the crown / air ratio per squared cm of missing snow",
        "MZ_PHENOL.for:412",
        "TEMPCN = 2.0 + TMIN*(0.4+0.0018*(XS-15.0)**2)",
    )
    crown_offset: float = _c(
        2.0,
        "degC",
        "crown temperature at 0 degC air temperature under snow",
        "MZ_PHENOL.for:412",
        "TEMPCN = 2.0 + TMIN*(0.4+0.0018*(XS-15.0)**2)",
    )
    acoef_srad: float = _c(
        0.01061,
        "(MJ m-2 d-1)-1",
        "radiation dependence of the daytime soil-temperature weight",
        "MZ_PHENOL.for:462",
        "ACOEF  = 0.01061 * SRAD + 0.5902",
    )
    acoef_base: float = _c(
        0.5902,
        "-",
        "daytime soil-temperature weight of TMAX at zero radiation",
        "MZ_PHENOL.for:462",
        "ACOEF  = 0.01061 * SRAD + 0.5902",
    )
    tnsoil_tmax_w: float = _c(
        0.36354,
        "-",
        "weight of TMAX in the night soil temperature",
        "MZ_PHENOL.for:464",
        "TNSOIL = 0.36354 * TMAX + 0.63646 * TMIN",
    )
    tnsoil_tmin_w: float = _c(
        0.63646,
        "-",
        "weight of TMIN in the night soil temperature",
        "MZ_PHENOL.for:464",
        "TNSOIL = 0.36354 * TMAX + 0.63646 * TMIN",
    )
    hourly_pi: float = _c(
        3.14,
        "-",
        "value of pi in the 24-point sine temperature interpolation",
        "MZ_PHENOL.for:495",
        "TH = (TMAX+TMIN)/2. + (TMAX-TMIN)/2. * SIN(3.14/12.*I)",
    )
    leafno_soil: int = _c(
        10,
        "-",
        "leaves up to which the growing point is below ground (soil temperature)",
        "MZ_PHENOL.for:449",
        "ELSEIF (LEAFNO.LE.10) THEN",
        static=True,
    )
    # ---- germination and emergence
    swsd_w_seed: float = _c(
        0.65,
        "-",
        "weight of the seed layer in the germination soil water",
        "MZ_PHENOL.for:558",
        "SWSD = (SW(L0)-LL(L0))*0.65 +",
    )
    swsd_w_below: float = _c(
        0.35,
        "-",
        "weight of the layer below the seed in the germination soil water",
        "MZ_PHENOL.for:559",
        "(SW(L0+1)-LL(L0+1))*0.35",
    )
    p9_base: float = _c(
        45.0,
        "degC d",
        "thermal time germination -> emergence at zero sowing depth",
        "MZ_PHENOL.for:591",
        "P9    = 45.0 +  GDDE*SDEPTH",
    )
    tlno_emergence: float = _c(
        30.0, "-", "provisional total leaf number set at emergence", "MZ_PHENOL.for:634", "TLNO   = 30.0"
    )
    # ---- tassel initiation
    tlno_phint_frac: float = _c(
        0.5,
        "-",
        "fraction of PHINT per leaf initiated before tassel initiation",
        "MZ_PHENOL.for:734",
        "TLNO    = SUMDTT/(PHINT*0.5)+ 5.0",
    )
    tlno_offset: float = _c(
        5.0,
        "-",
        "leaves already initiated in the seed",
        "MZ_PHENOL.for:734",
        "TLNO    = SUMDTT/(PHINT*0.5)+ 5.0",
    )
    p3_leaf_offset: float = _c(
        0.5,
        "-",
        "leaves beyond TLNO that appear before silking",
        "MZ_PHENOL.for:735",
        "P3      = ((TLNO + 0.5) * PHINT) - SUMDTT",
    )
    # ---- grain number and ears (beginning of effective grain filling)
    psker_a: float = _c(
        3.4,
        "-",
        "numerator of the PSKER scaling (SUMP x 1000 / IDURP x 3.4 / 5.0)",
        "MZ_PHENOL.for:810",
        "PSKER = SUMP*1000.0/IDURP*3.4/5.0",
    )
    psker_b: float = _c(
        5.0, "-", "denominator of the PSKER scaling", "MZ_PHENOL.for:810", "PSKER = SUMP*1000.0/IDURP*3.4/5.0"
    )
    gpp_psker: float = _c(
        7200.0,
        "mg plant-1 d-1",
        "PSKER at which kernel number reaches G2 (less 50)",
        "MZ_PHENOL.for:811",
        "GPP   = G2*PSKER/7200.0 + 50.0",
    )
    gpp_offset: float = _c(
        50.0,
        "kernel plant-1",
        "kernel number at zero PSKER",
        "MZ_PHENOL.for:811",
        "GPP   = G2*PSKER/7200.0 + 50.0",
    )
    gpp_min: float = _c(
        51.0, "kernel plant-1", "minimum kernel number", "MZ_PHENOL.for:817", "GPP = AMAX1 (GPP,51.0)"
    )
    ears_low_frac: float = _c(
        0.15, "-", "GPP / G2 below which ears are lost", "MZ_PHENOL.for:823", "IF (GPP .LT. G2*0.15) THEN"
    )
    ears_low_exp: float = _c(
        0.33,
        "-",
        "exponent of the ear loss at low kernel number",
        "MZ_PHENOL.for:824",
        "EARS = PLTPOP*(GPP/(G2*0.15))**0.33",
    )
    barren_pltpop: float = _c(
        12.0,
        "plant m-2",
        "population above which plants can be barren",
        "MZ_PHENOL.for:829",
        "IF (PLTPOP .GT. 12.0) THEN",
    )
    barren_gpp_frac: float = _c(
        0.5,
        "-",
        "GPP / G2 below which dense stands are partly barren",
        "MZ_PHENOL.for:833",
        "IF (GPP .LT. G2*0.5) THEN",
    )
    barfac_coef: float = _c(
        0.0085,
        "(plant m-2)-1.5",
        "barrenness exponent per unit (1 - GPP / G2) x PLTPOP^1.5",
        "MZ_PHENOL.for:839",
        "BARFAC = 0.0085*(1.0-GPP/G2)*PLTPOP**1.5",
    )
    barfac_exp: float = _c(
        1.5,
        "-",
        "exponent of population in the barrenness factor",
        "MZ_PHENOL.for:839",
        "BARFAC = 0.0085*(1.0-GPP/G2)*PLTPOP**1.5",
    )
    # ---- grain filling and maturity
    efg_end_frac: float = _c(
        0.95,
        "-",
        "fraction of P5 at which effective grain filling ends",
        "MZ_PHENOL.for:870",
        "IF (SUMDTT .LT. P5*0.95) RETURN",
    )
    dtt_maturity: float = _c(
        2.0,
        "degC d",
        "daily thermal time below which a stage-6 crop matures at once",
        "MZ_PHENOL.for:881",
        "IF (DTT .LT. 2.0) SUMDTT = P5",
    )


class RootgrCoefficients(Params):
    """Literals of ``MZ_ROOTGR`` (``DYNAMIC = INTEGR``): root front and root length density."""

    rtdep_emerg: float = _c(
        0.15,
        "cm (degC d)-1",
        "root front advance per degree day before emergence",
        "MZ_ROOTS.for:99",
        "IF(ISTAGE.EQ.9) RTDEP  = RTDEP + 0.15*DTT",
    )
    rlv_emergence: float = _c(
        0.20,
        "cm cm-2 plant-1 m2",
        "root length per unit area and plant at emergence",
        "MZ_ROOTS.for:106",
        "RLV(L) = 0.20*PLTPOP/DLAYR (L)",
    )
    grort_min: float = _c(
        0.0001,
        "g plant-1 d-1",
        "root growth at or below which roots do not grow",
        "MZ_ROOTS.for:125",
        "IF (GRORT.LE.0.0001) GOTO 999",
    )
    swdf_esw_frac: float = _c(
        0.25,
        "-",
        "fraction of extractable water below which root growth is water limited",
        "MZ_ROOTS.for:140",
        "IF (SW(L)-LL(L) .LT. 0.25*ESW(L)) THEN",
    )
    swdf_slope: float = _c(
        4.0,
        "-",
        "slope of the root water-deficit factor in relative extractable water",
        "MZ_ROOTS.for:141",
        "SWDF = 4.0*(SW(L)-LL(L))/ESW(L)",
    )
    rtexf: float = _c(
        0.1,
        "-",
        "root loss at full waterlogging of the deepest rooted layer",
        "MZ_ROOTS.for:171",
        "RTEXF = 0.1",
    )
    rtdep_cumdtt: float = _c(
        275.0,
        "degC d",
        "thermal time since germination at which the root front speeds up",
        "MZ_ROOTS.for:181",
        "IF (CUMDTT .LT. 275.0) THEN",
    )
    rtdep_rate_early: float = _c(
        0.1,
        "cm (degC d)-1",
        "root front advance per degree day before rtdep_cumdtt",
        "MZ_ROOTS.for:182",
        "RTDEP = RTDEP + DTT*0.1*SQRT(SHF(L)*AMIN1(SWFAC*2.0,SWDF))",
    )
    rtdep_rate_late: float = _c(
        0.2,
        "cm (degC d)-1",
        "root front advance per degree day after rtdep_cumdtt",
        "MZ_ROOTS.for:185",
        "RTDEP = RTDEP + DTT*0.2*SQRT(SHF(L)*AMIN1(SWFAC*2.0,SWDF))",
    )
    rtdep_swfac_mult: float = _c(
        2.0,
        "-",
        "multiplier of SWFAC in the water limitation of the root front",
        "MZ_ROOTS.for:182",
        "RTDEP = RTDEP + DTT*0.1*SQRT(SHF(L)*AMIN1(SWFAC*2.0,SWDF))",
    )
    trldf_min_frac: float = _c(
        0.00001,
        "-",
        "minimum total root-length distribution factor per new root length",
        "MZ_ROOTS.for:196",
        "IF (TRLDF .GE. RLNEW*0.00001) THEN",
    )
    rlv_decay: float = _c(
        0.005,
        "d-1",
        "daily root length density loss",
        "MZ_ROOTS.for:199",
        "RLV(L) = RLV(L) + RLDF(L)*RNLF/DLAYR(L)-0.005*RLV(L)",
    )
    rlv_max: float = _c(
        4.0, "cm cm-3", "maximum root length density", "MZ_ROOTS.for:204", "RLV(L) = AMIN1 (RLV(L),4.0)"
    )


class CeresCoefficients(Params):
    """All hard-coded CERES-Maize coefficients (``CeresMaizeParams.coefficients``)."""

    grosub: GrosubCoefficients = eqx.field(default_factory=GrosubCoefficients)
    phenol: PhenolCoefficients = eqx.field(default_factory=PhenolCoefficients)
    roots: RootgrCoefficients = eqx.field(default_factory=RootgrCoefficients)

    def as_arrays(self, dtype: Any = None) -> CeresCoefficients:
        """The same coefficients with every (non-static) leaf a JAX array, for ``jax.grad``."""
        return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=dtype), self)


DSSAT_COEFFICIENTS = CeresCoefficients()
"""The DSSAT-CSM v4.8.6.0 values (what ``CeresMaizeParams.coefficients = None`` means)."""


def coefficient_table() -> list[dict[str, Any]]:
    """One row per coefficient: ``path``, ``value``, ``unit``, ``description``, ``source``,
    ``fortran`` and ``static``, in field order (``grosub.*``, ``phenol.*``, ``roots.*``)."""
    rows = []
    for group in dataclasses.fields(CeresCoefficients):
        inst = getattr(DSSAT_COEFFICIENTS, group.name)
        for f in dataclasses.fields(inst):
            rows.append(
                {
                    "path": f"{group.name}.{f.name}",
                    "value": getattr(inst, f.name),
                    "unit": f.metadata["unit"],
                    "description": f.metadata["description"],
                    "source": f.metadata["source"],
                    "fortran": f.metadata["fortran"],
                    "static": bool(f.metadata.get("static", False)),
                }
            )
    return rows
