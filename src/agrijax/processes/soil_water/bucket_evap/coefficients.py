"""The numbers of the DSSAT-CSM v4.8.6.0 soil and mulch evaporation, declared once with provenance.

* :class:`SoilevCoefficients`: Ritchie's two-stage soil evaporation ``SOILEV`` / ``ESUP``
  (``SPAM/SOILEV.for``, J. T. Ritchie and B. Baer; ``MESEV = 'R'``);
* :class:`EsrCoefficients`: the Suleiman-Ritchie layered soil evaporation ``ESR_SoilEvap``
  (``SPAM/ESR_SoilEvap.for``, J. T. Ritchie and C. Porter; ``MESEV = 'S'``);
* :class:`MulchEvapCoefficients`: the evaporation of the surface mulch ``MULCH_EVAP``
  (``Soil/Mulch/MULCHEVAP.for``, C. H. Porter; Scopel et al. 2004);
* :class:`EvapGateCoefficients`: the thresholds with which ``SPAM`` decides whether the mulch and
  soil evaporation routines are called (``SPAM/SPAM.for``).

DSSAT-CSM is BSD-3 (Copyright 1998-2026 DSSAT Foundation, University of Florida, International
Fertilizer Development Center): every coefficient quotes its source statement and line. Defaults
are Python floats, so a run with the default sets traces exactly the literals of the equations;
pass a set with array leaves (``.as_arrays()``) to calibrate or differentiate them. Thresholds
that only guard the reference's arithmetic are leaves kept out of the calibration vector.

Published equations: Ritchie, J.T. (1972), Water Resour. Res. 8, 1204-1213 (two-stage
evaporation); Suleiman, A.A. and Ritchie, J.T. (2003), Soil Sci. Soc. Am. J. 67, 377-386;
Ritchie, J.T., Porter, C.H., Judge, J., Jones, J.W., Suleiman, A.A. (2009), Soil Sci. Soc. Am. J.
73, 792-801; Scopel, E. et al. (2004), Agronomie 24, 383-395 (mulch evaporation).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx

from agrijax.core.coefficients import Coefficients, Provenance, coef, coefficient_table

__all__ = [
    "EVAP_COEFFICIENTS",
    "REF_VERSION",
    "EsrCoefficients",
    "EvapCoefficients",
    "EvapGateCoefficients",
    "MulchEvapCoefficients",
    "SoilevCoefficients",
    "evap_coefficient_table",
]

#: the reference of every coefficient here (the registry keys end in ``@dssat-4.8.6.0``)
REF_VERSION = "dssat-4.8.6.0"
_RITCHIE72 = "Ritchie (1972)"
_SR03 = "Suleiman and Ritchie (2003)"
_R09 = "Ritchie et al. (2009)"
_SCOPEL = "Scopel et al. (2004)"


def _at(file_line: str, routine: str, statement: str, paper: str = "", note: str = "") -> Provenance:
    return Provenance.at(REF_VERSION, file_line, routine=routine, statement=statement, paper=paper, note=note)


def _soilev(line: int, statement: str, routine: str = "SOILEV", note: str = "") -> Provenance:
    return _at(f"SPAM/SOILEV.for:{line}", routine, statement, _RITCHIE72, note)


def _esr(line: int, statement: str, paper: str = _R09, note: str = "") -> Provenance:
    return _at(f"SPAM/ESR_SoilEvap.for:{line}", "ESR_SoilEvap", statement, paper, note)


def _mulch(line: int, statement: str, note: str = "") -> Provenance:
    return _at(f"Soil/Mulch/MULCHEVAP.for:{line}", "MULCH_EVAP", statement, _SCOPEL, note)


class SoilevCoefficients(Coefficients):
    """``SOILEV`` / ``ESUP``: Ritchie's two-stage soil evaporation."""

    stage2_rate: float = coef(
        3.5,
        "mm",
        "stage-2 cumulative evaporation per square root of days, mm d-1/2 (SUMES2 = 3.5 sqrt(T))",
        _soilev(109, "ES = 3.5 * T**0.5 - SUMES2", note="also lines 78, 119, 145, 153"),
    )
    stage2_infil_frac: float = coef(
        0.8,
        "-",
        "fraction of the day's infiltrating water evaporated in stage 2 on a wet day",
        _soilev(111, "ESX = 0.8 * WINF"),
    )
    esup_es_frac: float = coef(
        0.4,
        "-",
        "reduction of the day's evaporation per mm of stage-1 excess over U (ES = EOS - 0.4 (SUMES1 - U))",
        _soilev(220, "ES = EOS - 0.4 * (SUMES1 - U)", routine="ESUP"),
    )
    esup_s2_frac: float = coef(
        0.6,
        "-",
        "fraction of the stage-1 excess over U that starts stage 2 (SUMES2 = 0.6 (SUMES1 - U))",
        _soilev(221, "SUMES2 = 0.6 * (SUMES1 - U)", routine="ESUP"),
    )
    swef_max: float = coef(
        0.9,
        "-",
        "air-dry fraction of LL of a 30 cm top layer (SWEF = 0.9 - 0.00038 (DLAYR - 30)^2)",
        _soilev(85, "SWEF = 0.9-0.00038*(DLAYR(1)-30.)**2"),
    )
    swef_curvature: float = coef(
        0.00038,
        "cm-2",
        "decrease of SWEF with the squared departure of the top-layer thickness from 30 cm",
        _soilev(85, "SWEF = 0.9-0.00038*(DLAYR(1)-30.)**2"),
    )
    swef_dlayr_ref: float = coef(
        30.0,
        "cm",
        "top-layer thickness at which SWEF is largest",
        _soilev(85, "SWEF = 0.9-0.00038*(DLAYR(1)-30.)**2"),
    )
    pm_min: float = coef(
        1e-6,
        "-",
        "plastic-mulch fraction above which ES is reduced by (1 - PMFRACTION)",
        _soilev(162, "IF (PMFRACTION .GT. 1.E-6) THEN"),
        calibrate=False,
    )


class EsrCoefficients(Coefficients):
    """``ESR_SoilEvap``: the Suleiman-Ritchie layered evaporation (``MESEV = 'S'``)."""

    air_dry_frac: float = coef(
        0.30,
        "-",
        "air-dry water content as a fraction of LL (SWAD = 0.30 LL)",
        _esr(86, "SWAD(L) = 0.30 * LL(L)"),
    )
    pseudo_integration: float = coef(
        0.5,
        "-",
        "fraction of a positive drainage change SWDELTS counted in the day's water content",
        _esr(94, "SWTEMP(L) = SW(L) + 0.5 * SWDELTS(L)"),
    )
    wet_depth: float = coef(
        100.0,
        "cm",
        "mean depth above which a layer wetter than DUL makes the profile wet",
        _esr(101, "IF (MEANDEP(L) < 100. .AND. SWTEMP(L) > DUL(L)) THEN"),
    )
    thr_a: float = coef(
        0.275,
        "-",
        "linear term of the top-layer wet-profile threshold in DUL",
        _esr(111, "SW_threshold = 0.275*DUL(1) + 1.165*DUL(1)*DUL(1) +"),
    )
    thr_b: float = coef(
        1.165,
        "-",
        "quadratic term of the top-layer wet-profile threshold in DUL",
        _esr(111, "SW_threshold = 0.275*DUL(1) + 1.165*DUL(1)*DUL(1) +"),
    )
    thr_c: float = coef(
        1.2,
        "cm-1",
        "depth term of the threshold per cm of mean depth: 1.2 DUL^3.75 MEANDEP",
        _esr(112, "&          (1.2*DUL(1)**3.75)*MEANDEP(1)"),
    )
    thr_exp: float = coef(
        3.75,
        "-",
        "exponent of DUL in the depth term of the threshold",
        _esr(112, "&          (1.2*DUL(1)**3.75)*MEANDEP(1)"),
    )
    dry_a0: float = coef(
        0.5,
        "-",
        "intercept of the dry-profile coefficient A = 0.5 + 0.24 DUL",
        _esr(127, "A =  0.5  + 0.24 * DUL(L)"),
    )
    dry_a1: float = coef(
        0.24, "-", "slope of the dry-profile coefficient A in DUL", _esr(127, "A =  0.5  + 0.24 * DUL(L)")
    )
    dry_b0: float = coef(
        -2.04,
        "-",
        "intercept of the dry-profile exponent B = -2.04 + 0.20 DUL",
        _esr(128, "B = -2.04 + 0.20 * DUL(L)"),
    )
    dry_b1: float = coef(
        0.20, "-", "slope of the dry-profile exponent B in DUL", _esr(128, "B = -2.04 + 0.20 * DUL(L)")
    )
    equilibrium_coef: float = coef(
        0.011,
        "-",
        "daily fraction of (SW - SWAD) evaporated from every layer of an intermediate profile",
        _esr(133, "ES_Coef(L) = 0.011"),
    )
    wet_a: float = coef(
        0.26, "-", "coefficient A of the wet-profile evaporation A MEANDEP^B", _esr(138, "A = 0.26")
    )
    wet_b: float = coef(
        -0.70, "-", "exponent B of the wet-profile evaporation A MEANDEP^B", _esr(139, "B = -0.70")
    )
    pm_min: float = coef(
        1e-6,
        "-",
        "plastic-mulch fraction above which SWDELTU is reduced by (1 - PMFRACTION)",
        _esr(148, "IF (PMFRACTION .GT. 1.E-6) THEN"),
        calibrate=False,
    )


class MulchEvapCoefficients(Coefficients):
    """``MULCH_EVAP``: evaporation of the water held by the surface mulch (Scopel et al. 2004)."""

    mass_min: float = coef(
        0.1,
        "kg ha-1",
        "mulch mass above which the mulch evaporates and shades the soil",
        _mulch(41, "IF (MULCHMASS .GT. 0.1) THEN"),
        calibrate=False,
    )
    cover_min: float = coef(
        1e-6,
        "-",
        "mulch cover above which the mulch area index is computed (else MAI = 0)",
        _mulch(69, "IF (MULCHCOVER > 1.E-6) THEN"),
        calibrate=False,
    )
    am_unit: float = coef(
        1e-5,
        "ha kg-1 cm-2 g",
        "the reference's scale of AM * MULCHMASS in the mulch area index (1 cm2 g-1 = 1e-5 ha kg-1)",
        _mulch(
            70,
            "MAI = (AM * 1.E-5 * MULCHMASS) / MULCHCOVER",
            note="unit conversion as the reference writes it",
        ),
        calibrate=False,
    )
    water_frac: float = coef(
        0.85,
        "-",
        "largest fraction of the mulch water evaporated in a day",
        _mulch(79, "EM2 = MIN(EOM, MULCHWAT * 0.85)"),
    )


class EvapGateCoefficients(Coefficients):
    """The thresholds of ``SPAM`` that decide whether the mulch and soil evaporation run."""

    eos_min_mulch: float = coef(
        1e-6,
        "mm d-1",
        "potential evaporation above which SPAM calls MULCH_EVAP (with MEINF in R, S, M)",
        _at("SPAM/SPAM.for:338", "SPAM", "IF (EOS_SOIL > 1.E-6 .AND. INDEX('RSM',MEINF) > 0) THEN"),
        calibrate=False,
    )
    eos_min_soil: float = coef(
        1e-6,
        "mm d-1",
        "potential soil evaporation above which SPAM calls SOILEV / ESR_SoilEvap (else ES = 0)",
        _at("SPAM/SPAM.for:349", "SPAM", "IF (EOS_SOIL > 1.E-6) THEN"),
        calibrate=False,
    )


class EvapCoefficients(Coefficients):
    """Every coefficient of the soil-side evaporation, grouped by routine."""

    soilev: SoilevCoefficients = eqx.field(default_factory=SoilevCoefficients)
    esr: EsrCoefficients = eqx.field(default_factory=EsrCoefficients)
    mulch: MulchEvapCoefficients = eqx.field(default_factory=MulchEvapCoefficients)
    gate: EvapGateCoefficients = eqx.field(default_factory=EvapGateCoefficients)


#: the DSSAT-CSM v4.8.6.0 values
EVAP_COEFFICIENTS = EvapCoefficients()


def evap_coefficient_table() -> list[dict[str, Any]]:
    """One row per coefficient of :data:`EVAP_COEFFICIENTS` (:func:`coefficient_table`)."""
    return coefficient_table(EVAP_COEFFICIENTS)
