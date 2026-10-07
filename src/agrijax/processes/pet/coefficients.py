"""The coefficients of the PET kernels, declared once with unit, meaning and provenance.

Every number of the ASCE reference ET (:mod:`.penman_monteith`) and Priestley-Taylor
(:mod:`.priestley_taylor`) equations that is not an input, a forcing or a parameter is a field
below, declared with :func:`agrijax.core.coefficients.coef`:

* :class:`ASCECoefficients` (ASCE-EWRI 2005, reference ``asce-ewri-2005``), with the three
  constants that the reference-ET routine ``REF_ET`` of RZWQM2 sets differently
  (``variant="rzwqm"``), cited by file and line only (no statement).
* :class:`PTCoefficients` (DSSAT-CSM v4.8.6.0 ``SPAM/PET.for`` ``PETPT``, BSD-3: the Fortran
  statement is quoted).

Physical constants (Stefan-Boltzmann, the gas constant of air, the specific heat of air) carry
the value the reference sets, a ``note`` with the physical value, and ``calibrate=False``. Every
other coefficient is calibratable.

Defaults are Python floats, so a run with the default sets (:data:`ASCE_2005`, :data:`DSSAT_PT`,
both in :data:`PET_COEFFICIENTS`) traces exactly the literals the equations had before they were
named. To calibrate or differentiate a coefficient, pass a set with array leaves
(``ASCE_2005.as_arrays()``) to the kernel's ``coefficients`` argument, or put a
:class:`PETCoefficients` into ``PETSiteParams.coefficients`` for the processes.

Not coefficients, and therefore not declared here: unit conversions (the named constants of
:mod:`agrijax.core.units`), arithmetic means, integer exponents, and the numerical guards the port
adds for finite gradients (:func:`agrijax.core.coefficients.numerical_guard`).

Sources of the published equations cited below: ASCE-EWRI (2005), The ASCE Standardized Reference
Evapotranspiration Equation; Allen, R.G. et al. (1998), FAO Irrigation and Drainage Paper 56;
Ritchie, J.T. (1972), Water Resour. Res. 8, 1204-1213; Priestley, C.H.B. and Taylor, R.J. (1972),
Mon. Weather Rev. 100, 81-92.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx

from agrijax.core.coefficients import Coefficients, Provenance, coef
from agrijax.core.coefficients import coefficient_table as _coefficient_table

__all__ = [
    "ASCE_2005",
    "DSSAT_PT",
    "PET_COEFFICIENTS",
    "ASCECoefficients",
    "PETCoefficients",
    "PTCoefficients",
    "coefficient_table",
]

#: registry references of the PET processes
REF_RZWQM = "rzwqm2-4.6"
REF_ASCE = "asce-ewri-2005"
REF_DSSAT = "dssat-4.8.6.0"
#: reference source files (relative to the reference source trees)
REFET = "RZWQM/REF_ET.FOR"
PETFOR = "SPAM/PET.for"

ASCE = "ASCE-EWRI (2005)"
FAO56 = "Allen et al. (1998), FAO-56"
RITCHIE = "Ritchie (1972)"


def _rz(
    line: int, routine: str, paper: str, *, equation: str = "", note: str = "", file: str = REFET
) -> Provenance:
    """``REF_ET.FOR`` provenance: file, line and routine of the reference-ET routine plus the published
    equation (no statement)."""
    return Provenance(
        REF_RZWQM, file=file, line=line, routine=routine, paper=paper, equation=equation, note=note
    )


def _asce(equation: str = "", note: str = "") -> Provenance:
    """ASCE-EWRI (2005) provenance (a published standard: paper and equation only)."""
    return Provenance(REF_ASCE, paper=ASCE, equation=equation, note=note)


def _pt(line: int, statement: str, paper: str = "", note: str = "") -> Provenance:
    """DSSAT-CSM ``PETPT`` provenance with the quoted Fortran statement (BSD-3)."""
    return Provenance.at(
        REF_DSSAT, f"{PETFOR}:{line}", routine="PETPT", statement=statement, paper=paper, note=note
    )


_PHYS = "physical constant, not calibrated"


# ================================================================================================
# ASCE standardized reference ET (ASCE-EWRI 2005; RZWQM2 REF_ET.FOR variant)
# ================================================================================================
class ASCECoefficients(Coefficients):
    """ASCE-EWRI (2005) daily standardized reference ET, and the three ``REF_ET.FOR`` departures."""

    tetens_a: float = coef(
        0.6108,
        "kPa",
        "saturation vapour pressure at 0 degC (Tetens)",
        _asce("7"),
    )
    tetens_b: float = coef(
        17.27,
        "-",
        "coefficient of the Tetens exponent 17.27 T / (T + 237.3)",
        _asce("7"),
    )
    tetens_c: float = coef(
        237.3,
        "degC",
        "temperature offset of the Tetens curve",
        _asce("7"),
    )
    slope_a: float = coef(
        2503.0,
        "kPa degC",
        "slope of the saturation curve: 2503 exp(17.27 T/(T + 237.3)) / (T + 237.3)^2",
        _asce("5", note="slope of the saturation vapour pressure curve"),
    )
    wind_a: float = coef(
        4.87,
        "-",
        "numerator of the log-law wind adjustment to 2 m over clipped grass",
        _asce("33"),
    )
    wind_b: float = coef(
        67.8,
        "m-1",
        "height coefficient of the log-law wind adjustment ln(67.8 z - 5.42)",
        _asce("33"),
    )
    wind_c: float = coef(
        5.42,
        "-",
        "constant of the log-law wind adjustment",
        _asce("33"),
    )
    p0: float = coef(
        101.3,
        "kPa",
        "sea-level atmospheric pressure",
        _asce("3", note="atmospheric pressure from elevation"),
    )
    t0: float = coef(
        293.0,
        "K",
        "standard air temperature of the pressure-elevation formula",
        _asce("3", note="atmospheric pressure from elevation"),
    )
    lapse_rate: float = coef(
        0.0065,
        "K m-1",
        "temperature lapse rate of the pressure-elevation formula",
        _asce("3", note="atmospheric pressure from elevation"),
    )
    pressure_exp: float = coef(
        5.26,
        "-",
        "exponent of the pressure-elevation formula (g / (lapse R))",
        _asce("3", note="atmospheric pressure from elevation"),
    )
    psychrometric: float = coef(
        0.000665,
        "degC-1",
        "psychrometric constant per unit pressure (cp / (epsilon lambda))",
        _asce("4", note="psychrometric constant gamma = 0.000665 P"),
    )
    eccentricity: float = coef(
        0.033,
        "-",
        "amplitude of the inverse relative Earth-Sun distance dr = 1 + a cos(w J)",
        _asce("23"),
    )
    decl_amp: float = coef(
        0.409,
        "rad",
        "amplitude of the solar declination 0.409 sin(w J - 1.39)",
        _asce("24"),
    )
    decl_phase: float = coef(
        1.39,
        "rad",
        "phase of the solar declination",
        _asce("24"),
    )
    solar_const_hourly: float = coef(
        4.92,
        "MJ m-2 h-1",
        "solar constant Gsc = 0.0820 MJ m-2 min-1 per hour",
        _asce("21", note="Ra = (24/pi) Gsc dr (ws sin(lat) sin(dec) + cos(lat) cos(dec) sin(ws))"),
    )
    days_per_year: float = coef(
        365.0,
        "d",
        "days per year of the day angle 2 pi J / 365 (variant 'asce')",
        _asce("23", note="also eq. 24; astronomical constant, not calibrated"),
        calibrate=False,
    )
    rso_a: float = coef(
        0.75,
        "-",
        "clear-sky transmissivity at sea level: Rso = (a + b z) Ra",
        _asce("19", note="clear-sky solar radiation"),
    )
    rso_b: float = coef(
        2.0e-5,
        "m-1",
        "increase of the clear-sky transmissivity with elevation",
        _asce("19", note="clear-sky solar radiation"),
    )
    relsol_min: float = coef(
        0.3,
        "-",
        "lower limit of the relative shortwave Rs/Rso",
        _asce("18", note="limit of Rs/Rso in fcd"),
    )
    fcd_a: float = coef(
        1.35,
        "-",
        "cloudiness function fcd = a Rs/Rso - b",
        _asce("18", note="cloudiness function"),
    )
    fcd_b: float = coef(
        0.35,
        "-",
        "constant of the cloudiness function",
        _asce("18", note="cloudiness function"),
    )
    net_shortwave: float = coef(
        0.77,
        "-",
        "net shortwave fraction 1 - albedo of the reference surface (albedo 0.23)",
        _asce("16", note="net shortwave radiation"),
    )
    emissivity_a: float = coef(
        0.34,
        "-",
        "net emissivity at zero vapour pressure: 0.34 - 0.14 sqrt(ea)",
        _asce("17", note="net long-wave radiation"),
    )
    emissivity_b: float = coef(
        0.14,
        "kPa-0.5",
        "decrease of the net emissivity with the square root of the vapour pressure",
        _asce("17", note="net long-wave radiation"),
    )
    t_kelvin_longwave: float = coef(
        273.16,
        "K",
        "Kelvin offset of the long-wave temperatures (both variants)",
        _asce("17", note="net long-wave radiation; REF_ET.FOR line 295 uses the same value"),
        calibrate=False,
    )
    stefan_boltzmann: float = coef(
        4.903e-9,
        "MJ m-2 K-4 d-1",
        "Stefan-Boltzmann constant (variant 'asce'; the FAO-56 value, see note)",
        Provenance(
            REF_ASCE,
            paper=FAO56,
            equation="39",
            note=(
                _PHYS + ". 4.903e-9 is the FAO-56 value; ASCE-EWRI (2005) eq. 17 states 4.901e-9 (as "
                "REF_ET.FOR, variant 'rzwqm'), a documented deviation of variant 'asce' "
                "(CODATA 5.670374e-8 W m-2 K-4 = 4.899e-9)"
            ),
        ),
        calibrate=False,
    )
    t_kelvin: float = coef(
        273.0,
        "K",
        "Kelvin offset of the mean temperature in the aerodynamic term Cn/(T + 273) (variant 'asce')",
        _asce("1"),
        calibrate=False,
    )
    radiation_to_et: float = coef(
        0.408,
        "kg MJ-1",
        "inverse latent heat 1/lambda that converts Rn - G to mm d-1",
        _asce("1"),
    )
    cn_short: float = coef(
        900.0,
        "K mm s3 t-1 d-1",
        "numerator constant Cn of the short (grass) reference, daily",
        _asce("Table 1"),
    )
    cd_short: float = coef(
        0.34,
        "s m-1",
        "denominator constant Cd of the short (grass) reference, daily",
        _asce("Table 1"),
    )
    cn_tall: float = coef(
        1600.0,
        "K mm s3 t-1 d-1",
        "numerator constant Cn of the tall (alfalfa) reference, daily",
        _asce("Table 1"),
    )
    cd_tall: float = coef(
        0.38,
        "s m-1",
        "denominator constant Cd of the tall (alfalfa) reference, daily",
        _asce("Table 1"),
    )
    # ---- REF_ET.FOR (variant "rzwqm")
    stefan_boltzmann_rzwqm: float = coef(
        4.901e-9,
        "MJ m-2 K-4 d-1",
        "Stefan-Boltzmann constant as REF_ET.FOR writes it (variant 'rzwqm')",
        _rz(159, "REF_ET", ASCE, file=REFET, note=_PHYS),
        calibrate=False,
    )
    day_angle_rzwqm: float = coef(
        0.0172,
        "rad d-1",
        "day angle per day of dr and the declination (variant 'rzwqm', 2 pi / 365.3)",
        _rz(217, "REF_ET", ASCE, equation="23", file=REFET, note="also line 220 (eq. 24)"),
        calibrate=False,
    )
    t_kelvin_rzwqm: float = coef(
        273.16,
        "K",
        "Kelvin offset of the mean temperature in Cn/(T + 273.16) (variant 'rzwqm')",
        _rz(349, "REF_ET", ASCE, equation="1", file=REFET, note="also line 326"),
        calibrate=False,
    )


# ================================================================================================
# Priestley-Taylor (DSSAT-CSM PETPT)
# ================================================================================================
class PTCoefficients(Coefficients):
    """DSSAT-CSM ``PETPT`` (Ritchie's Priestley-Taylor form)."""

    td_tmax_weight: float = coef(
        0.6,
        "-",
        "weight of TMAX in the daytime temperature TD",
        _pt(895, "TD = 0.60*TMAX+0.40*TMIN", RITCHIE),
    )
    td_tmin_weight: float = coef(
        0.4,
        "-",
        "weight of TMIN in the daytime temperature TD",
        _pt(895, "TD = 0.60*TMAX+0.40*TMIN", RITCHIE),
    )
    canopy_albedo: float = coef(
        0.23,
        "-",
        "albedo of a full crop canopy",
        _pt(900, "ALBEDO = 0.23-(0.23-MSALB)*EXP(-0.75*XHLAI)", RITCHIE),
    )
    albedo_lai_decay: float = coef(
        0.75,
        "-",
        "extinction coefficient of the soil albedo's weight with LAI",
        _pt(900, "ALBEDO = 0.23-(0.23-MSALB)*EXP(-0.75*XHLAI)", RITCHIE),
    )
    eeq_a: float = coef(
        2.04e-4,
        "mm cal-1 cm2 degC-1",
        "equilibrium evaporation per langley and degC at zero albedo",
        _pt(904, "EEQ = SLANG*(2.04E-4-1.83E-4*ALBEDO)*(TD+29.0)", "Priestley and Taylor (1972)"),
    )
    eeq_b: float = coef(
        1.83e-4,
        "mm cal-1 cm2 degC-1",
        "decrease of the equilibrium evaporation with albedo",
        _pt(904, "EEQ = SLANG*(2.04E-4-1.83E-4*ALBEDO)*(TD+29.0)", "Priestley and Taylor (1972)"),
    )
    eeq_t_offset: float = coef(
        29.0,
        "degC",
        "temperature offset of the equilibrium evaporation (TD + 29)",
        _pt(904, "EEQ = SLANG*(2.04E-4-1.83E-4*ALBEDO)*(TD+29.0)", RITCHIE),
    )
    alpha: float = coef(
        1.1,
        "-",
        "Priestley-Taylor coefficient EO = alpha EEQ (also the offset of the hot branch)",
        _pt(905, "EO = EEQ*1.1", RITCHIE, note="line 908 uses the same 1.1, so EO is continuous at 35 degC"),
    )
    hot_threshold: float = coef(
        35.0,
        "degC",
        "TMAX above which the advection correction raises EO",
        _pt(907, "IF (TMAX .GT. 35.0) THEN", RITCHIE, note="line 908 uses the same 35.0 as the offset"),
    )
    hot_slope: float = coef(
        0.05,
        "degC-1",
        "increase of EO / EEQ per degC of TMAX above the hot threshold",
        _pt(908, "EO = EEQ*((TMAX-35.0)*0.05+1.1)", RITCHIE),
    )
    cold_threshold: float = coef(
        5.0,
        "degC",
        "TMAX below which the cold correction lowers EO",
        _pt(909, "ELSE IF (TMAX .LT. 5.0) THEN", RITCHIE),
    )
    cold_factor: float = coef(
        0.01,
        "-",
        "EO / EEQ factor of the cold branch",
        _pt(910, "EO = EEQ*0.01*EXP(0.18*(TMAX+20.0))", RITCHIE),
    )
    cold_exp: float = coef(
        0.18,
        "degC-1",
        "exponential rate of the cold branch",
        _pt(910, "EO = EEQ*0.01*EXP(0.18*(TMAX+20.0))", RITCHIE),
    )
    cold_offset: float = coef(
        20.0,
        "degC",
        "temperature offset of the cold branch",
        _pt(910, "EO = EEQ*0.01*EXP(0.18*(TMAX+20.0))", RITCHIE),
    )
    eo_floor: float = coef(
        1.0e-4,
        "mm d-1",
        "floor of the potential evapotranspiration",
        _pt(914, "EO = MAX(EO,0.0001)"),
    )


class PETCoefficients(Coefficients):
    """The coefficient sets of the PET processes (``PETSiteParams.coefficients``)."""

    asce: ASCECoefficients = eqx.field(default_factory=ASCECoefficients)
    pt: PTCoefficients = eqx.field(default_factory=PTCoefficients)


#: the ASCE-EWRI (2005) coefficients (with the REF_ET.FOR variant constants)
ASCE_2005 = ASCECoefficients()
#: the DSSAT-CSM v4.8.6.0 PETPT coefficients
DSSAT_PT = PTCoefficients()
#: the default sets together (what the processes use when ``PETSiteParams.coefficients`` is None)
PET_COEFFICIENTS = PETCoefficients(asce=ASCE_2005, pt=DSSAT_PT)


def coefficient_table(tree: Any = PET_COEFFICIENTS) -> list[dict[str, Any]]:
    """One row per PET coefficient (:func:`agrijax.core.coefficients.coefficient_table`)."""
    return _coefficient_table(tree)
