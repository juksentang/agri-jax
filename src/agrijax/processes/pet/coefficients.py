"""The coefficients of the PET kernels, declared once with unit, meaning and provenance.

Every number of the Shuttleworth-Wallace (:mod:`.shuttleworth_wallace`), ASCE reference ET
(:mod:`.penman_monteith`) and Priestley-Taylor (:mod:`.priestley_taylor`) equations that is not an
input, a forcing or a parameter read from the reference model's input files is a field below,
declared with :func:`agrijax.core.coefficients.coef`:

* :class:`SWCoefficients` (RZWQM2 ``Rzpet.for``, reference ``rzwqm2-4.6``), grouped by routine:
  :class:`EconstCoefficients` (psychrometrics), :class:`MaxswCoefficients` (clear-sky
  radiation), :class:`AlbedoCoefficients`, :class:`WindCoefficients` (wind-height adjustment),
  :class:`ResistCoefficients` (``RESISThr``), :class:`NetradCoefficients` and
  :class:`PotevpCoefficients` (the daily driver ``POTEVPHR``). RZWQM2 source carries no licence file: each
  field records file, line and routine of the reference source and the published equation
  (paper), never the source statement.
* :class:`ASCECoefficients` (ASCE-EWRI 2005, reference ``asce-ewri-2005``), with the three
  constants that RZWQM2's ``REF_ET.FOR`` sets differently (``variant="rzwqm"``) cited from that
  file.
* :class:`PTCoefficients` (DSSAT-CSM v4.8.6.0 ``SPAM/PET.for`` ``PETPT``, BSD-3: the Fortran
  statement is quoted).

Physical constants (Stefan-Boltzmann, von Karman, gravity, the gas constant of air, the specific
heat of air, the molecular-weight ratio of water vapour and air) carry the value the reference
model sets, a ``note`` with the physical value, and ``calibrate=False``; astronomical and
geometric constants (hours per radian, solar noon) likewise. Every other coefficient is
calibratable.

Defaults are Python floats, so a run with the default sets (:data:`RZWQM_SW`,
:data:`ASCE_2005`, :data:`DSSAT_PT`, all in :data:`PET_COEFFICIENTS`) traces exactly the literals
the equations had before they were named. To calibrate or differentiate a coefficient, pass a set
with array leaves (``RZWQM_SW.as_arrays()``) to the kernel's ``coefficients`` argument, or put a
:class:`PETCoefficients` into ``PETSiteParams.coefficients`` for the processes.

Not coefficients, and therefore not declared here: unit conversions (the named constants of
:mod:`agrijax.core.units` and the SI-prefix constants of the kernel modules), arithmetic means
(``_HALF``), the 10-point Gauss-Legendre nodes and weights of the clear-sky quadrature (exact
numerical constants of the quadrature rule), integer exponents, and the numerical guards the port
adds for finite gradients (:func:`agrijax.core.coefficients.numerical_guard`).

Sources of the published equations cited below: Shuttleworth, W.J. and Wallace, J.S. (1985),
Q. J. R. Meteorol. Soc. 111, 839-855; Shuttleworth, W.J. and Gurney, R.J. (1990), Q. J. R.
Meteorol. Soc. 116, 497-519; Farahani, H.J. and Ahuja, L.R. (1996), Trans. ASAE 39(6), 2051-2064;
Ahuja, L.R. et al. (eds.) (2000), Root Zone Water Quality Model, Water Resources Publications;
ASCE-EWRI (2005), The ASCE Standardized Reference Evapotranspiration Equation; Allen, R.G. et al.
(1998), FAO Irrigation and Drainage Paper 56; Ritchie, J.T. (1972), Water Resour. Res. 8,
1204-1213; Priestley, C.H.B. and Taylor, R.J. (1972), Mon. Weather Rev. 100, 81-92.
"""

from __future__ import annotations

import math
from typing import Any

import equinox as eqx

from agrijax.core.coefficients import Coefficients, Provenance, coef
from agrijax.core.coefficients import coefficient_table as _coefficient_table

__all__ = [
    "ASCE_2005",
    "DSSAT_PT",
    "PET_COEFFICIENTS",
    "RZWQM_SW",
    "ASCECoefficients",
    "AlbedoCoefficients",
    "EconstCoefficients",
    "MaxswCoefficients",
    "NetradCoefficients",
    "PETCoefficients",
    "PTCoefficients",
    "PotevpCoefficients",
    "ResistCoefficients",
    "SWCoefficients",
    "WindCoefficients",
    "coefficient_table",
]

#: registry references of the three PET processes
REF_RZWQM = "rzwqm2-4.6"
REF_ASCE = "asce-ewri-2005"
REF_DSSAT = "dssat-4.8.6.0"
#: reference source files (relative to the reference source trees)
RZPET = "RZWQM/Rzpet.for"
REFET = "RZWQM/REF_ET.FOR"
PETFOR = "SPAM/PET.for"

SW1985 = "Shuttleworth and Wallace (1985)"
SG1990 = "Shuttleworth and Gurney (1990)"
FA1996 = "Farahani and Ahuja (1996)"
RZWQM_BOOK = "Ahuja et al. (2000)"
ASCE = "ASCE-EWRI (2005)"
FAO56 = "Allen et al. (1998), FAO-56"
RITCHIE = "Ritchie (1972)"


def _rz(
    line: int, routine: str, paper: str = RZWQM_BOOK, *, equation: str = "", note: str = "", file: str = RZPET
) -> Provenance:
    """RZWQM2 provenance: file, line and routine of the reference source plus the published equation."""
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
# Shuttleworth-Wallace (RZWQM2 Rzpet.for)
# ================================================================================================
class EconstCoefficients(Coefficients):
    """``ECONST``: saturation vapour pressure, psychrometric constant, air density, latent heat."""

    svp_a: float = coef(
        16.78,
        "-",
        "linear coefficient of T in the exponent of the saturation vapour pressure (kPa)",
        _rz(193, "ECONST", "Bosen (1960)", note="e_s = exp((a T - b)/(T + c)), T degC"),
    )
    svp_b: float = coef(
        116.9,
        "degC",
        "constant of the exponent of the saturation vapour pressure",
        _rz(193, "ECONST", "Bosen (1960)"),
    )
    svp_c: float = coef(
        237.3,
        "degC",
        "temperature offset of the saturation vapour pressure curve",
        _rz(193, "ECONST", "Bosen (1960)"),
    )
    slope_a: float = coef(
        4098.0,
        "degC",
        "slope of the saturation curve: delta = a e_s(T) / (T + c)^2",
        _rz(213, "ECONST", "Bosen (1960)", note="4098 = b + c a of the Bosen form (116.9 + 237.3 * 16.78)"),
    )
    p0: float = coef(
        101.3,
        "kPa",
        "sea-level atmospheric pressure of the polytropic atmosphere",
        _rz(189, "ECONST", RZWQM_BOOK, note="PARAMETER P0; PA = P0 ((T0 - lapse z)/T0)^(g/(lapse R))"),
    )
    t0: float = coef(
        288.0,
        "K",
        "sea-level air temperature of the polytropic atmosphere",
        _rz(189, "ECONST", RZWQM_BOOK, note="PARAMETER T0"),
    )
    lapse_rate: float = coef(
        0.01,
        "K m-1",
        "temperature lapse rate of the polytropic atmosphere",
        _rz(190, "ECONST", RZWQM_BOOK, note="PARAMETER ETTA (continuation of line 189)"),
    )
    gravity: float = coef(
        9.8,
        "m s-2",
        "gravitational acceleration",
        _rz(189, "ECONST", RZWQM_BOOK, note="PARAMETER GRAV; " + _PHYS + " (standard 9.80665 m s-2)"),
        calibrate=False,
    )
    gas_constant: float = coef(
        286.9,
        "J kg-1 K-1",
        "specific gas constant of dry air",
        _rz(190, "ECONST", RZWQM_BOOK, note="PARAMETER R; " + _PHYS + " (287.05 J kg-1 K-1)"),
        calibrate=False,
    )
    virtual_t_offset: float = coef(
        273.2,
        "K",
        "Kelvin offset of the virtual temperature (the reference writes 273.2, not 273.15)",
        _rz(223, "ECONST", RZWQM_BOOK),
        calibrate=False,
    )
    virtual_vapour: float = coef(
        0.378,
        "-",
        "vapour term of the virtual temperature, 1 - epsilon: Tv = T / (1 - 0.378 e/p)",
        _rz(223, "ECONST", RZWQM_BOOK, note=_PHYS + " (1 - 0.622)"),
        calibrate=False,
    )
    latent_heat_0: float = coef(
        2.501,
        "MJ kg-1",
        "latent heat of vaporisation at 0 degC",
        _rz(233, "ECONST", FAO56, equation="3-1", note="lambda = 2.501 - 2.361e-3 T (Harrison 1963)"),
    )
    latent_heat_slope: float = coef(
        2.361e-3,
        "MJ kg-1 degC-1",
        "decrease of the latent heat of vaporisation with temperature",
        _rz(233, "ECONST", FAO56, equation="3-1"),
    )
    mw_ratio: float = coef(
        0.622,
        "-",
        "ratio of the molecular weights of water vapour and dry air (epsilon)",
        _rz(238, "ECONST", FAO56, note=_PHYS + " (18.015 / 28.966)"),
        calibrate=False,
    )
    cp_air: float = coef(
        1.013e-3,
        "MJ kg-1 degC-1",
        "specific heat of moist air at constant pressure",
        _rz(189, "ECONST", FAO56, note="PARAMETER CP (also POTEVPHR line 1798); " + _PHYS),
        calibrate=False,
        fortran_name="CP",
    )


class MaxswCoefficients(Coefficients):
    """``MAXSW`` / ``CSRAD``: extraterrestrial and clear-sky shortwave radiation (horizontal)."""

    solar_const_hourly: float = coef(
        4.9212,
        "MJ m-2 h-1",
        "solar constant (1367 W m-2) per hour",
        _rz(627, "MAXSW", "Swift (1976)", note="PARAMETER W"),
        fortran_name="W",
    )
    turbidity: float = coef(
        3.5,
        "-",
        "atmospheric turbidity factor of the clear-sky direct beam",
        _rz(627, "MAXSW", "Eagleson (1970)", note="PARAMETER B (continued on line 628), passed to CSRAD"),
        fortran_name="B",
    )
    min_sin_altitude: float = coef(
        4.2622e-3,
        "-",
        "sine of the solar altitude below which the direct beam is zero",
        _rz(118, "CSRAD", "Eagleson (1970)"),
    )
    scatter_a: float = coef(
        0.128,
        "-",
        "molecular scattering coefficient at unit air mass",
        _rz(123, "CSRAD", "Eagleson (1970)", note="A1 = a - b log10(air mass)"),
    )
    scatter_b: float = coef(
        0.054,
        "-",
        "decrease of the scattering coefficient per decade of air mass",
        _rz(123, "CSRAD", "Eagleson (1970)"),
    )
    eccentricity: float = coef(
        0.0167,
        "-",
        "orbital eccentricity: E = 1 - e cos(w (J - J_p)), solar constant / E^2",
        _rz(640, "MAXSW", "Swift (1976)"),
    )
    orbit_freq: float = coef(
        0.0172,
        "rad d-1",
        "angular frequency of the orbit in the eccentricity term",
        _rz(640, "MAXSW", "Swift (1976)"),
    )
    perihelion_day: float = coef(
        3.0,
        "d",
        "day of year of the perihelion",
        _rz(640, "MAXSW", "Swift (1976)"),
    )
    decl_amp: float = coef(
        0.39785,
        "-",
        "sine of the obliquity in the declination formula",
        _rz(648, "MAXSW", RZWQM_BOOK, note="RZ-SHAW declination: asin(a sin(b + c J + d sin(e + f J)))"),
    )
    decl_phase: float = coef(
        4.868961,
        "rad",
        "phase of the declination formula",
        _rz(648, "MAXSW", RZWQM_BOOK),
    )
    decl_freq: float = coef(
        0.017203,
        "rad d-1",
        "angular frequency of the declination formula",
        _rz(648, "MAXSW", RZWQM_BOOK),
    )
    decl_ecc_amp: float = coef(
        0.033446,
        "rad",
        "amplitude of the eccentricity correction of the declination",
        _rz(648, "MAXSW", RZWQM_BOOK),
    )
    decl_ecc_phase: float = coef(
        6.224111,
        "rad",
        "phase of the eccentricity correction of the declination",
        _rz(649, "MAXSW", RZWQM_BOOK, note="continuation of line 648"),
    )
    decl_ecc_freq: float = coef(
        0.017202,
        "rad d-1",
        "angular frequency of the eccentricity correction of the declination",
        _rz(649, "MAXSW", RZWQM_BOOK, note="continuation of line 648"),
    )
    hours_per_radian: float = coef(
        12.0 / math.pi,
        "h rad-1",
        "hours per radian of the Earth's rotation, 12/pi",
        _rz(
            627,
            "MAXSW",
            RZWQM_BOOK,
            note="PARAMETER PPCNST = 12/PI; the reference's PI is 3.141592654, "
            "the port uses math.pi; astronomical constant, not calibrated",
        ),
        calibrate=False,
        fortran_name="PPCNST",
    )
    diffuse_fraction: float = coef(
        0.91,
        "-",
        "fraction of the extraterrestrial radiation reaching the ground on a clear day",
        _rz(877, "MAXSW", "Shaffer and Larson (1982)", note="RDIF = max(0, (a RP - RCHD) f)"),
    )
    diffuse_share: float = coef(
        0.5,
        "-",
        "share of the scattered clear-sky radiation that reaches the ground as diffuse",
        _rz(877, "MAXSW", "Shaffer and Larson (1982)"),
    )
    solar_noon_hour: float = coef(
        12.0,
        "h",
        "solar noon (hour angle 0)",
        _rz(
            812, "MAXSW", RZWQM_BOOK, note="sunrise/sunset hour = 12 + hour angle / (rad per hour); geometric"
        ),
        calibrate=False,
    )
    radians_per_hour: float = coef(
        0.2618,
        "rad h-1",
        "hour angle per hour, pi/12 as the reference rounds it",
        _rz(812, "MAXSW", RZWQM_BOOK, note="also line 813; astronomical constant, not calibrated"),
        calibrate=False,
    )


class AlbedoCoefficients(Coefficients):
    """``ALBSWS`` (soil albedo) and the residue-albedo block of ``POTEVPHR``."""

    roughness_reduction: float = coef(
        0.08,
        "cm-1",
        "relative reduction of the soil albedo per cm of random roughness",
        _rz(40, "ALBSWS", RZWQM_BOOK, note="PARAMETER RR8"),
        fortran_name="RR8",
    )
    residue_aged_ratio: float = coef(
        1.06,
        "-",
        "residue albedo relative to the dry-soil albedo after full weathering",
        _rz(1936, "POTEVPHR", RZWQM_BOOK, note="AR = A0 (r + (ARI/A0 - r) exp(-k RESAGE))"),
    )
    residue_ageing_rate: float = coef(
        0.0255,
        "d-1",
        "rate of the residue albedo's approach to its weathered value",
        _rz(1936, "POTEVPHR", RZWQM_BOOK),
    )
    wet_residue_factor: float = coef(
        0.75,
        "-",
        "residue albedo of wet residue relative to dry residue",
        _rz(1938, "POTEVPHR", RZWQM_BOOK),
    )


class WindCoefficients(Coefficients):
    """``POTEVPHR`` wind block: wind run moved from the anemometer to a height above the canopy."""

    airport_height: float = coef(
        10.0,
        "m",
        "anemometer height from which the station is taken as an airport (40 cm cover)",
        _rz(1845, "POTEVPHR", FA1996),
    )
    ref_offset_airport: float = coef(
        1.33,
        "m",
        "reference height above the canopy's 2/3 h, airport anemometer",
        _rz(1849, "POTEVPHR", FA1996, note="XWNEW = offset + (2/3) h"),
    )
    ref_offset_micro: float = coef(
        1.93,
        "m",
        "reference height above the canopy's 2/3 h, micro-tower anemometer (10 cm cover)",
        _rz(1861, "POTEVPHR", FA1996),
    )
    ref_bare_airport: float = coef(
        1.6,
        "m",
        "reference height without a canopy, airport anemometer",
        _rz(1853, "POTEVPHR", FA1996),
    )
    ref_bare_micro: float = coef(
        2.0,
        "m",
        "reference height without a canopy, micro-tower anemometer",
        _rz(1865, "POTEVPHR", FA1996),
    )
    canopy_height_factor: float = coef(
        2.0 / 3.0,
        "-",
        "zero-plane displacement of the target canopy as a fraction of its height",
        _rz(1798, "POTEVPHR", FA1996, note="PARAMETER TWOTRD = 2/3"),
        fortran_name="TWOTRD",
    )
    canopy_roughness_factor: float = coef(
        0.123,
        "-",
        "roughness length of the target canopy as a fraction of its height",
        _rz(1798, "POTEVPHR", FA1996, note="PARAMETER PEN123 (continued on line 1799)"),
        fortran_name="PEN123",
    )
    bare_roughness: float = coef(
        0.01,
        "m",
        "roughness length of the bare target surface",
        _rz(1854, "POTEVPHR", FA1996, note="also line 1866"),
    )
    airport_displacement: float = coef(
        0.27,
        "m",
        "zero-plane displacement of the 40-cm airport cover",
        _rz(1851, "POTEVPHR", FA1996),
    )
    airport_roughness: float = coef(
        0.05,
        "m",
        "roughness length of the 40-cm airport cover",
        _rz(1851, "POTEVPHR", FA1996),
    )
    micro_displacement: float = coef(
        0.07,
        "m",
        "zero-plane displacement of the 10-cm micro-tower cover",
        _rz(1863, "POTEVPHR", FA1996),
    )
    micro_roughness: float = coef(
        0.0123,
        "m",
        "roughness length of the 10-cm micro-tower cover",
        _rz(1863, "POTEVPHR", FA1996),
    )


class ResistCoefficients(Coefficients):
    """``RESISThr``: aerodynamic, canopy, soil-surface and residue resistances (daily, no stubble)."""

    von_karman: float = coef(
        0.41,
        "-",
        "von Karman constant",
        _rz(2511, "RESISThr", SW1985, note="PARAMETER K; " + _PHYS),
        calibrate=False,
        fortran_name="K",
    )
    z0_soil: float = coef(
        0.01,
        "m",
        "effective roughness length of the bare soil surface",
        _rz(2511, "RESISThr", SW1985, note="PARAMETER Z0P"),
        fortran_name="Z0P",
    )
    eddy_decay: float = coef(
        2.5,
        "-",
        "eddy-diffusivity decay constant within the canopy",
        _rz(2511, "RESISThr", SW1985, equation="41-43", note="PARAMETER N; Shuttleworth and Gurney (1990)"),
        fortran_name="N",
    )
    drag_coeff: float = coef(
        0.07,
        "-",
        "mean drag coefficient of the leaves (X = cd LAI)",
        _rz(2511, "RESISThr", SG1990, note="PARAMETER CD"),
        fortran_name="CD",
    )
    leaf_boundary_resistance: float = coef(
        10.0,
        "s m-1",
        "mean boundary-layer resistance of a leaf (rac = rb / (2 LAI))",
        _rz(2512, "RESISThr", SW1985, note="PARAMETER RB (continuation of line 2511)"),
        fortran_name="RB",
    )
    min_plant_height: float = coef(
        0.05,
        "m",
        "floor of the plant height in the canopy resistances",
        _rz(2563, "RESISThr", FA1996),
    )
    min_lai: float = coef(
        0.05,
        "m2 m-2",
        "floor of the effective LAI (mean of green and total LAI)",
        _rz(2569, "RESISThr", FA1996),
    )
    displacement_pref: float = coef(
        0.63,
        "-",
        "preferred zero-plane displacement of the canopy as a fraction of its height",
        _rz(2578, "RESISThr", SW1985, note="Shuttleworth and Gurney (1990) preferred value"),
    )
    roughness_pref: float = coef(
        0.13,
        "-",
        "preferred roughness length of the canopy as a fraction of its height",
        _rz(2579, "RESISThr", SW1985, note="Shuttleworth and Gurney (1990) preferred value"),
    )
    displacement_factor: float = coef(
        1.1,
        "-",
        "displacement d = f h ln(1 + X^p)",
        _rz(2590, "RESISThr", SG1990),
    )
    displacement_exp: float = coef(
        0.25,
        "-",
        "exponent p of X in the zero-plane displacement",
        _rz(2590, "RESISThr", SG1990),
    )
    sparse_x_limit: float = coef(
        0.2,
        "-",
        "X = cd LAI up to which the sparse-canopy roughness applies",
        _rz(2592, "RESISThr", SG1990),
    )
    roughness_factor: float = coef(
        0.3,
        "-",
        "roughness factor: z0 = z0s + 0.3 h sqrt(X) (sparse), 0.3 h (1 - d/h) (dense)",
        _rz(2593, "RESISThr", SG1990, note="also line 2595"),
    )
    rac_lai_factor: float = coef(
        2.0,
        "-",
        "sides of a leaf in the bulk boundary-layer resistance rb / (2 LAI)",
        _rz(2598, "RESISThr", SW1985),
    )
    rsc_low_lai: float = coef(
        2.0,
        "m2 m-2",
        "effective LAI below which the bulk stomatal resistance is RST / (2 LAI)",
        _rz(2604, "RESISThr", FA1996, note="CERES-style active-leaf fraction (Jagtap and Jones)"),
    )
    rsc_low_divisor: float = coef(
        2.0,
        "-",
        "divisor of RST LAI-1 below the low-LAI limit",
        _rz(2606, "RESISThr", FA1996),
    )
    rsc_high_lai: float = coef(
        3.0,
        "m2 m-2",
        "effective LAI above which the bulk stomatal resistance is RST / 3",
        _rz(2607, "RESISThr", FA1996),
    )
    rsc_high_divisor: float = coef(
        3.0,
        "m2 m-2",
        "active LAI of a dense canopy (RST / 3)",
        _rz(2609, "RESISThr", FA1996),
    )
    ras_height_fraction: float = coef(
        0.5,
        "-",
        "second height of the neutral log profile of ras, as a fraction of the reference height",
        _rz(2646, "RESISThr", SW1985, note="ras = ln(z/z0) ln(f z/z0) / (k^2 u); also line 2769"),
    )
    residue_mass_threshold: float = coef(
        1.0e-6,
        "kg ha-1",
        "flat residue mass above which the residue layer exists",
        _rz(2675, "RESISThr", FA1996),
    )
    residue_cover_default: float = coef(
        1.32,
        "-",
        "randomness factor CRES of the residue cover when the input file gives <= 0",
        _rz(2680, "RESISThr", FA1996, note="the cover of rzwqm.dat is an input (residue_cover_factor)"),
        fortran_name="CRES",
    )
    residue_cover_coeff: float = coef(
        1.27e-2,
        "g cm-2 t-1 ha",
        "exposed soil fraction CS = exp(-CRES a M / (d rho)), M in t ha-1",
        _rz(2687, "RESISThr", FA1996),
    )
    residue_bulk_ratio: float = coef(
        0.2,
        "-",
        "bulk density of the residue layer as a fraction of the residue specific density",
        _rz(2698, "RESISThr", FA1996),
    )
    residue_roughness_ratio: float = coef(
        0.197,
        "-",
        "roughness length of the residue layer as a fraction of its thickness",
        _rz(2706, "RESISThr", FA1996),
    )
    porosity_low: float = coef(
        0.5,
        "-",
        "residue porosity at or below which the default porosity is used",
        _rz(2714, "RESISThr", FA1996),
    )
    porosity_high: float = coef(
        0.95,
        "-",
        "residue porosity above which the default porosity is used",
        _rz(2714, "RESISThr", FA1996),
    )
    porosity_default: float = coef(
        0.8,
        "-",
        "default porosity of the residue layer",
        _rz(2715, "RESISThr", FA1996),
    )
    u2_height: float = coef(
        2.0,
        "m",
        "height above the residue of the wind speed in the residue resistance",
        _rz(2718, "RESISThr", FA1996),
    )
    rsr_tortuosity: float = coef(
        1.1,
        "-",
        "tortuosity of the vapour path through the residue layer",
        _rz(
            2719,
            "RESISThr",
            FA1996,
            note="rsr = f HR / (D (1 + a max(0, T - Tr)) (1 + b rhob^-c u2) porosity)",
        ),
    )
    vapour_diffusivity: float = coef(
        2.12e-5,
        "m2 s-1",
        "molecular diffusivity of water vapour in air at the reference temperature",
        _rz(2719, "RESISThr", FA1996, note=_PHYS),
        calibrate=False,
    )
    diffusivity_temp_coeff: float = coef(
        0.007,
        "degC-1",
        "relative increase of the vapour diffusivity per degC above the reference",
        _rz(2719, "RESISThr", FA1996),
    )
    diffusivity_ref_temp: float = coef(
        20.0,
        "degC",
        "reference temperature of the vapour diffusivity",
        _rz(2719, "RESISThr", FA1996),
    )
    residue_wind_coeff: float = coef(
        1.25e-3,
        "g^1.79 cm^-5.37 s m-1",
        "wind enhancement of the vapour transfer through the residue layer",
        _rz(2720, "RESISThr", FA1996, note="continuation of line 2719"),
    )
    residue_wind_exp: float = coef(
        1.79,
        "-",
        "exponent of the residue bulk density in the wind enhancement (rhob^-c)",
        _rz(2720, "RESISThr", FA1996, note="continuation of line 2719"),
    )
    residue_diameter_corn: float = coef(
        1.0,
        "cm",
        "mean diameter of corn residue",
        _rz(2522, "RESISThr", FA1996, note="DATA RDIA (1)"),
    )
    residue_diameter_soybean: float = coef(
        0.5,
        "cm",
        "mean diameter of soybean residue",
        _rz(2522, "RESISThr", FA1996, note="DATA RDIA (2)"),
    )
    residue_diameter_wheat: float = coef(
        0.25,
        "cm",
        "mean diameter of wheat residue",
        _rz(2522, "RESISThr", FA1996, note="DATA RDIA (3)"),
    )
    residue_density_corn: float = coef(
        0.15,
        "g cm-3",
        "specific density of corn residue",
        _rz(2521, "RESISThr", FA1996, note="DATA RHORS (1)"),
    )
    residue_density_soybean: float = coef(
        0.17,
        "g cm-3",
        "specific density of soybean residue",
        _rz(2521, "RESISThr", FA1996, note="DATA RHORS (2)"),
    )
    residue_density_wheat: float = coef(
        0.18,
        "g cm-3",
        "specific density of wheat residue",
        _rz(2521, "RESISThr", FA1996, note="DATA RHORS (3)"),
    )


class NetradCoefficients(Coefficients):
    """``NETRAD`` (no plastic mulch): net radiation of the field and of the substrate surfaces."""

    negative_rn_divisor: float = coef(
        3.0,
        "-",
        "divisor of the absorbed shortwave that replaces a negative net radiation",
        _rz(996, "NETRAD", RZWQM_BOOK),
    )
    canopy_absorb_base: float = coef(
        0.5,
        "-",
        "shortwave absorption of the canopy at zero cover (AAC = a + b CCL)",
        _rz(998, "NETRAD", RZWQM_BOOK),
    )
    canopy_absorb_slope: float = coef(
        0.44,
        "-",
        "increase of the canopy shortwave absorption with the canopy cover",
        _rz(998, "NETRAD", RZWQM_BOOK),
    )


class PotevpCoefficients(Coefficients):
    """``POTEVPHR`` daily branch: long-wave radiation, canopy cover, night test, flux units."""

    stefan_boltzmann: float = coef(
        4.903e-9,
        "MJ m-2 K-4 d-1",
        "Stefan-Boltzmann constant (daily)",
        _rz(1832, "POTEVPHR", FAO56, note=_PHYS + " (CODATA 5.670374e-8 W m-2 K-4 = 4.899e-9)"),
        calibrate=False,
        fortran_name="SIGMA",
    )
    emissivity_a: float = coef(
        0.39,
        "-",
        "net emissivity of the surface-atmosphere at zero vapour pressure (Brunt form)",
        _rz(1978, "POTEVPHR", RZWQM_BOOK, note="RB0 = (a - b sqrt(ed)) sigma (Tmax_K^4 + Tmin_K^4)/2"),
    )
    emissivity_b: float = coef(
        0.158,
        "kPa-0.5",
        "decrease of the net emissivity with the square root of the vapour pressure",
        _rz(1978, "POTEVPHR", RZWQM_BOOK),
    )
    lw_a_arid: float = coef(
        1.2,
        "-",
        "cloudiness coefficient a of RNL = -(a Rs/Rso + b) RB0, arid zone",
        _rz(1816, "POTEVPHR", RZWQM_BOOK, note="DATA al(1)"),
    )
    lw_b_arid: float = coef(
        -0.2,
        "-",
        "cloudiness coefficient b, arid zone",
        _rz(1816, "POTEVPHR", RZWQM_BOOK, note="DATA bl(1)"),
    )
    lw_a_semiarid: float = coef(
        1.1,
        "-",
        "cloudiness coefficient a, semi-arid zone",
        _rz(1816, "POTEVPHR", RZWQM_BOOK, note="DATA al(2)"),
    )
    lw_b_semiarid: float = coef(
        -0.1,
        "-",
        "cloudiness coefficient b, semi-arid zone",
        _rz(1816, "POTEVPHR", RZWQM_BOOK, note="DATA bl(2)"),
    )
    lw_a_humid: float = coef(
        1.0,
        "-",
        "cloudiness coefficient a, humid zone",
        _rz(1816, "POTEVPHR", RZWQM_BOOK, note="DATA al(3)"),
    )
    lw_b_humid: float = coef(
        0.0,
        "-",
        "cloudiness coefficient b, humid zone",
        _rz(1816, "POTEVPHR", RZWQM_BOOK, note="DATA bl(3)"),
    )
    canopy_extinction: float = coef(
        0.594,
        "-",
        "extinction coefficient of the canopy cover CCL = 1 - exp(-k TLAI)",
        _rz(1960, "POTEVPHR", FA1996),
    )
    night_radiation: float = coef(
        10.0,
        "W m-2",
        "shortwave below which the canopy resistance takes its night value",
        _rz(
            1957,
            "POTEVPHR",
            RZWQM_BOOK,
            note="applied to the daily total as the reference does (effective threshold 0.036 MJ m-2 d-1)",
        ),
    )
    night_rsc_factor: float = coef(
        10.0,
        "-",
        "night-time canopy resistance relative to the day-time value",
        _rz(1957, "POTEVPHR", RZWQM_BOOK),
    )


class SWCoefficients(Coefficients):
    """Every coefficient of the RZWQM2 Shuttleworth-Wallace PET (``Rzpet.for``), by routine."""

    econst: EconstCoefficients = eqx.field(default_factory=EconstCoefficients)
    maxsw: MaxswCoefficients = eqx.field(default_factory=MaxswCoefficients)
    albedo: AlbedoCoefficients = eqx.field(default_factory=AlbedoCoefficients)
    wind: WindCoefficients = eqx.field(default_factory=WindCoefficients)
    resist: ResistCoefficients = eqx.field(default_factory=ResistCoefficients)
    netrad: NetradCoefficients = eqx.field(default_factory=NetradCoefficients)
    potevp: PotevpCoefficients = eqx.field(default_factory=PotevpCoefficients)


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
    """The coefficient sets of the three PET processes (``PETSiteParams.coefficients``)."""

    sw: SWCoefficients = eqx.field(default_factory=SWCoefficients)
    asce: ASCECoefficients = eqx.field(default_factory=ASCECoefficients)
    pt: PTCoefficients = eqx.field(default_factory=PTCoefficients)


#: the RZWQM2 4.6 Shuttleworth-Wallace coefficients (the defaults)
RZWQM_SW = SWCoefficients()
#: the ASCE-EWRI (2005) coefficients (with the REF_ET.FOR variant constants)
ASCE_2005 = ASCECoefficients()
#: the DSSAT-CSM v4.8.6.0 PETPT coefficients
DSSAT_PT = PTCoefficients()
#: the three default sets together (what the processes use when ``PETSiteParams.coefficients`` is None)
PET_COEFFICIENTS = PETCoefficients(sw=RZWQM_SW, asce=ASCE_2005, pt=DSSAT_PT)


def coefficient_table(tree: Any = PET_COEFFICIENTS) -> list[dict[str, Any]]:
    """One row per PET coefficient (:func:`agrijax.core.coefficients.coefficient_table`)."""
    return _coefficient_table(tree)
