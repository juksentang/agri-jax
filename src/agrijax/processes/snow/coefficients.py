"""Coefficients of the PRMS snowpack of RZWQM2 4.6 (``RZWQM/Snowprms.for``), each declared once.

The values are the literals of the reference source (RZWQM2 4.6, ``Snowprms.for``, the ``SNOWCOMP_PRMS``
interface of Ken Rojas to the PRMS ``snocomp`` module; ``Rzpet.for`` for the sublimation potential;
``Rzday.for`` for the daily call). RZWQM2 source statements are not reproduced: every coefficient
cites the file, line and routine, and the published PRMS documentation of the equation
(Leavesley, Lichty, Troutman and Saindon 1983, *Precipitation-runoff modeling system: user's
manual*, USGS Water-Resources Investigations Report 83-4238, the snowpack energy and mass balance
of Obled and Rosse 1977 and Anderson 1968 as used by PRMS).

The parameters that ``READ_SNOW`` reads from the scenario's ``.sno`` file (densities, albedo
resets, cover, emissivity, the areal depletion curve ...) are not coefficients: they are the
fields of :class:`~agrijax.processes.snow.prms.PrmsSnowParams`.

Several coefficients are written by the reference with fewer digits than the physical value they
stand for (``0.5556`` for 5/9, ``1.27`` for the heat capacity of ice times the centimetres of an
inch, ``1.2`` in ``CALOSS`` where the other routines have ``1.27``). They are kept as written: the
module follows the reference arithmetic, and each is calibratable like any other coefficient.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx

from agrijax.core.coefficients import Coefficients, Provenance, coef
from agrijax.core.coefficients import coefficient_table as _coefficient_table

__all__ = [
    "ALBEDO_DAYS",
    "PRMS_SNOW",
    "AccumulationAlbedo",
    "MeltAlbedo",
    "PrmsSnowCoefficients",
    "coefficient_table",
]

REF = "rzwqm2-4.6"
SNOWPRMS = "RZWQM/Snowprms.for"
RZPET = "RZWQM/Rzpet.for"
RZDAY = "RZWQM/Rzday.for"
PRMS = "Leavesley et al. (1983), PRMS user's manual, USGS WRI 83-4238"
_CONST = "reference arithmetic; a unit or calendar constant, not calibrated"


def _rz(line: int, routine: str, *, file: str = SNOWPRMS, note: str = "") -> Provenance:
    """RZWQM2 4.6 provenance: file, line, routine and the PRMS documentation (no statement)."""
    return Provenance(REF, file=file, line=line, routine=routine, paper=PRMS, note=note)


#: the days-since-snowfall entries of the two albedo decay tables (``SNALBEDO``)
ALBEDO_DAYS: tuple[str, ...] = tuple(f"d{i:02d}" for i in range(1, 16))


def _albedo(table: str, stage: str, i: int, value: float, line: int) -> Any:
    """One row of an albedo table of ``SNALBEDO`` (``line``: the source line of the DATA statement)."""
    return coef(
        value,
        "-",
        f"albedo of old snow {i} day(s) after the last snowfall, {stage} stage (table {table})",
        _rz(line, "SNALBEDO"),
        bounds=(0.0, 1.0),
    )


class AccumulationAlbedo(Coefficients):
    """``ACUM``: albedo against days since the last snowfall in the accumulation stage."""

    d01: float = _albedo("ACUM", "accumulation", 1, 0.80, 691)
    d02: float = _albedo("ACUM", "accumulation", 2, 0.77, 691)
    d03: float = _albedo("ACUM", "accumulation", 3, 0.75, 691)
    d04: float = _albedo("ACUM", "accumulation", 4, 0.72, 691)
    d05: float = _albedo("ACUM", "accumulation", 5, 0.70, 691)
    d06: float = _albedo("ACUM", "accumulation", 6, 0.69, 691)
    d07: float = _albedo("ACUM", "accumulation", 7, 0.68, 691)
    d08: float = _albedo("ACUM", "accumulation", 8, 0.63, 692)
    d09: float = _albedo("ACUM", "accumulation", 9, 0.62, 692)
    d10: float = _albedo("ACUM", "accumulation", 10, 0.61, 692)
    d11: float = _albedo("ACUM", "accumulation", 11, 0.6, 692)
    d12: float = _albedo("ACUM", "accumulation", 12, 0.6, 692)
    d13: float = _albedo("ACUM", "accumulation", 13, 0.6, 692)
    d14: float = _albedo("ACUM", "accumulation", 14, 0.6, 692)
    d15: float = _albedo("ACUM", "accumulation", 15, 0.6, 692)


class MeltAlbedo(Coefficients):
    """``AMLT``: albedo against days since the last snowfall in the melt stage."""

    d01: float = _albedo("AMLT", "melt", 1, 0.72, 693)
    d02: float = _albedo("AMLT", "melt", 2, 0.65, 693)
    d03: float = _albedo("AMLT", "melt", 3, 0.60, 693)
    d04: float = _albedo("AMLT", "melt", 4, 0.58, 693)
    d05: float = _albedo("AMLT", "melt", 5, 0.56, 693)
    d06: float = _albedo("AMLT", "melt", 6, 0.54, 693)
    d07: float = _albedo("AMLT", "melt", 7, 0.52, 693)
    d08: float = _albedo("AMLT", "melt", 8, 0.43, 694)
    d09: float = _albedo("AMLT", "melt", 9, 0.42, 694)
    d10: float = _albedo("AMLT", "melt", 10, 0.41, 694)
    d11: float = _albedo("AMLT", "melt", 11, 0.4, 694)
    d12: float = _albedo("AMLT", "melt", 12, 0.4, 694)
    d13: float = _albedo("AMLT", "melt", 13, 0.4, 694)
    d14: float = _albedo("AMLT", "melt", 14, 0.4, 694)
    d15: float = _albedo("AMLT", "melt", 15, 0.4, 694)


class PrmsSnowCoefficients(Coefficients):
    """Every literal of the RZWQM2 4.6 PRMS snowpack (``SNOWCOMP_PRMS`` and the routines it calls)."""

    # ---------------------------------------------------------------- unit conversions (as written)
    langley_per_mj_m2: float = coef(
        100.0 / 4.186,
        "cal cm-2 MJ-1 m2",
        "MJ m-2 to langley, the reference's 1.0D2/4.186D0 (4.186 J per calorie)",
        _rz(17, "SNOWCOMP_PRMS", note=_CONST),
        calibrate=False,
        fortran_name="CONV1",
    )
    inch_per_cm: float = coef(
        1.0 / 2.54,
        "in cm-1",
        "centimetre to inch, the reference's 1.0D0/2.54D0",
        _rz(17, "SNOWCOMP_PRMS", note=_CONST),
        calibrate=False,
        fortran_name="CONV2",
    )
    cm_per_inch: float = coef(
        2.54,
        "cm in-1",
        "centimetres of an inch: the heat of rain [cal cm-2] per inch and degree (PPT_TO_PACK)",
        _rz(503, "PPT_TO_PACK", note=_CONST),
        calibrate=False,
    )
    celsius_per_fahrenheit: float = coef(
        0.5556,
        "degC degF-1",
        "degree Fahrenheit to Celsius as written (5/9 to four digits), both directions",
        _rz(69, "SNOWCOMP_PRMS", note=_CONST + "; also SNORUN 352-353 and PPT_TO_PACK"),
        calibrate=False,
    )
    fahrenheit_offset: float = coef(
        32.0,
        "degF",
        "0 degC in degrees Fahrenheit",
        _rz(69, "SNOWCOMP_PRMS", note=_CONST + "; also SNORUN 352-353 and PPT_TO_PACK 487"),
        calibrate=False,
    )
    mean_weight: float = coef(
        0.5,
        "-",
        "weight of each extreme in a mean of two temperatures (daily mean, half-day means, "
        "the PPT_TO_PACK rain temperature)",
        _rz(68, "SNOWCOMP_PRMS", note="also SNORUN 387, 398, PPT_TO_PACK 487; Rzday.for 876 (TM)"),
        calibrate=False,
    )
    # ---------------------------------------------------------------- daily call (Rzday.for, Rzpet.for)
    sublimation_potential: float = coef(
        0.011,
        "cm d-1",
        "potential snow sublimation ESN handed to PRMS as its potential ET (constant as written: the "
        "SNOWQE estimate and its cap by the soil evaporation are overwritten)",
        Provenance(
            REF,
            file=RZPET,
            line=1425,
            routine="POTEVP",
            paper=PRMS,
            note="the same in POTEVPHR, Rzpet.for 2383",
        ),
        bounds=(0.0, 1.0),
        fortran_name="ESN",
    )
    reset_doy_north: int = coef(
        200,
        "d",
        "day of year that re-initialises the pack routine (SSTART) at a northern-hemisphere site",
        _rz(869, "PHYSCL", file=RZDAY, note=_CONST),
        static=True,
    )
    reset_doy_south: int = coef(
        10,
        "d",
        "day of year that re-initialises the pack routine (SSTART) at a southern-hemisphere site",
        _rz(870, "PHYSCL", file=RZDAY, note=_CONST),
        static=True,
    )
    # ---------------------------------------------------------------- pack heat (CALIN, CALOSS, ...)
    latent_heat_inch: float = coef(
        203.2,
        "cal cm-2 in-1",
        "latent heat of fusion of one inch of water over a square centimetre (80 cal g-1 x 2.54)",
        _rz(636, "CALIN", note="also CALOSS 584, 595"),
    )
    latent_heat_fusion: float = coef(
        80.0,
        "cal g-1",
        "latent heat of fusion of ice in the rain-on-pack balance",
        _rz(503, "PPT_TO_PACK"),
    )
    ice_heat_inch: float = coef(
        1.27,
        "cal cm-2 in-1 degC-1",
        "heat capacity of one inch of pack water (0.5 cal g-1 degC-1 x 2.54): pack temperature from "
        "the cold content",
        _rz(
            627, "CALIN", note="also PPT_TO_PACK 516, 551, 553, 559; SNOWBAL 853, 873, 876, 884; SNOWEVAP 934"
        ),
    )
    ice_heat_inch_caloss: float = coef(
        1.2,
        "cal cm-2 in-1 degC-1",
        "heat capacity of one inch of pack water as written in CALOSS (1.2, where every other routine "
        "has 1.27)",
        _rz(601, "CALOSS"),
    )
    # ---------------------------------------------------------------- pack conduction (SNORUN)
    conductivity_per_density: float = coef(
        0.0154,
        "cal cm-1 s-1 degC-1",
        "effective thermal conductivity of the pack per unit density (EFFK = 0.0154 PK_DEN)",
        _rz(366, "SNORUN"),
    )
    half_day_seconds_over_pi: float = coef(
        13751.0,
        "s",
        "time constant of the pack's daily temperature wave (43200 s / pi), in CST = PK_DEN sqrt(EFFK 13751)",
        _rz(367, "SNORUN"),
    )
    forced_melt_days: int = coef(
        4,
        "d",
        "days of an isothermal pack after MELT_LOOK that switch the pack to the melt stage (LSO <= 4)",
        _rz(374, "SNORUN"),
        static=True,
    )
    cec_half_day: float = coef(
        0.5,
        "-",
        "share of the monthly convection-condensation coefficient applied in each half-day balance",
        _rz(358, "SNORUN"),
    )
    # ---------------------------------------------------------------- energy balance (SNOWBAL)
    stefan_boltzmann_half_day: float = coef(
        0.585e-7,
        "cal cm-2 K-4",
        "Stefan-Boltzmann constant over a half day (langley per half day and K^4)",
        _rz(809, "SNOWBAL"),
    )
    kelvin_offset: float = coef(
        273.16,
        "K",
        "0 degC in kelvin as written",
        _rz(809, "SNOWBAL", note=_CONST),
        calibrate=False,
    )
    snow_longwave_melting: float = coef(
        325.7,
        "cal cm-2",
        "half-day long-wave emission of a snow surface at 0 degC (the value for air at or above 0 degC)",
        _rz(817, "SNOWBAL"),
    )
    # ---------------------------------------------------------------- albedo (SNALBEDO)
    albedo_new_snow_melt: float = coef(
        0.81, "-", "albedo of new snow in the melt stage", _rz(779, "SNALBEDO"), bounds=(0.0, 1.0)
    )
    albedo_new_snow_accumulation: float = coef(
        0.91, "-", "albedo of new snow in the accumulation stage", _rz(782, "SNALBEDO"), bounds=(0.0, 1.0)
    )
    albedo_days_round: float = coef(
        0.5,
        "d",
        "rounding offset of the snowfall counter to a table row, L = INT(SLST + 0.5)",
        _rz(757, "SNALBEDO", note=_CONST),
        calibrate=False,
    )
    albedo_table_rows: int = coef(
        15,
        "d",
        "rows of the albedo tables ACUM and AMLT (the counter is capped to the last row)",
        _rz(764, "SNALBEDO", note=_CONST + "; also 768, 773"),
        static=True,
    )
    albedo_melt_offset: int = coef(
        12,
        "d",
        "rows by which an old accumulation-stage snow past the ACUM table enters AMLT (L - 12)",
        _rz(767, "SNALBEDO"),
        static=True,
    )
    accumulation: AccumulationAlbedo = eqx.field(default_factory=AccumulationAlbedo)
    melt: MeltAlbedo = eqx.field(default_factory=MeltAlbedo)
    # ---------------------------------------------------------------- snow-covered area (SNOWCOV)
    new_snow_cover_fraction: float = coef(
        0.25,
        "-",
        "fraction of a new snowfall above which the pack keeps full cover (SCRV = PKWE - 0.25 NET_SNOW)",
        _rz(997, "SNOWCOV", note="also 1002"),
    )
    curve_points: float = coef(
        10.0,
        "-",
        "intervals of the areal depletion curve (11 points at FRAC = 0, 0.1, ..., 1)",
        _rz(1010, "SNOWCOV", note=_CONST + "; also 1013"),
        calibrate=False,
    )
    curve_index_offset: float = coef(
        0.2,
        "-",
        "offset of the curve index, IDX = INT(MIN(10 (FRAC + 0.2), 11))",
        _rz(1010, "SNOWCOV", note=_CONST),
        calibrate=False,
    )
    curve_last_point: float = coef(
        11.0,
        "-",
        "last point of the areal depletion curve (the cap of IDX)",
        _rz(1010, "SNOWCOV", note=_CONST),
        calibrate=False,
    )


#: the coefficients as the reference writes them
PRMS_SNOW = PrmsSnowCoefficients()


def coefficient_table(tree: Any = PRMS_SNOW) -> list[dict[str, Any]]:
    """One row per PRMS snow coefficient (:func:`agrijax.core.coefficients.coefficient_table`)."""
    return _coefficient_table(tree)
