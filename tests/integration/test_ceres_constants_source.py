"""Every named constant and coefficient of ``ceres_maize.constants`` stands in the DSSAT-CSM source
line it cites.

The coefficients (:class:`SolarCoefficients`, :class:`XstageCoefficients`) carry file, line and
statement in their :class:`~agrijax.core.coefficients.Provenance`; the codes and conversions are
plain module constants, cited here with the line that sets them. For each, the cited line of
DSSAT-CSM v4.8.6.0 (the tree the local ``dscsm048`` was built from) must hold the quoted statement
(case and whitespace aside, trailing ``!`` comments dropped) and the value must be one of the
numeric literals of that statement.

Skips (``allow_skip``) when the DSSAT source tree is absent (it is not part of the repository).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from agrijax.core.coefficients import coefficient_table
from agrijax.processes.crop.ceres_maize import constants as k

DSSAT_ENGINE = Path(
    os.environ.get("AGRI_JAX_DSSAT", "~/AFSoil/Formal_Analysis/02_DSSAT/dssat_engine")
).expanduser()
SOURCE = DSSAT_ENGINE / "source"
_NUM = re.compile(r"(?<![A-Za-z_0-9])(\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+|\d+)")
_PHENOL = "Plant/CERES-Maize/MZ_PHENOL.for"
_GROSUB = "Plant/CERES-Maize/MZ_GROSUB.for"
_ROOTS = "Plant/CERES-Maize/MZ_ROOTS.for"

pytestmark = [
    pytest.mark.allow_skip(reason="needs the local DSSAT-CSM v4.8.6.0 source tree (not in the repository)"),
    pytest.mark.skipif(not SOURCE.is_dir(), reason=f"DSSAT-CSM source not found under {SOURCE}"),
]

#: (name, value, file, line, statement) of the named constants
CONSTANTS = [
    ("ISTAGE_JUVENILE", k.ISTAGE_JUVENILE, _PHENOL, 632, "ISTAGE = 1"),
    ("ISTAGE_END_JUVENILE", k.ISTAGE_END_JUVENILE, _PHENOL, 676, "ISTAGE = 2"),
    ("ISTAGE_TASSEL_INIT", k.ISTAGE_TASSEL_INIT, _PHENOL, 731, "ISTAGE = 3"),
    ("ISTAGE_END_LEAF_GROWTH", k.ISTAGE_END_LEAF_GROWTH, _PHENOL, 775, "ISTAGE = 4"),
    ("ISTAGE_EFG", k.ISTAGE_EFG, _PHENOL, 848, "ISTAGE = 5"),
    ("ISTAGE_MATURITY", k.ISTAGE_MATURITY, _PHENOL, 875, "ISTAGE = 6"),
    ("ISTAGE_SOWING", k.ISTAGE_SOWING, _PHENOL, 351, "ISTAGE = 7"),
    ("ISTAGE_GERMINATION", k.ISTAGE_GERMINATION, _PHENOL, 534, "ISTAGE = 8"),
    ("ISTAGE_EMERGENCE", k.ISTAGE_EMERGENCE, _PHENOL, 587, "ISTAGE = 9"),
    ("ISTAGE_AFTER_MATURITY", k.ISTAGE_AFTER_MATURITY, _PHENOL, 899, "ISTAGE = 10"),
    ("CROP_STATUS_MATURE", k.CROP_STATUS_MATURE, _PHENOL, 897, "CropStatus = 1"),
    ("CROP_STATUS_NO_GERMINATION", k.CROP_STATUS_NO_GERMINATION, _PHENOL, 576, "CropStatus = 12"),
    ("CROP_STATUS_NO_EMERGENCE", k.CROP_STATUS_NO_EMERGENCE, _PHENOL, 624, "CropStatus = 13"),
    ("CROP_STATUS_COLD", k.CROP_STATUS_COLD, _GROSUB, 1663, "CropStatus = 32"),
    ("CROP_STATUS_DROUGHT", k.CROP_STATUS_DROUGHT, _GROSUB, 1706, "CropStatus = 33"),
    ("XSTAGE_SEASINIT", k.XSTAGE_SEASINIT, _PHENOL, 352, "XSTAGE = 0.1"),
    ("MDATE_NONE", -k.MDATE_NONE, _PHENOL, 353, "MDATE = -99"),
    ("MG_PER_G", k.MG_PER_G, _PHENOL, 810, "PSKER = SUMP*1000.0/IDURP*3.4/5.0"),
    ("G_PER_MG", k.G_PER_MG, _GROSUB, 1468, "GROGRN = RGFILL*GPP*G3*0.001*(0.45+0.55*SWFAC)"),
    ("RLV_PRECISION", k.RLV_PRECISION, _ROOTS, 202, "RLV(L) = REAL(INT(RLV(L)*1000.))/1000."),
    ("PAIR_MEAN_DIVISOR", k.PAIR_MEAN_DIVISOR, _PHENOL, 505, "DTT = (TMAX+TMIN)/2.0 - TBASE"),
    ("PAIR_MEAN_WEIGHT", k.PAIR_MEAN_WEIGHT, _GROSUB, 1022, "TEMPM = (TMAX + TMIN)*0.5"),
    ("HOURS_PER_HALF_DAY", k.HOURS_PER_HALF_DAY, "Weather/SOLAR.for", 133, "DAYL = 12.0 + 24.0*ASIN(SOC)/PI"),
    (
        "DEG_PER_HALF_TURN",
        k.DEG_PER_HALF_TURN,
        "Weather/SOLAR.for",
        121,
        "PARAMETER (PI=3.14159, RAD=PI/180.0)",
    ),
    (
        "FULL_TURN_PER_PI",
        k.FULL_TURN_PER_PI,
        "Weather/SOLAR.for",
        126,
        "DEC = -23.45 * COS(2.0*PI*(DOY+10.0)/365.0)",
    ),
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()


_LINES: dict[str, list[str]] = {}


def _code(fname: str, lineno: int) -> str:
    """The cited fixed-form Fortran line without its trailing ``!`` comment."""
    if fname not in _LINES:
        _LINES[fname] = (SOURCE / fname).read_text(errors="replace").splitlines()
    return _LINES[fname][lineno - 1].split("!")[0]


def _check(value: float, fname: str, line: int, statement: str) -> None:
    code = _code(fname, line)
    assert _norm(statement) in _norm(code), f"{fname}:{line} is {code.strip()!r}, not {statement!r}"
    nums = [float(x) for x in _NUM.findall(statement)]
    assert any(abs(float(value)) == n for n in nums), (value, statement)


ROWS = [*coefficient_table(k.SOLAR_COEFFICIENTS), *coefficient_table(k.XSTAGE_COEFFICIENTS)]


@pytest.mark.parametrize("row", ROWS, ids=[r["path"] for r in ROWS])
def test_coefficient_cites_its_source_line(row):
    assert row["ref_version"] == "dssat-4.8.6.0"
    assert not row["calibratable"]
    _check(row["value"], row["file"], row["line"], row["statement"])


@pytest.mark.parametrize(
    ("name", "value", "fname", "line", "statement"), CONSTANTS, ids=[c[0] for c in CONSTANTS]
)
def test_constant_cites_its_source_line(name, value, fname, line, statement):
    _check(value, fname, line, statement)


def test_every_constant_is_cited():
    """Every public code and conversion of the module appears in :data:`CONSTANTS`."""
    public = {n for n in k.__all__ if n.isupper() and isinstance(getattr(k, n), (int, float))}
    not_cited = public - {c[0] for c in CONSTANTS} - {"DEN_MIN", "YRDOY_SCALE"}  # a guard, a date format
    assert not not_cited, sorted(not_cited)
