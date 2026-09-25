"""Unit constants, named conversions and the parser of the ``field(unit=...)`` strings.

Internal units follow the RZWQM ``.ana`` convention so that comparison with the
Fortran oracle needs no conversion: length in cm, temperature in degC, radiation
in MJ m-2 d-1, mass in kg ha-1, time in days.

Every conversion is a named adapter registered in :data:`ADAPTERS` with its source and
target units (round trips and factors are tested against :func:`parse_unit`). Every numeric
constant of this module is declared in :data:`CONSTANTS` with its unit and meaning: a conversion
factor names the two units it converts between (its value is checked against
:func:`conversion_factor`), a physical constant or model coefficient cites where its value comes
from (paper and equation, or the reference source). These are the only conversion factors and
physical constants process and kernel code may use (lint rule AJ007, :mod:`agrijax.core.lint`).
:func:`parse_unit` makes the unit strings machine-readable (dimension and scale) with no
runtime dependency; it is for tests and tools and never runs inside a traced function.
"""

from __future__ import annotations

import re as _re
from collections.abc import Callable as _Callable
from dataclasses import dataclass as _dataclass
from fractions import Fraction as _Fraction
from typing import Any as _Any

# ---- length ---------------------------------------------------------------
CM_PER_MM: float = 0.1
MM_PER_CM: float = 10.0
CM_PER_M: float = 100.0
M_PER_CM: float = 0.01

# ---- area -----------------------------------------------------------------
M2_PER_CM2: float = 1.0e-4
CM2_PER_M2: float = 1.0e4

# ---- time -----------------------------------------------------------------
SECONDS_PER_HOUR: float = 3600.0
SECONDS_PER_DAY: float = 86400.0
HOURS_PER_DAY: float = 24.0

# ---- energy / radiation ---------------------------------------------------
#: multiply an energy flux in MJ m-2 d-1 by this to get W m-2 (1e6 J / 86400 s).
W_PER_M2_PER_MJ_M2_DAY: float = 1.0e6 / SECONDS_PER_DAY
#: multiply a flux in W m-2 by this to get MJ m-2 d-1.
MJ_M2_DAY_PER_W_PER_M2: float = SECONDS_PER_DAY / 1.0e6
#: latent heat of vaporisation of water at ~20 degC [MJ kg-1] (FAO-56 constant).
LATENT_HEAT_MJ_PER_KG: float = 2.45
#: Stefan-Boltzmann constant [MJ m-2 K-4 d-1] (FAO-56 value).
STEFAN_BOLTZMANN_MJ_M2_K4_DAY: float = 4.903e-9
#: Stefan-Boltzmann constant [W m-2 K-4].
STEFAN_BOLTZMANN_W_M2_K4: float = 5.670374419e-8
#: PAR fraction of global shortwave radiation used by CERES and RZWQM.
PAR_FRACTION: float = 0.5

# ---- mass -----------------------------------------------------------------
G_M2_PER_KG_HA: float = 0.1
KG_HA_PER_G_M2: float = 10.0

# ---- temperature ----------------------------------------------------------
KELVIN_OFFSET: float = 273.15

# ---- water ----------------------------------------------------------------
#: density of liquid water [kg m-3].
WATER_DENSITY_KG_M3: float = 1000.0
#: 1 mm of water over 1 m2 is 1 kg; 1 cm over 1 ha is 100 t.
KG_PER_M2_PER_MM: float = 1.0


def mm_to_cm(x):
    """Convert a length or water depth from mm to cm."""
    return x * CM_PER_MM


def cm_to_mm(x):
    """Convert a length or water depth from cm to mm."""
    return x * MM_PER_CM


def cm2_to_m2(x):
    """Convert an area from cm2 to m2 (leaf area per m2 of ground to LAI: cm2 m-2 -> m2 m-2)."""
    return x * M2_PER_CM2


def m2_to_cm2(x):
    """Convert an area from m2 to cm2."""
    return x * CM2_PER_M2


def mj_m2_day_to_w_m2(x):
    """Convert a daily energy flux from MJ m-2 d-1 to a mean power density in W m-2."""
    return x * W_PER_M2_PER_MJ_M2_DAY


def w_m2_to_mj_m2_day(x):
    """Convert a mean power density in W m-2 to a daily energy flux in MJ m-2 d-1."""
    return x * MJ_M2_DAY_PER_W_PER_M2


def celsius_to_kelvin(x):
    """Convert degC to K."""
    return x + KELVIN_OFFSET


def kelvin_to_celsius(x):
    """Convert K to degC."""
    return x - KELVIN_OFFSET


def kg_ha_to_g_m2(x):
    """Convert an areal mass from kg ha-1 to g m-2."""
    return x * G_M2_PER_KG_HA


def g_m2_to_kg_ha(x):
    """Convert an areal mass from g m-2 to kg ha-1."""
    return x * KG_HA_PER_G_M2


def mj_m2_day_to_mm_water(x, latent_heat=LATENT_HEAT_MJ_PER_KG):
    """Convert an energy flux in MJ m-2 d-1 into the equivalent evaporated water depth in mm d-1."""
    return x / latent_heat


def mm_water_to_mj_m2_day(x, latent_heat=LATENT_HEAT_MJ_PER_KG):
    """Convert an evaporated water depth in mm d-1 into an energy flux in MJ m-2 d-1."""
    return x * latent_heat


# =====================================================================================
# machine-readable unit strings
# =====================================================================================
#
# The ``unit=`` strings of :func:`agrijax.core.state.field` follow one small grammar, parsed here
# without any runtime dependency (the parser runs in tests and tools, never inside a traced
# function):
#
#     unit    := "-" | product [ "per" product ]
#     product := factor ( [ws | "x" | "*"] factor )*
#     factor  := (SYMBOL | NUMBER | "(" product ")") [exponent]
#     exponent:= ["^"] (SIGNED_NUMBER | ["-"] "(" NUMBER "/" NUMBER ")" | NAME)
#
# ``cm3 cm-3``, ``MJ m-2 d-1``, ``(degC d)-1``, ``g cm-2.5``, ``cm2 g-(1/1.25)``, ``cm g-1 x 1e4``,
# ``kg ha-1 per MJ m-2``, ``%``.
# A NAME exponent (``cm^eps``) makes the unit *parametric*: its dimension depends on a parameter,
# so it parses but has no fixed dimension. ``YYYYDDD`` is a date code, not a quantity.
# ``degC`` and ``K`` share the temperature dimension with scale 1 (a temperature *difference*);
# the 273.15 offset is applied only by the named adapters :func:`celsius_to_kelvin` /
# :func:`kelvin_to_celsius`.


class UnitError(ValueError):
    """A unit string does not parse, or two units are not convertible."""


#: base dimensions: length, mass, time, temperature, plus counted things (per plant, per kernel,
#: per ear) kept as their own dimensions so that ``g plant-1`` and ``g m-2`` never mix silently,
#: and ``date`` for the ``YYYYDDD`` code.
BASE_DIMENSIONS: tuple[str, ...] = ("L", "M", "T", "Theta", "plant", "kernel", "ear", "date")

_L, _M, _T, _TH = "L", "M", "T", "Theta"

#: symbol -> (scale to the base SI unit, {dimension: exponent})
_SYMBOLS: dict[str, tuple[float, dict[str, int]]] = {
    # length
    "m": (1.0, {_L: 1}),
    "cm": (1e-2, {_L: 1}),
    "mm": (1e-3, {_L: 1}),
    "km": (1e3, {_L: 1}),
    "ha": (1e4, {_L: 2}),
    # mass
    "kg": (1.0, {_M: 1}),
    "g": (1e-3, {_M: 1}),
    "mg": (1e-6, {_M: 1}),
    "t": (1e3, {_M: 1}),
    # time
    "s": (1.0, {_T: 1}),
    "min": (60.0, {_T: 1}),
    "h": (3600.0, {_T: 1}),
    "hr": (3600.0, {_T: 1}),
    "d": (86400.0, {_T: 1}),
    # temperature (difference)
    "K": (1.0, {_TH: 1}),
    "degC": (1.0, {_TH: 1}),
    # energy, power, pressure
    "J": (1.0, {_M: 1, _L: 2, _T: -2}),
    "kJ": (1e3, {_M: 1, _L: 2, _T: -2}),
    "MJ": (1e6, {_M: 1, _L: 2, _T: -2}),
    "W": (1.0, {_M: 1, _L: 2, _T: -3}),
    "Pa": (1.0, {_M: 1, _L: -1, _T: -2}),
    "kPa": (1e3, {_M: 1, _L: -1, _T: -2}),
    "bar": (1e5, {_M: 1, _L: -1, _T: -2}),
    # dimensionless
    "rad": (1.0, {}),
    "percent": (1e-2, {}),
    "ppm": (1e-6, {}),
    "%": (1e-2, {}),
    # calorie (thermochemical, 4.184 J): the langley is cal cm-2
    "cal": (4.184, {_M: 1, _L: 2, _T: -2}),
    # counted things
    "plant": (1.0, {"plant": 1}),
    "plants": (1.0, {"plant": 1}),
    "kernel": (1.0, {"kernel": 1}),
    "kernels": (1.0, {"kernel": 1}),
    "ear": (1.0, {"ear": 1}),
    "ears": (1.0, {"ear": 1}),
    # date code
    "YYYYDDD": (1.0, {"date": 1}),
}


@_dataclass(frozen=True)
class Unit:
    """A parsed unit: ``scale`` to base SI, integer/rational ``dims``, and free ``parametric`` exponents."""

    text: str
    scale: float
    dims: tuple[tuple[str, _Fraction], ...]
    parametric: tuple[str, ...] = ()

    @property
    def dimensionless(self) -> bool:
        return not self.dims and not self.parametric

    @property
    def is_date(self) -> bool:
        return dict(self.dims).get("date", 0) != 0

    def same_dimension(self, other: Unit) -> bool:
        """True when both units have the same fixed dimension (parametric units never match)."""
        return not self.parametric and not other.parametric and self.dims == other.dims

    def __str__(self) -> str:
        return self.text


def _norm(d: dict[str, _Fraction]) -> tuple[tuple[str, _Fraction], ...]:
    return tuple(sorted((k, v) for k, v in d.items() if v != 0))


_TOKEN = _re.compile(r"\s*(?:(?P<num>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)|(?P<sym>[A-Za-z_]+|%)|(?P<op>[()^*/-]))")


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.toks: list[tuple[str, str]] = []
        pos = 0
        while pos < len(text):
            if text[pos:].strip() == "":
                break
            m = _TOKEN.match(text, pos)
            if m is None or m.end() == pos:
                raise UnitError(f"cannot parse unit {text!r} at {text[pos:]!r}")
            kind = m.lastgroup
            assert kind is not None
            self.toks.append((kind, m.group(kind)))
            pos = m.end()
        self.i = 0

    def peek(self) -> tuple[str, str] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> tuple[str, str]:
        t = self.peek()
        if t is None:
            raise UnitError(f"unexpected end of unit {self.text!r}")
        self.i += 1
        return t

    def expect(self, value: str) -> None:
        t = self.take()
        if t[1] != value:
            raise UnitError(f"expected {value!r} in unit {self.text!r}, got {t[1]!r}")

    def number(self) -> _Fraction:
        kind, val = self.take()
        if kind != "num":
            raise UnitError(f"expected a number in unit {self.text!r}, got {val!r}")
        return _Fraction(val)

    def product(self) -> tuple[float, dict[str, _Fraction], set[str]]:
        scale, dims, par = 1.0, dict[str, _Fraction](), set[str]()
        n = 0
        while True:
            t = self.peek()
            if t is None or t[1] == ")":
                break
            if t == ("sym", "x") or t[1] == "*":
                self.take()
                continue
            s, d, p = self.factor()
            scale *= s
            for k, v in d.items():
                dims[k] = dims.get(k, _Fraction(0)) + v
            par |= p
            n += 1
        if n == 0:
            raise UnitError(f"empty unit expression in {self.text!r}")
        return scale, dims, par

    def factor(self) -> tuple[float, dict[str, _Fraction], set[str]]:
        kind, val = self.take()
        if kind == "num":
            base: tuple[float, dict[str, _Fraction], set[str]] = (float(val), {}, set())
        elif kind == "sym":
            if val not in _SYMBOLS:
                raise UnitError(f"unknown unit symbol {val!r} in {self.text!r}")
            sc, d = _SYMBOLS[val]
            base = (sc, {k: _Fraction(v) for k, v in d.items()}, set())
        elif val == "(":
            base = self.product()
            self.expect(")")
        else:
            raise UnitError(f"unexpected {val!r} in unit {self.text!r}")
        exp = self.exponent()
        if exp is None:
            return base
        sc, d, p = base
        if isinstance(exp, str):
            return sc, d, p | {exp}  # parametric: dimension depends on a parameter
        if p:
            return sc, d, p
        return sc ** float(exp), {k: v * exp for k, v in d.items()}, p

    def exponent(self) -> _Fraction | str | None:
        t = self.peek()
        if t is None:
            return None
        caret = False
        if t[1] == "^":
            self.take()
            caret = True
            t = self.peek()
            if t is None:
                raise UnitError(f"dangling '^' in unit {self.text!r}")
        sign = 1
        if t[1] == "-":
            self.take()
            sign = -1
            t = self.peek()
            if t is None:
                raise UnitError(f"dangling '-' in unit {self.text!r}")
        if t[0] == "num":
            self.take()
            return sign * _Fraction(t[1])
        if t[1] == "(" and (sign == -1 or caret):
            self.take()
            num = self.number()
            self.expect("/")
            den = self.number()
            self.expect(")")
            return sign * num / den
        if t[0] == "sym" and caret:
            self.take()
            return t[1]
        if sign == -1 or caret:
            raise UnitError(f"bad exponent in unit {self.text!r}")
        return None


def parse_unit(text: str) -> Unit:
    """Parse a unit string of the grammar above; ``"-"`` is dimensionless. Raises :class:`UnitError`."""
    if not isinstance(text, str) or not text.strip():
        raise UnitError(f"empty unit {text!r} (use '-' for dimensionless)")
    if text.strip() == "-":
        return Unit(text, 1.0, ())
    num, sep, den = text.partition(" per ")
    if sep:  # ``a per b`` = a b-1 (the right-hand side as a whole)
        a, b = parse_unit(num), parse_unit(den)
        dims = dict(a.dims)
        for k, v in b.dims:
            dims[k] = dims.get(k, _Fraction(0)) - v
        return Unit(text, a.scale / b.scale, _norm(dims), tuple(sorted({*a.parametric, *b.parametric})))
    p = _Parser(text)
    scale, dims, par = p.product()
    rest = p.peek()
    if rest is not None:
        raise UnitError(f"trailing {rest[1]!r} in unit {text!r}")
    return Unit(text, scale, _norm(dims), tuple(sorted(par)))


def conversion_factor(src: str | Unit, dst: str | Unit) -> float:
    """``x_dst = factor * x_src`` for two units of the same fixed dimension (no offsets, no dates)."""
    a = parse_unit(src) if isinstance(src, str) else src
    b = parse_unit(dst) if isinstance(dst, str) else dst
    if not a.same_dimension(b):
        raise UnitError(f"{a} and {b} are not convertible")
    if a.is_date:
        raise UnitError(f"{a} is a date code")
    return a.scale / b.scale


# ------------------------------------------------------------------------- named adapters
@_dataclass(frozen=True)
class Adapter:
    """A named unit conversion: ``fn`` maps ``src`` to ``dst``; ``inverse`` is the adapter back.

    ``via`` lists the units of the physical constants the conversion goes through (latent heat,
    water density): ``src / via == dst`` for one direction and ``src * via == dst`` for its
    inverse, so the dimension and factor checks cover these conversions too; ``affine`` marks an
    offset conversion (temperatures).
    """

    name: str
    fn: _Callable[[_Any], _Any]
    src: str
    dst: str
    inverse: str
    via: tuple[str, ...] = ()
    affine: bool = False


#: every unit conversion of the package, by name. A conversion elsewhere is a bug (see
#: ``tests/unit/test_dims_units.py``).
ADAPTERS: dict[str, Adapter] = {}


def _register(fn: _Callable[..., _Any], src: str, dst: str, inverse: str, **kw: _Any) -> None:
    ADAPTERS[fn.__name__] = Adapter(fn.__name__, fn, src, dst, inverse, **kw)


_register(mm_to_cm, "mm", "cm", "cm_to_mm")
_register(cm_to_mm, "cm", "mm", "mm_to_cm")
_register(cm2_to_m2, "cm2", "m2", "m2_to_cm2")
_register(m2_to_cm2, "m2", "cm2", "cm2_to_m2")
_register(mj_m2_day_to_w_m2, "MJ m-2 d-1", "W m-2", "w_m2_to_mj_m2_day")
_register(w_m2_to_mj_m2_day, "W m-2", "MJ m-2 d-1", "mj_m2_day_to_w_m2")
_register(celsius_to_kelvin, "degC", "K", "kelvin_to_celsius", affine=True)
_register(kelvin_to_celsius, "K", "degC", "celsius_to_kelvin", affine=True)
_register(kg_ha_to_g_m2, "kg ha-1", "g m-2", "g_m2_to_kg_ha")
_register(g_m2_to_kg_ha, "g m-2", "kg ha-1", "kg_ha_to_g_m2")
_register(mj_m2_day_to_mm_water, "MJ m-2 d-1", "mm d-1", "mm_water_to_mj_m2_day", via=("MJ kg-1", "kg m-3"))
_register(mm_water_to_mj_m2_day, "mm d-1", "MJ m-2 d-1", "mj_m2_day_to_mm_water", via=("kg m-3", "MJ kg-1"))


# ------------------------------------------------------------------------- declared constants
@_dataclass(frozen=True)
class Constant:
    """A numeric constant of this module, declared with its unit, meaning and provenance.

    ``kind`` is ``"conversion"`` (a pure unit-conversion factor: ``src`` and ``dst`` name the units,
    and ``value == conversion_factor(src, dst)``), ``"physical"`` (a physical constant or a
    standard value) or ``"coefficient"`` (a model coefficient kept here for reuse). A physical
    constant or coefficient cites its ``paper`` (author, year; ``equation`` its number there) or its
    reference source (``ref_version`` as in the process registry key, ``file``, ``line``); a
    derived value names the constants it is derived from in ``note``.
    """

    name: str
    value: float
    unit: str
    description: str
    kind: str
    src: str = ""
    dst: str = ""
    ref_version: str = "none"
    file: str = ""
    line: int | None = None
    paper: str = ""
    equation: str = ""
    note: str = ""


#: every numeric module constant of :mod:`agrijax.core.units`, by name (``tests/unit/test_units.py``
#: checks that none is missing and that every conversion factor matches :func:`conversion_factor`).
CONSTANTS: dict[str, Constant] = {}

_FAO56 = "Allen et al. (1998), FAO Irrigation and Drainage Paper 56"


def _declare(name: str, unit: str, description: str, kind: str, **kw: _Any) -> None:
    CONSTANTS[name] = Constant(name, globals()[name], unit, description, kind, **kw)


def _conversion(name: str, src: str, dst: str, description: str) -> None:
    _declare(name, f"({dst}) ({src})-1", description, "conversion", src=src, dst=dst)


_conversion("CM_PER_MM", "mm", "cm", "centimetres per millimetre")
_conversion("MM_PER_CM", "cm", "mm", "millimetres per centimetre")
_conversion("CM_PER_M", "m", "cm", "centimetres per metre")
_conversion("M_PER_CM", "cm", "m", "metres per centimetre")
_conversion("M2_PER_CM2", "cm2", "m2", "square metres per square centimetre")
_conversion("CM2_PER_M2", "m2", "cm2", "square centimetres per square metre")
_conversion("SECONDS_PER_HOUR", "h", "s", "seconds per hour")
_conversion("SECONDS_PER_DAY", "d", "s", "seconds per day")
_conversion("HOURS_PER_DAY", "d", "h", "hours per day")
_conversion("W_PER_M2_PER_MJ_M2_DAY", "MJ m-2 d-1", "W m-2", "W m-2 per MJ m-2 d-1 (1e6 J / 86400 s)")
_conversion("MJ_M2_DAY_PER_W_PER_M2", "W m-2", "MJ m-2 d-1", "MJ m-2 d-1 per W m-2 (86400 s / 1e6 J)")
_conversion("G_M2_PER_KG_HA", "kg ha-1", "g m-2", "g m-2 per kg ha-1")
_conversion("KG_HA_PER_G_M2", "g m-2", "kg ha-1", "kg ha-1 per g m-2")
_declare(
    "LATENT_HEAT_MJ_PER_KG",
    "MJ kg-1",
    "latent heat of vaporisation of water at about 20 degC",
    "physical",
    ref_version="fao56-1998",
    paper=_FAO56,
    equation="8",
    note="the constant lambda = 2.45 MJ kg-1 of the psychrometric constant",
)
_declare(
    "STEFAN_BOLTZMANN_MJ_M2_K4_DAY",
    "MJ m-2 K-4 d-1",
    "Stefan-Boltzmann constant per day (FAO-56 rounding)",
    "physical",
    ref_version="fao56-1998",
    paper=_FAO56,
    equation="39",
)
_declare(
    "STEFAN_BOLTZMANN_W_M2_K4",
    "W m-2 K-4",
    "Stefan-Boltzmann constant",
    "physical",
    paper="Tiesinga et al. (2021), CODATA recommended values of the fundamental physical constants: 2018",
)
_declare(
    "PAR_FRACTION",
    "-",
    "fraction of global shortwave radiation that is photosynthetically active",
    "coefficient",
    ref_version="dssat-4.8.6.0",
    file="Plant/CERES-Rice/RI_Grosub.for",
    line=588,
    note="the PAR = 0.5*SRAD of the CERES family; no kernel of the package uses this constant yet",
)
_declare(
    "KELVIN_OFFSET",
    "K",
    "0 degC in kelvin (applied only by celsius_to_kelvin / kelvin_to_celsius)",
    "physical",
    paper="BIPM (2019), The International System of Units (SI), 9th edition",
)
_declare(
    "WATER_DENSITY_KG_M3",
    "kg m-3",
    "density of liquid water",
    "physical",
    ref_version="fao56-1998",
    paper=_FAO56,
    note="rho_w = 1000 kg m-3 of the radiation-to-equivalent-evaporation conversion (chapter 3)",
)
_declare(
    "KG_PER_M2_PER_MM",
    "kg m-2 mm-1",
    "mass of 1 mm of water over 1 m2",
    "physical",
    note="derived: WATER_DENSITY_KG_M3 x 1 mm",
)
