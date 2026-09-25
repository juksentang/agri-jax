"""Coefficients and numerical settings of the soil-water modules, each declared once with its origin.

Two kinds of numbers appear in the Richards, Green-Ampt and soil-water-day kernels besides the
file-read soil parameters:

* **model coefficients**, numbers of the equations: declared with
  :func:`agrijax.core.coefficients.coef` in :class:`GreenAmptCoefficients` (value, unit, meaning,
  :class:`~agrijax.core.coefficients.Provenance`). They are calibratable pytree leaves by
  default; ``GreenAmptParams.coefficients = None`` means the RZWQM2 values
  (:data:`RZWQM2_GREEN_AMPT`), which are Python floats, so a default run traces the same
  literals as before they were named. The Richards redistribution has no hard-coded model
  coefficient: every number of its equations is a soil parameter read from ``rzwqm.dat``.
* **numerical settings**, numbers of the scheme (sub-steps, Newton iterations, damping,
  tolerances, clamp bounds, time weights, grid geometry): declared with
  :func:`numerical_setting` (a module constant) or :func:`setting_field` (a static field of a
  config such as :class:`~agrijax.processes.soil_water.richards.RichardsConfig`), with their
  unit, meaning and origin: ``"rzwqm2-4.6"`` for a convention of the reference model (file,
  line and routine of the reference source plus the published description; the reference
  source text is never quoted, see :data:`agrijax.core.coefficients.REFERENCES`), or
  ``"agrijax"`` for a choice of this implementation (with the design note or test that
  justifies it). They are not calibrated; :data:`SETTINGS` lists every one.

Numerical guards (floors that only keep a quotient or a logarithm finite) are declared with
:func:`agrijax.core.coefficients.numerical_guard` next to the kernel that uses them.

Source: Ahuja, L.R., Rojas, K.W., Hanson, J.D., Shaffer, M.J., Ma, L. (eds.), 2000. Root Zone
Water Quality Model, ch. 3; RZWQM2 4.6 ``RZTEST.for`` (``EVNTRO``, ``INFIL``), ``Rzrich.for``
(``RICHRD``, ``CNHEAD``), ``Rzmain.for`` (grid), read for conventions only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx

from agrijax.core.coefficients import Coefficients, Provenance, coef, coefficient_table
from agrijax.core.units import parse_unit

__all__ = [
    "AHUJA_2000",
    "REF_VERSION",
    "RZWQM2_GREEN_AMPT",
    "SETTINGS",
    "GreenAmptCoefficients",
    "NumericalSetting",
    "green_ampt_coefficient_table",
    "numerical_setting",
    "rzwqm2",
    "setting_field",
    "settings_table",
]

#: the reference of the soil-water processes (their registry keys end in ``@rzwqm2-4.6``)
REF_VERSION = "rzwqm2-4.6"
#: the published description of the RZWQM2 soil-water equations every RZWQM2 citation carries
AHUJA_2000 = "Ahuja et al. (2000), Root Zone Water Quality Model, ch. 3"
#: origins a numerical setting may have
_ORIGINS = frozenset({REF_VERSION, "agrijax"})


def rzwqm2(file_line: str, routine: str, *, note: str = "", equation: str = "") -> Provenance:
    """Provenance of an RZWQM2 4.6 convention: ``<file>:<line>`` of the reference source tree
    (``RZWQM/...``), the routine, and the published description (:data:`AHUJA_2000`); no
    statement text (the reference has no licence that allows quoting it)."""
    return Provenance.at(
        REF_VERSION, file_line, routine=routine, paper=AHUJA_2000, equation=equation, note=note
    )


@dataclass(frozen=True)
class NumericalSetting:
    """A setting of the numerical scheme: value, unit, meaning and origin.

    ``origin`` is ``"rzwqm2-4.6"`` (a convention of the reference model; ``provenance`` gives the
    file, line and routine) or ``"agrijax"`` (a choice of this implementation; ``basis`` names
    the design note, paper or test that justifies it, and ``provenance`` may cite a paper).
    """

    name: str
    value: Any
    unit: str
    description: str
    origin: str
    provenance: Provenance | None = None
    basis: str = ""

    def __post_init__(self) -> None:
        if self.origin not in _ORIGINS:
            raise ValueError(f"setting {self.name!r}: origin must be one of {sorted(_ORIGINS)}")
        if not self.description.strip():
            raise ValueError(f"setting {self.name!r} needs a description")
        parse_unit(self.unit)
        if self.origin == REF_VERSION and (self.provenance is None or not self.provenance.file):
            raise ValueError(f"setting {self.name!r}: an RZWQM2 convention cites the reference source file")
        if self.origin == "agrijax" and not (self.basis or self.provenance):
            raise ValueError(f"setting {self.name!r}: an own choice names its basis (note, paper or test)")
        if self.provenance is not None and self.provenance.ref_version not in (REF_VERSION, "none"):
            raise ValueError(f"setting {self.name!r}: unexpected reference {self.provenance.ref_version!r}")

    @property
    def source(self) -> str:
        """One-line citation of the origin."""
        parts = [self.provenance.source] if self.provenance is not None else []
        if self.basis:
            parts.append(self.basis)
        return "; ".join(parts) if parts else self.origin


#: every numerical setting of the soil-water modules, by dotted name
SETTINGS: dict[str, NumericalSetting] = {}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _register(s: NumericalSetting) -> None:
    old = SETTINGS.setdefault(s.name, s)
    _require(old == s, f"numerical setting {s.name!r} already declared as {old}")


def numerical_setting(
    name: str,
    value: Any,
    unit: str,
    description: str,
    *,
    origin: str,
    provenance: Provenance | None = None,
    basis: str = "",
) -> Any:
    """Declare a numerical setting used as a module constant and return its value unchanged."""
    _register(NumericalSetting(name, value, unit, description, origin, provenance, basis))
    return value


def setting_field(
    name: str,
    default: Any,
    unit: str,
    description: str,
    *,
    origin: str,
    provenance: Provenance | None = None,
    basis: str = "",
) -> Any:
    """A static config field holding a numerical setting (hashable, part of the compiled program).

    The field default is ``default``; its metadata carries the :class:`NumericalSetting` (key
    ``"setting"``), and the setting is registered in :data:`SETTINGS` under ``name``.
    """
    s = NumericalSetting(name, default, unit, description, origin, provenance, basis)
    _register(s)
    meta = {"unit": unit, "description": description, "setting": s, "source": s.source}
    return eqx.field(static=True, default=default, metadata=meta)


def settings_table() -> list[dict[str, Any]]:
    """One row per numerical setting: ``name``, ``value``, ``unit``, ``description``, ``origin``,
    ``source`` and the :class:`~agrijax.core.coefficients.Provenance` fields (empty when none)."""
    rows = []
    for s in SETTINGS.values():
        prov = s.provenance.as_dict() if s.provenance is not None else {}
        rows.append(
            {
                "name": s.name,
                "value": s.value,
                "unit": s.unit,
                "description": s.description,
                "origin": s.origin,
                "source": s.source,
                "basis": s.basis,
                **prov,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Green-Ampt event (RZWQM2 EVNTRO / INFIL)
# ---------------------------------------------------------------------------


class GreenAmptCoefficients(Coefficients):
    """Hard-coded numbers of the RZWQM2 Green-Ampt event (``GreenAmptParams.coefficients``)."""

    vrcf: float = coef(
        2.0,
        "-",
        "reduction factor of the layered Green-Ampt infiltration capacity (V = V_GA / vrcf)",
        rzwqm2(
            "RZWQM/RZTEST.for:1283",
            "INFIL",
            note="VRCF; applied to the capacity at RZTEST.for:1382. The published chapter gives the "
            "layered Green-Ampt capacity it scales, not the factor.",
        ),
        bounds=(1.0e-3, 1.0e3),
        fortran_name="VRCF",
    )
    suction_offset: float = coef(
        1.0,
        "cm",
        "head added to the conductivity integral in the wetting-front suction "
        "(S_f = offset + int K/K_s ds + pond)",
        rzwqm2("RZWQM/RZTEST.for:649", "EVNTRO", note="SWF; Mein & Larson (1973) form of the suction"),
        fortran_name="SWF",
    )
    suction_dry_limit: float = coef(
        1.0,
        "cm",
        "value of the conductivity integral when the initial suction is at most suction_lower",
        rzwqm2("RZWQM/RZTEST.for:646", "EVNTRO", note="SWF for SI <= 1 cm, before the division by DC1"),
        fortran_name="SWF",
    )
    suction_lower: float = coef(
        1.0,
        "cm",
        "lower limit of the conductivity integral of the wetting-front suction (the first "
        "centimetre is not integrated); also the floor of hb_k in the integral's segment split",
        rzwqm2("RZWQM/RZTEST.for:643", "EVNTRO", note="SI > 1 cm branch and the lower limit 1 cm"),
        static=True,
        fortran_name="SI",
    )


RZWQM2_GREEN_AMPT = GreenAmptCoefficients()
"""The RZWQM2 4.6 values (what ``GreenAmptParams.coefficients = None`` means)."""


def green_ampt_coefficient_table() -> list[dict[str, Any]]:
    """:func:`agrijax.core.coefficients.coefficient_table` of :data:`RZWQM2_GREEN_AMPT`."""
    return coefficient_table(RZWQM2_GREEN_AMPT)
