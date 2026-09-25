"""Model coefficients declared once: value, unit, meaning and provenance.

Every number of a model's equations that is not a state, a forcing or a file-read parameter is a
*coefficient*. It is declared exactly once, as a field of a :class:`Coefficients` pytree, with

* its default ``value`` (a Python ``float``, or an ``int`` for a static threshold),
* its ``unit`` in the grammar of :func:`agrijax.core.units.parse_unit` (checked when the class is
  defined: a unit that does not parse raises :class:`~agrijax.core.units.UnitError`),
* a one-line ``description``, and
* a structured :class:`Provenance`: the reference it follows (``ref_version``, the same string as
  the ``@ref_version`` of the process registry key, e.g. ``dssat-4.8.6.0``, ``rzwqm2-4.6``,
  ``asce-ewri-2005``), the ``file`` / ``line`` / ``routine`` of the reference source, the original
  ``statement`` (only for references whose source licence allows quoting it, see
  :data:`REFERENCES`), and the published ``paper`` / ``equation``.

A coefficient is **calibratable by default**: it is a pytree leaf, so a :class:`Coefficients`
instance whose leaves are arrays (:meth:`Coefficients.as_arrays`) can be differentiated, batched
and calibrated. ``static=True`` makes it a static field instead (integer day counts, leaf numbers:
not a leaf, and changing one retraces); ``calibrate=False`` keeps it a leaf but leaves it out of
:meth:`Coefficients.calibratable_paths` and the calibration vector (a constant such as the value
of pi a reference uses).

Defaults stay Python floats, so a model run with the default instance traces exactly the literals
the equations had before they were named. :func:`coefficient_table` lists every declared
coefficient (one row per field, recursing into nested coefficient groups);
:meth:`Coefficients.to_vector` / :meth:`Coefficients.from_vector` map the calibratable leaves to a
flat array and back.

Numerical guards (``1e-12`` floors that keep gradients finite) are not coefficients; declare them
as named module constants with :func:`numerical_guard` so that the kernels carry no bare literal
(lint rule AJ007, :mod:`agrijax.core.lint`).

Example::

    class BucketCoefficients(Coefficients):
        drain_frac: float = coef(
            0.5, "d-1", "fraction of the water above field capacity drained per day",
            Provenance("dssat-4.8.6.0", file="SPAM/WATBAL.for", line=120, routine="WATBAL",
                       statement="DRAIN = SWCON * (SW - DUL)"),
        )
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp

from agrijax.core.state import Params, get_path, set_path
from agrijax.core.units import UnitError, parse_unit

__all__ = [
    "GUARDS",
    "REFERENCES",
    "Coefficients",
    "Provenance",
    "Reference",
    "coef",
    "coefficient_table",
    "is_coefficient_field",
    "iter_coefficients",
    "numerical_guard",
    "register_reference",
]

_C = TypeVar("_C", bound="Coefficients")

#: ``ref_version`` of a process registry key (:mod:`agrijax.core.process`): ``name-version``
_REF_RE = re.compile(r"^(?P<name>[a-z][a-z0-9]*(?:-[a-z][a-z0-9]*)*)-(?P<version>[0-9][a-z0-9.\-]*)$")
NO_REFERENCE = "none"


@dataclass(frozen=True)
class Reference:
    """A reference a coefficient can cite: its display name and what its licence allows.

    ``display`` is a template with ``{version}`` (``"DSSAT-CSM v{version}"``). ``statement_allowed``
    says whether the original source statement may be quoted in the public repository: true for
    an open-source reference such as the BSD-3 DSSAT-CSM; false for a reference whose source
    carries no licence file (RZWQM2; statements are not quoted by project policy), whose
    coefficients cite file, line, routine and the *published* equation instead.
    ``needs_paper`` requires the published equation (``paper``) on every coefficient that cites
    this reference's source.
    """

    name: str
    display: str
    licence: str
    statement_allowed: bool
    needs_paper: bool = False


#: the references coefficients may cite, by the name part of ``ref_version`` (``dssat`` of
#: ``dssat-4.8.6.0``). A new reference is added with :func:`register_reference`, which is where
#: its licence decision is recorded.
REFERENCES: dict[str, Reference] = {}


def register_reference(
    name: str, display: str, licence: str, *, statement_allowed: bool, needs_paper: bool = False
) -> Reference:
    """Add (or confirm) a reference of :data:`REFERENCES`; a conflicting redefinition raises."""
    ref = Reference(name, display, licence, statement_allowed, needs_paper)
    old = REFERENCES.get(name)
    if old is not None and old != ref:
        raise ValueError(f"reference {name!r} already registered as {old}")
    REFERENCES[name] = ref
    return ref


register_reference("dssat", "DSSAT-CSM v{version}", "BSD-3-Clause", statement_allowed=True)
register_reference(
    "rzwqm2", "RZWQM2 {version}", "no licence file in source tree", statement_allowed=False, needs_paper=True
)
register_reference("asce-ewri", "ASCE-EWRI {version}", "published standard", statement_allowed=False)
register_reference("fao56", "FAO-56 ({version})", "published report", statement_allowed=False)


def _split_ref(ref_version: str) -> tuple[Reference | None, str]:
    if ref_version == NO_REFERENCE:
        return None, ""
    m = _REF_RE.match(ref_version)
    if m is None:
        raise ValueError(
            f"ref_version {ref_version!r} is not a registry reference 'name-version' "
            "(e.g. 'dssat-4.8.6.0', 'rzwqm2-4.6', 'asce-ewri-2005') or 'none'"
        )
    ref = REFERENCES.get(m["name"])
    if ref is None:
        raise ValueError(
            f"unknown reference {m['name']!r} of ref_version {ref_version!r}: register it with "
            f"register_reference() (known: {sorted(REFERENCES)})"
        )
    return ref, m["version"]


@dataclass(frozen=True)
class Provenance:
    """Where a coefficient comes from.

    ``ref_version`` is the reference the implementation follows, spelled as in the process
    registry key (``dssat-4.8.6.0``, ``rzwqm2-4.6``, ``asce-ewri-2005``, or ``none`` for a
    coefficient with no reference model, which then cites a ``paper``). ``file`` (relative to the
    reference source tree), ``line`` and ``routine`` locate it in the reference source;
    ``statement`` quotes the source statement that holds the value and is accepted only when the
    reference's licence allows it (:attr:`Reference.statement_allowed`). ``paper`` (author, year)
    and ``equation`` (number in that paper) name the published equation; ``note`` is free text.

    At least a ``file`` or a ``paper`` is required; a reference with ``needs_paper`` requires the
    ``paper`` whenever its source is cited.
    """

    ref_version: str
    file: str = ""
    line: int | None = None
    routine: str = ""
    statement: str = ""
    paper: str = ""
    equation: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        ref, _ = _split_ref(self.ref_version)
        if self.line is not None:
            if isinstance(self.line, bool) or not isinstance(self.line, int) or self.line < 1:
                raise ValueError(f"line must be a positive integer, got {self.line!r}")
            if not self.file:
                raise ValueError("a line number needs the file it is in")
        if (self.file or self.routine) and ref is None:
            raise ValueError("a source file or routine needs a reference ref_version, not 'none'")
        if not (self.file or self.paper):
            raise ValueError(f"provenance of {self.ref_version!r} cites neither a source file nor a paper")
        if self.statement:
            if ref is None or not ref.statement_allowed:
                who = "no reference" if ref is None else f"{ref.name} ({ref.licence})"
                raise ValueError(
                    f"the source statement of {who} may not be quoted; cite file, line, routine and "
                    "the published equation (paper, equation) instead"
                )
            if not (self.file and self.line):
                raise ValueError("a quoted statement needs the file and line it is on")
        if ref is not None and ref.needs_paper and self.file and not self.paper:
            raise ValueError(
                f"{ref.name} coefficients cite the published equation: give paper (and equation) "
                f"for {self.file}"
            )
        if self.equation and not self.paper:
            raise ValueError("an equation number needs the paper it is in")

    @classmethod
    def at(cls, ref_version: str, file_line: str, **kw: Any) -> Provenance:
        """``Provenance.at("dssat-4.8.6.0", "Plant/CERES-Maize/MZ_GROSUB.for:911", statement=...)``."""
        file, sep, line = file_line.rpartition(":")
        if not sep or not line.isdigit():
            raise ValueError(f"expected '<file>:<line>', got {file_line!r}")
        return cls(ref_version, file=file, line=int(line), **kw)

    @property
    def reference(self) -> Reference | None:
        return _split_ref(self.ref_version)[0]

    @property
    def source(self) -> str:
        """One-line citation: ``DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for:911``, or the paper."""
        ref, version = _split_ref(self.ref_version)
        parts: list[str] = []
        if self.file:
            where = self.file if self.line is None else f"{self.file}:{self.line}"
            parts.append(f"{ref.display.format(version=version)} {where}" if ref else where)
        if self.paper and (not self.file or not self.statement):
            parts.append(self.paper + (f", eq. {self.equation}" if self.equation else ""))
        return "; ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def coef(
    value: float,
    unit: str,
    description: str,
    provenance: Provenance,
    *,
    static: bool = False,
    calibrate: bool = True,
    bounds: tuple[float, float] | None = None,
    fortran_name: str = "",
) -> Any:
    """Declare a coefficient field of a :class:`Coefficients` class.

    ``value`` is the default (a Python float; an ``int`` only for a ``static`` threshold),
    ``unit`` a :func:`~agrijax.core.units.parse_unit` string (checked here), ``description`` its
    meaning and ``provenance`` where it comes from. ``static=True`` makes it a static field (not a
    pytree leaf, not calibratable); ``calibrate=False`` keeps it a leaf but out of the calibration
    vector; ``bounds`` are optional physical ``(low, high)`` limits the default must respect.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"coefficient {description!r}: value must be a Python float or int, got {value!r}")
    if isinstance(value, int) and not static:
        raise TypeError(
            f"coefficient {description!r}: an integer default must be static=True "
            "(write a float for a calibratable coefficient)"
        )
    if not isinstance(provenance, Provenance):
        raise TypeError(f"coefficient {description!r}: provenance must be a Provenance, got {provenance!r}")
    if not description.strip():
        raise ValueError("a coefficient needs a description")
    try:
        parse_unit(unit)
    except UnitError as e:
        raise UnitError(f"coefficient {description!r}: {e}") from None
    if bounds is not None:
        lo, hi = bounds
        if not lo <= value <= hi:
            raise ValueError(f"coefficient {description!r}: default {value} outside bounds {bounds}")
    meta: dict[str, Any] = {
        "unit": unit,
        "description": description,
        "fortran_name": fortran_name,
        "dims": (),
        "coefficient": True,
        "calibrate": bool(calibrate and not static),
        "provenance": provenance,
        "source": provenance.source,
        "fortran": provenance.statement,
    }
    if bounds is not None:
        meta["bounds"] = (float(bounds[0]), float(bounds[1]))
    return eqx.field(default=value, static=static, metadata=meta)


def is_coefficient_field(f: dataclasses.Field[Any]) -> bool:
    """True for a field declared with :func:`coef`."""
    return bool(f.metadata.get("coefficient", False))


def iter_coefficients(tree: Any, prefix: str = "") -> Iterator[tuple[str, dataclasses.Field[Any], Any]]:
    """``(dotted path, field, value)`` of every coefficient of ``tree``, in field order.

    Recurses into nested dataclass fields (coefficient groups); a class is instantiated with its
    defaults."""
    if isinstance(tree, type):
        tree = tree()
    for f in dataclasses.fields(tree):
        value = getattr(tree, f.name)
        path = f"{prefix}{f.name}"
        if is_coefficient_field(f):
            yield path, f, value
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            yield from iter_coefficients(value, prefix=path + ".")


def coefficient_table(tree: Any) -> list[dict[str, Any]]:
    """One row per coefficient of ``tree`` (an instance or a class), in field order.

    Keys: ``path``, ``value``, ``unit``, ``description``, ``source`` (one-line citation),
    ``fortran`` (the quoted statement, empty when none), ``static``, ``calibratable``, the
    :class:`Provenance` fields (``ref_version``, ``file``, ``line``, ``routine``, ``statement``,
    ``paper``, ``equation``, ``note``) and ``bounds`` (None when not declared).
    """
    rows = []
    for path, f, value in iter_coefficients(tree):
        prov: Provenance = f.metadata["provenance"]
        rows.append(
            {
                "path": path,
                "value": value,
                "unit": f.metadata["unit"],
                "description": f.metadata["description"],
                "source": f.metadata["source"],
                "fortran": f.metadata["fortran"],
                "static": bool(f.metadata.get("static", False)),
                "calibratable": bool(f.metadata["calibrate"]),
                **prov.as_dict(),
                "bounds": f.metadata.get("bounds"),
            }
        )
    return rows


class Coefficients(Params):
    """Base class of a coefficient set: fields declared with :func:`coef` (or nested sets)."""

    def as_arrays(self: _C, dtype: Any = None) -> _C:
        """The same coefficients with every (non-static) leaf a JAX array, for ``jax.grad``."""
        return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=dtype), self)

    def table(self) -> list[dict[str, Any]]:
        """:func:`coefficient_table` of this instance."""
        return coefficient_table(self)

    def calibratable_paths(self) -> list[str]:
        """Dotted paths of the coefficients a calibration may change (non-static, ``calibrate``)."""
        return [p for p, f, _ in iter_coefficients(self) if f.metadata["calibrate"]]

    def to_vector(
        self, paths: Sequence[str] | None = None, dtype: Any = None
    ) -> tuple[tuple[str, ...], jax.Array]:
        """``(paths, vector)``: the calibratable coefficients (or ``paths``) as one flat array."""
        paths = tuple(self.calibratable_paths() if paths is None else paths)
        allowed = set(self.calibratable_paths())
        bad = [p for p in paths if p not in allowed]
        if bad:
            raise KeyError(f"not calibratable coefficients: {bad}")
        if not paths:
            return paths, jnp.zeros((0,), dtype=dtype)
        return paths, jnp.stack([jnp.asarray(get_path(self, p), dtype=dtype) for p in paths])

    def from_vector(self: _C, vector: Any, paths: Sequence[str] | None = None) -> _C:
        """The inverse of :meth:`to_vector`: this set with ``paths`` replaced by ``vector[i]``."""
        paths = tuple(self.calibratable_paths() if paths is None else paths)
        if len(paths) != jnp.shape(vector)[0]:
            raise ValueError(f"{len(paths)} paths but a vector of shape {jnp.shape(vector)}")
        out = self
        for i, p in enumerate(paths):
            out = set_path(out, p, vector[i])
        return out


#: named numerical guards of the package, ``{name: (value, why)}`` (see :func:`numerical_guard`)
GUARDS: dict[str, tuple[float, str]] = {}


def numerical_guard(name: str, value: float, why: str) -> float:
    """Declare a numerical guard (a floor or tolerance that keeps a kernel finite) and return it.

    ``_TINY = numerical_guard("ceres.tiny", 1e-12, "floor of divisors ...")`` at module level: the
    kernel then uses ``_TINY``, not a bare ``1e-12`` (lint rule AJ007). The value is returned
    unchanged, so naming a guard does not change a result. A guard is not a model coefficient: it
    has no reference and is not calibrated.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"guard {name!r}: value must be a number, got {value!r}")
    if not why.strip():
        raise ValueError(f"guard {name!r} needs a reason")
    old = GUARDS.get(name)
    if old is not None and old[0] != value:
        raise ValueError(f"guard {name!r} already declared with value {old[0]}")
    GUARDS[name] = (value, why)
    return value
