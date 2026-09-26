"""Collecting conformance cases: this repository's and the plugins' (entry point group
``agrijax.conformance``).

A provider is a module (it has ``cases()`` or ``CASES``) or a callable returning a list of
:class:`~.case.ConformanceCase`. This repository's providers are :data:`._builtin.MODULES`; its own
entry point (``agrijax = "agrijax.testing.conformance.discover:builtin_cases"``) names the same
list, so the built-in and the plugin paths run the same code. Discovery is explicit: nothing is
collected at ``import agrijax`` (M3 coupling contract, decision 8); calling :func:`discover` imports
the providers, and importing a provider registers its processes.

:data:`EXEMPT` lists the registry keys of this repository that have no case yet, each with the
reason; ``tests/unit/conformance/test_registry_coverage.py`` requires every registered key to
have a case or an exemption, and every exemption to name a registered key without a case.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import importlib
import importlib.metadata as md
from collections.abc import Callable, Iterable
from typing import Any

from ._builtin import MODULES
from .case import ConformanceCase

__all__ = ["ENTRY_POINT_GROUP", "EXEMPT", "builtin_cases", "discover", "load_provider", "select"]

ENTRY_POINT_GROUP = "agrijax.conformance"
_BUILTIN_VALUE = "agrijax.testing.conformance.discover:builtin_cases"

_H1 = (
    "the soil-water package is being changed by the H1 hardening workflow on main; its cases are "
    "written after that work merges (M3 contract, section 4.6)"
)
_DEMO = "a demonstration model (ref_version none, variant demo), not an assembly slot"

#: registry keys of this repository without a case yet: key -> reason
EXEMPT: dict[str, str] = {
    "soil_water/day@rzwqm2-4.6:faithful": _H1,
    "soil_water/day@rzwqm2-4.6:replay_flux": _H1,
    "soil_water/infiltration_ga@rzwqm2-4.6:faithful": _H1,
    "soil_water/richards@rzwqm2-4.6:faithful": _H1,
    "pet/shuttleworth_wallace@rzwqm2-4.6:prescribed_canopy": (
        "the CA-TPA PET demonstration's variant (agrijax.models.catpa_pet_demo): the canopy comes from "
        "the forcing; the faithful process it wraps has a case"
    ),
    "diagnostic/catpa_pet_totals@none:demo": _DEMO,
    "crop/tobacco_demo.calendar@none:demo": _DEMO,
    "crop/tobacco_demo.leaves@none:demo": _DEMO,
    "crop/tobacco_demo.management@none:demo": _DEMO,
}


def _agrijax_origin() -> str:
    try:
        return f"agrijax {md.version('agrijax')}"
    except md.PackageNotFoundError:  # run from a source tree (PYTHONPATH) without an install
        return "agrijax (source tree)"


def load_provider(obj: Any) -> list[ConformanceCase]:
    """The cases of a provider: a module path, a module, or a callable."""
    if isinstance(obj, str):
        mod_name, _, attr = obj.partition(":")
        obj = importlib.import_module(mod_name)
        if attr:
            obj = getattr(obj, attr)
    if callable(obj) and not hasattr(obj, "__path__") and not hasattr(obj, "__file__"):
        cases = obj()
    elif hasattr(obj, "cases"):
        cases = obj.cases()
    elif hasattr(obj, "CASES"):
        cases = obj.CASES
    else:
        raise TypeError(f"conformance provider {obj!r} has neither cases() nor CASES")
    if not isinstance(cases, Iterable):
        raise TypeError(f"conformance provider {obj!r} did not return a list of cases")
    cases = list(cases)
    bad = [c for c in cases if not isinstance(c, ConformanceCase)]
    if bad:
        raise TypeError(f"conformance provider {obj!r} returned non-cases: {bad[:3]}")
    return cases


def builtin_cases() -> list[ConformanceCase]:
    """The cases of this repository (:data:`._builtin.MODULES`)."""
    return [c for m in MODULES for c in load_provider(m)]


def _with_origin(cases: Iterable[ConformanceCase], origin: str) -> list[ConformanceCase]:
    return [c if c.origin else dataclasses.replace(c, origin=origin) for c in cases]


def discover(*, builtin: bool = True, entry_points: bool = True) -> list[ConformanceCase]:
    """Every case: this repository's (``builtin``) and those of the ``agrijax.conformance`` entry
    points (``entry_points``), each with its origin (distribution and version). Two cases with
    the same registry key are an error."""
    found: list[ConformanceCase] = []
    if builtin:
        found += _with_origin(builtin_cases(), _agrijax_origin())
    if entry_points:
        for ep in md.entry_points(group=ENTRY_POINT_GROUP):
            if builtin and ep.value.replace(" ", "") == _BUILTIN_VALUE:
                continue  # this repository's own entry point: already collected
            dist = getattr(ep, "dist", None)
            origin = f"{dist.name} {dist.version}" if dist is not None else f"entry point {ep.name}"
            found += _with_origin(load_provider(ep.load()), origin)
    seen: dict[str, str] = {}
    for c in found:
        if c.key in seen:
            raise ValueError(f"two conformance cases for {c.key!r} ({seen[c.key]}, {c.origin})")
        seen[c.key] = c.origin
    return found


def select(
    cases: Iterable[ConformanceCase], *, key: str = "*", package: str | None = None
) -> list[ConformanceCase]:
    """Cases whose key matches the glob ``key`` (``'pet/*'``) and, when given, whose origin is the
    distribution ``package``."""
    match: Callable[[ConformanceCase], bool] = lambda c: fnmatch.fnmatchcase(c.key, key)  # noqa: E731
    out = [c for c in cases if match(c)]
    if package is not None:
        out = [c for c in out if c.origin.split(" ", 1)[0] == package]
    return out
