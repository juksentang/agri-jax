"""Checked dimensions and machine-readable units (plan D7 / A9, minimal part).

* every leaf field of every State / Params / Forcing class of the package declares a unit that
  :func:`agrijax.core.units.parse_unit` accepts and dims that :func:`agrijax.core.dims.parse_dims`
  resolves;
* the Brooks-Corey, PET and CERES-Maize fields carry the units their docstrings state, and the
  equations those docstrings write are dimensionally consistent;
* unit conversions live in the named adapters of :data:`agrijax.core.units.ADAPTERS` (round trip
  and factor tested); the few conversion constants elsewhere are pinned against the parser;
* :func:`agrijax.core.dims.check_tree_dims` binds axis sizes across a tree, ignores leading batch
  axes, names the field path on a mismatch, and runs inside :func:`agrijax.core.runtime.run`.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import math
import pkgutil
import re
import typing
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

import agrijax
from agrijax.core import units as U
from agrijax.core.dims import DIMS, DimsError, check_tree_dims, parse_dim, parse_dims
from agrijax.core.model import Model
from agrijax.core.runtime import run, run_batch
from agrijax.core.state import Forcing, Params, State, _Base, field
from agrijax.core.units import ADAPTERS, UnitError, conversion_factor, parse_unit

SRC = Path(agrijax.__file__).resolve().parent


# ======================================================================== every field of the package
def _import_all() -> list[type]:
    for m in pkgutil.walk_packages(agrijax.__path__, "agrijax."):
        importlib.import_module(m.name)

    out: list[type] = []
    stack: list[type] = [_Base]
    while stack:
        c = stack.pop()
        for s in c.__subclasses__():
            stack.append(s)
            if s.__module__.startswith("agrijax.") and s not in out:
                out.append(s)
    from agrijax.core.organs import OrganQueue

    out.append(OrganQueue)
    return sorted(out, key=lambda c: f"{c.__module__}.{c.__qualname__}")


CLASSES = _import_all()


def _is_subtree(cls: type, f: dataclasses.Field[Any]) -> bool:
    """A field holding a nested module (not an array leaf)."""
    hints = typing.get_type_hints(cls, include_extras=False)
    t = hints.get(f.name)
    args = typing.get_args(t) or (t,)
    return any(isinstance(a, type) and dataclasses.is_dataclass(a) for a in args)


def _leaf_fields() -> list[tuple[str, dataclasses.Field[Any]]]:
    out = []
    for cls in CLASSES:
        for f in dataclasses.fields(cls):
            if f.metadata.get("static", False) or _is_subtree(cls, f):
                continue
            out.append((f"{cls.__module__}.{cls.__qualname__}.{f.name}", f))
    return out


LEAVES = _leaf_fields()


def test_walk_finds_the_known_classes() -> None:
    names = {c.__qualname__ for c in CLASSES}
    for expected in (
        "SoilHydraulicParams",
        "RichardsParams",
        "SoilWater",
        "PETSiteParams",
        "PETFluxes",
        "CeresMaizeState",
        "CeresCultivar",
        "GrosubCoefficients",
        "EventTable",
        "OrganQueue",
    ):
        assert expected in names
    assert len(LEAVES) > 250


def test_every_leaf_unit_parses() -> None:
    bad = []
    for path, f in LEAVES:
        try:
            parse_unit(f.metadata.get("unit", ""))
        except UnitError as e:
            bad.append(f"{path}: {e}")
    assert not bad, "\n".join(bad)


def test_every_leaf_declares_dims() -> None:
    bad = []
    for path, f in LEAVES:
        dims = f.metadata.get("dims")
        if dims is None:
            bad.append(f"{path}: no dims")
            continue
        try:
            parse_dims(dims)
        except DimsError as e:
            bad.append(f"{path}: {e}")
    assert not bad, "\n".join(bad)


# ======================================================================== units the docstrings state
def _unit(cls: type, name: str) -> str:
    return {f.name: f for f in dataclasses.fields(cls)}[name].metadata["unit"]


def _same(a: str, b: str) -> bool:
    return parse_unit(a).same_dimension(parse_unit(b)) and math.isclose(
        parse_unit(a).scale, parse_unit(b).scale, rel_tol=1e-12
    )


def test_brooks_corey_units_match_docstring() -> None:
    from agrijax.processes.soil_water import hydraulics
    from agrijax.processes.soil_water.hydraulics import SoilHydraulicParams as P

    doc = hydraulics.__doc__ or ""
    assert "``h`` [cm]" in doc  # the head is in cm, so are the bubbling pressures
    for name in ("hb", "hb_k"):
        assert _same(_unit(P, name), "cm")
    for name in ("theta_r", "theta_s", "fc13", "fc110", "wp"):
        assert _same(_unit(P, name), "cm3 cm-3")
    for name in ("lambda_", "eps", "n1"):
        assert parse_unit(_unit(P, name)).dimensionless
    assert _same(_unit(P, "ksat"), "cm hr-1")  # RZWQM SOILHP(4)
    # theta_s + a1 h: a1 h has the dimension of theta ([cm3 cm-3 cm-1] in the module source)
    assert "[cm3 cm-3 cm-1]" in Path(hydraulics.__file__).read_text()
    assert parse_unit(f"({_unit(P, 'a1')}) cm").same_dimension(parse_unit(_unit(P, "theta_s")))
    # C2 = Ks hb_k^(eps - N1): its dimension depends on eps
    assert parse_unit(_unit(P, "c2")).parametric == ("eps",)


def test_pet_units_match_docstring() -> None:
    from agrijax.processes.pet import daily
    from agrijax.processes.pet.daily import DailyWeather, PETFluxes

    doc = daily.__doc__ or ""
    rows = re.findall(r"state\.pet\.(\{[^}]*\}|\w+)\s+(\S+ \S+) \(", doc)
    assert len(rows) == 3
    seen = set()
    for names, unit in rows:
        for name in names.strip("{}").split(","):
            name = name.strip()
            seen.add(name)
            assert _unit(PETFluxes, name) == unit, name
    assert seen == {f.name for f in dataclasses.fields(PETFluxes)}
    # forcing sentence of the docstring
    for name, unit in (("tmin", "degC"), ("tmax", "degC"), ("srad", "MJ m-2 d-1"), ("rh", "percent")):
        assert _unit(DailyWeather, name) == unit
    assert "``srad`` MJ m-2 d-1, ``rh`` percent, ``wind_run``\nkm d-1" in doc
    assert _unit(DailyWeather, "wind_run") == "km d-1"


def test_ceres_units_match_docstring_and_equations() -> None:
    from agrijax.core.organs import OrganQueue
    from agrijax.processes.crop.ceres_maize import state as cs

    doc = cs.__doc__ or ""
    assert "``PLA`` [cm2 plant-1]" in doc and "``LFWT`` [g plant-1]" in doc
    assert _unit(OrganQueue, "area") == "cm2 plant-1"
    assert _unit(OrganQueue, "mass") == "g plant-1"
    assert "``YYYYDDD`` date" in doc
    assert parse_unit(_unit(cs.CeresPhenologyState, "stgdoy")).is_date
    assert parse_unit(_unit(cs.CeresMaizeParams, "yrplt")).is_date

    cul, spe, par = cs.CeresCultivar, cs.CeresSpecies, cs.CeresMaizeParams
    grow, fo = cs.CeresGrowthState, cs.CeresForcing

    def dim(*factors: str) -> Any:
        return parse_unit(" ".join(f"({x})" if not x.endswith(")-1") else x for x in factors))

    # CARBO = RUE * PAR / PLTPOP: g MJ-1 * MJ m-2 d-1 / plant m-2 = g plant-1 d-1
    carbo = dim(_unit(cul, "rue"), _unit(fo, "srad"), f"({_unit(par, 'pltpop')})-1")
    assert carbo.same_dimension(parse_unit(_unit(grow, "carbo")))
    # grain growth: G3 [mg kernel-1 d-1] * GPP [kernel plant-1] -> g plant-1 d-1
    grogrn = dim(_unit(cul, "g3"), _unit(cs.CeresPhenologyState, "gpp"))
    assert grogrn.same_dimension(parse_unit(_unit(grow, "grogrn")))
    # emergence thermal time: GDDE [degC d cm-1] * SDEPTH [cm] -> degC d, as P1 and PHINT
    assert dim(_unit(cul, "gdde"), _unit(par, "sdepth")).same_dimension(parse_unit(_unit(cul, "p1")))
    assert parse_unit(_unit(cul, "p1")).same_dimension(parse_unit(_unit(cul, "phint")))
    # leaf area index: PLA [cm2 plant-1] * PLTPOP [plant m-2] -> m2 m-2 (factor 1e-4)
    lai = dim(_unit(OrganQueue, "area"), _unit(par, "pltpop"))
    assert lai.dimensionless and math.isclose(lai.scale, 1e-4)
    assert _unit(spe, "plae") == "cm2 plant-1"


def test_ceres_coefficients_are_scalar_with_units() -> None:
    from agrijax.processes.crop.ceres_maize.coefficients import CeresCoefficients

    n = 0
    for sub in dataclasses.fields(CeresCoefficients):
        for f in dataclasses.fields(typing.get_type_hints(CeresCoefficients)[sub.name]):
            parse_unit(f.metadata["unit"])
            assert f.metadata["dims"] == ()
            n += 1
    assert n >= 100


# ======================================================================== the unit parser
@pytest.mark.parametrize(
    ("text", "other", "factor"),
    [
        ("mm", "cm", 0.1),
        ("MJ m-2 d-1", "W m-2", 1e6 / 86400),
        ("kg ha-1", "g m-2", 0.1),
        ("km d-1", "m s-1", 1e3 / 86400),
        ("cm hr-1", "cm d-1", 24.0),
        ("(degC d)-1", "K-1 d-1", 1.0),
        ("cm2 g-(1/1.25)", "cm2 g-0.8", 1.0),
        ("cm g-1 x 1e4", "m g-1", 100.0),
        ("kg ha-1 per MJ m-2", "kg MJ-1 ha-1 m2", 1.0),
        ("%", "-", 0.01),
        ("percent", "%", 1.0),
        ("MJ m-2", "cal cm-2", 1e6 / 4.184 / 1e4),
    ],
)
def test_conversion_factor(text: str, other: str, factor: float) -> None:
    assert math.isclose(conversion_factor(text, other), factor, rel_tol=1e-12)


@pytest.mark.parametrize("bad", ["", "furlong", "cm^", "(cm", "cm)", "cm -", "m-(1/)"])
def test_parse_unit_rejects(bad: str) -> None:
    with pytest.raises(UnitError):
        parse_unit(bad)


def test_incompatible_and_special_units() -> None:
    with pytest.raises(UnitError):
        conversion_factor("cm", "g")
    with pytest.raises(UnitError):
        conversion_factor("g plant-1", "g m-2")  # counted things never mix silently
    with pytest.raises(UnitError):
        conversion_factor("cm hr-1 cm^eps", "cm hr-1")  # parametric
    with pytest.raises(UnitError):
        conversion_factor("YYYYDDD", "YYYYDDD")  # a date code is not a quantity
    assert parse_unit("-").dimensionless
    assert parse_unit("g cm-2.5").same_dimension(parse_unit("g cm-5 cm2.5"))


# ======================================================================== named adapters
@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_adapter_round_trip_and_factor(name: str) -> None:
    a = ADAPTERS[name]
    inv = ADAPTERS[a.inverse]
    assert (inv.src, inv.dst, inv.inverse) == (a.dst, a.src, a.name)
    x = np.array([-40.0, 0.0, 1.0, 3.7, 1234.5])
    np.testing.assert_allclose(inv.fn(a.fn(x)), x, rtol=1e-14, atol=1e-12)
    np.testing.assert_allclose(np.asarray(a.fn(jnp.asarray(x))), a.fn(x), rtol=1e-6)  # works on jax arrays
    slope = a.fn(1.0) - a.fn(0.0)
    if a.affine:
        assert parse_unit(a.src).same_dimension(parse_unit(a.dst))
        assert math.isclose(slope, 1.0)
        zero = {("degC", "K"): U.KELVIN_OFFSET, ("K", "degC"): -U.KELVIN_OFFSET}[(a.src, a.dst)]
        assert math.isclose(a.fn(0.0), zero)
        return
    assert math.isclose(a.fn(0.0), 0.0, abs_tol=0.0)
    if not a.via:
        assert math.isclose(slope, conversion_factor(a.src, a.dst), rel_tol=1e-12)
        return
    # a physical constant in between: src / via = dst (forward) or src * via = dst (inverse)
    values = {"MJ kg-1": U.LATENT_HEAT_MJ_PER_KG, "kg m-3": U.WATER_DENSITY_KG_M3}
    prod = math.prod(values[v] for v in a.via)
    vtext = " ".join(f"({v})" for v in a.via)
    divided = parse_unit(f"({a.src}) ({vtext})-1")
    multiplied = parse_unit(f"({a.src}) {vtext}")
    dst = parse_unit(a.dst)
    if divided.same_dimension(dst):
        expect = divided.scale / dst.scale / prod
    else:
        assert multiplied.same_dimension(dst)
        expect = multiplied.scale / dst.scale * prod
    assert math.isclose(slope, expect, rel_tol=1e-12)


def test_every_conversion_function_of_units_is_an_adapter() -> None:
    funcs = {n for n, v in vars(U).items() if callable(v) and re.fullmatch(r"[a-z0-9_]+_to_[a-z0-9_]+", n)}
    assert funcs == set(ADAPTERS)


#: conversion constants defined outside ``core/units.py``: ``(module, name) -> (src, dst, rtol)``
#: with ``value = conversion_factor(src, dst)``; a new one must be added here (or moved into an
#: adapter), so conversions never hide in process code.
PINNED: dict[tuple[str, str], Any] = {
    ("agrijax.io.rzwqm.events", "_IRRIG_UNIT_TO_CM"): {
        "": ("cm", "cm"),
        "cm": ("cm", "cm"),
        "mm": ("mm", "cm"),
    },
    ("agrijax.processes.soil_water.richards", "HOURS_PER_DAY"): ("d", "h", 0.0),
    ("agrijax.processes.pet.shuttleworth_wallace", "KM_DAY_TO_M_S"): ("km d-1", "m s-1", 0.0),
    ("agrijax.processes.pet.shuttleworth_wallace", "SECONDS_PER_DAY"): ("d", "s", 0.0),
    # 12/pi hours per radian of the Earth's rotation: an astronomical constant, not a unit change
    ("agrijax.processes.pet.shuttleworth_wallace", "_HOURS_PER_RADIAN"): 12.0 / math.pi,
    # RDPD = RTDEP / 100 in MZ_OPGROW: the value is cm per m (the name reads the other way)
    ("agrijax.processes.crop.ceres_maize.model", "_CM_TO_M"): ("m", "cm", 0.0),
    # DSSAT PETPT SLANG = SRAD * 23.923 (the exact thermochemical value is 23.9006)
    ("agrijax.processes.pet.priestley_taylor", "LANGLEY_PER_MJ_M2"): ("MJ m-2", "cal cm-2", 1e-3),
    # CERES-Maize: G3 in mg kernel-1 d-1 (MZ_GROSUB GROGRN), SUMP g -> mg (MZ_PHENOL PSKER)
    ("agrijax.processes.crop.ceres_maize.constants", "G_PER_MG"): ("mg", "g", 0.0),
    ("agrijax.processes.crop.ceres_maize.constants", "MG_PER_G"): ("g", "mg", 0.0),
    # sun geometry of DSSAT SOLAR.for / MZ_PHENOL: hours of half a day, degrees in pi radians, 2 pi
    ("agrijax.processes.crop.ceres_maize.constants", "HOURS_PER_HALF_DAY"): 12.0,
    ("agrijax.processes.crop.ceres_maize.constants", "DEG_PER_HALF_TURN"): 180.0,
    ("agrijax.processes.crop.ceres_maize.constants", "FULL_TURN_PER_PI"): 2.0,
}

_CONV_NAME = re.compile(r"_TO_|_PER_")


def _numeric(v: Any) -> bool:
    if isinstance(v, dict):
        return bool(v) and all(isinstance(x, (int, float)) for x in v.values())
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _module_constants() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC.parent).with_suffix("")
        mod = ".".join(rel.parts).removesuffix(".__init__")
        if mod == "agrijax.core.units":
            continue
        for node in ast.parse(path.read_text()).body:
            targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
            for t in targets:
                if isinstance(t, ast.Name) and _CONV_NAME.search(t.id):
                    if _numeric(getattr(importlib.import_module(mod), t.id)):
                        found.add((mod, t.id))
    return found


def test_conversion_constants_outside_units_are_pinned() -> None:
    found = _module_constants()
    assert found == set(PINNED), f"unpinned: {found - set(PINNED)}, stale: {set(PINNED) - found}"
    for (mod, name), spec in PINNED.items():
        value = getattr(importlib.import_module(mod), name)
        if isinstance(spec, float):
            assert math.isclose(value, spec, rel_tol=1e-15)
        elif isinstance(spec, dict):
            assert value.keys() == spec.keys()
            for k, (a, b) in spec.items():
                assert math.isclose(value[k], conversion_factor(a, b), rel_tol=1e-12), (mod, name, k)
        else:
            a, b, rtol = spec
            assert math.isclose(value, conversion_factor(a, b), rel_tol=max(rtol, 1e-12)), (mod, name)


# ======================================================================== dims registry and check
def test_parse_dim_forms() -> None:
    assert parse_dim("n_node") == (("n_node", 0, False))
    assert parse_dim("n_node-1") == ("n_node", -1, False)
    assert parse_dim("n_horizon?") == ("n_horizon", 0, True)
    assert parse_dim("4") == (None, 4, False)
    assert parse_dim("hour") == (None, 24, False)  # a fixed-size name is a literal
    assert parse_dim("n_day") == ("T", 0, False)  # alias
    assert parse_dims(None) is None
    assert parse_dims(()) == ()
    for bad in ("n_bogus", "n node", "-3", ""):
        with pytest.raises(DimsError):
            parse_dim(bad)
    with pytest.raises(DimsError):
        parse_dims(("n_crop", "n_horizon?"))  # optional axes lead
    for name in ("n_crop", "n_cohort", "n_node", "n_layer", "T", "n_pool"):
        assert name in DIMS


def test_unknown_axis_fails_at_class_definition() -> None:
    with pytest.raises(DimsError, match="n_bogus"):
        field(unit="cm", dims=("n_bogus",))


class _S(State):
    x: Array = field(unit="cm", dims=("n_node",))
    face: Array = field(unit="cm", dims=("n_node-1",))
    q: Array = field(unit="cm", dims=("n_crop", "3"))


class _P(Params):
    k: Array = field(unit="-", dims=())
    soil: Array = field(unit="cm", dims=("n_horizon?",))


class _F(Forcing):
    r: Array = field(unit="cm d-1", dims=("T",))
    layered: Array = field(unit="cm d-1", dims=("T", "n_node"))


def _good(n: int = 5, t: int = 4) -> tuple[_S, _P, _F]:
    s = _S(x=jnp.zeros(n), face=jnp.zeros(n - 1), q=jnp.zeros((2, 3)))
    p = _P(k=jnp.asarray(0.5), soil=jnp.asarray(1.0))
    f = _F(r=jnp.ones(t), layered=jnp.ones((t, n)))
    return s, p, f


def test_check_tree_dims_binds_across_the_tree() -> None:
    s, p, f = _good()
    sizes = check_tree_dims({"state": s, "params": p, "forcing": f})
    assert sizes == {"n_node": 5, "n_crop": 2, "T": 4}
    # optional axis present, and a leading batch axis, are both accepted
    p3 = _P(k=jnp.zeros((7,)), soil=jnp.ones(3))
    assert check_tree_dims(p3) == {"n_horizon": 3}


@pytest.mark.parametrize(
    ("change", "path"),
    [
        (lambda s, p, f: (s.replace(face=jnp.zeros(5)), p, f), "state.face"),
        (lambda s, p, f: (s.replace(q=jnp.zeros((2, 4))), p, f), "state.q"),
        (lambda s, p, f: (s, p, f.replace(layered=jnp.ones((4, 6)))), "forcing.layered"),
        (lambda s, p, f: (s, p, f.replace(r=jnp.ones(3))), "forcing.r"),
        (lambda s, p, f: (s.replace(x=jnp.asarray(0.0)), p, f), "state.x"),
    ],
)
def test_check_tree_dims_names_the_path(change: Any, path: str) -> None:
    s, p, f = change(*_good())
    with pytest.raises(DimsError, match=re.escape(path)):
        check_tree_dims({"params": p, "forcing": f, "state": s})


def _step(state: _S, params: _P, forcing_t: _F) -> _S:
    return state.replace(x=state.x + params.k * forcing_t.r + forcing_t.layered)


def test_runtime_run_checks_dims_at_trace_time() -> None:
    model = Model(_S, [_step], outputs=("x",))
    s, p, f = _good()
    out = jax.jit(lambda p, f, s: run(model, p, f, s))(p, f, s)
    assert out["x"].shape == (4, 5)
    batched = _P(k=jnp.linspace(0.0, 1.0, 3), soil=jnp.ones(3))
    assert run_batch(model, batched, f, s)["x"].shape == (3, 4, 5)
    bad = s.replace(face=jnp.zeros(2))
    with pytest.raises(DimsError, match=r"state\.face"):
        jax.jit(lambda p, f, s: run(model, p, f, s))(p, f, bad)


def test_check_adds_nothing_to_the_program() -> None:
    model = Model(_S, [_step], outputs=("x",))
    s, p, f = _good()
    from agrijax.core import runtime

    with_check = jax.make_jaxpr(lambda p, f, s: run(model, p, f, s))(p, f, s)
    orig = runtime.check_tree_dims
    try:
        runtime.check_tree_dims = lambda *a, **k: {}  # type: ignore[assignment]
        without = jax.make_jaxpr(lambda p, f, s: run(model, p, f, s))(p, f, s)
    finally:
        runtime.check_tree_dims = orig
    assert str(with_check) == str(without)


def test_real_pytrees_pass() -> None:
    from agrijax.models import tobacco_demo
    from agrijax.processes.crop.ceres_maize import CeresMaizeState

    from .test_ceres_phenology import make_params
    from .toy import toy_inputs

    st = tobacco_demo.initial_state(n_cohort=12, n_crop=2)
    sizes = check_tree_dims({"params": tobacco_demo.default_params(), "state": st})
    assert sizes == {"n_crop": 2, "n_cohort": 12}
    p = make_params()
    sizes = check_tree_dims({"params": p, "state": CeresMaizeState.initial(p, n_crop=2)})
    assert sizes["n_crop"] == 2 and sizes["n_layer"] == 9
    check_tree_dims(toy_inputs())
