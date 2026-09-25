"""``agrijax.core.coefficients``: the coefficient declaration helper (``coef`` / ``Provenance``).

* a coefficient carries value, unit (checked by the units parser at class definition), meaning and
  a structured provenance; bad units, integer leaves, out-of-bounds defaults are refused;
* provenance follows the licence rules: the source statement may be quoted only for a reference
  that allows it (DSSAT-CSM, BSD-3), RZWQM2 coefficients cite file, line, routine and the
  published equation instead; ``ref_version`` has the form of a process registry key and names a
  registered reference, and every ``ref_version`` of the process registry is one;
* a coefficient set is calibratable by default: array leaves, a flat vector and back, gradients;
* the CERES-Maize coefficients are declared through the helper (same values and table keys).
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import agrijax.models  # registers the processes of the assembled models
import agrijax.processes.crop.ceres_maize as ceres
import agrijax.processes.pet
import agrijax.processes.soil_water  # noqa: F401
from agrijax.core import Coefficients, Provenance, coef, coefficient_table
from agrijax.core import coefficients as cc
from agrijax.core.process import list_processes
from agrijax.core.units import UnitError, parse_unit

DSSAT = Provenance("dssat-4.8.6.0", file="SPAM/WATBAL.for", line=12, routine="WATBAL", statement="X = 0.5*Y")
RZ = Provenance(
    "rzwqm2-4.6", file="src/phys.for", line=40, routine="RICH", paper="Ahuja et al. (2000)", equation="3.12"
)


class _Inner(Coefficients):
    a: float = coef(0.5, "d-1", "a rate", DSSAT)
    n: int = coef(3, "d", "a day count", DSSAT, static=True)
    pi: float = coef(3.14, "-", "pi as the reference writes it", DSSAT, calibrate=False)


class _Outer(Coefficients):
    k: float = coef(2.0, "cm d-1", "a conductivity", RZ, bounds=(0.0, 10.0))
    inner: _Inner = eqx.field(default_factory=_Inner)


# ------------------------------------------------------------------------------ declaration
def test_table_rows_carry_value_unit_meaning_and_provenance() -> None:
    rows = coefficient_table(_Outer)
    assert [r["path"] for r in rows] == ["k", "inner.a", "inner.n", "inner.pi"]
    k, a, n, pi = rows
    assert k["value"] == 2.0 and k["unit"] == "cm d-1" and k["bounds"] == (0.0, 10.0)
    assert k["ref_version"] == "rzwqm2-4.6" and k["routine"] == "RICH" and k["statement"] == ""
    assert k["source"] == "RZWQM2 4.6 src/phys.for:40; Ahuja et al. (2000), eq. 3.12"
    assert a["source"] == "DSSAT-CSM v4.8.6.0 SPAM/WATBAL.for:12" and a["fortran"] == "X = 0.5*Y"
    assert (a["static"], a["calibratable"]) == (False, True)
    assert (n["static"], n["calibratable"]) == (True, False)
    assert (pi["static"], pi["calibratable"]) == (False, False)
    for r in rows:
        parse_unit(r["unit"])
    assert coefficient_table(_Outer()) == rows == _Outer().table()


def test_units_are_checked_at_declaration() -> None:
    with pytest.raises(UnitError, match="a bad unit"):

        class _Bad(Coefficients):
            x: float = coef(1.0, "furlong fortnight-1", "a bad unit", DSSAT)


@pytest.mark.parametrize(
    ("value", "kw", "exc"),
    [
        (3, {}, TypeError),  # an integer leaf would not be differentiable
        (True, {"static": True}, TypeError),
        ("1.0", {}, TypeError),
        (11.0, {"bounds": (0.0, 10.0)}, ValueError),
    ],
)
def test_bad_values_are_refused(value, kw, exc) -> None:
    with pytest.raises(exc):
        coef(value, "-", "x", DSSAT, **kw)


def test_description_and_provenance_are_required() -> None:
    with pytest.raises(ValueError):
        coef(1.0, "-", "  ", DSSAT)
    with pytest.raises(TypeError):
        coef(1.0, "-", "x", "MZ_GROSUB.for:12")  # type: ignore[arg-type]


# ------------------------------------------------------------------------------ provenance
@pytest.mark.parametrize(
    ("kw", "match"),
    [
        (
            {"ref_version": "rzwqm2-4.6", "file": "a.for", "line": 3, "statement": "X=1", "paper": "P"},
            "quoted",
        ),
        ({"ref_version": "rzwqm2-4.6", "file": "a.for", "line": 3}, "published equation"),
        ({"ref_version": "DSSAT 4.8", "file": "a.for"}, "not a registry reference"),
        ({"ref_version": "apsim-7.10", "file": "a.for"}, "unknown reference"),
        ({"ref_version": "dssat-4.8.6.0", "line": 3, "paper": "P"}, "needs the file"),
        ({"ref_version": "dssat-4.8.6.0", "file": "a.for", "statement": "X=1"}, "file and line"),
        ({"ref_version": "dssat-4.8.6.0", "file": "a.for", "line": 0}, "positive"),
        ({"ref_version": "dssat-4.8.6.0"}, "neither"),
        ({"ref_version": "none", "file": "a.for"}, "needs a reference"),
        ({"ref_version": "none", "paper": "P", "statement": "X=1"}, "quoted"),
        ({"ref_version": "dssat-4.8.6.0", "file": "a.for", "equation": "7"}, "needs the paper"),
    ],
)
def test_provenance_rules(kw, match) -> None:
    with pytest.raises(ValueError, match=match):
        Provenance(**kw)


def test_provenance_forms() -> None:
    p = Provenance.at("dssat-4.8.6.0", "Plant/CERES-Maize/MZ_GROSUB.for:911", statement="SWMIN = STMWT*0.85")
    assert (p.file, p.line) == ("Plant/CERES-Maize/MZ_GROSUB.for", 911)
    assert p.source == "DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for:911"
    assert p.reference is not None and p.reference.statement_allowed
    paper = Provenance("asce-ewri-2005", paper="ASCE-EWRI (2005)", equation="1")
    assert paper.source == "ASCE-EWRI (2005), eq. 1"
    assert Provenance("none", paper="Priestley and Taylor (1972)").source == "Priestley and Taylor (1972)"
    with pytest.raises(ValueError):
        Provenance.at("dssat-4.8.6.0", "MZ_GROSUB.for")
    rz = cc.REFERENCES["rzwqm2"]
    assert not rz.statement_allowed and rz.needs_paper


def test_register_reference_records_the_licence_decision_once() -> None:
    assert cc.register_reference("dssat", "DSSAT-CSM v{version}", "BSD-3-Clause", statement_allowed=True)
    with pytest.raises(ValueError, match="already registered"):
        cc.register_reference("dssat", "DSSAT-CSM v{version}", "BSD-3-Clause", statement_allowed=False)


def test_every_registry_ref_version_is_a_known_reference() -> None:
    """``Provenance.ref_version`` and the process registry keys use the same references."""
    refs = {p.info.ref_version for p in list_processes() if p.info is not None}
    assert {"dssat-4.8.6.0", "rzwqm2-4.6", "asce-ewri-2005"} <= refs
    for r in refs:
        Provenance(r, paper="P")  # parses and names a registered reference (or is 'none')


# ------------------------------------------------------------------------------ calibration
def test_calibratable_by_default_array_leaves_and_vector_round_trip() -> None:
    c = _Outer()
    assert c.calibratable_paths() == ["k", "inner.a"]
    assert len(jax.tree_util.tree_leaves(c)) == 3  # k, a, pi (n is static)
    arr = c.as_arrays()
    assert all(isinstance(x, jax.Array) for x in jax.tree_util.tree_leaves(arr))
    assert arr.inner.n == 3 and isinstance(arr.inner.n, int)
    paths, v = c.to_vector()
    assert paths == ("k", "inner.a") and np.array_equal(np.asarray(v), [2.0, 0.5])
    back = c.from_vector(v * 2.0)
    assert float(back.k) == 4.0 and float(back.inner.a) == 1.0 and back.inner.pi == 3.14
    with pytest.raises(KeyError):
        c.to_vector(["inner.n"])
    with pytest.raises(ValueError):
        c.from_vector(jnp.ones(3))


def test_gradient_through_the_vector() -> None:
    c = _Outer()

    def loss(v):
        s = c.from_vector(v)
        return s.k**2 * s.inner.a + s.inner.pi

    _, v = c.to_vector()
    g = np.asarray(jax.grad(loss)(v))
    np.testing.assert_allclose(g, [2 * 2.0 * 0.5, 2.0**2])


def test_numerical_guard_is_named_and_unchanged() -> None:
    assert cc.numerical_guard("test.tiny", 1e-12, "floor of a test divisor") == 1e-12
    assert cc.GUARDS["test.tiny"] == (1e-12, "floor of a test divisor")
    with pytest.raises(ValueError):
        cc.numerical_guard("test.tiny", 1e-9, "other value")
    with pytest.raises(ValueError):
        cc.numerical_guard("test.nowhy", 1e-9, " ")


# ------------------------------------------------------------------------------ CERES-Maize migration
def test_ceres_coefficients_are_declared_through_the_helper() -> None:
    rows = ceres.coefficient_table()
    assert len(rows) == 112
    assert issubclass(ceres.CeresCoefficients, Coefficients)
    for r in rows:
        assert {"path", "value", "unit", "description", "source", "fortran", "static"} <= set(r)
        assert r["ref_version"] == ceres.coefficients.REF_VERSION == "dssat-4.8.6.0"
        assert r["routine"] in {"MZ_GROSUB", "MZ_PHENOL", "MZ_ROOTGR"}
        assert r["statement"] == r["fortran"] and r["source"].endswith(f"{r['file']}:{r['line']}")
        parse_unit(r["unit"])
    for group in dataclasses.fields(ceres.CeresCoefficients):
        for f in dataclasses.fields(getattr(ceres.DSSAT_COEFFICIENTS, group.name)):
            assert cc.is_coefficient_field(f), f.name
    assert ceres.DSSAT_COEFFICIENTS.calibratable_paths() == [r["path"] for r in rows if not r["static"]]
