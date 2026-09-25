"""Process registry: versioned keys, duplicate detection, provenance metadata (plan D6)."""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

import agrijax
from agrijax.core.process import (
    FAITHFUL,
    GRIDS,
    PROVENANCE,
    Deviation,
    DuplicateProcessError,
    Process,
    ProcessKey,
    ProcessRegistry,
    Source,
    list_processes,
    lookup,
    metadata_problems,
    process,
    registry,
)

#: every module of the package that defines processes
PROCESS_MODULES = (
    "agrijax.processes.soil_water.richards",
    "agrijax.processes.soil_water.day",
    "agrijax.processes.soil_water.uptake",
    "agrijax.processes.pet.daily",
    "agrijax.processes.crop.ceres_maize.phenology",
    "agrijax.processes.crop.ceres_maize.growth",
    "agrijax.processes.crop.ceres_maize.roots",
    "agrijax.processes.crop.ceres_maize.model",
    "agrijax.models.catpa_pet_demo",
    "agrijax.models.tobacco_demo",
)

EXPECTED = {
    "soil_water/richards@rzwqm2-4.6:faithful": (
        "richards_redistribution",
        "reference_only_conventions",
        "rzwqm2_nodes",
    ),
    "soil_water/day@rzwqm2-4.6:faithful": ("soil_water_day", "reference_only_conventions", "rzwqm2_nodes"),
    "soil_water/day@rzwqm2-4.6:replay_flux": (
        "soil_water_day_replay",
        "reference_only_conventions",
        "rzwqm2_nodes",
    ),
    "soil_water/infiltration_ga@rzwqm2-4.6:faithful": (
        "infiltration_ga",
        "reference_only_conventions",
        "rzwqm2_nodes",
    ),
    "water_supply/rootwu@dssat-4.8.6.0:faithful": ("rootwu_supply", "translated_bsd3", "dssat_layers"),
    "water_supply/forcing_replay@none:replay": ("ceres_water_replay", "equations_only", "dssat_layers"),
    "pet/shuttleworth_wallace@rzwqm2-4.6:faithful": (
        "pet_shuttleworth_wallace",
        "reference_only_conventions",
        "point",
    ),
    "pet/shuttleworth_wallace@rzwqm2-4.6:prescribed_canopy": (
        "sw_pet_from_forcing",
        "reference_only_conventions",
        "point",
    ),
    "pet/asce_reference@asce-ewri-2005:faithful": ("pet_asce_reference", "equations_only", "point"),
    "pet/priestley_taylor@dssat-4.8.6.0:faithful": ("pet_priestley_taylor", "translated_bsd3", "point"),
    "crop/ceres_maize.phenology@dssat-4.8.6.0:faithful": (
        "ceres_phenology",
        "translated_bsd3",
        "dssat_layers",
    ),
    "crop/ceres_maize.stress@dssat-4.8.6.0:faithful": ("ceres_stress", "translated_bsd3", "dssat_layers"),
    "crop/ceres_maize.growth@dssat-4.8.6.0:faithful": ("ceres_growth", "translated_bsd3", "point"),
    "crop/ceres_maize.roots@dssat-4.8.6.0:faithful": ("ceres_roots", "translated_bsd3", "dssat_layers"),
    "crop/ceres_maize.publish@dssat-4.8.6.0:faithful": ("ceres_publish", "translated_bsd3", "dssat_layers"),
    "diagnostic/catpa_pet_totals@none:demo": ("accumulate_totals", "equations_only", "point"),
    "crop/tobacco_demo.calendar@none:demo": ("crop_calendar", "equations_only", "point"),
    "crop/tobacco_demo.leaves@none:demo": ("leaf_appearance_growth", "equations_only", "point"),
    "crop/tobacco_demo.management@none:demo": ("management", "equations_only", "point"),
}

META = {
    "provenance": "equations_only",
    "sources": (("test equation", "unit test"),),
    "grid": "point",
    "deviates": (),
}


@pytest.fixture(scope="module", autouse=True)
def _import_process_modules() -> None:
    for m in PROCESS_MODULES:
        importlib.import_module(m)


def _noop(state, params, forcing_t):
    return state


# ---------------------------------------------------------------------------------- the package


def test_every_package_process_has_complete_metadata() -> None:
    own = [p for p in registry.values() if getattr(p.fn, "__module__", "").startswith("agrijax.")]
    assert len(own) >= len(EXPECTED)
    bad = {p.name: metadata_problems(p) for p in own if metadata_problems(p)}
    assert not bad, bad
    for p in own:
        assert p.info is not None
        assert p.info.provenance in PROVENANCE
        assert p.info.grid in GRIDS
        assert p.info.sources and all(s.what and s.ref for s in p.info.sources)
        assert registry.lookup(str(p.info.key)) is p


def _decorated_processes(root: Path) -> list[tuple[str, str, set[str]]]:
    out = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                call = dec if isinstance(dec, ast.Call) else None
                target = call.func if call is not None else dec
                if isinstance(target, ast.Name) and target.id == "process":
                    kws = {k.arg for k in call.keywords if k.arg} if call is not None else set()
                    out.append((str(path.relative_to(root)), node.name, kws))
    return out


def test_every_decorated_process_in_the_source_is_keyed() -> None:
    """Static guard: a new ``@process`` in the package cannot land without registry metadata."""
    root = Path(agrijax.__file__).parent
    found = _decorated_processes(root)
    assert len(found) == len(EXPECTED)
    required = {"key", "provenance", "sources", "grid", "deviates"}
    missing = [(f, n, sorted(required - kws)) for f, n, kws in found if not required <= kws]
    assert not missing, missing


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_expected_keys_registered(key: str) -> None:
    name, prov, grid = EXPECTED[key]
    p = lookup(key)
    assert p.name == name and p.key == key
    assert registry[name] is p and registry[key] is p
    assert p.info is not None and p.info.provenance == prov and p.info.grid == grid


def test_reference_versions_and_builds() -> None:
    for p in list_processes(ref_version="dssat-4.8.6.0"):
        assert p.info is not None and p.info.provenance == "translated_bsd3" and p.info.ref_build
    for p in list_processes(ref_version="rzwqm2-4.6"):
        # RZWQM2 has no licence text: its source is read for conventions only, never translated
        assert p.info is not None and p.info.provenance == "reference_only_conventions" and p.info.ref_build
    for p in list_processes(ref_version="none"):
        assert p.info is not None and p.info.variant != FAITHFUL


def test_variants_list_their_deviations() -> None:
    for p in list_processes():
        i = p.info
        assert i is not None
        if i.variant != FAITHFUL and i.ref_version != "none":
            assert i.deviates
            faithful = lookup(ProcessKey(i.slot, i.impl, i.ref_version, FAITHFUL))
            assert faithful is not p


def test_list_processes_filters() -> None:
    ceres = list_processes(slot="crop", impl="ceres_maize")
    assert [p.info.impl for p in ceres if p.info] == [
        "ceres_maize.growth",
        "ceres_maize.phenology",
        "ceres_maize.publish",
        "ceres_maize.roots",
        "ceres_maize.stress",
    ]
    assert {p.name for p in list_processes(slot="pet", variant=FAITHFUL)} == {
        "pet_shuttleworth_wallace",
        "pet_asce_reference",
        "pet_priestley_taylor",
    }
    assert {p.name for p in list_processes(grid="rzwqm2_nodes")} == {
        "richards_redistribution",
        "soil_water_day",
        "soil_water_day_replay",
        "infiltration_ga",
    }
    assert {p.name for p in list_processes(slot="soil_water", impl="day")} == {
        "soil_water_day",
        "soil_water_day_replay",
    }
    assert {p.name for p in list_processes(slot="water_supply")} == {"rootwu_supply", "ceres_water_replay"}
    keys = [str(p.key) for p in list_processes()]
    assert keys == sorted(keys) and set(EXPECTED) <= set(keys)


def test_m3_variants_and_their_faithful_siblings() -> None:
    """The M1 replay day is a variant of the faithful RZWQM2 day, and says how it deviates."""
    replay = lookup("soil_water/day@rzwqm2-4.6:replay_flux")
    faithful = lookup("soil_water/day@rzwqm2-4.6:faithful")
    assert replay.info is not None and faithful.info is not None
    assert replay.info.deviates and faithful.info.deviates
    assert replay.writes == faithful.writes == ("soil_water",)
    ga = lookup("soil_water/infiltration_ga@rzwqm2-4.6:faithful")
    assert ga.info is not None and ga.fortran_name == "EVNTRO"
    rootwu = lookup("water_supply/rootwu@dssat-4.8.6.0:faithful")
    assert rootwu.info is not None and rootwu.info.ref_build and rootwu.info.deviates
    for key in ("crop/ceres_maize.publish@dssat-4.8.6.0:faithful", "water_supply/forcing_replay@none:replay"):
        p = lookup(key)
        assert p.info is not None and p.writes and metadata_problems(p) == []


def test_lookup_unknown_key_hints_at_siblings() -> None:
    with pytest.raises(KeyError, match=re.escape("shuttleworth_wallace@rzwqm2-4.6:faithful")):
        lookup("pet/shuttleworth_wallace@rzwqm2-4.6:does_not_exist")
    with pytest.raises(ValueError, match="invalid process key"):
        lookup("richards_redistribution")


def test_as_dict_round_trip() -> None:
    p = lookup("soil_water/richards@rzwqm2-4.6:faithful")
    assert p.info is not None
    d = p.info.as_dict()
    assert d["key"] == p.key and d["slot"] == "soil_water" and d["variant"] == FAITHFUL
    assert d["provenance"] == "reference_only_conventions" and d["deviates"] and d["sources"]
    assert all(set(s) == {"what", "ref"} for s in d["sources"])


# ---------------------------------------------------------------------------------- keys


@pytest.mark.parametrize(
    "text",
    [
        "soil_water/richards@rzwqm2-4.6:faithful",
        "soil_water/richards@rzwqm2-4.6:implicit_tile",
        "crop/ceres_maize.phenology@dssat-4.8.6.0:faithful",
        "pet/asce_reference@asce-ewri-2005:faithful",
        "diagnostic/totals@none:demo",
    ],
)
def test_key_round_trip(text: str) -> None:
    k = ProcessKey.parse(text)
    assert str(k) == text and ProcessKey.parse(k) is k


@pytest.mark.parametrize(
    "text",
    [
        "richards",
        "soil_water/richards",
        "soil_water/richards@rzwqm2-4.6",
        "soil_water/richards:faithful",
        "Soil_water/richards@rzwqm2-4.6:faithful",
        "soil_water/richards@rzwqm2-4.6:",
        "soil_water/richards@-4.6:faithful",
        "soil/water/richards@rzwqm2-4.6:faithful",
        "soil_water/richards@rzwqm2:4.6:faithful",
        "soil_water/.richards@rzwqm2-4.6:faithful",
    ],
)
def test_invalid_keys_raise(text: str) -> None:
    with pytest.raises(ValueError):
        ProcessKey.parse(text)


def test_key_constructor_validates() -> None:
    assert str(ProcessKey("pet", "x", "dssat-4.8.6.0")) == "pet/x@dssat-4.8.6.0:faithful"
    with pytest.raises(ValueError):
        ProcessKey("pet", "x", "DSSAT")


# ---------------------------------------------------------------------------------- duplicates


def test_duplicate_key_raises_and_keeps_the_first() -> None:
    key = "test_slot/dup@none:demo"

    @process(writes=("water",), key=key, **META)
    def first_dup(state, params, forcing_t):
        return state

    try:
        with pytest.raises(DuplicateProcessError, match=re.escape(key)):

            @process(writes=("water",), key=key, **META)
            def second_dup(state, params, forcing_t):
                return state

        assert lookup(key) is first_dup
        assert "second_dup" not in registry
    finally:
        registry.pop("first_dup", None)
    assert key not in registry and "first_dup" not in registry


def test_duplicate_name_raises() -> None:
    """Today's bug: a second process with the same function name silently replaced the first."""

    @process(writes=("water",))
    def same_name(state, params, forcing_t):
        return state

    try:
        with pytest.raises(DuplicateProcessError, match="same_name"):

            @process(writes=("biomass",))
            def same_name(state, params, forcing_t):
                return state

        assert registry["same_name"].writes == ("water",)
    finally:
        registry.pop("same_name", None)


def test_duplicate_name_of_a_package_process_raises() -> None:
    with pytest.raises(DuplicateProcessError, match="richards_redistribution"):
        process(_noop, name="richards_redistribution", writes=("soil_water",))
    with pytest.raises(DuplicateProcessError):
        process(_noop, name="other_name", key="soil_water/richards@rzwqm2-4.6:faithful", **META)
    assert lookup("soil_water/richards@rzwqm2-4.6:faithful").name == "richards_redistribution"
    assert "other_name" not in registry


def test_same_definition_again_replaces() -> None:
    """A module reload re-executes the decorator on the same definition: not a duplicate."""
    key = "test_slot/reload@none:demo"
    a = process(_noop, name="reloaded", key=key, **META)
    try:
        b = process(_noop, name="reloaded", key=key, **META)
        assert a is not b
        assert registry["reloaded"] is b and lookup(key) is b
    finally:
        registry.pop("reloaded", None)
    assert key not in registry


def test_register_false_skips_duplicate_check() -> None:
    p = process(_noop, name="richards_redistribution", writes=("soil_water",), register=False)
    assert p.name == "richards_redistribution"
    assert registry["richards_redistribution"] is not p


def test_local_registry_mapping_protocol() -> None:
    reg = ProcessRegistry()
    p = process(_noop, name="local", key="test_slot/local@none:demo", register=False, **META)
    reg["local"] = p
    assert len(reg) == 1 and list(reg) == ["local"]
    assert reg["test_slot/local@none:demo"] is p and "local" in reg
    with pytest.raises(ValueError):
        reg["other"] = p
    q = process(_noop, name="local2", key="test_slot/local2@none:demo", register=False, **META)
    reg.add(q)
    assert {x.name for x in reg.select(slot="test_slot")} == {"local", "local2"}
    del reg["test_slot/local@none:demo"]
    assert "local" not in reg and len(reg) == 1
    unkeyed = process(_noop, name="plain", register=False)
    reg.add(unkeyed)
    assert unkeyed.key is None and reg["plain"] is unkeyed
    assert reg.select() == [q] and reg.select(include_unkeyed=True) == [q, unkeyed]
    assert set(reg.keyed()) == {"test_slot/local2@none:demo"}


# ---------------------------------------------------------------------------------- metadata checks


def test_metadata_requires_a_key() -> None:
    with pytest.raises(ValueError, match="without a key"):
        process(_noop, provenance="equations_only", register=False)


@pytest.mark.parametrize("missing", ["provenance", "sources", "grid", "deviates"])
def test_key_requires_all_metadata(missing: str) -> None:
    meta = {k: v for k, v in META.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        process(_noop, key="test_slot/m@none:demo", register=False, **meta)


@pytest.mark.parametrize(
    ("key", "overrides", "match"),
    [
        ("test_slot/m@none:demo", {"provenance": "copied"}, "provenance"),
        ("test_slot/m@none:demo", {"grid": "hexagons"}, "grid"),
        ("test_slot/m@none:demo", {"sources": ()}, "no sources"),
        ("test_slot/m@none:demo", {"sources": (("", "x"),)}, "incomplete source"),
        ("test_slot/m@none:demo", {"deviates": (("a", "", "c"),)}, "incomplete deviation"),
        ("test_slot/m@none:faithful", {}, "without reference"),
        ("test_slot/m@none:demo", {"provenance": "translated_bsd3"}, "needs a reference"),
        ("test_slot/m@dssat-4.8.6.0:improved", {}, "lists no deviations"),
    ],
)
def test_incomplete_metadata_raises(key: str, overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        process(_noop, key=key, register=False, **{**META, **overrides})


def test_malformed_source_and_deviation_entries() -> None:
    with pytest.raises(TypeError):
        process(_noop, key="test_slot/m@none:demo", register=False, **{**META, "sources": ("just a string",)})
    with pytest.raises(TypeError):
        process(_noop, key="test_slot/m@none:demo", register=False, **{**META, "deviates": (("a", "b"),)})


def test_variant_with_deviations_accepted() -> None:
    p = process(
        _noop,
        key="test_slot/m@dssat-4.8.6.0:improved",
        register=False,
        provenance="translated_bsd3",
        sources=[Source("eq. 1", "DSSAT-CSM X.for")],
        grid="dssat_layers",
        deviates=[Deviation("jacobi order", "order independence", "unit test")],
        ref_build="dscsm048",
    )
    assert isinstance(p, Process) and p.info is not None
    assert p.key == "test_slot/m@dssat-4.8.6.0:improved"
    assert (p.info.slot, p.info.impl, p.info.ref_version, p.info.variant) == (
        "test_slot",
        "m",
        "dssat-4.8.6.0",
        "improved",
    )
    assert metadata_problems(p) == []


def test_unkeyed_process_reports_missing_metadata() -> None:
    p = process(_noop, name="adhoc", register=False)
    assert p.key is None and p.info is None
    assert metadata_problems(p)
