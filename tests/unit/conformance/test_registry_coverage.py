"""Every registry key has a conformance case or a written exemption; discovery, selection, the
entry-point path and the command line of the kit (no JAX run)."""

from __future__ import annotations

import importlib
import importlib.metadata as md
import pkgutil
from types import SimpleNamespace

import pytest

import agrijax.models
import agrijax.processes
from agrijax.core.ports import NAMESPACES, RESERVED_NAMESPACES
from agrijax.core.process import registry
from agrijax.testing.conformance import (
    EXEMPT,
    SLOT_CONTRACTS,
    ConformanceCase,
    builtin_cases,
    discover,
    select,
)
from agrijax.testing.conformance import __main__ as cli
from agrijax.testing.conformance.case import DEFAULT_OWN

#: the module (the package re-exports the function ``discover`` under the same name)
discover_mod = importlib.import_module("agrijax.testing.conformance.discover")


@pytest.fixture(scope="module", autouse=True)
def _import_every_process_module() -> None:
    for pkg in (agrijax.processes, agrijax.models):
        for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            importlib.import_module(m.name)


def test_every_registered_key_has_a_case_or_an_exemption() -> None:
    cases = builtin_cases()
    have = {c.key for c in cases}
    registered = set(registry.keyed())
    missing = sorted(registered - have - set(EXEMPT))
    assert not missing, f"registry keys without a conformance case or an exemption: {missing}"
    assert not sorted(set(EXEMPT) & have), "an exempted key has a case: drop the exemption"
    assert not sorted(set(EXEMPT) - registered), "an exemption names no registered key"
    assert all(why.strip() for why in EXEMPT.values())
    # a case either tests a registered process or brings its own (kit fixtures)
    assert all(c.key in registered or c.process is not None for c in cases)
    assert len(have) == len(cases)  # one case per key


def test_first_cases_cover_crop_uptake_grids_and_pet() -> None:
    have = {c.key for c in builtin_cases()}
    for key in (
        "crop/ceres_maize.phenology@dssat-4.8.6.0:faithful",
        "crop/ceres_maize.stress@dssat-4.8.6.0:faithful",
        "crop/ceres_maize.growth@dssat-4.8.6.0:faithful",
        "crop/ceres_maize.growth@dssat-4.8.6.0:nstress_replay",
        "crop/ceres_maize.roots@dssat-4.8.6.0:faithful",
        "crop/ceres_maize.publish@dssat-4.8.6.0:faithful",
        "water_supply/rootwu@dssat-4.8.6.0:faithful",
        "water_supply/forcing_replay@none:replay",
        "n_supply/forcing_replay@none:replay",
        "crop_iface/remap_in@none:kit_fixture",
        "pet/shuttleworth_wallace@rzwqm2-4.6:faithful",
        "pet/asce_reference@asce-ewri-2005:faithful",
        "pet/priestley_taylor@dssat-4.8.6.0:faithful",
    ):
        assert key in have, key


def test_cases_name_their_slot_contract_and_own_path() -> None:
    for c in builtin_cases():
        assert c.contract_name is None or c.contract_name in SLOT_CONTRACTS, c.key
        assert c.own_path.split(".", 1)[0] in {*NAMESPACES, *RESERVED_NAMESPACES}, c.key
    # the producers of P1 trwup and P10 live outside the crop subtree (M3 contract section 11, items 1-2)
    assert (
        DEFAULT_OWN["water_supply"] == "water_supply.{slot}" and DEFAULT_OWN["n_supply"] == "n_supply.{slot}"
    )
    assert set(DEFAULT_OWN) <= set(SLOT_CONTRACTS)


def test_discover_records_the_origin_and_selects_by_glob() -> None:
    cases = discover(entry_points=False)
    assert cases and all(c.origin.startswith("agrijax") for c in cases)
    pet = select(cases, key="pet/*")
    assert {c.slot for c in pet} == {"pet"} and len(pet) == 3
    assert select(cases, key="crop/ceres_maize.growth@*") and not select(cases, key="nothing/*")
    assert select(cases, package="agrijax") == cases
    assert select(cases, package="someone-else") == []


def _plugin_case() -> ConformanceCase:
    base = next(c for c in builtin_cases() if c.key == "n_supply/forcing_replay@none:replay")
    return ConformanceCase(
        key="n_supply/plugin_replay@none:replay",
        make=base.make,
        process=base.proc,
        n_days=base.n_days,
        no_balance="plugin fixture",
        grad=None,
        no_grad="plugin fixture",
    )


class _EP(SimpleNamespace):
    def load(self):
        return self.target


def test_entry_point_cases_are_collected_with_their_distribution(monkeypatch) -> None:
    dist = SimpleNamespace(name="my-agrijax-plugin", version="0.3.1")
    eps = [
        _EP(name="mine", value="my_plugin.conformance:cases", dist=dist, target=lambda: [_plugin_case()]),
        _EP(name="agrijax", value=discover_mod._BUILTIN_VALUE, dist=None, target=builtin_cases),
    ]
    monkeypatch.setattr(
        md, "entry_points", lambda group: eps if group == discover_mod.ENTRY_POINT_GROUP else []
    )
    cases = discover()
    plugin = [c for c in cases if c.key.startswith("n_supply/plugin_replay")]
    assert len(plugin) == 1 and plugin[0].origin == "my-agrijax-plugin 0.3.1"
    assert len(cases) == len(builtin_cases()) + 1  # this repository's own entry point is not collected twice
    assert [c.key for c in select(cases, package="my-agrijax-plugin")] == [
        "n_supply/plugin_replay@none:replay"
    ]
    only = discover(builtin=False)  # a plugin testing itself: its cases and the repository's entry point
    assert "n_supply/plugin_replay@none:replay" in {c.key for c in only}
    dup = [_EP(name="dup", value="x:y", dist=dist, target=lambda: [_plugin_case(), _plugin_case()])]
    monkeypatch.setattr(md, "entry_points", lambda group: dup)
    with pytest.raises(ValueError, match="two conformance cases"):
        discover(builtin=False)


def test_a_provider_must_return_cases() -> None:
    with pytest.raises(TypeError, match="non-cases"):
        discover_mod.load_provider(lambda: [object()])
    with pytest.raises(TypeError, match="neither"):
        discover_mod.load_provider(SimpleNamespace(__file__="x.py"))


def test_the_command_line_lists_cases(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--list", "--key", "pet/*"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 3 and all(line.startswith("pet/") and "(agrijax" in line for line in out)
