"""The planned part of the coupling contract, ``agrijax.iface.contract`` (modules not written yet).

* the planned ports and rows change nothing in the RZWQM2-order day: :func:`day_entries`,
  :func:`allowed_lags` and :func:`day_table` give its contract by default, the skeleton day is unchanged;
* every planned port's record matches its spec (unit strings, dims, grid), says what runs before
  its module exists (``off``), and its constructors give the declared shapes;
* the merged day puts each planned row after its anchor, drops the rows it supersedes, and stays
  in phase order;
* the mechanical checks AJ012-AJ019 find nothing in the tables, and each one fails on a known-bad
  fixture;
* the markdown renderer lists every port and row, deterministically;
* the planned slot contracts name contract ports only.
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
import subprocess
import sys

import pytest

import agrijax.iface.contract as c
from agrijax.core import Day, Phase
from agrijax.core.day import DayError, PhasedWrite
from agrijax.core.dims import check_tree_dims
from agrijax.iface.contract import (
    DAY_TABLE,
    EXTENDED_PHASES,
    PHASED_WRITES,
    PORTS,
    POST_M3_DAY_TABLE,
    STAGES,
    AllowedLag,
    DayEntry,
    FieldSpec,
    PortSpec,
    allowed_lags,
    contract_problems,
    day_entries,
    day_table,
    path_consumers,
    phased_writes,
    record_problems,
    registry_slot_problems,
    variant_problems,
)
from agrijax.iface.render import render, render_day, render_ports

PLANNED = [p for p, s in PORTS.items() if s.stage == "post_m3"]


# ------------------------------------------------------------------------ the RZWQM2-order day is untouched
def test_the_m3_day_is_the_default_and_unchanged() -> None:
    assert day_table() == DAY_TABLE
    phases = day_entries("maize")
    assert [e for _, es in phases for e in es] == [e.name("maize") for e in DAY_TABLE]
    assert {lag.pair for lag in allowed_lags("maize")} == {
        ("pet.sw_daily", "iface.canopy.maize"),
        ("pet.sw_daily", "soil_water.theta"),
        ("soil_water.uptake_limit", "iface.root_uptake.maize"),
        ("soil_water.uptake_limit", "iface.crop_water.maize.trwup"),
        ("water_supply.maize.rootwu", "iface.root.maize"),
    }
    from agrijax.models.day_rzwqm46 import day_rzwqm46

    day = day_rzwqm46()
    assert day.entries == tuple(e.name("maize") for e in DAY_TABLE)


def test_the_extended_day_builds_a_valid_day() -> None:
    phases = day_entries("maize", STAGES)
    assert tuple(ph for ph, _ in phases) == EXTENDED_PHASES
    day = Day(
        ref="rzwqm2-4.6",
        bare=True,  # a fixture of part of the day
        phases=tuple(Phase(p, e) for p, e in phases),
        lags=allowed_lags("maize", stages=STAGES),
    )
    flat = day.entries
    assert "n_supply.maize.replay" not in flat  # superseded by the soil nitrogen path
    assert flat.index("soil_water.day") < flat.index("soil_heat.heatfx") < flat.index("soil_om.day")
    assert flat.index("drainage.control") < flat.index("drainage.tile") < flat.index("soil_water.day")
    assert flat.index("crops.maize.remap_in") + 1 == flat.index("crops.maize.n_remap_in")
    assert flat.index("crops.maize.publish_uptake") < flat.index("crops.maize.residue_out")
    assert flat[-2:] == ("phosphorus.day", "ledger.close")
    assert len(flat) == len(DAY_TABLE) + len(POST_M3_DAY_TABLE) - 1


# ------------------------------------------------------------------------ the planned ports
@pytest.mark.parametrize("pid", PLANNED)
def test_planned_port_records_match_and_say_what_runs_before_them(pid: str) -> None:
    spec = PORTS[pid]
    assert record_problems(spec) == []
    assert spec.off.strip()
    assert spec.record is not None and spec.record.__module__.startswith("agrijax.iface.")


def test_a_planned_port_without_off_is_rejected() -> None:
    with pytest.raises(ValueError, match="off="):
        PortSpec("PX", "iface.x", None, (), ("a.b",), (), "same_day", "-", stage="post_m3")
    with pytest.raises(ValueError, match="unknown stage"):
        PortSpec("PX", "iface.x", None, (), (), (), "same_day", "-", stage="m4")  # type: ignore[arg-type]


def test_planned_rows_need_an_anchor() -> None:
    with pytest.raises(ValueError, match="after"):
        DayEntry("x", "chem", "soil_om.day", "", (), "none", stage="post_m3")
    with pytest.raises(ValueError, match="after"):
        DayEntry("x", "plant", "crops.x.y", "", (), "none", after="crops.x.z")
    with pytest.raises(ValueError, match="unknown phase"):
        DayEntry("x", "chem", "soil_om.day", "", (), "none")  # chem is a planned phase only


def test_record_shapes() -> None:
    import agrijax.iface as iface

    sizes = check_tree_dims(
        {
            "trace": iface.WaterStepTrace.zeros(3, 6),
            "n": iface.NodeNUptake.zeros(2, 6),
            "r": iface.CropResidueOut.zeros(2, 4),
        }
    )
    assert sizes == {**sizes, "n_step": 3, "n_node": 6, "n_crop": 2, "n_layer": 4}


# ------------------------------------------------------------------------ mechanical checks
def test_the_contract_tables_have_no_problem() -> None:
    assert contract_problems() == []


def _first(text: str) -> str:
    return text.split(" ", 1)[0]


def _with_port(monkeypatch: pytest.MonkeyPatch, pid: str, **changes: object) -> None:
    monkeypatch.setitem(c.PORTS, pid, dataclasses.replace(PORTS[pid], **changes))


def test_aj012_a_port_without_a_producer_row(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_port(monkeypatch, "P19", producers=("soil_om.nowhere",))
    assert any(p.startswith("AJ012 P19") and "not a row" in p for p in contract_problems())


def test_aj013_two_writers_of_one_field(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_port(
        monkeypatch, "P4", writers=(("soil_water.uptake_limit", ("uptake",)), ("drainage.tile", ("uptake",)))
    )
    assert any(p.startswith("AJ013 P4") and "2 entries" in p for p in contract_problems())
    _with_port(monkeypatch, "P4", writers=(("soil_water.uptake_limit", ("uptake", "nope")),))
    assert any(p.startswith("AJ013 P4") and "nope" in p for p in contract_problems())


def test_aj013_counts_writer_modules() -> None:
    """Fan-in is per module: ROOTWU writes P1 trwup in its uptake entry and in its
    own season end, one module; a second module writing it is a second writer."""
    ws = PORTS["P1"].field_writers()["trwup"]
    assert ws == ("water_supply.{slot}.rootwu", "water_supply.{slot}.season_end")
    assert not any(p.startswith("AJ013 P1") for p in contract_problems())


def test_aj013_a_second_module_on_a_field_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    w = (*PORTS["P1"].writers, ("crops.{slot}.harvest", ("trwup",)))
    _with_port(monkeypatch, "P1", writers=w)
    assert any(p.startswith("AJ013 P1") and "2 modules" in p for p in contract_problems())


def test_aj013_a_cross_module_reset_is_a_second_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    @dataclasses.dataclass(frozen=True)
    class RowWithReset(DayEntry):
        resets: tuple[tuple[str, str], ...] = ()

    base = DAY_TABLE[-1]
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(DayEntry)}
    bad = RowWithReset(**fields, resets=(("water_supply.{slot}", "harvest"),))
    monkeypatch.setattr(c, "DAY_TABLE", (*DAY_TABLE[:-1], bad))
    assert any(p.startswith("AJ013 row") and "resets" in p for p in contract_problems())


# ------------------------------------------------------------------------ AJ013 execution phases
def _without_declaration(entry: str, path: str) -> tuple[PhasedWrite, ...]:
    kept = tuple(d for d in PHASED_WRITES if (d.entry, d.path) != (entry, path))
    assert len(kept) == len(PHASED_WRITES) - 1
    return kept


def test_aj013_the_extra_writes_of_p1_p2_p6_are_declared_season_ends() -> None:
    """One owning module per field; the owner's other writes carry an
    execution phase and a meaning. P1 trwup (ROOTWU's uptake + its season end) and P2, P6 (the
    crop's publish, canopy + its harvest) are declared season_end writes after every consumer."""
    decl = {(d.entry, d.path): d for d in PHASED_WRITES}
    for key in (
        ("water_supply.{slot}.season_end", "iface.crop_water.{slot}.trwup"),
        ("crops.{slot}.harvest", "iface.root.{slot}"),
        ("crops.{slot}.harvest", "iface.canopy.{slot}"),
    ):
        assert decl[key].when == "season_end" and decl[key].meaning.strip(), key
    assert all(d.is_reset for d in PHASED_WRITES)
    rows = [r.entry for r in DAY_TABLE]
    last_consumer = max(rows.index(_first(cn)) for cn in PORTS["P1"].consumers)
    assert rows.index("water_supply.{slot}.season_end") > last_consumer
    assert not any(p.startswith("AJ013") for p in contract_problems())
    got = phased_writes("maize")
    assert ("water_supply.maize.season_end", "iface.crop_water.maize.trwup") in {
        (d.entry, d.path) for d in got
    }


def test_aj013_an_undeclared_extra_write_of_the_owner_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        c,
        "PHASED_WRITES",
        _without_declaration("water_supply.{slot}.season_end", "iface.crop_water.{slot}.trwup"),
    )
    probs = contract_problems()
    assert any(p.startswith("AJ013 P1 trwup") and "undeclared" in p for p in probs), probs


def test_aj013_an_undeclared_harvest_write_of_p2_or_p6_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    for pid, path in (("P2", "iface.root.{slot}"), ("P6", "iface.canopy.{slot}")):
        monkeypatch.setattr(c, "PHASED_WRITES", _without_declaration("crops.{slot}.harvest", path))
        assert any(p.startswith(f"AJ013 {pid}") and "undeclared" in p for p in contract_problems()), pid


def test_aj013_a_season_end_before_a_consumer_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """ROOTWU's season end moved to right after its uptake entry (before the crop reads P1)."""
    rows = list(DAY_TABLE)
    end = next(r for r in rows if r.entry == "water_supply.{slot}.season_end")
    rows.remove(end)
    at = next(i for i, r in enumerate(rows) if r.entry == "water_supply.{slot}.rootwu")
    rows.insert(at + 1, end)
    monkeypatch.setattr(c, "DAY_TABLE", tuple(rows))
    probs = contract_problems()
    assert any(
        p.startswith("AJ013 P1 trwup") and "before the consumer" in p and "crops.{slot}.stress" in p
        for p in probs
    ), probs
    # and before its producer
    rows.remove(end)
    rows.insert(at, end)
    monkeypatch.setattr(c, "DAY_TABLE", tuple(rows))
    assert any(
        p.startswith("AJ013 P1 trwup") and "the first writer is the undeclared producer" in p
        for p in contract_problems()
    )


def test_aj013_orders_a_reset_of_module_state_after_its_readers(monkeypatch: pytest.MonkeyPatch) -> None:
    """ROOTWU's rwu is not a port field, but the layer -> node publish reads it the same day
    (SHARED_STATE): the season end moved before the publish fails."""
    rows = list(DAY_TABLE)
    end = next(r for r in rows if r.entry == "water_supply.{slot}.season_end")
    rows.remove(end)
    rows.insert(next(i for i, r in enumerate(rows) if r.entry == "crops.{slot}.publish_uptake"), end)
    monkeypatch.setattr(c, "DAY_TABLE", tuple(rows))
    probs = contract_problems()
    assert any("water_supply.{slot}.rwu" in p and "crops.{slot}.publish_uptake" in p for p in probs), probs
    assert not any(p.startswith("AJ013 P1") for p in probs)  # trwup's consumers all ran before


def test_aj013_a_daily_extra_write_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    daily = tuple(
        dataclasses.replace(d, when="daily") if d.path.endswith("trwup") else d for d in PHASED_WRITES
    )
    monkeypatch.setattr(c, "PHASED_WRITES", daily)
    assert any(p.startswith("AJ013 P1 trwup") and "daily" in p for p in contract_problems())


def test_aj013_every_multi_module_field_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cross-module field does not hide the phase check of the port's other fields."""
    w = (*PORTS["P1"].writers, ("water_supply.{slot}.season_end", ("sw",)))
    _with_port(monkeypatch, "P1", writers=w)
    monkeypatch.setattr(
        c,
        "PHASED_WRITES",
        _without_declaration("water_supply.{slot}.season_end", "iface.crop_water.{slot}.trwup"),
    )
    probs = contract_problems()
    assert any(p.startswith("AJ013 P1 sw") and "2 modules" in p for p in probs), probs
    assert any(p.startswith("AJ013 P1 trwup") and "undeclared" in p for p in probs), probs


def test_p1_consumers_name_the_fields_they_read() -> None:
    spec = PORTS["P1"]
    assert spec.field_consumers("trwup") == ("crops.{slot}.stress", "soil_water.uptake_limit")
    assert spec.field_consumers("sw") == (
        "crops.{slot}.phenology",
        "crops.{slot}.stress",
        "crops.{slot}.roots",
    )
    assert path_consumers("iface.crop_water.{slot}.trwup") == spec.field_consumers("trwup")
    assert path_consumers("water_supply.{slot}.rwu") == ("crops.{slot}.publish_uptake",)
    assert path_consumers("iface.canopy.{slot}") == ("pet.sw_daily",)


def test_aj013_declarations_are_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = (
        # the harvest does not write P1 trwup (not in the port's writers)
        PhasedWrite("crops.{slot}.harvest", "iface.crop_water.{slot}.trwup", "season_end", "fixture"),
        # a crop entry declaring ROOTWU's state: a cross-module reset
        PhasedWrite("crops.{slot}.harvest", "water_supply.{slot}.tss", "season_end", "fixture"),
        PhasedWrite("crops.{slot}.nowhere", "crops.{slot}.x", "reset", "fixture"),
    )
    monkeypatch.setattr(c, "PHASED_WRITES", (*PHASED_WRITES, *bad))
    probs = contract_problems()
    assert any("P1 does not list crops.{slot}.harvest" in p for p in probs), probs
    assert any("water_supply.{slot}.tss" in p and "own writes only" in p for p in probs), probs
    assert any("crops.{slot}.nowhere" in p and "not a row" in p for p in probs), probs
    monkeypatch.setattr(c, "PHASED_WRITES", (*PHASED_WRITES, PHASED_WRITES[0]))
    assert any("declared twice" in p for p in contract_problems())


def test_phased_write_declarations_are_validated() -> None:
    with pytest.raises(DayError, match="execution phase"):
        PhasedWrite("a.b", "a.x", "sometimes", "fixture")
    with pytest.raises(DayError, match="meaning"):
        PhasedWrite("a.b", "a.x", "reset", " ")


def test_aj014_oversized_records(monkeypatch: pytest.MonkeyPatch) -> None:
    many = tuple((f"f{i}", FieldSpec("-", ())) for i in range(c.MAX_PORT_FIELDS + 1))
    _with_port(monkeypatch, "P24", fields=many, record=None)
    assert any(p.startswith("AJ014 P24") for p in contract_problems())


def test_aj015_a_read_before_the_writer_needs_an_allowed_lag(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_port(monkeypatch, "P17", lags=(), time="same_day")
    assert any(p.startswith("AJ015 P17") and "pet.sw_daily" in p for p in contract_problems())
    _with_port(
        monkeypatch,
        "P14",
        time="mixed",
        lags=(AllowedLag("phosphorus.day", "", "fixture: a lag on a reader after the writer"),),
    )
    assert any(p.startswith("AJ015 P14") and "hidden lag" in p for p in contract_problems())


def test_aj016_an_m3_row_cannot_write_a_planned_port(monkeypatch: pytest.MonkeyPatch) -> None:
    k = next(i for i, e in enumerate(DAY_TABLE) if e.entry == "soil_water.day")
    bad = dataclasses.replace(DAY_TABLE[k], ports_out=("P7", "P12"))
    monkeypatch.setattr(c, "DAY_TABLE", (*DAY_TABLE[:k], bad, *DAY_TABLE[k + 1 :]))
    assert any(p.startswith("AJ016 row 7") for p in contract_problems())
    monkeypatch.setattr(c, "DAY_TABLE", DAY_TABLE)
    _with_port(monkeypatch, "P24", path="erosionx.sediment")
    assert any(p.startswith("AJ016 P24") for p in contract_problems())


def _registry_entries() -> list[tuple[str, str]]:
    """Every key the package registers, with its defining module: the process packages and the
    assembled models are all imported (the registry is global, so the result must not depend on
    which modules other tests imported first); keys registered by test fixtures are left out."""
    import agrijax.models as models
    import agrijax.processes as procs
    from agrijax.core.process import list_processes

    for pkg in (procs, models):
        for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            importlib.import_module(m.name)
    return [
        (str(p.info.key), p.fn.__module__)
        for p in list_processes()
        if p.info is not None and p.fn.__module__.startswith("agrijax.")
    ]


#: registry keys defined outside their slot's directory, each with the move that closes it (AJ018);
#: the test fails when an entry is closed and not removed, or a new mismatch appears
KNOWN_SLOT_MISMATCHES: dict[str, str] = {
    "water_supply/forcing_replay@none:replay": "move from processes/crop/ceres_maize/model.py to processes/water_supply/",
    "crop_iface/eop_from_pet@rzwqm2-4.6:faithful": "move from processes/pet/eop.py to processes/crop_iface/",
    "crop_iface/publish_uptake@rzwqm2-4.6:faithful": (
        "move from processes/water_supply/publish.py to processes/crop_iface/"
    ),
    "soil_water/wuf@rzwqm2-4.6:faithful": "rename the key to water_supply/ (it lives in processes/water_supply/)",
    # the demo assemblies register their own entries in models/
    "pet/shuttleworth_wallace@rzwqm2-4.6:prescribed_canopy": "defined in models/catpa_pet_demo.py",
    "diagnostic/catpa_pet_totals@none:demo": "defined in models/catpa_pet_demo.py",
    "crop/tobacco_demo.calendar@none:demo": "defined in models/tobacco_demo.py",
    "crop/tobacco_demo.leaves@none:demo": "defined in models/tobacco_demo.py",
    "crop/tobacco_demo.management@none:demo": "defined in models/tobacco_demo.py",
}
#: variant labels outside the vocabulary (AJ019), each with its new name (c.LEGACY_VARIANTS)
KNOWN_VARIANT_LABELS: frozenset[str] = frozenset(
    {
        "crop/ceres_maize.growth@dssat-4.8.6.0:nstress_replay",
        "soil_water/day@rzwqm2-4.6:replay_flux",
        "pet/shuttleworth_wallace@rzwqm2-4.6:prescribed_canopy",
        # the soil-water day's RZWQM2 convention variants (DRAIN cap, flux-mode evaporation limit)
        "soil_water/day@rzwqm2-4.6:drain_cap",
        "soil_water/day@rzwqm2-4.6:flux_evap",
        "soil_water/day@rzwqm2-4.6:rzwqm2_conventions",
        "soil_water/day@rzwqm2-4.6:replay_flux_conventions",
    }
)


def test_aj018_registry_slot_is_the_directory() -> None:
    found = {p.split(" ", 2)[1].rstrip(":") for p in registry_slot_problems(_registry_entries())}
    assert found == set(KNOWN_SLOT_MISMATCHES)
    assert registry_slot_problems([("pet/x@none:replay", "my_pkg.processes.pet.x")]) == []
    assert registry_slot_problems([("pet/x@none:replay", "agrijax.models.day_rzwqm46")])


def test_aj019_variant_vocabulary() -> None:
    bad = {k for k, _ in _registry_entries() if variant_problems(k)}
    assert bad == set(KNOWN_VARIANT_LABELS)
    assert variant_problems("pet/x@rzwqm2-4.6:ref_istress0") == []
    assert variant_problems("n_supply/forcing_replay@none:replay") == []
    assert variant_problems("n_supply/forcing_replay@rzwqm2-4.6:replay")  # a replay has no reference
    assert variant_problems("crop/x@none:port_crop_n")  # a port variant follows a reference
    assert "port_crop_n" in variant_problems("crop/x@dssat-4.8.6.0:nstress_replay")[0]


# ------------------------------------------------------------------------ rendering
def test_render_lists_every_port_and_row_deterministically() -> None:
    ports = render_ports()
    for pid in PORTS:
        assert f"| {pid} |" in ports
    day = render_day("maize")
    for e in day_table(STAGES):
        assert f"`{e.name('maize')}`" in day
    assert "`n_supply.maize.replay`" not in day
    assert "`n_supply.maize.replay`" in render_day("maize", ("m3",))
    assert render() == render() and render("ports", stages=("m3",)).count("| P1") == 3  # P1, P10, P11
    out = subprocess.run(
        [sys.executable, "-m", "agrijax.iface.render", "--what", "ports", "--stage", "m3"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "| P11 |" in out and "| P12 |" not in out


# ------------------------------------------------------------------------ planned slot contracts
def test_planned_slot_contracts_name_contract_ports() -> None:
    from agrijax.testing.conformance.contracts import (
        POST_M3_SLOT_CONTRACTS,
        POST_M3_SLOT_PORTS,
        slot_contract,
    )

    for name, sc in POST_M3_SLOT_CONTRACTS.items():
        assert sc.phase in EXTENDED_PHASES, name
        assert all(sp.port in PORTS for sp in sc.ports), name
    for name, extra in POST_M3_SLOT_PORTS.items():
        merged = {sp.port for sp in slot_contract(name).ports}
        assert {sp.port for sp in extra} <= merged
    # every planned state port has a slot that may write it
    writable = {
        sp.port
        for sc in (*POST_M3_SLOT_CONTRACTS.values(), *(slot_contract(n) for n in POST_M3_SLOT_PORTS))
        for sp in sc.ports
        if sp.direction != "in"
    }
    assert {p for p in PLANNED if PORTS[p].kind == "state"} <= writable


def test_iface_init_lists_every_name_once_in_its_three_places() -> None:
    """``agrijax.iface.__init__`` names each export three times (``TYPE_CHECKING`` imports for the
    type checker, ``__all__``, the lazy table ``_WHERE``); they must agree, and each name must be in
    the ``__all__`` of the submodule it is loaded from, so adding a record cannot half-register it."""
    import ast
    from pathlib import Path

    import agrijax.iface as iface

    tree = ast.parse(Path(iface.__file__).read_text(encoding="utf-8"))
    typed = {
        a.asname or a.name
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
        for imp in node.body
        if isinstance(imp, ast.ImportFrom)
        for a in imp.names
    }
    where = iface._WHERE
    assert typed == set(iface.__all__) == set(where)
    for name, mod in where.items():
        m = (
            importlib.import_module(mod, "agrijax.iface")
            if mod.startswith(".")
            else importlib.import_module(mod)
        )
        assert name in getattr(m, "__all__", ()), (name, mod)
