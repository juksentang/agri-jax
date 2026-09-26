"""Port records of the M3 coupling contract (``agrijax.iface``): P1-P11 as code.

* the records defined before ``agrijax.iface`` existed are re-exported, not copied (the old
  import paths name the same classes);
* every port spec of :data:`agrijax.iface.contract.PORTS` agrees with its record class field by
  field: the same unit string (not only the same dimension), the same dims, the same grid;
  a spec that differs is reported;
* the contract's allowed lags are the lag table of the M3 day, and a day built from them accepts
  an implementation that uses only some of them (decision 3);
* the new records' constructors have the declared shapes; ``SinkInputs`` with only ``uptake``
  active builds exactly the M1 sink record;
* the crop package takes its records from ``agrijax.iface``, not from another slot's package.
"""

from __future__ import annotations

import ast
import dataclasses
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import agrijax
import agrijax.iface as iface
from agrijax.core import Day, Phase, process
from agrijax.core.dims import check_tree_dims
from agrijax.core.ledger import WaterLedger
from agrijax.core.units import parse_unit
from agrijax.iface.contract import (
    PORTS,
    AllowedLag,
    FieldSpec,
    PortSpec,
    allowed_lags,
    port_spec,
    record_problems,
)
from agrijax.processes.pet.daily import DailyWeather, PETFluxes
from agrijax.processes.soil_water.sinks import SinkChannels
from agrijax.processes.soil_water.uptake import CropWaterIn, RootRecord


def test_existing_records_are_re_exported_not_copied() -> None:
    assert iface.CropWaterIn is CropWaterIn and iface.RootRecord is RootRecord
    assert iface.SinkChannels is SinkChannels
    assert iface.PETFluxes is PETFluxes and iface.DailyWeather is DailyWeather
    assert iface.WaterLedger is WaterLedger
    from agrijax.processes.crop.ceres_maize.state import CeresMaizeState

    hints = {f.name: f.type for f in dataclasses.fields(CeresMaizeState)}
    assert set(hints) >= {"water_in", "root_out", "n_in"}
    with pytest.raises(AttributeError):
        _ = iface.NoSuchRecord  # type: ignore[attr-defined]
    assert set(iface.__all__) <= set(dir(iface))


def test_port_table_covers_p1_to_p11() -> None:
    assert list(PORTS) == [f"P{i}" for i in range(1, 12)]
    assert port_spec("P3") is PORTS["P3"]
    with pytest.raises(KeyError, match="P12"):
        port_spec("P12")
    assert PORTS["P1"].global_path("maize") == "iface.crop_water.maize"
    assert PORTS["P5"].global_path() == "iface.pet"
    with pytest.raises(ValueError, match="per crop slot"):
        PORTS["P2"].global_path()


@pytest.mark.parametrize("pid", list(PORTS))
def test_every_port_spec_matches_its_record(pid: str) -> None:
    spec = PORTS[pid]
    assert record_problems(spec) == []
    for _, fs in spec.fields:
        parse_unit(fs.unit)
    if spec.time in ("lag1", "mixed"):
        assert spec.lags and all(lag.evidence for lag in spec.lags)


def test_unit_string_differences_are_reported_even_with_the_same_dimension() -> None:
    """Contract section 2.3 item 3: ``eop`` is mm d-1 next to ``trwup`` in cm d-1; the check
    compares unit strings, so a producer publishing ``eop`` in cm d-1 would be caught."""
    p1 = dataclasses.replace(PORTS["P1"], time="same_day", lags=())  # field-only variants below
    wrong = dataclasses.replace(
        p1, fields=tuple((n, FieldSpec("cm d-1", fs.dims) if n == "eop" else fs) for n, fs in p1.fields)
    )
    assert parse_unit("cm d-1").same_dimension(parse_unit("mm d-1"))
    probs = record_problems(wrong)
    assert len(probs) == 1 and "P1.eop: unit 'mm d-1'" in probs[0]
    bad_dims = dataclasses.replace(p1, fields=(("sw", FieldSpec("cm3 cm-3", ("n_node",), "dssat_layers")),))
    assert "dims" in record_problems(bad_dims)[0]
    bad_grid = dataclasses.replace(p1, fields=(("sw", FieldSpec("cm3 cm-3", ("n_layer",), "rzwqm2_nodes")),))
    assert "grid" in record_problems(bad_grid)[0]
    missing = dataclasses.replace(p1, fields=(("nope", FieldSpec("-", ())),))
    assert "not a field" in record_problems(missing)[0]
    stale = dataclasses.replace(p1, pending=(("eop", FieldSpec("mm d-1", ("n_crop",)), "gap"),))
    assert "pending" in record_problems(stale)[0]


def test_pending_fields_are_the_open_gaps() -> None:
    """RTH (``srad_horizontal``) belongs to P8 but is not in ``DailyWeather`` yet (gap G3)."""
    pending = {(pid, name) for pid, spec in PORTS.items() for name, _, _ in spec.pending}
    assert pending == {("P8", "srad_horizontal")}
    assert "srad_horizontal" not in {f.name for f in dataclasses.fields(DailyWeather)}


def test_spec_validation() -> None:
    with pytest.raises(ValueError, match="needs the allowed lags"):
        PortSpec("PX", "iface.x", None, (), (), (), "lag1", "-")
    with pytest.raises(ValueError, match="unknown time"):
        PortSpec("PX", "iface.x", None, (), (), (), "tomorrow", "-")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown field"):
        PortSpec("PX", "iface.x", None, (), (), (), "lag1", "-", lags=(AllowedLag("a.b", "f", "e"),))
    with pytest.raises(ValueError):
        FieldSpec("furlongs", ())
    with pytest.raises(ValueError):
        FieldSpec("-", ("n_nope",))


# ------------------------------------------------------------------------ lags hung on the ports
M3_LAGS = {
    ("pet.sw_daily", "iface.canopy.maize"),
    ("pet.sw_daily", "soil_water.theta"),
    ("soil_water.uptake_limit", "iface.root_uptake.maize"),
    ("soil_water.uptake_limit", "iface.crop_water.maize.trwup"),
    ("crops.maize.rootwu", "iface.root.maize"),
}


def test_allowed_lags_are_the_m3_lag_table() -> None:
    lags = allowed_lags("maize")
    assert {lag.pair for lag in lags} == M3_LAGS and len(lags) == len(M3_LAGS)
    assert all(lag.evidence and lag.days == 1 for lag in lags)
    assert {lag.pair for lag in allowed_lags("maize", ("P2",))} == {
        ("crops.maize.rootwu", "iface.root.maize")
    }
    assert {lag.reader for lag in allowed_lags("soy", ("P2",))} == {"crops.soy.rootwu"}


def _toy(name: str, reads: tuple[str, ...], writes: tuple[str, ...]):
    def fn(state, params, forcing_t):
        """Source: fixture."""
        return state

    return process(fn, reads=reads, writes=writes, name=name, register=False, source="fixture")


def _m3_like_procs(*, uptake_limit_reads: tuple[str, ...], rootwu_reads: tuple[str, ...]) -> dict:
    return {
        "pet.sw_daily": _toy("pet", ("iface.canopy.maize", "soil_water.theta"), ("iface.pet",)),
        "soil_water.uptake_limit": _toy("wuf", ("iface.pet", *uptake_limit_reads), ("soil_water.sink_in",)),
        "soil_water.day": _toy("day", ("soil_water.sink_in", "soil_water.theta"), ("soil_water.theta",)),
        "crops.maize.rootwu": _toy(
            "rootwu", ("iface.crop_water.maize.sw", *rootwu_reads), ("iface.crop_water.maize.trwup",)
        ),
        "crops.maize.publish": _toy(
            "publish",
            ("crops.maize.lai",),
            ("iface.canopy.maize", "iface.root.maize", "iface.root_uptake.maize", "crops.maize.lai"),
        ),
    }


M3_LIKE_DAY = Day(
    ref="rzwqm2-4.6",
    phases=(
        Phase("physcl", ("pet.sw_daily", "soil_water.uptake_limit", "soil_water.day")),
        Phase("plant", ("crops.maize.rootwu", "crops.maize.publish")),
    ),
    lags=allowed_lags("maize"),
)


def test_a_day_on_the_contract_lags_accepts_implementations_with_fewer_lagged_reads() -> None:
    """Swapping an implementation for one that reads less (here an uptake limit that ignores
    yesterday's uptake and TRWUP, and a ROOTWU without roots) needs no change to the Day: the
    unused allowed lags are reported."""
    full = _m3_like_procs(
        uptake_limit_reads=("iface.root_uptake.maize", "iface.crop_water.maize.trwup"),
        rootwu_reads=("iface.root.maize",),
    )
    rep = M3_LIKE_DAY.check(M3_LIKE_DAY.compile(full))
    # crops.maize.rootwu and crops.maize.publish are one module for the Day (the entry name without
    # its last component), so ROOTWU's read of yesterday's root record counts as the module's own
    # carried state, not as a lag: the contract's P2 lag is reported unused (an open contract issue)
    p2 = ("crops.maize.rootwu", "iface.root.maize")
    assert set(rep.used) == M3_LAGS - {p2} and rep.unused_pairs == (p2,)
    assert p2 in M3_LIKE_DAY.carried_reads(M3_LIKE_DAY.compile(full))
    lean = _m3_like_procs(uptake_limit_reads=(), rootwu_reads=())
    rep = M3_LIKE_DAY.check(M3_LIKE_DAY.compile(lean))
    assert set(rep.used) == {("pet.sw_daily", "iface.canopy.maize"), ("pet.sw_daily", "soil_water.theta")}
    assert set(rep.unused_pairs) == M3_LAGS - set(rep.used)
    # a lag outside the contract is still rejected
    rogue = _m3_like_procs(uptake_limit_reads=("crops.maize.lai",), rootwu_reads=())
    with pytest.raises(ValueError, match=r"undeclared lags.*soil_water\.uptake_limit <- crops\.maize\.lai"):
        M3_LIKE_DAY.compile(rogue)


# ------------------------------------------------------------------------ the new records
def test_new_record_constructors_have_the_declared_dims() -> None:
    recs = {
        "canopy": iface.CanopyRecord.zeros(2),
        "n": iface.CropNIn.initial(2),
        "uptake": iface.NodeUptake.zeros(2, 7),
        "sink": iface.SinkInputs.zeros(7),
        "snow": iface.SnowOut.zeros(),
    }
    sizes = check_tree_dims(recs)
    assert sizes["n_crop"] == 2 and sizes["n_node"] == 7
    np.testing.assert_array_equal(np.asarray(recs["n"].nstres), 1.0)
    for r in (recs["canopy"], recs["uptake"], recs["sink"], recs["snow"]):
        assert all(float(jnp.max(jnp.abs(x))) == 0.0 for x in jax.tree_util.tree_leaves(r))
    f32 = iface.CanopyRecord.zeros(1, jnp.float32)
    assert f32.lai.dtype == jnp.float32


def test_sink_inputs_with_uptake_only_is_the_m1_sink_record() -> None:
    rng = np.random.default_rng(7)
    u = jnp.asarray(rng.uniform(0.0, 0.1, 9))
    si = iface.SinkInputs.zeros(9).replace(uptake=u)
    got = si.to_sink_channels()
    want = SinkChannels.from_uptake(u)
    assert jax.tree_util.tree_structure(got) == jax.tree_util.tree_structure(want)
    for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want), strict=True):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    assert got.only_daily_uptake and got.solute_channels() == ()
    both = si.to_sink_channels(("uptake", "tile"))
    assert not both.only_daily_uptake and both.solute_channels() == ("tile",)
    with pytest.raises(ValueError, match="unknown sink channels"):
        si.to_sink_channels(("drainpipe",))


def test_crop_package_imports_records_from_iface_only() -> None:
    root = Path(agrijax.__file__).parent / "processes" / "crop"
    offending = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("agrijax.processes.")
            ):
                if not node.module.startswith("agrijax.processes.crop"):
                    offending.append((path.name, node.module))
    assert offending == []


def test_importing_the_crop_records_does_not_import_pet() -> None:
    code = (
        "import sys; import agrijax.iface.crop, agrijax.iface; "
        "assert 'agrijax.processes.pet' not in sys.modules, 'pet imported'; "
        "assert 'agrijax.iface.contract' not in sys.modules; "
        "import agrijax.iface as i; i.PORTS; assert 'agrijax.processes.pet.daily' in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
