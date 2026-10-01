"""Checks every dump case against the Fortran index and against invariants of the reference model.

These are the prerequisites of the per-routine differential tests: the case files must hold every
variable the index says the routine reads or writes, with consistent shapes and types, and the
values must obey facts that do not depend on any port (inputs the routine never assigns are
unchanged at exit; the date hook agrees with the routine's own date argument; bounded quantities
stay in their bounds).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agrijax.port import dumps


def _entry(index_entries: dict[str, dict[str, Any]], routine: str) -> dict[str, Any]:
    if routine not in index_entries:
        pytest.skip(f"{routine} is not in the Fortran index under the data dir")
    return index_entries[routine]


def _has(name: str, where: dict[str, np.ndarray]) -> bool:
    return name in where or any(k.startswith(name + "%") for k in where)


def test_case_holds_every_indexed_variable(dump_case: Path, index_entries: dict[str, Any]) -> None:
    case = dumps.load_case(dump_case)
    e = _entry(index_entries, case.routine)
    names = [str(a).upper() for a in e["args"] if a != "*"]
    names += [str(s["name"]).upper() for s in e["saved_vars"]]
    used = {str(v).upper() for v in [*e["read_vars"], *e["assigned_vars"]]}
    for blk in e["common_blocks"]:
        names += [str(v).upper() for v in blk["vars"] if str(v).upper() in used]
    missing = [n for n in names if not _has(n, case.entry) or not _has(n, case.exit)]
    assert not missing, f"{case.routine}: not dumped: {missing}"
    if e["kind"] == "function":
        assert case.routine in case.exit and case.routine not in case.entry


def test_entry_and_exit_agree_in_shape_and_type(dump_case: Path) -> None:
    case = dumps.load_case(dump_case)
    assert case.call >= 1 and case.day_call >= 1
    for k, v in case.entry.items():
        w = case.exit[k]
        assert w.shape == v.shape and w.dtype == v.dtype, k
    assert set(case.exit) - set(case.entry) <= {case.routine}


def test_never_assigned_inputs_are_unchanged(dump_case: Path, index_entries: dict[str, Any]) -> None:
    """Arguments the index classifies as ``in`` (never assigned, also not through calls)."""
    case = dumps.load_case(dump_case)
    e = _entry(index_entries, case.routine)
    intents = e["intent_guess"]
    assert isinstance(intents, dict)
    changed = []
    for a, it in intents.items():
        if it != "in":
            continue
        a = a.upper()
        for k in [x for x in case.entry if x == a or x.startswith(a + "%")]:
            v, w = case.entry[k], case.exit[k]
            if v.dtype.kind == "f":
                same = np.array_equal(v, w, equal_nan=True)
            else:
                same = np.array_equal(v, w)
            if not same:
                changed.append(k)
    assert not changed, f"{case.routine} changed inputs: {changed}"


DATE_ARGS = {"MZ_PHENOL": "YRDOY", "MZ_GROSUB": "YRDOY", "MZ_ROOTGR": "YRDOY"}


def test_date_hook_matches_date_argument(dump_case: Path) -> None:
    case = dumps.load_case(dump_case)
    assert 1_900_001 <= case.date <= 2_100_366
    arg = DATE_ARGS.get(case.routine)
    if arg is not None:
        assert int(case.entry[arg]) == case.date
    if case.routine == "POTEVPHR":  # the hook is IYYY*1000+JDAY in PHYSCL, the caller
        assert int(case.entry["IYYY"]) * 1000 + int(case.entry["JDAY"]) == case.date


@pytest.mark.routines("RICHRD")
def test_richrd_water_contents_are_physical(dump_case: Path) -> None:
    case = dumps.load_case(dump_case)
    nn = int(case.entry["NN"])
    assert 1 <= nn <= case.exit["THETA"].shape[0]
    th = case.exit["THETA"][:nn]
    assert np.all(np.isfinite(th)) and np.all(th > 0.0) and np.all(th < 1.0)
    # saturation content is row 6 of the Brooks-Corey parameter table, per horizon
    ws = case.entry["SOILHP"][5, case.entry["NDXN2H"][:nn] - 1]
    assert np.all(th <= ws + 1e-9)
    assert np.all(case.entry["TL"][:nn] > 0.0)


@pytest.mark.routines("MZ_PHENOL", "MZ_GROSUB", "MZ_ROOTGR")
def test_ceres_soil_layers_consistent(dump_case: Path) -> None:
    case = dumps.load_case(dump_case)
    nl = int(case.entry["NLAYR"])
    assert 1 <= nl <= case.entry["DLAYR"].shape[0]
    assert np.all(case.entry["DLAYR"][:nl] > 0.0)
    if "SW" in case.entry and "LL" in case.entry:
        assert np.all(case.entry["SW"][:nl] >= 0.0)
