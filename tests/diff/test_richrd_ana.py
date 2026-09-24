"""Consistency of the RICHRD dumps with the whole-run output of the same instrumented run.

RZWQM2 reports the profile water storage (``.ana`` column 2, cm) at the end of each day. The last
RICHRD call of a day leaves the profile in its end-of-day state, so the exit water contents of
that call, integrated over the numerical layers (sum of THETA * TL over the NN nodes), must give
the ``.ana`` storage of that day. This ties the dumps to an output the reference model writes
itself, independently of any port.

Cases are named ``<run>_d<date>_c<call>.npz``; the run's model outputs are in
``dumps/_runs/<run>/``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from agrijax.io.rzwqm.ana import KEY_COLUMNS, read_ana
from agrijax.port import dumps

pytestmark = pytest.mark.allow_skip(reason="dumps are private data")

ANA_SIG_DIGITS = 6
"""The ``.ana`` columns carry 6 significant digits (e.g. ``28.7217``)."""


def _print_tol(x: float) -> float:
    """Half a unit in the last printed digit of ``x`` (plus float slack)."""
    return 0.5 * 10.0 ** (np.floor(np.log10(abs(x))) - (ANA_SIG_DIGITS - 1)) * (1 + 1e-9)


def _storage(c: dumps.DumpCase) -> float:
    nn = int(c.entry["NN"])
    return float(np.sum(c.exit["THETA"][:nn] * c.exit["TL"][:nn]))


def _cases_by_run(root: Path) -> dict[str, list[dumps.DumpCase]]:
    runs: dict[str, list[dumps.DumpCase]] = defaultdict(list)
    for p in dumps.list_cases(root, "RICHRD"):
        runs[p.stem.split("_d")[0]].append(dumps.load_case(p))
    return runs


def _ana_storage(run_dir: Path) -> dict[int, float]:
    anas = sorted(run_dir.glob("*.ana"))
    if not anas:
        pytest.skip(f"no .ana in {run_dir}")
    ds = read_ana(anas[0])
    name = ds.attrs["columns"][str(KEY_COLUMNS["profile_water_cm"])]
    days = [round(float(t) * 1000) for t in ds["yyyyddd"].values]
    return dict(zip(days, ds[name].values.tolist(), strict=True))


def test_richrd_last_call_storage_matches_ana(dumps_dir: Path) -> None:
    runs = _cases_by_run(dumps_dir)
    if not runs:
        pytest.skip("no RICHRD dumps")
    checked = 0
    for run, cases in runs.items():
        storage = _ana_storage(dumps_dir / "_runs" / run)
        last = [c for c in cases if c.last_of_date]
        assert last, f"{run}: no case is flagged as the last call of its date"
        bad = []
        for c in last:
            s = _storage(c)
            ref = storage[c.date]
            if abs(s - ref) > _print_tol(ref):
                bad.append((c.date, s, ref))
        checked += len(last)
        assert not bad, (
            f"{run}: sum(theta*TL) differs from the .ana storage beyond print precision: {bad[:5]}"
        )
    assert checked >= 30


def test_richrd_storage_changes_within_a_day(dumps_dir: Path) -> None:
    """Earlier calls of a day are intermediate states: storage then differs from the day's end
    value on rainy days (the check above is therefore not satisfied trivially)."""
    runs = _cases_by_run(dumps_dir)
    if not runs:
        pytest.skip("no RICHRD dumps")
    moved = 0
    for cases in runs.values():
        by_date: dict[int, list[dumps.DumpCase]] = defaultdict(list)
        for c in cases:
            by_date[c.date].append(c)
        for cs in by_date.values():
            vals = [_storage(c) for c in cs]
            if max(vals) - min(vals) > 10 * _print_tol(max(vals)):
                moved += 1
    assert moved >= 1
