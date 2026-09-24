"""CERES-Maize reproduces its frozen ``b65f1a7`` baseline (refactoring guard).

The snapshot ``<data-dir>/validation/ceres/baseline_b65f1a7.npz`` (written by
``tests/integration/ceres_baseline.py`` from the implementation validated against dscsm048 on 58
maize treatments) holds, per treatment, the parameters, the daily forcing, every leaf of the daily
CERES state, every daily output, and for two treatments the reverse-mode gradients of season
yield, maximum LAI and final above-ground biomass with respect to every cultivar and species
field. Each test reruns dscsm048 and the current model for one treatment and requires every array
to equal the snapshot within atol 1e-12, rtol 1e-12 (float64; integer and boolean leaves exactly).
A failure names the earliest differing day, the first field differing on it, and every differing
field of that treatment.

Skips (``allow_skip``) when the snapshot or the DSSAT drivers (the local dscsm048 binary and the
example data) are absent; slow (58 reference runs): run with ``--runslow`` or by file with
``-m slow``.
"""

from __future__ import annotations

import json
from pathlib import Path

import ceres_baseline as bl
import jax
import numpy as np
import pytest


def _needs_baseline(fn):
    """Slow (reruns dscsm048), float64, and allowed to skip without the snapshot / DSSAT drivers."""
    fn = pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the baseline is float64")(fn)
    fn = pytest.mark.allow_skip(reason="needs the local dscsm048 build and the private baseline snapshot")(fn)
    return pytest.mark.slow(fn)


@pytest.fixture(scope="module")
def snapshot(data_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    path = bl.snapshot_path(data_dir)
    if not path.is_file():
        pytest.skip(f"CERES baseline snapshot not found at {path}")
    ok, why = bl.drivers_available()
    if not ok:
        pytest.skip(why)
    side = path.with_suffix(".json")
    if side.is_file():  # the file is the one the sidecar describes
        assert bl.sha256(path) == json.loads(side.read_text())["sha256"], f"{path} does not match {side}"
    return bl.load_snapshot(path)


@_needs_baseline
def test_manifest(snapshot):
    arrays, man = snapshot
    assert man["git_hash"] == bl.BASELINE_COMMIT
    assert man["cases"] == [bl.case_id(e, t) for e, t in bl.CASES] and len(man["cases"]) == 58
    assert man["grad_cases"] == [bl.case_id(e, t) for e, t in bl.GRAD_CASES]
    assert bl.enumerate_cases(bl._drivers().MAIZE) == list(bl.CASES), "the maize example set changed"
    for cid in man["cases"]:
        for kind in ("params", "forcing", "state", "out"):
            names = [k.split("/", 2)[2] for k in arrays if k.startswith(f"{cid}/{kind}/")]
            assert names == man["fields"][kind], (cid, kind)
    for cid in man["grad_cases"]:
        got = sorted(k.split("/", 2)[2] for k in arrays if k.startswith(f"{cid}/grad/"))
        assert got == man["fields"]["grad"], cid
        for t in bl.GRAD_TARGETS:
            assert all(np.all(np.isfinite(v)) for k, v in arrays.items() if k.startswith(f"{cid}/grad/{t}/"))


@_needs_baseline
@pytest.mark.parametrize(("exp", "trno"), bl.CASES, ids=[bl.case_id(e, t) for e, t in bl.CASES])
def test_matches_baseline(snapshot, tmp_path, exp, trno):
    ref, _ = snapshot
    got = bl.compute_case(exp, trno, tmp_path / f"{exp}_{trno}")
    msg = bl.compare_case(bl.case_id(exp, trno), got, ref)
    assert msg is None, msg


def test_compare_names_first_field_and_day():
    """The comparison reports the earliest differing day and its field (no data needed)."""
    ref = {
        "X_t1/forcing/yrdoy": np.array([2000001, 2000002, 2000003]),
        "X_t1/state/a": np.array([1.0, 2.0, 3.0]),
        "X_t1/state/b": np.array([[1, 1], [2, 2], [3, 3]], dtype=np.int32),
        "X_t1/grad/yield/cultivar.g3": np.array(5.0),
    }
    assert bl.compare_case("X_t1", dict(ref), ref) is None
    got = dict(ref)
    got["X_t1/state/a"] = np.array([1.0, 2.0, 3.0 + 1e-6])
    got["X_t1/state/b"] = np.array([[1, 1], [2, 9], [3, 3]], dtype=np.int32)
    msg = bl.compare_case("X_t1", got, ref)
    assert msg is not None and "'X_t1/state/b'" in msg and "day 1 (YRDOY 2000002)" in msg
    assert "2 differing fields" in msg
    got = dict(ref)
    got["X_t1/state/a"] = np.array([1.0, 2.0, 3.0 * (1 + 2e-12)])
    assert bl.compare_case("X_t1", got, ref) is not None  # beyond atol + rtol |ref|
    got["X_t1/state/a"] = np.array([1.0, 2.0, 3.0 * (1 + 5e-13)])
    assert bl.compare_case("X_t1", got, ref) is None
    got = dict(ref)
    got["X_t1/grad/yield/cultivar.g3"] = np.array(5.1)
    assert "a season gradient" in (bl.compare_case("X_t1", got, ref) or "")
    got["X_t1/state/a"] = np.array([1.0, 2.5, 3.0])  # a daily difference is named before a gradient
    assert "'X_t1/state/a' on day 1" in (bl.compare_case("X_t1", got, ref) or "")
