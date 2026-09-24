"""Regression test for the throughput skeleton's numerical guards (poc/README.md, "NaN gradients").

Runs ``poc/bench_skeleton.py --check 1`` on the exact configuration that produced NaN gradients before
the fix (n=1000 samples, 60 days, 12 sub-steps x 2 Newton, x64, tridiagonal solve): 17/1000 samples then
had a non-finite gradient because the Richards iterate ran away to overflow. ``--check`` asserts finite
forward outputs, a final pressure head inside [H_MIN, H_MAX] for every sample and a finite gradient for
every parameter of every sample; the script exits non-zero otherwise. No data needed (~15-28 s on CPU),
but it lives in the integration tier (slow-marked) so the unit tier stays under its 2-minute budget.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

SKELETON = Path(__file__).resolve().parents[2] / "poc" / "bench_skeleton.py"


@pytest.mark.slow
def test_skeleton_check_passes_on_previously_failing_batch() -> None:
    cmd = [
        sys.executable,
        str(SKELETON),
        "--days",
        "60",
        "--sub",
        "12",
        "--newton",
        "2",
        "--n",
        "1000",
        "--x64",
        "1",
        "--solver",
        "tridiag",
        "--grad",
        "1",
        "--check",
        "1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    assert "finite=True" in res.stdout and "check: OK" in res.stdout, res.stdout
    assert "nan=False" in res.stdout, res.stdout


def _summary(solver: str) -> dict[str, float]:
    cmd = [sys.executable, str(SKELETON), "--days", "60", "--sub", "12", "--newton", "3", "--n", "64"]
    cmd += ["--x64", "1", "--solver", solver, "--check", "1"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    line = next(ln for ln in res.stdout.splitlines() if ln.startswith("n="))
    m = re.search(r"mean_aet=([-\d.e+]+) mean_lai=([-\d.e+]+)\s+h_final=\[([-\d.e+]+),([-\d.e+]+)\]", line)
    assert m is not None, line
    return dict(zip(("aet", "lai", "hmin", "hmax"), (float(g) for g in m.groups()), strict=True))


@pytest.mark.slow
def test_skeleton_tridiagonal_solve_equals_dense_lu() -> None:
    """Second independent method for the Newton linear solve: ``lax.linalg.tridiagonal_solve``
    (Thomas / gtsv) vs ``jnp.linalg.solve`` (dense LU) of the same Jacobian give the same run.

    This covers the linear algebra of the skeleton only. The skeleton is a computational stand-in,
    not a model, so there is no reference solution for its physics (see the validation matrix).
    """
    tri, dense = _summary("tridiag"), _summary("dense")
    for k in tri:
        assert tri[k] == pytest.approx(dense[k], rel=1e-4, abs=1e-6), (k, tri, dense)
