"""Bit-identity of the free-run DSSAT day between two source trees (measurement only).

``dump --out DIR``: the free configuration of ``tests/integration/day_dssat486_free_harness.py`` on the
58 DSSAT maize example treatments and the 7 CA-TPA seasons (dscsm048 reference runs for the inputs,
float64), every daily output of :func:`agrijax.models.day_dssat486.day_outputs` saved per run as
``DIR/<run>.npz``. Run it once per tree (e.g. ``main`` and a branch), then ``compare --a DIR1 --b DIR2``
reports, per run and output, whether the arrays are equal bit for bit (NaN == NaN). ``compare`` fails
(exit 1) unless both directories hold the same, non-empty set of runs, ``--expect`` of them (default
:data:`N_RUNS`: 58 + 7; ``--expect 0`` skips the count check).

Only the harness and the faithful day are imported, so the script runs unchanged on any tree that has
them. Runs on rorqual (dscsm048, the dump tables of the harness).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

#: runs of one dump: the 58 DSSAT maize example treatments and the 7 CA-TPA seasons
N_RUNS = 65

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "tests" / "integration"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def dump(a: argparse.Namespace) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import day_dssat486_free_harness as h

    data = Path(os.environ["AGRI_JAX_DATA"])
    work = Path(os.environ.get("AGRI_JAX_RUN_ROOT", "/tmp")) / "bitid"
    keys = h.a12_keys(data) + h.catpa_keys()
    refs = h.run_references(keys, work, data, a.jobs)
    runs = [h.build(e, t, refs[h.key_of(e, t)], data) for e, t in keys]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for grp in sorted({(r.nl, r.mesev) for r in runs}):
        rs = [r for r in runs if (r.nl, r.mesev) == grp]
        o = h.run_group(rs, "free")
        for i, r in enumerate(rs):
            np.savez(out / f"{r.key}.npz", **{k: np.asarray(v[i]) for k, v in o.items()})
    print("wrote", len(runs), "runs to", out, flush=True)


def compare(a: argparse.Namespace) -> None:
    da, db = Path(a.a), Path(a.b)
    names_a = {p.name for p in da.glob("*.npz")}
    names_b = {p.name for p in db.glob("*.npz")}
    problems = []
    if not names_a:
        problems.append(f"no runs in {da}")
    if names_a != names_b:
        problems.append(
            f"different run sets: only in --a {sorted(names_a - names_b)}, "
            f"only in --b {sorted(names_b - names_a)}"
        )
    if a.expect and len(names_a) != a.expect:
        problems.append(f"{len(names_a)} runs in --a, expected {a.expect}")
    if problems:
        raise SystemExit("compare: " + "; ".join(problems))
    names = sorted(names_a)
    report: dict[str, list[str]] = {}
    n_arrays = 0
    for n in names:
        x, y = np.load(da / n), np.load(db / n)
        bad = []
        for k in sorted(set(x.files) | set(y.files)):
            if k not in x.files or k not in y.files:
                bad.append(f"{k}: missing on one side")
                continue
            n_arrays += 1
            u, v = x[k], y[k]
            if (
                u.shape != v.shape
                or u.dtype != v.dtype
                or not np.array_equal(u, v, equal_nan=u.dtype.kind == "f")
            ):
                bad.append(k)
        if bad:
            report[n] = bad
    res = {"runs": len(names), "arrays": n_arrays, "different": report, "bit_identical": not report}
    print(json.dumps(res, indent=1), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    if report:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("dump", "compare"))
    ap.add_argument("--out", default="")
    ap.add_argument("--a", default="")
    ap.add_argument("--b", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--expect", type=int, default=N_RUNS, help="compare: required run count (0: any)")
    a = ap.parse_args()
    {"dump": dump, "compare": compare}[a.step](a)


if __name__ == "__main__":
    main()
