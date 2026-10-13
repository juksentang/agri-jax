"""Measure how much nitrogen limits each DSSAT v4.8.6 maize example treatment: the scope table of
``agrijax.calib.workflow.NITROGEN_STRESS``.

Every treatment of the example experiments that has the free-run input tables under the data
directory is run twice with ``dscsm048`` (build486): nitrogen on (``NITRO = Y``, the FileX as
distributed) and off (``NITRO = N``), published cultivar, one-treatment batch
(:func:`agrijax.sites.dssat_free_run.run_reference`). The measure is
``dHWAM = (HWAM_on - HWAM_off) / HWAM_off``: how far the nitrogen-off model, which ``calibrate``
runs, is from DSSAT's nitrogen-on run of the same treatment.

    python scripts/calib/nitrogen_stress_table.py [--data-dir D] [--jobs N] [--out table.json]

Also recorded per treatment: whether ``ADAT`` and ``MDAT`` are the same with nitrogen on and off. The
output marks the ones that are not ``dates differ`` in the table and lists them as the literal of
``NITROGEN_DATES_DIFFER`` (in scope with a warning, whatever the yield change).

Prints the table as a Python literal (paste it into ``NITROGEN_STRESS``, and the list of the
``dates differ`` treatments into ``NITROGEN_DATES_DIFFER``) and writes it as JSON.
Needs the DSSAT engine (``AGRI_JAX_DSSAT``); runs are staged under ``AGRI_JAX_RUN_ROOT``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agrijax.io.dssat import read_summary
from agrijax.io.dssat.filex import read_filex
from agrijax.port.run_fortran import DSSAT_ENGINE
from agrijax.sites.dssat_free_run import missing_tables, run_reference, treatment_key


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--data-dir", default=os.environ.get("AGRI_JAX_DATA", "~/agri_jax_data"))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="nitrogen_stress_table.json")
    a = ap.parse_args()
    data = Path(a.data_dir).expanduser()
    maize = DSSAT_ENGINE / "example_data" / "Maize"
    keys = []
    for fx in sorted(maize.glob("*.MZX")):
        for t in read_filex(fx)["TREATMENTS"]:
            n = int(t["N"])
            if not missing_tables(fx.stem, n, data):
                keys.append((fx, n))
    root = Path(os.environ.get("AGRI_JAX_RUN_ROOT") or tempfile.gettempdir())
    work = Path(tempfile.mkdtemp(prefix="ajn_", dir=root))

    def one(k: tuple[Path, int]) -> tuple[str, dict[str, float]]:
        fx, n = k
        key = treatment_key(fx.stem, n)
        rows = {}
        for label, on in (("on", True), ("off", False)):
            out = run_reference(fx, n, work / f"{key}_{label}", nitrogen=on)
            r = read_summary(out / "Summary.OUT").iloc[0]
            rows[label] = {c: float(r[c]) for c in ("HWAM", "CWAM", "ADAT", "MDAT")}
        on, off = rows["on"]["HWAM"], rows["off"]["HWAM"]
        d = (on - off) / off if off > 0 else float("nan")
        return key, {
            "hwam_on": on,
            "hwam_off": off,
            "d_hwam": d,
            "dates_equal": rows["on"]["ADAT"] == rows["off"]["ADAT"]
            and rows["on"]["MDAT"] == rows["off"]["MDAT"],
        }

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        table = dict(ex.map(one, keys))
    Path(a.out).write_text(json.dumps(table, indent=1))
    print("NITROGEN_STRESS = {")
    for k, v in table.items():
        print(
            f'    "{k}": {v["d_hwam"]:.4f},  # HWAM N on {v["hwam_on"]:.0f}, off {v["hwam_off"]:.0f}'
            + ("" if v["dates_equal"] else "; dates differ")
        )
    print("}")
    print("NITROGEN_DATES_DIFFER = frozenset({")
    for k, v in table.items():
        if not v["dates_equal"]:
            print(f'    "{k}",')
    print("})")


if __name__ == "__main__":
    main()
