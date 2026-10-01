"""Build the cached CA-TPA reference data under $AGRI_JAX_DATA/catpa (see scripts/data/README.md).

    uv run --project ~/Agri_JAX python scripts/data/make_catpa_refs.py [--skip-runs] [--extract-npy]

Steps (each run takes the shipped Scenario dir unchanged except for IPNAMES/RZX paths):
  ref_2015/          one-year RZWQM2 run (2015): LAYER.PLT, CA-TPA.ana, MANAGE.OUT, OVERVIEW.OUT, IPNAMES.DAT
  base_2015_2023/    full-period base run, same files                          (~15 s)
  lhs_run0/          full-period run with rzwqm.dat edited by row 0 of AutoAnalysis/parameter.csv,
                     exactly as GenerateDat.py writes 0_rzwqm.dat                (~15 s)
  events.csv         build_events(rzwqm.dat, 2015-01-01, 2023-12-31), checked against base MANAGE.OUT
  --extract-npy      catpa_lhs/npy/*.npy (7.9 GB) so load_catpa_lhs() can memory-map
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import tempfile
from pathlib import Path

import numpy as np

from agrijax.io.catpa import (
    CATPA_DIR,
    PARAMETER_CSV,
    SCENARIO_DIR,
    build_events,
    extract_lhs_npy,
    read_manage_out,
    write_events,
)
from agrijax.io.rzwqm.dat import read_rzwqm_dat, set_value, write_rzwqm_dat
from agrijax.port.run_fortran import RUN_ROOT, run_rzwqm

ALL_PARAMETERS = SCENARIO_DIR.parent.parent / "all_parameters.csv"
KEEP = ("*.ana", "OVERVIEW.OUT", "MANAGE.OUT", "LAYER.PLT", "IPNAMES.DAT")


def lhs_dat(row: int, out: Path) -> Path:
    """``{row}_rzwqm.dat``: GenerateDat's (line_number, location_at_line) edits with row ``row`` values."""
    addr: dict[str, tuple[int, int]] = {}
    with open(ALL_PARAMETERS, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["scenario"].strip() == "CA-TPA" and r["file"].strip() == "rzwqm.dat":
                addr[r["parameter"].strip()] = (int(r["line_number"]), int(r["location_at_line"]))
    with open(PARAMETER_CSV, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        vals = next(itertools.islice(rd, row, None), None)
    if vals is None:
        raise IndexError(row)
    dat = read_rzwqm_dat(SCENARIO_DIR / "rzwqm.dat")
    for name, v in zip(header, vals, strict=True):
        ln, tok = addr[name.strip()]
        dat = set_value(dat, ln, tok, float(v), inplace=True)
    return write_rzwqm_dat(dat, out, newline="\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-runs", action="store_true", help="only rebuild events.csv")
    ap.add_argument("--extract-npy", action="store_true", help="also extract catpa_lhs/npy/*.npy")
    a = ap.parse_args(argv)
    CATPA_DIR.mkdir(parents=True, exist_ok=True)
    root = RUN_ROOT / "catpa_refs"

    if not a.skip_runs:
        r = run_rzwqm(
            SCENARIO_DIR,
            CATPA_DIR / "ref_2015",
            start="2015-01-01",
            end="2015-12-31",
            keep_files=KEEP,
            run_root=root,
        )
        print(f"ref_2015: {r.elapsed_s:.1f} s")
        r = run_rzwqm(
            SCENARIO_DIR,
            CATPA_DIR / "base_2015_2023",
            start="2015-01-01",
            end="2023-12-31",
            keep_files=KEEP,
            run_root=root,
        )
        print(f"base_2015_2023: {r.elapsed_s:.1f} s")
        with tempfile.TemporaryDirectory(dir=RUN_ROOT) as tmp:
            dat0 = lhs_dat(0, Path(tmp) / "0_rzwqm.dat")
            (CATPA_DIR / "lhs_run0").mkdir(exist_ok=True)
            (CATPA_DIR / "lhs_run0" / "0_rzwqm.dat").write_bytes(dat0.read_bytes())
            r = run_rzwqm(
                SCENARIO_DIR,
                CATPA_DIR / "lhs_run0",
                dat_override=dat0,
                start="2015-01-01",
                end="2023-12-31",
                keep_files=KEEP,
                run_root=root,
            )
        print(f"lhs_run0: {r.elapsed_s:.1f} s")

    ev = build_events(SCENARIO_DIR / "rzwqm.dat", "2015-01-01", "2023-12-31")
    man = read_manage_out(CATPA_DIR / "base_2015_2023" / "MANAGE.OUT")
    man = man[~(man["event"].str.startswith("fertilizer_") & (man["value"] == 0.0))]
    a_keys = sorted(zip(ev["date"].astype("datetime64[ns]"), ev["event"], strict=True))
    b_keys = sorted(zip(man["date"].astype("datetime64[ns]"), man["event"], strict=True))
    if a_keys != b_keys:
        print("events.csv does not match MANAGE.OUT:", set(a_keys) ^ set(b_keys), file=sys.stderr)
        return 1
    p = write_events(ev, CATPA_DIR / "events.csv")
    print(f"{p}: {len(ev)} events, matches MANAGE.OUT ({np.unique(ev['event']).tolist()})")

    if a.extract_npy:
        print("npy ->", extract_lhs_npy())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
