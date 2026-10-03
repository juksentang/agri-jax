"""G1: the weather sensitivity calendar on UFGA8201 (rainfed t2, irrigated t4, several weather years) and
the CA-TPA seasons, with the trust checks and the measured cost (run on a rorqual CPU node).

    python scripts/diag/g1_weather_sensitivity.py [--parts ufga,years,catpa,cost,dssat] [--out DIR]

Writes JSON / CSV under ``$AGRI_JAX_DATA/validation/aj_g1`` (default) and prints the tables. The
reference program ``dscsm048`` is only timed (``--parts dssat``), for the per-day-perturbation cost
estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={_cpus}")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)

YEARS = list(range(1978, 1988))
CATPA_CASE = Path("dssat_catpa/catpa")


def _save(out: Path, name: str, df: pd.DataFrame) -> None:
    df.to_csv(out / f"{name}.csv", index=False)


def _summ(ws, out: Path, tag: str) -> dict:
    print(f"\n==================== {tag}")
    print(ws)
    print("\n-- stages")
    print(ws.stages.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    rec = {
        "name": ws.name,
        "values": ws.values,
        "trust": ws.trust,
        "timing": ws.timing,
        "dates": {k: str(v) for k, v in ws.dates.items()},
    }
    _save(out, f"{tag}_daily", ws.daily)
    _save(out, f"{tag}_stages", ws.stages)
    if ws.checks is not None:
        print("\n-- single-day check summary")
        print(ws.checks.summary.to_string(index=False))
        print("\n-- single-day check")
        print(ws.checks.single_day.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
        print("\n-- whole season")
        print(
            ws.checks.whole_season.drop(columns=["note"]).to_string(
                index=False, float_format=lambda v: f"{v:.5g}"
            )
        )
        _save(out, f"{tag}_single_day", ws.checks.single_day)
        _save(out, f"{tag}_whole_season", ws.checks.whole_season)
        _save(out, f"{tag}_check_summary", ws.checks.summary)
        rec["check_summary"] = ws.checks.summary.to_dict("records")
        rec["whole_season"] = ws.checks.whole_season.drop(columns=["note"]).to_dict("records")
    # the most sensitive days per variable (first output)
    o = ws.outputs[0]
    d = ws.daily[ws.daily["output"] == o]
    top = {}
    for v in ws.variables:
        t = d[d["variable"] == v].reindex(columns=["date", "stage", "value", "derivative"])
        t = t.loc[t["derivative"].abs().sort_values(ascending=False).index[:8]]
        top[v] = [
            {k: (str(r[k]) if k == "date" else float(r[k])) for k in t.columns} for _, r in t.iterrows()
        ]
        tot = float(d.loc[d["variable"] == v, "derivative"].abs().sum())
        share8 = float(t["derivative"].abs().sum()) / tot if tot else 0.0
        print(f"{v}: top-8 days hold {100 * share8:.1f} % of the season's sum |d{o}/d{v}|")
        top[v + "_top8_share"] = share8
    rec["top_days"] = top
    # ste vs exact
    rel = {}
    for v in ws.variables:
        dv = d[d["variable"] == v]
        num = float(np.abs(dv["derivative"] - dv["derivative_exact"]).sum())
        den = float(np.abs(dv["derivative"]).sum())
        rel[v] = num / den if den else 0.0
    rec["ste_vs_exact_l1"] = rel
    print("ste vs exact (L1 relative):", {k: f"{v:.3g}" for k, v in rel.items()})
    return rec


def _phenofree(ws, out: Path, tag: str) -> list:
    pf = ws.phenology_free([-2.0, -1.0, 1.0, 2.0])
    print("\n-- phenology free (temperature offsets)")
    print(pf.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    _save(out, f"{tag}_phenology_free", pf)
    return pf.to_dict("records")


def _attrib(ws, out: Path, tag: str, years=None) -> dict:
    att = ws.attribution(years=years)
    print("\n-- attribution")
    print(att)
    print(att.by_stage.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    _save(out, f"{tag}_attr_total", att.total)
    _save(out, f"{tag}_attr_stage", att.by_stage)
    return {"years": att.years, "total": att.total.to_dict("records")}


def _brute(ws, out: Path, tag: str) -> dict:
    bf = ws.brute_force()
    ws.brute_force(one_device=True)  # compiles the one-device program
    bf1 = ws.brute_force(one_device=True)
    _save(out, f"{tag}_brute_force", bf)
    stats = {}
    for (o, v), g in bf.groupby(["output", "variable"]):
        big = g["derivative"].abs() > 1e-3 * g["derivative"].abs().max()
        stats[f"{o}/{v}"] = {
            "days": len(g),
            "median_rel_diff_on_nonzero": float(g.loc[big, "rel_diff"].median()) if big.any() else 0.0,
            "frac_within_5pct_nonzero": float((g.loc[big, "rel_diff"] <= 0.05).mean()) if big.any() else 1.0,
            "sum_fd": float(g["fd"].sum()),
            "sum_ad": float(g["derivative"].sum()),
        }
    print("\n-- brute force (every day one-sided at the middle step)")
    print(json.dumps(stats, indent=1))
    rec = {
        "rows": bf.attrs["rows"],
        "run_s": bf.attrs["run_s"],
        "devices": bf.attrs["devices"],
        "run_one_device_s": bf1.attrs["run_s"],
        "stats": stats,
    }
    print(
        f"brute force: {rec['rows']} runs in {rec['run_s']:.2f} s on {rec['devices']} devices; "
        f"{bf1.attrs['run_s']:.2f} s on one device; gradient {ws.timing['gradient_s']:.3f} s (one device)"
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ufga,years,catpa,cost,dssat")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    parts = set(a.parts.split(","))
    import jax

    import agrijax as aj
    from agrijax import facade_weather as fw
    from agrijax.port.run_fortran import DSSAT_ENGINE

    data = Path(os.environ.get("AGRI_JAX_DATA", "~/agri_jax_data")).expanduser()
    out = Path(a.out) if a.out else data / "validation" / "aj_g1"
    out.mkdir(parents=True, exist_ok=True)
    print(f"devices {len(jax.devices())} x {jax.devices()[0].device_kind}; out {out}")
    res: dict = {"devices": len(jax.devices())}
    exp = aj.dssat.experiment("UFGA8201", data_root=DSSAT_ENGINE)
    if "ufga" in parts or "cost" in parts:
        for trno, vs in (
            (2, ["SRAD", "TMAX", "TMIN", "RAIN"]),
            (4, ["SRAD", "TMAX", "TMIN", "RAIN", "IRRD"]),
        ):
            t0 = time.perf_counter()
            ws = exp.weather_sensitivity(trno, outputs=["HWAM", "CWAM"], variables=vs)
            tag = f"ufga_t{trno}"
            rec = _summ(ws, out, tag)
            rec["first_call_wall_s"] = time.perf_counter() - t0
            if "ufga" in parts:
                rec["phenology_free"] = _phenofree(ws, out, tag)
                rec["attribution"] = _attrib(ws, out, tag)
            if "cost" in parts:
                rec["brute_force"] = _brute(ws, out, tag)
                # one season forward on one device and on all devices (single row)
                progs = ws._ctx.progs
                z = np.zeros((1, progs.n_days, len(fw._VORDER)))
                progs.values(z, np.zeros(1, np.int64), "one")
                _, s1 = progs.values(z, np.zeros(1, np.int64), "one")
                rec["forward_one_season_one_device_s"] = s1
                _, g2 = progs.gradient([0], "ste")[1:]
                rec["gradient_repeat_s"] = g2
                print(f"one season forward on one device {s1:.3f} s; gradient pass (repeat) {g2:.3f} s")
            res[tag] = rec
            (out / "g1_results.json").write_text(json.dumps(res, indent=1, default=str))
    if "years" in parts:
        for trno in (2, 4):
            vs = ["SRAD", "TMAX", "TMIN", "RAIN"] + (["IRRD"] if trno == 4 else [])
            t0 = time.perf_counter()
            scen = exp.scenarios(trno, years=YEARS)
            many = scen.weather_sensitivity(outputs=["HWAM"], variables=vs)
            wall = time.perf_counter() - t0
            recs = []
            for w in many:
                tag = f"years_t{trno}_{w.name.split()[1]}"
                r = _summ(w, out, tag)
                if trno == 2:
                    r["attribution"] = _attrib(w, out, tag)
                recs.append(r)
            res[f"years_t{trno}"] = {"wall_s": wall, "seasons": recs}
            (out / "g1_results.json").write_text(json.dumps(res, indent=1, default=str))
    if "catpa" in parts:
        from agrijax.sites.dssat_free_run import NativeInputError, free_run_inputs

        case = data / CATPA_CASE
        runs, names = [], []
        for trno in range(1, 8):
            try:
                x = free_run_inputs(
                    case / "CTPA1501.MZX",
                    trno,
                    None,
                    engine=DSSAT_ENGINE,
                    source="native",
                    weather_dir=case,
                    soil_dirs=[case],
                    genotype_dir=case,
                    extend_days=60,
                )
            except (NativeInputError, FileNotFoundError, ValueError) as e:
                print(f"CTPA1501 t{trno}: {e}")
                continue
            runs.append(x)
            names.append(f"CTPA1501_t{trno:02d} {int(x.days[0]) // 1000}")
        t0 = time.perf_counter()
        many = fw.weather_sensitivity(
            runs,
            names,
            outputs=["HWAM", "CWAM"],
            variables=["SRAD", "TMAX", "TMIN", "RAIN", "IRRD"],
            weather_files=fw.station_files(case, "CTPA"),
        )
        recs = []
        for w in many:
            tag = f"catpa_{w.name.split()[1]}"
            r = _summ(w, out, tag)
            r["phenology_free"] = _phenofree(w, out, tag)
            r["attribution"] = _attrib(w, out, tag)
            recs.append(r)
        res["catpa"] = {"wall_s": time.perf_counter() - t0, "seasons": recs}
        (out / "g1_results.json").write_text(json.dumps(res, indent=1, default=str))
    if "dssat" in parts:
        scen = exp.scenarios(2, years=[1979, 1982, 1985])
        try:
            ref = scen.reference()
            res["dssat_season_s"] = [float(s) for s in scen.dssat_s]
            print(ref)
            print("dscsm048 per season [s]:", scen.dssat_s)
        except Exception as e:
            print("dscsm048 timing failed:", e)
        (out / "g1_results.json").write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
