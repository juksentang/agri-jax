"""Priestley-Taylor PET against dscsm048 (DSSAT-CSM v4.8.6.0), day by day (H1 item 2).

Every Maize example of the v4.8.6.0 engine, every treatment, is run with ``EVAPO = R`` (the
harness, conventions and tolerance derivation are in :mod:`pt_dssat`). Two levels:

1. Printed output. :func:`agrijax.processes.pet.priestley_taylor`, driven with the weather DSSAT
   reads and the previous day's printed LAI, top-layer water, DUL(1) and mulch cover, must give
   ``ET.OUT``'s ``EOAA`` on every day to within the print-precision box of those inputs plus the
   ``EOAA`` print step (0.0005 mm) and 1e-6 relative for DSSAT's REAL arithmetic.
2. PETPT dumps. A private instrumented build of the same source (one WRITE after ``CALL PETPT``,
   every REAL exactly) gives the inputs PETPT received and the EO it returned. The kernel on those
   inputs must give EO to 1e-6 relative (float32 rounding of the reference; measured <= 2.5e-7),
   on the Maize examples and on three winter-wheat examples that reach the cold branch
   (TMAX < 5 degC). The instrumented build must print the same outputs as the shipped binary, the
   dumped EO must round to ``EOAA``, and the harness inputs of level 1 must lie inside their print
   box around the dumped ones.

Measured 2026-09-25 on rorqual. Level 1: 11 of the 12 Maize experiments, 68 runs, 12 607 days
(GAGR0201's six growth-chamber treatments and EBPL8501's two fallow runs are level 2 only); per
experiment max |ours - EOAA| 0.0023-0.0044 mm, RMSE 0.00064-0.00097 mm, no day outside its bound
(largest bound half-width 0.011 mm, UFGA8201: sandy top layer, DUL(1) = 0.096). Level 2: 13 413
Maize and 5 194 wheat days (1 439 hot, 462 cold); kernel against the dumped EO at most 2.5e-7
relative on the Maize and 5.2e-7 on the wheat (cold branch, EXP) days; the instrumented and the
shipped binary print identical ET, SoilWat, PlantGro, Mulch, SoilDyn and Summary files. The
per-experiment tables are written to ``<data-dir>/validation/h1_2_pt_dssat/`` by
``python tests/integration/pt_dssat.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pt_dssat as ptd
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.allow_skip(reason="needs the DSSAT-CSM v4.8.6.0 engine (AGRI_JAX_DSSAT) and its examples"),
]

MAIZE_EXPERIMENTS = ptd.experiments()
#: every treatment of GAGR0201 is a growth chamber (FileX environment modification)
ALL_ENV_MODIFIED = {"GAGR0201"}


@pytest.fixture(scope="session")
def engine_ok() -> None:
    from agrijax.port.run_fortran import dscsm_paths

    exe, _ = dscsm_paths(ptd.DSSAT_ENGINE)
    if not exe.is_file() or not ptd.MAIZE.is_dir():
        pytest.skip(f"dscsm048 v4.8.6.0 or the Maize examples not found under {ptd.DSSAT_ENGINE}")


@pytest.fixture(scope="session")
def dump_engine(data_dir: Path) -> Path:
    eng = data_dir / ptd.DUMP_ENGINE_SUBDIR
    if not (eng / "bin" / "dscsm048").is_file():
        pytest.skip(f"private instrumented dscsm048 (PETPT dump) not found at {eng}")
    return eng


@pytest.fixture(scope="session")
def outputs(engine_ok: None, tmp_path_factory: pytest.TempPathFactory) -> Callable[[str, Path | None], Path]:
    """``outputs(name, engine)``: output directory of the (cached) run of experiment ``name``."""
    cache: dict[tuple[str, Path | None], Path] = {}

    def get(name: str, engine: Path | None = None) -> Path:
        key = (name, engine)
        if key not in cache:
            cache[key] = ptd.run_experiment(name, tmp_path_factory.mktemp(name), engine=engine)
        return cache[key]

    return get


# --------------------------------------------------------------------------- level 1: printed output


def test_every_maize_example_is_compared() -> None:
    if not ptd.MAIZE.is_dir():
        pytest.skip(f"{ptd.MAIZE} not found")
    assert len(MAIZE_EXPERIMENTS) == 12


def test_patch_sets_the_switches_and_keeps_the_rest() -> None:
    text = (
        "*SIMULATION CONTROLS\n"
        "@N METHODS     WTHER INCON LIGHT EVAPO INFIL PHOTO\n"
        " 1 ME              M     M     E     F     S     R\n"
        "@N OUTPUTS     FNAME OVVEW SUMRY FROPT GROUT CAOUT WAOUT NIOUT MIOUT DIOUT VBOSE\n"
        " 1 OU              N     Y     Y     1     Y     N     N     N     N     N     Y\n"
    )
    out, changed = ptd.patch_filex(text)
    assert changed == {"EVAPO": 1, "WAOUT": 1, "VBOSE": 1}
    lines = out.splitlines()
    assert lines[2] == " 1 ME              M     M     E     R     S     R"
    assert lines[4] == " 1 OU              N     Y     Y     1     Y     N     Y     N     N     N     D"
    assert ptd.patch_filex(out)[1] == {"EVAPO": 0, "WAOUT": 0, "VBOSE": 0}


@pytest.mark.parametrize("name", MAIZE_EXPERIMENTS)
def test_priestley_taylor_matches_et_out_within_print_precision(
    name: str, outputs: Callable[..., Path]
) -> None:
    out = outputs(name)
    runs, skipped = ptd.run_days(name, out)
    if name in ALL_ENV_MODIFIED:
        assert not runs and len(skipped.env_modified) == 6
        return
    assert runs
    for r in runs:
        # the run used Priestley-Taylor, and the station weather is the weather DSSAT simulated with
        assert (r.soil.meevp, r.soil.meinf) == ("R", "S"), (name, r.run)
        d = r.days
        assert not d[["srad", "tmax", "tmin"]].isna().any().any(), (name, r.run)
        w = np.abs(d[["srad", "tmax", "tmin"]].to_numpy() - d[["sraa", "tmaxa", "tmina"]].to_numpy())
        assert w.max() <= ptd.WEATHER_HALF, (name, r.run)
    days = pd.concat([ptd.compare(r) for r in runs], ignore_index=True)
    bad = days[days["excess"] > 0]
    assert bad.empty, bad[["run", "date", "lai", "sw1", "dul1", "cover", "eo", "eo_ref", "lo", "hi"]].head(10)
    row = ptd.summarise(name, days, len(runs))
    # guard against a vacuous bound: the print box stays narrow (measured half-width <= 0.011 mm)
    assert row["max_halfwidth_mm"] < 0.02


# --------------------------------------------------------------------------- level 2: PETPT dumps


@pytest.mark.parametrize("name", [*MAIZE_EXPERIMENTS, *ptd.WHEAT_EXPERIMENTS])
def test_kernel_reproduces_the_petpt_dump(name: str, outputs: Callable[..., Path], dump_engine: Path) -> None:
    if name in ptd.WHEAT_EXPERIMENTS and not (ptd.WHEAT / f"{name}.WHX").is_file():
        pytest.skip(f"{name}.WHX not found")
    out, out_d = outputs(name), outputs(name, dump_engine)
    # the dump WRITE does not change what DSSAT prints
    assert all(n == 0 for n in ptd.builds_differ(out, out_d).values()), ptd.builds_differ(out, out_d)
    dump = ptd.read_ptdump(out_d / "PTDUMP.OUT")
    assert len(dump) > 100
    k = ptd.kernel_on_dump(dump)
    assert k["max_rel"] <= ptd.REAL_REL, k
    # ET.OUT's EOAA is the dumped EO rounded to 3 decimals, every day
    assert ptd.eoaa_vs_dump(out, dump) <= ptd.EO_HALF + 1e-6


def test_the_dumps_reach_every_branch(outputs: Callable[..., Path], dump_engine: Path) -> None:
    """Hot (TMAX > 35), cold (TMAX < 5) and normal days all occur in the dumped runs."""
    names = [
        n
        for n in ("GHWA0401", *ptd.WHEAT_EXPERIMENTS)
        if (ptd.crop_of(n)[0] / (n + ptd.crop_of(n)[1])).is_file()
    ]
    dumps = [ptd.read_ptdump(outputs(n, dump_engine) / "PTDUMP.OUT") for n in names]
    tmax = np.concatenate([d["tmax"].to_numpy() for d in dumps])
    assert (tmax > 35).sum() > 100 and (tmax < 5).sum() > 100 and ((tmax >= 5) & (tmax <= 35)).sum() > 100


@pytest.mark.parametrize("name", [n for n in MAIZE_EXPERIMENTS if n not in ALL_ENV_MODIFIED])
def test_harness_inputs_are_the_dumped_ones_within_print_precision(
    name: str, outputs: Callable[..., Path], dump_engine: Path
) -> None:
    out, out_d = outputs(name), outputs(name, dump_engine)
    runs, _ = ptd.run_days(name, out)
    days = pd.concat([ptd.compare(r) for r in runs], ignore_index=True)
    dump = ptd.read_ptdump(out_d / "PTDUMP.OUT")
    m = days.merge(dump, on=["run", "date"], suffixes=("", "_dump"), validate="one_to_one")
    assert len(m) == len(days)
    f32 = np.float32
    for c in ("srad", "tmax", "tmin"):
        np.testing.assert_array_equal(m[c].to_numpy().astype(f32), m[f"{c}_dump"].to_numpy().astype(f32))
    assert np.abs(m["lai"] - m["xhlai"]).max() <= ptd.LAI_HALF + 1e-6
    # the dumped albedo lies in the albedo range of the print box of SW1, DUL(1) and the mulch cover
    sa = np.asarray([next(r.soil.salb for r in runs if r.run == k) for k in m["run"]])
    corners = [
        ptd.dssat_msalb(sa, m["sw1"] + ds, m["dul1"] + du, np.clip(m["cover"] + dc, 0.0, 1.0))
        for ds in (-ptd.SW_HALF, ptd.SW_HALF)
        for du in (-ptd.DUL_HALF, ptd.DUL_HALF)
        for dc in (-ptd.COVER_HALF, ptd.COVER_HALF)
    ]
    lo, hi = np.min(corners, axis=0), np.max(corners, axis=0)
    alb = m["et_alb"].to_numpy()
    assert ((alb >= lo - 1e-6) & (alb <= hi + 1e-6)).all(), name
