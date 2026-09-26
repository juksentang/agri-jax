"""Shuttleworth-Wallace kernel against RZWQM2 4.6 ``POTEVPHR`` dumps: 9 scenarios x 3 years (H1 item 1).

The instrumented RZWQM2 4.6 build dumped every ``POTEVPHR`` call (entry and exit) of the 9 daily
S-W scenarios of ``RZWQM_sw_batch`` for the first 3 calendar years of each (9,862 days); the
reduction to daily tables and the input mapping are in :mod:`pet_dumps`. The kernel is driven
with the dumped entry values, so the comparison has no reconstruction in it.

Tolerances (all measured 2026-09-25, float64):

* kernel vs dump, with MAXSW's own ``PI = 3.141592654`` (:func:`pet_dumps.reference_pi_coefficients`):
  float64 rounding. The two codes evaluate the same expressions in a different order and with
  different ``exp``/``log``/``pow`` implementations; :data:`REL_ROUNDING` = 1e-12 relative plus
  :data:`ABS_ROUNDING` = 1e-15 cm d-1 absolute covers that.
* kernel vs dump with the default coefficients: the port's ``maxsw.hours_per_radian`` is
  ``12/math.pi``; MAXSW's PI is rounded up at the tenth digit, so the port's extraterrestrial and
  clear-sky radiation is 1.3057e-10 larger in relative terms (measured on every day,
  :func:`test_clear_sky_radiation`).
  That enters the fluxes through the net long-wave cloudiness ratio and is amplified by at most
  the ratio of the flux to its radiation-driven part; :data:`REL_PI` = 1e-8 bounds it (measured
  max 9.4e-10).
* dump vs ``.ana`` cols 8 and 9 of the same run: the ``.ana`` prints these in cm with 6
  decimals, so half a unit of the last digit, 5e-7 cm.
* the ``pet_scenarios`` reconstruction on the same run: its per-day error is the sum of the
  kernel error (above) and the effect of the reconstructed inputs, which :func:`pet_dumps.attribution`
  measures group by group; the harness on the run's own ``.ana`` must equal the kernel on the
  reconstructed inputs against the dump to the ``.ana`` print precision (5e-6 mm d-1).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import numpy as np
import pet_dumps as pdm
import pet_scenarios as ps
import pytest

pytestmark = pytest.mark.allow_skip(reason="needs the RZWQM2 POTEVPHR dump tables of RZWQM_sw_batch")

REL_ROUNDING = 1.0e-12
ABS_ROUNDING = 1.0e-15  # cm d-1
REL_PI = 1.0e-8
#: relative excess of the port's radiation over MAXSW's (whose PI is too large): 3.141592654 / pi - 1
PI_EXCESS = pdm.REFERENCE_PI / np.pi - 1.0
ANA_HALF_DIGIT_CM = 5.0e-7
ANA_HALF_DIGIT_MM = 5.0e-6
FLUXES = ("pt", "pes", "per")


@pytest.fixture(scope="module")
def dumped(data_dir: Path) -> dict[str, pdm.DumpDays]:
    out = {}
    for site in pdm.SITES:
        pe, px = pdm.table_paths(data_dir, site)
        if not (pe.is_file() and px.is_file()):
            pytest.skip(f"POTEVPHR dump tables of {site} not found under {pe.parent}")
        out[site] = pdm.load_site(data_dir, site)
    return out


@pytest.fixture(scope="module")
def kernel(dumped: dict[str, pdm.DumpDays]) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Per site: reference outputs, kernel with the reference PI, kernel with default coefficients."""
    cf = pdm.reference_pi_coefficients()
    res = {}
    for site, d in dumped.items():
        x = pdm.kernel_inputs(d)
        res[site] = {
            "x": x,
            "ref": pdm.reference_outputs(d),
            "pi": pdm.run_kernel(x, coefficients=cf),
            "default": pdm.run_kernel(x),
            "rts_only": pdm.run_kernel(x, override={"srad_h": x["srad"]}, coefficients=cf),
        }
    return res


@pytest.mark.parametrize("site", pdm.SITES)
def test_tables_cover_three_years_one_call_per_day(dumped: dict[str, pdm.DumpDays], site: str) -> None:
    d = dumped[site]
    days = pdm.dates_of(d)
    start = _dt.date.fromisoformat(d.meta["start"])
    assert days[0].date() == start and days[-1].date() == _dt.date(start.year + ps.N_YEARS - 1, 12, 31)
    assert len(days) == (days[-1] - days[0]).days + 1  # every day
    assert (d.n_calls == 1).all()  # the daily path calls POTEVPHR once per day


@pytest.mark.parametrize("site", pdm.SITES)
def test_inputs_outside_the_kernel_are_inactive(dumped: dict[str, pdm.DumpDays], site: str) -> None:
    """Stubble, standing dead residue, slope, hourly/SHAW/PENFLUX and the RTSTOT = 0 path are off."""
    assert pdm.inactive_inputs(dumped[site]) == dict.fromkeys(pdm.inactive_inputs(dumped[site]), 0.0)


@pytest.mark.parametrize("site", pdm.SITES)
def test_dump_is_the_ana_output(dumped: dict[str, pdm.DumpDays], data_dir: Path, site: str) -> None:
    """The dumped PES + PER and PET are the ``.ana`` cols 8 and 9 of the same run (6 decimals, cm)."""
    import pandas as pd

    from agrijax.io.rzwqm import read_ana

    run = data_dir / pdm.RUNS / site
    if not run.is_dir():
        pytest.skip(f"{run} not found")
    d = dumped[site]
    ref = pdm.reference_outputs(d)
    ds = read_ana(next(p for p in run.iterdir() if p.suffix.lower() == ".ana"))
    col = {int(k): v for k, v in ds.attrs["columns"].items()}
    idx = pd.DatetimeIndex(ds.time.values).get_indexer(pdm.dates_of(d))
    assert (idx >= 0).all()
    pe = ds[col[8]].values.astype(float)[idx]
    pt = ds[col[9]].values.astype(float)[idx]
    assert np.abs(ref["pes"] + ref["per"] - pe).max() <= ANA_HALF_DIGIT_CM + 1e-12
    assert np.abs(ref["pt"] - pt).max() <= ANA_HALF_DIGIT_CM + 1e-12


@pytest.mark.parametrize("site", pdm.SITES)
def test_clear_sky_radiation(kernel: dict[str, Any], site: str) -> None:
    """``RCS`` (flat site: the clear-sky total, floored at RTS): exact with MAXSW's PI, +1.3e-10 without."""
    k = kernel[site]
    x, rcs = k["x"], k["ref"]["rcs"]
    pi = pdm.clear_sky(x, pdm.reference_pi_coefficients())
    np.testing.assert_allclose(pi, rcs, rtol=REL_ROUNDING, atol=0.0)
    ratio = pdm.clear_sky(x) / rcs - 1.0
    floored = rcs == x["srad"]  # days on which RTS exceeds the clear-sky value
    np.testing.assert_allclose(ratio[~floored], PI_EXCESS, rtol=1e-3, atol=0.0)


@pytest.mark.parametrize("site", pdm.SITES)
def test_intermediates_match(kernel: dict[str, Any], site: str) -> None:
    """Soil and residue albedo, exposed soil fraction and adjusted wind run: rounding level."""
    k = kernel[site]
    for v in ("as", "ar", "cs", "unew"):
        np.testing.assert_allclose(k["pi"][v], k["ref"][v], rtol=REL_ROUNDING, atol=ABS_ROUNDING, err_msg=v)


@pytest.mark.parametrize("site", pdm.SITES)
def test_kernel_matches_potevphr(kernel: dict[str, Any], site: str) -> None:
    """PT, soil and residue evaporation on every day, from the dumped inputs: float64 rounding."""
    k = kernel[site]
    for v in FLUXES:
        np.testing.assert_allclose(k["pi"][v], k["ref"][v], rtol=REL_ROUNDING, atol=ABS_ROUNDING, err_msg=v)


@pytest.mark.parametrize("site", pdm.SITES)
def test_kernel_default_coefficients(kernel: dict[str, Any], site: str) -> None:
    """With ``12/math.pi`` the only difference left is MAXSW's truncated PI (relative <= 1e-8)."""
    k = kernel[site]
    for v in FLUXES:
        np.testing.assert_allclose(k["default"][v], k["ref"][v], rtol=REL_PI, atol=ABS_ROUNDING, err_msg=v)


def test_both_radiations_are_needed(kernel: dict[str, Any]) -> None:
    """Using the field radiation RTS also for the cloudiness ratio (instead of RTH) is far off.

    The two differ by up to 0.11 MJ m-2 d-1; the effect is largest on days whose net radiation
    is close to the ``RN < 0`` replacement of NETRAD (a jump), e.g. CA-TPA 2016-05-09 and US-Tw2
    2008-10-17, and around 1e-3..1e-2 mm d-1 otherwise.
    """
    worst = 0.0
    for site in pdm.SITES:
        k = kernel[site]
        for v in FLUXES:
            worst = max(worst, float(np.abs(k["rts_only"][v] - k["ref"][v]).max()) * 10.0)
    assert worst > 1.0e-1, worst  # measured 0.30 mm d-1 (US-Tw2 2008-10-17, PT)


@pytest.fixture(scope="module")
def attributions(dumped: dict[str, pdm.DumpDays], data_dir: Path) -> dict[str, tuple[Any, dict[str, Any]]]:
    from agrijax.io.rzwqm import read_rzwqm_dat
    from agrijax.io.rzwqm.layers import simulation_start

    out = {}
    for site, d in dumped.items():
        run = data_dir / pdm.RUNS / site
        src = data_dir / pdm.RUNS / "_stage" / site / "Scenario"
        if not src.is_dir():
            src = data_dir / ps.BATCH / site / "Scenario"
        if not (run / "LAYER.PLT").is_file():
            pytest.skip(f"{run} not found")
        st = simulation_start(src / "IPNAMES.DAT").astype(object)
        rec = dict(site=site, ok=True, start=st, end=_dt.date(st.year + 2, 12, 31), source=src, out=run)
        x = ps.site_inputs(rec, data_dir)
        x.site_consts["params"] = ps.pet_params(
            read_rzwqm_dat(data_dir / ps.BATCH / site / "Scenario" / "rzwqm.dat")
        )
        out[site] = (x, pdm.attribution(d, x))
    return out


@pytest.mark.parametrize("site", pdm.SITES)
def test_harness_error_is_the_reconstruction(
    attributions: dict[str, Any], dumped: dict[str, pdm.DumpDays], site: str
) -> None:
    """On the dumped run, the ``.ana`` harness error equals the reconstruction error against the dump.

    ``harness - .ana`` = ``(kernel(reconstructed) - dump) + (dump - .ana)``; the second term is the
    ``.ana`` print precision, so the harness residual is entirely due to the reconstructed inputs.
    """
    x, a = attributions[site]
    assert np.array_equal(x.days, pdm.dates_of(dumped[site]))  # same calendar days
    sim = ps.simulate(x, "ana")
    ok = a["ok"]
    np.testing.assert_allclose(
        (sim["pe"] - x.ref["pe"])[ok], a["pe_err"][ok], rtol=0.0, atol=ANA_HALF_DIGIT_MM
    )
    np.testing.assert_allclose(
        (sim["pt"] - x.ref["pt"])[ok], a["pt_err"][ok], rtol=0.0, atol=ANA_HALF_DIGIT_MM
    )


@pytest.mark.parametrize("site", pdm.SITES)
def test_reconstruction_attribution(attributions: dict[str, Any], site: str) -> None:
    """Report of the reconstruction error by input group (printed; see the module docstring)."""
    _, a = attributions[site]
    rows = {k: v for k, v in a.items() if isinstance(v, dict)}
    print(site, {k: (f"{v['pe_max_mm']:.2e}", f"{v['pt_max_mm']:.2e}") for k, v in rows.items()})
    assert rows["dumped inputs"]["pe_max_mm"] < 1e-8 and rows["dumped inputs"]["pt_max_mm"] < 1e-8


def test_write_summary(kernel: dict[str, Any], attributions: dict[str, Any], data_dir: Path) -> None:
    """Measured maxima per site to ``<data>/validation/pet_dumps/pet_dumps_summary.json``."""
    import json

    out: dict[str, Any] = {}
    for site in pdm.SITES:
        k = kernel[site]
        row: dict[str, Any] = {"n_days": len(k["x"]["doy"])}
        for case in ("pi", "default", "rts_only"):
            for v in FLUXES:
                e = np.abs(k[case][v] - k["ref"][v])
                row[f"{case}_{v}_max_abs_cm"] = float(e.max())
                row[f"{case}_{v}_max_rel"] = float((e / np.maximum(np.abs(k["ref"][v]), 1e-300)).max())
        _, a = attributions[site]
        row["attribution_mm"] = {key: v for key, v in a.items() if isinstance(v, dict)}
        out[site] = row
    p = data_dir / "validation" / "pet_dumps" / "pet_dumps_summary.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    assert p.stat().st_size > 0
