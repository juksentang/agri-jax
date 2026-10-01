"""The PET *process* on the coupling-contract ports against RZWQM2 4.6 dumps: 9 scenarios x 3 years.

:mod:`test_pet_dumps` validated the Shuttleworth-Wallace **kernel** on the ``POTEVPHR``
entry/exit dumps of the 9 daily S-W scenarios of ``RZWQM_sw_batch``, 3 years each (9,862 days).
This module runs the **registered process** ``pet/shuttleworth_wallace@rzwqm2-4.6:faithful`` on
the same days through the path an assembled day uses:

* a :class:`~agrijax.core.day.Day` in the contract's order with the contract's allowed lags
  (:func:`agrijax.iface.contract.allowed_lags`), compiled and checked by ``Day.compile``;
* the PET entry is the process bound with :func:`agrijax.core.ports.bind` to its own subtree
  ``surface.pet`` and the ports P6 ``iface.canopy.maize`` (canopy), P7 ``soil_water.theta``
  (node water) and P5 ``iface.pet``; the day's weather is a :class:`DailyWeather` with
  ``srad = RTS`` and ``srad_horizontal = RTH`` (the dumped values);
* the producers of P6 and P7 run **after** PET, as in the reference (the crop and the soil-water
  time loop follow PET in PHYSCL): replay entries write, at the end of day ``d``, the record the
  reference's ``POTEVPHR`` received on the morning of day ``d + 1``. PET therefore sees
  yesterday's records through the one-day lags of the day, never a same-day value;
* the ``EOP`` entry (``crop_iface/eop_from_pet``) reads P5 the same day and writes P1 ``eop``;
* the run is one ``jit`` of ``vmap`` over the 9 sites of ``lax.scan`` over the days. Parameters
  that the reference changes during a run (the plant's ``RST`` at a crop change, ``WC13`` / ``WC15``
  after management, the residue mass, age and type) are sliced per day by this harness, as an
  assembled day would with date-indexed parameter tables.

Tolerances are the measured bounds of the kernel test (:mod:`test_pet_dumps`): with MAXSW's own
``PI = 3.141592654`` float64 rounding (1e-12 relative plus 1e-15 cm d-1), with the default
coefficients 1e-8 relative (the truncated PI).

Negative control: the same day with the replays writing the morning record of day ``d`` instead
of ``d + 1`` (PET one day behind the reference) fails those bounds, so the comparison detects a
wrong lag.

Also here (same data): the evaporation demand the soil-water day takes from P5
(:func:`agrijax.iface.surface.evaporation_demand`, ``PES + PER``) bounds the reference's actual
evaporation (``.ana`` column 6) on every day of the 9 runs; and ``EOP = 10 PET`` against the
reference's ``DSSATDRV`` exit ``EOP`` of CA-TPA 2015-2023 (dump tables).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pet_dumps as pdm
import pytest
import test_pet_dumps as tpd

from agrijax.core.day import Day, Phase
from agrijax.core.ports import bind, compose
from agrijax.core.state import get_path
from agrijax.iface.contract import allowed_lags
from agrijax.iface.crop import CanopyRecord, CropWaterIn
from agrijax.iface.surface import EVAPORATION_DEMAND_FIELDS, DailyWeather, PETFluxes, evaporation_demand
from agrijax.models.day_rzwqm46 import DEFAULT_EVAPORATION_DEMAND, replay_entry
from agrijax.processes.pet import (
    PET_COEFFICIENTS,
    PETParams,
    PETSiteParams,
    PETState,
    SurfaceResidue,
    pet_shuttleworth_wallace,
)
from agrijax.processes.pet.eop import EOPState, eop_from_pet

pytestmark = [
    pytest.mark.allow_skip(reason="needs the RZWQM2 POTEVPHR dump tables of RZWQM_sw_batch"),
    pytest.mark.skipif(not jax.config.read("jax_enable_x64"), reason="the dump comparison runs in float64"),
]

SLOT = "maize"
CANOPY, THETA, PET, CROP_WATER = (
    f"iface.canopy.{SLOT}",
    "soil_water.theta",
    "iface.pet",
    f"iface.crop_water.{SLOT}",
)
#: the part of the contract's day this module exercises (entry names of DAY_TABLE)
DAY = Day(
    ref="rzwqm2-4.6",
    bare=True,  # part of the contract's day with stand-ins
    phases=(
        Phase("physcl", ("pet.sw_daily", "soil_water.day")),
        Phase("plant", (f"crops.{SLOT}.eop", f"crops.{SLOT}.canopy")),
    ),
    lags=allowed_lags(SLOT, ("P6", "P7")),  # the lags of the ports this part of the day reads
)
#: the per-day inputs of the harness (kernel_inputs names; RM and IPR raw from the dump)
PER_DAY = (
    "tmin tmax srad srad_h rh wind_run doy elevation latitude wc13 wc15 wind_height trat albedo_dry "
    "albedo_wet albedo_crop albedo_residue rss rst rm resage wres ipr"
).split()
#: rainfall zone and CRES of every scenario (static parameters; test_static_inputs_are_one_program)
STATIC_ZONE, STATIC_CRES = 3, 2.5
A12 = Path("dumps/tables/rzwqm46_catpa2015_2023")
OUT = Path("validation/aj_w3")


def day_processes() -> dict[str, Any]:
    return {
        "pet.sw_daily": bind(
            pet_shuttleworth_wallace,
            own="surface.pet",
            ports={"canopy": CANOPY, "theta": THETA, "pet": PET},
            params="pet",
            forcing="weather",
            name="pet.sw_daily",
        ),
        "soil_water.day": replay_entry("soil_water.day", {THETA: "replay.theta"}),
        f"crops.{SLOT}.eop": bind(
            eop_from_pet,
            own="surface.crop_iface",
            ports={"pet": PET, "crop_water": CROP_WATER},
            name=f"crops.{SLOT}.eop",
        ),
        f"crops.{SLOT}.canopy": replay_entry(f"crops.{SLOT}.canopy", {CANOPY: "replay.canopy"}),
    }


def _scalar(a: Any) -> np.ndarray:
    return pdm._scalar(np.asarray(a)).astype(float)


def site_days(d: pdm.DumpDays) -> dict[str, np.ndarray]:
    """Per-day inputs of one site: the dumped ``POTEVPHR`` entry values."""
    x = pdm.kernel_inputs(d)
    out = {k: np.asarray(x[k], dtype=float) for k in PER_DAY if k in x}
    out["rm"] = _scalar(d.entry["RM"])  # raw: the process applies the IPR = 0 test itself
    out["ipr"] = _scalar(d.entry["IPR"])
    out["resage"], out["wres"] = x["residue_age"], x["residue_wet"]
    for k in ("lai", "tlai", "height", "theta"):
        out[k] = np.asarray(x[k], dtype=float)
    return out


def _stack(sites: dict[str, pdm.DumpDays]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-day inputs of all sites, padded to the longest run (last day repeated) and the mask."""
    rows = {s: site_days(d) for s, d in sites.items()}
    n = max(len(r["doy"]) for r in rows.values())
    mask = np.zeros((len(rows), n), dtype=bool)
    out: dict[str, list[np.ndarray]] = {}
    for i, r in enumerate(rows.values()):
        m = len(r["doy"])
        mask[i, :m] = True
        for k, v in r.items():
            out.setdefault(k, []).append(np.concatenate([v, np.repeat(v[-1:], n - m)]))
    return {k: np.stack(v) for k, v in out.items()}, mask


def _shift(a: np.ndarray, lag: bool) -> np.ndarray:
    """The record the producer writes at the end of day ``d``: the morning value of ``d + 1``
    (``lag``; the last day repeats), or of ``d`` itself (the negative control)."""
    return np.concatenate([a[:, 1:], a[:, -1:]], axis=1) if lag else a


def run(x: dict[str, np.ndarray], coefficients: Any, *, lag: bool = True) -> dict[str, np.ndarray]:
    """The day on every site and day: P5 fluxes [cm d-1] and P1 ``eop`` [mm d-1], ``[site, day]``."""
    model = DAY.compile(day_processes())
    step = model.compile()

    def site_params(v: dict[str, Any]) -> PETSiteParams:
        return PETSiteParams(
            pet=PETParams(
                albedo_dry=v["albedo_dry"],
                albedo_wet=v["albedo_wet"],
                albedo_maturity=v["albedo_crop"],
                albedo_residue=v["albedo_residue"],
                soil_resistance=v["rss"],
                stomatal_resistance=v["rst"],
            ),
            elevation=v["elevation"],
            latitude=v["latitude"],
            wc13=v["wc13"],
            wc15=v["wc15"],
            wind_height=v["wind_height"],
            albedo_soil=jnp.zeros_like(v["rss"]),  # Priestley-Taylor only
            trat=v["trat"],
            rainfall_zone=STATIC_ZONE,
            residue_cover_factor=STATIC_CRES,
            coefficients=coefficients,
            residue=SurfaceResidue(mass=v["rm"], age=v["resage"], wet=v["wres"], kind=v["ipr"]),
        )

    def canopy(v: dict[str, Any], sfx: str = "") -> CanopyRecord:
        return CanopyRecord(
            lai=v["lai" + sfx][None], tlai=v["tlai" + sfx][None], height=v["height" + sfx][None]
        )

    def one_site(xs: dict[str, Any]) -> Any:
        first = jax.tree_util.tree_map(lambda a: a[0], xs)
        s0 = compose(
            {
                "surface.pet": PETState(),
                "surface.crop_iface": EOPState(),
                CANOPY: canopy(first),
                THETA: first["theta"][None],
                PET: PETFluxes.zeros(),
                CROP_WATER: CropWaterIn.zeros(1, 1),
            }
        )

        def body(s: Any, v: dict[str, Any]) -> tuple[Any, Any]:
            forcing = {
                "weather": DailyWeather(
                    tmin=v["tmin"],
                    tmax=v["tmax"],
                    srad=v["srad"],
                    rh=v["rh"],
                    wind_run=v["wind_run"],
                    doy=v["doy"],
                    srad_horizontal=v["srad_h"],
                ),
                "replay": {"canopy": canopy(v, "_next"), "theta": v["theta_next"][None]},
            }
            s, _ = step(s, {"pet": site_params(v)}, forcing)
            p = get_path(s, PET)
            return s, (
                p.transpiration,
                p.soil_evaporation,
                p.residue_evaporation,
                get_path(s, CROP_WATER).eop[0],
            )

        _, ys = jax.lax.scan(body, s0, xs)
        return ys

    xs = dict(x)
    for k in ("lai", "tlai", "height", "theta"):
        xs[k + "_next"] = _shift(x[k], lag)
    pt, pes, per, eop = jax.jit(jax.vmap(one_site))({k: jnp.asarray(v) for k, v in xs.items()})
    return {"pt": np.asarray(pt), "pes": np.asarray(pes), "per": np.asarray(per), "eop": np.asarray(eop)}


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
def inputs(dumped: dict[str, pdm.DumpDays]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    return _stack(dumped)


@pytest.fixture(scope="module")
def reference(dumped: dict[str, pdm.DumpDays], inputs: Any) -> dict[str, np.ndarray]:
    _, mask = inputs
    n = mask.shape[1]
    out: dict[str, np.ndarray] = {}
    for i, d in enumerate(dumped.values()):
        r = pdm.reference_outputs(d)
        for v in tpd.FLUXES:
            a = out.setdefault(v, np.zeros(mask.shape))
            a[i, : len(r[v])] = r[v]
            a[i, len(r[v]) :] = r[v][-1]
    assert n == max(len(d.date) for d in dumped.values())
    return out


@pytest.fixture(scope="module")
def runs(inputs: Any) -> dict[str, dict[str, np.ndarray]]:
    x, _ = inputs
    return {
        "pi": run(x, PET_COEFFICIENTS.replace(sw=pdm.reference_pi_coefficients())),
        "default": run(x, None),
        "no_lag": run(x, PET_COEFFICIENTS.replace(sw=pdm.reference_pi_coefficients()), lag=False),
    }


def test_day_uses_exactly_the_two_contract_lags() -> None:
    """The compiled day passes Day.check; PET's reads of P6 and P7 are its only lagged reads."""
    model = DAY.compile(day_processes())
    assert sorted(DAY.lagged_reads(model)) == [("pet.sw_daily", CANOPY), ("pet.sw_daily", THETA)]
    used = DAY.lag_report(model).used
    assert set(used) == {("pet.sw_daily", CANOPY), ("pet.sw_daily", THETA)}


def test_static_inputs_are_one_program(dumped: dict[str, pdm.DumpDays]) -> None:
    """Rainfall zone and CRES are the same on every day of every site (static parameters of the
    one vmapped program); crust and roughness are 0 (fixed to 0 in the process)."""
    for d in dumped.values():
        x = pdm.kernel_inputs(d)
        assert set(np.unique(x["rain_zone"])) == {float(STATIC_ZONE)}
        assert set(np.unique(x["cres"])) == {STATIC_CRES}
        assert not np.any(x["crust"]) and not np.any(x["roughness"])


def _check(out: dict[str, np.ndarray], ref: dict[str, np.ndarray], mask: np.ndarray, rtol: float) -> None:
    for i, site in enumerate(pdm.SITES):
        m = mask[i]
        for v in tpd.FLUXES:
            np.testing.assert_allclose(
                out[v][i, m], ref[v][i, m], rtol=rtol, atol=tpd.ABS_ROUNDING, err_msg=f"{site} {v}"
            )


def test_process_matches_potevphr(runs: Any, reference: Any, inputs: Any) -> None:
    """The process on the ports, with MAXSW's PI: float64 rounding on every day of the 9 x 3 years."""
    _check(runs["pi"], reference, inputs[1], tpd.REL_ROUNDING)


def test_process_default_coefficients(runs: Any, reference: Any, inputs: Any) -> None:
    """With the default ``12/math.pi``: the kernel-test bound of the truncated reference PI (1e-8)."""
    _check(runs["default"], reference, inputs[1], tpd.REL_PI)


def test_process_equals_the_kernel_path(runs: Any, dumped: dict[str, pdm.DumpDays], inputs: Any) -> None:
    """The bound process in the scanned day against the kernel harness on the same inputs."""
    cf = pdm.reference_pi_coefficients()
    worst = 0.0
    for i, d in enumerate(dumped.values()):
        k = pdm.run_kernel(pdm.kernel_inputs(d), coefficients=cf)
        n = len(d.date)
        for v in tpd.FLUXES:
            a, b = runs["pi"][v][i, :n], k[v]
            worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))))
            np.testing.assert_allclose(a, b, rtol=tpd.REL_ROUNDING, atol=tpd.ABS_ROUNDING)
    print("W3A process_vs_kernel_max_rel", worst)


def test_eop_is_ten_pet(runs: Any, inputs: Any) -> None:
    """P1 ``eop`` [mm d-1] written the same day from P5 ``transpiration`` [cm d-1]."""
    out = runs["pi"]
    np.testing.assert_array_equal(out["eop"][inputs[1]], (out["pt"] * 10.0)[inputs[1]])


def test_wrong_lag_is_detected(runs: Any, reference: Any, inputs: Any) -> None:
    """Negative control: PET one day behind the reference's canopy and surface water fails the bound."""
    mask = inputs[1]
    worst = 0.0
    for v in tpd.FLUXES:
        e = np.abs(runs["no_lag"][v] - reference[v])[mask]
        worst = max(worst, float(e.max()))
    assert worst > 1e6 * tpd.ABS_ROUNDING + tpd.REL_ROUNDING, worst
    print("W3A no_lag_max_abs_cm", worst)


def _ana_evaporation(data_dir: Path, site: str, d: pdm.DumpDays) -> np.ndarray:
    from agrijax.io.rzwqm import read_ana

    run_dir = data_dir / pdm.RUNS / site
    if not run_dir.is_dir():
        pytest.skip(f"{run_dir} not found")
    ds = read_ana(next(p for p in run_dir.iterdir() if p.suffix.lower() == ".ana"))
    col = {int(k): v for k, v in ds.attrs["columns"].items()}
    import pandas as pd

    idx = pd.DatetimeIndex(ds.time.values).get_indexer(pdm.dates_of(d))
    assert (idx >= 0).all()
    return ds[col[6]].values.astype(float)[idx]


def test_evaporation_demand_bounds_the_reference(dumped: dict[str, pdm.DumpDays], data_dir: Path) -> None:
    """The soil-water day's demand ``PES + PER`` against the reference's actual
    evaporation (``.ana`` column 6, cm, 6 decimals): never exceeded (print precision), and exceeded
    by the soil evaporation alone on some days (so the residue evaporation belongs to the demand)."""
    assert EVAPORATION_DEMAND_FIELDS == DEFAULT_EVAPORATION_DEMAND
    rows: dict[str, Any] = {}
    for site, d in dumped.items():
        ref = pdm.reference_outputs(d)
        z = np.zeros_like(ref["pes"])
        demand = np.asarray(evaporation_demand(PETFluxes(ref["pt"], ref["pes"], ref["per"], z, z, z)))
        e = _ana_evaporation(data_dir, site, d)
        assert np.all(e <= demand + tpd.ANA_HALF_DIGIT_CM), site
        rows[site] = {
            "days": len(e),
            "days_E_eq_demand": int(np.sum(np.abs(e - demand) <= tpd.ANA_HALF_DIGIT_CM)),
            "days_E_gt_PES": int(np.sum(e > ref["pes"] + tpd.ANA_HALF_DIGIT_CM)),
            "max_deficit_cm": float(np.max(demand - e)),
        }
    print("W3A evaporation_demand " + json.dumps(rows))
    assert sum(r["days_E_gt_PES"] for r in rows.values()) > 0


def test_eop_against_the_dssatdrv_dumps(data_dir: Path) -> None:
    """``eop_from_pet`` on the PHYSCL exit ``PET`` (REAL*8) against the DSSATDRV exit ``EOP``
    (REAL*4) of CA-TPA 2015-2023 on every crop day. The reference forms ``EOP`` in single
    precision from ``PET`` rounded to REAL*4 (reproduced bit for bit here); ours, in float64, is
    within those two roundings (measured: 822 of 1088 days equal after rounding to float32, the
    rest one float32 ulp apart)."""
    from agrijax.port import dumps

    tab = data_dir / A12
    if not (tab / "physcl_exit.npz").is_file():
        pytest.skip(f"{tab} not found")
    px, _ = dumps.load_table(tab / "physcl_exit.npz")
    dx, _ = dumps.load_table(tab / "dssatdrv_exit.npz")
    j = px.index_of(list(dx.date))
    assert (j >= 0).all()
    pet = np.asarray(px.values["PET"], dtype=np.float64)[j]
    eop_ref = np.asarray(dx.values["EOP"], dtype=np.float32)
    st = EOPState.module(
        PETFluxes.zeros().replace(transpiration=jnp.asarray(pet)),
        CropWaterIn(
            sw=jnp.zeros((len(pet), 1)), eop=jnp.zeros((len(pet), 1)), trwup=jnp.zeros((len(pet), 1))
        ),
    )
    eop = np.asarray(eop_from_pet(st, None, None).crop_water.eop[:, 0])
    # the reference's arithmetic: REAL*4 EOP = REAL(PET) * 10 in single precision (every day)
    ref32 = (pet.astype(np.float32) * np.float32(10.0)).astype(np.float32)
    assert np.array_equal(ref32, eop_ref)
    # ours in float64 differs by the reference's two single-precision roundings, no more
    bound = 10.0 * np.abs(pet - pet.astype(np.float32)) + 0.5 * np.spacing(np.abs(eop_ref)).astype(np.float64)
    err = np.abs(eop - eop_ref.astype(np.float64))
    exact = int(np.sum(eop.astype(np.float32) == eop_ref))
    print("W3A eop_days", len(eop), "float32_equal", exact, "max_abs_mm", float(err.max()))
    # err and the bound are evaluated in float64 at the magnitude of EOP (33 days sit exactly on
    # the bound): a few float64 ulps of |EOP| of slack
    assert np.all(err <= bound + 4.0 * np.finfo(np.float64).eps * np.abs(eop))


def test_write_summary(runs: Any, reference: Any, inputs: Any, data_dir: Path) -> None:
    """Measured maxima per site to ``<data>/validation/aj_w3/pet_process_dumps.json``."""
    mask = inputs[1]
    out: dict[str, Any] = {}
    for i, site in enumerate(pdm.SITES):
        m = mask[i]
        row: dict[str, Any] = {"n_days": int(m.sum())}
        for case in ("pi", "default", "no_lag"):
            for v in tpd.FLUXES:
                e = np.abs(runs[case][v][i, m] - reference[v][i, m])
                row[f"{case}_{v}_max_abs_cm"] = float(e.max())
                row[f"{case}_{v}_max_rel"] = float((e / np.maximum(np.abs(reference[v][i, m]), 1e-300)).max())
        out[site] = row
    p = data_dir / OUT / "pet_process_dumps.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    assert p.stat().st_size > 0
