"""Milestone M1: the Richards redistribution driven by RZWQM2's own fluxes reproduces CA-TPA 2015.

Reference run: ``<data>/catpa/ref_2015`` (RZWQM2 ``main_ryzen5_avx512``, 2015-01-01 .. 2015-12-31,
``CA-TPA.ana`` and ``LAYER.PLT``). Only the model's *own* water inputs are taken from it, and the
drainage and the profile are computed here:

* initial profile: ``rzinit.dat`` (water content 0.1917 on every horizon; the ``.ana`` 2015.000 row,
  28.755 cm, is checked against it);
* surface supply: the day's infiltration, ``.ana`` column 5 (RZWQM routes rain through its
  Green-Ampt ``INFIL`` and hands Richards the infiltrated water), placed on the hours of the
  ``.BRK`` breakpoint storm of that day, any remainder (snowmelt) spread over the day;
* soil evaporation demand: the actual evaporation, ``.ana`` column 6, spread over the day;
* root water uptake: ``LAYER.PLT`` PLANT WATER UPTAKE per layer and day (its daily sum equals
  ``.ana`` column 7 to 1e-6 cm);
* grid and Brooks-Corey parameters: ``rzwqm.dat`` node records and horizon hydraulics.

Compared: daily profile storage against ``.ana`` column 2 (M1: RMSE < 0.05 cm), per-layer theta
against ``LAYER.PLT`` (RMSE < 0.01), drainage against ``.ana`` column 10.

The convergence table (sub-steps x iterations against 96 x 8) is written to
``$AGRI_JAX_RICHARDS_REPORT`` (a directory) when that variable is set.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax import lax

from agrijax.io.rzwqm import read_ana, read_brk, read_rzwqm_dat
from agrijax.io.rzwqm.layers import read_layer_output
from agrijax.processes.soil_water.hydraulics import SoilHydraulicParams
from agrijax.processes.soil_water.richards import (
    RichardsConfig,
    RichardsForcing,
    RichardsGrid,
    RichardsParams,
    SoilWater,
    richards_day,
)

pytestmark = [
    pytest.mark.allow_skip(reason="needs the private CA-TPA 2015 reference run under the data dir"),
    pytest.mark.skipif(not jax.config.read("jax_enable_x64"), reason="M1 is a float64 comparison"),
]

REF = Path("catpa/ref_2015")
SCENARIO = Path("narval_mirror/RZWQM_sw_batch/CA-TPA/Scenario")
THETA_INIT = 0.1917  # rzinit.dat, record 2 of every horizon (form 1: water content)
INCH_CM = 2.54


class Catpa2015:
    """Inputs and reference series of the CA-TPA 2015 run."""

    def __init__(self, data_dir: Path) -> None:
        ref, sc = data_dir / REF, data_dir / SCENARIO
        for p in (ref / "CA-TPA.ana", ref / "LAYER.PLT", sc / "rzwqm.dat", sc / "CA-TPA.BRK"):
            if not p.is_file():
                pytest.skip(f"{p} not found")
        dat = read_rzwqm_dat(sc / "rzwqm.dat")
        tlt = dat.node_depths_cm
        self.grid = RichardsGrid.from_rzwqm(tlt, dat.node_spacing_cm)
        nh = np.searchsorted(dat.horizon_depths_cm, tlt, side="left")
        self.soil = SoilHydraulicParams.from_rzwqm_dat(dat.hydraulics, node_horizon=nh)
        ana = read_ana(ref / "CA-TPA.ana")
        cols = {int(k): v for k, v in ana.attrs["columns"].items()}

        def col(n: int) -> np.ndarray:
            return np.asarray(ana[cols[n]].values, dtype=float)

        self.storage0 = col(2)[0]
        self.storage = col(2)[1:]
        self.infiltration = col(5)[1:]
        self.evaporation = col(6)[1:]
        self.transpiration = col(7)[1:]
        self.deep_seepage = col(10)[1:]
        self.days = np.asarray(ana.time.values[1:], dtype="datetime64[D]")
        lay = read_layer_output(ref / "LAYER.PLT", start="2015-01-01")
        self.theta = np.asarray(lay["soil_water_content"].values)
        self.uptake = np.asarray(lay["plant_water_uptake"].values)
        self.thickness = np.asarray(lay["thickness"].values)
        self.supply = self._hourly_supply(read_brk(sc / "CA-TPA.BRK"))
        self.evap_hourly = np.repeat(self.evaporation[:, None] / 24.0, 24, axis=1)

    def _hourly_supply(self, brk: Any) -> np.ndarray:
        ev, bp = brk.events, brk.breakpoints
        out = np.zeros((len(self.days), 24))
        hours = np.arange(25.0)
        for d, day in enumerate(self.days):
            rest = self.infiltration[d]
            if rest <= 0.0:
                continue
            e = ev[ev["date"] == day]
            if len(e):
                b = bp[bp["event"] == e.index[0]]
                cum = np.interp(
                    hours, b["time_min"].to_numpy() / 60.0, b["cum_depth_in"].to_numpy() * INCH_CM
                )
                rain = np.diff(cum)
                if rain.sum() > 0.0:
                    part = min(rest, rain.sum())
                    out[d] += rain / rain.sum() * part
                    rest -= part
            out[d] += rest / 24.0
        return out

    def run(self, **cfg: Any) -> dict[str, np.ndarray]:
        params = RichardsParams(soil=self.soil, grid=self.grid, config=RichardsConfig(**cfg))
        w0 = SoilWater.from_theta(jnp.full(self.grid.n_node, THETA_INIT), self.soil)
        forcing = RichardsForcing(
            supply=jnp.asarray(self.supply),
            evaporation=jnp.asarray(self.evap_hourly),
            uptake=jnp.asarray(self.uptake),
        )

        def body(w, f):
            w2 = richards_day(w, params, f.supply, f.evaporation, f.uptake)
            return w2, (w2.theta, w2.storage(self.grid), w2.flux)

        _, (theta, storage, flux) = jax.jit(lambda w, f: lax.scan(body, w, f))(w0, forcing)
        out = {k: np.asarray(v) for k, v in flux.items()}
        out["theta"] = np.asarray(theta)
        out["storage"] = np.asarray(storage)
        return out


@pytest.fixture(scope="module")
def catpa(data_dir: Path) -> Catpa2015:
    return Catpa2015(data_dir)


@pytest.fixture(scope="module")
def converged(catpa: Catpa2015) -> dict[str, np.ndarray]:
    return catpa.run(n_sub=96, n_iter=8)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def test_inputs_are_consistent(catpa: Catpa2015) -> None:
    """The driving series close RZWQM's own balance and match the grid and the initial profile."""
    assert catpa.grid.n_node == 37
    np.testing.assert_allclose(catpa.grid.tl, catpa.thickness, atol=1e-12)
    assert THETA_INIT * 150.0 == pytest.approx(catpa.storage0, abs=1e-6)
    np.testing.assert_allclose(catpa.uptake.sum(axis=1), catpa.transpiration, atol=2e-6)
    np.testing.assert_allclose(catpa.supply.sum(axis=1), catpa.infiltration, atol=1e-12)
    # RZWQM's own balance with these columns (print precision 1e-4 cm)
    ds = np.diff(np.r_[catpa.storage0, catpa.storage])
    res = ds - (catpa.infiltration - catpa.evaporation - catpa.transpiration - catpa.deep_seepage)
    assert np.max(np.abs(res)) < 1e-4


def test_m1_profile_storage_and_layer_theta(catpa: Catpa2015, converged: dict[str, np.ndarray]) -> None:
    """M1 at the converged configuration (96 x 8): storage RMSE < 0.05 cm, per-layer theta RMSE < 0.01."""
    r = converged
    storage_rmse = _rmse(r["storage"], catpa.storage)
    theta_rmse = _rmse(r["theta"], catpa.theta)
    # measured: storage RMSE 0.0257 cm (max 0.094 cm), theta RMSE 0.0011
    assert storage_rmse < 0.05
    assert theta_rmse < 0.01
    assert np.max(np.abs(r["storage"] - catpa.storage)) < 0.15
    # the drainage is computed, not prescribed: 19.901 cm against RZWQM's 19.907 cm
    assert r["drainage"].sum() == pytest.approx(catpa.deep_seepage.sum(), rel=2e-3)
    # every prescribed input went in; the only loss is a small supply limit on evaporation
    assert r["infiltration"].sum() == pytest.approx(catpa.infiltration.sum(), rel=1e-10)
    assert r["uptake"].sum() == pytest.approx(catpa.uptake.sum(), rel=1e-10)
    assert r["runoff"].sum() == 0.0
    assert r["evaporation_deficit"].sum() < 0.03  # measured 0.013 cm of 19.67 cm
    assert np.max(np.abs(r["balance_error"])) < 1e-10
    assert r["n_clamp"].sum() == 0.0


def test_m1_working_configuration(catpa: Catpa2015) -> None:
    """24 x 3 (the working point) also meets M1; 12 x 2 does not (recorded, not required)."""
    r = catpa.run(n_sub=24, n_iter=3)
    # measured: 24 x 3 storage RMSE 0.0483 cm, theta RMSE 0.0012, |imbalance| 0.054 cm over the year
    assert _rmse(r["storage"], catpa.storage) < 0.05
    assert _rmse(r["theta"], catpa.theta) < 0.01
    assert np.sum(np.abs(r["balance_error"])) < 0.1
    r12 = catpa.run(n_sub=12, n_iter=2)
    # measured: 12 x 2 storage RMSE 0.221 cm, 0.75 cm imbalance; 6 x 1 2.85 cm (see the convergence table)
    assert 0.05 < _rmse(r12["storage"], catpa.storage) < 0.4


CONVERGENCE_CONFIGS = [
    (6, 1),
    (6, 2),
    (12, 1),
    (12, 2),
    (24, 2),
    (24, 3),
    (48, 3),
    (48, 4),
    (96, 8),
    (240, 10),
]


@pytest.mark.slow
def test_convergence_table_catpa(catpa: Catpa2015, converged: dict[str, np.ndarray]) -> None:
    """Sub-steps x iterations against 96 x 8 on CA-TPA 2015 (grade-C tolerance evidence)."""
    rows = []
    for n_sub, n_iter in CONVERGENCE_CONFIGS:
        for scheme in ("implicit", "rzwqm"):
            r = catpa.run(n_sub=n_sub, n_iter=n_iter, time_scheme=scheme)
            rows.append(
                {
                    "n_sub": n_sub,
                    "n_iter": n_iter,
                    "time_scheme": scheme,
                    "storage_maxabs_vs_96x8_cm": float(np.max(np.abs(r["storage"] - converged["storage"]))),
                    "theta_rmse_vs_96x8": _rmse(r["theta"], converged["theta"]),
                    "storage_rmse_vs_rzwqm_cm": _rmse(r["storage"], catpa.storage),
                    "theta_rmse_vs_layer_plt": _rmse(r["theta"], catpa.theta),
                    "drainage_cm": float(r["drainage"].sum()),
                    "sum_abs_balance_error_cm": float(np.sum(np.abs(r["balance_error"]))),
                    "max_theta_residual": float(np.max(r["max_theta_residual"])),
                    "n_clamp": float(np.sum(r["n_clamp"])),
                }
            )
    table = pd.DataFrame(rows)
    out = os.environ.get("AGRI_JAX_RICHARDS_REPORT")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        table.to_csv(Path(out) / "convergence_catpa2015.csv", index=False, float_format="%.6g")
    imp = table[table["time_scheme"] == "implicit"].set_index(["n_sub", "n_iter"])
    err = imp["storage_maxabs_vs_96x8_cm"]
    assert err[(240, 10)] < 0.05  # the reference is itself converged to this level
    assert err[(6, 1)] > err[(12, 2)] > err[(24, 3)] > err[(48, 4)]
    assert imp.loc[(24, 3), "storage_rmse_vs_rzwqm_cm"] < 0.05
    assert (imp["n_clamp"] == 0).all()
