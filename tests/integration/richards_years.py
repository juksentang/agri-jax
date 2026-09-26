"""M1 over many years: the Richards replay of an RZWQM2 run, year by year (H1 item 4).

Helper module of ``test_richards_years.py`` and a script
(``python tests/integration/richards_years.py [site ...]``) that writes the per-year table.

The replay is the M1 recipe of ``test_richards_catpa.py``, generalised from CA-TPA 2015 to a
whole reference run and to other scenarios: RZWQM2's own water inputs are taken from the run and
the drainage and the profile are computed by :func:`~agrijax.processes.soil_water.richards.richards_day`:

* surface supply: the day's infiltration (``.ana`` column 5), placed on the hours of the day's
  ``.BRK`` breakpoint storm, any remainder (snowmelt, irrigation) spread over the day;
* soil evaporation demand: the actual evaporation (``.ana`` column 6), spread over the day;
* root water uptake: ``LAYER.PLT`` PLANT WATER UPTAKE per layer and day;
* grid and Brooks-Corey parameters: ``rzwqm.dat`` (start-up parameters, no tillage);
* initial profile: ``rzinit.dat`` (form 1, water content per horizon, mapped to the nodes of each
  horizon as the soil parameters are).

Two ways of cutting the run into years:

* ``restart`` (the M1 check of each year): every calendar year starts from RZWQM2's profile at
  the end of the previous year (``LAYER.PLT``, printed to 6 digits), the first year from
  ``rzinit.dat``. Each year is then an independent one-year M1 run; the first year of CA-TPA is
  exactly the M1 run of ``test_richards_catpa.py``;
* ``free``: the whole run from ``rzinit.dat`` without any reset (drift accumulates, reported).

Compared per year: daily profile storage against ``.ana`` column 2 (M1: RMSE < 0.05 cm), per-layer
theta against ``LAYER.PLT`` (RMSE < 0.01) and the drainage against ``.ana`` column 10.

A scenario is *comparable* when the replay's physics is the reference run's: free drainage at the
bottom (control record item 3 ``IREBOT = 2``), no perched water table (item 9), no tile drains
(item 11), no macropores (macropore record item 1.1), and none of ``.ana`` tile drainage, lateral
flow or "water added due to measured SWC" non-zero. :func:`comparability` reads the flags.

Diagnostics for CA-TPA (the instrumented 2015-2023 run, ``dumps/tables/rzwqm46_catpa2015_2023``):
``PHYSCL`` entry dumps carry the day's ``SOILHP`` (post-tillage parameters) and the entry profile
``THETA``; :meth:`RunReplay.run` accepts per-day soil parameters (the two-segment post-tillage
curve, :class:`~agrijax.processes.soil_water.hydraulics.TilledSoilHydraulicParams`) and per-day
profile resets, so a year's error can be attributed to tillage.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # script use: import the sibling test modules' staging helpers
    sys.path.insert(0, str(HERE))

BATCH = Path("narval_mirror/RZWQM_sw_batch")
CATPA_BASE = Path("catpa/base_2015_2023")
CATPA_DUMPS = Path("dumps/tables/rzwqm46_catpa2015_2023")
OUT_SUBDIR = Path("validation/h1_4_richards_m1/final")
INCH_CM = 2.54
CONFIGS = ((96, 8), (24, 3), (12, 2))
#: M1 bounds (validation matrix): storage RMSE [cm] and per-layer theta RMSE
M1_STORAGE_RMSE = 0.05
M1_THETA_RMSE = 0.01
#: all local scenarios of the batch (CFIA_Ottawa has no rzwqm.dat, see NOTES.md)
SITES = (
    "CA-ER1",
    "CA-MA1",
    "CA-TPA",
    "US-LYS_NW",
    "US-LYS_SE",
    "US-LYS_SW",
    "US-Mj1",
    "US-S2",
    "US-TW3",
    "US-Tw2",
    "US-UA1_HartFarm",
    "US-manilacotton",
    "US_OPE",
    "US_Rockfish",
    "US_Rockford_Alfalfa",
)
KEEP = ("*.ana", "LAYER.PLT", "MANAGE.OUT")


# ------------------------------------------------------------------------------ scenario files


def _find_ci(folder: Path, name: str) -> Path | None:
    low = name.lower()
    return next((p for p in folder.iterdir() if p.name.lower() == low), None)


def _ipnames_file(scenario: Path, line: int) -> Path:
    """File named on ``line`` (0-based) of ``IPNAMES.DAT``, found case-insensitively in ``scenario``."""
    ip = _find_ci(scenario, "IPNAMES.DAT")
    if ip is None:
        raise FileNotFoundError(f"{scenario}: no IPNAMES.DAT")
    name = re.split(r"[\\/]", ip.read_bytes().decode("latin-1").splitlines()[line].strip())[-1]
    p = _find_ci(scenario, name)
    if p is None:
        raise FileNotFoundError(f"{name} not in {scenario}")
    return p


def brk_path(scenario: Path) -> Path:
    """The breakpoint rain file of a scenario (``IPNAMES.DAT`` line 4)."""
    return _ipnames_file(scenario, 3)


def rzinit_path(scenario: Path) -> Path:
    """The initial-state file of a scenario (``IPNAMES.DAT`` line 5)."""
    return _ipnames_file(scenario, 4)


def _data_rows(path: Path) -> list[list[str]]:
    rows = []
    for ln in path.read_bytes().decode("latin-1").splitlines():
        s = ln.strip()
        if s and not s.startswith("="):
            rows.append(s.split())
    return rows


def read_rzinit_theta(path: Path, n_horizon: int) -> np.ndarray:
    """Initial water content per horizon of ``rzinit.dat`` (record 1: form, restore; record 2 per horizon).

    Only form 1 (water content) with restore 0 (continuous run) is accepted; every local scenario
    has it.
    """
    rows = _data_rows(path)
    form, restore = int(float(rows[0][0])), int(float(rows[0][1]))
    if form != 1 or restore != 0:
        raise ValueError(f"{path}: form {form} / restore {restore} not supported (need 1 / 0)")
    return np.array([float(r[0]) for r in rows[1 : 1 + n_horizon]])


def _record_after(lines: Sequence[str], marker: str) -> list[str]:
    """First data record (not a ``=`` comment) after the line containing ``marker``."""
    for i, ln in enumerate(lines):
        if marker in ln:
            for nxt in lines[i + 1 :]:
                s = nxt.strip()
                if s and not s.startswith("="):
                    return s.split()
    raise LookupError(marker)


def comparability_record(dat_path: Path) -> list[str]:
    """The Richards control record of ``rzwqm.dat`` (19 items, see its header comment)."""
    lines = dat_path.read_bytes().decode("latin-1").splitlines()
    return _record_after(lines, "Wilting Point water suction head")


def comparability(dat_path: Path) -> dict[str, Any]:
    """Richards-relevant switches of ``rzwqm.dat`` and whether the M1 replay covers them."""
    lines = dat_path.read_bytes().decode("latin-1").splitlines()
    ctl = comparability_record(dat_path)
    mac = _record_after(lines, "fraction of dead end pores")
    flags: dict[str, Any] = {
        "irebot": int(float(ctl[2])),
        "water_table": int(float(ctl[8])),
        "drains": int(float(ctl[10])),
        "macropores": int(float(mac[0])),
    }
    why = []
    if flags["irebot"] != 2:
        why.append(f"bottom boundary IREBOT={flags['irebot']} (the solver has free drainage only)")
    if flags["water_table"]:
        why.append("perched water table")
    if flags["drains"]:
        why.append("tile drains")
    if flags["macropores"]:
        why.append("macropores")
    flags["comparable"] = not why
    flags["why_not"] = "; ".join(why)
    return flags


# ------------------------------------------------------------------------------ reference runs


def run_scenario(site: str, data_dir: Path, out_root: Path, stage_root: Path) -> dict[str, Any]:
    """Run RZWQM2 on one scenario for its whole ``IPNAMES.DAT`` period; never raises."""
    from test_rzwqm_all_scenarios import _stage_source

    import agrijax.port.run_fortran as rf

    rec: dict[str, Any] = {"site": site, "ok": False, "error": ""}
    out = out_root / site
    try:
        scenario = data_dir / BATCH / site / "Scenario"
        run_root = Path(os.environ.get("AGRI_JAX_RUN_ROOT", str(data_dir / "run")))
        src = _stage_source(site, scenario, stage_root, run_root)
        t0 = time.perf_counter()
        rf.run_rzwqm(src, out, keep_files=KEEP, timeout=3600, run_root=run_root)
        rec.update(ok=True, wall_s=time.perf_counter() - t0, source=src, out=out)
    except Exception as e:  # reported per scenario
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


# ------------------------------------------------------------------------------ replay


class RunReplay:
    """Inputs and reference series of one RZWQM2 run, for the M1 replay year by year."""

    def __init__(self, site: str, run_dir: Path, scenario: Path) -> None:
        from agrijax.io.rzwqm import read_ana, read_brk, read_rzwqm_dat
        from agrijax.io.rzwqm.layers import read_layer_output, simulation_start
        from agrijax.processes.soil_water.hydraulics import SoilHydraulicParams
        from agrijax.processes.soil_water.richards import RichardsGrid

        self.site = site
        ana_path = next(p for p in run_dir.iterdir() if p.suffix.lower() == ".ana")
        dat_path = scenario / "rzwqm.dat"
        self.flags = comparability(dat_path)
        dat = read_rzwqm_dat(dat_path)
        tlt = dat.node_depths_cm
        self.grid = RichardsGrid.from_rzwqm(tlt, dat.node_spacing_cm)
        self.node_horizon = np.searchsorted(dat.horizon_depths_cm, tlt, side="left")
        self.soil = SoilHydraulicParams.from_rzwqm_dat(dat.hydraulics, node_horizon=self.node_horizon)
        self.n_horizon = dat.n_horizon
        theta_h = read_rzinit_theta(rzinit_path(scenario), dat.n_horizon)
        # RZWQM2 keeps the head at or above Hmin (CNHEAD/HYDPAR, see H_CLAMP_RZWQM): a horizon given
        # drier than theta(Hmin) starts at theta(Hmin)
        from agrijax.processes.soil_water.hydraulics import H_CLAMP_RZWQM, h_of_theta, theta_of_h

        self.theta_file = theta_h[self.node_horizon]
        h0 = np.maximum(np.asarray(h_of_theta(self.theta_file, self.soil)), H_CLAMP_RZWQM)
        self.theta_init = np.where(h0 > H_CLAMP_RZWQM, self.theta_file, np.asarray(theta_of_h(h0, self.soil)))

        ana = read_ana(ana_path)
        cols = {int(k): v for k, v in ana.attrs["columns"].items()}

        def col(n: int) -> np.ndarray:
            return np.asarray(ana[cols[n]].values, dtype=float)

        def named(name: str) -> np.ndarray:
            return np.asarray(ana[name].values, dtype=float)[1:]

        self.storage0 = col(2)[0]
        self.storage = col(2)[1:]
        self.precipitation = col(3)[1:]
        self.infiltration = col(5)[1:]
        self.evaporation = col(6)[1:]
        self.transpiration = col(7)[1:]
        self.deep_seepage = col(10)[1:]
        self.runoff = col(12)[1:]
        snow_depth, melt = col(92), col(105)
        #: snow days (a snowpack at the start or end of the day, or snowmelt), as test_infiltration_catpa
        self.snow = (snow_depth[1:] > 0.0) | (snow_depth[:-1] > 0.0) | (melt[1:] > 0.0)
        self.aef = float(comparability_record(dat_path)[3])
        self.irrigation = named("irrigation")
        self.other_terms = {
            k: named(k)
            for k in ("tile_drainage", "lateral_water_flow", "water_added_due_to_using_measured_swc")
        }
        self.days = np.asarray(ana.time.values[1:], dtype="datetime64[D]")
        start = simulation_start(_find_ci(run_dir, "IPNAMES.DAT") or scenario / "IPNAMES.DAT")
        lay = read_layer_output(run_dir / "LAYER.PLT", start=start)
        if len(lay.time) != len(self.days):
            raise ValueError(f"{site}: LAYER.PLT has {len(lay.time)} days, .ana {len(self.days)}")
        self.theta = np.asarray(lay["soil_water_content"].values)
        self.uptake = np.asarray(lay["plant_water_uptake"].values)
        self.bulk_density = np.asarray(lay["soil_bulk_density"].values)
        self.thickness = np.asarray(lay["thickness"].values)
        self.years = self.days.astype("datetime64[Y]").astype(int) + 1970
        self.brk = read_brk(brk_path(scenario))
        self.supply = self._hourly_supply()
        self.evap_hourly = np.repeat(self.evaporation[:, None] / 24.0, 24, axis=1)

    def _hourly_supply(self) -> np.ndarray:
        """The M1 placement of ``.ana`` column 5 on the breakpoint hours (``test_richards_catpa.py``)."""
        ev, bp = self.brk.events, self.brk.breakpoints
        by_date = {np.datetime64(d, "D"): k for k, d in zip(ev.index, ev["date"], strict=True)}
        out = np.zeros((len(self.days), 24))
        hours = np.arange(25.0)
        for d, day in enumerate(self.days):
            rest = self.infiltration[d]
            if rest <= 0.0:
                continue
            k = by_date.get(day)
            if k is not None:
                b = bp[bp["event"] == k]
                t_h = np.asarray(b["time_min"], dtype=float) / 60.0
                cum = np.interp(hours, t_h, np.asarray(b["cum_depth_in"], dtype=float) * INCH_CM)
                rain = np.diff(cum)
                if rain.sum() > 0.0:
                    part = min(rest, rain.sum())
                    out[d] += rain / rain.sum() * part
                    rest -= part
            out[d] += rest / 24.0
        return out

    # -------------------------------------------------------------------- consistency

    def input_checks(self) -> dict[str, float]:
        """The driving series against RZWQM's own balance, grid and initial profile (print precision)."""
        ds = np.diff(np.r_[self.storage0, self.storage])
        res = ds - (self.infiltration - self.evaporation - self.transpiration - self.deep_seepage)
        return {
            "grid_vs_layer_plt_maxabs_cm": float(np.max(np.abs(np.asarray(self.grid.tl) - self.thickness))),
            "init_storage_vs_ana_cm": float(np.sum(self.theta_init * self.thickness) - self.storage0),
            "uptake_vs_col7_maxabs_cm": float(np.max(np.abs(self.uptake.sum(axis=1) - self.transpiration))),
            "supply_vs_col5_maxabs_cm": float(np.max(np.abs(self.supply.sum(axis=1) - self.infiltration))),
            "balance_residual_maxabs_cm": float(np.max(np.abs(res))),
            **{f"{k}_sum_cm": float(np.abs(v).sum()) for k, v in self.other_terms.items()},
        }

    # -------------------------------------------------------------------- run

    def restart_profiles(self, mode: str) -> tuple[np.ndarray, np.ndarray]:
        """Per-day reset flag and profile: ``restart`` resets on Jan 1 from the previous day's ``LAYER.PLT``."""
        n = len(self.days)
        reset = np.zeros(n, bool)
        start = np.concatenate([self.theta_init[None, :], self.theta[:-1]])
        if mode == "restart":
            reset[1:] = self.years[1:] != self.years[:-1]
        elif mode != "free":
            raise ValueError(mode)
        return reset, start

    def run(
        self,
        *,
        mode: str = "restart",
        soil_days: Any = None,
        pond_max: float = 0.0,
        evap_hourly: np.ndarray | None = None,
        reset_days: np.ndarray | None = None,
        reset_theta: np.ndarray | None = None,
        **cfg: Any,
    ) -> dict[str, np.ndarray]:
        """The whole run with :func:`richards_day`; ``cfg`` goes to :class:`RichardsConfig`.

        ``soil_days``: per-day soil parameters (a pytree with a leading day axis) in place of the
        static ``rzwqm.dat`` ones. ``reset_days``/``reset_theta``: extra per-day profile resets
        (e.g. the ``PHYSCL`` entry profile on tillage days). ``pond_max`` [cm]: surface ponding
        allowed before runoff (0 in M1, RZWQM2's Richards step has no ponding layer).
        ``evap_hourly`` [day, 24] replaces the uniform spread of the evaporation.
        """
        import jax
        import jax.numpy as jnp
        from jax import lax

        from agrijax.processes.soil_water.richards import (
            RichardsConfig,
            RichardsParams,
            SoilWater,
            richards_day,
        )

        reset, start = self.restart_profiles(mode)
        if reset_days is not None:
            assert reset_theta is not None
            start = np.where(reset_days[:, None], reset_theta, start)
            reset = reset | reset_days
        config = RichardsConfig(**cfg)
        grid = self.grid
        static_soil = self.soil
        w0 = SoilWater.from_theta(jnp.asarray(self.theta_init), self.soil)

        def body(w: Any, f: Any) -> tuple[Any, Any]:
            sup, eva, upt, flag, th, soil = f
            soil = static_soil if soil is None else soil
            params = RichardsParams(soil=soil, grid=grid, config=config, pond_max=jnp.asarray(pond_max))
            fresh = SoilWater.from_theta(th, soil)
            w = jax.tree_util.tree_map(lambda a, b: jnp.where(flag, a, b), fresh, w)
            w2 = richards_day(w, params, sup, eva, upt)
            return w2, (w2.theta, w2.storage(grid), w2.flux)

        xs = (
            jnp.asarray(self.supply),
            jnp.asarray(self.evap_hourly if evap_hourly is None else evap_hourly),
            jnp.asarray(self.uptake),
            jnp.asarray(reset),
            jnp.asarray(start),
            soil_days,
        )
        _, (theta, storage, flux) = jax.jit(lambda w, f: lax.scan(body, w, f))(w0, xs)
        out = {k: np.asarray(v) for k, v in flux.items()}
        out["theta"] = np.asarray(theta)
        out["storage"] = np.asarray(storage)
        return out

    def run_events(self, *, mode: str = "restart", n_sub: int = 96, n_iter: int = 8) -> dict[str, np.ndarray]:
        """Diagnostic: the same years with our Green-Ampt event day instead of the replayed infiltration.

        :func:`~agrijax.processes.soil_water.day.soil_water_day_kernel` (redistribution, the day's
        ``.BRK`` storm through Green-Ampt, redistribution; ``DayConfig(n_pre=0, n_post=n_sub)``) as in
        ``test_infiltration_catpa.py``: snow days get no event and their infiltration is replayed;
        on event days the infiltration beyond the storm's net rain (``.ana`` rain - runoff, i.e.
        irrigation) is replayed over the day as well. Two storms on one day are not supported by
        the event kernel (``ValueError``).
        """
        import jax
        import jax.numpy as jnp
        from jax import lax
        from test_infiltration_catpa import storm_forcing_from_breakpoints

        from agrijax.processes.soil_water.day import DayConfig, SoilWaterDayParams, soil_water_day_kernel
        from agrijax.processes.soil_water.infiltration import GreenAmptConfig, GreenAmptParams, StormForcing
        from agrijax.processes.soil_water.richards import RichardsConfig, RichardsParams, SoilWater

        storms = storm_forcing_from_breakpoints(self.days, self.brk.events, self.brk.breakpoints)
        rain = np.asarray(storms.depth).sum(axis=1)
        event = (rain > 0.0) & ~self.snow
        keep = jnp.asarray(event)[:, None]
        storm = StormForcing(
            ts0=storms.ts0,
            duration=jnp.where(keep, storms.duration, 0.0),
            depth=jnp.where(keep, storms.depth, 0.0),
        )
        extra = np.maximum(self.infiltration - (self.precipitation - self.runoff), 0.0)
        replay = np.where(event, extra, self.infiltration)
        supply = np.repeat(replay[:, None] / 24.0, 24, axis=1)
        rp = RichardsParams(soil=self.soil, grid=self.grid, config=RichardsConfig(n_sub=n_sub, n_iter=n_iter))
        gp = GreenAmptParams(
            aef=jnp.asarray(self.aef), config=GreenAmptConfig.for_grid(np.asarray(self.grid.tl))
        )
        params = SoilWaterDayParams(richards=rp, infiltration=gp, config=DayConfig(n_pre=0, n_post=n_sub))
        reset, start = self.restart_profiles(mode)
        w0 = SoilWater.from_theta(jnp.asarray(self.theta_init), self.soil)
        soil = self.soil

        def body(w: Any, f: Any) -> tuple[Any, Any]:
            sup, eva, upt, st, flag, th = f
            fresh = SoilWater.from_theta(th, soil)
            w = jax.tree_util.tree_map(lambda a, b: jnp.where(flag, a, b), fresh, w)
            w2, _, _ = soil_water_day_kernel(w, params, sup, eva, upt, st)
            return w2, (w2.theta, w2.storage(self.grid), w2.flux)

        xs = (
            jnp.asarray(supply),
            jnp.asarray(self.evap_hourly),
            jnp.asarray(self.uptake),
            storm,
            jnp.asarray(reset),
            jnp.asarray(start),
        )
        _, (theta, storage, flux) = jax.jit(lambda w, f: lax.scan(body, w, f))(w0, xs)
        out = {k: np.asarray(v) for k, v in flux.items()}
        out["theta"] = np.asarray(theta)
        out["storage"] = np.asarray(storage)
        out["n_event_days"] = np.asarray(event)
        return out

    # -------------------------------------------------------------------- per-year table

    @staticmethod
    def _budget_terms(r: dict[str, np.ndarray], m: np.ndarray, ref_drainage: np.ndarray) -> dict[str, float]:
        return {
            "budget_runoff_cm": float(-r["runoff"][m].sum()),
            "budget_evaporation_deficit_cm": float(r["evaporation_deficit"][m].sum()),
            "budget_uptake_cut_cm": float(r["uptake_cut"][m].sum()),
            "budget_drainage_cm": float(-(r["drainage"][m] - ref_drainage[m]).sum()),
            "budget_balance_cm": float(r["balance_error"][m].sum()),
        }

    def _budget(self, r: dict[str, np.ndarray], m: np.ndarray) -> dict[str, Any]:
        """The year's storage error budget and its largest term.

        With the supply prescribed, the end-of-year storage error is exactly (up to the start
        profile's 6-digit print) ``- runoff + evaporation deficit + uptake cut - (drainage -
        reference drainage) + balance error``: water the replay could not put in, could not
        take out, drained differently, or lost to an unconverged solve.
        """
        b = self._budget_terms(r, m, self.deep_seepage)
        dominant = max(b, key=lambda k: abs(b[k]))
        return {**b, "budget_dominant": dominant.removeprefix("budget_").removesuffix("_cm")}

    def fingerprint(self) -> dict[str, float]:
        """Sums of the reference series the replay reads (a changed reference run shows up here)."""
        return {
            "sum_storage_cm": float(self.storage.sum()),
            "sum_infiltration_cm": float(self.infiltration.sum()),
            "sum_evaporation_cm": float(self.evaporation.sum()),
            "sum_uptake_cm": float(self.uptake.sum()),
            "sum_deep_seepage_cm": float(self.deep_seepage.sum()),
            "sum_theta": float(self.theta.sum()),
        }

    def year_table(self, r: dict[str, np.ndarray], **tags: Any) -> pd.DataFrame:
        """Per-year M1 metrics of one replay ``r`` against the reference series."""
        rows = []
        for y in np.unique(self.years):
            m = self.years == y
            ds = r["storage"][m] - self.storage[m]
            rows.append(
                {
                    "site": self.site,
                    **tags,
                    "year": int(y),
                    "n_days": int(m.sum()),
                    "storage_rmse_cm": float(np.sqrt(np.mean(ds**2))),
                    "storage_maxabs_cm": float(np.max(np.abs(ds))),
                    "storage_bias_cm": float(np.mean(ds)),
                    "storage_end_diff_cm": float(ds[-1]),
                    "theta_rmse": float(np.sqrt(np.mean((r["theta"][m] - self.theta[m]) ** 2))),
                    "drainage_cm": float(r["drainage"][m].sum()),
                    "ref_drainage_cm": float(self.deep_seepage[m].sum()),
                    "infiltration_cm": float(self.infiltration[m].sum()),
                    "max_day_infiltration_cm": float(self.infiltration[m].max()),
                    "irrigation_cm": float(self.irrigation[m].sum()),
                    "runoff_cm": float(r["runoff"][m].sum()),
                    "ref_runoff_cm": float(self.runoff[m].sum()),
                    "evaporation_deficit_cm": float(r["evaporation_deficit"][m].sum()),
                    "uptake_cut_cm": float(r["uptake_cut"][m].sum()),
                    "sum_abs_balance_error_cm": float(np.abs(r["balance_error"][m]).sum()),
                    "n_clamp": float(r["n_clamp"][m].sum()),
                    "worst_day": str(self.days[m][int(np.argmax(np.abs(ds)))]),
                    **self._budget(r, m),
                }
            )
        return pd.DataFrame(rows)


def catpa_replay(data_dir: Path) -> RunReplay:
    """The CA-TPA 2015-2023 reference run (``catpa/base_2015_2023``)."""
    return RunReplay("CA-TPA", data_dir / CATPA_BASE, data_dir / BATCH / "CA-TPA" / "Scenario")


# ------------------------------------------------------------------------------ RZWQM2 conventions
#
# Two conventions of RZWQM2 4.6 that the M1 replay (and richards.py) do not have. They are emulated
# here, in the test harness only, to show that they explain most of the failing site-years; porting
# them is a model change for M3 (private design note 16, section 11), not part of this check.
#
# * DRAIN (Rzday.for:3975, called by REDIST at Rzrich.for:1092 after every RICHRD step): a node whose
#   water content exceeds the field-saturated porosity PORI = AEF * theta_s passes the excess to the
#   node below at once; the bottom node's excess leaves as seepage (counted in the deep seepage), and
#   the heads are recomputed from the water contents (WCHEAD). With it the reference never holds more
#   than AEF * theta_s in a node (Ahuja et al. 2000, ch. 3: field saturation).
# * flux-mode surface limit (CHKBC, Rzrich.for:3-110; CNHEAD, Rzrich.for:120): the evaporation demand
#   is a flux boundary at the ghost node as long as the ghost head found by the iteration stays above
#   Hmin; only then does the top switch to the head boundary Hmin. With the geometric-mean face K and
#   eps > 2 the Darcy flux to the ghost node is not monotone in the ghost head: it peaks near
#   eps / (eps - 2) times the node head and is smaller at Hmin. So RZWQM2 delivers any demand up to
#   that peak, while richards.surface_fluxes caps the evaporation at the flux with the ghost at Hmin.

#: ghost heads sampled (log-spaced between the surface node head and Hmin) for the flux-mode peak
N_GHOST = 256
#: smallest |head| [cm] of the ghost-head sample (keeps the log finite on a saturated surface)
GHOST_ABS_H_MIN = 1.0e-3


def pori_of(rep: RunReplay) -> np.ndarray:
    """Field-saturated porosity ``AEF * theta_s`` per node (RZWQM2 ``PORI``, Rzmain.for:538)."""
    import jax
    import jax.numpy as jnp

    from agrijax.processes.soil_water.hydraulics import theta_of_h

    n = rep.grid.n_node
    soil = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,)), rep.soil.at_nodes())
    return rep.aef * np.asarray(theta_of_h(jnp.zeros(n), soil))


def drain_cascade(theta: Any, tl: Any, pori: Any) -> tuple[Any, Any]:
    """RZWQM2 ``DRAIN``: excess above ``pori`` moves down node by node; returns ``(theta, seepage [cm])``."""
    import jax.numpy as jnp
    from jax import lax

    def f(carry: Any, x: Any) -> tuple[Any, Any]:
        th, t, p = x
        w = th * t + carry
        cap = p * t
        keep = jnp.where(w > cap, cap, w)
        return w - keep, keep / t

    seep, th = lax.scan(f, jnp.zeros((), theta.dtype), (theta, tl, pori))
    return th, seep


def surface_fluxes_flux_mode(h: Any, h_k: Any, a: Any) -> tuple[Any, Any, Any]:
    """:func:`richards.surface_fluxes` with the dry limit at the peak of the flux-mode Darcy flux."""
    import jax
    import jax.numpy as jnp

    from agrijax.processes.soil_water import richards
    from agrijax.processes.soil_water.hydraulics import k_of_h

    _, q_wet, q_dry_hmin = _SURFACE_FLUXES(h, h_k, a)
    ht0 = a.alpha * h[0] + (1.0 - a.alpha) * a.h_old[0]
    hk0 = a.alpha * h_k[0] + (1.0 - a.alpha) * a.h_old[0]
    soil0 = jax.tree_util.tree_map(lambda x: x[:1], a.soil)
    k_node = richards._log_k(k_of_h(hk0[None], soil0))[0]
    lo = jnp.log(jnp.where(-ht0 > GHOST_ABS_H_MIN, -ht0, GHOST_ABS_H_MIN))
    hg = -jnp.exp(jnp.linspace(0.0, 1.0, N_GHOST) * (jnp.log(-a.h_min) - lo) + lo)
    soil_g = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[:1], (N_GHOST,)), a.soil)
    kg = richards._log_k(k_of_h(hg, soil_g))
    qg = -jnp.exp(0.5 * (k_node + kg)) * ((ht0 - hg) / a.dz_top - 1.0)
    q_peak = jnp.min(jnp.concatenate([qg, q_dry_hmin[None]]))
    q_dry = jnp.where(q_peak < 0.0, q_peak, 0.0)
    q_top = jnp.where(a.q_demand > q_wet, q_wet, jnp.where(a.q_demand < q_dry, q_dry, a.q_demand))
    return q_top, q_wet, q_dry


def _original_surface_fluxes() -> Any:
    from agrijax.processes.soil_water import richards

    return richards.surface_fluxes


_SURFACE_FLUXES = _original_surface_fluxes()


def run_conventions(
    rep: RunReplay,
    *,
    n_sub: int = 96,
    n_iter: int = 8,
    drain: bool = True,
    flux_mode: bool = True,
    mode: str = "restart",
    **cfg: Any,
) -> dict[str, np.ndarray]:
    """The replay of :meth:`RunReplay.run` with the DRAIN cap after every sub-step and/or the flux-mode limit.

    With ``drain=False, flux_mode=False`` it is the M1 replay (same values as :meth:`RunReplay.run`).
    Extra outputs: ``drain_seep`` [cm d-1] (DRAIN's bottom seepage, included in ``drainage``) and
    ``cut_nodes`` [day, node] (the uptake removed by the ``h_min`` cap, per node).
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    from agrijax.processes.soil_water import richards
    from agrijax.processes.soil_water.hydraulics import H_CLAMP_RZWQM, h_of_theta, theta_of_h

    grid = rep.grid
    n = grid.n_node
    config = richards.RichardsConfig(n_sub=n_sub, n_iter=n_iter, **cfg)
    soil = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,)), rep.soil.at_nodes())
    pori = jnp.asarray(pori_of(rep))
    h_min = jnp.asarray(H_CLAMP_RZWQM)
    pond0 = jnp.zeros(())
    alphas = richards.day_alphas(n_sub, config, jnp.float64)
    th_min = theta_of_h(jnp.full(n, H_CLAMP_RZWQM), soil)

    def day(w: Any, sup_h: Any, eva_h: Any, upt: Any) -> tuple[Any, list[Any]]:
        t = richards.substep_edges(sup_h, n_sub, config.rain_fraction)
        dts = t[1:] - t[:-1]
        sup = richards._interval_means(sup_h, t)
        eva = richards._interval_means(eva_h, t)
        rate = upt / (richards.HOURS_PER_DAY * grid.tl)

        def body(carry: Any, xs: Any) -> tuple[Any, Any]:
            h, th = carry
            s_k, e_k, a_k, dt = xs
            live = dt > 0.0
            dt_ = jnp.where(live, dt, 1.0)
            r = richards.richards_step(
                h, th, pond0, soil, grid, s_k, e_k, rate, dt_, a_k, h_min, pond0, config
            )
            avail = th - th_min - config.sink_cutoff
            s_av = jnp.where(avail > 0.0, avail, 0.0) / dt_
            cut_nodes = grid.tl * jnp.where(rate > s_av, rate - s_av, 0.0) * dt_
            th2, h2, seep = r.theta, r.h, jnp.zeros(())
            if drain:
                th2, seep = drain_cascade(r.theta, grid.tl, pori)
                h2 = jnp.where(th2 < r.theta, h_of_theta(th2, soil), r.h)

            def z(x: Any) -> Any:
                return jnp.where(live, x, 0.0)

            out = (
                r.drainage + seep,
                seep,
                r.runoff,
                r.evaporation_deficit,
                r.uptake_cut,
                r.balance_error,
                r.n_clamp,
            )
            return (jnp.where(live, h2, h), jnp.where(live, th2, th)), (*(z(x) for x in out), z(cut_nodes))

        (h, th), o = lax.scan(body, (w.h, w.theta), (sup, eva, alphas, dts))
        return w.replace(h=h, theta=th), [jnp.sum(x, axis=0) for x in o]

    reset, start = rep.restart_profiles(mode)
    w0 = richards.SoilWater.from_theta(jnp.asarray(rep.theta_init), rep.soil)

    def scan_body(w: Any, f: Any) -> tuple[Any, Any]:
        sup, eva, upt, flag, th = f
        fresh = richards.SoilWater.from_theta(th, rep.soil)
        w = jax.tree_util.tree_map(lambda a, b: jnp.where(flag, a, b), fresh, w)
        w2, tot = day(w, sup, eva, upt)
        return w2, (w2.theta, jnp.sum(w2.theta * grid.tl), *tot)

    xs = tuple(jnp.asarray(x) for x in (rep.supply, rep.evap_hourly, rep.uptake, reset, start))
    richards.surface_fluxes = surface_fluxes_flux_mode if flux_mode else _SURFACE_FLUXES
    try:
        _, o = jax.jit(lambda w, f: lax.scan(scan_body, w, f))(w0, xs)
    finally:
        richards.surface_fluxes = _SURFACE_FLUXES
    keys = ("theta", "storage", "drainage", "drain_seep", "runoff", "evaporation_deficit", "uptake_cut")
    keys += ("balance_error", "n_clamp", "cut_nodes")
    return {k: np.asarray(v) for k, v in zip(keys, o, strict=True)}


# ------------------------------------------------------------------------------ CA-TPA dumps


def _yyyyddd(d: np.ndarray) -> np.ndarray:
    """``YYYYDDD`` integers (the dump date hook) as ``datetime64[D]``."""
    year = (d // 1000 - 1970).astype("datetime64[Y]").astype("datetime64[D]")
    return year + (d % 1000 - 1).astype("timedelta64[D]")


def catpa_physcl(data_dir: Path) -> dict[str, np.ndarray]:
    """Day-by-day ``PHYSCL`` entry arrays of the instrumented 2015-2023 run (``dumps/_runs/catpa2015_2023``).

    ``SOILHP[day, 13, MAXHOR]`` and ``HORTHK[day, MAXHOR]`` are on RZWQM2's *internal* horizons:
    with tillage in the run, start-up (``TILADJ``, ``Rzmain.for`` 6398-6441) splits the first
    ``rzwqm.dat`` horizon at the tillage depth, so CA-TPA has ``NHOR = 6`` internal horizons for
    the 5 of the file (the first two identical at start-up).
    """
    return physcl_table(data_dir / CATPA_DUMPS / "physcl_entry.npz")


def physcl_table(path: Path) -> dict[str, np.ndarray]:
    """A ``PHYSCL`` entry table (:func:`agrijax.port.dumps.save_table`) as the arrays used here."""
    z = np.load(path)
    return {
        "date": _yyyyddd(np.asarray(z["date"])),
        "SOILHP": np.asarray(z["v.SOILHP"], float),
        "HORTHK": np.asarray(z["v.HORTHK"], float),
        "NHOR": np.asarray(z["v.NHOR"]).astype(int),
        "THETA": np.asarray(z["v.THETA"], float),
        "BD": np.asarray(z["v.BD"], float),
        "HREVAP": np.asarray(z["v.HREVAP"], float),
    }


def internal_node_horizon(ph: dict[str, np.ndarray], node_depths: np.ndarray) -> tuple[int, np.ndarray]:
    """``(NHOR, node -> internal horizon)`` from the dumped ``HORTHK`` (constant over the run, asserted).

    ``node_depths`` are the node (layer-bottom) depths; the map is the one of the ``rzwqm.dat``
    horizons in :class:`RunReplay` (``searchsorted(..., side="left")``).
    """
    nhor = int(ph["NHOR"][0])
    assert (ph["NHOR"] == nhor).all(), "NHOR changes during the run"
    bottoms = ph["HORTHK"][:, :nhor]  # lower depth of each internal horizon [cm] (despite the name)
    assert (bottoms == bottoms[0]).all(), "HORTHK changes during the run"
    return nhor, np.searchsorted(bottoms[0], node_depths, side="left")


def soil_days_from_soilhp(soilhp: np.ndarray, n_horizon: int, node_horizon: np.ndarray, tilled: bool) -> Any:
    """Per-day parameters from ``SOILHP[day, 13, horizon]``; ``tilled`` pairs them with day 0 (start-up).

    ``n_horizon``/``node_horizon`` are the internal horizons (:func:`internal_node_horizon`).
    ``tilled=True`` gives :class:`TilledSoilHydraulicParams` (current = the day's ``SOILHP``,
    original = the first day's, which is the start-up ``TRHYDP`` before any tillage), the curve
    RZWQM2 evaluates after tillage; ``False`` the day's single-segment curve.
    """
    import jax
    import jax.numpy as jnp

    from agrijax.processes.soil_water.hydraulics import SoilHydraulicParams, TilledSoilHydraulicParams

    hp = np.transpose(soilhp[:, :, :n_horizon], (0, 2, 1))  # [day, horizon, 13]
    cur = SoilHydraulicParams.from_rzwqm_records(hp[..., 0:6], hp[..., 6:13], node_horizon=node_horizon)
    if not tilled:
        return cur
    o = SoilHydraulicParams.from_rzwqm_records(hp[0, :, 0:6], hp[0, :, 6:13], node_horizon=node_horizon)
    n = soilhp.shape[0]
    orig = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (n, *a.shape)), o)
    return TilledSoilHydraulicParams(current=cur, original=orig)


# ------------------------------------------------------------------------------ script


def main(argv: Sequence[str]) -> int:
    import jax

    jax.config.update("jax_enable_x64", True)
    data_dir = Path(os.environ.get("AGRI_JAX_DATA", str(Path.home() / "agri_jax_data")))
    out = data_dir / OUT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    sites = list(argv) or list(SITES)
    flags = []
    for s in SITES:
        dat = data_dir / BATCH / s / "Scenario" / "rzwqm.dat"
        if dat.is_file():
            flags.append({"site": s, **comparability(dat)})
    pd.DataFrame(flags).to_csv(out / "comparability.csv", index=False)
    print(pd.DataFrame(flags).to_string())
    run_root = Path(os.environ.get("AGRI_JAX_H14_RUNS", str(data_dir / "rzwqm_runs" / "h1_4")))
    tables, checks = [], []
    for s in sites:
        f = next(x for x in flags if x["site"] == s)
        if not f["comparable"]:
            continue
        if s == "CA-TPA":
            rep = catpa_replay(data_dir)
        else:
            rec = run_scenario(s, data_dir, run_root, run_root / "_stage")
            if not rec["ok"]:
                print(s, "RUN FAILED", rec["error"])
                checks.append({"site": s, "error": rec["error"]})
                continue
            rep = RunReplay(s, Path(rec["out"]), Path(rec["source"]))
        checks.append({"site": s, **rep.input_checks(), **rep.fingerprint()})
        print(s, checks[-1], flush=True)
        for mode in ("restart", "free"):
            for n_sub, n_iter in CONFIGS:
                t0 = time.perf_counter()
                r = rep.run(mode=mode, n_sub=n_sub, n_iter=n_iter)
                t = rep.year_table(r, mode=mode, n_sub=n_sub, n_iter=n_iter)
                t["wall_s"] = time.perf_counter() - t0
                tables.append(t)
                print(
                    t[
                        ["site", "mode", "n_sub", "n_iter", "year", "storage_rmse_cm", "theta_rmse"]
                    ].to_string()
                )
                pd.concat(tables).to_csv(out / "years.csv", index=False, float_format="%.6g")
        pd.DataFrame(checks).to_csv(out / "input_checks.csv", index=False, float_format="%.6g")
    print("written", out, _dt.datetime.now())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
