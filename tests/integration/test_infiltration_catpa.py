"""Plan 19 A5/A6 levels 2-3: the Green-Ampt event day against RZWQM2 on CA-TPA 2015.

Reference run: ``<data>/catpa/ref_2015`` (RZWQM2 ``main_ryzen5_avx512``, ``CA-TPA.ana``,
``LAYER.PLT``) and the scenario's ``CA-TPA.BRK`` breakpoint rain. The day is
:func:`~agrijax.processes.soil_water.day.soil_water_day_kernel`: redistribution, the day's storm
through our Green-Ampt event, redistribution. RZWQM's own inputs other than the rain are taken
from the reference run as in M1 (evaporation ``.ana`` column 6, uptake ``LAYER.PLT``).

Snow is not in the event kernel yet (L-snow; CA-TPA runs RZWQM2 with PRMS snow on, and snow is
added in the M3 assembly): on days with a snowpack or snowmelt (``.ana`` columns 92 and
105) the rain goes into the snowpack in RZWQM2, so those days get no event and their
infiltration (``.ana`` column 5) is replayed as a surface flux, as in M1. Every other storm day
is a Green-Ampt event.

Level 3 (resynchronised = ``resync``): each day starts from RZWQM's end-of-previous-day profile
(``LAYER.PLT``), so infiltration and runoff are compared day by day without drift.
Free-running (``free``): the year from ``rzinit.dat``; storage against ``.ana`` column 2.
The M1 replay on the same days separates infiltration from redistribution error.

Level 2 (per-event ``EVNTRO``/``INFIL`` entry/exit dumps, each storm from its dumped entry state)
is in ``test_infiltration_dumps.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from agrijax.core import Day, Phase, State, run
from agrijax.core.process import CHECK_ENV
from agrijax.io.rzwqm import read_ana, read_brk, read_rzwqm_dat
from agrijax.io.rzwqm.layers import read_layer_output
from agrijax.processes.soil_water.day import (
    SOIL_WATER_LEDGER_OUTFLOWS,
    DayConfig,
    SoilWaterDayForcing,
    SoilWaterDayParams,
    soil_water_day,
    soil_water_day_kernel,
    soil_water_ledger,
    soil_water_ledger_init,
)
from agrijax.processes.soil_water.hydraulics import SoilHydraulicParams
from agrijax.processes.soil_water.infiltration import (
    GreenAmptConfig,
    GreenAmptParams,
    StormForcing,
)
from agrijax.processes.soil_water.richards import (
    RichardsConfig,
    RichardsGrid,
    RichardsParams,
    SoilWater,
    richards_day,
)

pytestmark = [
    pytest.mark.allow_skip(reason="needs the private CA-TPA 2015 reference run under the data dir"),
    pytest.mark.skipif(not jax.config.read("jax_enable_x64"), reason="level-3 comparison in float64"),
]

REF = Path("catpa/ref_2015")
SCENARIO = Path("narval_mirror/RZWQM_sw_batch/CA-TPA/Scenario")
THETA_INIT = 0.1917
INCH_CM = 2.54
AEF_CATPA = 0.9  # rzwqm.dat Richards control record, item 4


#: RZWQM2 skips a storm whose total in the breakpoint file is below 0.01 in (STMINP)
MIN_STORM_IN = 0.01


def storm_forcing_from_breakpoints(days: np.ndarray, events: Any, breakpoints: Any) -> StormForcing:
    """Per-day storm segments from ``read_brk`` tables: intervals from breakpoint differences.

    Breakpoints are (clock time [min], cumulative depth [in]); an interval crossing midnight is
    split in proportion to time and continues the next day at ``ts0 = 0`` (RZWQM2 ``CHSPAN``).
    Storms below :data:`MIN_STORM_IN` are dropped, as RZWQM2 ``STMINP`` does.
    """
    index = {d: k for k, d in enumerate(days)}
    segs: dict[int, list[tuple[float, float]]] = {}
    start: dict[int, float] = {}
    owner: dict[int, Any] = {}
    for ev_id, ev in events.iterrows():
        if float(ev["depth_in"]) < MIN_STORM_IN:
            continue
        b = breakpoints[breakpoints["event"] == ev_id]
        t = b["time_min"].to_numpy(dtype=float) / 60.0
        c = b["cum_depth_in"].to_numpy(dtype=float) * INCH_CM
        day0 = np.datetime64(ev["date"], "D")
        for k in range(len(t) - 1):
            t0, t1, dd = t[k], t[k + 1], c[k + 1] - c[k]
            while t1 > t0:
                off = int(np.floor(t0 / 24.0 + 1e-12))
                cut = min(t1, 24.0 * (off + 1))
                frac = (cut - t0) / (t1 - t0)
                i = index.get(day0 + np.timedelta64(off, "D"))
                if i is not None:
                    if owner.setdefault(i, ev_id) != ev_id:
                        raise ValueError(f"two storms on {days[i]}: RZWQM2 runs one event per day")
                    start.setdefault(i, t0 - 24.0 * off)
                    segs.setdefault(i, []).append((cut - t0, dd * frac))
                dd *= 1.0 - frac
                t0 = cut
    width = max([len(v) for v in segs.values()] + [1])
    ts0 = np.full(len(days), 24.0)
    dur = np.zeros((len(days), width))
    dep = np.zeros((len(days), width))
    for i, v in segs.items():
        ts0[i] = start[i]
        dur[i, : len(v)] = [a for a, _ in v]
        dep[i, : len(v)] = [b for _, b in v]
    return StormForcing(ts0=jnp.asarray(ts0), duration=jnp.asarray(dur), depth=jnp.asarray(dep))


class Catpa2015Events:
    """CA-TPA 2015 inputs for the event day."""

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
        self.precipitation = col(3)[1:]
        self.infiltration = col(5)[1:]
        self.evaporation = col(6)[1:]
        self.deep_seepage = col(10)[1:]
        self.runoff = col(12)[1:]
        snow_depth, melt = col(92), col(105)
        self.days = np.asarray(ana.time.values[1:], dtype="datetime64[D]")
        lay = read_layer_output(ref / "LAYER.PLT", start="2015-01-01")
        self.theta = np.asarray(lay["soil_water_content"].values)
        self.uptake = np.asarray(lay["plant_water_uptake"].values)
        brk = read_brk(sc / "CA-TPA.BRK")
        storms = storm_forcing_from_breakpoints(self.days, brk.events, brk.breakpoints)
        rain = np.asarray(storms.depth).sum(axis=1)
        #: snow days: a snowpack at the start or end of the day, or snowmelt
        self.snow = (snow_depth[1:] > 0.0) | (snow_depth[:-1] > 0.0) | (melt[1:] > 0.0)
        self.event = (rain > 0.0) & ~self.snow
        keep = jnp.asarray(self.event)[:, None]
        self.storm = StormForcing(
            ts0=storms.ts0,
            duration=jnp.where(keep, storms.duration, 0.0),
            depth=jnp.where(keep, storms.depth, 0.0),
        )
        self.rain = rain
        # replayed surface flux on the days without an event (snowmelt infiltration), spread over the day
        replay = np.where(self.event, 0.0, self.infiltration)
        self.supply = np.repeat(replay[:, None] / 24.0, 24, axis=1)
        self.evap_hourly = np.repeat(self.evaporation[:, None] / 24.0, 24, axis=1)
        self.ga_config = GreenAmptConfig.for_grid(np.asarray(self.grid.tl))

    def params(self, day: DayConfig, **cfg: Any) -> SoilWaterDayParams:
        rp = RichardsParams(soil=self.soil, grid=self.grid, config=RichardsConfig(**cfg))
        gp = GreenAmptParams(aef=jnp.asarray(AEF_CATPA), config=self.ga_config)
        return SoilWaterDayParams(richards=rp, infiltration=gp, config=day)

    def forcing(self) -> tuple[Any, ...]:
        return (
            jnp.asarray(self.supply),
            jnp.asarray(self.evap_hourly),
            jnp.asarray(self.uptake),
            self.storm,
        )

    def run(self, day: DayConfig, *, resync: bool = False, **cfg: Any) -> dict[str, np.ndarray]:
        """The year with the event day; ``resync`` starts each day from RZWQM's previous profile."""
        params = self.params(day, **cfg)
        w0 = SoilWater.from_theta(jnp.full(self.grid.n_node, THETA_INIT), self.soil)
        start = np.concatenate([np.full((1, self.grid.n_node), THETA_INIT), self.theta[:-1]])

        def body(w: SoilWater, f: tuple[Any, ...]) -> tuple[SoilWater, Any]:
            sup, eva, upt, storm, th_start = f
            if resync:
                w = SoilWater.from_theta(th_start, self.soil)
            w2, ev, shift = soil_water_day_kernel(w, params, sup, eva, upt, storm)
            return w2, (w2.theta, w2.storage(self.grid), w2.flux, ev.front_depth, shift)

        xs = (*self.forcing(), jnp.asarray(start))
        _, (theta, storage, flux, front, shift) = jax.jit(lambda w, f: lax.scan(body, w, f))(w0, xs)
        out = {k: np.asarray(v) for k, v in flux.items()}
        out |= {"theta": np.asarray(theta), "storage": np.asarray(storage)}
        out |= {"front_depth": np.asarray(front), "event_shift": np.asarray(shift)}
        return out

    def run_replay(self, **cfg: Any) -> dict[str, np.ndarray]:
        """M1 on the same year (all infiltration replayed, spread uniformly), for the attribution."""
        params = RichardsParams(soil=self.soil, grid=self.grid, config=RichardsConfig(**cfg))
        w0 = SoilWater.from_theta(jnp.full(self.grid.n_node, THETA_INIT), self.soil)
        sup = np.repeat(self.infiltration[:, None] / 24.0, 24, axis=1)

        def body(w: SoilWater, f: tuple[Any, ...]) -> tuple[SoilWater, Any]:
            w2 = richards_day(w, params, f[0], f[1], f[2])
            return w2, w2.storage(self.grid)

        xs = (jnp.asarray(sup), jnp.asarray(self.evap_hourly), jnp.asarray(self.uptake))
        _, storage = jax.jit(lambda w, f: lax.scan(body, w, f))(w0, xs)
        return {"storage": np.asarray(storage)}


@pytest.fixture(scope="module")
def catpa(data_dir: Path) -> Catpa2015Events:
    return Catpa2015Events(data_dir)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def test_storm_forcing_matches_the_breakpoint_file(catpa: Catpa2015Events) -> None:
    """122 storms in 2015 (3 below 0.01 in dropped), all at 00:00, <= 2 h; totals equal ``.ana`` rain."""
    rain = catpa.rain
    assert (rain > 0).sum() == 122
    np.testing.assert_array_equal(rain > 0, catpa.precipitation > 0)
    ts0 = np.asarray(catpa.storm.ts0)[rain > 0]
    assert np.all(ts0 == 0.0)
    dur = np.asarray(catpa.storm.duration).sum(axis=1)[catpa.event]
    assert np.all(dur <= 2.0 + 1e-12)
    # .ana precipitation equals the .BRK storm (0.001 in in the file, 1e-4 cm printed)
    m = catpa.event
    assert np.max(np.abs(catpa.precipitation - rain)) < 1e-4
    # RZWQM2 lets every rain-only storm infiltrate at CA-TPA 2015: infiltration = rain, no runoff
    assert catpa.event.sum() == 87
    np.testing.assert_allclose(catpa.infiltration[m], catpa.precipitation[m], atol=1e-6)
    assert np.all(catpa.runoff[m] == 0.0)


@pytest.mark.parametrize("n_sub", [24, 96])
def test_level3_resync_daily_infiltration_and_runoff(catpa: Catpa2015Events, n_sub: int) -> None:
    """Each day from RZWQM's profile: event infiltration equals RZWQM's, no runoff, ledger closed."""
    day = DayConfig(n_pre=0, n_post=n_sub)
    r = catpa.run(day, resync=True, n_sub=n_sub, n_iter=8 if n_sub == 96 else 3)
    m = catpa.event
    # measured: every CA-TPA 2015 storm infiltrates completely (Green-Ampt capacity >= rain rate)
    np.testing.assert_allclose(r["event_infiltration"][m], catpa.rain[m], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(r["event_infiltration"][m], catpa.infiltration[m], atol=1e-4)
    assert np.max(np.abs(r["event_runoff"])) < 1e-12
    assert np.all(r["seepage"] == 0.0)
    assert np.all(r["event_shift"] == 0.0)
    assert np.all(r["event_infiltration"][~m] == 0.0)
    assert np.max(np.abs(r["balance_error"])) < (1e-10 if n_sub == 96 else 0.06)
    assert r["n_clamp"].sum() == 0.0


def test_level3_free_running_storage(catpa: Catpa2015Events) -> None:
    """The year free-running with our events: storage against ``.ana`` and against the M1 replay."""
    ga = catpa.run(DayConfig(n_pre=0, n_post=96), n_sub=96, n_iter=8)
    m1 = catpa.run_replay(n_sub=96, n_iter=8)
    rmse_ga = _rmse(ga["storage"], catpa.storage)
    rmse_m1 = _rmse(m1["storage"], catpa.storage)
    theta_rmse = _rmse(ga["theta"], catpa.theta)
    # the numbers are recorded (see the report in the validation matrix); the bounds are the M1 thresholds
    assert rmse_ga < 0.05, (rmse_ga, rmse_m1)
    assert theta_rmse < 0.01
    assert ga["drainage"].sum() == pytest.approx(catpa.deep_seepage.sum(), rel=5e-3)
    assert ga["infiltration"].sum() == pytest.approx(catpa.infiltration.sum(), abs=0.03)
    assert np.max(np.abs(ga["balance_error"])) < 1e-10
    assert ga["n_clamp"].sum() == 0.0


class _LedgerState(State):
    soil_water: SoilWater
    ledger: dict


def test_ledger_books_every_sink_channel_catpa_2015(
    catpa: Catpa2015Events, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H1 item 6: the free-running 2015 day through the runtime with the channel ledger as its last entry.

    Every day closes to 1e-10 cm under ``AGRI_JAX_CHECK=1`` (96 sub-steps, 8 iterations, the
    configuration whose ``balance_error`` is < 1e-10); each channel of ``SINK_LEDGER_OUTFLOWS`` is
    booked (only ``uptake`` is non-zero at CA-TPA, the others exactly 0); the soil water is
    bit-identical to the kernel scan without a ledger.
    """
    day = DayConfig(n_pre=0, n_post=96)
    params = catpa.params(day, n_sub=96, n_iter=8)
    sup, eva, upt, storm = catpa.forcing()
    forcing = SoilWaterDayForcing(storm=storm, supply=sup, evaporation=eva, uptake=upt)
    w0 = SoilWater.from_theta(jnp.full(catpa.grid.n_node, THETA_INIT), catpa.soil)
    s0 = _LedgerState(soil_water=w0, ledger={"water": soil_water_ledger_init(w0, catpa.grid)})
    model = Day(
        ref="rzwqm2-4.6",
        phases=(Phase("physcl", ("soil_water.day",)), Phase("ledger", ("ledger.close",))),
    ).compile(
        {"soil_water.day": soil_water_day, "ledger.close": soil_water_ledger()},
        outputs=lambda s, p, f: (s.ledger["water"], s.soil_water.theta, s.soil_water.flux),
    )
    monkeypatch.setenv(CHECK_ENV, "1")
    led, theta, flux = jax.jit(lambda f_, s_: run(model, params, f_, s_))(forcing, s0)
    res = np.asarray(led.residual)
    last = jax.tree_util.tree_map(lambda x: x[-1], led)
    tout, tin = last.total_out(), last.total_in()
    print(
        "CA-TPA 2015 ledger: max |daily residual| =", np.max(np.abs(res)), "cm; whole-year closure =",
        float(last.closure()), "cm; in =", {k: float(v) for k, v in tin.items()},
        "out =", {k: float(v) for k, v in tout.items()},
    )  # fmt: skip
    assert res.shape == (len(catpa.days),)
    assert np.max(np.abs(res)) < 1e-10
    np.testing.assert_allclose(res, np.asarray(flux.balance_error), rtol=0, atol=1e-12)
    assert abs(float(last.closure())) < 1e-10
    for name in SOIL_WATER_LEDGER_OUTFLOWS:
        daily = np.asarray(getattr(flux, name), np.float64)
        assert float(tout[name]) == pytest.approx(float(np.sum(daily)), rel=1e-12, abs=1e-14), name
    for name in ("tile", "lateral", "subirrigation", "macropore_to_drain"):
        assert float(tout[name]) == 0.0, name
    assert float(tout["uptake"]) > 0.0
    assert float(tin["rain"]) == pytest.approx(float(catpa.rain[catpa.event].sum()), rel=1e-12)
    assert float(tin["supply"]) == pytest.approx(float(catpa.supply.sum()), rel=1e-12)
    # the ledger adds an entry and reads the day; it does not change the soil water (bit for bit)
    ref = catpa.run(day, n_sub=96, n_iter=8)
    np.testing.assert_array_equal(np.asarray(theta), ref["theta"])
    for name in ("infiltration", "drainage", "runoff", "uptake", "evaporation", "balance_error"):
        np.testing.assert_array_equal(np.asarray(getattr(flux, name)), ref[name], err_msg=name)
