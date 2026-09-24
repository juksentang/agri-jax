"""OrganQueue + EventTable inside ``lax.scan`` on the real CA-TPA 2015-2023 forcing (3287 days).

Model: :mod:`agri_jax.models.tobacco_demo` (leaves by rank, topping and primings from the event
table). Every claim is checked against something that does not go through the queue or the scan:

* the model vs a plain NumPy/Python day loop of the same rules (lists of leaves, no masks, no JAX);
* the model vs a closed form: with at most one leaf per day and the leaf clock restarting at the
  overshoot, leaf ``k`` (0-based) appears when the season's cumulative thermal time reaches
  ``(k + 1) * phyllochron`` and its age on day ``t`` is ``TT(t) - (k + 1) * phyllochron``, so each
  priming harvests ``sum_k M(TT(t) - (k + 1) * phyllochron)`` over its ranks;
* ``n_cohort = 1`` vs the lumped single-pool closed form (CERES-style degenerate queue);
* mass conservation of the model's own bookkeeping (alive + harvested == added) on every day;
* ``run_batch`` over 64 parameter sets vs a Python loop of ``run``;
* ``jax.grad`` of the total harvest wrt ``phyllochron`` and ``growth_rate`` vs central differences;
* ``EventTable.from_csv`` on the real CA-TPA ``events.csv``: sow / harvest / fertilizer days seen
  inside a ``lax.scan`` vs the csv rows counted with the stdlib ``csv`` module; the tobacco model on
  that real calendar vs the NumPy loop.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax import lax

from agri_jax.core import EventTable, run, run_batch
from agri_jax.io.rzwqm import read_met
from agri_jax.models.tobacco_demo import (
    DEFAULT_PRIMINGS,
    build_forcing,
    default_params,
    initial_state,
    synthetic_events,
    tobacco_model,
)

START, END = "2015-01-01", "2023-12-31"
N_DAYS = 3287
N_COHORT = 30
N_BATCH = 64
EVENTS_CSV = Path("catpa/events.csv")

X64 = bool(jax.config.jax_enable_x64)
pytestmark = pytest.mark.skipif(not X64, reason="scan-level checks are defined in float64")
RTOL = 1e-10


# ------------------------------------------------------------------------------ references


def _m(a: float, p: dict[str, float]) -> float:
    """Leaf mass of age ``a``: the rescaled logistic of the module docstring, in plain ``math``."""
    s = lambda x: 1.0 / (1.0 + math.exp(-x))  # noqa: E731
    s0 = s(-p["growth_rate"] * p["age_half"])
    return p["leaf_mass_max"] * (s(p["growth_rate"] * (a - p["age_half"])) - s0) / (1.0 - s0)


def numpy_reference(
    tmin: np.ndarray, tmax: np.ndarray, ev: dict[str, np.ndarray], p: dict[str, float], n_cohort: int
) -> dict[str, np.ndarray]:
    """The demo's rules as a plain day loop over a Python list of leaves (no JAX, no masks)."""
    n = len(tmin)
    out = {k: np.zeros(n) for k in ("harvested", "harvest_today", "lai", "n_active", "mass_added")}
    leaves: list[dict[str, float]] = []  # rank = list index
    cap, active, tt, harvested, added = n_cohort, False, 0.0, 0.0, 0.0
    for t in range(n):
        if ev["sow"][t]:
            leaves, cap, active, tt = [], n_cohort, True, 0.0
        dtt = max(0.5 * (tmin[t] + tmax[t]) - p["tbase"], 0.0) if active else 0.0
        for leaf in leaves:
            if leaf["alive"]:
                dm = _m(leaf["age"] + dtt, p) - _m(leaf["age"], p)
                leaf["mass"] += dm
                leaf["area"] += p["sla"] * dm
                leaf["age"] += dtt
                added += dm
        tt += dtt
        if active and tt >= p["phyllochron"] and len(leaves) < cap:
            over = tt - p["phyllochron"]
            m0 = _m(over, p)
            leaves.append({"age": over, "mass": m0, "area": p["sla"] * m0, "alive": True})
            added += m0
            tt = over
        if ev["topping"][t]:
            cap = min(cap, len(leaves))
        today = 0.0
        lo, hi = int(ev["priming_lo"][t]), int(ev["priming_hi"][t])
        if ev["harvest"][t]:
            lo2, hi2 = 0, n_cohort
        else:
            lo2, hi2 = -1, -1
        for k, leaf in enumerate(leaves):
            if leaf["alive"] and (lo <= k < hi or lo2 <= k < hi2):
                today += leaf["mass"]
                leaf["alive"] = False
                leaf["mass"] = leaf["area"] = 0.0
        if ev["harvest"][t]:
            active = False
        harvested += today
        out["harvested"][t] = harvested
        out["harvest_today"][t] = today
        out["mass_added"][t] = added
        out["n_active"][t] = len(leaves)
        out["lai"][t] = sum(lf["area"] for lf in leaves if lf["alive"]) * p["density"] * 1e-4
    return out


def closed_form_primings(
    tmin: np.ndarray, tmax: np.ndarray, sow_days: list[int], p: dict[str, float], n_cohort: int
) -> list[float]:
    """Mass of every priming / harvest of the synthetic calendar from cumulative thermal time only."""
    phyl = p["phyllochron"]
    dtt_all = np.maximum(0.5 * (tmin + tmax) - p["tbase"], 0.0)
    out: list[float] = []
    for s in sow_days:
        tt = np.cumsum(dtt_all[s:])  # TT(t) on day s + i, sow day included
        n_top = min(int(np.floor(tt[60] / phyl)), n_cohort)  # leaves present at topping (sow + 60)
        taken: set[int] = set()
        for d, (lo, hi) in (*DEFAULT_PRIMINGS, (100, (0, n_cohort))):
            ranks = [k for k in range(max(lo, 0), min(hi, n_top)) if k not in taken]
            taken.update(ranks)
            out.append(sum(_m(tt[d] - (k + 1) * phyl, p) for k in ranks))
    return out


def _events_np(ev: EventTable) -> dict[str, np.ndarray]:
    return {k: np.asarray(getattr(ev, k)) for k in ("sow", "harvest", "topping", "priming_lo", "priming_hi")}


def _pdict(params: Any) -> dict[str, float]:
    return {k: float(v) for k, v in params.items()}


# ------------------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def met(catpa_scenario: Path) -> pd.DataFrame:
    return read_met(catpa_scenario / "CA-TPA.MET", prepare=True)


@pytest.fixture(scope="module")
def tob(met: pd.DataFrame) -> dict[str, Any]:
    dates = pd.date_range(START, END, freq="D")
    events = synthetic_events(dates)
    forcing = build_forcing(met, events, START, END)
    params = default_params()
    model = tobacco_model()
    state0 = initial_state(N_COHORT)
    single = jax.jit(lambda p: run(model, p, forcing, state0))
    outs = jax.block_until_ready(single(params))
    tmin, tmax = np.asarray(forcing.tmin), np.asarray(forcing.tmax)
    return dict(
        dates=dates,
        events=events,
        forcing=forcing,
        params=params,
        model=model,
        state0=state0,
        single=single,
        outs=outs,
        tmin=tmin,
        tmax=tmax,
    )


# ------------------------------------------------------------------------------ tobacco demo


def test_forcing_is_real_catpa(tob: dict[str, Any], met: pd.DataFrame) -> None:
    f = tob["forcing"]
    assert f.n_days == N_DAYS
    np.testing.assert_array_equal(tob["tmax"], met.loc[START:END, "tmax"].to_numpy())
    ev = _events_np(tob["events"])
    assert ev["sow"].sum() == 9 and ev["harvest"].sum() == 9 and ev["topping"].sum() == 9
    assert (ev["priming_lo"] >= 0).sum() == 27


def test_harvest_matches_numpy_loop(tob: dict[str, Any]) -> None:
    """Every daily output of the scan equals the plain-Python day loop of the same rules."""
    p = _pdict(tob["params"])
    ref = numpy_reference(tob["tmin"], tob["tmax"], _events_np(tob["events"]), p, N_COHORT)
    outs = tob["outs"]
    for key in ("harvested", "harvest_today", "mass_added", "lai"):
        np.testing.assert_allclose(np.asarray(outs[key])[:, 0], ref[key], rtol=RTOL, atol=1e-10, err_msg=key)
    np.testing.assert_array_equal(np.asarray(outs["n_active"])[:, 0], ref["n_active"])
    # the season is non-trivial: several primings harvest a positive mass, queue grows past 18 ranks
    assert float(outs["harvested"][-1, 0]) > 100.0
    assert int(np.asarray(outs["n_active"]).max()) >= 18


def test_harvest_matches_closed_form(tob: dict[str, Any]) -> None:
    """Each priming / final harvest equals the closed form in cumulative thermal time."""
    p = _pdict(tob["params"])
    ev = _events_np(tob["events"])
    sow_days = list(np.flatnonzero(ev["sow"]))
    expected = closed_form_primings(tob["tmin"], tob["tmax"], sow_days, p, N_COHORT)
    today = np.asarray(tob["outs"]["harvest_today"])[:, 0]
    harvest_days = np.flatnonzero((ev["priming_lo"] >= 0) | ev["harvest"])
    assert len(harvest_days) == len(expected) == 36
    np.testing.assert_allclose(today[harvest_days], expected, rtol=1e-10, atol=1e-10)
    other = np.setdiff1d(np.arange(N_DAYS), harvest_days)
    assert (today[other] == 0.0).all()


def test_mass_conservation_every_day(tob: dict[str, Any]) -> None:
    outs = tob["outs"]
    alive = np.asarray(outs["alive_mass"])[:, 0]
    added = np.asarray(outs["mass_added"])[:, 0]
    harvested = np.asarray(outs["harvested"])[:, 0]
    np.testing.assert_allclose(alive + harvested, added, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(np.cumsum(np.asarray(outs["harvest_today"])[:, 0]), harvested, rtol=1e-12)


def test_n_cohort_1_is_lumped(tob: dict[str, Any]) -> None:
    """One slot: a single pool that starts when TT reaches one phyllochron and is taken by the first
    priming (ranks 0-5); LAI = sla * M(age) * density * 1e-4 while it lives."""
    p = _pdict(tob["params"])
    state1 = initial_state(1)
    outs = jax.jit(lambda q: run(tob["model"], q, tob["forcing"], state1))(tob["params"])
    ev = _events_np(tob["events"])
    dtt = np.maximum(0.5 * (tob["tmin"] + tob["tmax"]) - p["tbase"], 0.0)
    exp_today = np.zeros(N_DAYS)
    exp_lai = np.zeros(N_DAYS)
    for s in np.flatnonzero(ev["sow"]):
        tt = np.cumsum(dtt[s : s + 71])  # sow .. first priming (day 70)
        age_pool = tt - p["phyllochron"]
        alive = age_pool >= 0.0
        exp_lai[s : s + 70] = np.where(
            alive[:70], p["sla"] * np.array([_m(a, p) for a in age_pool[:70]]) * p["density"] * 1e-4, 0.0
        )
        exp_today[s + 70] = _m(age_pool[70], p) if alive[70] else 0.0
    np.testing.assert_allclose(np.asarray(outs["harvest_today"])[:, 0], exp_today, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(outs["lai"])[:, 0], exp_lai, rtol=1e-10, atol=1e-12)
    assert int(np.asarray(outs["n_active"]).max()) == 1


def _batch_params(params: Any, n: int, seed: int = 20150501) -> Any:
    rng = np.random.default_rng(seed)
    phyl = 20.0 + 25.0 * (rng.permutation(n) + rng.random(n)) / n  # > max daily dtt (18 degC d)
    rate = 0.01 + 0.03 * (rng.permutation(n) + rng.random(n)) / n
    batched = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n, *jnp.shape(x))), params)
    return batched.set("phyllochron", jnp.asarray(phyl)).set("growth_rate", jnp.asarray(rate))


def test_run_batch_equals_loop(tob: dict[str, Any]) -> None:
    pb = _batch_params(tob["params"], N_BATCH)
    batched = run_batch(tob["model"], pb, tob["forcing"], tob["state0"])
    assert batched["harvested"].shape == (N_BATCH, N_DAYS, 1)
    final = np.asarray(batched["harvested"][:, -1, 0])
    for i in range(N_BATCH):
        pi = jax.tree_util.tree_map(lambda x, i=i: x[i], pb)
        single = tob["single"](pi)
        for key in ("harvested", "lai", "n_active"):
            np.testing.assert_allclose(
                np.asarray(batched[key][i]), np.asarray(single[key]), rtol=1e-12, atol=1e-12, err_msg=key
            )
    # the batch spans a real range of outcomes, and two members also match the NumPy loop
    assert final.max() > 1.2 * final.min()
    for i in (0, N_BATCH - 1):
        pi = _pdict(jax.tree_util.tree_map(lambda x, i=i: x[i], pb))
        ref = numpy_reference(tob["tmin"], tob["tmax"], _events_np(tob["events"]), pi, N_COHORT)
        np.testing.assert_allclose(final[i], ref["harvested"][-1], rtol=RTOL)


def test_grad_matches_central_differences(tob: dict[str, Any]) -> None:
    model, forcing, state0 = tob["model"], tob["forcing"], tob["state0"]
    base = tob["params"]

    def total(phyl: jax.Array, rate: jax.Array) -> jax.Array:
        p = base.set("phyllochron", phyl).set("growth_rate", rate)
        return run(model, p, forcing, state0)["harvested"][-1, 0]

    x = (base.phyllochron, base.growth_rate)
    grads = jax.jit(jax.grad(total, argnums=(0, 1)))(*x)
    f = jax.jit(total)
    for i, h in ((0, 1e-4), (1, 1e-7)):
        g = float(grads[i])
        assert math.isfinite(g) and g != 0.0
        xp = list(x)
        xm = list(x)
        xp[i] = x[i] + h
        xm[i] = x[i] - h
        cd = (float(f(*xp)) - float(f(*xm))) / (2 * h)
        assert g == pytest.approx(cd, rel=1e-5), (i, g, cd)
    # signs: a longer phyllochron gives fewer, younger leaves; a faster growth rate more mass
    assert float(grads[0]) < 0.0 < float(grads[1])


# ------------------------------------------------------------------------------ real events.csv


@pytest.fixture(scope="module")
def csv_events(data_dir: Path) -> Path:
    p = data_dir / EVENTS_CSV
    if not p.is_file():
        pytest.skip(f"CA-TPA events.csv not found at {p}")
    return p


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return [r for r in csv.DictReader(fh) if START <= r["date"] <= END]


def test_from_csv_counts_seen_inside_scan(csv_events: Path) -> None:
    dates = pd.date_range(START, END, freq="D")
    table = EventTable.from_csv(csv_events, dates)
    rows = _csv_rows(csv_events)

    def body(carry: tuple[jax.Array, ...], ev: EventTable) -> tuple[tuple[jax.Array, ...], jax.Array]:
        n_sow, n_harv, fert, t = carry
        return (n_sow + ev.sow, n_harv + ev.harvest, fert + ev.fert_kg_ha, t + 1), jnp.stack(
            [ev.sow, ev.harvest]
        )

    z = jnp.asarray(0)
    (n_sow, n_harv, fert, n_t), flags = jax.jit(lambda tb: lax.scan(body, (z, z, jnp.asarray(0.0), z), tb))(
        table
    )
    kinds = [r["event"] for r in rows]
    assert int(n_t) == N_DAYS
    assert int(n_sow) == kinds.count("planting") == 7
    assert int(n_harv) == kinds.count("harvest") == 7
    fert_rows = [float(r["value"]) for r in rows if r["event"].startswith("fertilizer")]
    assert float(fert) == pytest.approx(sum(fert_rows), rel=1e-12)
    flags = np.asarray(flags)
    seen_sow = {str(dates[t].date()) for t in np.flatnonzero(flags[:, 0])}
    seen_harv = {str(dates[t].date()) for t in np.flatnonzero(flags[:, 1])}
    assert seen_sow == {r["date"] for r in rows if r["event"] == "planting"}
    assert seen_harv == {r["date"] for r in rows if r["event"] == "harvest"}


def test_tobacco_on_real_calendar(csv_events: Path, met: pd.DataFrame) -> None:
    """The real CA-TPA sow/harvest calendar (no topping/priming rows) drives the demo; every season is
    harvested once, in full, and the scan equals the NumPy loop."""
    dates = pd.date_range(START, END, freq="D")
    table = EventTable.from_csv(csv_events, dates)
    forcing = build_forcing(met, table, START, END)
    params = default_params()
    outs = jax.jit(lambda p: run(tobacco_model(), p, forcing, initial_state(N_COHORT)))(params)
    ev = _events_np(table)
    ref = numpy_reference(np.asarray(forcing.tmin), np.asarray(forcing.tmax), ev, _pdict(params), N_COHORT)
    np.testing.assert_allclose(np.asarray(outs["harvested"])[:, 0], ref["harvested"], rtol=RTOL, atol=1e-10)
    today = np.asarray(outs["harvest_today"])[:, 0]
    assert set(np.flatnonzero(today > 0)) == set(np.flatnonzero(ev["harvest"]))
    alive = np.asarray(outs["alive_mass"])[:, 0]
    assert (alive[np.flatnonzero(ev["harvest"])] == 0.0).all()
