"""Tobacco-like organ-queue demo: leaves by rank on an :class:`~agrijax.core.OrganQueue`, driven by an
:class:`~agrijax.core.EventTable` (sow, topping, primings, final harvest).

A pipeline model, not a science model. It exercises the ``n_cohort`` axis end to end inside ``lax.scan``:
appearance by thermal time, per-leaf growth, topping and
priming from the event table, ``vmap`` over parameters and ``grad`` through a multi-year run.

Rules (one crop slot, ``n_crop = 1``; all quantities per plant):

* thermal time ``dtt = max((tmin + tmax) / 2 - tbase, 0)`` [degC d], accumulated only while the crop
  is active (between a ``sow`` and a ``harvest`` event);
* a sow day empties the queue (a new crop in the reused slot), activates the crop and resets the
  leaf clock ``tt_leaf``;
* every day: alive leaves age by ``dtt`` and grow; the leaf clock gains ``dtt``; when it reaches
  ``phyllochron`` and the queue can accept a leaf (``n_active < cap``), one leaf appears with age
  equal to the overshoot ``tt_leaf - phyllochron`` (it appeared part-way through the day) and the
  clock restarts at that overshoot. At most one leaf per day (``phyllochron`` must exceed the
  largest daily ``dtt``; 18 degC d for CA-TPA 2015-2023 with ``tbase = 10``);
* leaf mass is a logistic of age, rescaled to start at zero:
  ``M(a) = m_max * (s(r (a - a50)) - s(-r a50)) / (1 - s(-r a50))`` with ``s`` the logistic
  function and ``r = growth_rate``; a leaf's daily increment is ``M(a + dtt) - M(a)``, its area
  is ``sla * mass``;
* management, after growth: ``topping`` caps the queue at its current count; a ``priming`` day
  harvests the alive ranks ``[lo, hi)``; a ``harvest`` day harvests every alive leaf and
  deactivates the crop.

Bookkeeping: ``crop.mass_added`` accumulates every gram put into the queue (appearance mass and
growth increments) and ``crop.harvested`` every gram removed by priming or harvest, so
``sum(alive mass) + harvested == mass_added`` holds on every day (no senescence in this demo).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pandas as pd
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.coefficients import Provenance, coef
from agrijax.core.events import EventTable
from agrijax.core.model import Model
from agrijax.core.organs import OrganQueue, age, aggregate, appear, grow, prime, top
from agrijax.core.process import process
from agrijax.core.state import Forcing, Params, State, field

__all__ = [
    "DEFAULT_PRIMINGS",
    "TobaccoCrop",
    "TobaccoForcing",
    "TobaccoParams",
    "TobaccoState",
    "build_forcing",
    "crop_calendar",
    "default_params",
    "initial_state",
    "leaf_appearance_growth",
    "leaf_mass",
    "management",
    "synthetic_events",
    "tobacco_model",
    "tobacco_outputs",
]

#: (days after sowing, (rank_lo, rank_hi)) of the three primings of :func:`synthetic_events`.
DEFAULT_PRIMINGS: tuple[tuple[int, tuple[int, int]], ...] = ((70, (0, 6)), (80, (6, 12)), (90, (12, 18)))


#: default values of the :class:`TobaccoParams` fields (units and meaning in the field declarations).
#: Illustrative: plausible flue-cured tobacco orders of magnitude chosen for the demo, not taken from
#: a reference model or a paper; every one is a calibratable parameter of :class:`TobaccoParams`.
_DEFAULT_VALUES: dict[str, float] = {
    "phyllochron": 22.0,  # degC d
    "tbase": 10.0,  # degC
    "leaf_mass_max": 5.0,  # g plant-1 (one leaf)
    "growth_rate": 0.02,  # (degC d)-1
    "age_half": 250.0,  # degC d
    "sla": 180.0,  # cm2 g-1
    "density": 1.8,  # plants m-2
}
#: scenario defaults of :func:`synthetic_events` and :func:`initial_state` (calendar, not model)
_DEFAULT_SOW_MONTH_DAY: tuple[int, int] = (5, 1)
_DEFAULT_TOPPING_AFTER: int = 60  # days after sowing
_DEFAULT_HARVEST_AFTER: int = 100  # days after sowing
_DEFAULT_N_COHORT: int = 30  # leaf positions of the organ queue


class TobaccoParams(Params):
    """Run-constant parameters of the demo (all scalars; batch with a leading axis)."""

    phyllochron: Array = field(
        dims=(), unit="degC d", description="thermal time between two leaf appearances"
    )
    tbase: Array = field(dims=(), unit="degC", description="base temperature of thermal time")
    leaf_mass_max: Array = field(dims=(), unit="g plant-1", description="asymptotic dry mass of one leaf")
    growth_rate: Array = field(dims=(), unit="(degC d)-1", description="logistic rate of leaf mass in age")
    age_half: Array = field(dims=(), unit="degC d", description="leaf age at the logistic inflection")
    sla: Array = field(dims=(), unit="cm2 g-1", description="specific leaf area")
    density: Array = field(dims=(), unit="plants m-2", description="plant density")
    tmean_weight: float | Array = coef(
        0.5,
        "-",
        "weight of tmin and of tmax in the daily mean temperature of thermal time",
        Provenance(
            "none",
            paper="McMaster and Wilhelm (1997), Agric. For. Meteorol. 87: 291-300",
            equation="1",
            note="growing degree-days from the mean of the daily extremes, (tmax + tmin) / 2",
        ),
    )


class TobaccoCrop(State):
    """The crop slot: organ queue plus calendar and mass bookkeeping (``[n_crop]`` leaves)."""

    organs: OrganQueue = field(description="leaves by rank")
    active: Array = field(unit="-", description="crop sown and not yet harvested", dims="n_crop")
    tt_leaf: Array = field(
        unit="degC d", description="thermal time since the last leaf appeared", dims="n_crop"
    )
    mass_added: Array = field(
        unit="g plant-1", description="cumulative mass put into the queue", dims="n_crop"
    )
    harvested: Array = field(
        unit="g plant-1", description="cumulative primed + harvested mass", dims="n_crop"
    )
    harvest_today: Array = field(
        unit="g plant-1", description="mass primed or harvested today", dims="n_crop"
    )


class TobaccoState(State):
    """State of the tobacco demo."""

    crop: TobaccoCrop = field(description="the single crop slot")


class TobaccoForcing(Forcing):
    """Daily temperatures plus the management event table (time axis first when stacked)."""

    tmin: Array = field(unit="degC", dims="T", fortran_name="TMIN")
    tmax: Array = field(unit="degC", dims="T", fortran_name="TMAX")
    events: EventTable = field(description="management events of the day")


def leaf_mass(age_tt: ArrayLike, params: TobaccoParams) -> Array:
    """``M(a)`` [g plant-1]: logistic in age, rescaled so that ``M(0) = 0`` and ``M(inf) = leaf_mass_max``."""
    r, a50 = params.growth_rate, params.age_half
    s0 = jax.nn.sigmoid(-r * a50)
    return params.leaf_mass_max * (jax.nn.sigmoid(r * (jnp.asarray(age_tt) - a50)) - s0) / (1.0 - s0)


# ---------------------------------------------------------------------------------- processes


@process(
    reads=("crop.organs", "crop.active", "crop.tt_leaf"),
    writes=("crop.organs", "crop.active", "crop.tt_leaf"),
    source="bookkeeping; (rotation reuses the crop slot)",
    key="crop/tobacco_demo.calendar@none:demo",
    provenance="equations_only",
    grid="point",
    sources=(("sow event empties the organ queue and restarts the leaf clock", "bookkeeping"),),
    deviates=(),
)
def crop_calendar(state: TobaccoState, params: TobaccoParams, forcing_t: TobaccoForcing) -> TobaccoState:
    """On a sow day: empty the organ queue, activate the crop, restart the leaf clock.

    Source: bookkeeping, no literature (rotation = several sow events reusing the crop slot).
    """
    c = state.crop
    sow = forcing_t.events.sow
    q = c.organs
    fresh = OrganQueue.empty(q.n_crop, q.n_cohort, dtype=q.area.dtype)
    organs = jtu.tree_map(lambda new, old: jnp.where(sow, new, old), fresh, q)
    return eqx.tree_at(
        lambda s: (s.crop.organs, s.crop.active, s.crop.tt_leaf),
        state,
        (organs, c.active | sow, jnp.where(sow, jnp.zeros_like(c.tt_leaf), c.tt_leaf)),
    )


@process(
    reads=("crop.organs", "crop.active", "crop.tt_leaf", "crop.mass_added"),
    writes=("crop.organs", "crop.tt_leaf", "crop.mass_added"),
    source="phyllochron appearance + logistic leaf growth in thermal age (generic; e.g. CROPGRO / tobacco "
    "leaf-position models)",
    key="crop/tobacco_demo.leaves@none:demo",
    provenance="equations_only",
    grid="point",
    sources=(
        ("one leaf appears per phyllochron of thermal time above tbase", "generic phyllochron model"),
        ("leaf mass as a logistic function of thermal age", "generic logistic growth (module docstring)"),
    ),
    deviates=(),
)
def leaf_appearance_growth(
    state: TobaccoState, params: TobaccoParams, forcing_t: TobaccoForcing
) -> TobaccoState:
    """Age and grow the alive leaves by today's thermal time; let one leaf appear per phyllochron.

    Source: phyllochron-driven leaf appearance with a logistic mass-age curve (module docstring).
    """
    c = state.crop
    q = c.organs
    tmean = params.tmean_weight * (forcing_t.tmin + forcing_t.tmax)
    dtt = jnp.where(c.active, jnp.maximum(tmean - params.tbase, 0.0), 0.0)  # [n_crop]
    a_old = q.age_tt
    d_mass = jnp.where(q.alive, leaf_mass(a_old + dtt[:, None], params) - leaf_mass(a_old, params), 0.0)
    q = grow(age(q, dtt), d_area=params.sla * d_mass, d_mass=d_mass)
    tt = c.tt_leaf + dtt
    due = c.active & (tt >= params.phyllochron) & (q.n_active < q.cap)
    over = jnp.maximum(tt - params.phyllochron, 0.0)
    m0 = jnp.where(due, leaf_mass(over, params), 0.0)
    slot = (q.rank == q.n_active[:, None]) & due[:, None]  # the slot appear() will fill
    q = appear(q, due, area0=params.sla * m0, mass0=m0)
    q = eqx.tree_at(lambda o: o.age_tt, q, jnp.where(slot, over[:, None], q.age_tt))
    added = jnp.sum(d_mass, axis=-1) + m0
    return eqx.tree_at(
        lambda s: (s.crop.organs, s.crop.tt_leaf, s.crop.mass_added),
        state,
        (q, jnp.where(due, over, tt), c.mass_added + added),
    )


@process(
    reads=("crop.organs", "crop.active", "crop.harvested"),
    writes=("crop.organs", "crop.active", "crop.harvested", "crop.harvest_today"),
    source="(topping caps n_active, priming harvests a rank range)",
    key="crop/tobacco_demo.management@none:demo",
    provenance="equations_only",
    grid="point",
    sources=(("topping, priming of a rank range, final harvest from the event table", "bookkeeping"),),
    deviates=(),
)
def management(state: TobaccoState, params: TobaccoParams, forcing_t: TobaccoForcing) -> TobaccoState:
    """Topping, priming of ranks ``[lo, hi)``, and the final harvest of every alive leaf.

    Source: EventTable topping / priming / harvest fields.
    """
    ev = forcing_t.events
    c = state.crop
    q = top(c.organs, c.organs.n_active, mask=ev.topping)
    q, h_prime = prime(q, ev.priming_lo, ev.priming_hi)
    all_hi = jnp.where(ev.harvest, q.n_cohort, -1)
    q, h_final = prime(q, jnp.where(ev.harvest, 0, -1), all_hi)
    today = h_prime + h_final
    return eqx.tree_at(
        lambda s: (s.crop.organs, s.crop.active, s.crop.harvested, s.crop.harvest_today),
        state,
        (q, c.active & ~ev.harvest, c.harvested + today, today),
    )


def tobacco_outputs(state: TobaccoState, params: TobaccoParams, forcing_t: TobaccoForcing) -> dict[str, Any]:
    """Per-day outputs: cumulative and daily harvest, mass bookkeeping, LAI, leaf count, alive mass."""
    c = state.crop
    q = c.organs
    return {
        "harvested": c.harvested,
        "harvest_today": c.harvest_today,
        "mass_added": c.mass_added,
        "alive_mass": jnp.sum(jnp.where(q.alive, q.mass, 0.0), axis=-1),
        "lai": aggregate(q, params.density)["lai"],
        "n_active": q.n_active,
    }


def tobacco_model() -> Model:
    """The demo model: calendar, appearance + growth, management."""
    return Model(
        TobaccoState,
        [crop_calendar, leaf_appearance_growth, management],
        outputs=tobacco_outputs,
        name="tobacco_demo",
    )


# ---------------------------------------------------------------------------------- inputs


def default_params(**overrides: float) -> TobaccoParams:
    """Plausible tobacco-like values (flue-cured tobacco order of magnitude); override by keyword."""
    vals = dict(_DEFAULT_VALUES)
    vals.update(overrides)
    return TobaccoParams(**{k: jnp.asarray(float(v)) for k, v in vals.items()})


def initial_state(n_cohort: int = _DEFAULT_N_COHORT, n_crop: int = 1) -> TobaccoState:
    """No crop sown yet; an empty queue of ``n_cohort`` leaf positions."""
    z = jnp.zeros((n_crop,))
    return TobaccoState(
        crop=TobaccoCrop(
            organs=OrganQueue.empty(n_crop, n_cohort, dtype=z.dtype),
            active=jnp.zeros((n_crop,), dtype=bool),
            tt_leaf=z,
            mass_added=z,
            harvested=z,
            harvest_today=z,
        )
    )


def synthetic_events(
    dates: pd.DatetimeIndex,
    *,
    sow_month_day: tuple[int, int] = _DEFAULT_SOW_MONTH_DAY,
    topping_after: int = _DEFAULT_TOPPING_AFTER,
    primings: Sequence[tuple[int, tuple[int, int]]] = DEFAULT_PRIMINGS,
    harvest_after: int = _DEFAULT_HARVEST_AFTER,
) -> EventTable:
    """One tobacco season per calendar year of ``dates``: sow, topping, primings, final harvest.

    Offsets are days after sowing; events falling outside ``dates`` are dropped.
    """
    records: list[tuple[Any, str, Any]] = []
    for year in sorted({pd.Timestamp(d).year for d in dates}):
        sow = pd.Timestamp(year=int(year), month=sow_month_day[0], day=sow_month_day[1])
        records.append((sow, "sow", 1.0))
        records.append((sow + pd.Timedelta(days=topping_after), "topping", 1.0))
        for d, ranks in primings:
            records.append((sow + pd.Timedelta(days=d), "priming", ranks))
        records.append((sow + pd.Timedelta(days=harvest_after), "harvest", 1.0))
    return EventTable.from_records(records, dates, outside="drop")


def build_forcing(met: pd.DataFrame, events: EventTable, start: str, end: str) -> TobaccoForcing:
    """Stack ``tmin`` / ``tmax`` of the prepared ``.MET`` (:func:`agrijax.io.rzwqm.read_met`) over
    ``[start, end]`` with an event table of the same days."""
    days = pd.date_range(start, end, freq="D")
    missing = days.difference(met.index)
    if len(missing):
        raise ValueError(f".MET has no record for {len(missing)} days, first {str(missing[0])[:10]}")
    if events.n_days != len(days):
        raise ValueError(f"event table has {events.n_days} days, forcing {len(days)}")
    w = met.loc[days]
    return TobaccoForcing(
        tmin=jnp.asarray(w["tmin"].to_numpy(dtype=np.float64)),
        tmax=jnp.asarray(w["tmax"].to_numpy(dtype=np.float64)),
        events=events,
    )
