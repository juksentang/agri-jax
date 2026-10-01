"""The soil-water day of RZWQM2: redistribution, one infiltration event, redistribution.

RZWQM2 advances the soil water of a day by Richards redistribution up to the start of the
day's storm, then runs the storm as an instantaneous infiltration event
(:mod:`~agrijax.processes.soil_water.infiltration`), then redistributes to the end of the day;
the day's clock is not advanced by the storm duration. Here

``day = redistribute(n_pre sub-steps, [0, ts0]) -> green_ampt_event -> redistribute(n_post, [ts0, 24])``

with ``n_pre`` and ``n_post`` static. ``ts0`` is the storm start from the forcing; on a day
without a storm it is ``24 n_pre / (n_pre + n_post)``, so the sub-steps are the uniform ones
of a single segment. The event kernel runs every day and is the identity without rain. The
schedule depends on the forcing only, never on the state, so shapes stay static. At most one
event per day (RZWQM2 reads one storm per day); a storm spanning midnight arrives as one
segment per day (the forcing splits it as RZWQM2 ``CHSPAN`` does, the continuation starting at
``ts0 = 0``), so no state carries a remaining storm.

With ``n_pre = 0`` every event is placed at ``t = 0`` and the hours it was moved are reported
(``event_shift``); CA-TPA storms all start at midnight (the ``.BRK`` file is built from daily
totals), which makes ``n_pre = 0`` exact there and gives the post-event redistribution all the
sub-steps.

After the event, a fraction of the post-event sub-steps can be graded towards the event
(``post_grading = p``: edges ``ts0 + (24 - ts0) u^p`` on event days, uniform otherwise):
the profile just behind the wetting front redistributes fastest (RZWQM2 restarts from its
smallest time step after an event).

The sink is the typed channel record :class:`~agrijax.processes.soil_water.sinks.SinkChannels`
(``uptake``, ``tile``, ``lateral``, ``subirrigation``, ``macropore_to_drain``), each with its own
daily total in the fluxes and its own term in the balance. In the coupled model only ``uptake`` is
non-zero: the day's per-layer uptake spread uniformly (RZWQM2 with a DSSAT crop and
``ISTRESS = 0``); a plain array is taken as that channel. Surface supply (``supply``) is still
accepted as a prescribed surface flux of the Richards step. CA-TPA runs RZWQM2 with PRMS snow
on (snowmelt on 195 days of 2015-2023, 10-25 cm per year, 80 % of it infiltrating); this kernel
holds no snow pack: the melt of the snow module (:mod:`agrijax.processes.snow`) enters through
``supply``, and the tests replay the reference melt days through ``supply``.

Water ledger: :func:`soil_water_ledger` is the day's ``ledger.close`` entry
(:func:`agrijax.core.ledger.water_ledger`) and :func:`soil_water_ledger_init` its zero ledger.
Storage is profile plus pond; the inflows are the prescribed surface supply and the event rain;
the outflows are evaporation, drainage (seepage included), runoff (event runoff included) and
**every sink channel** of :data:`~agrijax.processes.soil_water.sinks.SINK_LEDGER_OUTFLOWS`, each
booked as its own cumulative channel (subirrigation as a negative outflow). The ledger residual is
then the day's ``balance_error`` recomputed from the booked channels.

Variants: ``soil_water_day`` (key ``soil_water/day@rzwqm2-4.6:faithful``, the Green-Ampt event
with the branches CA-TPA uses) and ``soil_water_day_replay`` (``...:replay_flux``, the
prescribed-supply day: the event is ignored and ``supply`` carries the water that infiltrates; bit-identical
to :func:`~agrijax.processes.soil_water.richards.richards_day`).

Source: Ahuja, L.R., Rojas, K.W., Hanson, J.D., Shaffer, M.J., Ma, L. (eds.), 2000. Root Zone
Water Quality Model, ch. 3 (event hydrology and redistribution); RZWQM2 ``PHYSCL`` day loop
(``Rzday.for``), read for conventions only.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.ledger import Channel, WaterLedger, water_ledger
from agrijax.core.process import Process, check_enabled, process
from agrijax.core.state import Forcing, Params, field

from .coefficients import setting_field
from .fixed_cn import day_edges
from .infiltration import GAResult, GreenAmptParams, StormForcing, green_ampt_event
from .integrator import DayEvent, DayPlan, integrator_for
from .problem import RichardsProblem
from .richards import (
    RichardsGrid,
    RichardsParams,
    RichardsState,
    SoilWater,
    day_fluxes,
    node_pori,
    other_sinks,
    richards_day,
)
from .sinks import SINK_LEDGER_OUTFLOWS, SinkChannels, as_sink_channels

__all__ = [
    "SOIL_WATER_LEDGER_INFLOWS",
    "SOIL_WATER_LEDGER_OUTFLOWS",
    "DayConfig",
    "SoilWaterDayForcing",
    "SoilWaterDayParams",
    "day_edges",
    "infiltration_ga",
    "soil_water_day",
    "soil_water_day_drain_cap",
    "soil_water_day_flux_evap",
    "soil_water_day_kernel",
    "soil_water_day_replay",
    "soil_water_day_replay_conventions",
    "soil_water_day_rzwqm2_conventions",
    "soil_water_ledger",
    "soil_water_ledger_init",
    "with_conventions",
]


class DayConfig(eqx.Module):
    """Static schedule of the day: ``n_pre`` / ``n_post`` sub-steps before / after the event.

    ``post_grading`` ``>= 1`` grades the post-event sub-steps towards the event on event days.
    Numerical settings of this implementation (``soil_water.coefficients.SETTINGS``).
    """

    n_pre: int = setting_field(
        "soil_water_day.n_pre",
        12,
        "-",
        "Richards sub-steps before the day's event (0 places every event at t = 0)",
        origin="agrijax",
        basis="the day is two redistribution segments around the event, 12 sub-steps each by default "
        "(day.py module docstring); tests/unit/test_infiltration_day.py",
    )
    n_post: int = setting_field(
        "soil_water_day.n_post",
        12,
        "-",
        "Richards sub-steps after the day's event (12 + 12 = the 24 sub-steps a day of the fixed-step "
        "baseline)",
        origin="agrijax",
        basis="the day is two redistribution segments around the event, 12 sub-steps each by default "
        "(day.py module docstring); tests/integration/test_infiltration_catpa.py",
    )
    post_grading: float = setting_field(
        "soil_water_day.post_grading",
        2.0,
        "-",
        "exponent p of the post-event edges ts0 + (24 - ts0) u^p on event days (1 = uniform); "
        "RZWQM2 restarts from its smallest time step after an event",
        origin="agrijax",
        basis="a choice of this implementation (day.py module docstring)",
    )

    def __check_init__(self) -> None:
        if self.n_pre < 0 or self.n_post < 1 or self.post_grading < 1.0:
            raise ValueError("n_pre >= 0, n_post >= 1 and post_grading >= 1 are required")


class SoilWaterDayParams(Params):
    """Parameters of the soil-water day: Richards (soil, grid, numerics) and the event."""

    richards: RichardsParams
    infiltration: GreenAmptParams
    config: DayConfig = eqx.field(static=True, default=DayConfig())

    def __check_init__(self) -> None:
        self.infiltration.config.check_grid(self.richards.grid.tl)


class SoilWaterDayForcing(Forcing):
    """One day's water inputs: prescribed surface flux, evaporation demand, uptake, and the storm."""

    storm: StormForcing
    supply: Array = field(unit="cm h-1", dims=("T", "hour"), description="prescribed surface water flux")
    evaporation: Array = field(unit="cm h-1", dims=("T", "hour"), description="potential soil evaporation")
    uptake: Array = field(unit="cm d-1", dims=("T", "n_node"), description="root water uptake per layer")


def soil_water_day_kernel(
    water: SoilWater,
    params: SoilWaterDayParams,
    supply: Array,
    evaporation: Array,
    uptake: Array | SinkChannels,
    storm: StormForcing,
) -> tuple[SoilWater, GAResult, Array]:
    """One day with the Green-Ampt event: returns ``(new state with the day's fluxes, event, event_shift)``.

    ``uptake`` is the sink-channel record, or the per-layer root water uptake [cm d-1] alone. The
    Richards redistribution before and after the event is the integrator of
    ``params.richards.stepping``, called through its protocol
    (:class:`~agrijax.processes.soil_water.integrator.RichardsIntegrator`): it places the event
    (:class:`~agrijax.processes.soil_water.integrator.DayEvent`) and runs it once; the day books the
    event terms and the balance.

    Source: Ahuja et al. (2000) ch. 3; RZWQM2 ``PHYSCL`` / ``EVNTRO`` / ``RICHRD`` (conventions).
    """
    rp = params.richards
    cfg = params.config
    dtype = water.theta.dtype
    n = rp.grid.n_node
    soil = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,)), rp.soil.at_nodes())
    depth = jnp.asarray(storm.depth, dtype)
    has_event = jnp.sum(depth) > 0.0
    uptake = as_sink_channels(uptake)
    pori = node_pori(rp, params.infiltration.aef, dtype)
    problem = RichardsProblem.of_day(rp, supply, evaporation, uptake, dtype, pori)

    def event(w_pre: SoilWater) -> tuple[SoilWater, GAResult]:
        ev = green_ampt_event(
            w_pre.theta,
            w_pre.h,
            soil,
            rp.grid.tl,
            jnp.asarray(params.infiltration.aef, dtype),
            storm.duration,
            depth,
            params.infiltration.config,
            params.infiltration.coef(),
        )
        return w_pre.replace(theta=ev.theta, h=ev.h), ev

    plan = DayPlan(DayEvent(storm.ts0, has_event, cfg.n_pre, cfg.n_post, cfg.post_grading, event))
    w_post, diag = integrator_for(rp.stepping).step_day(problem, water, plan)
    ev = diag.event
    tot = diag.totals
    w0 = water.storage(rp.grid) + water.pond
    w1 = w_post.storage(rp.grid) + w_post.pond
    drainage = tot.drainage + ev.seepage
    runoff = tot.runoff + ev.runoff
    sinks = tot.uptake + other_sinks(tot)
    balance = (w1 - w0) - (tot.supply + ev.rain - tot.evaporation - drainage - sinks - runoff)
    flux = day_fluxes(
        tot,
        diag.stats,
        rp.config,
        dtype,
        infiltration=tot.infiltration + ev.infiltration,
        drainage=drainage,
        runoff=runoff,
        balance_error=balance,
        rain=ev.rain,
        event_infiltration=ev.infiltration,
        event_runoff=ev.runoff,
        seepage=ev.seepage,
    )
    t_ev = cast(Array, diag.t_event)  # an event day always reports its event time
    shift = jnp.where(has_event, t_ev - jnp.asarray(storm.ts0, dtype), 0.0)
    return w_post.replace(flux=flux), ev, shift


def _checked(h: Array, where: str) -> Array:
    if check_enabled():
        return eqx.error_if(h, ~jnp.all(jnp.isfinite(h)), f"{where}: non-finite head")
    return h


def _checked_water(new: SoilWater, rp: RichardsParams, where: str) -> Array:
    """``_checked`` of the heads, plus the integrator's own check (the adaptive one: no unconverged
    step and no exhausted budget)."""
    h = _checked(new.h, where)
    if check_enabled():
        h = integrator_for(rp.stepping).check(h, new.flux, where)
    return h


_SOURCES = (
    ("Richards redistribution segments", "Ahuja et al. (2000) RZWQM ch. 3; Celia et al. (1990)"),
    (
        "Green-Ampt event, layered capacity",
        "Green & Ampt (1911); Mein & Larson (1973); Ahuja et al. (2000) ch. 3",
    ),
    (
        "day = redistribution / instantaneous event / redistribution; slices, AEF porosity, VRCF, DTMIN",
        "RZWQM2 PHYSCL, EVNTRO, INFIL, UNSATFLO, MIXRUNOFF (Rzday.for, RZTEST.for), read for conventions",
    ),
)
_DEVIATES_GA = (
    (
        "only the Green-Ampt branches used at CA-TPA: no crust, macropores, water table, tile drains, "
        "unit-gradient flow below the front, ponded irrigation or snowmelt events",
        "the CA-TPA rzwqm.dat switches the crust, macropores, water table, drains, unit-gradient flow and "
        "ponded irrigation off; snow is on at CA-TPA (PRMS, 10-25 cm melt per year), its melt events are "
        "left out of this kernel and the melt enters through the surface supply (the snow module)",
        "infiltration.py module docstring; tests/integration/test_infiltration_catpa.py",
    ),
    (
        "water above the available porosity theta_s*AEF is kept (the slice counts as saturated), "
        "not deleted as the reference's initial cap does",
        "RZWQM2 drains theta above AEF*theta_s after every Richards step (DRAIN), so its event never sees "
        "it; that drainage is not ported and deleting water would break the ledger",
        "infiltration.py green_ampt_event; tests/unit/test_infiltration.py mass-balance test",
    ),
    (
        "the Richards default numerics (implicit time weights, damped Newton) apply to both segments",
        "as for soil_water/richards@rzwqm2-4.6:faithful",
        "tests/unit/test_richards.py convergence study",
    ),
    (
        "adaptive mode: the first Richards step after the event has alpha = 1 (RZWQM2: 1/2, the day "
        "clock has advanced)",
        "the event leaves a steep profile behind the front on which Crank-Nicolson may oscillate",
        "richards_adaptive.py AdaptiveCN docstring; tests/unit/test_richards_adaptive.py",
    ),
)
#: the two RZWQM2 conventions the faithful keys leave out (they are the convention variants below)
_NO_DRAIN = (
    "no DRAIN cap: after a Richards step a node may hold water up to theta_s; RZWQM2 passes the water "
    "above aef * theta_s down at once after every RICHRD step (DRAIN)",
    "kept off in the faithful keys so that the validated prescribed-supply-day and Green-Ampt values stay "
    "as pinned; the cap is the labelled variant soil_water/day@rzwqm2-4.6:drain_cap",
    "conventions.py module docstring; tests/unit/test_richards_conventions.py; "
    "tests/integration/test_richards_conventions_years.py (site-years with and without the cap)",
)
_NO_FLUX_PEAK = (
    "the evaporation dry limit is the Darcy flux with the ghost head at Hmin; RZWQM2 keeps the flux "
    "condition while the ghost head stays above Hmin, which delivers up to the peak of that flux over "
    "the ghost head",
    "kept off in the faithful keys so that the validated values stay as pinned; the flux-mode limit is "
    "the labelled variant soil_water/day@rzwqm2-4.6:flux_evap",
    "conventions.py module docstring; tests/unit/test_richards_conventions.py; "
    "tests/integration/test_richards_conventions_years.py (site-years with and without the limit)",
)
_DRAIN_STEPS = (
    "the DRAIN cap acts after every accepted sub-step of this implementation's stepping (fixed or "
    "adaptive), whose steps are not RZWQM2's; it is not applied after tillage reconsolidation "
    "(RZWQM2 also calls DRAIN there, Rzday.for:1471)",
    "RZWQM2 applies it per RICHRD step, so the capped water depends on the step table and converges as "
    "the steps shrink; tillage is a parameter change in this implementation",
    "conventions.py module docstring",
)
_FLUX_PEAK_EVERY_STEP = (
    "the flux-mode limit is the peak on every sub-step; RZWQM2 switches to the head condition Hmin once "
    "the demand exceeds the peak and keeps it for the rest of the day",
    "the peak rule is the one measured against the reference (US_Rockfish evaporation deficits of "
    "1.3-6.1 cm a year closed to <= 0.008 cm); the sticky switch is not ported",
    "conventions.py module docstring",
)
_EVENT_EXCESS = (
    "the event keeps water above theta_s*AEF that it meets (with the DRAIN cap only possible in the "
    "initial profile of a run)",
    "deleting water would break the ledger; after a capped Richards step no node holds more",
    "infiltration.py green_ampt_event; tests/unit/test_infiltration.py mass-balance test",
)
_DEVIATES_FAITHFUL = (*_DEVIATES_GA, _NO_DRAIN, _NO_FLUX_PEAK)
_CONVENTION_SOURCES = (
    (
        "DRAIN cap: water above aef * theta_s cascades down after every Richards step, heads from WCHEAD",
        "Ahuja et al. (2000) ch. 3 (field saturation); RZWQM2 DRAIN (Rzday.for:3975) called by REDIST "
        "(Rzrich.for:1092), read for conventions",
    ),
    (
        "flux-mode evaporation limit: the peak of the ghost-face Darcy flux over the ghost head",
        "Ahuja et al. (2000) ch. 3; RZWQM2 CHKBC / CNHEAD (Rzrich.for:3-110), read for conventions",
    ),
)


def _variant_deviates(drain: bool, flux: bool, base: tuple[Any, ...] = _DEVIATES_GA) -> tuple[Any, ...]:
    out = [_EVENT_EXCESS if (drain and d is _DEVIATES_GA[1]) else d for d in base]
    out += [_DRAIN_STEPS] if drain else [_NO_DRAIN]
    out += [_FLUX_PEAK_EVERY_STEP] if flux else [_NO_FLUX_PEAK]
    return tuple(out)


def with_conventions(params: SoilWaterDayParams, *, drain: bool, flux_peak: bool) -> SoilWaterDayParams:
    """``params`` with the RZWQM2 convention switches of ``RichardsConfig`` set (static fields).

    Source: :mod:`~agrijax.processes.soil_water.conventions` (RZWQM2 ``DRAIN`` and ``CHKBC``).
    """
    rp = params.richards
    cfg = dataclasses.replace(
        rp.config, drain_cap=drain, evaporation_limit="flux_peak" if flux_peak else "hmin"
    )
    return dataclasses.replace(params, richards=dataclasses.replace(rp, config=cfg))


def _without_conventions(params: SoilWaterDayParams, key: str) -> None:
    on = params.richards.config.conventions
    if len(on) > 0:
        raise ValueError(
            f"{key} runs without the RZWQM2 convention switches {on}; use its convention variants "
            "(soil_water/day@rzwqm2-4.6:drain_cap, :flux_evap, :rzwqm2_conventions, :replay_flux_conventions)"
        )


@process(
    reads=("soil_water.h", "soil_water.theta", "soil_water.pond", "soil_water.dt_next"),
    writes=("soil_water",),
    source="Ahuja et al. (2000) RZWQM ch. 3; RZWQM2 Rzday.for PHYSCL, RZTEST.for EVNTRO/INFIL",
    fortran_name="PHYSCL",
    key="soil_water/day@rzwqm2-4.6:faithful",
    provenance="reference_only_conventions",
    grid="rzwqm2_nodes",
    ref_build="RZWQM2 4.6 main_ryzen5_avx512",
    sources=_SOURCES,
    deviates=_DEVIATES_FAITHFUL,
)
def soil_water_day(
    state: RichardsState, params: SoilWaterDayParams, forcing_t: SoilWaterDayForcing
) -> RichardsState:
    """One soil-water day: redistribution, the day's Green-Ampt event, redistribution.

    Thin wrapper of :func:`soil_water_day_kernel`; writes the new ``soil_water`` sub-tree with
    the day's fluxes (event terms in ``rain``, ``event_infiltration``, ``event_runoff``,
    ``seepage``). Under ``AGRI_JAX_CHECK=1`` a non-finite head raises.

    Source: Ahuja et al. (2000) ch. 3; RZWQM2 ``PHYSCL`` (``Rzday.for``), ``EVNTRO``/``INFIL``
    (``RZTEST.for``). The RZWQM2 convention switches of ``RichardsConfig`` must be off (they are the
    convention variants).
    """
    _without_conventions(params, "soil_water/day@rzwqm2-4.6:faithful")
    new, _, _ = soil_water_day_kernel(
        state.soil_water, params, forcing_t.supply, forcing_t.evaporation, forcing_t.uptake, forcing_t.storm
    )
    h = _checked_water(new, params.richards, "soil_water_day")
    return eqx.tree_at(lambda s: s.soil_water, state, new.replace(h=h))


@process(
    reads=("soil_water.h", "soil_water.theta", "soil_water.pond", "soil_water.dt_next"),
    writes=("soil_water",),
    source="Ahuja et al. (2000) RZWQM ch. 3; Celia et al. (1990); RZWQM2 Rzrich.for RICHRD",
    fortran_name="RICHRD",
    key="soil_water/day@rzwqm2-4.6:replay_flux",
    provenance="reference_only_conventions",
    grid="rzwqm2_nodes",
    ref_build="RZWQM2 4.6 main_ryzen5_avx512",
    sources=_SOURCES[:1],
    deviates=(
        (
            "no infiltration event: the storm forcing is ignored and supply is the water that infiltrates "
            "(e.g. .ana column 5), applied as a surface flux",
            "the prescribed-supply configuration: it separates redistribution error from infiltration error",
            "tests/integration/test_richards_catpa.py; tests/unit/test_infiltration_day.py",
        ),
        _NO_DRAIN,
        _NO_FLUX_PEAK,
    ),
)
def soil_water_day_replay(
    state: RichardsState, params: SoilWaterDayParams, forcing_t: SoilWaterDayForcing
) -> RichardsState:
    """The prescribed-supply day on the day forcing: :func:`richards_day` with ``supply``; the storm is
    not used.

    Source: Ahuja et al. (2000) ch. 3; Celia et al. (1990); RZWQM2 ``RICHRD`` (``Rzrich.for``).
    """
    _without_conventions(params, "soil_water/day@rzwqm2-4.6:replay_flux")
    new = _replay_day(state, params, forcing_t, "soil_water_day_replay")
    return eqx.tree_at(lambda s: s.soil_water, state, new)


def _replay_day(
    state: RichardsState, params: SoilWaterDayParams, forcing_t: SoilWaterDayForcing, where: str
) -> SoilWater:
    new = richards_day(
        state.soil_water,
        params.richards,
        forcing_t.supply,
        forcing_t.evaporation,
        forcing_t.uptake,
        aef=params.infiltration.aef,
    )
    return new.replace(h=_checked_water(new, params.richards, where))


def _event_day(
    state: RichardsState, params: SoilWaterDayParams, forcing_t: SoilWaterDayForcing, where: str
) -> SoilWater:
    new, _, _ = soil_water_day_kernel(
        state.soil_water, params, forcing_t.supply, forcing_t.evaporation, forcing_t.uptake, forcing_t.storm
    )
    return new.replace(h=_checked_water(new, params.richards, where))


# ---------------------------------------------------------------------------
# RZWQM2 convention variants: the DRAIN cap and the flux-mode evaporation limit are off in the
# faithful keys, so that their validated values stay as pinned, and on in these variants
# ---------------------------------------------------------------------------

_DAY_READS = ("soil_water.h", "soil_water.theta", "soil_water.pond", "soil_water.dt_next")


@process(
    reads=_DAY_READS,
    writes=("soil_water",),
    source="Ahuja et al. (2000) RZWQM ch. 3; RZWQM2 Rzday.for PHYSCL/DRAIN, RZTEST.for EVNTRO/INFIL",
    fortran_name="PHYSCL",
    key="soil_water/day@rzwqm2-4.6:drain_cap",
    provenance="reference_only_conventions",
    grid="rzwqm2_nodes",
    ref_build="RZWQM2 4.6 main_ryzen5_avx512",
    sources=(*_SOURCES, _CONVENTION_SOURCES[0]),
    deviates=_variant_deviates(drain=True, flux=False),
)
def soil_water_day_drain_cap(
    state: RichardsState, params: SoilWaterDayParams, forcing_t: SoilWaterDayForcing
) -> RichardsState:
    """:func:`soil_water_day` with RZWQM2's DRAIN cap after every Richards sub-step (``drain_cap``).

    Source: Ahuja et al. (2000) ch. 3; RZWQM2 ``PHYSCL``, ``DRAIN`` (``Rzday.for``), ``REDIST``
    (``Rzrich.for``), read for conventions.
    """
    p = with_conventions(params, drain=True, flux_peak=False)
    new = _event_day(state, p, forcing_t, "soil_water_day_drain_cap")
    return eqx.tree_at(lambda s: s.soil_water, state, new)


@process(
    reads=_DAY_READS,
    writes=("soil_water",),
    source="Ahuja et al. (2000) RZWQM ch. 3; RZWQM2 Rzrich.for CHKBC, RZTEST.for EVNTRO/INFIL",
    fortran_name="PHYSCL",
    key="soil_water/day@rzwqm2-4.6:flux_evap",
    provenance="reference_only_conventions",
    grid="rzwqm2_nodes",
    ref_build="RZWQM2 4.6 main_ryzen5_avx512",
    sources=(*_SOURCES, _CONVENTION_SOURCES[1]),
    deviates=_variant_deviates(drain=False, flux=True),
)
def soil_water_day_flux_evap(
    state: RichardsState, params: SoilWaterDayParams, forcing_t: SoilWaterDayForcing
) -> RichardsState:
    """:func:`soil_water_day` with RZWQM2's flux-mode evaporation limit (``flux_peak``).

    Source: Ahuja et al. (2000) ch. 3; RZWQM2 ``CHKBC`` / ``CNHEAD`` (``Rzrich.for``), read for
    conventions.
    """
    p = with_conventions(params, drain=False, flux_peak=True)
    new = _event_day(state, p, forcing_t, "soil_water_day_flux_evap")
    return eqx.tree_at(lambda s: s.soil_water, state, new)


@process(
    reads=_DAY_READS,
    writes=("soil_water",),
    source="Ahuja et al. (2000) RZWQM ch. 3; RZWQM2 Rzday.for PHYSCL/DRAIN, Rzrich.for CHKBC",
    fortran_name="PHYSCL",
    key="soil_water/day@rzwqm2-4.6:rzwqm2_conventions",
    provenance="reference_only_conventions",
    grid="rzwqm2_nodes",
    ref_build="RZWQM2 4.6 main_ryzen5_avx512",
    sources=(*_SOURCES, *_CONVENTION_SOURCES),
    deviates=_variant_deviates(drain=True, flux=True),
)
def soil_water_day_rzwqm2_conventions(
    state: RichardsState, params: SoilWaterDayParams, forcing_t: SoilWaterDayForcing
) -> RichardsState:
    """:func:`soil_water_day` with both RZWQM2 conventions: the DRAIN cap and the flux-mode limit.

    Source: Ahuja et al. (2000) ch. 3; RZWQM2 ``PHYSCL``, ``DRAIN``, ``CHKBC`` (conventions).
    """
    p = with_conventions(params, drain=True, flux_peak=True)
    new = _event_day(state, p, forcing_t, "soil_water_day_rzwqm2_conventions")
    return eqx.tree_at(lambda s: s.soil_water, state, new)


@process(
    reads=_DAY_READS,
    writes=("soil_water",),
    source="Ahuja et al. (2000) RZWQM ch. 3; Celia et al. (1990); RZWQM2 Rzrich.for RICHRD/CHKBC, "
    "Rzday.for DRAIN",
    fortran_name="RICHRD",
    key="soil_water/day@rzwqm2-4.6:replay_flux_conventions",
    provenance="reference_only_conventions",
    grid="rzwqm2_nodes",
    ref_build="RZWQM2 4.6 main_ryzen5_avx512",
    sources=(_SOURCES[0], *_CONVENTION_SOURCES),
    deviates=(
        (
            "no infiltration event: the storm forcing is ignored and supply is the water that infiltrates "
            "(e.g. .ana column 5), applied as a surface flux",
            "the prescribed-supply configuration with both RZWQM2 conventions: it separates redistribution "
            "error from infiltration error",
            "tests/integration/test_richards_conventions_years.py "
            "(site-years with and without the conventions)",
        ),
        _DRAIN_STEPS,
        _FLUX_PEAK_EVERY_STEP,
    ),
)
def soil_water_day_replay_conventions(
    state: RichardsState, params: SoilWaterDayParams, forcing_t: SoilWaterDayForcing
) -> RichardsState:
    """The prescribed-supply day (:func:`soil_water_day_replay`) with the DRAIN cap and the flux-mode
    limit.

    Source: Ahuja et al. (2000) ch. 3; Celia et al. (1990); RZWQM2 ``RICHRD``, ``DRAIN``, ``CHKBC``.
    """
    p = with_conventions(params, drain=True, flux_peak=True)
    new = _replay_day(state, p, forcing_t, "soil_water_day_replay_conventions")
    return eqx.tree_at(lambda s: s.soil_water, state, new)


@process(
    reads=("soil_water.h", "soil_water.theta"),
    writes=(
        "soil_water.h",
        "soil_water.theta",
        "soil_water.flux.rain",
        "soil_water.flux.event_infiltration",
        "soil_water.flux.event_runoff",
        "soil_water.flux.seepage",
    ),
    source="Green & Ampt (1911); Mein & Larson (1973); RZWQM2 RZTEST.for EVNTRO/INFIL",
    fortran_name="EVNTRO",
    key="soil_water/infiltration_ga@rzwqm2-4.6:faithful",
    provenance="reference_only_conventions",
    grid="rzwqm2_nodes",
    ref_build="RZWQM2 4.6 main_ryzen5_avx512",
    sources=_SOURCES[1:],
    deviates=_DEVIATES_GA[:2],
)
def infiltration_ga(
    state: RichardsState, params: SoilWaterDayParams, forcing_t: SoilWaterDayForcing
) -> RichardsState:
    """The day's Green-Ampt event alone, as its own day entry: fills ``theta``/``h``, writes the event terms.

    Source: Green & Ampt (1911); Mein & Larson (1973); Ahuja et al. (2000) ch. 3; RZWQM2 ``EVNTRO``.
    """
    w = state.soil_water
    rp = params.richards
    n = rp.grid.n_node
    soil = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,)), rp.soil.at_nodes())
    ev = green_ampt_event(
        w.theta,
        w.h,
        soil,
        rp.grid.tl,
        jnp.asarray(params.infiltration.aef, w.theta.dtype),
        forcing_t.storm.duration,
        forcing_t.storm.depth,
        params.infiltration.config,
        params.infiltration.coef(),
    )
    flux = w.flux.replace(
        rain=ev.rain, event_infiltration=ev.infiltration, event_runoff=ev.runoff, seepage=ev.seepage
    )
    new = w.replace(theta=ev.theta, h=_checked(ev.h, "infiltration_ga"), flux=flux)
    return eqx.tree_at(lambda s: s.soil_water, state, new)


# ---------------------------------------------------------------------------
# water ledger of the soil-water day
# ---------------------------------------------------------------------------

#: ledger inflows of the soil-water day (``supply`` from the forcing, ``rain`` the event rain)
SOIL_WATER_LEDGER_INFLOWS: tuple[str, ...] = ("supply", "rain")

#: ledger outflows: ``{name: state path of its daily total}``; the surface and bottom terms, then
#: every sink channel of :data:`~agrijax.processes.soil_water.sinks.SINK_LEDGER_OUTFLOWS`
SOIL_WATER_LEDGER_OUTFLOWS: dict[str, str] = {
    "evaporation": "soil_water.flux.evaporation",
    "drainage": "soil_water.flux.drainage",
    "runoff": "soil_water.flux.runoff",
    **SINK_LEDGER_OUTFLOWS,
}


def _day_grid(params: Any) -> RichardsGrid:
    return params.richards.grid


def _daily_supply(state: Any, params: Any, forcing_t: Any) -> Array:
    """The day's prescribed surface supply [cm]: the 24 hourly rates [cm h-1] of 1 h each.

    Source: ``SoilWaterDayForcing.supply``, the prescribed surface supply, booked as an inflow of the
    day's water balance (Ahuja et al. (2000) ch. 3).
    """
    return jnp.sum(forcing_t.supply, axis=-1)


def soil_water_ledger(
    *,
    grid: Callable[[Any], RichardsGrid] = _day_grid,
    supply: Channel = _daily_supply,
    at: str = "ledger.water",
    atol: float | None = None,
    rtol: float | None = None,
    name: str = "ledger.close",
) -> Process:
    """The ``ledger.close`` entry of the soil-water day, booking every sink channel separately.

    ``grid(params)`` gives the node grid (default ``params.richards.grid``, for
    :class:`SoilWaterDayParams`), ``supply`` the day's surface supply [cm] (default: the sum of
    ``forcing_t.supply``). Storage is ``sum(theta * tl) + pond``. The channels are
    :data:`SOIL_WATER_LEDGER_INFLOWS` and :data:`SOIL_WATER_LEDGER_OUTFLOWS`; start the ledger
    with :func:`soil_water_ledger_init`.

    Source: Ahuja et al. (2000) ch. 3 (water balance: change in storage = inflows - outflows); every
    sink channel is booked as an outflow of its own.
    """

    def storage(state: Any, params: Any, forcing_t: Any) -> Array:
        w = state.soil_water
        return jnp.sum(w.theta * grid(params).tl, axis=-1) + w.pond

    return water_ledger(
        storage=storage,
        inflows={"supply": supply, "rain": "soil_water.flux.rain"},
        outflows=SOIL_WATER_LEDGER_OUTFLOWS,
        reads=("soil_water.theta", "soil_water.pond"),
        at=at,
        atol=atol,
        rtol=rtol,
        name=name,
    )


def soil_water_ledger_init(water: SoilWater, grid: RichardsGrid) -> WaterLedger:
    """A zero ledger with the channels of :func:`soil_water_ledger`, storage from ``water``."""
    return WaterLedger.init(
        water.storage(grid) + water.pond,
        inflows=SOIL_WATER_LEDGER_INFLOWS,
        outflows=tuple(SOIL_WATER_LEDGER_OUTFLOWS),
    )
