"""The soil-water day of RZWQM2: redistribution, one infiltration event, redistribution (plan 19 A6).

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
(``uptake``, ``tile``, ``lateral``, ``subirrigation``, ``macropore_to_drain``; plan 19 A6), each
with its own daily total in the fluxes and its own term in the balance. In M3 only ``uptake`` is
non-zero: the day's per-layer uptake spread uniformly (RZWQM2 with a DSSAT crop and
``ISTRESS = 0``); a plain array is taken as that channel. Surface supply (``supply``) is still
accepted as a prescribed surface flux of the Richards step. CA-TPA runs RZWQM2 with PRMS snow
on (snowmelt on 195 days of 2015-2023, 10-25 cm per year, 80 % of it infiltrating); this kernel
has no snow yet, so the tests replay those days through ``supply``, and snow is added in the M3
assembly.

Variants: ``soil_water_day`` (key ``soil_water/day@rzwqm2-4.6:faithful``, the Green-Ampt event
with the branches CA-TPA uses) and ``soil_water_day_replay`` (``...:replay_flux``, the M1 day:
the event is ignored and ``supply`` carries the water that infiltrates; bit-identical to
:func:`~agrijax.processes.soil_water.richards.richards_day`).

Source: Ahuja, L.R., Rojas, K.W., Hanson, J.D., Shaffer, M.J., Ma, L. (eds.), 2000. Root Zone
Water Quality Model, ch. 3 (event hydrology and redistribution); RZWQM2 ``PHYSCL`` day loop
(``Rzday.for``), read for conventions only.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.process import check_enabled, process
from agrijax.core.state import Forcing, Params, field

from .infiltration import GAResult, GreenAmptParams, StormForcing, green_ampt_event
from .richards import (
    HOURS_PER_DAY,
    RichardsParams,
    RichardsState,
    SoilWater,
    _zero_fluxes,
    day_alphas,
    other_sinks,
    richards_day,
    richards_substeps,
    sink_fluxes,
)
from .sinks import SinkChannels, as_sink_channels

__all__ = [
    "DayConfig",
    "SoilWaterDayForcing",
    "SoilWaterDayParams",
    "day_edges",
    "infiltration_ga",
    "soil_water_day",
    "soil_water_day_kernel",
    "soil_water_day_replay",
]


class DayConfig(eqx.Module):
    """Static schedule of the day: ``n_pre`` / ``n_post`` sub-steps before / after the event.

    ``post_grading`` ``>= 1`` grades the post-event sub-steps towards the event on event days.
    """

    n_pre: int = eqx.field(static=True, default=12)
    n_post: int = eqx.field(static=True, default=12)
    post_grading: float = eqx.field(static=True, default=2.0)

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


def day_edges(ts0: Array, has_event: Array, cfg: DayConfig, dtype: Any) -> tuple[Array, Array, Array]:
    """``(t_pre[n_pre+1], t_post[n_post+1], t_event)``: sub-step edges [h] around the event time.

    Source: plan 19 A6 (private design note); RZWQM2 ``PHYSCL`` (the storm starts a new period).
    """
    n_sub = cfg.n_pre + cfg.n_post
    t_dry = jnp.asarray(HOURS_PER_DAY * cfg.n_pre / n_sub, dtype)
    t_ev = jnp.where(has_event, jnp.clip(jnp.asarray(ts0, dtype), 0.0, HOURS_PER_DAY), t_dry)
    t_ev = jnp.where(cfg.n_pre > 0, t_ev, 0.0)  # static choice: no pre-event segment
    t_pre = t_ev * jnp.linspace(0.0, 1.0, cfg.n_pre + 1, dtype=dtype)
    u = jnp.linspace(0.0, 1.0, cfg.n_post + 1, dtype=dtype)
    u_graded = u**cfg.post_grading  # u in [0, 1], post_grading >= 1
    graded = jnp.where(has_event, u_graded, u)
    t_post = t_ev + (HOURS_PER_DAY - t_ev) * graded
    t_post = t_post.at[-1].set(HOURS_PER_DAY)
    return t_pre, t_post, t_ev


def soil_water_day_kernel(
    water: SoilWater,
    params: SoilWaterDayParams,
    supply: Array,
    evaporation: Array,
    uptake: Array | SinkChannels,
    storm: StormForcing,
) -> tuple[SoilWater, GAResult, Array]:
    """One day with the Green-Ampt event: returns ``(new state with the day's fluxes, event, event_shift)``.

    ``uptake`` is the sink-channel record, or the per-layer root water uptake [cm d-1] alone.

    Source: Ahuja et al. (2000) ch. 3; RZWQM2 ``PHYSCL`` / ``EVNTRO`` / ``RICHRD`` (conventions).
    """
    rp = params.richards
    cfg = params.config
    dtype = water.theta.dtype
    n = rp.grid.n_node
    soil = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,)), rp.soil.at_nodes())
    depth = jnp.asarray(storm.depth, dtype)
    has_event = jnp.sum(depth) > 0.0
    t_pre, t_post, t_ev = day_edges(storm.ts0, has_event, cfg, dtype)
    alphas = day_alphas(cfg.n_pre + cfg.n_post, rp.config, dtype)
    supply = jnp.asarray(supply, dtype)
    uptake = as_sink_channels(uptake)

    w_pre, tot_pre = richards_substeps(water, rp, t_pre, supply, evaporation, uptake, alphas[: cfg.n_pre])
    ev = green_ampt_event(
        w_pre.theta,
        w_pre.h,
        soil,
        rp.grid.tl,
        jnp.asarray(params.infiltration.aef, dtype),
        storm.duration,
        depth,
        params.infiltration.config,
    )
    w_ev = w_pre.replace(theta=ev.theta, h=ev.h)
    w_post, tot_post = richards_substeps(w_ev, rp, t_post, supply, evaporation, uptake, alphas[cfg.n_pre :])

    def both(name: str) -> Array:
        return getattr(tot_pre, name) + getattr(tot_post, name)

    tot = tot_pre._replace(sinks=both("sinks"), sinks_cut=both("sinks_cut"))

    w0 = water.storage(rp.grid) + water.pond
    w1 = w_post.storage(rp.grid) + w_post.pond
    drainage = both("drainage") + ev.seepage
    runoff = both("runoff") + ev.runoff
    sinks = both("uptake") + other_sinks(tot)
    balance = (w1 - w0) - (both("supply") + ev.rain - both("evaporation") - drainage - sinks - runoff)
    flux = _zero_fluxes(dtype).replace(
        infiltration=both("infiltration") + ev.infiltration,
        evaporation=both("evaporation"),
        drainage=drainage,
        uptake=both("uptake"),
        runoff=runoff,
        evaporation_deficit=both("evaporation_deficit"),
        uptake_cut=both("uptake_cut"),
        balance_error=balance,
        max_theta_residual=jnp.maximum(tot_pre.max_theta_residual, tot_post.max_theta_residual),
        n_clamp=both("n_clamp"),
        rain=ev.rain,
        event_infiltration=ev.infiltration,
        event_runoff=ev.runoff,
        seepage=ev.seepage,
        **sink_fluxes(tot),
    )
    shift = jnp.where(has_event, t_ev - jnp.asarray(storm.ts0, dtype), 0.0)
    return w_post.replace(flux=flux), ev, shift


def _checked(h: Array, where: str) -> Array:
    if check_enabled():
        return eqx.error_if(h, ~jnp.all(jnp.isfinite(h)), f"{where}: non-finite head")
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
        "ponded irrigation off (L-GA+); snow is on at CA-TPA (PRMS, 10-25 cm melt per year), its melt "
        "events are left out of this kernel for now and snow is added in the M3 assembly (L-snow)",
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
)


@process(
    reads=("soil_water.h", "soil_water.theta", "soil_water.pond"),
    writes=("soil_water",),
    source="Ahuja et al. (2000) RZWQM ch. 3; RZWQM2 Rzday.for PHYSCL, RZTEST.for EVNTRO/INFIL",
    fortran_name="PHYSCL",
    key="soil_water/day@rzwqm2-4.6:faithful",
    provenance="reference_only_conventions",
    grid="rzwqm2_nodes",
    ref_build="RZWQM2 4.6 main_ryzen5_avx512",
    sources=_SOURCES,
    deviates=_DEVIATES_GA,
)
def soil_water_day(
    state: RichardsState, params: SoilWaterDayParams, forcing_t: SoilWaterDayForcing
) -> RichardsState:
    """One soil-water day: redistribution, the day's Green-Ampt event, redistribution.

    Thin wrapper of :func:`soil_water_day_kernel`; writes the new ``soil_water`` sub-tree with
    the day's fluxes (event terms in ``rain``, ``event_infiltration``, ``event_runoff``,
    ``seepage``). Under ``AGRI_JAX_CHECK=1`` a non-finite head raises.

    Source: Ahuja et al. (2000) ch. 3; RZWQM2 ``PHYSCL`` (``Rzday.for``), ``EVNTRO``/``INFIL``
    (``RZTEST.for``).
    """
    new, _, _ = soil_water_day_kernel(
        state.soil_water, params, forcing_t.supply, forcing_t.evaporation, forcing_t.uptake, forcing_t.storm
    )
    h = _checked(new.h, "soil_water_day")
    return eqx.tree_at(lambda s: s.soil_water, state, new.replace(h=h))


@process(
    reads=("soil_water.h", "soil_water.theta", "soil_water.pond"),
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
            "the M1 configuration: it separates redistribution error from infiltration error",
            "tests/integration/test_richards_catpa.py (M1); tests/unit/test_infiltration_day.py",
        ),
    ),
)
def soil_water_day_replay(
    state: RichardsState, params: SoilWaterDayParams, forcing_t: SoilWaterDayForcing
) -> RichardsState:
    """The M1 day on the day forcing: :func:`richards_day` with ``supply``; the storm is not used.

    Source: Ahuja et al. (2000) ch. 3; Celia et al. (1990); RZWQM2 ``RICHRD`` (``Rzrich.for``).
    """
    new = richards_day(
        state.soil_water, params.richards, forcing_t.supply, forcing_t.evaporation, forcing_t.uptake
    )
    h = _checked(new.h, "soil_water_day_replay")
    return eqx.tree_at(lambda s: s.soil_water, state, new.replace(h=h))


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
    )
    flux = w.flux.replace(
        rain=ev.rain, event_infiltration=ev.infiltration, event_runoff=ev.runoff, seepage=ev.seepage
    )
    new = w.replace(theta=ev.theta, h=_checked(ev.h, "infiltration_ga"), flux=flux)
    return eqx.tree_at(lambda s: s.soil_water, state, new)
