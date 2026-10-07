r"""Implicit mixed-form Richards equation for the soil-water redistribution step (RZWQM2 style).

One day of vertical unsaturated flow on a vertex-centred finite-volume grid. The physics (state,
grid, fluxes, residual, Jacobian, sinks, switch predicates, convention hooks, balance accumulators)
is :mod:`~agrijax.processes.soil_water.problem` (``RichardsProblem``, ``PROBLEM_VERSION``); the time
integration is an integrator of :mod:`~agrijax.processes.soil_water.integrator`, chosen by the
config in ``RichardsParams.stepping``: :class:`~agrijax.processes.soil_water.fixed_cn.FixedStepping`
(the default: a fixed number of sub-steps, each solved with a fixed number of Newton or modified
Picard iterations and a tridiagonal linear solve) or
:class:`~agrijax.processes.soil_water.richards_adaptive.AdaptiveStepping` (adaptive sub-steps, each
solved by Newton to a tolerance). This module holds the parameters, the process and the
prescribed-supply day (:func:`richards_day`), which calls the integrator only through its protocol,
and re-exports the names of both layers. The retention and conductivity curves are the modified
Brooks-Corey functions of :mod:`agrijax.processes.soil_water.hydraulics`.

Conventions
-----------
* depth ``z`` [cm] positive downward, ``z = 0`` at the surface; matric head ``h`` [cm]
  negative when unsaturated; total head ``H = h - z``;
* volumetric flux ``q`` [cm h-1] **positive downward** (infiltration and drainage > 0,
  evaporation < 0); Darcy-Buckingham ``q = -K(h) (dh/dz - 1)``, so a uniform head drains
  downward at ``q = +K``;
* continuity (mixed form, theta is the conserved variable) ``d theta / dt = -dq/dz - S`` with
  the root-water-uptake sink ``S`` [h-1] positive for extraction;
* internal units cm and **hours** (``ksat`` in cm h-1, sub-step ``dt`` in h); daily inputs are
  converted in the process wrapper.

Grid
----
``n`` nodes with cell thicknesses ``tl[n]`` (RZWQM ``TL``) and node spacings ``delz[n-1]``
(RZWQM ``DELZ``); the two are different arrays on a non-uniform grid and are never mixed.
Faces: ``q[0]`` is the surface flux, ``q[i]`` the flux between nodes ``i-1`` and ``i``
(0-based), ``q[n]`` the bottom flux. Storage is ``W = sum(theta * tl)``, the convention of
``.ana`` column 2 and ``LAYER.PLT``. An upper ghost node lies ``dz_top`` above node 0
(RZWQM: ``DELZ(1)``); boundary conditions act on the ghost-node / node-0 segment.

Discrete equations (Celia et al. 1990 mixed form)
-------------------------------------------------
For node ``i``, time level ``n -> n+1`` and time weight ``alpha``
(``h~ = alpha h^{n+1} + (1 - alpha) h^n``)::

    R_i(h) = tl_i (theta_i(h_i) - theta_i^n) / dt + (q_{i+1/2} - q_{i-1/2}) + tl_i S_i = 0
    q_{i+1/2} = -K_{i+1/2} ((h~_{i+1} - h~_i) / delz_i - 1),
    K_{i+1/2} = sqrt(K_i(h~_i) K_{i+1}(h~_{i+1}))          (geometric mean, as RZWQM)

Time weights: ``alpha = 1`` on every sub-step (``time_scheme="implicit"``, the fixed default), or
RZWQM2's pattern ``alpha = 1`` on the first sub-step of a day and ``1/2`` afterwards
(``"rzwqm"``, for per-call comparison with the reference model). With few iterations the
Crank-Nicolson sub-steps do not damp an unconverged iterate and can oscillate; the
convergence study (``tests/unit/test_richards.py``) quantifies both.

The storage term is ``theta(h^{n+1}) - theta^n``, not ``C (h^{n+1} - h^n)``, and the new water
content is ``theta(h^{n+1})`` of the final iterate. The water balance of a sub-step is then
``dW - (q_top - q_bot - sum tl S) dt = dt sum R_i``: it closes to rounding at a converged
root, and an unconverged solve reports its imbalance (``balance_error``) and the worst node
residual ``dt |R_i| / tl_i`` (``max_theta_residual``). A flux-conservative update
(``theta^{n+1} = theta^n - dt/tl (q_{i+1/2} - q_{i-1/2}) - dt S``) closes the balance
by construction but, with 1-2 iterations, leaves ``theta`` outside ``[theta_r, theta_s]`` after
intense rain and was measured to diverge on CA-TPA 2015, so it is not offered.

Iteration: Newton on the transformed variable ``v`` (:func:`head_of_v`: ``log(hb/|h|)`` on
the Brooks-Corey segment, linear in ``h`` above ``-hb``), with the exact tridiagonal Jacobian
of ``R`` (including ``dK/dh``) assembled from three JVPs with the seeds ``[i % 3 == k]``
(each row has three non-zeros, one per colour); ``jacobian="picard"`` freezes ``K`` at the
current iterate (RZWQM's modified Picard). The iteration is damped without touching the
residual (so the converged root is unchanged): a storage floor ``c_floor`` on the Jacobian
diagonal keeps it non-singular on a fully saturated profile (``C = 0`` and ``dK/dh = 0``);
each update is limited to ``|dv| <= dv_max`` (a factor ``e`` in ``|h|`` by default); an update
that leaves the saturated side across the air-entry kink ``h = -hb`` stops on the kink
(``chop``, by analogy with the trust regions of Wang & Tchelepi 2013, who cut Newton updates of
two-phase transport back at the inflection, unit-flux and end points of the fractional-flow
function; here the only boundary is the air-entry kink of the retention curve),
which removes the alternation across the kink while a saturated surface drains; and the
iterate is clamped to ``[h_min, h_upper + z_i]`` (a saturated layered profile carries
positive heads that grow with depth, ~24 cm at CA-TPA under ponding). Clamp activations
are counted (``n_clamp``) and should be zero.

Sub-steps (``FixedStepping``): ``n_sub`` per day. A fraction ``rain_fraction`` of them is placed
in proportion to the hourly supply, the rest spread uniformly (:func:`substep_edges`); the schedule depends
on the forcing only, so the count stays fixed while a wetting front moves by less than a
node per sub-step.

Boundary conditions
-------------------
*Bottom*: free drainage (RZWQM ``IREBOT = 2``, unit gradient): ``q_bot = K(h~_n)``.

*Top*: the requested surface flux is ``q_d = w - e`` with ``w = supply + pond/dt`` the water
available at the surface and ``e`` the potential soil-evaporation demand. Two head
(Dirichlet) limits on the ghost node bound it (RZWQM ``CHKBC``):

* ponding, ghost head ``0``: ``q_wet = -K_s ((h~_0 - 0)/dz_top - 1)``, the infiltration
  capacity, with the upstream (saturated ghost) conductivity: with the geometric mean the
  capacity would grow as a dry surface node wets and Newton would be driven to the dry end;
* dry end, ghost head ``h_min``: ``q_dry = -sqrt(K(h_min) K(h~_0)) ((h~_0 - h_min)/dz_top - 1)``
  (geometric mean as RZWQM ``CHKBC``/``NODFLX``), the largest evaporation the profile can
  supply;

and ``q_top = q_wet`` where ``q_d > q_wet``, ``q_dry`` where ``q_d < q_dry``, else ``q_d``
(selected with ``jnp.where`` inside the residual, both branches finite). Supply that
cannot infiltrate goes to the pond up to ``pond_max`` [cm] and runs off above it
(``pond_max = 0``, the default, is RZWQM's "no ponding layer in Richards"); an evaporation
demand the soil cannot meet is reported as ``evaporation_deficit``. The surface flux is
always diagnosed from the final heads.

Note: RZWQM2 does not use the rain as the upper boundary flux of the Richards step: an
infiltration event fills the profile and Richards only redistributes (Ahuja et al. 2000, ch. 3).
That event is not part of this package; an integrator takes one through its day plan
(:class:`~agrijax.processes.soil_water.integrator.DayPlan`). In this module's own day
(:func:`richards_day`, the prescribed-supply configuration) ``supply`` is the
water that *infiltrates* (e.g. ``.ana`` column 5), spread over the rain event.

Sink: an input, a typed record of sink channels (:mod:`~agrijax.processes.soil_water.sinks`:
``uptake``, ``tile``, ``lateral``, ``subirrigation``, ``macropore_to_drain``), each a per-layer
daily amount [cm d-1] spread uniformly over the day and/or a per-sub-step callable. A plain array
is the per-layer root water uptake (the only channel of the prescribed-supply day and of the coupled
model). Each channel is capped per sub-step by the water above ``theta(h_min)`` in the node (RZWQM:
no uptake at ``Hmin``), the channels in turn, and the cut is reported (``uptake_cut``, ``sink_cut``). Each
channel has its own daily total in :class:`SoilWaterFluxes` and its own term in ``balance_error``.

Gradients (``FixedStepping.grad``): ``"unrolled"`` differentiates through the fixed iterations;
``"implicit"`` gives each sub-step solve an implicit-function-theorem VJP
(``J(h*)^T lambda = g``, tridiagonal, then the VJP of ``R`` with respect to the inputs),
which is exact at a converged root and independent of the iteration path.

Source: Ahuja, L.R., Rojas, K.W., Hanson, J.D., Shaffer, M.J., Ma, L. (eds.), 2000. Root
Zone Water Quality Model, ch. 3 (Richards equation, boundary switching, free drainage);
Celia, M.A., Bouloutas, E.T., Zarba, R.L., 1990. A general mass-conservative numerical
solution for the unsaturated flow equation. Water Resour. Res. 26, 1483-1496. Wang, X.,
Tchelepi, H.A., 2013. Trust-region based solver for nonlinear transport in heterogeneous
porous media. J. Comput. Phys. 253, 114-137 (the analogy behind ``chop``). The
corresponding RZWQM2 subroutines are ``RICHRD``, ``CHKBC``, ``NODFLX`` and ``POINTK``
(``Rzrich.for``).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.process import check_enabled, process
from agrijax.core.state import Forcing, Params, State, field

from .fixed_cn import (
    FixedCN,
    FixedStepping,
    _iterate,
    _solve_implicit,
    _SolveCfg,
    day_alphas,
    day_edges,
    richards_step,
    richards_substeps,
    substep_edges,
)
from .hydraulics import H_CLAMP_RZWQM, AnyHydraulicParams
from .integrator import (
    _ALPHA_CN,
    _ALPHA_FIRST,
    DayDiagnostics,
    DayPlan,
    RichardsIntegrator,
    SteppingConfig,
    integrator_for,
    integrator_info,
)
from .problem import (
    _CELL_CENTRE,
    _GEOMETRIC_MEAN,
    _N_COLOURS,
    _N_HOUR_EDGES,
    BC_CLIP,
    BC_FLUX,
    BC_PONDED,
    DT_START,
    HOURS_PER_DAY,
    PROBLEM_VERSION,
    RichardsConfig,
    RichardsGrid,
    RichardsProblem,
    SoilWater,
    SoilWaterFluxes,
    StepResult,
    SubstepTotals,
    _dh_dv,
    _face_fluxes,
    _interval_means,
    _log_k,
    _step_args,
    _step_result,
    _StepArgs,
    _StepSinks,
    _tridiag_solve,
    _zero_fluxes,
    head_of_v,
    other_sinks,
    richards_residual,
    sink_fluxes,
    surface_fluxes,
    tridiagonal_jacobian,
    v_of_head,
)
from .richards_adaptive import DT_MAX_FAST, AdaptiveCN, AdaptiveStepping
from .sinks import SinkChannels

__all__ = [
    "DT_MAX_FAST",
    "DT_START",
    "HOURS_PER_DAY",
    "PROBLEM_VERSION",
    "AdaptiveCN",
    "AdaptiveStepping",
    "FixedCN",
    "FixedStepping",
    "RichardsConfig",
    "RichardsForcing",
    "RichardsGrid",
    "RichardsParams",
    "RichardsProblem",
    "RichardsState",
    "SoilWater",
    "SoilWaterFluxes",
    "StepResult",
    "SubstepTotals",
    "day_alphas",
    "day_edges",
    "day_fluxes",
    "head_of_v",
    "other_sinks",
    "richards_day",
    "richards_day_with",
    "richards_redistribution",
    "richards_residual",
    "richards_step",
    "richards_substeps",
    "sink_fluxes",
    "substep_edges",
    "surface_fluxes",
    "tridiagonal_jacobian",
    "v_of_head",
]

#: names of the two layers kept importable from this module (existing callers and the tests)
_REEXPORTED = (
    _ALPHA_CN, _ALPHA_FIRST, _CELL_CENTRE, _GEOMETRIC_MEAN, _N_COLOURS, _N_HOUR_EDGES, BC_CLIP, BC_FLUX,
    BC_PONDED, _dh_dv, _face_fluxes, _interval_means, _iterate, _log_k, _solve_implicit, _SolveCfg,
    _step_args, _step_result, _StepArgs, _StepSinks, _tridiag_solve,
)  # fmt: skip


# ---------------------------------------------------------------------------
# parameters, state, forcing
# ---------------------------------------------------------------------------


class RichardsParams(Params):
    """Parameters of the Richards redistribution process.

    ``config`` the settings of the problem (:class:`~agrijax.processes.soil_water.problem.RichardsConfig`),
    ``stepping`` the integrator's config, a tagged union (its class selects the registered
    integrator, :func:`~agrijax.processes.soil_water.integrator.integrator_for`): the default
    :class:`~agrijax.processes.soil_water.fixed_cn.FixedStepping` (24 x 3) or
    :class:`~agrijax.processes.soil_water.richards_adaptive.AdaptiveStepping`.
    """

    soil: AnyHydraulicParams
    grid: RichardsGrid
    h_min: Array = field(
        dims=(),
        unit="cm",
        description="dry-end head limit (ghost head of the dry BC, clamp)",
        fortran_name="HMIN",
        default=H_CLAMP_RZWQM,
    )
    pond_max: Array = field(
        dims=(), unit="cm", description="largest surface ponding depth before runoff (0 = RZWQM)", default=0.0
    )
    config: RichardsConfig = eqx.field(static=True, default=RichardsConfig())
    stepping: SteppingConfig = eqx.field(static=True, default=FixedStepping())

    def __check_init__(self) -> None:
        if not isinstance(self.config, RichardsConfig):
            raise TypeError(f"RichardsParams.config is a RichardsConfig, got {type(self.config).__name__}")
        if not isinstance(self.stepping, SteppingConfig):
            raise TypeError(
                "RichardsParams.stepping is an integrator config (FixedStepping, AdaptiveStepping, ...), "
                f"got {type(self.stepping).__name__}"
            )
        integrator_info(self.stepping)  # a registered integrator (KeyError otherwise)


class RichardsState(State):
    """Minimal model state for the Richards process: ``state.soil_water``."""

    soil_water: SoilWater


class RichardsForcing(Forcing):
    """One day's water inputs (time axis first when stacked over days).

    ``supply`` and ``evaporation`` are hourly rates for the 24 hours of the day; each
    sub-step uses their average over its interval. ``uptake`` is the root water uptake of each
    layer for the day [cm d-1] (e.g. ``LAYER.PLT`` PLANT WATER UPTAKE), spread uniformly.
    """

    supply: Array = field(unit="cm h-1", dims=("T", "hour"), description="water arriving at the surface")
    evaporation: Array = field(unit="cm h-1", dims=("T", "hour"), description="potential soil evaporation")
    uptake: Array = field(unit="cm d-1", dims=("T", "n_node"), description="root water uptake per layer")


# ---------------------------------------------------------------------------
# the prescribed-supply day
# ---------------------------------------------------------------------------


def day_fluxes(tot: SubstepTotals, stats: Any, dtype: Any, **fields: Array) -> SoilWaterFluxes:
    """The day's :class:`SoilWaterFluxes` (``dtype``) from the integrator's totals and counters
    (``stats``, ``None`` or a NamedTuple of diagnostic fields) and the other fields given
    (``infiltration``, ``drainage``, ``runoff``, ``balance_error`` and the event terms).

    Source: Ahuja et al. (2000) ch. 3 (water balance).
    """
    return _zero_fluxes(dtype).replace(
        evaporation=tot.evaporation,
        uptake=tot.uptake,
        evaporation_deficit=tot.evaporation_deficit,
        uptake_cut=tot.uptake_cut,
        max_theta_residual=tot.max_theta_residual,
        n_clamp=tot.n_clamp,
        **fields,
        **sink_fluxes(tot),
        **({} if stats is None else stats._asdict()),
    )


def richards_day(
    water: SoilWater,
    params: RichardsParams,
    supply: Array,
    evaporation: Array,
    uptake: Array | SinkChannels,
) -> SoilWater:
    """One day of Richards redistribution by the integrator of ``params.stepping``: ``n_sub`` sub-steps
    placed by :func:`substep_edges` (``FixedStepping``), or adaptive sub-steps
    (``AdaptiveStepping``, :mod:`~agrijax.processes.soil_water.richards_adaptive`).

    ``supply``/``evaporation`` hourly rates ``[24]`` [cm h-1], ``uptake`` the sink channels or
    the per-layer root water uptake [cm d-1] alone.
    Returns the new state with the day's totals in ``flux``. This is the prescribed-supply day (the
    surface supply is prescribed; no infiltration event).

    Source: Ahuja et al. (2000) ch. 3; Celia et al. (1990); RZWQM2 ``RICHRD`` (``Rzrich.for``).
    """
    integrator = integrator_for(params.stepping)
    return richards_day_with(integrator, water, params, supply, evaporation, uptake)


def richards_day_with(
    integrator: RichardsIntegrator,
    water: SoilWater,
    params: RichardsParams,
    supply: Array,
    evaporation: Array,
    uptake: Array | SinkChannels,
) -> SoilWater:
    """:func:`richards_day` with an explicit integrator (one not in the registry, such as a conformance
    fixture); ``params.stepping`` is not read.

    Source: Ahuja et al. (2000) ch. 3; Celia et al. (1990); RZWQM2 ``RICHRD`` (``Rzrich.for``).
    """
    dtype = water.theta.dtype
    problem = RichardsProblem.of_day(params, supply, evaporation, uptake, dtype)
    new, diag = integrator.step_day(problem, water, DayPlan())
    return new.replace(flux=_m1_fluxes(water, new, diag, params))


def _m1_fluxes(
    water: SoilWater, new: SoilWater, diag: DayDiagnostics, params: RichardsParams
) -> SoilWaterFluxes:
    tot = diag.totals
    w0 = water.storage(params.grid) + water.pond
    w1 = new.storage(params.grid) + new.pond
    sinks = tot.uptake + other_sinks(tot)
    balance = (w1 - w0) - (tot.supply - tot.evaporation - tot.drainage - sinks - tot.runoff)
    return day_fluxes(
        tot,
        diag.stats,
        water.theta.dtype,
        infiltration=tot.infiltration,
        drainage=tot.drainage,
        runoff=tot.runoff,
        balance_error=balance,
    )


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------


@process(
    reads=("soil_water.h", "soil_water.theta", "soil_water.pond", "soil_water.dt_next"),
    writes=("soil_water",),
    source="Ahuja et al. (2000) RZWQM ch. 3; Celia et al. (1990); RZWQM2 Rzrich.for RICHRD",
    fortran_name="RICHRD",
    key="soil_water/richards@rzwqm2-4.6:faithful",
    provenance="reference_only_conventions",
    grid="rzwqm2_nodes",
    ref_build="RZWQM2 4.6 main_ryzen5_avx512",
    sources=(
        ("mixed-form residual, theta(h) storage term", "Celia, Bouloutas & Zarba (1990) WRR 26, 1483-1496"),
        ("Richards equation, boundary switching, free drainage", "Ahuja et al. (2000) RZWQM, ch. 3"),
        ("modified Brooks-Corey theta(h), K(h)", "Ahuja et al. (2000) ch. 3 (hydraulics.py)"),
        (
            "geometric-mean face K, CHKBC head limits, IREBOT=2 bottom, no uptake at Hmin",
            "RZWQM2 RICHRD / CHKBC / NODFLX / POINTK (Rzrich.for), read for conventions",
        ),
    ),
    deviates=(
        (
            "the surface supply is the water that infiltrates; an infiltration event is not part of it",
            "RZWQM2 fills the profile with an infiltration event and Richards only redistributes; the event "
            "is outside this process (an integrator takes it through its day plan)",
            "richards.py module docstring; tests/integration/test_richards_catpa.py (.ana column 5)",
        ),
        (
            "ponded infiltration capacity uses the upstream conductivity K_s, not the geometric mean",
            "the geometric mean makes the capacity grow as a dry surface node wets and sends Newton to the "
            "dry end; RZWQM2 never meets the case because its rain goes through INFIL",
            "richards.py comment at the ponded limit in the surface-flux function",
        ),
        (
            "default time_scheme='implicit' (alpha = 1 on every sub-step) and Newton on a transformed head; "
            "RZWQM2 uses alpha = 1/2 after the first sub-step and modified Picard",
            "stable with a fixed small iteration count; time_scheme='rzwqm', jacobian='picard' give the "
            "reference scheme",
            "tests/unit/test_richards.py convergence study; the RICHRD dump comparison (data tier outside "
            "this repository)",
        ),
    ),
)
def richards_redistribution(
    state: RichardsState, params: RichardsParams, forcing_t: RichardsForcing
) -> RichardsState:
    """One day of implicit mixed-form Richards redistribution with supply-limited surface fluxes.

    Thin wrapper of :func:`richards_day`: reads the node heads, water contents and pond,
    writes the new ``soil_water`` sub-tree including the day's fluxes. Under
    ``AGRI_JAX_CHECK=1`` a non-finite head raises (``equinox.error_if``), because a NaN head
    would otherwise be read as saturation by the ``h < -hb`` branch tests; with an integrator that
    tests convergence (the adaptive one) an unconverged sub-step or an exhausted step budget raises too.

    Source: Ahuja et al. (2000) ch. 3; Celia et al. (1990); RZWQM2 ``RICHRD`` (``Rzrich.for``).
    """
    new = richards_day(state.soil_water, params, forcing_t.supply, forcing_t.evaporation, forcing_t.uptake)
    h = new.h
    if check_enabled():
        h = eqx.error_if(h, ~jnp.all(jnp.isfinite(h)), "richards_redistribution: non-finite head")
        h = integrator_for(params.stepping).check(h, new.flux, "richards_redistribution")
    return eqx.tree_at(lambda s: s.soil_water, state, new.replace(h=h))
