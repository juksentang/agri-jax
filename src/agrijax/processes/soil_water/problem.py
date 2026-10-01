r"""The Richards problem: the physics of one soil-water day, separate from its time integration.

This module is the **physics layer** of the Richards redistribution: what is solved, with no choice
of how (the time integration is a layer of its own, :mod:`~agrijax.processes.soil_water.integrator`).
It defines, once,

* the state and its conserved variable: the node water content ``theta`` [cm3 cm-3] with the
  storage ``W = sum(theta tl)`` [cm], the matric head ``h`` [cm] with ``theta = theta(h)`` (the
  modified Brooks-Corey curves of :mod:`~agrijax.processes.soil_water.hydraulics`), the surface pond
  (:class:`SoilWater`, :class:`SoilWaterFluxes`);
* the grid (:class:`RichardsGrid`) and the settings of the problem (:class:`RichardsConfig`: the
  sink cutoff at ``h_min`` and the two RZWQM2 conventions);
* the semi-discrete flux balance of a node, ``tl_i d theta_i / dt = q_{i-1/2} - q_{i+1/2} -
  tl_i S_i``, with the face fluxes (:func:`surface_fluxes`, :func:`face_fluxes`: Darcy-Buckingham
  with the geometric-mean face conductivity, free drainage at the bottom, the requested surface
  flux bounded by its ponded and dry-end head limits), the sinks capped at ``h_min``
  (:func:`step_args`), and the residual of the one-step theta-method family
  ``R_i = tl_i (theta_i(h) - theta_i^n)/dt + (q_{i+1/2} - q_{i-1/2}) + tl_i S_i`` evaluated at the
  time-weighted head ``alpha h + (1 - alpha) h^n`` (:func:`richards_residual`; ``alpha`` is the
  integrator's choice, the equation is the problem's);
* its exact tridiagonal Jacobian (three JVPs, :func:`tridiagonal_jacobian`), the capacity ``C(h)``
  (:func:`capacity`) and the transformed head of the Newton iterations (:func:`head_of_v`);
* the switch predicates of the surface condition (:func:`surface_switches`: ponding, i.e. the
  infiltration-capacity limit binds, the dry-end evaporation limit, nodes at ``h_min``) and the
  residual under a fixed surface condition (:func:`residual_bc`, RZWQM2 ``CHKBC``);
* the post-step convention hooks (:func:`post_step`: RZWQM2's DRAIN cap with the head recomputed on
  the changed nodes; the flux-mode evaporation limit is a dry limit of :func:`surface_fluxes`), both
  switched by :class:`RichardsConfig` and selected by the registered convention variants of the
  soil-water day;
* the accumulators of the water balance: per step :class:`StepResult` (``int q_top dt``,
  ``int q_bot dt``, ``int sum tl S dt`` and the step balance ``sum tl dtheta - dt (q_top - q_bot -
  sum tl S)``), per run of steps :class:`SubstepTotals`.

:class:`RichardsProblem` bundles the problem of one day (soil on the node axis, grid, limits, the
hourly surface supply and evaporation demand, the sink channels, the field-saturated porosity of
the DRAIN cap) with these functions as methods; an integrator
(:mod:`~agrijax.processes.soil_water.integrator`) takes it and advances the state over the day.

``PROBLEM_VERSION`` is the version of this definition. It changes whenever the output of the physics
can change (an equation, a limit, a convention, a sink rule), never for a change of an integrator;
the integrator conformance of :mod:`agrijax.testing.conformance.integrators` records it, and
``tests/unit/test_richards_integrators.py`` pins the problem's residual and fluxes on a frozen input
to it.

Conventions, grid, discrete equations and boundary conditions are those of the module docstring of
:mod:`~agrijax.processes.soil_water.richards`.

Source: Ahuja, L.R., Rojas, K.W., Hanson, J.D., Shaffer, M.J., Ma, L. (eds.), 2000. Root Zone Water
Quality Model, ch. 3 (Richards equation, boundary switching, free drainage, field saturation);
Celia, M.A., Bouloutas, E.T., Zarba, R.L., 1990. A general mass-conservative numerical solution for
the unsaturated flow equation. Water Resour. Res. 26, 1483-1496 (mixed form); Curtis, A.R., Powell,
M.J.D., Reid, J.K., 1974. On the estimation of sparse Jacobian matrices. IMA J. Appl. Math. 13,
117-119 (column colouring). The corresponding RZWQM2 subroutines are ``RICHRD``, ``CHKBC``,
``NODFLX``, ``POINTK`` (``Rzrich.for``) and ``DRAIN`` (``Rzday.for``), read for conventions only.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, ClassVar, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array

from agrijax.core import units
from agrijax.core.coefficients import numerical_guard
from agrijax.core.state import Params, State, field

from .coefficients import numerical_setting, rzwqm2, setting_field
from .conventions import drain_cap, dry_end_eps, field_saturation, flux_peak_head
from .hydraulics import AnyHydraulicParams, h_of_theta, k_of_h, theta_of_h
from .sinks import SINK_CHANNELS, SinkChannels, as_sink_channels

__all__ = [
    "BC_CLIP",
    "BC_FLUX",
    "BC_PONDED",
    "DT_START",
    "HOURS_PER_DAY",
    "PROBLEM_VERSION",
    "RichardsConfig",
    "RichardsGrid",
    "RichardsProblem",
    "SoilWater",
    "SoilWaterFluxes",
    "StepArgs",
    "StepResult",
    "StepSinks",
    "SubstepTotals",
    "SurfaceSwitches",
    "capacity",
    "drain_fluxes",
    "face_fluxes",
    "head_of_v",
    "node_pori",
    "other_sinks",
    "post_step",
    "residual_bc",
    "richards_residual",
    "sink_fluxes",
    "step_args",
    "step_result",
    "surface_fluxes",
    "surface_switches",
    "top_fluxes",
    "tridiagonal_jacobian",
    "v_of_head",
]

#: version of the problem definition: bumped whenever the physics output can change (module docstring)
PROBLEM_VERSION: int = 1

#: hours of the redistribution day (the unit conversion of :mod:`agrijax.core.units`)
HOURS_PER_DAY: float = units.HOURS_PER_DAY
#: hour boundaries 0..24 of the hourly forcing (a count, not a model number)
_N_HOUR_EDGES: int = int(HOURS_PER_DAY) + 1
#: floor of K inside the logarithm of the geometric mean [cm h-1]; K(h_min) is ~1e-9 cm/h for CA-TPA.
_K_FLOOR: float = numerical_guard(
    "richards.k_floor", 1.0e-300, "floor of K [cm h-1] inside the log of the geometric mean (float64)"
)
_K_FLOOR_F32: float = numerical_guard(
    "richards.k_floor_f32", 1.0e-37, "floor of K [cm h-1] inside the log of the geometric mean (float32)"
)
_GRID_ATOL: float = numerical_guard(
    "richards.grid_atol", 1.0e-9, "absolute tolerance [cm] of the vertex-centred grid check of rzwqm.dat"
)
_CELL_CENTRE: float = numerical_setting(
    "richards.cell_centre",
    0.5,
    "-",
    "vertex-centred grid: the first node at half the first cell, cell boundaries at the midpoints "
    "between nodes (TL(i) = (DELZ(i-1) + DELZ(i)) / 2)",
    origin="rzwqm2-4.6",
    provenance=rzwqm2("RZWQM/Rzmain.for:5921", "INPUT", note="ZN(1) and TL(i) at Rzmain.for:5921-5929"),
)
_GEOMETRIC_MEAN: float = numerical_setting(
    "richards.face_k_mean_exponent",
    0.5,
    "-",
    "face conductivity is the geometric mean exp(0.5 (log K_i + log K_i+1)) of the two nodes",
    origin="rzwqm2-4.6",
    provenance=rzwqm2("RZWQM/Rzrich.for:322", "CNHEAD", note="HKBAR, also Rzrich.for:350"),
)
#: colours of the tridiagonal band: row i couples to i-1, i, i+1, so seeds [i % 3 == k] recover it
_N_COLOURS: int = numerical_setting(
    "richards.jacobian_colours",
    3,
    "-",
    "JVP seeds of the tridiagonal Jacobian (one per colour of the three-point stencil)",
    origin="agrijax",
    basis="Curtis, Powell & Reid (1974) column colouring; tests/unit/test_richards.py dense jacfwd check",
)
#: sub-step the adaptive stepping starts from before its first day [h] (SoilWater.dt_next)
DT_START: float = numerical_setting(
    "richards.dt_start",
    1.0e-4,
    "h",
    "initial value of the carried adaptive sub-step SoilWater.dt_next",
    origin="rzwqm2-4.6",
    provenance=rzwqm2(
        "RZWQM/Rzday.for:724", "PHYSCL", note="DELT initial value; kept across days (SAVE, :717)"
    ),
)

# ---------------------------------------------------------------------------
# pytrees
# ---------------------------------------------------------------------------


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


class RichardsGrid(Params):
    """Vertex-centred finite-volume grid (RZWQM ``TL``, ``DELZ``)."""

    tl: Array = field(
        dims=("n_node",),
        unit="cm",
        description="cell (layer) thickness, sums to the profile depth",
        fortran_name="TL",
    )
    delz: Array = field(
        dims=("n_node-1",), unit="cm", description="distance between node i and node i+1", fortran_name="DELZ"
    )
    dz_top: Array = field(
        dims=(), unit="cm", description="distance from the upper ghost node to node 0", fortran_name="DELZ(1)"
    )

    @property
    def n_node(self) -> int:
        """Number of nodes."""
        return int(np.shape(self.tl)[-1])

    def node_depth(self) -> Array:
        """Node depths [cm]: ``tl[0]/2`` for the first node, then cumulative ``delz``."""
        z0 = _CELL_CENTRE * self.tl[..., :1]
        return jnp.concatenate([z0, z0 + jnp.cumsum(self.delz, axis=-1)], axis=-1)

    @classmethod
    def from_rzwqm(cls, layer_bottom: Any, node_spacing: Any) -> RichardsGrid:
        """Host-side: build from the rzwqm.dat node records, columns 2 (``TLT``) and 3 (``DELZ``).

        ``tl = diff([0, TLT])``; the file is checked for the vertex-centred rule of
        ``Rzmain.for`` (``TL(i) = (DELZ(i-1) + DELZ(i)) / 2``, ``TL(1) = DELZ(1)``,
        ``TL(n) = DELZ(n-1)``), which makes the cell boundaries the midpoints between nodes.
        NumPy on the records: they must be concrete (NumPy or Python values), never traced.
        """
        tlt = np.asarray(layer_bottom, dtype=float)
        dz = np.asarray(node_spacing, dtype=float)
        if tlt.ndim != 1 or dz.shape != tlt.shape:
            raise ValueError(f"expected two [n] arrays, got {tlt.shape} and {dz.shape}")
        tl = np.diff(np.concatenate([[0.0], tlt]))
        delz = dz[:-1]
        expect = np.concatenate([[delz[0]], _CELL_CENTRE * (delz[:-1] + delz[1:]), [delz[-1]]])
        _require(
            bool(np.allclose(tl, expect, atol=_GRID_ATOL) and np.all(delz > 0)),
            "node records are not a vertex-centred grid (TL != (DELZ(i-1)+DELZ(i))/2)",
        )
        return cls(tl=jnp.asarray(tl), delz=jnp.asarray(delz), dz_top=jnp.asarray(delz[0]))

    @classmethod
    def uniform(cls, n_node: int, dz: float) -> RichardsGrid:
        """``n_node`` cells of thickness ``dz``: nodes at the cell centres, ghost node ``dz`` above node 0."""
        return cls(
            tl=jnp.full((n_node,), float(dz)),
            delz=jnp.full((n_node - 1,), float(dz)),
            dz_top=jnp.asarray(float(dz)),
        )


class RichardsConfig(eqx.Module):
    """Static settings of the Richards *problem* (hashable; part of the compiled program).

    ``sink_cutoff`` [cm3 cm-3] water kept above ``theta(h_min)`` by the sink cap; ``drain_cap`` and
    ``evaporation_limit`` switch the two RZWQM2 conventions of
    :mod:`~agrijax.processes.soil_water.conventions` (off by default; the convention variants of the
    soil-water day set them). These are the settings that change the equations; the numerics of
    the time integration (sub-steps, Newton iterations, damping, tolerances, step control) are the
    settings of the integrator (:class:`~agrijax.processes.soil_water.fixed_cn.FixedStepping`,
    :class:`~agrijax.processes.soil_water.richards_adaptive.AdaptiveStepping`: the ``stepping``
    field of :class:`~agrijax.processes.soil_water.richards.RichardsParams`).

    Every field is a numerical setting with its origin
    (:data:`~agrijax.processes.soil_water.coefficients.SETTINGS`, ``richards.*``).
    """

    sink_cutoff: float = setting_field(
        "richards.sink_cutoff",
        1.0e-9,
        "cm3 cm-3",
        "water kept above theta(h_min) by the sink cap (RZWQM2: no uptake from a node at Hmin)",
        origin="agrijax",
        basis="a node at h_min must give no uptake (as in RZWQM2), otherwise the clamp would swallow the "
        "water; tests/unit/test_richards.py uptake-cap test",
    )
    drain_cap: bool = setting_field(
        "richards.drain_cap",
        False,
        "-",
        "apply RZWQM2's DRAIN cap after every accepted sub-step: water above the field-saturated "
        "porosity aef * theta_s cascades down at once and leaves the bottom node as seepage "
        "(conventions.drain_cap); needs aef (the soil-water day passes GreenAmptParams.aef)",
        origin="agrijax",
        provenance=rzwqm2(
            "RZWQM/Rzrich.for:1092", "REDIST", note="DRAIN (Rzday.for:3975) after every RICHRD step"
        ),
        basis="an RZWQM2 convention kept off in the faithful keys so that their pinned values stay; a "
        "labelled convention variant, measured with and without "
        "(tests/integration/test_richards_conventions_years.py)",
    )
    evaporation_limit: str = setting_field(
        "richards.evaporation_limit",
        "hmin",
        "-",
        "dry limit of the surface flux: 'hmin' (Darcy flux with the ghost head at h_min) or "
        "'flux_peak' (RZWQM2's flux-mode limit: the peak of that flux over the ghost head, "
        "conventions.flux_peak_head)",
        origin="agrijax",
        provenance=rzwqm2(
            "RZWQM/Rzrich.for:59", "CHKBC", note="flux condition while the ghost head stays above Hmin"
        ),
        basis="an RZWQM2 convention kept off in the faithful keys so that their pinned values stay; a "
        "labelled convention variant, measured with and without "
        "(tests/integration/test_richards_conventions_years.py)",
    )

    @property
    def conventions(self) -> tuple[str, ...]:
        """The RZWQM2 conventions switched on: ``"drain_cap"``, ``"flux_peak"`` (empty by default)."""
        on = ("drain_cap",) if self.drain_cap else ()
        return on + (("flux_peak",) if self.evaporation_limit == "flux_peak" else ())

    def __check_init__(self) -> None:
        if self.evaporation_limit not in ("hmin", "flux_peak"):
            raise ValueError(
                f"evaporation_limit must be 'hmin' or 'flux_peak', got {self.evaporation_limit!r}"
            )


class SoilWaterFluxes(State):
    """Daily totals [cm d-1] and diagnostics of one Richards day."""

    infiltration: Array = field(
        dims=(), unit="cm d-1", description="water that entered the soil at the surface", fortran_name="TQF"
    )
    evaporation: Array = field(
        dims=(), unit="cm d-1", description="actual soil evaporation", fortran_name="AEVAP"
    )
    drainage: Array = field(
        dims=(), unit="cm d-1", description="free drainage out of the profile bottom", fortran_name="DEEP"
    )
    uptake: Array = field(
        dims=(), unit="cm d-1", description="actual root water uptake (sink after the h_min cut)"
    )
    runoff: Array = field(
        dims=(), unit="cm d-1", description="supply that neither infiltrated nor stayed ponded"
    )
    evaporation_deficit: Array = field(
        dims=(), unit="cm d-1", description="evaporation demand the soil could not supply"
    )
    uptake_cut: Array = field(
        dims=(), unit="cm d-1", description="uptake removed because a node was at h_min"
    )
    balance_error: Array = field(
        dims=(),
        unit="cm",
        description="d(storage + pond) - (supply + rain - evaporation - drainage - sinks - runoff)",
    )
    max_theta_residual: Array = field(
        dims=(),
        unit="cm3 cm-3",
        description="max |theta - theta(h)| over the day's sub-steps (unconverged iterations)",
    )
    n_clamp: Array = field(
        dims=(), unit="-", description="clamp activations of the head iterate during the day (should be 0)"
    )
    rain: Array = field(
        dims=(), unit="cm d-1", description="storm rain of the day's infiltration event (0 without an event)"
    )
    event_infiltration: Array = field(
        dims=(),
        unit="cm d-1",
        description="part of infiltration that entered through the event",
        fortran_name="CII",
    )
    event_runoff: Array = field(
        dims=(), unit="cm d-1", description="part of runoff produced by the event", fortran_name="ROI"
    )
    seepage: Array = field(
        dims=(),
        unit="cm d-1",
        description="event rain passed through a saturated profile (part of drainage)",
        fortran_name="CDNCI",
    )
    tile: Array = field(
        dims=(),
        unit="cm d-1",
        description="sink channel: tile drainage (0 in the coupled model: no module writes it yet)",
        fortran_name="TOTDRN",
    )
    lateral: Array = field(
        dims=(),
        unit="cm d-1",
        description="sink channel: lateral flow out (0 in the coupled model: no module writes it yet)",
    )
    subirrigation: Array = field(
        dims=(),
        unit="cm d-1",
        description="sink channel: subirrigation, negative when it adds water (0 in the coupled model: "
        "no module writes it yet)",
    )
    macropore_to_drain: Array = field(
        dims=(),
        unit="cm d-1",
        description="sink channel: macropore flow to the drains (0 in the coupled model: no module writes "
        "it yet)",
    )
    sink_cut: Array = field(
        dims=(), unit="cm d-1", description="sink of the channels other than uptake removed at h_min"
    )
    drain_seepage: Array = field(
        dims=(),
        unit="cm d-1",
        description="water the DRAIN cap passed out of the bottom node (part of drainage; 0 unless "
        "RichardsConfig.drain_cap)",
        fortran_name="TSEEP",
    )
    drain_moved: Array = field(
        dims=(),
        unit="cm d-1",
        description="water the DRAIN cap passed down from nodes above the field-saturated porosity, "
        "summed over the nodes and sub-steps (drain_seepage included; 0 unless RichardsConfig.drain_cap)",
    )
    # ---- adaptive stepping diagnostics (0 in the fixed modes) ----
    n_steps: Array = field(dims=(), unit="-", description="accepted adaptive sub-steps of the day")
    n_rejects: Array = field(
        dims=(), unit="-", description="rejected adaptive tries of the day (failed Newton, redone)"
    )
    n_newton: Array = field(
        dims=(), unit="-", description="Newton evaluations of the adaptive step search (rejected tries too)"
    )
    n_unconverged: Array = field(
        dims=(),
        unit="-",
        description="accepted adaptive sub-steps whose Newton solve did not converge (should be 0)",
    )
    budget_exhausted: Array = field(
        dims=(),
        unit="-",
        description="segments that ran out of steps or tries and ended in one unconverged step (should be 0)",
    )
    n_active: Array = field(
        dims=(),
        unit="-",
        description="accepted adaptive sub-steps with a node held at the dry guard (active set; should be 0)",
    )
    dt_min_used: Array = field(
        dims=(), unit="h", description="smallest accepted adaptive sub-step of the day"
    )
    step_balance_max: Array = field(
        dims=(),
        unit="cm",
        description="largest |water balance error| of an accepted adaptive sub-step of the day "
        "(sum(tl dtheta) - dt (q_top - q_bot - sum tl S); rounding level on converged steps)",
    )


class SoilWater(State):
    """Soil-water state on the nodes plus the surface pond and the last day's fluxes."""

    flux: SoilWaterFluxes
    h: Array = field(unit="cm", description="matric head at the nodes", fortran_name="H", dims="n_node")
    theta: Array = field(
        unit="cm3 cm-3",
        description="volumetric water content (conserved)",
        fortran_name="THETA",
        dims="n_node",
    )
    pond: Array = field(dims=(), unit="cm", description="surface ponding depth")
    dt_next: Array = field(
        dims=(),
        unit="h",
        description="adaptive sub-step carried to the next day (RZWQM2 keeps DELT across days); "
        "not differentiated, unused by the fixed modes",
        fortran_name="DELT",
        default=DT_START,
    )

    @classmethod
    def from_theta(cls, theta: Any, soil: AnyHydraulicParams) -> SoilWater:
        """Initial state from a water-content profile (``h = h(theta)``, RZWQM ``WCH``)."""
        th = jnp.asarray(theta, dtype=jnp.result_type(float))
        h = h_of_theta(th, soil)
        return cls(
            h=h,
            theta=theta_of_h(h, soil),
            pond=jnp.zeros((), th.dtype),
            flux=_zero_fluxes(th.dtype),
            dt_next=jnp.asarray(DT_START, th.dtype),
        )

    @classmethod
    def from_head(cls, h: Any, soil: AnyHydraulicParams) -> SoilWater:
        """Initial state from a head profile (``theta = theta(h)``)."""
        hh = jnp.asarray(h, dtype=jnp.result_type(float))
        return cls(
            h=hh,
            theta=theta_of_h(hh, soil),
            pond=jnp.zeros((), hh.dtype),
            flux=_zero_fluxes(hh.dtype),
            dt_next=jnp.asarray(DT_START, hh.dtype),
        )

    def storage(self, grid: RichardsGrid) -> Array:
        """Profile storage ``sum(theta * tl)`` [cm] (``.ana`` column 2)."""
        return jnp.sum(self.theta * grid.tl, axis=-1)


def _zero_fluxes(dtype: Any) -> SoilWaterFluxes:
    z = jnp.zeros((), dtype)
    names = [f.name for f in dataclasses.fields(SoilWaterFluxes)]
    return SoilWaterFluxes(**dict.fromkeys(names, z))


# ---------------------------------------------------------------------------
# residual, fluxes and Jacobian of one sub-step
# ---------------------------------------------------------------------------


class _StepArgs(NamedTuple):
    """Differentiable inputs of one sub-step solve (``soil`` already gathered on the nodes)."""

    soil: AnyHydraulicParams
    tl: Array
    delz: Array
    dz_top: Array
    theta_old: Array
    h_old: Array
    q_demand: Array  # requested surface flux w - e [cm h-1], downward positive
    sink: Array  # [h-1], after the h_min cut
    dt: Array  # [h]
    alpha: Array
    h_min: Array
    h_hi: Array  # upper clamp of the iterate per node [cm] (h_upper + node depth)
    # dry-end K exponent of node 0 for the flux-mode evaporation limit; None: the h_min limit (static)
    eps_peak: Array | None = None


def _log_k(k: Array) -> Array:
    floor = _K_FLOOR_F32 if k.dtype == jnp.float32 else _K_FLOOR
    return jnp.log(jnp.where(k > floor, k, floor))


def surface_fluxes(h: Array, h_k: Array, a: _StepArgs) -> tuple[Array, Array, Array]:
    """``(q_top, q_wet, q_dry)`` [cm h-1] for iterate ``h`` with conductivities at ``h_k``.

    ``q_wet`` (ghost head 0) and ``q_dry`` (ghost head ``h_min``) are the Dirichlet limits of
    the surface flux; ``q_top`` is the requested flux ``q_demand`` clipped to them. ``q_wet`` is
    floored at 0 and ``q_dry`` capped at 0 so that a limit never reverses the flux direction.
    With ``a.eps_peak`` set (``RichardsConfig.evaporation_limit = "flux_peak"``) ``q_dry`` is
    RZWQM2's flux-mode limit: the larger evaporation of the ghost head ``h_min`` and of the peak
    head :func:`~agrijax.processes.soil_water.conventions.flux_peak_head`.
    """
    ht0 = a.alpha * h[0] + (1.0 - a.alpha) * a.h_old[0]
    hk0 = a.alpha * h_k[0] + (1.0 - a.alpha) * a.h_old[0]
    soil0 = jax.tree_util.tree_map(lambda x: x[:1], a.soil)
    k_node = _log_k(k_of_h(hk0[None], soil0))[0]
    k_sat = _log_k(k_of_h(jnp.zeros_like(hk0)[None], soil0))[0]
    k_dry = _log_k(k_of_h(jnp.broadcast_to(a.h_min, hk0.shape)[None], soil0))[0]
    # ponded limit: upstream (ghost-node) conductivity K(0) = K_sat. The geometric mean of the
    # interior faces makes the capacity *increase* as a dry surface node wets (d q_wet / d h_0 > 0
    # for 0.5 eps > 1), which sends Newton towards the dry end; RZWQM never meets this case
    # because its rain goes through the Green-Ampt INFIL routine, not through Richards.
    kw = jnp.exp(k_sat)
    # dry limit: geometric mean with K(h_min) as in RZWQM (monotone decreasing in h_0)
    kd = jnp.exp(_GEOMETRIC_MEAN * (k_node + k_dry))
    q_wet_raw = -kw * (ht0 / a.dz_top - 1.0)
    q_dry_raw = -kd * ((ht0 - a.h_min) / a.dz_top - 1.0)
    if a.eps_peak is not None:  # static: RZWQM2's flux-mode limit, the peak over the ghost head
        h_pk = flux_peak_head(ht0, a.eps_peak, a.h_min, a.dz_top, _GEOMETRIC_MEAN)
        k_pk = _log_k(k_of_h(h_pk[None], soil0))[0]
        q_pk = -jnp.exp(_GEOMETRIC_MEAN * (k_node + k_pk)) * ((ht0 - h_pk) / a.dz_top - 1.0)
        q_dry_raw = jnp.where(q_pk < q_dry_raw, q_pk, q_dry_raw)
    q_wet = jnp.where(q_wet_raw > 0.0, q_wet_raw, 0.0)
    q_dry = jnp.where(q_dry_raw < 0.0, q_dry_raw, 0.0)
    q_top = jnp.where(a.q_demand > q_wet, q_wet, jnp.where(a.q_demand < q_dry, q_dry, a.q_demand))
    return q_top, q_wet, q_dry


def _face_fluxes(h: Array, h_k: Array, a: _StepArgs) -> Array:
    """Face fluxes ``q[n+1]`` [cm h-1] (downward positive); gradients at ``h``, conductivities at ``h_k``."""
    ht = a.alpha * h + (1.0 - a.alpha) * a.h_old
    hk = a.alpha * h_k + (1.0 - a.alpha) * a.h_old
    logk = _log_k(k_of_h(hk, a.soil))
    k_face = jnp.exp(_GEOMETRIC_MEAN * (logk[:-1] + logk[1:]))
    q_int = -k_face * ((ht[1:] - ht[:-1]) / a.delz - 1.0)
    q_top, _, _ = surface_fluxes(h, h_k, a)
    q_bot = jnp.exp(logk[-1])
    return jnp.concatenate([q_top[None], q_int, q_bot[None]])


def richards_residual(h: Array, h_k: Array, a: _StepArgs) -> Array:
    """Mixed-form residual ``R_i`` [cm h-1] of one sub-step (rows scaled by ``tl``).

    ``h_k`` is where the conductivities are evaluated (``h`` itself for Newton, the current
    iterate held fixed for modified Picard).
    """
    q = _face_fluxes(h, h_k, a)
    return a.tl * (theta_of_h(h, a.soil) - a.theta_old) / a.dt + (q[1:] - q[:-1]) + a.tl * a.sink


def tridiagonal_jacobian(fun: Any, h: Array) -> tuple[Array, Array, Array, Array]:
    """``(f(h), dl, d, du)``: value and the three diagonals of the Jacobian of a tridiagonal-coupled ``fun``.

    Three JVPs with the seeds ``s_k = [i % 3 == k]``: row ``i`` of ``J s_k`` is the single
    entry of row ``i`` in a column of colour ``k``, so the three products hold the whole band.
    ``dl[0] = du[-1] = 0`` (``lax.linalg.tridiagonal_solve`` layout).
    """
    n = h.shape[-1]
    idx = np.arange(n)
    seeds = jnp.asarray(
        np.stack([(idx % _N_COLOURS == k) for k in range(_N_COLOURS)]).astype(float), dtype=h.dtype
    )
    r, lin = jax.linearize(fun, h)
    cols = jax.vmap(lin)(seeds)  # [3, n]
    d = cols[idx % _N_COLOURS, idx]
    dl = jnp.where(idx > 0, cols[(idx - 1) % _N_COLOURS, idx], 0.0)
    du = jnp.where(idx < n - 1, cols[(idx + 1) % _N_COLOURS, idx], 0.0)
    return r, dl, d, du


def _tridiag_solve(dl: Array, d: Array, du: Array, b: Array) -> Array:
    return lax.linalg.tridiagonal_solve(dl, d, du, b[:, None])[:, 0]


def head_of_v(v: Array, scale: Array) -> Array:
    """Transformed iteration variable -> head: ``-s exp(-v)`` for ``v <= 0``, ``s (v - 1)`` above.

    ``s`` is the node's bubbling pressure ``hb``; the map is C1 at ``v = 0`` (``h = -hb``).
    On the Brooks-Corey segment ``v = log(hb/|h|)``, so ``theta - theta_r`` and ``K`` are
    exponentials of ``v`` instead of steep power laws of ``h``: Newton steps in ``v`` do not
    overshoot on a wetting front entering dry soil (Pan & Wierenga 1995 use a transform of
    the same kind). Both branches are finite for every finite ``v``.
    """
    v_dry = jnp.where(v <= 0.0, v, 0.0)
    return jnp.where(v <= 0.0, -scale * jnp.exp(-v_dry), scale * (v - 1.0))


def v_of_head(h: Array, scale: Array) -> Array:
    """Inverse of :func:`head_of_v`: ``log(hb/|h|)`` for ``h <= -hb``, ``h/hb + 1`` above."""
    absh = jnp.where(-h >= scale, -h, scale)
    v_dry = jnp.log(scale / absh)
    v_wet = h / scale + 1.0
    return jnp.where(h <= -scale, v_dry, v_wet)


def _dh_dv(v: Array, scale: Array) -> Array:
    """Derivative of :func:`head_of_v`: ``s exp(-v)`` for ``v <= 0``, ``s`` above (both finite)."""
    v_dry = jnp.where(v <= 0.0, v, 0.0)
    return jnp.where(v <= 0.0, scale * jnp.exp(-v_dry), scale)


def _capacity(h: Array, soil: AnyHydraulicParams) -> Array:
    """Exact ``C(h) = d theta / dh`` (one JVP of :func:`theta_of_h`)."""
    return jax.jvp(lambda x: theta_of_h(x, soil), (h,), (jnp.ones_like(h),))[1]


class StepResult(NamedTuple):
    """Outcome of one sub-step (fluxes as depths over the sub-step [cm])."""

    h: Array
    theta: Array
    pond: Array
    infiltration: Array
    evaporation: Array
    drainage: Array
    uptake: Array
    runoff: Array
    evaporation_deficit: Array
    uptake_cut: Array
    balance_error: Array
    theta_residual: Array
    n_clamp: Array
    sinks: Array  # [n_channel] depth of each sink channel (SINK_CHANNELS order) over the sub-step
    sinks_cut: Array  # [n_channel] part of each channel removed at h_min


class _StepSinks(NamedTuple):
    """The sink of one sub-step after the ``h_min`` cap: node rate and per-channel depths [cm]."""

    s_cut: Array  # [n] total sink rate after the cut [h-1]
    uptake: Array
    uptake_cut: Array
    sinks: Array  # [n_channel]
    sinks_cut: Array  # [n_channel]


def _step_args(
    h: Array,
    theta: Array,
    pond: Array,
    soil: AnyHydraulicParams,
    grid: RichardsGrid,
    supply: Array,
    evaporation: Array,
    sink: Array,
    dt: Array,
    alpha: Array,
    h_min: Array,
    cfg: RichardsConfig,
    h_upper: float,
) -> tuple[_StepArgs, _StepSinks]:
    """Inputs of one sub-step solve: the ``h_min`` sink cap and the :class:`_StepArgs`.

    See :func:`~agrijax.processes.soil_water.fixed_cn.richards_step` for the arguments; ``cfg`` is the
    problem's config, ``h_upper`` [cm] the integrator's upper clamp of the iterate above hydrostatic.

    Source: RZWQM2 ``CNHEAD`` (no uptake at ``Hmin``).
    """
    # RZWQM: no uptake from a node at h_min. Written as an availability cap so that a prescribed
    # uptake can never take a node below theta(h_min) within the sub-step.
    theta_min = theta_of_h(jnp.broadcast_to(h_min, h.shape), soil)
    avail = theta - theta_min - cfg.sink_cutoff
    s_avail = jnp.where(avail > 0.0, avail, 0.0) / dt
    k_upt = SINK_CHANNELS.index("uptake")
    zeros = jnp.zeros(len(SINK_CHANNELS), theta.dtype)
    if sink.ndim == 1:  # static: uptake alone
        s_cut = jnp.where(sink < s_avail, sink, s_avail)
        uptake = jnp.sum(grid.tl * s_cut) * dt
        uptake_cut = jnp.sum(grid.tl * (sink - s_cut)) * dt
        sinks = zeros.at[k_upt].set(uptake)
        sinks_cut = zeros.at[k_upt].set(uptake_cut)
    else:
        rows = []
        left = s_avail
        for k, _ in enumerate(SINK_CHANNELS):  # five named channels, in turn
            s_k = jnp.where(sink[k] < left, sink[k], left)
            left = left - s_k
            rows.append(s_k)
        s_rows = jnp.stack(rows)
        s_cut = jnp.sum(s_rows, axis=0)
        sinks = jnp.sum(grid.tl * s_rows, axis=1) * dt
        sinks_cut = jnp.sum(grid.tl * (sink - s_rows), axis=1) * dt
        uptake, uptake_cut = sinks[k_upt], sinks_cut[k_upt]
    w = supply + pond / dt
    a = _StepArgs(
        soil=soil,
        tl=grid.tl,
        delz=grid.delz,
        dz_top=grid.dz_top,
        theta_old=theta,
        h_old=h,
        q_demand=w - evaporation,
        sink=s_cut,
        dt=dt,
        alpha=alpha,
        h_min=h_min,
        h_hi=h_upper + grid.node_depth(),
        eps_peak=dry_end_eps(soil)[0] if cfg.evaporation_limit == "flux_peak" else None,
    )
    return a, _StepSinks(s_cut, uptake, uptake_cut, sinks, sinks_cut)


def _step_result(
    h_new: Array,
    nclamp: Array,
    a: _StepArgs,
    sk: _StepSinks,
    soil: AnyHydraulicParams,
    grid: RichardsGrid,
    evaporation: Array,
    dt: Array,
    pond_max: Array,
) -> StepResult:
    """Fluxes, pond, balance and residual of a sub-step from its solved heads ``h_new``.

    Source: Celia et al. (1990) mixed form (the balance is ``dt sum R``); RZWQM2 ``NODFLX``.
    """
    theta = a.theta_old
    s_cut, uptake, uptake_cut, sinks, sinks_cut = sk
    q = _face_fluxes(h_new, h_new, a)
    resid = richards_residual(h_new, h_new, a)
    theta_new = theta_of_h(h_new, soil)
    q_top = q[0]
    surplus = jnp.where(a.q_demand > q_top, a.q_demand - q_top, 0.0) * dt
    deficit = jnp.where(q_top > a.q_demand, q_top - a.q_demand, 0.0) * dt
    pond_new = jnp.where(surplus < pond_max, surplus, pond_max)
    evap = evaporation * dt - deficit
    return StepResult(
        h=h_new,
        theta=theta_new,
        pond=pond_new,
        infiltration=q_top * dt + evap,
        evaporation=evap,
        drainage=q[-1] * dt,
        uptake=uptake,
        runoff=surplus - pond_new,
        evaporation_deficit=deficit,
        uptake_cut=uptake_cut,
        balance_error=jnp.sum(grid.tl * (theta_new - theta))
        - dt * (q_top - q[-1] - jnp.sum(grid.tl * s_cut)),
        theta_residual=jnp.max(jnp.abs(resid) * dt / grid.tl),
        n_clamp=nclamp,
        sinks=sinks,
        sinks_cut=sinks_cut,
    )


def _interval_means(hourly: Array, t: Array) -> Array:
    """Average of an hourly piecewise-constant rate over each interval ``[t[k], t[k+1]]`` (0 if empty)."""
    cum = jnp.concatenate([jnp.zeros_like(hourly[:1]), jnp.cumsum(hourly)])
    c = jnp.interp(t, jnp.arange(_N_HOUR_EDGES, dtype=hourly.dtype), cum)
    width = t[1:] - t[:-1]
    return (c[1:] - c[:-1]) / jnp.where(width > 0.0, width, 1.0)


class SubstepTotals(NamedTuple):
    """Totals [cm] of a run of sub-steps, and its worst residual and clamp count."""

    supply: Array
    infiltration: Array
    evaporation: Array
    drainage: Array
    uptake: Array
    runoff: Array
    evaporation_deficit: Array
    uptake_cut: Array
    max_theta_residual: Array
    n_clamp: Array
    sinks: Array  # [n_channel] daily depth of each sink channel (SINK_CHANNELS order)
    sinks_cut: Array  # [n_channel]
    drain_seepage: Array  # DRAIN cap seepage out of the bottom node (included in drainage)
    drain_moved: Array  # water the DRAIN cap passed down, summed over nodes and sub-steps


def sink_fluxes(tot: SubstepTotals) -> dict[str, Array]:
    """Flux fields of the sink channels other than ``uptake``, and their total ``h_min`` cut."""
    out = {name: tot.sinks[k] for k, name in enumerate(SINK_CHANNELS) if name != "uptake"}
    k_upt = SINK_CHANNELS.index("uptake")
    out["sink_cut"] = jnp.sum(tot.sinks_cut) - tot.sinks_cut[k_upt]
    return out


def other_sinks(tot: SubstepTotals) -> Array:
    """Total of the sink channels other than ``uptake`` [cm] (exactly 0 when they are absent)."""
    k_upt = SINK_CHANNELS.index("uptake")
    return jnp.sum(jnp.where(jnp.arange(len(SINK_CHANNELS)) == k_upt, 0.0, tot.sinks))


def node_pori(params: Any, aef: Any, dtype: Any) -> Array | None:
    """Field-saturated porosity ``aef theta_s`` on the nodes for the DRAIN cap, ``None`` without it.

    Source: Ahuja et al. (2000) ch. 3; RZWQM2 ``PORI`` (``Rzmain.for:538``), read for conventions.
    """
    if "drain_cap" not in params.config.conventions:  # static
        return None
    _require(aef is not None, "RichardsConfig.drain_cap needs the field-saturation fraction aef")
    n = params.grid.n_node
    soil = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,)), params.soil.at_nodes())
    return field_saturation(soil, jnp.asarray(aef, dtype))


def drain_fluxes(tot: SubstepTotals, cfg: RichardsConfig) -> dict[str, Array]:
    """``{"drain_seepage": ..., "drain_moved": ...}`` with the DRAIN cap, else ``{}`` (the fields keep
    their zeros)."""
    if "drain_cap" not in cfg.conventions:  # static
        return {}
    return {"drain_seepage": tot.drain_seepage, "drain_moved": tot.drain_moved}


#: surface condition of a solve: the clipped condition of the fixed modes, ponded, or flux
BC_CLIP: float = 0.0
BC_PONDED: float = 1.0
BC_FLUX: float = 2.0


def _top_fluxes(h: Array, a: _StepArgs, bc: Array) -> tuple[Array, Array]:
    """``(q_clip, q_bc)`` [cm h-1]: the clipped surface flux of the fixed modes
    (:func:`~agrijax.processes.soil_water.richards.surface_fluxes`) and the flux under the surface
    condition ``bc``: ``BC_CLIP`` the same, ``BC_PONDED`` the ponded (ghost head 0) Darcy flux without
    its floor at 0, ``BC_FLUX`` the requested flux ``q_demand``. Each fixed condition is smooth in
    ``h``; where the clipped flux equals it, the two residuals are the same.

    Source: RZWQM2 ``CHKBC`` (``Rzrich.for``: flux or head condition chosen outside the iteration),
    read for conventions; Ahuja et al. (2000) ch. 3.
    """
    q_clip, _, _ = surface_fluxes(h, h, a)
    ht0 = a.alpha * h[0] + (1.0 - a.alpha) * a.h_old[0]
    soil0 = jax.tree_util.tree_map(lambda x: x[:1], a.soil)
    k_sat = jnp.exp(_log_k(k_of_h(jnp.zeros_like(ht0)[None], soil0))[0])
    q_ponded = -k_sat * (ht0 / a.dz_top - 1.0)
    q_bc = jnp.where(bc == BC_PONDED, q_ponded, jnp.where(bc == BC_FLUX, a.q_demand, q_clip))
    return q_clip, q_bc


def _residual_bc(h: Array, a: _StepArgs, bc: Array) -> Array:
    """The residual of :func:`~agrijax.processes.soil_water.richards.richards_residual` with the
    surface condition ``bc`` (``BC_CLIP``: that residual itself)."""
    r = richards_residual(h, h, a)
    q_clip, q_bc = _top_fluxes(h, a, bc)
    return jnp.where(bc == BC_CLIP, r, r.at[0].add(q_clip - q_bc))


def _rate(hourly: Array, t0: Array, dt: Array) -> Array:
    """Mean of an hourly piecewise-constant rate over ``[t0, t0 + dt]`` (0 for an empty step)."""
    cum = jnp.concatenate([jnp.zeros_like(hourly[:1]), jnp.cumsum(hourly)])
    hours = jnp.arange(_N_HOUR_EDGES, dtype=hourly.dtype)
    c = jnp.interp(jnp.stack([t0, t0 + dt]), hours, cum)
    return (c[1] - c[0]) / jnp.where(dt > 0.0, dt, 1.0)


SinkOf = Callable[[Array, Array, Array, Array], Array]


def _sink_provider(channels: SinkChannels, grid: RichardsGrid, dtype: Any) -> SinkOf:
    """``sink_of(t0, dt, theta, h)``: the node sink rate [h-1] of a step, as in ``richards_substeps``."""
    daily = channels.daily_uptake_only()
    if daily is not None:  # static: the uptake-only sink, one daily array
        uptake_rate = jnp.asarray(daily, dtype) / (HOURS_PER_DAY * grid.tl)

        def sink_daily(t0: Array, dt: Array, th: Array, h: Array) -> Array:
            return uptake_rate

        return sink_daily

    def sink_channels(t0: Array, dt: Array, th: Array, h: Array) -> Array:
        return jnp.stack([c.node_rate(grid.tl, t0, dt, th, h) for c in channels.channels()])

    return sink_channels


# ---------------------------------------------------------------------------
# post-step convention hooks, switch predicates
# ---------------------------------------------------------------------------


def post_step(
    r: StepResult, soil: AnyHydraulicParams, tl: Array, pori: Array | None, config: RichardsConfig
) -> tuple[StepResult, tuple[Array, Array] | None]:
    """The post-step convention hooks of an accepted sub-step: RZWQM2's DRAIN cap
    (``RichardsConfig.drain_cap``; heads and water contents capped, the seepage added to the
    drainage; returns ``(step, (seepage, moved))``), else the step unchanged and ``None``.

    Source: RZWQM2 ``DRAIN`` after every ``RICHRD`` step (``Rzrich.for:1092``), read for conventions;
    :func:`~agrijax.processes.soil_water.conventions.drain_cap`.
    """
    if "drain_cap" not in config.conventions:  # static
        return r, None
    dr = drain_cap(r.theta, r.h, soil, tl, cast(Array, pori))
    return r._replace(h=dr.h, theta=dr.theta, drainage=r.drainage + dr.seepage), (dr.seepage, dr.moved)


class SurfaceSwitches(NamedTuple):
    """The switch predicates of a sub-step at an iterate (booleans)."""

    ponding: Array  # the requested flux exceeds the ponded (ghost head 0) infiltration capacity
    evaporation_limited: Array  # the evaporation demand exceeds the dry-end (ghost head h_min) limit
    dry_end: Array  # [n] nodes at or below h_min


def surface_switches(h: Array, a: _StepArgs) -> SurfaceSwitches:
    """Which limit of the surface flux binds at ``h`` (the clipped condition of :func:`surface_fluxes`)
    and which nodes are at the dry end.

    Source: RZWQM2 ``CHKBC`` (``Rzrich.for``: flux or head condition), read for conventions;
    Ahuja et al. (2000) ch. 3.
    """
    _, q_wet, q_dry = surface_fluxes(h, h, a)
    return SurfaceSwitches(a.q_demand > q_wet, a.q_demand < q_dry, h <= a.h_min)


# ---------------------------------------------------------------------------
# the problem of a day
# ---------------------------------------------------------------------------


def _node_soil(soil: AnyHydraulicParams, n: int) -> AnyHydraulicParams:
    """The hydraulic parameters on the ``n`` nodes (broadcast)."""
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,)), soil.at_nodes())


class RichardsProblem(eqx.Module):
    """The semi-discrete Richards problem of one day (a pytree; ``config`` static).

    ``soil`` on the node axis, ``h_min``/``pond_max`` [cm], ``supply``/``evaporation`` the hourly
    surface supply and evaporation demand ``[24]`` [cm h-1], ``channels`` the sink channels,
    ``pori`` the field-saturated porosity per node (``None`` without the DRAIN cap), ``hydraulic``
    the hydraulic parameters as given (``RichardsParams.soil``, from which ``soil`` is derived).
    Build it with :meth:`of_day`; the methods are the functions of this module on the problem's
    data. An integrator advances a segment of the day (the day, or the part before or after an
    event) on :meth:`for_segment`.
    """

    soil: AnyHydraulicParams
    hydraulic: AnyHydraulicParams
    grid: RichardsGrid
    h_min: Array
    pond_max: Array
    supply: Array
    evaporation: Array
    channels: SinkChannels
    pori: Array | None
    config: RichardsConfig = eqx.field(static=True)

    version: ClassVar[int] = PROBLEM_VERSION

    @classmethod
    def of_day(
        cls,
        params: Any,
        supply: Any,
        evaporation: Any,
        uptake: Any,
        dtype: Any,
        pori: Array | None = None,
    ) -> RichardsProblem:
        """The problem of a day from :class:`~agrijax.processes.soil_water.richards.RichardsParams`:
        the hourly supply and evaporation ``[24]`` [cm h-1], the sink channels (or the per-layer
        uptake [cm d-1] alone), the state's dtype and ``pori`` (:func:`node_pori`; needed with
        ``RichardsConfig.drain_cap``)."""
        config = params.config
        _require(
            "drain_cap" not in config.conventions or pori is not None,
            "RichardsConfig.drain_cap needs pori (aef * theta_s on the nodes)",
        )
        return cls(
            soil=_node_soil(params.soil, params.grid.n_node),
            hydraulic=params.soil,
            grid=params.grid,
            h_min=jnp.asarray(params.h_min, dtype),
            pond_max=jnp.asarray(params.pond_max, dtype),
            supply=jnp.asarray(supply, dtype),
            evaporation=jnp.asarray(evaporation, dtype),
            channels=as_sink_channels(uptake),
            pori=None if "drain_cap" not in config.conventions else jnp.asarray(pori, dtype),
            config=config,
        )

    def for_segment(self) -> RichardsProblem:
        """The problem as one segment of the day sees it: ``soil`` derived afresh from ``hydraulic``.

        One derivation per segment keeps the reverse pass of an event day as it was before the
        physics and the integrator were separated (each segment differentiates its own node
        parameters; one shared derivation sums the two segments' cotangents first and rounds
        differently, 1-4 ulp); the values are the same.

        Source: implementation structure with no reference-model counterpart (``_node_soil`` applied per
        segment); the existing modes are kept bit for bit across the split of physics and integrator.
        """
        return dataclasses.replace(self, soil=_node_soil(self.hydraulic, self.grid.n_node))

    @property
    def dtype(self) -> Any:
        return self.supply.dtype

    # ---- storage and the conserved variable
    def theta(self, h: Array) -> Array:
        """Water content ``theta(h)`` of the nodes [cm3 cm-3] (the conserved variable)."""
        return theta_of_h(h, self.soil)

    def head(self, theta: Array) -> Array:
        """Head ``h(theta)`` of the nodes [cm] (RZWQM2 ``WCH``)."""
        return h_of_theta(theta, self.soil)

    def capacity(self, h: Array) -> Array:
        """``C(h) = d theta / dh`` [cm-1] (:func:`capacity`)."""
        return _capacity(h, self.soil)

    def storage(self, theta: Array) -> Array:
        """Profile storage ``sum(theta tl)`` [cm]."""
        return jnp.sum(theta * self.grid.tl, axis=-1)

    # ---- forcing of a sub-step
    def rates(self, t0: Array, dt: Array) -> tuple[Array, Array]:
        """Mean supply and evaporation demand [cm h-1] over ``[t0, t0 + dt]``."""
        return _rate(self.supply, t0, dt), _rate(self.evaporation, t0, dt)

    def interval_rates(self, t: Array) -> tuple[Array, Array]:
        """Mean supply and evaporation demand [cm h-1] over each interval ``[t[k], t[k+1]]``."""
        return _interval_means(self.supply, t), _interval_means(self.evaporation, t)

    def sink_of(self) -> SinkOf:
        """``sink_of(t0, dt, theta, h)``: the node sink rate [h-1] of a sub-step before the cap."""
        return _sink_provider(self.channels, self.grid, self.dtype)

    # ---- one sub-step of the theta-method family
    def step_args(
        self,
        h: Array,
        theta: Array,
        pond: Array,
        supply: Array,
        evaporation: Array,
        sink: Array,
        dt: Array,
        alpha: Array,
        h_upper: float,
    ) -> tuple[_StepArgs, _StepSinks]:
        """The sub-step problem (:func:`step_args`) from the state at its start."""
        return _step_args(
            h, theta, pond, self.soil, self.grid, supply, evaporation, sink, dt, alpha, self.h_min,
            self.config, h_upper,
        )  # fmt: skip

    def residual(self, h: Array, a: _StepArgs, bc: Array | None = None) -> Array:
        """Residual ``R`` [cm h-1] at ``h`` (:func:`richards_residual`; under the surface condition
        ``bc`` with :func:`residual_bc`)."""
        return richards_residual(h, h, a) if bc is None else _residual_bc(h, a, bc)

    def jacobian(self, h: Array, a: _StepArgs, bc: Array | None = None) -> tuple[Array, Array, Array, Array]:
        """``(R, dl, d, du)``: the residual and its exact tridiagonal Jacobian ``dR/dh``."""
        return tridiagonal_jacobian(lambda x: self.residual(x, a, bc), h)

    def face_fluxes(self, h: Array, a: _StepArgs) -> Array:
        """Face fluxes ``q[n+1]`` [cm h-1] at the time-weighted head of ``h``."""
        return _face_fluxes(h, h, a)

    def switches(self, h: Array, a: _StepArgs) -> SurfaceSwitches:
        """:func:`surface_switches` at ``h``."""
        return surface_switches(h, a)

    def result(
        self, h_new: Array, n_clamp: Array, a: _StepArgs, sk: _StepSinks, evaporation: Array, dt: Array
    ) -> StepResult:
        """The accumulators of a solved sub-step (:func:`step_result`)."""
        return _step_result(h_new, n_clamp, a, sk, self.soil, self.grid, evaporation, dt, self.pond_max)

    def post_step(self, r: StepResult) -> tuple[StepResult, tuple[Array, Array] | None]:
        """The convention hooks after an accepted sub-step (:func:`post_step`)."""
        return post_step(r, self.soil, self.grid.tl, self.pori, self.config)


#: public names of the sub-step functions (the underscore names are kept for existing callers)
StepArgs = _StepArgs
StepSinks = _StepSinks
face_fluxes = _face_fluxes
step_args = _step_args
step_result = _step_result
top_fluxes = _top_fluxes
residual_bc = _residual_bc
capacity = _capacity
tridiag_solve = _tridiag_solve
interval_means = _interval_means
