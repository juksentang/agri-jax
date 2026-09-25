r"""Implicit mixed-form Richards equation for the soil-water redistribution step (RZWQM2 style).

One day of vertical unsaturated flow on a vertex-centred finite-volume grid, advanced with a
fixed number of sub-steps, each solved with a fixed number of Newton (or modified Picard)
iterations and a tridiagonal linear solve. The retention and conductivity curves are the
modified Brooks-Corey functions of :mod:`agrijax.processes.soil_water.hydraulics`.

Conventions (G3 review, private note 16)
----------------------------------------
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

Time weights: ``alpha = 1`` on every sub-step (``time_scheme="implicit"``, the default), or
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
(``chop``; Wang & Tchelepi 2013 limit Newton updates at kinks of the flux function alike),
which removes the alternation across the kink while a saturated surface drains; and the
iterate is clamped to ``[h_min, h_upper + z_i]`` (a saturated layered profile carries
positive heads that grow with depth, ~24 cm at CA-TPA under ponding). Clamp activations
are counted (``n_clamp``) and should be zero.

Sub-steps: ``n_sub`` per day. A fraction ``rain_fraction`` of them is placed in proportion to
the hourly supply, the rest spread uniformly (:func:`substep_edges`); the schedule depends
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

Note: RZWQM2 does not use the rain as the upper boundary flux of the Richards step. Its
``INFIL`` routine fills the profile with a Green-Ampt wetting front and Richards only
redistributes; that event is :mod:`~agrijax.processes.soil_water.infiltration`, and the day
of redistribution / event / redistribution is :mod:`~agrijax.processes.soil_water.day`. In
this module's own day (:func:`richards_day`, the M1 configuration) ``supply`` is the water
that *infiltrates* (e.g. ``.ana`` column 5), spread over the rain event.

Sink: an input, a typed record of sink channels (:mod:`~agrijax.processes.soil_water.sinks`:
``uptake``, ``tile``, ``lateral``, ``subirrigation``, ``macropore_to_drain``), each a per-layer
daily amount [cm d-1] spread uniformly over the day and/or a per-sub-step callable. A plain array
is the per-layer root water uptake (the only channel of M1 and M3). Each channel is capped per
sub-step by the water above ``theta(h_min)`` in the node (RZWQM: no uptake at ``Hmin``), the
channels in turn, and the cut is reported (``uptake_cut``, ``sink_cut``). Each channel has its own
daily total in :class:`SoilWaterFluxes` and its own term in ``balance_error``.

Gradients: ``grad="unrolled"`` differentiates through the fixed iterations;
``grad="implicit"`` gives each sub-step solve an implicit-function-theorem VJP
(``J(h*)^T lambda = g``, tridiagonal, then the VJP of ``R`` with respect to the inputs),
which is exact at a converged root and independent of the iteration path.

Source: Ahuja, L.R., Rojas, K.W., Hanson, J.D., Shaffer, M.J., Ma, L. (eds.), 2000. Root
Zone Water Quality Model, ch. 3 (Richards equation, boundary switching, free drainage);
Celia, M.A., Bouloutas, E.T., Zarba, R.L., 1990. A general mass-conservative numerical
solution for the unsaturated flow equation. Water Resour. Res. 26, 1483-1496. Wang, X.,
Tchelepi, H.A., 2013. Trust-region based solver for nonlinear transport in heterogeneous
porous media. J. Comput. Phys. 253, 114-137 (Newton updates limited at kinks and inflection
points). The
corresponding RZWQM2 subroutines are ``RICHRD``, ``CHKBC``, ``NODFLX`` and ``POINTK``
(``Rzrich.for``).
"""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array

from agrijax.core.process import check_enabled, process
from agrijax.core.state import Forcing, Params, State, field

from .hydraulics import H_CLAMP_RZWQM, SoilHydraulicParams, h_of_theta, k_of_h, theta_of_h
from .sinks import SINK_CHANNELS, SinkChannels, as_sink_channels

__all__ = [
    "HOURS_PER_DAY",
    "RichardsConfig",
    "RichardsForcing",
    "RichardsGrid",
    "RichardsParams",
    "RichardsState",
    "SoilWater",
    "SoilWaterFluxes",
    "StepResult",
    "SubstepTotals",
    "day_alphas",
    "head_of_v",
    "other_sinks",
    "richards_day",
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

HOURS_PER_DAY: float = 24.0
#: floor of K inside the logarithm of the geometric mean [cm h-1]; K(h_min) is ~1e-9 cm/h for CA-TPA.
_K_FLOOR: float = 1.0e-300
_K_FLOOR_F32: float = 1.0e-37

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
        z0 = 0.5 * self.tl[..., :1]
        return jnp.concatenate([z0, z0 + jnp.cumsum(self.delz, axis=-1)], axis=-1)

    @classmethod
    def from_rzwqm(cls, layer_bottom: Any, node_spacing: Any) -> RichardsGrid:
        """Build from the rzwqm.dat node records: column 2 (layer bottom ``TLT``) and column 3 (``DELZ``).

        ``tl = diff([0, TLT])``; the file is checked for the vertex-centred rule of
        ``Rzmain.for`` (``TL(i) = (DELZ(i-1) + DELZ(i)) / 2``, ``TL(1) = DELZ(1)``,
        ``TL(n) = DELZ(n-1)``), which makes the cell boundaries the midpoints between nodes.
        """
        tlt = np.asarray(layer_bottom, dtype=float)
        dz = np.asarray(node_spacing, dtype=float)
        if tlt.ndim != 1 or dz.shape != tlt.shape:
            raise ValueError(f"expected two [n] arrays, got {tlt.shape} and {dz.shape}")
        tl = np.diff(np.concatenate([[0.0], tlt]))
        delz = dz[:-1]
        expect = np.concatenate([[delz[0]], 0.5 * (delz[:-1] + delz[1:]), [delz[-1]]])
        _require(
            bool(np.allclose(tl, expect, atol=1e-9) and np.all(delz > 0)),
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
    """Static numerical settings (hashable; part of the compiled program).

    ``n_sub`` sub-steps per day x ``n_iter`` Newton iterations per sub-step; ``jacobian``
    ``"newton"`` | ``"picard"``; ``time_scheme`` ``"implicit"`` | ``"rzwqm"``; ``grad``
    ``"unrolled"`` | ``"implicit"`` (implicit-function-theorem VJP per sub-step);
    ``h_upper`` [cm] upper clamp of the iterate *above hydrostatic*: node ``i`` is clamped at
    ``h_upper + z_i`` (a saturated layered profile under ponding carries positive heads that
    grow with depth, up to ``z_i`` plus the surface head in downward flow); ``sink_cutoff``
    [cm3 cm-3] water kept above ``theta(h_min)`` by the uptake cap; ``dv_max`` largest Newton
    update of the transformed variable; ``c_floor`` [cm-1] storage floor added to the
    Jacobian diagonal only (never to the residual); ``chop`` stops an update that leaves the
    saturated side across the air-entry kink ``h = -hb`` on the kink (see :func:`_iterate`);
    ``rain_fraction`` share of the sub-steps placed on the supply hours.
    """

    n_sub: int = eqx.field(static=True, default=24)
    n_iter: int = eqx.field(static=True, default=3)
    jacobian: str = eqx.field(static=True, default="newton")
    time_scheme: str = eqx.field(static=True, default="implicit")
    grad: str = eqx.field(static=True, default="unrolled")
    h_upper: float = eqx.field(static=True, default=10.0)
    sink_cutoff: float = eqx.field(static=True, default=1.0e-9)
    dv_max: float = eqx.field(static=True, default=1.0)
    c_floor: float = eqx.field(static=True, default=1.0e-7)
    chop: bool = eqx.field(static=True, default=True)
    rain_fraction: float = eqx.field(static=True, default=0.5)

    def __check_init__(self) -> None:
        if self.n_sub < 1 or self.n_iter < 1:
            raise ValueError("n_sub and n_iter must be >= 1")
        if self.jacobian not in ("newton", "picard"):
            raise ValueError(f"jacobian must be 'newton' or 'picard', got {self.jacobian!r}")
        if self.time_scheme not in ("rzwqm", "implicit"):
            raise ValueError(f"time_scheme must be 'rzwqm' or 'implicit', got {self.time_scheme!r}")
        if self.grad not in ("unrolled", "implicit"):
            raise ValueError(f"grad must be 'unrolled' or 'implicit', got {self.grad!r}")
        if not 0.0 <= self.rain_fraction < 1.0 or self.dv_max <= 0.0:
            raise ValueError("rain_fraction must be in [0, 1) and dv_max > 0")
        if self.c_floor < 0.0:
            raise ValueError("c_floor must be >= 0")


class RichardsParams(Params):
    """Parameters of the Richards redistribution process."""

    soil: SoilHydraulicParams
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
        dims=(), unit="cm d-1", description="sink channel: tile drainage (0 in M3)", fortran_name="TOTDRN"
    )
    lateral: Array = field(dims=(), unit="cm d-1", description="sink channel: lateral flow out (0 in M3)")
    subirrigation: Array = field(
        dims=(),
        unit="cm d-1",
        description="sink channel: subirrigation, negative when it adds water (0 in M3)",
    )
    macropore_to_drain: Array = field(
        dims=(), unit="cm d-1", description="sink channel: macropore flow to the drains (0 in M3)"
    )
    sink_cut: Array = field(
        dims=(), unit="cm d-1", description="sink of the channels other than uptake removed at h_min"
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

    @classmethod
    def from_theta(cls, theta: Any, soil: SoilHydraulicParams) -> SoilWater:
        """Initial state from a water-content profile (``h = h(theta)``, RZWQM ``WCH``)."""
        th = jnp.asarray(theta, dtype=jnp.result_type(float))
        h = h_of_theta(th, soil)
        return cls(h=h, theta=theta_of_h(h, soil), pond=jnp.zeros((), th.dtype), flux=_zero_fluxes(th.dtype))

    @classmethod
    def from_head(cls, h: Any, soil: SoilHydraulicParams) -> SoilWater:
        """Initial state from a head profile (``theta = theta(h)``)."""
        hh = jnp.asarray(h, dtype=jnp.result_type(float))
        return cls(
            h=hh, theta=theta_of_h(hh, soil), pond=jnp.zeros((), hh.dtype), flux=_zero_fluxes(hh.dtype)
        )

    def storage(self, grid: RichardsGrid) -> Array:
        """Profile storage ``sum(theta * tl)`` [cm] (``.ana`` column 2)."""
        return jnp.sum(self.theta * grid.tl, axis=-1)


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


def _zero_fluxes(dtype: Any) -> SoilWaterFluxes:
    z = jnp.zeros((), dtype)
    names = [f.name for f in dataclasses.fields(SoilWaterFluxes)]
    return SoilWaterFluxes(**dict.fromkeys(names, z))


# ---------------------------------------------------------------------------
# residual, fluxes and Jacobian of one sub-step
# ---------------------------------------------------------------------------


class _StepArgs(NamedTuple):
    """Differentiable inputs of one sub-step solve (``soil`` already gathered on the nodes)."""

    soil: SoilHydraulicParams
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


def _log_k(k: Array) -> Array:
    floor = _K_FLOOR_F32 if k.dtype == jnp.float32 else _K_FLOOR
    return jnp.log(jnp.where(k > floor, k, floor))


def surface_fluxes(h: Array, h_k: Array, a: _StepArgs) -> tuple[Array, Array, Array]:
    """``(q_top, q_wet, q_dry)`` [cm h-1] for iterate ``h`` with conductivities at ``h_k``.

    ``q_wet`` (ghost head 0) and ``q_dry`` (ghost head ``h_min``) are the Dirichlet limits of
    the surface flux; ``q_top`` is the requested flux ``q_demand`` clipped to them. ``q_wet`` is
    floored at 0 and ``q_dry`` capped at 0 so that a limit never reverses the flux direction.
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
    kd = jnp.exp(0.5 * (k_node + k_dry))
    q_wet_raw = -kw * (ht0 / a.dz_top - 1.0)
    q_dry_raw = -kd * ((ht0 - a.h_min) / a.dz_top - 1.0)
    q_wet = jnp.where(q_wet_raw > 0.0, q_wet_raw, 0.0)
    q_dry = jnp.where(q_dry_raw < 0.0, q_dry_raw, 0.0)
    q_top = jnp.where(a.q_demand > q_wet, q_wet, jnp.where(a.q_demand < q_dry, q_dry, a.q_demand))
    return q_top, q_wet, q_dry


def _face_fluxes(h: Array, h_k: Array, a: _StepArgs) -> Array:
    """Face fluxes ``q[n+1]`` [cm h-1] (downward positive); gradients at ``h``, conductivities at ``h_k``."""
    ht = a.alpha * h + (1.0 - a.alpha) * a.h_old
    hk = a.alpha * h_k + (1.0 - a.alpha) * a.h_old
    logk = _log_k(k_of_h(hk, a.soil))
    k_face = jnp.exp(0.5 * (logk[:-1] + logk[1:]))
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
    seeds = jnp.asarray(np.stack([(idx % 3 == k) for k in range(3)]).astype(float), dtype=h.dtype)
    r, lin = jax.linearize(fun, h)
    cols = jax.vmap(lin)(seeds)  # [3, n]
    d = cols[idx % 3, idx]
    dl = jnp.where(idx > 0, cols[(idx - 1) % 3, idx], 0.0)
    du = jnp.where(idx < n - 1, cols[(idx + 1) % 3, idx], 0.0)
    return r, dl, d, du


def _tridiag_solve(dl: Array, d: Array, du: Array, b: Array) -> Array:
    return lax.linalg.tridiagonal_solve(dl, d, du, b[:, None])[:, 0]


class _SolveCfg(NamedTuple):
    n_iter: int
    jacobian: str
    dv_max: float
    c_floor: float
    chop: bool


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


def _residual_newton(h: Array, a: _StepArgs) -> Array:
    return richards_residual(h, h, a)


def _iterate(cfg: _SolveCfg, h0: Array, a: _StepArgs) -> tuple[Array, Array]:
    """Damped fixed-count Newton / Picard iterations in the transformed variable; returns ``(h, n_clamp)``.

    Each iteration solves the tridiagonal Newton system in ``v`` and damps the update in three
    ways, none of which changes the residual (so the converged root is the same):

    1. a storage floor ``tl c_floor / dt`` on the Jacobian diagonal: a saturated profile has
       ``C = 0`` and, under a flux top and free drainage, ``dK/dh = 0`` at every node, which
       makes the plain Jacobian singular;
    2. the update is limited to ``|dv| <= dv_max`` per node;
    3. an update that would carry a node from the saturated side across the air-entry kink
       ``v = 0`` (``h = -hb``, where ``C`` jumps from ``a1`` to the Brooks-Corey value) stops
       on the kink, and the next iteration continues from there with the Jacobian of the
       unsaturated side (a "chop": Wang & Tchelepi 2013 limit Newton updates at the kinks
       and inflection points of the flux function in the same way). On the saturated side
       ``C = a1`` (0 for CA-TPA), so the step there is set by the conductances alone and
       overshoots into the dry side;
       without the chop the iterate alternates across the kink while a saturated surface
       drains. Measured on a pond-emptying sub-step (``tests/unit/test_richards.py``): 8
       iterations left a 0.015 cm imbalance without the chop and 3e-6 cm with it; chopping
       the dry-to-wet crossing as well was worse (1.5e-5 cm), chopping each node only once
       per solve worse still (0.03 cm).

    The iterate is then clamped to ``[h_min, h_hi]`` and the clamp activations are counted.
    """
    s = a.soil.hb
    lo = v_of_head(jnp.broadcast_to(a.h_min, h0.shape), s)
    hi = v_of_head(jnp.broadcast_to(a.h_hi, h0.shape), s)
    reg_h = a.tl * cfg.c_floor / a.dt  # storage floor in h units [cm h-1 cm-1]

    def residual_v(x: Array, h_k: Array | None) -> Array:
        hh = head_of_v(x, s)
        return richards_residual(hh, hh if h_k is None else h_k, a)

    def body(_: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        v, nclamp = carry
        h_k = head_of_v(v, s) if cfg.jacobian == "picard" else None
        r, dl, d, du = tridiagonal_jacobian(lambda x: residual_v(x, h_k), v)
        d = d + reg_h * _dh_dv(v, s)
        dv = jnp.clip(_tridiag_solve(dl, d, du, r), -cfg.dv_max, cfg.dv_max)
        v_raw = v - dv
        if cfg.chop:
            v_raw = jnp.where((v > 0.0) & (v_raw < 0.0), 0.0, v_raw)
        clamped = (v_raw < lo) | (v_raw > hi)
        v_new = jnp.clip(v_raw, lo, hi)
        return v_new, nclamp + jnp.sum(clamped).astype(h0.dtype)

    init = (v_of_head(h0, s), jnp.zeros((), h0.dtype))
    v, nclamp = lax.fori_loop(0, cfg.n_iter, body, init)
    return head_of_v(v, s), nclamp


def _dh_dv(v: Array, scale: Array) -> Array:
    """Derivative of :func:`head_of_v`: ``s exp(-v)`` for ``v <= 0``, ``s`` above (both finite)."""
    v_dry = jnp.where(v <= 0.0, v, 0.0)
    return jnp.where(v <= 0.0, scale * jnp.exp(-v_dry), scale)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _solve_implicit(cfg: _SolveCfg, h0: Array, a: _StepArgs) -> tuple[Array, Array]:
    return _iterate(cfg, h0, a)


def _solve_implicit_fwd(
    cfg: _SolveCfg, h0: Array, a: _StepArgs
) -> tuple[tuple[Array, Array], tuple[Array, _StepArgs]]:
    h, nclamp = _iterate(cfg, h0, a)
    return (h, nclamp), (h, a)


def _solve_implicit_bwd(
    cfg: _SolveCfg, res: tuple[Array, _StepArgs], g: tuple[Array, Array]
) -> tuple[Array, _StepArgs]:
    """Implicit-function-theorem VJP: ``dh*/da = -J^{-1} dR/da``, so ``a_bar = -(dR/da)^T J^{-T} g``."""
    h, a = res
    g_h = g[0]
    _, dl, d, du = tridiagonal_jacobian(lambda x: richards_residual(x, x, a), h)
    # transpose of the band: (J^T)_{i,i-1} = J_{i-1,i} = du[i-1], (J^T)_{i,i+1} = J_{i+1,i} = dl[i+1]
    zero = jnp.zeros_like(d[:1])
    dl_t = jnp.concatenate([zero, du[:-1]])
    du_t = jnp.concatenate([dl[1:], zero])
    lam = _tridiag_solve(dl_t, d, du_t, g_h)
    _, vjp = jax.vjp(lambda aa: richards_residual(h, h, aa), a)
    (a_bar,) = vjp(-lam)
    return jnp.zeros_like(h), a_bar


_solve_implicit.defvjp(_solve_implicit_fwd, _solve_implicit_bwd)


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


def richards_step(
    h: Array,
    theta: Array,
    pond: Array,
    soil: SoilHydraulicParams,
    grid: RichardsGrid,
    supply: Array,
    evaporation: Array,
    sink: Array,
    dt: Array,
    alpha: Array,
    h_min: Array,
    pond_max: Array,
    cfg: RichardsConfig,
) -> StepResult:
    """Advance one sub-step of length ``dt`` [h].

    ``supply`` and ``evaporation`` are rates [cm h-1] (``>= 0``); ``sink`` is the sink rate
    [h-1] per node before the ``h_min`` cut: ``[n]`` the root water uptake alone, or
    ``[n_channel, n]`` one row per channel in the order
    :data:`~agrijax.processes.soil_water.sinks.SINK_CHANNELS` (each row capped by the water left
    after the rows before it). ``soil`` must already be on the node axis
    (``SoilHydraulicParams.at_nodes()``).

    Source: Celia et al. (1990) mixed form; Ahuja et al. (2000) ch. 3; RZWQM2 ``RICHRD``/``CHKBC``.
    """
    # RZWQM: no uptake from a node at h_min. Written as an availability cap so that a prescribed
    # uptake can never take a node below theta(h_min) within the sub-step.
    theta_min = theta_of_h(jnp.broadcast_to(h_min, h.shape), soil)
    avail = theta - theta_min - cfg.sink_cutoff
    s_avail = jnp.where(avail > 0.0, avail, 0.0) / dt
    k_upt = SINK_CHANNELS.index("uptake")
    zeros = jnp.zeros(len(SINK_CHANNELS), theta.dtype)
    if sink.ndim == 1:  # static: uptake alone (M1)
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
        h_hi=cfg.h_upper + grid.node_depth(),
    )
    scfg = _SolveCfg(cfg.n_iter, cfg.jacobian, cfg.dv_max, cfg.c_floor, cfg.chop)
    if cfg.grad == "implicit":
        h_new, nclamp = cast(tuple[Array, Array], _solve_implicit(scfg, h, a))
    else:
        h_new, nclamp = _iterate(scfg, h, a)
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


def substep_edges(supply: Array, n_sub: int, rain_fraction: float) -> Array:
    """Sub-step boundaries ``t[n_sub + 1]`` [h] within the day, ``t[0] = 0``, ``t[-1] = 24``.

    A fraction ``1 - rain_fraction`` of the sub-steps is spread uniformly over the day and the
    rest is placed in proportion to the hourly ``supply``: the boundaries split the cumulative
    weight ``w_h = (1 - f)/24 + f supply_h / sum(supply)`` into equal parts. On a dry day (or
    with ``f = 0``) the sub-steps are equal. The longest sub-step is ``24 / ((1 - f) n_sub)`` h,
    and a wetting front moves by less than a node per sub-step during intense rain, which a
    few Newton iterations can follow. The schedule depends on the forcing only, never on
    the state, so the number of sub-steps stays fixed.
    """
    dtype = supply.dtype
    hours = jnp.arange(25, dtype=dtype)
    pos = jnp.where(supply > 0.0, supply, 0.0)
    total = jnp.sum(pos)
    wet = total > 0.0
    share = pos / jnp.where(wet, total, 1.0)
    f = jnp.where(wet, rain_fraction, 0.0)
    w = (1.0 - f) / HOURS_PER_DAY + f * share
    cumw = jnp.concatenate([jnp.zeros_like(w[:1]), jnp.cumsum(w)])
    targets = jnp.linspace(0.0, 1.0, n_sub + 1, dtype=dtype) * cumw[-1]
    t = jnp.interp(targets, cumw, hours)
    return t.at[0].set(0.0).at[-1].set(HOURS_PER_DAY)


def _interval_means(hourly: Array, t: Array) -> Array:
    """Average of an hourly piecewise-constant rate over each interval ``[t[k], t[k+1]]`` (0 if empty)."""
    cum = jnp.concatenate([jnp.zeros_like(hourly[:1]), jnp.cumsum(hourly)])
    c = jnp.interp(t, jnp.arange(25, dtype=hourly.dtype), cum)
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


def richards_substeps(
    water: SoilWater,
    params: RichardsParams,
    t: Array,
    supply: Array,
    evaporation: Array,
    uptake: Array | SinkChannels,
    alphas: Array,
) -> tuple[SoilWater, SubstepTotals]:
    """Advance ``water`` over the sub-steps ``[t[k], t[k+1]]`` [h] of a day; empty ones are the identity.

    ``supply``/``evaporation`` hourly rates ``[24]`` [cm h-1] (averaged over each sub-step),
    ``uptake`` the sink channels (:class:`~agrijax.processes.soil_water.sinks.SinkChannels`) or
    the per-layer root water uptake [cm d-1] alone (spread uniformly over the 24 h), ``alphas``
    the time weight of each sub-step. The returned state keeps ``water.flux``; the totals are
    separate.

    Source: Ahuja et al. (2000) ch. 3; Celia et al. (1990); RZWQM2 ``RICHRD`` (``Rzrich.for``).
    """
    cfg = params.config
    grid = params.grid
    n = grid.n_node
    soil = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,)), params.soil.at_nodes())
    dtype = water.theta.dtype
    dts = t[1:] - t[:-1]
    sup = _interval_means(jnp.asarray(supply, dtype), t)
    eva = _interval_means(jnp.asarray(evaporation, dtype), t)
    channels = as_sink_channels(uptake)
    daily = channels.daily_uptake_only()
    if daily is not None:  # static: the M1/M3 sink, one daily array
        uptake_rate = jnp.asarray(daily, dtype) / (HOURS_PER_DAY * grid.tl)

        def sink_of(t0: Array, dt: Array, th: Array, h: Array) -> Array:
            return uptake_rate

    else:

        def sink_of(t0: Array, dt: Array, th: Array, h: Array) -> Array:
            return jnp.stack([c.node_rate(grid.tl, t0, dt, th, h) for c in channels.channels()])

    h_min = jnp.asarray(params.h_min, dtype)
    pond_max = jnp.asarray(params.pond_max, dtype)

    def body(
        carry: tuple[Array, Array, Array], xs: tuple[Array, Array, Array, Array, Array]
    ) -> tuple[Any, Any]:
        h, th, pd = carry
        s_k, e_k, a_k, dt, t0 = xs
        live = dt > 0.0
        dt_safe = jnp.where(live, dt, 1.0)
        sink = sink_of(t0, dt_safe, th, h)
        r = richards_step(h, th, pd, soil, grid, s_k, e_k, sink, dt_safe, a_k, h_min, pond_max, cfg)
        out = tuple(
            jnp.where(live, x, 0.0)
            for x in (
                r.infiltration,
                r.evaporation,
                r.drainage,
                r.uptake,
                r.runoff,
                r.evaporation_deficit,
                r.uptake_cut,
                r.theta_residual,
                r.n_clamp,
                r.sinks,
                r.sinks_cut,
            )
        )
        new = (jnp.where(live, r.h, h), jnp.where(live, r.theta, th), jnp.where(live, r.pond, pd))
        return new, out

    xs = (sup, eva, alphas, dts, t[:-1])
    (h, th, pd), outs = lax.scan(body, (water.h, water.theta, water.pond), xs)
    infil, evap, drain, upt, runoff, deficit, cut, resid, nclamp, snk, snk_cut = outs
    totals = SubstepTotals(
        supply=jnp.sum(sup * dts),
        infiltration=jnp.sum(infil),
        evaporation=jnp.sum(evap),
        drainage=jnp.sum(drain),
        uptake=jnp.sum(upt),
        runoff=jnp.sum(runoff),
        evaporation_deficit=jnp.sum(deficit),
        uptake_cut=jnp.sum(cut),
        max_theta_residual=jnp.max(resid, initial=0.0),
        n_clamp=jnp.sum(nclamp),
        sinks=jnp.sum(snk, axis=0),
        sinks_cut=jnp.sum(snk_cut, axis=0),
    )
    return water.replace(h=h, theta=th, pond=pd), totals


def day_alphas(n_sub: int, cfg: RichardsConfig, dtype: Any) -> Array:
    """Time weights of a day's sub-steps: 1 on the first, then 1 (``"implicit"``) or 1/2 (``"rzwqm"``)."""
    a_rest = 0.5 if cfg.time_scheme == "rzwqm" else 1.0
    return jnp.where(jnp.arange(n_sub) == 0, 1.0, a_rest).astype(dtype)


def richards_day(
    water: SoilWater,
    params: RichardsParams,
    supply: Array,
    evaporation: Array,
    uptake: Array | SinkChannels,
) -> SoilWater:
    """One day of Richards redistribution: ``n_sub`` sub-steps placed by :func:`substep_edges`.

    ``supply``/``evaporation`` hourly rates ``[24]`` [cm h-1], ``uptake`` the sink channels or
    the per-layer root water uptake [cm d-1] alone.
    Returns the new state with the day's totals in ``flux``. This is the M1 day (the surface
    supply is prescribed; no infiltration event).

    Source: Ahuja et al. (2000) ch. 3; Celia et al. (1990); RZWQM2 ``RICHRD`` (``Rzrich.for``).
    """
    cfg = params.config
    dtype = water.theta.dtype
    supply = jnp.asarray(supply, dtype)
    t = substep_edges(supply, cfg.n_sub, cfg.rain_fraction)
    new, tot = richards_substeps(
        water, params, t, supply, evaporation, uptake, day_alphas(cfg.n_sub, cfg, dtype)
    )
    w0 = water.storage(params.grid) + water.pond
    w1 = new.storage(params.grid) + new.pond
    sinks = tot.uptake + other_sinks(tot)
    balance = (w1 - w0) - (tot.supply - tot.evaporation - tot.drainage - sinks - tot.runoff)
    flux = _zero_fluxes(dtype).replace(
        infiltration=tot.infiltration,
        evaporation=tot.evaporation,
        drainage=tot.drainage,
        uptake=tot.uptake,
        runoff=tot.runoff,
        evaporation_deficit=tot.evaporation_deficit,
        uptake_cut=tot.uptake_cut,
        balance_error=balance,
        max_theta_residual=tot.max_theta_residual,
        n_clamp=tot.n_clamp,
        **sink_fluxes(tot),
    )
    return new.replace(flux=flux)


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------


@process(
    reads=("soil_water.h", "soil_water.theta", "soil_water.pond"),
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
            "the surface supply is the water that infiltrates; the Green-Ampt INFIL step is not part of it",
            "RZWQM2 fills the profile with INFIL and Richards only redistributes; the event is the separate "
            "soil_water/infiltration_ga process, composed with this one in soil_water/day",
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
            "tests/unit/test_richards.py convergence study; tests/integration/test_richards_dump.py",
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
    would otherwise be read as saturation by the ``h < -hb`` branch tests.

    Source: Ahuja et al. (2000) ch. 3; Celia et al. (1990); RZWQM2 ``RICHRD`` (``Rzrich.for``).
    """
    new = richards_day(state.soil_water, params, forcing_t.supply, forcing_t.evaporation, forcing_t.uptake)
    h = new.h
    if check_enabled():
        h = eqx.error_if(h, ~jnp.all(jnp.isfinite(h)), "richards_redistribution: non-finite head")
    return eqx.tree_at(lambda s: s.soil_water, state, new.replace(h=h))
