r"""Green-Ampt infiltration of one storm event, with the conventions of RZWQM2 (branches used at CA-TPA).

RZWQM2 does not pass rain through the Richards equation. Each storm is an *event*: a
Green-Ampt wetting front fills the profile from the top, slice by slice, to the available
porosity; rain in excess of the infiltration capacity runs off; and the Richards step only
redistributes afterwards. The event is instantaneous on the redistribution clock (the day's
time does not advance by the storm duration). This module implements that event from the
published equations, with the conventions RZWQM2 documents and uses:

Grid
----
An infiltration grid of slices of thickness ``ds`` (1 cm in RZWQM2, :class:`GreenAmptConfig`)
covering the node cells exactly; slice ``j`` belongs to the node whose cell contains its centre.
Node -> slice copies the node water content; slice -> node is the thickness-weighted sum, so
storage is conserved exactly.

Per node (vectorised)
---------------------
* available porosity ``theta_a = theta_s AEF`` (``AEF`` the field-saturation fraction, 0.9 at
  CA-TPA); a slice is saturated when ``theta >= theta_a``;
* wetting-front suction from the initial suction ``s_i = -h(min(theta, theta_a))`` (Mein &
  Larson 1973; Ahuja et al. 2000 ch. 3), the Brooks-Corey conductivity integral
  ``S_f = 1 + int_1^{s_i} K(-s)/K_s ds + pond`` evaluated in closed form on the two
  power-law segments of :func:`~agrijax.processes.soil_water.hydraulics.k_of_h` (RZWQM2 takes
  the integral from 1 cm and counts the first centimetre as 1; for ``s_i <= 1`` the integral
  term is 1). ``pond = 0`` for rain;
* front conductance ``C_i``: ``K`` at the air-entry head, capped at ``K_s``, and forced
  non-increasing with depth (a running minimum); the surface node uses ``K_s`` (no crust);
* series resistance above node ``i``: ``P_i = sum_{m<i} tl_m / C_m`` (a prefix sum).

Per slice (vectorised)
----------------------
With the front at the centre of slice ``j``, depth ``z_f``, in node ``i`` whose cell starts at
depth ``z_i``, the layered Green-Ampt infiltration capacity is

.. math::

    V_j = \frac{C_i (S_{f,i} + z_f)}{C_i P_i + (z_f - z_i)} \Big/ r_c ,

the flux through the saturated layers above the front in series (Darcy with the
wetting-front suction as the head at the front; Green & Ampt 1911, layered form as in
Ahuja et al. 2000 ch. 3); ``r_c = 2`` is RZWQM2's reduction factor ``VRCF``.

Recurrence over the slices (:func:`agrijax.core.depth_scan.depth_scan`)
-----------------------------------------------------------------------
The front advances one unsaturated slice per step: the step fills ``dq = (theta_a - theta) ds``
at the rate ``v = min(V_j, r)``, where ``r = max(r(R), r_min)`` is the breakpoint rain
intensity at the cumulative rain ``R`` consumed so far (the interval that contains ``R``);
the step lasts ``dt = dq / v``, floored at ``dt_min`` by lowering ``v`` (``1e-5`` h), and the
rain of the step is ``r dt``, of which ``r dt - dq`` runs off. The step that would pass the
storm total is cut to the rain time left, ``dt = (R_tot - R) / r``, and fills ``v dt`` only;
the event ends there. Saturated slices are skipped. Because ``r`` depends on the rain
consumed above, this is a true recurrence along depth; everything else is vectorised.

If the whole profile saturates before the rain ends, the remaining rain passes through the
saturated profile as bottom seepage (RZWQM2 with no water table), or runs off when the
profile was saturated before the storm.

Branches not implemented (inactive at CA-TPA, deferred as L-GA+): surface crust, macropore
flow, water table / tile drains / saturated flow below the front, unit-gradient flow below the
front, ponded irrigation (``INFLPD``), snowmelt events (L-snow).

Differentiability: every quotient is on a clamped argument (``r >= r_min``, ``V > 0``,
``dq > 0`` on active slices, guarded elsewhere); the slice at which the rain runs out and the
breakpoint interval are masks with zero gradient.

Source: Green, W.H., Ampt, G.A., 1911. Studies on soil physics: 1. The flow of air and water
through soils. J. Agric. Sci. 4, 1-24. Mein, R.G., Larson, C.L., 1973. Modeling infiltration
during a steady rain. Water Resour. Res. 9, 384-394. Ahuja, L.R., Rojas, K.W., Hanson, J.D.,
Shaffer, M.J., Ma, L. (eds.), 2000. Root Zone Water Quality Model, ch. 3 (Green-Ampt
infiltration in layered soil, wetting-front suction, event hydrology). The corresponding
RZWQM2 routines are ``EVNTRO``, ``INFIL``, ``UNSATFLO`` and ``MIXRUNOFF`` (``RZTEST.for``), read
for conventions only.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.core import Tracer
from jaxtyping import Array

from agrijax.core.depth_scan import depth_scan
from agrijax.core.dims import register_dim
from agrijax.core.state import Forcing, Params, field

from .hydraulics import SoilHydraulicParams, c2_of_params, h_of_theta, k_of_h

__all__ = [
    "GAResult",
    "GreenAmptConfig",
    "GreenAmptParams",
    "StormForcing",
    "front_conductance",
    "green_ampt_capacity",
    "green_ampt_event",
    "slice_nodes",
    "wetting_front_suction",
]

register_dim("n_bp", "breakpoint intervals of one day's storm segment (padded with zeros)")

#: RZWQM2 constants of the event (documented values of the reference model)
VRCF: float = 2.0  # reduction factor of the Green-Ampt capacity
DT_MIN: float = 1.0e-5  # [h] shortest front step
RR_MIN: float = 1.0e-2  # [cm h-1] floor of the rain intensity
_SAT_TOL: float = 1.0e-12  # [cm3 cm-3] a slice within this of theta_s*AEF is saturated


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


class GreenAmptConfig(eqx.Module):
    """Static settings of the event: slice count ``n_slice`` and thickness ``ds`` [cm], constants."""

    n_slice: int = eqx.field(static=True)
    ds: float = eqx.field(static=True, default=1.0)
    vrcf: float = eqx.field(static=True, default=VRCF)
    dt_min: float = eqx.field(static=True, default=DT_MIN)
    rr_min: float = eqx.field(static=True, default=RR_MIN)

    def __check_init__(self) -> None:
        if self.n_slice < 1 or self.ds <= 0.0 or self.vrcf <= 0.0 or self.dt_min <= 0.0 or self.rr_min <= 0.0:
            raise ValueError("n_slice >= 1 and ds, vrcf, dt_min, rr_min > 0 are required")

    @classmethod
    def for_grid(cls, tl: Any, ds: float = 1.0, **kw: Any) -> GreenAmptConfig:
        """Slices of ``ds`` [cm] over the cells ``tl``; every cell boundary must be a multiple of ``ds``."""
        tlt = np.cumsum(np.asarray(tl, dtype=float))
        k = tlt / ds
        _require(
            bool(np.allclose(k, np.round(k), atol=1e-9, rtol=0.0)),
            f"cell boundaries {tlt} are not multiples of the slice thickness {ds}",
        )
        return cls(n_slice=round(float(k[-1])), ds=float(ds), **kw)

    def check_grid(self, tl: Any) -> None:
        """Raise unless the slices span exactly the cells ``tl`` (skipped when ``tl`` is traced).

        A mismatch would silently misplace water: slices below the grid are clipped into the
        last cell, and a grid deeper than the slices stops the front early.
        """
        if isinstance(tl, Tracer):
            return
        depth = np.sum(np.asarray(tl, dtype=float), axis=-1)
        span = self.n_slice * self.ds
        _require(
            bool(np.all(np.abs(depth - span) <= 1e-6 * max(1.0, span))),
            f"Green-Ampt slices span {span} cm but the grid is {depth} cm deep; "
            "build the config with GreenAmptConfig.for_grid(tl)",
        )


class GreenAmptParams(Params):
    """Parameters of the event that are not soil hydraulics (those come from the Richards parameters)."""

    config: GreenAmptConfig = eqx.field(static=True)  # no default: build it with GreenAmptConfig.for_grid(tl)
    aef: Array = field(
        dims=(),
        unit="-",
        description="field-saturation fraction: available porosity = theta_s * aef",
        fortran_name="AEF",
        default=0.9,
    )


class StormForcing(Forcing):
    """One day's storm segment: start time and breakpoint intervals (padded, time axis first when stacked).

    ``depth.sum() == 0`` means no event that day. Storms that span midnight are split into
    one segment per day (the continuation starts at ``ts0 = 0``), as RZWQM2 ``CHSPAN`` does.
    """

    ts0: Array = field(unit="h", dims=("T",), description="clock time of the storm start", fortran_name="TS0")
    duration: Array = field(
        unit="h", dims=("T", "n_bp"), description="length of each breakpoint interval", fortran_name="STMSEG1"
    )
    depth: Array = field(
        unit="cm", dims=("T", "n_bp"), description="rain in each breakpoint interval", fortran_name="STMSEG2"
    )


class GAResult(NamedTuple):
    """Outcome of one event (depths [cm])."""

    theta: Array
    h: Array
    rain: Array
    infiltration: Array
    runoff: Array
    seepage: Array
    error: Array  # rain - infiltration - runoff - seepage (rounding)
    duration: Array  # [h] event time on the Green-Ampt clock (not added to the day)
    front_depth: Array  # [cm] bottom of the deepest slice filled


# ---------------------------------------------------------------------------
# vectorised pieces
# ---------------------------------------------------------------------------


def _where_min(a: Array, b: Array) -> Array:
    """``min`` that sends the whole cotangent to ``a`` on ties (``jnp.minimum`` would split it)."""
    return jnp.where(b < a, b, a)


def _int_pow(a: Array, b: Array, p: Array) -> Array:
    """``int_a^b s^-p ds`` for ``b >= a >= 1``; at ``p = 1`` the logarithm (both branches finite)."""
    q = 1.0 - p
    near = jnp.abs(q) < 1.0e-6
    q_safe = jnp.where(near, 1.0, q)
    a_safe = jnp.where(a > 1.0, a, 1.0)
    b_safe = jnp.where(b > a_safe, b, a_safe)
    log_ratio = jnp.log(b_safe) - jnp.log(a_safe)
    general = (b_safe**q_safe - a_safe**q_safe) / q_safe
    return jnp.where(near, log_ratio, general)


def wetting_front_suction(
    theta: Array, soil: SoilHydraulicParams, theta_avail: Array, pond: Array | float = 0.0
) -> Array:
    """Wetting-front suction ``S_f`` [cm] per node from the initial water content (Mein & Larson 1973).

    ``S_f = 1 + int_1^{s_i} K(-s)/K_s ds + pond`` with ``s_i = -h(min(theta, theta_avail))``; the
    integral is closed-form on the two power-law segments of ``K`` (``K_s s^-n1`` up to
    ``hb_k``, ``C2 s^-eps`` beyond) and is 1 for ``s_i <= 1`` (RZWQM2 convention). ``soil`` on
    the node axis.

    Source: Mein & Larson (1973); Ahuja et al. (2000) ch. 3; RZWQM2 ``EVNTRO`` (conventions).
    """
    w_init = _where_min(theta, theta_avail)
    s_i = -h_of_theta(w_init, soil)
    hbk = jnp.where(soil.hb_k > 1.0, soil.hb_k, 1.0)
    s_wet = jnp.clip(s_i, 1.0, hbk)
    s_dry = jnp.where(s_i > hbk, s_i, hbk)
    wet = _int_pow(jnp.ones_like(s_wet), s_wet, soil.n1)
    dry = c2_of_params(soil) / soil.ksat * _int_pow(hbk, s_dry, soil.eps)
    integral = jnp.where(s_i > 1.0, wet + dry, 1.0)
    return integral + pond + 1.0


def front_conductance(soil: SoilHydraulicParams) -> Array:
    """Conductance of each node for the front ``C_i`` [cm h-1]: ``K(-hb)`` capped at ``K_s``, non-increasing.

    The surface node takes ``K_s`` (no crust). The running minimum over depth is an
    associative scan, not a loop.

    Source: Ahuja et al. (2000) ch. 3; RZWQM2 ``EVNTRO`` (conventions).
    """
    k_air = k_of_h(-soil.hb, soil)
    c = _where_min(k_air, soil.ksat)
    c = jnp.concatenate([soil.ksat[:1], c[1:]])
    return lax.associative_scan(_where_min, c)


def slice_nodes(tl: Array, cfg: GreenAmptConfig) -> tuple[Array, Array]:
    """``(node_of_slice[n_slice], slice_centre_depth[n_slice])``: the cell that contains each slice centre."""
    tlt = jnp.cumsum(tl)
    zc = (jnp.arange(cfg.n_slice, dtype=tl.dtype) + 0.5) * cfg.ds
    idx = jnp.searchsorted(tlt, zc, side="left")
    return jnp.clip(idx, 0, tl.shape[-1] - 1), zc


def green_ampt_capacity(
    tl: Array, conductance: Array, suction: Array, node: Array, z_front: Array, cfg: GreenAmptConfig
) -> Array:
    """Layered Green-Ampt capacity ``V_j`` [cm h-1], front at depth ``z_front[j]`` in node ``node[j]``.

    ``V = C_i (S_f,i + z_f) / (C_i P_i + z_f - z_i) / r_c`` with ``P_i = sum_{m<i} tl_m / C_m``
    and ``z_i`` the top of cell ``i`` (Green & Ampt 1911, series form).

    Source: Green & Ampt (1911); Ahuja et al. (2000) ch. 3; RZWQM2 ``INFIL`` (conventions).
    """
    resist = jnp.cumsum(tl / conductance) - tl / conductance  # exclusive prefix sum
    top = jnp.cumsum(tl) - tl
    c = conductance[node]
    below_top = z_front - top[node]
    return c * (suction[node] + z_front) / (c * resist[node] + below_top) / cfg.vrcf


# ---------------------------------------------------------------------------
# the event
# ---------------------------------------------------------------------------


def _rain_intensity(consumed: Array, cum: Array, rate: Array) -> Array:
    """Breakpoint intensity at cumulative rain ``consumed``: the first interval with ``consumed <= cum``."""
    first = jnp.argmax(consumed <= cum)
    return rate[first]


def green_ampt_event(
    theta: Array,
    h: Array,
    soil: SoilHydraulicParams,
    tl: Array,
    aef: Array,
    duration: Array,
    depth: Array,
    cfg: GreenAmptConfig,
) -> GAResult:
    """Fill the profile with one storm (RZWQM2 conventions); the identity when ``depth.sum() == 0``.

    ``theta``, ``h`` on the nodes, ``soil`` on the node axis, ``tl`` the cell thicknesses,
    ``duration``/``depth`` the breakpoint intervals [h]/[cm]. Nodes that receive water get
    ``h = h(theta)``; the others keep their head and water content bit for bit.

    Source: Green & Ampt (1911); Mein & Larson (1973); Ahuja et al. (2000) ch. 3; RZWQM2
    ``EVNTRO``/``INFIL``/``UNSATFLO``/``MIXRUNOFF`` (conventions).
    """
    cfg.check_grid(tl)
    dtype = theta.dtype
    n = tl.shape[-1]
    theta_avail = soil.theta_s * aef
    node, zc = slice_nodes(tl, cfg)
    deficit = theta_avail[node] - theta[node]
    active = deficit > _SAT_TOL
    dq = jnp.where(active, deficit, 0.0) * cfg.ds

    suction = wetting_front_suction(theta, soil, theta_avail)
    cond = front_conductance(soil)
    cap = green_ampt_capacity(tl, cond, suction, node, zc, cfg)

    depth = jnp.asarray(depth, dtype)
    dur = jnp.asarray(duration, dtype)
    cum = jnp.cumsum(depth)
    total = jnp.sum(depth)
    rate = jnp.where(dur > 0.0, depth / jnp.where(dur > 0.0, dur, 1.0), 0.0)
    rr_min = jnp.asarray(cfg.rr_min, dtype)

    def step(carry: tuple[Array, Array, Array], x: tuple[Array, Array]) -> tuple[Any, Any]:
        consumed, done, filled_any = carry
        v_cap, dqj = x
        rr = _rain_intensity(consumed, cum, rate)
        rr = jnp.where(rr > rr_min, rr, rr_min)
        v = _where_min(v_cap, rr)
        short = dqj / v < cfg.dt_min
        v_floor = _where_min(dqj / cfg.dt_min, rr)
        v = jnp.where(short, jnp.where(v_floor > 0.0, v_floor, rr), v)
        dt_full = dqj / v
        rain_full = rr * dt_full
        # the last step: rain stops after (R_tot - R) / r, and the slice fills at v only that long
        over = consumed + rain_full > total
        dt_left = (total - consumed) / rr  # rr >= rr_min > 0
        dt = jnp.where(over, dt_left, dt_full)
        fill = jnp.where(over, v * dt_left, dqj)
        rain = jnp.where(over, total - consumed, rain_full)
        go = ~done
        fill = jnp.where(go, fill, 0.0)
        rain = jnp.where(go, rain, 0.0)
        dt = jnp.where(go, dt, 0.0)
        consumed_new = jnp.where(go, jnp.where(over, total, consumed + rain), consumed)
        done_new = done | (consumed_new >= total)
        return (consumed_new, done_new, filled_any | go), (fill, rain, dt)

    zero = jnp.zeros((), dtype)
    init = (zero, total <= 0.0, jnp.zeros((), bool))
    (consumed, done, filled_any), (fill, rain, dt) = depth_scan(step, init, (cap, dq), mask=active)
    remaining = jnp.where(done, 0.0, total - consumed)
    seepage = jnp.where(filled_any, remaining, 0.0)
    extra_runoff = jnp.where(filled_any, 0.0, remaining)
    infiltration = jnp.sum(fill)
    runoff = jnp.sum(rain - fill) + extra_runoff

    added = jax.ops.segment_sum(fill, node, num_segments=n)
    theta_new = theta + added / tl
    h_new = jnp.where(added > 0.0, h_of_theta(theta_new, soil), h)
    deepest = jnp.max(jnp.where(fill > 0.0, zc + 0.5 * cfg.ds, 0.0))
    return GAResult(
        theta=theta_new,
        h=h_new,
        rain=total,
        infiltration=infiltration,
        runoff=runoff,
        seepage=seepage,
        error=total - infiltration - runoff - seepage,
        duration=jnp.sum(dt),
        front_depth=deepest,
    )
