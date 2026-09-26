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

Coefficients and settings: the numbers RZWQM2 hard-codes in the event (``VRCF``, the offset and
the 1-cm lower limit of the suction integral) are calibratable coefficients,
:class:`~agrijax.processes.soil_water.coefficients.GreenAmptCoefficients`
(``GreenAmptParams.coefficients``, ``None`` for the RZWQM2 values); the slice thickness, the
shortest front step, the rain-intensity floor and the saturation tolerance are RZWQM2
numerical settings (:data:`~agrijax.processes.soil_water.coefficients.SETTINGS`).

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

After tillage (``soil`` a :class:`~agrijax.processes.soil_water.hydraulics.TilledSoilHydraulicParams`)
the initial suction ``s_i`` and the post-event heads come from the two-segment retention curve
RZWQM2 evaluates in ``WCH`` (pre-tillage curve at the dry end), and everything else (porosity,
the conductivity integral, the front conductance at the air-entry head) from the current
parameters, as ``EVNTRO`` reads ``SOILHP`` directly there.

Branches not implemented (deferred as L-GA+; inactive on the 15 ``RZWQM_sw_batch`` scenarios,
active on the tile-drained ones, ``tests/integration/test_green_ampt_scenarios.py``): surface
crust, macropore flow, water table / tile drains / saturated flow below the front, unit-gradient
flow below the front, bottom-flux cap (``IREBOT = 3``), ice-limited porosity (``PORI``), plastic
mulch, ponded irrigation (``INFLPD``), snowmelt events (L-snow). Irrigation events that enter
with breakpoints do not match from their entry state either; the cause is open (L-irr).

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

from agrijax.core.coefficients import numerical_guard
from agrijax.core.depth_scan import depth_scan
from agrijax.core.dims import register_dim
from agrijax.core.state import Forcing, Params, field

from .coefficients import RZWQM2_GREEN_AMPT, GreenAmptCoefficients, numerical_setting, rzwqm2
from .hydraulics import (
    AnyHydraulicParams,
    SoilHydraulicParams,
    TilledSoilHydraulicParams,
    c2_of_params,
    h_of_theta,
    k_of_h,
)

__all__ = [
    "GAResult",
    "GreenAmptCoefficients",
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

#: reduction factor of the Green-Ampt capacity (the coefficient ``GreenAmptCoefficients.vrcf``)
VRCF: float = RZWQM2_GREEN_AMPT.vrcf
#: RZWQM2 numerical settings of the event (conventions of the reference model)
DT_MIN: float = numerical_setting(
    "green_ampt.dt_min",
    1.0e-5,
    "h",
    "shortest front step: a slice that would fill faster is filled at the rate dq / dt_min",
    origin="rzwqm2-4.6",
    provenance=rzwqm2("RZWQM/RZTEST.for:1284", "INFIL", note="DTMIN; applied at RZTEST.for:1394"),
)
RR_MIN: float = numerical_setting(
    "green_ampt.rr_min",
    1.0e-2,
    "cm h-1",
    "floor of the breakpoint rain intensity the front advances at (keeps dt = dq / v finite)",
    origin="rzwqm2-4.6",
    provenance=rzwqm2("RZWQM/RZTEST.for:1384", "INFIL"),
)
_SAT_TOL: float = numerical_setting(
    "green_ampt.sat_tol",
    1.0e-12,
    "cm3 cm-3",
    "a slice within this of the available porosity theta_s*AEF counts as saturated (skipped)",
    origin="rzwqm2-4.6",
    provenance=rzwqm2("RZWQM/RZTEST.for:1348", "INFIL"),
)
SLICE_DS: float = numerical_setting(
    "green_ampt.ds",
    1.0,
    "cm",
    "thickness of the infiltration slices the front fills one by one",
    origin="rzwqm2-4.6",
    provenance=rzwqm2("RZWQM/RZTEST.for:4090", "NTRPTR", note="the infiltration grid of 1-cm layers"),
)
#: centre of a slice as a fraction of its thickness (slice j spans [j ds, (j + 1) ds])
_SLICE_CENTRE: float = numerical_setting(
    "green_ampt.slice_centre",
    0.5,
    "-",
    "position of the front within a slice, as a fraction of its thickness (the centre)",
    origin="agrijax",
    basis="infiltration.py module docstring (slice j belongs to the node whose cell contains its centre)",
)
_P_NEAR_ONE = numerical_guard(
    "green_ampt.p_near_one",
    1.0e-6,
    "|1 - p| below which int s^-p ds is taken in its logarithmic form (both forms finite)",
)
_GRID_ATOL = numerical_guard(
    "green_ampt.grid_atol", 1.0e-9, "absolute tolerance of 'cell boundary is a multiple of ds' (file check)"
)
_SPAN_RTOL = numerical_guard(
    "green_ampt.span_rtol", 1.0e-6, "relative tolerance of 'the slices span the grid depth' (config check)"
)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


class GreenAmptConfig(eqx.Module):
    """Static numerical settings of the event: slice count ``n_slice`` and thickness ``ds`` [cm],
    the shortest front step ``dt_min`` [h] and the rain-intensity floor ``rr_min`` [cm h-1]
    (RZWQM2 conventions, :data:`~agrijax.processes.soil_water.coefficients.SETTINGS`). The
    capacity reduction factor is a model coefficient (:class:`GreenAmptCoefficients`)."""

    n_slice: int = eqx.field(static=True, metadata={"unit": "-", "description": "number of slices"})
    ds: float = eqx.field(
        static=True,
        default=SLICE_DS,
        metadata={"unit": "cm", "description": "slice thickness", "setting": "green_ampt.ds"},
    )
    dt_min: float = eqx.field(
        static=True,
        default=DT_MIN,
        metadata={"unit": "h", "description": "shortest front step", "setting": "green_ampt.dt_min"},
    )
    rr_min: float = eqx.field(
        static=True,
        default=RR_MIN,
        metadata={"unit": "cm h-1", "description": "rain-intensity floor", "setting": "green_ampt.rr_min"},
    )

    def __check_init__(self) -> None:
        if self.n_slice < 1 or self.ds <= 0.0 or self.dt_min <= 0.0 or self.rr_min <= 0.0:
            raise ValueError("n_slice >= 1 and ds, dt_min, rr_min > 0 are required")

    @classmethod
    def for_grid(cls, tl: Any, ds: float = SLICE_DS, **kw: Any) -> GreenAmptConfig:
        """Slices of ``ds`` [cm] over the cells ``tl``; every cell boundary must be a multiple of ``ds``."""
        tlt = np.cumsum(np.asarray(tl, dtype=float))
        k = tlt / ds
        _require(
            bool(np.allclose(k, np.round(k), atol=_GRID_ATOL, rtol=0.0)),
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
            bool(np.all(np.abs(depth - span) <= _SPAN_RTOL * max(1.0, span))),
            f"Green-Ampt slices span {span} cm but the grid is {depth} cm deep; "
            "build the config with GreenAmptConfig.for_grid(tl)",
        )


class GreenAmptParams(Params):
    """Parameters of the event that are not soil hydraulics (those come from the Richards parameters)."""

    config: GreenAmptConfig = eqx.field(static=True)  # no default: build it with GreenAmptConfig.for_grid(tl)
    aef: Array = field(
        dims=(),
        unit="-",
        description="field-saturation fraction: available porosity = theta_s * aef (rzwqm.dat Richards "
        "control record; 0.9 at CA-TPA)",
        fortran_name="AEF",
        default=0.9,
    )
    coefficients: GreenAmptCoefficients | None = field(
        description="the coefficients RZWQM2 hard-codes in the event (None: the RZWQM2 4.6 values)",
        default=None,
    )

    def coef(self) -> GreenAmptCoefficients:
        """The event coefficients in force: :attr:`coefficients`, or the RZWQM2 values."""
        return RZWQM2_GREEN_AMPT if self.coefficients is None else self.coefficients


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
    near = jnp.abs(q) < _P_NEAR_ONE
    q_safe = jnp.where(near, 1.0, q)
    a_safe = jnp.where(a > 1.0, a, 1.0)
    b_safe = jnp.where(b > a_safe, b, a_safe)
    log_ratio = jnp.log(b_safe) - jnp.log(a_safe)
    general = (b_safe**q_safe - a_safe**q_safe) / q_safe
    return jnp.where(near, log_ratio, general)


def _current(soil: AnyHydraulicParams) -> SoilHydraulicParams:
    """The parameters of the curve now (``SOILHP``): ``soil`` itself, or its ``current`` after tillage."""
    return soil.current if isinstance(soil, TilledSoilHydraulicParams) else soil


def wetting_front_suction(
    theta: Array,
    soil: AnyHydraulicParams,
    theta_avail: Array,
    pond: Array | float = 0.0,
    coefs: GreenAmptCoefficients = RZWQM2_GREEN_AMPT,
) -> Array:
    """Wetting-front suction ``S_f`` [cm] per node from the initial water content (Mein & Larson 1973).

    ``S_f = 1 + int_1^{s_i} K(-s)/K_s ds + pond`` with ``s_i = -h(min(theta, theta_avail))``; the
    integral is closed-form on the two power-law segments of ``K`` (``K_s s^-n1`` up to
    ``hb_k``, ``C2 s^-eps`` beyond) and is 1 for ``s_i <= 1`` (RZWQM2 convention). ``soil`` on
    the node axis. The offset, the lower limit (1 cm) and the value below it are the
    ``coefs`` (:class:`GreenAmptCoefficients`). After tillage ``s_i`` is read from the
    two-segment retention curve of ``soil`` (``WCH``) and the integral from the current
    parameters (``SOILHP``).

    Source: Mein & Larson (1973); Ahuja et al. (2000) ch. 3; RZWQM2 ``EVNTRO`` (conventions).
    """
    lower = coefs.suction_lower
    w_init = _where_min(theta, theta_avail)
    s_i = -h_of_theta(w_init, soil)
    soil = _current(soil)
    hbk = jnp.where(soil.hb_k > lower, soil.hb_k, lower)
    s_wet = jnp.clip(s_i, lower, hbk)
    s_dry = jnp.where(s_i > hbk, s_i, hbk)
    wet = _int_pow(jnp.full_like(s_wet, lower), s_wet, soil.n1)
    dry = c2_of_params(soil) / soil.ksat * _int_pow(hbk, s_dry, soil.eps)
    integral = jnp.where(s_i > lower, wet + dry, coefs.suction_dry_limit)
    return integral + pond + coefs.suction_offset


def front_conductance(soil: AnyHydraulicParams) -> Array:
    """Conductance of each node for the front ``C_i`` [cm h-1]: ``K(-hb)`` capped at ``K_s``, non-increasing.

    The surface node takes ``K_s`` (no crust). The running minimum over depth is an
    associative scan, not a loop. ``K`` is evaluated on ``soil`` (the two-segment curve after
    tillage, ``POINTK``) at the current air-entry head.

    Source: Ahuja et al. (2000) ch. 3; RZWQM2 ``EVNTRO`` (conventions).
    """
    cur = _current(soil)
    k_air = k_of_h(-cur.hb, soil)
    c = _where_min(k_air, cur.ksat)
    c = jnp.concatenate([cur.ksat[:1], c[1:]])
    return lax.associative_scan(_where_min, c)


def slice_nodes(tl: Array, cfg: GreenAmptConfig) -> tuple[Array, Array]:
    """``(node_of_slice[n_slice], slice_centre_depth[n_slice])``: the cell that contains each slice centre."""
    tlt = jnp.cumsum(tl)
    zc = (jnp.arange(cfg.n_slice, dtype=tl.dtype) + _SLICE_CENTRE) * cfg.ds
    idx = jnp.searchsorted(tlt, zc, side="left")
    return jnp.clip(idx, 0, tl.shape[-1] - 1), zc


def green_ampt_capacity(
    tl: Array,
    conductance: Array,
    suction: Array,
    node: Array,
    z_front: Array,
    cfg: GreenAmptConfig,
    coefs: GreenAmptCoefficients = RZWQM2_GREEN_AMPT,
) -> Array:
    """Layered Green-Ampt capacity ``V_j`` [cm h-1], front at depth ``z_front[j]`` in node ``node[j]``.

    ``V = C_i (S_f,i + z_f) / (C_i P_i + z_f - z_i) / r_c`` with ``P_i = sum_{m<i} tl_m / C_m``
    and ``z_i`` the top of cell ``i`` (Green & Ampt 1911, series form); ``r_c`` is
    ``coefs.vrcf``. ``cfg`` is kept for the call signature (the slices are in ``z_front``).

    Source: Green & Ampt (1911); Ahuja et al. (2000) ch. 3; RZWQM2 ``INFIL`` (conventions).
    """
    resist = jnp.cumsum(tl / conductance) - tl / conductance  # exclusive prefix sum
    top = jnp.cumsum(tl) - tl
    c = conductance[node]
    below_top = z_front - top[node]
    return c * (suction[node] + z_front) / (c * resist[node] + below_top) / coefs.vrcf


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
    soil: AnyHydraulicParams,
    tl: Array,
    aef: Array,
    duration: Array,
    depth: Array,
    cfg: GreenAmptConfig,
    coefs: GreenAmptCoefficients | None = None,
) -> GAResult:
    """Fill the profile with one storm (RZWQM2 conventions); the identity when ``depth.sum() == 0``.

    ``theta``, ``h`` on the nodes, ``soil`` on the node axis, ``tl`` the cell thicknesses,
    ``duration``/``depth`` the breakpoint intervals [h]/[cm]. Nodes that receive water get
    ``h = h(theta)``; the others keep their head and water content bit for bit. ``coefs`` are
    the event coefficients (``None``: the RZWQM2 values, :data:`RZWQM2_GREEN_AMPT`). ``soil``
    may be the post-tillage :class:`TilledSoilHydraulicParams` (see the module docstring).

    Source: Green & Ampt (1911); Mein & Larson (1973); Ahuja et al. (2000) ch. 3; RZWQM2
    ``EVNTRO``/``INFIL``/``UNSATFLO``/``MIXRUNOFF`` (conventions).
    """
    cfg.check_grid(tl)
    coefs = RZWQM2_GREEN_AMPT if coefs is None else coefs
    dtype = theta.dtype
    n = tl.shape[-1]
    theta_avail = _current(soil).theta_s * aef
    node, zc = slice_nodes(tl, cfg)
    deficit = theta_avail[node] - theta[node]
    active = deficit > _SAT_TOL
    dq = jnp.where(active, deficit, 0.0) * cfg.ds

    suction = wetting_front_suction(theta, soil, theta_avail, coefs=coefs)
    cond = front_conductance(soil)
    cap = green_ampt_capacity(tl, cond, suction, node, zc, cfg, coefs)

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
    deepest = jnp.max(jnp.where(fill > 0.0, zc + _SLICE_CENTRE * cfg.ds, 0.0))
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
