r"""The fixed-step integrator of the Richards problem (``FixedCN``, key ``:alt_fixed_steps``).

A day of ``n_sub`` sub-steps with ``n_iter`` damped Newton (or modified Picard) iterations each and
no convergence test: the modes the soil-water kernels had before the adaptive stepping (the 24 x 3
baseline, the near-converged 96 x 8 configuration, the Crank-Nicolson ``time_scheme="rzwqm"`` and the
fully implicit default), kept bit for bit as the legacy scheme and as the baseline the adaptive
scheme is compared with. Its config is :class:`FixedStepping`.

Sub-steps: ``n_sub`` per day. A fraction ``rain_fraction`` of them is placed in proportion to the
hourly supply, the rest spread uniformly (:func:`substep_edges`); the schedule depends on the
forcing only, so the count stays fixed while a wetting front moves by less than a node per
sub-step. On an event day the ``n_pre`` / ``n_post`` sub-steps of the day's schedule are placed
before and after the event (:func:`day_edges`). Time weights: ``alpha = 1`` on every
sub-step (``time_scheme="implicit"``, the default), or RZWQM2's pattern ``alpha = 1`` on the first
sub-step of a day and ``1/2`` afterwards (``"rzwqm"``).

Iteration: Newton on the transformed head ``v`` (:func:`~agrijax.processes.soil_water.problem.head_of_v`)
with the exact tridiagonal Jacobian of the problem's residual, damped without touching the residual
(:func:`_iterate`: storage floor ``c_floor`` on the diagonal, ``|dv| <= dv_max``, the air-entry
``chop``, the clamp to ``[h_min, h_upper + z_i]``); ``jacobian="picard"`` freezes ``K`` at the
current iterate (RZWQM's modified Picard). Gradients: ``grad="unrolled"`` differentiates through the
fixed iterations; ``grad="implicit"`` gives each sub-step solve an implicit-function-theorem VJP
(``J(h*)^T lambda = g``, tridiagonal), exact at a converged root.

Source: Ahuja, L.R., Rojas, K.W., Hanson, J.D., Shaffer, M.J., Ma, L. (eds.), 2000. Root Zone Water
Quality Model, ch. 3; Celia, M.A., Bouloutas, E.T., Zarba, R.L., 1990. Water Resour. Res. 26,
1483-1496 (mixed form); Wang, X., Tchelepi, H.A., 2013. J. Comput. Phys. 253, 114-137 (the analogy
behind ``chop``); RZWQM2 ``RICHRD`` (``Rzrich.for``), read for conventions only.
"""

from __future__ import annotations

from functools import partial
from typing import Any, ClassVar, NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from agrijax.core.coefficients import Provenance
from agrijax.core.process import Deviation, Source

from . import richards_adaptive as _faithful  # noqa: F401  (registers the faithful sibling first)
from .coefficients import numerical_setting, setting_field
from .hydraulics import AnyHydraulicParams
from .integrator import (
    _ALPHA_CN,
    _ALPHA_FIRST,
    Capabilities,
    DayDiagnostics,
    DayPlan,
    IntegratorInfo,
    SteppingConfig,
    combine_totals,
    register_integrator,
)
from .problem import (
    _N_HOUR_EDGES,
    HOURS_PER_DAY,
    RichardsConfig,
    RichardsGrid,
    RichardsProblem,
    SoilWater,
    StepResult,
    SubstepTotals,
    _dh_dv,
    _step_args,
    _step_result,
    _StepArgs,
    _tridiag_solve,
    head_of_v,
    richards_residual,
    tridiagonal_jacobian,
    v_of_head,
)
from .sinks import SinkChannels

__all__ = [
    "FixedCN",
    "FixedStepping",
    "day_alphas",
    "day_edges",
    "richards_step",
    "richards_substeps",
    "substep_edges",
]

#: key of the fixed-step integrator
KEY = "soil_water/richards_time@rzwqm2-4.6:alt_fixed_steps"

#: the 96 x 8 configuration of the fixed steps (FixedStepping.m1())
M1_N_SUB: int = numerical_setting(
    "richards.m1_n_sub",
    96,
    "-",
    "sub-steps per day of the 96 x 8 configuration FixedStepping.m1() (near-converged fixed steps)",
    origin="agrijax",
    basis="the year-by-year replays of CA-TPA 2015-2023 with the prescribed supply: 96 x 8 meets the "
    "0.05 cm storage bound in all nine years, 24 x 3 does not (the RZWQM2 4.6 replays, data tier "
    "outside this repository)",
)
M1_N_ITER: int = numerical_setting(
    "richards.m1_n_iter",
    8,
    "-",
    "Newton iterations per sub-step of the 96 x 8 configuration FixedStepping.m1()",
    origin="agrijax",
    basis="the year-by-year replays of CA-TPA 2015-2023 with the prescribed supply: 96 x 8 meets the "
    "0.05 cm storage bound in all nine years, 24 x 3 does not (the RZWQM2 4.6 replays, data tier "
    "outside this repository)",
)


class FixedStepping(SteppingConfig):
    """Config of :class:`FixedCN` (static settings; hashable, part of the compiled program).

    ``n_sub`` sub-steps per day x ``n_iter`` Newton iterations per sub-step; ``jacobian``
    ``"newton"`` | ``"picard"``; ``time_scheme`` ``"implicit"`` | ``"rzwqm"``; ``grad``
    ``"unrolled"`` | ``"implicit"`` (implicit-function-theorem VJP per sub-step); ``h_upper`` [cm]
    upper clamp of the iterate *above hydrostatic*: node ``i`` is clamped at ``h_upper + z_i``;
    ``dv_max`` largest Newton update of the transformed variable; ``c_floor`` [cm-1] storage floor
    added to the Jacobian diagonal only (never to the residual); ``chop`` stops an update that leaves
    the saturated side across the air-entry kink ``h = -hb`` on the kink (see :func:`_iterate`);
    ``rain_fraction`` share of the sub-steps placed on the supply hours.

    Every field is a numerical setting with its origin
    (:data:`~agrijax.processes.soil_water.coefficients.SETTINGS`, ``richards.*``): RZWQM2 uses an
    adaptive step of 1e-4 to 0.1 h with modified Picard iterations to a tolerance, so the fixed
    counts, the damping and the clamps are choices of this implementation, measured in
    ``tests/unit/test_richards.py``. Presets: :meth:`baseline` (24 x 3), :meth:`m1` (96 x 8, the
    near-converged configuration).
    """

    KEY: ClassVar[str] = KEY

    n_sub: int = setting_field(
        "richards.n_sub",
        24,
        "-",
        "sub-steps per day (fixed; 24 = the 24 x 3 baseline, 96 x 8 the near-converged reference)",
        origin="agrijax",
        basis="tests/unit/test_richards.py convergence study (24 sub-steps a day are steps of 1 h, ten "
        "times the largest RZWQM2 step of 0.1 h)",
    )
    n_iter: int = setting_field(
        "richards.n_iter",
        3,
        "-",
        "Newton (or Picard) iterations per sub-step (fixed count, no tolerance test)",
        origin="agrijax",
        basis="tests/unit/test_richards.py convergence study (3 iterations in the 24 x 3 baseline, 8 in "
        "the 96 x 8 configuration)",
    )
    jacobian: str = setting_field(
        "richards.jacobian",
        "newton",
        "-",
        "'newton' (exact tridiagonal Jacobian incl. dK/dh) or 'picard' (K frozen, RZWQM2's scheme)",
        origin="agrijax",
        basis="RZWQM2 RICHRD iterates a modified Picard scheme; the exact Newton Jacobian is a choice of "
        "this implementation and reaches the same root",
    )
    time_scheme: str = setting_field(
        "richards.time_scheme",
        "implicit",
        "-",
        "'implicit' (alpha = 1 on every sub-step) or 'rzwqm' (1, then 1/2: richards.alpha_cn)",
        origin="agrijax",
        basis="tests/unit/test_richards.py convergence study (Crank-Nicolson oscillates unconverged)",
    )
    grad: str = setting_field(
        "richards.grad",
        "unrolled",
        "-",
        "'unrolled' (through the fixed iterations) or 'implicit' (implicit-function VJP per sub-step)",
        origin="agrijax",
        basis="tests/unit/test_richards_grad.py",
    )

    h_upper: float = setting_field(
        "richards.h_upper",
        10.0,
        "cm",
        "upper clamp of the head iterate above hydrostatic (node i clamped at h_upper + z_i); a "
        "divergence guard that should never activate (n_clamp)",
        origin="agrijax",
        basis="+10 cm above hydrostatic is a divergence guard of this implementation; RZWQM2 caps the "
        "surface head at HMAX = 0",
    )
    dv_max: float = setting_field(
        "richards.dv_max",
        1.0,
        "-",
        "largest Newton update of the transformed variable v per node (a factor e in |h|)",
        origin="agrijax",
        basis="richards.py module docstring (damping); tests/unit/test_richards.py",
    )
    c_floor: float = setting_field(
        "richards.c_floor",
        1.0e-7,
        "cm-1",
        "storage floor added to the Jacobian diagonal only (never the residual): keeps it "
        "non-singular on a saturated profile",
        origin="agrijax",
        basis="added to the Jacobian only because C(h) must stay the exact derivative of theta(h): the "
        "floor changes the iteration path, not the residual or the root",
    )
    chop: bool = setting_field(
        "richards.chop",
        True,
        "-",
        "stop an update that leaves the saturated side across the air-entry kink h = -hb on the kink",
        origin="agrijax",
        provenance=Provenance("none", paper="Wang & Tchelepi (2013), J. Comput. Phys. 253, 114-137"),
        basis="tests/unit/test_richards.py pond-emptying sub-step (measured with and without)",
    )
    rain_fraction: float = setting_field(
        "richards.rain_fraction",
        0.5,
        "-",
        "share of the sub-steps placed in proportion to the hourly supply (the rest uniform)",
        origin="agrijax",
        basis="richards.py substep_edges; tests/unit/test_richards.py",
    )

    @classmethod
    def baseline(cls, **overrides: Any) -> FixedStepping:
        """The 24 x 3 baseline (the defaults)."""
        return cls(**overrides)

    @classmethod
    def m1(cls, **overrides: Any) -> FixedStepping:
        """The 96 x 8 configuration (near-converged fully implicit fixed steps)."""
        kw: dict[str, Any] = {"n_sub": M1_N_SUB, "n_iter": M1_N_ITER}
        return cls(**(kw | overrides))

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


class _SolveCfg(NamedTuple):
    n_iter: int
    jacobian: str
    dv_max: float
    c_floor: float
    chop: bool


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
       unsaturated side (a "chop", by analogy with the trust-region Newton of Wang & Tchelepi
       2013 for two-phase transport; not their method). On the saturated side
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


def richards_step(
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
    pond_max: Array,
    stepping: FixedStepping | None = None,
    config: RichardsConfig | None = None,
) -> StepResult:
    """Advance one sub-step of length ``dt`` [h].

    ``supply`` and ``evaporation`` are rates [cm h-1] (``>= 0``); ``sink`` is the sink rate
    [h-1] per node before the ``h_min`` cut: ``[n]`` the root water uptake alone, or
    ``[n_channel, n]`` one row per channel in the order
    :data:`~agrijax.processes.soil_water.sinks.SINK_CHANNELS` (each row capped by the water left
    after the rows before it). ``soil`` must already be on the node axis
    (``SoilHydraulicParams.at_nodes()``).

    ``stepping`` the integrator's settings (default :class:`FixedStepping`), ``config`` the problem's
    (default :class:`~agrijax.processes.soil_water.problem.RichardsConfig`).

    Source: Celia et al. (1990) mixed form; Ahuja et al. (2000) ch. 3; RZWQM2 ``RICHRD``/``CHKBC``.
    """
    cfg = FixedStepping() if stepping is None else stepping
    if not isinstance(cfg, FixedStepping):
        raise TypeError(f"richards_step takes a FixedStepping, got {type(cfg).__name__}")
    pcfg = RichardsConfig() if config is None else config
    a, sk = _step_args(
        h, theta, pond, soil, grid, supply, evaporation, sink, dt, alpha, h_min, pcfg, cfg.h_upper
    )
    scfg = _SolveCfg(cfg.n_iter, cfg.jacobian, cfg.dv_max, cfg.c_floor, cfg.chop)
    if cfg.grad == "implicit":
        h_new, nclamp = cast(tuple[Array, Array], _solve_implicit(scfg, h, a))
    else:
        h_new, nclamp = _iterate(scfg, h, a)
    return _step_result(h_new, nclamp, a, sk, soil, grid, evaporation, dt, pond_max)


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
    hours = jnp.arange(_N_HOUR_EDGES, dtype=dtype)
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


def richards_substeps(
    water: SoilWater,
    params: Any,
    t: Array,
    supply: Array,
    evaporation: Array,
    uptake: Array | SinkChannels,
    alphas: Array,
) -> tuple[SoilWater, SubstepTotals]:
    """Advance ``water`` over the sub-steps ``[t[k], t[k+1]]`` [h] of a day; empty ones are the identity.

    ``params`` a :class:`~agrijax.processes.soil_water.richards.RichardsParams` with a
    :class:`FixedStepping`, ``supply``/``evaporation`` hourly rates ``[24]`` [cm h-1] (averaged over
    each sub-step), ``uptake`` the sink channels
    (:class:`~agrijax.processes.soil_water.sinks.SinkChannels`) or the per-layer root water uptake
    [cm d-1] alone (spread uniformly over the 24 h), ``alphas`` the time weight of each sub-step.
    The returned state keeps ``water.flux``; the totals are separate.

    Source: Ahuja et al. (2000) ch. 3; Celia et al. (1990); RZWQM2 ``RICHRD`` (``Rzrich.for``).
    """
    problem = RichardsProblem.of_day(params, supply, evaporation, uptake, water.theta.dtype)
    return _substeps(problem, _fixed(params.stepping), water, t, alphas)


def _fixed(stepping: Any) -> FixedStepping:
    if not isinstance(stepping, FixedStepping):
        raise TypeError(f"the fixed-step kernels take a FixedStepping, got {type(stepping).__name__}")
    return stepping


def _substeps(
    problem: RichardsProblem, cfg: FixedStepping, water: SoilWater, t: Array, alphas: Array
) -> tuple[SoilWater, SubstepTotals]:
    """:func:`richards_substeps` on the problem of the day (one segment,
    :meth:`~agrijax.processes.soil_water.problem.RichardsProblem.for_segment`).

    Source: Ahuja et al. (2000) ch. 3; Celia et al. (1990); RZWQM2 ``RICHRD`` (``Rzrich.for``).
    """
    problem = problem.for_segment()
    grid = problem.grid
    soil = problem.soil
    dts = t[1:] - t[:-1]
    sup, eva = problem.interval_rates(t)
    sink_of = problem.sink_of()
    h_min = problem.h_min
    pond_max = problem.pond_max
    pcfg = problem.config

    def body(
        carry: tuple[Array, Array, Array], xs: tuple[Array, Array, Array, Array, Array]
    ) -> tuple[Any, Any]:
        h, th, pd = carry
        s_k, e_k, a_k, dt, t0 = xs
        live = dt > 0.0
        dt_safe = jnp.where(live, dt, 1.0)
        sink = sink_of(t0, dt_safe, th, h)
        r = richards_step(h, th, pd, soil, grid, s_k, e_k, sink, dt_safe, a_k, h_min, pond_max, cfg, pcfg)
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


def day_alphas(n_sub: int, cfg: FixedStepping, dtype: Any) -> Array:
    """Time weights of a day's sub-steps: 1 on the first, then 1 (``"implicit"``) or 1/2 (``"rzwqm"``)."""
    a_rest = _ALPHA_CN if cfg.time_scheme == "rzwqm" else _ALPHA_FIRST
    return jnp.where(jnp.arange(n_sub) == 0, _ALPHA_FIRST, a_rest).astype(dtype)


def day_edges(ts0: Array, has_event: Array, cfg: Any, dtype: Any) -> tuple[Array, Array, Array]:
    """``(t_pre[n_pre+1], t_post[n_post+1], t_event)``: sub-step edges [h] around the event time.

    ``cfg`` the day's schedule (``n_pre``, ``n_post``, ``post_grading``: a ``DayConfig`` or a
    :class:`~agrijax.processes.soil_water.integrator.DayEvent`).

    Source: RZWQM2 ``PHYSCL`` (the storm starts a new period).
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


# ---------------------------------------------------------------------------
# the integrator
# ---------------------------------------------------------------------------


class FixedCN:
    """The fixed-step integrator (:class:`~agrijax.processes.soil_water.integrator.RichardsIntegrator`).

    A day without an event: ``n_sub`` sub-steps placed by :func:`substep_edges`. An event day: the
    ``n_pre`` sub-steps of the day's schedule up to the event time of :func:`day_edges`, the event,
    the ``n_post`` sub-steps to the end of the day, with the time weights of the whole day
    (:func:`day_alphas`) split between them.
    """

    def __init__(self, config: FixedStepping) -> None:
        self.config = _fixed(config)

    @property
    def capabilities(self) -> Capabilities:
        unrolled = self.config.grad == "unrolled"
        return Capabilities(
            gradient="autodiff" if unrolled else "custom",
            dtypes=("float64", "float32"),
            batching="static shapes: every lane runs the same n_sub x n_iter program, no lane waits",
            forward_mode=unrolled,
            conserved="theta",
            convergence=False,
            gradient_note=""
            if unrolled
            else "implicit-function derivative of each sub-step, exact only where the fixed iterations "
            "reach the root: 3.7e-2 relative error against central differences at 24 x 3 on the "
            "frozen problem, 2.5e-7 at 96 x 8 (integrator conformance kit, case fixed_96x8_ift); use it "
            "with n_iter >= 8",
        )

    def step_day(
        self, problem: RichardsProblem, water: SoilWater, plan: DayPlan
    ) -> tuple[SoilWater, DayDiagnostics]:
        """One day (:class:`~agrijax.processes.soil_water.integrator.RichardsIntegrator`).

        Source: Ahuja et al. (2000) ch. 3; RZWQM2 ``PHYSCL`` / ``RICHRD`` (conventions).
        """
        cfg = self.config
        dtype = water.theta.dtype
        ev = plan.event
        if ev is None:  # static: a day without an event, one run of sub-steps
            t = substep_edges(problem.supply, cfg.n_sub, cfg.rain_fraction)
            new, tot = _substeps(problem, cfg, water, t, day_alphas(cfg.n_sub, cfg, dtype))
            return new, DayDiagnostics(tot)
        t_pre, t_post, t_ev = day_edges(ev.ts0, ev.has_event, ev, dtype)
        alphas = day_alphas(ev.n_pre + ev.n_post, cfg, dtype)
        w_pre, tot_pre = _substeps(problem, cfg, water, t_pre, alphas[: ev.n_pre])
        w_ev, result = ev.apply(w_pre)
        w_post, tot_post = _substeps(problem, cfg, w_ev, t_post, alphas[ev.n_pre :])
        return w_post, DayDiagnostics(combine_totals(tot_pre, tot_post), None, result, t_ev)

    def check(self, h: Array, flux: Any, where: str) -> Array:
        """No convergence test: the identity."""
        return h


INFO = register_integrator(
    IntegratorInfo(
        key=KEY,
        config_type=FixedStepping,
        factory=FixedCN,
        summary="fixed sub-steps per day, a fixed number of damped Newton/Picard iterations each, fully "
        "implicit or Crank-Nicolson (the legacy modes, bit for bit)",
        sources=(
            Source("mixed-form residual, theta(h) storage term", "Celia, Bouloutas & Zarba (1990) WRR 26"),
            Source(
                "damped Newton on the transformed head, chop at the air-entry kink",
                "fixed_cn.py; Wang & Tchelepi (2013) J. Comput. Phys. 253 (analogy)",
            ),
        ),
        deviates=(
            Deviation(
                "a fixed number of sub-steps and iterations without a convergence test; RZWQM2 adapts "
                "the step (ADJDT) and iterates to a tolerance",
                "static shapes and a fixed cost; the error is reported in balance_error and "
                "max_theta_residual",
                "tests/unit/test_richards.py convergence study; the 24 x 3 and 96 x 8 replays of RZWQM2 4.6 "
                "(data tier outside this repository)",
            ),
        ),
    )
)
