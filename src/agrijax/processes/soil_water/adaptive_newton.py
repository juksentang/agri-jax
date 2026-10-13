"""Newton to a tolerance (fixes N1-N3) with an implicit-function VJP, and one adaptive sub-step.

Part of :mod:`~agrijax.processes.soil_water.richards_adaptive` (split out as a pure move); the
solver and the fixes N1-N3 are described in that module's docstring.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from .adaptive_stepping import AdaptiveStepping
from .hydraulics import AnyHydraulicParams
from .problem import (
    BC_CLIP,
    RichardsConfig,
    RichardsGrid,
    StepResult,
    _capacity,
    _dh_dv,
    _residual_bc,
    _step_args,
    _step_result,
    _StepArgs,
    _top_fluxes,
    _tridiag_solve,
    head_of_v,
    tridiagonal_jacobian,
    v_of_head,
)

#: ``live`` of an ordinary solve, and of the last-resort solve (``dt_min``, ``alpha = 1``)
_LIVE: float = 1.0
_LIVE_LAST: float = 2.0

# ---------------------------------------------------------------------------
# Newton to a tolerance (fixes N1-N3) with an implicit-function VJP
# ---------------------------------------------------------------------------


class _TolCfg(NamedTuple):
    """Static settings of one Newton solve to a tolerance (dtype-resolved)."""

    kmax: int
    kmax_last: int
    tol_theta: float
    tol_balance: float
    polish: bool
    div_after: int
    div_patience: int
    dv_max: float
    c_floor: float
    chop: bool
    lambdas: tuple[float, ...]
    dry_guard: float


def _tol_cfg(cfg: AdaptiveStepping, f32: bool) -> _TolCfg:
    return _TolCfg(
        kmax=cfg.newton_max_iter,
        kmax_last=cfg.newton_max_iter_last,
        tol_theta=cfg.newton_tol_theta_f32 if f32 else cfg.newton_tol_theta,
        tol_balance=cfg.newton_tol_balance_f32 if f32 else cfg.newton_tol_balance,
        polish=cfg.newton_polish,
        div_after=cfg.newton_div_after,
        div_patience=cfg.newton_div_patience,
        dv_max=cfg.dv_max,
        c_floor=cfg.c_floor,
        chop=cfg.chop,
        lambdas=cfg.line_search,
        dry_guard=cfg.dry_guard,
    )


class NewtonInfo(NamedTuple):
    """Outcome of one Newton solve to a tolerance (floats, so that they pass through ``custom_vjp``)."""

    conv: Array  # 1 when the convergence test passed (before the polish update)
    k: Array  # Newton evaluations (the polish update included)
    n_clamp: Array  # clamp activations of the iterate over the whole solve (diagnostic)
    clamp_last: Array  # clamp activations of the last applied update (a converged step needs 0)
    n_active: Array  # nodes held at the dry guard at the solution (fix N3; should be 0)
    residual: Array  # max_i |R_i| dt / tl_i at the last evaluation (active rows excluded)
    balance: Array  # |dt sum_i R_i| at the last evaluation (active rows excluded)


class _NewtonCarry(NamedTuple):
    v: Array
    n_clamp: Array
    clamp_last: Array
    k: Array
    done: Array
    conv: Array
    r_prev: Array
    rise: Array
    active: Array
    residual: Array
    balance: Array


def _dry_bound(cfg: _TolCfg, a: _StepArgs, shape: tuple[int, ...]) -> Array:
    """Lower bound of the head iterate: the guard ``dry_guard h_min``, or the step's start head
    where that is drier (a clamp never wets a node, which would create water).

    The bound is a divergence guard, not the dry end of the physics: a node at ``h_min`` that still
    loses water drains below it (conservatively), while the uptake cap and the surface dry limit
    (ghost head ``h_min``) keep the sink and evaporation from taking a node past ``h_min``.
    """
    guard = cfg.dry_guard * jnp.broadcast_to(a.h_min, shape)
    return jnp.minimum(guard, a.h_old)


def newton_to_tolerance(
    cfg: _TolCfg, h0: Array, a: _StepArgs, live: Array, bc: Array
) -> tuple[Array, NewtonInfo, Array]:
    """Damped Newton on the transformed head to a tolerance; returns ``(h, info, active)``.

    The damping of the fixed modes (``|dv| <= dv_max``, the air-entry chop, the clamp to
    ``[bound, h_hi]``) is kept, with the dry bound at the guard ``dry_guard h_min`` (N3); the
    storage floor is ``max(C, c_floor)`` (N1),
    the update is backtracked on the scaled residual (N2) and a node at the dry bound pushed
    further out is held there (N3). ``live = 0`` runs no iteration (an empty step of the replay);
    ``live = 2`` marks the last-resort solve (``dt_min``, ``alpha = 1``), allowed ``kmax_last``
    evaluations. ``bc`` is the surface condition (:func:`_top_fluxes`).

    Source: the fixes N1-N3 of the
    :mod:`~agrijax.processes.soil_water.richards_adaptive` docstring; Dennis & Schnabel (1996)
    ch. 6 (backtracking); Celia et al. (1990) (mixed-form residual).
    """
    s = a.soil.hb
    lo = v_of_head(_dry_bound(cfg, a, h0.shape), s)
    hi = v_of_head(jnp.broadcast_to(a.h_hi, h0.shape), s)
    scale = a.dt / a.tl
    lambdas = jnp.asarray(cfg.lambdas, h0.dtype)
    last = len(cfg.lambdas) - 1

    def residual_v(x: Array) -> Array:
        return _residual_bc(head_of_v(x, s), a, bc)

    def trial(v: Array, dv: Array, lam: Array) -> tuple[Array, Array]:
        v_raw = v - lam * dv
        if cfg.chop:
            v_raw = jnp.where((v > 0.0) & (v_raw < 0.0), 0.0, v_raw)
        return v_raw, jnp.clip(v_raw, lo, hi)

    def cond(c: _NewtonCarry) -> Array:
        kmax = jnp.where(live > _LIVE, cfg.kmax_last, cfg.kmax)  # live = 2: the last-resort solve
        return (live > 0.0) & ~c.done & (c.k < kmax)

    def body(c: _NewtonCarry) -> _NewtonCarry:
        v = c.v
        r, dl, d, du = tridiagonal_jacobian(residual_v, v)
        finite = jnp.all(jnp.isfinite(r))
        cap = _capacity(head_of_v(v, s), a.soil)
        c_add = jnp.where(cap < cfg.c_floor, cfg.c_floor - cap, 0.0)  # N1: floor at max(C, c_floor)
        d = d + a.tl * c_add / a.dt * _dh_dv(v, s)
        dv0 = _tridiag_solve(dl, d, du, r)
        active = (v <= lo) & (dv0 > 0.0)  # N3: at the dry bound and pushed further out
        keep = ~active
        dv = _tridiag_solve(
            jnp.where(keep, dl, 0.0),
            jnp.where(keep, d, 1.0),
            jnp.where(keep, du, 0.0),
            jnp.where(keep, r, 0.0),
        )
        r_in = jnp.where(active, 0.0, r)
        rt = jnp.max(jnp.abs(r_in) * scale)
        bal = jnp.abs(a.dt * jnp.sum(r_in))
        conv = (rt <= cfg.tol_theta) & (bal <= cfg.tol_balance) & finite
        dv = jnp.clip(dv, -cfg.dv_max, cfg.dv_max)
        if len(cfg.lambdas) > 1:  # N2: backtracking on the scaled residual
            raws, cands = jax.vmap(lambda lam: trial(v, dv, lam))(lambdas)
            merit = jax.vmap(lambda x: jnp.sum((residual_v(x) * scale) ** 2))(cands)
            better = (merit < jnp.sum((r * scale) ** 2)) & jnp.isfinite(merit)
            j = jnp.where(jnp.any(better), jnp.argmax(better), last)
            v_raw, v_c = raws[j], cands[j]
        else:
            v_raw, v_c = trial(v, dv, lambdas[0])
        applied = finite if cfg.polish else finite & ~conv
        clamped = jnp.where(applied, jnp.sum((v_raw < lo) | (v_raw > hi)), 0)
        rise = jnp.where(rt > c.r_prev, c.rise + 1, 0)
        diverged = ~finite | (
            (c.k >= cfg.div_after) & (rise >= cfg.div_patience) & (live <= _LIVE)
        )  # the last-resort solve (live = 2) runs to kmax_last
        return _NewtonCarry(
            v=jnp.where(applied, v_c, v),
            n_clamp=c.n_clamp + clamped.astype(h0.dtype),
            clamp_last=clamped.astype(h0.dtype),
            k=c.k + 1,
            done=conv | diverged,
            conv=conv,
            r_prev=rt,
            rise=rise,
            active=active,
            residual=rt,
            balance=bal,
        )

    z = jnp.zeros((), h0.dtype)
    i0 = jnp.zeros((), jnp.int32)
    init = _NewtonCarry(
        v=v_of_head(h0, s),
        n_clamp=z,
        clamp_last=z,
        k=i0,
        done=jnp.zeros((), bool),
        conv=jnp.zeros((), bool),
        r_prev=jnp.full((), jnp.inf, h0.dtype),
        rise=i0,
        active=jnp.zeros(h0.shape, bool),
        residual=z,
        balance=z,
    )
    c = lax.while_loop(cond, body, init)
    info = NewtonInfo(
        conv=c.conv.astype(h0.dtype),
        k=c.k.astype(h0.dtype),
        n_clamp=c.n_clamp,
        clamp_last=c.clamp_last,
        n_active=jnp.sum(c.active).astype(h0.dtype),
        residual=c.residual,
        balance=c.balance,
    )
    return head_of_v(c.v, s), info, c.active


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _solve_tol(cfg: _TolCfg, h0: Array, a: _StepArgs, live: Array, bc: Array) -> tuple[Array, NewtonInfo]:
    h, info, _ = newton_to_tolerance(cfg, h0, a, live, bc)
    return h, info


def _solve_tol_fwd(
    cfg: _TolCfg, h0: Array, a: _StepArgs, live: Array, bc: Array
) -> tuple[tuple[Array, NewtonInfo], tuple[Array, _StepArgs, Array, Array, Array]]:
    h, info, active = newton_to_tolerance(cfg, h0, a, live, bc)
    return (h, info), (h, a, active, live, bc)


def _solve_tol_bwd(
    cfg: _TolCfg, res: tuple[Array, _StepArgs, Array, Array, Array], g: tuple[Array, NewtonInfo]
) -> tuple[Array, _StepArgs, Array, Array]:
    """Implicit-function VJP at the root: rows ``R_i(h, a) = 0``, active rows ``h_i - bound(a) = 0``.

    ``a_bar = -(dF/da)^T J^{-T} g`` with ``J = dF/dh`` the exact tridiagonal Jacobian (``dK/dh``
    included, no storage floor). A fully saturated profile under a flux top has ``C = 0`` and
    ``dK/dh = 0`` at every node and a singular ``J``; there (only) the storage floor
    ``tl c_floor / dt`` is added to the diagonal, so the gradient stays finite. An empty step (``live = 0``)
    passes no cotangent.
    """
    h, a, active, live, bc = res
    g_h = g[0]

    def constrained(x: Array, aa: _StepArgs) -> Array:
        return jnp.where(active, x - _dry_bound(cfg, aa, x.shape), _residual_bc(x, aa, bc))

    _, dl, d, du = tridiagonal_jacobian(lambda x: constrained(x, a), h)
    saturated = jnp.all(_capacity(h, a.soil) <= 0.0)
    reg = a.tl * cfg.c_floor / a.dt
    d = d + jnp.where(saturated & ~active, reg, 0.0)
    zero = jnp.zeros_like(d[:1])
    dl_t = jnp.concatenate([zero, du[:-1]])
    du_t = jnp.concatenate([dl[1:], zero])
    lam = _tridiag_solve(dl_t, d, du_t, g_h)
    lam = jnp.where(live > 0.0, lam, 0.0)
    _, vjp = jax.vjp(lambda aa: constrained(h, aa), a)
    (a_bar,) = vjp(-lam)
    return jnp.zeros_like(h), a_bar, jnp.zeros_like(live), jnp.zeros_like(bc)


_solve_tol.defvjp(_solve_tol_fwd, _solve_tol_bwd)


def adaptive_step(
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
    cfg: AdaptiveStepping,
    live: Array,
    h_start: Array,
    bc: Array | float = BC_CLIP,
    config: RichardsConfig | None = None,
) -> tuple[StepResult, NewtonInfo, Array]:
    """One sub-step of the adaptive mode: :func:`~agrijax.processes.soil_water.richards.richards_step`
    with the Newton solve to a tolerance (from ``h_start``) under the surface condition ``bc`` and
    its implicit-function VJP; also returns whether the clipped surface flux equals the ``bc`` one
    at the solution (always under ``BC_CLIP``), i.e. whether it solves the clipped problem.
    ``config`` the problem's settings (default
    :class:`~agrijax.processes.soil_water.problem.RichardsConfig`).

    Source: Celia et al. (1990); RZWQM2 ``RICHRD`` / ``CHKBC``.
    """
    pcfg = RichardsConfig() if config is None else config
    a, sk = _step_args(
        h, theta, pond, soil, grid, supply, evaporation, sink, dt, alpha, h_min, pcfg, cfg.h_upper
    )
    bc_ = jnp.asarray(bc, theta.dtype)
    solved = _solve_tol(_tol_cfg(cfg, theta.dtype == jnp.float32), h_start, a, live, bc_)
    h_new, info = cast(tuple[Array, NewtonInfo], solved)
    q_clip, q_bc = _top_fluxes(lax.stop_gradient(h_new), lax.stop_gradient(a), bc_)
    consistent = (bc_ == BC_CLIP) | (q_clip == q_bc)
    res = _step_result(h_new, info.n_clamp, a, sk, soil, grid, evaporation, dt, pond_max)
    return res, info, consistent
