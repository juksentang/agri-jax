"""Batched optimisers for calibration: Adam, L-BFGS and Levenberg-Marquardt (gradient), CMA-ES and
random search (gradient-free).

Every optimiser runs ``R`` independent problems at once (replicates, starting points or sites):
the objective is evaluated on a ``[R, n]`` (or ``[R * lambda, n]``) batch of unconstrained vectors
in one call, so the model runs as one ``vmap`` batch. They count **model calls**: one call is one
parameter vector simulated over all treatments of its problem; a value-and-gradient call is
counted separately from a forward call (its cost relative to a forward call is measured by the
caller, e.g. :func:`time_calls`).

* :func:`adam` - Adam (Kingma & Ba 2015, Algorithm 1) on ``value_and_grad(z)``; the optax
  package is not in the cluster environment, so this is the textbook update, with the
  best-so-far iterate kept.
* :func:`secant_gradient` - wraps a loss so that the gradient components of chosen parameters
  (those the gradient-trust report flags as zero or step-like) are central secants of a chosen
  width instead of the AD derivative, evaluated in the same batch;
* :func:`pair_gradient` - the same per (treatment, parameter) pair: AD for the pairs a
  :class:`~agrijax.calib.trust.GradientPlan` trusts, central secants of that treatment's loss for
  the others (a jumpy pair falls back to derivative-free without giving up AD elsewhere).
* :func:`cma_es` - (mu/mu_w, lambda)-CMA-ES with the default strategy parameters of Hansen,
  "The CMA Evolution Strategy: A Tutorial" (arXiv:1604.00772, 2016), Table 1 and Fig. 7.
* :func:`random_search` - uniform sampling in the unconstrained box, best kept.
* :func:`lbfgs` - L-BFGS (Nocedal & Wright 2006, Algorithms 7.4 / 7.5) with an Armijo
  backtracking line search (written here: optax is not in the cluster environment); and :func:`lm` -
  Levenberg-Marquardt with Marquardt scaling on a residual vector. Both run the rows with their
  own step, line search / damping, stop decision and call counts: a row that stops is frozen
  (state unchanged; still evaluated in the fixed-shape batch, but its results are discarded and
  no longer counted). They stop on a per-row loss target or forward-equivalent budget.

Row invariance. The algebra of :func:`lbfgs` and :func:`lm` is row-wise: given the same values
from the objective, a row follows the same iterates, counts and stop decisions bit for bit
whatever the other rows hold and whatever the batch size. Whether the *objective* gives a row the
same bits is the caller's property: a compiled ``jit(vmap(...))`` objective (e.g.
:func:`batched_value_and_grad`) may round a row differently at another batch size (XLA picks
fused multiply-adds and reduction orders per program shape), so a row run alone at ``R = 1`` can
end at a different point than the same row inside a batch of ``R`` rows. At a *fixed* batch shape
the program is the same, so the bitwise single-row reference of a row ``i`` is the run of the
same ``[R, n]`` batch with only row ``i`` active (``active = (arange(R) == i)``: the other lanes
are frozen at their start). Across batch shapes or devices, single-row and batch runs agree only
within rounding; the caller checks that on its own objective and device.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.coefficients import numerical_guard

__all__ = [
    "LM_MINPACK_TOL",
    "AdamConfig",
    "CmaConfig",
    "LbfgsConfig",
    "LmConfig",
    "OptResult",
    "adam",
    "batched_loss",
    "batched_residual_jacobian",
    "batched_value_and_grad",
    "cma_es",
    "lbfgs",
    "lm",
    "pair_gradient",
    "pair_gradient_forwards",
    "random_search",
    "secant_gradient",
    "time_calls",
]

#: floor of the CMA-ES covariance eigenvalue roots (keeps C^-1/2 finite)
_EIG_FLOOR = numerical_guard("calib.cma_eig_floor", 1e-300, "floor of the CMA-ES covariance eigenvalue roots")
#: Adam's denominator guard (Kingma & Ba 2015, Algorithm 1: epsilon = 1e-8)
_ADAM_EPS = numerical_guard("calib.adam_eps", 1e-8, "Adam denominator guard (Kingma & Ba 2015, Alg. 1)")


@dataclass(frozen=True)
class AdamConfig:
    """Adam settings. Defaults ``b1 = 0.9``, ``b2 = 0.999`` are Kingma & Ba (2015), Algorithm 1;
    ``lr`` is in unconstrained units (logit of the bound fraction), a harness choice."""

    lr: float = 0.05
    b1: float = 0.9
    b2: float = 0.999
    steps: int = 200


DEFAULT_ADAM = AdamConfig()


@dataclass
class OptResult:
    """Result of a batched optimisation over ``R`` problems."""

    z_best: np.ndarray  #: [R, n] best unconstrained vector found
    loss_best: np.ndarray  #: [R]
    loss_history: np.ndarray  #: [steps, R] loss of the current iterate (Adam) or generation best
    z_history: np.ndarray  #: [steps, R, n]
    n_forward: int  #: forward model calls per problem
    n_grad: int  #: value-and-gradient model calls per problem
    wall_s: float  #: wall time of the whole loop (warm the jitted function before: no compilation)
    #: [steps, R, 2] cumulative (forward, derivative) calls of each row when each history entry was
    #: recorded; the last entry equals the final counts (``n_forward`` / ``n_grad`` for every row,
    #: or the per-row counts of :func:`lbfgs` / :func:`lm`)
    cost_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 2), dtype=np.int64))
    #: [steps] seconds since the start of the loop when each history entry was recorded (one clock
    #: for the whole batch: the rows advance together)
    time_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    extra: dict[str, Any] = field(default_factory=dict)

    def forward_equivalents(self, grad_cost: float) -> np.ndarray:
        """``[steps, R]`` cumulative cost ``forward + grad_cost * derivative`` in forward calls;
        ``grad_cost`` is the measured cost of one derivative call (value-and-gradient, or the
        Jacobian of :func:`lm`) relative to one forward call."""
        return self.cost_history[..., 0] + grad_cost * self.cost_history[..., 1]


def _stack_costs(entries: list[np.ndarray], r: int) -> np.ndarray:
    """The ``[steps, R, 2]`` cost history from its per-step entries."""
    return np.stack(entries) if entries else np.zeros((0, r, 2), dtype=np.int64)


def _cost_entry(r: int, n_fwd: Any, n_grad: Any) -> np.ndarray:
    """``[R, 2]`` cumulative (forward, derivative) calls of the ``R`` rows (scalars broadcast)."""
    out = np.zeros((r, 2), dtype=np.int64)
    out[:, 0] = n_fwd
    out[:, 1] = n_grad
    return out


def batched_loss(loss: Callable[[Array], Array]) -> Callable[[Array], Array]:
    """``jit(vmap(loss))``: ``[R, n] -> [R]``."""
    return jax.jit(jax.vmap(loss))


def batched_value_and_grad(
    loss: Callable[[Array], Array],
) -> Callable[[Array], tuple[Array, Array]]:
    """``jit(vmap(value_and_grad(loss)))``: ``[R, n] -> ([R], [R, n])``."""
    return jax.jit(jax.vmap(jax.value_and_grad(loss)))


def secant_gradient(
    loss: Callable[[Array], Array], index: Sequence[int], delta: Sequence[float]
) -> Callable[[Array], tuple[Array, Array]]:
    """``z -> (loss(z), g)`` where ``g`` is the AD gradient except at ``index[k]``, which is the
    central secant ``(loss(z + d e_i) - loss(z - d e_i)) / (2 d)`` with ``d = delta[k]`` (in
    ``z`` units). The ``2 * len(index)`` extra points run in the same ``vmap`` batch. Costs one
    value-and-gradient call and ``2 * len(index)`` forward calls per evaluation."""
    idx = np.asarray(index, dtype=int)
    d = np.asarray(delta, dtype=float)
    if idx.size == 0:
        return jax.value_and_grad(loss)
    n_sec = idx.size

    def vg(z: Array) -> tuple[Array, Array]:
        v, g = jax.value_and_grad(loss)(z)
        n = z.shape[-1]
        steps = jnp.zeros((n_sec, n), dtype=z.dtype).at[jnp.arange(n_sec), idx].set(jnp.asarray(d, z.dtype))
        pts = jnp.concatenate([z[None] + steps, z[None] - steps], axis=0)
        vals = jax.vmap(loss)(pts)
        sec = (vals[:n_sec] - vals[n_sec:]) / (2.0 * jnp.asarray(d, z.dtype))
        return v, g.at[idx].set(sec)

    return vg


def pair_gradient(
    losses: Callable[[Array], Array], secant: Any, delta: float | Sequence[float]
) -> Callable[[Array], tuple[Array, Array]]:
    """``z -> (sum_b L_b(z), g)`` for per-treatment losses ``losses: z [n] -> [B]``.

    ``g_i = sum_b dL_b/dz_i`` (AD) over the pairs with ``secant[b, i]`` false, plus
    ``sum_b (L_b(z + d_i e_i) - L_b(z - d_i e_i)) / (2 d_i)`` over the pairs with ``secant[b, i]``
    true (``delta``: ``d`` in ``z`` units, one value or one per parameter). The AD part is one
    forward pass and ``B`` reverse passes (``vmap``-ed); the secants are ``2 k`` forward points in
    one ``vmap`` batch, ``k`` the number of parameters with a secant pair
    (:func:`pair_gradient_forwards` gives ``2 k``, the ``forward_per_derivative`` of
    :func:`lbfgs`). With no secant pair it is ``value_and_grad`` of the sum."""
    m = np.asarray(secant, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"pair_gradient: secant mask must be [B, n], got shape {m.shape}")
    cols = np.nonzero(m.any(axis=0))[0]
    d_all = np.broadcast_to(np.asarray(delta, dtype=float), (m.shape[1],))

    def total(z: Array) -> Array:
        return jnp.sum(losses(z))

    if cols.size == 0:
        return jax.value_and_grad(total)
    keep = jnp.asarray(~m)
    msel = jnp.asarray(m[:, cols])  # [B, k]
    d = d_all[cols]

    def vg(z: Array) -> tuple[Array, Array]:
        lb, pullback = jax.vjp(losses, z)
        b = lb.shape[0]
        (jac,) = jax.vmap(pullback)(jnp.eye(b, dtype=lb.dtype))  # [B, n]
        g = jnp.sum(jnp.where(keep, jac, 0.0), axis=0)
        n = z.shape[-1]
        steps = (
            jnp.zeros((cols.size, n), dtype=z.dtype)
            .at[jnp.arange(cols.size), cols]
            .set(jnp.asarray(d, z.dtype))
        )
        vals = jax.vmap(losses)(jnp.concatenate([z[None] + steps, z[None] - steps], axis=0))  # [2k, B]
        sec = (vals[: cols.size] - vals[cols.size :]) / (2.0 * jnp.asarray(d, z.dtype))[:, None]  # [k, B]
        g = g.at[cols].add(jnp.sum(jnp.where(msel.T, sec, 0.0), axis=1))
        return jnp.sum(lb), g

    return vg


def pair_gradient_forwards(secant: Any) -> int:
    """Forward calls :func:`pair_gradient` makes per evaluation with the ``[B, n]`` mask
    ``secant``: ``2 k``, ``k`` the number of parameters with at least one secant pair."""
    m = np.asarray(secant, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"pair_gradient_forwards: secant mask must be [B, n], got shape {m.shape}")
    return 2 * int(np.count_nonzero(m.any(axis=0)))


def time_calls(fn: Callable[[Any], Any], arg: Any, repeat: int = 3) -> tuple[float, float]:
    """``(first_call_s, steady_call_s)``: compilation-inclusive time of the first call and the
    median of ``repeat`` further calls (blocking on the result)."""
    t0 = time.perf_counter()
    jax.block_until_ready(fn(arg))
    first = time.perf_counter() - t0
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(arg))
        ts.append(time.perf_counter() - t0)
    return first, float(np.median(ts))


# ------------------------------------------------------------------------------------ Adam
def adam(
    value_and_grad: Callable[[Array], tuple[Array, Array]],
    z0: Any,
    cfg: AdamConfig = DEFAULT_ADAM,
    *,
    forward_per_step: int = 0,
    callback: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
) -> OptResult:
    """Batched Adam on ``value_and_grad: [R, n] -> ([R], [R, n])`` from ``z0 [R, n]``.

    ``forward_per_step`` is the number of extra forward calls per step inside
    ``value_and_grad`` (e.g. :func:`secant_gradient`), for the call count. Non-finite losses or
    gradients are reported in ``extra["n_nonfinite"]`` and the step is skipped for that row.
    """
    z = np.asarray(z0, dtype=float).copy()
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    best_z = z.copy()
    best_l = np.full(z.shape[0], np.inf)
    lh, zh = [], []
    ch: list[np.ndarray] = []
    th: list[float] = []
    n_bad = 0
    t0 = time.perf_counter()
    for k in range(cfg.steps):
        loss, g = value_and_grad(jnp.asarray(z))
        loss = np.asarray(loss, dtype=float)
        g = np.asarray(g, dtype=float)
        ok = np.isfinite(loss) & np.all(np.isfinite(g), axis=-1)
        n_bad += int(np.sum(~ok))
        better = ok & (loss < best_l)
        best_l = np.where(better, loss, best_l)
        best_z[better] = z[better]
        lh.append(loss)
        zh.append(z.copy())
        ch.append(_cost_entry(z.shape[0], (k + 1) * forward_per_step, k + 1))
        th.append(time.perf_counter() - t0)
        if callback is not None:
            callback(k, loss, z)
        g = np.where(ok[:, None], g, 0.0)
        m = cfg.b1 * m + (1.0 - cfg.b1) * g
        v = cfg.b2 * v + (1.0 - cfg.b2) * g**2
        mh = m / (1.0 - cfg.b1 ** (k + 1))
        vh = v / (1.0 - cfg.b2 ** (k + 1))
        z = z - np.where(ok[:, None], cfg.lr * mh / (np.sqrt(vh) + _ADAM_EPS), 0.0)
    wall = time.perf_counter() - t0
    return OptResult(
        z_best=best_z,
        loss_best=best_l,
        loss_history=np.asarray(lh),
        z_history=np.asarray(zh),
        n_forward=cfg.steps * forward_per_step,
        n_grad=cfg.steps,
        wall_s=wall,
        cost_history=_stack_costs(ch, z.shape[0]),
        time_history=np.asarray(th, dtype=float),
        extra={"n_nonfinite": n_bad},
    )


# ------------------------------------------------------------------------------------ CMA-ES
@dataclass(frozen=True)
class CmaConfig:
    """CMA-ES settings: ``sigma0`` (initial step, ``z`` units, harness choice), ``popsize``
    (``None``: ``4 + floor(3 ln n)``, Hansen 2016 Table 1) and the evaluation budget
    ``max_evals`` per problem (rounded down to whole generations)."""

    sigma0: float = 0.5
    popsize: int | None = None
    max_evals: int = 200


DEFAULT_CMA = CmaConfig()


def _cma_defaults(n: int, lam: int) -> dict[str, Any]:
    """Default strategy parameters (Hansen 2016, Table 1)."""
    mu = lam // 2
    w = np.log((lam + 1) / 2) - np.log(np.arange(1, mu + 1))
    w = w / w.sum()
    mueff = 1.0 / np.sum(w**2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chin = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
    return dict(mu=mu, w=w, mueff=mueff, cc=cc, cs=cs, c1=c1, cmu=cmu, damps=damps, chin=chin)


def cma_es(
    loss_batch: Callable[[Array], Array],
    z0: Any,
    cfg: CmaConfig = DEFAULT_CMA,
    *,
    seed: int = 0,
) -> OptResult:
    """Batched CMA-ES: ``R`` independent searches from the means ``z0 [R, n]``; every generation
    evaluates the ``R * lambda`` candidates in one call of ``loss_batch: [M, n] -> [M]``."""
    z0 = np.asarray(z0, dtype=float)
    r, n = z0.shape
    lam = int(cfg.popsize or 4 + int(np.floor(3 * np.log(n))))
    p = _cma_defaults(n, lam)
    n_gen = max(1, cfg.max_evals // lam)
    rng = np.random.default_rng(seed)
    mean = z0.copy()
    sigma = np.full(r, cfg.sigma0)
    c = np.tile(np.eye(n), (r, 1, 1))
    pc = np.zeros((r, n))
    ps = np.zeros((r, n))
    best_z = z0.copy()
    best_l = np.full(r, np.inf)
    lh, zh = [], []
    ch: list[np.ndarray] = []
    th: list[float] = []
    t0 = time.perf_counter()
    for gen in range(n_gen):
        evals, bvecs = np.linalg.eigh(c)  # [r, n], [r, n, n]
        dsq = np.sqrt(np.maximum(evals, 0.0))
        y = np.einsum("rij,rkj->rki", bvecs * dsq[:, None, :], rng.standard_normal((r, lam, n)))
        x = mean[:, None, :] + sigma[:, None, None] * y  # [r, lam, n]
        f = np.asarray(loss_batch(jnp.asarray(x.reshape(r * lam, n))), dtype=float).reshape(r, lam)
        f = np.where(np.isfinite(f), f, np.inf)
        order = np.argsort(f, axis=1)
        gbest = f[np.arange(r), order[:, 0]]
        better = gbest < best_l
        best_l = np.where(better, gbest, best_l)
        best_z[better] = x[np.arange(r), order[:, 0]][better]
        lh.append(gbest)
        zh.append(mean.copy())
        ch.append(_cost_entry(r, (gen + 1) * lam, 0))
        th.append(time.perf_counter() - t0)
        ysel = np.take_along_axis(y, order[:, : p["mu"], None], axis=1)  # [r, mu, n]
        yw = np.einsum("m,rmn->rn", p["w"], ysel)
        mean = mean + sigma[:, None] * yw
        # C^{-1/2} yw
        inv_sqrt = np.einsum("rij,rj,rkj->rik", bvecs, 1.0 / np.maximum(dsq, _EIG_FLOOR), bvecs)
        ps = (1 - p["cs"]) * ps + np.sqrt(p["cs"] * (2 - p["cs"]) * p["mueff"]) * np.einsum(
            "rij,rj->ri", inv_sqrt, yw
        )
        ps_norm = np.linalg.norm(ps, axis=1)
        hsig = (
            ps_norm / np.sqrt(1 - (1 - p["cs"]) ** (2 * (gen + 1))) / p["chin"] < 1.4 + 2 / (n + 1)
        ).astype(float)
        pc = (1 - p["cc"]) * pc + hsig[:, None] * np.sqrt(p["cc"] * (2 - p["cc"]) * p["mueff"]) * yw
        rank_mu = np.einsum("m,rmi,rmj->rij", p["w"], ysel, ysel)
        dh = (1 - hsig) * p["cc"] * (2 - p["cc"])
        c = (
            (1 - p["c1"] - p["cmu"]) * c
            + p["c1"] * (np.einsum("ri,rj->rij", pc, pc) + dh[:, None, None] * c)
            + p["cmu"] * rank_mu
        )
        c = 0.5 * (c + np.transpose(c, (0, 2, 1)))
        sigma = sigma * np.exp((p["cs"] / p["damps"]) * (ps_norm / p["chin"] - 1))
    wall = time.perf_counter() - t0
    return OptResult(
        z_best=best_z,
        loss_best=best_l,
        loss_history=np.asarray(lh),
        z_history=np.asarray(zh),
        n_forward=n_gen * lam,
        n_grad=0,
        wall_s=wall,
        cost_history=_stack_costs(ch, r),
        time_history=np.asarray(th, dtype=float),
        extra={"popsize": lam, "generations": n_gen, "sigma_final": sigma},
    )


def random_search(
    loss_batch: Callable[[Array], Array],
    lower_z: Any,
    upper_z: Any,
    n_problems: int,
    max_evals: int,
    *,
    batch: int = 16,
    seed: int = 0,
) -> OptResult:
    """Uniform random search in the box ``[lower_z, upper_z]`` (unconstrained units), ``R``
    independent problems, ``max_evals`` candidates each, ``batch`` candidates per call."""
    lo = np.asarray(lower_z, dtype=float)
    hi = np.asarray(upper_z, dtype=float)
    n = lo.size
    rng = np.random.default_rng(seed)
    best_z = np.zeros((n_problems, n))
    best_l = np.full(n_problems, np.inf)
    lh, zh = [], []
    ch: list[np.ndarray] = []
    th: list[float] = []
    n_rounds = max(1, max_evals // batch)
    t0 = time.perf_counter()
    for i_round in range(n_rounds):
        x = lo + (hi - lo) * rng.random((n_problems, batch, n))
        f = np.asarray(loss_batch(jnp.asarray(x.reshape(-1, n))), dtype=float).reshape(n_problems, batch)
        f = np.where(np.isfinite(f), f, np.inf)
        j = np.argmin(f, axis=1)
        fb = f[np.arange(n_problems), j]
        better = fb < best_l
        best_l = np.where(better, fb, best_l)
        best_z[better] = x[np.arange(n_problems), j][better]
        lh.append(best_l.copy())
        zh.append(best_z.copy())
        ch.append(_cost_entry(n_problems, (i_round + 1) * batch, 0))
        th.append(time.perf_counter() - t0)
    return OptResult(
        z_best=best_z,
        loss_best=best_l,
        loss_history=np.asarray(lh),
        z_history=np.asarray(zh),
        n_forward=n_rounds * batch,
        n_grad=0,
        wall_s=time.perf_counter() - t0,
        cost_history=_stack_costs(ch, n_problems),
        time_history=np.asarray(th, dtype=float),
    )


# ------------------------------------------------------------------------------- per-row stop
#: per-row status of :func:`lbfgs` and :func:`lm` (``OptResult.extra["status"]``). For a
#: classification of the outcome: ``target`` and ``gradient`` mean the row reached a point where
#: it cannot go lower (a loss target, a gradient at most ``gtol``); L-BFGS ``ftol`` (checked on
#: accepted steps only) means an accepted step decreased the loss by at most ``ftol`` relative, a
#: flat point or a minimum. LM ``ftol`` and ``xtol`` follow MINPACK and are checked on every trial,
#: accepted or rejected: after an accepted trial (``extra["rejects_in_a_row"]`` 0) they mean
#: convergence, but after a run of rejected trials (``rejects_in_a_row`` > 0: the damping grew and
#: shrank the step) they may only mean that the row could not find a lower point, which on a
#: step-like loss is a failure to descend, not convergence; a classification must use
#: ``rejects_in_a_row`` to tell the two apart. ``stall`` means slow progress; ``linesearch``,
#: ``rejects`` and ``lambda`` mean the steps kept failing (usually a non-smooth or inconsistent
#: loss, sometimes a smooth minimum reached at the rounding floor of the loss, where no trial is
#: lower any more); ``budget`` and ``max_iter`` are caps (an L-BFGS row at a minimum ends
#: ``ftol``, ``linesearch`` or ``max_iter`` unless ``gtol`` or ``ftol`` catches it first);
#: ``nonfinite`` a model failure at the current point.
STATUS_RUNNING = "running"  #: still running when the loop ended (only with an external cap)
#: inactive from the start (``active`` false): evaluated in the fixed-shape batch at its start
#: point, but its results are discarded and never counted
STATUS_FROZEN = "frozen"
STATUS_TARGET = "target"  #: the loss reached the row's ``target``
#: ``max |g| <= gtol`` at an accepted point (L-BFGS: the gradient; LM: ``J'r``; with ``gtol = 0``:
#: an exactly zero gradient)
STATUS_GRADIENT = "gradient"
#: L-BFGS: an accepted step decreased the loss by at most ``ftol`` relative; LM: the actual and
#: the predicted relative decrease of a trial step (accepted or not) are both at most ``ftol``
STATUS_FTOL = "ftol"
#: LM: the trial step (accepted or not) is at most ``xtol (|z| + xtol)`` long (Euclidean, ``z`` units)
STATUS_XTOL = "xtol"
STATUS_LINESEARCH = "linesearch"  #: L-BFGS: ``max_backtracks`` trial steps in a row were rejected
#: LM: ``stall_steps`` accepted steps in a row with a relative decrease below ``stall_rel``
STATUS_STALL = "stall"
STATUS_REJECTS = "rejects"  #: LM: ``max_rejects`` trial steps in a row were rejected
STATUS_LAMBDA = "lambda"  #: LM: the damping exceeded ``lambda_max``
STATUS_MAX_ITER = "max_iter"  #: ``max_iter`` accepted steps (L-BFGS) or trial steps (LM)
STATUS_BUDGET = "budget"  #: the next call the row needs would take its cost above its budget
STATUS_NONFINITE = "nonfinite"  #: the loss or the derivative at the current point is not finite


class _Rows:
    """Per-row bookkeeping shared by :func:`lbfgs` and :func:`lm`: the active mask, the status, the
    forward / derivative call counts, the per-row target and budget.

    A row is counted only for the calls it requests; a call runs on the whole fixed-shape batch
    (frozen and stopped rows included), but the results of the rows that did not request it are
    discarded. So the bookkeeping of a row does not depend on the other rows; its iterates do not
    either, provided the objective gives the row the same values whatever the other rows hold
    (see the module docstring). ``forward_per_derivative`` (one value or one per row) is the
    number of forward calls a row's derivative call makes inside the derivative callable
    (secant or finite-difference columns), added to its forward count."""

    def __init__(
        self,
        r: int,
        active: Any,
        target: Any,
        budget: Any,
        grad_cost: float | None,
        name: str,
        forward_per_derivative: Any = 0,
    ) -> None:
        self.active = np.ones(r, bool) if active is None else np.array(active, dtype=bool).reshape(r)
        self.status = np.where(self.active, STATUS_RUNNING, STATUS_FROZEN).astype(object)
        self.n_fwd = np.zeros(r, dtype=np.int64)
        self.n_der = np.zeros(r, dtype=np.int64)
        self.target = np.broadcast_to(
            np.asarray(-np.inf if target is None else target, dtype=float), (r,)
        ).copy()
        if budget is not None and grad_cost is None:
            raise ValueError(f"{name}: a budget needs grad_cost (a derivative call's cost in forward calls)")
        self.budget = None if budget is None else np.broadcast_to(np.asarray(budget, float), (r,)).copy()
        self.grad_cost = 0.0 if grad_cost is None else float(grad_cost)
        fpd = np.asarray(forward_per_derivative)
        if not np.issubdtype(fpd.dtype, np.integer) or np.any(fpd < 0):
            raise ValueError(f"{name}: forward_per_derivative must be non-negative integers, got {fpd!r}")
        self.fpd = np.broadcast_to(fpd.astype(np.int64), (r,)).copy()

    def cost(self) -> np.ndarray:
        """Forward-equivalent cost so far per row: ``n_forward + grad_cost * n_grad`` (the forward
        calls made inside derivative calls are in ``n_forward``)."""
        return self.n_fwd + self.grad_cost * self.n_der

    def count_derivative(self, rows: np.ndarray) -> None:
        """Count one derivative call, and its inner forward calls, for the rows of ``rows``."""
        self.n_der[rows] += 1
        self.n_fwd[rows] += self.fpd[rows]

    def stop(self, rows: np.ndarray, status: str) -> None:
        """Freeze the rows of the boolean mask ``rows`` that are still active with ``status``."""
        rows = rows & self.active
        self.status[rows] = status
        self.active[rows] = False

    def request(self, want: np.ndarray, derivative: bool) -> np.ndarray:
        """The rows of ``want`` that may make the call: rows whose budget the call would exceed are
        stopped (``budget``) and left out."""
        want = want & self.active
        if self.budget is not None:
            add = self.grad_cost + self.fpd if derivative else np.ones(want.shape[0])
            self.stop(want & (self.cost() + add > self.budget), STATUS_BUDGET)
            want = want & self.active
        return want

    def counts(self) -> dict[str, Any]:
        return {
            "n_forward_rows": self.n_fwd.copy(),
            "n_grad_rows": self.n_der.copy(),
            "status": self.status.astype(str),
        }


# ------------------------------------------------------------------------------------ L-BFGS
@dataclass(frozen=True)
class LbfgsConfig:
    """L-BFGS settings (Nocedal & Wright, "Numerical Optimization", 2nd ed. 2006, Algorithms 7.4
    and 7.5, with an Armijo backtracking line search, Algorithm 3.1).

    ``memory`` stored pairs; ``c1`` the Armijo sufficient-decrease constant (Nocedal & Wright
    2006, Section 3.1: "c1 = 1e-4" as a typical value); ``backtrack`` the step shrink factor of a
    rejected trial and ``max_backtracks`` the rejected trials in a row after which the row stops
    (``linesearch``); ``init_step`` the Euclidean length (``z`` units) of the first trial step,
    taken along the steepest descent before any pair is stored, and after a memory reset;
    ``gtol`` stops a row when ``max |g| <= gtol`` (0: only an exactly zero gradient, the
    zero-derivative plateau of a step-like loss); ``ftol`` stops a row when an accepted step
    decreases the loss by at most ``ftol max(|f_k|, |f_k+1|)`` (``ftol``; 0: only an accepted step
    with no decrease at all, which the Armijo test accepts once ``c1 alpha g.d`` is below the
    rounding of the loss); ``max_iter`` accepted steps per row. The ``ftol`` default is the
    relative-reduction tolerance of L-BFGS-B, ``factr * eps`` with ``factr = 1e7`` (Zhu, Byrd, Lu &
    Nocedal 1997, ACM TOMS 23(4), Algorithm 778, driver comment: "factr = 1.d+7 for moderate
    accuracy"; SciPy ``minimize(method="L-BFGS-B")`` ``ftol`` default 2.22e-9); unlike L-BFGS-B
    the denominator has no floor at 1, so the test stays relative on losses far below 1.
    ``memory``, ``backtrack``, ``max_backtracks``, ``init_step``, ``gtol`` and ``max_iter`` are
    harness choices."""

    memory: int = 10
    c1: float = 1e-4
    backtrack: float = 0.5
    max_backtracks: int = 20
    init_step: float = 1.0
    gtol: float = 0.0
    ftol: float = 1e7 * float(np.finfo(np.float64).eps)
    max_iter: int = 100


DEFAULT_LBFGS = LbfgsConfig()

#: the opt-in ``ftol`` / ``xtol`` of :class:`LmConfig`: the square root of the double-precision
#: machine epsilon (More, Garbow & Hillstrom 1980, "User Guide for MINPACK-1", ANL-80-74,
#: Section 2: the recommended ``tol`` of ``lmder1`` is "the square root of the machine precision")
LM_MINPACK_TOL = float(np.sqrt(np.finfo(np.float64).eps))

#: a pair (s, y) is stored only when s.y > eps * y.y (the curvature condition with the guard of
#: Byrd, Lu, Nocedal & Zhu 1995, L-BFGS-B, which skips the update when s'y <= eps y'y; eps the
#: double-precision machine epsilon)
_LBFGS_CURV = numerical_guard(
    "calib.lbfgs_curvature", float(np.finfo(np.float64).eps), "L-BFGS pair skipped unless s.y > eps * y.y"
)
#: divisor guard of the steepest-descent step scaling init_step / |g| (a zero gradient stops the row)
_LBFGS_TINY = numerical_guard("calib.lbfgs_tiny", 1e-300, "divisor guard of the L-BFGS first-step scaling")


def _seq_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``sum_i a[:, i] b[:, i]`` per row of ``[R, n]`` arrays, added in index order with
    elementwise NumPy operations over the rows. Each operation is one correctly rounded IEEE
    operation per element, so a row's value does not depend on the other rows or the batch size
    (a compiled reduction may contract into fused multiply-adds or reorder the sum differently
    for different batch shapes)."""
    acc = a[:, 0] * b[:, 0]
    for i in range(1, a.shape[1]):
        acc = acc + a[:, i] * b[:, i]
    return acc


def _lbfgs_direction(
    upd: np.ndarray,
    has_prev: np.ndarray,
    z: np.ndarray,
    g: np.ndarray,
    z_prev: np.ndarray,
    g_prev: np.ndarray,
    s_mem: np.ndarray,
    y_mem: np.ndarray,
    rho: np.ndarray,
    head: np.ndarray,
    init_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For the rows of ``upd``: store the pair ``(z - z_prev, g - g_prev)`` when ``has_prev`` and
    the curvature condition hold, then the two-loop recursion (Nocedal & Wright 2006, Algorithm
    7.4) with ``H0 = gamma I``, ``gamma = s.y / y.y`` of the newest pair. Without a stored pair, or
    when the recursion's direction is not a descent direction, the row's memory is cleared and its
    direction is the steepest descent of length ``init_step``. Rows with ``upd`` false come back
    unchanged. Elementwise over the rows (see :func:`_seq_dot`). Returns
    ``(s_mem, y_mem, rho, head, d, g.d)``; ``s_mem``, ``y_mem`` ``[R, m, n]``, ``rho`` ``[R, m]``
    (0 for an empty slot), ``head`` ``[R]`` the next slot."""
    r, m, _ = s_mem.shape
    ar = np.arange(r)
    s = z - z_prev
    y = g - g_prev
    sy = _seq_dot(s, y)
    yy = _seq_dot(y, y)
    with np.errstate(invalid="ignore", over="ignore"):
        store = upd & has_prev & np.isfinite(sy) & (sy > _LBFGS_CURV * yy)
    s_mem = s_mem.copy()
    y_mem = y_mem.copy()
    rho = rho.copy()
    s_mem[ar[store], head[store]] = s[store]
    y_mem[ar[store], head[store]] = y[store]
    rho[ar[store], head[store]] = 1.0 / sy[store]
    head = np.where(store, (head + 1) % m, head)
    newest = (head - 1) % m
    have = rho[ar, newest] > 0.0
    s_n = s_mem[ar, newest]
    y_n = y_mem[ar, newest]
    yy_n = _seq_dot(y_n, y_n)
    gamma = np.where(have, _seq_dot(s_n, y_n) / np.where(have, yy_n, 1.0), 1.0)
    q = g.copy()
    alphas = np.zeros((r, m))
    for i in range(m):  # newest to oldest; an empty slot (rho = 0, s = y = 0) changes nothing
        j = (head - 1 - i) % m
        a = rho[ar, j] * _seq_dot(s_mem[ar, j], q)
        q = q - a[:, None] * y_mem[ar, j]
        alphas[ar, j] = a
    rr = gamma[:, None] * q
    for i in range(m):  # oldest to newest
        j = (head + i) % m
        b = rho[ar, j] * _seq_dot(y_mem[ar, j], rr)
        rr = rr + s_mem[ar, j] * (alphas[ar, j] - b)[:, None]
    d_qn = -rr
    gd_qn = _seq_dot(g, d_qn)
    use_qn = have & np.isfinite(gd_qn) & (gd_qn < 0.0)
    gnorm = np.sqrt(_seq_dot(g, g))
    d_sd = -g * (init_step / np.maximum(gnorm, _LBFGS_TINY))[:, None]
    d = np.where(use_qn[:, None], d_qn, d_sd)
    gd = _seq_dot(g, d)
    clear = upd & ~use_qn  # a fallback to steepest descent clears the memory
    s_mem[clear] = 0.0
    y_mem[clear] = 0.0
    rho[clear] = 0.0
    head = np.where(clear, 0, head)
    return s_mem, y_mem, rho, head, d, gd


def lbfgs(
    value_and_grad: Callable[[Array], tuple[ArrayLike, ArrayLike]],
    loss_batch: Callable[[Array], ArrayLike],
    z0: Any,
    cfg: LbfgsConfig = DEFAULT_LBFGS,
    *,
    target: Any = None,
    budget: Any = None,
    grad_cost: float | None = None,
    active: Any = None,
    forward_per_derivative: Any = 0,
) -> OptResult:
    """Batched L-BFGS with an Armijo backtracking line search, ``R`` independent rows from
    ``z0 [R, n]`` in the unconstrained space.

    ``value_and_grad: [R, n] -> ([R], [R, n])`` (e.g. :func:`batched_value_and_grad`) gives the
    gradient at a row's accepted point; ``loss_batch: [R, n] -> [R]`` (:func:`batched_loss`)
    evaluates the trial points of the line search. Each loop round makes at most one derivative
    call (the rows that accepted a step in the previous round, and every row in the first round)
    and one forward call (the rows in their line search). Each row has its own step length,
    backtracking count, memory and stop decision; a call always runs on the full ``[R, n]`` batch
    (one compiled shape), and only the rows that requested it are updated and counted.

    Per row, the run stops (``extra["status"]``) when the loss reaches ``target`` (a value or one
    per row; checked at the first point and at every accepted step), on a zero gradient
    (``gtol``), on an accepted step with a relative decrease at most ``ftol``, after
    ``max_backtracks`` rejected trials in a row, after ``max_iter`` accepted
    steps, when the loss or gradient at its current point is not finite, or when the next call it
    needs would take its cost ``n_forward + grad_cost * n_grad`` above ``budget`` (a value or one
    per row; ``grad_cost`` is the measured cost of a value-and-gradient call in forward calls).
    ``forward_per_derivative`` (one integer or one per row) is the number of forward calls made
    inside one call of ``value_and_grad`` for that row, e.g. :func:`pair_gradient_forwards` for
    :func:`pair_gradient`, ``2 * len(index)`` for :func:`secant_gradient`, or ``n`` for a
    forward-difference gradient (then ``grad_cost = 1``, the base point); they are added to the
    row's ``n_forward`` and to its budget check. Rows with ``active`` false are frozen from the
    start: they are still evaluated in the fixed-shape batch (at their start point), but their
    results are discarded and never counted. A stopped row keeps its state bit for bit and is no
    longer counted.

    A row's iterates in a batch equal those of the row alone bit for bit only when the objective
    callables give the row the same values at both batch shapes; the algebra here is row-wise.
    The same-shape single-row reference is ``active = (arange(R) == i)`` (module docstring).

    ``n_forward`` / ``n_grad`` are the largest per-row counts; the per-row counts are in
    ``extra["n_forward_rows"]`` / ``extra["n_grad_rows"]``. The accepted losses decrease, so
    ``z_best`` is the last accepted point. History: one entry per loop round, the current point
    and loss of every row (a frozen row repeats its last entry)."""
    z = np.array(z0, dtype=float, copy=True)
    r, n = z.shape
    m = int(cfg.memory)
    rows = _Rows(r, active, target, budget, grad_cost, "lbfgs", forward_per_derivative)
    f = np.full(r, np.inf)
    g = np.zeros((r, n))
    z_prev = z.copy()
    g_prev = np.zeros((r, n))
    has_prev = np.zeros(r, bool)
    s_mem = np.zeros((r, m, n))
    y_mem = np.zeros((r, m, n))
    rho = np.zeros((r, m))
    head = np.zeros(r, dtype=np.int64)
    d = np.zeros((r, n))
    gd = np.zeros(r)
    alpha = np.ones(r)
    n_back = np.zeros(r, dtype=np.int64)
    n_acc = np.zeros(r, dtype=np.int64)
    need_grad = rows.active.copy()
    in_ls = np.zeros(r, bool)
    lh, zh = [], []
    ch: list[np.ndarray] = []
    th: list[float] = []
    t0 = time.perf_counter()
    while rows.active.any():
        # ---- derivative call: rows at a new accepted point
        req = rows.request(need_grad, derivative=True)
        if req.any():
            fv, gv = value_and_grad(jnp.asarray(z))
            fv = np.asarray(fv, dtype=float)
            gv = np.asarray(gv, dtype=float)
            rows.count_derivative(req)
            ok = np.isfinite(fv) & np.all(np.isfinite(gv), axis=-1)
            rows.stop(req & ~ok, STATUS_NONFINITE)
            got = req & ok
            f = np.where(got, fv, f)
            g = np.where(got[:, None], gv, g)
            s_mem, y_mem, rho, head, dj, gdj = _lbfgs_direction(
                got, has_prev, z, g, z_prev, g_prev, s_mem, y_mem, rho, head, cfg.init_step
            )
            d = np.where(got[:, None], dj, d)
            gd = np.where(got, gdj, gd)
            z_prev = np.where(got[:, None], z, z_prev)
            g_prev = np.where(got[:, None], g, g_prev)
            has_prev = has_prev | got
            need_grad = need_grad & ~got
            rows.stop(got & (f <= rows.target), STATUS_TARGET)
            rows.stop(got & (np.max(np.abs(g), axis=-1) <= cfg.gtol), STATUS_GRADIENT)
            start = got & rows.active
            in_ls = in_ls | start
            alpha = np.where(start, 1.0, alpha)
            n_back = np.where(start, 0, n_back)
        # ---- forward call: the trial points of the rows in their line search
        req = rows.request(in_ls, derivative=False)
        if req.any():
            trial = z + alpha[:, None] * d
            ft = np.asarray(loss_batch(jnp.asarray(trial)), dtype=float)
            rows.n_fwd[req] += 1
            acc = req & np.isfinite(ft) & (ft <= f + cfg.c1 * alpha * gd)
            with np.errstate(invalid="ignore", over="ignore"):
                flat = acc & (f - ft <= cfg.ftol * np.maximum(np.abs(f), np.abs(ft)))
            z = np.where(acc[:, None], trial, z)
            f = np.where(acc, ft, f)
            n_acc = n_acc + acc
            need_grad = need_grad | acc
            in_ls = in_ls & ~acc
            rej = req & ~acc
            alpha = np.where(rej, alpha * cfg.backtrack, alpha)
            n_back = n_back + rej
            rows.stop(rej & (n_back >= cfg.max_backtracks), STATUS_LINESEARCH)
            rows.stop(acc & (f <= rows.target), STATUS_TARGET)
            rows.stop(flat, STATUS_FTOL)
            rows.stop(acc & (n_acc >= cfg.max_iter), STATUS_MAX_ITER)
        lh.append(f.copy())
        zh.append(z.copy())
        ch.append(_cost_entry(r, rows.n_fwd, rows.n_der))
        th.append(time.perf_counter() - t0)
    wall = time.perf_counter() - t0
    return OptResult(
        z_best=z,
        loss_best=f,
        loss_history=np.asarray(lh).reshape(len(lh), r),
        z_history=np.asarray(zh).reshape(len(zh), r, n),
        n_forward=int(rows.n_fwd.max(initial=0)),
        n_grad=int(rows.n_der.max(initial=0)),
        wall_s=wall,
        cost_history=_stack_costs(ch, r),
        time_history=np.asarray(th, dtype=float),
        extra={**rows.counts(), "accepted": n_acc},
    )


# ---------------------------------------------------------------------- Levenberg-Marquardt
@dataclass(frozen=True)
class LmConfig:
    """Levenberg-Marquardt settings: the damped Gauss-Newton step
    ``-(J'J + lambda D)^-1 J'r`` with ``D = diag(J'J)`` (Marquardt 1963 scaling, floored),
    ``lambda`` divided by ``lambda_factor`` after an accepted step (not below ``lambda_min``) and
    multiplied by it after a rejected one. A row stops when ``max |J'r| <= gtol`` at an accepted
    point (``gradient``; 0: only an exactly zero gradient), when a trial step's actual and
    predicted relative decreases are both at most ``ftol`` (``ftol``: the convergence test of
    MINPACK ``lmder``, More, Garbow & Hillstrom 1980, "User Guide for MINPACK-1", ANL-80-74,
    Section 2: ``|actred| <= ftol``, ``prered <= ftol`` and ``actred <= 2 prered``; a row at its
    minimum stops after one trial instead of burning ``max_rejects`` forward calls), when the trial
    step is at most ``xtol (|z| + xtol)`` long (``xtol``: the relative step test of MINPACK
    ``lmder``, Section 2, in Euclidean ``z`` units; it ends a zero-residual fit whose loss is at
    the rounding floor, where relative decreases are noise), after
    ``stall_steps`` accepted steps in a row with a relative decrease below ``stall_rel``, after
    ``max_rejects`` rejected trials in a row, when ``lambda`` exceeds ``lambda_max``, or after
    ``max_iter`` trial steps. As in MINPACK, the ``ftol`` and ``xtol`` tests are made on every
    trial, accepted or rejected (see the status notes: ``extra["rejects_in_a_row"]`` tells a
    converged row from one that stopped after a run of rejections).

    ``ftol = xtol = 0`` (the default) switches both tests off: the stop rules are then the
    default stop rules of the per-row damping loop (stall, rejects, damping, ``max_iter``,
    target). :data:`LM_MINPACK_TOL` (the square root of the double-precision
    machine epsilon, the value MINPACK-1 Section 2 recommends for ``tol`` of ``lmder1``) is the
    opt-in value; with it a row at a minimum or at the rounding floor stops earlier, so its call
    counts differ from the default rules. The other values are harness choices (``lambda0 =
    1e-3``, damping factor 3)."""

    lambda0: float = 1e-3
    lambda_factor: float = 3.0
    lambda_min: float = 1e-12
    lambda_max: float = 1e10
    max_iter: int = 40
    stall_rel: float = 1e-6
    stall_steps: int = 3
    max_rejects: int = 12
    ftol: float = 0.0
    xtol: float = 0.0
    gtol: float = 0.0


DEFAULT_LM = LmConfig()

#: floor of the Marquardt scaling diag(J'J) (a zero column keeps the damped matrix invertible)
_LM_DIAG_FLOOR = numerical_guard("calib.lm_diag_floor", 1e-12, "floor of the Marquardt scaling diag(J'J)")
#: divisor guard of the LM relative decrease (loss - trial) / loss
_LM_TINY = numerical_guard("calib.lm_tiny", 1e-300, "divisor guard of the LM relative decrease")


def _seq_sumsq(res: np.ndarray) -> np.ndarray:
    """``sum_k res[:, k]^2`` per row, added in index order with elementwise operations over the
    rows (a row's value does not depend on the other rows or the batch size)."""
    acc = res[:, 0] ** 2
    for k in range(1, res.shape[1]):
        acc = acc + res[:, k] ** 2
    return acc


def _seq_normal_equations(jac: np.ndarray, res: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(J'J, J'r)`` per row of ``jac [R, m, n]``, ``res [R, m]``, accumulated over the residuals
    in index order with elementwise operations over the rows."""
    a_m = jac[:, 0, :, None] * jac[:, 0, None, :]
    g = jac[:, 0, :] * res[:, 0, None]
    for k in range(1, jac.shape[1]):
        a_m = a_m + jac[:, k, :, None] * jac[:, k, None, :]
        g = g + jac[:, k, :] * res[:, k, None]
    return a_m, g


def _lm_step(a_m: np.ndarray, g: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """The damped Gauss-Newton step ``p = -(A + lam D)^-1 g`` (``A = J'J``, ``g = J'r``,
    ``D = diag(A)`` floored) and its predicted decrease of ``sum(r^2)``, ``|r|^2 - |r + J p|^2 =
    p'Ap + 2 lam p'Dp`` (from ``(A + lam D) p = -g``; MINPACK ``lmder``'s ``prered``). A singular
    system gives no step and a NaN prediction (a rejected trial, never an ``ftol`` stop)."""
    ddiag = np.maximum(np.diag(a_m), _LM_DIAG_FLOOR)
    damped = np.ascontiguousarray(a_m + lam * np.diag(ddiag))
    try:
        step = -np.linalg.solve(damped, np.ascontiguousarray(g))
    except np.linalg.LinAlgError:
        return np.zeros_like(g), float("nan")
    return step, float(step @ a_m @ step + 2.0 * lam * np.sum(ddiag * step**2))


def batched_residual_jacobian(
    residuals: Callable[[Array], Array],
) -> Callable[[Array], tuple[Array, Array]]:
    """``jit(vmap(z -> (r(z), jacfwd(r)(z))))``: ``[R, n] -> ([R, m], [R, m, n])``; the Jacobian
    is forward-mode AD (``n`` tangent directions)."""

    def rj(z: Array) -> tuple[Array, Array]:
        return residuals(z), jax.jacfwd(residuals)(z)

    return jax.jit(jax.vmap(rj))


def lm(
    residual_and_jacobian: Callable[[Array], tuple[ArrayLike, ArrayLike]],
    residual_batch: Callable[[Array], ArrayLike],
    z0: Any,
    cfg: LmConfig = DEFAULT_LM,
    *,
    target: Any = None,
    budget: Any = None,
    grad_cost: float | None = None,
    active: Any = None,
    forward_per_derivative: Any = 0,
) -> OptResult:
    """Batched Levenberg-Marquardt on the loss ``sum(r(z)^2)``, ``R`` independent rows from
    ``z0 [R, n]`` in the unconstrained space.

    ``residual_and_jacobian: [R, n] -> ([R, m], [R, m, n])`` (e.g.
    :func:`batched_residual_jacobian`, or a Jacobian with secant columns) at a row's accepted
    point, ``residual_batch: [R, n] -> [R, m]`` at the trial points. Each loop round makes at most
    one Jacobian call (rows at a new accepted point, every row in the first round) and one forward
    call (one trial step per active row). The step is solved per row on the host. A Jacobian call
    is counted in ``n_grad``; ``grad_cost`` is its measured cost in forward calls.

    Stops, per row (``extra["status"]``): the loss reaches ``target`` (at the first point or at an
    accepted step), the gradient / ``ftol`` / stall / reject / damping / ``max_iter`` rules of
    :class:`LmConfig`, a non-finite loss or Jacobian at the current point, or the budget (as in
    :func:`lbfgs`). The ``ftol`` and ``xtol`` stops (off by default) are checked after every
    trial's forward call, accepted or rejected (an accepted trial is kept);
    ``extra["rejects_in_a_row"]`` is the number of rejected trials in a row at the end (0: the last
    trial was accepted), which tells a converged ``ftol`` / ``xtol`` row from one that stopped
    after a run of rejections. ``forward_per_derivative`` counts the forward calls made inside one
    ``residual_and_jacobian`` call for a row (e.g. two per secant column of that row's Jacobian),
    as in :func:`lbfgs`. Rows with ``active`` false are frozen from the start (evaluated in the
    fixed-shape batch, results discarded, never counted); a stopped row keeps its state bit for
    bit and is no longer counted. Row invariance holds as in :func:`lbfgs`: bit for bit provided
    the residual callables give a row the same values at both batch shapes. ``z_best`` is the
    last accepted point (accepted losses decrease). The per-row counts are in
    ``extra["n_forward_rows"]`` / ``extra["n_grad_rows"]``; ``n_forward`` / ``n_grad`` are the
    largest of them.

    Default rules: with the default settings (``ftol = xtol = gtol = 0``, no target) a row follows
    the plain per-row damping loop (one Jacobian at each new accepted point, one trial per round;
    accept when the trial loss is lower; ``lambda`` divided by ``lambda_factor`` after an accept,
    multiplied by it after a reject; stop on stall, rejects, damping, ``max_iter`` or target).
    ``J'J``, ``J'r`` and ``sum(r^2)`` are summed in index order (needed for row invariance), so an
    implementation of the same loop with BLAS products and ``np.sum`` gives the same decisions but
    iterates that differ at rounding level. Far from the minimum this changes
    nothing visible, but near the noise floor of the loss, where acceptance (``trial < loss``) and
    the stall test are decided by the last bits, a row's status and call counts can differ. Two
    rules also differ by design: a row whose loss meets ``target`` at its first point stops before
    any trial (the plain loop makes one trial first), and a row with an exactly zero ``J'r`` stops
    (``gradient``, with ``gtol = 0``) instead of making ``max_rejects`` zero steps."""
    z = np.array(z0, dtype=float, copy=True)
    r_n, n = z.shape
    rows = _Rows(r_n, active, target, budget, grad_cost, "lm", forward_per_derivative)
    lam = np.full(r_n, cfg.lambda0)
    loss = np.full(r_n, np.inf)
    cur_a = np.zeros((r_n, n, n))  # J'J and J'r at each row's current point
    cur_g = np.zeros((r_n, n))
    need_j = rows.active.copy()
    n_acc = np.zeros(r_n, dtype=np.int64)
    n_it = np.zeros(r_n, dtype=np.int64)
    stall = np.zeros(r_n, dtype=np.int64)
    rejects = np.zeros(r_n, dtype=np.int64)
    lh, zh = [], []
    ch: list[np.ndarray] = []
    th: list[float] = []
    t0 = time.perf_counter()
    while rows.active.any():
        # ---- Jacobian call: rows at a new accepted point
        req = rows.request(need_j, derivative=True)
        if req.any():
            res, jac = residual_and_jacobian(jnp.asarray(z))
            res = np.asarray(res, dtype=float)
            jac = np.asarray(jac, dtype=float)
            rows.count_derivative(req)
            lv = _seq_sumsq(res)
            ok = np.isfinite(lv) & np.all(np.isfinite(jac), axis=(-2, -1))
            rows.stop(req & ~ok, STATUS_NONFINITE)
            got = req & ok
            with np.errstate(invalid="ignore", over="ignore"):
                a_new, g_new = _seq_normal_equations(jac, res)
            cur_a[got] = a_new[got]
            cur_g[got] = g_new[got]
            loss = np.where(got, lv, loss)
            need_j = need_j & ~got
            rows.stop(got & (loss <= rows.target), STATUS_TARGET)
            rows.stop(got & (np.max(np.abs(cur_g), axis=-1) <= cfg.gtol), STATUS_GRADIENT)
        # ---- forward call: one trial step per active row
        req = rows.request(rows.active & ~need_j, derivative=False)
        if req.any():
            trial = z.copy()
            pred = np.zeros(r_n)  # predicted decrease of sum(r^2) by the damped step
            small = np.zeros(r_n, bool)  # the step is below xtol
            for i in np.nonzero(req)[0]:  # host-side n x n solves, one per row
                step, pred[i] = _lm_step(cur_a[i], cur_g[i], float(lam[i]))
                trial[i] = z[i] + step
                small[i] = (
                    cfg.xtol > 0.0
                    and bool(np.isfinite(pred[i]))
                    and bool(np.sqrt(np.sum(step**2)) <= cfg.xtol * (np.sqrt(np.sum(z[i] ** 2)) + cfg.xtol))
                )
            lt = _seq_sumsq(np.asarray(residual_batch(jnp.asarray(trial)), dtype=float))
            loss_old = loss.copy()
            rows.n_fwd[req] += 1
            n_it = n_it + req
            acc = req & np.isfinite(lt) & (lt < loss)
            rel = np.where(acc, (loss - lt) / np.maximum(np.where(acc, loss, 1.0), _LM_TINY), 0.0)
            z = np.where(acc[:, None], trial, z)
            loss = np.where(acc, lt, loss)
            lam = np.where(acc, np.maximum(lam / cfg.lambda_factor, cfg.lambda_min), lam)
            need_j = need_j | acc
            n_acc = n_acc + acc
            rejects = np.where(acc, 0, rejects)
            stall = np.where(acc, np.where(rel < cfg.stall_rel, stall + 1, 0), stall)
            rej = req & ~acc
            lam = np.where(rej, lam * cfg.lambda_factor, lam)
            rejects = rejects + rej
            rows.stop(req & (loss <= rows.target), STATUS_TARGET)
            with np.errstate(invalid="ignore", over="ignore"):
                denom = np.maximum(loss_old, _LM_TINY)
                actred = np.where(np.isfinite(lt), (loss_old - lt) / denom, np.inf)
                prered = pred / denom
                conv = (
                    (cfg.ftol > 0.0)
                    & (np.abs(actred) <= cfg.ftol)
                    & (prered <= cfg.ftol)
                    & (actred <= 2.0 * prered)
                )
            rows.stop(req & conv, STATUS_FTOL)
            rows.stop(req & small & np.isfinite(lt), STATUS_XTOL)
            rows.stop(req & (stall >= cfg.stall_steps), STATUS_STALL)
            rows.stop(req & (rejects >= cfg.max_rejects), STATUS_REJECTS)
            rows.stop(req & (lam > cfg.lambda_max), STATUS_LAMBDA)
            rows.stop(req & (n_it >= cfg.max_iter), STATUS_MAX_ITER)
            # an accepted row that stopped needs no Jacobian at its last point
            need_j = need_j & rows.active
        lh.append(loss.copy())
        zh.append(z.copy())
        ch.append(_cost_entry(r_n, rows.n_fwd, rows.n_der))
        th.append(time.perf_counter() - t0)
    wall = time.perf_counter() - t0
    return OptResult(
        z_best=z,
        loss_best=loss,
        loss_history=np.asarray(lh).reshape(len(lh), r_n),
        z_history=np.asarray(zh).reshape(len(zh), r_n, n),
        n_forward=int(rows.n_fwd.max(initial=0)),
        n_grad=int(rows.n_der.max(initial=0)),
        wall_s=wall,
        cost_history=_stack_costs(ch, r_n),
        time_history=np.asarray(th, dtype=float),
        extra={**rows.counts(), "accepted": n_acc, "lambda": lam, "rejects_in_a_row": rejects},
    )
