"""Batched optimisers for calibration: Adam (gradient), CMA-ES and random search (gradient-free).

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
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from agrijax.core.coefficients import numerical_guard

__all__ = [
    "AdamConfig",
    "CmaConfig",
    "OptResult",
    "adam",
    "batched_loss",
    "batched_value_and_grad",
    "cma_es",
    "pair_gradient",
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
    extra: dict[str, Any] = field(default_factory=dict)


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
    one ``vmap`` batch, ``k`` the number of parameters with a secant pair. With no secant pair it
    is ``value_and_grad`` of the sum."""
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
    n_rounds = max(1, max_evals // batch)
    t0 = time.perf_counter()
    for _ in range(n_rounds):
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
    return OptResult(
        z_best=best_z,
        loss_best=best_l,
        loss_history=np.asarray(lh),
        z_history=np.asarray(zh),
        n_forward=n_rounds * batch,
        n_grad=0,
        wall_s=time.perf_counter() - t0,
    )
