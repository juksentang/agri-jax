"""Gradient-trust report: are the model's derivatives usable for calibration?

Three levels of trust in a derivative:

1. **plausible outputs** - every output is finite along every line scan;
2. **correct derivatives** - the AD derivative equals the central finite differences at small
   steps (the derivative of the forward program is computed correctly). Three-valued: the
   differences at the adjacent steps ``fd_small_steps`` (1e-4, 1e-5, 1e-6 of the width, fixed in
   advance) agree with each other and with the derivative (**pass**), with each other but not with
   it (**fail**: a derivative error, level 1), or not with each other (**undecidable**: the
   differences straddle a quantum or a jump and may not veto the derivative; no step is picked);
3. **fit for inference** - the AD derivative also predicts the response at the steps an
   optimiser takes: it agrees with the central secants at 1, 2 and 5 % of the width
   (``fd_large_steps``), and a line scan along the parameter shows no jumps the derivative does not
   explain and no stretches where the derivative is 0 while the output still moves. Where level 2
   is undecidable, level 3 alone decides (passing it gives level 3, failing it level 1): it is the
   gate the calibration relies on in any case.

**Straight-through derivatives.** A function bound to the ``ste`` gradient mode
(:func:`agrijax.core.grad.bind_gradient_mode`) differentiates on purpose not its own staircase:
through every quantiser of :mod:`agrijax.core.grad` (``trunc_st``, ``round_st``: Fortran's
``REAL(INT(x*1000))/1000``, ``ANINT(x*1E6)/1E6``) the derivative is 1 (through ``real4_store`` the
cast's, the tangent rounded to binary32), where the forward program's own derivative (the ``exact``
mode) is 0. Level 2 of such a surrogate is decided on the exact path, and the surrogate is then
correct by a three-link argument, each link tested:

(a) the exact-mode derivative ``ad_exact`` agrees with the small-step central differences (the
    three-valued level 2 above, at every point checked);
(b) the surrogate differs from the exact derivative only at the registered sites: with every
    registered straight-through call site traced in the exact mode, the ``ste`` program equals the
    exact one bit for bit (``tests/integration/test_facade_grad.py``, all coefficients, nine
    scenarios of the DSSAT maize day);
(c) at each registered site the surrogate's derivative is 1 by definition: every call of a
    straight-through helper in the package is a :class:`agrijax.core.process.GradientConvention`
    of its process (the AST call-site test of ``tests/unit/test_process_registry.py``), and the
    helpers' identity derivative is tested in ``tests/unit/test_grad_helpers.py``.

So the surrogate is the exact derivative with the registered sites' derivative set to 1, and nothing
else. The **unrounded model** (:func:`agrijax.core.grad.unrounded`: every quantiser, ``real4_store``
included, the identity) is reported as a diagnostic only: its derivative ``ad_unrounded``, its
small-step agreement ``rel_err_small_unrounded`` and ``ste_offset`` (``ad / ad_unrounded - 1``: the
surrogate is that model's derivative along the rounded trajectory, the two trajectories drift apart by
up to a quantum a step). It is not a gate because that model is not smooth either (comparisons the
rounding held exactly switch at tiny steps), so its differences often straddle a jump; the argument
above does not need it, which is why its diagnostic status is not leniency. Level 3 tests the
surrogate value itself against the optimiser-step secants and the line scan of the real model.
Measured case: the CERES-Maize derivatives of the DSSAT day (:mod:`agrijax.facade_grad`).

Tools, all batched with ``vmap`` (one call per step size / scan):

* :func:`fd_check` - AD Jacobian against central differences at the small and large step sets
  (:func:`fd_diagnostics`: the verdicts alone, :func:`fd_agreement`: one comparison);
* :func:`line_scan` - outputs and directional AD derivatives on a grid along one parameter, with
  the jump count, the zero-derivative fraction and longest stretch, and the secant / AD ratio
  (:func:`scan_summary`: the diagnostics alone, from points computed elsewhere, e.g. in a batch
  over scenarios; :func:`classify_pair`: the class and level of one pair from the two);
* :func:`sensitivity` - normalised local sensitivities ``d y / d theta * width / |y|`` and the
  scan ranges;
* :func:`trust_report` - all of it for every parameter and output, with a per-(parameter,
  output) class (``smooth``, ``kinked``, ``jumpy``, ``step``, ``inert``) and trust level;
* :func:`gradient_plan` - from a report, the method of every (parameter, treatment) pair: the
  AD derivative where every output of the treatment is trusted for that parameter (class
  ``smooth`` at level 3, or ``inert``), else derivative-free (central secants,
  :func:`agrijax.calib.optim.pair_gradient`); parameters named derivative-free by default (the
  phenology parameters of CERES) are derivative-free on every treatment.

``f`` is always a function ``theta [n] -> y [m]`` of physical parameters (named outputs).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from agrijax.core.coefficients import numerical_guard

__all__ = [
    "L2_FAIL",
    "L2_PASS",
    "L2_STATUSES",
    "L2_UNDECIDABLE",
    "GradientPlan",
    "TrustConfig",
    "classify_pair",
    "counterparts",
    "fd_agreement",
    "fd_check",
    "fd_diagnostics",
    "gradient_plan",
    "line_scan",
    "scan_summary",
    "sensitivity",
    "trust_report",
]

_TINY = numerical_guard("calib.trust_tiny", 1e-300, "floor of denominators in the trust report")


@dataclass(frozen=True)
class TrustConfig:
    """Thresholds of the report (harness choices, documented in the report itself).

    * ``fd_steps`` - the reported small and large central-difference steps (columns ``fd_small``,
      ``fd_large``), as fractions of the bound width; each is one of the step sets below;
    * ``fd_small_steps`` - the adjacent small steps of the level-2 test, fixed in advance: the
      central differences there must agree with each other (to ``fd_rtol[0]``) to decide anything;
    * ``fd_large_steps`` - the optimiser-scale steps of level 3 (1, 2 and 5 % of the width: the
      steps Adam and the staged Levenberg-Marquardt take, :mod:`agrijax.facade_grad`); the AD
      derivative must agree with the central secant at every one of them;
    * ``fd_rtol`` - relative agreement at the small steps (the derivative of the program) and at
      the large steps (looser: a smooth response has O(h^2) curvature error);
    * ``fd_rtol_unrounded`` - the small-step agreement reported for the unrounded model's derivative
      (diagnostic only, never a gate: module docstring);
    * ``abs_floor`` - derivatives with ``|d y| * width`` below ``abs_floor * max(|y|, 1)`` count
      as zero;
    * ``jump_frac`` - a grid interval of a line scan is a jump when the change the AD derivative
      does not explain (trapezoid rule) exceeds this fraction of the output's scan range;
    * ``n_scan`` - scan points; ``scan_frac`` - half-width of the scan around the centre, as a
      fraction of the bound width (clipped to the bounds);
    * ``secant_band`` - level 3 requires the mean AD derivative over the scan to be within this
      factor of the scan's end-to-end secant.
    """

    fd_steps: tuple[float, float] = (1e-5, 2e-2)
    fd_small_steps: tuple[float, ...] = (1e-4, 1e-5, 1e-6)
    fd_large_steps: tuple[float, ...] = (1e-2, 2e-2, 5e-2)
    fd_rtol: tuple[float, float] = (1e-3, 5e-2)
    fd_rtol_unrounded: float = 1e-2
    abs_floor: float = 1e-10
    jump_frac: float = 0.01
    n_scan: int = 41
    scan_frac: float = 0.1
    secant_band: float = 2.0

    def __post_init__(self) -> None:
        if self.fd_steps[0] not in self.fd_small_steps or self.fd_steps[1] not in self.fd_large_steps:
            raise ValueError(
                f"fd_steps {self.fd_steps} must be one of fd_small_steps {self.fd_small_steps} and one of "
                f"fd_large_steps {self.fd_large_steps}"
            )


DEFAULT_TRUST = TrustConfig()

#: the level-2 outcomes: the small-step differences agree with each other and with the derivative
#: (pass), with each other but not with it (fail: a derivative error), or not with each other (the
#: differences straddle a quantum or a jump: they cannot judge the derivative, level 3 decides)
L2_PASS, L2_FAIL, L2_UNDECIDABLE = "pass", "fail", "undecidable (FD unreliable)"
L2_STATUSES: tuple[str, ...] = (L2_PASS, L2_FAIL, L2_UNDECIDABLE)


# ------------------------------------------------------------------------------ finite differences
def _jac(f: Callable[[Array], Array], x: Array) -> Array:
    return jax.jacfwd(f)(x)


def counterparts(
    f: Callable[[Array], Array],
) -> tuple[Callable[[Array], Array] | None, Callable[[Array], Array] | None]:
    """``(f_exact, f_unrounded)`` of a :class:`~agrijax.core.grad.ModeBound` ``f`` of a mode other
    than ``exact`` (its derivative is straight-through: module docstring): the same function in the
    ``exact`` mode, and the unrounded model (:func:`agrijax.core.grad.bind_unrounded`) in ``f``'s
    mode; ``(None, None)`` for any other ``f`` (its derivative is already the program's own)."""
    from agrijax.core.grad import ModeBound, bind_gradient_mode, bind_unrounded

    if isinstance(f, ModeBound) and f.mode != "exact":
        return bind_gradient_mode(f.fn, "exact"), bind_gradient_mode(bind_unrounded(f.fn), f.mode)
    return None, None


def fd_check(
    f: Callable[[Array], Array],
    x: Any,
    width: Any,
    cfg: TrustConfig = DEFAULT_TRUST,
    *,
    f_exact: Callable[[Array], Array] | None = None,
    f_unrounded: Callable[[Array], Array] | None = None,
) -> dict[str, np.ndarray]:
    """AD Jacobian ``[m, n]``, central differences at the small and large step sets (fractions of
    ``width``; all points of one step in one ``vmap`` call) and their verdicts
    (:func:`fd_diagnostics`).

    For an ``f`` whose derivative is straight-through (module docstring): ``f_exact`` (the same
    function in the ``exact`` mode) gives ``ad_exact``, the derivative level 2 judges; the large
    steps judge ``ad``. ``f_unrounded`` (the unrounded model) gives the diagnostic ``ad_unrounded``,
    its small-step difference ``fdu0`` and their disagreement ``rel_erru0`` (never a gate). Without
    them ``ad_exact`` is ``ad`` and the unrounded keys are absent."""
    x = jnp.asarray(x, dtype=float)
    width = np.asarray(width, dtype=float)
    n = x.shape[0]
    ad = np.asarray(jax.jit(lambda z: _jac(f, z))(x))
    ad_exact = ad if f_exact is None else np.asarray(jax.jit(lambda z: _jac(f_exact, z))(x))
    y0 = np.asarray(jax.jit(f)(x))
    fv = jax.jit(jax.vmap(f))

    def central(fn: Any, frac: float) -> np.ndarray:
        h = frac * width
        pts = np.concatenate([np.asarray(x) + np.diag(h), np.asarray(x) - np.diag(h)])
        vals = np.asarray(fn(jnp.asarray(pts)))
        return ((vals[:n] - vals[n:]) / (2 * h[:, None])).T  # [m, n]

    small = [central(fv, fr) for fr in cfg.fd_small_steps]
    large = [central(fv, fr) for fr in cfg.fd_large_steps]
    out: dict[str, np.ndarray] = {"ad": ad, "ad_exact": ad_exact, "y0": y0}
    out.update(fd_diagnostics(ad, ad_exact, small, large, y0, width, cfg))
    out["h0"], out["h1"] = cfg.fd_steps[0] * width, cfg.fd_steps[1] * width
    if f_unrounded is not None:
        adu = np.asarray(jax.jit(lambda z: _jac(f_unrounded, z))(x))
        yu = np.asarray(jax.jit(f_unrounded)(x))
        fdu = central(jax.jit(jax.vmap(f_unrounded)), cfg.fd_steps[0])
        relu, agu = fd_agreement(adu, fdu, yu, width, 0, cfg, cfg.fd_rtol_unrounded)
        out.update(ad_unrounded=adu, fdu0=fdu, rel_erru0=relu, agreeu0=agu)
    return out


def fd_diagnostics(
    ad: Any,
    ad_exact: Any,
    small: Sequence[Any],
    large: Sequence[Any],
    y0: Any,
    width: Any,
    cfg: TrustConfig = DEFAULT_TRUST,
) -> dict[str, np.ndarray]:
    """The level-2 and level-3 verdicts of the derivatives ``ad`` (as differentiated) and ``ad_exact``
    (the program's own; ``ad`` itself for an ordinary function), ``[m, n]``, from the central
    differences ``small`` at ``cfg.fd_small_steps`` and ``large`` at ``cfg.fd_large_steps``.

    Level 2 (``status0``, :data:`L2_STATUSES`): the small-step differences agree with each other
    to ``cfg.fd_rtol[0]`` (or are all zero within ``cfg.abs_floor``) and ``ad_exact`` with the
    one at ``cfg.fd_steps[0]`` -> pass; they agree with each other but not with ``ad_exact`` ->
    fail; they do not agree with each other -> undecidable (no step is picked: a difference that
    straddles a quantum or a jump says nothing about the derivative). ``agree0`` = pass,
    ``rel_err0`` = the disagreement with ``fd0``, ``spread0`` = the spread of the small-step
    differences. Level 3 input (``agree1``): ``ad`` agrees with every large-step secant to
    ``cfg.fd_rtol[1]``; ``rel_err1`` is the disagreement at ``cfg.fd_steps[1]`` (``fd1``),
    ``rel_err1_max`` the largest over the large steps."""
    ad, ad_exact = np.asarray(ad), np.asarray(ad_exact)
    y0, width = np.asarray(y0), np.asarray(width)
    fs, fl = np.stack([np.asarray(a) for a in small]), np.stack([np.asarray(a) for a in large])
    fd0 = fs[list(cfg.fd_small_steps).index(cfg.fd_steps[0])]
    fd1 = fl[list(cfg.fd_large_steps).index(cfg.fd_steps[1])]
    floor = cfg.abs_floor * np.maximum(np.abs(y0), 1.0)[:, None] / width[None, :]
    spread = (fs.max(axis=0) - fs.min(axis=0)) / np.maximum(np.abs(fs).max(axis=0), _TINY)
    all_zero = np.all(np.abs(fs) <= floor[None], axis=0)
    consistent = all_zero | (spread <= cfg.fd_rtol[0])
    rel0, ag0 = fd_agreement(ad_exact, fd0, y0, width, 0, cfg)
    status = np.where(~consistent, L2_UNDECIDABLE, np.where(ag0, L2_PASS, L2_FAIL)).astype(object)
    rels, ags = zip(*(fd_agreement(ad, f, y0, width, 1, cfg) for f in fl), strict=True)
    rel1, _ = fd_agreement(ad, fd1, y0, width, 1, cfg)
    return {
        "fd0": fd0,
        "fd1": fd1,
        "rel_err0": rel0,
        "status0": status,
        "agree0": status == L2_PASS,
        "spread0": np.where(all_zero, 0.0, spread),
        "rel_err1": rel1,
        "rel_err1_max": np.max(np.stack(rels), axis=0),
        "agree1": np.all(np.stack(ags), axis=0),
    }


def fd_agreement(
    ad: Any,
    fd: Any,
    y0: Any,
    width: Any,
    k: int,
    cfg: TrustConfig = DEFAULT_TRUST,
    rtol: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Relative AD / finite-difference disagreement and whether they agree, ``[m, n]``, at step
    ``k`` of ``cfg.fd_steps`` (the comparison of :func:`fd_check`: ``ad``, ``fd`` ``[m, n]``,
    ``y0`` ``[m]``, ``width`` ``[n]``; both derivatives zero within ``cfg.abs_floor`` count as
    agreeing). ``rtol`` overrides ``cfg.fd_rtol[k]``."""
    ad, fd = np.asarray(ad), np.asarray(fd)
    y0, width = np.asarray(y0), np.asarray(width)
    den = np.maximum(np.maximum(np.abs(ad), np.abs(fd)), _TINY)
    rel = np.abs(ad - fd) / den
    floor = cfg.abs_floor * np.maximum(np.abs(y0), 1.0)[:, None] / width[None, :]
    both_zero = (np.abs(ad) <= floor) & (np.abs(fd) <= floor)
    tol = cfg.fd_rtol[k] if rtol is None else rtol
    return np.where(both_zero, 0.0, rel), both_zero | (rel <= tol)


# ------------------------------------------------------------------------------ line scans
def line_scan(
    f: Callable[[Array], Array],
    x: Any,
    i: int,
    lo: float,
    hi: float,
    cfg: TrustConfig = DEFAULT_TRUST,
    *,
    width: float | None = None,
) -> dict[str, np.ndarray]:
    """Outputs ``y [P, m]`` and directional derivatives ``g [P, m]`` (``d y / d theta_i``, AD)
    on ``P = cfg.n_scan`` points from ``lo`` to ``hi`` along parameter ``i``, with per-output
    diagnostics: ``n_jumps``, ``max_jump`` (unexplained change / range), ``zero_frac``,
    ``zero_run`` (longest run of zero-derivative points), ``zero_run_moves`` (the output changes
    across that run), ``range``, ``secant`` and ``ad_mean``."""
    x = np.asarray(x, dtype=float)
    grid = np.linspace(lo, hi, cfg.n_scan)
    pts = np.repeat(x[None], cfg.n_scan, axis=0)
    pts[:, i] = grid
    e = jnp.zeros(x.shape[0]).at[i].set(1.0)

    def one(t: Array) -> tuple[Array, Array]:
        return jax.jvp(f, (t,), (e,))

    y, g = jax.jit(jax.vmap(one))(jnp.asarray(pts))
    return scan_summary(grid, np.asarray(y), np.asarray(g), float(hi - lo) if width is None else width, cfg)


def scan_summary(
    grid: Any, y: Any, g: Any, width: float, cfg: TrustConfig = DEFAULT_TRUST
) -> dict[str, np.ndarray]:
    """The diagnostics of a line scan (the second half of :func:`line_scan`) from its points:
    ``grid [P]`` along the parameter, outputs ``y [P, m]`` and directional AD derivatives
    ``g [P, m]`` there, ``width`` the parameter's bound width (the keys of :func:`line_scan`)."""
    grid, y, g = np.asarray(grid), np.asarray(y), np.asarray(g)
    w = float(width)
    dx = np.diff(grid)[:, None]
    actual = np.diff(y, axis=0)
    pred = 0.5 * (g[1:] + g[:-1]) * dx
    rng = y.max(axis=0) - y.min(axis=0)
    scale = np.maximum(np.abs(y).max(axis=0), 1.0)
    unexplained = np.abs(actual - pred)
    jump = unexplained > cfg.jump_frac * np.maximum(rng, cfg.abs_floor * scale)
    zero = np.abs(g) * w <= cfg.abs_floor * scale
    runs = np.zeros(y.shape[1], dtype=int)
    moves = np.zeros(y.shape[1], dtype=bool)
    for j in range(y.shape[1]):
        best, cur, start, best_start = 0, 0, 0, 0
        for k, z in enumerate(zero[:, j]):
            if z:
                if cur == 0:
                    start = k
                cur += 1
                if cur > best:
                    best, best_start = cur, start
            else:
                cur = 0
        runs[j] = best
        if best > 1:
            seg = y[best_start : best_start + best, j]
            moves[j] = bool(seg.max() - seg.min() > cfg.abs_floor * scale[j])
    secant = (y[-1] - y[0]) / (grid[-1] - grid[0])
    return {
        "grid": grid,
        "y": y,
        "g": g,
        "finite": np.all(np.isfinite(y), axis=0) & np.all(np.isfinite(g), axis=0),
        "n_jumps": jump.sum(axis=0),
        "max_jump": (unexplained / np.maximum(rng, _TINY)).max(axis=0),
        "zero_frac": zero.mean(axis=0),
        "zero_run": runs,
        "zero_run_moves": moves,
        "range": rng,
        "secant": secant,
        "ad_mean": g.mean(axis=0),
    }


def sensitivity(ad: Any, y0: Any, width: Any) -> np.ndarray:
    """Normalised local sensitivity ``d y / d theta * width / max(|y|, tiny)`` ``[m, n]``."""
    ad = np.asarray(ad)
    return ad * np.asarray(width)[None, :] / np.maximum(np.abs(np.asarray(y0)), _TINY)[:, None]


# ------------------------------------------------------------------------------ the report
def _classify(fd: dict[str, np.ndarray], sc: dict[str, np.ndarray], j: int, i: int, cfg: TrustConfig) -> str:
    if sc["range"][j] <= cfg.abs_floor * max(float(np.abs(sc["y"][:, j]).max()), 1.0):
        return "inert"
    if sc["zero_frac"][j] >= 0.5 and sc["zero_run_moves"][j]:
        return "step"
    if sc["n_jumps"][j] > 0:
        return "jumpy"
    if not bool(fd["agree1"][j, i]):
        return "kinked"
    return "smooth"


def _level(fd: dict[str, np.ndarray], sc: dict[str, np.ndarray], j: int, i: int, cfg: TrustConfig) -> int:
    if not bool(sc["finite"][j]):
        return 0
    status = fd["status0"][j, i] if "status0" in fd else (L2_PASS if bool(fd["agree0"][j, i]) else L2_FAIL)
    if status == L2_FAIL:
        return 1
    sec, mean = float(sc["secant"][j]), float(sc["ad_mean"][j])
    scale = max(float(np.abs(sc["y"][:, j]).max()), 1.0)
    tiny = cfg.abs_floor * scale / max(float(sc["grid"][-1] - sc["grid"][0]), _TINY)
    if abs(sec) <= tiny and abs(mean) <= tiny:
        ratio_ok = True
    else:
        ratio = mean / sec if abs(sec) > tiny else np.inf
        ratio_ok = 1.0 / cfg.secant_band <= ratio <= cfg.secant_band
    fit = (
        bool(fd["agree1"][j, i])
        and int(sc["n_jumps"][j]) == 0
        and not (sc["zero_run_moves"][j] and sc["zero_frac"][j] > 0.0)
        and ratio_ok
    )
    if status == L2_UNDECIDABLE:  # level 3 decides; failing it, the derivative is not shown correct
        return 3 if fit else 1
    return 3 if fit else 2


def classify_pair(
    agree: tuple[bool, bool],
    scan: Mapping[str, np.ndarray],
    j: int,
    cfg: TrustConfig = DEFAULT_TRUST,
    status: str | None = None,
) -> tuple[str, int]:
    """``(class, level)`` of output ``j`` for one parameter, as :func:`trust_report` assigns them:
    ``agree`` = (level 2 passed, the AD derivative agrees with every large-step secant)
    (:func:`fd_diagnostics` ``agree0`` / ``agree1``), ``status`` the level-2 outcome
    (:data:`L2_STATUSES`; default from ``agree[0]``), ``scan`` = the parameter's :func:`line_scan`
    / :func:`scan_summary` of that output set."""
    st = (L2_PASS if agree[0] else L2_FAIL) if status is None else status
    fd = {
        "agree0": np.full((j + 1, 1), agree[0]),
        "agree1": np.full((j + 1, 1), agree[1]),
        "status0": np.full((j + 1, 1), st, dtype=object),
    }
    return _classify(fd, dict(scan), j, 0, cfg), _level(fd, dict(scan), j, 0, cfg)


def trust_report(
    f: Callable[[Array], Array],
    x: Any,
    lower: Any,
    upper: Any,
    param_names: Sequence[str],
    output_names: Sequence[str],
    cfg: TrustConfig = DEFAULT_TRUST,
    *,
    f_exact: Callable[[Array], Array] | None = None,
    f_unrounded: Callable[[Array], Array] | None = None,
    diagnostics: bool = True,
) -> dict[str, Any]:
    """The gradient-trust report of ``f`` at ``x`` (JSON-serialisable dict).

    For every parameter ``i`` a line scan over ``x_i +- cfg.scan_frac * width_i`` (clipped to
    the bounds); for every (parameter, output) the FD agreement at both steps, the scan
    diagnostics, the normalised sensitivity, the class and the trust level (0-3).

    ``f_exact``, ``f_unrounded``: the counterparts of a straight-through ``f`` (:func:`fd_check`);
    default :func:`counterparts` of ``f`` (for a ``ModeBound`` in the ``ste`` / ``implicit`` mode).
    Each output then also reports ``ad_exact`` (``rel_err_small`` is measured on it),
    ``ad_unrounded``, ``fd_small_unrounded``, ``rel_err_small_unrounded`` and ``ste_offset``
    (``ad / ad_unrounded - 1``). ``diagnostics=False`` skips the unrounded model (it is never a gate,
    so the verdicts are the same; a calibration need not compile it) unless ``f_unrounded`` is
    given."""
    x = np.asarray(x, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    width = hi - lo
    dx, du = counterparts(f)
    fd = fd_check(
        f,
        x,
        width,
        cfg,
        f_exact=dx if f_exact is None else f_exact,
        f_unrounded=(du if diagnostics else None) if f_unrounded is None else f_unrounded,
    )
    sens = sensitivity(fd["ad"], fd["y0"], width)
    params: dict[str, Any] = {}
    for i, pn in enumerate(param_names):
        a = max(lo[i], x[i] - cfg.scan_frac * width[i])
        b = min(hi[i], x[i] + cfg.scan_frac * width[i])
        sc = line_scan(f, x, i, a, b, cfg, width=width[i])
        outs = {}
        for j, on in enumerate(output_names):
            outs[on] = {
                "ad": float(fd["ad"][j, i]),
                "ad_exact": float(fd["ad_exact"][j, i]),
                **(
                    {
                        "ad_unrounded": float(fd["ad_unrounded"][j, i]),
                        "fd_small_unrounded": float(fd["fdu0"][j, i]),
                        "rel_err_small_unrounded": float(fd["rel_erru0"][j, i]),
                        "ste_offset": float(fd["ad"][j, i] / fd["ad_unrounded"][j, i] - 1.0)
                        if fd["ad_unrounded"][j, i] != 0.0
                        else 0.0,
                    }
                    if "ad_unrounded" in fd
                    else {}
                ),
                "fd_small": float(fd["fd0"][j, i]),
                "fd_large": float(fd["fd1"][j, i]),
                "rel_err_small": float(fd["rel_err0"][j, i]),
                "level2": str(fd["status0"][j, i]),
                "fd_spread_small": float(fd["spread0"][j, i]),
                "rel_err_large": float(fd["rel_err1"][j, i]),
                "rel_err_large_max": float(fd["rel_err1_max"][j, i]),
                "sensitivity": float(sens[j, i]),
                "scan_range": float(sc["range"][j]),
                "scan_secant": float(sc["secant"][j]),
                "scan_ad_mean": float(sc["ad_mean"][j]),
                "n_jumps": int(sc["n_jumps"][j]),
                "max_jump_frac": float(sc["max_jump"][j]),
                "zero_frac": float(sc["zero_frac"][j]),
                "zero_run": int(sc["zero_run"][j]),
                "zero_run_moves": bool(sc["zero_run_moves"][j]),
                "class": _classify(fd, sc, j, i, cfg),
                "level": _level(fd, sc, j, i, cfg),
            }
        params[pn] = {
            "x": float(x[i]),
            "scan": [float(a), float(b)],
            "outputs": outs,
            "scan_y": sc["y"].tolist(),
            "scan_g": sc["g"].tolist(),
            "scan_grid": sc["grid"].tolist(),
        }
    return {
        "config": asdict(cfg),
        "x": x.tolist(),
        "lower": lo.tolist(),
        "upper": hi.tolist(),
        "outputs": list(output_names),
        "y0": np.asarray(fd["y0"]).tolist(),
        "params": params,
    }


# ------------------------------------------------------------------------------ gradient plan
def _pair_ok(o: Mapping[str, Any]) -> bool:
    """A (parameter, output) whose AD derivative a calibration uses: smooth at level 3, or inert."""
    return o["class"] == "inert" or (o["class"] == "smooth" and int(o["level"]) == 3)


@dataclass(frozen=True)
class GradientPlan:
    """Method of every (treatment, parameter) pair of a calibration.

    ``use_ad[b, i]`` is true where treatment ``b``'s loss is differentiated by AD with respect to
    parameter ``i``; elsewhere a derivative-free estimate is used (central secants of that
    treatment's loss, :func:`agrijax.calib.optim.pair_gradient`, or a derivative-free optimiser
    for a parameter derivative-free on every treatment). ``method[name]`` is ``"gradient"`` (AD
    on every treatment), ``"hybrid"`` (AD on some) or ``"derivative_free"``; ``reasons[name]``
    lists why pairs fell back."""

    params: tuple[str, ...]
    treatments: tuple[str, ...]
    use_ad: np.ndarray
    method: dict[str, str]
    reasons: dict[str, list[str]] = field(default_factory=dict)

    @property
    def secant_mask(self) -> np.ndarray:
        """``[B, n]`` booleans: the pairs estimated by secants (``~use_ad``)."""
        return ~self.use_ad

    @property
    def gradient_params(self) -> tuple[str, ...]:
        """Parameters with AD on at least one treatment (``gradient`` or ``hybrid``)."""
        return tuple(p for p in self.params if self.method[p] != "derivative_free")

    @property
    def derivative_free_params(self) -> tuple[str, ...]:
        return tuple(p for p in self.params if self.method[p] == "derivative_free")

    def table(self) -> list[dict[str, Any]]:
        """One row per parameter (for reports)."""
        return [
            {
                "param": p,
                "method": self.method[p],
                "ad_treatments": [t for b, t in enumerate(self.treatments) if self.use_ad[b, i]],
                "reasons": list(self.reasons.get(p, [])),
            }
            for i, p in enumerate(self.params)
        ]


def gradient_plan(
    report: Mapping[str, Any],
    treatments: Mapping[str, Sequence[str]] | None = None,
    *,
    derivative_free: Sequence[str] = (),
) -> GradientPlan:
    """The :class:`GradientPlan` of a :func:`trust_report`.

    ``treatments`` maps each treatment to the report outputs that belong to it (``None``: every
    output is its own treatment; the natural report for a calibration is the one of the
    per-treatment losses, :meth:`agrijax.calib.objective.Objective.treatment_losses`). A pair
    uses AD when every output of the treatment is trusted for the parameter (class ``smooth`` at
    level 3, or ``inert``: a zero derivative that is right); a pair with jumps, zero-derivative
    stretches, a kink or an FD disagreement falls back to derivative-free. Parameters in
    ``derivative_free`` are derivative-free on every treatment, whatever the report says (a
    default, e.g. the phenology parameters, whose event gradients are experimental).

    Limitation: the plan is only as good as the report's line scans. A jump narrower than the
    grid spacing (``2 scan_frac width / (n_scan - 1)``; 2.3 % of the bound width with the default
    41 points) can fall between two points and go unseen: on IUAF9901 treatment 1 the default
    scan classes (G2, t1) smooth, while a 401-point scan of the same interval finds +165 / -161
    kg ha-1 yield jumps (the stage-5 stem-floor count going 2 -> 1 -> 2 on day 173).
    For a decision that matters, pass a report with a finer scan near the start point
    (``TrustConfig(n_scan=401)`` or a smaller ``scan_frac``)."""
    params = tuple(report["params"])
    outputs = list(report["outputs"])
    groups: dict[str, list[str]] = (
        {o: [o] for o in outputs} if treatments is None else {k: list(v) for k, v in treatments.items()}
    )
    unknown = sorted({o for v in groups.values() for o in v} - set(outputs))
    if unknown:
        raise ValueError(f"gradient_plan: outputs not in the report: {unknown}")
    df = set(derivative_free)
    tnames = tuple(groups)
    use = np.zeros((len(tnames), len(params)), dtype=bool)
    reasons: dict[str, list[str]] = {p: [] for p in params}
    for i, p in enumerate(params):
        outs = report["params"][p]["outputs"]
        for b, t in enumerate(tnames):
            if p in df:
                continue
            bad = [o for o in groups[t] if not _pair_ok(outs[o])]
            use[b, i] = not bad
            for o in bad:
                r = outs[o]
                reasons[p].append(f"{t}: {o} class {r['class']} level {r['level']} ({r['n_jumps']} jumps)")
        if p in df:
            reasons[p].append("derivative-free by default")
    method = {
        p: ("gradient" if use[:, i].all() else "hybrid" if use[:, i].any() else "derivative_free")
        for i, p in enumerate(params)
    }
    return GradientPlan(params, tnames, use, method, {p: r for p, r in reasons.items() if r})
