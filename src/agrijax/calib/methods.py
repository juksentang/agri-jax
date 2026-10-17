"""Per-parameter method selection: which calibrated parameters get a gradient, which a derivative-free search.

:func:`plan_methods` runs the gradient-trust report (:func:`agrijax.calib.trust.trust_report`) of a
per-treatment loss at one point and splits the parameters into two roles:

* ``"gradient"`` - the AD derivative is trusted on **every** treatment (each (treatment, parameter)
  pair smooth at trust level 3, or inert: :func:`agrijax.calib.trust.gradient_plan`);
* ``"derivative_free"`` - at least one treatment's pair is not trusted (jumps, steps, kinks, an
  AD / finite-difference disagreement, a failed level 3 or a non-finite output), or the parameter
  carries a prior label (by default the CERES-Maize phenology coefficients
  :data:`agrijax.calib.ceres.CERES_DERIVATIVE_FREE`, whose event gradients are experimental).

The split is recorded in a :class:`MethodPlan` with its source (``"diagnosis"``, ``"prior"`` or
``"assumed"``), the diagnosis-only class even where a prior label overrides it, the per-pair verdicts
and the report itself: which parameters a gradient method (:func:`agrijax.calib.optim.lbfgs`,
:func:`~agrijax.calib.optim.adam`) may move and which a derivative-free one
(:func:`~agrijax.calib.optim.cma_es`) should, for example in alternating blocks.

The correspondence with :class:`~agrijax.calib.trust.GradientPlan` (the per-pair plan,
:attr:`MethodPlan.pairs`): its ``method`` ``"gradient"`` is a ``"gradient"`` role here; its
``"hybrid"`` (AD on some treatments only) and ``"derivative_free"`` are both ``"derivative_free"``
here (a method that alternates blocks gives a parameter either a gradient everywhere or none).

This module imports no model: the CERES prior is imported when the default prior is used.
"""

from __future__ import annotations

import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from agrijax.calib.trust import L2_FAIL, GradientPlan, TrustConfig, counterparts, gradient_plan, trust_report

__all__ = [
    "ALTERNATING_ROUNDS",
    "NOT_RUN",
    "ROLES",
    "SOURCES",
    "TRUST_SCAN",
    "CalibrationWarning",
    "MethodPlan",
    "plan_methods",
]

#: line-scan points of the trust report a plan runs (design choice: the scan spans ``2 scan_frac`` of
#: the bound width in ``n_scan - 1`` steps, so with the default ``scan_frac`` 0.1 the report's own 41
#: points are 0.5 % of the width apart and can miss a narrower jump; 201 points are 0.1 % apart)
TRUST_SCAN = 201
#: rounds of a block-alternating calibration: derivative-free block, then gradient block, this many
#: times (design choice)
ALTERNATING_ROUNDS = 3

ROLE_GRADIENT = "gradient"
ROLE_DERIVATIVE_FREE = "derivative_free"
ROLES: tuple[str, ...] = (ROLE_GRADIENT, ROLE_DERIVATIVE_FREE)
SOURCE_DIAGNOSIS = "diagnosis"
SOURCE_PRIOR = "prior"
SOURCE_ASSUMED = "assumed"
SOURCES: tuple[str, ...] = (SOURCE_DIAGNOSIS, SOURCE_PRIOR, SOURCE_ASSUMED)
#: the diagnosis-only class of a parameter when the diagnosis did not run (``diagnose=False``)
NOT_RUN = "not_run"
#: the worst-pair class of a non-finite output (trust level 0)
NONFINITE = "nonfinite"

#: severity of a (parameter, treatment) pair, worst first (:meth:`MethodPlan.table` ``worst_class``)
_SEV_NONFINITE, _SEV_JUMPY, _SEV_STEP, _SEV_KINKED, _SEV_L2_FAIL, _SEV_L3_FAIL, _SEV_SMOOTH, _SEV_INERT = (
    7,
    6,
    5,
    4,
    3,
    2,
    1,
    0,
)
_SEV_CLASS = {"jumpy": _SEV_JUMPY, "step": _SEV_STEP, "kinked": _SEV_KINKED}
#: the trust level at which the derivative is fit for inference (trust module docstring)
_LEVEL_FIT = 3

#: JAX events whose durations count as compilation (tracing, lowering, backend compile)
_COMPILE_EVENTS = (
    "/jax/core/compile/jaxpr_trace_duration",
    "/jax/core/compile/jaxpr_to_mlir_module_duration",
    "/jax/core/compile/backend_compile_duration",
)


class CalibrationWarning(UserWarning):
    """Something the calibration did or found that the user should know (also in the result)."""


# ------------------------------------------------------------------ the plan
def _jsonable(x: Any) -> Any:
    """``x`` with tuples as lists and NumPy values as Python ones (what ``json.loads`` gives back)."""
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def _pair_severity(o: Mapping[str, Any]) -> tuple[int, str]:
    """``(severity, class)`` of one (parameter, output) entry of a trust report (order: module)."""
    level, cls = int(o["level"]), str(o["class"])
    if level == 0:
        return _SEV_NONFINITE, NONFINITE
    if cls in _SEV_CLASS:
        return _SEV_CLASS[cls], cls
    if cls == "smooth":
        if o.get("level2") == L2_FAIL:
            return _SEV_L2_FAIL, cls
        return (_SEV_SMOOTH if level >= _LEVEL_FIT else _SEV_L3_FAIL), cls
    return _SEV_INERT, cls


@dataclass(frozen=True, eq=False)
class MethodPlan:
    """The method split of :func:`plan_methods` (module docstring).

    ``params`` / ``treatments`` the names (theta order / loss order); ``theta0`` the diagnosis point
    (physical values); ``role[name]`` ``"gradient"`` or ``"derivative_free"`` (the recommended split);
    ``source[name]`` ``"diagnosis"``, ``"prior"`` (a prior label) or ``"assumed"`` (no diagnosis, no
    label: taken as gradient); ``diagnosed[name]`` the diagnosis-only class (``"gradient"``,
    ``"derivative_free"``, or ``"not_run"``), recorded even where a prior label overrides it;
    ``inert[name]`` the parameter moves no treatment's loss within the scan; ``pairs`` the per-pair
    :class:`~agrijax.calib.trust.GradientPlan` computed **without** the prior labels (so ``diagnosed``
    can be recomputed from it; ``None`` without a diagnosis); ``reasons[name]`` readable reasons;
    ``report`` the trust report (JSON-serialisable; ``None`` without a diagnosis); ``config`` the
    trust configuration, the prior labels, ``allow_gradient``, the gradient mode and float64;
    ``calls`` the diagnosis cost (forward, JVP and Jacobian calls, ``wall_s``, ``compile_s``).

    A plan holds for the problem it was made on: it records the treatment names, not the inputs
    behind them."""

    params: tuple[str, ...]
    treatments: tuple[str, ...]
    theta0: np.ndarray
    role: dict[str, str]
    source: dict[str, str]
    diagnosed: dict[str, str]
    inert: dict[str, bool]
    pairs: GradientPlan | None
    reasons: dict[str, list[str]] = field(default_factory=dict)
    report: dict[str, Any] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    calls: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        # short: the generated repr would print the whole trust report (every scan)
        grad = tuple(p for p in self.params if self.role.get(p) == ROLE_GRADIENT)
        dfree = tuple(p for p in self.params if self.role.get(p) != ROLE_GRADIENT)
        return (
            f"MethodPlan(params={self.params!r}, gradient={grad!r}, derivative_free={dfree!r}, "
            f"treatments={len(self.treatments)}, diagnosed={self.report is not None})"
        )

    # ------------------------------------------------------------------ views
    @property
    def gradient_params(self) -> tuple[str, ...]:
        """Parameters whose role is ``"gradient"`` (trusted on every treatment). Unlike
        :attr:`agrijax.calib.trust.GradientPlan.gradient_params`, a parameter trusted on some
        treatments only is not among them."""
        return tuple(p for p in self.params if self.role[p] == ROLE_GRADIENT)

    @property
    def derivative_free_params(self) -> tuple[str, ...]:
        """Parameters whose role is ``"derivative_free"``."""
        return tuple(p for p in self.params if self.role[p] == ROLE_DERIVATIVE_FREE)

    def _worst(self, p: str) -> tuple[str | None, int | None]:
        if self.report is None:
            return None, None
        outs = self.report["params"][p]["outputs"]
        worst = max(outs.values(), key=lambda o: _pair_severity(o)[0])
        return _pair_severity(worst)[1], int(worst["level"])

    def _ad_treatments(self, p: str) -> list[str] | None:
        if self.pairs is None:
            return None
        i = self.pairs.params.index(p)
        return [t for b, t in enumerate(self.pairs.treatments) if bool(self.pairs.use_ad[b, i])]

    def table(self) -> list[dict[str, Any]]:
        """One row per parameter: ``param, role, source, diagnosed, inert, ad_treatments`` (the
        treatments whose pair is trusted, as in :meth:`GradientPlan.table`), ``worst_class`` /
        ``worst_level`` (the worst pair over the treatments, severity non-finite > jumpy > step >
        kinked > level-2 fail > smooth failing level 3 > smooth at level 3 > inert; ``worst_level``
        is that pair's trust level 0-3; both ``None`` without a diagnosis) and ``reasons``."""
        rows = []
        for p in self.params:
            wc, wl = self._worst(p)
            rows.append(
                {
                    "param": p,
                    "role": self.role[p],
                    "source": self.source[p],
                    "diagnosed": self.diagnosed[p],
                    "inert": bool(self.inert[p]),
                    "ad_treatments": self._ad_treatments(p),
                    "worst_class": wc,
                    "worst_level": wl,
                    "reasons": list(self.reasons.get(p, [])),
                }
            )
        return rows

    def __str__(self) -> str:
        head = ("param", "role", "source", "diagnosed", "AD on", "worst")
        body = []
        for r in self.table():
            ad = "-" if r["ad_treatments"] is None else f"{len(r['ad_treatments'])}/{len(self.treatments)}"
            worst = "-" if r["worst_class"] is None else f"{r['worst_class']} L{r['worst_level']}"
            body.append((r["param"], r["role"], r["source"], r["diagnosed"], ad, worst))
        widths = [max(len(str(x[k])) for x in (head, *body)) for k in range(len(head))]
        lines = [
            "  ".join(str(x[k]).ljust(widths[k]) for k in range(len(head))).rstrip() for x in (head, *body)
        ]
        c = self.calls
        if self.report is None:
            lines.append("diagnosis: not run (prior labels only)")
        else:
            lines.append(
                f"diagnosis: {c.get('forward', 0)} forward + {c.get('jvp', 0)} JVP + {c.get('jacobian', 0)} "
                f"Jacobian calls, {c.get('wall_s', 0.0):.1f} s (compile {c.get('compile_s', 0.0):.1f} s)"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ JSON
    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view; :meth:`from_dict` rebuilds the plan."""
        pairs = None
        if self.pairs is not None:
            pairs = {
                "params": list(self.pairs.params),
                "treatments": list(self.pairs.treatments),
                "use_ad": np.asarray(self.pairs.use_ad, dtype=bool).tolist(),
                "method": dict(self.pairs.method),
                "reasons": {k: list(v) for k, v in self.pairs.reasons.items()},
            }
        return _jsonable(
            {
                "params": list(self.params),
                "treatments": list(self.treatments),
                "theta0": np.asarray(self.theta0, dtype=float).tolist(),
                "role": dict(self.role),
                "source": dict(self.source),
                "diagnosed": dict(self.diagnosed),
                "inert": {k: bool(v) for k, v in self.inert.items()},
                "pairs": pairs,
                "reasons": {k: list(v) for k, v in self.reasons.items()},
                "report": self.report,
                "config": self.config,
                "calls": self.calls,
            }
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> MethodPlan:
        """The plan of a :meth:`to_dict` (after a JSON round trip too)."""
        try:
            params = tuple(str(p) for p in d["params"])
            treatments = tuple(str(t) for t in d["treatments"])
            pd_ = d["pairs"]
            pairs = None
            if pd_ is not None:
                pp, pt = tuple(pd_["params"]), tuple(pd_["treatments"])
                pairs = GradientPlan(
                    pp,
                    pt,
                    np.asarray(pd_["use_ad"], dtype=bool).reshape(len(pt), len(pp)),
                    dict(pd_["method"]),
                    {k: list(v) for k, v in pd_["reasons"].items()},
                )
            plan = cls(
                params=params,
                treatments=treatments,
                theta0=np.asarray(d["theta0"], dtype=float),
                role={str(k): str(v) for k, v in d["role"].items()},
                source={str(k): str(v) for k, v in d["source"].items()},
                diagnosed={str(k): str(v) for k, v in d["diagnosed"].items()},
                inert={str(k): bool(v) for k, v in d["inert"].items()},
                pairs=pairs,
                reasons={str(k): list(v) for k, v in d["reasons"].items()},
                report=None if d["report"] is None else dict(d["report"]),
                config=dict(d["config"]),
                calls=dict(d["calls"]),
            )
        except (KeyError, TypeError, AttributeError) as e:
            raise ValueError(f"MethodPlan.from_dict: not a MethodPlan.to_dict() ({e!r})") from e
        missing = [p for p in params if p not in plan.role or plan.role[p] not in ROLES]
        if missing or plan.theta0.shape != (len(params),):
            raise ValueError(
                f"MethodPlan.from_dict: roles or theta0 do not match the parameters {list(params)}"
            )
        return plan


# ------------------------------------------------------------------ plan_methods
def _specs(space: Any) -> list[Any] | None:
    """The :class:`~agrijax.calib.space.ParamSpec` of every name of ``space`` (theta order), or
    ``None`` for a space that does not expose them."""
    from agrijax.calib.space import ParamBlocks, ParamSpace

    if isinstance(space, ParamSpace):
        return list(space.specs)
    if isinstance(space, ParamBlocks):
        out = list(space.shared.specs)
        for sp in space.per_site.values():
            out += list(sp.specs)
        return out
    return None


def _names_of(seq: Any, what: str) -> tuple[str, ...]:
    if isinstance(seq, str):
        raise TypeError(f"{what}: a sequence of names, not a string ({seq!r})")
    return tuple(str(n) for n in seq)


def _loss_of(
    problem: Any, space: Any, theta0: Any, treatments: Any
) -> tuple[Callable[[Any], Any], tuple[str, ...], Any, Any, bool]:
    """``(f, treatment names, space, theta0, whether the loss's shape still has to be checked)``."""
    from agrijax.calib.objective import Objective

    fit_mod = sys.modules.get("agrijax.calib.fit")  # a CultivarProblem exists only once fit is imported
    if fit_mod is not None and isinstance(problem, fit_mod.CultivarProblem):
        raise TypeError(
            "plan_methods: pass a cultivar problem's per-treatment loss as a function theta -> [B] with "
            "treatments= (e.g. problem.jax_loss(obs, winv) on the calibration treatments), its space "
            "and theta0"
        )
    if space is None or theta0 is None:
        raise ValueError("plan_methods: space and theta0 are required for this problem")
    if isinstance(problem, Objective):
        if treatments is not None:
            raise ValueError("treatments=: only for a plain function (an Objective names its treatments)")
        th = jnp.asarray(np.asarray(theta0, dtype=float))
        names: list[str] = []
        for g in problem.groups:
            shape = jax.eval_shape(lambda t, g=g: jnp.atleast_1d(g.treatment_losses(t)), th).shape
            names.extend(g.treatment_names(int(shape[0])))
        return problem.treatment_losses, tuple(names), space, theta0, False
    if callable(problem):
        if treatments is None:
            raise ValueError("treatments=: a plain function needs the name of each of its outputs")
        return problem, _names_of(treatments, "treatments"), space, theta0, True
    raise TypeError(
        f"plan_methods: problem must be an Objective or a function theta -> [B], got {type(problem).__name__}"
    )


class _CompileClock:
    """Seconds JAX spends tracing, lowering and compiling while the block runs."""

    def __init__(self) -> None:
        self.seconds = 0.0

    def _listen(self, event: str, duration_secs: float, **_: str | int) -> None:
        if event in _COMPILE_EVENTS:
            self.seconds += float(duration_secs)

    def __enter__(self) -> _CompileClock:
        jax.monitoring.register_event_duration_secs_listener(self._listen)
        return self

    def __exit__(self, *exc: Any) -> None:
        jax.monitoring.unregister_event_duration_listener(self._listen)


def plan_methods(
    problem: Any,
    space: Any = None,
    theta0: Any = None,
    *,
    treatments: Sequence[str] | None = None,
    derivative_free: Sequence[str] | None = None,
    allow_gradient: Sequence[str] = (),
    diagnose: bool = True,
    trust: TrustConfig | None = None,
) -> MethodPlan:
    """The :class:`MethodPlan` of ``problem`` at ``theta0`` (module docstring).

    ``problem``: an :class:`~agrijax.calib.objective.Objective` (its per-treatment losses) or a
    JAX-traceable function ``theta [n] -> [B]`` with ``treatments=`` (the name of each output), e.g.
    the function :meth:`agrijax.calib.dssat_day.DaySimulator.jax_loss` returns; a
    :class:`~agrijax.calib.fit.CultivarProblem` is passed as its loss function in that form. A
    function bound to a gradient mode other than ``exact`` (:func:`agrijax.core.grad.bind_gradient_mode`)
    is checked at level 2 on its exact counterpart; any other function is differentiated as it is.
    ``space``: a :class:`~agrijax.calib.space.ParamSpace`, :class:`~agrijax.calib.space.ParamBlocks`
    or any object with ``names``, ``lower`` and ``upper``.

    Roles, by precedence: a name among the prior labels -> ``derivative_free`` (source ``prior``);
    else, with the diagnosis, ``gradient`` when the pair is trusted on every treatment and
    ``derivative_free`` otherwise (source ``diagnosis``); else ``gradient`` (source ``assumed``).
    ``derivative_free``: the prior labels; ``None`` = the CERES-Maize phenology coefficients
    (:data:`agrijax.calib.ceres.CERES_DERIVATIVE_FREE`) wherever the space holds them with the CERES
    specs (:data:`agrijax.calib.ceres.CERES_SPECS`, compared spec for spec, not by name); ``()`` = no
    label. ``allow_gradient``: names taken out of the prior labels, which the diagnosis then decides
    (as in :func:`agrijax.calib.ceres.ceres_gradient_plan`). ``diagnose=False``: no trust report (no
    derivative call, one forward call at ``theta0``), roles from the labels alone. ``trust``: the
    report's settings, default ``TrustConfig(n_scan=TRUST_SCAN)`` (the report's own default of 41
    scan points misses narrow jumps).

    The diagnosis needs float64 (its smallest central differences are 1e-6 of the bound width):
    without ``jax_enable_x64`` it raises ``RuntimeError``. A ``theta0`` outside or on its bounds
    gives a :class:`CalibrationWarning` and is diagnosed all the same; a parameter that moves no
    treatment's loss within the scan, and a non-finite output on the scans, also warn."""
    t_start = time.perf_counter()
    df_given = None if derivative_free is None else _names_of(derivative_free, "derivative_free")
    allow = _names_of(allow_gradient, "allow_gradient")
    if trust is not None and not isinstance(trust, TrustConfig):
        raise TypeError("trust: a TrustConfig")
    cfg = TrustConfig(n_scan=TRUST_SCAN) if trust is None else trust
    if diagnose and not jax.config.jax_enable_x64:
        raise RuntimeError(
            "plan_methods runs the trust diagnosis in 64-bit (its central differences are 1e-6 of the bound "
            "width): call jax.config.update('jax_enable_x64', True) before creating arrays, or pass "
            "diagnose=False"
        )
    f, tnames, space, theta0, check_shape = _loss_of(problem, space, theta0, treatments)
    pnames = tuple(str(n) for n in space.names)
    lower = np.asarray(space.lower, dtype=float)
    upper = np.asarray(space.upper, dtype=float)
    x0 = np.asarray(theta0, dtype=float)
    if x0.shape != (len(pnames),):
        raise ValueError(f"theta0: shape {x0.shape}, expected ({len(pnames)},) for {list(pnames)}")
    if not np.all(np.isfinite(x0)):
        bad_x = [n for n, v in zip(pnames, x0, strict=True) if not np.isfinite(v)]
        raise ValueError(f"theta0: non-finite values for {bad_x}")
    edge = [n for i, n in enumerate(pnames) if not lower[i] < x0[i] < upper[i]]
    if edge:
        warnings.warn(
            f"theta0 outside or on the bounds for {edge}: diagnosed all the same",
            CalibrationWarning,
            stacklevel=2,
        )
    # ---------------------------------------------------------------- treatment names and prior labels
    if check_shape:
        shape = jax.eval_shape(f, jnp.asarray(x0)).shape
        if shape != (len(tnames),):
            raise ValueError(
                f"treatments: {len(tnames)} names for a loss of shape {shape} (one value per treatment)"
            )
    for what, names in (("derivative_free", df_given or ()), ("allow_gradient", allow)):
        unknown = [n for n in names if n not in pnames]
        if unknown:
            raise ValueError(f"{what}: {unknown} not in the space {list(pnames)}")
    if df_given is not None and set(df_given) & set(allow):
        raise ValueError(f"derivative_free and allow_gradient both name {sorted(set(df_given) & set(allow))}")
    if df_given is None:
        specs = _specs(space)
        prior_all: tuple[str, ...] = ()
        if specs is not None:
            from agrijax.calib.ceres import CERES_DERIVATIVE_FREE, CERES_SPECS

            prior_all = tuple(
                n
                for n, s in zip(pnames, specs, strict=True)
                if s.name in CERES_DERIVATIVE_FREE and s == CERES_SPECS[s.name]
            )
    else:
        prior_all = df_given
    prior = tuple(n for n in prior_all if n not in set(allow))
    released = tuple(n for n in prior_all if n in set(allow))
    # ---------------------------------------------------------------- the diagnosis
    rep: dict[str, Any] | None = None
    pairs: GradientPlan | None = None
    calls: dict[str, Any]
    with _CompileClock() as clock:
        # the input check first: a non-finite loss at theta0 is refused before the diagnosis compiles
        y0 = np.asarray(jax.jit(f)(jnp.asarray(x0)), dtype=float)
        bad = [t for t, v in zip(tnames, np.atleast_1d(y0), strict=True) if not np.isfinite(v)]
        if bad:
            raise ValueError(f"the loss at theta0 is not finite on the treatments {bad}")
        calls = {"forward": 1, "jvp": 0, "jacobian": 0}
        if diagnose:
            rep = trust_report(f, x0, lower, upper, pnames, tnames, cfg, diagnostics=False)
            pairs = gradient_plan(rep)
            exact = counterparts(f)[0] is not None
            n_steps = len(cfg.fd_small_steps) + len(cfg.fd_large_steps)
            # the report evaluates theta0 again (its own FD centre)
            calls = {
                "forward": 2 + 2 * len(pnames) * n_steps,
                "jvp": len(pnames) * cfg.n_scan,
                "jacobian": 1 + int(exact),
            }
    calls.update(wall_s=time.perf_counter() - t_start, compile_s=clock.seconds)
    # ---------------------------------------------------------------- roles
    role: dict[str, str] = {}
    source: dict[str, str] = {}
    diagnosed: dict[str, str] = {}
    inert: dict[str, bool] = {}
    reasons: dict[str, list[str]] = {}
    nonfinite: list[str] = []
    for i, p in enumerate(pnames):
        why: list[str] = []
        if rep is not None and pairs is not None:
            outs = rep["params"][p]["outputs"]
            diagnosed[p] = ROLE_GRADIENT if bool(pairs.use_ad[:, i].all()) else ROLE_DERIVATIVE_FREE
            inert[p] = all(o["class"] == "inert" for o in outs.values())
            if any(int(o["level"]) == 0 for o in outs.values()):
                nonfinite.append(p)
            why += list(pairs.reasons.get(p, []))
        else:
            diagnosed[p] = NOT_RUN
            inert[p] = False
        if p in prior:
            role[p], source[p] = ROLE_DERIVATIVE_FREE, SOURCE_PRIOR
            why.append("derivative-free by default" if df_given is None else "derivative-free by label")
        elif rep is not None:
            role[p], source[p] = diagnosed[p], SOURCE_DIAGNOSIS
            if p in released:
                why.append("released from the default by allow_gradient")
        else:
            role[p], source[p] = ROLE_GRADIENT, SOURCE_ASSUMED
        if why:
            reasons[p] = why
    dead = [p for p in pnames if inert[p]]
    if dead:
        warnings.warn(
            f"inert on every treatment within the scan (the data do not constrain them): {dead}",
            CalibrationWarning,
            stacklevel=2,
        )
    if nonfinite:
        warnings.warn(
            f"non-finite loss on the trust scans of {nonfinite} (untrusted)", CalibrationWarning, stacklevel=2
        )
    config = {
        "trust": asdict(cfg),
        "derivative_free": list(prior),
        "prior": "default" if df_given is None else "given",
        "allow_gradient": list(allow),
        "gradient_mode": getattr(f, "mode", None),
        "x64": bool(jax.config.jax_enable_x64),
        "diagnose": bool(diagnose),
    }
    return MethodPlan(
        params=pnames,
        treatments=tuple(tnames),
        theta0=x0.copy(),
        role=role,
        source=source,
        diagnosed=diagnosed,
        inert=inert,
        pairs=pairs,
        reasons=reasons,
        report=rep,
        config=_jsonable(config),
        calls=calls,
    )
