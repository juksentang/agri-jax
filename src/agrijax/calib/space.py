"""Parameter spaces: named, bounded model parameters and their unconstrained coordinates.

A calibration works on an unconstrained vector ``z`` (what the optimiser moves); the model sees
the physical values ``theta`` inside the declared bounds. :class:`ParamSpec` names one parameter
by its dotted path in the parameter tree (``"cultivar.p1"``, :func:`agrijax.core.state.set_path`),
with its unit, bounds and the source of the bounds; :class:`ParamSpace` maps ``z <-> theta`` with
a scaled logistic transform (``theta = lo + (hi - lo) * sigmoid(z)``, the default), a log
transform (positive, unbounded above) or the identity, and writes ``theta`` into a parameter
tree.

Every parameter also declares its **valid domain** (:class:`Domain`): where the model's equations
are defined for it, e.g. ``(0, inf)`` for a coefficient that divides. The domain is a property of
the model, the bounds are a choice of the calibration (reference-file ``MINIMA`` / ``MAXIMA``,
documented ranges); a spec whose bounds leave the domain is rejected, so an optimiser moving ``z``
can never reach an invalid value (the logistic transform stays inside the bounds). ``index``
selects one element (the last axis) of an array leaf, e.g. one soil layer.

**Ordered chains** (:class:`OrderedChain`) calibrate parameters that must stay ordered, such as a
soil layer's ``LL < DUL < SAT``, jointly: the first member has the usual logistic transform; each
next member ``k`` is mapped into ``[max(lower_k, theta_{k-1} + gap), upper_k]``, i.e.
``theta_k = b_k + (upper_k - b_k) sigmoid(z_k)`` with ``b_k = max(lower_k, theta_{k-1} + gap)``.
The map is continuous, strictly increasing in each ``z_k`` and invertible given the earlier
members; every ``z`` gives ``theta_k >= theta_{k-1} + gap`` (``gap > 0``) and ``theta_k`` inside
its own bounds.

Everything here is a pure JAX function of arrays, so ``jax.grad`` / ``jax.vmap`` go through it.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from agrijax.core.coefficients import numerical_guard
from agrijax.core.state import get_path, set_path

__all__ = [
    "UNBOUNDED",
    "Domain",
    "OrderedChain",
    "ParamSpace",
    "ParamSpec",
    "Transform",
    "non_negative",
    "positive",
]

Transform = Literal["logit", "log", "none"]

#: the logistic transform keeps ``theta`` this fraction of the bound width away from either bound
#: when mapping back to ``z`` (``logit(0)`` is infinite)
_EDGE = numerical_guard(
    "calib.logit_edge", 1e-9, "fraction of the bound width kept from each bound in the inverse logit"
)
#: tolerance of :meth:`ParamSpace.violations`, in units in the last place of theta's precision
_ULP_TOL = numerical_guard("calib.violations_ulp", 4.0, "ulps of rounding allowed at a bound or chain gap")
#: floor of a log-transformed value in the inverse transform (``log(0)`` is infinite)
_LOG_FLOOR = numerical_guard("calib.log_floor", 1e-300, "floor of a log-transformed value")


@dataclass(frozen=True)
class Domain:
    """Valid domain of a parameter: the interval where the model's equations are defined for it
    (``lower_open`` / ``upper_open``: the end itself is excluded), with the reason (``source``:
    the equation that needs it, file:line)."""

    lower: float = -math.inf
    upper: float = math.inf
    lower_open: bool = False
    upper_open: bool = False
    source: str = ""

    def __post_init__(self) -> None:
        if not self.lower < self.upper:
            raise ValueError(f"empty domain [{self.lower}, {self.upper}]")

    def contains(self, lo: float, hi: float) -> bool:
        """``[lo, hi]`` lies inside the domain."""
        ok_lo = lo > self.lower if self.lower_open else lo >= self.lower
        ok_hi = hi < self.upper if self.upper_open else hi <= self.upper
        return bool(ok_lo and ok_hi)

    def __str__(self) -> str:
        lo = "(" if self.lower_open else "["
        hi = ")" if self.upper_open else "]"
        return f"{lo}{self.lower:g}, {self.upper:g}{hi}"


#: no restriction (the default of :class:`ParamSpec`)
UNBOUNDED = Domain()


def positive(source: str) -> Domain:
    """``(0, inf)``: e.g. a divisor, a rate constant."""
    return Domain(0.0, math.inf, lower_open=True, source=source)


def non_negative(source: str) -> Domain:
    """``[0, inf)``: e.g. a sensitivity, a thermal-time duration that may be 0."""
    return Domain(0.0, math.inf, source=source)


@dataclass(frozen=True)
class ParamSpec:
    """One calibrated parameter.

    ``path`` is the dotted path of the leaf in the parameter tree, ``lower`` / ``upper`` the
    bounds (physical units), ``bounds_source`` where the bounds come from (a reference file and
    line, or "harness choice" with the reason), ``transform`` how ``z`` maps to ``theta``.
    """

    path: str
    lower: float
    upper: float
    unit: str = "-"
    bounds_source: str = ""
    transform: Transform = "logit"
    label: str = ""
    domain: Domain = UNBOUNDED
    index: int | None = None

    def __post_init__(self) -> None:
        if not (np.isfinite(self.lower) and np.isfinite(self.upper)) and self.transform == "logit":
            raise ValueError(f"{self.path}: the logit transform needs finite bounds")
        if self.lower >= self.upper:
            raise ValueError(f"{self.path}: lower bound {self.lower} >= upper bound {self.upper}")
        if self.transform == "log" and self.lower < 0.0:
            raise ValueError(f"{self.path}: the log transform needs a non-negative lower bound")
        # the values the transform can produce: [lower, upper] (logit), (lower, inf) (log), any (none)
        if self.transform == "none" and self.domain != UNBOUNDED:
            raise ValueError(
                f"{self.path}: the identity transform cannot keep theta inside the domain {self.domain}"
            )
        reach_hi = math.inf if self.transform == "log" else self.upper
        lo_ok = self.domain.lower < self.lower or (
            self.lower == self.domain.lower and (not self.domain.lower_open or self.transform == "log")
        )
        hi_ok = reach_hi < self.domain.upper or (reach_hi == self.domain.upper and not self.domain.upper_open)
        if self.transform != "none" and not (lo_ok and hi_ok):
            raise ValueError(
                f"{self.path}: bounds [{self.lower}, {self.upper}] ({self.transform}) leave the valid domain "
                f"{self.domain} ({self.domain.source})"
            )

    @property
    def name(self) -> str:
        """Short name (the label, else the last path component, with the index)."""
        if self.label:
            return self.label
        base = self.path.rsplit(".", 1)[-1]
        return base if self.index is None else f"{base}[{self.index}]"

    @property
    def key(self) -> tuple[str, int | None]:
        """``(path, index)``: what the spec writes."""
        return (self.path, self.index)


@dataclass(frozen=True)
class OrderedChain:
    """Parameters (by name) that must satisfy ``theta_0 < theta_1 < ...`` with at least ``gap``
    between neighbours (units of the parameters), calibrated with the joint transform of the
    module docstring; ``source`` says where the order comes from."""

    names: tuple[str, ...]
    gap: float
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(self.names))
        if len(self.names) < 2:
            raise ValueError("an ordered chain needs at least two parameters")
        if not self.gap > 0.0:
            raise ValueError(f"ordered chain {self.names}: the gap must be > 0 (got {self.gap})")


@dataclass(frozen=True)
class ParamSpace:
    """An ordered set of :class:`ParamSpec`; ``theta[i]`` and ``z[i]`` belong to ``specs[i]``."""

    specs: tuple[ParamSpec, ...]
    ordered: tuple[OrderedChain, ...] = field(default=())
    #: ``_prev[i]``: ``(index of the previous chain member, gap)`` or ``None``
    _prev: tuple[tuple[int, float] | None, ...] = field(default=(), repr=False, compare=False)

    def __init__(self, specs: Sequence[ParamSpec], ordered: Sequence[OrderedChain] = ()) -> None:
        keys = [s.key for s in specs]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate parameter paths: {[s.path for s in specs]}")
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate parameter names: {names}")
        specs = tuple(specs)
        prev: list[tuple[int, float] | None] = [None] * len(specs)
        seen: set[str] = set()
        for ch in ordered:
            missing = [n for n in ch.names if n not in names]
            if missing:
                raise ValueError(f"ordered chain {ch.names}: unknown parameters {missing}")
            if seen & set(ch.names):
                raise ValueError(f"ordered chain {ch.names}: a parameter is in two chains")
            seen |= set(ch.names)
            idx = [names.index(n) for n in ch.names]
            if idx != sorted(idx):
                raise ValueError(f"ordered chain {ch.names}: members must appear in the space in chain order")
            for a, b in itertools.pairwise(idx):
                sa, sb = specs[a], specs[b]
                if sa.transform != "logit" or sb.transform != "logit":
                    raise ValueError(f"ordered chain {ch.names}: members need the logit transform")
                if not sa.upper + ch.gap < sb.upper:
                    raise ValueError(
                        f"ordered chain {ch.names}: {sb.name} needs upper > upper({sa.name}) + gap "
                        f"({sb.upper} <= {sa.upper} + {ch.gap})"
                    )
                prev[b] = (a, float(ch.gap))
        object.__setattr__(self, "specs", specs)
        object.__setattr__(self, "ordered", tuple(ordered))
        object.__setattr__(self, "_prev", tuple(prev))

    # ------------------------------------------------------------------ description
    @property
    def n(self) -> int:
        return len(self.specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.specs)

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(s.path for s in self.specs)

    @property
    def lower(self) -> np.ndarray:
        return np.asarray([s.lower for s in self.specs], dtype=float)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray([s.upper for s in self.specs], dtype=float)

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower

    def table(self) -> list[dict[str, Any]]:
        """One row per parameter (for reports)."""
        return [
            {
                "name": s.name,
                "path": s.path,
                "unit": s.unit,
                "lower": s.lower,
                "upper": s.upper,
                "transform": s.transform,
                "bounds_source": s.bounds_source,
                "index": s.index,
                "domain": str(s.domain),
                "domain_source": s.domain.source,
            }
            for s in self.specs
        ]

    # ------------------------------------------------------------------ transforms
    def _base(self, i: int, cols: Sequence[Array]) -> Array | float:
        """Lower end of the logistic box of parameter ``i``: its lower bound, or for an ordered
        chain member ``max(lower_i, theta_prev + gap)``."""
        s = self.specs[i]
        pv = self._prev[i]
        if pv is None:
            return s.lower
        return jnp.maximum(s.lower, cols[pv[0]] + pv[1])

    def to_physical(self, z: Array) -> Array:
        """``theta`` of an unconstrained ``z`` (last axis = parameters; any leading batch axes)."""
        z = jnp.asarray(z)
        cols: list[Array] = []
        for i, s in enumerate(self.specs):
            zi = z[..., i]
            if s.transform == "logit":
                base = self._base(i, cols)
                # clipped: at a saturated z the sum can round 1 ulp past upper (clip keeps an
                # interior derivative of 1 and a finite one at the ends)
                cols.append(jnp.clip(base + (s.upper - base) * jax.nn.sigmoid(zi), base, s.upper))
            elif s.transform == "log":
                cols.append(s.lower + jnp.exp(zi))
            else:
                cols.append(zi)
        return jnp.stack(cols, axis=-1)

    def to_unconstrained(self, theta: Any) -> Array:
        """``z`` of a physical ``theta`` (clamped just inside the bounds for the logit)."""
        theta = jnp.asarray(theta)
        cols = []
        phys = [theta[..., i] for i in range(self.n)]
        for i, s in enumerate(self.specs):
            ti = theta[..., i]
            if s.transform == "logit":
                base = self._base(i, phys)
                u = (ti - base) / (s.upper - base)
                u = jnp.clip(u, _EDGE, 1.0 - _EDGE)
                cols.append(jnp.log(u) - jnp.log1p(-u))
            elif s.transform == "log":
                cols.append(jnp.log(jnp.maximum(ti - s.lower, _LOG_FLOOR)))
            else:
                cols.append(ti)
        return jnp.stack(cols, axis=-1)

    def clip(self, theta: Any) -> np.ndarray:
        """``theta`` clipped into the bounds (NumPy, for building starting points; the order of
        a chain is not restored: map through ``to_physical(to_unconstrained(theta))`` for that)."""
        return np.clip(np.asarray(theta, dtype=float), self.lower, self.upper)

    def violations(self, theta: Any, dtype: Any = None) -> list[str]:
        """What is wrong with a physical ``theta`` (one vector): values outside the bounds or
        the valid domain, broken chain orders. Empty when ``theta`` is admissible.

        Bounds and chain gaps are checked with a tolerance of ``_ULP_TOL`` units in the last place
        of the precision ``theta`` was computed in (float32 values carry the float32 rounding of a
        bound): ``dtype`` if given, else ``theta``'s own dtype. A float32 result upcast to float64
        before the call gets the float64 tolerance unless ``dtype`` says float32. The open end of a
        domain is checked strictly."""
        raw = np.asarray(theta)
        dt = np.dtype(dtype) if dtype is not None else raw.dtype
        eps = float(np.finfo(dt if np.issubdtype(dt, np.floating) else np.float64).eps)
        t = raw.astype(float)
        out = []
        for i, s in enumerate(self.specs):
            tol = _ULP_TOL * eps * max(1.0, abs(s.lower), abs(s.upper))
            if not s.lower - tol <= t[i] <= s.upper + tol and s.transform == "logit":
                out.append(f"{s.name} = {t[i]:g} outside the bounds [{s.lower:g}, {s.upper:g}]")
            if not s.domain.contains(t[i], t[i]):
                out.append(f"{s.name} = {t[i]:g} outside the valid domain {s.domain}")
            pv = self._prev[i]
            if pv is not None and not t[i] >= t[pv[0]] + pv[1] - tol:
                out.append(
                    f"{s.name} = {t[i]:g} not >= {self.specs[pv[0]].name} + {pv[1]:g} = {t[pv[0]] + pv[1]:g}"
                )
        return out

    # ------------------------------------------------------------------ parameter trees
    def get(self, params: Any) -> Array:
        """The current values of the parameters in ``params`` as a vector."""
        vals = []
        for s in self.specs:
            v = jnp.asarray(get_path(params, s.path), dtype=float)
            vals.append(v if s.index is None else v[..., s.index])
        return jnp.stack(vals)

    def apply(self, params: Any, theta: Array) -> Any:
        """``params`` with every path set to ``theta[i]`` (dtype of the existing leaf kept; with
        an ``index``, only that element of the last axis)."""
        out = params
        for i, s in enumerate(self.specs):
            old = jnp.asarray(get_path(out, s.path))
            val = jnp.asarray(theta[i], dtype=old.dtype)
            new = val * jnp.ones_like(old) if s.index is None else old.at[..., s.index].set(val)
            out = set_path(out, s.path, new)
        return out

    def relative_error(self, theta: Any, truth: Any) -> np.ndarray:
        """``|theta - truth| / |truth|`` per parameter (NumPy; any leading batch axes)."""
        t = np.asarray(truth, dtype=float)
        return np.abs(np.asarray(theta, dtype=float) - t) / np.maximum(np.abs(t), _LOG_FLOOR)

    def width_error(self, theta: Any, truth: Any) -> np.ndarray:
        """``|theta - truth| / (upper - lower)`` per parameter (error as a fraction of the box)."""
        return np.abs(np.asarray(theta, dtype=float) - np.asarray(truth, dtype=float)) / self.width
