"""Daily resynchronised comparison against a reference model (plan A12, level 2).

A free-running comparison mixes every error of a coupled model into one trajectory: a small
infiltration error on day 10 is still visible in the soil water of day 200. In a
*resynchronised* run, chosen state subtrees are overwritten each morning with the reference
model's value of that morning, so each day's residual measures only what the model gets wrong
within that day, given the reference's state. This module gives the pieces:

* :class:`DailyReference` -- reference values per date (``[n_day, ...]`` arrays and the dates),
  built from the daily dump tables of :mod:`agrijax.port.dumps` (for example the ``PHYSCL`` entry
  state of RZWQM2) or from any other per-day arrays;
* :func:`resync_scan` -- a ``lax.scan`` over days that, before each day's step, replaces the
  leaves picked by ``where`` with the reference values of that morning (``jnp.where`` on a
  per-day mask; days without reference keep the model's own state). Everything is traced, so it
  jits and differentiates like :func:`agrijax.core.runtime.run`; gradients do not flow into a
  replaced leaf through the previous day (the overwrite cuts the chain, which is the point);
* :class:`Tolerance`, :func:`compare_daily`, :func:`save_tolerances` / :func:`load_tolerances` --
  per-quantity ``(rel, abs)`` tolerance pairs stored next to the golden data, and the daily
  comparison that reports the worst day and every failing day;
* :func:`provenance` -- the record kept with every reference data set:
  ``{model, version, binary_sha256, flags, case_hash}``.

The runtime owns the day loop (the three process rules bind process functions; this is a
runtime-level helper like ``run``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from agrijax.port.dumps import DailyTable, load_table

__all__ = [
    "DailyComparison",
    "DailyReference",
    "Tolerance",
    "align_dates",
    "compare_daily",
    "file_sha256",
    "load_tolerances",
    "morning_values",
    "provenance",
    "resync_scan",
    "save_tolerances",
]


# ---------------------------------------------------------------------------------------- reference
@dataclass
class DailyReference:
    """Reference quantities per date: ``values[name]`` has the date axis first."""

    dates: np.ndarray
    values: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dates = np.asarray(self.dates, dtype=np.int64)
        if self.dates.ndim != 1:
            raise ValueError("dates must be one-dimensional")
        if np.any(np.diff(self.dates) <= 0):
            raise ValueError("dates must be strictly increasing")
        for k, v in self.values.items():
            if np.shape(v)[:1] != self.dates.shape:
                raise ValueError(f"{k}: leading axis {np.shape(v)[:1]} differs from {self.dates.shape}")

    @classmethod
    def from_table(
        cls, table: DailyTable, names: Mapping[str, str] | Sequence[str] | None = None
    ) -> DailyReference:
        """From a dump table; ``names`` maps reference names to table variables (or lists them)."""
        if names is None:
            sel = {k: k for k in table.values}
        elif isinstance(names, Mapping):
            sel = dict(names)
        else:
            sel = {k: k for k in names}
        vals = {k: np.asarray(table.values[v.upper()]) for k, v in sel.items()}
        meta = {"routine": table.routine, "phase": table.phase, "which": table.which}
        return cls(np.asarray(table.date), vals, meta)

    @classmethod
    def from_npz(
        cls, path: str | Path, names: Mapping[str, str] | Sequence[str] | None = None
    ) -> DailyReference:
        """From a table file written by :func:`agrijax.port.dumps.save_table`."""
        tab, meta = load_table(path)
        ref = cls.from_table(tab, names)
        ref.meta.update(meta)
        return ref

    def at(self, dates: Sequence[int] | np.ndarray, name: str, fill: float = np.nan) -> np.ndarray:
        """``values[name]`` on ``dates`` (rows without a reference date are ``fill``)."""
        idx, ok = align_dates(dates, self.dates)
        v = np.asarray(self.values[name])
        out = np.full((len(idx), *v.shape[1:]), fill, dtype=np.result_type(v.dtype, np.float64))
        out[ok] = v[idx[ok]]
        return out


def align_dates(
    dates: Sequence[int] | np.ndarray, ref_dates: Sequence[int] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each of ``dates``: the row in ``ref_dates`` and whether it exists (``(idx, ok)``)."""
    d = np.asarray(dates, dtype=np.int64)
    r = np.asarray(ref_dates, dtype=np.int64)
    if r.size == 0:
        return np.zeros(d.shape, np.int64), np.zeros(d.shape, bool)
    pos = np.clip(np.searchsorted(r, d), 0, r.size - 1)
    ok = r[pos] == d
    return np.where(ok, pos, 0), ok


def morning_values(
    ref: DailyReference,
    dates: Sequence[int] | np.ndarray,
    names: Sequence[str],
    *,
    lag: int = 0,
    dtype: Any = None,
) -> tuple[tuple[jax.Array, ...], jax.Array]:
    """Per-day overwrite values and mask for :func:`resync_scan`.

    Row ``t`` of each output is the reference value on ``dates[t - lag]``: with an *entry* table
    (the reference state at the start of its day) use ``lag=0``; with an *end-of-day* table use
    ``lag=1`` (yesterday's end is this morning). Rows without a reference value have mask
    ``False`` and hold zeros (never used, finite)."""
    d = np.asarray(dates, dtype=np.int64)
    src = np.concatenate([np.full(lag, -1, np.int64), d[: len(d) - lag]]) if lag else d
    idx, ok = align_dates(src, ref.dates)
    if lag:
        ok[:lag] = False
    out = []
    for nm in names:
        v = np.asarray(ref.values[nm])
        x = np.where(ok.reshape((-1,) + (1,) * (v.ndim - 1)), v[idx], 0).astype(dtype or v.dtype)
        out.append(jnp.asarray(x))
    return tuple(out), jnp.asarray(ok)


# ---------------------------------------------------------------------------------------- the scan
def resync_scan(
    step: Callable[[Any, Any], tuple[Any, Any]],
    state0: Any,
    xs: Any,
    *,
    where: Callable[[Any], Any],
    values: Sequence[Any],
    mask: Any,
) -> tuple[Any, Any]:
    """``lax.scan`` of ``step(state, x_t) -> (state, out_t)`` with a morning overwrite.

    Before day ``t``: ``leaf_i <- where(mask[t], values[i][t], leaf_i)`` for every leaf of the
    tuple ``where(state)`` (an :func:`equinox.tree_at` selector), cast to the leaf's dtype and
    shape. ``values`` and ``mask`` have the day axis first. Returns ``(final_state, outs)``; the
    outputs of day ``t`` are computed from the overwritten morning state.

    A model step from :meth:`agrijax.core.model.Model.compile` takes ``(state, params, forcing)``;
    pass ``lambda s, f: step(s, params, f)``.
    """
    import equinox as eqx

    vals = tuple(values)
    m = jnp.asarray(mask, dtype=bool)
    leaves0 = where(state0)
    leaves0 = leaves0 if isinstance(leaves0, tuple) else (leaves0,)
    if len(leaves0) != len(vals):
        raise ValueError(f"where() selects {len(leaves0)} leaves but {len(vals)} value arrays were given")
    n = jax.tree_util.tree_leaves(xs)[0].shape[0] if jax.tree_util.tree_leaves(xs) else m.shape[0]
    for i, (lf, v) in enumerate(zip(leaves0, vals, strict=True)):
        if jnp.shape(v)[0] != n or jnp.shape(v)[1:] != jnp.shape(lf):
            raise ValueError(f"values[{i}] has shape {jnp.shape(v)}, expected ({n}, *{jnp.shape(lf)})")
    if m.shape != (n,):
        raise ValueError(f"mask has shape {m.shape}, expected ({n},)")

    def body(state: Any, inp: tuple[Any, tuple[Any, ...], Any]) -> tuple[Any, Any]:
        x_t, v_t, m_t = inp
        cur = where(state)
        cur = cur if isinstance(cur, tuple) else (cur,)
        new = tuple(jnp.where(m_t, jnp.asarray(v, dtype=c.dtype), c) for c, v in zip(cur, v_t, strict=True))
        state = eqx.tree_at(lambda s: _as_tuple(where(s)), state, new)
        return step(state, x_t)

    return lax.scan(body, state0, (xs, vals, m))


def _as_tuple(x: Any) -> tuple[Any, ...]:
    return x if isinstance(x, tuple) else (x,)


# ---------------------------------------------------------------------------------------- tolerances
@dataclass(frozen=True)
class Tolerance:
    """Pass when ``|sim - ref| <= abs + rel * |ref|`` (numpy ``allclose`` semantics)."""

    rel: float = 0.0
    abs: float = 0.0
    note: str = ""

    def __post_init__(self) -> None:
        if self.rel < 0 or self.abs < 0 or not (np.isfinite(self.rel) and np.isfinite(self.abs)):
            raise ValueError(f"tolerances must be finite and non-negative: rel={self.rel}, abs={self.abs}")


def save_tolerances(
    path: str | Path, tols: Mapping[str, Tolerance], *, meta: Mapping[str, Any] | None = None
) -> Path:
    """Write ``{"meta": ..., "tolerances": {name: {"rel", "abs", "note"}}}`` as JSON."""
    body = {
        "meta": dict(meta or {}),
        "tolerances": {k: {"rel": t.rel, "abs": t.abs, "note": t.note} for k, t in sorted(tols.items())},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body, indent=1) + "\n")
    return p


def load_tolerances(path: str | Path) -> dict[str, Tolerance]:
    """Read a file written by :func:`save_tolerances`."""
    body = json.loads(Path(path).read_text())
    tol = body["tolerances"]
    return {k: Tolerance(float(v["rel"]), float(v["abs"]), str(v.get("note", ""))) for k, v in tol.items()}


@dataclass
class DailyComparison:
    """Result of :func:`compare_daily` for one quantity."""

    name: str
    tolerance: Tolerance
    n_days: int
    max_abs: float
    max_rel: float
    worst_date: int | None
    fail_dates: list[int]

    @property
    def passed(self) -> bool:
        return not self.fail_dates

    def __str__(self) -> str:
        s = "ok" if self.passed else f"FAIL on {len(self.fail_dates)} days (first {self.fail_dates[:5]})"
        return (
            f"{self.name}: {s}; max |d| {self.max_abs:.3e}, max rel {self.max_rel:.3e} "
            f"(worst {self.worst_date}) over {self.n_days} days, "
            f"tol rel {self.tolerance.rel:g} abs {self.tolerance.abs:g}"
        )


def compare_daily(
    name: str,
    sim: Any,
    ref: Any,
    dates: Sequence[int] | np.ndarray,
    tol: Tolerance,
    *,
    mask: Any = None,
) -> DailyComparison:
    """Compare ``sim`` and ``ref`` (day axis first) day by day; NaN in ``ref`` means no reference.

    A day fails when any element exceeds ``tol``; a NaN in ``sim`` where ``ref`` is finite fails.
    ``max_rel`` is over elements with ``ref != 0``."""
    s = np.asarray(sim, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    d = np.asarray(dates, dtype=np.int64)
    if s.shape != r.shape or s.shape[:1] != d.shape:
        raise ValueError(f"{name}: shapes sim {s.shape}, ref {r.shape}, dates {d.shape} do not match")
    have = np.isfinite(r)
    if mask is not None:
        have &= np.asarray(mask, bool).reshape((-1,) + (1,) * (r.ndim - 1))
    diff = np.where(have, np.abs(s - r), 0.0)
    bad = have & ~(diff <= tol.abs + tol.rel * np.abs(r))  # NaN in sim fails here
    axes = tuple(range(1, r.ndim))
    day_bad = bad.any(axis=axes) if axes else bad
    day_have = have.any(axis=axes) if axes else have
    diff = np.where(np.isfinite(diff), diff, np.inf)
    rel = np.where(have & (r != 0), diff / np.where(r != 0, np.abs(r), 1.0), 0.0)
    per_day = diff.max(axis=axes) if axes else diff
    worst = int(d[int(np.argmax(np.where(day_have, per_day, -1.0)))]) if day_have.any() else None
    return DailyComparison(
        name=name,
        tolerance=tol,
        n_days=int(day_have.sum()),
        max_abs=float(diff.max()) if diff.size else 0.0,
        max_rel=float(rel.max()) if rel.size else 0.0,
        worst_date=worst,
        fail_dates=[int(x) for x in d[day_bad]],
    )


# ---------------------------------------------------------------------------------------- provenance
def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(
    model: str,
    version: str,
    binary: str | Path,
    *,
    flags: Mapping[str, Any] | None = None,
    case_files: Sequence[str | Path] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """``{model, version, binary_sha256, flags, case_hash}`` for a reference data set.

    ``case_hash`` is the SHA-256 over the sorted ``(file name, file SHA-256)`` pairs of the input
    files that define the case (``rzwqm.dat``, ``.BRK``, FileX ...), so it does not depend on
    where the case lives."""
    pairs = sorted((Path(f).name, file_sha256(f)) for f in case_files)
    ch = hashlib.sha256(json.dumps(pairs).encode()).hexdigest() if pairs else None
    rec: dict[str, Any] = {
        "model": model,
        "version": version,
        "binary_sha256": file_sha256(binary),
        "flags": dict(flags or {}),
        "case_hash": ch,
        "case_files": [p[0] for p in pairs],
    }
    rec.update(dict(extra or {}))
    return rec
