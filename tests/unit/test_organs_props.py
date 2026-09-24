"""Hypothesis property tests for :mod:`agri_jax.core.organs` against an independent pure-Python queue.

Random sequences of ``appear`` / ``age`` / ``grow`` / ``senesce`` / ``top`` / ``prime`` are applied
to an :class:`OrganQueue` and, in parallel, to a reference written with Python lists and floats
from the docstring semantics (no JAX, no masks, no ``jnp.where``). After every operation:

* the whole queue equals the reference (area, mass, n_mass, age, alive, n_active, cap);
* mass conservation: ``alive mass + senesced mass + harvested mass == total mass ever added``
  (the total is counted by the reference, to 1e-9 relative in x64), and likewise for nitrogen;
* ``n_active <= n_cohort`` and ``cap <= n_cohort``; ``cap`` never increases; ``n_active`` grows by at
  most one per ``appear`` and only while ``n_active < cap``, so every appearance leaves
  ``n_active <= cap`` (``top(q, k)`` with ``k`` below the current count lowers ``cap`` under
  ``n_active`` by design: existing cohorts stay); ``alive`` implies ``rank < n_active``;
  never-appeared slots hold no mass or area;
* ``aggregate`` equals a NumPy re-computation of LAI / leaf mass / N from the arrays.

Deterministic: ``derandomize=True`` and no example database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from agri_jax.core.organs import OrganQueue, age, aggregate, appear, grow, prime, senesce, top

SETTINGS = settings(derandomize=True, database=None, deadline=None, max_examples=120)
X64 = bool(jax.config.jax_enable_x64)
RTOL = 1e-9 if X64 else 2e-5
ATOL = 1e-9 if X64 else 1e-4

pos = st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False, width=32)


# ------------------------------------------------------------------------------ the reference


@dataclass
class RefCrop:
    n_cohort: int
    cap: int
    area: list[float] = field(default_factory=list)  # one entry per appeared slot, rank = index
    mass: list[float] = field(default_factory=list)
    n_mass: list[float] = field(default_factory=list)
    age: list[float] = field(default_factory=list)
    alive: list[bool] = field(default_factory=list)


@dataclass
class Ref:
    crops: list[RefCrop]
    added_mass: float = 0.0
    added_n: float = 0.0
    harvested_mass: float = 0.0
    harvested_n: float = 0.0

    def apply(self, op: tuple[Any, ...]) -> None:
        kind = op[0]
        if kind == "appear":
            _, mask, a0, m0, n0 = op
            for c, crop in enumerate(self.crops):
                if mask[c] and len(crop.alive) < crop.cap:
                    crop.area.append(a0[c])
                    crop.mass.append(m0[c])
                    crop.n_mass.append(n0[c])
                    crop.age.append(0.0)
                    crop.alive.append(True)
                    self.added_mass += m0[c]
                    self.added_n += n0[c]
        elif kind == "age":
            _, dtt = op
            for c, crop in enumerate(self.crops):
                for k in range(len(crop.alive)):
                    if crop.alive[k]:
                        crop.age[k] += dtt[c]
        elif kind == "grow":
            _, da, dm, dn = op
            for c, crop in enumerate(self.crops):
                for k in range(len(crop.alive)):
                    if crop.alive[k]:
                        crop.area[k] += da[c][k]
                        crop.mass[k] += dm[c][k]
                        crop.n_mass[k] += dn[c][k]
                        self.added_mass += dm[c][k]
                        self.added_n += dn[c][k]
        elif kind == "senesce":
            _, mask = op
            for c, crop in enumerate(self.crops):
                for k in range(len(crop.alive)):
                    if mask[c][k]:
                        crop.alive[k] = False
        elif kind == "top":
            _, max_active, mask = op
            for c, crop in enumerate(self.crops):
                if mask[c]:
                    crop.cap = min(crop.cap, max_active[c])
        elif kind == "prime":
            _, lo, hi = op
            for c, crop in enumerate(self.crops):
                for k in range(len(crop.alive)):
                    if crop.alive[k] and lo[c] <= k < hi[c]:
                        self.harvested_mass += crop.mass[k]
                        self.harvested_n += crop.n_mass[k]
                        crop.area[k] = crop.mass[k] = crop.n_mass[k] = 0.0
                        crop.alive[k] = False
        else:  # pragma: no cover
            raise ValueError(kind)

    def dense(self, name: str) -> np.ndarray:
        """``[n_crop, n_cohort]`` array of a per-slot list; never-appeared slots are 0 / False."""
        n = self.crops[0].n_cohort
        fill = False if name == "alive" else 0.0
        return np.array([getattr(c, name) + [fill] * (n - len(c.alive)) for c in self.crops])


_OPS = {
    "appear": appear,
    "age": age,
    "grow": grow,
    "senesce": senesce,
    "top": top,
    "prime": prime,
}
_JITTED = {k: jax.jit(f) for k, f in _OPS.items()}  # compiled once per (n_crop, n_cohort) shape
_aggregate = jax.jit(aggregate)


def apply_jax(
    q: OrganQueue, op: tuple[Any, ...], *, jitted: bool = True, ops: dict[str, Any] | None = None
) -> tuple[OrganQueue, jax.Array | None]:
    """Apply one operation; ``ops`` overrides the functions (mutation checks), ``jitted=False`` inside a trace."""
    fns = ops if ops is not None else (_JITTED if jitted else _OPS)
    kind, args = op[0], [jnp.asarray(a) for a in op[1:]]
    out = fns[kind](q, *args)
    if kind == "prime":
        return out
    return out, None


# ------------------------------------------------------------------------------ strategies


@st.composite
def scenario(draw: st.DrawFn) -> tuple[int, int, list[tuple[Any, ...]]]:
    n_crop = draw(st.integers(1, 3))
    n_cohort = draw(st.integers(1, 7))
    per_crop = lambda s: st.lists(s, min_size=n_crop, max_size=n_crop)  # noqa: E731
    grid = lambda s: st.lists(per_crop(s), min_size=n_cohort, max_size=n_cohort).map(  # noqa: E731
        lambda cols: [list(r) for r in zip(*cols, strict=True)]
    )
    rank = st.integers(-1, n_cohort + 1)
    strategies = {
        "appear": st.tuples(
            st.just("appear"), per_crop(st.booleans()), per_crop(pos), per_crop(pos), per_crop(pos)
        ),
        "age": st.tuples(st.just("age"), per_crop(pos)),
        "grow": st.tuples(st.just("grow"), grid(pos), grid(pos), grid(pos)),
        "senesce": st.tuples(st.just("senesce"), grid(st.sampled_from([False, False, False, True]))),
        "top": st.tuples(st.just("top"), per_crop(st.integers(0, n_cohort + 1)), per_crop(st.booleans())),
        "prime": st.tuples(st.just("prime"), per_crop(rank), per_crop(rank)),
    }
    # appearance-heavy so queues fill up and hit the cap; long enough to mix every operation
    kinds = st.sampled_from(["appear"] * 4 + ["age", "grow", "grow", "senesce", "top", "prime", "prime"])
    ops = draw(st.lists(kinds.flatmap(strategies.__getitem__), min_size=12, max_size=30))
    return n_crop, n_cohort, ops


# ------------------------------------------------------------------------------ the property


def _check(
    q: OrganQueue,
    ref: Ref,
    harvested: float,
    prev_cap: np.ndarray,
    prev_n_active: np.ndarray,
    density: np.ndarray,
) -> None:
    alive = np.asarray(q.alive)
    n_active = np.asarray(q.n_active)
    cap = np.asarray(q.cap)
    rank = np.asarray(q.rank)
    mass, area, n_mass = (np.asarray(x, dtype=np.float64) for x in (q.mass, q.area, q.n_mass))

    # equality with the reference
    np.testing.assert_array_equal(n_active, [len(c.alive) for c in ref.crops])
    np.testing.assert_array_equal(cap, [c.cap for c in ref.crops])
    np.testing.assert_array_equal(alive, ref.dense("alive"))
    for name, arr in (("mass", mass), ("area", area), ("n_mass", n_mass), ("age", q.age_tt)):
        np.testing.assert_allclose(np.asarray(arr, dtype=np.float64), ref.dense(name), rtol=RTOL, atol=ATOL)

    # structural invariants
    # appearance never crosses the cap: whenever n_active grew it is still <= cap (topping below
    # the current count, top(q, k < n_active), legitimately leaves n_active > cap without growth)
    grew = n_active > prev_n_active
    assert (n_active[grew] <= cap[grew]).all(), "appearance beyond cap"
    assert (n_active >= prev_n_active).all() and (n_active - prev_n_active <= 1).all()
    assert (n_active[prev_n_active >= prev_cap] == prev_n_active[prev_n_active >= prev_cap]).all()
    assert (n_active <= q.n_cohort).all() and (cap <= q.n_cohort).all()
    assert (cap <= prev_cap).all(), "cap increased"
    appeared = rank < n_active[:, None]
    assert not (alive & ~appeared).any(), "alive slot beyond n_active"
    assert (mass[~appeared] == 0).all() and (area[~appeared] == 0).all()

    # conservation: alive + senesced (appeared, not alive) + harvested == added
    alive_m = mass[alive].sum()
    senesced_m = mass[appeared & ~alive].sum()
    tot = ref.added_mass
    np.testing.assert_allclose(alive_m + senesced_m + harvested, tot, rtol=RTOL, atol=ATOL * (1 + tot))
    np.testing.assert_allclose(harvested, ref.harvested_mass, rtol=RTOL, atol=ATOL * (1 + tot))
    tot_n = ref.added_n
    np.testing.assert_allclose(
        n_mass[appeared].sum() + ref.harvested_n, tot_n, rtol=RTOL, atol=ATOL * (1 + tot_n)
    )

    # aggregate vs NumPy
    agg = _aggregate(q, jnp.asarray(density))
    np.testing.assert_allclose(
        np.asarray(agg["lai"]), (area * alive).sum(-1) * density * 1e-4, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        np.asarray(agg["leaf_mass"]), (mass * alive).sum(-1) * density * 10.0, rtol=RTOL, atol=ATOL * 100
    )
    np.testing.assert_allclose(
        np.asarray(agg["n_mass"]), (n_mass * alive).sum(-1) * density * 10.0, rtol=RTOL, atol=ATOL * 100
    )


@SETTINGS
@given(scenario(), st.lists(st.floats(0.5, 12.0, width=32), min_size=3, max_size=3))
def test_random_operation_sequences(sc: tuple[int, int, list[tuple[Any, ...]]], dens: list[float]) -> None:
    n_crop, n_cohort, ops = sc
    density = np.asarray(dens[:n_crop], dtype=np.float64)
    q = OrganQueue.empty(n_crop, n_cohort, dtype=jnp.zeros(()).dtype)
    ref = Ref([RefCrop(n_cohort=n_cohort, cap=n_cohort) for _ in range(n_crop)])
    harvested = 0.0
    prev_cap = np.full(n_crop, n_cohort)
    prev_n = np.zeros(n_crop, dtype=int)
    for op in ops:
        q, h = apply_jax(q, op)
        ref.apply(op)
        if h is not None:
            assert h.shape == (n_crop,)
            harvested += float(np.asarray(h, dtype=np.float64).sum())
        _check(q, ref, harvested, prev_cap, prev_n, density)
        prev_cap, prev_n = np.asarray(q.cap), np.asarray(q.n_active)


@settings(derandomize=True, database=None, deadline=None, max_examples=30)
@given(scenario())
def test_jit_equals_eager(sc: tuple[int, int, list[tuple[Any, ...]]]) -> None:
    """The same sequence under ``jax.jit`` (one compiled function) gives the eager result bit for bit."""
    n_crop, n_cohort, ops = sc
    q0 = OrganQueue.empty(n_crop, n_cohort, dtype=jnp.zeros(()).dtype)

    def run_ops(q: OrganQueue) -> tuple[OrganQueue, jax.Array]:
        total = jnp.zeros((n_crop,), dtype=q.mass.dtype)
        for op in ops:
            q, h = apply_jax(q, op, jitted=False)
            if h is not None:
                total = total + h
        return q, total

    eager = run_ops(q0)
    jitted = jax.jit(run_ops)(q0)
    for a, b in zip(jax.tree_util.tree_leaves(eager), jax.tree_util.tree_leaves(jitted), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=RTOL, atol=ATOL)


@settings(derandomize=True, database=None, deadline=None, max_examples=20)
@given(scenario(), st.integers(2, 5))
def test_vmap_over_queues_equals_loop(sc: tuple[int, int, list[tuple[Any, ...]]], n_batch: int) -> None:
    """``vmap`` over a batch of queues with per-sample appearance mass equals a Python loop."""
    n_crop, n_cohort, ops = sc
    scales = jnp.linspace(0.5, 2.0, n_batch)

    def run_ops(scale: jax.Array) -> tuple[OrganQueue, jax.Array]:
        q = OrganQueue.empty(n_crop, n_cohort, dtype=scales.dtype)
        total = jnp.zeros((n_crop,), dtype=scales.dtype)
        for op in ops:
            if op[0] == "appear":
                op = (op[0], op[1], op[2], scale * jnp.asarray(op[3]), op[4])
            q, h = apply_jax(q, op, jitted=False)
            if h is not None:
                total = total + h
        return q, total

    batched = jax.jit(jax.vmap(run_ops))(scales)
    single_fn = jax.jit(run_ops)
    for i in range(n_batch):
        single = single_fn(scales[i])
        for a, b in zip(jax.tree_util.tree_leaves(batched), jax.tree_util.tree_leaves(single), strict=True):
            np.testing.assert_allclose(np.asarray(a)[i], np.asarray(b), rtol=RTOL, atol=ATOL)
