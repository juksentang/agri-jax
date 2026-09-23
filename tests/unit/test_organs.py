"""OrganQueue (docs/en/02_architecture.md section 3.7): appearance, topping, priming, aggregation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agri_jax.core import OrganQueue as CoreOrganQueue
from agri_jax.core.organs import OrganQueue, age, aggregate, appear, grow, prime, senesce, top
from agri_jax.core.state import OrganQueue as StateOrganQueue
from agri_jax.core.state import field_metadata


def test_empty_shapes_and_reexports() -> None:
    q = OrganQueue.empty(2, 5)
    assert StateOrganQueue is OrganQueue and CoreOrganQueue is OrganQueue
    for name in ("rank", "area", "mass", "age_tt", "n_mass", "alive"):
        assert getattr(q, name).shape == (2, 5)
    assert q.n_active.shape == (2,) and q.cap.shape == (2,)
    assert q.alive.dtype == jnp.bool_ and jnp.issubdtype(q.rank.dtype, jnp.integer)
    np.testing.assert_array_equal(np.asarray(q.rank[1]), np.arange(5))
    assert not bool(q.alive.any()) and int(q.n_active.sum()) == 0
    assert q.n_crop == 2 and q.n_cohort == 5
    assert field_metadata(OrganQueue)["area"]["unit"] == "cm2 plant-1"
    with pytest.raises(ValueError):
        OrganQueue.empty(0, 3)


def test_appearance_up_to_n_cohort_then_noop() -> None:
    q = OrganQueue.empty(2, 3)
    mask = jnp.array([True, False])
    for k in range(3):
        q = appear(q, mask, area0=float(k + 1), mass0=0.1 * (k + 1))
        q = age(q, 10.0)
    assert q.n_active.tolist() == [3, 0]
    np.testing.assert_array_equal(np.asarray(q.alive), [[True, True, True], [False, False, False]])
    np.testing.assert_allclose(np.asarray(q.area[0]), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(q.age_tt[0]), [30.0, 20.0, 10.0])  # ageing starts at appearance
    assert float(q.age_tt[1].sum()) == 0.0
    full = appear(q, True, area0=99.0, mass0=99.0)  # queue full: no-op, no wrap-around
    assert full.n_active.tolist() == [3, 1]
    np.testing.assert_array_equal(np.asarray(full.area[0]), np.asarray(q.area[0]))
    assert float(full.area[1, 0]) == 99.0


def test_topping_stops_appearance() -> None:
    q = OrganQueue.empty(2, 6)
    q = appear(appear(q, True, 1.0, 1.0), True, 1.0, 1.0)
    q = top(q, q.n_active, mask=jnp.array([True, False]))  # top crop 0 at its current count
    for _ in range(3):
        q = appear(q, True, 1.0, 1.0)
    assert q.n_active.tolist() == [2, 5]
    assert int(q.alive[0].sum()) == 2  # existing leaves survive topping
    q = top(q, 4)  # a later, looser cap never raises an earlier one
    assert q.cap.tolist() == [2, 4]
    q = appear(q, True, 1.0, 1.0)
    assert q.n_active.tolist() == [2, 5]


def test_priming_removes_ranks_and_returns_mass() -> None:
    q = OrganQueue.empty(2, 6)
    for k in range(6):
        q = appear(q, True, area0=10.0 * (k + 1), mass0=jnp.array([1.0, 2.0]) * (k + 1))
    q = senesce(q, q.rank == 0)  # rank 0 already senesced: not harvested
    q2, h = prime(q, 0, 3)
    np.testing.assert_allclose(np.asarray(h), [2.0 + 3.0, 2 * (2.0 + 3.0)])
    np.testing.assert_array_equal(np.asarray(q2.alive[0]), [False, False, False, True, True, True])
    assert float(q2.mass[0, 1]) == 0.0 and float(q2.mass[0, 0]) == 1.0  # senesced mass kept
    # per-crop bounds; the -1/-1 "no priming" sentinel is a no-op
    q3, h3 = prime(q2, jnp.array([3, -1]), jnp.array([5, -1]))
    np.testing.assert_allclose(np.asarray(h3), [4.0 + 5.0, 0.0])
    np.testing.assert_array_equal(np.asarray(q3.alive[1]), np.asarray(q2.alive[1]))
    np.testing.assert_array_equal(np.asarray(q3.alive[0]), [False] * 5 + [True])
    agg = aggregate(q3, density=jnp.array([2.0, 1.0]))
    assert float(agg["leaf_mass"][0]) == pytest.approx(6.0 * 2.0 * 10.0)


def _grown_queue(scale: jax.Array, n_crop: int = 2, n_cohort: int = 4) -> OrganQueue:
    q = OrganQueue.empty(n_crop, n_cohort)
    for k in range(3):
        q = appear(q, True, area0=scale * (k + 1.0), mass0=0.5 * scale, n_mass0=0.01 * scale)
    return grow(q, d_area=scale, d_mass=0.1)


def test_aggregate_values() -> None:
    q = _grown_queue(jnp.asarray(1.0))
    agg = aggregate(q, density=8.0)
    area = (1 + 1) + (2 + 1) + (3 + 1)
    assert float(agg["lai"][0]) == pytest.approx(area * 8.0 * 1e-4)
    assert float(agg["leaf_mass"][0]) == pytest.approx(3 * 0.6 * 8.0 * 10.0)
    assert float(agg["n_mass"][0]) == pytest.approx(3 * 0.01 * 8.0 * 10.0)
    dead = senesce(q, True)
    assert float(aggregate(dead, 8.0)["lai"].sum()) == 0.0


def test_vmap_over_8_batches(tol: float) -> None:
    scales = jnp.linspace(0.5, 4.0, 8)
    densities = jnp.linspace(4.0, 11.0, 8)

    def pipeline(scale: jax.Array, density: jax.Array) -> dict[str, jax.Array]:
        q = _grown_queue(scale)
        q = age(q, 12.0)
        q = top(q, 3)
        q = appear(q, True, 5.0, 5.0)  # blocked by topping
        q, _ = prime(q, 0, 1)
        return aggregate(q, density)

    batched = jax.jit(jax.vmap(pipeline))(scales, densities)
    assert batched["lai"].shape == (8, 2)
    for i in range(8):
        single = pipeline(scales[i], densities[i])
        for key in ("lai", "leaf_mass", "n_mass"):
            np.testing.assert_allclose(np.asarray(batched[key][i]), np.asarray(single[key]), rtol=tol)

    # vmap directly over a batch of queues (leading axis on every leaf)
    qs = jax.vmap(_grown_queue)(scales)
    assert qs.area.shape == (8, 2, 4)
    qs = jax.vmap(lambda q: appear(q, jnp.array([True, False]), 7.0, 1.0))(qs)
    assert qs.n_active[:, 0].tolist() == [4] * 8 and qs.n_active[:, 1].tolist() == [3] * 8
    lai = jax.vmap(lambda q, d: aggregate(q, d)["lai"])(qs, densities)
    assert lai.shape == (8, 2)


def test_jit_grad_finite() -> None:
    def loss(area: jax.Array, mass: jax.Array) -> jax.Array:
        q = OrganQueue.empty(1, 3)
        q = appear(q, True, area, mass)
        q = appear(q, True, 2.0 * area, mass)
        q = grow(q, d_area=area)
        q = senesce(q, q.rank == 0)
        q, harvested = prime(q, 1, 2)
        agg = aggregate(q, 6.0)
        return jnp.sum(agg["lai"]) + jnp.sum(harvested) + jnp.sum(agg["leaf_mass"])

    g_area, g_mass = jax.jit(jax.grad(loss, argnums=(0, 1)))(jnp.asarray(30.0), jnp.asarray(0.4))
    assert bool(jnp.isfinite(g_area)) and bool(jnp.isfinite(g_mass))
    assert float(g_mass) == pytest.approx(1.0)  # only the primed cohort's mass reaches the loss

    def lai_of_area(area0: jax.Array) -> jax.Array:
        q = appear(OrganQueue.empty(2, 4), True, area0, 0.0)
        return jnp.sum(aggregate(q, 5.0)["lai"])

    g = jax.grad(lai_of_area)(jnp.array([10.0, 20.0]))
    np.testing.assert_allclose(np.asarray(g), [5.0e-4, 5.0e-4])


def test_n_cohort_1_equals_lumped_lai(tol: float) -> None:
    """CERES-Maize: n_cohort = 1, LAI = PLA * PLTPOP * 1e-4 with the single pool in slot 0."""
    rng = np.random.default_rng(0)
    d_pla = rng.uniform(0.0, 40.0, size=30)
    senes = rng.uniform(0.0, 5.0, size=30)
    pltpop = 7.5
    pla = 0.0
    q = appear(OrganQueue.empty(1, 1), True, area0=0.0, mass0=0.0)
    for t in range(30):
        pla = pla + d_pla[t] - senes[t]
        q = grow(q, d_area=d_pla[t] - senes[t], d_mass=0.01 * d_pla[t])
        q = appear(q, True, 123.0, 1.0)  # further appearance is a no-op for a lumped pool
        lai = float(aggregate(q, pltpop)["lai"][0])
        assert lai == pytest.approx(pla * pltpop * 1e-4, rel=tol)
    assert int(q.n_active[0]) == 1
