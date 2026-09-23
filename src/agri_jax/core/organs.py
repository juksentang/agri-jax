"""Within-crop organ queue (the ``n_cohort`` axis, docs/en/02_architecture.md section 3.7).

An :class:`OrganQueue` tracks organs (leaves, bolls, tillers, fruit) by rank in fixed-shape
``[n_crop, n_cohort]`` arrays. ``n_cohort`` is a compile-time constant: the maximum number of
organ positions (about 30 for tobacco leaves). Slot ``k`` holds the organ of rank ``k``
(0-based, in order of appearance), so appearance never needs dynamic allocation: it marks slot
``n_active`` alive and increments the counter. Senescence, topping and priming only change
masks. Every helper is a pure function of arrays with ``jnp.where`` / one-hot masks (no Python
branch on values, no loop over cohorts), so it works unchanged under ``jax.jit``, ``jax.vmap``
over a leading batch axis and ``jax.grad`` with respect to ``area`` / ``mass``.

Lumped crop models degenerate to ``n_cohort = 1``: **CERES-Maize uses** ``n_cohort = 1``; its
single leaf pool lives in slot 0, ``area[:, 0]`` is the plant leaf area ``PLA`` [cm2/plant] and
the lumped ``LAI = PLA * PLTPOP * 1e-4`` equals ``aggregate(q, density)["lai"]``, so lumped crop
code keeps working on the queue (``tests/unit/test_organs.py`` checks the equivalence).

Units (internal convention): ``area`` cm2/plant, ``mass`` and ``n_mass`` g/plant, ``age_tt``
degC d, plant ``density`` plants m-2; :func:`aggregate` returns LAI [m2 m-2] and masses in kg ha-1.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agri_jax.core.state import field

__all__ = ["OrganQueue", "age", "aggregate", "appear", "grow", "prime", "senesce", "top"]

_CM2_PER_PLANT_TO_LAI = 1e-4  # cm2/plant * plants/m2 -> m2/m2
_G_M2_TO_KG_HA = 10.0  # g/m2 -> kg/ha

_DIMS = ("n_crop", "n_cohort")


class OrganQueue(eqx.Module):
    """Organ cohorts of every crop, indexed by rank, with static shape ``[n_crop, n_cohort]``.

    ``alive`` distinguishes the three slot states together with ``n_active``: slots
    ``k >= n_active`` have not appeared yet; slots ``k < n_active`` are alive or, when
    ``alive`` is False, senesced or harvested. ``cap`` is the number of slots that may ever
    appear (``n_cohort`` until :func:`top` lowers it).

    CERES-Maize uses ``n_cohort = 1``; the lumped LAI then equals ``aggregate(q, density)["lai"]``.
    """

    rank: Array = field(
        unit="-", description="organ rank (leaf / boll position, tiller index), 0-based", dims=_DIMS
    )
    area: Array = field(unit="cm2 plant-1", description="organ (leaf) area per plant", dims=_DIMS)
    mass: Array = field(unit="g plant-1", description="organ dry mass per plant", dims=_DIMS)
    age_tt: Array = field(unit="degC d", description="thermal time since appearance", dims=_DIMS)
    n_mass: Array = field(unit="g plant-1", description="organ nitrogen mass per plant", dims=_DIMS)
    alive: Array = field(
        unit="-", description="cohort has appeared and is neither senesced nor harvested", dims=_DIMS
    )
    n_active: Array = field(unit="-", description="number of cohorts that have appeared", dims="n_crop")
    cap: Array = field(
        unit="-", description="maximum number of cohorts that may appear (topping)", dims="n_crop"
    )

    @classmethod
    def empty(cls, n_crop: int, n_cohort: int, dtype: Any = None) -> OrganQueue:
        """A queue with no organ appeared yet; ``rank[c, k] = k``, ``cap = n_cohort``."""
        if n_crop < 1 or n_cohort < 1:
            raise ValueError(f"n_crop and n_cohort must be >= 1, got {n_crop}, {n_cohort}")
        shape = (n_crop, n_cohort)
        z = jnp.zeros(shape, dtype=dtype)
        return cls(
            rank=jnp.broadcast_to(jnp.arange(n_cohort, dtype=jnp.int32), shape),
            area=z,
            mass=z,
            age_tt=z,
            n_mass=z,
            alive=jnp.zeros(shape, dtype=bool),
            n_active=jnp.zeros((n_crop,), dtype=jnp.int32),
            cap=jnp.full((n_crop,), n_cohort, dtype=jnp.int32),
        )

    @property
    def n_crop(self) -> int:
        return int(self.alive.shape[-2])

    @property
    def n_cohort(self) -> int:
        return int(self.alive.shape[-1])


def _per_crop(q: OrganQueue, x: ArrayLike) -> Array:
    """Broadcast a scalar or ``[n_crop]`` value to ``[n_crop, 1]`` (vmap-safe: shapes are static)."""
    return jnp.broadcast_to(jnp.asarray(x), (q.n_crop,))[:, None]


def appear(
    q: OrganQueue,
    crop_idx_mask: ArrayLike,
    area0: ArrayLike,
    mass0: ArrayLike,
    n_mass0: ArrayLike = 0.0,
) -> OrganQueue:
    """One new cohort appears in every crop where ``crop_idx_mask`` is True.

    Slot ``n_active`` is marked alive with ``area0`` / ``mass0`` / ``n_mass0`` (scalars or
    ``[n_crop]``), age 0, and ``n_active`` is incremented. A crop whose queue is full
    (``n_active == n_cohort``) or topped (``n_active >= cap``) is left unchanged.
    """
    can = jnp.broadcast_to(jnp.asarray(crop_idx_mask, dtype=bool), (q.n_crop,)) & (q.n_active < q.cap)
    onehot = (jnp.arange(q.n_cohort) == q.n_active[:, None]) & can[:, None]
    return eqx.tree_at(
        lambda s: (s.area, s.mass, s.n_mass, s.age_tt, s.alive, s.n_active),
        q,
        (
            jnp.where(onehot, _per_crop(q, area0), q.area),
            jnp.where(onehot, _per_crop(q, mass0), q.mass),
            jnp.where(onehot, _per_crop(q, n_mass0), q.n_mass),
            jnp.where(onehot, jnp.zeros_like(q.age_tt), q.age_tt),
            q.alive | onehot,
            (q.n_active + can).astype(q.n_active.dtype),
        ),
    )


def age(q: OrganQueue, dtt: ArrayLike) -> OrganQueue:
    """Add thermal time ``dtt`` [degC d] (scalar or ``[n_crop]``) to every alive cohort."""
    return eqx.tree_at(lambda s: s.age_tt, q, q.age_tt + jnp.where(q.alive, _per_crop(q, dtt), 0.0))


def grow(
    q: OrganQueue, d_area: ArrayLike = 0.0, d_mass: ArrayLike = 0.0, d_n_mass: ArrayLike = 0.0
) -> OrganQueue:
    """Add increments (broadcastable to ``[n_crop, n_cohort]``) to the alive cohorts only."""
    return eqx.tree_at(
        lambda s: (s.area, s.mass, s.n_mass),
        q,
        (
            q.area + jnp.where(q.alive, d_area, 0.0),
            q.mass + jnp.where(q.alive, d_mass, 0.0),
            q.n_mass + jnp.where(q.alive, d_n_mass, 0.0),
        ),
    )


def senesce(q: OrganQueue, mask: ArrayLike) -> OrganQueue:
    """Set ``alive = False`` where ``mask`` (broadcastable to ``[n_crop, n_cohort]``) is True.

    Area, mass and nitrogen of senesced cohorts are kept (litter bookkeeping is the caller's).
    """
    return eqx.tree_at(lambda s: s.alive, q, q.alive & ~jnp.asarray(mask, dtype=bool))


def top(q: OrganQueue, max_active: ArrayLike, mask: ArrayLike = True) -> OrganQueue:
    """Topping: no cohort beyond ``max_active`` (scalar or ``[n_crop]``) will ever appear.

    Lowers ``cap`` to ``min(cap, max_active)`` where ``mask`` (scalar or ``[n_crop]``) is True.
    Cohorts that already appeared stay alive; pass ``max_active = q.n_active`` to stop
    appearance at the current count.
    """
    m = jnp.broadcast_to(jnp.asarray(mask, dtype=bool), (q.n_crop,))
    new_cap = jnp.minimum(q.cap, jnp.broadcast_to(jnp.asarray(max_active), (q.n_crop,)))
    return eqx.tree_at(lambda s: s.cap, q, jnp.where(m, new_cap, q.cap).astype(q.cap.dtype))


def prime(q: OrganQueue, rank_lo: ArrayLike, rank_hi: ArrayLike) -> tuple[OrganQueue, Array]:
    """Priming: harvest the alive cohorts with ``rank_lo <= rank < rank_hi``.

    Bounds are scalars or ``[n_crop]``; an empty range (e.g. the ``-1, -1`` "no priming"
    sentinel of :class:`~agri_jax.core.events.EventTable`) is a no-op. Harvested cohorts are
    marked dead and their area, mass and nitrogen are removed from the queue.

    Returns ``(q_new, harvested_mass)`` with ``harvested_mass`` [g/plant] of shape ``[n_crop]``.
    """
    hit = q.alive & (q.rank >= _per_crop(q, rank_lo)) & (q.rank < _per_crop(q, rank_hi))
    harvested = jnp.sum(jnp.where(hit, q.mass, 0.0), axis=-1)
    q_new = eqx.tree_at(
        lambda s: (s.area, s.mass, s.n_mass, s.alive),
        q,
        (
            jnp.where(hit, 0.0, q.area),
            jnp.where(hit, 0.0, q.mass),
            jnp.where(hit, 0.0, q.n_mass),
            q.alive & ~hit,
        ),
    )
    return q_new, harvested


def aggregate(q: OrganQueue, density: ArrayLike) -> dict[str, Array]:
    """Lumped canopy quantities per crop, summed over alive cohorts times plant ``density``.

    ``density`` [plants m-2] is a scalar or ``[n_crop]``. Returns ``[n_crop]`` arrays:
    ``lai`` [m2 m-2] ``= sum(area * alive) * density * 1e-4``, ``leaf_mass`` [kg ha-1]
    ``= sum(mass * alive) * density * 10`` and ``n_mass`` [kg ha-1] likewise.
    """
    dens = jnp.broadcast_to(jnp.asarray(density), (q.n_crop,))

    def total(x: Array) -> Array:
        return jnp.sum(jnp.where(q.alive, x, 0.0), axis=-1) * dens

    return {
        "lai": total(q.area) * _CM2_PER_PLANT_TO_LAI,
        "leaf_mass": total(q.mass) * _G_M2_TO_KG_HA,
        "n_mass": total(q.n_mass) * _G_M2_TO_KG_HA,
    }
