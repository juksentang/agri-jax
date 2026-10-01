"""The RZWQM2 uptake limit ``WUF``: yesterday's node uptake limited by today's PET (entry 6 of the coupling
contract's day, :data:`agrijax.iface.contract.DAY_TABLE`).

Before the soil-water time loop of the day, RZWQM2 ``PHYSCL`` limits the root water uptake the
embedded DSSAT crop published the evening before (node ``qsr``, one day old) by the ratio of
today's potential transpiration to the crop's potential uptake of the day before
[RZWQM2 4.6, ``RZWQM/Rzday.for:1341-1357``, PHYSCL]:

.. math::

    \\mathrm{WUF} = \\min\\!\\Big(1, \\frac{\\mathrm{PET}}{\\mathrm{TRWUP}}\\Big), \\qquad
    q_i \\leftarrow \\begin{cases} q_i\\,\\mathrm{WUF} & \\theta_i > \\theta_{9,i} \\\\
                                   0 & \\theta_i < \\theta_{9,i} \\\\
                                   q_i & \\theta_i = \\theta_{9,i} \\end{cases}

on every node ``i``, but only when ``PET > 0`` and ``TRWUP != 0``; otherwise the uptake is kept
as it is. ``PET`` is today's potential transpiration (port P5, ``iface.pet.transpiration``;
RZWQM2 ``PETPLANT = PET``), ``TRWUP`` yesterday's potential root water uptake of the crop (port
P1 ``trwup``, read before ``ROOTWU`` writes it: a one-day lag), ``q`` yesterday's node uptake
(port P3, one-day lag), ``theta`` the node water content at the start of the day (port P7,
``soil_water.theta``) and ``theta_9`` the wilting point ``SOILHP(9)`` of the node's horizon. The
result is the day's uptake sink of the soil-water day (port P4, ``soil_water.sink_in.uptake``).

In the reference ``WUF`` is a double-precision quotient rounded to REAL*4 and the product is
REAL*4; this process runs in the working precision (float64 with x64), which differs from the
reference by at most two REAL*4 roundings (measured on CA-TPA 2015-2023: 1.05e-7 relative,
``tests/integration/test_water_supply_rzwqm.py``). The node sink of the soil-water day is set to
zero on the nodes whose Newton update is clamped at ``HMIN`` inside ``RICHRD``: that belongs to
the soil-water day, not to this limit (there the sink is cut to the water available above the
water content at ``HMIN`` instead; ``RICHRD`` zeroes it in place on 1.3 % of the node-days of the
CA-TPA dump tables of the instrumented RZWQM2 4.6 build, all at the dry limit).

Crops: the ports are per crop slot, the record axis ``n_crop`` holds the crops of the slot, each
limited by its own ``TRWUP``; the sink is their sum over ``n_crop`` (RZWQM2 has one crop, and
the sum over a single crop is that crop's uptake bit for bit).

Source: RZWQM2 4.6 PHYSCL (``Rzday.for:1341-1357``), conventions only (no RZWQM2 statement is
reproduced); verified day by day on the dump tables of the instrumented RZWQM2 4.6 build for CA-TPA
2015-2023.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.ports import port
from agrijax.core.process import process
from agrijax.core.state import Forcing, Params, State, field
from agrijax.iface.crop import CropWaterIn
from agrijax.iface.soil import NodeUptake, SinkInputs
from agrijax.iface.surface import PETFluxes

__all__ = ["UptakeLimitParams", "UptakeLimitState", "rzwqm_uptake_limit", "uptake_limit_factor"]

_N = ("n_node",)


class UptakeLimitParams(Params):
    """The node wilting point of the limit (``SOILHP(9)`` of each node's horizon), ``[n_node]``."""

    theta_wp: Array = field(
        unit="cm3 cm-3",
        description="wilting point (15 bar) of the node's horizon: below it a node gives no uptake",
        fortran_name="SOILHP(9)",
        dims=_N,
        grid="rzwqm2_nodes",
    )


class UptakeLimitState(State):
    """The uptake limit holds no state of its own: only the five records it reads and writes.

    ``theta`` (P7) and ``root_uptake`` (P3) and ``water`` (P1, ``trwup``) and ``pet`` (P5) are
    read, ``sink_in`` (P4) gets its ``uptake`` field written. Run on its own, fill the ports.
    """

    theta: Array = port(
        unit="cm3 cm-3",
        description="node water content at the start of the day (P7, read)",
        fortran_name="THETA",
        dims=_N,
        grid="rzwqm2_nodes",
    )
    root_uptake: NodeUptake = port(description="yesterday's node uptake of the crops (P3, read)")
    water: CropWaterIn = port(description="crop water record: yesterday's trwup read (P1)")
    pet: PETFluxes = port(description="today's potential fluxes: transpiration read (P5)")
    sink_in: SinkInputs = port(description="sink record of the soil-water day: uptake written (P4)")


def uptake_limit_factor(pet: Any, trwup: Any) -> tuple[Array, Array]:
    """``(on, wuf)``: where the limit applies (``PET > 0`` and ``TRWUP != 0``) and
    ``WUF = min(1, PET / TRWUP)`` there (1 elsewhere), per crop.

    ``pet`` [cm d-1] broadcasts against ``trwup`` [cm d-1, ``(n_crop)``]. The quotient is taken on a
    divisor that is 1 where the limit does not apply, so every branch is finite.

    Source: RZWQM2 4.6 PHYSCL, ``Rzday.for:1341-1346`` (conventions only).
    """
    pet = jnp.asarray(pet)
    trwup = jnp.asarray(trwup)
    on = (pet > 0.0) & (trwup != 0.0)
    divisor = jnp.where(on, trwup, 1.0)
    ratio = pet / divisor
    wuf = jnp.where(on & (pet <= divisor), ratio, 1.0)
    return on, wuf


@process(
    reads=("theta", "root_uptake", "water.trwup", "pet.transpiration"),
    writes=("sink_in.uptake",),
    source="RZWQM2 4.6 PHYSCL (Rzday.for:1341-1357): WUF = min(1, PET/TRWUP) on yesterday's qsr",
    fortran_name="PHYSCL",
    key="soil_water/wuf@rzwqm2-4.6:faithful",
    provenance="reference_only_conventions",
    grid="rzwqm2_nodes",
    ref_build=(
        "main_ryzen5_avx512 (RZWQM2 4.6), instrumented build that dumps the daily entry and exit tables; "
        "its outputs equal those of the plain build"
    ),
    sources=(
        ("WUF = min(1, PET / TRWUP) when PET > 0 and TRWUP != 0", "RZWQM2 4.6 Rzday.for:1341-1346"),
        (
            "nodes above SOILHP(9) scaled by WUF, below it zeroed, equal kept",
            "RZWQM2 4.6 Rzday.for:1348-1356",
        ),
        ("PET is today's plant PET (PETPLANT = PET)", "RZWQM2 4.6 Rzday.for:1358"),
    ),
    deviates=(
        (
            "REAL*4 rounding of WUF and of the product is not reproduced",
            "the limit runs in the working precision; at most two REAL*4 roundings from the reference",
            "tests/integration/test_water_supply_rzwqm.py (1.05e-7 relative, CA-TPA 2015-2023)",
        ),
    ),
)
def rzwqm_uptake_limit(
    state: UptakeLimitState, params: UptakeLimitParams, forcing_t: Forcing
) -> UptakeLimitState:
    """``sink_in.uptake`` [n_node] = the sum over the slot's crops of yesterday's node uptake
    ``root_uptake.uptake`` [n_crop, n_node] limited by ``WUF`` (see the module docstring).

    Reads the node water content ``theta``, yesterday's ``root_uptake`` and ``water.trwup``,
    today's ``pet.transpiration``, and the node wilting point ``params.theta_wp``; no forcing.

    Source: RZWQM2 4.6 PHYSCL, Rzday.for:1341-1357 (conventions only).
    """
    q = state.root_uptake.uptake
    on, wuf = uptake_limit_factor(state.pet.transpiration, state.water.trwup)
    theta = jnp.asarray(state.theta)
    theta_wp = jnp.asarray(params.theta_wp, dtype=theta.dtype)
    kept = jnp.where(theta < theta_wp, 0.0, q)  # theta == theta_wp keeps the node as it is
    limited = jnp.where(theta > theta_wp, q * wuf[..., None].astype(q.dtype), kept)
    node = jnp.where(on[..., None], limited, q)
    old = state.sink_in.uptake
    return eqx.tree_at(lambda s: s.sink_in.uptake, state, jnp.sum(node, axis=-2).astype(old.dtype))
