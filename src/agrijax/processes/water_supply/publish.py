"""Layer uptake to node uptake at the end of the crop day (entry 16 of the contract's day, port P3).

After ``ROOTWU`` and the crop, the RZWQM2 crop driver ``DSSATDRV`` turns the day's layer uptake
into the node uptake the soil side uses the next morning [RZWQM2 4.6,
``RZWQM/DSSATDRV.for:1543-1569``]: on each crop layer ``L`` the rate
``RWU_L / 24 / DLAYR_L`` when ``SW_L > LL_L`` and 0 when ``SW_L < LL_L``, all layers 0 when the
day's ``PET <= 0``; then the thickness-weighted mean of the layer rates onto the nodes
(``REALMATCH``). This process publishes the result as :class:`~agrijax.iface.soil.NodeUptake`,
extensive (water depth taken from each node cell per day): the node rate times the node
thickness, which is RZWQM2's ``qsr * 24 * TL``.

The ``SW == LL`` quirk. A layer whose ``SW`` equals ``LL`` exactly is
neither set nor zeroed in the reference: its element of the rate array keeps the value the array
held when ``DSSATDRV`` was entered, which is the **node** ``L`` value of the day's sink (the
morning's limited ``qsr``, port P4 ``soil_water.sink_in.uptake``; at CA-TPA 2015-2023 the
``DSSATDRV`` entry ``QSR`` equals the ``PHYSCL`` exit ``QSR`` on all 1088 crop days). This
process reproduces it: such a layer gets the node rate ``sink_in.uptake[L] / TL_L``. The
comparison of ``SW`` with ``LL`` is made on the REAL*4-rounded values, as in the reference: in
float64 a layer the reference sees at ``SW == LL`` would almost always count as ``SW > LL``
(the precision trap of this quirk). At CA-TPA the quirk acts on 22 of 1088 crop days (2016-243 to
2016-264, 33 layer-days on layers 4 and 5); giving such a layer 0 instead misses up to 0.64 cm d-1
of the day's node uptake there (61 %, median 20 %; ``tests/integration/test_water_supply_rzwqm.py``).
With ``SW`` remapped in float64 from the reference's node ``THETA``, none of the 33 layer-days is
equal in float64 (5 count as ``SW > LL``, the rest as ``SW < LL``, within 2.2e-9) and all 33 are
equal on the REAL*4-rounded values, with no other layer-day equal.

``PET > 0`` is read as ``EOP > 0`` (port P1 ``eop``): with ``ISTRESS = 0`` RZWQM2 sets
``EOP = 10 PET``, and on the CA-TPA crop days both select the same 118 days without uptake.

Crops: ``rwu`` and ``eop`` are per crop (``n_crop``), ``sw`` and the layer and node grids are
shared by the crops of the slot; the quirk's node value is the day's total sink (one crop in
RZWQM2).

Source: RZWQM2 4.6 DSSATDRV (``DSSATDRV.for:1543-1569``), conventions only (no RZWQM2 statement is
reproduced); the layer -> node map is DSSAT-CSM v4.8.6.0 ``LMATCH`` (thickness-weighted mean,
:func:`agrijax.core.grids.remap_intensive`).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.grad import real4_store
from agrijax.core.grids import SoilGrid, remap_intensive
from agrijax.core.process import process
from agrijax.core.state import Forcing, Params, field

from .rootwu import RootwuState

__all__ = ["PublishUptakeParams", "rzwqm_publish_uptake"]


class PublishUptakeParams(Params):
    """The two grids of the map (soil-water nodes, crop layers) and the crop layers' lower limit."""

    nodes: SoilGrid = field(description="soil-water node grid (RZWQM2 TLT)", static=True)
    layers: SoilGrid = field(description="crop layers (LYRSET on the node grid)", static=True)
    ll: Array = field(
        unit="cm3 cm-3",
        description="lower limit of the crop layers (as the crop driver holds it, REAL*4 in RZWQM2)",
        fortran_name="SOILPROP%LL",
        dims=("n_layer",),
        grid="dssat_layers",
    )


@process(
    reads=("rwu", "water.sw", "water.eop", "sink_in.uptake"),
    writes=("root_uptake.uptake",),
    source="RZWQM2 4.6 DSSATDRV (DSSATDRV.for:1543-1569): layer uptake -> node qsr, SW == LL quirk",
    fortran_name="DSSATDRV",
    key="crop_iface/publish_uptake@rzwqm2-4.6:faithful",
    provenance="reference_only_conventions",
    grid="rzwqm2_nodes",
    ref_build=(
        "main_ryzen5_avx512 (RZWQM2 4.6), instrumented build that dumps the daily entry and exit tables; "
        "its outputs equal those of the plain build"
    ),
    sources=(
        ("layer rate RWU / DLAYR where SW > LL, 0 where SW < LL", "RZWQM2 4.6 DSSATDRV.for:1550-1562"),
        (
            "SW == LL: the layer keeps the node value of the array as found",
            "RZWQM2 4.6 DSSATDRV.for:1550-1562",
        ),
        ("no uptake when PET <= 0", "RZWQM2 4.6 DSSATDRV.for:1543, 1563-1568"),
        ("thickness-weighted mean onto the nodes (REALMATCH)", "RZWQM2 4.6 DSSATDRV.for:1569"),
    ),
    deviates=(
        (
            "the layer and node arithmetic is not rounded to REAL*4",
            "working precision; the SW/LL comparison alone is made on REAL*4-rounded values",
            "tests/integration/test_water_supply_rzwqm.py (CA-TPA 2015-2023, 1088 crop days)",
        ),
        (
            "RWU is not rewritten (the reference zeroes RWU of layers with SW < LL and on PET <= 0 days)",
            "rwu stays ROOTWU's output; only the node uptake is published",
            "no effect at CA-TPA: DSSATDRV exit RWU equals ROOTWU exit RWU on all 1088 crop days",
        ),
    ),
)
def rzwqm_publish_uptake(state: RootwuState, params: PublishUptakeParams, forcing_t: Forcing) -> RootwuState:
    """``root_uptake.uptake`` [n_crop, n_node] from the day's layer uptake ``rwu`` [n_crop, n_layer].

    Reads ``rwu`` (the producer's own state), ``water.sw`` and ``water.eop`` (P1) and the day's
    node sink ``sink_in.uptake`` (P4, for the ``SW == LL`` layers); no forcing. The layer
    thicknesses are ``params.layers``, the node thicknesses ``params.nodes``.

    Source: RZWQM2 4.6 DSSATDRV, DSSATDRV.for:1543-1569 (conventions only); DSSAT-CSM v4.8.6.0 LMATCH.
    """
    rwu = state.rwu
    dt = rwu.dtype
    dlayr = jnp.asarray(params.layers.thickness, dtype=dt)
    tl = jnp.asarray(params.nodes.thickness, dtype=dt)
    n_layer = params.layers.n
    # the REAL*4 values the reference compares (reduce-precision: XLA may drop a convert on GPU)
    sw32 = real4_store(state.water.sw)
    ll32 = real4_store(params.ll)
    rate = rwu / dlayr  # d-1
    found = jnp.asarray(state.sink_in.uptake, dtype=dt)[..., :n_layer] / tl[:n_layer]  # node L as found
    kept = jnp.where(sw32 < ll32, 0.0, found)
    layer = jnp.where(sw32 > ll32, rate, kept)
    layer = jnp.where((jnp.asarray(state.water.eop) > 0.0)[..., None], layer, 0.0)
    node = remap_intensive(layer, params.layers, params.nodes) * tl
    old = state.root_uptake.uptake
    return eqx.tree_at(lambda s: s.root_uptake.uptake, state, node.astype(old.dtype))
