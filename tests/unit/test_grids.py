"""Soil grids and remapping (plan 19 A7): LYRSET layers, the REALMATCH operator, conservation.

Data-free checks:

* the intensive operator equals, to 1e-12 (float64), a plain-Python transcription of the layer
  sweep of DSSAT-CSM ``LMATCH`` (``Soil/SoilUtilities/LMATCH.for``, BSD-3): a running pair of
  input / output cells, partial sums ``SUMZ``, ``SUMV`` of the overlaps, ``VS = SUMV / SUMZ``.
  RZWQM2's ``REALMATCH`` maps its nodes to the embedded crop's layers and back by the same
  thickness-weighted sweep (its dumps are checked in ``tests/integration/test_rootwu_dssat.py``).
  Checked on the CA-TPA node grid <-> its LYRSET layers in both directions and on random grids;
* column totals are preserved to 1e-12 (extensive map; intensive map with thickness weights);
* both maps are the identity on coinciding grids;
* gradients are finite and equal the transposed weights;
* LYRSET: the CA-TPA layers, the cut at the profile, the merge of a thin last layer, the
  20-layer limit, and the reserved surface cell.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core.grids import (
    SoilGrid,
    lyrset,
    overlap,
    remap,
    remap_extensive,
    remap_intensive,
    remap_weights,
    rzwqm_lyrset,
    rzwqm_nodes,
)

X64 = jax.config.jax_enable_x64
KINDS: tuple[Literal["intensive", "extensive"], ...] = ("intensive", "extensive")
TOL = 1e-12 if X64 else 2e-6

#: CA-TPA rzwqm.dat node records, column 2 (layer bottom TLT, cm)
CATPA_TLT = [1, 2, 4, 7, 11, 15, 19, 23, 26, 30, 34, 38, 43, 48, 53, 58, 63, 67, 70, 73, 77, 82, 86,
             90, 94, 98, 103, 108, 113, 118, 123, 128, 133, 138, 143, 147, 150]  # fmt: skip


def ref_sweep(dsi: list[float], vi: list[float], dso: list[float]) -> list[float]:
    """The thickness-weighted layer sweep of ``LMATCH`` (bottoms ``dsi`` -> ``dso``, same profile
    depth): Python floats, one input and one output cell at a time."""
    vs: list[float] = []
    k = 0
    zil = zol = 0.0
    for L in range(len(dso)):
        sumz = sumv = 0.0
        last = False
        while True:
            zt = max(zol, zil)
            zb = min(dso[L], dsi[k])
            sumz += zb - zt
            sumv += vi[k] * (zb - zt)
            if dso[L] < dsi[k]:
                break
            if k == len(dsi) - 1:
                last = True
                break
            zil = dsi[k]
            k += 1
        vs.append(sumv / sumz if sumz > 0.0 else vi[k])
        zol = dso[L]
        if last:
            break
    return vs


def catpa() -> tuple[SoilGrid, SoilGrid]:
    nodes = rzwqm_nodes(CATPA_TLT)
    return nodes, rzwqm_lyrset(nodes)


def test_catpa_layers_are_lyrset() -> None:
    nodes, layers = catpa()
    assert nodes.n == 37 and nodes.depth == 150.0
    assert layers.bottom == (5.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0)
    np.testing.assert_array_equal(layers.thickness, [5, 10, 15, 15, 15, 30, 30, 30])


@pytest.mark.parametrize("seed", range(5))
def test_intensive_map_is_the_layer_sweep_on_the_catpa_grids(seed: int) -> None:
    nodes, layers = catpa()
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.05, 0.45, (4, nodes.n))  # a batch of profiles
    got = np.asarray(remap_intensive(jnp.asarray(theta), nodes, layers))
    for b in range(4):
        want = ref_sweep(list(nodes.bottom), list(theta[b]), list(layers.bottom))
        np.testing.assert_allclose(got[b], want, rtol=TOL, atol=0.0)
    q = rng.uniform(0.0, 0.01, layers.n)  # a per-thickness rate on the layers, back to the nodes
    back = np.asarray(remap_intensive(jnp.asarray(q), layers, nodes))
    np.testing.assert_allclose(
        back, ref_sweep(list(layers.bottom), list(q), list(nodes.bottom)), rtol=TOL, atol=0.0
    )


@pytest.mark.parametrize("seed", range(10))
def test_intensive_map_is_the_layer_sweep_on_random_grids(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    depth = float(rng.uniform(40.0, 300.0))
    a = np.sort(rng.uniform(0.5, depth - 0.5, int(rng.integers(3, 40))))
    b = np.sort(rng.uniform(0.5, depth - 0.5, int(rng.integers(2, 20))))
    if seed % 3 == 0:
        b = np.union1d(b, a[::3])  # shared boundaries
    src = SoilGrid("a", (*np.unique(a).tolist(), depth))
    dst = SoilGrid("b", (*np.unique(b).tolist(), depth))
    x = rng.normal(size=src.n)
    got = np.asarray(remap_intensive(jnp.asarray(x), src, dst))
    np.testing.assert_allclose(
        got, ref_sweep(list(src.bottom), list(x), list(dst.bottom)), rtol=TOL, atol=TOL
    )


@pytest.mark.parametrize("seed", range(6))
def test_totals_are_preserved(seed: int) -> None:
    rng = np.random.default_rng(200 + seed)
    nodes, layers = catpa()
    grids = [(nodes, layers), (layers, nodes)]
    depth = 150.0
    grids.append((SoilGrid("r", (*np.sort(rng.uniform(1, 149, 25)).tolist(), depth)), layers))
    for src, dst in grids:
        x = rng.uniform(0.0, 5.0, (3, src.n))
        ext = np.asarray(remap_extensive(jnp.asarray(x), src, dst))
        np.testing.assert_allclose(ext.sum(-1), x.sum(-1), rtol=TOL, atol=0.0)
        inten = np.asarray(remap_intensive(jnp.asarray(x), src, dst))
        np.testing.assert_allclose(inten @ dst.thickness, x @ src.thickness, rtol=TOL, atol=0.0)
        # extensive = intensive of the density, times the target thickness
        np.testing.assert_allclose(
            ext,
            np.asarray(remap_intensive(jnp.asarray(x / src.thickness), src, dst)) * dst.thickness,
            rtol=TOL,
            atol=TOL,
        )


def test_identity_on_coinciding_grids() -> None:
    nodes, layers = catpa()
    for g in (nodes, layers):
        for kind in KINDS:
            np.testing.assert_array_equal(remap_weights(g, g, kind), np.eye(g.n))
        x = jnp.asarray(np.linspace(0.1, 0.4, g.n))
        assert np.asarray(remap(x, g, g)).tobytes() == np.asarray(x).tobytes()
    same = SoilGrid("copy", nodes.bottom)
    np.testing.assert_array_equal(remap_weights(nodes, same), np.eye(nodes.n))


def test_gradients_are_the_transposed_weights_and_finite() -> None:
    nodes, layers = catpa()
    w = jnp.asarray(np.linspace(0.5, 2.0, layers.n))
    for kind in KINDS:

        def loss(x: jax.Array, kind: Literal["intensive", "extensive"] = kind) -> jax.Array:
            return jnp.sum(w * remap(x, nodes, layers, kind))

        g = jax.grad(loss)(jnp.asarray(np.full(nodes.n, 0.3)))
        assert np.all(np.isfinite(np.asarray(g)))
        np.testing.assert_allclose(
            np.asarray(g), remap_weights(nodes, layers, kind).T @ np.asarray(w), rtol=TOL
        )
    # jit and vmap see constant weights
    f = jax.jit(jax.vmap(lambda x: remap_intensive(x, nodes, layers)))
    assert f(jnp.ones((3, nodes.n))).shape == (3, layers.n)


def test_overlap_rows_and_columns_are_thicknesses() -> None:
    nodes, layers = catpa()
    o = overlap(nodes, layers)
    assert o.shape == (layers.n, nodes.n) and np.all(o >= 0)
    np.testing.assert_allclose(o.sum(axis=1), layers.thickness, rtol=0, atol=1e-12)
    np.testing.assert_allclose(o.sum(axis=0), nodes.thickness, rtol=0, atol=1e-12)


def test_target_below_the_source_gets_the_covered_mean_or_zero() -> None:
    src = SoilGrid("s", (10.0, 20.0))
    dst = SoilGrid("d", (15.0, 30.0, 40.0))
    y = np.asarray(remap_intensive(jnp.asarray([1.0, 3.0]), src, dst))
    np.testing.assert_allclose(y, [(10 * 1 + 5 * 3) / 15, 3.0, 0.0])


# ------------------------------------------------------------------ LYRSET
def test_lyrset_cut_merge_and_limits() -> None:
    # the last candidate at or below the profile bottom takes the profile bottom
    assert lyrset(SoilGrid("x", (10.0, 20.0, 40.0, 70.0, 100.0, 108.0))).bottom == (
        5.0, 15.0, 30.0, 45.0, 60.0, 90.0, 108.0,
    )  # fmt: skip
    # a thin last layer (5 cm < 30 cm and < 15 cm) is merged half-and-half with the one above
    assert lyrset(SoilGrid("x", (50.0, 95.0))).bottom == (5.0, 15.0, 30.0, 45.0, 60.0, 77.5, 95.0)
    # thin but not thinner than the one above: kept
    assert lyrset(SoilGrid("x", (4.0, 8.0, 12.0))).bottom == (5.0, 12.0)
    # a one-layer profile
    assert lyrset(SoilGrid("x", (3.0,))).bottom == (3.0,)
    # at most 20 layers (to 510 cm): deeper source cells do not take part
    deep = lyrset(SoilGrid("x", tuple(float(z) for z in range(10, 801, 10))))
    assert deep.n == 20 and deep.depth == 510.0 and deep.bottom[5] == 90.0
    # the DSSAT-CSM deep convention (60 cm from layer 18) is a parameter
    with pytest.raises(ValueError, match="within the deepest"):
        lyrset(SoilGrid("x", (600.0,)))


def test_grid_validation_and_reserved_surface_cell() -> None:
    with pytest.raises(ValueError):
        SoilGrid("x", (5.0, 5.0))
    with pytest.raises(ValueError):
        SoilGrid("x", (0.0, 5.0))
    with pytest.raises(ValueError):
        SoilGrid("x", ())
    with pytest.raises(NotImplementedError, match="layer 0"):
        SoilGrid("x", (5.0,), surface=True)
    g = SoilGrid.from_thickness("t", [5, 10, 15])
    assert g.bottom == (5.0, 15.0, 30.0) and hash(g) == hash(SoilGrid("t", (5, 15, 30)))
    with pytest.raises(ValueError, match="cells on its last axis"):
        remap(jnp.ones(4), g, g)
