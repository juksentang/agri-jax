"""Soil grids and conservative remapping between them (plan 19 A7; decision Q8).

A :class:`SoilGrid` is a static, hashable description of a 1-D soil column: the depths of its cell
bottoms from the surface (cm), cells stacked from depth 0. The grids of the M3 assembly are

* :func:`rzwqm_nodes` - the RZWQM2 node grid: cell ``i`` spans ``[TLT(i-1), TLT(i)]`` with
  ``TLT`` the layer-bottom column of the ``rzwqm.dat`` node records (37 cells to 150 cm at
  CA-TPA);
* :func:`lyrset` / :func:`rzwqm_lyrset` - the fixed layers of an embedded DSSAT crop: bottoms at
  5, 15, 30, 45, 60 cm, then every 30 cm, cut at the bottom of the source profile, with the
  last layer merged half-and-half into the one above when it is thinner than that layer and
  thinner than 15 cm (``LYRSET``).

The remapping operators are built from the overlap matrix ``O[i, j]`` = thickness of target cell
``i`` inside source cell ``j`` (:func:`overlap`):

* **intensive** quantities (water content, temperature, a rate per unit thickness such as the
  uptake ``qsr`` in cm d-1 per cm): ``y_i = sum_j O_ij x_j / sum_j O_ij``, the thickness-weighted
  mean over the part of the target cell the source covers (0 where it covers none). This is the
  ``LMATCH`` rule of DSSAT-CSM, which RZWQM2 applies between its nodes and the embedded crop's
  layers (``REALMATCH``), so the faithful mapping and the conservative one are the same operator;
* **extensive** quantities (cm of water, kg ha-1 per cell): ``y_i = sum_j (O_ij / dz_j) x_j``,
  each source cell's amount split by the fraction of its thickness inside each target cell.

Where the target covers the source (same profile depth), the extensive map preserves the column
total exactly and the intensive map preserves the thickness-weighted total ``sum_i dz_i y_i``; on
coinciding grids both are the identity. The operators are linear with constant weights (NumPy
float64, built once per grid pair), so they are differentiable everywhere with finite gradients.

The surface cell "layer 0" and layer-fraction weights (review R-8) are reserved:
``SoilGrid(surface=True)`` raises until they are implemented (L-remap+).

Source: DSSAT-CSM v4.8.6.0 ``Soil/SoilUtilities/LMATCH.for`` (``LMATCH``, ``LYRSET``; BSD-3,
Copyright 1998-2026 DSSAT Foundation, University of Florida, International Fertilizer Development
Center). The RZWQM2 conventions (the fixed layers +30 cm to 20 layers, the node cells up to the
deepest layer bottom, the node <-> layer mapping by thickness-weighted means) are validated
against RZWQM2 4.6 dumps of CA-TPA (``tests/integration/test_rootwu_dssat.py``).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array

__all__ = [
    "LYRSET_FIXED",
    "LYRSET_MERGE_BELOW",
    "LYRSET_N_MAX",
    "LYRSET_STEP",
    "SoilGrid",
    "lyrset",
    "overlap",
    "remap",
    "remap_extensive",
    "remap_intensive",
    "remap_weights",
    "rzwqm_lyrset",
    "rzwqm_nodes",
]

#: fixed layer bottoms of ``LYRSET`` [cm]
LYRSET_FIXED: tuple[float, ...] = (5.0, 15.0, 30.0, 45.0, 60.0)
#: thickness of the layers below 60 cm in RZWQM2's embedded crop [cm]
LYRSET_STEP = 30.0
#: a last layer thinner than this (and than the layer above) is merged with the layer above [cm]
LYRSET_MERGE_BELOW = 15.0
#: maximum number of crop layers (DSSAT ``NL``; RZWQM2 ``dslayer``)
LYRSET_N_MAX = 20

Kind = Literal["intensive", "extensive"]


@dataclass(frozen=True)
class SoilGrid:
    """A 1-D soil column: cell bottoms [cm] from the surface, strictly increasing, the first > 0.

    Static and hashable (usable as a static argument or a closure constant under ``jit``).
    """

    name: str
    bottom: tuple[float, ...]
    surface: bool = False

    def __post_init__(self) -> None:
        b = tuple(float(x) for x in self.bottom)
        object.__setattr__(self, "bottom", b)
        if not b:
            raise ValueError(f"grid {self.name!r}: no cells")
        arr = np.asarray(b)
        if not (np.all(np.isfinite(arr)) and arr[0] > 0.0 and np.all(np.diff(arr) > 0.0)):
            raise ValueError(f"grid {self.name!r}: cell bottoms must be finite, > 0 and increasing: {b}")
        if self.surface:
            raise NotImplementedError(
                "the surface cell 'layer 0' is reserved (plan 19 A7, review R-8) and not implemented in M3"
            )

    @classmethod
    def from_thickness(cls, name: str, thickness: Sequence[float] | np.ndarray) -> SoilGrid:
        """A grid from its cell thicknesses (cm), stacked from the surface."""
        return cls(name, tuple(np.cumsum(np.asarray(thickness, dtype=float)).tolist()))

    @property
    def n(self) -> int:
        """Number of cells."""
        return len(self.bottom)

    @property
    def bottoms(self) -> np.ndarray:
        """Cell bottoms [cm], float64."""
        return np.asarray(self.bottom, dtype=float)

    @property
    def tops(self) -> np.ndarray:
        """Cell tops [cm], float64 (the first is 0)."""
        return np.concatenate([[0.0], self.bottoms[:-1]])

    @property
    def thickness(self) -> np.ndarray:
        """Cell thicknesses [cm], float64."""
        return np.diff(np.concatenate([[0.0], self.bottoms]))

    @property
    def depth(self) -> float:
        """Profile depth [cm] (bottom of the last cell)."""
        return self.bottom[-1]


# ------------------------------------------------------------------------ the M3 grids
def rzwqm_nodes(layer_bottom: Sequence[float] | np.ndarray, name: str = "rzwqm2_nodes") -> SoilGrid:
    """The RZWQM2 node grid from the layer-bottom column ``TLT`` of the node records [cm]."""
    return SoilGrid(name, tuple(np.asarray(layer_bottom, dtype=float).tolist()))


def lyrset(
    source: SoilGrid,
    *,
    fixed: Sequence[float] = LYRSET_FIXED,
    step: float = LYRSET_STEP,
    n_max: int = LYRSET_N_MAX,
    name: str = "lyrset",
) -> SoilGrid:
    """Fixed crop layers cut at the source profile (DSSAT ``LYRSET``).

    Candidate bottoms: ``fixed``, then ``+step`` up to ``n_max`` layers. Only the source cells whose
    bottom lies within the deepest candidate take part (RZWQM2 ``DSSATDRV``); the first candidate
    at or below the last of them becomes the last layer, its bottom set to that source bottom.
    A last layer thinner than the one above and than 15 cm is merged with it: both get the mean
    thickness.

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilUtilities/LMATCH.for, SUBROUTINE LYRSET (BSD-3); the
    +30 cm step to 20 layers is RZWQM2's embedded-crop convention (DSSAT-CSM uses 60 cm from
    layer 18 on, which only matters below 450 cm).
    """
    ds = list(float(x) for x in fixed)
    while len(ds) < n_max:
        ds.append(ds[-1] + float(step))
    ds = ds[:n_max]
    di = source.bottoms[source.bottoms <= ds[-1]]
    if di.size == 0:
        raise ValueError(f"no cell of {source.name!r} lies within the deepest layer bottom {ds[-1]} cm")
    last = float(di[-1])
    n = next(i for i, d in enumerate(ds) if d >= last) + 1
    out = np.asarray(ds[:n], dtype=float)
    out[-1] = last
    dl = np.diff(np.concatenate([[0.0], out]))
    if n > 1 and dl[-1] < dl[-2] and dl[-1] < LYRSET_MERGE_BELOW:
        half = 0.5 * (dl[-1] + dl[-2])
        out[-2] = (out[-3] if n > 2 else 0.0) + half
    return SoilGrid(name, tuple(out.tolist()))


def rzwqm_lyrset(nodes: SoilGrid, name: str = "rzwqm2_lyrset") -> SoilGrid:
    """The embedded crop's layers of RZWQM2 on the node grid ``nodes`` (:func:`lyrset`)."""
    return lyrset(nodes, name=name)


# ------------------------------------------------------------------------ operators
def overlap(src: SoilGrid, dst: SoilGrid) -> np.ndarray:
    """``O[i, j]`` = thickness [cm] of target cell ``i`` inside source cell ``j``, ``[dst.n, src.n]``."""
    lo = np.maximum(dst.tops[:, None], src.tops[None, :])
    hi = np.minimum(dst.bottoms[:, None], src.bottoms[None, :])
    return np.clip(hi - lo, 0.0, None)


@functools.lru_cache(maxsize=64)
def _weights(src: SoilGrid, dst: SoilGrid, kind: Kind) -> np.ndarray:
    o = overlap(src, dst)
    if kind == "intensive":
        cover = o.sum(axis=1, keepdims=True)
        w = np.divide(o, cover, out=np.zeros_like(o), where=cover > 0.0)
    elif kind == "extensive":
        w = o / src.thickness[None, :]
    else:
        raise ValueError(f"kind must be 'intensive' or 'extensive', got {kind!r}")
    w.setflags(write=False)
    return w


def remap_weights(src: SoilGrid, dst: SoilGrid, kind: Kind = "intensive") -> np.ndarray:
    """The constant weight matrix ``W`` [dst.n, src.n] of :func:`remap` (``y = W x``), float64."""
    return _weights(src, dst, kind)


def remap(x: ArrayLike, src: SoilGrid, dst: SoilGrid, kind: Kind = "intensive") -> Array:
    """Map ``x[..., src.n]`` to ``[..., dst.n]`` (``kind``: see the module docstring).

    Linear with constant weights: leading axes are batch axes, gradients are ``W^T`` (finite).

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilUtilities/LMATCH.for (LMATCH, thickness-weighted mean).
    """
    x = jnp.asarray(x)
    if x.shape[-1] != src.n:
        raise ValueError(f"x has {x.shape[-1]} cells on its last axis, grid {src.name!r} has {src.n}")
    w = jnp.asarray(
        _weights(src, dst, kind), dtype=x.dtype if jnp.issubdtype(x.dtype, jnp.floating) else None
    )
    return jnp.einsum("ij,...j->...i", w, x)


def remap_intensive(x: ArrayLike, src: SoilGrid, dst: SoilGrid) -> Array:
    """Thickness-weighted means of ``x`` over each target cell (``LMATCH`` / ``REALMATCH``)."""
    return remap(x, src, dst, "intensive")


def remap_extensive(x: ArrayLike, src: SoilGrid, dst: SoilGrid) -> Array:
    """Per-cell amounts of ``x`` redistributed by thickness fraction (column total preserved)."""
    return remap(x, src, dst, "extensive")
