"""Kernels of the DSSAT-CSM v4.8.6.0 tipping bucket: one function per Fortran routine.

Each kernel is a pure function of explicit arrays, on ``[..., n_layer]`` layer arrays (leading
batch axes allowed) with padded layers marked by ``dlayr == 0`` (``NLAYR`` = the count of layers
with ``dlayr > 0``, which must come first). Units are DSSAT's: water contents cm3 cm-3, layer
thickness and per-layer flows cm (``DRN``, ``UPFLOW``: cm d-1), surface water mm (``RAIN``,
``RUNOFF``, ``WATAVL``, ``DRAIN``: mm d-1), as in the Fortran, so that every intermediate can be
compared with the reference's dump.

The true recurrences along depth (``INFIL``'s downward pass with its upward redistribution of an
excess, ``SATFLO``'s two passes, ``UP_FLOW``'s pairwise exchange) go through
:func:`agrijax.core.depth_scan.depth_scan`; everything else is elementwise. Every branch is a
``jnp.where`` on two finite values: divisions by a layer thickness, a curve number or an
extractable-water range use clamped denominators, and the result of a padded layer is discarded.

Source: DSSAT-CSM v4.8.6.0 ``Soil/SoilWater/WATBAL.for`` (RATE: lines 276-459; INTEGR: lines
465-533), ``RNOFF.for``, ``INFIL.for``, ``SATFLO.for``, ``WBSUBS.for`` (``SNOWFALL``, ``UP_FLOW``)
and ``Soil/Mulch/MULCHWAT.for`` (``MULCHWATER``), BSD-3: Copyright 1998-2026 DSSAT Foundation,
University of Florida, International Fertilizer Development Center (see
``THIRD_PARTY_NOTICES.md``). The Fortran statements are translated one to one; the line numbers
are cited per kernel.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.coefficients import numerical_guard
from agrijax.core.depth_scan import depth_scan
from agrijax.core.grad import real4_store, round_st
from agrijax.core.units import CM_PER_MM, HOURS_PER_DAY, MM_PER_CM

from .coefficients import WATBAL_COEFFICIENTS, WatbalCoefficients

__all__ = [
    "InfilResult",
    "MulchRate",
    "RunoffResult",
    "SnowResult",
    "UpflowResult",
    "infil",
    "integrate_sw",
    "mulch_integrate",
    "mulch_rate",
    "rnoff",
    "satflo",
    "snowfall",
    "up_flow",
]

#: floor of a layer thickness, an extractable-water range and the other divisors of the kernels
#: (padded layers have dlayr = 0; their results are discarded, the floor only keeps them finite)
_TINY = numerical_guard(
    "bucket.tiny", 1e-12, "floor of divisors (layer thickness, DUL - LL, CN, runoff denominator)"
)


def _moved(x: Array) -> Array:
    """Layer axis first (the depth axis of :func:`depth_scan`)."""
    return jnp.moveaxis(x, -1, 0)


def _back(x: Array) -> Array:
    """Layer axis back to last."""
    return jnp.moveaxis(x, 0, -1)


def _div(num: ArrayLike, den: ArrayLike) -> Array:
    """``num / den`` with the denominator floored at :data:`_TINY` (a thickness or range > 0)."""
    return jnp.asarray(num) / jnp.maximum(den, _TINY)


# --------------------------------------------------------------------------- SNOWFALL
class SnowResult(NamedTuple):
    """Result of :func:`snowfall`."""

    snow: Array  # mm, snow pack after the day's accumulation / melt
    watavl: Array  # mm d-1, water available for runoff and infiltration (rain plus melt)
    dropped: Array  # mm d-1, snow pack set to zero by the 0.001 mm cut (not accounted for)


def snowfall(
    tmax: ArrayLike, rain: ArrayLike, snow: ArrayLike, c: WatbalCoefficients = WATBAL_COEFFICIENTS
) -> SnowResult:
    """``WATBAL`` RATE lines 282-288 and ``SNOWFALL`` RATE: precipitation to snow or melt.

    ``SNOWFALL`` runs when ``TMAX <= 1`` or a pack exists; otherwise ``WATAVL = RAIN``.

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilWater/WATBAL.for:282-288, WBSUBS.for:44-60 (SNOWFALL).
    """
    tmax = jnp.asarray(tmax)
    rain = jnp.asarray(rain)
    snow = jnp.asarray(snow)
    called = (tmax <= c.snow_tmax) | (snow > 0.0)
    melting = tmax > c.snow_tmax
    melt = jnp.minimum(c.melt_per_degc * tmax + rain * c.melt_rain, snow)
    s_new = jnp.where(melting, snow - melt, snow + rain)
    w_new = jnp.where(melting, rain + melt, 0.0)
    cut = s_new < c.snow_min
    dropped = jnp.where(called & cut, s_new, 0.0)
    s_new = jnp.where(cut, 0.0, s_new)
    return SnowResult(
        snow=jnp.where(called, s_new, snow), watavl=jnp.where(called, w_new, rain), dropped=dropped
    )


# --------------------------------------------------------------------------- MULCHWATER
class MulchRate(NamedTuple):
    """Result of :func:`mulch_rate` (``MULCHWATER`` RATE)."""

    watavl: Array  # mm d-1, water left for runoff and infiltration after the interception
    mulwatadd: Array  # mm d-1, rain intercepted by the mulch
    reswatadd: Array  # mm d-1, water brought in by newly added residue
    mulchsat: Array  # mm, water holding capacity of the mulch


def mulch_rate(
    watavl: ArrayLike,
    mulchwat: ArrayLike,
    mulchevap_prev: ArrayLike,
    mass: ArrayLike,
    cover: ArrayLike,
    new_mass: ArrayLike,
    watfac: ArrayLike,
    on: ArrayLike,
    c: WatbalCoefficients = WATBAL_COEFFICIENTS,
) -> MulchRate:
    """``MULCHWATER`` RATE: rain interception by surface residue (``MEINF`` in ``'RSM'``: ``on``).

    ``mulchwat`` is the routine's saved mulch water [mm], ``mulchevap_prev`` the mulch evaporation
    of the previous SPAM call [mm d-1] (``MULCH%MULCHEVAP`` at the day's ``WATBAL`` RATE), ``mass``,
    ``cover``, ``new_mass`` and ``watfac`` the residue record the soil organic matter module left
    (``MULCHMASS`` kg ha-1, ``MULCHCOVER``, ``NEWMULCH`` kg ha-1, ``MUL_WATFAC`` mm per 1e-4 kg ha-1).

    Source: DSSAT-CSM v4.8.6.0 Soil/Mulch/MULCHWAT.for:98-151.
    """
    watavl = jnp.asarray(watavl)
    on = jnp.asarray(on, dtype=bool)
    mass = jnp.asarray(mass)
    new_mass = jnp.asarray(new_mass)
    mulchsat = watfac * c.mulch_sat_per_mass * mass
    reswatadd = jnp.where(
        new_mass > c.new_mulch_min, c.new_mulch_wet_fraction * watfac * c.mulch_sat_per_mass * new_mass, 0.0
    )
    wet = (mass > c.mulch_mass_min) & (watavl > 0.0)
    deficit = (
        mulchsat - (mulchwat + reswatadd) + jnp.minimum(mulchwat * c.mulch_evap_fraction, mulchevap_prev)
    )
    add = jnp.maximum(jnp.minimum(deficit, watavl * cover), 0.0)
    add = jnp.where(wet, add, 0.0)
    left = jnp.where(wet, jnp.maximum(watavl - add, 0.0), watavl)
    z = jnp.zeros_like(watavl)
    return MulchRate(
        watavl=jnp.where(on, left, watavl),
        mulwatadd=jnp.where(on, add, z),
        reswatadd=jnp.where(on, reswatadd, z),
        mulchsat=jnp.where(on, mulchsat, z),
    )


def mulch_integrate(
    mulchwat: ArrayLike,
    mulwatadd: ArrayLike,
    reswatadd: ArrayLike,
    mulchevap: ArrayLike,
    on: ArrayLike,
    c: WatbalCoefficients = WATBAL_COEFFICIENTS,
) -> tuple[Array, Array]:
    """``MULCHWATER`` INTEGR: ``MULCHWAT + MULWATADD + RESWATADD - MULCHEVAP``, cut at 1e-4 mm.

    Returns ``(mulchwat, dropped)``: the new mulch water [mm] and the amount the cut removed [mm]
    (not accounted for by the reference; booked by the ledger as a truncation).

    Source: DSSAT-CSM v4.8.6.0 Soil/Mulch/MULCHWAT.for:157-167.
    """
    on = jnp.asarray(on, dtype=bool)
    new = jnp.asarray(mulchwat) + mulwatadd + reswatadd - mulchevap
    cut = new < c.mulch_wat_min
    dropped = jnp.where(on & cut, new, 0.0)
    return jnp.where(on, jnp.where(cut, 0.0, new), mulchwat), dropped


# --------------------------------------------------------------------------- RNOFF
class RunoffResult(NamedTuple):
    """Result of :func:`rnoff`."""

    runoff: Array  # mm d-1
    smx: Array  # mm, soil retention of the curve number
    iabs: Array  # -, initial-abstraction ratio


def rnoff(
    cn: ArrayLike,
    ll: ArrayLike,
    sat: ArrayLike,
    sw: ArrayLike,
    watavl: ArrayLike,
    cover: ArrayLike,
    mulch_on: ArrayLike,
    pm_fraction: ArrayLike = 0.0,
    c: WatbalCoefficients = WATBAL_COEFFICIENTS,
) -> RunoffResult:
    """``RNOFF``: Williams-SCS curve-number runoff with Ritchie's abstraction index.

    ``ll``, ``sat``, ``sw`` are ``[..., n_layer]`` (only layers 1 and 2 are read); ``mulch_on``
    applies the mulch-cover increase of the abstraction ratio (``MEINF`` in ``'RSM'``);
    ``pm_fraction`` is the plastic-mulch cover (``GET('PM','PMFRACTION')``, 0 without plastic).

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilWater/RNOFF.for:76-116.
    """
    sw = jnp.asarray(sw)
    ll = jnp.asarray(ll)
    sat = jnp.asarray(sat)
    watavl = jnp.asarray(watavl)
    smx = c.scs_smx_scale * (_div(c.scs_cn_max, cn) - 1.0)
    frac = (sat[..., :2] - sw[..., :2]) / jnp.maximum(sat[..., :2] - ll[..., :2] * c.swabi_ll_fraction, _TINY)
    swabi = jnp.maximum(0.0, c.swabi_scale * (frac[..., 0] + frac[..., 1]))
    iabs_m = jnp.maximum(swabi, swabi + (c.mulch_iabs_max - swabi) * cover)
    iabs = jnp.where(jnp.asarray(mulch_on, dtype=bool), iabs_m, swabi)
    pb = watavl - iabs * smx
    den = jnp.maximum(watavl + (1.0 - iabs) * smx, _TINY)
    ro = jnp.where((watavl > c.runoff_watavl_min) & (pb > 0.0), pb**2 / den, 0.0)
    pm_fraction = jnp.asarray(pm_fraction)
    ro = jnp.where(pm_fraction > c.plastic_fraction_min, watavl * pm_fraction + ro * (1.0 - pm_fraction), ro)
    return RunoffResult(runoff=ro, smx=smx, iabs=iabs)


# --------------------------------------------------------------------------- INFIL
class InfilResult(NamedTuple):
    """Result of :func:`infil` and :func:`satflo`."""

    swdelts: Array  # [..., n_layer] cm3 cm-3, change of water content by infiltration / drainage
    drn: Array  # [..., n_layer] cm d-1, drainage through the bottom of each layer
    drain: Array  # mm d-1, drainage out of the profile (DRAIN)
    excs: Array  # cm d-1, excess water that did not infiltrate (added to runoff)
    lost: Array  # cm d-1, excess below 1e-4 cm dropped by INFIL's redistribution (not accounted for)


def _first(dl: Array) -> Array:
    """``[..., n]`` one-hot of layer 1."""
    return jnp.arange(dl.shape[-1]) == 0


def infil(
    dlayr: ArrayLike,
    ds: ArrayLike,
    dul: ArrayLike,
    sat: ArrayLike,
    sw: ArrayLike,
    swcn: ArrayLike,
    swcon: ArrayLike,
    pinf: ArrayLike,
    actwtd: ArrayLike,
    c: WatbalCoefficients = WATBAL_COEFFICIENTS,
) -> InfilResult:
    """``INFIL``: the day's infiltration ``PINF`` [cm] distributed down the profile.

    Layer by layer (a :func:`depth_scan` down the profile), water fills the layer to saturation
    and drains ``SWCON (SAT - DUL) DLAYR`` (0.9 of that in the top layer), capped by
    ``24 SWCN`` when the saturated conductivity is given; a layer that would exceed saturation
    under that cap pushes the excess back into the layers above (an inner reverse scan, stopped
    once the excess is below 1e-4 cm), the top layer's excess becoming ``EXCS``. Below the
    wetting front a layer at or above ``DUL + 0.003`` drains as on a dry day (only above the water
    table ``actwtd``). ``DRAIN`` is what leaves the last layer.

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilWater/INFIL.for:44-147.
    """
    dl = jnp.asarray(dlayr)
    sw = jnp.asarray(sw)
    shp = sw.shape
    n = shp[-1]
    idx = jnp.arange(n)
    active = dl > 0.0
    top = _first(dl)
    swcon = jnp.asarray(swcon)
    wtd = jnp.asarray(actwtd)[..., None]
    sat_a = jnp.broadcast_to(jnp.asarray(sat), shp)
    b = lambda v: _moved(jnp.broadcast_to(v, shp))  # noqa: E731
    xs = (
        idx,
        b(active),
        b(dl),
        b(sat_a),
        b(dul),
        b(jnp.where(top, c.top_drain_factor, 1.0)),
        b(jnp.asarray(swcn) * HOURS_PER_DAY),
        b(jnp.asarray(swcn) > 0.0),
        b((jnp.asarray(ds) > wtd) & (wtd > 0.0)),
        b((jnp.asarray(ds) < wtd) & (wtd > 0.0)),
    )
    zero = jnp.zeros(shp[:-1], sw.dtype)

    def redistribute(
        tmpexcs: Array, swtemp: Array, drn: Array, lay: Array
    ) -> tuple[Array, Array, Array, Array]:
        """Lines 91-104: push ``tmpexcs`` into the layers above ``lay`` (an inner reverse scan)."""

        def up(
            carry: tuple[Array, Array, Array], x: tuple[Array, ...]
        ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array]]:
            rem, stopped, excs_add = carry
            k, sw_k, drn_k, sat_k, dl_k = x
            below = k < lay
            stopped = stopped | (below & (rem < c.excess_min))
            go = (~stopped) & below
            hold = jnp.minimum((sat_k - sw_k) * dl_k, rem)
            sw_new = jnp.where(go, sw_k + _div(hold, dl_k), sw_k)
            drn_new = jnp.where(go, jnp.maximum(drn_k - rem, 0.0), drn_k)
            rem_new = jnp.where(go, rem - hold, rem)
            excs_add = jnp.where(go & (k == 0) & (rem_new > 0.0), excs_add + rem_new, excs_add)
            return (rem_new, stopped, excs_add), (sw_new, drn_new)

        xs_up = (idx, _moved(swtemp), _moved(drn), _moved(sat_a), _moved(jnp.broadcast_to(dl, shp)))
        init = (tmpexcs, jnp.zeros(tmpexcs.shape, bool), jnp.zeros_like(tmpexcs))
        (rem, stopped, excs_add), (sw_up, drn_up) = depth_scan(up, init, xs_up, reverse=True)
        return _back(sw_up), _back(drn_up), excs_add, jnp.where(stopped, rem, 0.0)

    def step(carry: tuple[Array, ...], x: tuple[Array, ...]) -> tuple[tuple[Array, ...], None]:
        pinf_c, swtemp, drn, excs, lost = carry
        lay, act, dl_l, sat_l, dul_l, fac_l, cap_l, capped_l, below_l, above_l = x
        onehot = idx == lay
        sw_l = jnp.sum(jnp.where(onehot, swtemp, 0.0), axis=-1)
        hold = (sat_l - sw_l) * dl_l
        wet = (pinf_c > c.infil_pinf_min) & (pinf_c > hold)
        # ---- wetting front (lines 55-109)
        drn_w = pinf_c - hold + fac_l * swcon * (sat_l - dul_l) * dl_l
        drn_w = jnp.where(capped_l & (drn_w > cap_l), cap_l, drn_w)
        drn_w = jnp.where(below_l, 0.0, drn_w)
        sw_w = sw_l + _div(pinf_c - drn_w, dl_l)
        tmpexcs = (sw_w - sat_l) * dl_l
        over = sw_w > sat_l
        sw_w = jnp.where(over, sat_l, sw_w)
        excs_w = jnp.where(over & (lay == 0) & (tmpexcs > 0.0), excs + tmpexcs, excs)
        swtemp_w = jnp.where(onehot, sw_w[..., None], swtemp)
        drn_wv = jnp.where(onehot, drn_w[..., None], drn)
        push = over & (lay > 0)
        sw_r, drn_r, excs_r, lost_r = redistribute(jnp.where(push, tmpexcs, 0.0), swtemp_w, drn_wv, lay)
        swtemp_w = jnp.where(push[..., None], sw_r, swtemp_w)
        drn_wv = jnp.where(push[..., None], drn_r, drn_wv)
        excs_w = jnp.where(push, excs_w + excs_r, excs_w)
        lost_w = jnp.where(push, lost + lost_r, lost)
        # ---- below the front (lines 111-139)
        sw_d = sw_l + _div(pinf_c, dl_l)
        drains = (sw_d >= dul_l + c.dul_margin) & above_l
        drcm_d = fac_l * (sw_d - dul_l) * swcon * dl_l
        drcm_d = jnp.where(capped_l & (drcm_d > cap_l), cap_l, drcm_d)
        drn_d = jnp.where(drains, drcm_d, 0.0)
        sw_d = jnp.where(drains, sw_d - _div(drcm_d, dl_l), sw_d)
        swtemp_d = jnp.where(onehot, sw_d[..., None], swtemp)
        drn_dv = jnp.where(onehot, drn_d[..., None], drn)
        # ---- select; a padded layer (dlayr = 0) passes everything on unchanged
        new = (
            jnp.where(wet, drn_w, drn_d),
            jnp.where(wet[..., None], swtemp_w, swtemp_d),
            jnp.where(wet[..., None], drn_wv, drn_dv),
            jnp.where(wet, excs_w, excs),
            jnp.where(wet, lost_w, lost),
        )
        keep = (act, act[..., None], act[..., None], act, act)
        return tuple(jnp.where(k, a, o) for k, a, o in zip(keep, new, carry, strict=True)), None

    init = (jnp.asarray(pinf, sw.dtype) + zero, sw, jnp.zeros_like(sw), zero, zero)
    (pinf_e, swtemp, drn, excs, lost), _ = depth_scan(step, init, xs)
    return InfilResult(
        swdelts=jnp.where(active, swtemp - sw, 0.0),
        drn=drn,
        drain=pinf_e * MM_PER_CM,
        excs=excs,
        lost=lost,
    )


# --------------------------------------------------------------------------- SATFLO
def satflo(
    dlayr: ArrayLike,
    dul: ArrayLike,
    sat: ArrayLike,
    sw: ArrayLike,
    swcn: ArrayLike,
    swcon: ArrayLike,
    c: WatbalCoefficients = WATBAL_COEFFICIENTS,
) -> InfilResult:
    """``SATFLO``: drainage of the layers above ``DUL + 0.003`` on a day without infiltration.

    Downward pass (a :func:`depth_scan`): ``DRN(L) = max(DRN(L-1) + DRMX(L) - HOLD(L), 0)`` with
    ``DRMX = SWCON (SW - DUL) DLAYR`` and ``HOLD`` the deficit below ``DUL``, capped by
    ``24 SWCN``; upward pass (a reverse :func:`depth_scan`) from the last layer: a layer that the
    inflow from above would bring above saturation reduces that inflow.

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilWater/SATFLO.for:46-116.
    """
    dl = jnp.asarray(dlayr)
    sw = jnp.asarray(sw)
    n = dl.shape[-1]
    idx = jnp.arange(n)
    active = dl > 0.0
    swcon = jnp.asarray(swcon)[..., None]
    drmx = jnp.where(sw >= dul + c.dul_margin, jnp.maximum(0.0, (sw - dul) * swcon * dl), 0.0)
    hold = jnp.where(sw < dul, (dul - sw) * dl, 0.0)
    cap24 = jnp.asarray(swcn) * HOURS_PER_DAY
    capped = jnp.asarray(swcn) > 0.0

    def down(prev: Array, x: tuple[Array, ...]) -> tuple[Array, Array]:
        k, drmx_k, hold_k, cap_k, capped_k = x
        d = jnp.where(k == 0, drmx_k, jnp.maximum(prev + drmx_k - hold_k, 0.0))
        d = jnp.where(capped_k & (d > cap_k), cap_k, d)
        return d, d

    shp = sw.shape
    b = lambda v: _moved(jnp.broadcast_to(v, shp))  # noqa: E731
    zero = jnp.zeros(shp[:-1], sw.dtype)
    _, drn = depth_scan(down, zero, (idx, b(drmx), b(hold), b(cap24), b(capped)))
    drn = jnp.where(active, _back(drn), 0.0)
    last = active & ~jnp.concatenate([active[..., 1:], jnp.zeros_like(active[..., :1])], axis=-1)
    drn_last = jnp.sum(jnp.where(last, drn, 0.0), axis=-1)
    drn_above = jnp.concatenate([jnp.zeros_like(drn[..., :1]), drn[..., :-1]], axis=-1)

    def up(drn_l: Array, x: tuple[Array, ...]) -> tuple[Array, tuple[Array, Array]]:
        k, sw_k, sat_k, dl_k, above_k, act_k = x
        go = act_k & (k >= 1)
        swt = sw_k + _div(above_k - drn_l, dl_k)
        oversat = (swt - sat_k) * dl_k
        big = oversat > above_k
        swt_o = jnp.where(big, swt - _div(above_k, dl_k), sat_k)
        above_o = jnp.where(big, 0.0, above_k - oversat)
        swt = jnp.where(oversat > 0.0, swt_o, swt)
        above_n = jnp.where(oversat > 0.0, above_o, above_k)
        # carry: the drainage out of the layer above (possibly reduced); outputs: this layer's
        # water content and its final drainage
        return jnp.where(go, above_n, drn_l), (jnp.where(go, swt, sw_k), drn_l)

    drn0, (swt, drn_fin) = depth_scan(
        up, drn_last, (idx, b(sw), b(sat), b(dl), b(drn_above), b(active)), reverse=True
    )
    swt = _back(swt)
    drn_fin = jnp.where(idx == 0, drn0[..., None], _back(drn_fin))
    drn_fin = jnp.where(active, drn_fin, 0.0)
    swt = jnp.where(idx == 0, sw - _div(drn0[..., None], dl), swt)
    return InfilResult(
        swdelts=jnp.where(active, swt - sw, 0.0),
        drn=drn_fin,
        drain=drn_last_final(drn_fin, last) * MM_PER_CM,
        excs=zero,
        lost=zero,
    )


def drn_last_final(drn: Array, last: Array) -> Array:
    """``DRN(NLAYR)``: the drainage out of the last layer.

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilWater/SATFLO.for:109 (``DRAIN = DRN(NLAYR) * 10.0``).
    """
    return jnp.sum(jnp.where(last, drn, 0.0), axis=-1)


# --------------------------------------------------------------------------- UP_FLOW
class UpflowResult(NamedTuple):
    """Result of :func:`up_flow`."""

    swdeltu: Array  # [..., n_layer] cm3 cm-3, change of water content by the unsaturated flow
    upflow: Array  # [..., n_layer] cm d-1, flow from layer L+1 into layer L (negative: downward)


def up_flow(
    dlayr: ArrayLike,
    dul: ArrayLike,
    ll: ArrayLike,
    sat: ArrayLike,
    sw: ArrayLike,
    sw_avail: ArrayLike,
    c: WatbalCoefficients = WATBAL_COEFFICIENTS,
) -> UpflowResult:
    """``UP_FLOW``: unsaturated flow between neighbouring layers driven by the gradient of the
    relative extractable water, with ``DBAR = 0.88 exp(35.4 THETA_mean)`` (at most 100 cm2 d-1).

    Pairs ``(L, L+1)`` from layer 1 (layer 2 when the top layer is thinner than 5 cm) to
    ``NLAYR - 1``, in order (a :func:`depth_scan`): each pair sees the water content its upper
    layer got from the pair above. ``sw`` is the day's starting water content, ``sw_avail`` the
    water after infiltration or drainage (``max(0, SW + SWDELTS)``), which limits the flow.

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilWater/WBSUBS.for:283-396 (UP_FLOW).
    """
    dl = jnp.asarray(dlayr)
    sw = jnp.asarray(sw)
    n = dl.shape[-1]
    idx = jnp.arange(n)
    active = dl > 0.0
    ll = jnp.asarray(ll)
    sat = jnp.asarray(sat)
    esw = jnp.asarray(dul) - ll
    avail = jnp.maximum(0.0, jnp.asarray(sw_avail) - ll)
    ist = jnp.where(dl[..., 0] >= c.upflow_top_min_dlayr, 0, 1)

    def nxt(v: Array) -> Array:
        return jnp.concatenate([v[..., 1:], jnp.zeros_like(v[..., :1])], axis=-1)

    shp = sw.shape
    b = lambda v: _moved(jnp.broadcast_to(v, shp))  # noqa: E731
    xs = (
        idx,
        b(dl),
        b(nxt(dl)),
        b(dul),
        b(ll),
        b(nxt(ll)),
        b(esw),
        b(nxt(esw)),
        b(sat),
        b(nxt(sat)),
        b(avail),
        b(nxt(avail)),
        b(nxt(sw)),
        b(nxt(jnp.asarray(sw_avail))),
        b(nxt(active)),
    )

    def step(
        carry: tuple[Array, Array], x: tuple[Array, ...]
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        swl, infl = carry  # this layer's water content and SW_INF as left by the pair above
        k, dl_l, dl_m, dul_l, ll_l, ll_m, esw_l, esw_m, sat_l, sat_m, av_l, av_m, sw_m0, inf_m0, act_m = x
        go = act_m & (k >= ist)
        swm, infm = sw_m0, inf_m0
        swold = swl
        t1 = jnp.maximum(0.0, jnp.minimum(swl - ll_l, esw_l))
        t2 = jnp.maximum(0.0, jnp.minimum(swm - ll_m, esw_m))
        dsum = jnp.maximum(dl_l + dl_m, _TINY)
        dbar = c.upflow_dbar0 * jnp.exp(
            c.upflow_dbar_slope * ((t1 * dl_l + t2 * dl_m) / dsum) * c.upflow_mean_factor
        )
        dbar = jnp.minimum(dbar, c.upflow_dbar_max)
        grad = (_div(t2, esw_m) - _div(t1, esw_l)) * (esw_m * dl_m + esw_l * dl_l) / dsum
        q = dbar * grad / (dsum * c.upflow_mean_factor)
        # ---- upward (lines 345-364)
        wet_l = swl <= dul_l
        swl_u = swl + _div(q, dl_l)
        infl_u = infl + _div(q, dl_l)
        fix = (swl_u > dul_l) | (infl_u > sat_l)
        flowfix = jnp.minimum(
            q, jnp.maximum(jnp.maximum(0.0, (swl_u - dul_l) * dl_l), (infl_u - sat_l) * dl_l)
        )
        q_u = jnp.where(fix, q - flowfix, q)
        swl_u = jnp.where(fix, swold + _div(q_u, dl_l), swl_u)
        q_u = jnp.where(wet_l, q_u, 0.0)
        swl_u = jnp.where(wet_l, swl_u, swl)
        lim = _div(q_u, dl_m) > av_m
        q_u = jnp.where(lim, av_m * dl_m, q_u)
        swl_u = jnp.where(lim, swold + _div(q_u, dl_l), swl_u)
        swm_u = swm - _div(q_u, dl_m)
        # ---- downward (lines 368-388)
        dry_ok = swl >= ll_l
        q_d = jnp.where(jnp.abs(_div(q, dl_l)) > av_l, -av_l * dl_l, q)
        swl_d = swl + _div(q_d, dl_l)
        swm_d = swm - _div(q_d, dl_m)
        infm_d = infm - _div(q_d, dl_m)
        over = infm_d > sat_m
        ff = jnp.minimum(jnp.abs(q_d), (infm_d - sat_m) * dl_m)
        q_d2 = jnp.where(over, q_d + ff, q_d)
        swl_d = jnp.where(over, swold + _div(q_d2, dl_l), swl_d)
        swm_d = jnp.where(over, swm_d - _div(ff, dl_m), swm_d)
        q_d2 = jnp.where(dry_ok, q_d2, 0.0)
        swl_d = jnp.where(dry_ok, swl_d, swl)
        swm_d = jnp.where(dry_ok, swm_d, swm)
        infm_d = jnp.where(dry_ok, infm_d, infm)
        # ---- select
        pos, neg = q > 0.0, q < 0.0
        q_f = jnp.where(pos, q_u, jnp.where(neg, q_d2, q))
        swl_f = jnp.where(pos, swl_u, jnp.where(neg, swl_d, swl))
        swm_f = jnp.where(pos, swm_u, jnp.where(neg, swm_d, swm))
        infm_f = jnp.where(neg, infm_d, infm)
        # a skipped pair leaves layer L as it is and hands layer M its starting values on
        return (jnp.where(go, swm_f, swm), jnp.where(go, infm_f, infm)), (
            jnp.where(go, swl_f, swl),
            jnp.where(go, q_f, 0.0),
        )

    init = (sw[..., 0], jnp.asarray(sw_avail)[..., 0])
    # the scan emits layer L's final value at pair L; the pair after the last layer is skipped
    # (its lower layer is padding), so that entry is the last layer's final value too
    _, (swt, q) = depth_scan(step, init, xs)
    swt = jnp.where(active, _back(swt), sw)
    q = _back(q)
    return UpflowResult(swdeltu=jnp.where(active, swt - sw, 0.0), upflow=jnp.where(active, q, 0.0))


# --------------------------------------------------------------------------- integration
def integrate_sw(
    sw: ArrayLike,
    dlayr: ArrayLike,
    dlayr_yest: ArrayLike,
    es: ArrayLike,
    deltas: ArrayLike,
    ritchie_es: ArrayLike,
    c: WatbalCoefficients = WATBAL_COEFFICIENTS,
    *,
    real4: bool = False,
) -> tuple[Array, Array]:
    """``WATBAL`` INTEGR: remove the soil evaporation from layer 1, add the day's changes of water
    content (on yesterday's layer thickness, as depths), convert back with today's thickness and
    round to 1e-6 (a content below 1e-4 becomes 0). With ``real4`` (a static switch) the rounded
    content is stored as DSSAT stores it, in REAL*4 (the nearest ``float32``, promoted exactly;
    :func:`~agrijax.core.grad.real4_store`, which XLA cannot fold away on GPU), so a layer dried
    to a REAL*4 lower limit equals it exactly, as in the reference.

    ``es`` [mm d-1] is SPAM's soil evaporation (removed here when ``ritchie_es``: ``MESEV /= 'S'``),
    ``deltas`` [..., n_layer] the sum ``SWDELTS + SWDELTU + SWDELTL + SWDELTX + SWDELTT + SWDELTW``
    [cm3 cm-3]. Returns ``(sw_new, rounding)``: the new water content and the water the rounding
    added to each layer [cm] (``(rounded - unrounded) DLAYR``; the reference does not account for it).

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilWater/WATBAL.for:467-506.
    """
    sw = jnp.asarray(sw)
    dy = jnp.asarray(dlayr_yest)
    dl = jnp.asarray(dlayr)
    active = dl > 0.0
    top = _first(dl)
    es_sw = jnp.where(
        top & jnp.asarray(ritchie_es, dtype=bool)[..., None],
        CM_PER_MM * jnp.asarray(es)[..., None] / jnp.maximum(dy, _TINY),
        0.0,
    )
    sw1 = sw - es_sw
    sw_mm = sw1 * dy * MM_PER_CM + deltas * dy * MM_PER_CM
    raw = sw_mm / jnp.maximum(dl, _TINY) / MM_PER_CM
    new = round_st(raw, c.sw_decimals)
    new = jnp.where(jnp.abs(new) < c.sw_zero, 0.0, new)
    if real4:  # static switch: the REAL*4 store of ANINT(SW * 1E6) / 1E6 (WATBAL.for:503-505)
        new = real4_store(new)  # reduce-precision: survives XLA (a convert pair does not on GPU)
    new = jnp.where(active, new, sw)
    return new, jnp.where(active, (new - raw) * dl, 0.0)
