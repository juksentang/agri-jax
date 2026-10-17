"""Calibratable soil parameters of the DSSAT tipping bucket: bounds with sources, valid domains and
the joint ordered transform of a layer's ``LL < DUL < SAT``.

DSSAT ``.SOL`` files have no ``MINIMA`` / ``MAXIMA`` rows (unlike the cultivar files), so the
bounds below come from the DSSAT soil-parameter estimation tables and the DSSAT-CSM source; each
spec says which. Where no documented range was found, nothing is invented: the caller passes the
bound and its source.

* ``CN`` (``SLRO``, runoff curve number): bounds 61-94, the range of the DSSAT estimation table
  by hydrologic group and slope (Ritchie, Godwin & Singh 1990, "Soil and weather inputs for the
  IBSNAT crop models", IBSNAT Symposium Proceedings Part I, Univ. Hawaii; reproduced in Gijsman
  et al. 2007, Comput. Electron. Agric. 56:85-100, and Romero et al. 2012, Environ. Model. Softw.
  35:163-170, Tables 3-4: "runoff curve number ranges from 61 to 94"). Valid domain 25-98: the
  range DSSAT-CSM v4.8.6.0 ``Soil/SoilUtilities/SOILDYN.for:453-466`` leaves ``CN`` in before the
  run (a value <= 0 is first replaced by 80, then ``CN`` is clamped to [25, 98]).
* ``SWCON`` (``SLDR``, drainage coefficient, fraction d-1): bounds 0.01-0.85, the seven drainage
  classes of the same table (very poorly drained 0.01 ... excessively drained 0.85; Romero et al.
  2012 Table 3 after Ritchie et al. 1990). Valid domain ``(0, 1]``: a daily fraction; SOILDYN.for:
  479-488 replaces values below 1e-4 by the default 0.25.
* ``U`` (``SLU1``, stage-1 soil evaporation limit, mm): only an upper bound is documented, 12 mm
  (Romero et al. 2012 Table 3, "less or equal to 12.0", after FAO 1990 World Soil Resources Report
  60). No sourced lower bound was found (DSSAT's default is 6 mm, SOILDYN.for:469-477), so
  :func:`u_spec` needs the caller's lower bound and its source. Valid domain ``(0, inf)``
  (SOILDYN.for:469 replaces ``U < 1e-4`` by 6 mm).
* ``LL``, ``DUL``, ``SAT`` of one layer (volumetric, cm3 cm-3): valid domain ``[0, 1]``; they are
  calibrated as one :class:`~agrijax.calib.space.OrderedChain` with a minimum gap of 0.01 cm3
  cm-3. The gap is **our design choice**, not a separation DSSAT enforces: DSSAT-CSM accepts
  ``DUL = SAT`` and ``LL = DUL`` and only repairs an inversion (SOILDYN.for:1382-1395, "Protection
  for LL < DUL < SAT": ``DUL > SAT`` gives ``SAT = DUL + 0.01``, ``LL > DUL`` gives
  ``LL = DUL - 0.01``); we take that repair offset as the minimum gap, so a calibrated profile
  never sits on an equality. Their bounds are site choices (a box around the ``.SOL`` values or a
  pedotransfer range): the caller passes them with their source.

* ``SLPF`` (soil photosynthesis factor, ``-``): bounds 0.70-1.00, the range of the 124 profiles of
  the ``SOIL.SOL`` distributed with DSSAT-CSM v4.8.6.0 (``example_data/Soil``) without the one 0.00
  (IBSG910010, a placeholder: DSSAT reads ``SLPF < 1e-4`` as 1.0, SOILDYN.for:443-444). Valid
  domain ``(0, 1]`` (``Data/SOIL.CDE``: "0 to 1 scale").
* ``U``: the same ``SOIL.SOL`` gives ``SLU1`` 2.0-12.7 mm and all 3826 profiles of the distributed
  ``example_data/Soil/*.SOL`` a smallest value of 1.0 mm (:data:`U_SOIL_SOL_RANGE`,
  :data:`U_ALL_SOL_MIN`): sourced lower bounds a caller may pass to :func:`u_spec`.
  :func:`soil_space` uses the 1.0 mm of all profiles: the ``SOIL.SOL`` minimum 2.0 mm is the value
  of some sites (the Millhopper sand IBMZ910014 among them), which would then sit on the bound.
* ``SRGF`` of a layer (root growth factor, the crop's ``SHF``): valid domain ``[0, 1]``
  (``Data/SOIL.CDE``: "Root growth factor, soil only, 0.0 to 1.0"); no documented range, the
  bounds are a box around the site value (:data:`SRGF_SITE_BOX`) or the caller's.
* ``SWCN`` of a layer (``SSKS``, saturated hydraulic conductivity, cm h-1): log transform, positive
  domain. A layer whose ``.SOL`` value is missing (-99) has no conductivity cap in the bucket
  (``swcn <= 0``, ``bucket/kernels.py`` INFIL and SATFLO); giving it a value would switch the cap on,
  a change of structure rather than of a parameter, so :func:`swcn_spec` refuses such a layer.

Paths default to the bucket's parameter tree (``soil.cn``, ``soil.swcon``, ``soil.ll`` ...,
:class:`agrijax.processes.soil_water.bucket.watbal.BucketSoil`); pass ``prefix`` for another tree.

**Spaces by ``.SOL`` layer.** :func:`soil_space` builds the space of a site's soil from its ``.SOL``
values (one value per layer of the soil file, not per model layer: DSSAT splits thick file layers,
and two model layers cut from one file layer are one soil property). Its paths are the keys of
that mapping (``"cn"``, ``"ll"`` with ``index`` = the ``.SOL`` layer, 0 the top one), the tree
:func:`agrijax.calib.day_soil.with_soil` reads; ``prefix`` puts them under another node. Bounds
come from :class:`Box` (fixed), :class:`RelativeBox` (around the site value) or
:class:`WidenedBox` (a table range widened to reach around the site value), each with its source;
:data:`SITE_BOUNDS` is one set of such boxes (a design choice, documented there). The site value
must lie inside its bounds, and every layer's ``LL + gap <= DUL``, ``DUL + gap <= SAT`` must hold at
the site values and stay feasible against the members that are not calibrated.

**A site value on a bound.** In the logit coordinate a value on (or very near) a bound maps to a
saturated ``z`` (about +-20 at the inverse transform's edge), where the derivative of the value
with respect to ``z`` is about 1e-9: a gradient method started at the site cannot move it, and
nothing can be drawn around it on one side. :func:`soil_space` therefore refuses a site value that
lies within :data:`SITE_MARGIN` of the width of its logistic box from either end (for an ordered
chain member the box starts at the previous member's site value plus the gap) unless the caller
accepts it (``site_on_bound=True``). The domain caps make this unavoidable for some quantities:
``SLPF = 1.0`` (IBMZ910013) and ``SRGF = 1.0`` (the top layers of most profiles) cannot lie inside a
box that stays inside ``(0, 1]`` and ``[0, 1]``.

**Leaving such coordinates out, on record.** A site value equal to an end of its quantity's valid
domain is not a missing value but a physical limit (``SRGF = 1.0``: roots grow unhindered;
``SLPF = 1.0``: no fertility reduction; ``SRGF = 0`` in a layer roots never enter), and no box
around it stays in the domain. :func:`without_domain_bounds` takes such coordinates out of a
requested set before :func:`soil_space` builds it and returns what it took out
(:class:`ExcludedCoordinate`: quantity, layer, value, the domain end and its source), so the
dimension of a space and the reason it shrank are both on record; nothing is dropped silently.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from agrijax.core.coefficients import numerical_guard

from .space import Domain, OrderedChain, ParamSpace, ParamSpec, positive

__all__ = [
    "CN_BOUNDS",
    "DOMAINS",
    "DSSAT_LAYER_GAP",
    "LAYER_QUANTITIES",
    "SCALAR_QUANTITIES",
    "SITE_BOUNDS",
    "SITE_MARGIN",
    "SLPF_BOUNDS",
    "SOIL_QUANTITIES",
    "SRGF_SITE_BOX",
    "SWCN_SITE_BOX",
    "SWCON_BOUNDS",
    "U_ALL_SOL_MIN",
    "U_ALL_SOL_SOURCE",
    "U_SOIL_SOL_RANGE",
    "U_SOIL_SOL_SOURCE",
    "U_UPPER",
    "Box",
    "ExcludedCoordinate",
    "RelativeBox",
    "WidenedBox",
    "cn_spec",
    "layer_water_chain",
    "slpf_spec",
    "soil_space",
    "srgf_spec",
    "swcn_spec",
    "swcon_spec",
    "u_spec",
    "without_domain_bounds",
]

_RGS = (
    "Ritchie, Godwin & Singh 1990 (IBSNAT Symposium Proc. Part I) estimation table, as reproduced in "
    "Romero et al. 2012 (Environ. Model. Softw. 35:163-170) Tables 3-4 and Gijsman et al. 2007"
)
_SOILDYN = "DSSAT-CSM v4.8.6.0 Soil/SoilUtilities/SOILDYN.for"

#: runoff curve number bounds: the DSSAT estimation table by hydrologic group A-D and slope
CN_BOUNDS = (61.0, 94.0)
#: drainage coefficient bounds [d-1]: very poorly drained (0.01) to excessively drained (0.85)
SWCON_BOUNDS = (0.01, 0.85)
#: documented upper limit of the stage-1 evaporation limit [mm] (Romero et al. 2012 Table 3, FAO 1990)
U_UPPER = 12.0
#: minimum separation of LL, DUL and SAT in one layer [cm3 cm-3]: a design choice, DSSAT-CSM's
#: repair offset for an inverted profile (SOILDYN.for:1382-1395), not a separation DSSAT enforces
DSSAT_LAYER_GAP = 0.01
#: ``SLU1`` range [mm] of the 124 profiles of DSSAT-CSM v4.8.6.0 ``example_data/Soil/SOIL.SOL``
U_SOIL_SOL_RANGE = (2.0, 12.7)
#: smallest ``SLU1`` [mm] of the 3826 profiles of DSSAT-CSM v4.8.6.0 ``example_data/Soil/*.SOL``
U_ALL_SOL_MIN = 1.0
#: ``SLPF`` range of the 124 ``SOIL.SOL`` profiles, the placeholder 0.00 of IBSG910010 left out
SLPF_BOUNDS = (0.70, 1.00)

#: rounding allowed when a layer's order is checked at the site values (decimals of the soil file)
_CHAIN_TOL = numerical_guard("calib.soil_chain_tol", 1e-12, "rounding allowed in the site-value order check")
#: smallest distance of a site value from either end of its logistic box, as a fraction of the box
#: width, below which :func:`soil_space` refuses it unless accepted (``|z| <= ln(99)``, about 4.6,
#: where ``d theta / d z`` is still about 1 % of the width)
SITE_MARGIN = numerical_guard(
    "calib.soil_site_margin", 0.01, "fraction of a box width a site value keeps from its bounds"
)
_SOIL_SOL = "DSSAT-CSM v4.8.6.0 example_data/Soil/SOIL.SOL (124 profiles)"
_SOIL_CDE = "DSSAT-CSM v4.8.6.0 Data/SOIL.CDE"
#: source of :data:`U_SOIL_SOL_RANGE`
U_SOIL_SOL_SOURCE = f"{_SOIL_SOL}: SLU1 ranges from 2.0 (IBMZ910014 and three more) to 12.7 mm (UFPT930001)"
#: source of :data:`U_ALL_SOL_MIN`
U_ALL_SOL_SOURCE = (
    "DSSAT-CSM v4.8.6.0 example_data/Soil/*.SOL (3826 profiles): smallest SLU1 1.0 mm (CPLI000021, CP.SOL)"
)

_CN_DOMAIN = Domain(
    25.0,
    98.0,
    source=f"{_SOILDYN}:453-466: CN <= 0 is set to 80, then clamped to [25, 98] "
    "(AMIN1(CN,98.0), AMAX1(CN,25.0))",
)
_SWCON_DOMAIN = Domain(
    0.0,
    1.0,
    lower_open=True,
    source=f"fraction of the water above DUL drained per day; {_SOILDYN}:479-488 replaces SWCON < 1e-4 "
    "by 0.25",
)
_U_DOMAIN = positive(f"{_SOILDYN}:469-477 replaces U < 1e-4 by 6 mm")
_THETA_DOMAIN = Domain(0.0, 1.0, source="volumetric water content (cm3 cm-3)")
_SLPF_DOMAIN = Domain(
    0.0,
    1.0,
    lower_open=True,
    source=f"{_SOIL_CDE}:74 'Soil fertility factor, 0 to 1 scale'; "
    f"{_SOILDYN}:443-444 reads SLPF < 1e-4 as 1.0",
)
_SRGF_DOMAIN = Domain(0.0, 1.0, source=f"{_SOIL_CDE}:89 'Root growth factor, soil only, 0.0 to 1.0'")
_SWCN_DOMAIN = positive(
    "saturated hydraulic conductivity (cm h-1); swcn <= 0 means 'not given, no cap' in the bucket "
    "(processes/soil_water/bucket/kernels.py INFIL, SATFLO)"
)


def _path(prefix: str | None, leaf: str) -> str:
    return leaf if not prefix else f"{prefix}.{leaf}"


def cn_spec(
    prefix: str | None = "soil", *, bounds: tuple[float, float] | None = None, bounds_source: str = ""
) -> ParamSpec:
    """``CN`` (``SLRO``) with the bounds of the DSSAT estimation table, or ``bounds`` with their
    ``bounds_source`` (required), e.g. for a site whose value lies outside 61-94."""
    if bounds is None:
        lo, hi = CN_BOUNDS
        src = f"{_RGS}: 'runoff curve number ranges from 61 to 94'"
    else:
        if not bounds_source.strip():
            raise ValueError("cn_spec: bounds other than the estimation table need their source")
        lo, hi = bounds
        src = bounds_source
    return ParamSpec(_path(prefix, "cn"), lo, hi, "-", src, label="CN", domain=_CN_DOMAIN)


def swcon_spec(prefix: str | None = "soil") -> ParamSpec:
    """``SWCON`` (``SLDR``) with the bounds of the DSSAT drainage classes."""
    return ParamSpec(
        _path(prefix, "swcon"), *SWCON_BOUNDS, "d-1",
        f"{_RGS}: drainage classes very poorly drained 0.01 ... excessively drained 0.85",
        label="SWCON", domain=_SWCON_DOMAIN,
    )  # fmt: skip


def u_spec(lower: float, lower_source: str, path: str = "u", upper: float = U_UPPER) -> ParamSpec:
    """``U`` (``SLU1``): the documented upper bound 12 mm and the caller's lower bound, which must
    come with its source (no documented lower bound was found)."""
    if not lower_source.strip():
        raise ValueError(
            "u_spec: no documented lower bound of U exists here; pass the one you use and its source"
        )
    src = f"upper: Romero et al. 2012 Table 3 (after FAO 1990): U <= 12 mm; lower: {lower_source}"
    if upper != U_UPPER:
        src = f"upper {upper:g} and lower bound: {lower_source}"
    return ParamSpec(path, lower, upper, "mm", src, label="U", domain=_U_DOMAIN)


def layer_water_chain(
    layer: int,
    ll: tuple[float, float],
    dul: tuple[float, float],
    sat: tuple[float, float],
    bounds_source: str,
    *,
    prefix: str | None = "soil",
    gap: float = DSSAT_LAYER_GAP,
) -> tuple[list[ParamSpec], OrderedChain]:
    """Specs of ``LL``, ``DUL``, ``SAT`` of one layer (``index = layer`` of the layer arrays) and
    their ordered chain (``LL + gap <= DUL``, ``DUL + gap <= SAT`` for every ``z``).

    ``ll`` / ``dul`` / ``sat`` are the ``(lower, upper)`` bounds, a site choice with
    ``bounds_source``; the chain needs ``upper(LL) + gap < upper(DUL)`` and
    ``upper(DUL) + gap < upper(SAT)``."""
    if not bounds_source.strip():
        raise ValueError("layer_water_chain: the LL / DUL / SAT bounds need their source")
    names = (f"LL{layer}", f"DUL{layer}", f"SAT{layer}")
    specs = [
        ParamSpec(
            _path(prefix, leaf), lo, hi, "cm3 cm-3", bounds_source, label=n, domain=_THETA_DOMAIN, index=layer
        )
        for leaf, n, (lo, hi) in zip(("ll", "dul", "sat"), names, (ll, dul, sat), strict=True)
    ]
    chain = OrderedChain(
        names,
        gap,
        f"design choice: DSSAT's inversion-repair offset 0.01 ({_SOILDYN}:1382-1395) as minimum gap",
    )
    return specs, chain


def slpf_spec(
    prefix: str | None = "soil",
    *,
    bounds: tuple[float, float] = SLPF_BOUNDS,
    bounds_source: str = "",
) -> ParamSpec:
    """``SLPF`` (soil photosynthesis factor, the crop's ``soil.slpf``) with the ``SOIL.SOL`` range,
    or ``bounds`` with their ``bounds_source`` (required when not the default)."""
    if tuple(bounds) == SLPF_BOUNDS and not bounds_source.strip():
        bounds_source = (
            f"{_SOIL_SOL}: SLPF ranges from 0.70 to 1.00 without the placeholder 0.00 of IBSG910010 "
            f"({_SOILDYN}:443-444 reads it as 1.0)"
        )
    if not bounds_source.strip():
        raise ValueError("slpf_spec: bounds other than the SOIL.SOL range need their source")
    return ParamSpec(_path(prefix, "slpf"), *bounds, "-", bounds_source, label="SLPF", domain=_SLPF_DOMAIN)


# ------------------------------------------------------------------------ bounds around a site value
@dataclass(frozen=True)
class Box:
    """Fixed bounds ``[lower, upper]`` with their source."""

    lower: float
    upper: float
    source: str

    def bounds(self, site: float) -> tuple[float, float]:
        """The bounds (the site value plays no part)."""
        return float(self.lower), float(self.upper)


@dataclass(frozen=True)
class RelativeBox:
    """Bounds around a site value ``v``: ``[v lower_factor, v upper_factor]``, intersected with
    ``[v - half_width, v + half_width]`` when ``half_width`` is given, then raised to ``floor`` and
    lowered to ``cap`` when given; ``source`` says why."""

    lower_factor: float
    upper_factor: float
    source: str
    half_width: float | None = None
    floor: float | None = None
    cap: float | None = None

    def bounds(self, site: float) -> tuple[float, float]:
        """The bounds around ``site``."""
        v = float(site)
        lo, hi = v * self.lower_factor, v * self.upper_factor
        if self.half_width is not None:
            lo, hi = max(lo, v - self.half_width), min(hi, v + self.half_width)
        if self.floor is not None:
            lo = max(lo, self.floor)
        if self.cap is not None:
            hi = min(hi, self.cap)
        return lo, hi


@dataclass(frozen=True)
class WidenedBox:
    """A table range ``[lower, upper]`` widened to reach ``margin`` around the site value ``v``:
    ``[min(lower, v - margin), max(upper, v + margin)]``; ``source`` says where the range and the
    margin come from."""

    lower: float
    upper: float
    margin: float
    source: str

    def bounds(self, site: float) -> tuple[float, float]:
        """The bounds for ``site``."""
        v = float(site)
        return min(self.lower, v - self.margin), max(self.upper, v + self.margin)


BoundsRule = Box | RelativeBox | WidenedBox

_DESIGN = "design choice, not a documented range"
#: ``SRGF`` box around the site value: x [0.5, 1.5], kept inside the valid domain [0, 1]
SRGF_SITE_BOX = RelativeBox(
    0.5, 1.5, f"{_DESIGN}: site SRGF x [0.5, 1.5], clipped to its domain [0, 1]", floor=0.0, cap=1.0
)
#: ``SWCN`` box around the site value: x [1/4, 4] (a factor of four either way on a log scale)
SWCN_SITE_BOX = RelativeBox(0.25, 4.0, f"{_DESIGN}: site SSKS x [1/4, 4] (a factor of 4 on a log scale)")
_WATER = (
    f"{_DESIGN}: site value x [0.6, 1.4] intersected with site +- 0.05 cm3 cm-3 (a relative box narrows "
    "on sands, where +- 0.05 would be +- 58 % of DUL 0.086, the absolute half-width caps it on clays)"
)
#: one set of boxes around a site's ``.SOL`` values (a design choice; each box carries its reason)
SITE_BOUNDS: dict[str, BoundsRule] = {
    "cn": WidenedBox(
        *CN_BOUNDS,
        10.0,
        f"estimation-table range 61-94 ({_RGS}) widened to the site value +- 10 ({_DESIGN}: a site value "
        "outside 61-94 still lies inside the box)",
    ),
    "ll": RelativeBox(0.6, 1.4, f"{_WATER}; LL at least 0.01", half_width=0.05, floor=0.01),
    "dul": RelativeBox(0.6, 1.4, _WATER, half_width=0.05),
    "sat": RelativeBox(0.6, 1.4, _WATER, half_width=0.05),
    "srgf": SRGF_SITE_BOX,
    "swcn": SWCN_SITE_BOX,
}


def _missing(v: Any) -> bool:
    x = float(v)
    return not math.isfinite(x) or x <= 0.0


def srgf_spec(
    layer: int, site_value: float, rule: BoundsRule = SRGF_SITE_BOX, *, prefix: str | None = "soil"
) -> ParamSpec:
    """``SRGF`` (root growth factor) of ``.SOL`` layer ``layer`` (``index``), bounds from ``rule``
    around ``site_value``."""
    lo, hi = rule.bounds(site_value)
    return ParamSpec(
        _path(prefix, "srgf"), lo, hi, "-", rule.source, label=f"SRGF_H{layer + 1}", domain=_SRGF_DOMAIN,
        index=layer,
    )  # fmt: skip


def swcn_spec(
    layer: int, site_value: float, rule: BoundsRule = SWCN_SITE_BOX, *, prefix: str | None = "soil"
) -> ParamSpec:
    """``SWCN`` (``SSKS``, cm h-1) of ``.SOL`` layer ``layer`` (``index``), log transform, bounds from
    ``rule`` around ``site_value``. A missing site value (-99, NaN or <= 0) is refused: the layer
    then has no conductivity cap, and a value would switch it on (module docstring).

    The log transform (``theta = lower + exp(z)``, :class:`~agrijax.calib.space.ParamSpec`) keeps the
    lower bound; the upper bound is recorded for reports and starting points, not enforced."""
    if _missing(site_value):
        raise ValueError(
            f"swcn_spec: SSKS of .SOL layer {layer} is missing (no conductivity cap in the model); "
            "a value there would switch the cap on, a change of structure: leave the layer out"
        )
    lo, hi = rule.bounds(site_value)
    return ParamSpec(
        _path(prefix, "swcn"), lo, hi, "cm h-1", rule.source, transform="log", label=f"SWCN_H{layer + 1}",
        domain=_SWCN_DOMAIN, index=layer,
    )  # fmt: skip


# ------------------------------------------------------------------------ a site's soil by .SOL layer
#: the soil quantities of one value per site
SCALAR_QUANTITIES: tuple[str, ...] = ("cn", "swcon", "u", "slpf")
#: the soil quantities of one value per ``.SOL`` layer
LAYER_QUANTITIES: tuple[str, ...] = ("ll", "dul", "sat", "srgf", "swcn")
#: every quantity :func:`soil_space` and :func:`agrijax.calib.day_soil.with_soil` know
SOIL_QUANTITIES: tuple[str, ...] = SCALAR_QUANTITIES + LAYER_QUANTITIES
_CHAIN = ("ll", "dul", "sat")


def _default_scalar_rule(q: str) -> BoundsRule:
    if q == "cn":
        return Box(*CN_BOUNDS, f"{_RGS}: 'runoff curve number ranges from 61 to 94'")
    if q == "swcon":
        return Box(*SWCON_BOUNDS, f"{_RGS}: drainage classes very poorly drained 0.01 ... excessively 0.85")
    if q == "u":
        return Box(U_ALL_SOL_MIN, U_UPPER, f"lower: {U_ALL_SOL_SOURCE}; upper: Romero et al. 2012 Table 3")
    return Box(*SLPF_BOUNDS, f"{_SOIL_SOL}: SLPF 0.70-1.00 without the placeholder 0.00 of IBSG910010")


def _scalar_spec(q: str, rule: BoundsRule, site: float, prefix: str | None) -> ParamSpec:
    lo, hi = rule.bounds(site)
    if q == "cn":
        return cn_spec(prefix, bounds=(lo, hi), bounds_source=rule.source)
    if q == "swcon":
        return ParamSpec(
            _path(prefix, "swcon"), lo, hi, "d-1", rule.source, label="SWCON", domain=_SWCON_DOMAIN
        )
    if q == "u":
        return u_spec(lo, rule.source, path=_path(prefix, "u"), upper=hi)
    return slpf_spec(prefix, bounds=(lo, hi), bounds_source=rule.source)


def soil_space(
    site: Mapping[str, Any],
    quantities: Sequence[str],
    layers: Mapping[str, Sequence[int]] | Sequence[int] = (),
    bounds: Mapping[str, BoundsRule] | None = None,
    *,
    prefix: str | None = None,
    gap: float = DSSAT_LAYER_GAP,
    site_on_bound: bool = False,
) -> ParamSpace:
    """The parameter space of a site's soil by ``.SOL`` layer (module docstring).

    ``site`` holds the site's ``.SOL`` values: the scalars of :data:`SCALAR_QUANTITIES` and, for
    :data:`LAYER_QUANTITIES`, one value per ``.SOL`` layer (top first; e.g.
    :attr:`agrijax.calib.day_soil.SoilLayerMap.sol`). ``quantities`` are the calibrated ones;
    ``layers`` the ``.SOL`` layers (0 = the top one) of the layered ones: one sequence for all of
    them, or ``{quantity: layers}``. ``bounds`` maps a quantity to its :class:`Box`,
    :class:`RelativeBox` or :class:`WidenedBox` (e.g. :data:`SITE_BOUNDS`); a layered quantity needs
    one, the scalars default to the documented ranges (``CN`` 61-94, ``SWCON`` 0.01-0.85, ``U``
    :data:`U_ALL_SOL_MIN` lower and 12 mm upper, ``SLPF`` :data:`SLPF_BOUNDS`).

    Order: the scalars in :data:`SCALAR_QUANTITIES` order, then layer by layer (top first)
    ``LL``, ``DUL``, ``SAT`` (one :class:`~agrijax.calib.space.OrderedChain` per layer with two or
    three of them), ``SRGF``, ``SWCN``. Labels: ``CN``, ``SWCON``, ``U``, ``SLPF``, ``LL_H<k>``,
    ``DUL_H<k>``, ``SAT_H<k>``, ``SRGF_H<k>``, ``SWCN_H<k>`` with ``k`` the ``.SOL`` layer counted from
    1. Raises ``ValueError`` when a site value lies outside its bounds or a layer's order cannot hold
    (at the site values, or against a member that is not calibrated), and when a site value lies on
    or within :data:`SITE_MARGIN` of an end of its logistic box unless ``site_on_bound`` accepts
    that (module docstring)."""
    bounds = dict(bounds or {})
    unknown = sorted(set(quantities) - set(SOIL_QUANTITIES))
    if unknown:
        raise ValueError(f"soil_space: unknown quantities {unknown} (known: {list(SOIL_QUANTITIES)})")
    if len(set(quantities)) != len(quantities):
        raise ValueError(f"soil_space: a quantity is listed twice in {list(quantities)}")
    layered = [q for q in LAYER_QUANTITIES if q in quantities]
    if isinstance(layers, Mapping):
        extra = sorted(set(layers) - set(layered))
        if extra:
            raise ValueError(f"soil_space: layers given for {extra}, which are not among the quantities")
        per = {q: tuple(int(h) for h in layers.get(q, ())) for q in layered}
    else:
        per = {q: tuple(int(h) for h in layers) for q in layered}
    problems: list[str] = []
    specs: list[ParamSpec] = []
    for q in SCALAR_QUANTITIES:
        if q not in quantities:
            continue
        v = float(np.asarray(site[q]))
        rule = bounds.get(q, _default_scalar_rule(q))
        sp = _scalar_spec(q, rule, v, prefix)
        problems += _outside(sp, v)
        specs.append(sp)
    cols = {q: np.asarray(site[q], dtype=float).reshape(-1) for q in LAYER_QUANTITIES if q in site}
    n_sol = max((c.size for c in cols.values()), default=0)
    for q in layered:
        if q not in bounds:
            raise ValueError(f"soil_space: {q} needs its bounds (bounds={{'{q}': ...}}, e.g. SITE_BOUNDS)")
        if q not in cols:
            raise ValueError(f"soil_space: the site values have no {q}")
        bad = [h for h in per[q] if not 0 <= h < cols[q].size]
        if bad:
            raise ValueError(f"soil_space: {q} layers {bad} outside the {cols[q].size} .SOL layers")
        if len(set(per[q])) != len(per[q]):
            raise ValueError(f"soil_space: a {q} layer is listed twice in {list(per[q])}")
        if not per[q]:
            raise ValueError(f"soil_space: no .SOL layers given for {q}")
    chains: list[OrderedChain] = []
    for h in range(n_sol):
        here = [q for q in _CHAIN if h in per.get(q, ())]
        layer_specs: dict[str, ParamSpec] = {}
        for q in here:
            v = float(cols[q][h])
            lo, hi = bounds[q].bounds(v)
            sp = ParamSpec(
                _path(prefix, q), lo, hi, "cm3 cm-3", bounds[q].source, label=f"{q.upper()}_H{h + 1}",
                domain=_THETA_DOMAIN, index=h,
            )  # fmt: skip
            problems += _outside(sp, v)
            layer_specs[q] = sp
            specs.append(sp)
        problems += _chain_problems(h, layer_specs, cols, gap)
        if len(here) >= 2:
            chains.append(
                OrderedChain(
                    tuple(layer_specs[q].name for q in here),
                    gap,
                    f"design choice: DSSAT's inversion-repair offset 0.01 ({_SOILDYN}:1382-1395) as "
                    "minimum gap",
                )
            )
        if h in per.get("srgf", ()):
            v = float(cols["srgf"][h])
            sp = srgf_spec(h, v, bounds["srgf"], prefix=prefix)
            problems += _outside(sp, v)
            specs.append(sp)
        if h in per.get("swcn", ()):
            specs.append(swcn_spec(h, float(cols["swcn"][h]), bounds["swcn"], prefix=prefix))
    if problems:
        raise ValueError("soil_space: " + "; ".join(problems))
    space = ParamSpace(specs, ordered=chains)
    if not site_on_bound:
        near = _near_bounds(space, _site_theta(space, site, prefix))
        if near:
            raise ValueError(
                "soil_space: site values on or near a bound of their logistic box (a gradient method "
                f"started there cannot move them; leave them out, widen the bounds, or pass "
                f"site_on_bound=True): {'; '.join(near)}"
            )
    return space


def _site_theta(space: ParamSpace, site: Mapping[str, Any], prefix: str | None) -> np.ndarray:
    """The site values in the order of ``space``."""
    out = []
    for s in space.specs:
        key = s.path if not prefix else s.path[len(prefix) + 1 :]
        v = np.asarray(site[key], dtype=float)
        out.append(float(v.reshape(-1)[s.index]) if s.index is not None else float(v))
    return np.asarray(out)


def _near_bounds(space: ParamSpace, theta: np.ndarray) -> list[str]:
    """Specs whose site value lies within :data:`SITE_MARGIN` of the box width from an end of its
    logistic box (from the lower bound for a log transform)."""
    out: list[str] = []
    for i, s in enumerate(space.specs):
        v = float(theta[i])
        if s.transform == "logit":
            prev = next((ch for ch in space.ordered if s.name in ch.names[1:]), None)
            lo = s.lower
            if prev is not None:
                a = prev.names[prev.names.index(s.name) - 1]
                lo = max(lo, float(theta[space.names.index(a)]) + prev.gap)
            width = s.upper - lo
            u = (v - lo) / width if width > 0.0 else 0.0
            if not SITE_MARGIN <= u <= 1.0 - SITE_MARGIN:
                out.append(f"{s.name} {v:g} in [{lo:g}, {s.upper:g}] ({s.bounds_source})")
        elif s.transform == "log" and v - s.lower < SITE_MARGIN * (s.upper - s.lower):
            out.append(f"{s.name} {v:g} at its lower bound {s.lower:g} ({s.bounds_source})")
    return out


def _outside(sp: ParamSpec, v: float) -> list[str]:
    """The complaint about a site value outside its spec's bounds (empty when inside)."""
    if sp.lower <= v <= sp.upper:
        return []
    box = f"[{sp.lower:g}, {sp.upper:g}]"
    return [f"{sp.name}: site value {v:g} outside its bounds {box} ({sp.bounds_source})"]


def _chain_problems(
    h: int, specs: Mapping[str, ParamSpec], cols: Mapping[str, np.ndarray], gap: float
) -> list[str]:
    """What breaks ``LL + gap <= DUL``, ``DUL + gap <= SAT`` of ``.SOL`` layer ``h``: at the site
    values of the calibrated members, and against the site value of a member that is not
    calibrated (a calibrated neighbour's bounds must leave room for it)."""
    out: list[str] = []
    k = f"H{h + 1}"
    for a, b in (("ll", "dul"), ("dul", "sat")):
        if a not in specs and b not in specs:
            continue
        va, vb = float(cols[a][h]), float(cols[b][h])
        if va + gap > vb + _CHAIN_TOL:
            out.append(f"{k}: site {a.upper()} {va:g} + gap {gap:g} > site {b.upper()} {vb:g}")
        if a in specs and b not in specs and specs[a].upper + gap > vb + _CHAIN_TOL:
            out.append(
                f"{k}: upper {a.upper()} {specs[a].upper:g} + gap {gap:g} > the fixed {b.upper()} {vb:g}"
            )
        if b in specs and a not in specs and specs[b].lower < va + gap - _CHAIN_TOL:
            out.append(
                f"{k}: lower {b.upper()} {specs[b].lower:g} < the fixed {a.upper()} {va:g} + gap {gap:g}"
            )
    return out


# ------------------------------------------------------------------------ coordinates on a domain end
#: the valid domain of every soil quantity (as its spec declares it)
DOMAINS: dict[str, Domain] = {
    "cn": _CN_DOMAIN,
    "swcon": _SWCON_DOMAIN,
    "u": _U_DOMAIN,
    "slpf": _SLPF_DOMAIN,
    "ll": _THETA_DOMAIN,
    "dul": _THETA_DOMAIN,
    "sat": _THETA_DOMAIN,
    "srgf": _SRGF_DOMAIN,
    "swcn": _SWCN_DOMAIN,
}


@dataclass(frozen=True)
class ExcludedCoordinate:
    """A soil coordinate left out of a site's space (:func:`without_domain_bounds`): ``quantity``,
    ``.SOL`` ``layer`` (0 = the top one; ``None`` for a scalar), the site ``value``, the domain end
    ``bound`` it equals and ``reason`` (the domain and its source)."""

    quantity: str
    layer: int | None
    value: float
    bound: float
    reason: str

    @property
    def name(self) -> str:
        """The label the coordinate would have had in :func:`soil_space` (``SRGF_H1``, ``SLPF``)."""
        q = self.quantity.upper()
        return q if self.layer is None else f"{q}_H{self.layer + 1}"


def without_domain_bounds(
    site: Mapping[str, Any],
    quantities: Sequence[str],
    layers: Mapping[str, Sequence[int]] | Sequence[int] = (),
) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]], tuple[ExcludedCoordinate, ...]]:
    """The requested soil coordinates without those whose site value equals an end of the
    quantity's valid domain (:data:`DOMAINS`; module docstring), and the record of what was left out.

    ``site``, ``quantities`` and ``layers`` as for :func:`soil_space`. Returns ``(quantities,
    layers, excluded)``: the quantities still requested (a layered one whose every requested layer
    was left out is dropped), ``{quantity: layers}`` of the layered ones, and one
    :class:`ExcludedCoordinate` per coordinate left out, in request order. Pass the first two to
    :func:`soil_space`. A missing value (``SSKS`` -99) is not a domain end and stays (``soil_space``
    refuses it)."""
    unknown = sorted(set(quantities) - set(SOIL_QUANTITIES))
    if unknown:
        raise ValueError(
            f"without_domain_bounds: unknown quantities {unknown} (known: {list(SOIL_QUANTITIES)})"
        )
    out_q: list[str] = []
    out_l: dict[str, tuple[int, ...]] = {}
    excluded: list[ExcludedCoordinate] = []

    def on_end(q: str, v: float) -> float | None:
        d = DOMAINS[q]
        for end in (d.lower, d.upper):
            if math.isfinite(end) and math.isfinite(v) and abs(v - end) <= _CHAIN_TOL:
                return end
        return None

    def record(q: str, h: int | None, v: float, end: float) -> None:
        d = DOMAINS[q]
        why = f"site value on the end {end:g} of the domain {d} ({d.source})"
        excluded.append(ExcludedCoordinate(q, h, v, end, why))

    for q in quantities:
        if q in SCALAR_QUANTITIES:
            v = float(np.asarray(site[q]))
            end = on_end(q, v)
            if end is None:
                out_q.append(q)
            else:
                record(q, None, v, end)
            continue
        req = tuple(int(h) for h in (layers.get(q, ()) if isinstance(layers, Mapping) else layers))
        col = np.asarray(site[q], dtype=float).reshape(-1)
        keep: list[int] = []
        for h in req:
            v = float(col[h]) if 0 <= h < col.size else math.nan
            end = on_end(q, v)
            if end is None:
                keep.append(h)
            else:
                record(q, h, v, end)
        if keep or not req:
            out_q.append(q)
            out_l[q] = tuple(keep)
    return tuple(out_q), out_l, tuple(excluded)
