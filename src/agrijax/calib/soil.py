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

Paths default to the bucket's parameter tree (``soil.cn``, ``soil.swcon``, ``soil.ll`` ...,
:class:`agrijax.processes.soil_water.bucket.watbal.BucketSoil`); pass ``prefix`` for another tree.
"""

from __future__ import annotations

from .space import Domain, OrderedChain, ParamSpec, positive

__all__ = [
    "CN_BOUNDS",
    "DSSAT_LAYER_GAP",
    "SWCON_BOUNDS",
    "U_UPPER",
    "cn_spec",
    "layer_water_chain",
    "swcon_spec",
    "u_spec",
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


def cn_spec(prefix: str = "soil") -> ParamSpec:
    """``CN`` (``SLRO``) with the bounds of the DSSAT estimation table."""
    return ParamSpec(
        f"{prefix}.cn", *CN_BOUNDS, "-", f"{_RGS}: 'runoff curve number ranges from 61 to 94'", label="CN",
        domain=_CN_DOMAIN,
    )  # fmt: skip


def swcon_spec(prefix: str = "soil") -> ParamSpec:
    """``SWCON`` (``SLDR``) with the bounds of the DSSAT drainage classes."""
    return ParamSpec(
        f"{prefix}.swcon", *SWCON_BOUNDS, "d-1",
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
    prefix: str = "soil",
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
            f"{prefix}.{leaf}", lo, hi, "cm3 cm-3", bounds_source, label=n, domain=_THETA_DOMAIN, index=layer
        )
        for leaf, n, (lo, hi) in zip(("ll", "dul", "sat"), names, (ll, dul, sat), strict=True)
    ]
    chain = OrderedChain(
        names,
        gap,
        f"design choice: DSSAT's inversion-repair offset 0.01 ({_SOILDYN}:1382-1395) as minimum gap",
    )
    return specs, chain
