"""Host-side: the soil properties and initial soil water of a DSSAT-CSM v4.8.6.0 run, from the public
``.SOL`` and FileX (NumPy).

``dscsm048`` derives the static ``SOILPROP`` the soil-water modules read in three steps, all
reproduced here in the reference's single precision:

1. **Input module** (``InputModule/IPSOIL_Inp.for``): the profile's layer table is read as written;
   a layer with ``SAT = DUL`` gets ``SAT = DUL + 0.01``, one with ``LL = DUL`` gets
   ``LL = DUL - 0.01`` (lines 508-513), a negative ``SSKS`` becomes missing (526-528); the layers
   are redistributed by the ``MESOL`` method (541-545: ``LYRSET2`` by default,
   ``Soil/SoilUtilities/LMATCH.for`` 263-350, which keeps 5 and 15 cm on top and splits thick
   layers into 2, 3 or 4 whole-centimetre layers; ``LYRSET3`` for ``MESOL = 3``: the file's layers)
   and every layer value is depth-averaged onto the new layers (``LMATCH``, LMATCH.for 29-120: a
   value missing in any overlapped layer stays missing). The initial soil water of the FileX
   ``*INITIAL CONDITIONS`` (``IPSLIN.for`` 130-176: ``ICBL`` / ``SH2O`` read with format 60,
   ``LMATCH`` onto the soil layers, a missing value -> ``DUL``; ``INSOIL.for`` 84-86) likewise.
2. **``DSSAT48.INP``**: the input module writes the soil (``optempy2k.for`` 505-560: depth
   ``F5.0``, ``LL`` / ``DUL`` / ``SAT`` / ``SRGF`` ``F5.3``, ``SSKS`` with a width chosen by its
   magnitude; ``SALB`` ``F5.2``, ``SLU1`` ``F5.1``, ``SLDR`` ``F5.2``, ``SLRO`` ``F5.0``, format 980
   at line 810) and the initial water (``optempy2k.for`` 318-334, ``SH2O`` ``F5.3``).
3. **``SOILDYN``** (``Soil/SoilUtilities/SOILDYN.for``) reads that file back (lines 267-300, 354-368):
   the ``REAL`` of each printed number; then ``SALB``, ``CN``, ``U``, ``SWCON`` defaults
   (434-486), ``DLAYR`` from the depths (495-500), and the initial water clipped to ``[LL, SAT]``
   (layer 1: to the air-dry ``0.30 LL`` at least; 369-420).

What a module reads is thus the ``REAL*4`` of a printed decimal; :class:`NativeSoil` holds the
decimal (float64), the convention of :class:`agrijax.models.day_dssat486.DssatSite` (the harness
builds its site from the instrumented ``SOILPROP`` with ``decimal_of``).

Not ported (raised): the ``LYRSET`` redistribution of ``MESOL = 1`` (fixed 5 / 10 / 15 ... cm
increments), a profile deeper than 20 layers after redistribution, and a missing (``-99``)
``SLLL``, ``SDUL`` or ``SSAT`` (DSSAT stops or prints asterisks). Reference: DSSAT-CSM v4.8.6.0
(BSD-3); file, line and routine are cited, the source is not reproduced.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from agrijax.core.coefficients import Coefficients, Provenance, coef

from ._f77 import decimal_of, nint, read_f, round_trip, write_f
from .sol import SoilProfile, read_sol

__all__ = [
    "DSSAT_SOIL_INPUT",
    "NL",
    "NativeSoil",
    "SoilInputCoefficients",
    "find_soil_profile",
    "initial_soil_water",
    "lmatch",
    "lyrset2",
    "lyrset3",
    "native_soil",
    "swcn_width",
]

REF = "dssat-4.8.6.0"
LMATCH_FOR = "Soil/SoilUtilities/LMATCH.for"
IPSOIL_INP = "InputModule/IPSOIL_Inp.for"
SOILDYN_FOR = "Soil/SoilUtilities/SOILDYN.for"
IPSLIN_FOR = "InputModule/IPSLIN.for"
OPTEMPY2K = "InputModule/optempy2k.for"
_ALG = "an algorithm threshold of the input module, not a physical parameter"
_DEF = "default when the soil file leaves the value missing or zero"

#: DSSAT's maximum number of soil layers (``ModuleDefs.for``: ``NL = 20``)
NL = 20
_F = np.float32


def _p(file_line: str, routine: str, note: str = "") -> Provenance:
    return Provenance.at(REF, file_line, routine=routine, note=note)


class SoilInputCoefficients(Coefficients):
    """The thresholds and defaults of the DSSAT-CSM v4.8.6.0 soil input chain (module docstring)."""

    top_layer_bottom: float = coef(
        5.0,
        "cm",
        "bottom of the first soil layer after LYRSET2",
        _p(f"{LMATCH_FOR}:278", "LYRSET2", _ALG),
        calibrate=False,
    )
    second_layer_bottom: float = coef(
        15.0,
        "cm",
        "bottom of the second soil layer after LYRSET2",
        _p(f"{LMATCH_FOR}:279", "LYRSET2", _ALG),
        calibrate=False,
    )
    thin_layer: float = coef(
        2.0,
        "cm",
        "a soil-file layer thinner than this is merged into the next one",
        _p(f"{LMATCH_FOR}:296", "LYRSET2", _ALG),
        calibrate=False,
    )
    deep_1: float = coef(
        90.0,
        "cm",
        "depth from which the thicker layer limits apply",
        _p(f"{LMATCH_FOR}:300", "LYRSET2", _ALG),
        calibrate=False,
    )
    deep_2: float = coef(
        200.0,
        "cm",
        "depth from which the thickest layer limits apply",
        _p(f"{LMATCH_FOR}:303", "LYRSET2", _ALG),
        calibrate=False,
    )
    one_shallow: float = coef(
        15.0,
        "cm",
        "thickest soil-file layer kept as one layer (bottom above deep_1)",
        _p(f"{LMATCH_FOR}:300", "LYRSET2", _ALG),
        calibrate=False,
    )
    one_deep: float = coef(
        30.0,
        "cm",
        "thickest soil-file layer kept as one layer (bottom at or below deep_1)",
        _p(f"{LMATCH_FOR}:302", "LYRSET2", _ALG),
        calibrate=False,
    )
    one_deepest: float = coef(
        90.0,
        "cm",
        "thickest soil-file layer kept as one layer (bottom at or below deep_2)",
        _p(f"{LMATCH_FOR}:303", "LYRSET2", _ALG),
        calibrate=False,
    )
    two_shallow: float = coef(
        30.0,
        "cm",
        "thickest soil-file layer split into two (bottom above deep_1)",
        _p(f"{LMATCH_FOR}:308", "LYRSET2", _ALG),
        calibrate=False,
    )
    two_deep: float = coef(
        60.0,
        "cm",
        "thickest soil-file layer split into two (bottom at or below deep_1)",
        _p(f"{LMATCH_FOR}:309", "LYRSET2", _ALG),
        calibrate=False,
    )
    two_deepest: float = coef(
        180.0,
        "cm",
        "thickest soil-file layer split into two (bottom at or below deep_2)",
        _p(f"{LMATCH_FOR}:310", "LYRSET2", _ALG),
        calibrate=False,
    )
    three_shallow: float = coef(
        45.0,
        "cm",
        "thickest soil-file layer split into three (bottom above deep_1)",
        _p(f"{LMATCH_FOR}:316", "LYRSET2", _ALG),
        calibrate=False,
    )
    three_deep: float = coef(
        90.0,
        "cm",
        "thickest soil-file layer split into three (bottom at or below deep_1)",
        _p(f"{LMATCH_FOR}:317", "LYRSET2", _ALG),
        calibrate=False,
    )
    three_deepest: float = coef(
        270.0,
        "cm",
        "thickest soil-file layer split into three (bottom at or below deep_2)",
        _p(f"{LMATCH_FOR}:318", "LYRSET2", _ALG),
        calibrate=False,
    )
    half: float = coef(
        0.5,
        "-",
        "split fraction of a layer divided into two (and the middle of four)",
        _p(f"{LMATCH_FOR}:312", "LYRSET2", _ALG),
        calibrate=False,
    )
    two_thirds: float = coef(
        0.66667,
        "-",
        "split fraction of the upper third of a layer divided into three",
        _p(f"{LMATCH_FOR}:320", "LYRSET2", _ALG),
        calibrate=False,
    )
    one_third: float = coef(
        0.33333,
        "-",
        "split fraction of the middle third of a layer divided into three",
        _p(f"{LMATCH_FOR}:321", "LYRSET2", _ALG),
        calibrate=False,
    )
    three_quarters: float = coef(
        0.75,
        "-",
        "split fraction of the upper quarter of a layer divided into four",
        _p(f"{LMATCH_FOR}:327", "LYRSET2", _ALG),
        calibrate=False,
    )
    one_quarter: float = coef(
        0.25,
        "-",
        "split fraction of the lowest quarter of a layer divided into four",
        _p(f"{LMATCH_FOR}:329", "LYRSET2", _ALG),
        calibrate=False,
    )
    missing_tol: float = coef(
        0.001,
        "-",
        "a layer value within this of -99 is missing",
        _p(f"{LMATCH_FOR}:65", "LMATCH", _ALG),
        calibrate=False,
    )
    retention_gap: float = coef(
        0.01,
        "cm3 cm-3",
        "SAT = DUL + gap where they are equal, LL = DUL - gap where they are equal",
        _p(f"{IPSOIL_INP}:509", "IPSOIL_Inp", "applied to LL at line 512 too"),
        calibrate=False,
    )
    salb_min: float = coef(
        1.0e-4,
        "-",
        "soil albedo below this is missing",
        _p(f"{SOILDYN_FOR}:434", "SOILDYN", _DEF),
        calibrate=False,
    )
    salb_default: float = coef(
        0.15,
        "-",
        "soil albedo used when SALB is missing",
        _p(f"{SOILDYN_FOR}:435", "SOILDYN", _DEF),
        calibrate=False,
    )
    cn_low: float = coef(
        25.0,
        "-",
        "lowest runoff curve number kept",
        _p(f"{SOILDYN_FOR}:460", "SOILDYN", "clamp of CN"),
        calibrate=False,
    )
    cn_high: float = coef(
        98.0,
        "-",
        "highest runoff curve number kept",
        _p(f"{SOILDYN_FOR}:459", "SOILDYN", "clamp of CN"),
        calibrate=False,
    )
    u_min: float = coef(
        1.0e-4,
        "mm",
        "stage-1 evaporation limit below this is missing",
        _p(f"{SOILDYN_FOR}:469", "SOILDYN", _DEF),
        calibrate=False,
    )
    u_default: float = coef(
        6.0,
        "mm",
        "stage-1 evaporation limit used when SLU1 is missing",
        _p(f"{SOILDYN_FOR}:470", "SOILDYN", _DEF),
        calibrate=False,
    )
    swcon_min: float = coef(
        1.0e-4,
        "d-1",
        "drainage constant below this is missing",
        _p(f"{SOILDYN_FOR}:479", "SOILDYN", _DEF),
        calibrate=False,
    )
    swcon_default: float = coef(
        0.25,
        "d-1",
        "drainage constant used when SLDR is missing",
        _p(f"{SOILDYN_FOR}:480", "SOILDYN", _DEF),
        calibrate=False,
    )
    air_dry_fraction: float = coef(
        0.30,
        "-",
        "air-dry water content of layer 1 as a fraction of LL (lower bound of the initial SW)",
        _p(f"{SOILDYN_FOR}:379", "SOILDYN"),
        calibrate=False,
    )
    sw_init_max: float = coef(
        0.75,
        "cm3 cm-3",
        "an initial soil water above this stops DSSAT",
        _p(f"{IPSLIN_FOR}:171", "IPSLIN"),
        calibrate=False,
    )
    swcn_tiny: float = coef(
        1.0e-6,
        "cm h-1",
        "SSKS below this is printed as missing (F5.0)",
        _p(f"{OPTEMPY2K}:544", "OPTEMPY2K"),
        calibrate=False,
    )


DSSAT_SOIL_INPUT = SoilInputCoefficients()


# ------------------------------------------------------------------------ soil file
def find_soil_profile(profile_id: str, dirs: Sequence[Path]) -> SoilProfile:
    """Host-side: the profile ``profile_id`` as ``dscsm048`` finds it (``InputModule/ipexp.for``
    580-640, ``IPSOIL_Inp.for`` 244-270): ``SOIL.SOL`` of the run directory, else
    ``<first two letters>.SOL``; when ``SOIL.SOL`` lacks the profile, the two-letter file of the same
    directory. ``dirs`` are the directories staged into the run directory, in order (the
    experiment's, then the soil directory's), and are searched as one directory."""
    pid = profile_id.strip().upper()

    def first(name: str) -> Path | None:
        return next((d / name for d in dirs if (d / name).is_file()), None)

    alt = f"{pid[:2]}.SOL"
    soil_sol = first("SOIL.SOL")
    if soil_sol is not None:
        prof = read_sol(soil_sol, dssat_spans=True)
        if pid in prof:
            return prof[pid]
    path = first(alt)
    if path is None:
        raise FileNotFoundError(
            f"soil profile {pid}: neither SOIL.SOL nor {alt} holds it in {[str(d) for d in dirs]}"
        )
    prof = read_sol(path, dssat_spans=True)
    if pid not in prof:
        raise KeyError(f"soil profile {pid} is not in {path}")
    return prof[pid]


# ------------------------------------------------------------------------ layer redistribution
def lyrset2(zlayr: ArrayLike, c: SoilInputCoefficients = DSSAT_SOIL_INPUT) -> np.ndarray:
    """Host-side: ``LYRSET2`` (``LMATCH.for`` 263-350): the new layer bottoms [cm, REAL*4] for the
    soil-file layer bottoms ``zlayr``."""
    z = np.asarray(zlayr, dtype=_F)
    ds = [_F(c.top_layer_bottom), _F(c.second_layer_bottom)]
    lim = (
        (c.one_shallow, c.one_deep, c.one_deepest),
        (c.two_shallow, c.two_deep, c.two_deepest),
        (c.three_shallow, c.three_deep, c.three_deepest),
    )
    for cum in z:
        th = _F(cum - ds[-1])
        if cum <= ds[1] or th < _F(c.thin_layer):
            continue

        def fits(k: int, cum: np.float32 = cum, th: np.float32 = th) -> bool:
            a, b, d = (_F(x) for x in lim[k])
            return bool(
                (cum < _F(c.deep_1) and th <= a)
                or (cum >= _F(c.deep_1) and th <= b)
                or (cum >= _F(c.deep_2) and th <= d)
            )

        if fits(0):
            ds.append(cum)
        elif fits(1):
            ds += [_F(cum - nint(th * _F(c.half))), cum]
        elif fits(2):
            ds += [_F(cum - nint(th * _F(c.two_thirds))), _F(cum - nint(th * _F(c.one_third))), cum]
        else:
            ds += [
                _F(cum - nint(th * _F(c.three_quarters))),
                _F(cum - nint(th * _F(c.half))),
                _F(cum - nint(th * _F(c.one_quarter))),
                cum,
            ]
    if z[-1] > ds[-1]:
        ds[-1] = z[-1]
    if len(ds) > NL:
        raise ValueError(f"LYRSET2 gives {len(ds)} soil layers, more than DSSAT's {NL}")
    return np.asarray(ds, dtype=_F)


def lyrset3(zlayr: ArrayLike) -> np.ndarray:
    """Host-side: ``LYRSET3`` (``LMATCH.for`` 365-392): the soil file's own layers."""
    return np.asarray(zlayr, dtype=_F)


def lmatch(
    dsi: ArrayLike,
    vi: ArrayLike,
    dso: ArrayLike,
    c: SoilInputCoefficients = DSSAT_SOIL_INPUT,
) -> np.ndarray:
    """Host-side: ``LMATCH`` (``LMATCH.for`` 29-120) in REAL*4: the depth-weighted mean of the input
    layers (bottoms ``dsi``, values ``vi``, ``-99`` missing) over each output layer (bottoms
    ``dso``); an output layer overlapping a missing input layer is missing, the part of the last
    output layer below the input profile takes the last input value."""
    dsi_ = np.asarray(dsi, dtype=_F)
    vi_ = np.asarray(vi, dtype=_F)
    dso_ = np.asarray(dso, dtype=_F)
    ni, no = len(dsi_), len(dso_)
    miss = _F(-99.0)
    vs = np.full(no, miss, dtype=_F)
    if ni < 1:
        return vs
    k = 0
    zil = zol = _F(0.0)
    for li in range(no):
        sumz = sumv = _F(0.0)
        missing = False
        last = False
        while True:
            zt = max(zol, zil)
            zb = min(dso_[li], dsi_[k])
            sumz = _F(sumz + _F(zb - zt))
            sumv = _F(sumv + _F(vi_[k] * _F(zb - zt)))
            if abs(float(vi_[k] - miss)) < c.missing_tol and zb > zt:
                missing = True
            if dso_[li] < dsi_[k]:
                break
            if k == ni - 1:
                last = True
                break
            zil = dsi_[k]
            k += 1
        if last:
            # LMATCH.for 101-110 (label 20): the last output layer, and nothing below it
            v = vi_[k]
            if abs(float(sumv - miss)) > c.missing_tol:
                if sumz > 0:
                    v = _F(sumv / sumz)
            else:
                v = miss
            vs[li] = v
            break
        v = vi_[k]
        if sumz > 0:
            v = _F(sumv / sumz)
        vs[li] = miss if missing else v
        zol = dso_[li]
    return vs


# ------------------------------------------------------------------------ DSSAT48.INP round trip
def swcn_width(swcn: float, c: SoilInputCoefficients = DSSAT_SOIL_INPUT) -> tuple[int, int]:
    """Host-side: the ``F5.d`` edit descriptor ``optempy2k.for`` 544-556 prints ``SSKS`` with."""
    v = _F(swcn)
    if v < _F(c.swcn_tiny):
        return 5, 0
    for bound, d in ((0.1, 4), (1.0, 3), (10.0, 2), (100.0, 1)):
        if v < _F(bound):
            return 5, d
    return 5, 0


def _inp(x: Any, w: int, d: int) -> np.float32:
    """The REAL*4 SOILDYN reads back (``F6.0`` fields) after the input module printed ``x`` ``Fw.d``."""
    text = write_f(x, w, d)
    if "*" in text:
        raise ValueError(f"DSSAT48.INP field overflows: {float(x)!r} in F{w}.{d}")
    return read_f(text, 0)


@dataclass(frozen=True)
class NativeSoil:
    """The static ``SOILPROP`` of a run as ``SOILDYN`` holds it (``nl`` layers; decimals, float64)
    and the input-module values the initial soil water is matched against (REAL*4)."""

    profile: str
    mesol: str
    ds: np.ndarray
    dlayr: np.ndarray
    ll: np.ndarray
    dul: np.ndarray
    sat: np.ndarray
    swcn: np.ndarray
    cn: float
    swcon: float
    salb: float
    u: float
    #: the input module's (unrounded) REAL*4 DUL, the default of a missing initial SW
    dul_input: np.ndarray
    #: the soil file's layer bottoms [cm] (before redistribution)
    zlayr: np.ndarray

    @property
    def nl(self) -> int:
        return int(self.ds.shape[0])


def _layer_col(p: SoilProfile, name: str, n: int) -> np.ndarray:
    if name not in p.layers:
        return np.full(n, -99.0, dtype=_F)
    v = np.asarray(p.layers[name], dtype=np.float64)
    return np.where(np.isnan(v), -99.0, v).astype(_F)


def _surface(p: SoilProfile, name: str) -> np.float32:
    v = p.surface.get(name)
    if v is None or (isinstance(v, float) and math.isnan(v)) or not isinstance(v, (int, float)):
        return _F(-99.0)
    return _F(v)


def native_soil(
    profile: SoilProfile, mesol: str = "2", c: SoilInputCoefficients = DSSAT_SOIL_INPUT
) -> NativeSoil:
    """Host-side: the static ``SOILPROP`` of ``profile`` (a :func:`~agrijax.io.dssat.read_sol` profile
    read with ``dssat_spans=True``) under the layer method ``mesol`` (module docstring)."""
    n = profile.n_layers
    if n < 1:
        raise ValueError(f"soil {profile.id}: no layers")
    z = _layer_col(profile, "SLB", n)
    ll, dul, sat = (_layer_col(profile, k, n) for k in ("SLLL", "SDUL", "SSAT"))
    swcn = _layer_col(profile, "SSKS", n)
    for name, v in (("SLLL", ll), ("SDUL", dul), ("SSAT", sat)):
        if np.any(np.abs(v + 99.0) < c.missing_tol):
            raise ValueError(f"soil {profile.id}: {name} is missing in a layer (not supported)")
    if np.any(dul > sat) or np.any(ll > dul) or np.any(dul < 0):
        raise ValueError(
            f"soil {profile.id}: LL <= DUL <= SAT does not hold (DSSAT stops, IPSOIL_Inp 499-507)"
        )
    gap = _F(c.retention_gap)
    sat = np.where(sat == dul, (dul + gap).astype(_F), sat).astype(_F)
    ll = np.where(ll == dul, (dul - gap).astype(_F), ll).astype(_F)
    swcn = np.where(swcn < 0, _F(-99.0), swcn).astype(_F)
    m = (mesol or "2").strip() or "2"
    if m == "1":
        raise NotImplementedError("MESOL = 1 (LYRSET fixed increments) is not ported; use MESOL 2 or 3")
    ds = lyrset3(z) if m == "3" else lyrset2(z, c)
    ll_m, dul_m, sat_m, swcn_m = (lmatch(z, v, ds, c) for v in (ll, dul, sat, swcn))
    # DSSAT48.INP: depth F5.0, LL/DUL/SAT F5.3, SSKS by magnitude; SOILDYN reads F6.0 fields
    ds_r = np.asarray([_inp(x, 5, 0) for x in ds], dtype=_F)
    rd = lambda v: np.asarray([_inp(x, 5, 3) for x in v], dtype=_F)  # noqa: E731
    ll_r, dul_r, sat_r = rd(ll_m), rd(dul_m), rd(sat_m)
    swcn_r = np.asarray([_inp(x, *swcn_width(float(x), c)) for x in swcn_m], dtype=_F)
    dlayr = np.concatenate([ds_r[:1], (ds_r[1:] - ds_r[:-1]).astype(_F)]).astype(_F)
    salb = round_trip(_surface(profile, "SALB"), 5, 2)
    u = round_trip(_surface(profile, "SLU1"), 5, 1)
    swcon = round_trip(_surface(profile, "SLDR"), 5, 2)
    cn = round_trip(_surface(profile, "SLRO"), 5, 0)
    if swcon < 0:
        raise ValueError(f"soil {profile.id}: SLDR < 0 (DSSAT stops, IPSOIL_Inp 477)")
    if cn <= 0:
        raise ValueError(f"soil {profile.id}: SLRO <= 0 (DSSAT stops, IPSOIL_Inp 478)")
    if salb <= 0:
        raise ValueError(f"soil {profile.id}: SALB <= 0 (DSSAT stops, IPSOIL_Inp 479-485)")
    if salb < _F(c.salb_min):
        salb = _F(c.salb_default)
    if cn <= _F(c.cn_low) or cn > _F(c.cn_high):
        cn = _F(min(max(float(cn), c.cn_low), c.cn_high))
    if u < _F(c.u_min):
        u = _F(c.u_default)
    if swcon < _F(c.swcon_min):
        swcon = _F(c.swcon_default)
    return NativeSoil(
        profile=profile.id,
        mesol=m,
        ds=decimal_of(ds_r),
        dlayr=decimal_of(dlayr),
        ll=decimal_of(ll_r),
        dul=decimal_of(dul_r),
        sat=decimal_of(sat_r),
        swcn=decimal_of(swcn_r),
        cn=float(decimal_of(cn)),
        swcon=float(decimal_of(swcon)),
        salb=float(decimal_of(salb)),
        u=float(decimal_of(u)),
        dul_input=dul_m,
        zlayr=z,
    )


# ------------------------------------------------------------------------ initial soil water
def _ic_field(v: Any) -> np.float32:
    """An ``*INITIAL CONDITIONS`` layer value as ``IPSLIN`` format 60 reads it (``F5.0``; missing -99)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return _F(-99.0)
    if isinstance(v, str):
        return read_f(v, 0) if v.strip() else _F(0.0)
    return _F(v)


def initial_soil_water(
    soil: NativeSoil,
    ic_rows: Sequence[Mapping[str, Any]] | None,
    c: SoilInputCoefficients = DSSAT_SOIL_INPUT,
) -> np.ndarray:
    """Host-side: the initial soil water ``SW`` [cm3 cm-3, decimals] ``SOILDYN`` / ``WATBAL`` start from:
    the FileX initial-condition layers ``ic_rows`` (``ICBL`` layer bottom, ``SH2O``; ``None`` or
    empty: no initial conditions, ``DUL`` everywhere) matched onto ``soil`` (module docstring)."""
    nl = soil.nl
    ds = np.asarray(soil.ds, dtype=_F)
    if ic_rows:
        dsi = np.asarray([_ic_field(r.get("ICBL")) for r in ic_rows], dtype=_F)
        swi = np.asarray([_ic_field(r.get("SH2O")) for r in ic_rows], dtype=_F)
        sw = lmatch(dsi, swi, ds, c)
    else:
        sw = np.full(nl, _F(-99.0), dtype=_F)
    if np.any(sw > _F(c.sw_init_max)):
        raise ValueError(f"initial soil water above {c.sw_init_max} (DSSAT stops, IPSLIN 171)")
    dul_in = np.asarray(soil.dul_input, dtype=_F)
    sw = np.where(sw <= 0, dul_in, sw).astype(_F)
    # DSSAT48.INP (F5.3) read back by SOILDYN (F5.1 field: the printed decimal)
    sw_r = np.asarray([_inp(x, 5, 3) for x in sw], dtype=_F)
    ll = np.asarray(soil.ll, dtype=_F)
    sat = np.asarray(soil.sat, dtype=_F)
    out = sw_r.copy()
    for li in range(nl):
        if out[li] < ll[li]:
            if li == 0:
                swad = _F(_F(c.air_dry_fraction) * ll[li])
                if out[li] < swad:
                    out[li] = swad
            else:
                out[li] = ll[li]
    out = np.where(out > sat, sat, out).astype(_F)
    return decimal_of(out)
