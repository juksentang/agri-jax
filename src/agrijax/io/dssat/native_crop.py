"""Host-side: CERES-Maize's inputs of a DSSAT-CSM v4.8.6.0 maize run from the public files, without
running DSSAT (NumPy).

What CERES-Maize reads from ``DSSAT48.INP`` and the crop weather it reads from the weather module
(printed in ``Weather.OUT``), rebuilt from the experiment's files by the input chain of the input
module, each value through the same print / read round trip (:mod:`agrijax.io.dssat._f77`):

* **Cultivar** (``InputModule/IPVAR.for``; ``InputModule/optempy2k.for`` format 1800, line 827):
  ``P1`` ``F6.1``, ``P2`` ``F6.3``, ``P5`` ``F6.1``, ``G2`` ``F6.1``, ``G3`` ``F6.2``, ``PHINT``
  ``F6.2`` of the cultivar's row of the ``.CUL`` file (read as ``REAL``), and its ecotype ``ECO#``.
* **Planting** (``InputModule/ipexp.for`` ``IPPLNT_Inp``, 972-1030; ``optempy2k.for`` format 70,
  775): ``YRPLT`` (``PDATE`` through ``Y4K_DOY``), ``PLTPOP`` = ``PPOE`` (``PPOP`` when ``PPOE`` is
  not positive) ``F6.1``, ``ROWSPC`` = ``PLRS`` ``F5.0`` (``100 / SQRT(PLTPOP)`` when not
  positive), ``SDEPTH`` = ``PLDP`` ``F5.1``.
* **Soil** (``InputModule/IPSOIL_Inp.for`` 404, 514, 580; ``optempy2k.for`` 520, 555-575, formats
  980): the root growth factor ``SHF`` (``SRGF``) matched onto the simulated layers (``LMATCH``)
  ``F5.3``, the fertility factor ``SLPF`` ``F5.2``; the layer depths and limits are those of
  :func:`agrijax.io.dssat.native_soil.native_soil` (the same ``DSSAT48.INP`` values).
* **Crop weather** (``Weather/OPWEATH.for`` format 300, 147-149): ``TMAX``, ``TMIN``, ``SRAD``
  ``F6.1`` and ``CO2`` ``F7.1`` of the daily weather record
  (:func:`agrijax.io.dssat.native_weather.native_weather`), the values the crop reads.
* **Weather file** (``InputModule/ipexp.for`` 697-707): ``<WSTA><YY>01.WTH`` for a four-character
  station, ``<WSTA><WSTA1>.WTH`` for an eight-character one.
* **Simulation start** (``ipexp.for`` 653-660): ``SDATE`` (``START = S``) or the planting date
  (``START = P``).

The values equal the ones read from a ``DSSAT48.INP`` / ``Weather.OUT`` of the same run on the
example treatments (``tests/integration/test_dssat_free_inputs.py``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from agrijax.core.units import CM_PER_M

from ._f77 import f32, write_f
from .observed import y4k_date

__all__ = [
    "CUL_INP_FORMAT",
    "crop_weather",
    "last_weather_day",
    "native_cultivar",
    "native_planting",
    "native_soil_crop",
    "season_start",
    "weather_file_name",
]

#: ``DSSAT48.INP`` edit descriptors ``(w, d)`` of the CERES-Maize cultivar line (format 1800)
CUL_INP_FORMAT: dict[str, tuple[int, int]] = {
    "P1": (6, 1),
    "P2": (6, 3),
    "P5": (6, 1),
    "G2": (6, 1),
    "G3": (6, 2),
    "PHINT": (6, 2),
}
#: planting line (format 70): ``PLTPOP`` ``F6.1``, ``ROWSPC`` ``F5.0``, ``SDEPTH`` ``F5.1``
_PLTPOP_F, _ROWSPC_F, _SDEPTH_F = (6, 1), (5, 0), (5, 1)
#: soil lines (format 980 and the layer format): ``SHF`` ``F5.3``, ``SLPF`` ``F5.2``
_SHF_F, _SLPF_F = (5, 3), (5, 2)
#: ``Weather.OUT`` (format 300): ``SRAD``, ``TMAX``, ``TMIN`` ``F6.1``, ``CO2`` ``F7.1``
_WEATHER_F, _CO2_F = (6, 1), (7, 1)
#: the largest planting depth the input module accepts (ipexp.for:1027) [cm]
_SDEPTH_MAX = 100.0
_MISSING = -99.0


def _printed(x: Any, fmt: tuple[int, int]) -> float:
    """The number a module reads (list-directed, as :func:`agrijax.sites.dssat_inputs.read_inp`) after
    the input module printed the ``REAL`` ``x`` with ``F<w>.<d>``."""
    text = write_f(f32(x), *fmt)
    if "*" in text:
        raise ValueError(f"DSSAT48.INP field overflows: {float(x)!r} in F{fmt[0]}.{fmt[1]}")
    return float(text)


def _num(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x


def native_cultivar(cul: str | Path, cultivar: str) -> dict[str, Any]:
    """``{"P1".."PHINT": float, "ECO": str}`` of ``cultivar`` (``VAR#``) of the ``.CUL`` file ``cul``
    as CERES-Maize reads it from ``DSSAT48.INP``."""
    from .genotype import read_cul

    t = read_cul(cul)
    ids = [str(i).strip() for i in t.index]
    if cultivar not in ids:
        raise ValueError(f"cultivar {cultivar} not in {Path(cul).name}")
    row = t.iloc[ids.index(cultivar)]
    out: dict[str, Any] = {}
    for name, fmt in CUL_INP_FORMAT.items():
        v = _num(row[name])
        if not math.isfinite(v):
            raise ValueError(f"cultivar {cultivar}: {name} missing in {Path(cul).name}")
        out[name] = _printed(v, fmt)
    out["ECO"] = str(row["ECO#"]).strip()
    return out


def native_planting(x: Mapping[str, Any], trno: int, first_weather: int | None = None) -> dict[str, Any]:
    """``{"yrplt", "pltpop", "rowspc", "sdepth"}`` of treatment ``trno`` of the FileX ``x``
    (:func:`agrijax.io.dssat.read_filex`) as CERES-Maize reads them from ``DSSAT48.INP``."""
    from .native_management import treatment_levels

    tr = treatment_levels(x, trno)
    lev = int(tr.get("MP", 0) or 0)
    p = x.get("PLANTING DETAILS", {}).get(lev) if lev else None
    if not p:
        raise ValueError(f"treatment {trno}: no *PLANTING DETAILS level {lev}")
    d = y4k_date(int(_num(p["PDATE"])), first_weather=first_weather)
    plants, ppoe = f32(_num(p.get("PPOP", _MISSING))), f32(_num(p.get("PPOE", _MISSING)))
    pltpop = ppoe if ppoe > 0 else plants
    if not pltpop > 0:
        raise ValueError(f"treatment {trno}: no plant population (PPOP / PPOE; DSSAT stops, IPPLNT 1001)")
    rowspc = f32(_num(p.get("PLRS", _MISSING)))
    if not rowspc > 0:
        # ROWSPC = 1 / SQRT(PLTPOP) * 100 (ipexp.for:1016): plants m-2 -> row spacing in cm
        rowspc = f32(f32(1.0) / f32(np.sqrt(np.float32(pltpop))) * f32(CM_PER_M))
    sdepth = f32(_num(p.get("PLDP", _MISSING)))
    if not 0 < sdepth <= _SDEPTH_MAX:
        raise ValueError(f"treatment {trno}: planting depth PLDP {sdepth} (DSSAT stops, IPPLNT 1027)")
    return {
        "yrplt": d.year * 1000 + d.timetuple().tm_yday,
        "pltpop": _printed(pltpop, _PLTPOP_F),
        "rowspc": _printed(rowspc, _ROWSPC_F),
        "sdepth": _printed(sdepth, _SDEPTH_F),
    }


def native_soil_crop(profile: Any, soil: Any) -> dict[str, Any]:
    """``{"ds", "ll", "dul", "sat", "shf", "slpf"}`` of the soil ``profile``
    (:func:`agrijax.io.dssat.read_sol`) on the layers of ``soil``
    (:func:`agrijax.io.dssat.native_soil.native_soil`) as CERES-Maize reads them from ``DSSAT48.INP``."""
    from .native_soil import lmatch, lyrset2, lyrset3

    n = profile.n_layers
    z = np.asarray(soil.zlayr, dtype=np.float32)
    if "SRGF" not in profile.layers:
        raise ValueError(f"soil {profile.id}: no root growth factor SRGF (DSSAT stops, IPSOIL_Inp 514)")
    srgf = np.asarray(profile.layers["SRGF"], dtype=np.float64)[:n]
    if np.any(~np.isfinite(srgf)) or np.any(srgf < 0):
        raise ValueError(f"soil {profile.id}: root growth factor SRGF missing (DSSAT stops, IPSOIL_Inp 514)")
    ds = lyrset3(z) if soil.mesol == "3" else lyrset2(z)  # the layers LMATCH matches onto (REAL*4)
    shf = lmatch(z, srgf.astype(np.float32), ds)
    slpf = _num(profile.surface.get("SLPF", _MISSING))
    return {
        "ds": np.asarray(soil.ds, dtype=float),
        "ll": np.asarray(soil.ll, dtype=float),
        "dul": np.asarray(soil.dul, dtype=float),
        "sat": np.asarray(soil.sat, dtype=float),
        "shf": np.asarray([_printed(v, _SHF_F) for v in shf]),
        "slpf": _printed(slpf if math.isfinite(slpf) else _MISSING, _SLPF_F),
    }


def crop_weather(tmax: Any, tmin: Any, srad: Any, co2: Any) -> dict[str, np.ndarray]:
    """``{"tmax", "tmin", "srad", "co2"}`` [float64] of the daily weather record (REAL*4 values, e.g.
    :func:`agrijax.io.dssat.native_weather.native_weather`) as ``Weather.OUT`` prints them."""

    def p(v: Any, fmt: tuple[int, int]) -> np.ndarray:
        return np.asarray([_printed(x, fmt) for x in np.asarray(v, dtype=np.float32)], dtype=float)

    return {
        "tmax": p(tmax, _WEATHER_F),
        "tmin": p(tmin, _WEATHER_F),
        "srad": p(srad, _WEATHER_F),
        "co2": p(co2, _CO2_F),
    }


#: ``WSTA`` characters that name the station (the rest, when given, names the file)
_WSTA_STATION = 4


def weather_file_name(wsta: str, year: int) -> str:
    """The weather file DSSAT opens for station ``wsta`` (FileX ``*FIELDS``) and simulation year
    ``year`` (ipexp.for:697-707)."""
    w = str(wsta).strip()
    if len(w) <= _WSTA_STATION:
        return f"{w:<4}"[:4] + f"{year % 100:02d}01.WTH"
    return f"{w[:8]}.WTH"


def season_start(x: Mapping[str, Any], trno: int, first_weather: int | None = None) -> int:
    """``YRSIM`` (``YYYYDDD``) of treatment ``trno``: ``SDATE`` for ``START = S``, the planting date for
    ``START = P`` (ipexp.for:653-660); ``START = E`` (emergence) is refused."""
    from .native_management import treatment_levels

    tr = treatment_levels(x, trno)
    sc = x.get("SIMULATION CONTROLS", {})
    gen = sc.get(int(tr.get("SM", 0) or 0), sc.get(1, {})).get("GENERAL", {})
    start = str(gen.get("START", "S") or "S").strip().upper()[:1] or "S"
    if start == "P":
        return int(native_planting(x, trno, first_weather)["yrplt"])
    if start != "S":
        raise NotImplementedError(f"START = {start}: simulation start at emergence is not supported")
    d = y4k_date(int(_num(gen["SDATE"])), first_weather=first_weather)
    return d.year * 1000 + d.timetuple().tm_yday


def last_weather_day(wth: str | Path, century: int | None = None) -> int:
    """The last day (``YYYYDDD``) of the weather that continues from the file ``wth`` through the
    station's next-year files (``<INSI><YY>01.WTH``), as the native weather reads it."""
    from .native_weather import _next_year_file
    from .wth import read_wth

    p = Path(wth)
    last = None
    while p.is_file():
        df = read_wth(p, dssat_spans=True, century=century)
        d = df["date"].iloc[-1]
        last = int(d.strftime("%Y%j"))
        try:
            p = _next_year_file(p, d.year + 1)
        except FileNotFoundError:
            break
        century = (d.year + 1) // 100
    if last is None:
        raise FileNotFoundError(f"weather file {p} not found")
    return last
