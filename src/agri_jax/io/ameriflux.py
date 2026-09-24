"""AmeriFlux eddy-covariance data for CA-TPA: BASE half-hourly and ONEFlux FLUXNET products.

Nothing here imports JAX; every reader returns a :class:`pandas.DataFrame`.

Files (under ``$AGRI_JAX_DATA/ameriflux``, CC-BY-4.0, see :data:`CITATION_BASE`)
--------------------------------------------------------------------------------
* ``base/AMF_CA-TPA_BASE_HH_3-5.csv``: AmeriFlux BASE, half-hourly 2020-01-01..2023-12-31
  (70 128 rows). Two ``# Site: / # Version:`` comment lines, then the header. ``-9999`` =
  missing. :func:`read_base_hh`.
* ``fluxnet/AMF_CA-TPA_FLUXNET_FLUXMET_{HH,DD,WW,MM,YY}_2020-2023_v1.3_r1.csv``: ONEFlux
  FULLSET (gap-filled, partitioned, energy-balance corrected). :func:`read_fluxnet`. The
  ``*_BIFVARINFO_*`` file of the same resolution gives unit, definition and height per
  variable (:func:`read_bifvarinfo`); :func:`read_fluxnet` attaches the units to ``attrs``.

Time: all timestamps are local standard time without DST (README.txt), UTC-5 for CA-TPA
(``UTC_OFFSET`` in the BIF file); indexes are tz-naive. Half-hourly rows are indexed by
``TIMESTAMP_START``; the value covers ``[start, start + 30 min)``.

Units (BASE; FP-Standard units, confirmed against the FLUXNET BIFVARINFO file)
------------------------------------------------------------------------------
:data:`BASE_UNITS`: fluxes and radiation W m-2 (LE, H, NETRAD, G, SW/LW in/out), TA and TS
deg C, RH %, WS m s-1, WD degrees, PA kPa, P mm per half-hour, **SWC in % (volumetric,
0-100)**, FC / NEE umol CO2 m-2 s-1, PPFD umol m-2 s-1, USTAR m s-1. The SWC unit is ``%`` in
BIFVARINFO (``SWC_F_MDS_#``, SWC-TDR), and the BASE values (0.1-41.6) are identical to
``SWC_F_MDS_#`` where both are measured. FLUXNET differs from BASE in VPD (``hPa``) and in
daily aggregates of P (``mm d-1``, a daily sum).

Position qualifiers: BASE names end in ``_H_V_R`` (horizontal, vertical, replicate;
:func:`base_variable_positions`). V is a per-variable depth rank (1 = shallowest), not a
depth. The CA-TPA depths (:data:`CATPA_DEPTH_M`) come from the FLUXNET BIFVARINFO heights of
``TS_F_MDS_#`` / ``SWC_F_MDS_#`` / ``G_F_MDS``; each BASE column equals exactly one of those
series where both are measured (max abs difference 0), and the AmeriFlux measurement-height
file (``ameriflux/CA-TPA_measurement_height.csv``, :func:`read_measurement_heights`) says the same:
SWC_1_2_1 / 1_3_1 / 1_5_1 = 5 / 20 / 50 cm, TS_1_1_1 / 1_2_1 / 1_3_1 / 1_5_1 =
2 / 5 / 10 / 50 cm, G_1_1_1 = 3 cm.

FLUXNET QC flags
----------------
Half-hourly ``*_QC``: 0 = measured, 1 = good-quality gap-fill, 2 = medium, 3 = poor. Daily and
coarser ``*_QC``: fraction (0-1) of measured + good-quality gap-filled half-hours. The
fraction of *measured* half-hours per day is derivable only from the HH file
(:func:`measured_fraction`); :func:`daily_et` uses it when given the HH frame.

ET conversion (:func:`daily_et`)
--------------------------------
ET [mm d-1] = LE [W m-2, daily mean] * 86 400 s / lambda [J kg-1], with 1 kg m-2 = 1 mm of
water. lambda is constant (default 2.45 MJ kg-1, FAO-56) or, with ``lambda_mj_kg="ta"``,
temperature dependent, lambda = 2.501 - 0.002361 * T [MJ kg-1] with T = daily mean ``TA_F``
(Harrison 1963, as in FAO-56 Annex 3). ``LE_CORR`` is ``LE_F_MDS`` scaled by the ONEFlux
energy-balance-closure correction factor, i.e. the ET a closed energy balance would imply.

Validation target (:func:`catpa_validation_set`)
------------------------------------------------
Flux measurements start 2020-06-29 (BIF ``FLUX_MEASUREMENTS_DATE_START``; LE_CORR exists on
206 days of 2020 and every day of 2021-2023). Only 2020 and 2021 are maize (then sweet potato
2022, tobacco 2023; BIF ``SITE_DESC``). Sowing / harvest dates (:data:`CATPA_MAIZE_SEASONS`)
are those of the CA-TPA RZWQM2 scenario (``rzwqm.dat`` plant-management records, i.e.
``catpa/events.csv``); the BIF file has no management records.

Attribution: any use needs :data:`CITATION_BASE` and/or :data:`CITATION_FLUXNET` plus
:func:`acknowledgments` (sentences copied from ``DATA_POLICY_LICENSE_AND_INSTRUCTIONS.txt``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, cast, overload

import numpy as np
import pandas as pd

__all__ = [
    "ACK_AMERIFLUX",
    "ACK_FLUXNET",
    "BASE_UNITS",
    "CATPA_DEPTH_M",
    "CATPA_FLUX_START",
    "CATPA_MAIZE_SEASONS",
    "CATPA_UTC_OFFSET_H",
    "CITATION_BASE",
    "CITATION_FLUXNET",
    "DATA_AVAILABILITY_TEMPLATE",
    "MISSING",
    "PROCESSING_PAPER",
    "acknowledgments",
    "base_units",
    "base_variable_positions",
    "catpa_validation_set",
    "daily_et",
    "data_availability",
    "energy_balance_closure",
    "find_fluxnet_file",
    "growing_season_mask",
    "latent_heat_mj_kg",
    "measured_fraction",
    "read_base_hh",
    "read_bifvarinfo",
    "read_fluxnet",
    "read_measurement_heights",
]

MISSING = -9999
"""Missing-value marker of both products."""

CATPA_FLUX_START = "2020-06-29"
"""First day of CA-TPA flux measurements (BIF ``FLUX_MEASUREMENTS_DATE_START``). The DD file has
``LE_CORR`` from 2020-06-09, but up to 2020-06-23 ``LE_F_MDS_QC`` is 0 (no measured or good
gap-filled half-hour: a repeating 3-day MDS fill), so :func:`catpa_validation_set` starts here."""

CATPA_UTC_OFFSET_H = -5
"""CA-TPA local standard time is UTC-5 (BIF ``UTC_OFFSET``); indexes are tz-naive in this time."""

Resolution = Literal["HH", "DD", "WW", "MM", "YY"]

# ----------------------------------------------------------------------------- data policy

CITATION_BASE = (
    "M. Altaf Arain (2025), AmeriFlux BASE CA-TPA Ontario Turkey Point Observatory Agricultural Site, "
    "Ver. 3-5, AmeriFlux AMP, (Dataset). https://doi.org/10.17190/AMF/2563529"
)
"""BASE citation, verbatim from the site BIF (``GRP_DOI / DOI_CITATION``, DOI 10.17190/AMF/2563529)."""

CITATION_FLUXNET = (
    "M. Altaf Arain (2026), AmeriFlux FLUXNET-1F CA-TPA Ontario Turkey Point Observatory Agricultural "
    "Site, Ver. 1.3_r1, AmeriFlux AMP, (Dataset). https://doi.org/10.17190/AMF/3027362"
)
"""FLUXNET-1F citation (DOI 10.17190/AMF/3027362, v1.3_r1, processed 2026-03-28), in the BASE
citation's pattern; the zip carries no citation string, so check it on the AmeriFlux site page."""

ACK_FLUXNET = (
    "FLUXNET data products were produced and harmonized by eddy covariance regional networks and data "
    "processing centers, including AmeriFlux, ChinaFlux, European Fluxes Database, ICOS, JapanFlux, "
    "KoFlux, OzFlux, SAEON, and TERN. These products also include a modified version of ERA5 hourly "
    "data provided by the Copernicus Climate Change Service."
)
"""Required acknowledgment for any use of FLUXNET data (policy item 2, verbatim)."""

ACK_AMERIFLUX = (
    "Funding for the AmeriFlux data service was provided by the U.S. Department of Energy Office of Science."
)
"""Required acknowledgment when AmeriFlux (AMF) data are used (policy item 2a, verbatim)."""

DATA_AVAILABILITY_TEMPLATE = (
    "The FLUXNET data products used in this work were downloaded from [SOURCE] on XX Month YYYY"
)
"""Data-availability sentence (policy item 3, verbatim); fill with :func:`data_availability`."""

PROCESSING_PAPER = (
    "Pastorello, G., Trotta, C., Canfora, E. et al. The FLUXNET2015 dataset and the ONEFlux processing "
    "pipeline for eddy covariance data. Sci Data 7, 225 (2020). https://doi.org/10.1038/s41597-020-0534-3."
)
"""Recommended (not required) citation of the ONEFlux processing paper."""


def acknowledgments(fluxnet: bool = True) -> str:
    """Acknowledgment text: the AmeriFlux sentence, preceded by the FLUXNET one if ``fluxnet``."""
    return f"{ACK_FLUXNET} {ACK_AMERIFLUX}" if fluxnet else ACK_AMERIFLUX


def data_availability(source: str, date: str) -> str:
    """Policy item 3 filled in: ``data_availability("the AmeriFlux data portal", "23 September 2026")``."""
    return DATA_AVAILABILITY_TEMPLATE.replace("[SOURCE]", source).replace("XX Month YYYY", date) + "."


# ----------------------------------------------------------------------------- BASE

BASE_UNITS: dict[str, str] = {
    "NEE": "umol CO2 m-2 s-1",
    "FC": "umol CO2 m-2 s-1",
    "SC": "umol CO2 m-2 s-1",
    "CO2": "umol CO2 mol-1",
    "H2O": "mmol H2O mol-1",
    "LE": "W m-2",
    "H": "W m-2",
    "G": "W m-2",
    "NETRAD": "W m-2",
    "SW_IN": "W m-2",
    "SW_OUT": "W m-2",
    "LW_IN": "W m-2",
    "LW_OUT": "W m-2",
    "PPFD_IN": "umol photon m-2 s-1",
    "USTAR": "m s-1",
    "TA": "deg C",
    "TS": "deg C",
    "RH": "%",
    "WS": "m s-1",
    "WD": "decimal degrees",
    "PA": "kPa",
    "P": "mm (per half-hour)",
    "SWC": "% (volumetric, 0-100)",
    "U_SIGMA": "m s-1",
    "V_SIGMA": "m s-1",
    "W_SIGMA": "m s-1",
}
"""Units of the BASE variables (FP-Standard), keyed by base name (qualifiers stripped)."""

CATPA_DEPTH_M: dict[str, float] = {
    "TS_1_1_1": 0.02,
    "TS_1_2_1": 0.05,
    "TS_1_3_1": 0.10,
    "TS_1_5_1": 0.50,
    "SWC_1_2_1": 0.05,
    "SWC_1_3_1": 0.20,
    "SWC_1_5_1": 0.50,
    "G_1_1_1": 0.03,
}
"""CA-TPA sensor depth below the surface [m] of each BASE soil column; equal to
``-Height`` in ``CA-TPA_measurement_height.csv`` (:func:`read_measurement_heights`)."""

_HVR_RE = re.compile(r"^(?P<base>.+?)_(?P<h>\d+)_(?P<v>\d+)_(?P<r>\d+)$")
_SUFFIX_RE = re.compile(r"^(?P<base>.+?)(?:_(?:PI|F|IU))*(?:_\d+)*$")


def read_measurement_heights(path: str | Path, site: str | None = "CA-TPA") -> pd.DataFrame:
    """AmeriFlux measurement-height CSV (``Site_ID, Variable, Start_Date, Height, Instrument_Model, ...``)
    -> DataFrame with columns ``site``, ``variable``, ``height_m`` (negative = depth below the surface),
    ``instrument``, ``start_date`` (NaT when blank). ``site=None`` keeps every site; with several
    rows per (site, variable) (sensor moved), the one with the latest ``Start_Date`` is kept."""
    raw = pd.read_csv(path, dtype=str, encoding="latin-1").fillna("")
    for col in ("Site_ID", "Variable", "Height"):
        if col not in raw.columns:
            raise ValueError(f"{path}: no {col} column")
    if site is not None:
        raw = raw[_col(raw, "Site_ID").str.strip() == site]
    raw = cast(pd.DataFrame, raw).copy()
    for col in ("Start_Date", "Instrument_Model", "Instrument_Model2"):
        if col not in raw.columns:
            raw[col] = ""
    inst = _col(raw, "Instrument_Model").str.strip()
    inst2 = _col(raw, "Instrument_Model2").str.strip()
    out = pd.DataFrame(
        {
            "site": _col(raw, "Site_ID").str.strip(),
            "variable": _col(raw, "Variable").str.strip(),
            "height_m": pd.to_numeric(raw["Height"], errors="coerce"),
            "instrument": [a if not b else f"{a};{b}" for a, b in zip(inst, inst2)],
            "start_date": pd.to_datetime(
                _col(raw, "Start_Date").str.strip().str[:8],
                format="%Y%m%d",
                errors="coerce",
            ),
        }
    )
    out = out.sort_values(["site", "variable", "start_date"], na_position="first", kind="stable")
    return out.drop_duplicates(["site", "variable"], keep="last").reset_index(drop=True)


@overload
def base_variable_positions(cols: Iterable[str], heights: None = None) -> dict[str, tuple[int, int, int]]: ...
@overload
def base_variable_positions(cols: Iterable[str], heights: pd.DataFrame) -> pd.DataFrame: ...
def base_variable_positions(
    cols: Iterable[str], heights: pd.DataFrame | None = None
) -> dict[str, tuple[int, int, int]] | pd.DataFrame:
    """``{column: (horizontal, vertical, replicate)}`` for every column with an ``_H_V_R`` suffix.

    ``SWC_1_2_1 -> (1, 2, 1)``; columns without the three-integer suffix (``LE``, ``P_PI_1``) are
    left out. The vertical index ranks the sensors of one variable by depth (1 = shallowest),
    it is not a depth. With ``heights`` (:func:`read_measurement_heights`) the result is a
    DataFrame indexed by column with ``h``, ``v``, ``r``, ``height_m`` (negative = depth) and
    ``depth_m`` (= -height_m below ground, NaN above), ``instrument``.
    """
    out: dict[str, tuple[int, int, int]] = {}
    for c in cols:
        m = _HVR_RE.match(c)
        if m:
            out[c] = (int(m.group("h")), int(m.group("v")), int(m.group("r")))
    if heights is None:
        return out
    tab = pd.DataFrame.from_dict(out, orient="index", columns=["h", "v", "r"])
    tab.index.name = "column"
    sites = heights["site"].unique() if "site" in heights.columns else []
    if len(sites) > 1:
        raise ValueError(f"heights cover several sites {list(sites)}; read them with site=...")
    hv = heights.set_index("variable")
    tab["height_m"] = hv["height_m"].reindex(tab.index).astype("float64")
    tab["depth_m"] = (-tab["height_m"]).where(tab["height_m"] < 0)
    tab["instrument"] = hv["instrument"].reindex(tab.index)
    return tab


def _base_name(col: str) -> str:
    m = _HVR_RE.match(col)
    name = m.group("base") if m else col
    if name in BASE_UNITS:
        return name
    m2 = _SUFFIX_RE.match(name)
    return m2.group("base") if m2 else name


def base_units(cols: Iterable[str]) -> dict[str, str]:
    """``{column: unit}`` for the BASE columns in :data:`BASE_UNITS` (e.g. ``NEE_PI``, ``P_PI_1``)."""
    out: dict[str, str] = {}
    for c in cols:
        u = BASE_UNITS.get(_base_name(c))
        if u is not None:
            out[c] = u
    return out


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, df[name])


def _day(d: Any) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(d)).floor("D")


def _floor(index: Any, unit: str) -> pd.DatetimeIndex:
    """Start of the day / month / year (``unit`` = "D" / "M" / "Y") of each timestamp."""
    days = np.asarray(pd.DatetimeIndex(index).values).astype(f"datetime64[{unit}]")
    return pd.DatetimeIndex(days.astype("datetime64[ns]"))


def _parse_stamp(values: pd.Series, fmt: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values.astype("int64").astype(str), format=fmt))


def read_base_hh(path: str | Path) -> pd.DataFrame:
    """AmeriFlux BASE half-hourly CSV -> DataFrame indexed by ``TIMESTAMP_START``.

    ``-9999`` becomes NaN; ``TIMESTAMP_END`` is kept as a datetime column. The index is tz-naive
    local standard time (UTC-5 for CA-TPA). ``attrs``: ``site``, ``version`` (from the comment
    lines), ``units`` (:func:`base_units`), ``positions`` (:func:`base_variable_positions`).
    """
    p = Path(path)
    meta: dict[str, str] = {}
    n_comment = 0
    with p.open(encoding="latin-1") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            n_comment += 1
            key, _, val = line.lstrip("#").partition(":")
            meta[key.strip().lower()] = val.strip()
    df = pd.read_csv(p, skiprows=n_comment, na_values=[MISSING, str(MISSING), "-9999.0"], encoding="latin-1")
    for col in ("TIMESTAMP_START", "TIMESTAMP_END"):
        if col not in df.columns:
            raise ValueError(f"{p}: no {col} column in the header")
    start = _parse_stamp(_col(df, "TIMESTAMP_START"), "%Y%m%d%H%M")
    df["TIMESTAMP_END"] = _parse_stamp(_col(df, "TIMESTAMP_END"), "%Y%m%d%H%M")
    df = df.drop(columns="TIMESTAMP_START")
    df.index = start.rename("TIMESTAMP_START")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError(f"{p}: TIMESTAMP_START is not strictly increasing")
    data_cols = [c for c in df.columns if c != "TIMESTAMP_END"]
    df[data_cols] = df[data_cols].astype("float64")
    df.attrs.update(
        site=meta.get("site", ""),
        version=meta.get("version", ""),
        units=base_units(data_cols),
        positions=base_variable_positions(data_cols),
    )
    return df


# ----------------------------------------------------------------------------- FLUXNET

_STAMP_FORMATS: dict[str, tuple[str, str]] = {
    "HH": ("TIMESTAMP_START", "%Y%m%d%H%M"),
    "DD": ("TIMESTAMP", "%Y%m%d"),
    "WW": ("TIMESTAMP_START", "%Y%m%d"),
    "MM": ("TIMESTAMP", "%Y%m"),
    "YY": ("TIMESTAMP", "%Y"),
}


def find_fluxnet_file(path_or_dir: str | Path, resolution: str = "DD", kind: str = "FLUXMET") -> Path:
    """Locate ``*_FLUXNET_<kind>_<resolution>_*.csv`` in a directory (or its ``fluxnet/`` /
    ``ameriflux/fluxnet/`` subdirectory); a file path is returned unchanged."""
    p = Path(path_or_dir)
    if p.is_file():
        return p
    pattern = f"*_FLUXNET_{kind}_{resolution}_*.csv"
    for d in (p, p / "fluxnet", p / "ameriflux" / "fluxnet"):
        hits = sorted(d.glob(pattern)) if d.is_dir() else []
        if len(hits) > 1:
            raise ValueError(f"several {pattern} files in {d}: {[h.name for h in hits]}")
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no {pattern} under {p}")


def read_bifvarinfo(path: str | Path) -> pd.DataFrame:
    """FLUXNET ``*_BIFVARINFO_*.csv`` -> one row per variable (index ``VARNAME``) with columns
    ``UNIT``, ``HEIGHT`` (m, negative = below ground), ``DEFINITION``, ``MODEL``, ``DATE``."""
    raw = pd.read_csv(path, encoding="latin-1", dtype=str)
    raw = raw[raw["VARIABLE_GROUP"] == "GRP_VAR_INFO"]
    wide = raw.pivot_table(index="GROUP_ID", columns="VARIABLE", values="DATAVALUE", aggfunc="first")
    wide.columns = [str(c).removeprefix("VAR_INFO_") for c in wide.columns]
    wide = wide.set_index("VARNAME")
    wide = wide[~wide.index.duplicated(keep="first")]
    if "HEIGHT" in wide.columns:
        wide["HEIGHT"] = pd.to_numeric(wide["HEIGHT"], errors="coerce")
    return cast(pd.DataFrame, wide)


def read_fluxnet(
    path_or_dir: str | Path,
    resolution: Resolution = "DD",
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """ONEFlux FLUXNET FLUXMET CSV -> DataFrame with a tz-naive DatetimeIndex (local standard time).

    Index: ``TIMESTAMP_START`` for HH / WW, ``TIMESTAMP`` (day / month / year start) for DD /
    MM / YY. ``-9999`` becomes NaN. ``columns`` restricts the columns read (faster on HH). If the
    matching ``*_BIFVARINFO_<res>_*.csv`` sits next to the file, ``attrs["units"]`` maps each
    column to its unit.
    """
    res = resolution.upper()
    if res not in _STAMP_FORMATS:
        raise ValueError(f"resolution must be one of {sorted(_STAMP_FORMATS)}, got {resolution!r}")
    p = find_fluxnet_file(path_or_dir, res)
    stamp, fmt = _STAMP_FORMATS[res]
    usecols = None
    if columns is not None:
        extra = ["TIMESTAMP_END"] if stamp == "TIMESTAMP_START" else []
        usecols = list(dict.fromkeys([stamp, *extra, *columns]))
    df = pd.read_csv(p, usecols=usecols, na_values=[MISSING, str(MISSING), "-9999.0"], encoding="latin-1")
    if stamp not in df.columns:
        raise ValueError(f"{p}: no {stamp} column for resolution {res}")
    idx = _parse_stamp(_col(df, stamp), fmt).rename(stamp)
    df = df.drop(columns=stamp)
    if "TIMESTAMP_END" in df.columns:
        df["TIMESTAMP_END"] = _parse_stamp(_col(df, "TIMESTAMP_END"), fmt)
    df.index = idx
    data_cols = [c for c in df.columns if c != "TIMESTAMP_END"]
    df[data_cols] = df[data_cols].astype("float64")
    units: dict[str, str] = {}
    try:
        info = read_bifvarinfo(find_fluxnet_file(p.parent, res, kind="BIFVARINFO"))
        units = {c: str(info.at[c, "UNIT"]) for c in data_cols if c in info.index and "UNIT" in info.columns}
    except (FileNotFoundError, KeyError, ValueError):
        pass
    df.attrs.update(resolution=res, source=p.name, units=units)
    return df


def measured_fraction(fluxnet_hh: pd.DataFrame, qc_col: str = "LE_F_MDS_QC") -> pd.Series:
    """Per-day fraction of half-hours whose HH ``qc_col`` is 0 (measured), index = day."""
    if qc_col not in fluxnet_hh.columns:
        raise KeyError(f"{qc_col} not in the HH frame")
    flag = (_col(fluxnet_hh, qc_col) == 0).astype("float64")
    day = _floor(fluxnet_hh.index, "D")
    return flag.groupby(day).mean().rename("qc")


# ----------------------------------------------------------------------------- ET


def latent_heat_mj_kg(t_c: np.ndarray | pd.Series | float) -> np.ndarray | pd.Series | float:
    """Latent heat of vaporization [MJ kg-1] = 2.501 - 0.002361 * T [deg C] (FAO-56 eq. 3-1)."""
    return 2.501 - 0.002361 * t_c


def _le_to_mm_d(le: pd.Series, lam_j_kg: pd.Series | float) -> pd.Series:
    return le * 86400.0 / lam_j_kg


def daily_et(
    fluxnet_dd: pd.DataFrame,
    source: Literal["LE_CORR", "LE_F_MDS"] = "LE_CORR",
    lambda_mj_kg: float | Literal["ta"] = 2.45,
    hh: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Daily ET [mm d-1] from a FLUXNET DD frame (:func:`read_fluxnet` with ``resolution="DD"``).

    ET = LE [W m-2, daily mean] * 86400 / lambda [J kg-1]. ``lambda_mj_kg`` is a constant in
    MJ kg-1 or ``"ta"`` for :func:`latent_heat_mj_kg` of the daily mean ``TA_F``.

    Columns: ``et_mm``; ``et_lo`` / ``et_hi`` from ``LE_CORR_25`` / ``LE_CORR_75`` (the spread of
    the closure correction factor; NaN for ``source="LE_F_MDS"`` or when absent); ``randunc_mm``
    from ``LE_RANDUNC`` (random uncertainty of the measured half-hours); ``qc``: fraction of
    measured half-hours (``LE_F_MDS_QC == 0``) when the HH frame ``hh`` is given, else the DD
    ``LE_F_MDS_QC`` (fraction measured + good gap-fill). ``attrs["qc_kind"]`` says which.
    """
    if source not in fluxnet_dd.columns:
        raise KeyError(f"{source} not in the DD frame")
    if lambda_mj_kg == "ta":
        if "TA_F" not in fluxnet_dd.columns:
            raise KeyError("lambda_mj_kg='ta' needs TA_F in the DD frame")
        lam: pd.Series | float = cast(pd.Series, latent_heat_mj_kg(_col(fluxnet_dd, "TA_F"))) * 1e6
    else:
        lam = float(lambda_mj_kg) * 1e6
    nan = pd.Series(np.nan, index=fluxnet_dd.index)
    out = pd.DataFrame(index=fluxnet_dd.index)
    out["et_mm"] = _le_to_mm_d(_col(fluxnet_dd, source), lam)
    has_q = source == "LE_CORR" and {"LE_CORR_25", "LE_CORR_75"} <= set(fluxnet_dd.columns)
    out["et_lo"] = _le_to_mm_d(_col(fluxnet_dd, "LE_CORR_25"), lam) if has_q else nan
    out["et_hi"] = _le_to_mm_d(_col(fluxnet_dd, "LE_CORR_75"), lam) if has_q else nan
    out["randunc_mm"] = (
        _le_to_mm_d(_col(fluxnet_dd, "LE_RANDUNC"), lam) if "LE_RANDUNC" in fluxnet_dd else nan
    )
    if hh is not None:
        out["qc"] = measured_fraction(hh).reindex(out.index)
        qc_kind = "measured_fraction_hh"
    elif "LE_F_MDS_QC" in fluxnet_dd.columns:
        out["qc"] = fluxnet_dd["LE_F_MDS_QC"]
        qc_kind = "LE_F_MDS_QC_dd"
    else:
        out["qc"] = nan
        qc_kind = "none"
    out.attrs.update(
        source=source, lambda_mj_kg=lambda_mj_kg, qc_kind=qc_kind, units={c: "mm d-1" for c in out.columns}
    )
    out.attrs["units"]["qc"] = "fraction"
    return out


# ----------------------------------------------------------------------------- closure

_CLOSURE_TERMS: tuple[tuple[str, str, str, str], ...] = (
    ("H", "LE", "NETRAD", "G_1_1_1"),  # BASE
    ("H_F_MDS", "LE_F_MDS", "NETRAD", "G_F_MDS"),  # FLUXNET
)
_WINDOWS = ("D", "M", "Y")


def _closure_columns(df: pd.DataFrame) -> tuple[str, str, str, str]:
    for terms in _CLOSURE_TERMS:
        if set(terms) <= set(df.columns):
            return terms
    g = sorted(c for c in df.columns if c.startswith("G_"))
    if {"H", "LE", "NETRAD"} <= set(df.columns) and g:
        return ("H", "LE", "NETRAD", g[0])
    raise KeyError("need H, LE, NETRAD, G (BASE) or H_F_MDS, LE_F_MDS, NETRAD, G_F_MDS (FLUXNET)")


def energy_balance_closure(df: pd.DataFrame, window: Literal["D", "M", "Y"] = "D") -> pd.DataFrame:
    """Energy-balance closure (H + LE) / (Rn - G) per ``window`` from BASE HH or FLUXNET frames.

    Sub-daily input uses only the half-hours where all four terms are present; for FLUXNET HH
    the gap-filled H / LE / G (``*_QC`` > 0) are also excluded, so the closure is that of the
    measurements. Daily (or coarser) FLUXNET input uses the rows where all four are present.

    Columns: ``ratio`` = sum(H + LE) / sum(Rn - G) over the valid records; ``slope`` and
    ``intercept`` [W m-2] of the OLS fit H + LE = slope * (Rn - G) + intercept; ``n`` valid
    records; ``coverage`` = n / records in the window.
    """
    if window not in _WINDOWS:
        raise ValueError(f"window must be one of {sorted(_WINDOWS)}, got {window!r}")
    h, le, rn, g = _closure_columns(df)
    valid = cast(pd.Series, df[[h, le, rn, g]].notna().all(axis=1))
    for c in (h, le, g):
        qc = f"{c}_QC"
        if qc in df.columns and bool(_col(df, qc).dropna().isin([0, 1, 2, 3]).all()):  # HH flags
            valid = valid & (_col(df, qc) == 0)
    turb = (_col(df, h) + _col(df, le)).where(valid)
    avail = (_col(df, rn) - _col(df, g)).where(valid)
    key = _floor(df.index, window)
    frame = pd.DataFrame(
        {"y": turb.to_numpy(), "x": avail.to_numpy(), "valid": valid.to_numpy(dtype=np.float64), "key": key}
    )
    rows: list[dict[str, float]] = []
    keys: list[pd.Timestamp] = []
    for k, g_any in frame.groupby("key", sort=True):
        grp = cast(pd.DataFrame, g_any)
        ok = _col(grp, "valid").to_numpy() > 0
        x = _col(grp, "x").to_numpy(dtype=np.float64)[ok]
        y = _col(grp, "y").to_numpy(dtype=np.float64)[ok]
        n = int(ok.sum())
        sx = float(x.sum()) if n else np.nan
        ratio = float(y.sum()) / sx if n and sx != 0.0 else np.nan
        slope = intercept = np.nan
        if n >= 3 and float(np.var(x)) > 0.0:
            slope, intercept = (float(v) for v in np.polyfit(x, y, 1))
        rows.append(
            {"ratio": ratio, "slope": slope, "intercept": intercept, "n": n, "coverage": n / len(grp)}
        )
        keys.append(cast(pd.Timestamp, pd.Timestamp(str(k))))
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(keys, name="window_start"))
    out.attrs.update(terms={"H": h, "LE": le, "Rn": rn, "G": g}, window=window)
    return out


# ----------------------------------------------------------------------------- seasons

CATPA_MAIZE_SEASONS: dict[int, tuple[str, str]] = {
    2020: ("2020-05-05", "2020-10-25"),
    2021: ("2021-05-05", "2021-10-20"),
}
"""Maize sowing and harvest dates at CA-TPA (``rzwqm.dat`` plant management / ``catpa/events.csv``)."""

DateLike = str | pd.Timestamp | np.datetime64


def growing_season_mask(
    index: pd.DatetimeIndex | Sequence[DateLike],
    sow: DateLike | Sequence[DateLike],
    harvest: DateLike | Sequence[DateLike],
) -> np.ndarray:
    """Boolean array: True where ``sow <= date <= harvest`` (by calendar day, both inclusive).

    ``sow`` / ``harvest`` may be sequences of equal length (several seasons, OR-ed together).
    """
    days = _floor(index, "D")
    sows = [sow] if isinstance(sow, str | pd.Timestamp | np.datetime64) else list(sow)
    harvs = [harvest] if isinstance(harvest, str | pd.Timestamp | np.datetime64) else list(harvest)
    if len(sows) != len(harvs):
        raise ValueError(f"{len(sows)} sowing dates but {len(harvs)} harvest dates")
    mask = np.zeros(len(days), dtype=bool)
    for s, h in zip(sows, harvs):
        s_ts, h_ts = _day(s), _day(h)
        if h_ts < s_ts:
            raise ValueError(f"harvest {h_ts.date()} before sowing {s_ts.date()}")
        mask |= np.asarray((days >= s_ts) & (days <= h_ts))
    return mask


def catpa_validation_set(
    data_dir: str | Path,
    years: Sequence[int] = (2021,),
    lambda_mj_kg: float | Literal["ta"] = 2.45,
    measured_qc: bool = True,
    seasons: dict[int, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Daily observed ET (``LE_CORR``) over the CA-TPA maize seasons of ``years``.

    ``data_dir`` is the data root (``~/agri_jax_data``), its ``ameriflux/`` folder or the
    ``fluxnet/`` folder. Returns the :func:`daily_et` columns plus ``season`` (the year) and
    ``le_qc_dd`` (DD ``LE_F_MDS_QC``), restricted to sow..harvest days on or after
    :data:`CATPA_FLUX_START` that have ``LE_CORR`` (so 2020 starts on 2020-06-29). With
    ``measured_qc`` the HH file is read to give ``qc`` = fraction of measured half-hours.
    """
    seasons = CATPA_MAIZE_SEASONS if seasons is None else seasons
    bad = [y for y in years if y not in seasons]
    if bad:
        raise ValueError(f"no maize season for {bad} (maize years: {sorted(seasons)})")
    dd = read_fluxnet(data_dir, "DD")
    hh = read_fluxnet(data_dir, "HH", columns=["LE_F_MDS_QC"]) if measured_qc else None
    et = daily_et(dd, "LE_CORR", lambda_mj_kg=lambda_mj_kg, hh=hh)
    et["le_qc_dd"] = dd["LE_F_MDS_QC"]
    et["season"] = pd.Series(np.nan, index=et.index)
    for y in years:
        s, h = seasons[y]
        et.loc[growing_season_mask(pd.DatetimeIndex(et.index), s, h), "season"] = y
    keep = _col(et, "season").notna() & _col(et, "et_mm").notna()
    keep &= np.asarray(pd.DatetimeIndex(et.index) >= pd.Timestamp(CATPA_FLUX_START))
    out = cast(pd.DataFrame, et[keep]).copy()
    out["season"] = _col(out, "season").astype("int64")
    out.attrs.update(
        {str(k): v for k, v in et.attrs.items()},
        seasons={y: seasons[y] for y in years},
        citation=CITATION_FLUXNET,
    )
    return out
