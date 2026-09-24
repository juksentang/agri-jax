"""Per-season summary from the DSSAT-style ``OVERVIEW.OUT`` written by RZWQM2.

Each season is a ``*RUN   k`` section with ``PLANTING DATE  : APR 28 2015  PLANTS/m2 : 8.0``,
``HARVEST DATE   : SEP 25 2015`` and ``<Crop> YIELD :     9916 kg/ha    [Dry weight]``
(CERES-Maize and the other DSSAT 4.0 crop modules). Cropsim-CERES (wheat, canola) writes no
``YIELD :`` line; its seasons carry ``CROP           : WHEAT`` in the header and a
``Product wt (kg dm/ha;no loss)      3275          -99`` row (predicted, measured) in the
end-of-season table, which is read as the yield. The file contains NUL padding, hence it is
decoded with ``errors="ignore"``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import pandas as pd

__all__ = ["read_overview_yields"]

_RUN_RE = re.compile(r"^\*RUN\s+(\d+)")
_DATE = r"([A-Z]{3})\s+(\d{1,2})\s+(\d{4})"
_PLANT_RE = re.compile(r"PLANTING DATE\s*:\s*" + _DATE + r"(?:.*?PLANTS/m2\s*:\s*([\d.]+))?")
_HARV_RE = re.compile(r"HARVEST DATE\s*:\s*" + _DATE)
_YIELD_RE = re.compile(r"^\s*(\w[\w ]*?)\s+YIELD\s*:\s*(-?[\d.]+)\s*kg/ha")
_CROP_RE = re.compile(r"^\s*CROP\s*:\s*(\S[\w ]*?)\s{2,}CULTIVAR")
_PRODUCT_RE = re.compile(r"^\s*Product wt \(kg dm/ha;no loss\)\s+(-?[\d.]+)")
_MONTHS = {m: i for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), start=1)}


def _ts(mon: str, day: str, year: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(year=int(year), month=_MONTHS[mon.upper()], day=int(day)))


def read_overview_yields(path: str | Path) -> pd.DataFrame:
    """``OVERVIEW.OUT`` -> DataFrame(season, crop, planting_date, harvest_date, plants_m2, yield_kg_ha).

    ``season`` is the ``*RUN`` number. Dates are NaT and yield NaN when a section lacks them.
    """
    text = Path(path).read_bytes().decode("latin-1", errors="ignore").replace("\x00", "")
    records: list[dict[str, object]] = []
    cur: dict[str, object] | None = None
    for ln in text.splitlines():
        m = _RUN_RE.match(ln)
        if m:
            cur = {
                "season": int(m.group(1)),
                "crop": "",
                "planting_date": pd.NaT,
                "harvest_date": pd.NaT,
                "plants_m2": float("nan"),
                "yield_kg_ha": float("nan"),
                "_yield_line": False,
            }
            records.append(cur)
            continue
        if cur is None:
            continue
        if m := _PLANT_RE.search(ln):
            cur["planting_date"] = _ts(*m.group(1, 2, 3))
            if m.group(4):
                cur["plants_m2"] = float(m.group(4))
        elif m := _HARV_RE.search(ln):
            cur["harvest_date"] = _ts(*m.group(1, 2, 3))
        elif m := _YIELD_RE.match(ln):
            cur["crop"] = m.group(1).strip()
            cur["yield_kg_ha"] = float(m.group(2))
            cur["_yield_line"] = True
        elif m := _CROP_RE.match(ln):
            if not cur["crop"]:
                cur["crop"] = m.group(1).strip().capitalize()
        elif (m := _PRODUCT_RE.match(ln)) and not cur["_yield_line"]:
            cur["yield_kg_ha"] = float(m.group(1))  # Cropsim-CERES layout
    for r in records:
        del r["_yield_line"]
    df = pd.DataFrame.from_records(
        records,
        columns=["season", "crop", "planting_date", "harvest_date", "plants_m2", "yield_kg_ha"],
    )
    for c in ("planting_date", "harvest_date"):
        df[c] = pd.to_datetime(df[c])
    return df
