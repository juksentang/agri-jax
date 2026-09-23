"""Reader for the RZWQM2 per-node daily profile file ``LAYER.PLT`` (the "vector plot file").

Where it comes from
-------------------
``cntrl.dat`` controls it in two places (``Rzout.for``: ``OUTIN``/``VWRITE``):

* ``==== NUMBER OF VECTOR VARIABLES`` / ``VARIABLES TO BE PLOTTED AGAINST TIME AND DEPTH``:
  up to 8 vector-variable ids. CA-TPA ships ``8`` / ``2 43 20 9 32 33 39 36``, i.e. soil water
  content, pressure head, bulk density, NO3-N, pesticide 1 and 2 total mass, plant water
  uptake and N uptake (the labels are written into the file header, so the reader never needs
  the id -> name table).
* the ``Output Control`` row ``VECTOR TABULAR / SCALAR TABULAR / VECTOR PLOT / SCALAR PLOT``:
  CA-TPA ships ``0 1 1 1``. The third flag (VECTOR PLOT FILE = 1) makes the model write
  ``LAYER.PLT`` (``EVENTL.PLT`` for event-scale runs). The first flag (VECTOR TABULAR = 1)
  would add ``LAYER1.OUT ... LAYERk.OUT`` (9 variables per file, same data in a paged layout);
  it is not needed. So **no cntrl.dat change is required**: the shipped CA-TPA scenario already
  writes theta per numerical node per day.

Layout (``RZWQM2-3D``)::

    RZWQM2-3D
    ***...  Subplot Definitions  ***
        8    1                       <- number of vector variables, number of subplots
    ***...  COLUMN LABELS  ***
    DAY
    DEPTH (CM)
    SOIL WATER CONTENT (VOL)
    ... one label per variable ...
    ****************** DATA STARTS HERE *****************
             1      1.00000       0.179042       -2933.61  ...   (FORMAT 1X,I10,3X,30G15.6E3)

One row per (day, node): ``DAY`` is the 1-based *simulation* day (day 1 = the IPNAMES start
date; it does not reset at year ends), ``DEPTH`` the node depth in cm. Node depths are the
layer *bottoms* of the numerical grid (CA-TPA: 37 nodes, 1 ... 150 cm), so the layer
thicknesses are ``diff([0, depth])``; with that convention
``sum(theta * thickness)`` equals ``.ana`` column 2 (STORED SOIL WATER, cm) to print
precision (6e-5 cm over 2015 for CA-TPA). Row ``DAY = k`` is the state at the end of calendar
day ``start + k - 1``, the same day as the ``.ana`` row ``YYYY.DDD`` of that date.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ._fortran import fortran_float

__all__ = ["layer_thickness_cm", "profile_storage_cm", "read_layer_output", "simulation_start"]

_LABEL_UNITS = re.compile(r"^(?P<name>.*?)\s*\((?P<units>[^()]*)\)\s*$")
_IPNAMES_DATE = re.compile(r"^\s*(\d{1,2})\s+(\d{1,2})\s+(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{4})\s*$")


def _slug(text: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", text.strip().lower()).strip("_")
    return s if s and not s[0].isdigit() else f"v_{s}"


def simulation_start(ipnames: str | Path) -> np.datetime64:
    """Start date (``datetime64[D]``) from line 9 of an ``IPNAMES.DAT`` (``DD MM YYYY DD MM YYYY``)."""
    lines = Path(ipnames).read_bytes().decode("latin-1").splitlines()
    for ln in lines[8:12]:
        m = _IPNAMES_DATE.match(ln)
        if m:
            d, mo, y = (int(g) for g in m.group(1, 2, 3))
            return np.datetime64(_dt.date(y, mo, d), "D")
    raise ValueError(f"{ipnames}: no 'DD MM YYYY DD MM YYYY' simulation-period line")


def read_layer_output(
    path: str | Path,
    start: str | np.datetime64 | _dt.date | None = None,
) -> xr.Dataset:
    """``LAYER.PLT`` -> Dataset with dims ``(time, depth)``, one variable per column label.

    ``start`` is the simulation start date; when omitted it is read from ``IPNAMES.DAT`` next
    to ``path`` (the run directory). Variables are slugified labels
    (``SOIL WATER CONTENT (VOL)`` -> ``soil_water_content``, ``attrs["units"] == "VOL"``;
    ``PRESSURE HEAD (CM)`` -> ``pressure_head``). Coordinates: ``time`` (datetime64),
    ``day`` (1-based simulation day, along ``time``), ``depth`` (node depth = layer bottom, cm)
    and ``thickness`` (layer thickness, cm, along ``depth``).
    """
    p = Path(path)
    if start is None:
        ip = p.parent / "IPNAMES.DAT"
        if not ip.is_file():
            raise ValueError(f"start not given and {ip} not found")
        start64 = simulation_start(ip)
    else:
        start64 = np.datetime64(start, "D")

    lines = p.read_bytes().decode("latin-1").splitlines()
    if not lines or not lines[0].strip().upper().startswith("RZWQM2-3D"):
        raise ValueError(f"{p}: not an RZWQM2-3D vector plot file")
    try:
        i_lab = next(i for i, ln in enumerate(lines) if "COLUMN LABELS" in ln.upper())
        i_dat = next(i for i, ln in enumerate(lines) if "DATA STARTS HERE" in ln.upper())
    except StopIteration:
        raise ValueError(f"{p}: missing 'COLUMN LABELS' or 'DATA STARTS HERE'") from None
    labels = [ln.strip() for ln in lines[i_lab + 1 : i_dat] if ln.strip() and not ln.lstrip().startswith("*")]
    if len(labels) < 3 or labels[0].upper() != "DAY" or not labels[1].upper().startswith("DEPTH"):
        raise ValueError(f"{p}: unexpected column labels {labels[:3]}")
    ncol = len(labels)

    rows = [ln.split() for ln in lines[i_dat + 1 :] if ln.strip()]
    bad = next((k for k, r in enumerate(rows) if len(r) != ncol), None)
    if bad is not None:
        raise ValueError(f"{p}: data row {bad + 1} has {len(rows[bad])} fields, labels declare {ncol}")
    try:
        data = np.array(rows, dtype=np.float64)
    except ValueError:
        data = np.array([[fortran_float(t) for t in r] for r in rows], dtype=np.float64)
    if data.shape[0] == 0:
        raise ValueError(f"{p}: no data rows")

    day = data[:, 0].astype(np.int64)
    days, first = np.unique(day, return_index=True)
    counts = np.diff(np.r_[first, len(day)])
    n_node = int(counts[0])
    if not (counts == n_node).all():
        raise ValueError(f"{p}: days have different node counts {sorted(set(counts.tolist()))}")
    if not (np.diff(day) >= 0).all():
        raise ValueError(f"{p}: DAY column is not sorted")
    block = data.reshape(len(days), n_node, ncol)
    depth = block[0, :, 1]
    if not np.allclose(block[:, :, 1], depth):
        raise ValueError(f"{p}: node depths change between days")

    time = start64 + (days - 1).astype("timedelta64[D]")
    data_vars = {}
    for c in range(2, ncol):
        m = _LABEL_UNITS.match(labels[c])
        name, units = (m.group("name"), m.group("units").strip()) if m else (labels[c], "")
        key = _slug(name)
        if key in data_vars:
            key = f"{key}_c{c + 1}"
        data_vars[key] = (
            ("time", "depth"),
            block[:, :, c],
            {"long_name": name, "units": units, "column": c + 1},
        )
    return xr.Dataset(
        data_vars,
        coords={
            "time": pd.DatetimeIndex(time.astype("datetime64[ns]")),
            "day": ("time", days),
            "depth": ("depth", depth, {"units": "cm", "long_name": "node depth (layer bottom)"}),
            "thickness": ("depth", layer_thickness_cm(depth), {"units": "cm"}),
        },
        attrs={"source": str(p), "start": str(start64)},
    )


def layer_thickness_cm(depth: np.ndarray) -> np.ndarray:
    """Thickness of each numerical layer from the node depths (layer bottoms): ``diff([0, depth])``."""
    z = np.asarray(depth, dtype=np.float64)
    return np.diff(np.concatenate([[0.0], z]))


def profile_storage_cm(ds: xr.Dataset, var: str = "soil_water_content") -> xr.DataArray:
    """Profile water storage (cm) = ``sum(theta * thickness)``; equals ``.ana`` column 2."""
    return (ds[var] * ds["thickness"]).sum("depth")
