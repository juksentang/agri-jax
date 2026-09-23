"""Calibration-parameter maps for ``rzwqm.dat``: ``all_parameters.csv`` -> (line, token) addresses.

``all_parameters.csv`` (``RZWQM_sw_batch``) lists, per scenario, the calibrated parameters and
where they live::

    scenario,parameter,min,max,file,line_number,location_at_line
    CA-TPA,Pore Size (c2),0.14,0.616,rzwqm.dat,135,2

A :class:`ParamSpec` is one such row with the CSV name translated to a canonical field name
(``"Pore Size (c2)"`` -> ``field="lam", horizon=1``). NOTE the label offset: the ``cK`` labels
of the CSV (and of the Setting.json files it was built from) are *hydraulic horizon K-1*, i.e.
``c2..c5`` = horizons 1..4 (``RZWQM_sw_batch/NOTES.md``; for CA-TPA ``(c2)`` is line 135, the
first record of horizon 1). :data:`CSV_HORIZON_LABEL_OFFSET` encodes this, and every map is
checked against the parsed layout before use. A *param map* is a list of specs for one
scenario; it can be derived from the CSV (:func:`param_map_from_csv`), from the layout of a
parsed dat file (:func:`layout_param_map`, every horizon of every hydraulic field), or saved
to / loaded from YAML (:func:`save_param_map`, :func:`load_param_map`).

:func:`params_from_dat` returns the soil hydraulic arrays ``[n_horizon]`` (always every
horizon, taken from the parsed hydraulic block) plus the scalar parameters of the map; the
map's horizon addresses are checked against the block layout, so a stale line number in the
CSV raises instead of silently reading the wrong token. :func:`params_to_dat` is the inverse
write-back, going through :func:`~agri_jax.io.rzwqm.dat.set_value` (GenerateDat formatting).
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .dat import HYDRAULIC_FIELDS, RzwqmDat, set_value

__all__ = [
    "CSV_HORIZON_LABEL_OFFSET",
    "CSV_NAME_TO_FIELD",
    "ParamSpec",
    "canonical_name",
    "layout_param_map",
    "load_param_map",
    "param_map_from_csv",
    "params_from_dat",
    "params_to_dat",
    "save_param_map",
]

CSV_NAME_TO_FIELD: dict[str, str] = {
    "pore size": "lam",
    "ksat": "ksat",
    "residual wc": "theta_r",
    "fc 1/3 wc": "theta_fc33",
    "fc 1/10 wc": "theta_fc10",
    "wp wc": "theta_wp",
    "albedo dry": "albedo_dry",
    "albedo wet": "albedo_wet",
    "albedo maturity": "albedo_crop",
    "albedo residue": "albedo_residue",
    "soil resistance": "soil_resistance",
    "stomatal resistance": "rs_min",
}

CSV_HORIZON_LABEL_OFFSET = 1
"""``(cK)`` in ``all_parameters.csv`` means hydraulic horizon ``K - CSV_HORIZON_LABEL_OFFSET``."""

_HORIZON_RE = re.compile(r"^(?P<base>.*?)\s*\(\s*[cC](?P<h>\d+)\s*\)\s*$")


@dataclass(frozen=True)
class ParamSpec:
    """One calibratable value: canonical field, optional horizon (1-based), address, bounds."""

    name: str  # unique key, e.g. "lam_2", "albedo_dry", "rs_min_corn"
    field: str  # canonical field, e.g. "lam"
    line_number: int  # 1-based
    location_at_line: int  # 0-based token index
    horizon: int | None = None  # 1-based horizon for hydraulic fields
    minimum: float | None = None
    maximum: float | None = None
    file: str = "rzwqm.dat"
    source_name: str = ""  # name as written in all_parameters.csv

    @property
    def address(self) -> tuple[int, int]:
        return (self.line_number, self.location_at_line)


def canonical_name(csv_name: str) -> tuple[str, str, int | None]:
    """CSV parameter name -> ``(name, field, horizon)``.

    ``"Pore Size (c2)"`` -> ``("lam_1", "lam", 1)`` (see :data:`CSV_HORIZON_LABEL_OFFSET`);
    ``"Stomatal Resistance corn"`` -> ``("rs_min_corn", "rs_min", None)``.
    """
    s = csv_name.strip()
    horizon: int | None = None
    m = _HORIZON_RE.match(s)
    if m:
        s, horizon = m.group("base"), int(m.group("h")) - CSV_HORIZON_LABEL_OFFSET
    low = s.lower()
    if low.startswith("stomatal resistance"):
        crop = re.sub(r"\W+", "_", low[len("stomatal resistance") :].strip()).strip("_")
        return (f"rs_min_{crop}" if crop else "rs_min", "rs_min", None)
    if low not in CSV_NAME_TO_FIELD:
        raise KeyError(f"unknown parameter name {csv_name!r} (known: {sorted(CSV_NAME_TO_FIELD)})")
    field = CSV_NAME_TO_FIELD[low]
    return (f"{field}_{horizon}" if horizon is not None else field, field, horizon)


def param_map_from_csv(csv_path: str | Path, scenario: str, *, file: str = "rzwqm.dat") -> list[ParamSpec]:
    """Param map of one scenario from ``all_parameters.csv`` (rows of other files are skipped)."""
    specs: list[ParamSpec] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["scenario"].strip() != scenario or row["file"].strip() != file:
                continue
            name, field, horizon = canonical_name(row["parameter"])
            specs.append(
                ParamSpec(
                    name=name,
                    field=field,
                    line_number=int(row["line_number"]),
                    location_at_line=int(row["location_at_line"]),
                    horizon=horizon,
                    minimum=float(row["min"]),
                    maximum=float(row["max"]),
                    file=file,
                    source_name=row["parameter"].strip(),
                )
            )
    if not specs:
        raise KeyError(f"scenario {scenario!r} has no {file} rows in {csv_path}")
    return specs


def layout_param_map(dat: RzwqmDat, fields: Iterable[str] = HYDRAULIC_FIELDS) -> list[ParamSpec]:
    """Every horizon of every hydraulic ``field``, addressed from the parsed block layout."""
    addrs = dat.hydraulic_addresses()
    return [
        ParamSpec(name=f"{f}_{h + 1}", field=f, line_number=a[0], location_at_line=a[1], horizon=h + 1)
        for f in fields
        for h, a in enumerate(addrs[f])
    ]


def save_param_map(specs: Iterable[ParamSpec], path: str | Path) -> Path:
    import yaml

    p = Path(path)
    p.write_text(yaml.safe_dump([asdict(s) for s in specs], sort_keys=False), encoding="utf-8")
    return p


def load_param_map(path: str | Path) -> list[ParamSpec]:
    import yaml

    rows = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    return [ParamSpec(**r) for r in rows]


def _check_address(dat: RzwqmDat, spec: ParamSpec) -> None:
    """Hydraulic specs must point where the parsed layout says that field/horizon lives."""
    if spec.field in HYDRAULIC_FIELDS and spec.horizon is not None:
        want = dat.hydraulic_addresses()[spec.field][spec.horizon - 1]
        if spec.address != want:
            raise ValueError(
                f"{spec.name} ({spec.source_name or spec.field}): map says line {spec.address}, "
                f"but the hydraulic block of {dat.path} puts it at {want}"
            )
    elif spec.field in dat.pet_addresses() and spec.address != dat.pet_addresses()[spec.field]:
        raise ValueError(
            f"{spec.name}: map says {spec.address}, PET block has {dat.pet_addresses()[spec.field]}"
        )
    elif spec.field == "rs_min":
        allowed = [d["rs_min"] for d in dat.plant_site_addresses() if "rs_min" in d]
        if spec.address not in allowed:
            raise ValueError(
                f"{spec.name}: map says {spec.address}, plant site params have rs_min at {allowed}"
            )


def params_from_dat(dat: RzwqmDat, param_map: Iterable[ParamSpec] | None = None) -> dict[str, np.ndarray]:
    """Soil hydraulic arrays ``[n_horizon]`` plus the scalar parameters of ``param_map``.

    Keys: every name of :data:`~agri_jax.io.rzwqm.dat.HYDRAULIC_FIELDS` (arrays over all
    horizons, from the parsed block), and ``spec.name`` for each non-horizon spec of the map
    (0-d arrays, read at the spec's address). ``param_map=None`` uses :func:`layout_param_map`.
    """
    specs = list(layout_param_map(dat) if param_map is None else param_map)
    for s in specs:
        _check_address(dat, s)
    out: dict[str, np.ndarray] = dict(dat.hydraulics)
    for s in specs:
        if s.horizon is None:
            out[s.name] = np.asarray(dat.get_float(*s.address))
    return out


def params_to_dat(
    dat: RzwqmDat,
    params: Mapping[str, object],
    param_map: Iterable[ParamSpec] | None = None,
    *,
    decimals: int | None = 5,
) -> RzwqmDat:
    """Write ``params`` back at the addresses of ``param_map``; returns a new :class:`RzwqmDat`.

    For a horizon spec the value is ``params[field][horizon - 1]`` (an array ``[n_horizon]``)
    or, if present, the per-spec scalar ``params[spec.name]``; for a scalar spec it is
    ``params[spec.name]``. Specs without a value in ``params`` are left untouched, so only the
    horizons the map covers are written (GenerateDat behaviour for the CSV map).
    """
    specs = list(layout_param_map(dat) if param_map is None else param_map)
    out = dat.copy()
    for s in specs:
        _check_address(out, s)
        if s.name in params:
            v = params[s.name]
        elif s.horizon is not None and s.field in params:
            v = np.asarray(params[s.field])[s.horizon - 1]
        else:
            continue
        set_value(
            out, s.line_number, s.location_at_line, float(np.asarray(v)), decimals=decimals, inplace=True
        )
    return out
