"""Reader / writer for the RZWQM2 main parameter file ``rzwqm.dat``.

File structure
--------------
``rzwqm.dat`` is a sequence of *comment blocks* (every line whose first character is ``=``)
and *data blocks* (maximal runs of non-comment lines). The data blocks appear in a fixed
order, each preceded by the comment block that documents it. Values are addressed the way
``RZWQM_Tool/LHS_ana_Gen/GenerateDat.py`` and ``all_parameters.csv`` address them:

* ``line_no``  -- 1-based line number in the file (line breaks as Python's universal newlines
  count them, i.e. ``\\r\\n``, ``\\n`` and ``\\r`` each end a line);
* ``token_idx`` -- 0-based index into ``line.split()``.

:class:`RzwqmDat` keeps the raw lines (terminators included) so that ``read -> write`` is
byte-identical, and parses the PoC-relevant sections on access (always from the current
lines, so they reflect :func:`set_value` edits):

=====================  ====================================================================
section                comment block that precedes it (matched case-insensitively)
=====================  ====================================================================
physiography           first data block (``area of the field`` ... ``ambient co2``)
horizons               ``SOIL SYSTEM PHYSICAL CONFIGURATION``
nodes                  ``DEPTH DISCRETIZATION FOR MOISTURE & HEAT MODELS``
soil_physical          ``SOIL HORIZON PHYSICAL PROPERTIES``
hydraulics             ``SOIL HORIZON HYDRAULIC PROPERTIES``
pet                    ``albedo of the dry soil`` (POTENTIAL EVAPORATION parameters)
plants                 ``PLANT MODEL CONTROL``
plant_site_params      ``SITE SPECIFIC PARAMETERS``
plantings              ``planting control and harvest parameters``
=====================  ====================================================================

Everything else stays available as raw lines / :attr:`RzwqmDat.blocks`.

Hydraulic block (per horizon, three records; verified against the comment block of the
RZWQM2 4.5/4.6 file and the use of ``SOILHP(1..13, IH)`` in ``Rzmain.for``/``RZTEST.for``)::

    rec 1:  horizon  hb  lam  eps  ksat  theta_r  theta_s
    rec 2:  theta_fc33  theta_fc10  theta_wp  hb_k  c2  n1  a1
    rec 3:  ksat_lat

    hb          SOILHP(1)   bubbling pressure of the theta(h) curve            [cm]
    lam         SOILHP(2)   Brooks-Corey pore size distribution index           [-]
    eps         SOILHP(3)   exponent of the K(h) curve (N2 in RZTEST)           [-]
    ksat        SOILHP(4)   saturated hydraulic conductivity (C1 in RZTEST)     [cm/hr]
    theta_r     SOILHP(5)   residual water content                              [cm3/cm3]
    theta_s     SOILHP(6)   saturated water content (porosity used by Richards) [cm3/cm3]
    theta_fc33  SOILHP(7)   field capacity, 1/3 bar                              [cm3/cm3]
    theta_fc10  SOILHP(8)   field capacity, 1/10 bar                             [cm3/cm3]
    theta_wp    SOILHP(9)   wilting point, 15 bar                                [cm3/cm3]
    hb_k        SOILHP(10)  bubbling pressure of the K(h) curve                  [cm]
    c2          SOILHP(11)  second intercept of the K(h) curve                   [-]
    n1          SOILHP(12)  first exponent of the K(h) curve                     [-]
    a1          SOILHP(13)  constant of the theta(h) curve                       [-]
    ksat_lat    --          lateral saturated conductivity to the drain          [cm/hr]

Formatting of edits (GenerateDat compatibility)
-----------------------------------------------
:func:`set_value` does exactly what ``GenerateDat._apply_edit_generic`` does to the edited
line: the value is rounded to 5 decimals (``np.round``, as the LHS matrix is) and formatted
with ``f"{value:<05}"``; the line is re-joined with single spaces (trailing spaces dropped).
Unedited lines are untouched. ``GenerateDat.py`` reads in universal-newline text mode, so on
Linux every line of its output ends in ``\\n``; pass ``newline="\\n"`` to
:func:`write_rzwqm_dat` to reproduce its output byte for byte. The default
(``newline=None``) keeps each line's own terminator (CRLF for files saved by the Windows GUI).
Deliberate difference: an out-of-range ``line_no``/``token_idx`` raises ``IndexError``
where GenerateDat silently skips the edit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "HYDRAULIC_FIELDS",
    "HYDRAULIC_UNITS",
    "PET_FIELDS",
    "PHYSIOGRAPHY_FIELDS",
    "PLANT_SITE_FIELDS",
    "Address",
    "Block",
    "Planting",
    "RzwqmDat",
    "format_value",
    "read_rzwqm_dat",
    "set_value",
    "write_rzwqm_dat",
]

Address = tuple[int, int]
"""``(line_no, token_idx)``: 1-based line number, 0-based token index (GenerateDat convention)."""

_LINE_RE = re.compile(r"[^\r\n]*(?:\r\n|\n|\r)|[^\r\n]+\Z")
_ENCODING = "latin-1"

PHYSIOGRAPHY_FIELDS: tuple[str, ...] = (
    "area_ha",
    "elevation_m",
    "aspect_rad",
    "latitude_rad",
    "slope_rad",
    "longitude_rad",
    "rainfall_zone",
    "co2_ppm",
)

# (record, token) of every hydraulic field inside one horizon's 3-record group.
_HYDRAULIC_LAYOUT: dict[str, tuple[int, int]] = {
    "hb": (0, 1),
    "lam": (0, 2),
    "eps": (0, 3),
    "ksat": (0, 4),
    "theta_r": (0, 5),
    "theta_s": (0, 6),
    "theta_fc33": (1, 0),
    "theta_fc10": (1, 1),
    "theta_wp": (1, 2),
    "hb_k": (1, 3),
    "c2": (1, 4),
    "n1": (1, 5),
    "a1": (1, 6),
    "ksat_lat": (2, 0),
}
HYDRAULIC_FIELDS: tuple[str, ...] = tuple(_HYDRAULIC_LAYOUT)
HYDRAULIC_UNITS: dict[str, str] = {
    "hb": "cm",
    "lam": "1",
    "eps": "1",
    "ksat": "cm/hr",
    "theta_r": "cm3/cm3",
    "theta_s": "cm3/cm3",
    "theta_fc33": "cm3/cm3",
    "theta_fc10": "cm3/cm3",
    "theta_wp": "cm3/cm3",
    "hb_k": "cm",
    "c2": "1",
    "n1": "1",
    "a1": "1",
    "ksat_lat": "cm/hr",
}

PET_FIELDS: tuple[str, ...] = (
    "albedo_dry",
    "albedo_wet",
    "albedo_crop",
    "albedo_residue",
    "wind_height_m",
    "sunshine_frac",
    "pan_coef",
    "hourly_weather",
    "use_shaw",
    "use_penflux",
    "soil_resistance",  # surface soil resistance [s/m]
    "water_stress_method",
    "plastic_cover_frac",
    "emit_canopy",
    "emit_residue",
    "emit_snow",
    "emit_soil",
    "emit_plastic",
    "pet_method",  # 0 = Shuttleworth-Wallace, 1 = FAO tall, 2 = FAO short
    "kc_init",
    "kc_max",
    "albedo_plastic",
    "transmissivity_plastic",
    "rain_intercept_plastic",
)

PLANT_SITE_FIELDS: tuple[str, ...] = (
    "cnup1",  # max N uptake rate [g/plant/day]
    "resp_frac",  # proportion of photosynthate respired
    "biomass_lai1",  # biomass for LAI = 1 [g]
    "density_ref",  # plant density on which biomass_lai1 is based
    "age_effect_propagule",
    "age_effect_seed",
    "root_depth_max_m",
    "rs_min",  # potential minimum leaf stomatal resistance [s/m]
    "nsi_threshold",
    "n_luxury_eff",
    "wsi",  # water stress sensitivity, (AT/PT)^WSI
    "leaf_angle_x",
    "t_transp_min",  # temperature above which the plant transpires [C]
    "leaf_psi_crit_m",
    "rs_exponent",
    "leaf_resistance",
    "p_frac_emergence",
    "p_frac_maturity",
    "p_frac_50",
    "p_uptake_dist",
    "p_stress",
)

# section name -> keyword that must appear in the comment block preceding it.
_SECTION_KEYWORDS: dict[str, str] = {
    "physiography": "area of the field",
    "horizons": "SOIL SYSTEM PHYSICAL CONFIGURATION",
    "nodes": "DEPTH DISCRETIZATION",
    "soil_physical": "SOIL HORIZON PHYSICAL PROPERTIES",
    "hydraulics": "SOIL HORIZON HYDRAULIC PROPERTIES",
    "heat": "SOIL HORIZON HEAT MODEL PARAMETERS",
    "pet": "albedo of the dry soil",
    "plants": "PLANT MODEL CONTROL",
    "plant_site_params": "SITE SPECIFIC PARAMETERS",
    "plantings": "planting control and harvest parameters",
}


def _split_lines(text: str) -> list[str]:
    return _LINE_RE.findall(text)


def _strip_eol(line: str) -> str:
    return line.rstrip("\r\n")


def _eol(line: str) -> str:
    return line[len(_strip_eol(line)) :]


def _num(tok: str) -> float:
    return float(tok.replace("D", "E").replace("d", "e"))


def format_value(value: float | int | str, decimals: int | None = 5) -> str:
    """Format a value exactly as GenerateDat.py does: ``np.round(v, 5)`` then ``f"{v:<05}"``.

    Strings are inserted verbatim. ``decimals=None`` skips the rounding.
    """
    if isinstance(value, str):
        return value
    v = np.float64(value)
    if decimals is not None:
        v = np.round(v, decimals)
    return f"{v:<05}"


@dataclass(frozen=True)
class Block:
    """One data block: a maximal run of non-comment lines and the comment block before it."""

    index: int
    start: int  # 1-based line number of the first data line
    stop: int  # 1-based line number of the last data line (inclusive)
    header: str  # preceding comment lines, joined with "\n", EOLs stripped

    @property
    def line_numbers(self) -> range:
        return range(self.start, self.stop + 1)


@dataclass(frozen=True)
class Planting:
    """One planting/harvest entry of the PLANT MANAGEMENT block (three records)."""

    plant_ref: int
    planting_date: np.datetime64
    row_spacing_cm: float
    planting_depth_layer: int
    density_seeds_ha: float
    planting_method: int
    harvest_option: int  # 1 growth stage, 2 growth class, 3 fixed date
    harvest_growth_stage: float
    harvest_growth_class: int
    harvest_threshold: float
    harvest_date: np.datetime64 | None  # the date of option 3 (None when not a valid date)
    stubble_height_cm: float
    harvest_efficiency: float
    harvest_type: int
    soil_water_at_planting_cm: float
    planting_window_days: int
    line_no: int  # 1-based line of the first record
    density_address: Address


def _date(dd: str, mm: str, yyyy: str) -> np.datetime64 | None:
    try:
        return np.datetime64(f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}")
    except ValueError:
        return None


class RzwqmDat:
    """Raw lines of an ``rzwqm.dat`` plus parsed views of the PoC-relevant sections."""

    def __init__(self, lines: list[str], path: Path | None = None) -> None:
        self.lines: list[str] = list(lines)
        self.path = path
        self.blocks: list[Block] = self._find_blocks()
        self._sections: dict[str, Block] = self._index_sections()

    # ------------------------------------------------------------------ structure
    def _find_blocks(self) -> list[Block]:
        blocks: list[Block] = []
        header: list[str] = []
        pending: list[str] = []
        start: int | None = None
        for i, raw in enumerate(self.lines, start=1):
            line = _strip_eol(raw)
            if line.startswith("="):
                if start is not None:
                    blocks.append(Block(len(blocks), start, i - 1, "\n".join(pending)))
                    start, header = None, []
                header.append(line)
            else:
                if start is None:
                    start, pending, header = i, header, []
        if start is not None:
            blocks.append(Block(len(blocks), start, len(self.lines), "\n".join(pending)))
        return blocks

    def _index_sections(self) -> dict[str, Block]:
        out: dict[str, Block] = {}
        for name, kw in _SECTION_KEYWORDS.items():
            for b in self.blocks:
                if kw.lower() in b.header.lower():
                    out[name] = b
                    break
        return out

    def section(self, name: str) -> Block:
        """The data block of a named section (see the module docstring)."""
        try:
            return self._sections[name]
        except KeyError:
            raise KeyError(
                f"section {name!r} not found (looked for a comment block containing "
                f"{_SECTION_KEYWORDS.get(name)!r})"
            ) from None

    def has_section(self, name: str) -> bool:
        return name in self._sections

    @property
    def newline(self) -> str:
        """Dominant line terminator of the file."""
        eols = [_eol(x) for x in self.lines if _eol(x)]
        return max(set(eols), key=eols.count) if eols else "\n"

    def line(self, line_no: int) -> str:
        """Content of a 1-based line without its terminator."""
        return _strip_eol(self.lines[line_no - 1])

    def tokens(self, line_no: int) -> list[str]:
        return self.line(line_no).split()

    def get(self, line_no: int, token_idx: int) -> str:
        """Raw token at ``(line_no, token_idx)``."""
        return self.tokens(line_no)[token_idx]

    def get_float(self, line_no: int, token_idx: int) -> float:
        return _num(self.get(line_no, token_idx))

    def copy(self) -> RzwqmDat:
        return RzwqmDat(self.lines, self.path)

    def to_text(self, newline: str | None = None) -> str:
        if newline is None:
            return "".join(self.lines)
        return "".join(_strip_eol(x) + newline for x in self.lines)

    # ------------------------------------------------------------------ addresses
    def _row(self, section: str, k: int = 0) -> int:
        return self.section(section).start + k

    @property
    def n_horizon(self) -> int:
        return int(self.get(self._row("horizons"), 0))

    @property
    def n_node(self) -> int:
        return int(self.get(self._row("nodes"), 0))

    def physiography_addresses(self) -> dict[str, Address]:
        row = self._row("physiography")
        return {name: (row, j) for j, name in enumerate(PHYSIOGRAPHY_FIELDS)}

    def pet_addresses(self) -> dict[str, Address]:
        row = self._row("pet")
        n = len(self.tokens(row))
        return {name: (row, j) for j, name in enumerate(PET_FIELDS[:n])}

    def hydraulic_addresses(self) -> dict[str, list[Address]]:
        """``field -> [address of horizon 1, ..., horizon n]`` for every hydraulic field."""
        b = self.section("hydraulics")
        nh = self.n_horizon
        if b.stop - b.start + 1 < 3 * nh:
            raise ValueError(f"hydraulic block at line {b.start} has fewer than 3*{nh} records")
        return {
            name: [(b.start + 3 * h + rec, tok) for h in range(nh)]
            for name, (rec, tok) in _HYDRAULIC_LAYOUT.items()
        }

    def plant_site_addresses(self) -> list[dict[str, Address]]:
        b = self.section("plant_site_params")
        out = []
        for row in b.line_numbers:
            n = len(self.tokens(row))
            out.append({name: (row, j) for j, name in enumerate(PLANT_SITE_FIELDS[:n])})
        return out

    # ------------------------------------------------------------------ parsed views
    @property
    def physiography(self) -> dict[str, float]:
        return {k: self.get_float(*a) for k, a in self.physiography_addresses().items()}

    @property
    def profile_depth_cm(self) -> float:
        return self.get_float(self._row("horizons"), 1)

    @property
    def horizon_depths_cm(self) -> np.ndarray:
        """Lower depth of each horizon [cm], shape ``[n_horizon]``."""
        row = self._row("horizons")
        return np.array([self.get_float(row + 1 + h, 0) for h in range(self.n_horizon)])

    @property
    def node_depths_cm(self) -> np.ndarray:
        """Depth of each numerical node [cm], shape ``[n_node]``."""
        row = self._row("nodes")
        return np.array([self.get_float(row + 1 + k, 1) for k in range(self.n_node)])

    @property
    def node_spacing_cm(self) -> np.ndarray:
        """Distance from each node to the next one [cm], shape ``[n_node]``."""
        row = self._row("nodes")
        return np.array([self.get_float(row + 1 + k, 2) for k in range(self.n_node)])

    @property
    def soil_physical(self) -> dict[str, Any]:
        """Texture name, texture code, particle/bulk density, porosity, sand/silt/clay per horizon."""
        row = self._row("soil_physical")
        names, rows = [], []
        for h in range(self.n_horizon):
            names.append(self.line(row + 2 * h).strip())
            rows.append([_num(t) for t in self.tokens(row + 2 * h + 1)[:7]])
        a = np.array(rows)
        return {
            "texture": names,
            "texture_code": a[:, 0].astype(int),
            "particle_density": a[:, 1],
            "bulk_density": a[:, 2],
            "porosity": a[:, 3],
            "sand": a[:, 4],
            "silt": a[:, 5],
            "clay": a[:, 6],
        }

    @property
    def hydraulics(self) -> dict[str, np.ndarray]:
        """Brooks-Corey parameters per horizon, ``field -> array[n_horizon]`` (see module docs)."""
        return {
            name: np.array([self.get_float(*a) for a in addrs])
            for name, addrs in self.hydraulic_addresses().items()
        }

    @property
    def pet(self) -> dict[str, float]:
        """POTENTIAL EVAPORATION parameters (albedos, soil resistance, PET method, ...)."""
        return {k: self.get_float(*a) for k, a in self.pet_addresses().items()}

    @property
    def plants(self) -> list[str]:
        """Plant names of the PLANT MODEL CONTROL block, in reference-number order."""
        row = self._row("plants")
        n = int(self.get(row, 0))
        return [self.line(row + 1 + k).strip() for k in range(n)]

    @property
    def plant_site_params(self) -> list[dict[str, float]]:
        """Generic-plant site parameters, one dict per parameterised plant."""
        return [{k: self.get_float(*a) for k, a in d.items()} for d in self.plant_site_addresses()]

    @property
    def plantings(self) -> list[Planting]:
        """Planting / harvest schedule of the PLANT MANAGEMENT block."""
        row = self._row("plantings")
        n = int(self.get(row, 0))
        out = []
        for k in range(n):
            r = row + 1 + 3 * k
            a, b, c = self.tokens(r), self.tokens(r + 1), self.tokens(r + 2)
            pdate = _date(a[1], a[2], a[3])
            if pdate is None:
                raise ValueError(f"invalid planting date on line {r}: {a[1:4]}")
            out.append(
                Planting(
                    plant_ref=int(a[0]),
                    planting_date=pdate,
                    row_spacing_cm=_num(a[4]),
                    planting_depth_layer=int(a[5]),
                    density_seeds_ha=_num(a[6]),
                    planting_method=int(a[7]),
                    harvest_option=int(b[0]),
                    harvest_growth_stage=_num(b[1]),
                    harvest_growth_class=int(b[2]),
                    harvest_threshold=_num(b[3]),
                    harvest_date=_date(b[4], b[5], b[6]) if len(b) >= 7 else None,
                    stubble_height_cm=_num(c[0]),
                    harvest_efficiency=_num(c[1]),
                    harvest_type=int(c[2]),
                    soil_water_at_planting_cm=_num(c[3]) if len(c) > 3 else float("nan"),
                    planting_window_days=int(c[4]) if len(c) > 4 else 0,
                    line_no=r,
                    density_address=(r, 6),
                )
            )
        return out

    def __repr__(self) -> str:
        return f"RzwqmDat(path={self.path!s}, n_lines={len(self.lines)}, n_blocks={len(self.blocks)})"


def read_rzwqm_dat(path: str | Path) -> RzwqmDat:
    """Read ``rzwqm.dat`` (latin-1, any line terminators) into an :class:`RzwqmDat`."""
    p = Path(path)
    text = p.read_bytes().decode(_ENCODING)
    return RzwqmDat(_split_lines(text), p)


def set_value(
    dat: RzwqmDat,
    line_no: int,
    token_idx: int,
    value: float | int | str,
    *,
    decimals: int | None = 5,
    inplace: bool = False,
) -> RzwqmDat:
    """Replace one token, formatted as GenerateDat.py does; returns the edited dat.

    The edited line becomes ``" ".join(tokens)`` + its original terminator.
    ``inplace=False`` (default) leaves ``dat`` untouched and returns a copy.
    """
    if not 1 <= line_no <= len(dat.lines):
        raise IndexError(f"line_no {line_no} outside 1..{len(dat.lines)}")
    raw = dat.lines[line_no - 1]
    parts = _strip_eol(raw).split()
    if not 0 <= token_idx < len(parts):
        raise IndexError(f"token_idx {token_idx} outside 0..{len(parts) - 1} on line {line_no}: {raw!r}")
    if _strip_eol(raw).startswith("="):
        raise ValueError(f"line {line_no} is a comment line")
    parts[token_idx] = format_value(value, decimals)
    out = dat if inplace else dat.copy()
    out.lines[line_no - 1] = " ".join(parts) + (_eol(raw) or "\n")
    return out


def write_rzwqm_dat(dat: RzwqmDat, path: str | Path, *, newline: str | None = None) -> Path:
    """Write ``dat`` to ``path`` (latin-1).

    ``newline=None`` keeps every line's own terminator (byte-identical round trip);
    ``newline="\\n"`` reproduces GenerateDat.py's output on Linux.
    """
    p = Path(path)
    p.write_bytes(dat.to_text(newline).encode(_ENCODING))
    return p
