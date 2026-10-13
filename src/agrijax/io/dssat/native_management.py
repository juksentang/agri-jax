"""Host-side: the management and switch inputs of the DSSAT-CSM v4.8.6.0 water balance from a public
FileX and the engine's standard data (NumPy).

* **Irrigation** (``Management/IRRIG.for``): the daily effective amount ``IRRAMT`` of an
  irrigation schedule "as reported" (``IRRIG = R``): the FileX ``*IRRIGATION`` level of the
  treatment (``IPMAN.for`` formats 55 / 60 at lines 227-228: ``EFIR``, ``IDATE``, ``IROP``,
  ``IRVAL``; ``EFIR <= 0`` -> 1, line 91) goes through ``DSSAT48.INP`` (``optempy2k.for`` formats
  75 / 76, lines 785-786: ``EFIR`` ``F5.3``, amounts ``F5.1``), is read back by ``IRRIG``
  (254-266) and applied on its date: ``DEPIR`` sums the day's events in file order (706; the scan
  stops at the first event after the day, 711, so an event listed after a later one is skipped),
  ``IRRAMT = DEPIR * EFFIRR`` (310-316, 970-971). A negative amount is skipped (387-391).
* **Residue parameters** (``Soil/CERES_OrganicMatter/SoilOrg_init.for`` 136, 152, 252-254;
  ``Soil/SoilUtilities/IPSOIL.for`` 171, 224-270): ``MULCH_AM``, ``MUL_EXTFAC`` and
  ``MUL_WATFAC`` of the previous crop ``PCR`` of the initial conditions, the first row of that crop
  in the engine's ``StandardData/RESCH048.SDA`` (the first row when the crop is blank or not
  listed).
* **Extinction coefficient** ``KSEVAP = KTRANS = KEP`` of CERES-Maize (``Plant/plant.for`` 406-407;
  ``Plant/CERES-Maize/MZ_PHENOL.for`` 338): ``KEP = KCAN / (1 - 0.07) * (1 - 0.25)`` with the
  ecotype's ``KCAN`` (``.ECO``).
* **Water table** (``Soil/SoilWater/WaterTable.f90`` 79-85): no initial water table (``ICWD``
  missing or not positive) -> ``ActWTD = 1000`` cm.
* **Switches** (``InputModule/IPSIM.for`` 253-256, 664-666, 694-696, 709-713): ``MESEV`` R / S
  (default R), ``MEINF``, ``MESOL`` (default 2), ``MESOM`` (default G), ``CO2`` (default M).

:func:`unsupported_features` lists what of a treatment the native inputs cannot reproduce without
porting another DSSAT process (automatic irrigation, flooding and water-table management, tillage,
surface residue and its decay, the CENTURY organic matter, environment modifications, plastic
mulch, tile drainage...); :func:`agrijax.sites.dssat_free_run.free_run_inputs` refuses such
treatments.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agrijax.core.coefficients import Coefficients, Provenance, coef

from ._f77 import round_trip
from ._fixed import read_lines
from .observed import y4k_date

__all__ = [
    "DSSAT_MANAGEMENT_INPUT",
    "ManagementInputCoefficients",
    "ResidueParameters",
    "Switches",
    "extinction_kep",
    "irrigation_amounts",
    "residue_parameters",
    "switches",
    "treatment_levels",
    "unsupported_features",
]

REF = "dssat-4.8.6.0"
_F = np.float32


def _p(file_line: str, routine: str, note: str = "") -> Provenance:
    return Provenance.at(REF, file_line, routine=routine, note=note)


class ManagementInputCoefficients(Coefficients):
    """Constants of the irrigation, residue, water-table and ``KEP`` inputs."""

    effirr_default: float = coef(
        1.0,
        "-",
        "irrigation efficiency used when EFIR is missing or not positive",
        _p("InputModule/IPMAN.for:91", "IPIRR"),
        calibrate=False,
    )
    effirr_min: float = coef(
        1.0e-3,
        "-",
        "an efficiency below this is replaced by the default",
        _p("Management/IRRIG.for:316", "IRRIG"),
        calibrate=False,
    )
    amount_min: float = coef(
        -1.0e-6,
        "mm",
        "an irrigation amount below this is skipped",
        _p("Management/IRRIG.for:387", "IRRIG"),
        calibrate=False,
    )
    kep_scatter: float = coef(
        0.07,
        "-",
        "the 1 - 0.07 denominator of KEP = KCAN / (1 - 0.07) * (1 - 0.25)",
        _p("Plant/CERES-Maize/MZ_PHENOL.for:338", "MZ_PHENOL"),
    )
    kep_reduction: float = coef(
        0.25,
        "-",
        "the 1 - 0.25 factor of KEP = KCAN / (1 - 0.07) * (1 - 0.25)",
        _p("Plant/CERES-Maize/MZ_PHENOL.for:338", "MZ_PHENOL"),
    )
    no_water_table: float = coef(
        1000.0,
        "cm",
        "water table depth meaning 'no water table'",
        _p("Soil/SoilWater/WaterTable.f90:80", "WaterTable"),
        calibrate=False,
    )
    wtd_min: float = coef(
        1.0e-6,
        "cm",
        "an initial water table depth below this is no water table",
        _p("Soil/SoilWater/WaterTable.f90:79", "WaterTable"),
        calibrate=False,
    )
    residue_am_default: float = coef(
        32.0,
        "cm2 g-1",
        "residue cover per unit mass when RESCH048.SDA has no row",
        _p("Soil/SoilUtilities/IPSOIL.for:98", "IPSOIL"),
        calibrate=False,
    )
    residue_watfac_default: float = coef(
        3.8,
        "-",
        "residue saturation water per unit mass (kg kg-1) when RESCH048.SDA has no row",
        _p("Soil/SoilUtilities/IPSOIL.for:99", "IPSOIL"),
        calibrate=False,
    )
    residue_extfac_default: float = coef(
        0.80,
        "-",
        "mulch light extinction coefficient when RESCH048.SDA has no row",
        _p("Soil/SoilUtilities/IPSOIL.for:100", "IPSOIL"),
        calibrate=False,
    )
    residue_incorporation_min: float = coef(
        0.01,
        "%",
        "an initial residue incorporation ICRIP below this is none (the residue stays on the surface)",
        _p("Soil/CERES_OrganicMatter/SoilOrg_init.for:144", "SoilOrg_init"),
        calibrate=False,
    )
    residue_depth_min: float = coef(
        0.01,
        "cm",
        "an initial residue incorporation depth ICRID below this is none (the residue stays on the surface)",
        _p("Soil/CERES_OrganicMatter/SoilOrg_init.for:145", "SoilOrg_init", "and ICRIP = 0 at line 146"),
        calibrate=False,
    )
    residue_surface_min: float = coef(
        1.0e-3,
        "kg ha-1",
        "initial surface residue above this makes a mulch layer",
        _p("Soil/CERES_OrganicMatter/SoilOrg_init.for:160", "SoilOrg_init"),
        calibrate=False,
    )


DSSAT_MANAGEMENT_INPUT = ManagementInputCoefficients()


# ------------------------------------------------------------------------ FileX helpers
def treatment_levels(x: Mapping[str, Any], trno: int) -> dict[str, Any]:
    """The ``*TREATMENTS`` row of treatment ``trno`` of a :func:`~agrijax.io.dssat.read_filex` dict."""
    tr = next((t for t in x["TREATMENTS"] if int(t["N"]) == int(trno)), None)
    if tr is None:
        raise ValueError(f"no treatment {trno} in the experiment file")
    return dict(tr)


def _sim(x: Mapping[str, Any], tr: Mapping[str, Any]) -> dict[str, dict[str, Any]] | None:
    """The treatment's ``*SIMULATION CONTROLS`` level; ``None`` when it has none (``SM = 0``, or a
    level the file does not hold): ``IPSIM`` then uses its built-in defaults (``IPSIM.for`` 110-160)."""
    sc = x.get("SIMULATION CONTROLS") or {}
    lev = int(tr.get("SM", 0) or 0)
    return sc.get(lev) if lev > 0 else None


#: the switch defaults of a blank column of a simulation-control level, and (``None``) of a
#: treatment without a level: ``IPSIM.for`` 110-160 (batch run) then has irrigation and residue
#: applications "as reported", tillage and nitrogen on
_BLANK = {"nitro": "N", "irrig": "N", "resid": "N", "till": "N"}
_NO_LEVEL = {"nitro": "Y", "irrig": "R", "resid": "R", "till": "Y"}


def _code(v: Any, default: str) -> str:
    s = "" if v is None else str(v).strip().upper()
    return s[:1] if s and s not in {"-99", "-9"} else default


@dataclass(frozen=True)
class Switches:
    """The simulation options of a treatment as ``IPSIM`` normalises them."""

    water: str
    nitro: str
    co2: str
    wther: str
    evapo: str
    infil: str
    mesom: str
    mesev: str
    mesol: str
    irrig: str
    resid: str
    till: str


def switches(x: Mapping[str, Any], trno: int) -> Switches:
    """Host-side: the treatment's simulation options (module docstring; batch-run defaults)."""
    tr = treatment_levels(x, trno)
    lev = _sim(x, tr)
    dflt = _BLANK if lev is not None else _NO_LEVEL
    sc = lev or {}
    op, me, ma = sc.get("OPTIONS", {}), sc.get("METHODS", {}), sc.get("MANAGEMENT", {})
    co2 = _code(op.get("CO2"), "M")
    mesom = _code(me.get("MESOM"), "G")
    mesev = _code(me.get("MESEV"), "R")
    mesol = str(me.get("MESOL", "2")).strip()[:1]
    return Switches(
        water=_code(op.get("WATER"), "Y"),
        nitro=_code(op.get("NITRO"), dflt["nitro"]),
        co2=co2 if co2 in "WMD" else "M",
        wther=_code(me.get("WTHER"), "M"),
        evapo=_code(me.get("EVAPO"), "R"),
        infil=_code(me.get("INFIL"), "S"),
        mesom=mesom if mesom in "PG" else "G",
        mesev=mesev if mesev in "RS" else "R",
        mesol=mesol if mesol and mesol in "123" else "2",
        irrig=_code(ma.get("IRRIG"), dflt["irrig"]),
        resid=_code(ma.get("RESID"), dflt["resid"]),
        till=_code(op.get("TILL"), dflt["till"]),
    )


def _num(v: Any) -> float:
    if v is None:
        return math.nan
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except ValueError:
        return math.nan


def _missing(v: Any) -> bool:
    x = _num(v)
    return math.isnan(x) or x <= -99.0


# ------------------------------------------------------------------------ irrigation
def irrigation_amounts(
    x: Mapping[str, Any],
    trno: int,
    days: Sequence[int],
    c: ManagementInputCoefficients = DSSAT_MANAGEMENT_INPUT,
) -> np.ndarray:
    """Host-side: the effective irrigation ``IRRAMT`` [mm, REAL*4] of each of ``days`` (``YYYYDDD``,
    the first the simulation start) for a schedule "as reported" (``IRRIG = R``) or none
    (``IRRIG = N``, or no irrigation level); other methods raise ``NotImplementedError``."""
    tr = treatment_levels(x, trno)
    sw = switches(x, trno)
    n = len(days)
    out = np.zeros(n, dtype=_F)
    lev = int(tr.get("MI", 0) or 0)
    if sw.irrig == "N" or lev == 0:
        return out
    if sw.irrig != "R":
        raise NotImplementedError(f"irrigation method IRRIG = {sw.irrig} is not implemented (R, N are)")
    rec = x.get("IRRIGATION AND WATER MANAGEMENT", {}).get(lev)
    if rec is None:
        raise ValueError(f"treatment {trno}: irrigation level {lev} not in the experiment file")
    eff = _num(rec.get("EFIR"))
    effirx = _F(c.effirr_default) if math.isnan(eff) or eff <= 0 else _F(eff)
    effirr = round_trip(effirx, 5, 3)
    if effirr < _F(c.effirr_min):
        effirr = _F(c.effirr_default)
    first = int(days[0])
    pos = {int(d): i for i, d in enumerate(days)}
    depir = np.zeros(n, dtype=_F)
    latest = -1  # the latest date of the events listed so far
    for r in rec.get("rows", []):
        code = str(r.get("IROP", "")).strip().upper()
        try:
            op = int(code[2:5])
        except ValueError as e:
            raise ValueError(f"treatment {trno}: irrigation operation {code!r}") from e
        if not 1 <= op <= 6:
            raise NotImplementedError(
                f"irrigation operation {code} (water table, flood, bund, puddling) is not implemented"
            )
        amt = _num(r.get("IRVAL"))
        if math.isnan(amt):
            amt = 0.0
        if _F(amt) < _F(c.amount_min):
            continue
        d = y4k_date(int(_num(r["IDATE"])), first_weather=first)
        yrdoy = d.year * 1000 + d.timetuple().tm_yday
        if yrdoy < first:
            raise ValueError(
                f"treatment {trno}: irrigation on {yrdoy} before the simulation start (DSSAT stops)"
            )
        # IRRIG.for 704-713 scans the events in file order and stops at the first one after the
        # day: an event listed after a later-dated one is never applied
        if yrdoy in pos and latest <= yrdoy:
            i = pos[yrdoy]
            depir[i] = _F(depir[i] + round_trip(amt, 5, 1))
        latest = max(latest, yrdoy)
    for i in range(n):
        out[i] = _F(depir[i] * effirr) if effirr > 0 else depir[i]
    return out


# ------------------------------------------------------------------------ residue parameters
@dataclass(frozen=True)
class ResidueParameters:
    """``MULCH_AM`` [cm2 g-1], ``MUL_WATFAC`` [kg kg-1] and ``MUL_EXTFAC`` [-] (REAL*4)."""

    am: float
    watfac: float
    extfac: float
    restype: str


def residue_parameters(
    crop: str, resch: str | Path | None, c: ManagementInputCoefficients = DSSAT_MANAGEMENT_INPUT
) -> ResidueParameters:
    """Host-side: the residue characteristics ``IPSOIL`` gives the previous crop ``crop`` (module
    docstring), from the engine's ``RESCH048.SDA`` (``None``: the built-in defaults)."""
    rows: list[tuple[str, str, np.float32, np.float32, np.float32]] = []
    if resch is not None:
        for ln in read_lines(resch):
            if not ln.strip() or ln[:1] in "*@!":
                continue
            try:
                vals = tuple(_F(float(ln[a:b])) for a, b in ((9, 17), (17, 25), (25, 33)))
            except ValueError:
                continue
            rows.append((ln[1:6], ln[7:9], vals[0], vals[1], vals[2]))
    if not rows:
        return ResidueParameters(
            float(_F(c.residue_am_default)),
            float(_F(c.residue_watfac_default)),
            float(_F(c.residue_extfac_default)),
            "RE001",
        )
    cr = (crop or "")[:2].ljust(2)
    pick = rows[0]
    if cr.strip():
        pick = next((r for r in rows if r[1] == cr), rows[0])
    return ResidueParameters(float(pick[2]), float(pick[3]), float(pick[4]), pick[0])


# ------------------------------------------------------------------------ KEP
def extinction_kep(kcan: float, c: ManagementInputCoefficients = DSSAT_MANAGEMENT_INPUT) -> float:
    """Host-side: CERES-Maize's ``KEP`` (= ``KSEVAP`` = ``KTRANS``) from the ecotype's ``KCAN``, REAL*4."""
    den = _F(_F(1.0) - _F(c.kep_scatter))
    fac = _F(_F(1.0) - _F(c.kep_reduction))
    return float(_F(_F(_F(kcan) / den) * fac))


# ------------------------------------------------------------------------ scope
def unsupported_features(
    x: Mapping[str, Any], trno: int, c: ManagementInputCoefficients = DSSAT_MANAGEMENT_INPUT
) -> list[str]:
    """Host-side: what of treatment ``trno`` the native inputs do not reproduce (empty: supported).

    Each entry names the FileX feature and the DSSAT process it needs."""
    tr = treatment_levels(x, trno)
    sw = switches(x, trno)
    out: list[str] = []
    if sw.water != "Y":
        out.append("WATER != Y: the free run simulates the soil water")
    if sw.wther != "M":
        out.append(f"WTHER = {sw.wther}: generated weather (WGEN / SIMMETEO) not implemented")
    if sw.evapo != "R":
        out.append(f"EVAPO = {sw.evapo}: the free-run day runs Priestley-Taylor (R) only")
    if sw.mesom == "P":
        out.append(
            "MESOM = P: CENTURY organic matter (surface residue decay, SOILDYN soil changes) not implemented"
        )
    if sw.mesol == "1":
        out.append("MESOL = 1: LYRSET soil-layer redistribution not implemented")
    if sw.irrig not in ("N", "R"):
        out.append(f"IRRIG = {sw.irrig}: automatic or days-after-planting irrigation (IRRIG) not implemented")
    mi = int(tr.get("MI", 0) or 0)
    if mi and sw.irrig == "R":
        for r in x.get("IRRIGATION AND WATER MANAGEMENT", {}).get(mi, {}).get("rows", []):
            code = str(r.get("IROP", "")).strip().upper()
            if not (code[2:5].isdigit() and 1 <= int(code[2:5]) <= 6):
                out.append(
                    f"irrigation operation {code}: water table / flood / bund / puddling not implemented"
                )
                break
    if int(tr.get("ME", 0) or 0) > 0:
        out.append("environment modifications (ME level; WTHMOD) not implemented")
    if int(tr.get("MT", 0) or 0) > 0 and sw.till in ("Y", "R"):
        out.append("tillage (MT level; SOILDYN TillEvent soil changes) not implemented")
    if int(tr.get("MR", 0) or 0) > 0 and sw.resid != "N":
        out.append("organic residue applications (MR level; SoilOrg residue, mulch, SOILDYN) not implemented")
    ic_lev = int(tr.get("IC", 0) or 0)
    ic = x.get("INITIAL CONDITIONS", {}).get(ic_lev, {}) if ic_lev else {}
    if ic:
        if not _missing(ic.get("ICWD")) and _num(ic.get("ICWD")) >= c.wtd_min:
            out.append("initial water table depth ICWD (WaterTable) not implemented")
        icres = _num(ic.get("ICRES"))
        icrip = _num(ic.get("ICRIP"))
        icrid = _num(ic.get("ICRID"))
        if not math.isnan(icres) and icres > c.residue_surface_min:
            none = math.isnan(icrip) or icrip < c.residue_incorporation_min
            none = none or math.isnan(icrid) or icrid < c.residue_depth_min
            rip = 0.0 if none else icrip
            surface = icres - icres * rip / 100.0
            if surface > c.residue_surface_min:
                out.append(
                    "initial surface residue (ICRES with ICRIP < 100): mulch decay of SoilOrg not implemented"
                )
    fl = x.get("FIELDS", {}).get(int(tr.get("FL", 0) or 0), {})
    if not _missing(fl.get("PMALB")) and not _missing(fl.get("PMWD")) and _num(fl.get("PMWD")) > 0:
        out.append("plastic mulch (PMALB / PMWD) not implemented")
    if not _missing(fl.get("FLDD")) and _num(fl.get("FLDD")) > 0:
        # TILEDRAIN.for 72-81: a positive tile depth puts a drain in the profile
        out.append("tile drainage (FLDD > 0; TILEDRAIN) not implemented")
    return out
