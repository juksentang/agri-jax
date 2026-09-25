"""Build CERES-Maize parameters and forcing from a DSSAT-CSM run (host-side NumPy, not traced).

* :func:`read_inp` parses the ``DSSAT48.INP`` the input module writes for a run; the crop reads
  its cultivar line, planting details and soil layers from this file, with the INP's own
  rounding (cultivar format ``(A6,1X,A16,1X,A6,1X,F6.1,F6.3,2F6.1,2F6.2)``, soil ``F5.3``), so
  these are the values the reference model used.
* :func:`ceres_params` adds the ecotype (``*.ECO``) and species (``*.SPE``) coefficients.
* :func:`ceres_forcing` builds the daily forcing: the weather the model used (``Weather.OUT``:
  ``TMXD``, ``TMND``, ``SRAD``, ``CO2D``, i.e. after any environment modification), daylength and
  twilight daylength from the weather-file latitude (DSSAT ``DAYLEN`` / ``TWILIGHT``), soil
  water ``SW1D..`` from ``SoilWat.OUT`` of the same treatment, and the potential transpiration
  ``EOP`` and potential root water uptake ``TRWUP`` of SPAM (``supply``: the reference run's
  daily values, which DSSAT prints nowhere at full precision; the A12 SPAM dumps). These drive the
  crop's ``water_in`` port (the isolation of the crop from the soil-water model); the crop
  computes ``SWFAC`` and ``TURFAC`` from them. Snow is taken as 0.
* :func:`spam_supply` reads ``(YRDOY, EOP, TRWUP)`` from a SPAM daily dump table.

Every argument is a plain path / number, so nothing here is traced by JAX.

Source: DSSAT-CSM v4.8.6.0 InputModule/optempy2k.for (INP layout), MZ_PHENOL.for / MZ_GROSUB.for
(the reads of the INP and the SPE / ECO files; BSD-3).
"""

from __future__ import annotations

from datetime import date, timedelta

import jax.numpy as jnp
import numpy as np
import pandas as pd

from agrijax.io.dssat import read_eco, read_out, read_soilwat, read_spe

from ._util import daylength, twilight_daylength
from .coefficients import BSGDD, CANHT_POT, DSSAT_COEFFICIENTS
from .state import CeresCultivar, CeresForcing, CeresMaizeParams, CeresSoil, CeresSpecies

__all__ = ["ceres_forcing", "ceres_params", "read_inp", "spam_supply", "yrdoy_range"]


def _section(text: str, name: str) -> list[str]:
    """Lines after the ``*NAME`` header of an INP text up to the next ``*`` header."""
    lines = text.splitlines()
    heads = [i for i, ln in enumerate(lines) if ln[:1] == "*"]
    start = next(i for i in heads if lines[i].split()[0].upper() == "*" + name)
    stop = next((i for i in heads if i > start), len(lines))
    return lines[start + 1 : stop]


def read_inp(path: str) -> dict[str, object]:
    """Cultivar, planting and soil values of a ``DSSAT48.INP`` (the last treatment written).

    Returns ``{"cultivar": {P1..PHINT, ECO}, "yrplt", "pltpop", "rowspc", "sdepth",
    "ds", "ll", "dul", "sat", "shf", "slpf"}``; ``pltpop`` is the plant population at emergence
    (``PPOE``, what ``MZ_PHENOL`` / ``MZ_GROSUB`` read), soil arrays are per layer with ``ds`` the
    layer bottom depths [cm].
    """
    with open(path, encoding="latin-1") as fh:
        text = fh.read()
    cul = next(ln for ln in _section(text, "CULTIVAR") if ln.strip())
    vals = [float(cul[31 + 6 * k : 37 + 6 * k]) for k in range(6)]
    plant = next(ln for ln in _section(text, "PLANTING") if ln.strip()).split()
    soil = _section(text, "SOIL")
    surface = soil[2].split()
    body = soil[3:]
    n_layer = next((k for k, ln in enumerate(body) if not ln.strip()), len(body))
    layers = [
        [float(ln[1:6]), float(ln[13:18]), float(ln[19:24]), float(ln[25:30]), float(ln[31:36])]
        for ln in body[:n_layer]
    ]
    arr = np.asarray(layers, dtype=float)
    return {
        "cultivar": dict(zip(("P1", "P2", "P5", "G2", "G3", "PHINT"), vals, strict=True))
        | {"ECO": cul[24:30]},
        "yrplt": int(plant[0]),
        "pltpop": float(plant[3]),
        "rowspc": float(plant[6]),
        "sdepth": float(plant[8]),
        "ds": arr[:, 0],
        "ll": arr[:, 1],
        "dul": arr[:, 2],
        "sat": arr[:, 3],
        "shf": arr[:, 4],
        "slpf": float(surface[6]),
    }


def ceres_params(
    inp_path: str, eco_path: str, spe_path: str, iswwat: bool = True, *, calibratable: bool = True
) -> CeresMaizeParams:
    """:class:`CeresMaizeParams` of a DSSAT run: INP cultivar / planting / soil + ECO + SPE.

    With ``calibratable=True`` (default) the coefficients DSSAT hard-codes are array leaves of the
    parameter tree (their DSSAT-CSM v4.8.6.0 values), so they can be calibrated and differentiated
    like the cultivar and species coefficients; ``False`` leaves them as trace-time constants.
    """
    inp = read_inp(inp_path)
    cul = inp["cultivar"]
    assert isinstance(cul, dict)
    eco = read_eco(eco_path).loc[str(cul["ECO"]).strip()]
    spe = read_spe(spe_path)

    def a(x: object) -> jnp.ndarray:
        return jnp.asarray(np.asarray(x, dtype=float))

    cultivar = CeresCultivar(
        p1=a(cul["P1"]),
        p2=a(cul["P2"]),
        p5=a(cul["P5"]),
        g2=a(cul["G2"]),
        g3=a(cul["G3"]),
        phint=a(cul["PHINT"]),
        tbase=a(eco["TBASE"]),
        topt=a(eco["TOPT"]),
        ropt=a(eco["ROPT"]),
        p2o=a(eco["P20"]),
        djti=a(eco["DJTI"]),
        gdde=a(eco["GDDE"]),
        dsgft=a(eco["DSGFT"]),
        rue=a(eco["RUE"]),
        tsen=a(eco["TSEN"]),
        cday=a(eco["CDAY"]),
    )
    species = CeresSpecies(
        prftc=a(spe["PRFTC"]),
        rgfil=a(spe["RGFIL"]),
        parsr=a(spe["PARSR"]),
        co2x=a(spe["CO2X"]),
        co2y=a(spe["CO2Y"]),
        fslfw=a(spe["FSLFW"]),
        rsgr=a(spe["RSGR"]),
        rsgrt=a(spe["RSGRT"]),
        carbot=a(spe["CARBOT"]),
        dsgt=a(spe["DSGT"]),
        dget=a(spe["DGET"]),
        swcg=a(spe["SWCG"]),
        stmwte=a(spe["STMWTE"]),
        rtwte=a(spe["RTWTE"]),
        lfwte=a(spe["LFWTE"]),
        seedrve=a(spe["SEEDRVE"]),
        leafnoe=a(spe["LEAFNOE"]),
        plae=a(spe["PLAE"]),
        pormin=a(spe["PORM"]),
        rwumx=a(spe["RWMX"]),
        rlwr=a(spe["RLWR"]),
        rwuep1=a(spe["RWUEP1"]),
        canht_pot=a(CANHT_POT),  # set in MZ_GROSUB SEASINIT, not read from the SPE file
        bsgdd=a(BSGDD),
    )
    ds = np.asarray(inp["ds"], dtype=float)
    soil = CeresSoil(
        dlayr=a(np.diff(np.concatenate([[0.0], ds]))),
        ll=a(inp["ll"]),
        dul=a(inp["dul"]),
        sat=a(inp["sat"]),
        shf=a(inp["shf"]),
        slpf=a(inp["slpf"]),
    )
    return CeresMaizeParams(
        cultivar=cultivar,
        species=species,
        soil=soil,
        pltpop=a(inp["pltpop"]),
        sdepth=a(inp["sdepth"]),
        rowspc=a(inp["rowspc"]),
        yrplt=jnp.asarray(int(inp["yrplt"]), dtype=jnp.int32),  # type: ignore[arg-type]
        iswwat=iswwat,
        coefficients=DSSAT_COEFFICIENTS.as_arrays() if calibratable else None,
    )


def yrdoy_range(first: int, last: int) -> list[int]:
    """``YYYYDDD`` integers from ``first`` to ``last`` inclusive."""
    d0 = date(first // 1000, 1, 1) + timedelta(days=first % 1000 - 1)
    d1 = date(last // 1000, 1, 1) + timedelta(days=last % 1000 - 1)
    n = (d1 - d0).days + 1
    days = [d0 + timedelta(days=k) for k in range(n)]
    return [d.year * 1000 + d.timetuple().tm_yday for d in days]


def spam_supply(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(yrdoy, eop, trwup)`` of a SPAM daily dump table (``<EXP>_t<NN>_spam.npz``).

    The table holds SPAM's exit values of the RATE call of every simulated day (REAL*4):
    ``v.YRDOY``, ``v.EOP`` [mm d-1] and ``v.TRWUP`` [cm d-1]; these are what PLANT receives on
    that day (``CSM_Main/LAND.for``).
    """
    with np.load(path, allow_pickle=False) as z:
        return (
            np.asarray(z["v.YRDOY"], dtype=np.int64),
            np.asarray(z["v.EOP"], dtype=np.float64),
            np.asarray(z["v.TRWUP"], dtype=np.float64),
        )


def ceres_forcing(
    out_dir: str,
    trno: int,
    first: int,
    last: int,
    n_layer: int,
    latitude: float,
    water: bool = True,
    daylength_from_output: bool = False,
    supply: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> CeresForcing:
    """Daily :class:`CeresForcing` for ``first..last`` (``YYYYDDD``) of treatment ``trno``.

    ``out_dir`` holds the reference run's ``Weather.OUT`` and ``SoilWat.OUT``. ``supply`` is
    ``(yrdoy, eop, trwup)`` of the same run (:func:`spam_supply`); it is required with
    ``water=True``, and days it does not list get ``EOP = TRWUP = 0`` (no demand: no stress). With
    ``water=False`` (``ISWWAT = N`` runs, no ``SoilWat.OUT``) the soil water, ``EOP`` and ``TRWUP``
    are 0 on every day (the crop sets its stress factors to 1). With
    ``daylength_from_output=True`` daylength and twilight daylength are the printed
    ``DAYLD`` / ``TWLD`` (0.1 h) instead of ``DAYLEN`` / ``TWILIGHT``: needed when an environment
    modification replaces the daylength (``WTHMOD`` sets both to the replacement value).
    """
    days = yrdoy_range(first, last)
    doy = np.asarray([d % 1000 for d in days], dtype=float)

    def by_day(df: pd.DataFrame, col: str, fill: float) -> np.ndarray:
        sub = df[df["TRNO"] == trno]
        key = np.asarray(sub["YEAR"], dtype=int) * 1000 + np.asarray(sub["DOY"], dtype=int)
        m = dict(zip(key.tolist(), np.asarray(sub[col], dtype=float).tolist(), strict=True))
        return np.asarray([m.get(d, fill) for d in days], dtype=float)

    wo = read_out(f"{out_dir}/Weather.OUT")
    co2 = by_day(wo, "CO2D", np.nan)
    dayl_out = by_day(wo, "DAYLD", np.nan)
    twld_out = by_day(wo, "TWLD", np.nan)
    tmax = by_day(wo, "TMXD", np.nan)
    tmin = by_day(wo, "TMND", np.nan)
    srad = by_day(wo, "SRAD", np.nan)
    if water:
        if supply is None:
            raise ValueError(
                "water=True needs supply = (yrdoy, eop, trwup) of the reference run (spam_supply)"
            )
        sw_df = read_soilwat(f"{out_dir}/SoilWat.OUT")
        sw = np.stack([by_day(sw_df, f"SW{k + 1}D", np.nan) for k in range(n_layer)], axis=-1)
        sd, se, st = (np.asarray(x) for x in supply)
        emap = dict(zip(sd.astype(int).tolist(), se.astype(float).tolist(), strict=True))
        tmap = dict(zip(sd.astype(int).tolist(), st.astype(float).tolist(), strict=True))
        eop = np.asarray([emap.get(d, 0.0) for d in days], dtype=float)
        trwup = np.asarray([tmap.get(d, 0.0) for d in days], dtype=float)
    else:
        sw = np.zeros((len(days), n_layer))
        eop = np.zeros(len(days))
        trwup = np.zeros(len(days))
    return CeresForcing(
        yrdoy=jnp.asarray(np.asarray(days, dtype=np.int32)),
        tmax=jnp.asarray(tmax),
        tmin=jnp.asarray(tmin),
        srad=jnp.asarray(srad),
        dayl=jnp.asarray(dayl_out) if daylength_from_output else daylength(jnp.asarray(doy), latitude),
        twilen=jnp.asarray(twld_out)
        if daylength_from_output
        else twilight_daylength(jnp.asarray(doy), latitude),
        co2=jnp.asarray(co2),
        snow=jnp.zeros(len(days)),
        sw=jnp.asarray(sw),
        eop=jnp.asarray(eop),
        trwup=jnp.asarray(trwup),
    )
