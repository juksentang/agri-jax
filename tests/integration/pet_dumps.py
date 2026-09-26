"""Shuttleworth-Wallace kernel against RZWQM2 4.6 ``POTEVPHR`` entry/exit dumps (helper module).

An instrumented RZWQM2 4.6 build (``-fp-model precise``; its ``.ana`` equals that of the plain
precise build) was run on the 9 daily S-W scenarios of ``RZWQM_sw_batch`` for the same 3 years as
:mod:`pet_scenarios`, with every ``POTEVPHR`` call dumped (one call per day on the daily path).
Each stream is reduced to two daily tables (:func:`agrijax.port.dumps.daily_table`), stored as
``<data>/dumps/tables/rzwqm46_pet_scen/<site>_{entry,exit}.npz``.

The kernel :func:`agrijax.processes.pet.shuttleworth_wallace` is driven from the entry values, so
every input is the one the reference routine received (no reconstruction from end-of-day output):

=====================  ===========================================================================
kernel input           dumped ``POTEVPHR`` entry value
=====================  ===========================================================================
weather                ``TMIN``, ``TMAX``, ``RTS`` (daily solar, hourly re-sum after the SHAW
                       split), ``RTH`` (measured horizontal solar), ``RH``, ``U`` (wind run, km/d)
canopy                 ``LAI``, ``TLAI``, ``HEIGHT``; ``RST`` of plant ``IPL`` (``/IPOTEV/``)
soil surface           ``THETA`` (node 1), ``WC13``, ``WC15`` (``SOILHP`` rows 7 and 9 of horizon 1),
                       ``ICRUST``, ``RR``
residue                ``RM`` [kg/ha], ``RESAGE``, ``WRES``, ``CRES``; diameter and density of the
                       residue-type index ``IPR`` (1 corn, 2 soybean, 3 wheat)
site / parameters      ``ELEV``, ``XLAT``, ``JDAY``, ``XW`` (anemometer height), ``IWZONE``,
                       ``A0``, ``AW``, ``AC``, ``ARI``, ``RSS``, ``TRAT``
=====================  ===========================================================================

Compared with the exit values ``PET`` (potential transpiration), ``PES`` (soil) and ``PER``
(residue evaporation), all cm d-1, and the intermediate outputs ``AS`` (soil albedo), ``AR``
(residue albedo), ``CS`` (exposed soil fraction) and ``UNEW`` (adjusted wind run).

The daily inputs the kernel does not model (standing dead residue, stubble, slope, the SHAW and
PENFLUX switches, hourly mode) are checked to be inactive on every dumped day
(:func:`inactive_inputs`), so a difference cannot hide behind them.

The same runs also give the ``.ana`` of the instrumented build, so the reconstruction of
:mod:`pet_scenarios` (start-of-day state rebuilt from end-of-day output) can be applied to exactly
the run the dumps come from, and each reconstructed input replaced by its dumped value
(:func:`attribution`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TABLES = Path("dumps/tables/rzwqm46_pet_scen")
RUNS = Path("pet_dump_runs")
SITES = (
    "CA-ER1",
    "CA-MA1",
    "CA-TPA",
    "US-Mj1",
    "US-Tw2",
    "US-UA1_HartFarm",
    "US-manilacotton",
    "US_OPE",
    "US_Rockfish",
)
RESIDUE_TYPES = ("corn", "soybean", "wheat")  # IPR = 1, 2, 3 (RESISThr RDIA / RHORS order)
OUTPUTS = {"pt": "PET", "pes": "PES", "per": "PER"}
#: PI of MAXSW (Rzpet.for line 627, PARAMETER statement)
REFERENCE_PI = 3.141592654


@dataclass
class DumpDays:
    site: str
    date: np.ndarray  # YYYYDDD
    n_calls: np.ndarray
    entry: dict[str, np.ndarray]
    exit: dict[str, np.ndarray]
    meta: dict[str, Any]


def table_paths(data_dir: Path, site: str) -> tuple[Path, Path]:
    d = Path(data_dir) / TABLES
    return d / f"{site}_entry.npz", d / f"{site}_exit.npz"


def load_site(data_dir: Path, site: str) -> DumpDays:
    from agrijax.port.dumps import load_table

    pe, px = table_paths(data_dir, site)
    te, meta = load_table(pe)
    tx, _ = load_table(px)
    assert np.array_equal(te.date, tx.date) and np.array_equal(te.call, tx.call), site
    return DumpDays(site, te.date, te.n_calls, te.values, tx.values, meta)


def _scalar(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a.reshape(a.shape[0], -1)[:, 0] if a.ndim > 1 else a


def kernel_inputs(d: DumpDays) -> dict[str, np.ndarray]:
    """Per-day kernel inputs from the entry table (see the module docstring)."""
    e = d.entry
    f = {k: _scalar(v).astype(float) for k, v in e.items() if k not in ("RST", "IRTYPE")}
    ipl = _scalar(e["IPL"]).astype(int)
    rst_all = np.asarray(e["RST"], dtype=float).reshape(len(ipl), -1)
    # IPL = 0 (no plant yet): RSC is not used (LAI = 0 gives the no-canopy branch); plant 1's value
    rst = rst_all[np.arange(len(ipl)), np.clip(ipl - 1, 0, rst_all.shape[1] - 1)]
    ipr = _scalar(e["IPR"]).astype(int)
    return dict(
        tmin=f["TMIN"],
        tmax=f["TMAX"],
        srad=f["RTS"],
        srad_h=f["RTH"],
        rh=f["RH"],
        wind_run=f["U"],
        lai=f["LAI"],
        tlai=f["TLAI"],
        height=f["HEIGHT"],
        theta=f["THETA"],
        wc13=f["WC13"],
        wc15=f["WC15"],
        elevation=f["ELEV"],
        latitude=f["XLAT"],
        doy=f["JDAY"],
        # RESISThr takes the residue branch only for IPR != 0 (Rzpet.for line 2675)
        residue_mass=np.where(ipr > 0, f["RM"], 0.0),
        residue_age=f["RESAGE"],
        residue_wet=f["WRES"],
        crust=f["ICRUST"],
        roughness=f["RR"],
        cres=f["CRES"],
        ipr=np.clip(ipr, 1, 3).astype(float),
        wind_height=f["XW"],
        trat=f["TRAT"],
        albedo_dry=f["A0"],
        albedo_wet=f["AW"],
        albedo_crop=f["AC"],
        albedo_residue=f["ARI"],
        rss=f["RSS"],
        rst=rst,
        rain_zone=f["IWZONE"],
    )


def inactive_inputs(d: DumpDays) -> dict[str, float]:
    """Largest value on any day of each dumped input the kernel does not model (all must be 0).

    Standing dead residue and stubble (``SDEAD``, ``SAI``: wind-height block and ``RESISThr``
    stubble branch), slope ``S``, the hourly / SHAW / PENFLUX switches, and ``RTSTOT = 0`` (which
    would make ``POTEVPHR`` replace the measured radiation by the sunshine-fraction estimate).
    """
    e = {k: _scalar(v).astype(float) for k, v in d.entry.items() if k not in ("RST", "IRTYPE")}
    return {
        "SDEAD": float(np.abs(e["SDEAD"]).max()),
        "SAI": float(np.abs(e["SAI"]).max()),
        "S": float(np.abs(e["S"]).max()),
        "IHOURLY": float(np.abs(e["IHOURLY"]).max()),
        "ISHAW": float(np.abs(e["ISHAW"]).max()),
        "IPENFLUX": float(np.abs(e["IPENFLUX"]).max()),
        "JPENFLUX": float(np.abs(e["JPENFLUX"]).max()),
        "RTSTOT_zero_days": float((e["RTSTOT"] == 0.0).sum()),
    }


def run_kernel(
    x: dict[str, np.ndarray],
    *,
    override: dict[str, np.ndarray] | None = None,
    coefficients: Any = None,
) -> dict[str, np.ndarray]:
    """Kernel outputs [cm d-1 and intermediates] for the per-day inputs ``x``.

    ``override`` replaces some inputs; ``coefficients`` replaces the kernel's ``SWCoefficients``
    (default :data:`~agrijax.processes.pet.RZWQM_SW`).
    """
    import jax
    import jax.numpy as jnp

    from agrijax.processes.pet import RZWQM_SW

    cf = RZWQM_SW if coefficients is None else coefficients

    from agrijax.processes.pet import PETParams, shuttleworth_wallace
    from agrijax.processes.pet.shuttleworth_wallace import RESIDUE_DENSITY_G_CM3, RESIDUE_DIAMETER_CM

    x = dict(x, **(override or {}))
    zones = np.unique(x["rain_zone"]).astype(int)
    assert len(zones) == 1, zones  # a static argument of the kernel
    zone = int(zones[0])
    ipr = x["ipr"].astype(int)
    rdia = np.array([RESIDUE_DIAMETER_CM[RESIDUE_TYPES[i - 1]] for i in ipr], dtype=float)
    rho = np.array([RESIDUE_DENSITY_G_CM3[RESIDUE_TYPES[i - 1]] for i in ipr], dtype=float)
    keys = (
        "tmin tmax srad srad_h rh wind_run lai tlai height theta wc13 wc15 elevation latitude doy residue_mass "
        "residue_age residue_wet crust roughness cres wind_height trat albedo_dry albedo_wet albedo_crop "
        "albedo_residue rss rst"
    ).split()

    def one(v: dict[str, Any], rd, rr):
        p = PETParams(
            albedo_dry=v["albedo_dry"],
            albedo_wet=v["albedo_wet"],
            albedo_maturity=v["albedo_crop"],
            albedo_residue=v["albedo_residue"],
            soil_resistance=v["rss"],
            stomatal_resistance=v["rst"],
        )
        return shuttleworth_wallace(
            v["tmin"],
            v["tmax"],
            v["srad"],
            v["rh"],
            v["wind_run"],
            v["lai"],
            v["height"],
            p,
            theta_surface=v["theta"],
            wc13=v["wc13"],
            wc15=v["wc15"],
            elevation=v["elevation"],
            latitude=v["latitude"],
            doy=v["doy"],
            tlai=v["tlai"],
            srad_horizontal=v["srad_h"],
            residue_mass=v["residue_mass"],
            residue_age=v["residue_age"],
            residue_wet=v["residue_wet"],
            crust=v["crust"],
            roughness_cm=v["roughness"],
            wind_height=v["wind_height"],
            trat=v["trat"],
            rainfall_zone=zone,
            residue_cover_factor=v["cres"],
            residue_diameter_cm=rd,
            residue_density=rr,
            coefficients=cf,
        )

    r = jax.jit(jax.vmap(one))({k: jnp.asarray(x[k]) for k in keys}, jnp.asarray(rdia), jnp.asarray(rho))
    return {
        "pt": np.asarray(r.transpiration),
        "pes": np.asarray(r.soil_evaporation),
        "per": np.asarray(r.residue_evaporation),
        "as": np.asarray(r.albedo_soil),
        "ar": np.asarray(r.albedo_residue),
        "cs": np.asarray(r.soil_fraction),
        "unew": np.asarray(r.wind_run_adjusted),
    }


def reference_pi_coefficients() -> Any:
    """``RZWQM_SW`` with the hours-per-radian constant of MAXSW built on the reference's PI.

    MAXSW takes pi as 3.141592654 (Rzpet.for line 627) in its hours-per-radian factor 12/pi;
    the port uses ``math.pi`` (a documented choice, ``maxsw.hours_per_radian``), a relative
    difference of 1.3e-10 in the extraterrestrial and clear-sky radiation. With this instance the kernel runs
    the reference's arithmetic.
    """
    from agrijax.core.state import set_path
    from agrijax.processes.pet import RZWQM_SW

    return set_path(RZWQM_SW, "maxsw.hours_per_radian", 12.0 / REFERENCE_PI)


def clear_sky(x: dict[str, np.ndarray], coefficients: Any = None) -> np.ndarray:
    """Kernel clear-sky radiation ``RCH`` [MJ m-2 d-1], floored at ``RTS`` like the dumped ``RCS``."""
    from agrijax.processes.pet import RZWQM_SW, clear_sky_radiation

    cf = RZWQM_SW if coefficients is None else coefficients
    return np.maximum(np.asarray(clear_sky_radiation(x["doy"], x["latitude"], cf.maxsw).total), x["srad"])


def reference_outputs(d: DumpDays) -> dict[str, np.ndarray]:
    x = d.exit
    return {
        "pt": _scalar(x["PET"]).astype(float),
        "pes": _scalar(x["PES"]).astype(float),
        "per": _scalar(x["PER"]).astype(float),
        "as": _scalar(x["AS"]).astype(float),
        "ar": _scalar(x["AR"]).astype(float),
        "cs": _scalar(x["CS"]).astype(float),
        "unew": _scalar(x["UNEW"]).astype(float),
        "rcs": _scalar(x["RCS"]).astype(float),
    }


def dates_of(d: DumpDays):
    import pandas as pd

    y, j = d.date // 1000, d.date % 1000
    return pd.to_datetime([f"{a:04d}{b:03d}" for a, b in zip(y, j, strict=True)], format="%Y%j")


#: input groups of :func:`attribution` (kernel input names of :func:`kernel_inputs`)
GROUPS = {
    "weather (.ana echo, RTH from .MET)": ("tmin", "tmax", "srad", "srad_h", "rh", "wind_run"),
    "canopy (LAI, TLAI, height)": ("lai", "tlai", "height"),
    "surface water (THETA)": ("theta",),
    "residue mass (RM)": ("residue_mass",),
    "residue age (RESAGE)": ("residue_age",),
    "residue type (IPR)": ("ipr",),
    "WC13 / WC15": ("wc13", "wc15"),
    "stomatal resistance (RST)": ("rst",),
    "site constants and parameters": (
        "elevation",
        "latitude",
        "wind_height",
        "cres",
        "albedo_dry",
        "albedo_wet",
        "albedo_crop",
        "albedo_residue",
        "rss",
    ),
}


def reconstructed_inputs(
    x: dict[str, np.ndarray], d: DumpDays, x_rec: Any
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """The dumped inputs with every value replaced by its :mod:`pet_scenarios` reconstruction.

    Returns ``(inputs, ok)``; ``ok`` marks the days both cover, day 1 of the run excluded (its
    start-of-day state is not printed). ``x_rec`` is :func:`pet_scenarios.site_inputs` of the
    instrumented run's own outputs; weather from its ``.ana`` echo.
    """
    days = dates_of(d)
    k = np.asarray(x_rec.days.get_indexer(days))
    ok = k >= 0
    ok[0] = False
    kk = k[ok]
    n = len(x_rec.days)
    w = x_rec.weather["ana"]
    c = x_rec.site_consts
    p = c["params"]
    src = {
        "tmin": w["tmin"],
        "tmax": w["tmax"],
        "srad": w["srad"],
        "srad_h": w["srad_h"],
        "rh": w["rh"],
        "wind_run": w["wind_run"],
        "lai": x_rec.lai,
        "tlai": x_rec.lai,
        "height": x_rec.height,
        "theta": x_rec.theta,
        "residue_mass": x_rec.residue_mass,
        "residue_age": x_rec.residue_age,
        "rst": x_rec.extra["rs_min"],
        "wc13": np.full(n, c["wc13"]),
        "wc15": np.full(n, c["wc15"]),
        "ipr": np.array([RESIDUE_TYPES.index(t) + 1 for t in x_rec.residue_type], dtype=float),
        "elevation": np.full(n, c["elevation"]),
        "latitude": np.full(n, c["latitude"]),
        "wind_height": np.full(n, c["wind_height"]),
        "cres": np.full(n, c["cover_factor"]),
        "albedo_dry": np.full(n, float(p.albedo_dry)),
        "albedo_wet": np.full(n, float(p.albedo_wet)),
        "albedo_crop": np.full(n, float(p.albedo_maturity)),
        "albedo_residue": np.full(n, float(p.albedo_residue)),
        "rss": np.full(n, float(p.soil_resistance)),
    }
    out = {}
    for name, v in x.items():
        a = np.array(v, dtype=float)
        if name in src:
            a[ok] = np.asarray(src[name], dtype=float)[kk]
        out[name] = a
    return out, ok


def attribution(d: DumpDays, x_rec: Any) -> dict[str, Any]:
    """Error of the :mod:`pet_scenarios` reconstruction on the dumped run, input group by group.

    The kernel is run (a) on the dumped inputs, (b) on the dumped inputs with one group of
    :data:`GROUPS` replaced by its reconstruction, (c) on all reconstructed inputs (what the
    harness computes). Per case the max |error| of PE (= PES + PER) and PT against the dumped
    outputs [mm d-1] over the days both cover, and for (c) the per-day errors (``pe_err``,
    ``pt_err``) and the days (``ok``).
    """
    ref = reference_outputs(d)
    x = kernel_inputs(d)
    xr, ok = reconstructed_inputs(x, d, x_rec)

    def err(out: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        pe = (out["pes"] + out["per"] - ref["pes"] - ref["per"]) * 10.0
        pt = (out["pt"] - ref["pt"]) * 10.0
        return pe, pt

    res: dict[str, Any] = {}
    for label, names in {"dumped inputs": (), **GROUPS}.items():
        pe, pt = err(run_kernel(x, override={n: xr[n] for n in names}))
        key = label if label == "dumped inputs" else f"reconstructed {label}"
        res[key] = {"pe_max_mm": float(np.abs(pe[ok]).max()), "pt_max_mm": float(np.abs(pt[ok]).max())}
    pe, pt = err(run_kernel(xr))
    res["all reconstructed"] = {
        "pe_max_mm": float(np.abs(pe[ok]).max()),
        "pt_max_mm": float(np.abs(pt[ok]).max()),
    }
    res["pe_err"], res["pt_err"], res["ok"] = pe, pt, ok
    return res
