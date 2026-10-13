"""One-call calibration of CERES-Maize cultivar coefficients on a DSSAT maize example: :func:`calibrate`.

::

    import jax
    jax.config.update("jax_enable_x64", True)
    from agrijax.calib import calibrate

    res = calibrate("UFGA8201", treatments=[4], holdout=[6], write_cul="out/MZCER048.CUL",
                    dssat_check=True)
    print(res)                  # coefficients, objective, fit per target, DSSAT check
    res.params                  # {"P1": ..., "P2": ..., ...} as written to the .CUL row
    res.fit                     # observed vs simulated, one row per observation (pandas)

**What runs.** The DSSAT-CSM v4.8.6.0 day (:mod:`agrijax.models.day_dssat486`) run free (soil water,
evaporation, transpiration, uptake and crop simulated; :mod:`agrijax.sites.dssat_free_run`),
**nitrogen off**: the crop grows without nitrogen limitation, so only treatments where nitrogen does
not matter can be fitted. The six ``MZCER048.CUL`` coefficients P1, P2, P5, G2, G3, PHINT are
calibrated inside the file's ``MINIMA`` / ``MAXIMA`` rows; the ecotype coefficients
(``MZCER048.ECO``) stay as published.

**Supported scope** (anything else is refused with the reason, :class:`ScopeError`, before anything
runs). The free-run day's soil and soil-water inputs come from one of two sources
(:func:`agrijax.sites.dssat_free_run.free_run_inputs`, ``inputs=``): ``"native"`` builds them from the
experiment's public files (a treatment that needs a DSSAT process Agri-JAX does not implement, e.g. automatic
irrigation or the CENTURY organic matter with surface residue, is refused naming it);
``"tables"`` reads the tables written by an instrumented DSSAT-CSM build for the DSSAT v4.8.6 maize
example treatments (not distributed; the validation harness); ``"auto"`` (the default) uses the
tables when every requested treatment has them under the data directory, the native inputs
otherwise. On the example treatments both give the same inputs except the hourly mean ``TAVG``
(within a few REAL*4 units on about 1 % of the days). Of the treatments, those where nitrogen
matters are refused: :data:`NITROGEN_STRESS` holds the
measured change of DSSAT's yield between nitrogen on and off, in scope up to
:data:`NITROGEN_STRESS_OK` (2 %), with a warning up to :data:`NITROGEN_STRESS_MAX` (5 %); a
treatment whose silking or maturity date also changes with nitrogen on
(:data:`NITROGEN_DATES_DIFFER`) is in scope with a warning even when its yield change is below 2 %
(its date targets are fitted with nitrogen off); the experiments in :data:`OUT_OF_SCOPE` (the growth
chamber, the nitrogen x phosphorus trial) are refused as a whole. One cultivar per call: the
treatments must share it (``cultivar=`` picks it). ``treatments=None`` takes every supported
treatment of the experiment(s) with that cultivar, except
the held-out ones.

**Steps.**

1. ``dscsm048`` runs every treatment once with the published cultivar (nitrogen off): the reference
   run gives the crop parameters, weather and the season (planting to maturity / harvest), whose
   days carry real forcing; the forcing is padded after them (:data:`~agrijax.calib.dssat_day.PAD_DAYS`)
   so a later-maturing candidate still matures.
2. **Targets**: ``targets="auto"`` uses :data:`BASE_TARGETS` (ADAT, MDAT, HWAM, CWAM, H#AM, LAID)
   where observed; ``"all"`` every observed quantity of the A / T files that maps to a model output
   (:data:`agrijax.calib.observations.DSSAT_TARGETS`, with the organ-weight series); or a list of
   codes. Unmapped codes are listed in :attr:`CalibrationResult.unsupported`. An A-file target (a
   date or a value at maturity) that cannot be placed on the simulated days raises
   :class:`ObservationError` - nothing observed is dropped silently; a T-file observation dated after
   the season (e.g. an LAI sampled after harvest) is dropped **with a warning** and listed in
   :attr:`CalibrationResult.dropped`; dates observed twice are averaged, with a warning.
3. :func:`agrijax.calib.fit.fit_cultivar`: the sensitivity check (``params="auto"`` fixes the
   coefficients the observations carry no information on, e.g. P2 without a photoperiod signal),
   then ``method``: ``"cma"`` joint CMA-ES (the default), ``"staged"`` (phenology, then growth, then a
   joint refinement; derivative-free), ``"adam"`` (only the coefficients whose gradient the trust
   report accepts); a warning when G2 / G3 are free without an observed grain number.
4. **Selection on the written coefficients**: the candidates, per start, are the best point of each
   run that ended in a restart, the start's overall best point and its 8 best points, plus the
   published cultivar; they are evaluated at the values a ``.CUL`` row holds (printed precision,
   :func:`agrijax.io.dssat.cultivar_write.cul_written`); the best at those values is returned, the
   published cultivar itself when nothing beats it (:attr:`CalibrationResult.improved`).
5. Optional: ``write_cul`` writes a copy of the engine's ``MZCER048.CUL`` with the new row
   (:func:`agrijax.io.dssat.cultivar_write.write_cultivar`; an existing file is refused before
   anything runs); ``dssat_check`` runs ``dscsm048`` with that copy on every treatment and compares
   ``HWAM``, ``ADAT``, ``MDAT`` with the prediction. Both are also methods of the result
   (:meth:`CalibrationResult.write_cul`, :meth:`CalibrationResult.check_dssat`).

**Performance.** The candidates of all starts are simulated together, sharded over ``jax.devices()``;
on a CPU node set ``XLA_FLAGS=--xla_force_host_platform_device_count=<cores>`` before importing JAX.
Runs in 64-bit (``jax_enable_x64``): the forward model is validated against DSSAT in float64.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from agrijax.calib.dssat_day import CUL_ORDER, E_DATE, E_DAY, E_FINAL, OUT_NAMES, Entry
from agrijax.calib.fit import (
    PROBES,
    CalibrationWarning,
    CultivarProblem,
    Observed,
    Treatment,
    _yrdoy_days_between,
    fit_cultivar,
)
from agrijax.calib.observations import ObservationError

__all__ = [
    "BASE_TARGETS",
    "DSSAT_YIELD_RTOL",
    "NITROGEN_DATES_DIFFER",
    "NITROGEN_STRESS",
    "NITROGEN_STRESS_MAX",
    "NITROGEN_STRESS_OK",
    "OUT_OF_SCOPE",
    "CalibrationResult",
    "ObservationError",
    "ScopeError",
    "calibrate",
]

#: experiments outside the supported scope as a whole, with the reason
OUT_OF_SCOPE: dict[str, str] = {
    "GHWA0401": "nitrogen x phosphorus trial (N0-N120 x P0-P90): phosphorus limitation is not simulated",
    "GAGR0201": "growth-chamber experiment (environment modifications): its seasons end before silking",
}
#: how far the nitrogen-off model is from DSSAT's nitrogen-on run of each treatment:
#: ``(HWAM with NITRO = Y - HWAM with NITRO = N) / HWAM with NITRO = N`` of ``dscsm048`` (DSSAT-CSM
#: v4.8.6.0 build486, published cultivar, one-treatment batch, the FileX otherwise as distributed),
#: measured 2026-09-30 on every example treatment with the free-run input tables by
#: ``scripts/calib/nitrogen_stress_table.py`` (rerun it after a DSSAT engine update). Criterion:
#: in scope when ``|dHWAM| <= NITROGEN_STRESS_OK``, in scope with a warning up to
#: ``NITROGEN_STRESS_MAX``, refused above (the nitrogen-off calibration would fit a different yield
#: than DSSAT simulates for the treatment). A line marked ``dates differ`` is also in
#: ``NITROGEN_DATES_DIFFER``
NITROGEN_STRESS: dict[str, float] = {
    "BRPI0202_t01": -0.0686,  # HWAM N on 4329, N off 4648
    "BRPI0202_t02": -0.0165,  # HWAM N on 4358, N off 4431
    "BRPI0202_t03": -0.0055,  # HWAM N on 4559, N off 4584
    "BRPI0202_t04": -0.0491,  # HWAM N on 4878, N off 5130
    "BRPI0202_t05": -0.0002,  # HWAM N on 5215, N off 5216
    "BRPI0202_t06": -0.0031,  # HWAM N on 5859, N off 5877
    "BRPI0202_t07": -0.0002,  # HWAM N on 5596, N off 5597
    "BRPI0202_t08": 0.0002,  # HWAM N on 6149, N off 6148
    "FLSC8101_t01": 0.0001,  # HWAM N on 10147, N off 10146
    "FLSC8101_t02": -0.1054,  # HWAM N on 5236, N off 5853
    "GHWA0401_t01": -0.8995,  # HWAM N on 449, N off 4468
    "GHWA0401_t02": -0.9253,  # HWAM N on 334, N off 4470
    "GHWA0401_t03": -0.9253,  # HWAM N on 334, N off 4470
    "GHWA0401_t04": -0.7906,  # HWAM N on 936, N off 4469
    "GHWA0401_t05": -0.5104,  # HWAM N on 2190, N off 4473
    "GHWA0401_t06": -0.5108,  # HWAM N on 2188, N off 4473
    "GHWA0401_t07": -0.7921,  # HWAM N on 929, N off 4469
    "GHWA0401_t08": -0.0418,  # HWAM N on 4285, N off 4472
    "GHWA0401_t09": -0.0550,  # HWAM N on 4226, N off 4472
    "IBWA8301_t01": -0.7311,  # HWAM N on 1930, N off 7177
    "IBWA8301_t02": -0.4324,  # HWAM N on 4074, N off 7177
    "IBWA8301_t03": -0.0105,  # HWAM N on 7102, N off 7177
    "IBWA8301_t04": -0.6825,  # HWAM N on 2339, N off 7366
    "IBWA8301_t05": -0.4071,  # HWAM N on 4367, N off 7366
    "IBWA8301_t06": -0.0175,  # HWAM N on 7237, N off 7366
    "IUAF9901_t01": -0.4446,  # HWAM N on 5140, N off 9255
    "IUAF9901_t02": -0.0003,  # HWAM N on 9309, N off 9312
    "IUAF9901_t03": -0.6219,  # HWAM N on 3992, N off 10558
    "IUAF9901_t04": -0.0311,  # HWAM N on 10189, N off 10516
    "SIAZ9501_t01": 0.0000,  # HWAM N on 10990, N off 10990
    "SIAZ9501_t02": 0.0016,  # HWAM N on 7656, N off 7644
    "SIAZ9501_t03": 0.1089,  # HWAM N on 1018, N off 918; dates differ
    "SIAZ9501_t04": 0.0434,  # HWAM N on 8600, N off 8242
    "SIAZ9501_t05": 0.0244,  # HWAM N on 10823, N off 10565
    "SIAZ9501_t06": 0.0445,  # HWAM N on 6671, N off 6387
    "SIAZ9501_t07": 0.0186,  # HWAM N on 6668, N off 6546
    "SIAZ9501_t08": 0.0406,  # HWAM N on 8922, N off 8574
    "SIAZ9601_t01": 0.0000,  # HWAM N on 10924, N off 10924
    "SIAZ9601_t02": 0.0136,  # HWAM N on 6834, N off 6742
    "SIAZ9601_t03": 0.0032,  # HWAM N on 5273, N off 5256
    "SIAZ9601_t04": -0.0001,  # HWAM N on 10608, N off 10609
    "SIAZ9601_t05": -0.0205,  # HWAM N on 9830, N off 10036
    "SIAZ9601_t06": -0.0002,  # HWAM N on 10873, N off 10875
    "SIAZ9601_t07": 0.0037,  # HWAM N on 7059, N off 7033; dates differ
    "SIAZ9601_t08": 0.0054,  # HWAM N on 7424, N off 7384
    "SIAZ9601_t09": 0.0059,  # HWAM N on 8636, N off 8585
    "UFGA8201_t01": 0.1408,  # HWAM N on 2293, N off 2010
    "UFGA8201_t02": 0.1408,  # HWAM N on 2293, N off 2010
    "UFGA8201_t03": -0.3080,  # HWAM N on 8207, N off 11859
    "UFGA8201_t04": -0.0004,  # HWAM N on 11854, N off 11859
    "UFGA8201_t05": -0.2559,  # HWAM N on 7718, N off 10372
    "UFGA8201_t06": -0.0076,  # HWAM N on 10293, N off 10372
}
#: the treatments of ``NITROGEN_STRESS`` where DSSAT's silking (``ADAT``) or maturity (``MDAT``) date
#: changes with nitrogen on (``dates_equal`` false in ``scripts/calib/nitrogen_stress_table.py``, the
#: ``dates differ`` mark of its output; same run and date as the table): in scope with a warning
#: even when the yield change is below ``NITROGEN_STRESS_OK``, because the nitrogen-off calibration
#: fits date targets that the nitrogen-on model would not reproduce. A treatment above
#: ``NITROGEN_STRESS_MAX`` is refused as before. (The growth-chamber experiment GAGR0201, refused as a
#: whole, also has differing dates on all its treatments and is not in the table.)
NITROGEN_DATES_DIFFER: frozenset[str] = frozenset(
    {
        "SIAZ9501_t03",  # refused on the yield (+10.9 %)
        "SIAZ9601_t07",  # the only one in scope: yield +0.37 %, dates differ
    }
)
NITROGEN_STRESS_OK = 0.02
NITROGEN_STRESS_MAX = 0.05
#: the targets of ``targets="auto"``: stage dates, yield, tops weight and grain number at maturity,
#: LAI series. The organ-weight series (``LWAD``, ``SWAD``, ``GWAD``, ``CWAD`` ...) are opt-in
#: (``targets="all"`` or a list): on UFGA8201 the observed LWAD is 20-60 % below the CERES-Maize leaf
#: weight and SWAD 30-60 % above the stem weight at the published cultivar while their sum agrees
#: within about 10 %. A hypothesis, not checked against the experiments' documentation: the leaf /
#: stem split is defined differently (CERES-Maize's leaf weight includes the sheaths, which the
#: observations may count as stem)
BASE_TARGETS: tuple[str, ...] = ("ADAT", "MDAT", "HWAM", "CWAM", "H#AM", "LAID")
#: DSSAT check: the relative yield difference accepted between dscsm048 and Agri-JAX at the written
#: coefficients (HWAM is printed to 1 kg ha-1; the free-run day's yield matches DSSAT's to 0.02 % on
#: the validated treatments, except BRPI0202 t04 at 1.5 %)
DSSAT_YIELD_RTOL = 1e-3
#: prefix of the default id of a written cultivar row (``AJ0001``, the first one not in the file)
CUL_ID_PREFIX = "AJ"
#: the sources of the free-run inputs (``inputs=``)
INPUTS = ("auto", "native", "tables")


class ScopeError(ValueError):
    """The experiment or a treatment is outside the supported scope (the message says why)."""


# ------------------------------------------------------------------ result
@dataclass
class CalibrationResult:
    """What :func:`calibrate` returns.

    ``params`` are the calibrated coefficients **as written** (``.CUL`` precision), ``published`` the
    starting cultivar's. ``loss``: the objective on the calibration treatments at the written values
    (``written``), at the unrounded optimum (``unrounded``) and at the published cultivar
    (``published``). ``fit`` has one row per observation (observed, calibrated, published; dates as
    ``YYYYDDD``). ``dssat_check`` (with ``dssat_check=True``): per treatment DSSAT's and Agri-JAX's
    ``HWAM`` / ``ADAT`` / ``MDAT`` at the written coefficients and whether they agree."""

    cultivar: str
    treatments: list[str]
    params: dict[str, float]
    params_unrounded: dict[str, float]
    published: dict[str, float]
    free: tuple[str, ...]
    fixed: dict[str, str]
    loss: dict[str, float]
    fit: Any
    targets: dict[str, int]
    scales: dict[str, float]
    dropped: list[dict[str, Any]]
    unsupported: dict[str, str]
    sensitivity: dict[str, dict[str, Any]]
    trust: dict[str, Any]
    predictions: dict[str, dict[str, dict[str, float]]]
    #: each treatment's season at the published cultivar, nitrogen off (per treatment): the reference
    #: run's ``Summary.OUT`` row (table inputs) or Agri-JAX's own season (native inputs, no DSSAT run)
    reference: dict[str, dict[str, float]]
    per_start: list[dict[str, Any]]
    method: str
    calls: dict[str, Any]
    wall_s: float
    holdout: dict[str, Any] | None = None
    cul_id: str | None = None
    cul_line: str | None = None
    cul_path: Path | None = None
    dssat_check: dict[str, Any] | None = None
    #: False when no written candidate beat the published cultivar (the published one is returned)
    improved: bool = True
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: the source of the free-run inputs (``"native"`` or ``"tables"``)
    inputs: str = "tables"
    #: what :meth:`write_cul` / :meth:`check_dssat` need (experiment files, engine, cultivar)
    context: dict[str, Any] | None = field(default=None, repr=False)

    def write_cul(self, path: str | os.PathLike[str], cul_id: str | None = None) -> str:
        """Write a copy of the engine's ``MZCER048.CUL`` with the calibrated row to ``path`` (a file
        or a directory; the file must not exist); returns the row. ``cul_id`` defaults to this
        result's id, else the first free ``AJNNNN``."""
        ctx = self._ctx()
        dest = _cul_dest(path)
        if dest.exists():
            raise FileExistsError(f"write_cul: {dest} exists (remove it or choose another path)")
        cid, line, warn = _write_row(ctx, self.params, dest, cul_id or self.cul_id)
        self.cul_id, self.cul_line, self.cul_path = cid, line, dest
        for m in warn:
            self.warnings.append(m)
            warnings.warn(m, CalibrationWarning, stacklevel=2)
        return line

    def check_dssat(self) -> Any:
        """Run ``dscsm048`` with the calibrated row on every treatment (calibrated and held out) and
        compare ``HWAM``, ``ADAT``, ``MDAT`` with Agri-JAX's prediction (stored in
        :attr:`dssat_check`); returns one row per treatment (pandas)."""
        import pandas as pd

        ctx = self._ctx()
        work = _run_root()
        try:
            dest = work / "cul" / "MZCER048.CUL"
            cid, _, _ = _write_row(ctx, self.params, dest, self.cul_id)
            if self.cul_id is None:
                self.cul_id = cid
            self.dssat_check = _dssat_check(ctx, self.predictions["calibrated"], cid, dest, work)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if not self.dssat_check["agree"]:
            m = (
                "DSSAT and Agri-JAX disagree at the written coefficients: "
                f"{[t['treatment'] for t in self.dssat_check['treatments'] if not t['agree']]}"
            )
            self.warnings.append(m)
            warnings.warn(m, CalibrationWarning, stacklevel=2)
        return pd.DataFrame(
            [
                {
                    "treatment": t["treatment"],
                    "HWAM DSSAT": t["dssat"]["HWAM"],
                    "HWAM Agri-JAX": t["agrijax"]["HWAM"],
                    "yield rel. diff": t["yield_rel_diff"],
                    "ADAT DSSAT": int(t["dssat"]["ADAT"]),
                    "ADAT Agri-JAX": int(t["agrijax"]["ADAT"]),
                    "MDAT DSSAT": int(t["dssat"]["MDAT"]),
                    "MDAT Agri-JAX": int(t["agrijax"]["MDAT"]),
                    "agree": t["agree"],
                }
                for t in self.dssat_check["treatments"]
            ]
        )

    def __repr__(self) -> str:
        names = ", ".join(f"{k}={v:.6g}" for k, v in self.params.items())
        lo, nan = self.loss, float("nan")
        return (
            f"CalibrationResult({self.cultivar}: {names}; objective {lo.get('published', nan):.4g} "
            f"-> {lo.get('written', nan):.4g}; {len(self.treatments)} treatments, "
            f"{self.inputs} inputs; print() for the report)"
        )

    def _ctx(self) -> dict[str, Any]:
        if self.context is None:
            raise RuntimeError("this result carries no calibration context (it was not made by calibrate)")
        return self.context

    def summary(self) -> str:
        """A plain-text report."""
        lines = [
            f"Agri-JAX calibration of cultivar {self.cultivar} (DSSAT-CSM 4.8.6 day, nitrogen off), "
            f"method {self.method}, {self.inputs} inputs",
            f"treatments: {', '.join(self.treatments)}"
            + (f"; held out: {', '.join(self.holdout['treatments'])}" if self.holdout else ""),
            "coefficient   published   calibrated (written)",
        ]
        for n in CUL_ORDER:
            tag = "" if n in self.free else f"   fixed: {self.fixed.get(n, '')}"
            lines.append(f"  {n:<6} {self.published[n]:>12.4g} {self.params[n]:>14.4g}{tag}")
        lo = self.loss
        lines.append(
            f"objective: published {lo['published']:.4g} -> calibrated {lo['written']:.4g} "
            f"(unrounded {lo['unrounded']:.4g})"
        )
        if not self.improved:
            lines.append(
                "no written candidate beat the published cultivar: the published cultivar is returned"
            )
        if self.holdout:
            h = self.holdout
            lines.append(
                f"held out: published {h['loss_published']:.4g} -> calibrated {h['loss_written']:.4g}"
            )
        lines.append("targets (observations): " + ", ".join(f"{c} ({n})" for c, n in self.targets.items()))
        fit = self.fit
        if fit is not None and len(fit):
            lines.append("fit on the calibration treatments (RMSE; dates in days):")
            cal = fit[fit["set"] == "calibration"]
            for c, g in cal.groupby("code", sort=False):
                obs = g["observed"].to_numpy(float)
                if (g["kind"] == "date").all():
                    ec = [
                        _yrdoy_days_between(int(a), int(b)) for a, b in zip(g["calibrated"], obs, strict=True)
                    ]
                    ep = [
                        _yrdoy_days_between(int(a), int(b)) for a, b in zip(g["published"], obs, strict=True)
                    ]
                else:
                    ec = list(g["calibrated"].to_numpy(float) - obs)
                    ep = list(g["published"].to_numpy(float) - obs)
                rc, rp = float(np.sqrt(np.mean(np.square(ec)))), float(np.sqrt(np.mean(np.square(ep))))
                lines.append(f"  {c:<5} n={len(g):<3} published {rp:>10.4g}   calibrated {rc:>10.4g}")
        if self.cul_line:
            lines.append(f".CUL row ({self.cul_path}):")
            lines.append("  " + self.cul_line)
        if self.dssat_check:
            d = self.dssat_check
            lines.append(
                f"DSSAT check (dscsm048 with the written row): {'agrees' if d['agree'] else 'DISAGREES'}"
            )
            for r in d["treatments"]:
                lines.append(
                    f"  {r['treatment']}: HWAM DSSAT {r['dssat']['HWAM']:.0f} / Agri-JAX "
                    f"{r['agrijax']['HWAM']:.1f} (rel {r['yield_rel_diff']:.1e}); "
                    f"ADAT {int(r['dssat']['ADAT'])} / {int(r['agrijax']['ADAT'])}; "
                    f"MDAT {int(r['dssat']['MDAT'])} / {int(r['agrijax']['MDAT'])}"
                )
        c = self.calls
        lines.append(
            f"calls: {c.get('candidates', 0)} candidates, {c.get('seasons', 0)} seasons, "
            f"{c.get('program_calls', 0)} program calls on {c.get('n_devices', '?')} devices; "
            f"wall {self.wall_s:.1f} s (compile {c.get('compile_s', 0.0):.1f} s)"
        )
        for w in self.warnings:
            lines.append(f"warning: {w}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (tables as lists of records)."""
        out = {k: v for k, v in self.__dict__.items() if k not in ("fit", "holdout", "cul_path", "context")}
        out["fit"] = [] if self.fit is None else self.fit.to_dict("records")
        out["cul_path"] = None if self.cul_path is None else str(self.cul_path)
        if self.holdout is not None:
            h = dict(self.holdout)
            h["fit"] = h["fit"].to_dict("records")
            out["holdout"] = h
        return out


# ------------------------------------------------------------------ inputs
def _filex_path(exp: str | os.PathLike[str], engine: Path) -> Path:
    p = Path(exp)
    if p.suffix.upper() == ".MZX" and p.is_file():
        return p.resolve()
    if p.suffix and p.suffix.upper() != ".MZX":
        raise ScopeError(f"{exp}: only maize experiments (.MZX) are supported")
    q = engine / "example_data" / "Maize" / f"{p.name.upper()}.MZX"
    if not q.is_file():
        raise ScopeError(f"{exp}: no maize experiment file {q.name} in {q.parent} (pass the .MZX path)")
    return q


def _parse_treatments(spec: Any, exps: Sequence[str], what: str) -> list[tuple[str, int]]:
    """``[(experiment, treatment)]`` of a treatment specification (ints for one experiment,
    ``(exp, n)`` pairs or ``"EXP_tNN"`` / ``"EXP:N"`` strings)."""
    if spec is None:
        return []
    one = isinstance(spec, (int, np.integer, str)) or (
        isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str)
    )
    items: list[Any] = [spec] if one else list(cast("Sequence[Any]", spec))
    out: list[tuple[str, int]] = []
    for it in items:
        if isinstance(it, (int, np.integer)):
            if len(exps) != 1:
                raise ValueError(
                    f"{what}: {it} is ambiguous with several experiments; use (experiment, number)"
                )
            out.append((exps[0], int(it)))
        elif isinstance(it, str):
            if "_t" in it:
                e, n = it.rsplit("_t", 1)
            elif ":" in it:
                e, n = it.split(":", 1)
            else:
                raise ValueError(f"{what}: {it!r} is not 'EXP_tNN' or 'EXP:N'")
            out.append((e.upper(), int(n)))
        else:
            e, n = it
            out.append((str(e).upper(), int(n)))
    unknown = sorted({e for e, _ in out} - set(exps))
    if unknown:
        raise ValueError(f"{what}: experiments {unknown} are not among {list(exps)}")
    return out


def _scope_reason(
    exp: str, trno: int, treatments: dict[int, str], data_dir: Path, inputs: str = "tables", filex: Any = None
) -> str | None:
    from agrijax.sites.dssat_free_run import missing_tables, treatment_key

    if exp in OUT_OF_SCOPE:
        return OUT_OF_SCOPE[exp]
    if trno not in treatments:
        return f"no treatment {trno} in {exp}.MZX (treatments {sorted(treatments)})"
    key = treatment_key(exp, trno)
    d = NITROGEN_STRESS.get(key)
    if d is None:
        return (
            "its nitrogen stress is not measured (the model runs nitrogen off; the change of DSSAT's "
            "yield with nitrogen on is known for the DSSAT v4.8.6 maize example treatments only)"
        )
    if abs(d) > NITROGEN_STRESS_MAX:
        return (
            f"nitrogen matters: DSSAT's yield with nitrogen on differs by {d:+.1%} from nitrogen off "
            f"(the calibrated model runs nitrogen off; limit +-{NITROGEN_STRESS_MAX:.0%})"
        )
    if inputs == "native":
        from agrijax.io.dssat.filex import read_filex
        from agrijax.io.dssat.native_management import unsupported_features

        bad = unsupported_features(read_filex(filex), trno)
        if bad:
            return "needs a DSSAT process Agri-JAX does not implement: " + "; ".join(bad)
        return None
    miss = missing_tables(exp, trno, data_dir)
    if miss:
        return (
            f"the free-run day's input tables are not under {data_dir}: "
            f"{[str(p.relative_to(data_dir)) for p in miss]}; they are written by an instrumented "
            "DSSAT-CSM build for the v4.8.6 maize example treatments and are not distributed"
        )
    return None


def _nitrogen_warning(key: str) -> str | None:
    """The warning of an in-scope treatment where nitrogen matters a little: its yield change between
    nitrogen on and off is in the warning band (above :data:`NITROGEN_STRESS_OK`, up to
    :data:`NITROGEN_STRESS_MAX`), or its silking / maturity date changes with nitrogen on
    (:data:`NITROGEN_DATES_DIFFER`); ``None`` otherwise."""
    d = NITROGEN_STRESS[key]
    why: list[str] = []
    if abs(d) > NITROGEN_STRESS_OK:
        why.append(
            f"DSSAT yield with nitrogen on differs by {d:+.1%} from nitrogen off "
            f"(within the +-{NITROGEN_STRESS_MAX:.0%} limit, above +-{NITROGEN_STRESS_OK:.0%})"
        )
    if key in NITROGEN_DATES_DIFFER:
        why.append("DSSAT's silking (ADAT) or maturity (MDAT) date changes with nitrogen on")
    if not why:
        return None
    return (
        f"{key}: "
        + "; ".join(why)
        + ": nitrogen limitation may bias the fit (the calibrated model runs nitrogen off)"
    )


def _cultivars(filex: Path) -> dict[int, str]:
    from agrijax.io.dssat.filex import read_filex

    x = read_filex(filex)
    return {int(t["N"]): str(x["CULTIVARS"][t["CU"]]["INGENO"]).strip() for t in x["TREATMENTS"]}


def _run_root() -> Path:
    root = Path(os.environ.get("AGRI_JAX_RUN_ROOT") or tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="ajcal_", dir=root))


def _observations(
    obsdata: Any, run: Any, codes: Sequence[str] | None
) -> tuple[
    list[tuple[str, Entry, float, int]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str], list[str]
]:
    """Targets of one treatment: ``[(code, entry, value, day)]``, the A-file targets that cannot be
    placed (errors), the T-file observations dropped (outside the simulated days), the unmapped codes
    and the notes on dates observed more than once (averaged)."""
    from agrijax.calib.observations import DSSAT_TARGETS, observation_targets
    from agrijax.io.dssat.observed import reada_date

    days = run.days
    ot = observation_targets(obsdata, [run.trno], days[None], codes=codes, sim_start=[int(days[0])])
    rows: list[tuple[str, Entry, float, int]] = []
    for t in ot.targets:
        v = np.asarray(ot.observed[t.name])
        out = OUT_NAMES.index(t.output) if t.output in OUT_NAMES else -1
        m = DSSAT_TARGETS[t.name]
        if t.kind == "date":
            idx = int(v[0])
            stage = t.code if t.code is not None else m.code_stage
            assert stage is not None
            rows.append((t.name, Entry(E_DATE, int(stage)), float(idx), int(days[idx])))
        elif t.kind == "final" or m.kind == "final":
            val = float(v[0]) if t.kind == "final" else float(v[np.asarray(t.mask)[:, 0], 0][0])
            rows.append((t.name, Entry(E_FINAL, out), val, -1))
        else:
            mask = np.asarray(t.mask)[:, 0]
            for d in np.nonzero(mask)[0]:
                rows.append((t.name, Entry(E_DAY, out, int(d)), float(v[d, 0]), int(days[d])))
    used = {r[0] for r in rows}
    errors: list[dict[str, Any]] = []
    fa = obsdata.filea
    if fa is not None:
        for c in fa.codes:
            m = DSSAT_TARGETS.get(c)
            if m is None or m.file != "A" or (codes is not None and c not in codes) or c in used:
                continue
            val = fa.value(run.trno, c)
            if val is None or not np.isfinite(val):
                continue
            if m.kind == "date":
                if np.trunc(val) <= 0:
                    continue
                d = reada_date(float(val), int(days[0]), None)
                if d is None:
                    errors.append(
                        {"treatment": run.key, "code": c, "observed": float(val), "unreadable": True}
                    )
                    continue
                errors.append(
                    {
                        "treatment": run.key,
                        "code": c,
                        "observed": d.year * 1000 + d.timetuple().tm_yday,
                        "first_day": int(days[0]),
                        "last_day": int(days[-1]),
                    }
                )
            else:
                errors.append({"treatment": run.key, "code": c, "observed": float(val)})
    dropped: list[dict[str, Any]] = []
    tt = obsdata.filet
    if tt is not None:
        real = set(days.tolist())
        for c in tt.codes:
            m = DSSAT_TARGETS.get(c)
            if m is None or m.file != "T" or (codes is not None and c not in codes):
                continue
            s = tt.series(run.trno, c)
            for yd, val in zip(
                s["YRDOY"].to_numpy(np.int64).tolist(), s["value"].to_numpy(float).tolist(), strict=True
            ):
                if int(yd) not in real and np.isfinite(val):
                    dropped.append(
                        {
                            "treatment": run.key,
                            "code": c,
                            "day": int(yd),
                            "observed": float(val),
                            "reason": f"outside the simulated days {int(days[0])}-{int(days[-1])}",
                        }
                    )
    averaged = [f"{run.key}: {n}" for n in ot.notes if "averaged" in n]
    return rows, errors, dropped, dict(ot.unsupported), averaged


def _reference_inputs(
    filex: Path,
    trno: int,
    dest: Path,
    data: Path,
    engine: Path | None,
    source: str = "tables",
    root: Path | None = None,
) -> Any:
    """The free-run inputs: native without any DSSAT run (``root``: DSSAT's data), or from the
    reference run (``dscsm048``, published cultivar, nitrogen off) and the instrumented tables."""
    from agrijax.sites.dssat_free_run import free_run_inputs, run_reference

    r = root if root is not None else engine
    if source == "native":
        return free_run_inputs(filex, trno, None, engine=r, source="native")
    assert engine is not None and r is not None
    out = run_reference(
        filex,
        trno,
        dest,
        engine=engine,
        weather_dir=r / "example_data" / "Weather",
        soil_dir=r / "example_data" / "Soil",
    )
    return free_run_inputs(filex, trno, out, data, engine=r, source=source)


def _day_simulator(runs: Sequence[Any], entries: Sequence[Sequence[Entry]]) -> Any:
    from agrijax.calib.dssat_day import DaySimulator

    return DaySimulator(runs, entries)


def _dssat_run(
    filex: Path,
    trno: int,
    dest: Path,
    engine: Path,
    cultivar: tuple[str, str],
    cul_file: Path,
    root: Path | None = None,
) -> dict[str, Any]:
    """``dscsm048`` with the written row: the summary values and whether the run read the new id."""
    from agrijax.io.dssat import read_summary
    from agrijax.sites.dssat_free_run import run_reference

    r = root if root is not None else engine
    out = run_reference(
        filex,
        trno,
        dest,
        engine=engine,
        cultivar=cultivar,
        cul_file=cul_file,
        weather_dir=r / "example_data" / "Weather",
        soil_dir=r / "example_data" / "Soil",
    )
    row = read_summary(out / "Summary.OUT").iloc[0]
    inp = (out / "DSSAT48.INP").read_text(errors="replace")
    return {
        "row": {c: float(row[c]) for c in ("HWAM", "CWAM", "H#AM", "ADAT", "MDAT")},
        "read_new_row": cultivar[1] in inp,
    }


def _cul_dest(path: str | os.PathLike[str]) -> Path:
    d = Path(path)
    if d.is_dir() or str(path).endswith(("/", os.sep)):
        d = d / "MZCER048.CUL"
    return d


def _write_row(
    ctx: dict[str, Any], params_w: dict[str, float], dest: Path, cul_id: str | None
) -> tuple[str, str, list[str]]:
    """Write the engine's ``MZCER048.CUL`` with the row ``params_w`` to ``dest``: ``(id, row,
    warnings)``."""
    from agrijax.io.dssat.cultivar_write import write_cultivar

    cid = cul_id or _next_cul_id(ctx["cul_src"])
    name = ctx["cultivar"]
    w = write_cultivar(ctx["cul_src"], dest, cid, f"AgriJAX {name}"[:16], params_w, base=name)
    if w.rounded:
        raise AssertionError(f"written values were rounded again: {w.rounded}")
    msgs = [f"written values outside MINIMA / MAXIMA: {w.out_of_range}"] if w.out_of_range else []
    return cid, w.line, msgs


def _dssat_check(
    ctx: dict[str, Any], predicted: dict[str, dict[str, float]], cid: str, cul: Path, work: Path
) -> dict[str, Any]:
    """``dscsm048`` with the row ``cid`` of ``cul`` on every treatment against ``predicted``."""
    from agrijax.sites.dssat_free_run import treatment_key

    allt = ctx["treatments"]
    eng = ctx.get("engine")
    if eng is None:  # calibrated without DSSAT: get the reference program now
        from agrijax.io.examples import install_reference

        eng = ctx["engine"] = install_reference(verbose=False)

    def one(k: tuple[str, int]) -> dict[str, Any]:
        e, t = k
        return _dssat_run(
            ctx["filex"][e],
            t,
            work / f"chk_{treatment_key(e, t)}",
            eng,
            (ctx["cultivar"], cid),
            cul,
            ctx.get("data"),
        )

    with ThreadPoolExecutor(max_workers=max(1, min(len(allt), os.cpu_count() or 1))) as ex:
        dres = list(ex.map(one, allt))
    trs = []
    for (e, t), d in zip(allt, dres, strict=True):
        key = treatment_key(e, t)
        ours = predicted[key]
        ds = d["row"]
        rel = abs(ours["HWAM"] - ds["HWAM"]) / max(abs(ds["HWAM"]), 1.0)
        dates = int(ours["ADAT"]) == int(ds["ADAT"]) and int(ours["MDAT"]) == int(ds["MDAT"])
        trs.append(
            {
                "treatment": key,
                "dssat": ds,
                "agrijax": {c: ours[c] for c in ("HWAM", "CWAM", "H#AM", "ADAT", "MDAT")},
                "yield_rel_diff": rel,
                "dates_equal": dates,
                "read_new_row": d["read_new_row"],
                "agree": dates and rel <= DSSAT_YIELD_RTOL and d["read_new_row"],
            }
        )
    return {
        "cultivar_id": cid,
        "yield_rtol": DSSAT_YIELD_RTOL,
        "treatments": trs,
        "agree": all(t["agree"] for t in trs),
    }


def _resolve_inputs(inputs: str, keys: Sequence[tuple[str, int]], data: Path) -> str:
    """``"native"`` or ``"tables"`` for ``inputs`` (``"auto"``: the tables when every treatment
    ``keys`` has them under ``data``)."""
    from agrijax.sites.dssat_free_run import missing_tables

    if inputs not in INPUTS:
        raise ValueError(f"inputs: one of {INPUTS}, got {inputs!r}")
    if inputs != "auto":
        return inputs
    return "tables" if keys and all(not missing_tables(e, t, data) for e, t in keys) else "native"


def _next_cul_id(cul: Path) -> str:
    from agrijax.io.dssat.genotype import read_cul

    have = {str(i).strip() for i in read_cul(cul).index}
    for k in range(1, 10000):
        cid = f"{CUL_ID_PREFIX}{k:04d}"
        if cid not in have:
            return cid
    raise ValueError(f"no free {CUL_ID_PREFIX}NNNN cultivar id in {cul}")


# ------------------------------------------------------------------ the call
def calibrate(
    experiment: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    treatments: Any = None,
    cultivar: str | None = None,
    *,
    params: str | Sequence[str] = "auto",
    targets: str | Sequence[str] = "auto",
    method: str = "cma",
    starts: int = 8,
    seed: int = 0,
    budget: int | None = None,
    write_cul: str | os.PathLike[str] | None = None,
    cul_id: str | None = None,
    dssat_check: bool = False,
    holdout: Any = None,
    data_dir: str | os.PathLike[str] | None = None,
    engine: str | os.PathLike[str] | None = None,
    inputs: str = "auto",
    data: str | os.PathLike[str] | None = None,
) -> CalibrationResult:
    """Calibrate a CERES-Maize cultivar on a DSSAT maize experiment (module docstring).

    ``experiment``: an example experiment name (``"UFGA8201"``), a ``.MZX`` path, or several of them;
    ``treatments``: treatment numbers (one experiment), ``(experiment, number)`` pairs or
    ``"EXP_tNN"`` names (``None``: every supported treatment of ``cultivar``); ``cultivar``: the FileX
    cultivar id (``INGENO``) when the treatments use several; ``params``: ``"auto"`` (the six ``.CUL``
    coefficients less the ones the observations carry no information on) or a sequence of names;
    ``targets``: ``"auto"`` (:data:`BASE_TARGETS`), ``"all"`` or observed codes; ``method``:
    ``"cma"``, ``"staged"`` or ``"adam"``; ``starts``, ``seed``, ``budget``: see
    :func:`agrijax.calib.fit.fit_cultivar`; ``write_cul``: a ``.CUL`` file (or a directory) to write
    the engine's ``MZCER048.CUL`` with the new row ``cul_id`` into (default id: the first free
    ``AJNNNN``; the file must not exist); ``dssat_check``: run ``dscsm048`` with that row and
    compare; ``holdout``: treatments evaluated, not fitted; ``data_dir``: the data directory with
    the free-run tables (default ``AGRI_JAX_DATA`` or ``~/agri_jax_data``); ``engine``: the DSSAT
    engine root (default ``AGRI_JAX_DSSAT``; only the table inputs and the DSSAT check run it, the
    native inputs need no DSSAT); ``inputs``: the source of the free-run inputs, ``"auto"`` (the
    tables when every requested treatment has them, else native), ``"native"`` or ``"tables"``
    (module docstring); ``data``: the root of DSSAT's data (``example_data``, the genotype and standard
    data files; default ``engine`` when given, else :func:`agrijax.io.examples.fetch_dssat_data`).
    """
    import jax

    from agrijax.calib.observations import DSSAT_TARGETS, UNSUPPORTED_REASONS
    from agrijax.io.dssat.cultivar_write import cul_precision, cul_written
    from agrijax.io.dssat.observed import read_observed
    from agrijax.port.run_fortran import DSSAT_ENGINE, dscsm_paths
    from agrijax.sites.dssat_free_run import dssat_data_dir, treatment_key

    t0 = time.perf_counter()
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "calibrate runs in 64-bit: call jax.config.update('jax_enable_x64', True) before creating arrays"
        )
    eng: Path | None = Path(engine) if engine is not None else None
    # the native inputs need no DSSAT program: it is not even looked for (the DSSAT check finds it)
    if eng is None and inputs != "native" and dscsm_paths(DSSAT_ENGINE)[0].is_file():
        eng = DSSAT_ENGINE
    if eng is not None and inputs != "native" and not dscsm_paths(eng)[0].is_file():
        raise FileNotFoundError(
            f"dscsm048 not found at {dscsm_paths(eng)[0]} (set AGRI_JAX_DSSAT or pass engine=)"
        )
    if data is not None:
        root = Path(data)
    elif eng is not None and (eng / "example_data").is_dir():
        root = eng
    else:
        from agrijax.io.examples import fetch_dssat_data

        root = fetch_dssat_data()
    cul_src = dssat_data_dir(root) / "Genotype" / "MZCER048.CUL"
    data_tab = Path(
        data_dir if data_dir is not None else os.environ.get("AGRI_JAX_DATA", "~/agri_jax_data")
    ).expanduser()
    exp_list = [experiment] if isinstance(experiment, (str, os.PathLike)) else list(experiment)
    filex = {}
    for e in exp_list:
        p = _filex_path(e, root)
        filex[p.stem.upper()] = p
    exps = list(filex)
    for e in exps:
        if e in OUT_OF_SCOPE:
            raise ScopeError(f"{e} is outside the supported scope: {OUT_OF_SCOPE[e]}")
    cul_dest: Path | None = None
    if write_cul is not None:
        cul_dest = _cul_dest(write_cul)
        if cul_dest.exists():
            raise FileExistsError(f"write_cul: {cul_dest} exists (remove it or choose another path)")
    cults = {e: _cultivars(filex[e]) for e in exps}
    msgs: list[str] = []
    notes: list[str] = []

    def warn(m: str) -> None:
        msgs.append(m)
        warnings.warn(m, CalibrationWarning, stacklevel=3)

    hold = _parse_treatments(holdout, exps, "holdout")
    asked = (
        [(e, t) for e in exps for t in sorted(cults[e])]
        if treatments is None
        else [*_parse_treatments(treatments, exps, "treatments"), *hold]
    )
    source = _resolve_inputs(inputs, asked, data_tab)
    if source == "tables" and eng is None:
        raise FileNotFoundError("the table inputs need dscsm048 (set AGRI_JAX_DSSAT or pass engine=)")

    def scope(e: str, t: int) -> str | None:
        return _scope_reason(e, t, cults[e], data_tab, source, filex[e])

    if treatments is None:
        cand = [(e, t) for e in exps for t in sorted(cults[e])]
        ok = [(e, t) for e, t in cand if scope(e, t) is None and (e, t) not in hold]
        excl = {treatment_key(e, t): r for e, t in cand if (r := scope(e, t)) is not None}
        if excl:
            notes.append(f"treatments outside the supported scope, not used: {excl}")
        if cultivar is not None:
            ok = [(e, t) for e, t in ok if cults[e][t] == cultivar]
        cal = ok
    else:
        cal = _parse_treatments(treatments, exps, "treatments")
    if not cal:
        raise ScopeError(
            f"no supported treatment to calibrate in {exps}"
            + (f" for cultivar {cultivar}" if cultivar else "")
        )
    if set(cal) & set(hold):
        raise ValueError(f"treatments both calibrated and held out: {sorted(set(cal) & set(hold))}")
    bad = {treatment_key(e, t): r for e, t in [*cal, *hold] if (r := scope(e, t)) is not None}
    if bad:
        raise ScopeError("outside the supported scope: " + "; ".join(f"{k}: {r}" for k, r in bad.items()))
    used_cul = {cults[e][t] for e, t in [*cal, *hold]}
    if cultivar is not None and used_cul != {cultivar}:
        raise ScopeError(f"treatments use cultivars {sorted(used_cul)}, not only {cultivar}")
    if len(used_cul) != 1:
        by = {
            c: [treatment_key(e, t) for e, t in [*cal, *hold] if cults[e][t] == c] for c in sorted(used_cul)
        }
        raise ScopeError(
            f"one cultivar per calibration; these treatments use {by}: pass cultivar= or treatments="
        )
    (cul_name,) = used_cul
    for e, t in [*cal, *hold]:
        if (nw := _nitrogen_warning(treatment_key(e, t))) is not None:
            warn(nw)
    # ---------------------------------------------------------------- targets requested
    codes: list[str] | None = None
    explicit = not isinstance(targets, str)
    if targets == "auto":
        codes = list(BASE_TARGETS)
    elif not isinstance(targets, str):
        codes = [str(c).upper() for c in targets]
        unk = {
            c: UNSUPPORTED_REASONS.get(c, "no model output for this code")
            for c in codes
            if c not in DSSAT_TARGETS
        }
        if unk:
            raise ObservationError(f"targets not mapped to a model output: {unk}")
    elif targets != "all":
        raise ValueError(f"targets: 'auto', 'all' or a sequence of observed codes, got {targets!r}")
    # ---------------------------------------------------------------- reference runs and inputs
    work = _run_root()
    try:
        allt = [*cal, *hold]

        def ref(k: tuple[str, int]) -> Any:
            e, t = k
            return _reference_inputs(filex[e], t, work / treatment_key(e, t), data_tab, eng, source, root)

        with ThreadPoolExecutor(max_workers=max(1, min(len(allt), os.cpu_count() or 1))) as ex:
            runs = list(ex.map(ref, allt))
        pubs = {tuple(np.round([r.published()[n] for n in CUL_ORDER], 6)) for r in runs}
        if len(pubs) != 1:
            raise ScopeError(
                f"the treatments' DSSAT48.INP hold different coefficients for {cul_name}: {pubs}"
            )
        published = runs[0].published()
        # ---------------------------------------------------------------- observations
        obs_cache = {e: read_observed(filex[e]) for e in exps}
        tr_list: list[Treatment] = []
        observed: list[Observed] = []
        errors: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        unsupported: dict[str, str] = {}
        for b, r in enumerate(runs):
            rows, err, drp, uns, avg = _observations(obs_cache[r.exp], r, codes)
            if avg:
                warn("; ".join(avg))
            errors += err
            dropped += drp
            unsupported.update(uns)
            entries = [e for _, e in PROBES] + [e for _, e, _, _ in rows]
            for j, (c, _, v, d) in enumerate(rows):
                observed.append(Observed(b, c, len(PROBES) + j, v, d))
            tr_list.append(Treatment(r.key, r.days, tuple(entries)))
            if not rows:
                warn(f"{r.key}: no usable observation")
        if errors:
            lines = []
            for e in errors:
                if e.get("unreadable"):
                    lines.append(
                        f"{e['treatment']}: observed {e['code']} {e['observed']:g} is not a readable date"
                    )
                elif "last_day" in e:
                    lines.append(
                        f"{e['treatment']}: observed {e['code']} {e['observed']} lies outside the simulated "
                        f"days {e['first_day']}-{e['last_day']} (the reference season, whose input tables "
                        "end there)"
                    )
                else:
                    lines.append(f"{e['treatment']}: observed {e['code']} = {e['observed']} cannot be used")
            raise ObservationError(
                "observed targets cannot be placed on the simulated days (nothing is dropped silently):\n  "
                + "\n  ".join(lines)
                + "\nleave the treatment out, or leave the code out with targets=(...)"
            )
        if dropped:
            drop_by: dict[tuple[str, int], list[str]] = {}
            for d in dropped:
                drop_by.setdefault((d["treatment"], d["day"]), []).append(d["code"])
            warn(
                f"{len(dropped)} T-file observations lie outside the simulated days and are not used "
                "(listed in CalibrationResult.dropped): "
                + "; ".join(f"{t} {day}: {', '.join(c)}" for (t, day), c in drop_by.items())
            )
        n_cal = len(cal)
        if explicit and codes is not None:
            seen = {o.code for o in observed if o.treatment < n_cal}
            miss = [c for c in codes if c not in seen]
            if miss:
                raise ObservationError(f"targets {miss} are not observed on the calibration treatments")
        # ---------------------------------------------------------------- the problem
        dec = cul_precision(cul_src)

        def written(theta: np.ndarray) -> np.ndarray:
            return np.asarray(
                [
                    [cul_written(dict(zip(CUL_ORDER, row, strict=True)), decimals=dec)[n] for n in CUL_ORDER]
                    for row in np.asarray(theta, dtype=float).tolist()
                ]
            )

        sim = _day_simulator(runs, [t.entries for t in tr_list])
        problem = CultivarProblem(
            simulate=sim,
            treatments=tr_list,
            observed=observed,
            published=np.asarray([published[n] for n in CUL_ORDER]),
            written=written,
            jax_loss=sim.jax_loss,
            stats=sim.stats,
        )
        res = fit_cultivar(
            problem,
            range(n_cal),
            holdout=range(n_cal, len(runs)),
            params=params,
            method=method,
            starts=starts,
            seed=seed,
            budget=budget,
        )
        msgs += res.warnings
        params_w = dict(zip(CUL_ORDER, res.theta_written.tolist(), strict=True))
        # ---------------------------------------------------------------- .CUL row and DSSAT check
        ctx = {
            "filex": dict(filex),
            "engine": eng,
            "data": root,
            "cul_src": cul_src,
            "cultivar": cul_name,
            "treatments": list(allt),
        }
        cul_line = cul_path = None
        cid = cul_id
        dest = cul_dest if cul_dest is not None else work / "cul" / "MZCER048.CUL"
        if write_cul is not None or dssat_check:
            cid, cul_line, wmsgs = _write_row(ctx, params_w, dest, cid)
            cul_path = dest if write_cul is not None else None
            for m in wmsgs:
                warn(m)
        check = None
        if dssat_check:
            assert cid is not None
            check = _dssat_check(ctx, res.predictions["calibrated"], cid, dest, work)
            if not check["agree"]:
                warn(
                    f"DSSAT and Agri-JAX disagree at the written coefficients: "
                    f"{[t['treatment'] for t in check['treatments'] if not t['agree']]}"
                )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    import pandas as pd

    fit_df = pd.DataFrame(res.fit)
    hold_res = None
    if res.holdout is not None:
        hold_res = {**res.holdout, "fit": pd.DataFrame(res.holdout["fit"])}
    return CalibrationResult(
        cultivar=cul_name,
        treatments=[r.key for r in runs[:n_cal]],
        params=params_w,
        params_unrounded=dict(zip(CUL_ORDER, res.theta_unrounded.tolist(), strict=True)),
        published=published,
        free=res.free,
        fixed=res.fixed,
        loss=res.loss,
        fit=fit_df,
        targets=res.targets,
        scales=res.scales,
        dropped=dropped,
        unsupported=unsupported,
        sensitivity=res.sensitivity,
        trust=res.trust,
        predictions=res.predictions,
        reference={r.key: dict(r.summary) for r in runs},
        per_start=res.per_start,
        method=method,
        calls=res.calls,
        wall_s=time.perf_counter() - t0,
        holdout=hold_res,
        cul_id=cid if (write_cul is not None or dssat_check) else None,
        cul_line=cul_line,
        cul_path=cul_path,
        dssat_check=check,
        improved=res.improved,
        notes=notes,
        warnings=msgs,
        inputs=source,
        context=ctx,
    )
