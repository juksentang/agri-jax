"""Inputs of the DSSAT-CSM v4.8.6.0 day run free (:mod:`agrijax.models.day_dssat486`) for one maize treatment.

The free-run day simulates the soil water, the evaporation, the transpiration, the root water uptake
and the crop itself; what it takes from DSSAT-CSM is what DSSAT-CSM reads or derives before these
modules run. For one treatment ``<EXP>_t<NN>`` of a DSSAT maize experiment:

* the **reference run** (``dscsm048`` build486, nitrogen off, a one-treatment batch:
  :func:`run_reference`) gives CERES-Maize's parameters (``DSSAT48.INP`` with the ``.ECO`` / ``.SPE``
  files of the engine), the crop weather (``Weather.OUT``) and the reference summary
  (``Summary.OUT``: planting, maturity and harvest dates);
* the soil and soil-water inputs as ``dscsm048`` holds them, from one of two sources:

  - ``source="native"`` (:func:`native_series`): built from the public files -- the experiment's
    FileX and ``.SOL``, the ``.WTH``, the engine's standard data and ecotype file -- by the native
    ports of the DSSAT input chain (:mod:`agrijax.io.dssat.native_soil`: ``LYRSET2`` / ``LMATCH``,
    the ``DSSAT48.INP`` round trip, ``SOILDYN`` defaults, the initial water;
    :mod:`agrijax.io.dssat.native_weather` with :mod:`agrijax.forcing.dssat_weather`: ``IPWTH``,
    ``HMET``'s hourly mean ``TAVG``, the 2 m wind, ``CO2VAL``;
    :mod:`agrijax.io.dssat.native_management`: the reported irrigation, the residue parameters,
    ``KEP``). Treatments that need a DSSAT process not ported (automatic irrigation, environment
    modifications, surface residue and its decay, the CENTURY organic matter, tillage, a water
    table ...) raise :class:`NativeInputError` naming it. On the 50 supported runs of the free-run
    acceptance every input equals the table one exactly except ``TAVG`` (within 4 REAL*4 units;
    ``tests/integration/test_dssat_native_inputs.py``);
  - ``source="tables"`` (the default; the validation harness): the **instrumented-engine tables** of
    the treatment under the data directory: the ``WATBAL`` tables
    (``validation/aj_dsw/d1a/tables/<key>/``: the static ``SOILPROP``, the initial ``SW``, snow and
    mulch water, the day's ``RAIN``, ``TMAX``, ``IRRAMT``, the residue record replayed from the
    organic matter module and, where ``SOILDYN`` changes it, the day's ``SOILPROP``) and the
    ``SPAM`` entry table (``validation/aj_det/tables/<key>_spam_in.npz``: the weather record
    ``SPAM`` reads with the hourly mean ``TAVG``, ``SALB``, ``U``, the residue cover and
    ``KSEVAP = KTRANS``).

This is the ``free`` configuration of the free-run acceptance
(``tests/integration/day_dssat486_free_harness.py``: REAL*4 soil values, the ``SOILDYN`` replay on for
runs whose soil changes, ``exact_lags=True``), for the example treatments; the integration test
``test_calib_workflow_dssat.py`` checks that the trees built here equal the harness's. The crop
forcing's soil water, ``EOP`` and ``TRWUP`` are NaN: the crop reads them from the day's own modules.

With the tables, only treatments that have them can run free (:func:`missing_tables` lists what is
missing). Everything here is host-side (NumPy, files); the trees it returns are JAX arrays.
"""

from __future__ import annotations

import functools
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from agrijax.core.units import MM_PER_CM

__all__ = [
    "CULTIVAR_FIELDS",
    "DSW_TABLES",
    "FILEX_DATE_FIELDS",
    "SPAM_TABLES",
    "FreeRunInputs",
    "NativeInputError",
    "cultivar_batch_filex",
    "dssat_data_dir",
    "dssat_free_inputs",
    "free_run_inputs",
    "missing_tables",
    "native_inp",
    "native_series",
    "nitrogen_off_filex",
    "run_reference",
    "set_filex_cultivar",
    "shift_filex_dates",
    "stack_trees",
    "treatment_key",
    "with_cultivar",
]

#: the ``WATBAL`` tables (one directory per treatment) and the ``SPAM`` entry tables under the data
#: directory (written by the instrumented DSSAT-CSM v4.8.6.0 engine runs)
DSW_TABLES = Path("validation/aj_dsw/d1a/tables")
SPAM_TABLES = Path("validation/aj_det/tables")
#: the ``WATBAL`` tables a run needs (``rnoff_in`` is optional: the plastic-mulch fraction)
_DSW_REQUIRED = ("wb_rate_in", "wb_integr_in", "mulch_rate_in", "wb_seasinit_in")
#: the soil properties SOILDYN may change (the SOILPROP components the day's modules read)
SOILPROP = ("dlayr", "ds", "ll", "dul", "sat", "swcn")
#: soil values under the REAL*4 convention (:func:`agrijax.models.day_dssat486.dssat_soil_values`)
SOIL_VALUES = "real4"
#: the FileX options a reference run sets: ``FROPT`` 1 (daily outputs), growth and water outputs on
_OUTPUT_SWITCHES = (("FROPT", "1"), ("GROUT", "Y"), ("WAOUT", "Y"))


def treatment_key(exp: str, trno: int) -> str:
    """``<EXP>_t<NN>``: the name of a treatment's tables."""
    return f"{exp}_t{int(trno):02d}"


# ------------------------------------------------------------------------ FileX and reference run
def nitrogen_off_filex(text: str, water: str | None = None, *, nitro: str = "N") -> str:
    """FileX text with ``NITRO = nitro`` (default ``N``; and optionally ``WATER``) and daily growth /
    water outputs.

    Values are right-aligned under their header names (``@N OPTIONS ... WATER NITRO``)."""
    out = []
    opt = outp = ""
    for ln in text.splitlines():
        if ln.startswith("@N OPTIONS"):
            opt = ln
        elif ln.startswith("@N OUTPUTS"):
            outp = ln
        elif opt and ln.split()[1:2] == ["OP"]:
            k = opt.index("NITRO") + 4
            ln = ln[:k] + nitro + ln[k + 1 :]
            if water is not None:
                k = opt.index("WATER") + 4
                ln = ln[:k] + water + ln[k + 1 :]
            opt = ""
        elif outp and ln.split()[1:2] == ["OU"]:
            for name, val in _OUTPUT_SWITCHES:
                k = outp.index(name) + 4
                ln = ln[:k] + val + ln[k + 1 :]
            outp = ""
        out.append(ln)
    return "\n".join(out) + "\n"


def set_filex_cultivar(text: str, old: str, new: str) -> str:
    """FileX text with the ``*CULTIVARS`` rows of cultivar ``old`` pointed at ``new`` (``INGENO``)."""
    out, sec = [], ""
    n = 0
    for ln in text.splitlines():
        if ln.startswith("*"):
            sec = ln
        if sec.startswith("*CULTIVARS") and not ln.startswith(("*", "@", "!")) and f" {old} " in ln:
            ln = ln.replace(f" {old} ", f" {new} ", 1)
            n += 1
        out.append(ln)
    if n == 0:
        raise ValueError(f"no *CULTIVARS row of {old!r} in the experiment file")
    return "\n".join(out) + "\n"


#: FileX columns holding a date (``YYDDD`` or ``YYYYDDD``): initial conditions, planting, emergence,
#: irrigation, fertilizer, residue, chemical, tillage, environment modification and harvest records,
#: the simulation start and the automatic planting / harvest windows
FILEX_DATE_FIELDS = frozenset(
    {"ICDAT", "PDATE", "EDATE", "IDATE", "FDATE", "RDATE", "CDATE", "TDATE", "ODATE", "HDATE", "SDATE"}
    | {"PFRST", "PLAST", "HFRST", "HLAST"}
)
#: two-digit FileX years below this are read as 20YY for the date arithmetic (only leap years matter)
_Y2K_PIVOT = 30
_YYDDD, _YYYYDDD = 5, 7


def _shift_date(tok: str, years: int, days: int) -> str:
    import datetime as _dt

    if not tok.isdigit() or len(tok) not in (_YYDDD, _YYYYDDD) or int(tok) <= 0:
        return tok
    y, doy = int(tok[:-3]), int(tok[-3:])
    if len(tok) == _YYDDD:
        y += 2000 if y < _Y2K_PIVOT else 1900
    d = _dt.date(y + years, 1, 1) + _dt.timedelta(days=doy - 1 + days)
    doy2 = d.timetuple().tm_yday
    return f"{d.year % 100:02d}{doy2:03d}" if len(tok) == _YYDDD else f"{d.year:04d}{doy2:03d}"


def _header_names(line: str) -> list[tuple[str, int]]:
    """``(name, start column)`` of the words of a FileX header line (dots and ``@`` stripped)."""
    import re

    out = []
    for m in re.finditer(r"[^\s@]+", line):
        word = m.group(0)
        lead = len(word) - len(word.lstrip("."))
        out.append((word.strip("."), m.start() + lead))
    return out


def shift_filex_dates(text: str, *, years: int = 0, days: int = 0) -> str:
    """FileX text with every management and simulation date (:data:`FILEX_DATE_FIELDS`) moved by
    ``years`` (same day of year) and then ``days``: the same experiment on another weather year or
    sown earlier / later, its whole calendar (irrigation, fertilizer ...) moving with it.

    Values stay right-aligned under their header names; ``-99`` and ``0`` stay as they are. The
    weather file DSSAT reads follows the shifted year (``<INSI><YY>01.WTH``)."""
    out: list[str] = []
    ends: list[int] = []
    for ln in text.splitlines():
        if ln.startswith("@"):
            ends = [k + len(name) for name, k in _header_names(ln) if name in FILEX_DATE_FIELDS]
        elif ln.startswith("*") or not ln.strip():
            ends = []
        elif ends and not ln.lstrip().startswith("!"):
            for end in ends:
                if end > len(ln) or ln[end - 1] == " ":
                    continue
                j = end
                while j > 0 and ln[j - 1] != " ":
                    j -= 1
                ln = ln[:j] + _shift_date(ln[j:end], years, days).rjust(end - j) + ln[end:]
        out.append(ln)
    return "\n".join(out) + "\n"


def cultivar_batch_filex(text: str, trno: int, cultivar_ids: Sequence[str]) -> str:
    """FileX text whose treatments are ``len(cultivar_ids)`` copies of treatment ``trno``, copy ``k``
    (numbered ``k``, from 1) growing cultivar ``cultivar_ids[k - 1]``: one ``dscsm048`` batch run
    simulates the same season for several cultivars (at most 99, the width of the FileX levels)."""
    ids = list(cultivar_ids)
    if not 1 <= len(ids) <= _MAX_LEVELS:
        raise ValueError(f"1 to {_MAX_LEVELS} cultivars per batch, got {len(ids)}")
    out: list[str] = []
    sec = hdr = ""
    done = False
    for ln in text.splitlines():
        if ln.startswith("*"):
            sec, hdr, done = ln, "", False
        elif ln.startswith("@"):
            hdr = ln
        elif hdr and ln.strip() and not ln.lstrip().startswith("!"):
            if sec.startswith("*TREATMENTS"):
                if int(ln[:2]) == int(trno) and not done:
                    cu = hdr.index(" CU") + 3
                    for k in range(1, len(ids) + 1):
                        out.append(f"{k:2d}" + ln[2 : cu - 2] + f"{k:2d}" + ln[cu:])
                    done = True
                continue
            if sec.startswith("*CULTIVARS"):
                if not done:
                    cr = ln.split()[1]
                    out += [f"{k:2d} {cr} {cid:<6} {cid}" for k, cid in enumerate(ids, start=1)]
                    done = True
                continue
        out.append(ln)
    return "\n".join(out) + "\n"


#: the widest FileX level number (``I2``)
_MAX_LEVELS = 99


def _batch_file(filex: str, trno: int | Sequence[int]) -> str:
    """A ``DSSBatch.v48`` running the treatments ``trno`` of ``filex`` once each."""
    head = "$BATCH(MAIZE)\n!\n@FILEX" + " " * 88 + "TRTNO     RP     SQ     OP     CO\n"
    trs = [trno] if isinstance(trno, int) else list(trno)
    return head + "".join(filex.ljust(92) + f"{int(t):7d}      1      0      0      0\n" for t in trs)


def run_reference(
    filex: str | os.PathLike[str],
    trno: int | Sequence[int],
    dest: str | os.PathLike[str],
    *,
    engine: str | os.PathLike[str] | None = None,
    cultivar: tuple[str, str] | None = None,
    cul_file: str | os.PathLike[str] | None = None,
    nitrogen: bool = False,
    filex_edit: Callable[[str], str] | None = None,
    weather_dir: str | os.PathLike[str] | None = None,
    soil_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Run treatment ``trno`` of the maize experiment ``filex`` (``<EXP>.MZX``) with ``dscsm048``,
    nitrogen off, as a one-treatment batch; returns the output directory (``*.OUT``,
    ``DSSAT48.INP``).

    The experiment's files ``<EXP>.MZ*`` (FileX, observed A / T files) are copied into ``dest``;
    the weather files ``<INSI>*.WTH`` come from the engine's ``example_data/Weather``. With
    ``cultivar = (old, new)`` the FileX's cultivar ``old`` is replaced by ``new``, read from the
    ``MZCER048.CUL`` copy ``cul_file`` (staged next to the experiment, so it replaces the engine's).
    ``nitrogen=True`` runs with ``NITRO = Y`` instead (the nitrogen-stress measurement of the scope).
    ``filex_edit`` changes the FileX text last (e.g. :func:`shift_filex_dates`); the edited copy is
    ``dest/<EXP>.MZX``, the file to build the treatment's inputs from. A sequence ``trno`` runs those
    treatments in one batch. ``weather_dir`` / ``soil_dir`` hold the ``.WTH`` / ``.SOL`` files
    (default the engine's ``example_data/Weather`` and ``example_data/Soil``).
    """
    from agrijax.port.run_fortran import DSSAT_ENGINE, run_dscsm

    src = Path(filex)
    exp = src.stem
    eng = Path(engine) if engine is not None else DSSAT_ENGINE
    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    for f in src.parent.glob(exp + ".MZ*"):
        shutil.copy2(f, d / f.name)
    x = d / f"{exp}.MZX"
    text = nitrogen_off_filex(x.read_text(errors="replace"), nitro="Y" if nitrogen else "N")
    if cultivar is not None:
        text = set_filex_cultivar(text, *cultivar)
    if filex_edit is not None:
        text = filex_edit(text)
    if cul_file is not None:
        shutil.copy2(cul_file, d / "MZCER048.CUL")
    x.write_text(text)
    (d / "DSSBatch.v48").write_text(_batch_file(f"{exp}.MZX", trno))
    weather = Path(weather_dir) if weather_dir is not None else eng / "example_data" / "Weather"
    out = d / "out"
    run_dscsm(
        d,
        out,
        run_mode="B",
        experiment_file="DSSBatch.v48",
        engine=eng,
        weather_dir=weather,
        soil_dir=soil_dir,
        extra_files=sorted(weather.glob(exp[:4] + "*.WTH")),
        keep_files=("*.OUT", "DSSAT48.INP"),
    )
    return out


# ------------------------------------------------------------------------ tables
def missing_tables(exp: str, trno: int, data_dir: str | os.PathLike[str]) -> list[Path]:
    """The instrumented-engine tables of treatment ``trno`` that are not under ``data_dir``."""
    key = treatment_key(exp, trno)
    root = Path(data_dir)
    need = [root / SPAM_TABLES / f"{key}_spam_in.npz"]
    need += [root / DSW_TABLES / key / f"{n}.npz" for n in _DSW_REQUIRED]
    return [p for p in need if not p.is_file()]


def _chr(a: Any) -> str:
    v = np.asarray(a).reshape(-1)
    if v.dtype.kind in "SU":
        s = v[0]
        return (s.decode() if isinstance(s, bytes) else str(s)).strip()
    return np.asarray(a).tobytes().decode(errors="replace").strip("\x00 ")


def decimal_of(x: Any) -> np.ndarray:
    """The shortest decimal that rounds to the REAL*4 value (the number as written in the input file)."""
    a = np.asarray(x, dtype=np.float32)
    return np.asarray([float(str(v)) for v in a.reshape(-1)], dtype=np.float64).reshape(a.shape)


def _table(d: Path, name: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    from agrijax.port import dumps

    t, _ = dumps.load_table(d / f"{name}.npz")
    return np.asarray(t.date, dtype=np.int64), {k: np.asarray(v) for k, v in t.values.items()}


def _pad(x: Any, n: int) -> np.ndarray:
    a = np.asarray(x)
    idx = np.minimum(np.arange(n), a.shape[0] - 1)
    return a[idx]


# ------------------------------------------------------------------------ one treatment
@dataclass
class FreeRunInputs:
    """Host-side inputs of one treatment's free run (build the trees with :meth:`params`,
    :meth:`forcing`, :meth:`state`)."""

    key: str
    exp: str
    trno: int
    #: the reference run's output directory (``None``: built without running DSSAT)
    out: Path | None
    #: the FileX cultivar (``INGENO``) of the treatment
    cultivar: str
    params_crop: Any
    forcing_crop: Any
    #: the reference run's ``Summary.OUT`` row (``SDAT``, ``ADAT``, ``MDAT``, ``HDAT``, ``HWAM`` ...)
    summary: dict[str, float]
    nl: int
    mesev: str
    site: Any
    soildyn: bool
    series: dict[str, np.ndarray]
    sw0: np.ndarray
    snow0: float
    mulch0: float
    mulch_evap0: float
    ksevap: float
    notes: dict[str, Any] = field(default_factory=dict)
    #: real days after the reference season (native inputs with ``extend_days``): the same series
    #: as :attr:`series` and the crop forcing of those days; :meth:`forcing` uses them before padding
    extension: dict[str, np.ndarray] = field(default_factory=dict)
    extension_crop: Any = None

    @property
    def n_ext(self) -> int:
        """Real days available after the reference season (0 without an extension)."""
        return 0 if self.extension_crop is None else int(np.asarray(self.extension_crop.yrdoy).shape[0])

    @property
    def days(self) -> np.ndarray:
        """The simulated days (``YYYYDDD``): planting to the reference maturity / harvest."""
        return np.asarray(self.forcing_crop.yrdoy, dtype=np.int64)

    @property
    def n_days(self) -> int:
        return int(self.days.shape[0])

    def published(self) -> dict[str, float]:
        """The cultivar coefficients the reference run read (``DSSAT48.INP``)."""
        c = self.params_crop.cultivar
        return {n: float(np.asarray(getattr(c, n.lower()))) for n in ("P1", "P2", "P5", "G2", "G3", "PHINT")}

    def params(self) -> dict[str, Any]:
        """The day's global params (:func:`agrijax.models.day_dssat486.day_params`)."""
        from agrijax.models.day_dssat486 import day_params

        return day_params(
            self.site, self.params_crop, ksevap=self.ksevap, ktrans=self.ksevap, soil_values=SOIL_VALUES
        )

    def forcing(self, n: int | None = None) -> dict[str, Any]:
        """The day's global forcing over ``n`` days (default the season): after the reference season
        the real days of :attr:`extension` (native inputs with ``extend_days``), then the last day
        repeated with rain and irrigation 0 (padding: a candidate cultivar that matures later is
        simulated on these days)."""
        from agrijax.models.day_dssat486 import dssat_soil_values
        from agrijax.processes.pet.spam_dssat import SpamWeather
        from agrijax.processes.soil_water.bucket import BucketForcing, BucketSoil, MulchForcing

        n = self.n_days if n is None else int(n)
        if n < self.n_days:
            raise ValueError(f"{self.key}: cannot pad a {self.n_days}-day season to {n} days")
        m = min(n - self.n_days, self.n_ext)
        s = self.series
        crop = self.forcing_crop
        if m:
            s = {k: np.concatenate([np.asarray(v), np.asarray(self.extension[k])[:m]]) for k, v in s.items()}
            crop = jax.tree.map(
                lambda a, b: np.concatenate([np.asarray(a), np.asarray(b)[:m]]), crop, self.extension_crop
            )
        n_real = self.n_days + m
        j = lambda x: jnp.asarray(_pad(np.asarray(x, dtype=np.float64), n))  # noqa: E731
        conv = lambda x: dssat_soil_values(x, SOIL_VALUES)  # noqa: E731
        if self.soildyn:
            soil = BucketSoil(**{k: j(conv(decimal_of(s[f"soil_{k}"]))) for k in (*SOILPROP, "cn", "swcon")})
            dlayr_end = j(conv(decimal_of(s["dlayr_end"])))
        else:
            st = self.site
            tile = lambda v: np.broadcast_to(conv(v), (n_real, *np.shape(v)))  # noqa: E731
            soil = BucketSoil(
                dlayr=j(tile(st.dlayr)),
                ds=j(tile(st.ds)),
                ll=j(tile(st.ll)),
                dul=j(tile(st.dul)),
                sat=j(tile(st.sat)),
                swcn=j(tile(st.swcn)),
                cn=j(tile(st.cn)),
                swcon=j(tile(st.swcon)),
            )
            dlayr_end = soil.dlayr
        mulch = MulchForcing(
            mass=j(s["m_mass"]), cover=j(s["m_cover"]), new_mass=j(s["m_new"]), watfac=j(s["m_watfac"])
        )
        after = np.arange(n) >= n_real
        zero_after = lambda x: jnp.where(jnp.asarray(after), 0.0, j(x))  # noqa: E731
        bf = BucketForcing(
            rain=zero_after(s["rain"]),
            tmax=j(s["tmax"]),
            irrigation=zero_after(s["irrigation"]),
            mulch=mulch,
            soil=soil,
            dlayr_end=dlayr_end,
        )
        weather = SpamWeather(
            tavg=j(s["tavg"]),
            wind_run=j(s["wind_run"]),
            co2=j(s["co2"]),
            srad=j(s["srad"]),
            tmax=j(s["tmax_w"]),
            tmin=j(s["tmin_w"]),
        )
        fc = jax.tree.map(lambda x: jnp.asarray(_pad(np.asarray(x), n)), crop)
        fc = fc.replace(
            sw=jnp.full_like(fc.sw, jnp.nan),
            eop=jnp.full_like(fc.eop, jnp.nan),
            trwup=jnp.full_like(fc.trwup, jnp.nan),
        )
        return {
            "crop": fc,
            "soil": bf,
            "spam": {"weather": weather, "mulch_am": j(s["mulch_am"]), "mulch_extfac": j(s["mulch_extfac"])},
        }

    def state(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """The morning state of the first day (:func:`agrijax.models.day_dssat486.initial_state`)."""
        from agrijax.models.day_dssat486 import dssat_soil_values, initial_state
        from agrijax.processes.crop.ceres_maize import CeresMaizeState
        from agrijax.processes.soil_water.bucket import BucketState
        from agrijax.processes.soil_water.bucket_evap import SoilEvapState
        from agrijax.processes.water_supply import RootwuState

        p = self.params() if params is None else params
        sw0 = jnp.asarray(dssat_soil_values(self.sw0, SOIL_VALUES))
        b = BucketState.initial(
            sw0, snow=self.snow0, mulch_wat=self.mulch0, mulch_evap_prev=self.mulch_evap0, dtype=jnp.float64
        )
        soil = p["soil"].soil
        se = SoilEvapState.initial(sw0, soil.dlayr, soil.ds, soil.dul, soil.ll, p["evap"].u)
        storage0 = jnp.sum(sw0 * soil.dlayr) + (self.snow0 + self.mulch0) / MM_PER_CM
        return initial_state(
            bucket=b,
            soil_evap=se,
            crop=CeresMaizeState.initial(p["crop"], 1),
            rootwu=RootwuState.initial(1, self.nl, jnp.float64),
            salb=p["albedo"].salb,
            storage0=storage0,
        )

    def simulate(
        self, cultivar: Mapping[str, float] | None = None, *, pad_days: int = 0
    ) -> dict[str, np.ndarray]:
        """Host-side: the free-run day over the season (plus ``pad_days`` padded days,
        :meth:`forcing`), the daily outputs of :func:`agrijax.models.day_dssat486.day_outputs` as
        NumPy arrays ``[days, ...]`` (crop outputs ``[days, 1]``, the soil water ``[days, layers]``).

        ``cultivar`` replaces ``MZCER048.CUL`` coefficients (``{"P1": ..., "G2": ...}``,
        :func:`with_cultivar`); the morning state is the published cultivar's, as in
        :class:`agrijax.calib.dssat_day.DaySimulator`. Needs ``jax_enable_x64`` (the day is validated
        in float64)."""
        from agrijax.core.runtime import run_sites

        if not jax.config.jax_enable_x64:
            raise RuntimeError("the DSSAT day runs in float64: jax.config.update('jax_enable_x64', True)")
        p0 = self.params()
        p = with_cultivar(p0, cultivar) if cultivar else p0
        n = self.n_days + int(pad_days)
        out = run_sites(  # the cached model: its compiled runner is reused for the same shapes
            _season_model(self.mesev),
            stack_trees([p]),
            stack_trees([self.forcing(n)]),
            stack_trees([self.state(p0)]),
        )
        return {k: np.asarray(v)[0] for k, v in out.items()}


#: the ``MZCER048.CUL`` coefficients a cultivar override may set, and their CERES-Maize field names
CULTIVAR_FIELDS: dict[str, str] = {
    "P1": "p1",
    "P2": "p2",
    "P5": "p5",
    "G2": "g2",
    "G3": "g3",
    "PHINT": "phint",
}


def with_cultivar(params: dict[str, Any], values: Mapping[str, float]) -> dict[str, Any]:
    """The day's params with the cultivar coefficients ``values`` (names of :data:`CULTIVAR_FIELDS`)."""
    import equinox as eqx

    bad = sorted(set(values) - set(CULTIVAR_FIELDS))
    if bad:
        raise ValueError(f"unknown cultivar coefficients {bad}; known: {list(CULTIVAR_FIELDS)}")
    c = params["crop"].cultivar
    names = [CULTIVAR_FIELDS[k] for k in values]
    new = tuple(
        jnp.asarray(v, dtype=getattr(c, f).dtype).reshape(getattr(c, f).shape)
        for f, v in zip(names, values.values(), strict=True)
    )
    cul = eqx.tree_at(lambda x: tuple(getattr(x, f) for f in names), c, new)
    return {**params, "crop": eqx.tree_at(lambda x: x.cultivar, params["crop"], cul)}


def _filex_cultivar(filex: Path, trno: int) -> str:
    from agrijax.io.dssat.filex import read_filex

    x = read_filex(filex)
    tr = next((t for t in x["TREATMENTS"] if int(t["N"]) == int(trno)), None)
    if tr is None:
        raise ValueError(f"{filex.name}: no treatment {trno}")
    return str(x["CULTIVARS"][tr["CU"]]["INGENO"]).strip()


class NativeInputError(NotImplementedError):
    """A treatment whose free-run inputs cannot be built natively: ``features`` lists the FileX
    features that need a DSSAT process Agri-JAX does not implement (see
    :func:`agrijax.io.dssat.native_management.unsupported_features`)."""

    def __init__(self, key: str, features: Sequence[str]) -> None:
        self.key = key
        self.features = list(features)
        super().__init__(
            f"{key}: needs a DSSAT process Agri-JAX does not implement: " + "; ".join(self.features)
        )


@dataclass
class _CropPart:
    """What both sources take from the reference run: CERES-Maize's parameters and forcing."""

    params: Any
    forcing: Any
    row: Any
    summary: dict[str, float]
    days: np.ndarray
    wth: Path
    inp: Path
    lat: float


def _crop_part(fx: Path, trno: int, out: Path, genotype: Path, weather_dir: Path) -> _CropPart:
    from agrijax.io.dssat import read_summary, read_wth
    from agrijax.sites.dssat_inputs import ceres_forcing, ceres_params

    inp = out / "DSSAT48.INP"
    p = ceres_params(str(inp), str(genotype / "MZCER048.ECO"), str(genotype / "MZCER048.SPE"), iswwat=True)
    summ = read_summary(out / "Summary.OUT")
    row = summ[summ["TRNO"] == int(trno)].iloc[0]
    text = inp.read_text(errors="replace").splitlines()
    wname = next(ln.split()[1] for ln in text if ln.startswith("WEATHERW"))
    wth = weather_dir / wname
    lat = float(read_wth(wth).attrs["site"]["LAT"])
    last = int(np.nanmax([row["MDAT"], row["HDAT"]]))
    # the crop's soil water, EOP and TRWUP come from the day's own modules (NaN in the forcing)
    f = ceres_forcing(
        str(out), int(trno), int(row["SDAT"]), last, int(p.soil.dlayr.shape[0]), lat, water=False
    )
    summary = {c: float(row[c]) for c in ("SDAT", "EDAT", "ADAT", "MDAT", "HDAT", "HWAM", "CWAM", "H#AM")}
    return _CropPart(p, f, row, summary, np.asarray(f.yrdoy, dtype=np.int64), wth, inp, lat)


def dssat_data_dir(root: str | os.PathLike[str]) -> Path:
    """The directory holding ``Genotype`` and ``StandardData`` under a DSSAT root: ``source/Data``
    (an engine source tree with its build, the validation layout), ``Data`` (the public data
    downloaded by :func:`agrijax.io.examples.fetch_dssat_data`) or ``bin`` (an installed engine)."""
    r = Path(root)
    for d in (r / "source" / "Data", r / "Data", r / "bin"):
        if (d / "Genotype").is_dir():
            return d
    raise FileNotFoundError(f"no Genotype directory under {r} (source/Data, Data or bin)")


def free_run_inputs(
    filex: str | os.PathLike[str],
    trno: int,
    ref_out: str | os.PathLike[str] | None,
    data_dir: str | os.PathLike[str] | None = None,
    *,
    engine: str | os.PathLike[str] | None = None,
    source: str = "tables",
    weather_dir: str | os.PathLike[str] | None = None,
    soil_dirs: Sequence[str | os.PathLike[str]] | None = None,
    genotype_dir: str | os.PathLike[str] | None = None,
    standard_dir: str | os.PathLike[str] | None = None,
    cul_file: str | os.PathLike[str] | None = None,
    extend_days: int = 0,
    window_days: int | None = None,
) -> FreeRunInputs:
    """The free-run inputs of treatment ``trno`` of ``filex`` from its reference run's outputs
    ``ref_out`` (:func:`run_reference`: CERES-Maize's parameters, the crop weather, the days).

    ``source="tables"`` takes the soil and soil-water inputs from the instrumented-engine tables
    under ``data_dir`` (module docstring); ``source="native"`` builds them from the public files
    (:func:`native_series`: the experiment's ``.SOL`` found in ``soil_dirs`` (default: the FileX
    directory, then the engine's ``example_data/Soil``), the weather files of ``weather_dir``
    (default the engine's ``example_data/Weather``), the engine's standard data and the ecotype's
    ``KCAN``) and raises :class:`NativeInputError` for a treatment that needs a DSSAT process not
    ported. ``genotype_dir`` holds the ``MZCER048.ECO`` / ``.SPE`` the reference run read (default
    the engine's ``Genotype``). ``extend_days`` (native only) also builds the real weather and
    management of that many days after the reference season (:attr:`FreeRunInputs.extension`: the
    crop weather at ``Weather.OUT``'s printed precision), fewer when the weather files end earlier;
    a later-maturing cultivar is then simulated on real days instead of padding.

    ``ref_out=None`` (native only) runs no DSSAT at all (:func:`dssat_free_inputs`): CERES-Maize's
    parameters come from the ``.CUL`` (``cul_file``, default ``<genotype_dir>/MZCER048.CUL``),
    ``.ECO``, ``.SPE`` and the FileX (:mod:`agrijax.io.dssat.native_crop`), the crop weather from the
    native weather chain, and the season from the model itself: the published cultivar is simulated
    over ``window_days`` from the simulation start and the season ends at its maturity (or at the
    reported harvest date). ``engine`` is then only the root of the DSSAT data (``example_data``
    and ``Genotype`` / ``StandardData``, :func:`dssat_data_dir`)."""
    from agrijax.port.run_fortran import DSSAT_ENGINE

    if source not in ("tables", "native"):
        raise ValueError(f"source must be 'tables' or 'native', got {source!r}")
    fx = Path(filex)
    exp = fx.stem
    key = treatment_key(exp, trno)
    eng = Path(engine) if engine is not None else DSSAT_ENGINE
    std = dssat_data_dir(eng) if genotype_dir is None or standard_dir is None else None
    genotype = Path(genotype_dir) if genotype_dir is not None else Path(str(std)) / "Genotype"
    stdd = Path(standard_dir) if standard_dir is not None else Path(str(std)) / "StandardData"
    wdir = Path(weather_dir) if weather_dir is not None else eng / "example_data" / "Weather"
    if ref_out is None:
        if source != "native":
            raise ValueError("inputs without a reference run (ref_out=None) need source='native'")
        sdirs = (
            [Path(d) for d in soil_dirs]
            if soil_dirs is not None
            else [fx.parent, eng / "example_data" / "Soil"]
        )
        return dssat_free_inputs(
            fx,
            trno,
            weather_dir=wdir,
            soil_dirs=sdirs,
            genotype_dir=genotype,
            standard_dir=stdd,
            cul_file=cul_file,
            extend_days=extend_days,
            window_days=window_days,
        )
    if source == "tables":
        if data_dir is None:
            raise ValueError("source='tables' needs data_dir")
        miss = missing_tables(exp, trno, data_dir)
        if miss:
            raise FileNotFoundError(f"{key}: free-run tables missing: {[str(p) for p in miss]}")
    out = Path(ref_out)
    crop = _crop_part(fx, trno, out, genotype, wdir)
    if source == "tables":
        assert data_dir is not None
        part = _table_series(key, crop, Path(data_dir))
    else:
        sdirs = (
            [Path(d) for d in soil_dirs]
            if soil_dirs is not None
            else [fx.parent, eng / "example_data" / "Soil"]
        )
        from agrijax.io.dssat.native_crop import last_weather_day

        eco = genotype / "MZCER048.ECO"
        ext = [_yrdoy_next(int(crop.days[-1]), i) for i in range(1, int(extend_days) + 1)]
        if ext:
            end = last_weather_day(crop.wth, int(crop.days[0]) // 100000)
            ext = [d for d in ext if d <= end]  # the weather files may end earlier
        days = np.concatenate([crop.days, np.asarray(ext, dtype=np.int64)])
        part = native_series(fx, trno, days, crop.inp, crop.wth, sdirs, stdd, eco)
        n = crop.days.shape[0]
        if ext:
            part["extension"] = {k: np.asarray(v)[n:] for k, v in part["series"].items()}
            part["series"] = {k: np.asarray(v)[:n] for k, v in part["series"].items()}
            part["extension_crop"] = _crop_extension(crop.forcing, part["extension"], ext, crop.lat)
        part["notes"]["extension_days"] = len(ext)
        if part["nl"] != int(crop.params.soil.dlayr.shape[0]):
            raise ValueError(
                f"{key}: {part['nl']} native soil layers, {crop.params.soil.dlayr.shape[0]} in DSSAT48.INP"
            )
    return FreeRunInputs(
        key=key,
        exp=exp,
        trno=int(trno),
        out=out,
        cultivar=_filex_cultivar(fx, trno),
        params_crop=crop.params,
        forcing_crop=crop.forcing,
        summary=crop.summary,
        **part,
    )


def _yrdoy_next(yrdoy: int, days: int) -> int:
    import datetime as _dt

    d = _dt.date(yrdoy // 1000, 1, 1) + _dt.timedelta(days=yrdoy % 1000 - 1 + days)
    return d.year * 1000 + d.timetuple().tm_yday


def _crop_extension(fc: Any, ext: dict[str, np.ndarray], days: Sequence[int], lat: float) -> Any:
    """The crop forcing of the extension days, as the reference run's ``Weather.OUT`` would print it
    (:func:`agrijax.sites.dssat_inputs.ceres_forcing`: the weather at one decimal, daylength and
    twilight daylength from the latitude; soil water, ``EOP``, ``TRWUP`` from the day's own modules)."""
    from agrijax.io.dssat.native_crop import crop_weather
    from agrijax.processes.crop.ceres_maize._util import daylength, twilight_daylength

    cw = crop_weather(ext["tmax_w"], ext["tmin_w"], ext["srad"], ext["co2"])
    doy = jnp.asarray(np.asarray(days, dtype=float) % 1000)
    m = len(days)
    nl = int(np.asarray(fc.sw).shape[-1])
    return fc.replace(
        yrdoy=jnp.asarray(np.asarray(days, dtype=np.int32)),
        tmax=jnp.asarray(cw["tmax"]),
        tmin=jnp.asarray(cw["tmin"]),
        srad=jnp.asarray(cw["srad"]),
        dayl=daylength(doy, lat),
        twilen=twilight_daylength(doy, lat),
        co2=jnp.asarray(cw["co2"]),
        snow=jnp.zeros(m),
        sw=jnp.full((m, nl), jnp.nan),
        eop=jnp.full(m, jnp.nan),
        trwup=jnp.full(m, jnp.nan),
    )


def _table_series(key: str, crop: _CropPart, data: Path) -> dict[str, Any]:
    """The soil / soil-water part of :class:`FreeRunInputs` from the instrumented-engine tables."""
    from agrijax.models.day_dssat486 import DssatSite

    days = crop.days
    si_dates, si = _table(data / SPAM_TABLES, f"{key}_spam_in")
    d = data / DSW_TABLES / key
    rate_dates, ri = _table(d, "wb_rate_in")
    _, ii = _table(d, "wb_integr_in")
    _, mri = _table(d, "mulch_rate_in")
    _, si0 = _table(d, "wb_seasinit_in")
    pmf = 0.0
    if (d / "rnoff_in.npz").is_file():
        pmf = float(_table(d, "rnoff_in")[1]["PMFRACTION"][0])
    pos = {int(x): i for i, x in enumerate(rate_dates.tolist())}
    spos = {int(x): i for i, x in enumerate(si_dates.tolist())}
    missing = [int(x) for x in days if int(x) not in pos or int(x) not in spos]
    if missing:
        raise ValueError(f"{key}: crop days without a WATBAL / SPAM table row: {missing[:5]}")
    at = np.asarray([pos[int(x)] for x in days])
    sat_ = np.asarray([spos[int(x)] for x in days])
    nl = int(ri["SP_NLAYR"][0])
    if nl != int(crop.params.soil.dlayr.shape[0]):
        raise ValueError(
            f"{key}: {nl} soil layers in the tables, {crop.params.soil.dlayr.shape[0]} in DSSAT48.INP"
        )

    def lay(tab: dict[str, np.ndarray], name: str, idx: Any = None) -> np.ndarray:
        v = np.asarray(tab[name], dtype=np.float64)[:, :nl]
        return v if idx is None else v[idx]

    def sca(tab: dict[str, np.ndarray], name: str, idx: Any = None) -> np.ndarray:
        v = np.asarray(tab[name], dtype=np.float64).reshape(-1)
        return v if idx is None else v[idx]

    src = si0 if "SP_DLAYR" in si0 else ri
    static = {k: lay(src, f"SP_{k.upper()}")[0] for k in SOILPROP}
    static_cn, static_swcon = float(sca(src, "SP_CN")[0]), float(sca(src, "SP_SWCON")[0])
    daily = {k: lay(ri, f"SP_{k.upper()}", at) for k in SOILPROP}
    daily["cn"] = sca(ri, "SP_CN", at)
    daily["swcon"] = sca(ri, "SP_SWCON", at)
    dlayr_end = lay(ii, "SP_DLAYR", at)
    changes = any(np.any(daily[k] != static[k][None, :]) for k in SOILPROP) or bool(
        np.any(dlayr_end != static["dlayr"][None, :])
    )
    changes |= bool(np.any(daily["cn"] != static_cn) or np.any(daily["swcon"] != static_swcon))
    mesev = _chr(si["MESEV"][0])
    meinf = _chr(ri["MEINF"][0])
    site = DssatSite(
        dlayr=decimal_of(static["dlayr"]),
        ds=decimal_of(static["ds"]),
        ll=decimal_of(static["ll"]),
        dul=decimal_of(static["dul"]),
        sat=decimal_of(static["sat"]),
        swcn=decimal_of(static["swcn"]),
        cn=float(decimal_of(static_cn)),
        swcon=float(decimal_of(static_swcon)),
        salb=float(decimal_of(si["SALB_S"][0])),
        u=float(decimal_of(si["U_S"][0])),
        mesev=mesev,
        meinf=meinf,
        actwtd=float(sca(ri, "ACTWTD")[0]),
        pmfraction=pmf,
    )
    series = {
        "rain": sca(ri, "RAIN", at),
        "tmax": sca(ri, "TMAX", at),
        "irrigation": sca(ri, "IRRAMT", at),
        "m_mass": sca(ri, "M_MASS", at),
        "m_cover": sca(ri, "M_COVER", at),
        "m_new": sca(ri, "M_NEW", at),
        "m_watfac": sca(ri, "M_WATFAC", at),
        "mulch_am": sca(si, "MULCH_AM", sat_),
        "mulch_extfac": sca(si, "MUL_EXTFAC", sat_),
        "tavg": sca(si, "TAVG_W", sat_),
        "wind_run": sca(si, "WINDSP_W", sat_),
        "co2": sca(si, "CO2_W", sat_),
        "srad": sca(si, "SRAD_W", sat_),
        "tmax_w": sca(si, "TMAX_W", sat_),
        "tmin_w": sca(si, "TMIN_W", sat_),
        "dlayr_end": dlayr_end,
        **{f"soil_{k}": v for k, v in daily.items()},
    }
    ks = np.unique(np.asarray(si["KSEVAP"], dtype=np.float64))
    kt = np.unique(np.asarray(si["KTRANS"], dtype=np.float64))
    if not (ks.size == 1 and kt.size == 1 and ks[0] == kt[0]):
        raise ValueError(f"{key}: KSEVAP {ks} and KTRANS {kt} differ or change during the season")
    return {
        "nl": nl,
        "mesev": mesev,
        "site": site,
        "soildyn": bool(changes),
        "series": series,
        "sw0": lay(ri, "SW", at)[0],
        "snow0": float(sca(ri, "SNOW")[at[0]]),
        "mulch0": float(sca(mri, "MULCHWAT")[at[0]]),
        "mulch_evap0": float(sca(mri, "MULCHEVAP")[at[0]]),
        "ksevap": float(ks[0]),
        "notes": {
            "source": "tables",
            "mesev": mesev,
            "meinf": meinf,
            "soildyn_replay": bool(changes),
            "first_day_is_simulation_start": int(days[0]) == int(rate_dates[0]),
        },
    }


def _sim_start(x: dict[str, Any], trno: int, first_weather: int) -> int:
    """``YRSIM`` of the treatment (the FileX ``SDATE`` through ``Y4K_DOY``; without a
    simulation-control level the planting date, ``IPSIM.for`` 117)."""
    from agrijax.io.dssat.native_management import treatment_levels
    from agrijax.io.dssat.observed import y4k_date

    tr = treatment_levels(x, trno)
    sm = int(tr.get("SM", 0) or 0)
    lev = (x.get("SIMULATION CONTROLS") or {}).get(sm) if sm > 0 else None
    if lev is not None:
        sdate = lev.get("GENERAL", {})["SDATE"]
    else:
        sdate = x["PLANTING DETAILS"][int(tr["MP"])]["PDATE"]
    d = y4k_date(int(sdate), first_weather=first_weather)
    return d.year * 1000 + d.timetuple().tm_yday


def native_series(
    filex: str | os.PathLike[str],
    trno: int,
    days: Any,
    inp: str | os.PathLike[str] | dict[str, Any],
    wth: str | os.PathLike[str],
    soil_dirs: Sequence[str | os.PathLike[str]],
    standard_data: str | os.PathLike[str],
    eco: str | os.PathLike[str],
) -> dict[str, Any]:
    """Host-side: the soil / soil-water part of :class:`FreeRunInputs` built from the public files
    (:mod:`agrijax.io.dssat.native_soil`, :mod:`~agrijax.io.dssat.native_weather`,
    :mod:`~agrijax.io.dssat.native_management`) for the simulated ``days`` (the first the
    simulation start): the static ``SOILPROP`` and initial water of the ``.SOL`` / FileX, the
    daily weather record of the ``.WTH`` ``wth`` (``TAVG`` of ``HMET``, the 2 m wind, ``CO2VAL``
    with ``standard_data/CO2048.WDA``), the reported irrigation, the residue parameters of
    ``standard_data/RESCH048.SDA``, ``KEP`` of the ecotype file ``eco``, no mulch, snow or water
    table. The soil is checked against the reference run's ``DSSAT48.INP`` ``inp`` (the layers the
    crop reads). Raises :class:`NativeInputError` for a treatment that needs a DSSAT process not
    ported (:func:`agrijax.io.dssat.native_management.unsupported_features`)."""
    from agrijax.io.dssat import read_eco, read_filex
    from agrijax.io.dssat.native_management import (
        extinction_kep,
        irrigation_amounts,
        residue_parameters,
        switches,
        treatment_levels,
        unsupported_features,
    )
    from agrijax.io.dssat.native_soil import find_soil_profile, initial_soil_water, native_soil
    from agrijax.io.dssat.native_weather import native_weather
    from agrijax.models.day_dssat486 import DssatSite
    from agrijax.sites.dssat_inputs import read_inp

    fx = Path(filex)
    key = treatment_key(fx.stem, trno)
    x = read_filex(fx)
    bad = unsupported_features(x, trno)
    if bad:
        raise NativeInputError(key, bad)
    dd = np.asarray(days, dtype=np.int64)
    start = _sim_start(x, trno, int(dd[0]))
    if start != int(days[0]):
        raise NativeInputError(
            key,
            [f"the crop days start on {int(days[0])}, not on the simulation start {start} (fallow spin-up)"],
        )
    sw = switches(x, trno)
    tr = treatment_levels(x, trno)
    fl = x["FIELDS"][int(tr["FL"])]
    prof = find_soil_profile(str(fl["ID_SOIL"]).strip(), [Path(d) for d in soil_dirs])
    soil = native_soil(prof, sw.mesol)
    ref = inp if isinstance(inp, dict) else read_inp(str(inp))
    for name, ours in (("ds", soil.ds), ("ll", soil.ll), ("dul", soil.dul), ("sat", soil.sat)):
        theirs = np.asarray(ref[name], dtype=np.float64)
        if theirs.shape != ours.shape or not np.array_equal(theirs, ours):
            raise ValueError(
                f"{key}: native soil {name} {ours.tolist()} differs from DSSAT48.INP {theirs.tolist()}"
            )
    ic_lev = int(tr.get("IC", 0) or 0)
    ic = x.get("INITIAL CONDITIONS", {}).get(ic_lev, {}) if ic_lev else {}
    sw0 = initial_soil_water(soil, ic.get("rows"))
    std = Path(standard_data)
    w = native_weather(Path(wth), dd.tolist(), co2_option=sw.co2, co2_file=std / "CO2048.WDA")
    irr = irrigation_amounts(x, trno, dd.tolist())
    res = residue_parameters(str(ic.get("PCR", "") or ""), std / "RESCH048.SDA")
    cul = ref["cultivar"]
    assert isinstance(cul, dict)
    kep = extinction_kep(float(read_eco(Path(eco)).loc[str(cul["ECO"]).strip()]["KCAN"]))
    site = DssatSite(
        dlayr=soil.dlayr,
        ds=soil.ds,
        ll=soil.ll,
        dul=soil.dul,
        sat=soil.sat,
        swcn=soil.swcn,
        cn=soil.cn,
        swcon=soil.swcon,
        salb=soil.salb,
        u=soil.u,
        mesev=sw.mesev,
        meinf=sw.infil,
        actwtd=1000.0,
        pmfraction=0.0,
    )
    n = int(dd.shape[0])
    f64 = lambda v: np.asarray(v, dtype=np.float32).astype(np.float64)  # noqa: E731
    zero = np.zeros(n)
    series = {
        "rain": f64(w.rain),
        "tmax": f64(w.tmax),
        "irrigation": f64(irr),
        "m_mass": zero,
        "m_cover": zero,
        "m_new": zero,
        "m_watfac": np.full(n, res.watfac),
        "mulch_am": np.full(n, res.am),
        "mulch_extfac": np.full(n, res.extfac),
        "tavg": f64(w.tavg),
        "wind_run": f64(w.windsp),
        "co2": f64(w.co2),
        "srad": f64(w.srad),
        "tmax_w": f64(w.tmax),
        "tmin_w": f64(w.tmin),
        "dlayr_end": np.broadcast_to(f64(np.asarray(soil.dlayr, np.float32)), (n, soil.nl)).copy(),
    }
    return {
        "nl": soil.nl,
        "mesev": sw.mesev,
        "site": site,
        "soildyn": False,
        "series": series,
        "sw0": f64(np.asarray(sw0, np.float32)),
        "snow0": 0.0,
        "mulch0": 0.0,
        "mulch_evap0": 0.0,
        "ksevap": kep,
        "notes": {
            "source": "native",
            "mesev": sw.mesev,
            "meinf": sw.infil,
            "soildyn_replay": False,
            "first_day_is_simulation_start": True,
            "soil_profile": prof.id,
            "weather_files": list(w.files),
            "residue_type": res.restype,
        },
    }


#: days simulated from the simulation start to find the season's end without DSSAT (the longest
#: example season is 207 days; the extension follows it)
SEASON_WINDOW = 240
#: harvest options the DSSAT-free inputs support: at maturity, on the reported date
_HARVS = ("M", "R")


@functools.lru_cache(maxsize=4)
def _season_model(mesev: str) -> Any:
    """The day with its daily outputs for ``mesev`` (one per soil evaporation, so its compiled runners
    are reused from season to season)."""
    from agrijax.models.day_dssat486 import SLOT, day_dssat486, day_outputs, day_processes

    return day_dssat486(SLOT).compile(
        day_processes(SLOT, mesev=mesev), outputs=day_outputs(SLOT), exact_lags=True
    )


def native_inp(
    x: dict[str, Any],
    trno: int,
    soil_dirs: Sequence[str | os.PathLike[str]],
    cul: str | os.PathLike[str],
    first_weather: int | None = None,
) -> dict[str, Any]:
    """Host-side: what :func:`agrijax.sites.dssat_inputs.read_inp` reads from a ``DSSAT48.INP``,
    rebuilt from the FileX ``x`` (:func:`agrijax.io.dssat.read_filex`), the ``.SOL`` in
    ``soil_dirs`` and the ``.CUL`` file ``cul`` (:mod:`agrijax.io.dssat.native_crop`)."""
    from agrijax.io.dssat.native_crop import native_cultivar, native_planting, native_soil_crop
    from agrijax.io.dssat.native_management import switches, treatment_levels
    from agrijax.io.dssat.native_soil import find_soil_profile, native_soil

    tr = treatment_levels(x, trno)
    fl = x["FIELDS"][int(tr["FL"])]
    prof = find_soil_profile(str(fl["ID_SOIL"]).strip(), [Path(d) for d in soil_dirs])
    soil = native_soil(prof, switches(x, trno).mesol)
    return {
        "cultivar": native_cultivar(Path(cul), str(x["CULTIVARS"][tr["CU"]]["INGENO"]).strip()),
        **native_planting(x, trno, first_weather),
        **native_soil_crop(prof, soil),
    }


def dssat_free_inputs(
    filex: str | os.PathLike[str],
    trno: int,
    *,
    weather_dir: str | os.PathLike[str],
    soil_dirs: Sequence[str | os.PathLike[str]],
    genotype_dir: str | os.PathLike[str],
    standard_dir: str | os.PathLike[str],
    cul_file: str | os.PathLike[str] | None = None,
    extend_days: int = 0,
    window_days: int | None = None,
) -> FreeRunInputs:
    """Host-side: the free-run inputs of treatment ``trno`` of ``filex`` without running DSSAT
    (:func:`free_run_inputs` with ``ref_out=None``).

    The FileX gives the simulation start (:func:`agrijax.io.dssat.native_crop.season_start`), the
    weather file (``<WSTA><YY>01.WTH`` in ``weather_dir``) and, with the ``.CUL`` / ``.ECO`` /
    ``.SPE`` of ``genotype_dir`` and the ``.SOL`` of ``soil_dirs``, CERES-Maize's parameters
    (:func:`native_inp`); the soil and soil-water inputs are :func:`native_series`, the crop weather
    their record at ``Weather.OUT``'s precision. The published cultivar is then simulated over
    ``window_days`` (default :data:`SEASON_WINDOW`) from the simulation start; the season ends at
    its maturity (``HARVS = M``) or at the reported harvest date (``R``), the days after it
    (up to ``extend_days``) become :attr:`FreeRunInputs.extension`. :attr:`FreeRunInputs.summary`
    holds that simulation's dates and values (``SDAT``, ``EDAT``, ``ADAT``, ``MDAT``, ``HDAT``,
    ``HWAM``, ``CWAM``, ``H#AM``). Needs ``jax_enable_x64``."""
    from agrijax.io.dssat import read_filex, read_wth
    from agrijax.io.dssat.native_crop import crop_weather, season_start, weather_file_name
    from agrijax.io.dssat.native_management import treatment_levels, unsupported_features
    from agrijax.io.dssat.observed import y4k_date
    from agrijax.processes.crop.ceres_maize._util import daylength, twilight_daylength
    from agrijax.sites.dssat_inputs import CeresReplayForcing, ceres_params_from, yrdoy_range

    if not jax.config.jax_enable_x64:
        raise RuntimeError("the DSSAT day runs in float64: jax.config.update('jax_enable_x64', True)")
    fx = Path(filex)
    key = treatment_key(fx.stem, trno)
    x = read_filex(fx)
    bad = list(unsupported_features(x, trno))
    tr = treatment_levels(x, trno)
    sc = x.get("SIMULATION CONTROLS", {})
    ma = sc.get(int(tr.get("SM", 0) or 0), sc.get(1, {})).get("MANAGEMENT", {})
    harvs = str(ma.get("HARVS", "M") or "M").strip().upper()[:1] or "M"
    if harvs not in _HARVS:
        bad.append(
            f"HARVS = {harvs}: harvest options other than at maturity (M) or reported (R) not implemented"
        )
    if bad:
        raise NativeInputError(key, bad)
    fl = x["FIELDS"][int(tr["FL"])]
    wdir = Path(weather_dir)
    year0 = season_start(x, trno) // 1000  # the cross-over rule, before the weather file is known
    wth = wdir / weather_file_name(str(fl["WSTA"]), year0)
    if not wth.is_file():
        raise FileNotFoundError(f"{key}: weather file {wth} not found")
    wdf = read_wth(wth, dssat_spans=True, century=year0 // 100)
    first = int(wdf["date"].iloc[0].strftime("%Y%j"))
    start = season_start(x, trno, first)
    lat = float(wdf.attrs["site"]["LAT"])
    genotype = Path(genotype_dir)
    cul = Path(cul_file) if cul_file is not None else genotype / "MZCER048.CUL"
    inp = native_inp(x, trno, soil_dirs, cul, first)
    params = ceres_params_from(
        inp, str(genotype / "MZCER048.ECO"), str(genotype / "MZCER048.SPE"), iswwat=True
    )
    from agrijax.io.dssat.native_crop import last_weather_day

    window = int(window_days or SEASON_WINDOW) + int(extend_days)
    eco = genotype / "MZCER048.ECO"
    last = min(_yrdoy_next(start, window - 1), last_weather_day(wth, year0 // 100))
    days = np.asarray(yrdoy_range(start, last), dtype=np.int64)
    part = native_series(fx, trno, days, inp, wth, soil_dirs, standard_dir, eco)
    n_win = days.shape[0]
    nl = int(part["nl"])
    cw = crop_weather(
        part["series"]["tmax_w"], part["series"]["tmin_w"], part["series"]["srad"], part["series"]["co2"]
    )
    doy = jnp.asarray(np.asarray(days % 1000, dtype=float))
    fc = CeresReplayForcing(
        yrdoy=jnp.asarray(days.astype(np.int32)),
        tmax=jnp.asarray(cw["tmax"]),
        tmin=jnp.asarray(cw["tmin"]),
        srad=jnp.asarray(cw["srad"]),
        dayl=daylength(doy, lat),
        twilen=twilight_daylength(doy, lat),
        co2=jnp.asarray(cw["co2"]),
        snow=jnp.zeros(n_win),
        sw=jnp.full((n_win, nl), jnp.nan),
        eop=jnp.full(n_win, jnp.nan),
        trwup=jnp.full(n_win, jnp.nan),
    )
    full = FreeRunInputs(
        key=key,
        exp=fx.stem,
        trno=int(trno),
        out=None,
        cultivar=_filex_cultivar(fx, trno),
        params_crop=params,
        forcing_crop=fc,
        summary={},
        **part,
    )
    # the season: the published cultivar simulated over the window, up to its maturity / harvest
    p0 = full.params()
    o = run_sites_one(full, p0)
    ist = np.asarray(o["istage"])[:, 0]

    def first_of(code: int) -> int | None:
        hit = np.nonzero(ist == code)[0]
        return int(hit[0]) if hit.size else None

    mat = first_of(_ISTAGE_MATURITY)
    if harvs == "R":
        mh = int(tr.get("MH", 0) or 0)
        rows = x.get("HARVEST DETAILS", {}).get(mh, {}).get("rows") or [
            x.get("HARVEST DETAILS", {}).get(mh, {})
        ]
        hd = y4k_date(int(float(rows[0]["HDATE"])), first_weather=first)
        hdat = hd.year * 1000 + hd.timetuple().tm_yday
        pos = np.nonzero(days == hdat)[0]
        if not pos.size:
            raise NativeInputError(key, [f"reported harvest date {hdat} outside {window} days of the start"])
        end = int(pos[0])
    else:
        if mat is None:
            raise NativeInputError(
                key, [f"the crop does not reach maturity within {n_win} days of the start"]
            )
        end = mat
    n = end + 1
    extend_days = min(int(extend_days), n_win - n)
    cut = lambda a: np.asarray(a)[:n]  # noqa: E731
    tail = lambda a: np.asarray(a)[n : n + int(extend_days)]  # noqa: E731
    series = {k: cut(v) for k, v in part["series"].items()}
    ext = {k: tail(v) for k, v in part["series"].items()} if extend_days else {}
    fc_season = jax.tree.map(lambda a: jnp.asarray(cut(a)), fc)
    fc_ext = jax.tree.map(lambda a: jnp.asarray(tail(a)), fc) if extend_days else None
    pos_of = lambda code: first_of(code)  # noqa: E731
    at = lambda i: float(days[i]) if i is not None and i < n else float("nan")  # noqa: E731
    summary = {
        "SDAT": float(days[0]),
        "EDAT": at(pos_of(_ISTAGE_EMERGENCE)),
        "ADAT": at(pos_of(_ISTAGE_SILKING)),
        "MDAT": at(mat),
        "HDAT": float(days[end]),
        "HWAM": float(np.asarray(o["gwad"])[end, 0]),
        "CWAM": float(np.asarray(o["cwad"])[end, 0]),
        "H#AM": float(np.asarray(o["g_ad"])[end, 0]),
    }
    notes = dict(part["notes"])
    notes.update(dssat_free=True, extension_days=int(extend_days), season_window=n_win, harvest=harvs)
    return FreeRunInputs(
        key=key,
        exp=fx.stem,
        trno=int(trno),
        out=None,
        cultivar=full.cultivar,
        params_crop=params,
        forcing_crop=fc_season,
        summary=summary,
        nl=nl,
        mesev=part["mesev"],
        site=part["site"],
        soildyn=part["soildyn"],
        series=series,
        sw0=part["sw0"],
        snow0=part["snow0"],
        mulch0=part["mulch0"],
        mulch_evap0=part["mulch_evap0"],
        ksevap=part["ksevap"],
        notes=notes,
        extension=ext,
        extension_crop=fc_ext,
    )


#: CERES-Maize stage codes (``ISTAGE`` output): first day of emergence, silking, maturity
_ISTAGE_EMERGENCE, _ISTAGE_SILKING, _ISTAGE_MATURITY = 1, 4, 10


def run_sites_one(x: FreeRunInputs, params: dict[str, Any]) -> dict[str, np.ndarray]:
    """Host-side: the day over all of ``x``'s days with ``params`` (the published cultivar's morning
    state), daily outputs ``[days, ...]`` (one compiled program per soil evaporation and shape)."""
    from agrijax.core.runtime import run_sites

    out = run_sites(
        _season_model(x.mesev),
        stack_trees([params]),
        stack_trees([x.forcing(x.n_days)]),
        stack_trees([x.state(x.params())]),
    )
    return {k: np.asarray(v)[0] for k, v in out.items()}


def stack_trees(trees: Sequence[Any]) -> Any:
    """Leaves of ``trees`` stacked on a new leading axis (NumPy)."""
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *trees)
