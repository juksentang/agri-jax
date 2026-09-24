"""CERES-Maize against DSSAT-CSM ``dscsm048`` (milestone M2), crop isolated from the soil water.

The reference model runs with nitrogen off (``NITRO = N`` in the FileX options) through
:func:`agrijax.port.run_fortran.run_dscsm` (batch mode, one treatment per run so the
``DSSAT48.INP`` holds that treatment's cultivar, planting and soil). The crop is driven by the
reference run itself: weather and CO2 from ``Weather.OUT``, soil water from ``SoilWat.OUT`` and
the water-stress factors ``SWFAC = 1 - WSPD``, ``TURFAC = 1 - WSGD`` from ``PlantGro.OUT``; so only
the crop (phenology, growth, roots and the saturation factor) is compared.

M2: daily LAI and above-ground biomass within 1 % where the printed value is at least 100 print
units (LAI >= 1.00, CWAD >= 100 kg/ha; below that the print rounding alone exceeds 0.5 %), within
one print unit on every day, grain yield within 0.5 %, emergence / silking / maturity dates equal
and the growth stage and leaf number equal on every day.

With the private instrumented build (``<data-dir>/dssat_dump/engine``: dscsm048 compiled from the
BSD-3 source with one extra ``WRITE`` of the crop's inputs and states in ``MZ_CERES``; the patch
and the binary live only in the data directory) the drivers come at full REAL precision and the
states agree to float32 accuracy (relative 1e-4 or better).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core import run
from agrijax.io.dssat import read_out, read_plantgro, read_summary, read_wth
from agrijax.processes.crop.ceres_maize import CeresMaizeState, ceres_maize_model
from agrijax.processes.crop.ceres_maize.dssat_inputs import ceres_forcing, ceres_params, yrdoy_range

DSSAT_ENGINE = Path(
    os.environ.get("AGRI_JAX_DSSAT", "~/AFSoil/Formal_Analysis/02_DSSAT/dssat_engine")
).expanduser()

pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="the reference comparison runs in float64"
)

MAIZE = DSSAT_ENGINE / "example_data" / "Maize"
WEATHER = DSSAT_ENGINE / "example_data" / "Weather"
GENOTYPE = DSSAT_ENGINE / "bin" / "Genotype"
DSCSM = DSSAT_ENGINE / "bin" / "dscsm048"

#: PlantGro column -> (model output, print unit)
COLUMNS = {
    "LAID": ("lai", 0.01),
    "CWAD": ("cwad", 1.0),
    "LWAD": ("lwad", 1.0),
    "SWAD": ("swad", 1.0),
    "GWAD": ("gwad", 1.0),
    "RWAD": ("rwad", 1.0),
    "G#AD": ("g_ad", 1.0),
    "RDPD": ("rdpd", 0.01),
    "EWSD": ("ewsd", 0.001),
    "GSTD": ("gstd", 1.0),
    "L#SD": ("lsd", 1.0),
}

_RUNNER = jax.jit(lambda p, f, s: run(ceres_maize_model(), p, f, s))


def _nitrogen_off(text: str, water: str | None = None) -> str:
    """FileX with ``NITRO = N`` (and optionally ``WATER``) and daily growth / water outputs.

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
            ln = ln[:k] + "N" + ln[k + 1 :]
            if water is not None:
                k = opt.index("WATER") + 4
                ln = ln[:k] + water + ln[k + 1 :]
            opt = ""
        elif outp and ln.split()[1:2] == ["OU"]:
            for name, val in (("FROPT", "1"), ("GROUT", "Y"), ("WAOUT", "Y")):
                k = outp.index(name) + 4
                ln = ln[:k] + val + ln[k + 1 :]
            outp = ""
        out.append(ln)
    return "\n".join(out) + "\n"


def run_reference(
    exp: str, trno: int, dest: Path, *, water: str | None = None, engine: Path | None = None
) -> Path:
    """Stage ``<exp>.MZX`` (nitrogen off) with a one-treatment batch file and run dscsm048."""
    from agrijax.port.run_fortran import run_dscsm

    dest.mkdir(parents=True, exist_ok=True)
    for f in MAIZE.glob(exp + ".MZ*"):
        shutil.copy2(f, dest / f.name)
    x = dest / f"{exp}.MZX"
    x.write_text(_nitrogen_off(x.read_text(errors="replace"), water))
    batch = "$BATCH(MAIZE)\n!\n@FILEX" + " " * 88 + "TRTNO     RP     SQ     OP     CO\n"
    batch += f"{exp}.MZX".ljust(92) + f"{trno:7d}      1      0      0      0\n"
    (dest / "DSSBatch.v48").write_text(batch)
    out = dest / "out"
    run_dscsm(
        dest,
        out,
        run_mode="B",
        experiment_file="DSSBatch.v48",
        extra_files=sorted(WEATHER.glob(exp[:4] + "*.WTH")),
        keep_files=("*.OUT", "DSSAT48.INP"),
        engine=engine,
    )
    return out


def simulate(out: Path, trno: int, *, water: bool = True, daylength_from_output: bool = False):
    """Parameters from the run's INP / ECO / SPE, forcing from its outputs; run the model."""
    inp = out / "DSSAT48.INP"
    p = ceres_params(str(inp), str(GENOTYPE / "MZCER048.ECO"), str(GENOTYPE / "MZCER048.SPE"), iswwat=water)
    summ = read_summary(out / "Summary.OUT")
    row = summ[summ["TRNO"] == trno].iloc[0]
    wname = next(ln.split()[1] for ln in inp.read_text().splitlines() if ln.startswith("WEATHERW"))
    lat = float(read_wth(WEATHER / wname).attrs["site"]["LAT"])
    last = int(np.nanmax([row["MDAT"], row["HDAT"]]))
    f = ceres_forcing(
        str(out),
        trno,
        int(row["SDAT"]),
        last,
        int(p.soil.dlayr.shape[0]),
        lat,
        water=water,
        daylength_from_output=daylength_from_output,
    )
    res = _RUNNER(p, f, CeresMaizeState.initial(p, 1))
    return p, f, {k: np.asarray(v) for k, v in res.items()}, row


def daily_errors(out: Path, trno: int, f, res) -> dict[str, dict[str, float]]:
    """Per PlantGro column: max |sim - ref| in print units, and max relative error on days with
    ``ref >= 100`` print units."""
    pg = read_plantgro(out / "PlantGro.OUT")
    pg = pg[pg["TRNO"] == trno]
    days = np.asarray(f.yrdoy)
    key = np.asarray(pg["YEAR"], dtype=int) * 1000 + np.asarray(pg["DOY"], dtype=int)
    idx = np.searchsorted(days, key)
    assert np.array_equal(days[idx], key)
    errs = {}
    for col, (name, unit) in COLUMNS.items():
        sim = res[name][idx, 0]
        ref = np.asarray(pg[col], dtype=float)
        d = np.abs(sim - ref)
        big = ref >= 100 * unit
        errs[col] = {
            "max_units": float(d.max() / unit),
            "max_rel_big": float((d[big] / ref[big]).max()) if big.any() else 0.0,
            "n_days": len(ref),
        }
    return errs


def check_m2(out: Path, trno: int, p, f, res, row, *, units: float = 1.0):
    errs = daily_errors(out, trno, f, res)
    # M2: LAI and above-ground biomass, 1 % where printed values carry 3 significant digits
    for col in ("LAID", "CWAD"):
        assert errs[col]["max_rel_big"] < 0.01, (col, errs[col])
    # every day within the print unit (plus the rounding of the printed drivers)
    for col in ("LAID", "CWAD", "LWAD", "SWAD", "GWAD", "RWAD", "G#AD"):
        assert errs[col]["max_units"] <= units, (col, errs[col])
    assert errs["GSTD"]["max_units"] == 0.0 and errs["L#SD"]["max_units"] == 0.0, errs
    # yield
    yld = float(res["gwad"][-1, 0])
    if float(row["HWAM"]) > 0:
        assert abs(yld - float(row["HWAM"])) / float(row["HWAM"]) < 0.005, (yld, row["HWAM"])
    # phenology dates: the day the stage code first changes
    days = np.asarray(f.yrdoy)
    stage = res["istage"][:, 0]

    def first(code: int) -> int:
        hit = np.nonzero(stage == code)[0]
        return int(days[hit[0]]) if hit.size else -99

    assert first(1) == int(row["EDAT"]), ("emergence", first(1), row["EDAT"])
    if float(row["ADAT"]) > 0:
        assert first(4) == int(row["ADAT"]), ("silking", first(4), row["ADAT"])
    if float(row["MDAT"]) > 0:
        assert first(10) == int(row["MDAT"]), ("maturity", first(10), row["MDAT"])
    return errs


# ------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def ref_dir(tmp_path_factory: pytest.TempPathFactory):
    if not DSCSM.is_file() or not MAIZE.is_dir():
        pytest.skip(f"dscsm048 / DSSAT example data not found under {DSSAT_ENGINE}")
    cache: dict[tuple, Path] = {}

    def get(exp: str, trno: int, water: str | None = None) -> Path:
        k = (exp, trno, water)
        if k not in cache:
            cache[k] = run_reference(exp, trno, tmp_path_factory.mktemp(f"{exp}_{trno}_{water}"), water=water)
        return cache[k]

    return get


# ------------------------------------------------------------------------------ M2 on UFGA8201
@pytest.mark.parametrize("trno", [1, 3, 5])
def test_ufga8201_m2(ref_dir, trno):
    """Rainfed (1), irrigated (3) and vegetative-stress (5) treatments, nitrogen off."""
    out = ref_dir("UFGA8201", trno)
    p, f, res, row = simulate(out, trno)
    errs = check_m2(out, trno, p, f, res, row)
    assert errs["RDPD"]["max_units"] <= 0.51 and errs["EWSD"]["max_units"] == 0.0, errs


def test_ufga8201_without_water_balance(ref_dir):
    """``WATER = N``: no stress, no roots (``MZ_ROOTGR`` is not called); equals the irrigated run."""
    out = ref_dir("UFGA8201", 3, water="N")
    assert not (out / "SoilWat.OUT").exists()
    p, f, res, row = simulate(out, 3, water=False)
    check_m2(out, 3, p, f, res, row)
    assert np.all(res["rdpd"] == 0.0) and np.all(res["rlv"] == 0.0)


def test_ufga8201_forcing_chain(ref_dir):
    """The weather the forcing takes from Weather.OUT is the .WTH weather; DAYLEN / TWILIGHT
    reproduce the printed DAYLD / TWLD (0.1 h); the INP cultivar is the CUL rounded to its F6.2."""
    out = ref_dir("UFGA8201", 1)
    _, f, _, _ = simulate(out, 1)
    wth = read_wth(WEATHER / "UFGA8201.WTH")
    wkey = {d.year * 1000 + d.timetuple().tm_yday: i for i, d in enumerate(wth["date"])}
    rows = [wkey[int(d)] for d in np.asarray(f.yrdoy)]
    np.testing.assert_array_equal(np.asarray(f.tmax), wth["tmax"].to_numpy(float)[rows])
    np.testing.assert_array_equal(np.asarray(f.tmin), wth["tmin"].to_numpy(float)[rows])
    np.testing.assert_array_equal(np.asarray(f.srad), wth["srad"].to_numpy(float)[rows])
    wo = read_out(out / "Weather.OUT")
    np.testing.assert_allclose(np.asarray(f.dayl), wo["DAYLD"].to_numpy(float), atol=0.051)
    np.testing.assert_allclose(np.asarray(f.twilen), wo["TWLD"].to_numpy(float), atol=0.051)
    p = ceres_params(str(out / "DSSAT48.INP"), str(GENOTYPE / "MZCER048.ECO"), str(GENOTYPE / "MZCER048.SPE"))
    assert float(p.cultivar.g3) == 8.17  # CUL 8.168 written with F6.2 into the INP
    assert float(p.soil.slpf) == 0.92 and int(p.soil.dlayr.shape[0]) == 9


# ------------------------------------------------------------------------------ a second site
@pytest.mark.parametrize("trno", [1, 3])
def test_iuaf9901_m2(ref_dir, trno):
    """Ames, Iowa 1999 (42 N): 4.7 and 7.5 plants/m2, cold spring, waterlogging days (SATFAC > 0)."""
    out = ref_dir("IUAF9901", trno)
    p, f, res, row = simulate(out, trno)
    errs = check_m2(out, trno, p, f, res, row)
    assert np.max(res["ewsd"]) > 0.0
    # SATFAC follows the printed SW (3 decimals) through (SAT - SW) / PORMIN: 1 % of the factor
    assert errs["EWSD"]["max_units"] <= 6.0, errs


@pytest.mark.parametrize("trno", [2, 5])
def test_gagr0201_environment_modification(ref_dir, trno):
    """Growth chamber: daylength replaced by 16 h, 35/25 or 25/15 degC, CO2 400 / 800 ppm."""
    out = ref_dir("GAGR0201", trno)
    _, f, res, row = simulate(out, trno, daylength_from_output=True)
    # the modification starts on the planting day; the simulation starts a month earlier
    grown = np.asarray(f.yrdoy) >= int(row["PDAT"])
    assert set(np.unique(np.asarray(f.dayl)[grown]).tolist()) == {16.0}
    assert set(np.unique(np.asarray(f.co2)[grown]).tolist()) == ({400.0} if trno == 2 else {800.0})
    errs = daily_errors(out, trno, f, res)
    for col in ("LAID", "CWAD", "LWAD", "SWAD", "RWAD"):
        assert errs[col]["max_units"] <= 1.0 and errs[col]["max_rel_big"] < 0.01, (col, errs[col])
    assert errs["GSTD"]["max_units"] == 0.0 and errs["L#SD"]["max_units"] == 0.0


@pytest.mark.slow
def test_every_maize_example_experiment(ref_dir, tmp_path_factory):
    """All maize example experiments that run without pests / missing weather, every treatment."""
    skip = {"EBPL8501", "IUAF9902", "IUAF9903"}  # missing EBCH8401.WTH; pest damage (not ported)
    n = 0
    for x in sorted(MAIZE.glob("*.MZX")):
        if x.stem in skip:
            continue
        sec = x.read_text(errors="replace").split("*TREATMENTS")[1].split("*")[0]
        trts = [int(ln[:3]) for ln in sec.splitlines() if ln[:3].strip().isdigit()]
        for t in trts:
            out = ref_dir(x.stem, t)
            _, f, res, row = simulate(out, t, daylength_from_output=x.stem == "GAGR0201")
            errs = daily_errors(out, t, f, res)
            for col in ("LAID", "CWAD"):
                assert errs[col]["max_rel_big"] < 0.01, (x.stem, t, col, errs[col])
            for col in ("LAID", "CWAD", "LWAD", "SWAD", "GWAD", "RWAD"):
                assert errs[col]["max_units"] <= 2.5, (x.stem, t, col, errs[col])
            assert errs["GSTD"]["max_units"] == 0.0 and errs["L#SD"]["max_units"] == 0.0, (x.stem, t)
            if float(row["HWAM"]) > 0:
                assert abs(float(res["gwad"][-1, 0]) - float(row["HWAM"])) / float(row["HWAM"]) < 0.005
            n += 1
    assert n >= 50


# ------------------------------------------------------------------------------ full precision (private build)
DUMP_NAMES = [
    "RUN",
    "YRDOY",
    "ISTAGE",
    "SWFAC",
    "TURFAC",
    "SATFAC",
    "EOP",
    "TRWUP",
    "DTT",
    "SUMDTT",
    "CUMDTT",
    "XLAI",
    "WTLF",
    "STMWTO",
    "SDWT",
    "RTWTO",
    "TOPWT",
    "GPP",
    "EARS",
    "RTDEP",
]
DUMP_TO_OUTPUT = {
    "XLAI": ("lai", 1.0),
    "WTLF": ("lwad", 0.1),
    "STMWTO": ("swad", 0.1),
    "SDWT": ("gwad", 0.1),
    "RTWTO": ("rwad", 0.1),
    "TOPWT": ("cwad", 0.1),
    "RTDEP": ("rdpd", 100.0),
    "SUMDTT": ("sumdtt", 1.0),
    "DTT": ("dttd", 1.0),
}


@pytest.fixture(scope="module")
def dump_engine(data_dir: Path) -> Path:
    eng = data_dir / "dssat_dump" / "engine"
    if not (eng / "bin" / "dscsm048").is_file():
        pytest.skip(f"private instrumented dscsm048 not found at {eng}")
    return eng


@pytest.mark.allow_skip(reason="needs the private instrumented DSSAT build in the data directory")
@pytest.mark.parametrize(
    ("exp", "trno"), [("UFGA8201", 1), ("UFGA8201", 5), ("IUAF9901", 1), ("SIAZ9501", 3)]
)
def test_full_precision_drivers_give_float32_agreement(dump_engine, tmp_path, exp, trno):
    base = run_reference(exp, trno, tmp_path / "base")
    out = run_reference(exp, trno, tmp_path / "dump", engine=dump_engine)

    # the extra WRITE does not change any output (all but the time-stamped header lines)
    def strip(t: str) -> list[str]:
        return [
            ln for ln in t.splitlines() if "DSSAT Cropping System" not in ln and not ln.startswith("*SUMMARY")
        ]

    for name in ("PlantGro.OUT", "SoilWat.OUT", "Summary.OUT", "ET.OUT"):
        assert strip((base / name).read_text(errors="replace")) == strip(
            (out / name).read_text(errors="replace")
        ), name
    p, f, _, row = simulate(out, trno)
    nl = int(p.soil.dlayr.shape[0])
    dump = np.loadtxt(out / "CERESDUMP.OUT", ndmin=2)
    col = {n: dump[:, i] for i, n in enumerate(DUMP_NAMES)}
    k = len(DUMP_NAMES)
    days = np.asarray(f.yrdoy)
    idx = np.searchsorted(days, col["YRDOY"].astype(int))
    sw = np.asarray(f.sw).copy()
    sw[idx] = dump[:, k : k + nl]
    swfac = np.asarray(f.swfac).copy()
    swfac[idx] = col["SWFAC"]
    turfac = np.asarray(f.turfac).copy()
    turfac[idx] = col["TURFAC"]
    f = eqx.tree_at(
        lambda x: (x.sw, x.swfac, x.turfac), f, (jnp.asarray(sw), jnp.asarray(swfac), jnp.asarray(turfac))
    )
    res = {kk: np.asarray(v) for kk, v in _RUNNER(p, f, CeresMaizeState.initial(p, 1)).items()}
    np.testing.assert_array_equal(res["istage"][idx, 0], col["ISTAGE"].astype(int))
    for name, (key, scale) in DUMP_TO_OUTPUT.items():
        sim = res[key][idx, 0] * scale
        ref = col[name]
        sig = np.abs(ref) > 1e-3 * max(np.abs(ref).max(), 1e-30)
        rel = np.abs(sim - ref)[sig] / np.abs(ref[sig])
        assert rel.max() < 1e-4, (name, rel.max())
    # RLV is truncated to 1e-3: float32 vs float64 may land one quantum apart
    assert np.abs(res["rlv"][idx, 0] - dump[:, k + nl : k + 2 * nl]).max() <= 1e-3 + 1e-6
    np.testing.assert_allclose(res["ewsd"][idx, 0], col["SATFAC"], atol=1e-3)
    if float(row["HWAM"]) > 0:
        assert abs(float(res["gwad"][-1, 0]) - float(row["HWAM"])) < 1.0  # HWAM is printed as an integer


def test_yrdoy_range_crosses_the_year():
    assert yrdoy_range(2003360, 2004003) == [
        2003360,
        2003361,
        2003362,
        2003363,
        2003364,
        2003365,
        2004001,
        2004002,
        2004003,
    ]
    assert len(yrdoy_range(2000001, 2000366)) == 366
