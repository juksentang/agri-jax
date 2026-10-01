"""DSSAT-CSM accepts the written cultivar file, and reads the observed data as our FILEA reader.

* **Cultivar round trip through the engine.** UFGA8201 is run with ``dscsm048`` (v4.8.6.0) twice:
  once as distributed (cultivar ``IB0035``), once with a copy of ``MZCER048.CUL`` holding a new
  row ``AJ0035`` written by :func:`~agrijax.io.dssat.cultivar_write.write_cultivar` with
  IB0035's published values and the experiment's ``INGENO`` pointed at it. Every ``Summary.OUT``
  and ``PlantGro.OUT`` value must be identical. A third run with P5 changed in the written row
  must change maturity (the engine really reads the new row).
* **Observed data as DSSAT reads it.** ``Evaluate.OUT`` lists the measured values the engine read
  from FILEA (``READA``): for every maize experiment of the example tree with a FILEA, every
  measured column equals our reading (``HWAMM`` = HWAM ..., ``ADAPM`` / ``MDAPM`` = the observed
  date minus the planting date).

Runs are staged under ``AGRI_JAX_RUN_ROOT`` when set (a short node-local directory: the engine's path
fields hold 80 characters), else in pytest's temporary directory.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agrijax.io.dssat import read_plantgro, read_summary, run_dssat, stage_run_dir
from agrijax.io.dssat.cultivar_write import CUL_COEFFICIENTS, write_cultivar
from agrijax.io.dssat.genotype import read_cul
from agrijax.io.dssat.observed import read_observed
from agrijax.io.dssat.outputs import read_evaluate
from agrijax.io.dssat.wth import parse_dssat_date
from agrijax.port.run_fortran import DEFAULT_DSSAT_ENGINE

DSSAT_ENGINE = Path(os.environ.get("AGRI_JAX_DSSAT", str(DEFAULT_DSSAT_ENGINE))).expanduser()
DSCSM = Path(
    os.environ.get("AGRI_JAX_DSCSM", str(DSSAT_ENGINE / "source" / "build486" / "bin" / "dscsm048"))
).expanduser()
UFGA_FILEX = Path("UFGA8201.MZX")  # staged by the ``stage_ufga8201`` fixture (conftest)
pytestmark = pytest.mark.slow

#: DSSAT's path fields are short (``PATHEX`` 80 characters)
MAX_RUN_DIR = 60


def _run_dir(tmp_path: Path, tag: str) -> Path:
    root = os.environ.get("AGRI_JAX_RUN_ROOT")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        d = Path(tempfile.mkdtemp(prefix=f"{tag}_", dir=root))
    else:
        d = tmp_path / tag
        d.mkdir()
    assert len(str(d)) <= MAX_RUN_DIR, f"run dir {d} too long for DSSAT; set AGRI_JAX_RUN_ROOT"
    return d


def _set_ingeno(filex: Path, ingeno: str) -> None:
    """Point every ``*CULTIVARS`` row of the (copied) experiment file at cultivar ``ingeno``."""
    raw = filex.read_bytes().decode("latin-1")
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.split(nl)
    sec = False
    col = None
    n = 0
    for i, ln in enumerate(lines):
        if ln.startswith("*"):
            sec = ln.upper().startswith("*CULTIVARS")
            continue
        if sec and ln.startswith("@"):
            col = ln.index("INGENO")
            continue
        if sec and col is not None and ln.strip() and not ln.startswith("!"):
            lines[i] = ln[:col] + ingeno + ln[col + 6 :]
            n += 1
    assert n >= 1, f"no cultivar row in {filex}"
    filex.write_bytes(nl.join(lines).encode("latin-1"))


def _run(run_dir: Path, filex_name: str) -> Path:
    r = run_dssat(run_dir, filex_name)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (run_dir / "Summary.OUT").is_file(), r.stdout[-2000:]
    return run_dir


def _summary_values(p: Path) -> pd.DataFrame:
    s = read_summary(p / "Summary.OUT")
    return s.drop(columns=[c for c in s.columns if c in {"CR", "MODEL", "EXNAME"}])


@pytest.mark.allow_skip(reason="needs dscsm048 and the DSSAT example tree ($AGRI_JAX_DSSAT)")
def test_dssat_accepts_written_cultivar(
    tmp_path: Path, stage_ufga8201: Callable[[Path], Path], capsys: pytest.CaptureFixture[str]
) -> None:
    ref_dir = _run(stage_ufga8201(_run_dir(tmp_path, "cul_ref")), UFGA_FILEX.name)

    src_cul = DSSAT_ENGINE / "source" / "Data" / "Genotype" / "MZCER048.CUL"
    src_bytes = src_cul.read_bytes()
    published = read_cul(src_cul).loc["IB0035"]
    values = {c: float(published[c]) for c in CUL_COEFFICIENTS}

    new_dir = stage_ufga8201(_run_dir(tmp_path, "cul_new"))
    staged = new_dir / "MZCER048.CUL"
    staged.unlink()  # the run dir's copy is replaced by the written one
    w = write_cultivar(src_cul, staged, "AJ0035", "agrijax copy", values, base="IB0035")
    assert w.rounded == {} and w.out_of_range == {}
    assert src_cul.read_bytes() == src_bytes  # the engine's file is untouched
    _set_ingeno(new_dir / UFGA_FILEX.name, "AJ0035")
    _run(new_dir, UFGA_FILEX.name)

    # the engine used the new row
    overview = (new_dir / "OVERVIEW.OUT").read_text(errors="replace")
    assert "AJ0035" in overview and "IB0035" not in overview

    a, b = _summary_values(ref_dir), _summary_values(new_dir)
    pd.testing.assert_frame_equal(a, b)
    for col in ("HWAM", "CWAM", "ADAT", "MDAT", "EDAT"):
        np.testing.assert_array_equal(a[col].to_numpy(), b[col].to_numpy(), err_msg=col)
    pg_a, pg_b = read_plantgro(ref_dir / "PlantGro.OUT"), read_plantgro(new_dir / "PlantGro.OUT")
    pd.testing.assert_frame_equal(pg_a, pg_b)

    # negative control: a changed P5 in the written row changes maturity
    ctl_dir = stage_ufga8201(_run_dir(tmp_path, "cul_ctl"))
    (ctl_dir / "MZCER048.CUL").unlink()
    write_cultivar(
        src_cul,
        ctl_dir / "MZCER048.CUL",
        "AJ0035",
        "agrijax P5-100",
        {**values, "P5": values["P5"] - 100.0},
        base="IB0035",
    )
    _set_ingeno(ctl_dir / UFGA_FILEX.name, "AJ0035")
    c = _summary_values(_run(ctl_dir, UFGA_FILEX.name))
    assert (c["MDAT"].to_numpy() < a["MDAT"].to_numpy()).all(), (c["MDAT"], a["MDAT"])
    with capsys.disabled():
        print(
            f"\nUFGA8201 with AJ0035 (written) == IB0035: Summary {a.shape}, PlantGro {pg_a.shape} identical; "
            f"HWAM {a['HWAM'].tolist()}, ADAT {a['ADAT'].tolist()}, MDAT {a['MDAT'].tolist()}; "
            f"P5-100 control MDAT {c['MDAT'].tolist()}"
        )


_MAIZE = DSSAT_ENGINE / "example_data" / "Maize"
_WITH_A = (
    sorted(x.stem for x in _MAIZE.glob("*.MZX") if any(_MAIZE.glob(x.stem + ".MZA")))
    if _MAIZE.is_dir()
    else []
)
#: FILEA date codes -> the Evaluate.OUT measured column (days after planting). Other codes: the
#: code + ``M``. FILEA codes are already READA_Y4K's aliases (HWAH -> HWAM, BWAH -> BWAM ...).
_DATES = {"ADAT": "ADAP", "MDAT": "MDAP", "EDAT": "EDAP", "PD1T": "PD1P", "PDFT": "PDFP"}

#: the FILEA codes of each experiment with no Evaluate.OUT column for CERES-Maize (not compared);
#: the maize Evaluate.OUT has ADAP PD1P PDFP MDAP HWAM PWAM H#AM HWUM H#UM CWAM BWAM LAIX HIAM THAM
#: GNAM CNAM SNAM GN%M CWAA CNAA L#SM EDAP
_NOT_IN_EVALUATE: dict[str, list[str]] = {
    "BRPI0202": ["L#SD"],
    "EBPL8501": [],
    "FLSC8101": [],
    "GAGR0201": ["CHTA"],
    "GHWA0401": [],
    "IBWA8301": [],
    "IUAF9901": [],
    "SIAZ9501": ["TNAM"],
    "SIAZ9601": ["TNAM"],
    "UFGA8201": [],
}


def _evaluate_column(code: str) -> str:
    return _DATES.get(code, code) + "M"


@pytest.mark.allow_skip(reason="needs dscsm048 and the DSSAT example tree ($AGRI_JAX_DSSAT)")
@pytest.mark.parametrize("exp", _WITH_A or ["none"])
def test_filea_equals_dssat_measured(exp: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    if not DSCSM.is_file() or not _WITH_A:
        pytest.skip(f"dscsm048 ({DSCSM}) or {_MAIZE} not found")
    filex = _MAIZE / f"{exp}.MZX"
    run_dir = stage_run_dir(
        filex,
        _run_dir(tmp_path, exp.lower()),
        data_dir=DSSAT_ENGINE / "source" / "Data",
        weather_dirs=[DSSAT_ENGINE / "example_data" / "Weather"],
        soil_dirs=[DSSAT_ENGINE / "example_data" / "Soil"],
        binary=DSCSM,
    )
    _run(run_dir, filex.name)
    ev = read_evaluate(run_dir / "Evaluate.OUT")
    summ = read_summary(run_dir / "Summary.OUT").drop_duplicates("TRNO").set_index("TRNO")
    obs = read_observed(filex)
    assert obs.filea is not None
    skipped = [c for c in obs.filea.codes if _evaluate_column(c) not in ev.columns]
    with capsys.disabled():
        print(f"\n{exp}: FILEA codes without an Evaluate.OUT column: {skipped}")
    assert skipped == _NOT_IN_EVALUATE[exp], (exp, skipped)
    n = 0
    for _, row in ev.iterrows():
        trno = int(row["TN"])
        for code in obs.filea.codes:
            col = _evaluate_column(code)
            if code in skipped:
                continue
            want = float(row[col])
            if code in _DATES:
                d = obs.filea.date(trno, code, int(summ.at[trno, "SDAT"]))
                if d is None:
                    assert math.isnan(want), (exp, trno, code, want)
                    continue
                pdat = parse_dssat_date(str(int(summ.at[trno, "PDAT"])))
                assert (d - pdat).days == want, (exp, trno, code, d, want)
                n += 1
                continue
            got = obs.filea.value(trno, code)
            assert (math.isnan(got) and math.isnan(want)) or got == want, (exp, trno, code, got, want)
            n += int(math.isfinite(got))
    assert n > 0 or len(skipped) == len(obs.filea.codes), f"{exp}: no FILEA value compared"
