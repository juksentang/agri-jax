"""The top-level facade without DSSAT: ``import agrijax`` stays light, the lazy names, readable errors,
the experiment's description, the FileX scenario helpers, the season comparisons and the figures
(on synthetic tables)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import agrijax as aj
from agrijax import dssat as ajd
from agrijax.sites.dssat_free_run import cultivar_batch_filex, shift_filex_dates

_FILEX = (
    "*EXP.DETAILS: TEST8201MZ synthetic experiment\n\n"
    "*TREATMENTS                        -------------FACTOR LEVELS------------\n"
    "@N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM\n"
    " 1 1 0 0 RAINFED                    1  1  0  1  1  0  0  0  0  0  0  0  1\n"
    " 2 1 0 0 GROWTH CHAMBER             1  1  0  1  1  0  0  0  0  0  1  0  1\n\n"
    "*CULTIVARS\n@C CR INGENO CNAME\n 1 MZ IB0035 McCurdy 84aa\n\n"
    "*FIELDS\n"
    "@L ID_FIELD WSTA....  FLSA  FLOB  FLDT  FLDD  FLDS  FLST SLTX  SLDP  ID_SOIL    FLNAME\n"
    " 1 UFGA0002 UFGA       -99     0 DR000     0     0 00000 -99    180  IBMZ910014 Field section\n\n"
    "*INITIAL CONDITIONS\n"
    "@C   PCR ICDAT  ICRT  ICND  ICRN  ICRE  ICWD ICRES ICREN ICREP ICRIP ICRID ICNAME\n"
    " 1    MZ 82056   100     0     1     1   -99     0    .8     0   100    15 -99\n\n"
    "*PLANTING DETAILS\n"
    "@P PDATE EDATE  PPOP  PPOE  PLME  PLDS  PLRS  PLRD  PLDP  PLWT  PAGE  PENV  PLPH  SPRL\n"
    " 1 82057   -99   7.2   7.2     S     R    61     0     7   -99   -99   -99   -99     0\n\n"
    "*SIMULATION CONTROLS\n"
    "@N GENERAL     NYERS NREPS START SDATE RSEED SNAME.................... SMODEL\n"
    " 1 GE              1     1     S 82056  2150 TEST\n"
    "@N HARVEST     HFRST HLAST HPCNP HPCNR\n"
    " 1 HA              0 83057   100     0\n"
)


def test_import_is_light_and_names_are_lazy():
    code = (
        "import sys, agrijax as aj; assert 'jax' not in sys.modules, 'jax imported'; "
        "print(sorted(n for n in ('dssat','calibrate','plot','machine') if n in dir(aj)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "calibrate" in out.stdout and "dssat" in out.stdout
    assert callable(aj.calibrate) and aj.dssat is ajd and callable(aj.dssat.install_reference)
    assert aj.plot.__name__ == "agrijax.report.plot"
    with pytest.raises(AttributeError):
        _ = aj.no_such_name  # type: ignore[attr-defined]
    assert "JAX" in aj.machine() and "logical cores" in aj.machine()


def test_missing_reference_program_says_how_to_get_it(tmp_path):
    with pytest.raises(ajd.DssatNotFoundError, match=r"install_reference\(\)"):
        ajd.engine(tmp_path)


def _engine(tmp_path: Path) -> Path:
    exe = tmp_path / "eng" / "source" / "build486" / "bin" / "dscsm048"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    (tmp_path / "eng" / "source" / "Data").mkdir(parents=True)
    maize = tmp_path / "eng" / "example_data" / "Maize"
    maize.mkdir(parents=True)
    (maize / "TEST8201.MZX").write_text(_FILEX)
    return tmp_path / "eng"


def test_experiment_description(tmp_path):
    exp = ajd.experiment("TEST8201", data_root=_engine(tmp_path))
    assert exp.name == "TEST8201" and exp.title == "TEST8201MZ synthetic experiment"
    assert exp.treatments == {1: "RAINFED", 2: "GROWTH CHAMBER"} and exp.cultivars == {
        1: "IB0035",
        2: "IB0035",
    }
    assert exp.soil == "IBMZ910014" and exp.station == "UFGA"
    assert Path(exp) == exp.filex  # os.PathLike: accepted where a path is
    t = exp.table()
    assert t.loc[1, "native inputs"] == "yes" and "WTHMOD" in t.loc[2, "native inputs"]
    assert t.loc[1, "yield change with nitrogen on"] == "not measured"
    with pytest.raises(ValueError, match="no treatment 7"):
        exp.reference(7)
    with pytest.raises(ValueError, match=r"\.MZX"):
        ajd.experiment("TEST8201.WHX", data_root=exp.data)


def test_shift_filex_dates():
    u = shift_filex_dates(_FILEX, years=3, days=-14)
    assert " 1    MZ 85042   100" in u and " 1 85043   -99   7.2" in u and "S 85042  2150" in u
    assert " 1 HA              0 86043" in u  # HLAST moves too; HFRST 0 stays
    assert shift_filex_dates(_FILEX) == _FILEX
    # across a year end and a leap day, 7-digit dates keep their width
    v = shift_filex_dates("@P   PDATE\n 1 2019365\n", days=1)
    assert v == "@P   PDATE\n 1 2020001\n"
    assert shift_filex_dates("@P PDATE\n 1 20059\n", days=1) == "@P PDATE\n 1 20060\n"
    assert shift_filex_dates("@P PDATE EDATE\n 1 82057   -99\n", days=5) == "@P PDATE EDATE\n 1 82062   -99\n"


def test_cultivar_batch_filex():
    u = cultivar_batch_filex(_FILEX, 1, ["AJ0001", "AJ0002"])
    lines = u.splitlines()
    i = lines.index(next(ln for ln in lines if ln.startswith("@N R O C")))
    assert lines[i + 1].startswith(" 1 1 0 0 RAINFED") and lines[i + 1][35:37] == " 1"
    assert lines[i + 2].startswith(" 2 1 0 0 RAINFED") and lines[i + 2][35:37] == " 2"
    assert "GROWTH CHAMBER" not in u
    assert " 1 MZ AJ0001 AJ0001" in u and " 2 MZ AJ0002 AJ0002" in u and "IB0035" not in u
    with pytest.raises(ValueError, match="99"):
        cultivar_batch_filex(_FILEX, 1, [f"AJ{i:04d}" for i in range(100)])


def test_samples_and_dates():
    pub = dict(zip(ajd.CULTIVAR, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], strict=True))
    assert ajd._samples(None, pub).tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    s = ajd._samples({"G2": [10.0, 20.0]}, pub)
    assert s.shape == (2, 6) and s[:, 3].tolist() == [10.0, 20.0] and s[:, 0].tolist() == [1.0, 1.0]
    s = ajd._samples(pd.DataFrame({"P1": [7.0, 8.0, 9.0]}), pub)
    assert s[:, 0].tolist() == [7.0, 8.0, 9.0]
    with pytest.raises(ValueError, match="unknown"):
        ajd._samples({"XX": [1.0]}, pub)
    with pytest.raises(ValueError, match="different lengths"):
        ajd._samples({"P1": [1.0, 2.0], "G2": [1.0, 2.0, 3.0]}, pub)
    assert ajd._yrdoy_add(1982365, 1) == 1983001 and ajd._yrdoy_add(1984059, 1) == 1984060


def _daily(n: int, scale: float = 1.0) -> pd.DataFrame:
    days = [1982057 + i for i in range(n)]
    t = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "date": ajd._dates(days),
            "yrdoy": days,
            "lai": scale * np.sin(t / n * np.pi) * 4,
            "cwad": scale * t * 100,
            "gwad": scale * np.maximum(t - n / 2, 0) * 80,
            "swtd": 200 - t * 0.3,
        }
    )


def test_season_comparisons():
    summ = {"ADAT": 1982140.0, "MDAT": 1982180.0, "HWAM": 10000.0, "CWAM": 20000.0}
    s = ajd.Season("T_t01", _daily(100), summ, {}, None, {})
    ref = ajd.Reference(
        "T_t01", _daily(90, 1.01), {**summ, "MDAT": 1982181.0, "HWAM": 10100.0}, Path("."), 0.1
    )
    d = s.compare_daily(ref)
    assert list(d.index) == ["lai", "cwad", "gwad", "swtd"] and (d["days"] == 90).all()
    assert d.loc["swtd", "RMSE"] == 0.0 and d.loc["cwad", "RMSE"] > 0
    c = s.compare_summary(ref)
    assert c.loc["MDAT", "difference"] == -1.0 and c.loc["ADAT", "difference"] == 0.0
    assert c.loc["HWAM", "difference"] == pytest.approx(-100 / 10100)


def test_figures_draw_from_tables():
    import matplotlib

    matplotlib.use("Agg")
    summ = {"ADAT": 1982140.0, "MDAT": 1982180.0, "HWAM": 1.0, "CWAM": 1.0}
    s = ajd.Season("T_t01", _daily(60), summ, {}, None, {})
    ref = ajd.Reference("T_t01", _daily(60, 0.98), summ, Path("."), 0.1)
    obs = _daily(60).iloc[::15][["date", "yrdoy", "lai", "cwad"]]
    assert len(aj.plot.season(s, reference=ref, observed=obs).axes) == 4
    assert len(aj.plot.compare(s, ref).axes) == 4
    tab = pd.DataFrame({"year": [1980, 1980, 1981, 1981], "HWAM": [1.0, 2.0, 3.0, 4.0]})
    assert len(aj.plot.batch(ajd.BatchResult(tab, {"seasons": 4})).axes) == 1
    fit = pd.DataFrame(
        {
            "set": ["calibration"] * 3,
            "code": ["ADAT", "HWAM", "HWAM"],
            "kind": ["date", "final", "final"],
            "observed": [1982140, 10.0, 11.0],
            "published": [1982142, 9.0, 10.0],
            "calibrated": [1982141, 10.0, 10.5],
        }
    )
    assert len(aj.plot.calibration(fit).axes) == 2
    import matplotlib.pyplot as plt

    plt.close("all")


def test_nitrogen_limited_treatments_warn_once(tmp_path, monkeypatch):
    """``exp.run`` on a treatment where DSSAT's yield changes with nitrogen on warns (the model runs
    nitrogen off); once per treatment, and not on a treatment where nitrogen does not matter."""
    import warnings

    from agrijax.calib import workflow as wf

    exp = ajd.experiment("TEST8201", data_root=_engine(tmp_path))
    monkeypatch.setitem(wf.NITROGEN_STRESS, "TEST8201_t01", -0.308)
    monkeypatch.setitem(wf.NITROGEN_STRESS, "TEST8201_t02", -0.004)
    monkeypatch.setattr(ajd.Experiment, "_build", lambda self, fx, t: object())
    with pytest.warns(ajd.NitrogenWarning, match=r"-30\.8%"):
        exp.inputs(1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exp.inputs(1)  # once per treatment
        exp.inputs(2)  # nitrogen does not matter here


def test_calibration_result_repr_is_short():
    from agrijax.calib.workflow import CalibrationResult

    r = CalibrationResult.__new__(CalibrationResult)
    r.cultivar, r.params, r.treatments, r.inputs = "IB0035", {"P1": 249.0, "G2": 906.3}, ["X_t04"], "native"
    r.loss = {"published": 1.015, "written": 0.01135}
    s = repr(r)
    assert s.startswith("CalibrationResult(IB0035: P1=249, G2=906.3") and "1.015 -> 0.01135" in s
    assert len(s) < 200
