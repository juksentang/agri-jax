"""The crop selection of the DSSAT facade (:mod:`agrijax.crops`): lookup by crop code and by FileX,
the refusal of a crop Agri-JAX does not implement, and the maize record as the one source of the
facade's maize names (no data, no DSSAT)."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from agrijax import crops, facade_export
from agrijax import dssat as ajd
from agrijax.calib import workflow as wf
from agrijax.crops import CROPS, MAIZE, UnsupportedCropError, crop, crop_of_filex

from .test_facade import _engine

SRC = Path(crops.__file__).resolve().parent


def test_lookup_by_code_and_filex():
    assert CROPS == {"MZ": MAIZE}
    assert crop("MZ") is MAIZE and crop(" mz ") is MAIZE
    assert crop_of_filex("UFGA8201.MZX") is MAIZE and crop_of_filex(Path("/x/ufga8201.mzx")) is MAIZE
    assert (MAIZE.cul, MAIZE.eco, MAIZE.spe) == ("MZCER048.CUL", "MZCER048.ECO", "MZCER048.SPE")
    assert MAIZE.describe() == "MZ Maize (CERES-Maize, .MZX)"


def test_unsupported_crop_lists_the_supported_ones():
    with pytest.raises(UnsupportedCropError, match=r"crop 'SB' is not supported; supported crops: MZ Maize"):
        crop("SB")
    with pytest.raises(UnsupportedCropError, match=r"crop SB \(\.SBX\) is not supported.*\.MZX"):
        crop_of_filex("UFGA8101.SBX")
    with pytest.raises(UnsupportedCropError, match=r"not a DSSAT experiment file.*MZ Maize"):
        crop_of_filex("notes.txt")
    assert issubclass(UnsupportedCropError, ValueError)


def test_facade_refuses_an_unsupported_crop(tmp_path):
    eng = _engine(tmp_path)
    (eng / "example_data" / "Maize" / "TEST8201.SBX").write_text("*EXP.DETAILS: soybean\n")
    with pytest.raises(wf.ScopeError, match=r"crop SB \(\.SBX\) is not supported; supported crops: MZ"):
        ajd.experiment(eng / "example_data" / "Maize" / "TEST8201.SBX", data_root=eng)
    with pytest.raises(wf.ScopeError, match=r"crop WH \(\.WHX\)"):
        wf._filex_path("TEST8201.WHX", eng)
    with pytest.raises(wf.ScopeError, match=r"no maize experiment file XXXX9901\.MZX"):
        wf._filex_path("XXXX9901", eng)
    exp = ajd.experiment("TEST8201", data_root=eng)
    assert exp.crop is MAIZE and repr(exp).startswith("Experiment(TEST8201")


def test_maize_record_points_at_the_maize_implementation():
    from agrijax.calib import ceres
    from agrijax.calib.dssat_day import CUL_ORDER

    # the rejection tables are the module's own objects, read when asked
    assert MAIZE.nitrogen_effect is wf.NITROGEN_STRESS
    assert MAIZE.nitrogen_dates_differ is wf.NITROGEN_DATES_DIFFER
    assert MAIZE.out_of_scope is wf.OUT_OF_SCOPE
    # the calibratable coefficients: names in .CUL order declared once (the record), bounds once (the adapter)
    from agrijax.io.dssat.cultivar_write import CUL_COEFFICIENTS

    assert CUL_ORDER is MAIZE.cultivar
    assert CUL_COEFFICIENTS == MAIZE.cultivar  # the io row format (the io layer imports no crop record)
    spec = importlib.import_module(MAIZE.calibration).CERES_SPECS
    assert set(MAIZE.cultivar) <= set(spec)
    assert all(
        spec[n].bounds_source.startswith("DSSAT-CSM v4.8.6.0 Genotype/MZCER048.CUL") for n in MAIZE.cultivar
    )
    assert MAIZE.stages == (ceres.ISTAGE_SILKING_OUT, ceres.ISTAGE_MATURITY_OUT)
    day = importlib.import_module(MAIZE.day)
    assert day.SLOT == "maize" and callable(day.day_dssat486)


def test_replaced_table_is_read_through_the_record(monkeypatch):
    monkeypatch.setattr(wf, "NITROGEN_DATES_DIFFER", frozenset({"TWIN_t04"}))
    assert MAIZE.nitrogen_dates_differ == {"TWIN_t04"}


def test_facade_maize_names_come_from_the_record():
    assert ajd.CULTIVAR is MAIZE.cultivar
    assert (ajd._SILKING, ajd._MATURITY) == MAIZE.stages
    assert (facade_export._CROP, facade_export._MODEL, facade_export._CROP_NAME) == (
        MAIZE.code,
        MAIZE.model,
        MAIZE.name,
    )
    assert f"{MAIZE.name.lower()} model ({MAIZE.implementation})" in facade_export._AGRIJAX_SOURCE


def _code_strings(path: Path) -> list[tuple[int, str]]:
    """The string constants of a module outside its docstrings (``(line, value)``)."""
    tree = ast.parse(path.read_text())
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docs.add(id(body[0].value))
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            docs.add(id(node.value))  # attribute docstrings
    return [
        (n.lineno, n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs
    ]


_FACADE = ["dssat.py", "calib/workflow.py", *sorted(p.name for p in SRC.glob("facade_*.py"))]


@pytest.mark.parametrize("module", _FACADE)
def test_no_maize_literal_in_the_facade_code(module):
    """The facade's crop choice reads the record: no crop code, model, genotype file or FileX
    extension of maize is spelled in the code of the facade modules (docstrings and comments may
    name them)."""
    bad = [
        (line, v)
        for line, v in _code_strings(SRC / module)
        if v in ("MZ", "Maize") or "MZCER048" in v or ".MZ" in v
    ]
    assert not bad, f"{module}: maize literals in code {bad}"
