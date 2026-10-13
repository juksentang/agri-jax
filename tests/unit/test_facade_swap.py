"""Swapping the soil evaporation from the facade, without DSSAT: the table of validated alternatives,
the readable errors, the experiment-file edit that carries the DSSAT ``MESEV`` option, and the plumbing
of ``soil_evaporation=`` through ``Experiment`` (inputs, reference, DSSAT batch, scenarios) and
``calibrate``. The input builder and the DSSAT and calibration runners are stood in for: nothing here
builds inputs or runs a model."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from agrijax import dssat as ajd
from agrijax import facade_swap as fs
from agrijax.io.dssat import read_filex
from agrijax.io.dssat.native_management import switches

_METHODS = "@N METHODS     WTHER INCON LIGHT EVAPO INFIL PHOTO HYDRO NSWIT MESOM MESEV MESOL\n"
_OPTIONS = "@N OPTIONS     WATER NITRO SYMBI PHOSP POTAS DISES  CHEM  TILL   CO2\n"
#: a synthetic experiment: treatment 1 runs DSSAT's ``MESEV = R`` (simulation controls level 1),
#: treatment 2 ``MESEV = S`` (level 2); both levels have a METHODS line
_FILEX = (
    "*EXP.DETAILS: SWAP8201MZ synthetic experiment\n\n"
    "*TREATMENTS                        -------------FACTOR LEVELS------------\n"
    "@N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM\n"
    " 1 1 0 0 RITCHIE                    1  1  0  1  1  0  0  0  0  0  0  0  1\n"
    " 2 1 0 0 SALUS                      1  1  0  1  1  0  0  0  0  0  0  0  2\n\n"
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
    f"{_OPTIONS}"
    " 1 OP              Y     N     N     N     N     N     N     Y     M\n"
    f"{_METHODS}"
    " 1 ME              M     M     E     R     S     R     R     1     G     R     2\n"
    "@N MANAGEMENT  PLANT IRRIG FERTI RESID HARVS\n"
    " 1 MA              R     R     R     N     M\n"
    "@N HARVEST     HFRST HLAST HPCNP HPCNR\n"
    " 1 HA              0 83057   100     0\n"
    "@N GENERAL     NYERS NREPS START SDATE RSEED SNAME.................... SMODEL\n"
    " 2 GE              1     1     S 82056  2150 TEST\n"
    f"{_OPTIONS}"
    " 2 OP              Y     N     N     N     N     N     N     Y     M\n"
    f"{_METHODS}"
    " 2 ME              M     M     E     R     S     R     R     1     G     S     2\n"
    "@N MANAGEMENT  PLANT IRRIG FERTI RESID HARVS\n"
    " 2 MA              R     R     R     N     M\n"
    "@N HARVEST     HFRST HLAST HPCNP HPCNR\n"
    " 2 HA              0 83057   100     0\n"
)
#: the same experiment with every simulation control line but the general, management and harvest ones left out
_NO_METHODS = "".join(
    f"{ln}\n"
    for ln in _FILEX.splitlines()
    if not ln.startswith(("@N METHODS", "@N OPTIONS")) and ln.split()[1:2] not in (["ME"], ["OP"])
)
#: the column of the MESEV letter in a METHODS line
_MESEV_COLUMN = _METHODS.index("MESEV") + len("MESEV") - 1


class _Boom(Exception):
    """Stops a stand-in runner after it has recorded what it was called with."""


def _mesev_of(text: str, treatment: int) -> str:
    """The ``MESEV`` of ``treatment`` in FileX ``text``, as DSSAT reads it."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "check.MZX"
        p.write_text(text)
        return fs.own_mesev(p, treatment)


@pytest.fixture
def exp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """An experiment on a synthetic engine layout (its FileX and the weather file of its station)."""
    return _experiment(tmp_path, monkeypatch)


def _experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str = _FILEX) -> Any:
    maize = tmp_path / "eng" / "example_data" / "Maize"
    weather = tmp_path / "eng" / "example_data" / "Weather"
    maize.mkdir(parents=True)
    weather.mkdir(parents=True)
    (maize / "SWAP8201.MZX").write_text(text)
    (weather / "UFGA8201.WTH").write_text("")
    e = ajd.experiment("SWAP8201", data_root=tmp_path / "eng")
    monkeypatch.setenv("AGRI_JAX_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setattr(ajd, "_x64", lambda: None)  # nothing here runs a model: leave the JAX setting alone
    monkeypatch.setattr(ajd, "data", lambda root=None: e.data)  # no download of DSSAT's data
    return e


def _fake_inputs(filex: Path) -> Any:
    """What ``Experiment._build`` stands in with: the fields of the inputs the facade reads."""
    return SimpleNamespace(
        filex=Path(filex),
        days=np.asarray([1982057, 1982058]),
        published=lambda: dict.fromkeys(ajd.CULTIVAR, 1.0),
        params_crop=SimpleNamespace(yrplt=1982057),
        summary=dict.fromkeys(ajd.SUMMARY, 1.0),
    )


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The experiment files the inputs were built from (``Experiment._build`` is a stand-in)."""
    calls: list[Path] = []

    def build(self: Any, filex: Path, trno: int) -> Any:
        calls.append(Path(filex))
        return _fake_inputs(filex)

    monkeypatch.setattr(ajd.Experiment, "_build", build)
    return calls


def _stop_run_ref(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """``Experiment._run_ref`` records its keyword arguments and stops."""
    seen: list[dict[str, Any]] = []

    def run_ref(self: Any, trno: Any, tag: str, **kw: Any) -> Any:
        seen.append(kw)
        raise _Boom

    monkeypatch.setattr(ajd.Experiment, "_run_ref", run_ref)
    return seen


# ------------------------------------------------------------------ the table and the choices
def test_alternatives_are_the_two_mesev_of_the_day():
    from agrijax.models.day_dssat486 import MESEV, SOIL_EVAPORATION_KEYS, resolve_soil_evaporation

    t = ajd.alternatives()
    assert list(t.index) == ["ritchie", "salus"] and set(t.index) == {a.name for a in fs.ALTERNATIVES}
    assert list(t["DSSAT option"]) == ["MESEV = R", "MESEV = S"]
    assert {a.mesev for a in fs.ALTERNATIVES} == set(MESEV)
    for (
        a
    ) in fs.ALTERNATIVES:  # each choice is a checked implementation of the day, under the key of its MESEV
        assert a.key == SOIL_EVAPORATION_KEYS[a.mesev]
        assert resolve_soil_evaporation(a.key).mesev == a.mesev
    assert t["validated against"].str.contains("dscsm048").all()
    assert "experiment's own" not in t.columns


def test_alternatives_says_which_one_the_experiment_uses(exp):
    assert ajd.alternatives(exp, 1)["experiment's own"].to_dict() == {"ritchie": True, "salus": False}
    assert ajd.alternatives(exp, 2)["experiment's own"].to_dict() == {"ritchie": False, "salus": True}
    with pytest.raises(ValueError, match="both the experiment and the treatment"):
        ajd.alternatives(exp)
    with pytest.raises(ValueError, match="no treatment 9"):
        ajd.alternatives(exp, 9)


@pytest.mark.parametrize(
    ("choice", "mesev"),
    [
        ("ritchie", "R"),
        ("Salus", "S"),
        ("  SALUS ", "S"),
        ("R", "R"),
        ("s", "S"),
        ("soil_water/soilev@dssat-4.8.6.0:faithful", "R"),
        ("soil_water/esr_soilevap@dssat-4.8.6.0:faithful", "S"),
    ],
)
def test_resolve_accepts_the_names_letters_and_keys(choice, mesev):
    alt = fs.resolve(choice)
    assert alt is not None and alt.mesev == mesev


def test_resolve_none_is_the_experiments_own():
    assert fs.resolve(None) is None


@pytest.mark.parametrize("choice", ["sauls", "ritchi", "foo", "", "mulch", 3, ["salus"], b"salus"])
def test_unknown_alternative_lists_the_valid_ones(choice):
    with pytest.raises(fs.SwapError) as e:
        fs.resolve(choice)
    msg = str(e.value)
    assert "'ritchie'" in msg and "'salus'" in msg and "MESEV = R" in msg and "MESEV = S" in msg
    assert isinstance(e.value, ValueError)


def test_close_name_gets_a_suggestion():
    with pytest.raises(fs.SwapError, match=r"Did you mean 'salus'\?"):
        fs.resolve("sauls")
    with pytest.raises(fs.SwapError, match=r"Did you mean 'ritchie'\?"):
        fs.resolve("ritchi")


def test_unvalidated_and_unregistered_keys_are_refused_with_the_reason():
    # registered in the process registry, but not a soil evaporation validated for the day
    with pytest.raises(fs.SwapError, match="registered, but is not a soil evaporation validated"):
        fs.resolve("soil_water/mulch_evap@dssat-4.8.6.0:faithful")
    with pytest.raises(fs.SwapError, match="no process is registered"):
        fs.resolve("soil_water/nothing@none:demo")
    with pytest.raises(fs.SwapError, match="'salus'"):  # not even a key
        fs.resolve("not/a key")


# ------------------------------------------------------------------ the experiment file
def test_filex_edit_changes_only_the_mesev_character():
    new = fs.filex_with_mesev(_FILEX, "S")
    old_lines, new_lines = _FILEX.splitlines(), new.splitlines()
    assert len(old_lines) == len(new_lines)
    diff = [(a, b) for a, b in zip(old_lines, new_lines, strict=True) if a != b]
    assert len(diff) == 1  # level 2 already says S: only level 1's METHODS line changes
    a, b = diff[0]
    assert a.split()[:2] == ["1", "ME"]
    (k,) = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
    assert k == _MESEV_COLUMN and (a[k], b[k]) == ("R", "S")
    assert [_mesev_of(new, t) for t in (1, 2)] == ["S", "S"]
    back = fs.filex_with_mesev(new, "R")
    assert [_mesev_of(back, t) for t in (1, 2)] == ["R", "R"]
    assert back == _FILEX.replace("G     S     2", "G     R     2")  # nothing else moved


def test_the_edited_file_still_reads_as_the_same_experiment(tmp_path):
    a, b = tmp_path / "a.MZX", tmp_path / "b.MZX"
    a.write_text(_FILEX)
    b.write_text(fs.filex_with_mesev(_FILEX, "S"))
    x, y = read_filex(a), read_filex(b)
    assert x["TREATMENTS"] == y["TREATMENTS"] and x["PLANTING DETAILS"] == y["PLANTING DETAILS"]
    ma, mb = x["SIMULATION CONTROLS"][1]["METHODS"], y["SIMULATION CONTROLS"][1]["METHODS"]
    assert ma["MESEV"] == "R" and mb["MESEV"] == "S"
    assert {k: v for k, v in ma.items() if k != "MESEV"} == {k: v for k, v in mb.items() if k != "MESEV"}
    assert switches(x, 1).mesev == "R" and switches(y, 1).mesev == "S"


def test_a_file_without_a_methods_line_cannot_carry_the_option(tmp_path):
    assert "METHODS" not in _NO_METHODS
    p = tmp_path / "n.MZX"
    p.write_text(_NO_METHODS)
    assert fs.own_mesev(p, 1) == "R"  # DSSAT's default
    with pytest.raises(fs.SwapError, match=r"NOM8201 treatment 1 has no METHODS line .* 'ritchie'"):
        fs.filex_with_mesev(_NO_METHODS, "S", "NOM8201 treatment 1")


def test_a_methods_header_without_mesev_is_refused():
    old = "*SIMULATION CONTROLS\n@N METHODS     WTHER INCON LIGHT EVAPO\n 1 ME              M     M     E     R\n"
    with pytest.raises(fs.SwapError, match="no MESEV column"):
        fs.filex_with_mesev(old, "S")


def test_stage_filex_copies_what_the_runs_read_and_leaves_the_source(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    raw = _FILEX.replace("McCurdy 84aa", "McCurdy \xe9\xe8").encode("latin-1") + b"\x1a trailing junk"
    (src / "T8201.MZX").write_bytes(raw)
    (src / "T8201.MZT").write_text("observed")
    (src / "T8201.MZA").write_text("averages")
    (src / "OTHER.MZT").write_text("another experiment")
    (src / "MY.SOL").write_text("soil")
    staged = fs.stage_filex(src / "T8201.MZX", "S", tmp_path / "dest")
    assert staged == tmp_path / "dest" / "T8201.MZX"
    assert sorted(p.name for p in staged.parent.iterdir()) == [
        "MY.SOL",
        "T8201.MZA",
        "T8201.MZT",
        "T8201.MZX",
    ]
    assert (staged.parent / "T8201.MZT").read_text() == "observed"
    out = staged.read_bytes()
    assert b"McCurdy \xe9\xe8" in out  # the bytes are kept as they are
    assert b"\x1a" not in out and b"junk" not in out  # the DOS end-of-file mark ends the text
    assert fs.own_mesev(staged, 1) == "S" and fs.own_mesev(src / "T8201.MZX", 1) == "R"
    assert (src / "T8201.MZX").read_bytes() == raw


# ------------------------------------------------------------------ the experiment
def test_swap_code_is_none_when_nothing_changes(exp):
    assert fs.swap_code(exp, 1, None) is None
    assert fs.swap_code(exp, 1, "ritchie") is None and fs.swap_code(exp, 2, "salus") is None
    assert fs.swap_code(exp, 1, "salus") == "S" and fs.swap_code(exp, 2, "ritchie") == "R"
    with pytest.raises(fs.SwapError, match="'salus'"):
        fs.swap_code(exp, 1, "sauls")


def test_inputs_are_built_once_per_swap_from_the_staged_file(exp, built):
    base = exp.inputs(1)
    assert built == [exp.filex] and exp.inputs(1) is base
    assert exp.inputs(1, soil_evaporation="ritchie") is base  # the experiment's own: nothing is rebuilt
    s = exp.inputs(1, soil_evaporation="salus")
    assert s is not base and len(built) == 2
    staged = built[1]
    assert staged.parent.name == "swap_S" and staged.name == exp.filex.name and staged != exp.filex
    assert (
        fs.own_mesev(staged, 1) == "S" and fs.own_mesev(exp.filex, 1) == "R"
    )  # the experiment's file is untouched
    assert exp.inputs(1, soil_evaporation="SALUS") is s and exp.inputs(1, soil_evaporation="s") is s
    assert len(built) == 2
    # treatment 2 runs MESEV = S itself: salus is its own, ritchie is the swap
    assert exp.inputs(2, soil_evaporation="salus") is exp.inputs(2)
    r = exp.inputs(2, soil_evaporation="ritchie")
    assert built[-1].parent.name == "swap_R" and fs.own_mesev(built[-1], 2) == "R" and r is not exp.inputs(2)


def test_an_unknown_alternative_fails_before_anything_is_built(exp, built):
    with pytest.raises(fs.SwapError, match=r"'ritchie'.*'salus'"):
        exp.inputs(1, soil_evaporation="sauls")
    with pytest.raises(fs.SwapError, match=r"'ritchie'.*'salus'"):
        exp.run(1, soil_evaporation="sauls")
    with pytest.raises(fs.SwapError):
        exp.scenarios(1, soil_evaporation="nope")
    with pytest.raises(fs.SwapError):
        exp.reference(1, soil_evaporation="nope")
    with pytest.raises(fs.SwapError):
        exp.dssat_batch(1, {"G2": [800.0]}, soil_evaporation="nope")
    with pytest.raises(fs.SwapError):
        ajd.calibrate(exp, [1], soil_evaporation="nope")
    assert built == []


def test_reference_runs_dssat_on_the_same_swap(exp, monkeypatch):
    seen = _stop_run_ref(monkeypatch)
    for choice in ("salus", None, "ritchie"):
        with pytest.raises(_Boom):
            exp.reference(1, soil_evaporation=choice)
    assert "filex_edit" in seen[0] and "filex_edit" not in seen[1] and "filex_edit" not in seen[2]
    assert _mesev_of(seen[0]["filex_edit"](_FILEX), 1) == "S"
    with pytest.raises(_Boom):
        exp.reference(2, soil_evaporation="ritchie")
    assert _mesev_of(seen[-1]["filex_edit"](_FILEX), 2) == "R"


def test_reference_cache_tells_swaps_apart(exp, monkeypatch, tmp_path):
    calls: list[Any] = []

    def run_ref(self: Any, trno: Any, tag: str, **kw: Any) -> Any:
        calls.append(kw.get("filex_edit"))
        out = tmp_path / tag
        out.mkdir(exist_ok=True)
        return out, 0.0

    import agrijax.io.dssat as io_dssat

    monkeypatch.setattr(ajd.Experiment, "_run_ref", run_ref)
    monkeypatch.setattr(io_dssat, "read_summary", lambda p: SimpleNamespace(iloc=[SimpleNamespace(index=[])]))
    monkeypatch.setattr(ajd, "_reference_daily", lambda out: None)
    a = exp.reference(1)
    b = exp.reference(1, soil_evaporation="salus")
    assert a is not b and len(calls) == 2
    assert exp.reference(1, soil_evaporation="s") is b and exp.reference(1, soil_evaporation=None) is a
    assert exp.reference(1, soil_evaporation="ritchie") is a  # the experiment's own: the same DSSAT run
    assert len(calls) == 2


def test_dssat_batch_composes_the_swap_with_the_cultivar_levels(exp, built, monkeypatch):
    seen = _stop_run_ref(monkeypatch)
    monkeypatch.setattr(
        ajd.Experiment, "_cul_rows", lambda self, trno, samples, tag: (["AJ0001"], Path("x"), [{}])
    )
    for choice in (None, "salus"):
        with pytest.raises(_Boom):
            exp.dssat_batch(1, {"G2": [800.0]}, soil_evaporation=choice)
    plain, swapped = (kw["filex_edit"](_FILEX) for kw in seen)
    assert " 1 MZ AJ0001 AJ0001" in plain and " 1 MZ AJ0001 AJ0001" in swapped
    assert _mesev_of(plain, 1) == "R" and _mesev_of(swapped, 1) == "S"


def test_scenarios_are_built_from_the_swapped_file(exp, built):
    plain = exp.scenarios(1, years=[1982], sowing_shift=[0, 7])
    swapped = exp.scenarios(1, years=[1982], sowing_shift=[0, 7], soil_evaporation="salus")
    own = exp.scenarios(1, years=[1982], sowing_shift=[0], soil_evaporation="ritchie")
    assert [fs.own_mesev(f, 1) for f in plain.filex] == ["R", "R"]
    assert [fs.own_mesev(f, 1) for f in swapped.filex] == ["S", "S"]
    assert [fs.own_mesev(f, 1) for f in own.filex] == ["R"]
    assert len({f.parent for f in swapped.filex}) == 2 and all(
        f.name == exp.filex.name for f in swapped.filex
    )
    # the swapped files sit apart from the experiment's own: the plain scenarios still say R after the swapped ones
    assert not {f.parent for f in swapped.filex} & {f.parent for f in plain.filex}
    # the sowing shift still applies: the planting date of the +7 d scenario moved, the other did not
    assert "82057" in swapped.filex[0].read_text() and "82064" in swapped.filex[1].read_text()
    assert list(swapped.table["sowing_shift"]) == [0, 7]
    # the scenarios' own inputs were built from those files
    assert built[-1] == own.filex[0] and swapped.runs[1].filex == swapped.filex[1]


def test_run_batch_passes_the_swap_on(exp, built, monkeypatch):
    seen: dict[str, Any] = {}

    def scenarios(self: Any, treatment: int, **kw: Any) -> Any:
        seen.update(kw)
        return SimpleNamespace(run=lambda cultivar: ("ran", cultivar))

    monkeypatch.setattr(ajd.Experiment, "scenarios", scenarios)
    assert exp.run_batch(1, {"G2": [1.0]}, years=[1982], soil_evaporation="salus") == ("ran", {"G2": [1.0]})
    assert seen["soil_evaporation"] == "salus" and seen["years"] == [1982]


# ------------------------------------------------------------------ calibrate
def _capture_calibrate(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, dict[str, Any]]]:
    """``agrijax.calib.workflow.calibrate`` records what it is called with."""
    import agrijax.calib.workflow as wf

    calls: list[tuple[Any, dict[str, Any]]] = []

    def fake(experiment: Any, treatments: Any = None, cultivar: Any = None, **kw: Any) -> Any:
        calls.append((experiment, kw))
        return SimpleNamespace(notes=["existing note"])

    monkeypatch.setattr(wf, "calibrate", fake)
    return calls


def test_calibrate_runs_on_the_staged_file(exp, monkeypatch):
    calls = _capture_calibrate(monkeypatch)
    res = ajd.calibrate(exp, [1], soil_evaporation="salus")
    staged, kw = calls[0]
    assert Path(staged) != exp.filex and Path(staged).name == "SWAP8201.MZX"
    assert fs.own_mesev(staged, 1) == "S" and fs.own_mesev(exp.filex, 1) == "R"
    assert kw["inputs"] == "native" and kw["data"] == exp.data
    assert res.notes[0] == "existing note" and "salus" in res.notes[-1] and "MESEV" in res.notes[-1]
    # the swap to ritchie changes treatment 2's level (the file's treatments are swapped together)
    ajd.calibrate(exp, [2], soil_evaporation="ritchie")
    assert fs.own_mesev(calls[1][0], 2) == "R"


def test_calibrate_without_swap_is_untouched(exp, monkeypatch):
    calls = _capture_calibrate(monkeypatch)
    res = ajd.calibrate(exp, [1])
    assert calls[0][0] is exp and res.notes == ["existing note"]


def test_calibrate_on_the_experiments_own_choice_stages_nothing(tmp_path, monkeypatch):
    only_r = _experiment(tmp_path, monkeypatch, _FILEX.replace("G     S     2", "G     R     2"))
    calls = _capture_calibrate(monkeypatch)
    ajd.calibrate(only_r, [1], soil_evaporation="ritchie")
    assert calls[0][0] is only_r


def test_calibrate_by_name_and_path_and_several(exp, monkeypatch):
    calls = _capture_calibrate(monkeypatch)
    ajd.calibrate("SWAP8201", [1], soil_evaporation="salus")
    assert fs.own_mesev(calls[0][0], 1) == "S"
    ajd.calibrate(str(exp.filex), [1], soil_evaporation="salus")
    assert fs.own_mesev(calls[1][0], 1) == "S"
    ajd.calibrate([exp, str(exp.filex)], [1], soil_evaporation="salus")
    both = calls[2][0]
    assert isinstance(both, list) and [fs.own_mesev(p, 1) for p in both] == ["S", "S"]


def test_calibrate_swap_needs_the_native_inputs(exp, monkeypatch):
    calls = _capture_calibrate(monkeypatch)
    for inputs in ("tables", "auto"):
        with pytest.raises(fs.SwapError, match="needs the native inputs"):
            ajd.calibrate(exp, [1], soil_evaporation="salus", inputs=inputs)
    assert calls == []
    ajd.calibrate(exp, [1], soil_evaporation="salus", inputs="native")
    ajd.calibrate(exp, [1], inputs="tables")  # without a swap the tables stay allowed
    assert len(calls) == 2


def test_the_swap_keeps_the_facade_light():
    code = (
        "import sys, agrijax as aj; t = aj.dssat.alternatives(); "
        "from agrijax import facade_swap; assert facade_swap.resolve('Salus').mesev == 'S'; "
        "assert 'jax' not in sys.modules, 'jax imported'; print(list(t.index))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "ritchie" in out.stdout and "salus" in out.stdout
