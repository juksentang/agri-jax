"""ROOTWU port against the reference models' own ROOTWU (plan 19 A4) and the CA-TPA remap (A7).

Data (private, ``allow_skip``): the A12 dump tables in ``<data-dir>/dumps/tables``:

* ``dssat486/<EXP>_t<NN>_{rootwu_in,rootwu_out,spam}.npz`` - DSSAT-CSM v4.8.6.0 (a gfortran build
  of the BSD-3 source instrumented at ROOTWU entry / exit and SPAM exit, printed outputs identical
  to build486) on the 58 M2 maize treatments;
* ``rzwqm46_catpa2015{,_2023}/{rootwu,dssatdrv,physcl}_{entry,exit}.npz`` - RZWQM2 4.6 at CA-TPA,
  whose embedded DSSAT crop calls its own ROOTWU from ``DSSATDRV`` after the day's soil physics.

Checks:

1. :func:`rootwu_estimate` on the dumped entry arguments gives the dumped ``RWU``, ``TRWUP`` to
   REAL*4 rounding and ``TSS`` exactly, on every call of every treatment; in float64 it equals a
   plain-Python transcription of the Fortran to 1e-12 on the same (float32-promoted) inputs;
2. the counter ``TSS`` carried by us from zero at the start of the season equals the Fortran's
   ``SAVE`` on every call; driven by SPAM's daily ``RLV``, ``SW`` and ``XHLAI`` on every day, the
   estimate reproduces SPAM's daily ``TRWUP`` including the days without canopy (not called);
3. the same kernel reproduces RZWQM2's embedded ROOTWU (DSSAT 4.0 lineage) at CA-TPA;
4. the grids of :mod:`agrijax.core.grids` are RZWQM2's: the node grid from the dumped ``TLT``, the
   crop layers ``SOILPROP%DS``, the node -> layer map of the soil water (``SW`` at ROOTWU entry
   from the end-of-physics node ``THETA``) and the layer -> node map of the uptake ``qsr``;
5. (with the local ``dscsm048``) the root record CERES publishes on day ``d - 1`` is the record
   ROOTWU reads on day ``d``, and our chain crop record -> ROOTWU gives DSSAT's ``TRWUP``;
6. (with the local DSSAT-CSM v4.8.6.0 source tree) every declared coefficient of ROOTWU, LYRSET
   and the crop's water-stress interface stands, with its quoted statement, on the line it cites,
   and (with the local RZWQM2 source tree) RZWQM2's embedded-crop layers use the same LYRSET
   numbers (values only: no RZWQM2 statement is quoted).
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core.coefficients import coefficient_table
from agrijax.core.grids import LyrsetCoefficients, SoilGrid, remap_intensive, rzwqm_lyrset, rzwqm_nodes
from agrijax.port import dumps
from agrijax.processes.crop.ceres_maize.growth import WaterStressCoefficients
from agrijax.processes.soil_water.uptake import RootRecord, RootwuCoefficients, SoilView, rootwu_estimate

pytestmark = [
    pytest.mark.allow_skip(reason="the ROOTWU dumps are private data"),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the dump comparison runs in float64"),
]

#: REAL*4 rounding of the Fortran: a few ulp per operation (measured max 5.6e-7)
F32_REL = 2e-6
_EST = jax.jit(rootwu_estimate)


@pytest.fixture(scope="module")
def dssat_tables(data_dir: Path) -> Path:
    d = data_dir / "dumps" / "tables" / "dssat486"
    if not (d / "collect_report.json").is_file():
        pytest.skip(f"no DSSAT A12 tables at {d}")
    return d


def _keys(d: Path) -> list[str]:
    return sorted(p.name.removesuffix("_spam.npz") for p in d.glob("*_spam.npz"))


def _promote(values: dict, nl: int, key: str) -> jnp.ndarray:
    return jnp.asarray(values[key][:, :nl].astype(np.float64))


def _call(values: dict, nl: int, tss, xhlai=None):
    """ROOTWU on dumped entry arguments (float32 promoted to float64)."""
    n = len(values["NLAYR"])
    root = RootRecord(
        rlv=_promote(values, nl, "RLV"),
        rtdep=jnp.zeros(n),
        rwumx=jnp.asarray(values["RWUMX"].astype(np.float64)),
        pormin=jnp.asarray(values["PORMIN"].astype(np.float64)),
        xhlai=jnp.ones(n) if xhlai is None else jnp.asarray(xhlai),
    )
    soil = SoilView(
        dlayr=_promote(values, nl, "DLAYR"),
        ll=_promote(values, nl, "LL"),
        sat=_promote(values, nl, "SAT"),
        sw=_promote(values, nl, "SW"),
    )
    return _EST(root, soil, tss)


def _loop(v: dict, nl: int, k: int) -> tuple[list[float], float, list[float]]:
    """Plain-Python ROOTWU of call ``k`` (float64 on the promoted REAL*4 arguments)."""
    tss = [float(x) for x in v["TSS"][k, :nl]]
    rwu = []
    for L in range(nl):
        ll, sw, sat, rlv = (float(v[n][k, L]) for n in ("LL", "SW", "SAT", "RLV"))
        rwumx, pormin = float(v["RWUMX"][k]), float(v["PORMIN"][k])
        swcon2 = 45.0 if ll > 0.30 else 120.0 - 250.0 * ll
        r = 0.0
        if not (rlv <= 0.00001 or sw <= ll):
            den = 7.01 - math.log(7.01) if rlv > math.exp(7.01) else 7.01 - math.log(rlv)
            r = 1.32e-3 * math.exp(min(swcon2 * (sw - ll), 40.0)) / den
            tss[L] = 0.0 if (sat - sw) >= pormin or pormin < 1e-6 else tss[L] + 1.0
            swexf = min(max((sat - sw) / pormin, 0.0), 1.0) if tss[L] > 2.0 else 1.0
            r = min(r, rwumx * swexf, rwumx)
        rwu.append(r * float(v["DLAYR"][k, L]) * rlv)
    return rwu, sum(rwu), tss


def _check_against_exit(got, out: dict, nl: int, where: str) -> None:
    ref = out["RWU"][:, :nl].astype(np.float64)
    rwu = np.asarray(got.rwu)
    np.testing.assert_allclose(rwu, ref, rtol=F32_REL, atol=1e-9, err_msg=where)
    np.testing.assert_allclose(
        np.asarray(got.trwup), out["TRWUP"].astype(np.float64), rtol=F32_REL, atol=1e-9
    )
    np.testing.assert_array_equal(np.asarray(got.tss), out["TSS"][:, :nl], err_msg=where)


# ------------------------------------------------------------------ 1-2: DSSAT-CSM v4.8.6.0
def test_estimate_equals_dssat_rootwu_on_every_call(dssat_tables: Path) -> None:
    n_calls = n_active = 0
    for key in _keys(dssat_tables):
        ri = dumps.load_table(dssat_tables / f"{key}_rootwu_in.npz")[0]
        ro = dumps.load_table(dssat_tables / f"{key}_rootwu_out.npz")[0]
        np.testing.assert_array_equal(ri.date, ro.date)
        nl = int(ri.values["NLAYR"][0])
        got = _call(ri.values, nl, _promote(ri.values, nl, "TSS"))
        _check_against_exit(got, ro.values, nl, key)
        # float64: a second implementation (plain Python) on the same promoted inputs, 1e-12
        for k in range(0, len(ri.date), 7):
            rwu, trwup, tss = _loop(ri.values, nl, k)
            np.testing.assert_allclose(np.asarray(got.rwu[k]), rwu, rtol=1e-12, atol=1e-18)
            np.testing.assert_allclose(float(got.trwup[k]), trwup, rtol=1e-12, atol=1e-18)
            np.testing.assert_array_equal(np.asarray(got.tss[k]), tss)
        n_calls += len(ri.date)
        n_active += int((ro.values["RWU"][:, :nl] > 0).sum())
    assert n_calls > 6000 and n_active > 30000, (n_calls, n_active)


def test_counter_is_carried_from_the_season_start(dssat_tables: Path) -> None:
    """Our TSS, carried from zero (SEASINIT) through every call, is the Fortran's SAVEd TSS."""
    n_sat = 0
    for key in _keys(dssat_tables):
        ri = dumps.load_table(dssat_tables / f"{key}_rootwu_in.npz")[0]
        ro = dumps.load_table(dssat_tables / f"{key}_rootwu_out.npz")[0]
        nl = int(ri.values["NLAYR"][0])
        np.testing.assert_array_equal(ri.values["TSS"][0, :nl], 0.0)
        tss = jnp.zeros((nl,))
        for k in range(len(ri.date)):
            one = {n: v[k : k + 1] for n, v in ri.values.items()}
            np.testing.assert_array_equal(
                np.asarray(tss), ri.values["TSS"][k, :nl], err_msg=f"{key} call {k}"
            )
            tss = _call(one, nl, tss[None, :]).tss[0]
        np.testing.assert_array_equal(np.asarray(tss), ro.values["TSS"][-1, :nl])
        n_sat += int((ro.values["TSS"][:, :nl] > 2).sum())
    assert n_sat > 0  # the two-day delay of the excess-water factor is exercised


def test_daily_trwup_from_spam_records_including_days_without_canopy(dssat_tables: Path) -> None:
    """Driven by SPAM's own daily RLV, SW, XHLAI (soil arrays as ROOTWU received them), the
    estimate gives SPAM's TRWUP on every simulated day; on days with XHLAI = 0 it is 0 and the
    counter is untouched (SPAM does not call ROOTWU)."""
    n_off = 0
    for key in _keys(dssat_tables):
        sp = dumps.load_table(dssat_tables / f"{key}_spam.npz")[0]
        ri = dumps.load_table(dssat_tables / f"{key}_rootwu_in.npz")[0]
        nl = int(ri.values["NLAYR"][0])
        v = dict(sp.values)
        # the soil arrays of the ROOTWU call of the day (they change with soil dynamics, e.g.
        # GHWA0401); days without a call take the latest call's (unused: ROOTWU is not called)
        call = np.searchsorted(ri.date, sp.date, side="right") - 1
        call = np.clip(call, 0, len(ri.date) - 1)
        for name in ("DLAYR", "LL", "SAT", "RWUMX", "PORMIN", "NLAYR"):
            v[name] = ri.values[name][call]
        n = len(sp.date)
        tss = jnp.zeros((nl,))
        trwup = []
        for k in range(n):
            one = {nm: np.asarray(x[k : k + 1]) for nm, x in v.items() if nm in _ARGS}
            r = _call(one, nl, tss[None, :], xhlai=np.asarray(v["XHLAI"][k : k + 1], dtype=np.float64))
            tss = r.tss[0]
            trwup.append(float(r.trwup[0]))
        np.testing.assert_allclose(
            trwup, sp.values["TRWUP"].astype(np.float64), rtol=F32_REL, atol=1e-9, err_msg=key
        )
        off = sp.values["XHLAI"] <= 0
        assert np.all(np.asarray(trwup)[off] == 0.0)
        n_off += int(off.sum())
    assert n_off > 500


_ARGS = ("DLAYR", "LL", "SAT", "SW", "RLV", "RWUMX", "PORMIN", "NLAYR")


# ------------------------------------------------------------------ 3-4: RZWQM2 4.6 at CA-TPA
RZ_RUNS = ("catpa2015", "catpa2015_2023")


@pytest.fixture(params=RZ_RUNS)
def rz_tables(request: pytest.FixtureRequest, data_dir: Path) -> Path:
    d = data_dir / "dumps" / "tables" / f"rzwqm46_{request.param}"
    if not (d / "rootwu_entry.npz").is_file():
        pytest.skip(f"no RZWQM2 A12 tables at {d}")
    return d


def test_estimate_equals_rzwqm_embedded_rootwu(rz_tables: Path) -> None:
    ri = dumps.load_table(rz_tables / "rootwu_entry.npz")[0]
    ro = dumps.load_table(rz_tables / "rootwu_exit.npz")[0]
    assert set(ri.values["ISTRESS"].tolist()) == {0}
    nl = int(ri.values["NLAYR"][0])
    got = _call(ri.values, nl, _promote(ri.values, nl, "TSS"))
    _check_against_exit(got, ro.values, nl, str(rz_tables))
    assert len(ri.date) > 100 and float(np.max(np.asarray(got.trwup))) > 0.0


def test_grids_and_remap_are_rzwqm_realmatch(rz_tables: Path) -> None:
    ph = dumps.load_table(rz_tables / "physcl_exit.npz")[0]
    dd_in = dumps.load_table(rz_tables / "dssatdrv_entry.npz")[0]
    dd = dumps.load_table(rz_tables / "dssatdrv_exit.npz")[0]
    ri = dumps.load_table(rz_tables / "rootwu_entry.npz")[0]
    nn, nl = int(ph.values["NN"][0]), int(ri.values["NLAYR"][0])
    nodes = rzwqm_nodes(ph.values["TLT"][0, :nn].astype(np.float64))
    layers = rzwqm_lyrset(nodes)
    assert nodes.n == 37 and nodes.depth == 150.0
    assert layers.bottom == tuple(dd.values["SOILPROP%DS"][0, :nl].astype(np.float64).tolist())
    np.testing.assert_array_equal(layers.thickness, dd.values["SOILPROP%DLAYR"][0, :nl])
    # nodes -> layers: SW at ROOTWU entry is the thickness-weighted mean of the day's end-of-physics THETA
    j = ph.index_of(ri.date)
    assert np.all(j >= 0)
    sw = np.asarray(
        remap_intensive(jnp.asarray(ph.values["THETA"][j, :nn].astype(np.float64)), nodes, layers)
    )
    np.testing.assert_allclose(ri.values["SW"][:, :nl], sw, rtol=0, atol=2e-7)  # REAL*4 of the mean
    # layers -> nodes: node qsr is the thickness-weighted mean of the layer rates rwu / 24 / dlayr (SW > LL);
    # a layer with SW == LL exactly keeps element L of the array as DSSATDRV found it (quirk, A12 note)
    v = dd.values
    swl, ll = v["SW"][:, :nl], v["SOILPROP%LL"][:, :nl]
    ql = np.where(swl > ll, v["RWU"][:, :nl] / 24.0 / v["SOILPROP%DLAYR"][:, :nl], 0.0).astype(np.float32)
    ql = np.where(swl == ll, dd_in.values["QSR"][:, :nl], ql).astype(np.float64)
    qn = np.asarray(remap_intensive(jnp.asarray(ql), layers, nodes))
    np.testing.assert_allclose(v["QSR"][:, :nn], qn, rtol=1e-6, atol=1e-12)
    assert float(np.max(qn)) > 0.0
    # the operator is the thickness-weighted overlap: the column totals of the map agree
    np.testing.assert_allclose(qn @ nodes.thickness, ql @ layers.thickness, rtol=1e-12)
    assert isinstance(layers, SoilGrid)


# ------------------------------------------------------------------ 5: CERES publishes what ROOTWU reads
@pytest.mark.parametrize(
    ("exp", "trno"), [("UFGA8201", 1), ("UFGA8201", 5), ("IUAF9901", 1), ("SIAZ9501", 3)]
)
def test_crop_root_record_is_what_rootwu_reads(
    dssat_tables: Path, tmp_path: Path, exp: str, trno: int
) -> None:
    """CERES driven by DSSAT's full-precision end-of-day SW (SPAM's start-of-day SW of the next
    day; the printed SoilWat.OUT is within half its print unit of it) publishes on day d - 1 the
    RLV, RWUMX, PORMIN and XHLAI that ROOTWU receives on day d; ROOTWU on our record and the soil
    it received gives DSSAT's TRWUP."""
    import test_ceres_dssat as t

    if not t.DSCSM.is_file() or not t.MAIZE.is_dir():
        pytest.skip(f"dscsm048 / DSSAT example data not found under {t.DSSAT_ENGINE}")
    from agrijax.core import run
    from agrijax.processes.crop.ceres_maize import CeresMaizeState, ceres_maize_model

    out = t.run_reference(exp, trno, tmp_path / "ref")
    p, f, _, _ = t.simulate(out, trno, exp=exp, tables=dssat_tables)
    key = f"{exp}_t{trno:02d}"
    sp = dumps.load_table(dssat_tables / f"{key}_spam.npz")[0]
    ri = dumps.load_table(dssat_tables / f"{key}_rootwu_in.npz")[0]
    ro = dumps.load_table(dssat_tables / f"{key}_rootwu_out.npz")[0]
    nl = int(p.soil.dlayr.shape[0])
    days = np.asarray(f.yrdoy)
    pos = {int(d): k for k, d in enumerate(sp.date.tolist())}
    nxt = np.asarray([pos.get(int(d), -2) + 1 for d in days], dtype=np.intp)
    have = (nxt > 0) & (nxt < len(sp.date))
    sw = np.asarray(f.sw).copy()
    sw_spam = np.asarray(sp.values["SW"], dtype=np.float64)
    assert np.abs(sw[have] - sw_spam[nxt[have], :nl]).max() <= 5e-4 + 1e-6
    sw[have] = sw_spam[nxt[have], :nl]
    f = f.replace(sw=jnp.asarray(sw))
    s = run(ceres_maize_model(outputs=lambda st, pp, ff: st.root_out), p, f, CeresMaizeState.initial(p, 1))
    j = np.searchsorted(days, ri.date) - 1  # the record published the day before each call
    assert np.all(days[j + 1] == ri.date) and np.all(j >= 0)
    # RLV is truncated to 1e-3 in MZ_ROOTGR: float32 vs float64 may land one quantum apart
    np.testing.assert_allclose(np.asarray(s.rlv[j, 0]), ri.values["RLV"][:, :nl], rtol=0, atol=1e-3 + 1e-6)
    np.testing.assert_allclose(np.asarray(s.rwumx[j, 0]), ri.values["RWUMX"], rtol=1e-7)
    np.testing.assert_allclose(np.asarray(s.pormin[j, 0]), ri.values["PORMIN"], rtol=1e-7)
    np.testing.assert_allclose(
        np.asarray(s.xhlai[j, 0]), sp.values["XHLAI"][sp.index_of(ri.date)], rtol=0, atol=5e-4
    )
    # the chain: our record + the soil ROOTWU received (its REAL*4 LL / SAT / DLAYR / SW)
    tss = jnp.zeros((1, nl))
    trwup = []
    for k in range(len(ri.date)):
        root = RootRecord(
            rlv=s.rlv[j[k]],
            rtdep=s.rtdep[j[k]],
            rwumx=s.rwumx[j[k]],
            pormin=s.pormin[j[k]],
            xhlai=s.xhlai[j[k]],
        )
        r = _EST(root, SoilView(*(_promote(ri.values, nl, n)[k] for n in ("DLAYR", "LL", "SAT", "SW"))), tss)
        tss = r.tss
        trwup.append(float(r.trwup[0]))
    ref = ro.values["TRWUP"].astype(np.float64)
    rel = np.abs(np.asarray(trwup) - ref) / np.maximum(ref, 1e-9)
    # float32 level (median 5e-8); one RLV truncation quantum apart moves a day by <= 1.5e-4 (IUAF9901)
    assert np.median(rel) < 1e-6 and rel.max() < 1e-3, (np.median(rel), rel.max())
    np.testing.assert_array_equal(np.asarray(tss)[0], ro.values["TSS"][-1, :nl])


# ------------------------------------------------------------------ 6. coefficients vs the source
DSSAT_SOURCE = (
    Path(os.environ.get("AGRI_JAX_DSSAT", "~/AFSoil/Formal_Analysis/02_DSSAT/dssat_engine")).expanduser()
    / "source"
)
RZWQM_SOURCE = (
    Path(
        os.environ.get("AGRI_JAX_RZWQM_SRC", "~/agri_jax_data/narval_mirror/RZWQM_Linux_Ver45/src")
    ).expanduser()
    / "RZWQM"
)
_NUM = re.compile(r"(?<![A-Za-z_0-9])(\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+|\d+(?:[eE][+-]?\d+)?)")
COEF_ROWS = [
    *(dict(r, group="rootwu") for r in coefficient_table(RootwuCoefficients)),
    *(dict(r, group="lyrset") for r in coefficient_table(LyrsetCoefficients)),
    *(dict(r, group="water_stress") for r in coefficient_table(WaterStressCoefficients)),
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text).upper()


@pytest.mark.parametrize("row", COEF_ROWS, ids=[f"{r['group']}.{r['path']}" for r in COEF_ROWS])
def test_coefficient_stands_on_its_cited_dssat_line(row: dict) -> None:
    path = DSSAT_SOURCE / row["file"]
    if not path.is_file():
        pytest.skip(f"DSSAT-CSM v4.8.6.0 source not found at {path}")
    assert row["ref_version"] == "dssat-4.8.6.0"
    line = path.read_text(errors="replace").splitlines()[row["line"] - 1]
    assert line[:1] not in "!cC*", f"{row['file']}:{row['line']} is a comment line: {line!r}"
    code = line.split("!")[0]
    assert _norm(row["statement"].split("!")[0]) in _norm(code), (row["statement"], line)
    assert float(row["value"]) in [float(x) for x in _NUM.findall(code)], (row["value"], line)


def test_rzwqm2_embedded_crop_layers_use_the_lyrset_numbers() -> None:
    """RZWQM2 4.6 ``DSSATDRV`` sets the same fixed bottoms, the +30 cm step and 20 layers (values
    compared on the lines, no statement quoted)."""
    path = RZWQM_SOURCE / "DSSATDRV.for"
    if not path.is_file():
        pytest.skip(f"RZWQM2 source not found at {path}")
    lines = path.read_text(errors="replace").splitlines()
    c = LyrsetCoefficients()

    def nums(lineno: int) -> list[float]:
        return [float(x) for x in _NUM.findall(lines[lineno - 1].split("!")[0])]

    for k, (lineno, want) in enumerate(zip(range(557, 562), c.fixed, strict=True)):
        assert nums(lineno) == [k + 1, want], (lineno, lines[lineno - 1])
    assert c.step in nums(563) and "DS(I - 1)" in lines[562].upper().replace("  ", " ")
    assert 20 in nums(69) and c.n_max == 20
