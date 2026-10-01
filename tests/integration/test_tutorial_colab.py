"""The Colab tutorial's path end to end, against ``dscsm048`` (slow).

* **DSSAT's data from DSSAT's repositories.** :func:`agrijax.io.examples.fetch_dssat_data` on a fresh
  root (from the download cache ``AGRI_JAX_CACHE`` when the machine has no network; every file
  SHA-256 checked) gives the same files as the validation engine, and the DSSAT-free inputs built
  from them equal the ones built from the engine's.
* **The reference bundle.** The prebuilt bundle (``scripts/release/make_dssat_bundle.py``, from
  ``AGRI_JAX_BUNDLE`` or the cache's ``release/``) unpacked into a fresh directory runs UFGA8201
  treatment 4 exactly as the validation engine does, and Agri-JAX agrees with it.
* **DSSAT built from the GitHub archive** (the fallback of ``install_reference(build=True)``):
  reproduces the reference engine's ``Summary.OUT`` and daily ``PlantGro.OUT`` on treatments 4 and 6.
* **The facade on UFGA8201.** One season without DSSAT against ``dscsm048``; the weather-year x
  sowing-date scenarios at the published cultivar against ``dscsm048`` on each scenario; perturbed
  cultivars against a ``dscsm048`` batch run of the same (written) coefficients.
* **calibrate: native inputs (no DSSAT run) = tables.** Same written coefficients and objective values.
* **The notebook.** ``tutorial/agrijax_tutorial.ipynb`` executed top to bottom (code cells in one
  namespace, shell lines skipped, ``AGRI_JAX_TUTORIAL_SMALL=1``).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import jax
import numpy as np
import pytest

from agrijax.port.run_fortran import DSSAT_ENGINE, dscsm_paths

pytestmark = [
    pytest.mark.slow,
    pytest.mark.allow_skip(reason="needs dscsm048 v4.8.6.0 (AGRI_JAX_DSSAT) and the DSSAT example data"),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the DSSAT day runs in float64"),
]

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "tutorial" / "agrijax_tutorial.ipynb"
#: native season against dscsm048 (the free-run acceptance: yield within 2 %, measured 3e-5 here)
SEASON_YIELD_RTOL = 1e-3
#: scenarios and cultivar samples against dscsm048 (HWAM printed to 1 kg ha-1; measured 5.5e-5
#: on 30 scenarios and 4.9e-4 on 20 perturbed cultivars)
BATCH_YIELD_RTOL = 1e-3


@pytest.fixture(scope="module")
def engine() -> Path:
    if not dscsm_paths(DSSAT_ENGINE)[0].is_file():
        pytest.skip(f"dscsm048 not found under {DSSAT_ENGINE}")
    return DSSAT_ENGINE


@pytest.fixture(scope="module")
def exp(engine: Path):
    import agrijax as aj

    return aj.dssat.experiment("UFGA8201", data_root=engine)


def _cached_or_skip(url: str, dest: Path, sha: str) -> None:
    import agrijax.io.examples as ex

    if not dest.is_file():
        try:
            ex.download(url, dest, sha, timeout=20)
        except OSError as e:
            pytest.skip(f"not in the download cache and no network: {e}")


def _same_runs(a: Path, b: Path, t: int) -> None:
    from agrijax.io.dssat import read_plantgro, read_summary

    sa, sb = read_summary(a / "Summary.OUT").iloc[0], read_summary(b / "Summary.OUT").iloc[0]
    for c in ("HWAM", "CWAM", "H#AM", "ADAT", "MDAT", "LAIX", "PRCM", "ETCM"):
        if c in sb.index:
            assert sa[c] == sb[c] or (np.isnan(sa[c]) and np.isnan(sb[c])), (t, c, sa[c], sb[c])
    pa, pb = read_plantgro(a / "PlantGro.OUT"), read_plantgro(b / "PlantGro.OUT")
    for c in ("LAID", "CWAD", "GWAD", "SWAD", "LWAD"):
        np.testing.assert_array_equal(pa[c].to_numpy(), pb[c].to_numpy(), err_msg=f"t{t} {c}")


def test_dssat_data_from_the_repositories_equal_the_engine_s(engine, tmp_path_factory):
    import agrijax.io.examples as ex
    from agrijax.sites.dssat_free_run import free_run_inputs

    _cached_or_skip(
        ex.DSSAT_CSM_OS.url,
        ex.cache_dir() / "downloads" / f"dssat-csm-os-{ex.DSSAT_CSM_OS.commit}.tar.gz",
        ex.DSSAT_CSM_OS.sha256,
    )
    store = ex.cache_dir() / "downloads" / f"dssat-csm-data-{ex.DSSAT_CSM_DATA_COMMIT}"
    for rel, sha in ex.EXAMPLE_FILES.items():
        _cached_or_skip(ex._DATA_URL.format(commit=ex.DSSAT_CSM_DATA_COMMIT, path=rel), store / rel, sha)
    root = ex.fetch_dssat_data(tmp_path_factory.mktemp("dssat_data"))
    for rel in ex.EXAMPLE_FILES:
        assert (root / "example_data" / rel).read_bytes() == (engine / "example_data" / rel).read_bytes(), rel
    for rel in ex.DATA_FILES:
        assert (root / rel).read_bytes() == (engine / "source" / rel).read_bytes(), rel
    for t in (4, 6):
        a = free_run_inputs(
            root / "example_data" / "Maize" / "UFGA8201.MZX", t, None, engine=root, source="native"
        )
        b = free_run_inputs(
            engine / "example_data" / "Maize" / "UFGA8201.MZX", t, None, engine=engine, source="native"
        )
        np.testing.assert_array_equal(a.days, b.days)
        for x, y in zip(jax.tree.leaves(a.forcing()), jax.tree.leaves(b.forcing()), strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(jax.tree.leaves(a.params()), jax.tree.leaves(b.params()), strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_the_reference_bundle_runs_the_example(engine, exp, tmp_path_factory):
    import agrijax.io.examples as ex
    from agrijax.sites.dssat_free_run import run_reference

    tgz = Path(os.environ.get("AGRI_JAX_BUNDLE") or ex.cache_dir() / "release" / ex.REFERENCE_BUNDLE.name)
    if not tgz.is_file():
        pytest.skip(f"reference bundle not found at {tgz}")
    assert ex.sha256_of(tgz) == ex.REFERENCE_BUNDLE.sha256
    root = ex.unpack_bundle(tgz, tmp_path_factory.mktemp("bundle") / "dssat")
    assert (root / "LICENSE").is_file() and "build486" in (root / "PROVENANCE.txt").read_text()
    exe = dscsm_paths(root)[0]
    assert exe == root / "bin" / "dscsm048" and exe.read_bytes() == dscsm_paths(engine)[0].read_bytes()
    work = tmp_path_factory.mktemp("bundle_runs")
    fx = engine / "example_data" / "Maize" / "UFGA8201.MZX"
    wx, sx = engine / "example_data" / "Weather", engine / "example_data" / "Soil"
    a = run_reference(fx, 4, work / "bundle", engine=root, weather_dir=wx, soil_dir=sx)
    b = run_reference(fx, 4, work / "engine", engine=engine)
    _same_runs(a, b, 4)
    # Agri-JAX (no DSSAT) against the bundle's run
    import agrijax as aj

    e2 = aj.dssat.experiment("UFGA8201", data_root=engine)
    e2._engine = root
    ref = e2.reference(treatment=4)
    s = exp.run(treatment=4).compare_summary(ref)
    assert s.loc["MDAT", "difference"] == 0 and abs(s.loc["HWAM", "difference"]) < SEASON_YIELD_RTOL


def test_dssat_built_from_the_github_archive_matches_the_reference(engine, tmp_path_factory, monkeypatch):
    import agrijax.io.examples as ex
    from agrijax.sites.dssat_free_run import run_reference

    if shutil.which("gfortran") is None or shutil.which("cmake") is None:
        pytest.skip("gfortran / cmake not available")
    _cached_or_skip(
        ex.DSSAT_CSM_OS.url,
        ex.cache_dir() / "downloads" / f"dssat-csm-os-{ex.DSSAT_CSM_OS.commit}.tar.gz",
        ex.DSSAT_CSM_OS.sha256,
    )
    root = tmp_path_factory.mktemp("dssat_gh")
    monkeypatch.setenv("AGRI_JAX_DSSAT", "")
    monkeypatch.setattr(ex, "_use", lambda _e: None)  # keep the session's default engine
    t0 = time.perf_counter()
    eng = ex.install_reference(root, build=True, jobs=int(os.environ.get("AGRI_JAX_BUILD_JOBS", "2")))
    print(
        f"install_reference(build=True) (archive checked, built with 2 jobs): {time.perf_counter() - t0:.0f} s"
    )
    assert dscsm_paths(eng)[0] == eng / "source" / "build486" / "bin" / "dscsm048"
    fx = engine / "example_data" / "Maize" / "UFGA8201.MZX"
    wx, sx = engine / "example_data" / "Weather", engine / "example_data" / "Soil"
    work = tmp_path_factory.mktemp("dssat_cmp")
    for t in (4, 6):
        a = run_reference(fx, t, work / f"gh{t}", engine=eng, weather_dir=wx, soil_dir=sx)
        b = run_reference(fx, t, work / f"ref{t}", engine=engine)
        _same_runs(a, b, t)


def test_one_season_against_dssat(exp):
    import agrijax as aj
    from agrijax.sites.dssat_free_run import free_run_inputs

    season = exp.run(treatment=4)  # no DSSAT run
    assert season.inputs.out is None and season.inputs.notes["dssat_free"]
    ref = exp.reference(treatment=4)
    s = season.compare_summary(ref)
    assert s.loc["ADAT", "difference"] == 0 and s.loc["MDAT", "difference"] == 0
    assert abs(s.loc["HWAM", "difference"]) < SEASON_YIELD_RTOL
    d = season.compare_daily(ref)
    assert (d["RMSE / max"] < 2e-3).all(), d
    assert set(exp.observed(4).columns) >= {"date", "yrdoy", "lai", "cwad", "gwad"}
    # the reference-run inputs (no extension) and the DSSAT-free ones agree over the season
    plain = free_run_inputs(exp.filex, 4, ref.out, engine=exp.data, source="native")
    ext = exp.inputs(4)
    assert ext.n_ext == aj.dssat.PAD_DAYS and plain.n_ext == 0
    np.testing.assert_array_equal(plain.days, ext.days)
    fa, fb = plain.forcing(plain.n_days), ext.forcing(plain.n_days)
    for a, b in zip(jax.tree.leaves(fa), jax.tree.leaves(fb), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # past the season: real rain on the extension, none on the padding
    rain = np.asarray(ext.forcing(ext.n_days + 70)["soil"].rain)
    assert rain[ext.n_days : ext.n_days + 60].sum() > 0 and rain[ext.n_days + 60 :].sum() == 0


def test_scenarios_and_cultivar_batches_against_dssat(exp):
    scen = exp.scenarios(treatment=4, years=[1979, 1982, 1985], sowing_shift=[-14, 0, 14])
    assert len(scen.runs) == 9 and scen.dssat_s == []  # no DSSAT run so far
    dss = scen.reference()
    assert len(dss) == 9 and len(scen.dssat_s) == 9
    rng = np.random.default_rng(0)
    pub = scen.published
    cul = {n: pub[n] * rng.uniform(0.9, 1.1, 16) for n in ("P1", "P5", "G2", "G3", "PHINT")}
    for n in cul:
        cul[n][0] = pub[n]
    first = scen.run(cul)
    assert first.timing["seasons"] == 9 * 16 and first.timing["compile_s"] > 0
    again = scen.run({n: v[::-1] for n, v in cul.items()})
    assert again.timing["compile_s"] == 0.0  # same program
    t = first.table[first.table["sample"] == 0].merge(dss, on=["year", "sowing_shift"], suffixes=("", "_d"))
    assert len(t) == 9
    assert (np.abs(t["HWAM"] - t["HWAM_d"]) / t["HWAM_d"]).max() < BATCH_YIELD_RTOL
    assert (t["ADAT"] == t["ADAT_d"]).all() and (t["MDAT"] == t["MDAT_d"]).all()
    db = exp.dssat_batch(treatment=4, cultivar={n: v[:12] for n, v in cul.items()})
    assert db.seasons == 12 and db.elapsed_s > 0
    ours = scen.run(db.cultivar).table
    ours = ours[(ours["year"] == 1982) & (ours["sowing_shift"] == 0)].reset_index(drop=True)
    assert (np.abs(ours["HWAM"] - db.table["HWAM"]) / db.table["HWAM"]).max() < BATCH_YIELD_RTOL
    assert (ours["ADAT"] == db.table["ADAT"]).all() and (ours["MDAT"] == db.table["MDAT"]).all()
    # a changed cultivar in one season agrees with dscsm048 run on the same written row
    c = {"P1": float(db.table.loc[3, "P1"]), "G2": float(db.table.loc[3, "G2"])}
    s = exp.run(treatment=4, cultivar=c)
    r = exp.reference(treatment=4, cultivar=c)
    assert r.cultivar is not None and r.cultivar["P1"] == c["P1"]
    assert abs(s.summary["HWAM"] - r.summary["HWAM"]) / r.summary["HWAM"] < BATCH_YIELD_RTOL
    assert s.summary["MDAT"] == r.summary["MDAT"]


def test_calibrate_native_inputs_equal_the_tables(engine, data_dir):
    from agrijax.calib import calibrate
    from agrijax.sites.dssat_free_run import missing_tables

    miss = [p for t in (4, 6) for p in missing_tables("UFGA8201", t, data_dir)]
    if miss:
        pytest.skip(f"free-run tables missing: {miss[:2]}")
    kw = {"treatments": [4], "holdout": [6], "starts": 2, "budget": 96, "seed": 0, "engine": engine}
    with pytest.warns(Warning):
        nat = calibrate("UFGA8201", inputs="native", **kw)
    with pytest.warns(Warning):
        tab = calibrate("UFGA8201", inputs="tables", data_dir=data_dir, **kw)
    assert nat.inputs == "native" and tab.inputs == "tables"
    print("native", nat.params, nat.loss, "\ntables", tab.params, tab.loss)
    assert nat.params == tab.params
    assert nat.free == tab.free
    for k in ("written", "unrounded", "published"):
        assert nat.loss[k] == pytest.approx(tab.loss[k], rel=1e-9, abs=1e-12), k
    assert nat.holdout is not None and tab.holdout is not None
    assert nat.holdout["loss_written"] == pytest.approx(tab.holdout["loss_written"], rel=1e-9)


def _code_of(nb: Path) -> str:
    cells = json.loads(nb.read_text())["cells"]
    out = []
    for c in cells:
        if c["cell_type"] != "code":
            continue
        for ln in "".join(c["source"]).splitlines():
            s = ln.lstrip()
            # shell and magic lines (the Colab install) are not Python
            out.append(ln[: len(ln) - len(s)] + "pass" if s.startswith(("!", "%")) else ln)
        out.append("")
    return "\n".join(out)


def test_the_notebook_runs_top_to_bottom(engine, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    monkeypatch.setenv("AGRI_JAX_TUTORIAL_SMALL", "1")
    monkeypatch.setenv("AGRI_JAX_DSSAT", str(engine))
    monkeypatch.chdir(tmp_path)
    code = _code_of(NOTEBOOK)
    t0 = time.perf_counter()
    ns: dict = {"__name__": "__tutorial__"}
    exec(compile(code, str(NOTEBOOK), "exec"), ns)
    print(f"notebook (small) in {time.perf_counter() - t0:.0f} s")
    assert ns["res"].dssat_check is not None and ns["res"].dssat_check["agree"]
    assert (tmp_path / "MZCER048_calibrated.CUL").is_file()
    import matplotlib.pyplot as plt

    plt.close("all")
