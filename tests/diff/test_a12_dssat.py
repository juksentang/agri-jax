"""DSSAT-CSM v4.8.6.0 ROOTWU and daily EOP / TRWUP dumps for the 58 M2 maize treatments (A12).

A private gfortran build of the BSD-3 source, instrumented at ``SPAM`` (exit, RATE call: EO, EOP,
EP, ES, TRWU, TRWUP, XHLAI, RWU, SW, RLV) and ``ROOTWU`` (entry and exit, with its SAVEd state
``TSS``), ran every treatment of ``tests/integration/test_ceres_dssat.py``'s maize set with the
M2 staging (nitrogen off, one-treatment batch). The tables are
``dumps/tables/dssat486/<EXP>_t<NN>_{spam,rootwu_in,rootwu_out}.npz`` and the run's printed
outputs are in ``dumps/tables/dssat486/_runs/<EXP>_t<NN>/``.

These are the inputs of the A3 replay (``EOP``, ``TRWUP`` instead of ``SWFAC``, ``TURFAC``) and
the reference of the A4 ``rootwu_estimate`` port. The checks tie them to the reference model's own
printed outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from agrijax.io.dssat import read_out, read_plantgro, read_summary
from agrijax.port import dumps
from agrijax.processes.crop.ceres_maize.growth import water_stress_factors

pytestmark = pytest.mark.allow_skip(reason="dumps are private data")


@pytest.fixture(scope="module")
def tabdir(dumps_dir: Path) -> Path:
    d = dumps_dir / "tables" / "dssat486"
    if not (d / "collect_report.json").is_file():
        pytest.skip(f"no DSSAT A12 tables at {d}")
    return d


def _keys(d: Path) -> list[str]:
    return sorted(p.name.removesuffix("_spam.npz") for p in d.glob("*_spam.npz"))


def test_all_58_treatments_with_unchanged_outputs(tabdir: Path) -> None:
    rep = json.loads((tabdir / "collect_report.json").read_text())
    assert rep["n_treatments"] == 58 and rep["n_with_output_diffs"] == 0
    assert len(_keys(tabdir)) == 58
    for k, v in rep["treatments"].items():
        assert v["out_diffs"] == [], k


def test_rootwu_trwup_is_the_layer_sum_and_reaches_spam(tabdir: Path) -> None:
    """TRWUP = sum of RWU over the layers (REAL*4), and SPAM passes ROOTWU's TRWUP on unchanged;
    ROOTWU runs on the days with canopy (XHLAI > 0)."""
    for key in _keys(tabdir):
        sp = dumps.load_table(tabdir / f"{key}_spam.npz")[0]
        ri = dumps.load_table(tabdir / f"{key}_rootwu_in.npz")[0]
        ro = dumps.load_table(tabdir / f"{key}_rootwu_out.npz")[0]
        np.testing.assert_array_equal(ri.date, ro.date)
        nl = int(ro.values["NLAYR"][0])
        np.testing.assert_allclose(
            ro.values["TRWUP"], ro.values["RWU"][:, :nl].sum(axis=1), rtol=1e-5, atol=1e-6
        )
        i = sp.index_of(ro.date)
        assert np.all(i >= 0), key
        np.testing.assert_array_equal(sp.values["TRWUP"][i], ro.values["TRWUP"])
        canopy = sp.date[sp.values["XHLAI"] > 0]
        assert set(ro.date.tolist()) == set(canopy.tolist()), key
        # the saturation-day counter is state carried from one call to the next
        np.testing.assert_array_equal(ri.values["TSS"][1:], ro.values["TSS"][:-1])


def test_eop_matches_printed_eopa(tabdir: Path) -> None:
    for key in _keys(tabdir):
        sp = dumps.load_table(tabdir / f"{key}_spam.npz")[0]
        et = read_out(tabdir / "_runs" / key / "ET.OUT")
        k = np.asarray(et["YEAR"], int) * 1000 + np.asarray(et["DOY"], int)
        j = sp.index_of(k)
        ok = j >= 0
        assert ok.sum() > 50, key
        d = np.abs(sp.values["EOP"][j[ok]] - np.asarray(et["EOPA"], float)[ok])
        assert d.max() <= 0.0005 + 1e-6, (key, d.max())  # EOPA is printed with 3 decimals


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the replay check runs in float64")
def test_stress_factors_from_eop_trwup_reproduce_plantgro(tabdir: Path) -> None:
    """A3 / O-DS1: ``water_stress_factors(EOP, TRWUP, RWUEP1 = 1.5)`` reproduces the printed
    ``WSPD = 1 - SWFAC`` and ``WSGD = 1 - TURFAC`` on every day before physiological maturity (from
    maturity on, CERES no longer updates SWFAC/TURFAC while SPAM still reports EOP, TRWUP)."""
    n = 0
    for key in _keys(tabdir):
        sp = dumps.load_table(tabdir / f"{key}_spam.npz")[0]
        pg = read_plantgro(tabdir / "_runs" / key / "PlantGro.OUT")
        s = read_summary(tabdir / "_runs" / key / "Summary.OUT")
        md = float(s["MDAT"].iloc[0])
        md_i = int(md) if np.isfinite(md) and md > 0 else 10**8
        kp = np.asarray(pg["YEAR"], int) * 1000 + np.asarray(pg["DOY"], int)
        j = sp.index_of(kp)
        ok = (j >= 0) & (kp < md_i)
        sw, tu = water_stress_factors(
            sp.values["EOP"][j[ok]].astype(np.float64), sp.values["TRWUP"][j[ok]].astype(np.float64), 1.5
        )
        wspd = np.asarray(pg["WSPD"], float)[ok]
        wsgd = np.asarray(pg["WSGD"], float)[ok]
        assert np.abs((1 - np.asarray(sw)) - wspd).max() <= 0.0005 + 1e-6, key
        assert np.abs((1 - np.asarray(tu)) - wsgd).max() <= 0.0005 + 1e-6, key
        n += int(ok.sum())
    assert n > 5000
