"""Soil parameters written into the free-run DSSAT day (``calib/day_soil.py``), no data.

A synthetic run: the soil profile of the DSSAT example UFGA8201 (IBMZ910014, eight ``.SOL`` layers,
nine model layers: the 30-60 cm layer is split), FileX-like initial water, the crop conformance
season. Its trees come from the native soil chain and :class:`FreeRunInputs`, as for a DSSAT
example.

* The map: one model layer per soil-file layer share, weight exactly 1 for a split layer; ``lmatch``
  is the weighted sum for any values; a thin file layer shares a model layer with its neighbour,
  which the map refuses unless the caller accepts the deviation. **Declared deviation**: in a shared
  model layer ``with_soil`` keeps the unrounded mean where the native chain (as DSSAT) reads it
  printed ``F5.3``; pinned to at most half a print step there, and none elsewhere.
* Writing the unchanged soil leaves every tree bit for bit as it was.
* **Independent check**: ``with_soil(original inputs, soil')`` equals, bit for bit, the trees the
  native chain builds from a ``.SOL`` holding ``soil'`` (written with ``write_sol`` and read back),
  for soils at the printed precision of ``DSSAT48.INP``; the initial water is limited by the new
  ``LL`` and ``SAT`` as DSSAT does (raised to ``LL``, the top layer to its air-dry value, lowered to
  ``SAT``, ``DUL`` where the FileX has no value) -- the same as the native chain's own limits; one
  soil dries the top layer below ``DUL`` by more than the new ``U``, so the ``SOILEV`` store starts
  in stage 2 and its sums depend on ``U``.
* After writing, every copy of the soil holds the same values (bucket, crop, ROOTWU, XTRACT,
  albedo, soil evaporation, the forcing on every day).
* A season run from the written trees equals the season run from the rebuilt ones; ``jit``,
  ``vmap`` and ``grad`` go through ``with_soil`` (the layer map as a static ``jit`` argument);
  on this synthetic run the season's derivatives of the summed soil evaporation and of the
  end-of-season storage against ``DUL`` and ``SWCON`` agree with central finite differences (other
  season derivatives here, and the storage on a DSSAT example, do not: ``calib/day_soil.py``,
  Gradients).
* A run whose soil changes during the season (``SOILDYN``: the map's flag, or a forcing whose daily
  soil varies) refuses soil parameters; ``SSKS`` cannot be given where the file has none.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from agrijax.calib.day_soil import SoilDynamicsError, clip_initial_water, sol_layer_map, with_soil
from agrijax.io.dssat.native_crop import native_soil_crop
from agrijax.io.dssat.native_soil import initial_soil_water, layer_weights, lmatch, native_soil
from agrijax.io.dssat.sol import SoilProfile, read_sol, write_sol
from agrijax.models.day_dssat486 import DssatSite
from agrijax.processes.crop.ceres_maize import CeresSoil
from agrijax.sites.dssat_free_run import FreeRunInputs, _season_model, stack_trees
from agrijax.testing.conformance._builtin import crop as crop_case

pytestmark = [
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the day's soil is compared in float64"),
    pytest.mark.allow_skip(reason="float64 comparison, skipped in the float32 tier"),
]

N_DAYS = 45
YRPLT = crop_case.START + 5
ZLAYR = [5.0, 15.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
#: initial water rows (ICBL, SH2O) of the FileX-like initial conditions; layer 3 (15-30 cm) has none
IC = [
    {"ICBL": 5, "SH2O": 0.086},
    {"ICBL": 15, "SH2O": 0.086},
    {"ICBL": 30, "SH2O": -99.0},
    {"ICBL": 60, "SH2O": 0.086},
    {"ICBL": 90, "SH2O": 0.076},
    {"ICBL": 120, "SH2O": 0.076},
    {"ICBL": 150, "SH2O": 0.13},
    {"ICBL": 180, "SH2O": 0.258},
]
BASE = {
    "SLLL": [0.026, 0.025, 0.025, 0.025, 0.028, 0.028, 0.029, 0.070],
    "SDUL": [0.096, 0.086, 0.086, 0.086, 0.090, 0.090, 0.130, 0.258],
    "SSAT": [0.23] * 7 + [0.36],
    "SRGF": [1.0, 1.0, 0.7, 0.3, 0.05, 0.03, 0.002, 0.0],
    "SSKS": [np.nan] * 8,
}
SURFACE = {"SCOM": "-99", "SALB": 0.18, "SLU1": 2.0, "SLDR": 0.65, "SLRO": 60.0, "SLNF": 1.0, "SLPF": 0.92}


def _profile(
    layers: dict | None = None, surface: dict | None = None, zlayr: list | None = None
) -> SoilProfile:
    lay = {"SLB": zlayr or ZLAYR, **{k: list(v) for k, v in BASE.items()}, **(layers or {})}
    df = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in lay.items()})
    return SoilProfile(
        id="IBMZ910014",
        source="TEST",
        texture="S",
        depth=float(df["SLB"].iloc[-1]),
        description="synthetic copy of the Millhopper sand",
        site={"SITE": "Gainesville", "COUNTRY": "USA", "LAT": 29.6, "LONG": -82.4, "SCS FAMILY": "TEST"},
        surface={**SURFACE, **(surface or {})},
        layers=df,
        layer_groups=[list(df.columns)],
    )


def _inputs(profile: SoilProfile, ic: list | None = IC) -> FreeRunInputs:
    """The free-run inputs of the synthetic run on ``profile``, built as ``native_series`` and
    ``ceres_params_from`` build them for a DSSAT example."""
    soil = native_soil(profile, "2")
    sw0 = initial_soil_water(soil, ic)
    crop = native_soil_crop(profile, soil)
    a = lambda x: jnp.asarray(np.asarray(x, dtype=float))  # noqa: E731
    pc = crop_case.params(jnp.float64, YRPLT)
    ds = np.asarray(crop["ds"], dtype=float)
    pc = pc.replace(
        soil=CeresSoil(
            dlayr=a(np.diff(np.concatenate([[0.0], ds]))),
            ll=a(crop["ll"]),
            dul=a(crop["dul"]),
            sat=a(crop["sat"]),
            shf=a(crop["shf"]),
            slpf=a(crop["slpf"]),
        )
    )
    site = DssatSite(
        dlayr=soil.dlayr, ds=soil.ds, ll=soil.ll, dul=soil.dul, sat=soil.sat, swcn=soil.swcn,
        cn=soil.cn, swcon=soil.swcon, salb=soil.salb, u=soil.u, mesev="R", meinf="S",
    )  # fmt: skip
    w = crop_case.weather(7, jnp.float64, N_DAYS)
    rng = np.random.default_rng(5)
    n = N_DAYS
    f64 = lambda v: np.asarray(v, dtype=np.float32).astype(np.float64)  # noqa: E731
    series = {
        "rain": f64(np.where(rng.uniform(size=n) < 0.3, rng.uniform(2.0, 30.0, n), 0.0)),
        "tmax": f64(w.tmax),
        "irrigation": np.zeros(n),
        "m_mass": np.zeros(n),
        "m_cover": np.zeros(n),
        "m_new": np.zeros(n),
        "m_watfac": np.full(n, 3.8),
        "mulch_am": np.full(n, 30.0),
        "mulch_extfac": np.full(n, 0.86),
        "tavg": f64((np.asarray(w.tmax) + np.asarray(w.tmin)) / 2.0),
        "wind_run": np.full(n, 150.0),
        "co2": f64(w.co2),
        "srad": f64(w.srad),
        "tmax_w": f64(w.tmax),
        "tmin_w": f64(w.tmin),
        "dlayr_end": np.broadcast_to(f64(np.asarray(soil.dlayr, np.float32)), (n, soil.nl)).copy(),
    }
    return FreeRunInputs(
        key="SYNTH_t01", exp="SYNTH", trno=1, out=None, cultivar="IB0035", params_crop=pc,
        forcing_crop=w, summary={}, nl=soil.nl, mesev="R", site=site, soildyn=False, series=series,
        sw0=f64(np.asarray(sw0, np.float32)), snow0=0.0, mulch0=0.0, mulch_evap0=0.0, ksevap=0.6854839,
    )  # fmt: skip


def _trees(x: FreeRunInputs):
    p = x.params()
    return p, x.forcing(), x.state(p)


def _assert_same(a, b, what: str) -> None:
    la, ta = jax.tree.flatten(a)
    lb, tb = jax.tree.flatten(b)
    assert ta == tb, what
    paths = [jax.tree_util.keystr(k) for k, _ in jax.tree_util.tree_flatten_with_path(a)[0]]
    bad = [
        p
        for p, x, y in zip(paths, la, lb, strict=True)
        if np.shape(x) != np.shape(y) or not np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)
    ]
    assert not bad, (what, bad[:10])


def _soil_of(layers: dict, surface: dict) -> dict:
    """``with_soil``'s mapping for the .SOL values ``layers`` / ``surface`` (file column names)."""
    names = {"SLLL": "ll", "SDUL": "dul", "SSAT": "sat", "SRGF": "srgf", "SSKS": "swcn"}
    snames = {"SLRO": "cn", "SLDR": "swcon", "SLU1": "u", "SLPF": "slpf"}
    out = {names[k]: jnp.asarray(np.asarray(v, dtype=float)) for k, v in layers.items()}
    out.update({snames[k]: jnp.asarray(float(v)) for k, v in surface.items()})
    return out


#: perturbed soils at DSSAT48.INP's printed precision (LL/DUL/SAT/SRGF 3 decimals, CN 0, U 1,
#: SWCON and SLPF 2); :data:`STAGE2` starts the SOILEV store in stage 2, the last one moves LL
#: above and SAT below the initial water
PERTURBED = [
    (
        {"SLLL": [0.03, 0.028, 0.022, 0.03, 0.025, 0.031, 0.026, 0.065],
         "SDUL": [0.11, 0.079, 0.095, 0.08, 0.097, 0.084, 0.121, 0.27],
         "SSAT": [0.25, 0.21, 0.24, 0.22, 0.23, 0.23, 0.23, 0.34],
         "SRGF": [1.0, 0.9, 0.8, 0.25, 0.06, 0.03, 0.002, 0.0]},
        {"SLRO": 66.0, "SLDR": 0.5, "SLU1": 3.5, "SLPF": 0.85},
    ),
    (
        {"SLLL": [0.02, 0.02, 0.03, 0.021, 0.028, 0.028, 0.029, 0.07],
         "SDUL": [0.08, 0.1, 0.07, 0.09, 0.09, 0.09, 0.13, 0.258]},
        {"SLRO": 55.0, "SLU1": 2.0},
    ),
    (  # DUL of the top layer raised above its initial water 0.086 by more than U / 50 mm: the
       # SOILEV store starts in stage 2 (USOIL 3.2 mm > U 2.5 mm); the top layer's SRGF moved too
        {"SDUL": [0.15, 0.086, 0.086, 0.086, 0.090, 0.090, 0.130, 0.258],
         "SRGF": [0.8, 1.0, 0.7, 0.3, 0.05, 0.03, 0.002, 0.0]},
        {"SLU1": 2.5},
    ),
    (
        {"SLLL": [0.045, 0.1, 0.025, 0.09, 0.08, 0.028, 0.029, 0.07],
         "SDUL": [0.12, 0.12, 0.086, 0.12, 0.11, 0.05, 0.13, 0.15],
         "SSAT": [0.23, 0.23, 0.23, 0.23, 0.23, 0.07, 0.23, 0.2]},
        {"SLDR": 0.7},
    ),
]  # fmt: skip
#: the soil of :data:`PERTURBED` whose top layer starts in SOILEV's stage 2
STAGE2 = 2


def _layer_map(profile: SoilProfile | None = None, ic: list | None = IC, soildyn: bool = False):
    return sol_layer_map(profile if profile is not None else _profile(), "2", ic, soildyn=soildyn)


# ------------------------------------------------------------------ the map
def test_the_layer_map_is_lmatch() -> None:
    m = _layer_map()
    assert (m.nl, m.n_sol) == (9, 8)
    assert m.sol_layer.tolist() == [0, 1, 2, 3, 3, 4, 5, 6, 7]  # the 30-60 cm layer split in two
    assert set(np.unique(m.weight).tolist()) == {0.0, 1.0} and m.index.shape == (9, 1)
    ds = np.asarray(native_soil(_profile(), "2").ds, dtype=np.float32)
    v = np.random.default_rng(0).uniform(0.01, 0.5, 8)
    np.testing.assert_allclose(m.weight @ v, lmatch(np.asarray(ZLAYR), v, ds), rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(m.expand(v)), v[m.sol_layer])
    # a file layer thinner than 2 cm is merged: a model layer then shares two file layers
    z = [1.0, 5.0, 15.0, 40.0]
    thin = {"SLLL": [0.03, 0.04, 0.05, 0.06], "SDUL": [0.1] * 4, "SSAT": [0.3] * 4, "SRGF": [1.0] * 4}
    p = _profile({**thin, "SSKS": [np.nan] * 4}, zlayr=z)
    with pytest.raises(ValueError, match="shared_layers"):
        sol_layer_map(p, "2", None, soildyn=False)
    mm = sol_layer_map(p, "2", None, soildyn=False, shared_layers=True)
    assert mm.shared_layers and not mm.exact and m.exact and not m.shared_layers
    w = layer_weights(np.asarray(z), np.asarray(native_soil(p, "2").ds, dtype=np.float32))
    assert np.allclose(w.sum(axis=1), 1.0) and np.count_nonzero(w[0]) == 2 and mm.index.shape[1] == 2
    x = np.array([0.03, 0.04, 0.05, 0.06])
    np.testing.assert_allclose(np.asarray(mm.expand(x)), w @ x, rtol=1e-15)


def test_initial_water_limits_are_dssat_s() -> None:
    """``clip_initial_water`` on the FileX input equals the native chain's limits (REAL*4) for
    random soils: below LL, below the air-dry top layer, above SAT, DUL where there is no value."""
    rng = np.random.default_rng(2)
    for _ in range(25):
        ll = np.round(rng.uniform(0.01, 0.2, 8), 3)
        dul = np.round(ll + rng.uniform(0.02, 0.15, 8), 3)
        sat = np.round(dul + rng.uniform(0.02, 0.2, 8), 3)
        ic = [{"ICBL": z, "SH2O": float(np.round(rng.uniform(0.0, 0.4), 3)) or -99.0} for z in ZLAYR]
        ic[0]["SH2O"] = float(np.round(rng.uniform(0.001, 0.02), 3))
        prof = _profile({"SLLL": ll, "SDUL": dul, "SSAT": sat})
        soil = native_soil(prof, "2")
        want = initial_soil_water(soil, ic)
        m = sol_layer_map(prof, "2", ic, soildyn=False)
        none = np.isnan(m.sw_input)
        r4 = lambda v: np.asarray(v, dtype=np.float32).astype(np.float64)  # noqa: E731
        raw = np.where(none, soil.dul, np.nan_to_num(m.sw_input))
        got = np.asarray(
            clip_initial_water(jnp.asarray(raw), jnp.asarray(r4(soil.ll)), jnp.asarray(r4(soil.sat)))
        )
        np.testing.assert_array_equal(got, r4(np.asarray(want, np.float32)))


#: a profile with a 1 cm top layer: model layer 0 (0-5 cm) averages file layers 0 and 1 (0.2 / 0.8)
THIN_Z = [1.0, *ZLAYR]
THIN = {k: [v[0], *v] for k, v in BASE.items()}
#: a soil on the thin profile whose shared-layer means fall between print steps
THIN_SOIL = {
    "SLLL": [0.031, 0.022, *BASE["SLLL"][1:]],
    "SDUL": [0.107, 0.083, *BASE["SDUL"][1:]],
    "SSAT": [0.25, 0.23, *BASE["SSAT"][1:]],
    "SRGF": [1.0, 0.93, *BASE["SRGF"][1:]],
}
#: half the F5.3 print step, plus a REAL*4 rounding quantum
HALF_STEP = 0.0005 + 1e-7


def test_shared_layer_deviation_is_at_most_half_a_print_step(tmp_path: Path) -> None:
    """Declared deviation: on a profile whose top model layer averages two file layers, ``with_soil``
    differs from the native rebuild (which reads the mean printed ``F5.3``) only in that layer, by at
    most half a print step in every soil copy, and nowhere else."""
    base = _profile(THIN, zlayr=THIN_Z)
    m = sol_layer_map(base, "2", IC, soildyn=False, shared_layers=True)
    shared = np.nonzero(np.any((m.weight != 0.0) & (m.weight != 1.0), axis=1))[0]
    assert shared.tolist() == [0]
    other = np.setdiff1d(np.arange(m.nl), shared)
    ours = with_soil(*_trees(_inputs(base)), _soil_of(THIN_SOIL, {}), layers=m)
    sol = write_sol([_profile({**THIN, **THIN_SOIL}, zlayr=THIN_Z)], tmp_path / "SOIL.SOL")
    ref = _trees(_inputs(read_sol(sol, dssat_spans=True)["IBMZ910014"]))
    worst = 0.0
    for name, a, b in (
        ("soil.ll", ours[0]["soil"].soil.ll, ref[0]["soil"].soil.ll),
        ("soil.dul", ours[0]["soil"].soil.dul, ref[0]["soil"].soil.dul),
        ("soil.sat", ours[0]["soil"].soil.sat, ref[0]["soil"].soil.sat),
        ("crop.ll", ours[0]["crop"].soil.ll, ref[0]["crop"].soil.ll),
        ("crop.dul", ours[0]["crop"].soil.dul, ref[0]["crop"].soil.dul),
        ("crop.shf", ours[0]["crop"].soil.shf, ref[0]["crop"].soil.shf),
        ("rootwu.ll", ours[0]["rootwu"].ll, ref[0]["rootwu"].ll),
        ("forcing.ll", ours[1]["soil"].soil.ll, ref[1]["soil"].soil.ll),
        ("evap.dul", ours[2]["surface"]["soil_evap"].dul, ref[2]["surface"]["soil_evap"].dul),
    ):
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_array_equal(a[..., other], b[..., other], err_msg=name)
        d = float(np.max(np.abs(a[..., shared] - b[..., shared])))
        assert d <= HALF_STEP, (name, d)
        worst = max(worst, d)
    assert worst > 1e-5  # the deviation exists: LL 0.0238 against the printed 0.024
    dl = np.asarray(ours[0]["soil"].soil.dlayr)
    d0 = abs(float(ours[2]["ledger"]["water"].storage) - float(ref[2]["ledger"]["water"].storage))
    assert d0 <= HALF_STEP * float(np.sum(dl[shared])) + 1e-9, d0


# ------------------------------------------------------------------ writing a soil
def test_the_unchanged_soil_changes_nothing() -> None:
    x = _inputs(_profile())
    m = _layer_map()
    p, f, s = _trees(x)
    for soil in (m.values(), {}):
        q, g, t = with_soil(p, f, s, soil, layers=m)
        _assert_same(q, p, "params")
        _assert_same(g, f, "forcing")
        _assert_same(t, s, "state")


@pytest.mark.parametrize("k", range(len(PERTURBED)))
def test_with_soil_equals_the_native_rebuild(k: int, tmp_path: Path) -> None:
    layers, surface = PERTURBED[k]
    x0 = _inputs(_profile())
    m = _layer_map()
    soil = _soil_of(layers, surface)
    ours = with_soil(*_trees(x0), soil, layers=m)
    # the independent rebuild: the soil written to a .SOL, read back, through the native chain
    sol = write_sol([_profile(layers, surface)], tmp_path / "SOIL.SOL")
    prof = read_sol(sol, dssat_spans=True)["IBMZ910014"]
    ref = _trees(_inputs(prof))
    for a, b, what in zip(ours, ref, ("params", "forcing", "state"), strict=True):
        _assert_same(a, b, what)
    if k == STAGE2:  # stage 2 at the start: the store's sums depend on the new U
        se = ours[2]["surface"]["soil_evap"]
        u = float(np.float32(surface["SLU1"]))
        assert float(se.sumes1) == u and float(se.sumes2) > 0.0 and float(se.t) > 0.0, se
        old_u = with_soil(*_trees(x0), {q: v for q, v in soil.items() if q != "u"}, layers=m)
        old = old_u[2]["surface"]["soil_evap"]
        assert float(old.sumes1) != float(se.sumes1) and float(old.sumes2) != float(se.sumes2)
    if k == len(PERTURBED) - 1:  # the initial water was limited both ways
        sw_in = np.where(np.isnan(m.sw_input), np.asarray(ref[0]["soil"].soil.dul), m.sw_input)
        sw0 = np.asarray(ours[2]["soil_water"].sw)
        assert np.any(sw0 > sw_in + 1e-6) and np.any(sw0 < sw_in - 1e-6)


def test_every_copy_holds_the_written_soil() -> None:
    layers, surface = PERTURBED[0]
    x0 = _inputs(_profile())
    m = _layer_map()
    p, f, s = with_soil(*_trees(x0), _soil_of(layers, surface), layers=m)
    bs = p["soil"].soil
    r4 = lambda v: np.asarray(v, dtype=np.float32).astype(np.float64)  # noqa: E731
    want_ll = r4(np.asarray(layers["SLLL"])[m.sol_layer])
    for name, v in (
        ("soil", bs.ll),
        ("crop", p["crop"].soil.ll),
        ("rootwu", p["rootwu"].ll),
        ("xtract", p["xtract"].ll),
        ("evap state", s["surface"]["soil_evap"].ll),
    ):
        np.testing.assert_array_equal(np.asarray(v), want_ll, err_msg=name)
    np.testing.assert_array_equal(np.asarray(f["soil"].soil.ll), np.broadcast_to(want_ll, (N_DAYS, 9)))
    for leaf in ("dul", "sat", "swcn", "cn", "swcon"):
        day = np.asarray(getattr(f["soil"].soil, leaf))
        assert np.all(day == np.asarray(getattr(bs, leaf))), leaf
    np.testing.assert_array_equal(np.asarray(p["rootwu"].sat), np.asarray(bs.sat))
    np.testing.assert_array_equal(np.asarray(p["crop"].soil.dul), np.asarray(bs.dul))
    assert float(p["albedo"].dul1) == float(bs.dul[0]) == float(r4(0.11))
    assert float(p["evap"].u) == float(r4(3.5)) and float(bs.cn) == 66.0 and float(bs.swcon) == float(r4(0.5))
    np.testing.assert_array_equal(np.asarray(p["crop"].soil.shf), np.asarray(layers["SRGF"])[m.sol_layer])
    assert float(p["crop"].soil.slpf) == 0.85
    sw0 = np.asarray(s["soil_water"].sw)
    np.testing.assert_array_equal(np.asarray(s["iface"]["crop_water"]["maize"].sw), sw0)
    assert float(s["ledger"]["water"].storage) == pytest.approx(
        float(np.sum(sw0 * np.asarray(bs.dlayr))), rel=1e-15
    )


def test_a_season_from_the_written_trees_equals_the_rebuilt_one(tmp_path: Path) -> None:
    layers, surface = PERTURBED[0]
    m = _layer_map()
    ours = with_soil(*_trees(_inputs(_profile())), _soil_of(layers, surface), layers=m)
    sol = write_sol([_profile(layers, surface)], tmp_path / "SOIL.SOL")
    ref = _trees(_inputs(read_sol(sol, dssat_spans=True)["IBMZ910014"]))
    base = _trees(_inputs(_profile()))
    from agrijax.core.runtime import run_sites

    model = _season_model("R")
    out = run_sites(model, *(stack_trees([a, b, c]) for a, b, c in zip(ours, ref, base, strict=True)))
    for k in ("soil_sw", "lai", "cwad", "es", "ep", "storage"):
        v = np.asarray(out[k])
        assert np.all(np.isfinite(v)), k
        np.testing.assert_array_equal(v[0], v[1], err_msg=k)
    assert not np.array_equal(np.asarray(out["soil_sw"])[0], np.asarray(out["soil_sw"])[2])


#: central finite-difference step on a relative scale of the soil values (smaller steps fall below a
#: REAL*4 quantum of the soil values or catch single threshold flips)
FD_STEP = 1e-3
#: pinned largest relative AD-FD difference (measured: 0.08 % to 0.8 %)
FD_REL = 0.02


def test_synthetic_sand_es_and_storage_gradients_match_finite_differences() -> None:
    """On this synthetic run only: the season's summed soil evaporation and its end-of-season
    storage against a relative change of ``DUL`` (all layers) and of ``SWCON``, automatic derivative
    through ``with_soil`` and the season against central finite differences. Not a property of these
    outputs: other season derivatives on this run disagree (the summed layer water against ``DUL``,
    aggregates against ``LL``, ``SAT``, ``SRGF``), and on UFGA8201 the end-of-season storage against
    a layer's ``DUL`` disagrees in sign, which is why a gradient-based soil calibration on the
    faithful day checks its objective on its own site first (``calib/day_soil.py``, Gradients)."""
    from agrijax.core.runtime import run

    x0 = _inputs(_profile())
    m = _layer_map()
    p, f, s = _trees(x0)
    base = m.values()
    model = _season_model("R")

    def season(scale: jax.Array, q: str) -> dict[str, jax.Array]:
        out = run(model, *with_soil(p, f, s, {**base, q: base[q] * scale}, layers=m))
        return {"es_sum": jnp.sum(out["es"]), "storage_end": out["storage"][-1]}

    one = jnp.asarray(1.0)
    for q in ("dul", "swcon"):
        jac = jax.jacrev(season)(one, q)
        up, down = season(one + FD_STEP, q), season(one - FD_STEP, q)
        for k, d in jac.items():
            ad, fd = float(d), (float(up[k]) - float(down[k])) / (2 * FD_STEP)
            rel = abs(ad - fd) / max(abs(fd), 1e-12)
            print(f"d {k} / d {q} scale: AD {ad:.6g} FD {fd:.6g} rel {rel:.3e}")
            assert np.isfinite(ad) and abs(fd) > 0.0 and rel <= FD_REL, (k, q, ad, fd, rel)


def test_jit_vmap_and_grad_go_through_with_soil() -> None:
    x0 = _inputs(_profile())
    m = _layer_map()
    p, f, s = _trees(x0)
    base = m.values()

    def store(dul_scale, u):
        soil = {**base, "dul": base["dul"] * dul_scale, "u": u}
        _, g, t = with_soil(p, f, s, soil, layers=m)
        se = t["surface"]["soil_evap"]
        return se.sumes1 + se.sumes2 + t["ledger"]["water"].storage + jnp.sum(g["soil"].soil.dul)

    val, grad = jax.jit(jax.value_and_grad(store, argnums=(0, 1)))(jnp.asarray(1.05), jnp.asarray(4.0))
    assert np.isfinite(float(val)) and all(np.isfinite(float(gr)) for gr in grad) and float(grad[0]) > 0.0
    many = jax.jit(jax.vmap(store))(jnp.asarray([0.9, 1.0, 1.1]), jnp.asarray([3.0, 4.0, 5.0]))
    assert np.all(np.isfinite(np.asarray(many))) and np.ptp(np.asarray(many)) > 0.0
    # the map hashes and compares by identity: a static jit argument
    assert hash(m) == hash(m) and m == m and m != _layer_map()
    jitted = jax.jit(with_soil, static_argnames=("layers", "slot"))
    soil = {**base, "dul": base["dul"] * 1.05}
    q, _, t = jitted(p, f, s, soil, layers=m)
    q0, _, t0 = with_soil(p, f, s, soil, layers=m)
    np.testing.assert_array_equal(np.asarray(q["soil"].soil.dul), np.asarray(q0["soil"].soil.dul))
    np.testing.assert_array_equal(np.asarray(t["soil_water"].sw), np.asarray(t0["soil_water"].sw))


def test_refusals() -> None:
    x0 = _inputs(_profile())
    p, f, s = _trees(x0)
    m = _layer_map(soildyn=True)
    with pytest.raises(SoilDynamicsError, match="SOILDYN"):
        with_soil(p, f, s, {"cn": jnp.asarray(70.0)}, layers=m)
    with_soil(p, f, s, {}, layers=m)  # nothing written: nothing to refuse
    m = _layer_map()
    # a forcing whose daily soil changes (a SOILDYN replay) is refused although the map says static
    fs = f["soil"].soil
    g = {**f, "soil": f["soil"].replace(soil=fs.replace(ll=fs.ll.at[3].add(0.01)))}
    with pytest.raises(SoilDynamicsError, match=r"daily soil\.ll"):
        with_soil(p, g, s, {"cn": jnp.asarray(70.0)}, layers=m)
    with pytest.raises(TypeError, match="soildyn"):
        sol_layer_map(_profile(), "2", IC)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="SSKS"):
        with_soil(p, f, s, {"swcn": jnp.ones(8)}, layers=m)
    with pytest.raises(ValueError, match="unknown"):
        with_soil(p, f, s, {"ksat": jnp.ones(8)}, layers=m)
    with pytest.raises(ValueError, match=r"one value per \.SOL layer"):
        with_soil(p, f, s, {"ll": jnp.ones(9) * 0.03}, layers=m)
    with pytest.raises(ValueError, match="one value"):
        with_soil(p, f, s, {"cn": jnp.ones(2)}, layers=m)


def test_ssks_where_the_file_gives_it(tmp_path: Path) -> None:
    ssks = [2.1, 1.5, 0.9, 0.8, 0.5, 0.4, 0.3, 0.2]
    prof = _profile({"SSKS": ssks})
    m = sol_layer_map(prof, "2", IC, soildyn=False)
    x0 = _inputs(prof)
    new = [3.0, 1.2, 0.7, 0.9, 0.45, 0.4, 0.33, 0.25]
    ours = with_soil(*_trees(x0), {"swcn": jnp.asarray(new)}, layers=m)
    sol = write_sol([_profile({"SSKS": new})], tmp_path / "SOIL.SOL")
    ref = _trees(_inputs(read_sol(sol, dssat_spans=True)["IBMZ910014"]))
    for a, b, what in zip(ours, ref, ("params", "forcing", "state"), strict=True):
        _assert_same(a, b, what)
    assert dataclasses.is_dataclass(m) and m.sol["swcn"].tolist() == ssks
