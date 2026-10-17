"""Shared and per-site parameter blocks (``calib/space.py`` :class:`ParamBlocks`) and the free-run
day simulator on any space (``calib/dssat_day.py`` :class:`DaySimulator`), no data.

* Blocks: the fixed layout (shared block, then the site blocks in order), the columns a site reads
  and the space they form, ``split`` / ``join``, transforms and violations block by block; a
  parameter in the shared block and a site block is refused.
* The simulator on synthetic runs (``test_calib_day_soil``: the crop conformance season on a soil
  profile, float64): the default space is :data:`CUL_ORDER` and runs the program it ran before
  (bit for bit the same entries as the explicit cultivar space, also reordered); a soil space
  gives the season of :func:`with_soil`'s trees (``run_sites``); the joint loss of two sites equals
  the sum of the single-site losses, treatment by treatment (one batch group, and two groups: a
  second profile with eight model layers); ``vmap`` of the loss equals one call per point; one
  compiled program per batch group; the loss's gradient is finite; refusals.
* Paths of the simulator pinned on the same runs: a treatment without soil coordinates or layer
  map in a group that writes soil keeps its default season; cultivar coefficients in the site
  blocks; interleaved batch groups (A, C, B) keep the treatment order; two treatments of one site,
  and soil coordinates in the shared space; single-precision storage; the joint gradient is the
  column-wise sum of the single-site gradients (one group, interleaved groups).

The simulators run the padded readout (``readout="padded"``: the value at the first ISTAGE 10 or
on the last simulated day, entries on any simulated day), except one test of the default season
readout: ``jax_loss`` on blocks with soil, interleaved groups, two end conventions and treatment
subsets. The season readout of a single run against the host readout is checked in
``test_dssat_day_readout``.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.calib.soil import SITE_BOUNDS, soil_space, without_domain_bounds
from agrijax.calib.space import ParamBlocks, ParamSpace, ParamSpec

#: the readout of every simulator of this module (module docstring)
READOUT = "padded"

#: marks of the tests on the synthetic runs (built in float64, as ``test_calib_day_soil``)
X64 = (
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the synthetic runs are built in float64"),
    pytest.mark.allow_skip(reason="float64 synthetic runs, skipped in the float32 tier"),
)


def _x64(f):
    for m in reversed(X64):
        f = m(f)
    return f


def _spec(path: str, lo: float, hi: float, **kw) -> ParamSpec:
    return ParamSpec(path, lo, hi, bounds_source="test box", **kw)


# ------------------------------------------------------------------------ blocks
SHARED = ParamSpace(
    [_spec("cultivar.p1", 5.0, 450.0, label="P1"), _spec("cultivar.g2", 248.0, 990.0, label="G2")]
)
SITE_A = ParamSpace([_spec("cn", 50.0, 94.0, label="CN"), _spec("u", 1.0, 12.0, label="U")])
SITE_B = ParamSpace(
    [_spec("ll", 0.2, 0.3, index=0, label="LL_H1"), _spec("dul", 0.25, 0.4, index=0, label="DUL_H1")],
)


def test_blocks_lay_theta_out_shared_first_then_sites_in_order() -> None:
    b = ParamBlocks(SHARED, {"A": SITE_A, "B": SITE_B})
    assert b.n == 6 and b.sites == ("A", "B")
    assert b.offsets == {None: 0, "A": 2, "B": 4}
    assert b.names == ("P1", "G2", "A:CN", "A:U", "B:LL_H1", "B:DUL_H1")
    assert b.columns("A").tolist() == [0, 1, 2, 3] and b.columns("B").tolist() == [0, 1, 4, 5]
    assert b.columns().tolist() == [0, 1]
    assert b.space_of("B").names == ("P1", "G2", "LL_H1", "DUL_H1")
    np.testing.assert_array_equal(b.lower, np.concatenate([SHARED.lower, SITE_A.lower, SITE_B.lower]))
    np.testing.assert_array_equal(b.upper, np.concatenate([SHARED.upper, SITE_A.upper, SITE_B.upper]))
    assert [r["site"] for r in b.table()] == [None, None, "A", "A", "B", "B"]
    with pytest.raises(KeyError, match="unknown site 'C'"):
        b.columns("C")
    # a plain space is the shared block of blocks without sites
    p = ParamBlocks.of(SHARED)
    assert p.n == 2 and p.sites == () and ParamBlocks.of(p) is p


def test_blocks_split_join_and_transform_block_by_block() -> None:
    b = ParamBlocks(SHARED, {"A": SITE_A, "B": SITE_B})
    z = jnp.asarray([[0.3, -1.2, 0.5, 2.0, -0.4, 0.1], [1.0, 0.0, -2.0, 0.2, 0.0, 3.0]])
    th = b.to_physical(z)
    sh, ps = b.split(th)
    np.testing.assert_array_equal(sh, SHARED.to_physical(z[:, :2]))
    np.testing.assert_array_equal(ps["A"], SITE_A.to_physical(z[:, 2:4]))
    np.testing.assert_array_equal(ps["B"], SITE_B.to_physical(z[:, 4:]))
    np.testing.assert_array_equal(b.join(sh, ps), th)
    np.testing.assert_allclose(b.to_unconstrained(th), z, rtol=1e-6, atol=1e-6)
    # the columns of a site are a theta of its space
    np.testing.assert_array_equal(th[:, b.columns("B")], b.space_of("B").to_physical(z[:, b.columns("B")]))
    bad = np.asarray(th[0]).copy()
    bad[3] = 99.0
    assert any(m.startswith("A:U = 99") for m in b.violations(bad))
    with pytest.raises(ValueError, match="no values for sites"):
        b.join(sh, {"A": ps["A"]})


def test_a_parameter_in_the_shared_block_and_a_site_block_is_refused() -> None:
    with pytest.raises(ValueError, match="repeats shared parameters"):
        ParamBlocks(SHARED, {"A": ParamSpace([_spec("cultivar.p1", 5.0, 450.0, label="P1x")])})
    with pytest.raises(ValueError, match="non-empty strings"):
        ParamBlocks(SHARED, {"": SITE_A})


# ------------------------------------------------------------------------ the simulator on any space
PAD = 5
#: a second synthetic site: the layers of BRPI020002 (BR.SOL, four .SOL layers, nine model layers),
#: its FileX initial water below LL in the top three layers (BRPI0202 initial condition 2)
B_ZLAYR = [20.0, 40.0, 120.0, 150.0]
B_LAYERS = {
    "SLLL": [0.280, 0.284, 0.280, 0.280],
    "SDUL": [0.349, 0.345, 0.311, 0.311],
    "SSAT": [0.53] * 4,
    "SRGF": [1.0, 1.0, 0.9, 0.1],
    "SSKS": [0.39, 0.63, 1.21, 1.21],
}
B_SURFACE = {"SLU1": 9.1, "SLDR": 0.20, "SLRO": 83.0, "SLPF": 0.96, "SALB": 0.14}
B_IC = [{"ICBL": 20, "SH2O": 0.26}, {"ICBL": 40, "SH2O": 0.26}, {"ICBL": 120, "SH2O": 0.25}]
#: a third: the layers of BRPI020001 (three .SOL layers, eight model layers: another batch group)
C_ZLAYR = [20.0, 40.0, 120.0]
C_LAYERS = {
    "SLLL": [0.17, 0.18, 0.18],
    "SDUL": [0.28, 0.29, 0.28],
    "SSAT": [0.45, 0.45, 0.46],
    "SRGF": [1.0, 1.0, 0.6],
    "SSKS": [0.10, 0.10, 0.10],
}
C_SURFACE = {"SLU1": 6.0, "SLDR": 0.30, "SLRO": 83.0, "SLPF": 0.90, "SALB": 0.14}
C_IC = [{"ICBL": 20, "SH2O": 0.25}, {"ICBL": 40, "SH2O": 0.25}, {"ICBL": 120, "SH2O": 0.26}]


def _synth():
    from . import test_calib_day_soil as synth

    return synth


@pytest.fixture(scope="module")
def sites():
    """``{site: (run, layer map)}`` of the three synthetic sites."""
    from agrijax.calib.day_soil import sol_layer_map

    synth = _synth()
    out = {}
    for name, key, layers, surface, zlayr, ic in (
        ("A", "SYNTH_t01", None, None, None, synth.IC),
        ("B", "SYNTH_t02", B_LAYERS, B_SURFACE, B_ZLAYR, B_IC),
        ("C", "SYNTH_t03", C_LAYERS, C_SURFACE, C_ZLAYR, C_IC),
    ):
        prof = synth._profile(layers, surface, zlayr)
        run = dataclasses.replace(synth._inputs(prof, ic), key=key)
        out[name] = (run, sol_layer_map(prof, "2", ic, soildyn=False))
    assert (out["A"][0].nl, out["B"][0].nl, out["C"][0].nl) == (9, 9, 8)
    return out


def _entries(run) -> list:
    from agrijax.calib.dssat_day import E_DATE, E_DAY, E_FINAL, OUT_NAMES, Entry

    n = run.n_days
    out = [Entry(E_DATE, 1), Entry(E_FINAL, OUT_NAMES.index("cwad"))]
    out += [Entry(E_DAY, OUT_NAMES.index("lai"), t) for t in range(10, n, 7)]
    out += [Entry(E_DAY, OUT_NAMES.index("et"), t) for t in range(2, n, 3)]
    out += [Entry(E_DAY, OUT_NAMES.index(f"sw{k}"), t) for k in (1, 2, 4) for t in range(1, n, 4)]
    return out


def _site_space(name: str, m) -> ParamSpace:
    """A soil space per synthetic site (requested coordinates without those on a domain end)."""
    if name == "A":
        req = (
            ["cn", "swcon", "u", "slpf", "ll", "dul", "srgf"],
            {"ll": [0, 1], "dul": [0, 1], "srgf": [0, 1, 2]},
        )
    elif name == "B":
        req = (
            ["cn", "u", "ll", "dul", "sat", "srgf", "swcn"],
            {"ll": [0, 1], "dul": [0, 1], "sat": [0], "srgf": [0, 1, 2, 3], "swcn": [0]},
        )
    else:
        req = (["swcon", "u", "dul", "srgf"], {"dul": [0, 1, 2], "srgf": [0, 2]})
    q, lay, _ = without_domain_bounds(m.sol, *req)
    return soil_space(m.sol, q, lay, SITE_BOUNDS)


def _cultivar(run) -> np.ndarray:
    from agrijax.calib.dssat_day import CUL_ORDER

    return np.asarray([run.published()[c] for c in CUL_ORDER])


def _site_theta(sp: ParamSpace, m, scale: float = 1.0) -> np.ndarray:
    """The site values of ``sp``'s coordinates, moved by ``scale`` toward the upper bound."""
    v = np.asarray([float(np.asarray(m.sol[s.path]).reshape(-1)[s.index or 0]) for s in sp.specs])
    return np.asarray(sp.to_physical(sp.to_unconstrained(v) + (scale - 1.0)))


@_x64
def test_default_space_is_the_cultivar_columns_and_its_program(sites) -> None:
    from agrijax.calib.ceres import ceres_space
    from agrijax.calib.dssat_day import CUL_ORDER, DaySimulator, default_space

    run = sites["A"][0]
    ents = [_entries(run)]
    sim = DaySimulator([run], ents, pad_days=PAD, readout=READOUT)
    assert CUL_ORDER == ("P1", "P2", "P5", "G2", "G3", "PHINT")
    assert sim.n_theta == 6 and sim.blocks.shared.names == CUL_ORDER and default_space().names == CUL_ORDER
    plan = sim.plans[(run.nl, run.mesev)]
    assert plan.cul == ("p1", "p2", "p5", "g2", "g3", "phint") and plan.cul_static == (0, 1, 2, 3, 4, 5)
    assert plan.soil == ()
    th = _cultivar(run)[None] * np.asarray([[1.1, 1.0, 0.95, 1.05, 0.9, 1.0]])
    y = sim(th, np.zeros(1, np.int64))
    explicit = DaySimulator([run], ents, space=ceres_space(CUL_ORDER), pad_days=PAD, readout=READOUT)
    np.testing.assert_array_equal(explicit(th, np.zeros(1, np.int64)), y)
    # any cultivar space: the same coefficients in another column order give the same entries
    order = ("G3", "PHINT", "P1", "G2", "P5", "P2")
    perm = DaySimulator([run], ents, space=ceres_space(order), pad_days=PAD, readout=READOUT)
    th_p = th[:, [CUL_ORDER.index(c) for c in order]]
    np.testing.assert_array_equal(perm(th_p, np.zeros(1, np.int64)), y)
    # a subset: the other coefficients keep the run's own values
    sub = DaySimulator([run], ents, space=ceres_space(("G2", "P1")), pad_days=PAD, readout=READOUT)
    pub = _cultivar(run)
    th_s = np.asarray([[pub[3] * 1.05, pub[0] * 1.1]])
    full = pub[None].copy()
    full[0, 3], full[0, 0] = th_s[0, 0], th_s[0, 1]
    np.testing.assert_array_equal(sub(th_s, np.zeros(1, np.int64)), sim(full, np.zeros(1, np.int64)))
    with pytest.raises(ValueError, match=r"theta must be \[samples, 6\]"):
        sim(th[:, :5], np.zeros(1, np.int64))


@_x64
def test_a_soil_space_runs_the_season_of_with_soil(sites) -> None:
    from agrijax.calib.ceres import ceres_space
    from agrijax.calib.day_soil import with_soil
    from agrijax.calib.dssat_day import CUL_ORDER, DaySimulator, entry_values
    from agrijax.core.runtime import run_sites
    from agrijax.sites.dssat_free_run import _season_model, stack_trees

    for name in ("A", "B"):
        run, m = sites[name]
        sp = _site_space(name, m)
        blocks = ParamBlocks(ceres_space(CUL_ORDER), {name: sp})
        ents = _entries(run)
        sim = DaySimulator([run], [ents], space=blocks, sites=[name], soil=[m], pad_days=PAD, readout=READOUT)
        soil_th = _site_theta(sp, m, 1.3)
        th = np.concatenate([_cultivar(run), soil_th])[None]
        y = sim(th, np.zeros(1, np.int64))[0]
        # the same soil through with_soil on the host, the season by run_sites
        soil = {k: jnp.asarray(v) for k, v in m.values().items()}
        paths = {s.path for s in sp.specs}
        written = {k: v for k, v in sp.apply(soil, jnp.asarray(soil_th)).items() if k in paths}
        n = sim.n_days[(run.nl, run.mesev)]
        p = run.params()
        pw, fw, sw = with_soil(p, run.forcing(n), run.state(p), written, layers=m)
        o = run_sites(_season_model(run.mesev), *(stack_trees([t]) for t in (pw, fw, sw)))
        out = {k: np.asarray(v)[0] for k, v in o.items()}
        # one value per day (the crop outputs carry a plant axis of one)
        out = {
            k: v if k == "soil_sw" or v.ndim < 2 else v.reshape(v.shape[0], -1)[:, 0] for k, v in out.items()
        }
        out.update({f"sw{k + 1}": out["soil_sw"][:, k] for k in range(run.nl)})
        out["et"] = out["evap"] + out["ep"]
        ref = entry_values(out, ents)
        # two compiled programs: fusion or FMA contraction may differ in the last bits
        np.testing.assert_allclose(y[: len(ents)], ref, rtol=1e-12, atol=1e-12)
        # and the soil moved the season (the comparison is not of an unchanged run)
        base = DaySimulator([run], [ents], pad_days=PAD, readout=READOUT)(
            _cultivar(run)[None], np.zeros(1, np.int64)
        )[0]
        assert np.max(np.abs(y[: len(ents)] - base[: len(ents)])) > 1e-3


def _problem(sites, names, readout: str = READOUT, entries=_entries):
    """``(simulator, blocks)`` for the sites ``names`` with shared CUL_ORDER (``entries(run)``: the
    entries of each run)."""
    from agrijax.calib.ceres import ceres_space
    from agrijax.calib.dssat_day import CUL_ORDER, DaySimulator

    spaces = {n: _site_space(n, sites[n][1]) for n in names}
    blocks = ParamBlocks(ceres_space(CUL_ORDER), spaces)
    runs = [sites[n][0] for n in names]
    ents = [entries(r) for r in runs]
    sim = DaySimulator(
        runs,
        ents,
        space=blocks,
        sites=list(names),
        soil=[sites[n][1] for n in names],
        pad_days=PAD,
        readout=readout,
    )
    return sim, blocks


def _truth_and_start(sites, blocks: ParamBlocks):
    pub = _cultivar(sites["A"][0])
    sh_true, sh_0 = pub * np.asarray([1.08, 1.0, 0.97, 1.06, 0.93, 1.02]), pub
    per_true = {n: _site_theta(sp, sites[n][1], 1.25) for n, sp in blocks.per_site.items()}
    per_0 = {n: _site_theta(sp, sites[n][1], 1.0) for n, sp in blocks.per_site.items()}
    return np.asarray(blocks.join(sh_true, per_true)), np.asarray(blocks.join(sh_0, per_0))


def _obs(sim, theta):
    y = sim(np.repeat(theta[None], len(sim.runs), axis=0), np.arange(len(sim.runs)))
    return y, 1.0 / np.maximum(np.abs(y), 1e-3)


@_x64
@pytest.mark.parametrize("pair", [("A", "B"), ("A", "C")], ids=["one_group", "two_groups"])
def test_joint_loss_equals_the_sum_of_single_site_losses(sites, pair) -> None:
    sim, blocks = _problem(sites, pair)
    assert len(sim.groups) == (1 if pair == ("A", "B") else 2)
    truth, start = _truth_and_start(sites, blocks)
    obs, winv = _obs(sim, truth)
    # one compiled program per batch group for the call that made the observations
    assert sim.stats()["n_programs"] == len(sim.groups)
    f = jax.jit(sim.jax_loss(obs, winv))
    joint = np.asarray(f(jnp.asarray(start)))
    assert joint.shape == (2,) and np.all(np.isfinite(joint)) and np.all(joint > 0.0)
    singles = []
    for b, name in enumerate(pair):
        s1, _ = _problem(sites, (name,))
        cols = blocks.columns(name)
        o1, w1 = obs[b : b + 1], winv[b : b + 1]
        f1 = jax.jit(s1.jax_loss(o1[:, : s1.n_entries], w1[:, : s1.n_entries]))
        singles.append(float(np.asarray(f1(jnp.asarray(start[cols])))[0]))
        # the single-site observations at the truth equal the joint ones (same treatment, same values)
        y1 = s1(truth[cols][None], np.zeros(1, np.int64))
        np.testing.assert_allclose(y1[0], obs[b, : s1.n_entries], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(joint, singles, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(joint.sum(), sum(singles), rtol=1e-12, atol=0.0)
    # vmap of the loss equals one call per point
    pts = jnp.asarray(np.stack([start, 0.5 * (start + truth), truth]))
    vm = np.asarray(jax.jit(jax.vmap(sim.jax_loss(obs, winv)))(pts))
    per = np.stack([np.asarray(f(p)) for p in pts])
    np.testing.assert_allclose(vm, per, rtol=1e-12, atol=1e-20)
    assert np.all(per[2] < 1e-20)  # the truth reproduces its own observations
    # a call over samples of both sites equals the calls site by site; the batch shapes differ (four
    # samples against one), so only within rounding (bit for bit only at one batch shape)
    th = np.stack([start, truth])
    y = sim(np.repeat(th, 2, axis=0), np.asarray([0, 1, 0, 1]))
    for k in range(4):
        np.testing.assert_allclose(
            y[k], sim(th[k // 2][None], np.asarray([k % 2]))[0], rtol=1e-13, atol=1e-15
        )
    # the gradient of the joint loss is finite and moves shared and site columns
    g = np.asarray(jax.jit(jax.grad(lambda t: jnp.sum(sim.jax_loss(obs, winv)(t))))(jnp.asarray(start)))
    assert np.all(np.isfinite(g)) and np.any(g[:6] != 0.0)
    assert all(
        np.any(g[blocks.offsets[n] : blocks.offsets[n] + sp.n] != 0.0) for n, sp in blocks.per_site.items()
    )


@_x64
def test_refusals(sites) -> None:
    from agrijax.calib.ceres import ceres_space
    from agrijax.calib.dssat_day import CUL_ORDER, DaySimulator

    run, m = sites["A"]
    ents = [_entries(run)]
    sp = _site_space("A", m)
    blocks = ParamBlocks(ceres_space(CUL_ORDER), {"A": sp})
    with pytest.raises(ValueError, match="no SoilLayerMap"):
        DaySimulator([run], ents, space=blocks, sites=["A"], pad_days=PAD, readout=READOUT)
    with pytest.raises(ValueError, match="site blocks need sites"):
        DaySimulator([run], ents, space=blocks, soil=[m], pad_days=PAD, readout=READOUT)
    with pytest.raises(ValueError, match="have no block"):
        DaySimulator([run], ents, space=blocks, sites=["Z"], soil=[m], pad_days=PAD, readout=READOUT)
    with pytest.raises(ValueError, match="cannot write"):
        DaySimulator(
            [run], ents, space=ParamSpace([_spec("crop.rue", 3.0, 5.5)]), pad_days=PAD, readout=READOUT
        )
    with pytest.raises(ValueError, match="has no coefficient"):
        DaySimulator(
            [run], ents, space=ParamSpace([_spec("cultivar.xyz", 0.0, 1.0)]), pad_days=PAD, readout=READOUT
        )
    # SSKS missing in the file: the conductivity cannot be calibrated there
    sw = ParamSpace(
        [ParamSpec("swcn", 0.1, 4.0, "cm h-1", "test box", transform="log", index=0, label="SWCN_H1")]
    )
    with pytest.raises(ValueError, match="SSKS"):
        DaySimulator([run], ents, space=sw, soil=[m], pad_days=PAD, readout=READOUT)
    # a soil whose run changes during the season (SOILDYN) refuses soil coordinates
    from agrijax.calib.day_soil import SoilDynamicsError

    dyn = dataclasses.replace(m, soildyn=True)
    with pytest.raises(SoilDynamicsError):
        DaySimulator([run], ents, space=sp, soil=[dyn], pad_days=PAD, readout=READOUT)


@_x64
def test_a_treatment_without_soil_coordinates_in_a_soil_group_keeps_its_season(sites) -> None:
    """A treatment with no soil block and no layer map shares a batch group with one that writes
    soil: its entries equal the default-space simulator's at the same cultivar, the other's equal
    its own single-site run, and the loss and its gradient are finite."""
    from agrijax.calib.ceres import ceres_space
    from agrijax.calib.dssat_day import CUL_ORDER, DaySimulator

    run_a, m_a = sites["A"]
    run_b, _ = sites["B"]
    sp_a = _site_space("A", m_a)
    blocks = ParamBlocks(ceres_space(CUL_ORDER), {"A": sp_a, "B": ParamSpace([])})
    ents = [_entries(run_a), _entries(run_b)]
    sim = DaySimulator(
        [run_a, run_b], ents, space=blocks, sites=["A", "B"], soil=[m_a, None], pad_days=PAD, readout=READOUT
    )
    assert len(sim.groups) == 1
    pub = _cultivar(run_a)
    th_a = _site_theta(sp_a, m_a, 1.3)
    th = np.asarray(blocks.join(pub, {"A": th_a, "B": np.zeros(0)}))
    y = sim(np.stack([th, th]), np.asarray([0, 1]))
    y_b = DaySimulator([run_b], [ents[1]], pad_days=PAD, readout=READOUT)(pub[None], np.zeros(1, np.int64))[0]
    n_b = len(ents[1])
    np.testing.assert_allclose(y[1, :n_b], y_b[:n_b], rtol=1e-12, atol=1e-12)
    b_a = ParamBlocks(ceres_space(CUL_ORDER), {"A": sp_a})
    sim_a = DaySimulator(
        [run_a], [ents[0]], space=b_a, sites=["A"], soil=[m_a], pad_days=PAD, readout=READOUT
    )
    y_a = sim_a(np.concatenate([pub, th_a])[None], np.zeros(1, np.int64))[0]
    n_a = len(ents[0])
    np.testing.assert_allclose(y[0, :n_a], y_a[:n_a], rtol=1e-12, atol=1e-12)
    obs, winv = y, 1.0 / np.maximum(np.abs(y), 1e-3)
    start = th * np.where(np.arange(th.size) < len(CUL_ORDER), 1.02, 1.0)
    loss = np.asarray(jax.jit(sim.jax_loss(obs, winv))(jnp.asarray(start)))
    g = np.asarray(jax.jit(jax.grad(lambda t: jnp.sum(sim.jax_loss(obs, winv)(t))))(jnp.asarray(start)))
    assert np.all(np.isfinite(loss)) and np.all(np.isfinite(g))


@_x64
def test_per_site_cultivar_blocks_give_each_treatment_its_own_coefficients(sites) -> None:
    """Cultivar coefficients in the site blocks (no static cultivar columns): each treatment runs
    with the shared coefficient, its own site coefficients and the run's values for the rest."""
    from agrijax.calib.ceres import ceres_space
    from agrijax.calib.dssat_day import DaySimulator

    run_a, _ = sites["A"]
    run_b, _ = sites["B"]
    pub = _cultivar(run_a)
    blocks = ParamBlocks(
        ceres_space(("G3",)), {"A": ceres_space(("P1", "G2")), "B": ceres_space(("P2", "G2", "PHINT"))}
    )
    ents = [_entries(run_a), _entries(run_b)]
    sim = DaySimulator([run_a, run_b], ents, space=blocks, sites=["A", "B"], pad_days=PAD, readout=READOUT)
    assert sim.plans[(run_a.nl, run_a.mesev)].cul_static is None
    g3 = pub[4] * 0.93
    site_a = np.asarray([pub[0] * 1.1, pub[3] * 1.05])
    site_b = np.asarray([pub[1] * 1.02, pub[3] * 0.97, pub[5] * 1.03])
    th = np.asarray(blocks.join(np.asarray([g3]), {"A": site_a, "B": site_b}))
    y = sim(np.stack([th, th]), np.asarray([0, 1]))
    full_a, full_b = pub.copy(), pub.copy()
    full_a[[0, 3, 4]] = [*site_a, g3]
    full_b[[1, 3, 5, 4]] = [*site_b, g3]
    for k, (run, full) in enumerate(((run_a, full_a), (run_b, full_b))):
        y1 = DaySimulator([run], [ents[k]], pad_days=PAD, readout=READOUT)(full[None], np.zeros(1, np.int64))[
            0
        ]
        n = len(ents[k])
        np.testing.assert_allclose(y[k, :n], y1[:n], rtol=1e-12, atol=1e-12)
    obs, winv = y, 1.0 / np.maximum(np.abs(y), 1e-3)
    g = np.asarray(jax.jit(jax.grad(lambda t: jnp.sum(sim.jax_loss(obs, winv)(t))))(jnp.asarray(th * 1.02)))
    assert np.all(np.isfinite(g))


@_x64
def test_interleaved_batch_groups_keep_the_treatment_order(sites) -> None:
    """Treatments A (nine layers), C (eight), B (nine): the groups are not contiguous; the loss of
    each treatment equals its single-site loss, and the outputs of a mixed call follow its order."""
    names = ("A", "C", "B")
    sim, blocks = _problem(sites, names)
    assert sorted(len(v) for v in sim.groups.values()) == [1, 2]
    truth, start = _truth_and_start(sites, blocks)
    obs, winv = _obs(sim, truth)
    joint = np.asarray(jax.jit(sim.jax_loss(obs, winv))(jnp.asarray(start)))
    singles = []
    for b, name in enumerate(names):
        s1, _ = _problem(sites, (name,))
        e, cols = s1.n_entries, blocks.columns(name)
        f1 = jax.jit(s1.jax_loss(obs[b : b + 1, :e], winv[b : b + 1, :e]))
        singles.append(float(np.asarray(f1(jnp.asarray(start[cols])))[0]))
    np.testing.assert_allclose(joint, singles, rtol=1e-12, atol=0.0)
    calls = ((2, start), (0, truth), (1, start), (2, truth))
    y = sim(np.stack([t for _, t in calls]), np.asarray([i for i, _ in calls]))
    for k, (tid, th) in enumerate(calls):
        np.testing.assert_allclose(y[k], sim(th[None], np.asarray([tid]))[0], rtol=1e-13, atol=1e-15)


@_x64
def test_a_treatment_subset_of_the_loss_simulates_those_treatments_only(sites, monkeypatch) -> None:
    """``jax_loss(..., treatments=)`` returns the subset's losses in its order, equal to the full
    loss's, and builds only the batch groups the subset touches."""
    names = ("A", "C", "B")
    sim, blocks = _problem(sites, names)
    truth, start = _truth_and_start(sites, blocks)
    obs, winv = _obs(sim, truth)
    th = jnp.asarray(start)
    joint = np.asarray(jax.jit(sim.jax_loss(obs, winv))(th))
    built: list = []
    real = sim._sim_fn

    def spy(g):
        built.append(g)
        return real(g)

    monkeypatch.setattr(sim, "_sim_fn", spy)
    for sub in ((2, 0), (1,), (1, 2, 0), (0, 2)):
        built.clear()
        f = sim.jax_loss(obs, winv, treatments=sub)
        np.testing.assert_allclose(np.asarray(jax.jit(f)(th)), joint[list(sub)], rtol=1e-12, atol=0.0)
        assert sorted(built) == sorted({sim.where[i][0] for i in sub})
    g_sub = np.asarray(
        jax.jit(jax.grad(lambda t: jnp.sum(sim.jax_loss(obs, winv, treatments=(0, 2))(t))))(th)
    )
    g_all = np.asarray(
        jax.jit(jax.grad(lambda t: jnp.sum(sim.jax_loss(obs, winv)(t)[jnp.asarray([0, 2])])))(th)
    )
    np.testing.assert_allclose(g_sub, g_all, rtol=1e-10, atol=1e-14)
    for bad in ((), (0, 0), (3,), (-1,)):
        with pytest.raises(ValueError, match="distinct run indices"):
            sim.jax_loss(obs, winv, treatments=bad)


def _season_entries(run) -> list:
    """Entries of the season readout: those of :func:`_entries` on the run's real days, the outputs
    of the last real day, then final values, the crop-end date and the end status."""
    from agrijax.calib.dssat_day import CROP_END, E_DATE, E_DAY, E_FINAL, E_STATUS, OUT_NAMES, Entry

    last = run.last_real_day(PAD)
    day = [e for e in _entries(run) if e.kind != E_DAY or e.t <= last]
    day += [Entry(E_DAY, OUT_NAMES.index(o), last) for o in ("lai", "et", "sw1")]
    end = [Entry(E_FINAL, OUT_NAMES.index(o)) for o in ("gwad", "et", "sw1")]
    return day + end + [Entry(E_DATE, CROP_END), Entry(E_STATUS, 0)]


def _cut(run, n: int, *, extend: bool):
    """``run`` on its first ``n`` days; ``extend``: the days after them become its real weather
    extension (read by the season readout under harvest at maturity), else they are dropped."""
    head = lambda a: np.asarray(a)[:n]  # noqa: E731
    tail = lambda a: np.asarray(a)[n:]  # noqa: E731
    kw = {
        "series": {k: head(v) for k, v in run.series.items()},
        "forcing_crop": jax.tree.map(head, run.forcing_crop),
    }
    if extend:
        kw["extension"] = {k: tail(v) for k, v in run.series.items()}
        kw["extension_crop"] = jax.tree.map(tail, run.forcing_crop)
    return dataclasses.replace(run, **kw)


@_x64
def test_season_readout_of_the_loss_on_blocks_and_interleaved_groups(sites) -> None:
    """The default (season) readout of ``jax_loss`` on parameter blocks with soil, three treatments
    in two batch groups out of order (A, C, B). In the group of A and B the two treatments differ in
    their last real day and their open status: A at maturity with 40 season days and a real weather
    extension of 5 (last day 44, censored), B harvested on its reported day after 38 days (last day
    37, harvested); C, alone in the other group, harvested after 45 days. Each treatment reads its
    own last day and open status, so its loss equals its single-site loss, a treatment subset equals
    those losses in its order, and the gradient is finite."""
    from agrijax.calib.dssat_day import E_DAY, E_FINAL, E_STATUS, END_CENSORED, END_HARVESTED, OUT_NAMES

    names = ("A", "C", "B")
    local = dict(sites)
    run_a, m_a = sites["A"]
    run_b, m_b = sites["B"]
    run_c, m_c = sites["C"]
    local["A"] = (_cut(run_a, 40, extend=True), m_a)
    local["B"] = (dataclasses.replace(_cut(run_b, 38, extend=False), harvest="R"), m_b)
    local["C"] = (dataclasses.replace(run_c, harvest="R"), m_c)
    assert local["A"][0].n_ext == 5 and local["B"][0].n_days == 38
    sim, blocks = _problem(local, names, readout="season", entries=_season_entries)
    assert sim.readout == "season" and sorted(len(v) for v in sim.groups.values()) == [1, 2]
    group_ab = next(v for v in sim.groups.values() if len(v) == 2)
    assert sorted(group_ab) == [0, 2]
    assert sim.last_days == [44, 44, 37]
    truth, start = _truth_and_start(local, blocks)
    obs, winv = _obs(sim, truth)
    # the end status of each treatment: the synthetic season does not reach maturity, so each
    # reads its own open status (A censored, B and C harvested, on their own last real day)
    ents = [_season_entries(local[n][0]) for n in names]
    i_status = [next(j for j, e in enumerate(es) if e.kind == E_STATUS) for es in ents]
    statuses = [int(obs[b, i_status[b]]) for b in range(3)]
    assert statuses == [END_CENSORED, END_HARVESTED, END_HARVESTED], statuses
    # each treatment's final values are its outputs of its own last real day
    i_sw1 = OUT_NAMES.index("sw1")
    for b, es in enumerate(ents):
        j_day = next(
            j for j, e in enumerate(es) if e.kind == E_DAY and e.out == i_sw1 and e.t == sim.last_days[b]
        )
        j_fin = next(j for j, e in enumerate(es) if e.kind == E_FINAL and e.out == i_sw1)
        assert obs[b, j_fin] == obs[b, j_day]
    th = jnp.asarray(start)
    joint = np.asarray(jax.jit(sim.jax_loss(obs, winv))(th))
    assert joint.shape == (3,) and np.all(np.isfinite(joint)) and np.all(joint > 0.0)
    singles = []
    for b, name in enumerate(names):
        s1, _ = _problem(local, (name,), readout="season", entries=_season_entries)
        assert s1.last_days == [sim.last_days[b]]
        e, cols = s1.n_entries, blocks.columns(name)
        np.testing.assert_allclose(
            s1(truth[cols][None], np.zeros(1, np.int64))[0], obs[b, :e], rtol=1e-12, atol=1e-12
        )
        f1 = jax.jit(s1.jax_loss(obs[b : b + 1, :e], winv[b : b + 1, :e]))
        singles.append(float(np.asarray(f1(jnp.asarray(start[cols])))[0]))
    np.testing.assert_allclose(joint, singles, rtol=1e-12, atol=0.0)
    for sub in ((2, 0), (1,), (1, 2, 0)):
        f = sim.jax_loss(obs, winv, treatments=sub)
        np.testing.assert_allclose(np.asarray(jax.jit(f)(th)), joint[list(sub)], rtol=1e-12, atol=0.0)
    g = np.asarray(jax.jit(jax.grad(lambda t: jnp.sum(sim.jax_loss(obs, winv, treatments=(1, 2))(t))))(th))
    assert np.all(np.isfinite(g)) and np.any(g[:6] != 0.0)


@_x64
def test_two_treatments_of_one_site_and_shared_soil_coordinates(sites) -> None:
    """Two treatments of one site read the same block and give the same entries; soil coordinates
    in the shared space are written into every site's soil."""
    from agrijax.calib.ceres import ceres_space
    from agrijax.calib.dssat_day import CUL_ORDER, DaySimulator

    run_a, m_a = sites["A"]
    run_a2 = dataclasses.replace(run_a, key="SYNTH_t09")
    sp_a = _site_space("A", m_a)
    blocks = ParamBlocks(ceres_space(CUL_ORDER), {"A": sp_a})
    sim = DaySimulator(
        [run_a, run_a2], [_entries(run_a), _entries(run_a2)], space=blocks, sites=["A", "A"],
        soil=[m_a, m_a], pad_days=PAD, readout=READOUT,
    )  # fmt: skip
    pub = _cultivar(run_a)
    th = np.concatenate([pub * 1.02, _site_theta(sp_a, m_a, 1.4)])
    y = sim(np.stack([th, th]), np.asarray([0, 1]))
    np.testing.assert_array_equal(y[0], y[1])
    # CN, U and LL of the first .SOL layer shared by sites A and B
    run_b, m_b = sites["B"]
    soil_specs = [s for s in sp_a.specs if s.path in ("cn", "u") or (s.path == "ll" and s.index == 0)]
    shared = ParamSpace([*ceres_space(("P1", "G2")).specs, *soil_specs])
    sim2 = DaySimulator(
        [run_a, run_b],
        [_entries(run_a), _entries(run_b)],
        space=shared,
        soil=[m_a, m_b],
        pad_days=PAD,
        readout=READOUT,
    )
    vals = np.asarray(
        [pub[0], pub[3]] + [float(np.asarray(m_a.sol[s.path]).reshape(-1)[s.index or 0]) for s in soil_specs]
    )
    vals = vals * 1.01
    y2 = sim2(np.stack([vals, vals]), np.asarray([0, 1]))
    for k, (run, m) in enumerate(((run_a, m_a), (run_b, m_b))):
        y1 = DaySimulator([run], [_entries(run)], space=shared, soil=[m], pad_days=PAD, readout=READOUT)(
            vals[None], np.zeros(1, np.int64)
        )[0]
        n = len(_entries(run))
        np.testing.assert_allclose(y2[k, :n], y1[:n], rtol=1e-12, atol=1e-12)


@_x64
def test_single_precision_storage_with_soil_blocks(sites) -> None:
    """``dtype=float32`` with soil blocks runs, stays finite and agrees with float64 to the
    precision of single storage."""
    from agrijax.calib.ceres import ceres_space
    from agrijax.calib.dssat_day import CUL_ORDER, DaySimulator

    names = ("A", "B")
    spaces = {n: _site_space(n, sites[n][1]) for n in names}
    blocks = ParamBlocks(ceres_space(CUL_ORDER), spaces)
    runs = [sites[n][0] for n in names]
    ents = [_entries(r) for r in runs]
    soil = [sites[n][1] for n in names]
    s64 = DaySimulator(runs, ents, space=blocks, sites=list(names), soil=soil, pad_days=PAD, readout=READOUT)
    s32 = DaySimulator(
        runs,
        ents,
        space=blocks,
        sites=list(names),
        soil=soil,
        pad_days=PAD,
        readout=READOUT,
        dtype=np.float32,
    )
    th = np.asarray(
        blocks.join(_cultivar(runs[0]), {n: _site_theta(sp, sites[n][1], 1.2) for n, sp in spaces.items()})
    )
    y64 = s64(np.stack([th, th]), np.asarray([0, 1]))
    y32 = s32(np.stack([th, th]), np.asarray([0, 1]))
    assert np.all(np.isfinite(y32))
    np.testing.assert_allclose(y32, y64, rtol=2e-3, atol=2e-3 * np.max(np.abs(y64)))


@_x64
@pytest.mark.parametrize("names", [("A", "B"), ("A", "C", "B")], ids=["one_group", "interleaved_groups"])
def test_joint_gradient_is_the_column_sum_of_single_site_gradients(sites, names) -> None:
    sim, blocks = _problem(sites, names)
    truth, start = _truth_and_start(sites, blocks)
    obs, winv = _obs(sim, truth)
    g_joint = np.asarray(jax.jit(jax.grad(lambda t: jnp.sum(sim.jax_loss(obs, winv)(t))))(jnp.asarray(start)))
    acc = np.zeros_like(g_joint)
    for b, name in enumerate(names):
        s1, _ = _problem(sites, (name,))
        e, cols = s1.n_entries, blocks.columns(name)
        o1, w1 = obs[b : b + 1, :e], winv[b : b + 1, :e]
        g1 = jax.jit(jax.grad(lambda t, s1=s1, o1=o1, w1=w1: jnp.sum(s1.jax_loss(o1, w1)(t))))
        np.add.at(acc, cols, np.asarray(g1(jnp.asarray(start[cols]))))
    assert np.any(g_joint != 0.0)
    np.testing.assert_allclose(g_joint, acc, rtol=1e-8, atol=1e-10 * np.max(np.abs(g_joint)))
