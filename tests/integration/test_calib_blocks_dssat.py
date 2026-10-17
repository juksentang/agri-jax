"""Shared cultivar and per-site soil blocks on DSSAT v4.8.6.0 example treatments (``calib/space.py``
:class:`ParamBlocks`, ``calib/dssat_day.py`` :class:`DaySimulator` with soil coordinates).

UFGA8201 treatment 2 (IBMZ910014, eight ``.SOL`` layers, nine model layers) and BRPI0202 treatment 5
(BRPI020002, four ``.SOL`` layers split into nine, ``SSKS`` given) share one batch group (nine
layers, ``MESEV = R``); BRPI0202 treatment 1 (BRPI020001, three ``.SOL`` layers, eight model
layers) is in another. Inputs are built from the public files (``source="native"``) after a
``dscsm048`` reference run; each site's soil space is requested by ``.SOL`` layer with the
coordinates on an end of their domain left out on record (``SRGF = 1.0`` in the top layers).

* The joint loss of two sites equals the sum of the single-site losses, treatment by treatment
  (1e-12), within one batch group and across two.
* One compiled program per batch group.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.calib.ceres import ceres_space
from agrijax.calib.day_soil import soil_layer_map
from agrijax.calib.dssat_day import CUL_ORDER, E_DATE, E_DAY, E_FINAL, OUT_NAMES, DaySimulator, Entry
from agrijax.calib.soil import SITE_BOUNDS, soil_space, without_domain_bounds
from agrijax.calib.space import ParamBlocks
from agrijax.port.run_fortran import DSSAT_ENGINE, dscsm_paths
from agrijax.sites.dssat_free_run import free_run_inputs, run_reference

pytestmark = [
    pytest.mark.allow_skip(reason="needs dscsm048 v4.8.6.0 (AGRI_JAX_DSSAT) and the DSSAT example data"),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the loss sums are compared in float64"),
]

MAIZE = DSSAT_ENGINE / "example_data" / "Maize"
#: site name -> (experiment, treatment)
CASES = {"UFGA": ("UFGA8201", 2), "BRPI_IRR": ("BRPI0202", 5), "BRPI_RF": ("BRPI0202", 1)}
#: requested soil coordinates (four scalars, LL and DUL of the top two .SOL
#: layers, SAT and SRGF of the top layers; SRGF = 1.0 is left out on record)
REQUEST = (
    ["cn", "swcon", "u", "slpf", "ll", "dul", "sat", "srgf"],
    {"ll": [0, 1], "dul": [0, 1], "sat": [0], "srgf": [0, 1, 2]},
)
PAD = 10


def _entries(run) -> list[Entry]:
    n = run.n_days
    out = [Entry(E_DATE, 4), Entry(E_DATE, 10), Entry(E_FINAL, OUT_NAMES.index("cwad"))]
    out += [Entry(E_DAY, OUT_NAMES.index("lai"), t) for t in range(20, n, 14)]
    out += [Entry(E_DAY, OUT_NAMES.index("et"), t) for t in range(3, n, 5)]
    out += [Entry(E_DAY, OUT_NAMES.index(f"sw{k}"), t) for k in (1, 3, 5) for t in range(2, n, 7)]
    return out


@pytest.fixture(scope="module")
def sites(tmp_path_factory: pytest.TempPathFactory) -> dict:
    if not dscsm_paths(DSSAT_ENGINE)[0].is_file() or not MAIZE.is_dir():
        pytest.skip(f"dscsm048 / DSSAT example data not found under {DSSAT_ENGINE}")
    root = tmp_path_factory.mktemp("blk")
    out = {}
    for name, (exp, trno) in CASES.items():
        fx = MAIZE / f"{exp}.MZX"
        ref = run_reference(fx, trno, Path(root) / f"{exp[:4]}{trno}")
        run = free_run_inputs(fx, trno, ref, source="native")
        m = soil_layer_map(run, fx)
        q, lay, excluded = without_domain_bounds(m.sol, *REQUEST)
        out[name] = (run, m, soil_space(m.sol, q, lay, SITE_BOUNDS), excluded)
    return out


def test_site_facts_and_exclusions(sites) -> None:
    run_u, m_u, sp_u, ex_u = sites["UFGA"]
    run_b, m_b, sp_b, ex_b = sites["BRPI_IRR"]
    run_r, m_r, _, _ = sites["BRPI_RF"]
    assert (run_u.nl, run_u.mesev) == (run_b.nl, run_b.mesev) == (9, "R") and run_r.nl == 8
    assert not (run_u.soildyn or run_b.soildyn or run_r.soildyn)
    assert (m_u.n_sol, m_b.n_sol, m_r.n_sol) == (8, 4, 3) and m_u.exact and m_b.exact and m_r.exact
    assert m_b.sol_layer.tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 3]
    assert [e.name for e in ex_u] == [e.name for e in ex_b] == ["SRGF_H1", "SRGF_H2"]
    assert sp_u.n == sp_b.n == 4 + 4 + 1 + 1


def _problem(sites, names):
    blocks = ParamBlocks(ceres_space(CUL_ORDER), {n: sites[n][2] for n in names})
    runs = [sites[n][0] for n in names]
    sim = DaySimulator(
        runs, [_entries(r) for r in runs], space=blocks, sites=list(names),
        soil=[sites[n][1] for n in names], pad_days=PAD,
    )  # fmt: skip
    return sim, blocks


def _site_theta(sp, m, shift: float) -> np.ndarray:
    v = np.asarray([float(np.asarray(m.sol[s.path]).reshape(-1)[s.index or 0]) for s in sp.specs])
    return np.asarray(sp.to_physical(sp.to_unconstrained(v) + shift))


@pytest.mark.parametrize("pair", [("UFGA", "BRPI_IRR"), ("UFGA", "BRPI_RF")], ids=["one_group", "two_groups"])
def test_joint_loss_equals_the_sum_of_single_site_losses(sites, pair) -> None:
    sim, blocks = _problem(sites, pair)
    assert len(sim.groups) == (1 if pair[1] == "BRPI_IRR" else 2)
    pub = np.asarray([sites["UFGA"][0].published()[c] for c in CUL_ORDER])
    sh_true = pub * np.asarray([1.08, 1.0, 0.97, 1.06, 0.93, 1.02])
    truth = np.asarray(blocks.join(sh_true, {n: _site_theta(sites[n][2], sites[n][1], 0.4) for n in pair}))
    start = np.asarray(blocks.join(pub, {n: _site_theta(sites[n][2], sites[n][1], 0.0) for n in pair}))
    obs = sim(np.repeat(truth[None], 2, axis=0), np.arange(2))
    assert sim.stats()["n_programs"] == len(sim.groups)
    winv = 1.0 / np.maximum(np.abs(obs), 1e-3)
    joint = np.asarray(jax.jit(sim.jax_loss(obs, winv))(jnp.asarray(start)))
    assert np.all(np.isfinite(joint)) and np.all(joint > 0.0)
    singles = []
    for b, name in enumerate(pair):
        s1, _ = _problem(sites, (name,))
        cols = blocks.columns(name)
        e = s1.n_entries
        singles.append(
            float(np.asarray(jax.jit(s1.jax_loss(obs[b : b + 1, :e], winv[b : b + 1, :e]))(start[cols]))[0])
        )
    np.testing.assert_allclose(joint, singles, rtol=1e-12, atol=0.0)
