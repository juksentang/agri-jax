"""The PR #1 Newton early-stopping bench scripts follow the Richards API of main.

``scripts/bench/collab/newton_early_stopping`` is outside ``src`` (no pyright) and needs the CA-TPA
inputs to run, so nothing else would notice an API change under it (splitting the Richards physics
from its integrator moved the stepping settings to ``FixedStepping`` and the fixed-step solver to
``fixed_cn``). This test runs, in
a subprocess because the scripts switch on x64 when imported: the constructors the scripts call,
the helpers they import, and the experiment's solver patch on a one-step problem, checking that the
patched solver is the one :func:`richards_step` uses (its audit callback fires).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "scripts" / "bench" / "collab" / "newton_early_stopping"

_PROBE = """
import jax, jax.numpy as jnp
from experiment_adaptive_richards import install_adaptive, R
from agrijax.processes.soil_water import fixed_cn
from agrijax.processes.soil_water.hydraulics import k_of_h, theta_of_h
from tests.unit.test_richards import SOIL_REC1, SOIL_REC2, SoilHydraulicParams, nodes

stepping = R.FixedStepping(n_sub=24, n_iter=12, grad="implicit")
soil = nodes(SoilHydraulicParams.from_rzwqm_records(SOIL_REC1[:1], SOIL_REC2[:1]), 3)
grid = R.RichardsGrid(tl=jnp.ones(3), delz=jnp.ones(2), dz_top=jnp.asarray(1.0))
R.RichardsParams(soil=soil, grid=grid, stepping=stepping)
audit = {"forward": [], "backward": []}
install_adaptive(1e-14, audit)
assert fixed_cn._solve_implicit is R._solve_implicit
h = jnp.full(3, -100.0)

def loss(supply):
    r = R.richards_step(h, theta_of_h(h, soil), jnp.asarray(0.0), soil, grid, supply,
                        jnp.asarray(0.0), jnp.zeros(3), jnp.asarray(1.0), jnp.asarray(1.0),
                        jnp.asarray(-15000.0), jnp.asarray(0.0), stepping)
    return jnp.sum(r.h)

jax.block_until_ready(jax.grad(loss)(jnp.asarray(0.01)))
jax.effects_barrier()
assert audit["forward"], "the experiment's solver was not used by richards_step"
assert audit["backward"], "the experiment's VJP was not used"
assert jnp.isfinite(k_of_h(h, soil)).all()
print("OK", len(audit["forward"]), len(audit["backward"]))
"""


def test_bench_scripts_follow_the_richards_api() -> None:
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        PYTHONPATH=os.pathsep.join([str(BENCH), str(REPO / "src"), str(REPO)]),
    )
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=REPO, env=env, capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.startswith("OK"), out.stdout
