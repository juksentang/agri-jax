"""Three-node, one-step checks for the experimental implicit-gradient guard."""
import json
from pathlib import Path

from experiment_adaptive_richards import install_adaptive, R, jax, jnp, np
from agrijax.processes.soil_water.hydraulics import k_of_h, theta_of_h
from tests.unit.test_richards import CATPA_REC1, CATPA_REC2, SoilHydraulicParams, nodes, step_args


def main():
    tol = 1e-14
    rec1 = CATPA_REC1[:1].copy()
    rec1[:, 3] = 1.0
    soil = nodes(SoilHydraulicParams.from_rzwqm_records(rec1, CATPA_REC2[:1]), 3)
    grid = R.RichardsGrid(tl=jnp.ones(3), delz=jnp.ones(2), dz_top=jnp.asarray(1.0))
    cfg = R._SolveCfg(n_iter=1, jacobian="newton", dv_max=1.0, c_floor=1e-7, chop=True)
    rows = []
    for name, head, demand, expected in [
        ("iteration_limit", -100.0, 1.0, False),
        ("saturated_singular_root", -1.0, 1.0, False),
        ("unsaturated_regular_root", -100.0, None, True),
    ]:
        audit = {"forward": [], "backward": []}
        install_adaptive(tol, audit)
        h0 = jnp.full(3, head)
        q = k_of_h(h0, soil)[0] if demand is None else demand
        a = step_args(h0, theta_of_h(h0, soil), soil, grid, q_demand=q)

        def loss(aa):
            h, _ = R._solve_implicit(cfg, h0, aa)
            return jnp.sum(h), h

        (value, h), gradient = jax.block_until_ready(
            jax.jit(jax.value_and_grad(loss, has_aux=True))(a))
        jax.effects_barrier()
        r, dl, d, du = R.tridiagonal_jacobian(lambda hh: R.richards_residual(hh, hh, a), h)
        residual = float(jnp.max(jnp.abs(r) * a.dt / a.tl))
        dense = np.diag(np.asarray(d)) + np.diag(np.asarray(dl[1:]), -1) + np.diag(np.asarray(du[:-1]), 1)
        finite_gradient = all(np.all(np.isfinite(x)) for x in jax.tree.leaves(gradient))
        assert np.isfinite(float(value)) and np.all(np.isfinite(h))
        assert len(audit["forward"]) == len(audit["backward"]) == 1
        assert audit["backward"][0]["accepted"] == expected
        assert finite_gradient == expected
        if name == "iteration_limit":
            assert audit["forward"][0]["iterations"] == 1 and residual > tol
        else:
            assert audit["forward"][0]["iterations"] == 0 and residual <= tol
        if name == "saturated_singular_root":
            np.testing.assert_array_equal(dense, [[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]])
            np.testing.assert_array_equal(dense @ np.ones(3), np.zeros(3))
            assert np.linalg.matrix_rank(dense) == 2
        else:
            assert np.linalg.matrix_rank(dense) == 3
        rows.append(dict(case=name, passed=True, nodes=3,
            iterations=audit["forward"][0]["iterations"], residual=residual,
            jacobian_rank=int(np.linalg.matrix_rank(dense)),
            gradient_finite=bool(finite_gradient), accepted=expected))
    dest = Path(__file__).with_suffix(".json")
    dest.write_text(json.dumps(dict(tolerance=tol, checks=rows), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)
    print("RESULT", dest, flush=True)


if __name__ == "__main__":
    main()
