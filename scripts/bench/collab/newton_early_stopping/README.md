# Newton early stopping with implicit-gradient checks (collaborator experiment)

Contributed by Jiaqi ZHANG. A bench experiment on the fixed-step Richards mode (the figures below were measured on the version the experiment was written against):

- Newton on each sub-step stops once the scaled residual `max(|R| dt / tl)` is at most 1e-14 (at most 12 updates), and the residual and tridiagonal Jacobian from the joint assembly are reused for the stopping test.
- The implicit gradient is kept. It is accepted only after checks on forward convergence, finiteness, and the scaled residual of the adjoint system (1e-10); a failed check gives a non-finite gradient, which the caller rejects.

Measured by the author on CA-TPA 2015-01-01..07 (37 nodes, one parameter set, 24 sub-steps a day, float64, CPU):

- objective and gradient 1.8-2.4x faster than fixed 12 iterations with the implicit gradient;
- gradient relative L2 difference 4e-15;
- central finite differences within 4.5e-6.

This is a small case. Full years, GPU batches and extreme wet or dry conditions are not covered.

The scripts use `FixedStepping` / `RichardsParams(stepping=...)` for the sub-step and iteration settings, and the experiment installs its solver on `agrijax.processes.soil_water.fixed_cn`, where `richards_step` looks it up; `tests/unit/test_bench_newton_early_stopping.py` checks that the patch is the solver in use.

The scripts expect this layout and do not change `src/`:

```
<root>/agri-jax/                          # the repository
<root>/output/profiling/<these scripts>
<root>/output/profiling/collab_inputs/data/agri_jax_data/   # CA-TPA reference inputs (not in the repo)
```

Run `python run_experiment.py --check-only | --smoke` from `<root>`.
