# Contributing to Agri-JAX

## Setup

```bash
uv sync --all-extras          # creates .venv with CPU JAX, all optional groups
uv run pre-commit install     # ruff, ruff-format, agrijax lint on every commit
```

Everything runs through `uv run ...` (or `.venv/bin/python`). Do not `pip install` into the venv by hand; add dependencies with `uv add` so `uv.lock` stays authoritative.

## The three rules for process functions

A process is a pure function `(state, params, forcing_t) -> state`, declared with `@process`:

```python
@process(reads=("crop.stage", "soil.theta"), writes=("crop.lai",), fortran_name="MZ_GROSUB")
def leaf_growth(state, params, forcing_t):
    """One-line summary; formula and units.

    Source: DSSAT-CSM MZ_GROSUB.for, eqs. 12-15.
    """
    ...
    return eqx.tree_at(lambda s: s.crop.lai, state, new_lai)
```

1. **Everything read is in the arguments; everything changed is in the return value.** No globals, no mutation. Update with `eqx.tree_at` / `.set(path, value)` / `.replace(...)` and declare every changed field in `writes`.
2. **Branch with `jnp.where` / `jnp.select`, never with Python `if` on state, params or forcing.** Both branches must stay finite (guard `log`, `sqrt`, `/` with `jnp.maximum` / `jnp.clip`).
3. **Never write a loop.** Time and samples are handled by the runtime (`lax.scan`, `vmap`); soil layers are an array axis.

`python -m agrijax.core.lint <paths>` enforces these as AJ001-AJ006 (AJ006: a Python loop over a shape-derived range or an array, i.e. an unrolled layer loop; a true recurrence over depth goes through `agrijax.core.depth_scan.depth_scan`). Set `AGRI_JAX_CHECK=1` to make every process call verify at runtime that only the declared `writes` changed.

## Commands

| What | Command |
|---|---|
| Format + lint | `uv run ruff format . && uv run ruff check .` |
| Types | `uv run pyright` |
| Three-rules lint | `uv run python -m agrijax.core.lint src/agrijax --strict` |
| Unit tier (every push, no data) | `uv run pytest tests/unit -q` |
| Unit tier in float32 | `AGRI_JAX_X64=0 uv run pytest tests/unit -q` |
| Diff tier (needs Fortran dumps) | `uv run pytest tests/diff -q --data-dir ~/agri_jax_data` |
| Integration tier (needs scenario dirs) | `uv run pytest tests/integration -q` |
| GPU tier | `uv run pytest tests/gpu -q` (skips without a GPU) |
| Everything with write checks on | `AGRI_JAX_CHECK=1 uv run pytest -q` |
| One tier by marker | `uv run pytest -m unit` / `-m "not gpu"` |
| Slow tests (`@pytest.mark.slow`, deselected by default) | `uv run pytest --runslow` (all) or `uv run pytest -m slow` (only them); any `-m` expression or an explicit node id also selects them |

Tests are marked with their tier automatically from their directory (`tests/<tier>/`). Data-dependent tiers skip when the data is missing; the default data root is `~/agri_jax_data`, overridable with `--data-dir` or `AGRI_JAX_DATA`.

## Layout

`src/agrijax/`: `core/` (State/Params/Forcing, `@process`, `Model`, runtime, units, lint), `processes/` (soil_water, pet, crop/ceres_maize, canopy, arbitration), `models/`, `io/` (dssat, rzwqm), `calib/`, `port/`, `report/`.

## Pull requests

Branch names `port/<subroutine>`, `proc/<name>`, `io/<format>`, `calib/<method>`, `fix/...`. CI runs ruff, pyright, the three-rules lint and the unit tier on Python 3.11/3.12/3.13 plus one float32 pass. PRs containing ported Fortran code state the number of dump cases, the tolerance grade passed. Never commit data, dumps or run directories.
