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

`python -m agrijax.core.lint <paths>` enforces these as AJ001-AJ006 (AJ006: a Python loop over a shape-derived range or an array, i.e. an unrolled layer loop; a true recurrence over depth goes through `agrijax.core.depth_scan.depth_scan`), and reports bare numeric literals as AJ007 (next section). Set `AGRI_JAX_CHECK=1` to make every process call verify at runtime that only the declared `writes` changed.

## Coefficients: declared once, with unit, meaning and provenance

Every number of a model's equations that is not a state, a forcing or a parameter read from an input file is a *coefficient*, declared once as a field of an `agrijax.core.coefficients.Coefficients` class with `coef(value, unit, description, provenance)`:

```python
from agrijax.core.coefficients import Coefficients, Provenance, coef


class GrosubCoefficients(Coefficients):
    sla_leaf: float = coef(
        267.0,
        "cm2 g-0.8",
        "leaf area per leaf weight^0.8 (stages 1-3)",
        Provenance.at(
            "dssat-4.8.6.0",
            "Plant/CERES-Maize/MZ_GROSUB.for:1246",
            routine="MZ_GROSUB",
            statement="PLA    = (LFWT+GROLF)**0.8*267.0",
        ),
    )
```

- `unit` follows the grammar of `agrijax.core.units.parse_unit` and is checked when the class is defined.
- `Provenance.ref_version` is spelled as the `@ref_version` of the process registry key (`dssat-4.8.6.0`, `rzwqm2-4.6`, `asce-ewri-2005`, or `none` with a `paper`). `file`, `line` and `routine` locate the coefficient in the reference source; `paper` and `equation` name the published equation.
- `statement` (the original source statement) is accepted only for a reference whose licence allows quoting it (DSSAT-CSM, BSD-3). The RZWQM2 source tree carries no licence file, so by project policy its coefficients record file, line, routine and the published equation (`paper`, `equation`), never the statement. A new reference is added once with `register_reference(...)`, which records that decision.
- A coefficient is calibratable by default (a pytree leaf). `static=True` is for integer thresholds (not a leaf; changing one retraces); `calibrate=False` keeps a leaf out of the calibration vector. `coefficients.as_arrays()` gives array leaves for `jax.grad`; `to_vector()` / `from_vector()` map the calibratable coefficients to a flat array and back; `coefficient_table(...)` lists them all.
- Defaults are Python floats, so naming a literal does not change a result: the traced program has the same constants in the same operation order.
- Numerical guards (the `1e-12` floors that keep gradients finite) are not coefficients: declare them as named module constants (`_TINY = numerical_guard("richards.tiny", 1e-12, "why")`). Unit conversions go through the named adapters of `agrijax.core.units` (`mm_to_cm`, `KG_HA_PER_G_M2`).

Lint rule **AJ007** reports the bare numeric literals left in `@process` functions and numerical kernels. The whitelist lives in `agrijax.core.lint` (`AJ007_TRIVIAL`, `AJ007_MAX_INDEX`, `AJ007_MAX_EXPONENT`, `AJ007_STRUCTURAL_CALLS`, `AJ007_STRUCTURAL_KEYWORDS`): `0`, `1` and `-1`; small integers used as indices, slice bounds, axes, shapes and counts; small integer exponents; comparisons with a shape query. Everything else is reported, powers of ten with a hint to use a unit adapter. AJ007 is a warning that `--strict` (run by CI and pre-commit) turns into a failure, like every other warning; `--strict-aj007` fails on AJ007 alone, `--aj007-report` prints the count per file, and `--ignore AJ007` turns it off. New code should not add AJ007 findings.

## Commands

| What | Command |
|---|---|
| Format + lint | `uv run ruff format . && uv run ruff check .` |
| Types | `uv run pyright` |
| Three-rules lint | `uv run python -m agrijax.core.lint src/agrijax --strict` |
| Bare-literal count per file (AJ007) | `uv run python -m agrijax.core.lint src/agrijax --quiet --aj007-report` |
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
