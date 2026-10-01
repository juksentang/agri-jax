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

`python -m agrijax.core.lint <paths>` enforces these as AJ001-AJ006, AJ020 and AJ021 (AJ020: a NumPy call on a value that depends on the arguments, use `jnp`; AJ021: an argument changed in place, return a new state with `eqx.tree_at` or `.replace`; AJ006: a Python loop over a shape-derived range or an array, i.e. an unrolled layer loop; a true recurrence over depth goes through `agrijax.core.depth_scan.depth_scan`), and reports bare numeric literals as AJ007 (next section). Set `AGRI_JAX_CHECK=1` to make every process call verify at runtime that only the declared `writes` changed.

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

Lint rule **AJ007** reports the bare numeric literals left in `@process` functions and numerical kernels. The whitelist lives in `agrijax.core.lint` (defined in `agrijax/core/_lint/rules.py`: `AJ007_TRIVIAL`, `AJ007_MAX_INDEX`, `AJ007_MAX_EXPONENT`, `AJ007_STRUCTURAL_CALLS`, `AJ007_STRUCTURAL_KEYWORDS`): `0`, `1` and `-1`; small integers used as indices, slice bounds, axes, shapes and counts; small integer exponents; comparisons with a shape query. Everything else is reported, powers of ten with a hint to use a unit adapter. AJ007 is a warning that `--strict` (run by CI and pre-commit) turns into a failure, like every other warning; `--strict-aj007` fails on AJ007 alone, `--aj007-report` prints the count per file, and `--ignore AJ007` turns it off. New code should not add AJ007 findings.

## Ports, registry keys and lags

Modules of different slots exchange data only through the port records of `agrijax.iface` (P1-P11: crop water, root record, node uptake, sink inputs, PET fluxes, canopy, surface soil water, weather, snow, crop nitrogen, water ledger). `agrijax.iface.contract.PORTS` gives each port's global path, fields (unit string in `agrijax.core.units` syntax, dims from `agrijax.core.dims.DIMS`, grid), producers, consumers and time semantics; a producer and a consumer of a port use the same record class, so the unit strings on both sides agree by construction. A slot package imports `agrijax.core` and `agrijax.iface`, never another slot's package: lint rule **AJ008** reports any import of `processes/<b>/` from code under `processes/<a>/` (absolute, `from agrijax.processes import b`, or a relative import that climbs into another slot, also inside functions), and `--strict` enforces it.

A process of the library is registered under `slot/impl@ref_version:variant`. A variant other than `faithful` of a key with a reference can only be registered when its faithful sibling `slot/impl@ref_version:faithful` is already registered (define the faithful process first); `ref_version = none` (replays, demonstrations) is exempt. The rule is enforced by the registry, so it binds plugins too.

A `Day` lists the one-day lags the coupling contract allows (`agrijax.iface.contract.allowed_lags(slot)`); `Day.check` rejects a lagged read no allowed lag covers and reports the allowed lags an implementation does not use (`Day.lag_report`), so swapping an implementation that reads less needs no change to the day. `exact_lags=True` asks for the two-way equality.

## Conformance kit

Every slot implementation, a plugin's included, provides a `ConformanceCase` (`agrijax.testing.conformance`): a `make(rng, dtype, variant)` that builds synthetic inputs from a NumPy generator (no data, never `jax.random`), the module binding (own subtree and ports), its conserved quantities (`Balance`) and its gradient spec (`GradSpec`). One command runs every generic check on it:

```bash
python -m agrijax.testing.conformance --key 'pet/*'          # this repository's cases
python -m agrijax.testing.conformance --list                 # the cases and their origin
python -m agrijax.testing.conformance --package my-plugin --no-builtin
```

The checks, in order: registry metadata and provenance (licence, `ref_build` of a faithful process, the `Source:` line, the case's origin); the lint with every rule (AJ001-AJ012, AJ020, AJ021; AJ009 and AJ011 are reported without failing the check) on the process module and the same-package modules it imports, every function as a kernel; coefficient labels; declared dims and shapes; units, and every bound port's record against `agrijax.iface.contract.PORTS` unit string for unit string; the slot contract (`SLOT_CONTRACTS`: reads and writes inside the own subtree and the slot's ports, never writing an `in` port); the writes under `AGRI_JAX_CHECK=1`; the reads by perturbation (every undeclared state leaf, and every undeclared forcing field when `forcing_fields` is given, set to NaN or a random value must leave the outputs bit for bit unchanged); daily closure of each balance in float64 and float32; eager against `jit`, `vmap(jit)` against per-sample `jit` and batch independence; float32 against float64 (no implicit upcast, finite); finite gradients in the case's and the `ste` gradient mode; gradients against central differences with steps `h` and `h/10` (a disagreement of the two is a kink next to the point: move the point or exempt the parameter in `fd_exempt`); replay against coupled binding and a `BindingError` for a used port left unbound. A case that leaves a check out gives the reason (`no_balance`, `no_grad`, `no_slot_contract`); `exempt_checks={name: reason}` records a known gap: the check must still fail (reported as xfail), and passing fails the test so the exemption is removed. A plugin registers its cases under the entry point group `agrijax.conformance`. `tests/unit/conformance/` runs every case of this repository in the unit tier, checks that every check fails on a fixture that breaks its rule, and that every registered key has a case or a written exemption (`agrijax.testing.conformance.EXEMPT`).

## Execution settings: the backend decides the layout, not the model

Some choices change how a program is laid out for a backend and nothing it computes. They live in `agrijax.core.execution`, are read at trace time by the sanctioned primitives, and are invisible to process authors: a process written once runs unchanged on a CPU or a GPU.

- `depth_unroll`: whether `depth_scan` unrolls its static-length layer recurrence (`lax.scan(..., unroll=True)`) or keeps it an XLA while loop. Default per target platform: unrolled on GPUs (a layer loop there is a few tiny kernels per layer and per day; unrolling made the DSSAT free-run day about 3x faster on an H100 in steady calls, at about 3x the compile time; the layer recurrences give the same bits, the code XLA fuses around them may round differently, measured at most 2e-16 relative in yield), a loop on CPUs and any other platform (unrolling is slower there and doubles the compile time). A `depth_scan(..., unroll=...)` call with an explicit value keeps it; kernel code should normally leave it out.
- Cost and opt-out: the unrolled programs compile about 3.5x longer on a GPU (for the DSSAT free-run day on a whole H100, about +45 s for the six group programs; a one-off run at 1e5 seasons on a cold compilation cache is therefore slower unrolled, break-even at about 50 repeated calls; with a warm persistent cache about +2.5 s). For one-off GPU runs set `AGRI_JAX_DEPTH_UNROLL=0` or use `execution(depth_unroll=False)`.
- Resolution: an `execution(depth_unroll=...)` context, else `AGRI_JAX_DEPTH_UNROLL` (`1`/`0`/`auto`), else the default of the target platform. The runtime entry points (`run`, `run_batch`, `run_sites`, `run_batch_chunked`, `run_and_grad`) set the target platform from the device of their committed inputs (uncommitted arrays run on the default device and do not decide) and key their cached jitted runners on the resolved settings; otherwise it is the `jax.default_device` or the default backend.
- Pitfall: settings are read at trace time, and JAX reuses the trace of any function whose identity is stable across traces: a `jax.jit` of the same function object, a `jax.checkpoint`-wrapped function, a `lax.scan` body or other inner function defined once and reused. Such a function keeps the settings of its first trace. When you change a setting yourself, build fresh closures (a new step, a new `jax.jit`) inside `execution(...)`; when you jit a model by hand for a non-default device, trace it inside `execution(platform=...)`.
- A new setting must leave the model unchanged: on a given backend the outputs are bit for bit the same, or differ only by the rounding of a different fusion with a measured, tested bound; it comes with a test of that (see `tests/unit/test_execution.py`, and the on/off comparison in `tests/integration/test_day_dssat486_free_gpu.py`).

## Commands

| What | Command |
|---|---|
| Format + lint | `uv run ruff format . && uv run ruff check .` |
| Types | `uv run pyright` |
| Three-rules lint (AJ001-AJ012, AJ020-AJ021) | `uv run python -m agrijax.core.lint src/agrijax --strict` |
| Conformance kit, one slot | `uv run python -m agrijax.testing.conformance --key 'crop/*'` (or `pytest tests/unit/conformance --agrijax-key 'crop/*'`) |
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

`src/agrijax/`: `core/` (State/Params/Forcing, `@process`, `Model`, runtime, units, lint), `iface/` (port records and the coupling contract table), `testing/conformance/` (the conformance kit and this repository's cases), `processes/` (soil_water, pet, crop/ceres_maize, n_supply, canopy, arbitration), `models/`, `io/` (dssat, rzwqm), `calib/`, `port/`, `report/`.

## Pull requests

Branch names `port/<subroutine>`, `proc/<name>`, `io/<format>`, `calib/<method>`, `fix/...`. CI runs ruff, pyright, the three-rules lint and the unit tier on Python 3.11/3.12/3.13 plus one float32 pass. PRs containing ported Fortran code state the number of dump cases, the tolerance grade passed. Never commit data, dumps or run directories.
