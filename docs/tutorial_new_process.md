# Writing a new process: a leaf-area formula from NumPy to a checked component

Agri-JAX is an independent implementation from published equations, validated against DSSAT-CSM (BSD-3) and RZWQM2 outputs. A model is a state plus an ordered list of processes, so trying a new formula means writing one process.

This tutorial takes a leaf-area expansion formula (logistic growth in thermal time, scaled by water stress, then senescence) from a few lines of NumPy to a process that runs in a model, passes the project's automated checks and could be merged. The coefficient values are illustrative, not a calibrated crop.

No data is needed. From a clone of the repository:

```bash
git clone https://github.com/juksentang/agri-jax && cd agri-jax
uv sync                                             # the core dependencies are enough here
uv run python examples/new_process_tutorial.py     # seconds to a minute on a CPU
```

Two files hold all the code:

* `examples/tutorial_lai_process.py` is the process module (steps 2 to 5). It is the file you would move into `src/agrijax/processes/<slot>/`.
* `examples/new_process_tutorial.py` is the scaffolding around it: the NumPy prototype, a synthetic season, the conformance case and the lint demonstration (steps 1 and 6 to 8).

The snippets below are copied from these two files. The import lines of each file are at its top; the snippets show the Agri-JAX imports where a name first appears (`np`, `jnp`, `jax` and `Any` are the usual ones). A `...` marks code that a snippet leaves out.

## Which steps do you need?

| Steps | What | Local try-out | Merge into the repository |
|---|---|---|---|
| 1 to 3 | NumPy prototype, pure JAX function, `@process` with `reads` / `writes` | needed | needed |
| 4, 5 | labelled coefficients, registry key | optional: plain floats and an unkeyed `@process` work | required (lint rule AJ007, registry metadata) |
| 6 | run it inside a model on a synthetic season | needed | needed |
| 7, 8 | conformance check, strict lint | not needed | required: CI runs the conformance tests and the strict lint |

You can stop after step 6 to try a formula on your own machine. Steps 7 and 8 are the merge gate, and steps 4 and 5 cost ten minutes and make step 7 pass.

Which modules join the library is the maintainers' decision: the README says new modules are added only after the existing ones pass automated comparisons against an independent reference. A plugin package (step 7) needs no merge.

## Step 1: the formula in plain NumPy

Start where you already are: floats in, float out, an ordinary `if`, the numbers written in place, a loop over days.

```python
def lai_step_numpy(lai: float, tt: float, dtt: float, swfac: float) -> float:
    if tt < 1200.0:  # expansion: logistic in thermal time, scaled by water stress
        dlai = 0.008 * dtt * lai * (1.0 - lai / 5.0) * swfac
    else:  # senescence
        dlai = -0.03 * lai
    return float(np.maximum(lai + dlai, 0.0))


def season_numpy(tmean: np.ndarray, swfac: np.ndarray, lai0: float = 0.05, tbase: float = 8.0) -> np.ndarray:
    lai, tt, out = lai0, 0.0, []
    for tm, sw in zip(tmean, swfac):
        dtt = max(tm - tbase, 0.0)
        tt += dtt
        lai = lai_step_numpy(lai, tt, dtt, sw)
        out.append(lai)
    return np.asarray(out)
```

`lai` is the leaf area index (m2 m-2), `tt` the thermal time since emergence (degC d), `dtt` today's thermal time and `swfac` a water stress factor from 1 (none) to 0. `season_numpy` adds the degree-day sum (base temperature 8 degC) and runs a whole season.

Keep both functions. They are the independent check of everything below: the script asserts that the JAX process agrees with them to 1e-12 (in the run shown in step 6 it agrees to the last bit).

## Step 2: the same formula as a pure JAX function

| NumPy prototype | JAX process |
|---|---|
| `np.maximum(a, 0.0)` | `jnp.maximum(a, 0.0)`; a NumPy function applied to a traced value fails under `jit` |
| `if tt < 1200.0: a else: b` | `jnp.where(tt < tt_senesce, a, b)`; there is no Python `if` on a value that comes from the state, the parameters or the forcing |
| `lai / 5.0` | divide by a clamped argument, `jnp.maximum(lai_max, _TINY)` (`_TINY` is a named guard, step 4), so the result is finite even when a branch is not selected |
| `x[i] = ...`, `lai += dlai` | return a new state with `state.replace(lai=...)`; nothing is mutated |
| a loop over days, samples or soil layers | no loop: the runtime owns time (`lax.scan`) and batches (`vmap`), and layers are an array axis |
| numbers inside the formula | named coefficients (step 4) |

`jnp.where` computes both branches and selects afterwards. A branch that is `nan` or `inf` where it is not selected still poisons the gradient, so guard every division, log and square root before the `where`. The body of the process:

```python
c = params.lai
ceiling = jnp.maximum(c.lai_max, _TINY)  # both branches below stay finite
grow = c.r_expand * state.dtt * state.lai * (1.0 - state.lai / ceiling) * forcing_t.swfac
lose = -c.k_senesce * state.lai
lai = jnp.maximum(state.lai + jnp.where(state.tt < c.tt_senesce, grow, lose), 0.0)
return state.replace(lai=lai, dlai=lai - state.lai)
```

## Step 3: wrap it as a `@process`

A process has the signature `(state, params, forcing_t) -> state`: everything it reads is an argument and everything it changes is in the return value. The state and the forcing are small classes whose fields carry a unit and named dimensions (`n_crop` is the crop axis, `T` the time axis of a forcing). The imports of the process module:

```python
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.coefficients import Coefficients, Provenance, coef, numerical_guard
from agrijax.core.process import process
from agrijax.core.state import Forcing, Params, State, field
```

The state, the forcing and the process:

```python
class LaiState(State):
    tt: Array = field(unit="degC d", description="thermal time since emergence", dims=("n_crop",))
    dtt: Array = field(unit="degC d", description="thermal time of today", dims=("n_crop",))
    lai: Array = field(unit="m2 m-2", description="green leaf area index", dims=("n_crop",))
    dlai: Array = field(unit="m2 m-2 d-1", description="change of the LAI today", dims=("n_crop",))


class LaiForcing(Forcing):
    tmean: Array = field(unit="degC", description="daily mean air temperature", dims="T")
    swfac: Array = field(unit="-", description="water stress factor (1 none, 0 full stress)", dims="T")


@process(reads=("tt", "dtt", "lai"), writes=("lai", "dlai"))   # step 5 adds the registry arguments
def lai_logistic(state: LaiState, params: DemoParams, forcing_t: LaiForcing) -> LaiState:
    """Logistic LAI expansion in thermal time, scaled by water stress, then senescence.

    Source: tutorial demo formula (docs/tutorial_new_process.md), no reference model.
    """
    ...  # the body of step 2
```

`reads` and `writes` are dotted paths into the state. With `AGRI_JAX_CHECK=1` (the script sets it) every call verifies that only the declared `writes` changed, and step 7 verifies the `reads` by perturbation. Units follow the grammar of `agrijax.core.units.parse_unit` (`"-"` is dimensionless). The docstring needs a `Source:` line.

The example has a second, unregistered helper process, `degree_days`, which writes `tt` and `dtt`. It only supplies the inputs of the first one.

## Step 4: declare the coefficients

Every number of the equations that is not a state, a forcing or a parameter read from an input file is a coefficient, declared once with its value, unit, meaning and provenance:

```python
class LaiCoefficients(Coefficients):
    r_expand: float = coef(
        0.008,
        "degC-1 d-1",
        "relative LAI expansion rate per unit of thermal time",
        Provenance("none", paper=_PAPER),
        bounds=(0.0, 0.05),
    )
    ...  # lai_max, tt_senesce and k_senesce are declared the same way


class DemoParams(Params):
    thermal: ThermalTimeCoefficients = field(description="degree-day coefficients")
    lai: LaiCoefficients = field(description="LAI expansion coefficients")
```

`ThermalTimeCoefficients` holds the base temperature of the helper process, and the coefficient classes go into `params`, here `DemoParams`.

`Provenance` says where the value comes from:

* With no reference model (`ref_version` `none`) it cites a published equation, `Provenance("none", paper="Author et al. (2001)", equation="7")`.
* For a value that a DSSAT-CSM routine hard-codes it cites the file and line, `Provenance.at("dssat-4.8.6.0", "Plant/CERES-Maize/MZ_GROSUB.for:1246", routine="MZ_GROSUB", ...)`.
* A provenance must cite a source file or a paper. The tutorial has neither, so `_PAPER` is a label saying that the value is illustrative and not from a publication. A real coefficient cites its paper there.

`bounds` are optional physical limits that the default must respect.

Coefficients are calibratable by default: `LaiCoefficients().as_arrays()` turns them into array leaves that `jax.grad` and `vmap` can use. A bare number in a process is lint finding AJ007 (only `0`, `1`, `-1`, small integer indices and exponents pass).

A floor that keeps a division finite is a named guard, `_TINY = numerical_guard("lai_logistic.tiny", 1e-12, "why")`, and a unit conversion goes through the adapters of `agrijax.core.units`.

## Step 5: register it under `slot/impl@ref_version:variant`

```python
KEY = "crop/lai_logistic@none:demo"


@process(
    reads=("tt", "dtt", "lai"),
    writes=("lai", "dlai"),
    key=KEY,
    provenance="equations_only",  # the nearest class: no reference code was used (see the tutorial)
    sources=[("logistic expansion in thermal time, exponential senescence", "tutorial demo formula")],
    grid="point",
    deviates=(),
)
```

* `crop` is the slot, the role the process fills in an assembly (`crop`, `soil_water`, `pet`, ...). `lai_logistic` names the implementation.
* `none` is the `ref_version`. Normally it names the reference model the process follows (`dssat-4.8.6.0`, `rzwqm2-4.6`). Every variant other than `faithful` of such a key is registered after its `faithful` sibling (`crop/...@dssat-4.8.6.0:faithful`) and lists how it deviates.
* `none` means there is no reference to be faithful to, which waives that rule. The price is that the process cannot claim validation against a reference model, and the variant name `demo` says so. A formula that modifies a DSSAT-CSM process uses that `ref_version` and registers the faithful process first.
* `provenance` records how the code was derived: `translated_bsd3` (translated from BSD-3 reference source), `equations_only` (written from published equations alone) or `reference_only_conventions` (written from published equations, a reference source without a licence read for conventions only). The registry accepts one of the three.
* The tutorial formula is not published, so `equations_only` is only the nearest class here: no reference code was used, and there is no paper to cite. A formula from a paper names it in `sources` and in the `Provenance` of its coefficients.
* `grid` is `point` for one value per crop. `sources` lists the equation or paper per block, and `deviates` lists deviations from the reference (`()` here).

Two different processes cannot share a key or a function name (`DuplicateProcessError`). `register=False` keeps a throwaway process out of the registry, as for `degree_days`.

## Step 6: run it inside a small model

A model is a state class plus an ordered list of processes, run by the runtime over a forcing with the time axis first. The script enables 64-bit floats (`jax.config.update("jax_enable_x64", True)`), so that the comparison with NumPy is to rounding error:

```python
from agrijax.core import Model, run
from agrijax.testing.conformance import synthetic

MODEL = Model(LaiState, [degree_days, lai_logistic], outputs=("lai", "tt"), name="lai_demo")

dtype = jnp.float64
n_days = synthetic.N_SEASON
forcing, state0, params = season_forcing(n_days, dtype), initial_state(1, dtype), demo_params(dtype)
out = jax.jit(lambda p, f, s: run(MODEL, p, f, s))(params, forcing, state0)
```

`initial_state` is the canopy at emergence (no thermal time yet and a seedling LAI) and `demo_params` gives the default coefficients as arrays (`as_arrays`, step 4). Both are a few lines in the script. The forcing is the synthetic weather of the conformance kit, `synthetic.ceres_weather` (maize-season temperatures at 29.6 degrees latitude), plus a dry spell every 37 days:

```python
def season_forcing(n_days: int, dtype: Any) -> LaiForcing:
    w = synthetic.ceres_weather(7, dtype, n_days)
    swfac = np.where(np.arange(n_days) % 37 > 25, 0.5, 1.0)
    return LaiForcing(tmean=(w.tmax + w.tmin) / 2.0, swfac=jnp.asarray(swfac, dtype))
```

The script prints:

```text
  dataflow (writer, reader, path): [('degree_days', 'lai_logistic', 'tt'), ('degree_days', 'lai_logistic', 'dtt')]
  peak LAI 4.856 on day 71; max |process - NumPy| = 0.0e+00
  r_expand 0.008 -> 0.006: LAI on day 30 0.78 -> 0.42
  d peak LAI / d r_expand   = +153.021897 (finite difference +153.021920)
  d peak LAI / d lai_max    = +0.939078 (finite difference +0.939078)
  d peak LAI / d tt_senesce  = +0.0 (a hard `where`: no gradient)
```

The declared `reads` and `writes` give the dataflow between processes. A model is a list, so another leaf-area formula replaces this one with `MODEL.replace("lai_logistic", other)`, and a coefficient changes by building `LaiCoefficients(r_expand=0.006)`.

`jax.grad` differentiates the whole season with respect to the coefficients and agrees with finite differences. The threshold `tt_senesce` has zero gradient because `jnp.where` switches hard: use a smooth switch if you want to calibrate a threshold.

**You can stop here** if you wanted to try the formula. Steps 7 and 8 are the merge gate.

## Step 7: a conformance case and the check (merge only)

The conformance kit runs generic checks on any process of a slot without data. You supply a `ConformanceCase`: the registry key, a `make(rng, dtype, variant)` that builds synthetic `(state0, params, forcing)` from a NumPy generator (never `jax.random`), and what to expect.

```python
from agrijax.testing.conformance import Balance, ConformanceCase, GradSpec, check_reads, run_checks


def make(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    tt = {"nominal": (200.0, 600.0), "senescence": (1300.0, 1800.0), "bare": (100.0, 300.0)}[variant]
    lai = {"nominal": (0.3, 2.0), "senescence": (2.0, 4.0), "bare": (0.0, 0.0)}[variant]
    ...  # a LaiState from these ranges, the forcing from the synthetic weather
    return state, demo_params(dtype), forcing


LAI_BOOKKEEPING = Balance(
    "leaf area",
    "m2 m-2",
    storage=lambda s, p: jnp.sum(s.lai),
    inflow=lambda before, after, p, f: jnp.sum(jnp.maximum(after.dlai, 0.0)),
    outflow=lambda before, after, p, f: jnp.sum(jnp.maximum(-after.dlai, 0.0)),
)
CASE = ConformanceCase(
    key=KEY,
    make=make,
    variants=("nominal", "senescence"),  # expansion branch, senescence branch
    n_days=N_DAYS,  # 3
    coefficient_sets=("lai",),  # the coefficient set this process uses
    forcing_fields=("swfac",),  # the only forcing field it reads
    balances=(LAI_BOOKKEEPING,),  # the day's change must be what the state records
    grad=GradSpec(edge_variants=("bare",)),  # also finite gradients at LAI = 0
    origin="examples/new_process_tutorial.py",
)
result = run_checks(CASE)  # {check name: None if it passed, else the message}
```

Give a variant to every branch of your `jnp.where`, so each is exercised. A check that does not apply needs a written reason (`no_balance="..."`, `no_grad="..."`, `no_slot_contract="..."`), and `exempt_checks={name: reason}` records a known gap: the check must still fail.

The slot contract lists the state paths and ports that the processes of a slot may touch. A port is a state field that holds a record exchanged with another slot (the water a crop receives, say); this process has none.

The paths of `reads` and `writes` (`tt`, `lai`) are relative to the small state of the example. The kit binds that state to its place in a full assembly, `crops.maize` for the `crop` slot, and checks that the process stays inside it and the slot's ports. For a slot without a contract, pass `slot_contract=None` with its reason.

The 14 checks, in order:

| Check | What it verifies |
|---|---|
| `registry` | key, provenance, sources, a `Source:` line, the case's origin |
| `lint` | the lint rules of step 8 on the process module and the modules of its package it imports |
| `coefficients`, `shapes_dims`, `units` | coefficient labels and bounds, declared dims against shapes, every leaf has a parsable unit |
| `slot_contract` | reads and writes stay inside the own subtree and the slot's ports |
| `writes`, `reads` | only declared `writes` change; every undeclared state leaf set to `nan` or random leaves the outputs bit for bit unchanged |
| `balance` | each declared balance closes every day, in float64 and float32 |
| `transforms` | eager equals `jit`; `vmap(jit)` equals per-sample `jit`; samples do not influence each other |
| `precision` | float32 agrees with float64, no implicit upcast, all finite |
| `grad_finite`, `grad_fd` | `jax.grad` is finite, and equals central differences at two step sizes |
| `binding` | replaying a port from data equals coupling it (no ports here, so nothing to bind) |

All 14 print `ok` for the example. A failure names the check and what differs. The script also builds a copy of the process that forgets to declare that it reads `dtt`, and runs only the `reads` check on it:

```python
import dataclasses

sloppy = dataclasses.replace(lai_logistic, reads=("tt", "lai"))
failed = run_checks(dataclasses.replace(CASE, process=sloppy), checks=[check_reads])["reads"]
```

```text
[crop/lai_logistic@none:demo] reads: variant 'nominal': outputs depend on reads outside ['tt', 'lai']: dtt (nan) changes ['lai', 'dlai']
```

The case lives in a separate file from the process because the lint treats every function of the process module as numerical kernel code, where the literals of synthetic test inputs would be findings.

In the repository the cases are in `src/agrijax/testing/conformance/_builtin/<slot>.py`, listed in `_builtin/__init__.py`, and run in the unit tier. A plugin package registers its cases under the entry point group `agrijax.conformance` and runs `python -m agrijax.testing.conformance --package my-plugin --no-builtin`.

The kit shows that a process can join an assembly without breaking it. Whether its numbers are right is answered by comparison with a reference model, in the integration tier.

## Step 8: run the strict lint and what to expect (merge only)

```bash
uv run python -m agrijax.core.lint examples/tutorial_lai_process.py --strict
```

```text
agrijax lint: 0 error(s), 0 warning(s) (0 AJ007) in 2 function(s) (2 @process, 0 kernel)
```

`--strict` turns warnings into failures, as CI and pre-commit do for `src`. The script also lints the module the way the conformance check does (every function as a kernel), then a process that breaks the rules; these are three of its five findings:

```text
lai_bad.py:3:0: AJ005 [warning] missing docstring or no 'Source:' line: lai_bad docstring has no 'Source:' line
lai_bad.py:5:4: AJ001 [error] Python branch on a value derived from the process arguments: if on state; use jnp.where / jnp.select
lai_bad.py:5:19: AJ007 [warning] bare numeric literal in process or kernel code: 4.5; declare it with core.coefficients.coef (or as a named constant)
```

| Rule | Reports | Fix |
|---|---|---|
| AJ001 (error) | `if` or `while` on a value derived from the arguments | `jnp.where` / `jnp.select` |
| AJ002, AJ006 (errors) | subscript assignment in a loop; a loop over a shape-derived range or an array | vectorise; a true recurrence over depth uses `depth_scan` |
| AJ003 | unguarded `log`, `sqrt` or `/` inside a `where` branch | clamp with `jnp.maximum` / `jnp.clip` first |
| AJ004, AJ005 | a return not built with `replace` / `tree_at`; no docstring or `Source:` line | `state.replace(...)`; add the line |
| AJ007 | bare numeric literal | a `coef` (step 4), a named guard or a unit adapter |
| AJ008 | a slot importing another slot's package | exchange data through the port records of `agrijax.iface` |

## To merge, and where to go next

A merge adds the process under `src/agrijax/processes/<slot>/`, its case under `_builtin/` and whatever else CI runs: `uv run ruff check . && uv run ruff format --check . && uv run pyright`, the lint over `src` with `--strict`, and the unit tier. These need the development tools, so run `uv sync --all-extras` first.

[CONTRIBUTING.md](../CONTRIBUTING.md) has the full list of rules, the checks and the pull-request conventions. [Swapping a process](swapping_a_process.md) shows how a registered process replaces another in the DSSAT-CSM day and what the framework checks when it does.
