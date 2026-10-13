# Debugging models and processes

A process is a pure function of `(state, params, forcing)`, so most debugging needs no special tool: call it, print it, step through it. The difficulty starts when the runtime takes over: `run` traces the day step once and compiles a `lax.scan` over the days, `run_batch` adds `vmap` and `jit`, and inside them a `print` shows a tracer (a placeholder for a value not yet known) instead of a number. The techniques below go from the simplest to the most specific. Every snippet is in [`examples/debugging_snippets.py`](../examples/debugging_snippets.py), which runs on a CPU without data (`uv run python examples/debugging_snippets.py`, after the `uv sync` of the README); docstrings and some `print` calls are left out here. The output lines come from that script (JAX 0.10.2; the wording of tracers and errors changes a little between JAX versions).

## The toy model

The snippets use a bucket that holds water and turns a fraction `k` of it into biomass each day, while it is not empty. `register=False` keeps throwaway processes out of the process registry.

```python
import os

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from jaxtyping import Array

from agrijax.core import Day, Forcing, Lag, Model, Params, Phase, State, field, process, run
from agrijax.core.coefficients import numerical_guard
from agrijax.core.day import DayLagError
from agrijax.core.lint import lint_source
from agrijax.core.process import ProcessWriteError
from agrijax.core.runtime import run_and_grad
from agrijax.core.state import get_path, set_path
from agrijax.port.compare import Tolerance, compare_series

jax.config.update("jax_enable_x64", True)  # float64, as in the output lines below

class Bucket(State):
    water: Array = field(unit="cm", description="bucket water")
    biomass: Array = field(unit="kg ha-1", description="above-ground biomass")

class BucketParams(Params):
    k: Array = field(unit="d-1", description="fraction of the water used per day")
    rue: Array = field(unit="kg ha-1 per MJ m-2", description="radiation use efficiency")

class Weather(Forcing):
    rain: Array = field(unit="cm", dims="T")
    srad: Array = field(unit="MJ m-2 d-1", dims="T")

@process(reads=("water",), writes=("water",), register=False)
def infiltrate(state, params, forcing_t):
    """..."""
    return state.set("water", state.water + forcing_t.rain)

@process(reads=("water", "biomass"), writes=("water", "biomass"), register=False)
def grow(state, params, forcing_t):
    """..."""
    uptake = params.k * state.water
    gain = jnp.where(state.water > 0.0, params.rue * forcing_t.srad * uptake, 0.0)
    return state.replace(water=state.water - uptake, biomass=state.biomass + gain)

def toy_inputs(n_days=6, water0=5.0, rain_first=1.0):
    t = np.arange(n_days)
    rain = np.where(t % 3 == 0, 1.0, 0.0)
    rain[0] = rain_first
    forcing = Weather(rain=jnp.asarray(rain), srad=jnp.asarray(15.0 + 5.0 * np.sin(t)))
    params = BucketParams(k=jnp.asarray(0.1), rue=jnp.asarray(2.0))
    state0 = Bucket(water=jnp.asarray(water0), biomass=jnp.asarray(0.0))
    return params, forcing, state0

MODEL = Model(Bucket, [infiltrate, grow], outputs=("water", "biomass"), name="bucket")
```

## 1. Call one process directly

`@process` returns a `Process`, callable with the signature of the function. Outside `run` nothing is traced, so `print`, `breakpoint()`, `pdb` and your editor's debugger all work. Give it one day of forcing (scalar leaves, no time axis).

```python
params, _, state = toy_inputs()
day = Weather(rain=jnp.asarray(1.0), srad=jnp.asarray(15.0))  # one day: scalar leaves, no time axis

after_rain = infiltrate(state, params, day)  # ordinary Python: print, breakpoint() and pdb work here

s = state  # the whole day, one process at a time
for p in MODEL.processes:
    s = p(s, params, day)
    print(f"after {p.name:<10} water={float(s.water):.4f} biomass={float(s.biomass):.4f}")

_, outputs = MODEL.compile()(state, params, day)  # the same day as the compiled step
```

```text
after infiltrate water=6.0000 biomass=0.0000
after grow       water=5.4000 biomass=18.0000
```

To debug a day in the middle of a season, take the state from the days before it (`run(..., return_final=True)` on the first `k` days of forcing, see section 6). `infiltrate.fn` is the undecorated function and skips the writes check of section 5.

## 2. `jax.disable_jit()`

Inside `run` the day step is traced once by `lax.scan`, so a `print` runs once, at trace time, and shows a tracer. Under `jax.disable_jit()` the scan becomes a Python loop over the days and every operation runs eagerly:

```python
@process(reads=("water",), writes=(), register=False)
def print_water(state, params, forcing_t):
    """..."""
    print("print_water sees", state.water)
    return state

params, forcing, state0 = toy_inputs(n_days=3)  # three days keep the printout short
model = Model(Bucket, [infiltrate, print_water, grow], outputs=("water", "biomass"))
run(model, params, forcing, state0)  # the scan body is traced once, the print shows a tracer
with jax.disable_jit():
    run(model, params, forcing, state0)  # one print per day, with values
```

```text
print_water sees JitTracer(float64[])
print_water sees 6.0
print_water sees 5.4
print_water sees 4.86
```

Now `print`, `breakpoint()` and exceptions behave as in ordinary Python, and a traceback points at your line. It is slow, so use a short forcing (`jax.tree_util.tree_map(lambda x: x[:10], forcing)`) and one parameter set with `run`: `disable_jit` does not switch `vmap` off, so `run_batch` still shows batch tracers. To watch a state field over the whole season without any print, add its path to `outputs=` and read the `[T]` array that `run` returns.

## 3. `jax.debug.print` inside compiled code

When the run has to stay compiled (a long season, `run_batch`, gradients), `jax.debug.print` prints values at run time from inside `jit`, `scan` and `vmap`. A tap process with `writes=()` passes the writes check, can go anywhere in the list and be removed again.

```python
@process(reads=("water", "biomass"), writes=(), register=False)
def tap(state, params, forcing_t):
    """..."""
    jax.debug.print("day: rain={r} water={w:.4f} biomass={b:.4f}", r=forcing_t.rain, w=state.water, b=state.biomass)
    return state

params, forcing, state0 = toy_inputs(n_days=3)
model = Model(Bucket, [infiltrate, grow, tap], outputs=("water", "biomass"))
run(model, params, forcing, state0)  # compiled as usual, and it prints every day
```

```text
day: rain=1.0 water=5.4000 biomass=18.0000
day: rain=0.0 water=4.8600 biomass=38.7439
day: rain=0.0 water=4.3740 biomass=57.7431
```

Under `vmap` the line appears once per sample and day, so keep the batch small. `jax.debug.breakpoint()` opens a debugger prompt at the same place.

## 4. NaNs

**The first NaN: `jax_debug_nans`.** A NaN in the forward pass does not raise. It travels through the state and shows up days later in the outputs. `sqrt_stress` adds the square root of `water - srad`, which is negative on the first day:

```python
@process(reads=("water", "biomass"), writes=("biomass",), register=False)
def sqrt_stress(state, params, forcing_t):
    """..."""
    return state.set("biomass", state.biomass + jnp.sqrt(state.water - forcing_t.srad))

params, forcing, state0 = toy_inputs()  # six days
model = Model(Bucket, [infiltrate, sqrt_stress], outputs=("water", "biomass"))
run(model, params, forcing, state0)["biomass"]  # [nan nan nan nan nan nan]

jax.config.update("jax_debug_nans", True)  # raise at the first NaN (jax_debug_infs: at the first inf)
with jax.disable_jit():  # a Python loop: every operation is checked on its own
    run(model, params, forcing, state0)  # raises FloatingPointError
jax.config.update("jax_debug_nans", False)  # switch it off again
```

```text
FloatingPointError: invalid value (nan) encountered in sqrt
  traceback: sqrt_stress, line 155: return state.set("biomass", state.biomass + jnp.sqrt(state.water - forcing_t.srad))
```

(`line 155` is the place of the process in the example file.) Without `disable_jit` the flag still stops the run, but it names the compiled operation (`invalid value (nan) encountered in scan`) and none of your lines, also when the call is wrapped in `jax.jit`. Use both together. `with jax.debug_nans(True):` turns the flag on for one block only; the gradient case below shows that the two forms do not always behave alike.

**A NaN gradient from `jnp.where`.** The forward value can be finite and the gradient NaN. `jnp.where(c, a, b)` selects a value, but the backward pass still differentiates the branch that was not selected and multiplies its derivative by the gradient flowing back into it, which is zero. When that derivative is infinite or NaN, `0 * inf` is NaN and poisons the gradient of every parameter the value depends on.

```python
_TINY = numerical_guard("debugging_snippets.tiny", 1e-12, "floor of a logarithm's argument ...")

def unsafe(w):
    return jnp.where(w > 0.0, jnp.log(w), 0.0)

def safe(w):  # clamp the argument, select afterwards
    return jnp.where(w > 0.0, jnp.log(jnp.maximum(w, _TINY)), 0.0)

@process(reads=("water", "biomass"), writes=("biomass",), register=False)
def log_growth_unsafe(state, params, forcing_t):
    """..."""
    return state.set("biomass", state.biomass + params.rue * unsafe(state.water))

@process(reads=("water", "biomass"), writes=("biomass",), register=False)
def log_growth_safe(state, params, forcing_t):
    """..."""
    return state.set("biomass", state.biomass + params.rue * safe(state.water))
```

```text
w=+0: unsafe value=+0.0 grad=+nan   safe value=+0.0 grad=+0.0
w=-1: unsafe value=+0.0 grad=-0.0   safe value=+0.0 grad=+0.0
sqrt, w=-1: grad nan (unsafe); 0.0 (safe)
```

The derivative of the unused branch is what matters: `log(0)` has derivative `1/0`, the square root of a negative number a NaN one. The log of a negative number has a finite derivative (`1/w`), so `w=-1` passes for `log` and not for `sqrt`; do not rely on it. The project's pattern is rule 2 of the [three process rules](../CONTRIBUTING.md#the-three-rules-for-process-functions) (see also the [README](../README.md#design)): compute every `log`, power, square root and division on a clamped argument (`jnp.maximum(w, _TINY)`, or `agrijax.core.grad.safe_div` for quotients), then select. Name the guard with `numerical_guard`, not a bare `1e-12`. Lint rule AJ003 flags the unsafe form (section 5).

In a model the symptom is a NaN in the gradient of one parameter, so check every leaf. The bucket starts empty, so the log sees 0 on the first day:

```python
params, forcing, state0 = toy_inputs(water0=0.0, rain_first=0.0)  # empty on day 1: no water, no rain
loss = lambda out: out["biomass"][-1]  # the final biomass
for proc in (log_growth_unsafe, log_growth_safe):
    model = Model(Bucket, [infiltrate, grow, proc], outputs=("water", "biomass"))
    value, grad = run_and_grad(model, loss, params, forcing, state0)
    finite = jax.tree_util.tree_map(lambda g: bool(jnp.isfinite(g).all()), grad)
    print(f"{proc.name:<17} loss={float(value):.4f}  gradient finite: {finite}")
```

```text
log_growth_unsafe loss=5.5489  gradient finite: BucketParams(k=False, rue=True)
log_growth_safe   loss=5.5489  gradient finite: BucketParams(k=True, rue=True)
```

`jax_debug_nans` is of little use on this gradient. Through the scan it only says `in scan`. Under `disable_jit` the result depends on how the flag is set (JAX 0.10.2). The config update raises nothing and returns the NaN gradient. The scoped form raises `invalid value (nan) encountered in div`, but its traceback shows no process, only the caller and `run_and_grad` (`frames:` lists the traceback frames that are in the script or in agrijax), so it does not name the process:

```python
model = Model(Bucket, [infiltrate, grow, log_growth_unsafe], outputs=("water", "biomass"))
jax.config.update("jax_debug_nans", True)
with jax.disable_jit():
    run_and_grad(model, loss, params, forcing, state0)  # no error, the gradient has a NaN
jax.config.update("jax_debug_nans", False)
with jax.debug_nans(True), jax.disable_jit():
    run_and_grad(model, loss, params, forcing, state0)  # FloatingPointError in div
```

```text
config flag: no error raised; gradient finite: BucketParams(k=False, rue=True)
scoped flag: FloatingPointError - invalid value (nan) encountered in div; frames: section_4_nans > run_and_grad
```

Narrow it down instead. Differentiate one process for one day, called directly (section 1), with respect to the suspect input, starting from the state of the day where it happens (`run(..., return_final=True)`):

```python
state = Bucket(water=jnp.asarray(0.0), biomass=jnp.asarray(0.0))  # the state of the day to reproduce
day = Weather(rain=jnp.asarray(0.0), srad=jnp.asarray(15.0))
for proc in (log_growth_unsafe, log_growth_safe):

    def biomass_of_water(w, proc=proc):
        return proc(state.replace(water=w), params, day).biomass

    try:
        with jax.debug_nans(True):  # the flag for this block only
            g = jax.grad(biomass_of_water)(jnp.asarray(0.0))
        print(f"{proc.name:<17} d biomass / d water = {float(g)}")
    except FloatingPointError as e:
        print(f"{proc.name:<17} {type(e).__name__} - {e}")
```

```text
log_growth_unsafe FloatingPointError - invalid value (nan) encountered in div
log_growth_safe   d biomass / d water = 0.0
```

The `div` is the derivative of `log` (the incoming gradient divided by `w`), which names the culprit.

## 5. What the checks catch

**`AGRI_JAX_CHECK=1`** makes every process call verify that the returned state differs from the input only in the declared `writes`. `sloppy` declares `writes=("water",)` but also books biomass:

```python
@process(reads=("water",), writes=("water",), register=False)
def sloppy(state, params, forcing_t):
    """..."""
    return state.replace(water=state.water - 1.0, biomass=state.biomass + 1.0)

params, forcing, state0 = toy_inputs(n_days=3)
model = Model(Bucket, [infiltrate, sloppy], outputs=("water", "biomass"))
run(model, params, forcing, state0)
os.environ["AGRI_JAX_CHECK"] = "1"  # the same as running the script with AGRI_JAX_CHECK=1
run(model, params, forcing, state0)  # raises ProcessWriteError
del os.environ["AGRI_JAX_CHECK"]  # switch it off again
```

```text
check off: runs, biomass [1. 2. 3.] (nobody declared it)
check on: ProcessWriteError - process 'sloppy' changed undeclared state fields ['biomass']; declared writes = ['water']
```

Inside `run` the check happens while the day step is traced, before any day is computed. Under `jit` and `scan` leaves are compared by identity, so a process that rebuilds a field it did not declare is flagged even when the value is the same. The flag is read at trace time: set it before the first call (`run_batch` includes it in its cache key; a function you `jax.jit` yourself keeps the choice it was traced with). The declared `reads` are not verified at run time.

**`Day.check`** reads the declarations of an assembled day without running it. A `Day` orders entries (`slot.process` names) in named phases and holds a table of allowed lags: a `Lag` lets an entry read the previous day's value of a path that a later entry writes. `Day.check` tests the entry order, the lags, one owning module per path and the declared extra writes; in a phase declared `discipline="start_of_day"` it also rejects a read of what an earlier entry of the same phase wrote (the day below has no such phase). Here `soil.day` reads `iface.uptake` (a path of the interface record the modules share), which `crop.publish` writes later in the same day, so it would see yesterday's value. `Day.compile` runs the check and refuses:

```python
def _soil_day(s, p, f):
    return set_path(s, "soil.w", get_path(s, "soil.w") + f.rain - get_path(s, "iface.uptake"))

def _grow_lai(s, p, f):
    return set_path(s, "crop.lai", get_path(s, "crop.lai") + 0.01 * get_path(s, "soil.w"))

def _publish(s, p, f):
    return set_path(s, "iface.uptake", 0.1 * get_path(s, "crop.lai"))

DAY_PROCS = {
    "soil.day": process(_soil_day, reads=("soil.w", "iface.uptake"), writes=("soil.w",), register=False),
    "crop.grow": process(_grow_lai, reads=("soil.w", "crop.lai"), writes=("crop.lai",), register=False),
    "crop.publish": process(_publish, reads=("crop.lai",), writes=("iface.uptake",), register=False),
}
ENTRIES = (Phase("day", ("soil.day", "crop.grow", "crop.publish")),)

day = Day(ref="toy", phases=ENTRIES)
day.compile(DAY_PROCS)  # raises DayLagError
unchecked = day.compile(DAY_PROCS, check=False)
day.lagged_reads(unchecked)  # [('soil.day', 'iface.uptake')]
allowed = Day(ref="toy", phases=ENTRIES, lags=(Lag("soil.day", "iface.uptake", evidence="published for tomorrow"),))
report = allowed.check(allowed.compile(DAY_PROCS, check=False))  # report.used, report.unused_pairs
MODEL.stale_reads()  # [('infiltrate', 'water'), ('grow', 'biomass')]
```

```text
DayLagError - undeclared lags (reader runs before another entry that writes the path, and the day does not allow the lag): soil.day <- iface.uptake (written by ['crop.publish'])
stale reads of the bucket model: [('infiltrate', 'water'), ('grow', 'biomass')]
```

A lag you want is declared with `Lag` and its evidence. `Model.stale_reads()` lists every read that sees a previous day's value: lags, which need a `Lag`, and carried state, which does not (`infiltrate` reads the `water` of the day before, and the bucket model has no lag). `Day.check` trusts the declared `reads` and `writes`, so run `AGRI_JAX_CHECK=1` as well.

**The strict lint** reads the source and enforces the three rules and the coefficient labelling: `python -m agrijax.core.lint src --strict` for a tree, or `lint_source` for a string. `LEAKY` is a process with an `if`, a bare `0.37`, an unguarded `log`, a loop over layers and no `Source:` line; the line numbers below count from its `import` line:

```python
LEAKY = '''\
import jax.numpy as jnp
from agrijax.core import process


@process(reads=("water",), writes=("biomass",))
def leaky(state, params, forcing_t):
    """Turn water into biomass."""
    w = state.water
    if w > 0.0:
        w = w * 0.37
    gain = jnp.where(w > 0.0, jnp.log(w), 0.0)
    total = 0.0
    for i in range(w.shape[0]):
        total = total + w[i]
    return state.replace(biomass=state.biomass + gain + total)
'''

for f in lint_source(LEAKY, "leaky.py"):
    print(f.format())
```

```text
leaky.py:6:0: AJ005 [warning] missing docstring or no 'Source:' line: leaky docstring has no 'Source:' line
leaky.py:9:4: AJ001 [error] Python branch on a value derived from the process arguments: if on w; ...
leaky.py:10:16: AJ007 [warning] bare numeric literal in process or kernel code: 0.37; ...
leaky.py:11:30: AJ003 [warning] unguarded log/sqrt/division inside a where/select branch: log(w) ...
leaky.py:13:13: AJ006 [error] Python loop over a shape-derived range or an array (unrolled layer loop): ...
```

| Check | When it runs | Catches | Does not catch |
|---|---|---|---|
| `AGRI_JAX_CHECK=1` | each process call (trace time under `run`) | a write outside the declared `writes` | wrong `reads`; wrong numbers |
| `Day.check` | when the day is compiled, no run | undeclared lags, owners, extra writes, same-phase reads in `start_of_day` phases | what a process does beyond its declarations |
| lint (`--strict`) | on the source text | Python branches on state, loops over layers, unsafe `where` branches, bare literals, missing `Source:` | what the code computes |

None of them checks the physics. For that, compare with a reference.

## 6. Compare with a reference, day by day

`agrijax.port.run_dscsm` (DSSAT-CSM) and `agrijax.port.run_rzwqm` (RZWQM2) run a reference binary that you provide on a staged copy of a case and keep its output files. `agrijax.io.dssat.read_plantgro`, `read_soilwat` and `read_et` return a pandas DataFrame with the reference's column names and a `DATE` column (one block per run and treatment, marked by `RUN` and `TRNO`: select one); `agrijax.io.rzwqm.read_ana` returns an xarray Dataset. Data and binaries live outside the repository (`--data-dir`, `AGRI_JAX_DATA`). `agrijax.port.compare.compare_series` inner-joins two xarray objects on `time` and reports bias, RMSE, maximum absolute difference, R² and NSE per variable, with optional `Tolerance` limits and a mapping `{sim_name: ref_name}` for differently named variables; a DataFrame goes through `df.set_index("DATE").rename_axis("time").to_xarray()` first. The snippet builds a stand-in reference in that shape (the biomass series with a drift from day 4 on, named `CWAD` as in DSSAT's PlantGro file) so that it runs without data.

```python
def to_dataset(outputs, start="2020-05-01"):
    """`run` outputs (one array per path, time first) as a Dataset with a `time` axis."""
    n = len(next(iter(outputs.values())))
    return xr.Dataset(
        {k: ("time", np.asarray(v)) for k, v in outputs.items()},
        coords={"time": pd.date_range(start, periods=n)},
    )

def first_divergence(sim, ref, atol):
    """First date on which |sim - ref| > atol (a NaN counts); None when the series agree."""
    s, r = xr.align(sim, ref, join="inner")
    bad = ~(np.abs(s.values - r.values) <= atol)
    return None if not bad.any() else s.time.values[int(np.argmax(bad))]

params, forcing, state0 = toy_inputs(n_days=8)
sim = to_dataset(run(MODEL, params, forcing, state0))

drift = np.where(np.arange(8) >= 3, 0.05 * (np.arange(8) - 2), 0.0)
frame = pd.DataFrame({"DATE": sim.time.values, "CWAD": sim["biomass"].values + drift})  # stand-in for a reader's output
ref = frame.set_index("DATE").rename_axis("time").to_xarray()  # the Dataset with a time axis

report = compare_series(sim, ref, {"biomass": "CWAD"}, Tolerance(max_abs=0.01), title="bucket vs stand-in reference")
print(report.to_markdown(metrics=("bias", "rmse", "max_abs"), heading_level=3))

date = first_divergence(sim["biomass"], ref["CWAD"], atol=0.01)
k = int((sim.time.values == date).argmax())
print("first day beyond the tolerance:", str(date)[:10], f"(day index {k})")

before = jax.tree_util.tree_map(lambda x: x[:k], forcing)  # the state after the k days before it
state_k, _ = run(MODEL, params, before, state0, return_final=True)
day_k = jax.tree_util.tree_map(lambda x: x[k], forcing)
for p in MODEL.processes:  # then process by process (section 1)
    state_k = p(state_k, params, day_k)
    print(f"after {p.name:<10} water={float(state_k.water):.4f} biomass={float(state_k.biomass):.4f}")
print("reference biomass that day:", float(ref["CWAD"].values[k]))
```

```text
### bucket vs stand-in reference

| period | variable | reference | units | bias | RMSE | max abs | tolerance | status |
|---|---|---|---|---:|---:|---:|---:|---:|
| all | `biomass` | `CWAD` |  | -0.094 | 0.131 | 0.250 | max_abs<=0.01 | FAIL (max_abs) |

first day beyond the tolerance: 2020-05-04 (day index 3)
after infiltrate water=5.3740 biomass=57.7431
after grow       water=4.8366 biomass=74.6235
reference biomass that day: 74.67350773564918
```

The first differing day is the day to reproduce; the first field that differs, after which process, tells which process to read against the published equations cited in its `Source:` line. Check at which point of the day the reference writes each output variable before comparing it with a state of the model.
