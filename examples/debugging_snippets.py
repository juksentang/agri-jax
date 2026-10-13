"""Every snippet of docs/debugging.md, runnable: debugging a model and its processes.

The script builds a two-process toy model (a water bucket and its biomass) and walks through the
techniques of the guide, one section each. It needs no data and runs on CPU in seconds:

    uv run python examples/debugging_snippets.py
"""

from __future__ import annotations

import os
import traceback

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

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------------- the toy model
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
    """Add today's rain to the bucket.

    Source: toy model, no literature.
    """
    return state.set("water", state.water + forcing_t.rain)


@process(reads=("water", "biomass"), writes=("water", "biomass"), register=False)
def grow(state, params, forcing_t):
    """Use a fraction of the water and turn it into biomass while the bucket is not empty.

    Source: toy model, no literature.
    """
    uptake = params.k * state.water
    gain = jnp.where(state.water > 0.0, params.rue * forcing_t.srad * uptake, 0.0)
    return state.replace(water=state.water - uptake, biomass=state.biomass + gain)


def toy_inputs(n_days: int = 6, water0: float = 5.0, rain_first: float = 1.0):
    t = np.arange(n_days)
    rain = np.where(t % 3 == 0, 1.0, 0.0)
    rain[0] = rain_first
    forcing = Weather(rain=jnp.asarray(rain), srad=jnp.asarray(15.0 + 5.0 * np.sin(t)))
    params = BucketParams(k=jnp.asarray(0.1), rue=jnp.asarray(2.0))
    state0 = Bucket(water=jnp.asarray(water0), biomass=jnp.asarray(0.0))
    return params, forcing, state0


MODEL = Model(Bucket, [infiltrate, grow], outputs=("water", "biomass"), name="bucket")


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


# ------------------------------------------------ 1. call a process directly (no runtime, no scan)
def section_1_direct_call() -> None:
    banner("1. a process called directly")
    params, _, state = toy_inputs()
    day = Weather(rain=jnp.asarray(1.0), srad=jnp.asarray(15.0))  # one day: scalar leaves, no time axis

    after_rain = infiltrate(state, params, day)  # ordinary Python: print, breakpoint() and pdb work here
    print("infiltrate: water", state.water, "->", after_rain.water)

    s = state  # the whole day, one process at a time
    for p in MODEL.processes:
        s = p(s, params, day)
        print(f"  after {p.name:<10} water={float(s.water):.4f} biomass={float(s.biomass):.4f}")

    _, outputs = MODEL.compile()(state, params, day)  # the same day as the compiled step
    print("compiled day step:", {k: float(v) for k, v in outputs.items()})


# ------------------------------------------------------------ 2. jax.disable_jit(): scan is a loop
@process(reads=("water",), writes=(), register=False)
def print_water(state, params, forcing_t):
    """Print the bucket water (a debugging tap that changes nothing).

    Source: debugging aid, no literature.
    """
    print("print_water sees", state.water)
    return state


def section_2_disable_jit() -> None:
    banner("2. jax.disable_jit()")
    params, forcing, state0 = toy_inputs(n_days=3)
    model = Model(Bucket, [infiltrate, print_water, grow], outputs=("water", "biomass"))

    print("plain run: the scan body is traced once, the print shows a tracer")
    run(model, params, forcing, state0)

    print("under disable_jit: one print per day, with values")
    with jax.disable_jit():
        run(model, params, forcing, state0)


# ------------------------------------------------------------ 3. jax.debug.print in scanned code
@process(reads=("water", "biomass"), writes=(), register=False)
def tap(state, params, forcing_t):
    """Print the state of the day from inside jit/scan (changes nothing).

    Source: debugging aid, no literature.
    """
    jax.debug.print(
        "day: rain={r} water={w:.4f} biomass={b:.4f}", r=forcing_t.rain, w=state.water, b=state.biomass
    )
    return state


def section_3_debug_print() -> None:
    banner("3. jax.debug.print")
    params, forcing, state0 = toy_inputs(n_days=3)
    model = Model(Bucket, [infiltrate, grow, tap], outputs=("water", "biomass"))
    run(model, params, forcing, state0)  # compiled as usual, and it prints every day


# ----------------------------------------------------------- 4. NaNs: jax_debug_nans, NaN gradients
@process(reads=("water", "biomass"), writes=("biomass",), register=False)
def sqrt_stress(state, params, forcing_t):
    """Biomass gain with a square-root factor: NaN when the water is below the day's radiation.

    Source: toy model, no literature.
    """
    return state.set("biomass", state.biomass + jnp.sqrt(state.water - forcing_t.srad))


_TINY = numerical_guard(
    "debugging_snippets.tiny", 1e-12, "floor of a logarithm's argument, keeps its derivative finite"
)


def unsafe(w):
    """log(w) where w > 0, else 0. Deliberately unsafe: the unused branch takes the log of 0 (rule AJ003)."""
    return jnp.where(w > 0.0, jnp.log(w), 0.0)


def safe(w):
    """The same with the project's pattern: clamp the argument, select afterwards."""
    return jnp.where(w > 0.0, jnp.log(jnp.maximum(w, _TINY)), 0.0)


@process(reads=("water", "biomass"), writes=("biomass",), register=False)
def log_growth_unsafe(state, params, forcing_t):
    """Biomass gain with the log of the water (0 for an empty bucket), unsafe form.

    Source: toy model, no literature.
    """
    return state.set("biomass", state.biomass + params.rue * unsafe(state.water))


@process(reads=("water", "biomass"), writes=("biomass",), register=False)
def log_growth_safe(state, params, forcing_t):
    """The same gain, safe form.

    Source: toy model, no literature.
    """
    return state.set("biomass", state.biomass + params.rue * safe(state.water))


def section_4_nans() -> None:
    banner("4a. jax_debug_nans: stop at the first NaN")
    params, forcing, state0 = toy_inputs()
    model = Model(Bucket, [infiltrate, sqrt_stress], outputs=("water", "biomass"))
    out = run(model, params, forcing, state0)
    print("without the flag the NaN travels silently:", np.asarray(out["biomass"]))

    jax.config.update("jax_debug_nans", True)  # raise at the first NaN (jax_debug_infs: at the first inf)
    try:
        try:
            run(model, params, forcing, state0)  # the scan is one compiled op: it is the one reported
        except FloatingPointError as e:
            print("compiled scan:  ", type(e).__name__, "-", e)
        with jax.disable_jit():  # a Python loop: every operation is checked on its own
            run(model, params, forcing, state0)
    except FloatingPointError as e:
        frame = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == __file__][-1]
        print("with disable_jit:", type(e).__name__, "-", e)
        print(f"  traceback: {frame.name}, line {frame.lineno}: {frame.line}")
    finally:
        jax.config.update("jax_debug_nans", False)

    banner("4b. a NaN gradient from jnp.where with an unsafe branch")

    for w in (0.0, -1.0):
        print(
            f"  w={w:+.0f}: unsafe value={float(unsafe(w)):+.1f} grad={float(jax.grad(unsafe)(w)):+.1f}"
            f"   safe value={float(safe(w)):+.1f} grad={float(jax.grad(safe)(w)):+.1f}"
        )
    print(
        "  sqrt, w=-1: grad",
        float(jax.grad(lambda w: jnp.where(w > 0.0, jnp.sqrt(w), 0.0))(-1.0)),
        "(unsafe);",
        float(jax.grad(lambda w: jnp.where(w > 0.0, jnp.sqrt(jnp.maximum(w, _TINY)), 0.0))(-1.0)),
        "(safe)",
    )

    # in a model: the bucket is empty on the first day (no water, no rain); is every gradient finite?
    params, forcing, state0 = toy_inputs(water0=0.0, rain_first=0.0)
    loss = lambda out: out["biomass"][-1]  # noqa: E731
    for proc in (log_growth_unsafe, log_growth_safe):
        model = Model(Bucket, [infiltrate, grow, proc], outputs=("water", "biomass"))
        value, grad = run_and_grad(model, loss, params, forcing, state0)
        finite = jax.tree_util.tree_map(lambda g: bool(jnp.isfinite(g).all()), grad)
        print(f"  {proc.name:<17} loss={float(value):.4f}  gradient finite: {finite}")

    # jax_debug_nans on that gradient, under disable_jit: what it reports depends on how it is set
    model = Model(Bucket, [infiltrate, grow, log_growth_unsafe], outputs=("water", "biomass"))
    jax.config.update("jax_debug_nans", True)
    try:
        with jax.disable_jit():
            _, grad = run_and_grad(model, loss, params, forcing, state0)  # no error, the NaN is returned
        finite = jax.tree_util.tree_map(lambda g: bool(jnp.isfinite(g).all()), grad)
        print(f"  config flag: no error raised; gradient finite: {finite}")
    finally:
        jax.config.update("jax_debug_nans", False)
    try:
        with jax.debug_nans(True), jax.disable_jit():
            run_and_grad(model, loss, params, forcing, state0)  # raises in div, without naming the process
    except FloatingPointError as e:
        ours = os.sep + "agrijax" + os.sep
        tb = traceback.extract_tb(e.__traceback__)
        frames = [f.name for f in tb if f.filename == __file__ or ours in f.filename]
        print(f"  scoped flag: {type(e).__name__} - {e}; frames: {' > '.join(frames)}")

    # find the culprit: differentiate one process for one day, called directly (section 1)
    state = Bucket(water=jnp.asarray(0.0), biomass=jnp.asarray(0.0))
    day = Weather(rain=jnp.asarray(0.0), srad=jnp.asarray(15.0))
    for proc in (log_growth_unsafe, log_growth_safe):

        def biomass_of_water(w, proc=proc):
            return proc(state.replace(water=w), params, day).biomass

        try:
            with jax.debug_nans(True):  # the flag for this block only
                g = jax.grad(biomass_of_water)(jnp.asarray(0.0))
            print(f"  {proc.name:<17} d biomass / d water = {float(g)}")
        except FloatingPointError as e:
            print(f"  {proc.name:<17} {type(e).__name__} - {e}")


# ---------------------------------------------- 5. AGRI_JAX_CHECK, Day.check and the strict lint
@process(reads=("water",), writes=("water",), register=False)
def sloppy(state, params, forcing_t):
    """Take a centimetre of water and book it as biomass, declaring only the water.

    Source: toy model, no literature.
    """
    return state.replace(water=state.water - 1.0, biomass=state.biomass + 1.0)


def _soil_day(s, p, f):
    """Source: fixture."""
    return set_path(s, "soil.w", get_path(s, "soil.w") + f.rain - get_path(s, "iface.uptake"))


def _grow_lai(s, p, f):
    """Source: fixture."""
    return set_path(s, "crop.lai", get_path(s, "crop.lai") + 0.01 * get_path(s, "soil.w"))


def _publish(s, p, f):
    """Source: fixture."""
    return set_path(s, "iface.uptake", 0.1 * get_path(s, "crop.lai"))


DAY_PROCS = {
    "soil.day": process(_soil_day, reads=("soil.w", "iface.uptake"), writes=("soil.w",), register=False),
    "crop.grow": process(_grow_lai, reads=("soil.w", "crop.lai"), writes=("crop.lai",), register=False),
    "crop.publish": process(_publish, reads=("crop.lai",), writes=("iface.uptake",), register=False),
}
ENTRIES = (Phase("day", ("soil.day", "crop.grow", "crop.publish")),)

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


def section_5_checks() -> None:
    banner("5a. AGRI_JAX_CHECK=1: undeclared writes")
    params, forcing, state0 = toy_inputs(n_days=3)
    model = Model(Bucket, [infiltrate, sloppy], outputs=("water", "biomass"))
    out = run(model, params, forcing, state0)
    print("check off: runs, biomass", np.asarray(out["biomass"]), "(nobody declared it)")

    before = os.environ.get("AGRI_JAX_CHECK")
    os.environ["AGRI_JAX_CHECK"] = "1"  # the same as running the script with AGRI_JAX_CHECK=1
    try:
        run(model, params, forcing, state0)
    except ProcessWriteError as e:
        print("check on:", type(e).__name__, "-", e)
    finally:
        if before is None:
            del os.environ["AGRI_JAX_CHECK"]
        else:
            os.environ["AGRI_JAX_CHECK"] = before

    banner("5b. Day.check: order and lags")
    day = Day(ref="toy", phases=ENTRIES)
    try:
        day.compile(DAY_PROCS)
    except DayLagError as e:
        print(type(e).__name__, "-", e)
    unchecked = day.compile(DAY_PROCS, check=False)
    print("lagged reads:", day.lagged_reads(unchecked))
    print("dataflow (writer, reader, path):", unchecked.dataflow())

    allowed = Day(
        ref="toy",
        phases=ENTRIES,
        lags=(Lag("soil.day", "iface.uptake", evidence="uptake is published for tomorrow"),),
    )
    report = allowed.check(allowed.compile(DAY_PROCS, check=False))
    print("with the lag declared: used", report.used, "unused", report.unused_pairs)
    # reads that see yesterday's value: carried state (infiltrate reads water), or a lag
    print("stale reads of the bucket model:", MODEL.stale_reads())

    banner("5c. the strict lint")
    for f in lint_source(LEAKY, "leaky.py"):
        print(" ", f.format())


# ------------------------------------------------------- 6. compare with a reference day by day
def to_dataset(outputs: dict, start: str = "2020-05-01") -> xr.Dataset:
    """``run`` outputs (one array per path, time first) as an xarray Dataset with a ``time`` axis."""
    n = len(next(iter(outputs.values())))
    return xr.Dataset(
        {k: ("time", np.asarray(v)) for k, v in outputs.items()},
        coords={"time": pd.date_range(start, periods=n)},
    )


def first_divergence(sim: xr.DataArray, ref: xr.DataArray, atol: float):
    """First date on which |sim - ref| > atol (a NaN counts); None when the series agree."""
    s, r = xr.align(sim, ref, join="inner")
    bad = ~(np.abs(s.values - r.values) <= atol)
    return None if not bad.any() else s.time.values[int(np.argmax(bad))]


def section_6_compare() -> None:
    banner("6. day-by-day comparison")
    params, forcing, state0 = toy_inputs(n_days=8)
    sim = to_dataset(run(MODEL, params, forcing, state0))

    # stand-in for a reference output: the biomass series with a drift from day 4 on, in the shape the
    # readers return (a pandas DataFrame with a DATE column and the reference's own variable names).
    # A real one comes from a reference run (agrijax.port.run_dscsm, run_rzwqm) read with agrijax.io
    # (read_plantgro: one block per run and treatment, select one; read_ana returns an xarray Dataset)
    drift = np.where(np.arange(8) >= 3, 0.05 * (np.arange(8) - 2), 0.0)
    frame = pd.DataFrame({"DATE": sim.time.values, "CWAD": sim["biomass"].values + drift})
    ref = frame.set_index("DATE").rename_axis("time").to_xarray()  # the Dataset with a time axis

    report = compare_series(
        sim, ref, {"biomass": "CWAD"}, Tolerance(max_abs=0.01), title="bucket vs stand-in reference"
    )
    print(report.to_markdown(metrics=("bias", "rmse", "max_abs"), heading_level=3))

    date = first_divergence(sim["biomass"], ref["CWAD"], atol=0.01)
    k = int((sim.time.values == date).argmax())
    print("first day beyond the tolerance:", str(date)[:10], f"(day index {k})")

    # step into that day: the state after the k days before it, then process by process (section 1)
    before = jax.tree_util.tree_map(lambda x: x[:k], forcing)
    state_k, _ = run(MODEL, params, before, state0, return_final=True)
    day_k = jax.tree_util.tree_map(lambda x: x[k], forcing)
    for p in MODEL.processes:
        state_k = p(state_k, params, day_k)
        print(f"  after {p.name:<10} water={float(state_k.water):.4f} biomass={float(state_k.biomass):.4f}")
    print("  reference biomass that day:", float(ref["CWAD"].values[k]))


if __name__ == "__main__":
    section_1_direct_call()
    section_2_disable_jit()
    section_3_debug_print()
    section_4_nans()
    section_5_checks()
    section_6_compare()
