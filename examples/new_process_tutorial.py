"""Try a new process from scratch: a leaf-area expansion formula, from NumPy to a checked process.

The script behind docs/tutorial_new_process.md. The process itself (steps 2 to 5) is in
``tutorial_lai_process.py``; this file holds the rest: the NumPy prototype (step 1), a model run on a
synthetic season (step 6), the conformance case (step 7) and the lint (step 8). No data needed.

    uv run python examples/new_process_tutorial.py
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
os.environ.setdefault("AGRI_JAX_CHECK", "1")  # every process call verifies that only its `writes` changed

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from tutorial_lai_process import (  # noqa: E402
    KEY,
    DemoParams,
    LaiCoefficients,
    LaiForcing,
    LaiState,
    ThermalTimeCoefficients,
    degree_days,
    lai_logistic,
)

from agrijax.core import Model, run  # noqa: E402
from agrijax.core.lint import lint_file, lint_source  # noqa: E402
from agrijax.testing.conformance import (  # noqa: E402
    Balance,
    ConformanceCase,
    GradSpec,
    check_reads,
    run_checks,
    synthetic,
)

HERE = Path(__file__).resolve().parent


# ---- step 1: the formula as plain NumPy, numbers written in place ---------------------------------
def lai_step_numpy(lai: float, tt: float, dtt: float, swfac: float) -> float:
    """One day of the new formula for one canopy: floats in, float out, an ordinary ``if``."""
    if tt < 1200.0:  # expansion: logistic in thermal time, scaled by water stress
        dlai = 0.008 * dtt * lai * (1.0 - lai / 5.0) * swfac
    else:  # senescence
        dlai = -0.03 * lai
    return float(np.maximum(lai + dlai, 0.0))


def season_numpy(tmean: np.ndarray, swfac: np.ndarray, lai0: float = 0.05, tbase: float = 8.0) -> np.ndarray:
    """The whole season with the prototype: a loop over days is fine in a prototype."""
    lai, tt, out = lai0, 0.0, []
    for tm, sw in zip(tmean, swfac):
        dtt = max(tm - tbase, 0.0)
        tt += dtt
        lai = lai_step_numpy(lai, tt, dtt, sw)
        out.append(lai)
    return np.asarray(out)


# ---- step 6: a small model on a synthetic season --------------------------------------------------
MODEL = Model(LaiState, [degree_days, lai_logistic], outputs=("lai", "tt"), name="lai_demo")


def demo_params(dtype: Any, **lai: float) -> DemoParams:
    """Default coefficients as arrays of ``dtype`` (``lai`` overrides LAI coefficients by name)."""
    return DemoParams(
        thermal=ThermalTimeCoefficients().as_arrays(dtype), lai=LaiCoefficients(**lai).as_arrays(dtype)
    )


def initial_state(n_crop: int, dtype: Any, lai0: float = 0.05) -> LaiState:
    """Emergence: no thermal time yet and a seedling LAI."""
    zeros = jnp.zeros(n_crop, dtype)
    return LaiState(tt=zeros, dtt=zeros, lai=jnp.full(n_crop, lai0, dtype), dlai=zeros)


def season_forcing(n_days: int, dtype: Any) -> LaiForcing:
    """The weather of the conformance kit (`synthetic.ceres_weather`) plus a dry spell every 37 days."""
    w = synthetic.ceres_weather(7, dtype, n_days)
    swfac = np.where(np.arange(n_days) % 37 > 25, 0.5, 1.0)
    return LaiForcing(tmean=(w.tmax + w.tmin) / 2.0, swfac=jnp.asarray(swfac, dtype))


# ---- step 7: the conformance case -----------------------------------------------------------------
N_DAYS = 3
N_CROP = 2


def make(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """Synthetic ``(state0, params, forcing)`` from a NumPy generator (never ``jax.random``).

    Variants put the canopy on the expansion branch (``nominal``), the senescence branch
    (``senescence``) or before emergence (``bare``: LAI 0, an edge for the gradients).
    """
    tt = {"nominal": (200.0, 600.0), "senescence": (1300.0, 1800.0), "bare": (100.0, 300.0)}[variant]
    lai = {"nominal": (0.3, 2.0), "senescence": (2.0, 4.0), "bare": (0.0, 0.0)}[variant]
    w = synthetic.ceres_weather(int(rng.integers(1_000_000)), dtype, N_DAYS)
    state = LaiState(
        tt=jnp.asarray(rng.uniform(*tt, N_CROP), dtype),
        dtt=jnp.asarray(rng.uniform(10.0, 20.0, N_CROP), dtype),
        lai=jnp.asarray(rng.uniform(*lai, N_CROP), dtype),
        dlai=jnp.zeros(N_CROP, dtype),
    )
    forcing = LaiForcing(
        tmean=(w.tmax + w.tmin) / 2.0, swfac=jnp.asarray(rng.uniform(0.4, 1.0, N_DAYS), dtype)
    )
    return state, demo_params(dtype), forcing


#: leaf area is not conserved, but the day's change must be what the state records
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
    variants=("nominal", "senescence"),
    n_days=N_DAYS,
    coefficient_sets=("lai",),  # the set this process uses (the degree-day set belongs to the helper)
    forcing_fields=("swfac",),  # the only forcing field the process reads
    balances=(LAI_BOOKKEEPING,),
    grad=GradSpec(edge_variants=("bare",)),
    origin="examples/new_process_tutorial.py",
)

# ---- step 8: what the lint says about a process that breaks the rules -----------------------------
BAD_PROCESS = '''
@process(reads=("lai",), writes=("lai",))
def lai_bad(state, params, forcing_t):
    """Bad on purpose."""
    if state.lai > 4.5:
        return state.replace(lai=state.lai * 0.97)
    return state.replace(lai=state.lai + 0.12 * forcing_t.swfac)
'''


def main() -> None:
    dtype = jnp.float64
    n_days = synthetic.N_SEASON
    forcing, state0, params = season_forcing(n_days, dtype), initial_state(1, dtype), demo_params(dtype)

    print("step 1 and 6: the NumPy prototype and the process, one season")
    ref = season_numpy(np.asarray(forcing.tmean), np.asarray(forcing.swfac))
    out = jax.jit(lambda p, f, s: run(MODEL, p, f, s))(params, forcing, state0)
    lai = np.asarray(out["lai"])[:, 0]
    gap = float(np.max(np.abs(lai - ref)))
    print(f"  dataflow (writer, reader, path): {MODEL.dataflow()}")
    print(f"  peak LAI {lai.max():.3f} on day {int(lai.argmax())}; max |process - NumPy| = {gap:.1e}")
    assert gap < 1e-12, "the process does not reproduce the NumPy prototype"
    assert lai.max() > 4.0 and lai[-1] < lai.max(), "the season should expand, peak, then senesce"

    print("step 6: change a coefficient, then differentiate the season")
    slow = run(MODEL, demo_params(dtype, r_expand=0.006), forcing, state0)["lai"][:, 0]
    print(f"  r_expand 0.008 -> 0.006: LAI on day 30 {lai[30]:.2f} -> {float(slow[30]):.2f}")

    def peak_lai(c: LaiCoefficients) -> Any:
        return run(MODEL, params.replace(lai=c), forcing, state0)["lai"].max()

    peak, grad = jax.jit(peak_lai), jax.grad(peak_lai)(params.lai)
    for name in ("r_expand", "lai_max"):
        x, h = getattr(params.lai, name), 1e-6
        up, down = params.lai.replace(**{name: x + h}), params.lai.replace(**{name: x - h})
        fd, ad = float(peak(up) - peak(down)) / (2 * h), float(getattr(grad, name))
        print(f"  d peak LAI / d {name:10s} = {ad:+.6f} (finite difference {fd:+.6f})")
        assert abs(ad - fd) < 1e-5 * max(1.0, abs(fd))
    print(f"  d peak LAI / d tt_senesce  = {float(grad.tt_senesce):+.1f} (a hard `where`: no gradient)")

    print(f"step 7: conformance checks of {KEY}")
    result = run_checks(CASE)
    for name, message in result.items():
        print(f"  {name:14s}{'ok' if message is None else 'FAILED ' + message}")
    assert all(m is None for m in result.values()), "a conformance check failed"
    print("  and a process that forgets to declare that it reads dtt:")
    sloppy = dataclasses.replace(lai_logistic, reads=("tt", "lai"))
    failed = run_checks(dataclasses.replace(CASE, process=sloppy), checks=[check_reads])["reads"]
    print(f"    {failed}")
    assert failed is not None and "dtt" in failed

    print("step 8: the lint (every function of the process module, as a kernel)")
    findings = lint_file(HERE / "tutorial_lai_process.py", kernel=True, processes=("lai_logistic",))
    print(f"  tutorial_lai_process.py: {len(findings)} finding(s)")
    assert not findings, findings
    print("  and a process that breaks the rules:")
    for f in lint_source(BAD_PROCESS, "lai_bad.py"):
        print(f"    {f.format()}")


if __name__ == "__main__":
    main()
