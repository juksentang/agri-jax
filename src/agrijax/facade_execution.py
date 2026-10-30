"""Execution options of the DSSAT facade: float32 and the device, without touching your JAX settings.

What this adds, as user code that continues the quick start of :mod:`agrijax`::

    import agrijax as aj

    exp = aj.dssat.experiment("UFGA8201")
    season32 = exp.run(treatment=4, precision="float32")             # opt-in; float64 is the default
    scen = exp.scenarios(treatment=4, years=range(1978, 1988), sowing_shift=[-14, 0, 14])
    on_gpu = scen.run({"G2": [800.0, 900.0, 1000.0]}, device="gpu")  # or "cpu"; a clear error without a GPU
    with aj.options(precision="float32", device="gpu"):              # the same for every call in a block
        quick = scen.run({"G2": [800.0, 900.0, 1000.0]})
    print(quick.timing["precision"], quick.timing["devices"])        # float32, e.g. "1 x NVIDIA H100 ..."

**Precision.** ``"float64"`` (the default) is the precision the model is validated in against
DSSAT-CSM. ``"float32"`` is an opt-in for forward runs (a season, a batch of scenarios and cultivar
samples) where speed and memory matter more than the last digits. What was measured, float32
against float64 on the same inputs, both on the CPU
(``tests/integration/test_facade_float32_examples.py``, which prints these figures; a Fir CPU node,
JAX 0.10.2, 2026-10-30):

* every DSSAT maize example treatment the facade runs (57 treatments of the ``example_data/Maize``
  experiments; the 19 others need a DSSAT process Agri-JAX does not implement and are refused): grain
  yield at the end of the season, relative difference median 2.7e-7, 56 of the 57 within 1e-3, the
  largest 1.5e-2 in one treatment (BRPI0202 t04; tops weight there 7.0e-3, median 5.2e-7);
  silking and maturity dates unchanged in all 57;
* UFGA8201 treatment 4 on its ten weather years 1978-1987 x 100 cultivar samples with P1, P5, G2, G3
  and PHINT drawn within +-10 % of the published values (1000 seasons): yield median 2.0e-7, 99th
  percentile 5.8e-6, largest 8.1e-6; stage dates unchanged in all 2000;
* float32 on a CPU and float32 on a GPU are not bit-identical (XLA's float32 division on a GPU is not
  correctly rounded; the cause is documented in :mod:`agrijax.core.grad`, ``_divide``): the figures
  of a GPU follow below.

A single season can move by a percent or so where a threshold of the model is crossed differently in
the last bits, while the bulk agrees to a few 1e-7. Float32 is therefore for point estimates and
ensemble statistics; do not use it to calibrate against the reference at the 0.1 % level or for
gradients (neither was tested for this model in float32), and compare float32 results with float64
ones to the tolerance above, not bit for bit. :func:`agrijax.calibrate` runs in float64 only and
refuses a block with float32 options.

The facade's own check on DSSAT's example experiment UFGA8201 is
``tests/integration/test_facade_execution.py`` (a CPU node, JAX 0.10.2): treatments 4 and 6 and a
season with changed cultivar coefficients agree to 3.3e-7 relative in yield and biomass, the daily LAI,
tops weight, grain weight and soil water to 2.0e-6 of the series' largest value, and the dates exactly;
a batch of 144 seasons (9 weather-year and sowing-date scenarios x 16 perturbed cultivars) has a median
yield difference of 1.6e-7, a largest of 2.7e-6, and no differing stage date (0 of 288). On an H100
(a 10 GB MIG slice) the same runs with ``device="gpu"`` against ``device="cpu"``: float64 to 1e-15
relative in yield and biomass; float32 to 4.2e-7 in yield and biomass of a season, 3.0e-6 in the daily
series, and in the batch a median yield difference of 1.0e-7, a largest of 2.1e-6 and identical dates.

**Device.** ``device=`` is ``"cpu"``, ``"gpu"`` (every GPU JAX sees; a batch is spread over them),
``"gpu:1"`` (one device by index) or ``None`` / ``"auto"`` (JAX's own default, as before). Asking for
a GPU that is not there raises :class:`DeviceNotFoundError` before anything is built, saying what JAX
does see and how to get a GPU build. On a GPU the first call of a batch compiles for longer (the
layer loops are unrolled by the backend setting of :mod:`agrijax.core.execution`; the option is not
repeated here); the compiled programs are kept per precision and device, so a repeated call is fast.

**Your JAX settings stay yours.** Every facade call (``exp.run``, ``exp.scenarios``, ``scen.run``,
``aj.calibrate``, the gradients and sensitivities) switches ``jax_enable_x64`` on for its own duration
only, when it was off, and switches it off again on return, also when the call raises
(:func:`facade_scope`). A result that computes later (``scen.run`` on scenarios made earlier, the
companions of a weather sensitivity) does the same on each call, so it works whatever the setting is
by then. Calls with ``precision=`` / ``device=`` (and a ``with aj.options(...)`` block) also snapshot
``jax_default_device`` and put it back. Float32 runs only inside the call, in a program traced with
``x64`` off, while the inputs are built in float64 and rounded once to float32. JAX's configuration is
process-wide: do not change it from another thread while a facade call runs.

**The compilation cache.** The facade's programs take seconds to compile; JAX's persistent compilation
cache keeps them on disk, so a new Python process does not compile them again. Facade calls use it by
default, in :func:`compile_cache_dir`: ``AGRI_JAX_COMPILE_CACHE`` when it names a directory, else
``<cache>/jax`` under Agri-JAX's cache (``AGRI_JAX_CACHE``, else ``$XDG_CACHE_HOME/agrijax``, else
``~/.cache/agrijax``). ``AGRI_JAX_COMPILE_CACHE=0`` switches it off. Your own JAX setting wins: when
``jax_compilation_cache_dir`` is set (``JAX_COMPILATION_CACHE_DIR`` or ``jax.config.update``) or the
cache is switched off (``jax_enable_compilation_cache``), the facade uses your setting and changes
nothing. Otherwise the cache is set for the facade call only and taken away after it, so your own
compilations outside facade calls are not cached by Agri-JAX. The cache holds compiled programs that
JAX runs: keep it in your own cache directory, not in one others can write to.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import os
import re
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

__all__ = [
    "COMPILE_CACHE_ENV",
    "PRECISIONS",
    "DeviceNotFoundError",
    "Options",
    "active",
    "compile_cache_dir",
    "facade_call",
    "facade_scope",
    "options",
]

_F = TypeVar("_F", bound=Callable[..., Any])

#: the environment variable of the facade's compilation cache: a directory, or ``0`` / ``off`` /
#: ``false`` / ``no`` to switch it off (unset or empty: the default directory, :func:`compile_cache_dir`)
COMPILE_CACHE_ENV = "AGRI_JAX_COMPILE_CACHE"
_OFF = frozenset({"0", "off", "false", "no"})

#: the precisions of a run: the validated default first
PRECISIONS = ("float64", "float32")
_PRECISION_NAMES = {
    "float64": "float64",
    "f64": "float64",
    "double": "float64",
    "64": "float64",
    "float32": "float32",
    "f32": "float32",
    "single": "float32",
    "32": "float32",
}
_DEVICE = re.compile(r"^(cpu|gpu|cuda|rocm)(?::(\d+))?$")
_AUTO = frozenset({"auto", "default", ""})


class DeviceNotFoundError(RuntimeError):
    """A device was asked for (``device="gpu"``) that JAX does not see on this machine."""


@dataclass(frozen=True)
class Options:
    """The execution options in force: :attr:`precision` (``"float64"`` or ``"float32"``) and
    :attr:`device` (``"auto"``: JAX's default; ``"cpu"``, ``"gpu"``, ``"gpu:<index>"``)."""

    precision: str = "float64"
    device: str = "auto"

    @property
    def is_default(self) -> bool:
        """The validated float64 on JAX's own default device (what a call without options does)."""
        return self.precision == "float64" and self.device == "auto"


#: the options of the enclosing :func:`options` blocks (innermost wins field by field)
_DEFAULT = Options()
_ACTIVE: contextvars.ContextVar[Options] = contextvars.ContextVar("agrijax_facade_options", default=_DEFAULT)


def active() -> Options:
    """The options a facade call made now would use (:func:`options`; the defaults outside any block)."""
    return _ACTIVE.get()


def _precision(value: Any) -> str:
    """The canonical precision name of ``value`` (``"float32"``, ``np.float32``, ``32`` ...)."""
    if isinstance(value, str | int):
        name = _PRECISION_NAMES.get(str(value).strip().lower())
    else:
        try:
            name = _PRECISION_NAMES.get(np.dtype(value).name)
        except TypeError:
            name = None
    if name is None:
        raise ValueError(
            f"precision={value!r} is not known: use 'float64' (the default, validated against DSSAT-CSM) "
            "or 'float32' (opt-in, for forward runs; its measured accuracy is in agrijax.facade_execution)"
        )
    return name


def _device(value: Any) -> str:
    """The canonical device name of ``value`` (``"cpu"``, ``"gpu"``, ``"gpu:1"``, ``"auto"``)."""
    text = "auto" if value is None else str(value).strip().lower()
    if text in _AUTO:
        return "auto"
    hit = _DEVICE.match(text)
    if hit is None:
        raise ValueError(
            f"device={value!r} is not known: use 'cpu', 'gpu', 'gpu:<index>' (one GPU) or None / 'auto' "
            "(JAX's own default)"
        )
    kind = "gpu" if hit.group(1) in ("gpu", "cuda", "rocm") else "cpu"
    return kind if hit.group(2) is None else f"{kind}:{int(hit.group(2))}"


def _devices(platform: str) -> list[Any]:
    """JAX's devices of ``platform`` (``"cpu"`` / ``"gpu"``); none when the platform is not there."""
    import jax

    try:
        return list(jax.devices(platform))
    except RuntimeError:
        return []


def _seen() -> str:
    import jax

    return ", ".join(sorted({d.platform for d in jax.devices()}))


def _resolve(opts: Options) -> list[Any] | None:
    """The devices ``opts`` names (``None`` for ``"auto"``: leave JAX's default alone). Raises
    :class:`DeviceNotFoundError` with what JAX sees when the device is not there."""
    if opts.device == "auto":
        return None
    kind, _, index = opts.device.partition(":")
    found = _devices(kind)
    if not found:
        hint = (
            "A GPU needs a GPU build of JAX (pip install -U 'jax[cuda12]') and an NVIDIA GPU visible to "
            "this process (on a cluster, request one from the scheduler)"
            if kind == "gpu"
            else "JAX has no CPU backend here"
        )
        env = os.environ.get("JAX_PLATFORMS")
        if env:
            hint += f"; the environment variable JAX_PLATFORMS={env} limits the platforms JAX uses"
        raise DeviceNotFoundError(
            f"device={opts.device!r} was requested but JAX sees no {kind.upper()} (it sees: {_seen()}). "
            f"{hint}. Use device='cpu' to run on the CPU."
        )
    if index:
        i = int(index)
        if i >= len(found):
            raise DeviceNotFoundError(
                f"device={opts.device!r} was requested but JAX sees {len(found)} {kind.upper()} "
                f"device(s) (indices 0 to {len(found) - 1})"
            )
        return [found[i]]
    return found


@dataclass(frozen=True)
class _Config:
    """The two JAX settings an options block puts back."""

    x64: bool
    default_device: Any


def _snapshot() -> _Config:
    import jax

    return _Config(bool(jax.config.jax_enable_x64), jax.config.jax_default_device)


def _restore(saved: _Config) -> None:
    import jax

    if bool(jax.config.jax_enable_x64) != saved.x64:
        jax.config.update("jax_enable_x64", saved.x64)
    if jax.config.jax_default_device is not saved.default_device:
        jax.config.update("jax_default_device", saved.default_device)


@contextlib.contextmanager
def options(precision: Any = None, device: Any = None) -> Iterator[Options]:
    """Run the facade calls of a block with execution options, and leave your JAX settings as they were.

    ::

        with aj.options(precision="float32", device="gpu"):
            batch = scen.run({"G2": [800.0, 900.0, 1000.0]})

    ``precision``: ``"float64"`` (the default, validated against DSSAT-CSM) or ``"float32"`` (opt-in
    for forward runs; accuracy in :mod:`agrijax.facade_execution`). ``device``: ``"cpu"``, ``"gpu"``
    (every GPU JAX sees), ``"gpu:<index>"`` or ``None`` / ``"auto"`` (JAX's default). ``None`` for
    either leaves the enclosing block's value (nested blocks inherit); ``device="auto"`` resets to
    JAX's default. The same options are the keywords ``precision=`` / ``device=`` of
    :meth:`Experiment.run <agrijax.dssat.Experiment.run>`, :meth:`Experiment.run_batch
    <agrijax.dssat.Experiment.run_batch>` and :meth:`Scenarios.run <agrijax.dssat.Scenarios.run>`.

    On exit (also when the block raises) ``jax_enable_x64`` and ``jax_default_device`` are put back to
    what they were on entry; ``with aj.options():`` is the way to run the defaults and keep them.
    Raises :class:`ValueError` for an unknown name and :class:`DeviceNotFoundError` when the device is
    not there, before anything runs. :func:`agrijax.calibrate` refuses a block that is not the
    default. Yields the :class:`Options` in force.
    """
    import jax

    enclosing = active()
    new = Options(
        enclosing.precision if precision is None else _precision(precision),
        enclosing.device if device is None else _device(device),
    )
    devs = _resolve(new)
    saved = _snapshot()
    token = _ACTIVE.set(new)
    try:
        with contextlib.ExitStack() as stack:
            if devs:
                stack.enter_context(jax.default_device(devs[0]))
            yield new
    finally:
        _ACTIVE.reset(token)
        _restore(saved)


def require_default_options(what: str) -> None:
    """Raise :class:`ValueError` when ``what`` (a facade function that runs in float64 on JAX's default
    device only) is called inside an :func:`options` block that asks for anything else."""
    opts = active()
    if not opts.is_default:
        raise ValueError(
            f"{what} runs in float64 on JAX's default device (the calibration is validated that way); "
            f"it was called inside aj.options(precision={opts.precision!r}, device={opts.device!r}). "
            "Leave the options for run() and scenarios().run(), or call it outside the block."
        )


# ------------------------------------------------------------------ the traced segments
@contextlib.contextmanager
def _x64(enabled: bool) -> Iterator[None]:
    """``jax_enable_x64`` set to ``enabled`` for the block, then what it was."""
    import jax

    before = bool(jax.config.jax_enable_x64)
    if before != enabled:
        jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        if bool(jax.config.jax_enable_x64) != before:
            jax.config.update("jax_enable_x64", before)


# ------------------------------------------------------------------ the scope of a facade call
#: nested facade scopes in this context: only the outermost changes JAX's settings
_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar("agrijax_facade_depth", default=0)


def compile_cache_dir() -> Path | None:
    """The directory of the facade's persistent compilation cache (module docstring): the one
    ``AGRI_JAX_COMPILE_CACHE`` names, ``None`` when it switches the cache off (``0``, ``off``,
    ``false``, ``no``), else ``<cache>/jax`` under :func:`agrijax.io.examples.cache_dir`. Used only
    when your JAX configuration sets no cache directory of its own."""
    raw = os.environ.get(COMPILE_CACHE_ENV, "").strip()
    if raw.lower() in _OFF:
        return None
    if raw:
        return Path(raw).expanduser()
    from agrijax.io.examples import cache_dir

    return cache_dir() / "jax"


def _cache_reset() -> Callable[[], None] | None:
    """JAX's ``reset_cache`` (the cache object is made again at the next compilation), ``None`` when
    this JAX has none (the facade then leaves the compilation cache alone)."""
    try:
        from jax.experimental.compilation_cache import compilation_cache as cc
    except ImportError:
        return None
    return getattr(cc, "reset_cache", None)


@contextlib.contextmanager
def _compile_cache() -> Iterator[None]:
    """The facade's compilation cache for the block, when your JAX configuration names none and does not
    switch the cache off (module docstring); then nothing set, as before."""
    import jax

    mine = jax.config.jax_compilation_cache_dir is None and bool(jax.config.jax_enable_compilation_cache)
    path = compile_cache_dir() if mine else None
    reset = _cache_reset() if path is not None else None
    if path is None or reset is None:
        yield
        return
    jax.config.update("jax_compilation_cache_dir", str(path))
    reset()  # JAX makes its cache object once, at the first compilation that sees a directory
    try:
        yield
    finally:
        if jax.config.jax_compilation_cache_dir == str(path):
            jax.config.update("jax_compilation_cache_dir", None)
            reset()  # later compilations outside facade calls see no cache, as before the call


@contextlib.contextmanager
def facade_scope() -> Iterator[None]:
    """The scope of one facade call: ``jax_enable_x64`` on (the facade's day is validated in float64)
    and the compilation cache of :func:`compile_cache_dir`, both put back as they were on exit, also
    when the block raises. Nested scopes change nothing (the outermost one owns the settings); an
    :func:`options` block inside one still switches x64 off for its float32 programs."""
    depth = _DEPTH.get()
    token = _DEPTH.set(depth + 1)
    try:
        if depth:
            yield
            return
        with _x64(True), _compile_cache():
            yield
    finally:
        _DEPTH.reset(token)


def facade_call(fn: _F) -> _F:
    """``fn`` run in a :func:`facade_scope` (the decorator of every facade entry point and of every
    result method that runs JAX later): the call sees float64, the caller's JAX settings are back
    after it."""

    @functools.wraps(fn)
    def call(*args: Any, **kwargs: Any) -> Any:
        with facade_scope():
            return fn(*args, **kwargs)

    call.__agrijax_facade_call__ = True  # type: ignore[attr-defined]
    return call  # type: ignore[return-value]


@contextlib.contextmanager
def _runtime(opts: Options, devices: Sequence[Any] | None) -> Iterator[None]:
    """The block that traces and runs a day program with ``opts``: ``jax_enable_x64`` on for float64,
    off for float32, on the first of ``devices`` as the default device. In float32 JAX's note that
    a float64 constant made before the switch (a module-level array) is truncated is silenced: that
    truncation is the float32 run."""
    import jax

    with contextlib.ExitStack() as stack:
        stack.enter_context(_x64(opts.precision == "float64"))
        if devices:
            stack.enter_context(jax.default_device(devices[0]))
        if opts.precision == "float32":
            stack.enter_context(warnings.catch_warnings())
            warnings.filterwarnings(
                "ignore", message="Explicitly requested dtype float64", category=UserWarning
            )
        yield


def _to_float32(tree: Any) -> Any:
    """``tree`` of NumPy arrays with every floating leaf rounded once to float32 (integers, booleans
    unchanged)."""
    import jax

    def cast(a: Any) -> Any:
        a = np.asarray(a)
        return a.astype(np.float32) if np.issubdtype(a.dtype, np.floating) else a

    return jax.tree.map(cast, tree)


def _widen(a: np.ndarray) -> np.ndarray:
    """A float32 result as float64 (exact), so a float32 run's tables have the dtypes of a float64 run's."""
    return a.astype(np.float64) if a.dtype == np.float32 else a


def simulate(x: Any, cultivar: Any = None, *, pad_days: int = 0) -> dict[str, np.ndarray]:
    """:meth:`FreeRunInputs.simulate <agrijax.sites.dssat_free_run.FreeRunInputs.simulate>` of ``x``
    under the options in force (the same call when they are the defaults): the daily outputs as
    NumPy arrays, in float64 whatever the precision of the run."""
    opts = active()
    if opts.is_default:
        return x.simulate(cultivar, pad_days=pad_days)
    devs = _resolve(opts)
    if opts.precision == "float64":
        with _runtime(opts, devs):
            return x.simulate(cultivar, pad_days=pad_days)
    from agrijax.core.runtime import run_sites
    from agrijax.models.day_dssat486 import SLOT, day_dssat486, day_processes
    from agrijax.sites.dssat_free_run import season_outputs, stack_trees, with_cultivar

    n = x.n_days + int(pad_days)
    with _x64(True):  # the inputs are built in float64, as validated, and rounded once
        p0 = x.params()
        p = with_cultivar(p0, cultivar) if cultivar else p0
        trees = _to_float32([stack_trees([t]) for t in (p, x.forcing(n), x.state(p0))])
    with _runtime(opts, devs):
        model = day_dssat486(SLOT).compile(
            day_processes(SLOT, mesev=x.mesev), outputs=season_outputs(), exact_lags=True
        )
        out = run_sites(model, *trees)
        return {k: _widen(np.asarray(v)[0]) for k, v in out.items()}


class _OptionSimulator:
    """A :class:`~agrijax.calib.dssat_day.DaySimulator` whose calls run under fixed options (the
    float32 or device-pinned simulator of a :class:`~agrijax.dssat.Scenarios`)."""

    def __init__(self, sim: Any, opts: Options, devices: Sequence[Any] | None) -> None:
        self._sim = sim
        self.options = opts
        self._devices = devices

    def __call__(self, theta: np.ndarray, tid: np.ndarray) -> np.ndarray:
        with _runtime(self.options, self._devices):
            return self._sim(theta, tid)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name == "_sim":
            raise AttributeError(name)
        return getattr(self._sim, name)


def simulator(scenarios: Any, build: Callable[..., Any]) -> Any:
    """The batched simulator of ``scenarios`` for the options in force: the one cached on it when they
    are the defaults, else one built (``build(devices=..., dtype=...)``: the inputs built in float64
    and stored in the precision asked for, on the devices asked for) and kept per precision and
    device. Its calls run under those options."""
    opts = active()
    if opts.is_default:
        if scenarios._sim is None:
            scenarios._sim = build()
        return scenarios._sim
    cache: dict[tuple[str, str], Any] = scenarios.__dict__.setdefault("_option_sims", {})
    key = (opts.precision, opts.device)
    if key not in cache:
        devs = _resolve(opts)
        with _x64(True):
            sim = build(devices=devs, dtype=np.float32 if opts.precision == "float32" else None)
        cache[key] = _OptionSimulator(sim, opts, devs)
    return cache[key]
