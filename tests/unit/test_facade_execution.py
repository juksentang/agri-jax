"""The facade's execution options (precision, device) without data or DSSAT: the names and their
errors, nesting, the JAX settings put back (x64 and the default device, also on an exception), a
GPU that is not there, and the plumbing from ``run`` / ``scenarios().run`` / ``run_batch`` to the
program (the dtype and the settings the program is traced and run with), on stand-ins for the
experiment's inputs and the batched simulator. The real float32 run on DSSAT's example experiment
is ``tests/integration/test_facade_execution.py``."""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import agrijax as aj
from agrijax import dssat as ajd
from agrijax import facade_execution as fx


@pytest.fixture(autouse=True)
def _jax_settings_put_back():
    """Whatever a test does to the two settings, the next test starts from this test's start."""
    x64, dev = bool(jax.config.jax_enable_x64), jax.config.jax_default_device
    yield
    jax.config.update("jax_enable_x64", x64)
    jax.config.update("jax_default_device", dev)


def _cpu() -> Any:
    return jax.devices("cpu")[0]


def _set_x64(value: bool) -> None:
    jax.config.update("jax_enable_x64", value)


# ------------------------------------------------------------------ names and errors
def test_import_stays_light_and_the_names_are_lazy():
    code = (
        "import sys, agrijax as aj; assert 'jax' not in sys.modules; "
        "aj.options; aj.DeviceNotFoundError; import agrijax.facade_execution as fx; "
        "assert 'jax' not in sys.modules, 'jax imported'; print(fx.active())"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "float64" in out.stdout and "auto" in out.stdout
    assert aj.options is fx.options and aj.DeviceNotFoundError is fx.DeviceNotFoundError
    assert {"options", "DeviceNotFoundError"} <= set(dir(aj))
    assert issubclass(fx.DeviceNotFoundError, RuntimeError)


@pytest.mark.parametrize(
    ("given", "name"),
    [
        ("float64", "float64"),
        ("F64", "float64"),
        (" double ", "float64"),
        (64, "float64"),
        (np.float64, "float64"),
        ("float32", "float32"),
        ("f32", "float32"),
        ("single", "float32"),
        (32, "float32"),
        (np.float32, "float32"),
        (np.dtype("float32"), "float32"),
    ],
)
def test_precision_names(given, name):
    assert fx._precision(given) == name


@pytest.mark.parametrize("given", ["float16", "bf16", "", 16, True, object()])
def test_unknown_precision_is_refused_with_the_choices(given):
    with pytest.raises(ValueError, match=r"'float64'.*'float32'"):
        fx._precision(given)
    with pytest.raises(ValueError, match="precision="), fx.options(precision=given):
        pass


@pytest.mark.parametrize(
    ("given", "name"),
    [
        (None, "auto"),
        ("auto", "auto"),
        ("default", "auto"),
        ("cpu", "cpu"),
        ("CPU", "cpu"),
        ("gpu", "gpu"),
        ("cuda", "gpu"),
        ("rocm", "gpu"),
        ("gpu:1", "gpu:1"),
        ("cuda:02", "gpu:2"),
        ("cpu:0", "cpu:0"),
    ],
)
def test_device_names(given, name):
    assert fx._device(given) == name


@pytest.mark.parametrize("given", ["tpu", "gpu:", "gpu:x", "gpu:-1", "nvidia", 0])
def test_unknown_device_is_refused_with_the_choices(given):
    with pytest.raises(ValueError, match=r"'cpu', 'gpu'"):
        fx._device(given)


def test_options_are_default_outside_any_block_and_nest_by_field():
    assert fx.active() == fx.Options("float64", "auto") and fx.active().is_default
    with fx.options(precision="f32") as a:
        assert a == fx.Options("float32", "auto") and not a.is_default
        with fx.options(device="cpu") as b:  # inherits the precision
            assert b == fx.Options("float32", "cpu") == fx.active()
            with fx.options(precision="float64", device="auto") as c:
                assert c.is_default
            assert fx.active() == b
        assert fx.active() == a
    assert fx.active().is_default


def test_options_are_reset_when_the_block_raises():
    with pytest.raises(KeyError), fx.options(precision="float32"):
        raise KeyError("body")
    assert fx.active().is_default


# ------------------------------------------------------------------ JAX settings put back
@pytest.mark.parametrize("before", [True, False])
@pytest.mark.parametrize("opts", [{}, {"precision": "float32"}, {"device": "cpu"}])
def test_x64_is_put_back_after_the_facade_switched_it(before, opts):
    _set_x64(before)
    with fx.options(**opts):
        _set_x64(not before)  # what the facade's float64 switch does
        assert bool(jax.config.jax_enable_x64) is (not before)
    assert bool(jax.config.jax_enable_x64) is before


def test_x64_and_the_default_device_are_put_back_when_the_block_raises():
    _set_x64(False)
    before = jax.config.jax_default_device
    with pytest.raises(RuntimeError, match="boom"), fx.options(precision="float32", device="cpu"):
        _set_x64(True)
        jax.config.update("jax_default_device", _cpu())
        raise RuntimeError("boom")
    assert bool(jax.config.jax_enable_x64) is False
    assert jax.config.jax_default_device is before


def test_the_device_block_sets_the_default_device_and_puts_it_back():
    before = jax.config.jax_default_device
    with fx.options(device="cpu"):
        assert jax.config.jax_default_device == _cpu()
        assert jnp.ones(2).devices() == {_cpu()}
    assert jax.config.jax_default_device is before


def test_a_block_with_the_default_device_leaves_it_alone():
    before = jax.config.jax_default_device
    with fx.options(precision="float32"):
        assert jax.config.jax_default_device is before


def test_runtime_traces_in_the_precision_asked_for_and_restores_x64():
    _set_x64(True)
    table = jnp.linspace(0.0, 1.0, 5)  # a float64 constant made before the switch, as a module-level one

    @jax.jit
    def f(x):
        return x * table[:2].sum() + table.sum()

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with fx._runtime(fx.Options("float32"), None):
            assert not jax.config.jax_enable_x64
            y32 = f(np.ones(3))
        assert jax.config.jax_enable_x64
        y64 = jax.jit(lambda x: x * 2.0)(np.ones(3))
        with fx._runtime(fx.Options("float64"), [_cpu()]):
            assert jax.config.jax_enable_x64 and jax.config.jax_default_device == _cpu()
    assert y32.dtype == np.float32 and y64.dtype == np.float64
    np.testing.assert_allclose(np.asarray(y32), np.ones(3) * 0.25 + 2.5, rtol=1e-6)
    assert not [w for w in seen if "Explicitly requested dtype float64" in str(w.message)]


def test_to_float32_rounds_floating_leaves_once_and_keeps_the_others():
    tree = {
        "w": np.array([0.1, 2.0**-30 + 1.0]),
        "n": np.arange(3),
        "flag": np.array([True, False]),
        "i32": np.arange(2, dtype=np.int32),
        "x32": np.ones(2, dtype=np.float32),
    }
    out = fx._to_float32(tree)
    assert out["w"].dtype == np.float32 and out["x32"].dtype == np.float32
    np.testing.assert_array_equal(out["w"], tree["w"].astype(np.float32))
    assert out["n"].dtype == tree["n"].dtype and out["flag"].dtype == bool and out["i32"].dtype == np.int32
    assert fx._widen(out["w"]).dtype == np.float64 and fx._widen(out["n"]).dtype == tree["n"].dtype


# ------------------------------------------------------------------ a GPU that is not there
def _no_gpu(monkeypatch, platforms: str = "cpu") -> None:
    monkeypatch.setattr(fx, "_devices", lambda kind: [_cpu()] if kind == "cpu" else [])
    monkeypatch.setattr(fx, "_seen", lambda: platforms)


def test_gpu_that_is_absent_is_a_readable_error_before_anything_runs(monkeypatch):
    _no_gpu(monkeypatch)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    ran = []
    with pytest.raises(fx.DeviceNotFoundError) as e, fx.options(device="gpu"):
        ran.append(1)
    msg = str(e.value)
    assert not ran and fx.active().is_default
    assert "device='gpu'" in msg and "no GPU" in msg and "it sees: cpu" in msg
    assert "jax[cuda12]" in msg and "device='cpu'" in msg and "JAX_PLATFORMS" not in msg
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    with pytest.raises(fx.DeviceNotFoundError, match="JAX_PLATFORMS=cpu"):
        fx._resolve(fx.Options(device="gpu"))
    with pytest.raises(fx.DeviceNotFoundError, match="no GPU"), fx.options(device="gpu:0"):
        pass


def test_gpu_index_out_of_range_names_the_range(monkeypatch):
    gpus = [SimpleNamespace(platform="gpu", id=i) for i in range(2)]
    monkeypatch.setattr(fx, "_devices", lambda kind: gpus if kind == "gpu" else [_cpu()])
    assert fx._resolve(fx.Options(device="gpu")) == gpus
    assert fx._resolve(fx.Options(device="gpu:1")) == [gpus[1]]
    assert fx._resolve(fx.Options(device="auto")) is None
    with pytest.raises(fx.DeviceNotFoundError, match=r"2 GPU device\(s\) \(indices 0 to 1\)"):
        fx._resolve(fx.Options(device="gpu:2"))


def test_the_run_keywords_raise_the_same_error_before_building_the_inputs(monkeypatch):
    _no_gpu(monkeypatch)
    exp = _experiment(inputs=None)  # building the inputs would fail: nothing may get that far
    with pytest.raises(fx.DeviceNotFoundError, match="device='gpu'"):
        exp.run(1, device="gpu")
    with pytest.raises(fx.DeviceNotFoundError, match="device='gpu'"):
        exp.run_batch(1, device="gpu")
    with pytest.raises(fx.DeviceNotFoundError, match="device='gpu'"):
        _scenarios(None).run(device="gpu")
    with pytest.raises(ValueError, match="precision="):
        exp.run(1, precision="half")


# ------------------------------------------------------------------ the plumbing of run
class _Inputs:
    """What ``Experiment.run`` reads of a treatment's inputs (3 days, 2 layers), recording the settings
    its ``simulate`` is called with."""

    key = "TEST_t01"
    trno = 1
    n_days = 3
    nl = 2
    mesev = "S"
    days = np.array([2020001, 2020002, 2020003])
    series = {"dlayr_end": np.full((3, 2), 10.0)}

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def published(self) -> dict[str, float]:
        return dict(zip(ajd.CULTIVAR, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0), strict=True))

    def simulate(self, cultivar: Any = None, *, pad_days: int = 0) -> dict[str, np.ndarray]:
        self.calls.append(
            {
                "x64": bool(jax.config.jax_enable_x64),
                "device": jax.config.jax_default_device,
                "cultivar": cultivar,
                "pad_days": pad_days,
            }
        )
        n = self.n_days + pad_days
        stage = np.full((n, 1), 4)
        stage[-1] = 10
        return {
            "lai": np.linspace(0, 3, n)[:, None],
            "cwad": np.linspace(0, 1e4, n)[:, None],
            "gwad": np.linspace(0, 5e3, n)[:, None],
            "istage": stage,
            "soil_sw": np.full((n, self.nl), 0.2),
        }

    # the float32 route builds the trees from these
    def params(self) -> dict[str, Any]:
        return {"c": jnp.asarray([1.5, 2.5], dtype=jnp.float64), "n": np.int64(3)}

    def forcing(self, n: int) -> dict[str, Any]:
        return {"rain": np.arange(n, dtype=np.float64), "doy": np.arange(n)}

    def state(self, params: Any) -> dict[str, Any]:
        return {"sw": np.full(2, 0.25), "flag": np.array(True)}


def _experiment(inputs: Any) -> ajd.Experiment:
    exp = ajd.Experiment(
        "TEST8201",
        Path("TEST8201.MZX"),
        Path("data"),
        "synthetic",
        {1: "RAINFED"},
        {1: "IB0035"},
        "IBMZ910014",
        "UFGA",
    )
    if inputs is not None:
        exp._inputs[1] = inputs
    return exp


def test_run_without_options_is_the_unchanged_float64_call():
    x = _Inputs()
    _set_x64(False)
    season = _experiment(x).run(1)
    assert season.precision == "float64" and isinstance(season.daily, pd.DataFrame)
    assert x.calls == [
        {"x64": True, "device": jax.config.jax_default_device, "cultivar": None, "pad_days": 0}
    ]
    # the plain call keeps the facade's float64 switch on, as before; options are how to undo it
    assert jax.config.jax_enable_x64


def test_run_with_device_runs_float64_on_that_device_and_puts_the_settings_back():
    x = _Inputs()
    _set_x64(False)
    before = jax.config.jax_default_device
    season = _experiment(x).run(1, device="cpu", cultivar={"G2": 777.0})
    (call,) = x.calls
    assert call == {"x64": True, "device": _cpu(), "cultivar": {"G2": 777.0}, "pad_days": ajd.PAD_DAYS}
    assert season.precision == "float64" and season.cultivar["G2"] == 777.0
    assert bool(jax.config.jax_enable_x64) is False and jax.config.jax_default_device is before


def test_run_with_float32_rounds_the_inputs_once_and_runs_with_x64_off(monkeypatch):
    seen: list[dict[str, Any]] = []

    def run_sites(model: Any, params: Any, forcing: Any, state0: Any, **kw: Any) -> dict[str, Any]:
        seen.append(
            {
                "x64": bool(jax.config.jax_enable_x64),
                "trees": (params, forcing, state0),
                "model": model,
            }
        )
        n = forcing["rain"].shape[1]
        stage = np.full((1, n, 1), 4, dtype=np.int32)
        stage[0, -1] = 10
        f = lambda v: np.full((1, n, 1), v, dtype=np.float32)  # noqa: E731
        return {
            "lai": f(2.0),
            "cwad": f(100.0),
            "gwad": f(50.0),
            "istage": stage,
            "soil_sw": np.full((1, n, 2), 0.2, dtype=np.float32),
        }

    monkeypatch.setattr("agrijax.core.runtime.run_sites", run_sites)
    x = _Inputs()
    _set_x64(True)
    season = _experiment(x).run(1, precision="float32")
    assert x.calls == [] and len(seen) == 1  # not x.simulate: the day was run in float32
    (call,) = seen
    assert call["x64"] is False
    params, forcing, state0 = call["trees"]
    for tree in (params, forcing, state0):  # one site on the leading axis, floats rounded to float32
        for leaf in jax.tree.leaves(tree):
            assert leaf.shape[0] == 1
            assert leaf.dtype in (np.float32, np.int64, np.bool_), leaf.dtype
    assert params["c"].dtype == np.float32 and forcing["rain"].dtype == np.float32
    assert forcing["doy"].dtype == np.int64 and state0["flag"].dtype == np.bool_
    np.testing.assert_array_equal(params["c"][0], np.asarray([1.5, 2.5], np.float32))
    assert call["model"].processes  # the day's model of this soil evaporation was built
    # the season's tables are float64 (the float32 values widened), labelled float32
    assert season.precision == "float32" and season.daily["lai"].dtype == np.float64
    assert season.outputs["lai"].dtype == np.float64 and season.summary["HWAM"] == 50.0
    assert bool(jax.config.jax_enable_x64) is True  # restored


def test_run_with_float32_applies_a_cultivar_in_float64_before_rounding(monkeypatch):
    got: dict[str, Any] = {}
    import agrijax.sites.dssat_free_run as fr

    def with_cultivar(p: Any, values: Any) -> Any:
        got["x64"] = bool(jax.config.jax_enable_x64)
        got["values"] = dict(values)
        return {**p, "c": jnp.asarray([9.0, 9.5], dtype=jnp.float64)}

    monkeypatch.setattr(fr, "with_cultivar", with_cultivar)
    monkeypatch.setattr(
        "agrijax.core.runtime.run_sites",
        lambda model, p, f, s, **kw: got.update(c=p["c"]) or _day_outputs(f["rain"].shape[1]),
    )
    _set_x64(False)  # the user's setting: the inputs are still built in float64
    _experiment(_Inputs()).run(1, precision="float32", cultivar={"G2": 700.0})
    assert got["x64"] is True and got["values"] == {"G2": 700.0}
    assert got["c"].dtype == np.float32 and bool(jax.config.jax_enable_x64) is False


def _day_outputs(n: int) -> dict[str, np.ndarray]:
    stage = np.full((1, n, 1), 4, dtype=np.int32)
    stage[0, -1] = 10
    z = np.ones((1, n, 1), dtype=np.float32)
    return {"lai": z, "cwad": z, "gwad": z, "istage": stage, "soil_sw": np.full((1, n, 2), 0.2, np.float32)}


def test_a_block_applies_to_run_and_the_plain_call_in_it_runs_the_block_options(monkeypatch):
    seen: list[bool] = []
    monkeypatch.setattr(
        "agrijax.core.runtime.run_sites",
        lambda model, p, f, s, **kw: seen.append(bool(jax.config.jax_enable_x64)) or _day_outputs(3),
    )
    exp = _experiment(_Inputs())
    with aj.options(precision="float32"):
        assert exp.run(1).precision == "float32"  # no keyword: the block's options
        assert exp.run(1, precision="float64").precision == "float64"  # the keyword wins inside the block
    assert seen == [False] and exp.inputs(1).calls[-1]["x64"] is True
    assert exp.run(1).precision == "float64"


def test_calibrate_refuses_options_it_does_not_honour():
    fx.require_default_options("aj.calibrate")  # the defaults pass
    with aj.options():
        fx.require_default_options("aj.calibrate")
    for kw in ({"precision": "float32"}, {"device": "cpu"}):
        with aj.options(**kw), pytest.raises(ValueError, match=r"aj\.calibrate runs in float64"):
            ajd.calibrate("UFGA8201")


# ------------------------------------------------------------------ the plumbing of scenarios().run
class _Sim:
    """A stand-in of the batched simulator: records the settings of each call."""

    def __init__(self, **kw: Any) -> None:
        self.kw = kw
        self.devs = kw.get("devices") or [SimpleNamespace(device_kind="stand-in")]
        self.ndev = len(self.devs)
        self.n_days = {"g": 30}
        self.where = {0: ("g", 0)}
        self.calls: list[dict[str, Any]] = []
        self._calls = 0

    def stats(self) -> dict[str, Any]:
        return {"compile_s": 0.5 * self._calls, "call_s": 0.25 * self._calls}

    def __call__(self, theta: np.ndarray, tid: np.ndarray) -> np.ndarray:
        self._calls += 1
        self.calls.append({"x64": bool(jax.config.jax_enable_x64), "device": jax.config.jax_default_device})
        return np.tile([4.0, 9.0, 5000.0, 12000.0], (theta.shape[0], 1))


def _scenarios(sim: Any) -> ajd.Scenarios:
    runs = [SimpleNamespace(days=np.array([2020100]), trno=1)]
    table = pd.DataFrame({"year": [2020], "sowing_shift": [0]})
    return ajd.Scenarios(
        "TEST_t01",
        table,
        runs,
        dict(zip(ajd.CULTIVAR, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0), strict=True)),
        _sim=sim,
    )


def test_scenarios_run_default_uses_the_cached_simulator_unchanged():
    sim = _Sim()
    batch = _scenarios(sim).run({"G2": [800.0, 900.0]})
    assert [c["x64"] for c in sim.calls] == [True]
    assert batch.timing["precision"] == "float64" and batch.timing["devices"] == "1 x stand-in"
    assert batch.timing["seasons"] == 2 and list(batch.table["HWAM"]) == [5000.0, 5000.0]
    assert list(batch.table["ADAT"]) == [2020104, 2020104]


def test_simulator_cache_is_per_options_and_the_calls_run_under_them():
    built: list[tuple[dict[str, Any], bool]] = []

    def build(**kw: Any) -> _Sim:
        built.append((kw, bool(jax.config.jax_enable_x64)))
        return _Sim(**kw)

    scen = SimpleNamespace(_sim=None)
    _set_x64(False)
    plain = fx.simulator(scen, build)
    assert scen._sim is plain and built == [({}, False)] and fx.simulator(scen, build) is plain
    with fx.options(precision="float32"):
        s32 = fx.simulator(scen, build)
        assert fx.simulator(scen, build) is s32 and s32 is not plain
        assert built[-1] == ({"devices": None, "dtype": np.float32}, True)  # inputs built in float64
        s32(np.ones((2, 6)), np.zeros(2, int))
        assert s32.calls[-1]["x64"] is False and s32.ndev == 1 and s32.stats()["call_s"] == 0.25
    with fx.options(device="cpu"):
        scpu = fx.simulator(scen, build)
        assert scpu is not s32 and built[-1] == ({"devices": [_cpu()], "dtype": None}, True)
        scpu(np.ones((2, 6)), np.zeros(2, int))
        assert scpu.calls[-1] == {"x64": True, "device": _cpu()}
        assert scpu.devs == [_cpu()]
    with fx.options(precision="float32", device="cpu"):
        assert fx.simulator(scen, build) not in (s32, scpu)
    with fx.options(precision="float32"):
        assert fx.simulator(scen, build) is s32  # kept per options
    assert len(built) == 4 and scen._sim is plain
    assert bool(jax.config.jax_enable_x64) is False


def test_scenarios_run_keyword_uses_the_simulator_of_those_options():
    sim32 = fx._OptionSimulator(_Sim(), fx.Options("float32", "auto"), None)
    scen = _scenarios(None)
    scen._option_sims = {("float32", "auto"): sim32}
    _set_x64(True)
    batch = scen.run({"G2": [800.0, 900.0, 1000.0]}, precision="float32")
    assert batch.timing["precision"] == "float32" and batch.timing["seasons"] == 3
    assert [c["x64"] for c in sim32.calls] == [False]  # traced and run with x64 off
    assert bool(jax.config.jax_enable_x64) is True and fx.active().is_default


def test_run_batch_forwards_the_options(monkeypatch):
    exp = _experiment(None)
    seen: list[tuple[Any, ...]] = []

    class _S:
        def run(self, cultivar: Any = None) -> str:
            seen.append((cultivar, fx.active()))
            return "batch"

    monkeypatch.setattr(
        ajd.Experiment,
        "scenarios",
        lambda self, treatment, *, years=None, sowing_shift=(0,), soil_evaporation=None: _S(),
    )
    assert exp.run_batch(1, {"G2": [1.0]}, device="cpu", precision="float32") == "batch"
    assert seen == [({"G2": [1.0]}, fx.Options("float32", "cpu"))]
    assert exp.run_batch(1) == "batch" and seen[-1] == (None, fx.Options())


def test_the_day_simulator_takes_devices_and_dtype_as_keywords():
    import inspect

    from agrijax.calib.dssat_day import DaySimulator

    p = inspect.signature(DaySimulator.__init__).parameters
    assert p["devices"].default is None and p["dtype"].default is None
    assert p["devices"].kind is p["dtype"].kind is inspect.Parameter.KEYWORD_ONLY
