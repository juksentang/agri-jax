"""The scope of a facade call (:func:`agrijax.facade_execution.facade_scope`): ``jax_enable_x64`` on for
the call only and the user's setting back after it (also on an exception, nested scopes), every facade
entry point and every result method that runs JAX later inside one, and the persistent compilation
cache: Agri-JAX's directory for the call only when the user's JAX configuration names none, the user's
own setting untouched, the off switch."""

from __future__ import annotations

import ast
import itertools
from collections.abc import Iterator
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax import dssat as ajd
from agrijax import facade_execution as fe
from agrijax import facade_grad as fg
from agrijax import facade_weather as fw

SRC = Path(fe.__file__).parent

#: every facade function and result method that runs JAX (directly or later, on kept programs): each
#: must run in a facade scope. A new one is added here (the AST test below finds no other x64 switch).
FACADE_CALLS = [
    ajd.Experiment.inputs,
    ajd.Experiment.run,
    ajd.Experiment.scenarios,
    ajd.Experiment.run_batch,
    ajd.Experiment.gradient,
    ajd.Experiment.weather_sensitivity,
    ajd.Experiment.dssat_batch,
    ajd.Scenarios.run,
    ajd.Scenarios.sensitivity,
    ajd.Scenarios.weather_sensitivity,
    ajd.calibrate,
    fg.gradient,
    fg.sensitivity,
    fw.weather_sensitivity,
    fw.WeatherSensitivity.phenology_free,
    fw.WeatherSensitivity.brute_force,
    fw.WeatherSensitivity.attribution,
]


@pytest.fixture
def x64_restored() -> Iterator[None]:
    before = bool(jax.config.jax_enable_x64)
    yield
    jax.config.update("jax_enable_x64", before)


def test_every_facade_call_runs_in_a_scope():
    missing = [f.__qualname__ for f in FACADE_CALLS if not getattr(f, "__agrijax_facade_call__", False)]
    assert not missing, missing


def test_only_the_scope_switches_x64():
    """No facade module but :mod:`agrijax.facade_execution` calls ``jax.config.update("jax_enable_x64", ...)``
    (the switch the plain calls used to leave on)."""
    found = []
    for path in [SRC / "dssat.py", *sorted(SRC.glob("facade_*.py"))]:
        if path.name == "facade_execution.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "jax_enable_x64"
            ):
                found.append(f"{path.name}:{node.lineno}")
    assert not found, found


@pytest.mark.parametrize("user", [False, True])
def test_the_scope_puts_the_users_x64_back(user, x64_restored):
    jax.config.update("jax_enable_x64", user)
    seen = []

    @fe.facade_call
    def call(fail: bool = False) -> str:
        seen.append(bool(jax.config.jax_enable_x64))
        if fail:
            raise RuntimeError("inside")
        return str(jnp.asarray(1.0).dtype)

    assert call() == "float64" and seen == [True]
    assert bool(jax.config.jax_enable_x64) is user
    with pytest.raises(RuntimeError, match="inside"):
        call(fail=True)
    assert bool(jax.config.jax_enable_x64) is user and seen == [True, True]
    # nested: the inner scope changes nothing, the outer one restores
    with fe.facade_scope():
        call()
        assert bool(jax.config.jax_enable_x64)
    assert bool(jax.config.jax_enable_x64) is user
    assert call.__name__ == "call" and call.__agrijax_facade_call__  # type: ignore[attr-defined]


def test_an_options_block_inside_a_scope_still_runs_float32(x64_restored):
    jax.config.update("jax_enable_x64", False)
    with fe.facade_scope():
        with fe._runtime(fe.Options("float32"), None):
            inside = bool(jax.config.jax_enable_x64)
        after_block = bool(jax.config.jax_enable_x64)
    assert (inside, after_block, bool(jax.config.jax_enable_x64)) == (False, True, False)


# ------------------------------------------------------------------ the compilation cache
_COUNT = itertools.count()


def _compile_new() -> None:
    """Compile a program never compiled before in this process (a new constant, a new function)."""
    c = float(next(_COUNT)) + 0.5
    jax.jit(lambda x: jnp.sin(x) * c + c).lower(np.arange(3.0)).compile()


def _entries(d: Path) -> set[str]:
    return {p.name for p in d.rglob("*") if p.is_file()} if d.is_dir() else set()


@pytest.fixture
def no_user_cache(monkeypatch) -> Iterator[None]:
    """JAX as a user without a cache directory has it (the clusters' runs set ``JAX_COMPILATION_CACHE_DIR``:
    cleared for the test and put back after), every compilation cached (minimum compile time 0)."""
    reset = fe._cache_reset()
    assert reset is not None, "this JAX has no compilation_cache.reset_cache"
    before = jax.config.jax_compilation_cache_dir, jax.config.jax_persistent_cache_min_compile_time_secs
    jax.config.update("jax_compilation_cache_dir", None)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    reset()
    yield
    jax.config.update("jax_compilation_cache_dir", before[0])
    jax.config.update("jax_persistent_cache_min_compile_time_secs", before[1])
    reset()


def test_the_facade_caches_its_compilations_for_the_call_only(tmp_path, monkeypatch, no_user_cache):
    d = tmp_path / "jc"
    monkeypatch.setenv(fe.COMPILE_CACHE_ENV, str(d))
    with fe.facade_scope():
        assert jax.config.jax_compilation_cache_dir == str(d)
        _compile_new()
    first = _entries(d)
    assert first, "nothing written to the facade's compilation cache"
    assert jax.config.jax_compilation_cache_dir is None
    _compile_new()  # the user's own compilation, outside facade calls: not cached by Agri-JAX
    assert _entries(d) == first
    with fe.facade_scope():  # a second call writes to the same cache again
        _compile_new()
    assert len(_entries(d)) > len(first)


def test_the_users_cache_setting_wins(tmp_path, monkeypatch, no_user_cache):
    mine = tmp_path / "mine"
    monkeypatch.setenv(fe.COMPILE_CACHE_ENV, str(tmp_path / "agrijax"))
    jax.config.update("jax_compilation_cache_dir", str(mine))
    with fe.facade_scope():
        assert jax.config.jax_compilation_cache_dir == str(mine)
    assert jax.config.jax_compilation_cache_dir == str(mine)
    assert not (tmp_path / "agrijax").exists()
    jax.config.update("jax_compilation_cache_dir", None)
    enabled = bool(jax.config.jax_enable_compilation_cache)
    jax.config.update("jax_enable_compilation_cache", False)
    try:
        with fe.facade_scope():  # the cache switched off by the user: no directory set
            assert jax.config.jax_compilation_cache_dir is None
    finally:
        jax.config.update("jax_enable_compilation_cache", enabled)


@pytest.mark.parametrize("off", ["0", "off", "False", "no"])
def test_the_off_switch(off, tmp_path, monkeypatch, no_user_cache):
    monkeypatch.setenv(fe.COMPILE_CACHE_ENV, off)
    assert fe.compile_cache_dir() is None
    with fe.facade_scope():
        assert jax.config.jax_compilation_cache_dir is None


def test_the_default_location_follows_the_cache_dir(tmp_path, monkeypatch):
    monkeypatch.delenv(fe.COMPILE_CACHE_ENV, raising=False)
    monkeypatch.setenv("AGRI_JAX_CACHE", "")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert fe.compile_cache_dir() == tmp_path / "xdg" / "agrijax" / "jax"
    monkeypatch.setenv("AGRI_JAX_CACHE", str(tmp_path / "own"))
    assert fe.compile_cache_dir() == tmp_path / "own" / "jax"
    monkeypatch.setenv(fe.COMPILE_CACHE_ENV, "")  # empty: the default
    assert fe.compile_cache_dir() == tmp_path / "own" / "jax"
    monkeypatch.setenv(fe.COMPILE_CACHE_ENV, str(tmp_path / "there"))
    assert fe.compile_cache_dir() == tmp_path / "there"
