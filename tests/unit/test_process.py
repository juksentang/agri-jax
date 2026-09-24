"""@process: registration, signature check, and the writes check under AGRI_JAX_CHECK=1."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from agrijax.core import process, registry
from agrijax.core.process import (
    CHECK_ENV,
    Process,
    ProcessSignatureError,
    ProcessWriteError,
    check_enabled,
)
from agrijax.core.state import tree_diff

from .toy import ToyForcing, ToyParams, ToyState, grow, infiltrate


@pytest.fixture
def check_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CHECK_ENV, "1")
    assert check_enabled()


@pytest.fixture
def check_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CHECK_ENV, raising=False)
    assert not check_enabled()


@pytest.fixture
def sample() -> tuple[ToyState, ToyParams, ToyForcing]:
    state = ToyState(water=jnp.asarray(3.0), biomass=jnp.asarray(1.0))
    params = ToyParams(k=jnp.asarray(0.2), rue=jnp.asarray(1.5))
    forcing_t = ToyForcing(rain=jnp.asarray(0.5), srad=jnp.asarray(10.0))
    return state, params, forcing_t


def test_decorator_returns_process_and_registers() -> None:
    @process(reads=("water",), writes=("water",), source="unit test", fortran_name="NOOP")
    def registered_noop(state, params, forcing_t):
        """No-op.

        Source: unit test.
        """
        return state

    try:
        assert isinstance(registered_noop, Process)
        assert registry["registered_noop"] is registered_noop
        assert registered_noop.reads == ("water",)
        assert registered_noop.writes == ("water",)
        assert registered_noop.source == "unit test"
        assert registered_noop.fortran_name == "NOOP"
        assert "Source:" in registered_noop.doc
    finally:
        registry.pop("registered_noop", None)


def test_register_false_keeps_registry_clean() -> None:
    assert "infiltrate" not in registry
    assert "grow" not in registry
    assert infiltrate.name == "infiltrate"


def test_bare_decorator_form() -> None:
    @process
    def bare(state, params, forcing_t):
        return state

    try:
        assert isinstance(bare, Process)
        assert bare.writes == ()
    finally:
        registry.pop("bare", None)


def test_signature_must_be_three_positional() -> None:
    with pytest.raises(ProcessSignatureError):

        @process(register=False)
        def two_args(state, params):
            return state

    with pytest.raises(ProcessSignatureError):

        @process(register=False)
        def four_args(state, params, forcing_t, extra):
            return state


def test_invalid_path_rejected() -> None:
    with pytest.raises(ValueError):
        process(lambda s, p, f: s, writes=(".water",), register=False)


def test_declared_writes_pass(check_on, sample) -> None:
    state, params, forcing_t = sample
    out = infiltrate(state, params, forcing_t)
    assert float(out.water) == pytest.approx(3.5)
    assert out.biomass is state.biomass
    out2 = grow(out, params, forcing_t)
    assert float(out2.biomass) > 1.0


def test_undeclared_write_raises(check_on, sample) -> None:
    state, params, forcing_t = sample

    @process(reads=("water",), writes=("water",), register=False)
    def sneaky(state, params, forcing_t):
        return eqx.tree_at(lambda s: (s.water, s.biomass), state, (state.water + 1.0, state.biomass + 1.0))

    with pytest.raises(ProcessWriteError, match="biomass"):
        sneaky(state, params, forcing_t)


def test_undeclared_write_ignored_when_check_off(check_off, sample) -> None:
    state, params, forcing_t = sample

    @process(reads=("water",), writes=("water",), register=False)
    def sneaky(state, params, forcing_t):
        return eqx.tree_at(lambda s: s.biomass, state, state.biomass + 1.0)

    out = sneaky(state, params, forcing_t)
    assert float(out.biomass) == pytest.approx(2.0)


def test_subtree_and_wildcard_writes_cover_leaves(check_on) -> None:
    class Inner(ToyState):
        pass

    class Outer(ToyState.__mro__[1]):  # agrijax.core.State
        soil: Inner
        day: jax.Array

    state = Outer(soil=Inner(water=jnp.asarray(1.0), biomass=jnp.asarray(0.0)), day=jnp.asarray(0))

    @process(writes=("soil",), register=False)
    def touch_subtree(state, params, forcing_t):
        return eqx.tree_at(lambda s: s.soil.water, state, state.soil.water * 2.0)

    @process(writes=("soil",), register=False)
    def touch_outside(state, params, forcing_t):
        return eqx.tree_at(lambda s: s.day, state, state.day + 1)

    @process(writes="*", register=False)
    def touch_everything(state, params, forcing_t):
        return eqx.tree_at(lambda s: (s.soil.water, s.day), state, (state.soil.water * 2.0, state.day + 1))

    assert float(touch_subtree(state, None, None).soil.water) == 2.0
    with pytest.raises(ProcessWriteError, match="day"):
        touch_outside(state, None, None)
    assert int(touch_everything(state, None, None).day) == 1


def test_check_works_under_jit(check_on, sample) -> None:
    """Inside jit the leaves are tracers; the check falls back to identity and still catches the write."""
    state, params, forcing_t = sample

    @process(writes=("water",), register=False)
    def sneaky(state, params, forcing_t):
        return eqx.tree_at(lambda s: s.biomass, state, state.biomass + 1.0)

    with pytest.raises(ProcessWriteError):
        jax.jit(sneaky)(state, params, forcing_t)

    out = jax.jit(infiltrate)(state, params, forcing_t)
    assert float(out.water) == pytest.approx(3.5)


def test_tree_diff_reports_changed_paths(sample) -> None:
    state, *_ = sample
    same = eqx.tree_at(lambda s: s.water, state, state.water)
    assert tree_diff(state, same) == []
    changed = eqx.tree_at(lambda s: s.water, state, state.water + 1.0)
    assert tree_diff(state, changed) == ["water"]
    with pytest.raises(ValueError):
        tree_diff(state, {"water": 1.0})


def test_check_writes_returns_changed_leaves(sample) -> None:
    state, params, forcing_t = sample
    out = infiltrate.fn(state, params, forcing_t)
    assert infiltrate.check_writes(state, out) == ["water"]
