"""State/Params/Forcing base classes: field metadata, dotted-path access, functional updates."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from agri_jax.core.state import get_path, leaf_paths, set_path

from .toy import ToyForcing, ToyParams, ToyState


def test_field_metadata_table() -> None:
    meta = ToyState.field_metadata()
    assert meta["water"]["unit"] == "cm"
    assert meta["water"]["fortran_name"] == "SW"
    assert meta["biomass"]["description"] == "above-ground biomass"
    assert ToyForcing.field_metadata()["rain"]["dims"] == ("T",)


def test_paths_get_set_replace() -> None:
    s = ToyState(water=jnp.asarray(1.0), biomass=jnp.asarray(2.0))
    assert leaf_paths(s) == ["water", "biomass"]
    assert float(s.get("water")) == 1.0
    assert float(get_path(s, "biomass")) == 2.0
    s2 = s.set("water", jnp.asarray(9.0))
    assert float(s2.water) == 9.0 and float(s.water) == 1.0
    s3 = set_path(s, "biomass", jnp.asarray(0.0))
    assert float(s3.biomass) == 0.0
    s4 = s.replace(water=jnp.asarray(4.0))
    assert float(s4.water) == 4.0
    assert dict(s.items()).keys() == {"water", "biomass"}
    assert get_path({"a": [1, 2]}, "a.1") == 2


def test_forcing_n_days() -> None:
    f = ToyForcing(rain=jnp.zeros(7), srad=jnp.zeros(7))
    assert f.n_days == 7


def test_params_are_pytrees() -> None:
    p = ToyParams(k=jnp.asarray(0.1), rue=jnp.asarray(2.0))
    with pytest.raises(AttributeError):
        p.k = jnp.asarray(0.2)  # type: ignore[misc]  # equinox modules are frozen
