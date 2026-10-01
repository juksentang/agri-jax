"""The tobacco demo's one model coefficient (the tmin/tmax weight of the daily mean temperature)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from agrijax.core.coefficients import coefficient_table
from agrijax.core.runtime import run
from agrijax.models import tobacco_demo as T


def test_declared_with_provenance() -> None:
    p = T.default_params()
    rows = {r["path"]: r for r in coefficient_table(p)}
    assert set(rows) == {"tmean_weight"}
    r = rows["tmean_weight"]
    assert r["value"] == 0.5 and isinstance(r["value"], float)
    assert r["unit"] == "-" and r["calibratable"] and not r["static"]
    assert r["ref_version"] == "none" and "McMaster" in r["paper"] and r["equation"] == "1"
    field = {f.name: f for f in dataclasses.fields(p)}["tmean_weight"]
    assert field.metadata["coefficient"]


def test_calibratable_and_used() -> None:
    dates = pd.date_range("2016-01-01", "2016-12-31", freq="D")
    doy = np.arange(1, len(dates) + 1)
    tmean = 10.0 + 12.0 * np.sin(2 * np.pi * (doy - 110) / 365)
    met = pd.DataFrame({"tmin": tmean - 6.0, "tmax": tmean + 6.0}, index=dates)
    forcing = T.build_forcing(met, T.synthetic_events(dates), "2016-01-01", "2016-12-31")
    s0 = T.initial_state(20)
    model = T.tobacco_model()
    base = T.default_params()

    def total(w: jax.Array) -> jax.Array:
        return run(model, base.set("tmean_weight", w), forcing, s0)["mass_added"][-1, 0]

    g = float(jax.grad(total)(jnp.asarray(0.5)))
    assert np.isfinite(g) and g > 0.0  # a warmer mean temperature grows more leaf mass
    # the default (a Python float) and the same value as a traced array give the same run
    assert float(total(jnp.asarray(0.5))) == pytest.approx(
        float(run(model, base, forcing, s0)["mass_added"][-1, 0]), rel=1e-12
    )
