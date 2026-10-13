"""GPU tier: skipped unless JAX sees a non-CPU device."""

from __future__ import annotations

import jax
import pytest


@pytest.fixture(scope="session", autouse=True)
def require_gpu() -> None:
    if all(d.platform == "cpu" for d in jax.devices()):
        pytest.skip("no GPU device visible to JAX")
