"""Placeholder so the tier collects; throughput and vmap-correctness tests arrive with the first model."""

import jax


def test_gpu_visible() -> None:
    assert any(d.platform != "cpu" for d in jax.devices())
