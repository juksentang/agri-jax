"""The machine a measurement ran on, printed next to every speed number (:func:`machine`)."""

from __future__ import annotations

import os
import platform
from pathlib import Path


def _cpu_model() -> str:
    try:
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if ln.lower().startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def machine() -> str:
    """One line: CPU model, logical cores available to this process, memory, Python, JAX version
    and devices (imports JAX)."""
    import jax

    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover (not Linux)
        cores = os.cpu_count() or 0
    mem = ""
    try:
        kb = int(
            next(
                ln for ln in Path("/proc/meminfo").read_text().splitlines() if ln.startswith("MemTotal")
            ).split()[1]
        )
        mem = f", {kb / 2**20:.1f} GiB RAM"
    except (OSError, StopIteration, ValueError):
        pass
    devs = jax.devices()
    kinds = sorted({d.device_kind for d in devs})
    return (
        f"{_cpu_model()}, {cores} logical cores{mem}; Python {platform.python_version()}, "
        f"JAX {jax.__version__} on {len(devs)} x {'/'.join(kinds)} ({jax.default_backend()})"
    )
