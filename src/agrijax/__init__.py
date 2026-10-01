"""Agri-JAX: differentiable, batch-parallel field-scale crop-soil models.

The short way in (``import agrijax as aj``; everything below is imported on first use, so
``import agrijax`` itself stays light and does not change any JAX setting):

* :mod:`dssat <agrijax.dssat>` - Agri-JAX's re-implementation of the DSSAT-CSM v4.8.6 maize model on
  DSSAT's own experiment files: run a season, batch scenarios (``aj.dssat.experiment("UFGA8201")``),
  and compare with the original DSSAT program (``aj.dssat.install_reference()``);
* :func:`calibrate` - CERES-Maize cultivar calibration on an experiment (:func:`agrijax.dssat.calibrate`);
* :mod:`plot <agrijax.report.plot>` - figures of seasons, comparisons, batches and calibrations
  (needs matplotlib: ``pip install "agrijax[plot]"``);
* :mod:`export <agrijax.facade_export>` - results to pandas, xarray, DSSAT-style CSV files and
  DSSAT-format ``.OUT`` files written by Agri-JAX, not by DSSAT (``season.to_frame()``,
  ``season.to_xarray()``, ``season.write_csv(directory)``, ``season.write_dssat_out(directory)``);
* :func:`machine` - the machine the numbers were measured on (CPU, cores, JAX devices);
* :func:`options` - float32 (opt-in) and the device (``"cpu"`` / ``"gpu"``) for the runs above, as a
  block (``with aj.options(precision="float32", device="gpu"):``) or as the ``precision=`` /
  ``device=`` keywords of ``run`` (:mod:`agrijax.facade_execution`); your JAX settings are put back.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.0.1"

# The names below are resolved on first use (no import statement here: ``agrijax`` is the parent of
# every module, and the layer rules check static import closures).
#: lazily served names: ``name -> (module, attribute or None for the module itself)``
_LAZY: dict[str, tuple[str, str | None]] = {
    "dssat": ("agrijax.dssat", None),
    "calibrate": ("agrijax.dssat", "calibrate"),
    "plot": ("agrijax.report.plot", None),
    "machine": ("agrijax._machine", "machine"),
    "options": ("agrijax.facade_execution", "options"),
    "DeviceNotFoundError": ("agrijax.facade_execution", "DeviceNotFoundError"),
    "export": ("agrijax.facade_export", None),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod, attr = _LAZY[name]
        m = importlib.import_module(mod)
        return m if attr is None else getattr(m, attr)
    raise AttributeError(f"module 'agrijax' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = ["__version__"]
