"""The crop models the DSSAT facade runs, selected by the experiment's crop: one record per crop.

::

    from agrijax.crops import CROPS, crop, crop_of_filex

    c = crop_of_filex("UFGA8201.MZX")   # the record of the FileX's crop (by its extension)
    c.code, c.model, c.cul               # "MZ", "MZCER048", "MZCER048.CUL"
    crop("SB")                           # UnsupportedCropError: lists the supported crops

A :class:`CropModel` holds what the facade (:mod:`agrijax.dssat`, :func:`agrijax.calibrate`,
:mod:`agrijax.facade_export`) needs to know about a crop: DSSAT's names for it (crop code, model,
genotype files, FileX extensions, example directory), the cultivar coefficients it varies and
calibrates, the stage codes of its anthesis and maturity, and where its implementation lives (the
module of its rejection tables, its calibration adapter, its assembled day model). The values of
those modules are not copied here: the bounds of the cultivar coefficients are declared once in the
calibration adapter, the nitrogen-effect table once in the rejection-table module.

DSSAT names an experiment file ``<name>.<CR>X`` with ``CR`` the crop code, so the extension gives
the crop even when Agri-JAX does not implement it; such a crop is refused with the list of the
supported ones. :data:`CROPS` holds the implemented crops: CERES-Maize (DSSAT-CSM v4.8.6 ``MZCER048``).
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["CROPS", "MAIZE", "CropModel", "UnsupportedCropError", "crop", "crop_of_filex"]


class UnsupportedCropError(ValueError):
    """The crop (a crop code or the crop of a FileX) is not one Agri-JAX implements."""


@dataclass(frozen=True)
class CropModel:
    """One crop model of the DSSAT facade (module docstring)."""

    #: DSSAT crop code (FileX ``*CULTIVARS`` ``CR``, ``Summary.OUT`` ``CR``)
    code: str
    #: DSSAT's name of the crop (run header ``MODEL : <model> - <name>``)
    name: str
    #: DSSAT's model name (run header, ``Summary.OUT`` ``MODEL``)
    model: str
    #: stem of the genotype files ``.CUL`` / ``.ECO`` / ``.SPE`` (not always ``model``: DSSAT's
    #: CROPGRO soybean reads ``SBGRO048.*`` under the model ``CRGRO048``)
    genotype: str
    #: the implementation that runs (for messages)
    implementation: str
    #: the FileX extensions of the crop (upper case, with the dot)
    filex: tuple[str, ...]
    #: the ``example_data`` subdirectory of DSSAT's example experiments of the crop
    examples: str
    #: the cultivar coefficients the facade varies and calibrates (names and column order of the ``.CUL``)
    cultivar: tuple[str, ...]
    #: the stage codes of the implementation's stage output on the first day of anthesis (silking)
    #: and of maturity
    stages: tuple[int, int]
    #: module of the crop's rejection tables: ``OUT_OF_SCOPE`` (experiments refused as a whole, with
    #: the reason), ``NITROGEN_STRESS`` (DSSAT's yield change with nitrogen on, per treatment) and
    #: ``NITROGEN_DATES_DIFFER`` (treatments whose stage dates change with nitrogen on)
    scope: str
    #: module of the calibration adapter (the bounds, units and sources of the cultivar coefficients);
    #: named, not yet dispatched on: :func:`agrijax.calibrate` and the gradient, weather and execution
    #: entry points import the CERES-Maize adapter directly while maize is the one registered crop
    calibration: str
    #: module of the assembled day model; named, not yet dispatched on (as ``calibration``)
    day: str

    @property
    def cul(self) -> str:
        """File name of the cultivar file (``MZCER048.CUL``)."""
        return f"{self.genotype}.CUL"

    @property
    def eco(self) -> str:
        """File name of the ecotype file."""
        return f"{self.genotype}.ECO"

    @property
    def spe(self) -> str:
        """File name of the species file."""
        return f"{self.genotype}.SPE"

    def _table(self, name: str) -> Any:
        """The ``scope`` module's ``name``, read when asked (so a replaced table is the one read)."""
        return getattr(importlib.import_module(self.scope), name)

    @property
    def out_of_scope(self) -> Mapping[str, str]:
        """Experiments refused as a whole: ``{name: reason}``."""
        return self._table("OUT_OF_SCOPE")

    @property
    def nitrogen_effect(self) -> Mapping[str, float]:
        """DSSAT's relative yield change with nitrogen on, per treatment (``EXP_tNN``)."""
        return self._table("NITROGEN_STRESS")

    @property
    def nitrogen_dates_differ(self) -> frozenset[str]:
        """Treatments (``EXP_tNN``) whose stage dates change with nitrogen on."""
        return self._table("NITROGEN_DATES_DIFFER")

    def describe(self) -> str:
        """``MZ Maize (CERES-Maize, .MZX)``."""
        return f"{self.code} {self.name} ({self.implementation}, {', '.join(self.filex)})"


#: CERES-Maize of DSSAT-CSM v4.8.6 (``MZCER048``). Cultivar coefficients: ``Genotype/MZCER048.CUL``
#: (``IPVAR.for`` case ``MZCER``); stage codes: the ``ISTAGE`` output, silking 4 and maturity 10
#: (:data:`agrijax.calib.ceres.ISTAGE_SILKING_OUT`, :data:`~agrijax.calib.ceres.ISTAGE_MATURITY_OUT`)
MAIZE = CropModel(
    code="MZ",
    name="Maize",
    model="MZCER048",
    genotype="MZCER048",
    implementation="CERES-Maize",
    filex=(".MZX",),
    examples="Maize",
    cultivar=("P1", "P2", "P5", "G2", "G3", "PHINT"),
    stages=(4, 10),
    scope="agrijax.calib.workflow",
    calibration="agrijax.calib.ceres",
    day="agrijax.models.day_dssat486",
)

#: the implemented crops by crop code, in lookup order
CROPS: dict[str, CropModel] = {MAIZE.code: MAIZE}


def _supported() -> str:
    return "supported crops: " + "; ".join(c.describe() for c in CROPS.values())


def crop(code: str) -> CropModel:
    """The record of the crop code ``code`` (case-insensitive); :class:`UnsupportedCropError` when
    Agri-JAX does not implement it."""
    c = CROPS.get(str(code).strip().upper())
    if c is None:
        raise UnsupportedCropError(f"crop {code!r} is not supported; {_supported()}")
    return c


def crop_of_filex(path: str | os.PathLike[str]) -> CropModel:
    """The record of the crop of a FileX (by its extension ``.<CR>X``); :class:`UnsupportedCropError`
    when the crop is not implemented or the name is not a DSSAT experiment file's."""
    p = Path(path)
    ext = p.suffix.upper()
    for c in CROPS.values():
        if ext in c.filex:
            return c
    if len(ext) == len(".MZX") and ext.endswith("X"):
        raise UnsupportedCropError(f"{p.name}: crop {ext[1:3]} ({ext}) is not supported; {_supported()}")
    raise UnsupportedCropError(
        f"{p.name}: not a DSSAT experiment file (.<crop code>X, e.g. .MZX); {_supported()}"
    )
