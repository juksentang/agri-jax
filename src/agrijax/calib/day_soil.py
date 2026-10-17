"""Soil parameters written into the free-run DSSAT-CSM v4.8.6.0 day (:mod:`agrijax.models.day_dssat486`).

The day holds the soil of a run in several places, all built once from the ``.SOL`` by the native
input chain (:mod:`agrijax.io.dssat.native_soil`, :mod:`agrijax.io.dssat.native_crop`) or the
instrumented-engine tables: the bucket's ``soil``, the crop's ``soil`` (``LL``, ``DUL``, ``SAT``,
``SHF`` = the root growth factor ``SRGF``, ``SLPF``), ROOTWU's ``LL`` and ``SAT``, XTRACT's ``LL``,
the albedo's top-layer ``DUL``, the soil evaporation's ``U``, the daily soil of the bucket forcing,
and in the morning state the initial soil water with what depends on it (the ``SOILEV`` store, the
crop's water record P1, the ledger's starting storage). Changing a soil parameter in one copy only
runs a model that DSSAT never runs. :func:`with_soil` writes a soil into all of them in one pure
JAX function, so a calibration can differentiate through it and ``vmap`` it.

**By soil-file layer.** The layered values are given per layer of the ``.SOL`` profile, the
quantity a soil file holds: DSSAT splits a thick file layer into two to four model layers
(``LYRSET2``) and averages every value onto them (``LMATCH``). :class:`SoilLayerMap` carries the
matching weights of that one map (:func:`agrijax.io.dssat.native_soil.layer_weights`, computed by
``lmatch`` itself), and :func:`with_soil` expands the file values onto the model layers by a
gather with those weights, no layering rule of its own. A model layer cut from one file layer gets
that layer's value exactly; one that overlaps several (a file layer thinner than 2 cm) gets their
depth-weighted mean, unrounded. DSSAT prints that mean to ``DSSAT48.INP`` (``F5.3`` for ``LL``,
``DUL``, ``SAT``, ``SRGF``) and reads the printed value back, so in such a layer :func:`with_soil`
and DSSAT differ by up to half a print step (5e-4), everywhere the value goes. The unrounded mean
is kept on purpose (it is continuous in the file values, the printed one is a staircase), but it is
a deviation from DSSAT: :func:`sol_layer_map` refuses a map with a shared model layer unless the
caller accepts it (``shared_layers=True``, recorded in :attr:`SoilLayerMap.shared_layers`).

**The initial soil water.** ``SOILDYN`` limits the initial water of the FileX to ``[LL, SAT]``
when the run starts (``Soil/SoilUtilities/SOILDYN.for`` 369-420: a layer below ``LL`` is raised to
``LL``, the top layer only to its air-dry content ``0.30 LL`` when below that; then a layer above
``SAT`` is lowered to ``SAT``), and a layer without an initial value starts at ``DUL``
(``InputModule/INSOIL.for`` 84-86). With another soil the limits move, so :func:`with_soil` starts
again from the FileX value before the limits (:attr:`SoilLayerMap.sw_input`, the new ``DUL`` where
there is none) and applies them with the new ``LL`` and ``SAT``, in REAL*4 under the ``real4``
soil-value convention as DSSAT does.

**Precision.** DSSAT reads its soil from ``DSSAT48.INP``, where the input module printed it
(``CN`` ``F5.0``, ``U`` ``F5.1``, ``SWCON`` and ``SLPF`` ``F5.2``, ``LL``, ``DUL``, ``SAT``, ``SRGF``
``F5.3``, ``SSKS`` by magnitude), and the input module repairs a few inputs (``SAT = DUL`` and
``LL = DUL`` moved 0.01 apart, defaults for a missing ``CN``, ``U``, ``SWCON``, ``SALB``, ``CN``
clamped to [25, 98]). :func:`with_soil` uses the values as given (a calibration moves them
continuously) and repeats none of the repairs: the parameter spaces of
:mod:`agrijax.calib.soil` keep the values inside the domains where no repair applies. For values
at the printed precision and a map whose weights are all 0 or 1 (every model layer cut from one
file layer, :attr:`SoilLayerMap.exact`), ``with_soil(inputs, soil)`` equals the trees the native
builder makes from a ``.SOL`` holding ``soil``, bit for bit
(``tests/integration/test_day_soil_dssat.py``); with a shared model layer it differs there by up
to half a print step (see above, pinned in ``tests/unit/test_calib_day_soil.py``).

**Gradients.** :func:`with_soil` is linear in the soil values apart from the limits on the initial
water and the ``SOILEV`` initialisation, so its own derivatives are exact. Derivatives of a whole
faithful season with respect to soil values are another matter: the faithful day has thresholds
(``SATFLO``'s drainage start at ``DUL + 0.003``, the runoff curve, stage changes) and keeps its soil
values in REAL*4, so a finite difference jumps whenever one layer-day crosses a threshold and is
zero below a REAL*4 quantum. Which outputs agree with finite differences depends on the site and
the run, not on the output alone; no aggregate is safe everywhere:

* On UFGA8201 treatment 2 (native inputs, 90 days) automatic and central finite differences agreed
  to a few percent for the season's summed layer water against ``DUL``, ``LL`` and ``SRGF``, while
  the canopy weight at the end against the top layer's ``DUL``, ``SAT`` or ``U``, and every output
  against ``CN``, disagreed in sign or size whatever the step. On the first 60 days of the same run
  the end-of-season storage against ``DUL`` of ``.SOL`` layer 3 disagreed too: automatic -0.39,
  central finite -1.62, -6.98 and +0.80 for steps 1e-3, 3e-4 and 1e-4. The summed layer water
  against the same ``DUL`` agreed (automatic 42.47, finite 42.36, 41.83, 42.44), as did it
  against ``SWCON`` (automatic -15.03, finite -14.6 to -15.1), ``LL`` of layer 2 (12.42; 12.2 to
  12.9) and ``SRGF`` of layer 4 (-0.36; -0.31 to -0.39).
* On the synthetic sand of ``tests/unit/test_calib_day_soil.py`` it is roughly the reverse: the
  summed soil evaporation and the end-of-season storage against a scale of all ``DUL`` and against
  ``SWCON`` agree within 1 % (pinned there), but the summed layer water against ``DUL`` does not
  (automatic 37.8; finite 41.5, 78.2, 39.0 for relative steps 1e-3, 1e-4, 1e-5), nor do several
  aggregates against ``LL``, ``SAT`` and ``SRGF``.

Every derivative through :func:`with_soil` and the season was finite for every soil quantity. A
gradient-based soil calibration on the faithful day should therefore check its own objective, on
its own site, against finite differences first, or use the smoothed comparator variant.

Runs whose soil changes during the season (the ``SOILDYN`` replay of the tables: organic matter,
tillage) take their daily soil from the forcing and are refused (:class:`SoilDynamicsError`).

Reference: DSSAT-CSM v4.8.6.0 (BSD-3); file, line and routine are cited, the source is not
reproduced.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from agrijax.calib.soil import LAYER_QUANTITIES, SCALAR_QUANTITIES, SOIL_QUANTITIES

__all__ = [
    "FORCING_SOIL_VALUES",
    "SoilDynamicsError",
    "SoilLayerMap",
    "clip_initial_water",
    "soil_layer_map",
    "sol_layer_map",
    "with_soil",
    "write_soil",
    "write_soil_forcing",
]

SOILDYN_FOR = "Soil/SoilUtilities/SOILDYN.for"


class SoilDynamicsError(ValueError):
    """Soil parameters requested for a run whose soil changes during the season (``SOILDYN``)."""


@dataclass(frozen=True, eq=False)
class SoilLayerMap:
    """How a run's ``.SOL`` profile maps onto its model layers (host-side, static).

    ``weight[i, j]`` is the share of soil-file layer ``j`` in model layer ``i``
    (:func:`agrijax.io.dssat.native_soil.layer_weights`); ``index`` / ``coef`` ``[nl, k]`` are its
    nonzero entries (padded with index 0 and weight 0), the gather :meth:`expand` does. ``sol`` holds
    the site's soil-file values (:data:`agrijax.calib.soil.SOIL_QUANTITIES`; layered ones per file
    layer, ``SSKS`` NaN where missing). ``sw_input`` is the initial soil water before ``SOILDYN``
    limits it, NaN in a layer where DSSAT takes ``DUL``
    (:func:`agrijax.io.dssat.native_soil.initial_soil_water_input`). ``soildyn``: the run's soil
    changes during the season. ``shared_layers``: the map has a model layer that averages several
    file layers (a weight strictly between 0 and 1), a deviation the caller accepted (module
    docstring).

    Hashed and compared by identity (its fields are arrays), so it can be a static argument of
    ``jax.jit`` (``static_argnames=("layers",)``) or a cache key."""

    profile: str
    weight: np.ndarray
    index: np.ndarray
    coef: np.ndarray
    sol: Mapping[str, np.ndarray]
    sw_input: np.ndarray
    soildyn: bool = False
    shared_layers: bool = False

    @property
    def exact(self) -> bool:
        """Every model layer is cut from one file layer (all weights 0 or 1): :func:`with_soil` then
        gives each model layer a file value exactly, as DSSAT does."""
        return bool(np.all((self.weight == 0.0) | (self.weight == 1.0)))

    @property
    def n_sol(self) -> int:
        """Layers of the soil file."""
        return int(self.weight.shape[1])

    @property
    def nl(self) -> int:
        """Model layers."""
        return int(self.weight.shape[0])

    @property
    def sol_layer(self) -> np.ndarray:
        """The soil-file layer of each model layer (the largest share)."""
        return np.argmax(self.weight, axis=1)

    def expand(self, v: Any) -> Any:
        """Soil-file layer values ``v [..., n_sol]`` on the model layers ``[..., nl]`` (a gather with
        the matching weights; exact for a model layer cut from one file layer, the unrounded
        depth-weighted mean for a shared one, where DSSAT reads the mean printed ``F5.3``)."""
        x = jnp.asarray(v)
        if x.shape[-1] != self.n_sol:
            raise ValueError(f"{self.profile}: {x.shape[-1]} layer values for {self.n_sol} .SOL layers")
        return jnp.sum(x[..., self.index] * jnp.asarray(self.coef, x.dtype), axis=-1)

    def values(self) -> dict[str, Any]:
        """The site's soil-file values as JAX arrays (the tree :mod:`agrijax.calib.soil` spaces
        built with ``prefix=None`` write into, and :func:`with_soil` reads); ``swcn`` only when the
        file gives ``SSKS`` in every layer (:func:`with_soil` refuses it otherwise)."""
        out = {k: jnp.asarray(np.asarray(v, dtype=np.float64)) for k, v in self.sol.items()}
        if not np.all(np.isfinite(np.asarray(self.sol["swcn"], dtype=float))):
            del out["swcn"]
        return out


def sol_layer_map(
    profile: Any,
    mesol: str = "2",
    ic_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    soildyn: bool,
    shared_layers: bool = False,
) -> SoilLayerMap:
    """Host-side: the :class:`SoilLayerMap` of ``profile`` (:func:`agrijax.io.dssat.read_sol` with
    ``dssat_spans=True``) under the layer method ``mesol``, the FileX initial-condition layers
    ``ic_rows`` (``None``: none) and the run's ``soildyn`` flag (required, and it must be the run's:
    :attr:`~agrijax.sites.dssat_free_run.FreeRunInputs.soildyn`; :func:`soil_layer_map` takes it
    from the run).

    Raises ``ValueError`` when a model layer averages several file layers (a file layer thinner than
    2 cm) unless ``shared_layers`` accepts the deviation that brings (module docstring)."""
    from agrijax.io.dssat.native_soil import initial_soil_water_input, layer_weights, native_soil

    soil = native_soil(profile, mesol)
    w = layer_weights(soil.zlayr, np.asarray(soil.ds, dtype=np.float32))
    k = max(1, int(np.max(np.sum(w != 0.0, axis=1))))
    order = np.argsort((w == 0.0).astype(np.int8), axis=1, kind="stable")[:, :k]
    coef = np.take_along_axis(w, order, axis=1)
    index = np.where(coef != 0.0, order, 0)
    shared = np.nonzero(np.any((w != 0.0) & (w != 1.0), axis=1))[0]
    if shared.size and not shared_layers:
        raise ValueError(
            f"sol_layer_map: model layers {shared.tolist()} of {profile.id} average several .SOL layers "
            "(a file layer thinner than 2 cm); DSSAT reads their mean printed F5.3, with_soil keeps it "
            "unrounded (up to 5e-4 apart); pass shared_layers=True to accept that deviation"
        )
    n = profile.n_layers

    def col(name: str) -> np.ndarray:
        if name not in profile.layers:
            return np.full(n, np.nan)
        return np.asarray(profile.layers[name], dtype=np.float64)[:n]

    def sca(name: str) -> np.ndarray:
        v = profile.surface.get(name)
        return np.asarray(float(v) if isinstance(v, (int, float)) else np.nan)

    swcn = col("SSKS")
    sol = {
        "cn": sca("SLRO"),
        "swcon": sca("SLDR"),
        "u": sca("SLU1"),
        "slpf": sca("SLPF"),
        "ll": col("SLLL"),
        "dul": col("SDUL"),
        "sat": col("SSAT"),
        "srgf": col("SRGF"),
        "swcn": np.where(swcn > 0, swcn, np.nan),
    }
    return SoilLayerMap(
        profile=str(profile.id),
        weight=w,
        index=index.astype(np.int32),
        coef=coef,
        sol=sol,
        sw_input=initial_soil_water_input(soil, ic_rows),
        soildyn=bool(soildyn),
        shared_layers=bool(shared.size),
    )


def soil_layer_map(
    inputs: Any,
    filex: str | os.PathLike[str],
    soil_dirs: Sequence[str | os.PathLike[str]] | None = None,
    *,
    engine: str | os.PathLike[str] | None = None,
    shared_layers: bool = False,
) -> SoilLayerMap:
    """Host-side: the :class:`SoilLayerMap` of the run ``inputs``
    (:class:`agrijax.sites.dssat_free_run.FreeRunInputs`) of treatment ``inputs.trno`` of ``filex``:
    its ``.SOL`` profile found in ``soil_dirs`` (default: the FileX directory, then the engine's
    ``example_data/Soil``, as :func:`~agrijax.sites.dssat_free_run.free_run_inputs` does) and the
    FileX initial conditions. Checks that the profile rebuilds the run's static soil (model layers,
    ``LL``, ``DUL``, ``SAT``) and the crop's ``SHF`` and ``SLPF``; ``shared_layers`` as for
    :func:`sol_layer_map`."""
    from agrijax.io.dssat import read_filex
    from agrijax.io.dssat.native_crop import native_soil_crop
    from agrijax.io.dssat.native_management import switches, treatment_levels
    from agrijax.io.dssat.native_soil import find_soil_profile, native_soil
    from agrijax.port.run_fortran import DSSAT_ENGINE

    fx = Path(filex)
    eng = Path(engine) if engine is not None else DSSAT_ENGINE
    dirs = (
        [Path(d) for d in soil_dirs] if soil_dirs is not None else [fx.parent, eng / "example_data" / "Soil"]
    )
    x = read_filex(fx)
    trno = int(inputs.trno)
    tr = treatment_levels(x, trno)
    prof = find_soil_profile(str(x["FIELDS"][int(tr["FL"])]["ID_SOIL"]).strip(), dirs)
    mesol = switches(x, trno).mesol
    ic_lev = int(tr.get("IC", 0) or 0)
    ic = x.get("INITIAL CONDITIONS", {}).get(ic_lev, {}) if ic_lev else {}
    m = sol_layer_map(prof, mesol, ic.get("rows"), soildyn=bool(inputs.soildyn), shared_layers=shared_layers)
    soil = native_soil(prof, mesol)
    crop = native_soil_crop(prof, soil)
    site, cs = inputs.site, inputs.params_crop.soil
    pairs = [(name, getattr(soil, name), getattr(site, name)) for name in ("ds", "ll", "dul", "sat")]
    pairs += [("SHF", crop["shf"], cs.shf), ("SLPF", crop["slpf"], cs.slpf)]
    for name, a, b in pairs:
        ours, theirs = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if ours.shape != theirs.shape or not np.array_equal(ours, theirs):
            raise ValueError(
                f"{inputs.key}: the .SOL {prof.id} gives {name} {ours.tolist()}, the run {theirs.tolist()}"
            )
    return m


# ------------------------------------------------------------------------ the initial soil water
def clip_initial_water(sw_in: Any, ll: Any, sat: Any, *, real4: bool = True) -> Any:
    """``SOILDYN``'s limits on the initial soil water (``SOILDYN.for`` 369-420): a layer below
    ``LL`` is raised to ``LL``, the top layer only to the air-dry ``0.30 LL`` when below that; then a
    layer above ``SAT`` is lowered to ``SAT``. ``real4``: in REAL*4 as DSSAT (the inputs rounded to
    ``float32``, the result promoted back), else in the inputs' precision. Pure JAX, last axis the
    layers."""
    from agrijax.io.dssat.native_soil import DSSAT_SOIL_INPUT

    out_dtype = jnp.result_type(jnp.asarray(sw_in), jnp.asarray(ll), jnp.asarray(sat))
    dt = jnp.float32 if real4 else out_dtype
    sw, lo, hi = (jnp.asarray(a).astype(dt) for a in (sw_in, ll, sat))
    swad = jnp.asarray(DSSAT_SOIL_INPUT.air_dry_fraction, dt) * lo
    top = jnp.arange(sw.shape[-1]) == 0
    raised = jnp.where(top, jnp.maximum(sw, swad), lo)
    sw = jnp.where(sw < lo, raised, sw)
    sw = jnp.where(sw > hi, hi, sw)
    return sw.astype(out_dtype)


# ------------------------------------------------------------------------ writing a soil
#: the bucket forcing's daily soil leaves (the same value every day when the run's soil is static)
_FORCING_SOIL = ("dlayr", "ds", "ll", "dul", "sat", "swcn", "cn", "swcon")


def _check_static_forcing(forcing: Mapping[str, Any], layers: SoilLayerMap) -> None:
    """Host-side: refuse a forcing whose daily soil changes over the days (a ``SOILDYN`` replay)
    although the map says the soil is static; traced leaves are not checked."""
    import jax

    bf = forcing["soil"]
    leaves = [(f"soil.{k}", getattr(bf.soil, k, None)) for k in _FORCING_SOIL] if bf.soil is not None else []
    leaves.append(("dlayr_end", getattr(bf, "dlayr_end", None)))
    for name, leaf in leaves:
        if leaf is None:
            continue
        try:
            a = np.asarray(leaf)
        except jax.errors.TracerArrayConversionError:
            continue
        if a.ndim and not np.array_equal(a, np.broadcast_to(a[:1], a.shape), equal_nan=True):
            raise SoilDynamicsError(
                f"with_soil: the forcing's daily {name} changes during this run of {layers.profile} "
                "(SOILDYN: organic matter, tillage) although the layer map says its soil is static; "
                "soil parameters cannot be written"
            )


def _check(soil: Mapping[str, Any], layers: SoilLayerMap) -> None:
    unknown = sorted(set(soil) - set(SOIL_QUANTITIES))
    if unknown:
        raise ValueError(f"with_soil: unknown soil quantities {unknown} (known: {list(SOIL_QUANTITIES)})")
    if layers.soildyn and soil:
        raise SoilDynamicsError(
            f"with_soil: the soil of {layers.profile} changes during this run (SOILDYN: organic matter, "
            "tillage); its daily soil comes from the forcing, so soil parameters cannot be written"
        )
    if "swcn" in soil:
        miss = np.nonzero(~np.isfinite(np.asarray(layers.sol["swcn"], dtype=float)))[0]
        if miss.size:
            raise ValueError(
                f"with_soil: SSKS of {layers.profile} is missing in .SOL layers {miss.tolist()} (no "
                "conductivity cap there); a value would switch the cap on, a change of structure"
            )
    for q in LAYER_QUANTITIES:
        if q in soil and jnp.shape(soil[q])[-1:] != (layers.n_sol,):
            raise ValueError(
                f"with_soil: {q} needs one value per .SOL layer ({layers.n_sol}), "
                f"got shape {jnp.shape(soil[q])}"
            )
    for q in SCALAR_QUANTITIES:
        if q in soil and jnp.shape(soil[q]) != ():
            raise ValueError(f"with_soil: {q} is one value, got shape {jnp.shape(soil[q])}")


def with_soil(
    params: Mapping[str, Any],
    forcing: Mapping[str, Any],
    state0: Any,
    soil: Mapping[str, Any],
    *,
    layers: SoilLayerMap,
    slot: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """``(params, forcing, state0)`` of the free-run day with the soil ``soil`` written everywhere
    the day holds it (module docstring). Pure JAX: ``soil``'s values may be traced.

    ``params``, ``forcing``, ``state0`` are the day's trees of one run
    (:meth:`~agrijax.sites.dssat_free_run.FreeRunInputs.params`, ``.forcing``, ``.state``; ``state0``
    the season's morning state). ``soil`` maps quantities of :data:`agrijax.calib.soil.SOIL_QUANTITIES`
    to values: ``cn``, ``swcon``, ``u``, ``slpf`` one each, ``ll``, ``dul``, ``sat``, ``srgf``, ``swcn``
    one per ``.SOL`` layer (``layers``, :class:`SoilLayerMap`); a quantity left out keeps its value.
    Under the ``real4`` soil-value convention (``params["soil"].real4_sw``) the soil copies get the
    REAL*4 value of each number, as :func:`agrijax.models.day_dssat486.day_params` gives them; ``SHF``
    and ``SLPF`` stay as given (the crop reads them as printed decimals). The initial soil water is
    limited again with the new ``LL`` and ``SAT`` (:func:`clip_initial_water`), and with it the
    ``SOILEV`` store, the crop's water record and the ledger's starting storage are rebuilt.

    Raises :class:`SoilDynamicsError` for a run whose soil changes during the season (the map's flag,
    or a concrete forcing whose daily soil is not the same every day), ``ValueError``
    for unknown quantities, wrong shapes, or ``swcn`` on a profile with missing ``SSKS``."""
    _check(soil, layers)
    if soil:
        _check_static_forcing(forcing, layers)
    none = np.isnan(np.asarray(layers.sw_input, dtype=float))  # no initial value: DSSAT takes DUL
    p, s, fsoil = write_soil(
        params,
        state0,
        soil,
        expand=layers.expand,
        sw_input=np.where(none, 0.0, layers.sw_input),
        sw_missing=none,
        slot=slot,
    )
    return p, write_soil_forcing(forcing, fsoil), s


#: the bucket forcing's soil values :func:`write_soil` returns for :func:`write_soil_forcing`
FORCING_SOIL_VALUES = ("ll", "dul", "sat", "swcn", "cn", "swcon")


def write_soil(
    params: Mapping[str, Any],
    state0: Any,
    soil: Mapping[str, Any],
    *,
    expand: Callable[[Any], Any],
    sw_input: Any,
    sw_missing: Any,
    slot: str | None = None,
    keep: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """The core of :func:`with_soil` without the host-side checks: ``(params, state0, forcing soil)``.

    ``expand`` maps soil-file layer values ``[..., n_sol]`` onto the model layers (e.g.
    :meth:`SoilLayerMap.expand`, or the same gather with traced indices); ``sw_input`` is the initial
    water before ``SOILDYN``'s limits and ``sw_missing`` where the FileX gives none (``DUL`` is taken
    there); both may be traced. The third value maps :data:`FORCING_SOIL_VALUES` to the new model-layer
    values the bucket forcing must carry (:func:`write_soil_forcing`).

    ``keep``: ``{quantity: bool}`` (traced allowed); where true, that quantity keeps its current value
    in every copy although it is in ``soil``, and when ``LL``, ``DUL``, ``SAT`` and ``U`` all keep
    theirs, the morning state is returned unchanged. A batched calibration uses it for runs that
    share a program with runs calibrating a quantity they do not calibrate themselves. With
    ``keep=None`` (:func:`with_soil`) every quantity in ``soil`` is written."""
    import jax

    from agrijax.core.state import get_path, set_path
    from agrijax.core.units import MM_PER_CM
    from agrijax.models.day_dssat486 import SLOT, ledger_initial
    from agrijax.processes.soil_water.bucket_evap.ritchie import soilev_init

    slot = SLOT if slot is None else slot
    bucket_p = params["soil"]
    real4 = bool(bucket_p.real4_sw)
    old = bucket_p.soil
    dt = jnp.asarray(old.ll).dtype

    def conv(x: Any) -> Any:
        x = jnp.asarray(x, dt)
        return x.astype(jnp.float32).astype(dt) if real4 else x

    def select(q: str, current: Any, new: Any) -> Any:
        if keep is None or q not in keep:
            return new
        return jnp.where(jnp.asarray(keep[q]), current, new)

    def layered(q: str, current: Any) -> Any:
        return current if q not in soil else select(q, current, conv(expand(jnp.asarray(soil[q], dt))))

    def scalar(q: str, current: Any) -> Any:
        return current if q not in soil else select(q, current, conv(soil[q]))

    ll, dul, sat = layered("ll", old.ll), layered("dul", old.dul), layered("sat", old.sat)
    swcn = layered("swcn", old.swcn)
    cn, swcon = scalar("cn", old.cn), scalar("swcon", old.swcon)
    new_soil = old.replace(ll=ll, dul=dul, sat=sat, swcn=swcn, cn=cn, swcon=swcon)

    p = dict(params)
    p["soil"] = bucket_p.replace(soil=new_soil)
    crop = p["crop"]
    cs = crop.soil
    shf_dt, slpf_dt = jnp.asarray(cs.shf).dtype, jnp.asarray(cs.slpf).dtype
    shf = cs.shf if "srgf" not in soil else select("srgf", cs.shf, expand(jnp.asarray(soil["srgf"], shf_dt)))
    slpf = cs.slpf if "slpf" not in soil else select("slpf", cs.slpf, jnp.asarray(soil["slpf"], slpf_dt))
    crop_water = (
        (ll, dul, sat)
        if real4
        else tuple(
            cur if q not in soil else select(q, cur, expand(jnp.asarray(soil[q], dt)))
            for q, cur in (("ll", cs.ll), ("dul", cs.dul), ("sat", cs.sat))
        )
    )
    p["crop"] = crop.replace(
        soil=cs.replace(ll=crop_water[0], dul=crop_water[1], sat=crop_water[2], shf=shf, slpf=slpf)
    )
    p["rootwu"] = p["rootwu"].replace(ll=ll, sat=sat)
    p["xtract"] = p["xtract"].replace(ll=ll)
    p["albedo"] = p["albedo"].replace(dul1=dul[..., 0])
    p["evap"] = p["evap"].replace(u=scalar("u", p["evap"].u))
    fsoil = {"ll": ll, "dul": dul, "sat": sat, "swcn": swcn, "cn": cn, "swcon": swcon}

    # the morning state: initial water limited with the new soil, and what is built from it
    b = get_path(state0, "soil_water")
    given = jnp.asarray(sw_input, dt)
    raw = jnp.where(jnp.asarray(sw_missing), dul, given)
    sw0 = clip_initial_water(raw, ll, sat, real4=real4).astype(jnp.asarray(b.sw).dtype)
    theta = sw0 if jnp.shape(b.theta) == jnp.shape(b.sw) else b.theta
    s = set_path(state0, "soil_water", b.replace(sw=sw0, theta=theta))
    se_path = "surface.soil_evap"
    se = get_path(s, se_path)
    u = p["evap"].u
    st = soilev_init(sw0[..., 0], ll[..., 0], dul[..., 0], new_soil.dlayr[..., 0], u)
    sdt = jnp.asarray(se.sw).dtype
    s = set_path(
        s,
        se_path,
        se.replace(
            sw=sw0,
            dul=dul,
            ll=ll,
            sumes1=jnp.asarray(st.sumes1, sdt),
            sumes2=jnp.asarray(st.sumes2, sdt),
            t=jnp.asarray(st.t, sdt),
            swef=jnp.asarray(st.swef, sdt),
        ),
    )
    cw_path = f"iface.crop_water.{slot}"
    cw = get_path(s, cw_path)
    s = set_path(
        s, cw_path, cw.replace(sw=jnp.broadcast_to(sw0, jnp.shape(cw.sw)).astype(jnp.asarray(cw.sw).dtype))
    )
    storage0 = jnp.sum(sw0 * new_soil.dlayr) + (b.snow + b.mulch_wat) / MM_PER_CM
    s = set_path(s, "ledger.water", ledger_initial(storage0))
    if keep is not None:
        same = [jnp.asarray(keep[q]) for q in ("ll", "dul", "sat", "u") if q in soil and q in keep]
        if len(same) == len([q for q in ("ll", "dul", "sat", "u") if q in soil]):
            unchanged = jnp.all(jnp.stack(same)) if same else jnp.asarray(True)
            s = jax.tree.map(lambda a, n: jnp.where(unchanged, a, n).astype(jnp.asarray(a).dtype), state0, s)
    return p, s, fsoil


def write_soil_forcing(forcing: Mapping[str, Any], fsoil: Mapping[str, Any]) -> dict[str, Any]:
    """``forcing`` with the bucket forcing's daily soil set to the values of :func:`write_soil` on
    every day (pure JAX; a forcing without a daily soil is returned as it is)."""
    f = dict(forcing)
    bf = f["soil"]
    if bf.soil is not None:
        fs = bf.soil
        tile = lambda new, cur: jnp.broadcast_to(new, jnp.shape(cur)).astype(jnp.asarray(cur).dtype)  # noqa: E731
        new = {k: tile(fsoil[k], getattr(fs, k)) for k in FORCING_SOIL_VALUES}
        f["soil"] = bf.replace(soil=fs.replace(**new))
    return f
