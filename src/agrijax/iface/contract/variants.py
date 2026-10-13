"""Registry-key checks of the coupling contract: the registry slot of a key against the directory of
its code (AJ018, :func:`registry_slot_problems`) and the closed variant vocabulary (AJ019,
:data:`VARIANTS`, :func:`variant_problems`).

Part of :mod:`agrijax.iface.contract`; import the names from there.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

#: registry slots whose code lives outside ``processes/<slot>/`` (none: an adapter slot gets its own
#: directory, e.g. ``processes/crop_iface/``, so that AJ008 checks its imports like any slot's)
ADAPTER_SLOTS: dict[str, str] = {}


def registry_slot_problems(entries: Iterable[tuple[str, str]]) -> list[str]:
    """AJ018: ``(key, defining module)`` pairs whose key slot is not the directory of its code
    (``agrijax.processes.<slot>...``, or a plugin's ``<pkg>.processes.<slot>...``) nor an
    :data:`ADAPTER_SLOTS` entry whose module matches."""
    out: list[str] = []
    for key, module in entries:
        slot = key.split("/", 1)[0]
        parts = module.split(".")
        idx = [i for i, p in enumerate(parts) if p == "processes"]
        where = parts[idx[-1] + 1] if idx and idx[-1] + 1 < len(parts) else None
        if where == slot:
            continue
        pref = ADAPTER_SLOTS.get(slot)
        if pref is not None and (module == pref or module.startswith(pref + ".")):
            continue
        out.append(f"AJ018 {key}: slot {slot!r} but defined in {module} (directory {where!r})")
    return out


#: the closed variant vocabulary (the part after ``:`` of a registry key), as regular expressions
VARIANTS: dict[str, str] = {
    "faithful": r"faithful",
    "ref": r"ref_[a-z0-9]+(?:_[a-z0-9]+)*",
    "port": r"port_[a-z0-9]+(?:_[a-z0-9]+)*",
    "alt": r"alt_[a-z0-9]+(?:_[a-z0-9]+)*",
    "replay": r"replay",
    "demo": r"demo",
}
#: the meaning of each variant kind
VARIANT_MEANING: dict[str, str] = {
    "faithful": "reproduces the reference in its default configuration",
    "ref": "reproduces the reference under a non-default switch (ref_<switch>, e.g. ref_istress0, ref_itbl1)",
    "port": "the faithful computation with one input read from a port instead of computed inside "
    "(port_<port>, e.g. port_crop_n)",
    "alt": "an alternative or improved formulation with declared deviations (alt_<name>)",
    "replay": "writes a port from recorded data (ref_version none); the one replay mechanism",
    "demo": "an example, never in an assembly",
}
#: labels in use before the vocabulary, and their names in it
LEGACY_VARIANTS: dict[str, str] = {
    "nstress_replay": "port_crop_n",
    "replay_flux": "alt_supply_as_infiltration",
    "prescribed_canopy": "alt_prescribed_canopy",
    "istress0": "ref_istress0",
    # the RZWQM2 convention variants of the soil-water day: their registry keys keep the old labels
    # until the keys are renamed in one step (the registry then keeps the old keys as aliases); the
    # new names are the ones proposed here
    "drain_cap": "ref_drain_cap",
    "flux_evap": "ref_flux_evap",
    "rzwqm2_conventions": "ref_rzwqm2_conventions",
    "replay_flux_conventions": "alt_supply_as_infiltration_conventions",
}


def variant_kind(variant: str) -> str | None:
    """The :data:`VARIANTS` kind of ``variant``, or ``None``."""
    return next((k for k, rx in VARIANTS.items() if re.fullmatch(rx, variant)), None)


def variant_problems(key: str) -> list[str]:
    """AJ019: the variant of registry key ``key`` against :data:`VARIANTS`: a known kind; ``replay``
    with ``ref_version none``; ``faithful``, ``ref_*`` and ``port_*`` with a reference."""
    head, _, variant = key.rpartition(":")
    ref = head.rpartition("@")[2]
    kind = variant_kind(variant)
    if kind is None:
        hint = f" (rename to {LEGACY_VARIANTS[variant]!r})" if variant in LEGACY_VARIANTS else ""
        return [f"AJ019 {key}: variant {variant!r} is not in the vocabulary {sorted(VARIANTS)}{hint}"]
    if kind == "replay" and ref != "none":
        return [f"AJ019 {key}: a replay has ref_version none"]
    if kind in ("faithful", "ref", "port") and ref == "none":
        return [f"AJ019 {key}: a {kind} variant follows a reference (ref_version is none)"]
    return []
