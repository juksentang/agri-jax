"""The PET module on the ports of the coupling contract: EOP for the crop, the soil's evaporation demand,
a bound PET entry with its one-day lag in a compiled day (no data)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core.day import Day, DayLagError, Lag, Phase
from agrijax.core.ports import bind, compose
from agrijax.core.state import get_path
from agrijax.iface.contract import PORTS, allowed_lags, day_entry
from agrijax.iface.crop import CanopyRecord, CropWaterIn
from agrijax.iface.surface import EVAPORATION_DEMAND_FIELDS, PETFluxes, evaporation_demand
from agrijax.models.entries import replay_entry
from agrijax.processes.pet import EOPState, PETState, eop_from_pet, pet_priestley_taylor

from .test_pet_process import site, weather

SLOT = "maize"
RTOL = 1e-12 if jax.config.read("jax_enable_x64") else 1e-5
CANOPY, THETA, PET, CROP_WATER = (
    f"iface.canopy.{SLOT}",
    "soil_water.theta",
    "iface.pet",
    f"iface.crop_water.{SLOT}",
)


def _f(x) -> jnp.ndarray:
    return jnp.asarray(x, dtype=float)


def _pet(t: float) -> PETFluxes:
    return PETFluxes.zeros().replace(
        transpiration=_f(t), soil_evaporation=_f(0.2), residue_evaporation=_f(0.05)
    )


# ------------------------------------------------------------------------------------ EOP
def test_eop_is_ten_times_the_pet_transpiration_for_every_crop() -> None:
    water = CropWaterIn.zeros(3, 4)
    out = eop_from_pet(EOPState.module(_pet(0.37), water), None, None)
    np.testing.assert_array_equal(np.asarray(out.crop_water.eop), np.full(3, float(_f(0.37) * 10.0)))
    # only eop changes
    assert out.crop_water.sw is water.sw and out.crop_water.trwup is water.trwup
    assert out.pet is not None and float(out.pet.soil_evaporation) == float(_f(0.2))


def test_eop_keeps_the_record_dtype_and_has_derivative_ten() -> None:
    water32 = CropWaterIn.zeros(1, 2, dtype=jnp.float32)
    out = eop_from_pet(EOPState.module(_pet(0.4), water32), None, None)
    assert out.crop_water.eop.dtype == jnp.float32
    g = jax.grad(
        lambda t: eop_from_pet(EOPState.module(_pet(t), CropWaterIn.zeros(1, 2)), None, None).crop_water.eop[
            0
        ]
    )(_f(0.3))
    assert float(g) == 10.0


def test_eop_entry_is_in_the_contract_day() -> None:
    e = day_entry(f"crops.{SLOT}.eop", SLOT)
    assert e.row == "9" and "P1" in e.ports_out
    assert eop_from_pet.reads == ("pet.transpiration",) and eop_from_pet.writes == ("crop_water.eop",)


# ------------------------------------------------------------------------------------ evaporation demand
def test_evaporation_demand_is_soil_plus_residue_evaporation() -> None:
    p = PETFluxes.zeros().replace(
        transpiration=_f(0.4), soil_evaporation=_f(0.21), residue_evaporation=_f(0.07)
    )
    assert float(evaporation_demand(p)) == float(_f(0.21) + _f(0.07))
    assert EVAPORATION_DEMAND_FIELDS == ("soil_evaporation", "residue_evaporation")
    assert "soil_water.day" in PORTS["P5"].consumers


# ------------------------------------------------------------------------------------ the bound PET entry
def _day() -> Day:
    return Day(
        ref="rzwqm2-4.6",
        bare=True,  # a fixture of part of the day
        phases=(
            Phase("physcl", ("pet.sw_daily", "soil_water.day")),
            Phase("plant", (f"crops.{SLOT}.eop", f"crops.{SLOT}.canopy")),
        ),
        lags=allowed_lags(SLOT, ("P6", "P7")),  # the lags of the ports this part of the day reads
    )


def _procs() -> dict:
    return {
        "pet.sw_daily": bind(
            pet_priestley_taylor,
            own="surface.pet",
            ports={"canopy": CANOPY, "theta": THETA, "pet": PET},
            params="pet",
            forcing="weather",
        ),
        "soil_water.day": replay_entry("soil_water.day", {THETA: "replay.theta"}),
        f"crops.{SLOT}.eop": bind(
            eop_from_pet, own="surface.crop_iface", ports={"pet": PET, "crop_water": CROP_WATER}
        ),
        f"crops.{SLOT}.canopy": replay_entry(f"crops.{SLOT}.canopy", {CANOPY: "replay.canopy"}),
    }


def _canopy(lai: float, height: float) -> CanopyRecord:
    return CanopyRecord(lai=_f([lai]), tlai=_f([lai + 0.1]), height=_f([height]))


def test_bound_pet_reads_yesterdays_canopy() -> None:
    model = _day().compile(_procs())
    assert sorted(_day().lagged_reads(model)) == [("pet.sw_daily", f"{CANOPY}.lai")]
    step = jax.jit(model.compile())
    canopies = [_canopy(1.0, 60.0), _canopy(2.0, 120.0), _canopy(3.0, 180.0)]
    thetas = [_f([0.18, 0.2]), _f([0.25, 0.26]), _f([0.3, 0.3])]
    s = compose(
        {
            "surface.pet": PETState(),
            "surface.crop_iface": EOPState(),
            CANOPY: canopies[0],
            THETA: thetas[0],
            PET: PETFluxes.zeros(),
            CROP_WATER: CropWaterIn.zeros(1, 2),
        }
    )
    p, w = site(), weather()
    for d in range(2):
        # the producers write, at the end of day d, the record PET reads on day d + 1
        f = {"weather": w, "replay": {"canopy": canopies[d + 1], "theta": thetas[d + 1]}}
        s, _ = step(s, {"pet": p}, f)
        alone = pet_priestley_taylor(PETState.module(canopies[d], thetas[d]), p, w).pet
        got = get_path(s, PET)
        # the jitted day against the eager process: the same inputs, rounding apart
        np.testing.assert_allclose(
            float(got.eo_priestley_taylor), float(alone.eo_priestley_taylor), rtol=RTOL
        )
        # a different day's inputs would differ by far more than rounding
        other = pet_priestley_taylor(PETState.module(canopies[d + 1], thetas[d + 1]), p, w).pet
        assert abs(float(other.eo_priestley_taylor) - float(got.eo_priestley_taylor)) > 1e3 * RTOL * float(
            got.eo_priestley_taylor
        )
        assert float(get_path(s, CROP_WATER).eop[0]) == float(got.transpiration * 10.0)
    assert float(get_path(s, CANOPY).lai[0]) == 3.0  # the last producer write stays for tomorrow


def test_a_day_without_the_contract_lags_is_rejected() -> None:
    """The PET entry reads P6 before its producer: without the allowed lag the day fails."""
    bad = Day(
        ref="rzwqm2-4.6",
        bare=True,  # a fixture of part of the day
        phases=_day().phases,
        lags=(Lag("pet.sw_daily", THETA, evidence="test: a lag the entry does not need, not P6's"),),
    )
    with pytest.raises(DayLagError):
        bad.compile(_procs())
