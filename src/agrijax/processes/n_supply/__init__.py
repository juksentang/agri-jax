"""Producers of the crop nitrogen port (P10, ``iface.crop_n.<slot>``).

M3 has no nitrogen module: :func:`crop_n_replay` writes the port from a forcing series (the
reference run's daily ``NSTRES``). A nitrogen module will register its own producer for the same
port; the consumer (the ``nstress_replay`` crop growth) does not change.
"""

from .replay import CropNReplayForcing, CropNReplayState, crop_n_replay

__all__ = ["CropNReplayForcing", "CropNReplayState", "crop_n_replay"]
