#!/usr/bin/env python3
"""Regenerate every showcase figure (English and Chinese) into ``docs/showcase/fig``.

    python scripts/showcase/make_all.py --data-dir ~/agri_jax_data

The daily validation figure needs a ``dscsm048`` reference run; it is made here when ``$AGRI_JAX_DSSAT`` points
to a DSSAT engine directory (otherwise pass ``--reference-dir`` or that figure is skipped). The calibration
figure needs ``contourpy`` >= 1.3.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_calibration
import fig_speed
import fig_swap
import fig_validation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir")
    ap.add_argument("--out")
    ap.add_argument(
        "--reference-dir", help="PlantGro.OUT / SoilWat.OUT / DSSAT48.INP of the daily figure's run"
    )
    a = ap.parse_args()
    common = [f"--{k.replace('_', '-')}={v}" for k, v in (("data_dir", a.data_dir), ("out", a.out)) if v]
    val = list(common)
    if a.reference_dir:
        val.append(f"--reference-dir={a.reference_dir}")
    elif os.environ.get("AGRI_JAX_DSSAT"):
        val.append("--run-reference")
    for mod, args in (
        (fig_validation, val),
        (fig_speed, common),
        (fig_calibration, common),
        (fig_swap, common),
    ):
        sys.argv = [mod.__name__, *args]
        mod.main()


if __name__ == "__main__":
    main()
