"""Reader for the RZWQM-DSSAT crop control file (``<CROP>DSSAT.RZX``, e.g. ``MZDSSAT.RZX``).

RZWQM2 reads it once per run for the embedded DSSAT crop (``READRZX``, called from DSSATDRV,
RZWQM2 4.5 DSSATDRV.for:801-809; the file is RZWQM2's own format, not a DSSAT one). Like
``rzwqm.dat`` it is a sequence of comment blocks (lines whose first character is ``=``) and data
blocks; the data blocks, in order, are:

1. ``SLNF SLPF EFINOC EFNFIX`` -- soil nitrification factor, soil fertility factor (DSSAT
   ``SOILPROP%SLPF``), inoculation and N-fixation efficiencies [0..1];
2. the simulation switches ``ISWWAT ISWNIT ISWSYM ISWPHO ISWPOT ISWDIS ISWPAR ISWPSN`` (Y/N, the
   last one C/L);
3. the output control (not interpreted here);
4. the root distribution: ``NLAYRO MAXDEPTH EXPONENT`` on the first line, then one soil root
   growth factor per crop soil layer (DSSAT ``SOILPROP%WR``, the ``.SOL`` column ``SRGF``);
5. the database file locations (not interpreted here).

Validated on CA-TPA against the ``DSSATDRV`` table of the instrumented RZWQM2 4.6 run
(``SOILPROP%WR``, ``SOILPROP%SLPF``; tests/integration/test_io_catpa_dssat.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["RzxControl", "read_rzx"]

_N_FACTORS = 4
_N_SWITCHES = 8
_ROOT_BLOCK = 3  # 0-based data block of the root distribution


@dataclass(frozen=True)
class RzxControl:
    """The parsed data blocks 1, 2 and 4 of a ``.RZX`` file (module docstring)."""

    slnf: float
    slpf: float
    efinoc: float
    efnfix: float
    switches: dict[str, str]
    n_root_layers: int
    max_root_depth_cm: float
    root_exponent: float
    srgf: np.ndarray


_SWITCH_NAMES = ("ISWWAT", "ISWNIT", "ISWSYM", "ISWPHO", "ISWPOT", "ISWDIS", "ISWPAR", "ISWPSN")


def _data_blocks(path: str | Path) -> list[list[list[str]]]:
    lines = Path(path).read_bytes().decode("latin-1").splitlines()
    blocks: list[list[list[str]]] = []
    cur: list[list[str]] = []
    for ln in lines:
        if ln.startswith("=") or not ln.strip():
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append(ln.split())
    if cur:
        blocks.append(cur)
    return blocks


def read_rzx(path: str | Path) -> RzxControl:
    """Parse a ``.RZX`` crop control file (module docstring)."""
    b = _data_blocks(path)
    if len(b) <= _ROOT_BLOCK:
        raise ValueError(f"{path}: expected at least {_ROOT_BLOCK + 1} data blocks, found {len(b)}")
    fac = [float(t) for t in b[0][0][:_N_FACTORS]]
    sw = b[1][0][:_N_SWITCHES]
    if len(fac) != _N_FACTORS or len(sw) != _N_SWITCHES:
        raise ValueError(f"{path}: malformed factor or switch record")
    root = b[_ROOT_BLOCK]
    n = int(float(root[0][0]))
    srgf = np.array([float(r[0]) for r in root[1 : 1 + n]], dtype=float)
    if len(srgf) != n:
        raise ValueError(f"{path}: {n} root layers declared, {len(srgf)} factors given")
    return RzxControl(
        slnf=fac[0],
        slpf=fac[1],
        efinoc=fac[2],
        efnfix=fac[3],
        switches=dict(zip(_SWITCH_NAMES, sw, strict=True)),
        n_root_layers=n,
        max_root_depth_cm=float(root[0][1]),
        root_exponent=float(root[0][2]),
        srgf=srgf,
    )
