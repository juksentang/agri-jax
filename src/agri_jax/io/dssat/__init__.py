"""DSSAT file formats: .WTH .SOL .CUL .ECO .SPE .MZX and OUT parsing, plus a runner.

Readers parse every ``@``-headed table by its header positions (not by whitespace), so
text columns with blanks (``TNAME``, ``TNAM``) and blank columns keep their alignment.
"""

from .filex import read_filex, read_filex_blocks
from .genotype import SpeciesFile, read_cul, read_eco, read_spe
from .outputs import read_et, read_out, read_plantgro, read_soilwat, read_summary
from .run import DssatRun, run_dssat, stage_run_dir
from .sol import SoilProfile, read_sol, write_sol
from .wth import parse_dssat_date, read_wth, write_wth

__all__ = [
    "DssatRun",
    "SoilProfile",
    "SpeciesFile",
    "parse_dssat_date",
    "read_cul",
    "read_eco",
    "read_et",
    "read_filex",
    "read_filex_blocks",
    "read_out",
    "read_plantgro",
    "read_soilwat",
    "read_sol",
    "read_spe",
    "read_summary",
    "read_wth",
    "run_dssat",
    "stage_run_dir",
    "write_sol",
    "write_wth",
]
