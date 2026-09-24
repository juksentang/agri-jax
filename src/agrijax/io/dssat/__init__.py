"""DSSAT file formats: .WTH .SOL .CUL .ECO .SPE .MZX and OUT parsing, plus a runner.

Readers parse every ``@``-headed table by its header positions (not by whitespace), so
text columns with blanks (``TNAME``, ``TNAM``) and blank columns keep their alignment.
"""

from .filex import read_filex, read_filex_blocks
from .genotype import SpeciesFile, read_cul, read_eco, read_spe, write_cul, write_eco, write_spe
from .outputs import (
    observed_date,
    read_et,
    read_evaluate,
    read_out,
    read_plantgro,
    read_soilwat,
    read_summary,
)
from .run import DssatRun, run_dssat, stage_run_dir, weather_stations
from .sol import SoilProfile, read_sol, write_sol
from .wth import parse_dssat_date, read_wth, write_wth

__all__ = [
    "DssatRun",
    "SoilProfile",
    "SpeciesFile",
    "observed_date",
    "parse_dssat_date",
    "read_cul",
    "read_eco",
    "read_et",
    "read_evaluate",
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
    "weather_stations",
    "write_cul",
    "write_eco",
    "write_sol",
    "write_spe",
    "write_wth",
]
