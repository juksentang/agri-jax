"""Fortran-to-JAX porting toolchain: fparser index, dump instrumentation, oracle runs, comparison reports.

Submodules: :mod:`~agrijax.port.fortran_index` (index), :mod:`~agrijax.port.instrument`
(entry/exit dump instrumentation), :mod:`~agrijax.port.run_fortran` (reference runs),
:mod:`~agrijax.port.dumps` (dump streams -> ``npz`` cases), :mod:`~agrijax.port.compare`.
"""

from agrijax.port.run_fortran import (
    DscsmResult,
    FortranRunError,
    RzwqmResult,
    check_dscsm_outputs,
    parse_overview_yields,
    run_dscsm,
    run_rzwqm,
)

__all__ = [
    "DscsmResult",
    "FortranRunError",
    "RzwqmResult",
    "check_dscsm_outputs",
    "parse_overview_yields",
    "run_dscsm",
    "run_rzwqm",
]
