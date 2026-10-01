"""Reader of the RZWQM2 ``.sno`` file: the PRMS snow parameters of a scenario.

The file is what ``READ_SNOW`` (``RZWQM/Snowprms.for`` 1024-1127) reads: five list-directed
records separated by ``=`` comment blocks (general pack dynamics; albedo resets; cover;
atmosphere, with the monthly convection coefficients and thunderstorm flags on two more lines;
snowmelt, followed by ``NDEPL`` areal depletion curves of 11 points). A record that is short on
its line continues on the next data line, as a Fortran list-directed read does. With
``HRU_DEPLCRV = 0`` the reference replaces every curve by full cover (lines 1115-1123), and so does
:attr:`SnoFile.depletion_curve`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["SnoFile", "parse_sno", "read_sno"]

#: the records of ``READ_SNOW`` in read order, each a tuple of field names (Snowprms.for 1092-1111)
_RECORDS: tuple[tuple[str, ...], ...] = (
    ("den_max", "den_init", "freeh2o_cap", "settle_const", "tmax_allsnow"),
    ("albset_rnm", "albset_rna", "albset_snm", "albset_sna"),
    ("cov_type", "covden_sum", "covden_win"),
    ("rad_trncf", "emis_noppt", "potet_sublim"),
)
_SNOWMELT: tuple[str, ...] = (
    "frac_infil",
    "melt_look",
    "melt_force",
    "snarea_thresh",
    "hru_deplcrv",
    "ndepl",
)
#: the integer fields (read into INTEGER variables)
_INTEGERS = frozenset({"cov_type", "melt_look", "melt_force", "hru_deplcrv", "ndepl"})
#: months of the monthly tables and points of a depletion curve (Snowprms.for 1106-1107, 1113)
_N_MONTHS = 12
_N_CURVE = 11
_COMMENT = "="


@dataclass(frozen=True)
class SnoFile:
    """The parameters of one ``.sno`` file, named as in ``READ_SNOW`` (lower case)."""

    den_max: float
    den_init: float
    freeh2o_cap: float
    settle_const: float
    tmax_allsnow: float
    albset_rnm: float
    albset_rna: float
    albset_snm: float
    albset_sna: float
    cov_type: int
    covden_sum: float
    covden_win: float
    rad_trncf: float
    emis_noppt: float
    potet_sublim: float
    cecn_coef: tuple[float, ...]
    tstorm_mo: tuple[int, ...]
    frac_infil: float
    melt_look: int
    melt_force: int
    snarea_thresh: float
    hru_deplcrv: int
    ndepl: int
    snarea_curve: tuple[tuple[float, ...], ...]

    def check_reproduced(self) -> None:
        """Raise ``NotImplementedError`` for what the PRMS snowpack here does not reproduce: a cover type
        above 1 or a thunderstorm month (see :mod:`agrijax.processes.snow.prms`)."""
        if self.cov_type > 1:
            raise NotImplementedError(f"COV_TYPE = {self.cov_type}: only cover types 0 and 1 are reproduced")
        if any(self.tstorm_mo):
            raise NotImplementedError("thunderstorm months (TSTORM_MO = 1) are not reproduced")

    @property
    def depletion_curve(self) -> tuple[float, ...]:
        """The curve the pack uses (``HRU_DEPLCRV``; full cover when it is 0)."""
        if self.hru_deplcrv == 0 or not self.snarea_curve:
            return tuple(1.0 for _ in range(_N_CURVE))
        return self.snarea_curve[self.hru_deplcrv - 1]


def _number(tok: str) -> float:
    return float(tok.replace("D", "E").replace("d", "e"))


class _Tokens:
    """The data tokens of the file, one list per data line (comment and blank lines dropped)."""

    def __init__(self, text: str) -> None:
        self.lines = [
            ln.split() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith(_COMMENT)
        ]
        self.i = 0

    def read(self, n: int) -> list[float]:
        """A list-directed ``READ`` of ``n`` values starting on the next data line."""
        vals: list[float] = []
        while len(vals) < n:
            if self.i >= len(self.lines):
                raise ValueError(f"the .sno file ends inside a record of {n} values")
            vals.extend(_number(t) for t in self.lines[self.i])
            self.i += 1
        return vals[:n]


def parse_sno(text: str) -> SnoFile:
    """Parse the text of a ``.sno`` file."""
    tok = _Tokens(text)
    kw: dict[str, object] = {}
    for rec in _RECORDS:
        kw.update(zip(rec, tok.read(len(rec)), strict=True))
    kw["cecn_coef"] = tuple(tok.read(_N_MONTHS))
    kw["tstorm_mo"] = tuple(int(v) for v in tok.read(_N_MONTHS))
    kw.update(zip(_SNOWMELT, tok.read(len(_SNOWMELT)), strict=True))
    kw.update({name: int(kw[name]) for name in _INTEGERS})  # type: ignore[call-overload]
    ndepl = kw["ndepl"]
    assert isinstance(ndepl, int)
    kw["snarea_curve"] = tuple(tuple(tok.read(_N_CURVE)) for _ in range(ndepl))
    return SnoFile(**kw)  # type: ignore[arg-type]


def read_sno(path: str | Path) -> SnoFile:
    """Read a ``.sno`` file (``IPNAMES.DAT`` names it; Latin-1, any line ending)."""
    return parse_sno(Path(path).read_bytes().decode("latin-1"))
