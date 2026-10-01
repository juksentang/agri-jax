"""Swap a process of the DSSAT-CSM maize day from the facade: the soil evaporation.

The code this adds to the quick start of :mod:`agrijax.dssat` (UFGA8201 uses DSSAT's ``MESEV = R``,
Ritchie's two-stage soil evaporation; the swap runs SALUS's, ``MESEV = S``)::

    import agrijax as aj
    exp = aj.dssat.experiment("UFGA8201")
    aj.dssat.alternatives()                                  # the swaps validated against dscsm048
    ritchie = exp.run(treatment=4)                           # the experiment's own soil evaporation
    salus = exp.run(treatment=4, soil_evaporation="salus")   # the same season with the other one
    ref = exp.reference(treatment=4, soil_evaporation="salus")   # dscsm048 with MESEV = S
    salus.compare_summary(ref)                               # dates and yield against DSSAT
    scen = exp.scenarios(treatment=4, years=range(1978, 1988), sowing_shift=[-14, 0, 14],
                         soil_evaporation="salus")
    batch = scen.run({"G2": [800.0, 900.0, 1000.0]})         # every sample on every scenario
    res = aj.calibrate(exp, treatments=[4], holdout=[6], soil_evaporation="salus")
    aj.dssat.alternatives(exp, 4)                            # which one the experiment itself uses

**What a swap is.** DSSAT-CSM v4.8.6.0 chooses its soil evaporation with the ``MESEV`` option of the
experiment file (``R``: Ritchie two-stage ``SOILEV``; ``S``: SALUS layered ``ESR_SoilEvap``), and
Agri-JAX's day has both, each validated against ``dscsm048`` where the experiment's own ``MESEV``
selects it (the maize free-run acceptance: the season's yield within 2 % of DSSAT's). A swap sets
``MESEV`` of the experiment file's ``METHODS`` line to the other one, in a copy staged in the
experiment's work directory, and builds everything from that copy: the season then ends at the
maturity of the swapped model (as in DSSAT with that option), the scenarios, the reference run
(:meth:`~agrijax.dssat.Experiment.reference`, :meth:`~agrijax.dssat.Scenarios.reference`,
:meth:`~agrijax.dssat.Experiment.dssat_batch`) and the calibration's DSSAT check read the same file,
so a swapped Agri-JAX run is compared with DSSAT run on the same swap. The swapped model is a different
model from the experiment's own: the integration test (``tests/integration/test_facade_swap.py``)
compares each swapped run with ``dscsm048`` on the same swap and with the swap test of the day
(``tests/integration/test_day_dssat486_swap.py``).

**Limits.** Only the choices of :func:`alternatives` are offered; any other name raises
:class:`SwapError` listing them (a process of your own: ``docs/swapping_a_process.md``). The
experiment file needs a ``METHODS`` line in its simulation controls to carry the option, unless the
swap is to DSSAT's default (``ritchie``). A swap needs the native inputs, so
:func:`agrijax.dssat.calibrate` refuses ``inputs="tables"`` with one.
"""

from __future__ import annotations

import difflib
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ALTERNATIVES",
    "PROCESS",
    "Alternative",
    "SwapError",
    "alternatives",
    "calibration_experiment",
    "filex_edit",
    "filex_with_mesev",
    "inputs",
    "note",
    "own_mesev",
    "resolve",
    "stage_filex",
    "swap_code",
    "swapped_text",
]

#: the argument name of the swap in :meth:`agrijax.dssat.Experiment.run` and the other calls
PROCESS = "soil_evaporation"


class SwapError(ValueError):
    """A swap the facade cannot make: an unknown or unvalidated alternative (the message lists the valid
    ones), or an experiment file that cannot carry the option."""


@dataclass(frozen=True)
class Alternative:
    """One validated choice of a process of the DSSAT day: its :attr:`name` (the value of the argument),
    the DSSAT option that selects it (:attr:`mesev`: the ``MESEV`` letter of the experiment file), what it
    is and where it comes from, and what it was validated against."""

    name: str
    mesev: str
    method: str
    module: str
    removal: str
    validated: str

    @property
    def key(self) -> str:
        """The process registry key of the implementation (the day's building blocks take it)."""
        from agrijax.models.day_dssat486 import SOIL_EVAPORATION_KEYS

        return SOIL_EVAPORATION_KEYS[self.mesev]

    def __str__(self) -> str:
        return f"{self.name!r} (DSSAT MESEV = {self.mesev}: {self.method})"


#: the validated alternatives of the soil evaporation: each is the implementation of one ``MESEV`` value
#: of DSSAT-CSM v4.8.6.0, run against ``dscsm048`` with that value in the experiment file
ALTERNATIVES: tuple[Alternative, ...] = (
    Alternative(
        name="ritchie",
        mesev="R",
        method="Ritchie two-stage soil evaporation",
        module="SOILEV (Ritchie 1972)",
        removal="the evaporation comes from the top layer",
        validated="dscsm048 with MESEV = R: the maize free-run acceptance (yield within 2 % of DSSAT's)",
    ),
    Alternative(
        name="salus",
        mesev="S",
        method="SALUS layered soil evaporation",
        module="ESR_SoilEvap (Suleiman and Ritchie 2003)",
        removal="the evaporation is taken layer by layer",
        validated="dscsm048 with MESEV = S: the maize free-run acceptance (yield within 2 % of DSSAT's)",
    ),
)
_BY_NAME = {a.name: a for a in ALTERNATIVES} | {a.mesev.lower(): a for a in ALTERNATIVES}


def _listing() -> str:
    return "; ".join(str(a) for a in ALTERNATIVES)


def resolve(choice: Any) -> Alternative | None:
    """The validated alternative named by ``choice`` (``None`` stays ``None``: the experiment's own).

    ``choice`` is a name of :data:`ALTERNATIVES` (``"ritchie"``, ``"salus"``, any case), the DSSAT
    ``MESEV`` letter (``"R"``, ``"S"``) or the registry key of one of them. Anything else raises
    :class:`SwapError` listing the valid choices (and the nearest name, when there is one)."""
    if choice is None:
        return None
    if isinstance(choice, str):
        s = choice.strip()
        alt = _BY_NAME.get(s.lower())
        if alt is not None:
            return alt
        if "/" in s:
            return _by_key(s)
        near = difflib.get_close_matches(s.lower(), [a.name for a in ALTERNATIVES], n=1)
        hint = f" Did you mean {near[0]!r}?" if near else ""
        raise SwapError(
            f"{PROCESS}={choice!r} is not a validated alternative. Valid choices: {_listing()}.{hint} "
            "aj.dssat.alternatives() lists them."
        )
    raise SwapError(
        f"{PROCESS} must be the name of an alternative or None, got {choice!r}. Valid choices: {_listing()}."
    )


def _by_key(key: str) -> Alternative:
    """The alternative of registry key ``key``; a registered process that is not one of them is refused."""
    from agrijax.core.process import lookup
    from agrijax.models.day_dssat486 import SOIL_EVAPORATION_KEYS

    for a in ALTERNATIVES:
        if SOIL_EVAPORATION_KEYS[a.mesev] == key:
            return a
    try:
        lookup(key)
    except (KeyError, ValueError):
        raise SwapError(
            f"{PROCESS}={key!r}: no process is registered under this key. Valid choices: {_listing()}."
        ) from None
    raise SwapError(
        f"{PROCESS}={key!r} is registered, but is not a soil evaporation validated against dscsm048 for "
        f"this day, so the facade does not offer it. Valid choices: {_listing()}. A process of your own "
        "plugs into the day directly: docs/swapping_a_process.md."
    )


def own_mesev(filex: str | os.PathLike[str], treatment: int) -> str:
    """The ``MESEV`` letter ``treatment`` of the experiment file ``filex`` runs with, as DSSAT reads it
    (``R`` without a ``METHODS`` line)."""
    from agrijax.io.dssat import read_filex
    from agrijax.io.dssat.native_management import switches

    return switches(read_filex(Path(filex)), int(treatment)).mesev


def alternatives(experiment: Any = None, treatment: int | None = None) -> Any:
    """The validated alternatives of the process that can be swapped (:data:`ALTERNATIVES`), one row each
    (pandas): the argument value (the index), the DSSAT option that selects it, the method and DSSAT
    module, how it removes the water, and what it was validated against. With an ``experiment`` and a
    ``treatment``, the column ``experiment's own`` says which one DSSAT uses on that treatment."""
    import pandas as pd

    if (experiment is None) != (treatment is None):
        raise ValueError("give both the experiment and the treatment, or neither")
    rows = [
        {
            "choice": a.name,
            "argument": PROCESS,
            "DSSAT option": f"MESEV = {a.mesev}",
            "method": a.method,
            "DSSAT-CSM module": a.module,
            "water removal": a.removal,
            "validated against": a.validated,
        }
        for a in ALTERNATIVES
    ]
    df = pd.DataFrame(rows).set_index("choice")
    if experiment is not None:
        own = own_mesev(experiment.filex, experiment._trno(treatment))
        df["experiment's own"] = [a.mesev == own for a in ALTERNATIVES]
    return df


def swap_code(experiment: Any, treatment: int, choice: Any) -> str | None:
    """The ``MESEV`` letter to set for ``choice`` on ``treatment`` of ``experiment``, or ``None`` when
    nothing changes (no choice, or the one the experiment's file already has). Raises
    :class:`SwapError` for an unknown choice."""
    alt = resolve(choice)
    if alt is None or alt.mesev == own_mesev(experiment.filex, treatment):
        return None
    return alt.mesev


def filex_with_mesev(text: str, mesev: str, what: str = "the experiment file") -> str:
    """FileX text with ``MESEV = mesev`` on every ``METHODS`` line of its simulation controls (values are
    right-aligned under their header names, as :func:`agrijax.sites.dssat_free_run.nitrogen_off_filex`
    writes them); every other character is kept. Raises :class:`SwapError` when ``what`` has no
    ``METHODS`` line with a ``MESEV`` column to carry the option."""
    out: list[str] = []
    hdr = ""
    n_rows = 0
    for ln in text.splitlines():
        if ln.startswith("@N METHODS"):
            hdr = ln
        elif ln.startswith(("@", "*")):
            hdr = ""
        elif hdr and ln.split()[1:2] == ["ME"]:
            if "MESEV" not in hdr:
                raise SwapError(f"{what}: the METHODS header has no MESEV column: {hdr.strip()}")
            k = hdr.index("MESEV") + len("MESEV") - 1
            ln = ln.ljust(k + 1)
            ln = ln[:k] + mesev + ln[k + 1 :]
            n_rows += 1
        out.append(ln)
    if not n_rows:
        raise SwapError(
            f"{what} has no METHODS line in its simulation controls, so MESEV = {mesev} cannot be set "
            "there (DSSAT's default, 'ritchie', applies without one)"
        )
    return "\n".join(out) + "\n"


def swapped_text(experiment: Any, treatment: int, choice: Any) -> str:
    """The experiment file's text with the swap of ``choice`` (the text itself when nothing changes)."""
    text = experiment.filex.read_text(errors="replace")
    code = swap_code(experiment, treatment, choice)
    if code is None:
        return text
    return filex_with_mesev(text, code, f"{experiment.name} treatment {treatment}")


def filex_edit(experiment: Any, treatment: int, choice: Any) -> Callable[[str], str] | None:
    """The FileX edit of a DSSAT run on the swap (``filex_edit=`` of
    :func:`agrijax.sites.dssat_free_run.run_reference`), ``None`` when nothing changes."""
    code = swap_code(experiment, treatment, choice)
    if code is None:
        return None
    what = f"{experiment.name} treatment {treatment}"
    return lambda text: filex_with_mesev(text, code, what)


def stage_filex(filex: Path, mesev: str, dest: Path, what: str | None = None) -> Path:
    """Stage a copy of the experiment file ``filex`` with ``MESEV = mesev`` in the directory ``dest``,
    with the files next to it that the runs read (``<name>.MZ*``: the observed files, and the ``.SOL``
    soil files); returns the staged file."""
    dest.mkdir(parents=True, exist_ok=True)
    for f in [*filex.parent.glob(filex.stem + ".MZ*"), *filex.parent.glob("*.SOL")]:
        shutil.copy2(f, dest / f.name)
    text = filex.read_bytes().decode("latin-1").split("\x1a", 1)[0]
    staged = dest / filex.name
    staged.write_bytes(filex_with_mesev(text, mesev, what or filex.name).encode("latin-1"))
    return staged


def inputs(experiment: Any, treatment: int, choice: Any) -> Any:
    """:meth:`agrijax.dssat.Experiment.inputs` of ``treatment`` on the swap of ``choice`` (built once per
    swap from the staged experiment file, so the season ends at the swapped model's maturity); the
    experiment's own inputs when nothing changes."""
    code = swap_code(experiment, treatment, choice)
    if code is None:
        return experiment.inputs(treatment)
    key = (int(treatment), code)
    if key not in experiment._inputs:
        staged = stage_filex(
            experiment.filex,
            code,
            experiment._dir(f"swap_{code}"),
            f"{experiment.name} treatment {treatment}",
        )
        experiment._inputs[key] = experiment._build(staged, int(treatment))
    return experiment._inputs[key]


def note(choice: Any) -> str:
    """The line a calibration on the swap of ``choice`` adds to its notes."""
    alt = resolve(choice)
    if alt is None:
        raise SwapError(f"no swap to describe: {PROCESS}=None")
    return f"soil evaporation swapped: {alt} instead of the experiment's own (DSSAT MESEV of its file)"


def calibration_experiment(experiments: Any, data_root: Path, choice: Any) -> Any:
    """What to hand :func:`agrijax.calib.workflow.calibrate` to calibrate on the swap of ``choice``: each
    experiment (an :class:`~agrijax.dssat.Experiment`, an example name or a ``.MZX`` path; or a sequence
    of them) as the path of its file staged with the option set, or unchanged when every treatment of
    the file already has it. The calibration, its inputs and its DSSAT check all read the staged file."""
    alt = resolve(choice)
    if alt is None:
        return experiments
    if isinstance(experiments, (str, os.PathLike)) or hasattr(experiments, "filex"):
        return _stage_one(experiments, data_root, alt)
    seq: Sequence[Any] = list(experiments)
    return [_stage_one(e, data_root, alt) for e in seq]


def _stage_one(experiment: Any, data_root: Path, alt: Alternative) -> Any:
    from agrijax.calib.workflow import _filex_path
    from agrijax.io.dssat import read_filex

    if hasattr(experiment, "filex"):
        fx, name = Path(experiment.filex), str(experiment.name)
    else:
        fx = _filex_path(experiment, data_root)
        name = fx.stem
    trs = sorted({int(t["N"]) for t in read_filex(fx)["TREATMENTS"]})
    if all(own_mesev(fx, t) == alt.mesev for t in trs):
        return experiment
    root = Path(os.environ.get("AGRI_JAX_RUN_ROOT") or tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"aj_swap_{name}_", dir=root))
    return stage_filex(fx, alt.mesev, work, name)
