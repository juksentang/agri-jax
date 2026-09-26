"""agrijax.core.lint: each rule AJ001-AJ005 and AJ007 fires on a violating fixture and a clean process passes
(AJ006: test_depth_scan.py)."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from agrijax.core import lint

CLEAN = '''
import equinox as eqx
import jax.numpy as jnp
from agrijax.core import process

_TINY = 1e-6  # a numerical guard is a named module constant (AJ007)

@process(reads=("water",), writes=("water",))
def clean_process(state, params, forcing_t):
    """Add today's rain to the bucket when it rains.

    Source: toy model, no literature.
    """
    rain = jnp.maximum(forcing_t.rain, 0.0)
    new = jnp.where(rain > 0.0, state.water + rain, state.water)
    ratio = jnp.where(state.water > 0.0, jnp.log(jnp.maximum(state.water, _TINY)), 0.0)
    return eqx.tree_at(lambda s: s.water, state, new + 0.0 * ratio)
'''

AJ001 = '''
@process(writes=("water",))
def bad_if(state, params, forcing_t):
    """Branch in Python.

    Source: fixture.
    """
    w = state.water
    if w > 0:
        w = w - 1.0
    return eqx.tree_at(lambda s: s.water, state, w)
'''

AJ001_WHILE = '''
@process(writes=("water",))
def bad_while(state, params, forcing_t):
    """Loop until converged.

    Source: fixture.
    """
    w = state.water
    while jnp.abs(w) > params.tol:
        w = w * 0.5
    return eqx.tree_at(lambda s: s.water, state, w)
'''

AJ001_IFEXP = '''
@process(writes=("water",))
def bad_ifexp(state, params, forcing_t):
    """Conditional expression on state.

    Source: fixture.
    """
    w = state.water - 1.0 if state.water > 0 else state.water
    return eqx.tree_at(lambda s: s.water, state, w)
'''

AJ002 = '''
@process(writes=("theta",))
def bad_loop(state, params, forcing_t):
    """Loop over soil layers.

    Source: fixture.
    """
    theta = state.theta
    for i in range(37):
        theta = theta.at[i].set(theta[i] * 0.9)
    out = np.zeros(37)
    for i in range(37):
        out[i] = theta[i]
    return eqx.tree_at(lambda s: s.theta, state, out)
'''

AJ003 = '''
@process(writes=("water",))
def bad_where(state, params, forcing_t):
    """Unguarded log and division inside where branches.

    Source: fixture.
    """
    w = state.water
    a = jnp.where(w > 0.0, jnp.log(w), 0.0)
    b = jnp.where(w > 0.0, 1.0 / w, 0.0)
    c = jnp.select([w > 1.0, w > 0.0], [jnp.sqrt(w), w], 0.0)
    return eqx.tree_at(lambda s: s.water, state, a + b + c)
'''

AJ004 = '''
@process(writes=("water",))
def bad_return(state, params, forcing_t):
    """Rebuild the state by hand instead of tree_at.

    Source: fixture.
    """
    return ToyState(water=state.water + 1.0, biomass=state.biomass)
'''

AJ005_NODOC = """
@process(writes=("water",))
def no_doc(state, params, forcing_t):
    return eqx.tree_at(lambda s: s.water, state, state.water + 1.0)
"""

AJ005_NOSOURCE = '''
@process(writes=("water",))
def no_source(state, params, forcing_t):
    """Has a docstring but no provenance line."""
    return eqx.tree_at(lambda s: s.water, state, state.water + 1.0)
'''

NOT_A_PROCESS = """
def helper(state):
    if state.water > 0:
        return 1
    return 0
"""

HEADER = (
    "import equinox as eqx\nimport jax.numpy as jnp\nimport numpy as np\nfrom agrijax.core import process\n"
)


def rules(src: str, **kw) -> set[str]:
    """Rules reported on a fixture; AJ007 (bare literals, which most fixtures have) is not run
    unless asked for with ``ignore=()``."""
    kw.setdefault("ignore", {"AJ007"})
    return {f.rule for f in lint.lint_source(HEADER + textwrap.dedent(src), "fixture.py", **kw)}


def test_clean_process_has_no_findings() -> None:
    assert lint.lint_source(CLEAN, "clean.py") == []


@pytest.mark.parametrize("src", [AJ001, AJ001_WHILE, AJ001_IFEXP], ids=["if", "while", "ifexp"])
def test_aj001_branch_on_state(src: str) -> None:
    found = rules(src)
    assert "AJ001" in found
    assert not found - {"AJ001"}


def test_aj001_static_tests_allowed() -> None:
    src = '''
    @process(writes=("water",))
    def static_ok(state, params, forcing_t):
        """Static Python-level tests are fine.

        Source: fixture.
        """
        if params is None:
            return state
        if isinstance(forcing_t, dict):
            return state
        return eqx.tree_at(lambda s: s.water, state, state.water)
    '''
    assert "AJ001" not in rules(src)


def test_aj002_subscript_assignment_in_loop() -> None:
    findings = [f for f in lint.lint_source(HEADER + AJ002, "f.py") if f.rule == "AJ002"]
    assert len(findings) == 2  # .at[i].set inside a loop and out[i] = ...
    assert all(f.level == "error" for f in findings)


def test_aj003_unguarded_where_branch() -> None:
    findings = [f for f in lint.lint_source(HEADER + AJ003, "f.py") if f.rule == "AJ003"]
    msgs = " | ".join(f.message for f in findings)
    assert "log(w)" in msgs
    assert "division by w" in msgs
    assert "sqrt(w)" in msgs
    assert all(f.level == "warning" for f in findings)
    assert len(findings) == 3


def test_aj004_return_not_via_tree_at() -> None:
    found = rules(AJ004)
    assert "AJ004" in found
    assert "AJ001" not in found


def test_aj004_accepts_replace_and_passthrough() -> None:
    src = '''
    @process(writes=("water",))
    def uses_replace(state, params, forcing_t):
        """Uses .replace and returns the argument unchanged on one path.

        Source: fixture.
        """
        new = state.replace(water=state.water + 1.0)
        return new

    @process(writes=())
    def passthrough(state, params, forcing_t):
        """No-op.

        Source: fixture.
        """
        return state
    '''
    assert "AJ004" not in rules(src)


def test_aj005_docstring_rules() -> None:
    f1 = [f for f in lint.lint_source(HEADER + AJ005_NODOC, "f.py") if f.rule == "AJ005"]
    f2 = [f for f in lint.lint_source(HEADER + AJ005_NOSOURCE, "f.py") if f.rule == "AJ005"]
    assert len(f1) == 1 and "no docstring" in f1[0].message
    assert len(f2) == 1 and "Source:" in f2[0].message


def test_only_process_functions_are_checked_unless_all() -> None:
    assert rules(NOT_A_PROCESS) == set()
    assert "AJ001" in rules(NOT_A_PROCESS, all_functions=True)


def test_syntax_error_is_a_finding() -> None:
    out = lint.lint_source("def broken(:\n", "bad.py")
    assert out and out[0].rule == "AJ000"


def test_lint_paths_and_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "clean.py").write_text(CLEAN)
    (tmp_path / "bad.py").write_text(HEADER + AJ001 + AJ005_NOSOURCE)
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.py").write_text(HEADER + AJ001)

    findings = lint.lint_paths([tmp_path], ignore={"AJ007"})
    assert {f.rule for f in findings} == {"AJ001", "AJ005"}
    assert all("__pycache__" not in f.path for f in findings)

    assert lint.main([str(tmp_path / "clean.py")]) == 0
    assert lint.main([str(tmp_path / "bad.py")]) == 1  # AJ001 is an error
    assert lint.main([str(tmp_path / "clean.py"), "--strict"]) == 0
    warn_only = tmp_path / "warn.py"
    warn_only.write_text(HEADER + AJ005_NOSOURCE)
    assert lint.main([str(warn_only)]) == 0
    assert lint.main([str(warn_only), "--strict"]) == 1
    captured = capsys.readouterr()
    assert "AJ001 [error]" in captured.out


def test_cli_entry_point(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text(HEADER + AJ001)
    proc = subprocess.run(
        [sys.executable, "-m", "agrijax.core.lint", str(bad)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 1
    assert "AJ001" in proc.stdout
    ok = subprocess.run(
        [sys.executable, "-m", "agrijax.core.lint", str(Path(lint.__file__).parent)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr


# ---------------------------------------------------------------------------
# guards (AJ003): sign-aware, so log(x + -5.0) and sqrt(-2.0 * x) are reported
# ---------------------------------------------------------------------------

AJ003_SIGNS = '''
@process(writes=("water",))
def signs(state, params, forcing_t):
    """Negative constants do not guard.

    Source: fixture.
    """
    w = state.water
    a = jnp.where(w > 0.0, jnp.log(w + -5.0), 0.0)
    b = jnp.where(w > 0.0, jnp.sqrt(-2.0 * w), 0.0)
    c = jnp.where(w > 0.0, jnp.log(2.0 * w), 0.0)
    d = jnp.where(w > 0.0, jnp.log(-jnp.maximum(w, 1e-6)), 0.0)
    e = jnp.where(w > 0.0, jnp.log(jnp.maximum(w, 1e-6) - 1.0), 0.0)
    return eqx.tree_at(lambda s: s.water, state, a + b + c + d + e)
'''

AJ003_GUARDED = '''
@process(writes=("water",))
def guarded(state, params, forcing_t):
    """Positive constants on guarded operands are fine; any non-zero constant divisor is fine.

    Source: fixture.
    """
    w = jnp.maximum(state.water, 1e-6)
    a = jnp.where(state.water > 0.0, jnp.log(w + 5.0), 0.0)
    b = jnp.where(state.water > 0.0, jnp.sqrt(2.0 * w), 0.0)
    c = jnp.where(state.water > 0.0, state.water / -2.0, 0.0)
    d = jnp.where(state.water > 0.0, jnp.log(w * jnp.exp(state.water) / (1.0 + w)), 0.0)
    return eqx.tree_at(lambda s: s.water, state, a + b + c + d)
'''


def test_aj003_negative_constants_do_not_guard() -> None:
    findings = [f for f in lint.lint_source(HEADER + AJ003_SIGNS, "f.py") if f.rule == "AJ003"]
    msgs = " | ".join(f.message for f in findings)
    assert "log(w + -5.0)" in msgs
    assert "sqrt(-2.0 * w)" in msgs
    assert "log(2.0 * w)" in msgs  # a positive factor on an unguarded operand is still unguarded
    assert "log(-jnp.maximum(w, 1e-06))" in msgs
    assert "log(jnp.maximum(w, 1e-06) - 1.0)" in msgs
    assert len(findings) == 5


def test_aj003_positive_guards_accepted() -> None:
    assert "AJ003" not in rules(AJ003_GUARDED)


# ---------------------------------------------------------------------------
# scope: kernels in processes/ and static arguments
# ---------------------------------------------------------------------------

KERNEL = """
from typing import Literal


def kernel(x, n: int, mode: str, flag: bool = True, scale: float = 2.0, kind: Literal["a", "b"] = "a"):
    y = x * 2.0 if flag else x
    if n > 3 and mode == "fast" and kind == "a":
        y = y / scale
    if x.shape[-1] != 6 or np.ndim(x) > 2 or len(x) == 0:
        raise ValueError("bad shape")
    if x > 0:
        y = y + 1.0
    return y


class Holder:
    @property
    def size(self):
        return int(np.prod(np.shape(self.v))) if np.ndim(self.v) else 1

    @classmethod
    def build(cls, rec, *, derive: bool = True):
        r = np.asarray(rec)
        if r.shape[-1] != 6:
            raise ValueError("bad")
        return cls() if derive else None
"""


def test_kernels_under_processes_get_numerical_rules(tmp_path: Path) -> None:
    f = tmp_path / "processes" / "pet" / "k.py"
    f.parent.mkdir(parents=True)
    f.write_text(KERNEL)
    found = lint.lint_file(f, ignore={"AJ007"})
    # only the traced ``if x > 0`` is reported: static args, self/cls, shape and str tests are not
    assert [(x.rule, x.line) for x in found] == [("AJ001", 11)]
    assert {x.rule for x in found} <= lint.KERNEL_RULES  # no AJ004/AJ005 on kernels
    # outside a processes/ directory the same file is not linted unless --all
    g = tmp_path / "io" / "k.py"
    g.parent.mkdir()
    g.write_text(KERNEL)
    assert lint.lint_file(g) == []
    assert {"AJ001", "AJ004", "AJ005"} <= {x.rule for x in lint.lint_file(g, all_functions=True)}
    # the default 2.0, the factor 2.0 and the threshold 3 are AJ007 warnings; 0 and 1.0 are
    # whitelisted, and so is the 6 of the shape test
    aj007 = [x for x in lint.lint_file(f) if x.rule == "AJ007"]
    assert [ln.split(";")[0].rsplit(": ", 1)[1] for ln in (x.message for x in aj007)] == ["2.0", "2.0", "3"]
    assert all(x.level == "warning" for x in aj007)


def test_lint_of_the_package_is_not_vacuous() -> None:
    """The src gate must actually visit the process code (it once visited 0 functions)."""
    processes = Path(lint.__file__).resolve().parents[1] / "processes"
    visited = lint.checked_functions([processes])
    names = {c.name for c in visited}
    assert len(visited) >= 20
    assert {"shuttleworth_wallace", "theta_of_h", "k_of_h", "h_of_theta", "asce_reference_et"} <= names
    procs = {c.name for c in visited if c.is_process}
    assert {"pet_shuttleworth_wallace", "pet_asce_reference", "pet_priestley_taylor"} <= procs
    assert all(c.rules == lint.ALL_RULES for c in visited if c.is_process)
    # and the package is clean under the strict gate that CI and pre-commit run (AJ007 included)
    findings = lint.lint_paths([Path(lint.__file__).resolve().parents[1]])
    assert findings == [], "\n".join(f.format() for f in findings)


# ---------------------------------------------------------------------------
# AJ007: bare numeric literals
# ---------------------------------------------------------------------------

AJ007_BAD = """
@process(writes=("water",))
def literals(state, params, forcing_t, eps=1e-9):
    \"\"\"Bare model numbers, unit conversions and guards.

    Source: fixture.
    \"\"\"
    w = state.water * 0.85 + 2.0
    cm = forcing_t.rain * 0.1
    safe = w / jnp.maximum(cm, 1e-12)
    k = jnp.where(w > 4.0, -0.5, safe)
    f = lambda x: x * 7
    return eqx.tree_at(lambda s: s.water, state, k + f(w) + 3)
"""

AJ007_OK = """
_EPS = 1e-12
_GAIN = 0.85


@process(writes=("theta",))
def allowed(state, params, forcing_t):
    \"\"\"Only whitelisted literals: 0, 1, -1, indices, axes, shapes and small integer exponents.

    Source: fixture.
    \"\"\"
    t = state.theta
    if t.ndim == 2 and t.shape[-1] != 6:
        raise ValueError("bad shape")
    a = t[..., 2] + t[:, -3:] ** 2 + jnp.power(t, 3) + t**-2
    b = jnp.sum(t, axis=-2) + jnp.reshape(t, (-1, 4)).sum(axis=1) + jnp.zeros((3, 2))
    c = jnp.maximum(t, _EPS) * _GAIN + 1.0 - 0 + t[..., : t.shape[-1] - 2]
    seeds = [t * k for k in range(3)]
    return eqx.tree_at(lambda s: s.theta, state, a + b + c - 1 + seeds[0])
"""


def _aj007(src: str) -> list[lint.Finding]:
    return [f for f in lint.lint_source(HEADER + textwrap.dedent(src), "f.py") if f.rule == "AJ007"]


def test_aj007_reports_bare_literals() -> None:
    found = _aj007(AJ007_BAD)
    texts = [f.message.split(";")[0].rsplit(": ", 1)[1] for f in found]
    assert sorted(texts) == sorted(["1e-09", "0.85", "2.0", "0.1", "1e-12", "4.0", "-0.5", "7", "3"])
    assert all(f.level == "warning" for f in found)
    # powers of ten carry the unit-adapter hint; a negative literal is one finding, not two
    by = {f.message.split(";")[0].rsplit(": ", 1)[1]: f.message for f in found}
    assert "core.units" in by["0.1"] and "core.units" in by["1e-12"] and "core.units" not in by["0.85"]
    assert "AJ007" in lint.KERNEL_RULES and lint.RULES["AJ007"][0] == "warning"


def test_aj007_whitelist() -> None:
    assert _aj007(AJ007_OK) == [], [f.format() for f in _aj007(AJ007_OK)]
    assert {0.0, 1.0, -1.0} == set(lint.AJ007_TRIVIAL)


def test_aj007_arithmetic_on_an_index_is_not_structural() -> None:
    src = """
    @process(writes=("water",))
    def p(state, params, forcing_t):
        \"\"\"Doc.

        Source: fixture.
        \"\"\"
        return eqx.tree_at(lambda s: s.water, state, jnp.sum(state.water * 2, axis=0) + state.water[3] ** 0.5)
    """
    texts = [f.message.split(";")[0].rsplit(": ", 1)[1] for f in _aj007(src)]
    assert texts == ["2", "0.5"]  # an argument of a structural call only when it is the argument itself


def test_aj007_decorators_annotations_and_module_constants_are_not_checked() -> None:
    src = """
    import functools
    from typing import Literal

    SCALE = 0.3

    @process(writes=("water",), version=2)
    def p(state, params, forcing_t, mode: Literal[5] = 5) -> "Annotated[int, 7]":
        \"\"\"Doc.

        Source: fixture.
        \"\"\"
        return eqx.tree_at(lambda s: s.water, state, state.water * SCALE)
    """
    assert [f.message for f in _aj007(src)] == [f.message for f in _aj007(src) if "5" in f.message]
    assert len(_aj007(src)) == 1  # the default value 5 is checked; decorator and annotations are not


def test_aj007_is_escalated_by_strict(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "lit.py"
    f.write_text(HEADER + AJ007_BAD)
    assert {x.rule for x in lint.lint_file(f)} == {"AJ007"}
    assert lint.main([str(f)]) == 0
    assert lint.main([str(f), "--strict"]) == 1
    assert lint.main([str(f), "--strict", "--ignore", "AJ007"]) == 0
    assert lint.main([str(f), "--strict-aj007"]) == 1
    assert lint.main([str(f), "--strict-aj007", "--ignore", "AJ007"]) == 0
    assert lint.main([str(f), "--aj007-report", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert f"AJ007     9  {f}" in out
    assert "(9 AJ007)" in out
    with pytest.raises(SystemExit):
        lint.main([str(f), "--ignore", "AJ999"])


def test_aj007_package_count() -> None:
    """The package has no AJ007 finding (every coefficient is labelled; --strict enforces it)."""
    src = Path(lint.__file__).resolve().parents[1]
    counts = lint.count_by_file(lint.lint_paths([src]), "AJ007")
    assert counts == {}, counts
    # the coefficient declarations themselves carry no finding (they are module-level fields)
    assert not any(p.endswith(("coefficients.py", "units.py")) for p in counts)


# ---------------------------------------------------------------------------
# AJ008: a slot imports only core, the port records and its own package
# ---------------------------------------------------------------------------

AJ008_SRC = """
import numpy as np
from agrijax.core import process
from agrijax.iface.crop import CropWaterIn
from agrijax.processes.soil_water.uptake import RootRecord
import agrijax.processes.pet.daily
from agrijax.processes import soil_water, crop
from ...pet import daily
from ... import pet
from .. import sibling
from . import local
from agrijax import processes
import agrijax.processes
import my_plugin.processes.bucket


def f():
    from agrijax.processes.soil_water import richards
"""


def _aj008(path: Path) -> list[tuple[int, str]]:
    out = []
    for f in lint.lint_file(path):
        assert f.rule == "AJ008" and f.level == "warning"
        out.append((f.line, f.message.split("imports processes/")[1].split(" ")[0]))
    return out


def test_aj008_reports_imports_of_other_slots(tmp_path: Path) -> None:
    f = tmp_path / "pkg" / "processes" / "crop" / "sub" / "m.py"
    f.parent.mkdir(parents=True)
    f.write_text(AJ008_SRC)
    # core, iface, numpy, the own slot (crop, ``..`` and ``.``) and the bare processes package pass
    assert _aj008(f) == [
        (5, "soil_water"),
        (6, "pet"),
        (7, "soil_water"),  # from agrijax.processes import soil_water, crop: crop is the own slot
        (8, "pet"),
        (9, "pet"),
        (14, "bucket"),  # any package's processes/ directory
        (18, "soil_water"),  # inside a function too
    ]
    assert lint.slot_of_path(f) == ("crop", ("crop", "sub"))


def test_aj008_file_directly_in_processes_is_its_own_slot(tmp_path: Path) -> None:
    f = tmp_path / "processes" / "bucket.py"
    f.parent.mkdir(parents=True)
    f.write_text("from . import local\nfrom .bucket_util import x\nfrom agrijax.processes.pet import y\n")
    assert _aj008(f) == [(1, "local"), (2, "bucket_util"), (3, "pet")]
    assert lint.slot_of_path(tmp_path / "processes" / "__init__.py") is None
    assert lint.slot_of_path(tmp_path / "io" / "x.py") is None
    g = tmp_path / "io" / "x.py"
    g.parent.mkdir()
    g.write_text("from agrijax.processes.pet import y\n")
    assert lint.lint_file(g) == [] and lint.lint_file(g, all_functions=True) == []  # outside processes/


def test_aj008_is_enforced_by_strict_and_can_be_ignored(tmp_path: Path) -> None:
    f = tmp_path / "processes" / "crop" / "m.py"
    f.parent.mkdir(parents=True)
    f.write_text("from agrijax.processes.soil_water.uptake import RootRecord\n")
    assert lint.main([str(f)]) == 0
    assert lint.main([str(f), "--strict"]) == 1
    assert lint.main([str(f), "--strict", "--ignore", "AJ008"]) == 0
    assert (
        "AJ008" in lint.RULES and lint.RULES["AJ008"][0] == "warning" and "AJ008" not in lint.NOT_STRICT_RULES
    )


def test_aj008_the_package_has_no_cross_slot_import() -> None:
    src = Path(lint.__file__).resolve().parents[1]
    found = [f for f in lint.lint_paths([src]) if f.rule == "AJ008"]
    assert found == [], "\n".join(f.format() for f in found)


def test_kernel_and_processes_options(tmp_path: Path) -> None:
    g = tmp_path / "plugin" / "k.py"
    g.parent.mkdir()
    g.write_text(HEADER + KERNEL)
    assert lint.lint_file(g) == []  # not under processes/: not linted by default
    assert [x.rule for x in lint.lint_file(g, kernel=True, ignore={"AJ007"})] == ["AJ001"]
    src = HEADER + textwrap.dedent("""
    def body(state, params, forcing_t):
        return state.water * 2
    """)
    h = tmp_path / "plugin" / "p.py"
    h.write_text(src)
    rules = {x.rule for x in lint.lint_file(h, kernel=True, processes=("body",))}
    assert {"AJ004", "AJ005", "AJ007"} <= rules  # a call-form process gets every rule
    assert {x.rule for x in lint.lint_file(h, kernel=True)} == {"AJ007"}


# AJ008 below the slots: core and iface import no processes module; iface only core and iface
BELOW_SRC = """
import numpy as np
from agrijax.core.state import State
from agrijax.processes.soil_water.uptake import RootRecord
from ..processes.pet import daily
from .. import processes
from agrijax import processes as p
from agrijax.models import day_rzwqm46
from . import crop
from .. import core


def f():
    import agrijax.processes.pet
    from agrijax.io import catpa
"""


def _below(path: Path) -> list[int]:
    out = []
    for f in lint.lint_file(path):
        assert f.rule == "AJ008" and f.level == "warning"
        out.append(f.line)
    return out


def test_aj008_iface_imports_only_core_and_iface(tmp_path: Path) -> None:
    f = tmp_path / "src" / "agrijax" / "iface" / "m.py"
    f.parent.mkdir(parents=True)
    f.write_text(BELOW_SRC)
    # processes (absolute, relative, inside a function) and any other agrijax package are reported
    assert _below(f) == [4, 5, 6, 7, 8, 14, 15]
    msg = next(x.message for x in lint.lint_file(f) if x.line == 8)
    assert "agrijax/iface imports agrijax.models" in msg


def test_aj008_core_imports_no_processes_module(tmp_path: Path) -> None:
    f = tmp_path / "agrijax" / "core" / "m.py"
    f.parent.mkdir(parents=True)
    f.write_text(BELOW_SRC)
    # core may import other agrijax packages lazily (io), never processes
    assert _below(f) == [4, 5, 6, 7, 14]
    assert lint.main([str(f), "--strict"]) == 1 and lint.main([str(f), "--strict", "--ignore", "AJ008"]) == 0


def test_module_imports_and_import_closure(tmp_path: Path) -> None:
    tree = ast.parse("from ..processes.pet import daily\nfrom . import crop\nimport a.b\nfrom x import *\n")
    nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    got = [lint.module_imports(n, ("agrijax", "iface")) for n in nodes]
    assert got == [
        ["agrijax.processes.pet", "agrijax.processes.pet.daily"],
        ["agrijax.iface", "agrijax.iface.crop"],
        ["a.b"],
        ["x"],
    ]
    root = tmp_path / "pkg"
    (root / "sub").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "a.py").write_text("from .sub import b\n")
    (root / "sub" / "__init__.py").write_text("import numpy\n")
    (root / "sub" / "b.py").write_text("def f():\n    from pkg import c\n")
    (root / "c.py").write_text("")
    assert lint.import_closure(["pkg.a"], tmp_path) == {
        "pkg": "",
        "pkg.a": "",
        "pkg.sub": "pkg.a",
        "pkg.sub.b": "pkg.a",
        "pkg.c": "pkg.sub.b",
    }
    assert set(lint.import_closure(["pkg.a"], tmp_path, depth=1)) == {"pkg", "pkg.a", "pkg.sub", "pkg.sub.b"}
