"""agrijax.core.lint: each rule AJ001-AJ005, AJ007-AJ012, AJ020 and AJ021 fires on a violating fixture and
a clean process passes (AJ006: test_depth_scan.py)."""

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
        '''Host-side: build from a file record (NumPy on the argument is fine here, AJ020).'''
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
    # and the package is clean under the strict gate that CI and pre-commit run (AJ007 included;
    # the rules --strict does not enforce yet are counted in test_aj009_to_aj011_on_the_package)
    findings = [
        f
        for f in lint.lint_paths([Path(lint.__file__).resolve().parents[1]])
        if f.rule not in lint.NOT_STRICT_RULES
    ]
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
from agrijax.processes.water_supply.rootwu import RootRecord
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
        (5, "water_supply"),
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
    f.write_text("from agrijax.processes.water_supply.rootwu import RootRecord\n")
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
from agrijax.processes.water_supply.rootwu import RootRecord
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
    for f in lint.lint_file(path, ignore={"AJ012"}):  # the io import of line 15 is AJ012's
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
    # core may import other agrijax packages (models), never processes; never io either (AJ012)
    assert _below(f) == [4, 5, 6, 7, 14]
    assert [x.line for x in lint.lint_file(f) if x.rule == "AJ012"] == [15]
    assert lint.main([str(f), "--strict"]) == 1
    assert lint.main([str(f), "--strict", "--ignore", "AJ008", "--ignore", "AJ012"]) == 0


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


# AJ008 sees dynamic imports: importlib.import_module and __import__ (plan 15 section 9.18 item 1)
DYNAMIC_SRC = """
import importlib
from importlib import import_module

TARGET = "agrijax.processes.water_supply.rootwu"
_t = importlib.import_module(TARGET)
__import__("agrijax.processes.pet.daily")
importlib.import_module("agrijax.processes.crop.x")
importlib.import_module(some_name)
import_module("..pet")
importlib.import_module("numpy")
"""


def test_aj008_reports_dynamic_imports_of_other_slots(tmp_path: Path) -> None:
    f = tmp_path / "processes" / "crop" / "m.py"
    f.parent.mkdir(parents=True)
    f.write_text(DYNAMIC_SRC)
    got = [(x.line, x.message) for x in lint.lint_file(f)]
    assert all(x.rule == "AJ008" for x in lint.lint_file(f))
    assert [line for line, _ in got] == [6, 7, 9, 10]
    assert "processes/water_supply" in got[0][1]  # a module-level constant is resolved
    assert "processes/pet" in got[1][1] and "processes/pet" in got[3][1]
    assert "cannot resolve" in got[2][1]
    assert lint.main([str(f), "--strict"]) == 1


def test_aj008_dynamic_imports_below_the_slots(tmp_path: Path) -> None:
    f = tmp_path / "src" / "agrijax" / "iface" / "m.py"
    f.parent.mkdir(parents=True)
    f.write_text(
        'import importlib\nWHERE = ".crop"\nimportlib.import_module(WHERE, __name__)\n'
        'importlib.import_module(name)\nimportlib.import_module("agrijax.processes.pet")\n'
    )
    # a lazy loader of the records' own modules passes; a processes module is reported
    assert _below(f) == [5]


# ---------------------------------------------------------------------------
# AJ008 (dynamic imports, forcing/), AJ009-AJ011, process(...) entries outside processes/
# ---------------------------------------------------------------------------


def test_aj008_sees_literal_dynamic_imports(tmp_path: Path) -> None:
    f = tmp_path / "processes" / "crop" / "m.py"
    f.parent.mkdir(parents=True)
    f.write_text(
        "import importlib\n"
        "def f():\n"
        "    a = importlib.import_module('agrijax.processes.water_supply.rootwu')\n"
        "    b = __import__('agrijax.processes.pet.daily')\n"
        "    return a, b\n"
    )
    assert _aj008(f) == [(3, "water_supply"), (4, "pet")]  # also inside functions
    g = tmp_path / "agrijax" / "forcing" / "gen.py"
    g.parent.mkdir(parents=True)
    g.write_text("from agrijax.processes.pet.daily import DailyWeather\nfrom agrijax.iface import PORTS\n")
    assert [x.line for x in lint.lint_file(g) if x.rule == "AJ008"] == [1]  # forcing/ is below the slots


AJ009_SRC = """
_CACHE = {}
_SEEN: list = []
LIMIT = 3


def remember(key, value):
    _CACHE[key] = value
    _SEEN.append(key)


def counter():
    global LIMIT
    LIMIT += 1


def fine(x):
    _CACHE = {}
    _CACHE[x] = 1
    return LIMIT + len(_SEEN)
"""


def test_aj009_reports_module_state_mutated_from_functions(tmp_path: Path) -> None:
    f = tmp_path / "processes" / "pet" / "m.py"
    f.parent.mkdir(parents=True)
    f.write_text(AJ009_SRC)
    got = sorted((x.line, x.message.split(": ", 1)[1]) for x in lint.lint_file(f) if x.rule == "AJ009")
    assert [line for line, _ in got] == [8, 9, 13]  # _CACHE[...] =, _SEEN.append, global; `fine` shadows
    g = tmp_path / "io" / "m.py"
    g.parent.mkdir()
    g.write_text(AJ009_SRC)
    assert [x for x in lint.lint_file(g) if x.rule == "AJ009"] == []  # outside processes/, iface/, forcing/
    assert "AJ009" in lint.NOT_STRICT_RULES  # until gap G24 moves the settings registry to core
    assert lint.main([str(f), "--strict"]) == 0 and lint.main([str(f), "--enforce", "AJ009"]) == 1


AJ010_SRC = '''
from agrijax.core import process


@process(reads=("soil_water.theta", "iface.canopy.maize"), writes=("ledger.water",))
def bad(state, params, forcing_t):
    """Source: fixture."""
    return state.replace()


@process(reads=("soil_water.theta", "canopy"), writes=("soil_water",))
def good(state, params, forcing_t):
    """Source: fixture."""
    return state.replace()
'''


def test_aj010_reports_global_paths_in_process_declarations(tmp_path: Path) -> None:
    f = tmp_path / "processes" / "pet" / "m.py"
    f.parent.mkdir(parents=True)
    f.write_text(AJ010_SRC)
    got = [x.message.split("bad ", 1)[1] for x in lint.lint_file(f) if x.rule == "AJ010"]
    assert got == [
        "reads 'iface.canopy.maize'; declare a port() field and bind it",
        "writes 'ledger.water'; declare a port() field and bind it",
    ]
    assert lint.main([str(f), "--strict"]) == 1


AJ011_SRC = '''
from agrijax.core import process


@process(reads=("soil_water",), writes=("soil_water",), key="soil_water/x@rzwqm2-4.6:faithful")
def shadow(state, params, forcing_t):
    """Source: fixture."""
    return state.replace(u=forcing_t.uptake + forcing_t.supply)


@process(reads=("soil_water",), writes=("soil_water",), key="soil_water/x@none:replay")
def replay(state, params, f):
    """Source: fixture."""
    return state.replace(u=f.uptake)
'''


def test_aj011_reports_forcing_that_shadows_a_port_field(tmp_path: Path) -> None:
    f = tmp_path / "processes" / "soil_water" / "m.py"
    f.parent.mkdir(parents=True)
    f.write_text(AJ011_SRC)
    got = [(x.line, x.message.rsplit(": ", 1)[1]) for x in lint.lint_file(f) if x.rule == "AJ011"]
    assert got == [(8, "shadow reads forcing_t.uptake; read it from the port")]  # supply is not a port field
    names = lint.port_field_names()
    assert {"uptake", "nstres", "swe", "trwup", "no3"} <= names and "tmin" not in names  # State records only
    assert lint.main([str(f), "--strict"]) == 0 and lint.main([str(f), "--strict-aj011"]) == 1


def test_aj009_to_aj011_on_the_package() -> None:
    """The tree has no AJ009 finding but the numerical-settings registry (gap G24) and the Richards
    integrator registry (``soil_water/integrator.py`` ``INTEGRATORS``, W1; a registry outside core,
    closed with G24) and no AJ010; the AJ011 findings are exactly gap G1 (the soil-water processes'
    uptake from the forcing)."""
    src = Path(lint.__file__).resolve().parents[1]
    found = [f for f in lint.lint_paths([src]) if f.rule in ("AJ009", "AJ010", "AJ011")]
    by_rule = {
        r: sorted({Path(f.path).name for f in found if f.rule == r}) for r in ("AJ009", "AJ010", "AJ011")
    }
    assert by_rule == {
        "AJ009": ["coefficients.py", "integrator.py"],
        "AJ010": [],
        "AJ011": ["day.py", "richards.py"],
    }


def test_process_calls_outside_processes_are_linted_as_processes(tmp_path: Path) -> None:
    src = HEADER + textwrap.dedent('''
    def _entry(state, params, forcing_t):
        """Source: fixture."""
        if state.x > 0:
            return state
        return state

    ENTRY = process(_entry, reads=("a",), writes=("a",), name="a.b", register=False)
    ''')
    for where in ("models", "forcing"):
        f = tmp_path / where / "m.py"
        f.parent.mkdir(parents=True)
        f.write_text(src)
        assert "AJ001" in {x.rule for x in lint.lint_file(f)}, where
    g = tmp_path / "testing" / "m.py"
    g.parent.mkdir(parents=True)
    g.write_text(src)
    assert lint.lint_file(g) == []  # fixtures elsewhere are not assembly entries


def test_forcing_preprocessing_gets_the_labelling_rule_only(tmp_path: Path) -> None:
    """``forcing/`` is host-side NumPy run before the day: a validation branch on an argument is
    fine there, a bare coefficient is not."""
    f = tmp_path / "forcing" / "m.py"
    f.parent.mkdir(parents=True)
    f.write_text(
        "import numpy as np\n"
        "def prep(x):\n"
        "    if np.any(x < 0):\n"
        "        raise ValueError('negative')\n"
        "    return x * 0.37\n"
    )
    assert [(x.rule, x.line) for x in lint.lint_file(f)] == [("AJ007", 5)]
    assert lint.HOST_RULES == {"AJ007"}


# ---------------------------------------------------------------------------
# AJ012: io below the processes, site assembly (agrijax.sites) above both
# ---------------------------------------------------------------------------

#: the import cycle of the tree before D2-1b, file by file: a process module that reads DSSAT
#: files (processes/crop/ceres_maize/dssat_inputs.py:32), a reader that builds process records
#: (io/catpa_m3.py:314-438) and core readers that import io (core/events.py:515,538)
AJ012_PROCESS_READS_IO = """
from agrijax.io.dssat import read_eco, read_out
from ._util import daylength
from agrijax.core.state import State
from agrijax.iface.surface import DailyWeather


def f():
    from agrijax.sites.catpa_m3 import season_table
    import agrijax.io.catpa
    from agrijax import io
"""
AJ012_IO_BUILDS_PROCESS_RECORDS = """
import numpy as np
from .dssat.genotype import read_cul
from ..rzwqm.dat import RzwqmDat
from agrijax.core.events import EventTable
from agrijax.forcing.radiation import horizontal_radiation


def f():
    from agrijax.processes.soil_water.infiltration import StormForcing
    from agrijax.iface.surface import DailyWeather
    from agrijax.port.run_fortran import run_dscsm
    from agrijax.sites import catpa_m3
    from agrijax.models import day_dssat486
    from ... import processes
"""
AJ012_CORE_READS_IO = """
def from_csv(path):
    from agrijax.io.catpa import load_events
    from agrijax.io.rzwqm.events import frame_records
    import importlib
    return importlib.import_module("agrijax.io.catpa")
"""


def _aj012(path: Path) -> list[int]:
    out = []
    for f in lint.lint_file(path):
        if f.rule == "AJ012":
            assert f.level == "warning"
            out.append(f.line)
    return out


def _write(tmp_path: Path, rel: str, src: str) -> Path:
    f = tmp_path.joinpath(*rel.split("/"))
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(src)
    return f


def test_aj012_processes_import_no_io_and_no_site(tmp_path: Path) -> None:
    f = _write(tmp_path, "src/agrijax/processes/crop/ceres_maize/dssat_inputs.py", AJ012_PROCESS_READS_IO)
    # io and sites at module level, inside a function, as ``from agrijax import io``
    assert _aj012(f) == [2, 9, 10, 11]
    msg = next(x.message for x in lint.lint_file(f) if x.rule == "AJ012")
    assert "processes imports agrijax.io.dssat" in msg
    # a slot plugin under a processes/ directory outside agrijax/ is of the processes layer too
    g = _write(tmp_path, "plugin/processes/crop/m.py", AJ012_PROCESS_READS_IO)
    assert _aj012(g) == [2, 9, 10, 11]
    for layer in ("models", "iface", "forcing"):
        h = _write(tmp_path, f"agrijax/{layer}/m.py", AJ012_PROCESS_READS_IO)
        assert _aj012(h) == [2, 9, 10, 11], layer
    assert lint.main([str(f), "--strict"]) == 1
    assert lint.main([str(f), "--strict", "--ignore", "AJ012"]) == 0


def test_aj012_io_imports_only_core_forcing_port_and_io(tmp_path: Path) -> None:
    f = _write(tmp_path, "src/agrijax/io/dssat/catpa_m3.py", AJ012_IO_BUILDS_PROCESS_RECORDS)
    # relative imports inside io, core, forcing and port pass; processes (absolute, relative),
    # iface, sites and models are reported
    assert _aj012(f) == [10, 11, 13, 14, 15]
    msg = next(x.message for x in lint.lint_file(f) if x.rule == "AJ012")
    assert "io imports agrijax.processes.soil_water.infiltration" in msg and "agrijax.sites" in msg


def test_aj012_core_imports_no_io(tmp_path: Path) -> None:
    f = _write(tmp_path, "agrijax/core/events.py", AJ012_CORE_READS_IO)
    assert _aj012(f) == [3, 4, 6]  # lazy imports and a literal dynamic import count


def test_aj012_the_site_layer_and_the_layers_above_import_both(tmp_path: Path) -> None:
    for rel in (
        "agrijax/sites/catpa_m3.py",
        "agrijax/calib/ceres.py",
        "tests/unit/test_x.py",
        "scripts/x.py",
    ):
        f = _write(tmp_path, rel, AJ012_PROCESS_READS_IO + AJ012_IO_BUILDS_PROCESS_RECORDS)
        assert _aj012(f) == [], rel
    assert "AJ012" in lint.FILE_RULES and "AJ012" not in lint.NOT_STRICT_RULES


def test_aj012_the_package_has_no_io_processes_cycle() -> None:
    """The tree has no AJ012 finding, and statically (every import statement, also inside
    functions): no module of ``agrijax.core`` or ``agrijax.processes`` reaches ``agrijax.io``, and
    ``agrijax.io`` reaches no ``agrijax.processes`` module."""
    src = Path(lint.__file__).resolve().parents[1]
    found = [f for f in lint.lint_paths([src]) if f.rule == "AJ012"]
    assert found == [], "\n".join(f.format() for f in found)
    root = src.parent
    for pkg in ("core", "processes"):
        mods = [
            ".".join(p.relative_to(root).with_suffix("").parts).removesuffix(".__init__")
            for p in sorted((src / pkg).rglob("*.py"))
        ]
        closure = lint.import_closure(mods, root)
        assert sorted(m for m in closure if m.startswith(("agrijax.io", "agrijax.sites"))) == [], pkg
    io_mods = [
        ".".join(p.relative_to(root).with_suffix("").parts).removesuffix(".__init__")
        for p in sorted((src / "io").rglob("*.py"))
    ]
    closure = lint.import_closure(io_mods, root)
    assert sorted(m for m in closure if m.startswith(("agrijax.processes", "agrijax.sites"))) == []
    # not vacuous: the site layer reaches both
    site = lint.import_closure(["agrijax.sites.catpa_m3"], root)
    assert "agrijax.io.rzwqm.dat" in site and "agrijax.processes.crop.ceres_maize.state" in site


# ---------------------------------------------------------------------------
# AJ020 / AJ021: NumPy on traced values and in-place mutation of an argument (rule 1, a pure
# function of (state, params, forcing) -> state; both were missed before, usability review 09-29)
# ---------------------------------------------------------------------------

REVIEWER = '''
@process(reads=("crop.lai",), writes=("crop.lai",))
def reviewer(state, params, forcing_t):
    """The review's deliberately bad process: NumPy on the state, then the input changed in place.

    Source: fixture.
    """
    lai = np.exp(state["crop"]["lai"]) * params.k
    state["crop"]["lai"] = lai
    return state
'''


def _new_rules(src: str, path: str = "fixture.py", header: str = HEADER) -> list[tuple[str, str, str]]:
    """``(rule, source line, message)`` of the AJ020/AJ021 findings on ``header + src``."""
    text = header + textwrap.dedent(src)
    lines = text.splitlines()
    return [
        (f.rule, lines[f.line - 1].strip(), f.message)
        for f in lint.lint_source(text, path)
        if f.rule in {"AJ020", "AJ021"}
    ]


def test_aj020_aj021_catch_the_reviewer_process(tmp_path: Path) -> None:
    found = _new_rules(REVIEWER)
    assert [(r, ln) for r, ln, _ in found] == [
        ("AJ020", 'lai = np.exp(state["crop"]["lai"]) * params.k'),
        ("AJ021", 'state["crop"]["lai"] = lai'),
    ]
    assert "np.exp(...) on state; use jnp.exp" in found[0][2]
    assert "mutates the argument state in place" in found[1][2]
    assert "eqx.tree_at(lambda s: s['crop']['lai'], state, value)" in found[1][2]
    assert lint.RULES["AJ020"][0] == lint.RULES["AJ021"][0] == "error"
    assert {"AJ020", "AJ021"} <= lint.KERNEL_RULES and not {"AJ020", "AJ021"} & lint.HOST_RULES
    f = tmp_path / "bad.py"
    f.write_text(HEADER + textwrap.dedent(REVIEWER))
    assert lint.main([str(f)]) == 1  # errors fail without --strict
    assert lint.main([str(f), "--ignore", "AJ020", "--ignore", "AJ021"]) == 0


AJ020_BAD = '''
import numpy
import numpy.linalg as la
from numpy import exp as np_exp


@process(writes=("theta",))
def numpy_on_traced(state, params, forcing_t):
    """NumPy on the arguments and on names derived from them.

    Source: fixture.
    """
    w = state.theta * params.k
    a = np.cumsum(np.asarray(w))
    b = numpy.linalg.norm(forcing_t.rain) + la.norm(w)
    c = np_exp(w)
    d = np.random.normal(params.mu)
    e = np.float32(state.theta)
    return eqx.tree_at(lambda s: s.theta, state, a + b + c + d + e)
'''


def test_aj020_reports_numpy_on_traced_values() -> None:
    found = _new_rules(AJ020_BAD)
    assert {r for r, _, _ in found} == {"AJ020"}
    by_line = [(ln.split(" = ")[0], m.split("; use ")[1].split(" (")[0]) for _, ln, m in found]
    assert by_line == [
        ("a", "jnp.cumsum"),  # one finding for np.cumsum(np.asarray(w)), the outermost call
        ("b", "jnp.linalg.norm"),
        ("b", "jnp.linalg.norm"),  # numpy.linalg.norm and la.norm (import numpy.linalg as la)
        ("c", "jnp.exp"),  # from numpy import exp as np_exp
        ("d", "jax.random with an explicit key"),
        ("e", "jnp.float32"),
    ]
    # np.random is reported whatever its arguments: a draw at trace time is a constant under jit
    assert all("a @process always runs traced" in m for _, ln, m in found if "random" not in ln)
    assert any("draws once at trace time" in m for _, _, m in found)


AJ020_OK = '''
_TABLE = (0.1, 0.2, 0.3)


@process(writes=("theta",))
def numpy_allowed(state, params, forcing_t, n_iter: int = 3):
    """NumPy for trace-time constants, shape queries and dtypes only.

    Source: fixture.
    """
    t = state.theta
    n = t.shape[-1]
    idx = np.arange(n)
    colours = np.stack([idx % 3 == k for k in range(3)])
    table = np.asarray(_TABLE)
    zeros = np.zeros((len(t), 2)) + np.ones(n_iter) + np.arange(params.n_layer)[0]
    eps = np.finfo(t.dtype).eps + np.ndim(t) + np.shape(t)[0] + np.size(t)
    w = jnp.exp(t).astype(np.float32) * np.pi
    u = jnp.asarray(w, dtype=np.float64)
    return eqx.tree_at(lambda s: s.theta, state, u + table[0] + zeros[0, 0] + eps + colours[0, 0] + idx[0])
'''


def test_aj020_allows_constants_shape_queries_and_dtypes() -> None:
    assert _new_rules(AJ020_OK) == []
    jnp_as_np = '''
    import equinox as eqx
    import jax.numpy as np


    @process(writes=("theta",))
    def jnp_named_np(state, params, forcing_t):
        """``np`` is jax.numpy in this file.

        Source: fixture.
        """
        return eqx.tree_at(lambda s: s.theta, state, np.exp(state.theta))
    '''
    assert _new_rules(jnp_as_np, header="") == []


def test_numpy_names_reads_the_imports() -> None:
    tree = ast.parse(
        "import numpy as onp\nimport numpy.linalg as la\nfrom numpy import exp, log as ln\nimport jax.numpy as np\n"
    )
    assert lint.numpy_names(tree) == {"numpy": "", "onp": "", "la": "linalg", "exp": "exp", "ln": "log"}
    assert lint.numpy_names(ast.parse("x = 1\n")) == {"np": "", "numpy": ""}


AJ020_KERNELS = '''
import jax
import jax.numpy as jnp
import numpy as np


def traced_kernel(h, soil):
    return jnp.maximum(np.exp(h / soil.hb), 0.0)


def from_records(rec, tl):
    """Host-side: build a record from the file reader's arrays (NumPy, before any trace)."""
    return np.cumsum(np.asarray(rec, dtype=float)) + np.sum(tl)


def _host_mean(w):
    return np.asarray(w).mean(axis=0)


def with_callback(w):
    return jax.pure_callback(_host_mean, jax.ShapeDtypeStruct(w.shape[1:], w.dtype), w)
'''


def test_aj020_on_kernels_and_the_host_side_exemption(tmp_path: Path) -> None:
    f = tmp_path / "processes" / "soil_water" / "k.py"
    f.parent.mkdir(parents=True)
    f.write_text(AJ020_KERNELS)
    found = [x for x in lint.lint_file(f) if x.rule == "AJ020"]
    # the traced kernel only: a docstring that says host-side and a pure_callback target are host code
    assert [x.line for x in found] == [8]
    assert "use jnp.exp" in found[0].message and "starts its docstring with 'Host-side:'" in found[0].message
    # outside processes/ the kernels are not linted; with --all every function is, the marker still holds
    g = tmp_path / "io" / "k.py"
    g.parent.mkdir()
    g.write_text(AJ020_KERNELS)
    assert lint.lint_file(g) == []
    assert [x.line for x in lint.lint_file(g, all_functions=True) if x.rule == "AJ020"] == [8]
    # a @process always runs traced: saying host-side does not exempt it
    claims_host = '''
    @process(writes=("theta",))
    def claims_host(state, params, forcing_t):
        """Host-side, it says, but a process always runs traced.

        Source: fixture.
        """
        return eqx.tree_at(lambda s: s.theta, state, np.exp(state.theta))
    '''
    assert [r for r, _, _ in _new_rules(claims_host)] == ["AJ020"]
    # a module docstring that says host-side covers every kernel of the module, not its processes
    h = tmp_path / "processes" / "soil_water" / "records.py"
    h.write_text('"""Host-side: readers of the file records (NumPy, not traced)."""\n' + AJ020_KERNELS)
    assert [x for x in lint.lint_file(h) if x.rule == "AJ020"] == []
    module_host = '"""Host-side: helpers."""\n' + HEADER + textwrap.dedent(claims_host)
    assert [x.rule for x in lint.lint_source(module_host, "fixture.py") if x.rule == "AJ020"] == ["AJ020"]


def test_aj020_host_marker_is_what_keeps_the_record_builders_clean() -> None:
    """``RichardsGrid.from_rzwqm`` (file records -> grid) uses NumPy on its arguments under
    ``processes/``; it is clean because its docstring declares it host-side, and reported without."""
    path = Path(lint.__file__).resolve().parents[1] / "processes" / "soil_water" / "problem.py"
    text = path.read_text(encoding="utf-8")
    assert [f for f in lint.lint_source(text, str(path)) if f.rule == "AJ020"] == []
    fn = next(
        n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == "from_rzwqm"
    )
    found = [f for f in lint.lint_source(text.replace("Host-side", "Built"), str(path)) if f.rule == "AJ020"]
    assert found and all(fn.lineno <= f.line <= (fn.end_lineno or 0) for f in found)


AJ021_BAD = '''
@process(writes=("crop", "theta"))
def mutates(state, params, forcing_t):
    """Every way of changing an input in place.

    Source: fixture.
    """
    state.crop.lai = forcing_t.rain
    state.crop.lai += 1.0
    setattr(state, "water", 0.0)
    object.__setattr__(state.crop, "lai", 2.0)
    del params.extra
    crop = state.crop
    crop.lai = 3.0
    theta = state.theta
    theta[0] = 0.0
    theta.fill(0.0)
    state.theta.sort()
    params.table.update({"a": 1})
    np.copyto(state.theta, forcing_t.rain)
    np.add.at(state.theta, 0, 1.0)
    jnp.exp(forcing_t.rain, out=state.theta)
    soil, weather = state.soil, forcing_t
    soil["sw"] = 0.0
    weather.rain = 0.0
    part = getattr(state, "roots")
    part.rlv[0] = 0.0
    for layer in (state.top, state.bottom):
        layer.w = 0.0
    if forcing_t.flag is None:
        theta = theta.copy()
    theta[1] = 0.0
    return state
'''


def test_aj021_reports_in_place_mutation_of_arguments() -> None:
    found = [(ln, m) for r, ln, m in _new_rules(AJ021_BAD) if r == "AJ021"]
    assert [ln for ln, _ in found] == [
        "state.crop.lai = forcing_t.rain",
        "state.crop.lai += 1.0",
        'setattr(state, "water", 0.0)',
        'object.__setattr__(state.crop, "lai", 2.0)',
        "del params.extra",
        "crop.lai = 3.0",
        "theta[0] = 0.0",
        "theta.fill(0.0)",
        "state.theta.sort()",
        'params.table.update({"a": 1})',
        "np.copyto(state.theta, forcing_t.rain)",
        "np.add.at(state.theta, 0, 1.0)",
        "jnp.exp(forcing_t.rain, out=state.theta)",
        'soil["sw"] = 0.0',
        "weather.rain = 0.0",
        "part.rlv[0] = 0.0",
        "layer.w = 0.0",
        "theta[1] = 0.0",  # a rebinding inside a branch may not happen: theta is still the state's array
    ]
    msg = dict(found)
    assert "the argument state in place;" in msg["state.crop.lai = forcing_t.rain"]
    assert "eqx.tree_at(lambda s: s.crop.lai, state, value)" in msg["state.crop.lai = forcing_t.rain"]
    assert "the argument params in place" in msg["del params.extra"]
    assert "the argument state in place (through crop)" in msg["crop.lai = 3.0"]
    assert "the argument forcing_t in place (through weather)" in msg["weather.rain = 0.0"]
    assert "theta.at[0].set(...)" in msg["theta[0] = 0.0"]
    assert "jnp.sort(x)" in msg["state.theta.sort()"]
    assert "new container" in msg['params.table.update({"a": 1})']
    assert "x.at[i].set(v)" in msg["np.copyto(state.theta, forcing_t.rain)"]


AJ021_OK = '''
@process(writes=("theta",))
def builds_new_values(state, params, forcing_t, *extra, **options):
    """Changing what the function built itself is fine; so are the functional updates.

    Source: fixture.
    """
    out = np.zeros(3)
    out[0] = 1.0
    buf = []
    buf.append(state.theta)
    d = {"theta": state.theta}
    d["rain"] = forcing_t.rain
    theta = state.theta.at[0].set(1.0)
    theta = theta.at[1].add(2.0)
    ordered = state.theta.sort() + jnp.sort(state.theta)
    options.pop("unused", None)
    options["seen"] = True
    head, _, tail = "a.b".partition(".")
    new = state.replace(theta=theta + ordered + out[0])
    return new
'''

AJ021_HOST = '''
import numpy as np


def clipped(x):
    """Host-side: a copy is changed, not the caller's array."""
    x = np.array(x, dtype=float)
    x[x < 0.0] = 0.0
    return x


def clipped_in_place(x):
    """Host-side: np.asarray returns the caller's array itself when no conversion is needed."""
    x = np.asarray(x)
    x[x < 0.0] = 0.0
    return x


class Grid:
    def __init__(self, tl):
        self.tl = tl
        object.__setattr__(self, "n", len(tl))
'''


def test_aj021_allows_local_values_functional_updates_and_self(tmp_path: Path) -> None:
    assert _new_rules(AJ021_OK) == []
    f = tmp_path / "processes" / "soil_water" / "host.py"
    f.parent.mkdir(parents=True)
    f.write_text(AJ021_HOST)
    found = lint.lint_file(f)
    # AJ021 applies to host-side kernels too; only the alias made by np.asarray is reported
    assert [(x.rule, x.line) for x in found] == [("AJ021", 15)], [x.format() for x in found]
    assert "the argument x in place" in found[0].message


# ---------------------------------------------------------------------------
# AJ020 / AJ021 review probes: one case per evasion or false positive found in review (fixtures
# linted as code under processes/, expected set of AJ020 / AJ021 findings)
# ---------------------------------------------------------------------------

_PHDR = "import numpy as np\nimport jax\nimport jax.numpy as jnp\nimport equinox as eqx\nfrom agrijax.core.process import process\n"


def _p(body: str) -> str:
    """``body`` as the body of a process ``p(state, params, forcing)``."""
    return "@process(reads=('x',), writes=('x',))\ndef p(state, params, forcing):\n" + textwrap.indent(
        textwrap.dedent(body), "    "
    )


def _k(body: str) -> str:
    """``body`` as the body of a kernel ``k(x, y, out=None)``."""
    return "def k(x, y, out=None):\n" + textwrap.indent(textwrap.dedent(body), "    ")


_R = "return state\n"
_H = '"""Host-side: a helper on concrete values."""\n'

# (id, header, source, expected rules)
PROBES: list[tuple[str, str, str, set[str]]] = [
    # ---- AJ020: NumPy reached through aliases, values, lambdas, comprehensions, self
    ("alias xp", "import numpy as xp\n", _p("return state.replace(x=xp.exp(state.x))"), {"AJ020"}),
    ("from numpy import", "from numpy import exp as e\n", _p("return state.replace(x=e(state.x))"), {"AJ020"}),
    ("getattr(np, name)", _PHDR, _p("return state.replace(x=getattr(np, 'exp')(state.x))"), {"AJ020"}),
    ("np.vectorize bound", _PHDR, _p("f = np.vectorize(lambda t: t + 1.0)\nreturn state.replace(x=f(state.x))"), {"AJ020"}),
    ("np.vectorize inline", _PHDR, _p("return state.replace(x=np.vectorize(float)(state.x))"), {"AJ020"}),
    ("np.frompyfunc", _PHDR, _p("return state.replace(x=np.frompyfunc(abs, 1, 1)(state.x))"), {"AJ020"}),
    ("importlib numpy", "import importlib\n_np = importlib.import_module('numpy')\n", _p("return state.replace(x=_np.exp(state.x))"), {"AJ020"}),
    ("module alias N = np", _PHDR + "N = np\n", _p("return state.replace(x=N.exp(state.x))"), {"AJ020"}),
    ("local alias m = np", _PHDR, _p("m = np\nreturn state.replace(x=m.exp(state.x))"), {"AJ020"}),
    ("local f = np.exp", _PHDR, _p("f = np.exp\nreturn state.replace(x=f(state.x))"), {"AJ020"}),
    ("module _EXP = np.exp", _PHDR + "_EXP = np.exp\n", _p("return state.replace(x=_EXP(state.x))"), {"AJ020"}),
    ("np.ndarray.sum unbound", _PHDR, _p("return state.replace(x=np.ndarray.sum(state.x))"), {"AJ020"}),
    ("lambda body", _PHDR, _p("f = lambda t: np.exp(t)\nreturn state.replace(x=f(state.x))"), {"AJ020"}),
    ("lambda closure", _PHDR, _p("f = lambda: np.exp(state.x)\nreturn state.replace(x=f())"), {"AJ020"}),
    ("vmap lambda", _PHDR, _p("return state.replace(x=jax.vmap(lambda t: np.exp(t))(state.x))"), {"AJ020"}),
    ("scan lambda", _PHDR, _p("x, _ = jax.lax.scan(lambda c, _: (np.exp(c), None), state.x, None, length=2)\nreturn state.replace(x=x)"), {"AJ020"}),
    ("comprehension target", _PHDR, _p("ys = [np.exp(v) for v in (state.x, state.y)]\nreturn state.replace(x=ys[0])"), {"AJ020"}),
    ("tree_map(np.asarray, state)", _PHDR, _p("return jax.tree_util.tree_map(np.asarray, state)"), {"AJ020"}),
    ("map(np.exp, ...)", _PHDR, _p("ys = list(map(np.exp, (state.x, state.y)))\nreturn state.replace(x=ys[0])"), {"AJ020"}),
    ("vmap(np.sum)(x)", _PHDR, _p("return state.replace(x=jax.vmap(np.sum)(state.x))"), {"AJ020"}),
    ("np.random always", _PHDR, _p("return state.replace(x=state.x + np.random.normal())"), {"AJ020"}),
    ("self of a Module", _PHDR, "class G(eqx.Module):\n    tl: jax.Array\n    def depth(self):\n        return np.cumsum(self.tl)\n", {"AJ020"}),
    ("self static field", _PHDR, "class G(eqx.Module):\n    n: int = eqx.field(static=True)\n    def idx(self):\n        return jnp.asarray(np.arange(self.n))\n", set()),
    ("local jnp import elsewhere", _PHDR, "def other(x):\n    import jax.numpy as np\n    return np.exp(x)\n\n" + _p("return state.replace(x=np.exp(state.x))"), {"AJ020"}),
    ("TYPE_CHECKING does not run", "import numpy as np\nfrom typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from jax import numpy as np\n", _p("return state.replace(x=np.exp(state.x))"), {"AJ020"}),
    ("local name np shadows", "import jax.numpy as jnp\n", _p("np = params.backend\nreturn state.replace(x=np.exp(state.x))"), set()),
    ("jax.numpy as np", "from jax import numpy as np\n", _p("return state.replace(x=np.exp(state.x))"), set()),
    # ---- AJ020 allowed: constants, shapes, dtypes, guards, callbacks
    ("np.shape static", _PHDR, _p("n = np.shape(state.x)[-1]\nreturn state.replace(x=jnp.zeros(n))"), set()),
    ("np.arange of a shape", _PHDR, _p("i = np.arange(state.x.shape[-1])\nreturn state.replace(x=state.x[i])"), set()),
    ("dtype values", _PHDR, _p("return state.replace(x=jnp.asarray(state.x, dtype=np.float32).astype(np.float64) * np.pi)"), set()),
    ("grid size", _PHDR, _p("return state.replace(x=jnp.asarray(np.ones(params.grid.n_node)))"), set()),
    ("Tracer guard", _PHDR + "from jax.core import Tracer\n", "def k(tl):\n    if isinstance(tl, Tracer):\n        return None\n    return np.sum(np.asarray(tl))\n", set()),
    ("module RNG constant", _PHDR + "_RNG = np.random.default_rng(0)\n", _p("return state.replace(x=state.x + _RNG.normal())"), set()),
    ("callback via partial", _PHDR + "import functools\n", "def cb(x, k):\n    return np.exp(x) * k\n\n" + _p("y = jax.pure_callback(functools.partial(cb, k=2.0), state.x, state.x)\nreturn state.replace(x=y)"), set()),
    ("from jax.debug import callback", _PHDR + "from jax.debug import callback\n", "def cb(x):\n    print(np.asarray(x))\n\n" + _p("callback(cb, state.x)\nreturn state"), set()),
    ("numpy callback function", _PHDR, _p("return state.replace(x=jax.pure_callback(np.sort, state.x, state.x))"), set()),
    # ---- host marker: structured, no loophole
    ("@process says Host-side:", _PHDR, '@process(reads=("x",), writes=("x",))\ndef p(state, params, forcing):\n    """Host-side: no."""\n    return state.replace(x=np.exp(state.x))\n', {"AJ020"}),
    ("process in a host module", '"""Host-side: helpers."""\n' + _PHDR, _p("return state.replace(x=np.exp(state.x))"), {"AJ020"}),
    ("traced call into a host helper", _PHDR, "def _helper(x):\n    " + _H + "    return np.exp(x)\n\n" + _p("return state.replace(x=_helper(state.x))"), {"AJ020"}),
    ("host helper on constants", _PHDR + "_T = (1.0, 2.0)\n", "def _helper(x):\n    " + _H + "    return np.exp(x)\n\n" + _p("return state.replace(x=state.x + _helper(_T))"), set()),
    ("negated mention", _PHDR, 'def k(x):\n    """This kernel is not host-side: it runs under jit."""\n    return np.exp(x)\n', {"AJ020"}),
    ("module mention in passing", '"""Traced kernels. The host-side readers live in agrijax.io."""\n' + _PHDR, "def k(x):\n    return np.exp(x)\n", {"AJ020"}),
    ("callback called directly", _PHDR, "def cb(x):\n    return np.exp(x)\n\n" + _p("y = jax.pure_callback(cb, state.x, state.x)\nreturn state.replace(x=cb(state.x) + y)"), {"AJ020"}),
    ("nested def in a host function", _PHDR, "def k(x):\n    " + _H + "    def inner(y):\n        return np.exp(y)\n    return inner(x)\n", set()),
    ("host function still gets AJ021", _PHDR, "def k(x):\n    " + _H + "    x[0] = 1\n    return x\n", {"AJ021"}),
    # ---- AJ021: forms of in-place change
    ("vars(state)[k] = v", _PHDR, _p("vars(state)['k'] = 1\n" + _R), {"AJ021"}),
    ("dict.update(state, d)", _PHDR, _p("dict.update(state, {'k': 1})\n" + _R), {"AJ021"}),
    ("dict.__setitem__", _PHDR, _p("dict.__setitem__(state, 'k', 1)\n" + _R), {"AJ021"}),
    ("list.append(state.xs, v)", _PHDR, _p("list.append(state.xs, 1)\n" + _R), {"AJ021"}),
    ("np.ndarray.fill(x, 0)", _PHDR, _p("np.ndarray.fill(state.x, 0)\n" + _R), {"AJ021"}),
    ("operator.setitem", _PHDR + "import operator\n", _p("operator.setitem(state, 'k', 1)\n" + _R), {"AJ021"}),
    ("from operator import setitem", _PHDR + "from operator import setitem\n", _p("setitem(state.x, 0, 1)\n" + _R), {"AJ021"}),
    ("operator.iadd", _PHDR + "import operator\n", _p("operator.iadd(state.xs, [1])\n" + _R), {"AJ021"}),
    ("x += on an array alias", _PHDR, _p("sw = state.sw\nsw += forcing.rain\nreturn state.replace(sw=sw)"), set()),
    ("x *= on a kernel argument", _PHDR, "def k(x, dt):\n    dt *= 0.5\n    return x * dt\n", set()),
    ("lst += [...] on an alias", _PHDR, _p("lst = state['log']\nlst += [1]\n" + _R), {"AJ021"}),
    ("alias only on a returning branch", _PHDR, _p("a = np.zeros(3)\nif params['c']:\n    a = state['x']\n    return state\na[0] = 1\n" + _R), set()),
    ("alias rebound in one branch", _PHDR, _p("a = state['x']\nif params['c']:\n    a = np.zeros(3)\na[0] = 1\n" + _R), {"AJ021"}),
    ("nested host def called", _PHDR, _p('def inner(y):\n    """Host-side: numpy."""\n    return np.exp(y)\nreturn state.replace(x=inner(state.x))'), {"AJ020"}),
    ("host fn bound to a name", _PHDR, "def h(x):\n    " + _H + "    return np.exp(x)\n\n" + _p("g = h\nreturn state.replace(x=g(state.x))"), {"AJ020"}),
    ("host fn passed to vmap", _PHDR, "def h(x):\n    " + _H + "    return np.exp(x)\n\n" + _p("return state.replace(x=jax.vmap(h)(state.x))"), {"AJ020"}),
    ("host fn passed to tree_map", _PHDR, "def h(x):\n    " + _H + "    return np.exp(x)\n\n" + _p("return jax.tree_util.tree_map(h, state)"), {"AJ020"}),
    ("host fn through partial", _PHDR + "import functools\n", "def h(x, a):\n    " + _H + "    return np.exp(x)\n\n" + _p("return state.replace(x=functools.partial(h, a=1)(state.x))"), {"AJ020"}),
    ("host method from a traced method", _PHDR, "class G(eqx.Module):\n    tl: jax.Array\n    def build(self):\n        " + _H + "        return np.cumsum(self.tl)\n    def depth(self):\n        return self.build()\n", {"AJ020"}),
    ("callback=lambda keyword", _PHDR, _p("return state.replace(x=jax.pure_callback(callback=lambda v: np.sort(v), result_shape_dtypes=None, x=state.x))"), set()),
    ("d |= on an alias", _PHDR, _p("d = state['crop']\nd |= {'lai': 1}\n" + _R), {"AJ021"}),
    ("argument += directly", _PHDR, _p("forcing += 1.0\n" + _R), {"AJ021"}),
    ("x.__iadd__", _PHDR, _p("state.x.__iadd__(1)\n" + _R), {"AJ021"}),
    ("lambda __setitem__", _PHDR, _p("(lambda: state.__setitem__('k', 1))()\n" + _R), {"AJ021"}),
    ("bound lambda setattr", _PHDR, _p("g = lambda s: setattr(s, 'k', 1)\ng(state)\n" + _R), {"AJ021"}),
    ("tree_map lambda fill", _PHDR, _p("jax.tree_util.tree_map(lambda a: a.fill(0.0), state)\n" + _R), {"AJ021"}),
    ("tree_map lambda copyto", _PHDR, _p("jax.tree_util.tree_map(lambda a, b: np.copyto(a, b), state, params)\n" + _R), {"AJ020", "AJ021"}),
    ("alias from one branch", _PHDR, _p("if forcing.flag:\n    x = state['crop']\nelse:\n    x = {}\nx['lai'] = 1\n" + _R), {"AJ021"}),
    ("comprehension update", _PHDR, _p("[d.update(k=1) for d in state.layers]\n" + _R), {"AJ021"}),
    ("dict(...) element", _PHDR, _p("crop = dict(state['crop'])\ncrop['lai'][0] = 1.0\n" + _R), {"AJ021"}),
    ("{**x} element", _PHDR, _p("crop = {**state['crop']}\ncrop['lai'][0] = 1.0\n" + _R), {"AJ021"}),
    ("copy.copy element", _PHDR + "import copy\n", _p("crop = copy.copy(state['crop'])\ncrop['lai'][0] = 1.0\n" + _R), {"AJ021"}),
    ("list(...) element", _PHDR, _p("ls = list(state.layers)\nls[0]['k'] = 1\n" + _R), {"AJ021"}),
    ("dict(state) nested set", _PHDR, _p("new = dict(state)\nnew['crop']['lai'] = 1.0\nreturn new"), {"AJ021"}),
    ("tuple element mutate", _PHDR, _p("parts = (state.x, state.y)\nparts[0][0] = 1.0\n" + _R), {"AJ021"}),
    ("np.ravel view", _PHDR, _p("a = np.ravel(state.x)\na[0] = 1\n" + _R), {"AJ020", "AJ021"}),
    ("x.flat", _PHDR, _p("a = state.x.flat\na[0] = 1\n" + _R), {"AJ021"}),
    ("match capture", _PHDR, _p("match state:\n    case {'crop': c}:\n        c['lai'] = 1\n" + _R), {"AJ021"}),
    ("star unpacking", _PHDR, _p("a, *rest = state.layers\nrest[0]['k'] = 1\n" + _R), {"AJ021"}),
    ("walrus alias", _PHDR, _p("if (c := state.crop) is not None:\n    c.lai = 1\n" + _R), {"AJ021"}),
    ("super(C, x).__setattr__", _PHDR, "class M(eqx.Module):\n    a: jax.Array\n    def f(self, other):\n        super(M, other).__setattr__('a', 1)\n        return other\n", {"AJ021"}),
    ("type(x).__setattr__", _PHDR, _p("type(state).__setattr__(state, 'k', 1)\n" + _R), {"AJ021"}),
    ("next(iter(...))", _PHDR, _p("c = next(iter(state.layers))\nc['k'] = 1\n" + _R), {"AJ021"}),
    ("try alias", _PHDR, _p("try:\n    c = state.crop\nexcept Exception:\n    c = None\nc.lai = 1\n" + _R), {"AJ021"}),
    ("*args element", _PHDR, "def k(*arrs):\n    arrs[0][0] = 1\n    return arrs\n", {"AJ021"}),
    ("**kw element", _PHDR, "def k(**kw):\n    kw['a'][0] = 1\n    return kw\n", {"AJ021"}),
    ("self of a Module mutated", _PHDR, "class M(eqx.Module):\n    a: jax.Array\n    def f(self):\n        self.a[0] = 1\n        return self\n", {"AJ021"}),
    # ---- AJ021 allowed: new objects, branches, functional methods, plain classes
    ("optional out in a branch", _PHDR, _k("if out is None:\n    out = np.zeros(3)\n    out[0] = 1.0\nreturn x + jnp.asarray(out)"), set()),
    ("optional dict in a branch", _PHDR, _k("if y is None:\n    y = {}\n    y['a'] = 1.0\nreturn x"), set()),
    ("rebind in a loop", _PHDR, _k("buf = x\nfor i in range(3):\n    buf = np.zeros(3)\n    buf[i] = 1.0\nreturn x"), set()),
    ("dict(state) top-level set", _PHDR, _p("d = dict(state)\nd['k'] = 1\nreturn type(state)(**d)"), set()),
    ("{**state} top-level set", _PHDR, _p("d = {**state}\nd['k'] = 1\nreturn d"), set()),
    ("list literal append", _PHDR, _p("parts = [state.x, 1.0]\nparts.append(2.0)\nreturn state.replace(x=jnp.stack(parts))"), set()),
    ("list literal setitem", _PHDR, _p("parts = [state.x, state.y]\nparts[0] = jnp.zeros(3)\nreturn state.replace(x=jnp.stack(parts))"), set()),
    ("[*x] rebuilt", _PHDR, _k("xs = [*x]\nxs.append(1.0)\nreturn xs"), set()),
    ("functional update method", _PHDR, _k("upd, st = y.update(x, out)\nreturn upd"), set()),
    ("functional add", _PHDR, _k("return y.add(x)"), set()),
    ("sorted copy used", _PHDR, _k("x = x.sort()\nreturn x"), set()),
    ("str.partition", _PHDR, _k("a, _, b = y.partition(':')\nreturn x"), set()),
    ("local set add", _PHDR, _k("seen = set()\nseen.add(1)\nreturn x"), set()),
    ("with target", _PHDR, _p("with ctx(state.crop) as c:\n    c.lai = 1\n" + _R), set()),
    ("host copy then set", _PHDR, "def k(x):\n    " + _H + "    a = np.array(x)\n    a[0] = 0\n    return a\n", set()),
    ("plain class self", _PHDR, "class Cursor:\n    def __init__(self):\n        self.i = 0\n    def step(self):\n        self.i += 1\n        return self.i\n", set()),
]  # fmt: skip


@pytest.mark.parametrize(("header", "src", "expected"), [c[1:] for c in PROBES], ids=[c[0] for c in PROBES])
def test_aj020_aj021_review_probes(header: str, src: str, expected: set[str]) -> None:
    found = {
        f.rule
        for f in lint.lint_source(header + src, "pkg/processes/demo/x.py")
        if f.rule in {"AJ020", "AJ021"}
    }
    assert found == expected


def test_aj020_host_helper_imported_from_another_module(tmp_path: Path) -> None:
    """A traced call into a host-side function imported with ``from <module> import <name>``."""
    pkg = tmp_path / "pkg" / "processes" / "demo"
    pkg.mkdir(parents=True)
    for d in (tmp_path / "pkg", tmp_path / "pkg" / "processes", pkg):
        (d / "__init__.py").write_text("")
    (pkg / "records.py").write_text(
        "import numpy as np\n\n\ndef read(x):\n    " + _H + "    return np.asarray(x)\n\n\n"
        "class Grid:\n    @classmethod\n    def of(cls, tl):\n        "
        + _H
        + "        return np.cumsum(tl)\n"
    )
    user = pkg / "user.py"
    user.write_text(
        _PHDR
        + "from .records import Grid, read\n\n"
        + _p(
            "a = read(state.x)\nb = Grid.of(state.y)\nc = read((1.0, 2.0))\nreturn state.replace(x=a + b + c)"
        )
    )
    found = [
        (f.rule, f.message.split(": ", 1)[1].split("(")[0]) for f in lint.lint_file(user) if f.rule == "AJ020"
    ]
    assert found == [("AJ020", "read"), ("AJ020", "Grid.of")]
    # re-exported by the package __init__: names imported from the package count the same way
    (pkg / "__init__.py").write_text("from .records import Grid, read\n")
    (pkg / "user2.py").write_text(
        _PHDR + "from . import read\n\n" + _p("return state.replace(x=read(state.x))")
    )
    assert [f.rule for f in lint.lint_file(pkg / "user2.py") if f.rule == "AJ020"] == ["AJ020"]
