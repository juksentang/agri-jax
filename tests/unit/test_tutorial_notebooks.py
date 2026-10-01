"""The tutorial notebooks are generated from one source: the committed files are up to date, the
English and the Chinese notebook have the same cells and the same code (except the language line and
the comments, which are written per language), no outputs, and their code compiles (the notebook
itself is executed by the slow tier, ``tests/integration/test_tutorial_colab.py``)."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = ROOT / "tutorial"


def _builder():
    spec = importlib.util.spec_from_file_location("build_notebooks", TUTORIAL / "build_notebooks.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load(name: str) -> dict:
    return json.loads((TUTORIAL / name).read_text())


def test_committed_notebooks_are_generated_from_the_source():
    b = _builder()
    for lang, path in b.LANGUAGES.items():
        assert path.read_text() == b.render(b.notebook(lang)), (
            f"{path.name}: run python tutorial/build_notebooks.py"
        )


def _code(cell: dict) -> str:
    """The code of a code cell with the shell and magic lines (``!pip ...``) replaced by ``pass``."""
    out = []
    for ln in "".join(cell["source"]).splitlines():
        s = ln.lstrip()
        out.append(ln[: len(ln) - len(s)] + "pass" if s.startswith(("!", "%")) else ln)
    return "\n".join(out)


def _structure(cell: dict) -> tuple[str, int]:
    """The syntax tree of a code cell without comments (the parser drops them) and docstrings, with
    the value of ``LANG`` blanked out, and the number of ``LANG`` assignments."""
    tree = ast.parse(_code(cell))
    langs = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "LANG" for t in node.targets
        ):
            node.value = ast.Constant("")
            langs += 1
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and (
            ast.get_docstring(node, clean=False) is not None
        ):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree), langs


def test_english_and_chinese_have_the_same_cells_and_code():
    en, zh = _load("agrijax_tutorial.ipynb"), _load("agrijax_tutorial_zh.ipynb")
    assert [(c["cell_type"], c["id"]) for c in en["cells"]] == [
        (c["cell_type"], c["id"]) for c in zh["cells"]
    ]
    lang_lines = 0
    for a, b in zip(en["cells"], zh["cells"], strict=True):
        if a["cell_type"] == "code":
            # the code is the same; the differences are the language of the printed messages and the comments
            (tree_a, n_a), (tree_b, n_b) = _structure(a), _structure(b)
            assert tree_a == tree_b, a["id"]
            assert n_a == n_b
            lang_lines += n_a
            assert a["outputs"] == [] == b["outputs"] and a["execution_count"] is None
        else:
            assert a["source"] != b["source"] or not "".join(a["source"]).strip()
    assert lang_lines == 1
    for nb, lang in ((en, "en"), (zh, "zh")):
        code = "\n".join(_code(c) for c in nb["cells"] if c["cell_type"] == "code")
        assert len(re.findall(rf'^LANG = "{lang}"', code, re.M)) == 1
    zh_text = "".join("".join(c["source"]) for c in zh["cells"] if c["cell_type"] == "markdown")
    assert re.search(r"[一-鿿]", zh_text)
    en_text = "".join("".join(c["source"]) for c in en["cells"] if c["cell_type"] == "markdown")
    assert not re.search(r"[一-鿿]{3}", en_text.replace("中文版", ""))


def test_badges_point_at_the_published_notebooks():
    base = "https://colab.research.google.com/github/juksentang/agri-jax/blob/main/tutorial/"
    for name in ("agrijax_tutorial.ipynb", "agrijax_tutorial_zh.ipynb"):
        first = "".join(_load(name)["cells"][0]["source"])
        assert base + "agrijax_tutorial.ipynb" in first and base + "agrijax_tutorial_zh.ipynb" in first


def test_the_notebook_code_compiles_and_uses_the_facade_only():
    code = []
    for c in _load("agrijax_tutorial.ipynb")["cells"]:
        if c["cell_type"] == "code":
            for ln in "".join(c["source"]).splitlines():
                s = ln.lstrip()
                code.append(ln[: len(ln) - len(s)] + "pass" if s.startswith(("!", "%")) else ln)
    text = "\n".join(code)
    compile(text, "agrijax_tutorial.ipynb", "exec")
    # a user of the package: the top-level facade only, no JAX settings or internals
    assert not re.search(r"from agrijax\.|import agrijax\.|jax_enable_x64|jax\.config|SLOT|exact_lags", text)
    assert "import agrijax as aj" in text
    # no speed number is written into the notebook: every time is measured and printed
    md = "".join(
        "".join(c["source"]) for c in _load("agrijax_tutorial.ipynb")["cells"] if c["cell_type"] == "markdown"
    )
    assert not re.search(r"\d+(\.\d+)?\s*(ms|x faster|times faster)", md)


def test_builder_refuses_inconsistent_sources():
    b = _builder()
    with pytest.raises(KeyError, match="no 'zh' text"):
        b.notebook("zh", "# %% [markdown] a\n", {"a": {"en": "x"}})
    with pytest.raises(KeyError, match="without a cell"):
        b.notebook("en", "# %% [markdown] a\n", {"a": {"en": "x"}, "b": {"en": "y"}})
    with pytest.raises(ValueError, match="without a key"):
        b.cells("# %% [markdown]\n")
    assert b.cells("# header\n# %%\nx = 1\n\n# %% [markdown] k\n") == [("code", "x = 1"), ("markdown", "k")]


_COMMENTS = {
    "trail": {"en": "EN trail", "zh": "ZH trail"},
    "whole": {"en": "EN whole line", "zh": "ZH whole line"},
    "long": {"en": "EN first\n\nEN third", "zh": "ZH first\n\nZH third"},
    "doc": {"en": "EN docstring", "zh": "ZH docstring"},
}
_CODE = '''\
x = 1  # @c:trail
if x:
    # @c:whole
    y = 2
    # @c:long


def f():
    """@c:doc"""
    return 1


def g():
    """@c:long"""
'''
_EXPECTED_EN = '''\
x = 1  # EN trail
if x:
    # EN whole line
    y = 2
    # EN first
    #
    # EN third


def f():
    """EN docstring"""
    return 1


def g():
    """EN first

    EN third
    """
'''


def test_comment_markers_are_replaced_by_the_text_of_each_language():
    b = _builder()
    used: set[str] = set()
    en = b.localize(_CODE, "en", _COMMENTS, used)
    assert used == set(_COMMENTS)
    assert en == _EXPECTED_EN
    assert b.localize(_CODE, "zh", _COMMENTS) == _EXPECTED_EN.replace("EN", "ZH")
    # the result is valid Python with the docstrings of the language
    ns: dict = {}
    exec(compile(en, "localized", "exec"), ns)
    assert ns["f"].__doc__ == "EN docstring" and ns["g"].__doc__ == "EN first\n\n    EN third\n    "


def test_comment_markers_in_a_notebook_cell_but_not_in_the_header():
    b = _builder()
    src = '# header that mentions # @c:nothing\n# %%\nLANG = "{LANG}"  # @c:trail\n'
    table = {"comment": {"trail": _COMMENTS["trail"]}}
    assert b.notebook("en", src, table)["cells"][0]["source"] == ['LANG = "en"  # EN trail']
    assert b.notebook("zh", src, table)["cells"][0]["source"] == ['LANG = "zh"  # ZH trail']


def test_comment_markers_refuse_what_cannot_be_substituted():
    b = _builder()
    one = {"k": {"en": "text", "zh": "text"}}
    for code, table in (("x = 1  # @c:nope", one), ("    # @c:nope", {}), ('"""@c:nope"""', one)):
        with pytest.raises(KeyError, match=r"no \[comment\.nope\]"):
            b.localize(code, "en", table)
    with pytest.raises(KeyError, match="no 'zh' text for comment 'k'"):
        b.localize("x = 1  # @c:k", "zh", {"k": {"en": "text"}})
    with pytest.raises(ValueError, match="one line"):
        b.localize("x = 1  # @c:k", "en", {"k": {"en": "a\nb"}})
    with pytest.raises(ValueError, match="empty text"):
        b.localize("# @c:k", "en", {"k": {"en": " \n"}})
    with pytest.raises(ValueError, match="triple quotes"):
        b.localize('"""@c:k"""', "en", {"k": {"en": 'a """ b'}})
    for bad in ("x = 1  # @c: k", "# @c:my-key", "x = 1  # @c:", "# @c:k extra"):
        with pytest.raises(ValueError, match="malformed comment marker"):
            b.localize(bad, "en", one)
    # a [comment] key of text.toml that no marker uses, and a marker without its key
    src = "# %%\nx = 1  # @c:k\n"
    b.notebook("en", src, {"comment": one})
    with pytest.raises(KeyError, match=r"without a marker.*'unused'"):
        b.notebook("en", src, {"comment": {**one, "unused": one["k"]}})
    with pytest.raises(KeyError, match=r"no \[comment\.k\]"):
        b.notebook("en", src, {})


def test_every_comment_marker_is_filled_in_both_notebooks():
    import tomllib

    comments = tomllib.loads((TUTORIAL / "source" / "text.toml").read_text())["comment"]
    body = (TUTORIAL / "source" / "tutorial.py").read_text().split("\n# %%", 1)[1]  # without the header
    assert set(re.findall(r"@c:(\w+)", body)) == set(comments)
    for name, lang in (("agrijax_tutorial.ipynb", "en"), ("agrijax_tutorial_zh.ipynb", "zh")):
        code = "\n".join("".join(c["source"]) for c in _load(name)["cells"] if c["cell_type"] == "code")
        assert "@c:" not in code
        for key, t in comments.items():
            for line in t[lang].strip().split("\n"):
                assert line.strip() in code, (name, key, line)


def test_messages_have_the_same_placeholders_in_both_languages():
    import string
    import tomllib

    tab = tomllib.loads((TUTORIAL / "source" / "text.toml").read_text())["print"]
    fields = lambda t: sorted(f for _, f, _, _ in string.Formatter().parse(t) if f)  # noqa: E731
    for key, t in tab.items():
        if t["zh"] != "TODO":
            assert fields(t["en"]) == fields(t["zh"]), key
    code = "".join(
        "".join(c["source"]) for c in _load("agrijax_tutorial.ipynb")["cells"] if c["cell_type"] == "code"
    )
    used = set(re.findall(r'say\("([a-z_]+)"', code))
    assert used == set(tab), used ^ set(tab)
