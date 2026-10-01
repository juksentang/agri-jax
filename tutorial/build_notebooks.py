"""Generate the tutorial notebooks from their one source.

``tutorial/source/tutorial.py`` holds the cells: ``# %% [markdown] <key>`` is a text cell whose text
is ``<key>`` of ``tutorial/source/text.toml`` (``en`` and ``zh``); ``# %% [messages]`` is the code cell
defining the printed messages (``MSG``, both languages, from text.toml's ``[print]`` table) and
``say(key, **values)``; ``# %%`` starts a code cell, the same in both notebooks except that
``"{LANG}"`` becomes the notebook's language. Comments are per language too: a whole-line comment
``# @c:<key>``, a trailing comment ``code  # @c:<key>`` or a docstring consisting of ``@c:<key>`` is
replaced by the text of ``[comment.<key>]`` of text.toml (``en`` and ``zh``), keeping the indentation
and the ``# `` prefix. Writes ``tutorial/agrijax_tutorial.ipynb`` (English) and
``tutorial/agrijax_tutorial_zh.ipynb`` (Chinese), outputs empty::

    python tutorial/build_notebooks.py           # write both
    python tutorial/build_notebooks.py --check   # exit 1 if a committed notebook is out of date
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source" / "tutorial.py"
TEXT = HERE / "source" / "text.toml"
LANGUAGES = {"en": HERE / "agrijax_tutorial.ipynb", "zh": HERE / "agrijax_tutorial_zh.ipynb"}
_MARK = "# %%"
_MD = "# %% [markdown]"
_MSG = "# %% [messages]"
LANG_TOKEN = '"{LANG}"'
_KEY = r"[A-Za-z0-9_]+"
_C_LINE = re.compile(rf"^(\s*)# @c:({_KEY})\s*$")  # a whole-line comment
_C_TRAIL = re.compile(rf"^(.*\S)(\s+)# @c:({_KEY})\s*$")  # a comment after code
_C_DOC = re.compile(rf'"""@c:({_KEY})"""')  # a docstring


def cells(source: str) -> list[tuple[str, str]]:
    """``[(kind, body)]`` of the source: ``("markdown", key)`` or ``("code", code)``; the text
    before the first marker (the file's header comment) is dropped."""
    out: list[tuple[str, str]] = []
    cur: tuple[str, list[str]] | None = None
    for ln in source.splitlines():
        if ln.startswith(_MARK):
            if cur is not None:
                out.append((cur[0], "\n".join(cur[1]).strip("\n")))
            if ln.startswith(_MSG):
                out.append(("messages", ""))
                cur = None
            elif ln.startswith(_MD):
                key = ln[len(_MD) :].strip()
                if not key:
                    raise ValueError(f"markdown cell without a key: {ln!r}")
                out.append(("markdown", key))
                cur = None
            else:
                cur = ("code", [])
        elif cur is not None:
            cur[1].append(ln)
        elif ln.strip() and not ln.startswith("#"):
            raise ValueError(f"text outside a cell: {ln!r}")
    if cur is not None:
        out.append((cur[0], "\n".join(cur[1]).strip("\n")))
    return out


def _lines(text: str) -> list[str]:
    ls = text.split("\n")
    return [ln + "\n" for ln in ls[:-1]] + [ls[-1]]


def _comment_text(comments: dict, key: str, lang: str) -> str:
    """The text of ``[comment.<key>]`` in ``lang``."""
    if key not in comments:
        raise KeyError(f"text.toml has no [comment.{key}] (used by '@c:{key}' in tutorial.py)")
    entry = comments[key]
    if not isinstance(entry, dict) or lang not in entry:
        raise KeyError(f"text.toml has no {lang!r} text for comment {key!r}")
    text = "\n".join(ln.rstrip() for ln in entry[lang].strip().split("\n"))
    if not text:
        raise ValueError(f"comment {key!r} ({lang}): empty text")
    return text


def _docstring(text: str, key: str, lang: str, indent: str) -> str:
    if '"""' in text or "\\" in text or text.endswith('"'):
        raise ValueError(
            f"comment {key!r} ({lang}): a docstring has no triple quotes, backslash or final quote"
        )
    first, *rest = text.split("\n")
    if not rest:
        return f'"""{first}"""'
    return '"""' + first + "".join("\n" + (indent + ln if ln else "") for ln in rest) + "\n" + indent + '"""'


def _localize_docstrings(line: str, lang: str, comments: dict, used: set[str]) -> str:
    indent = line[: len(line) - len(line.lstrip())]

    def sub(m: re.Match[str]) -> str:
        used.add(m[1])
        return _docstring(_comment_text(comments, m[1], lang), m[1], lang, indent)

    return _C_DOC.sub(sub, line)


def localize(code: str, lang: str, comments: dict, used: set[str] | None = None) -> str:
    """``code`` with every comment marker replaced by the text of ``lang`` (``comments`` is the
    ``[comment]`` table of text.toml); the keys found are added to ``used``.

    ``# @c:<key>`` on a line of its own becomes one ``# `` line per line of the text, at the same
    indentation; after code it becomes ``# <text>`` (the text is one line); a docstring consisting of
    ``@c:<key>`` becomes the docstring of the text (the further lines of a long text indented like the
    docstring, the closing quotes on a line of their own)."""
    used = set() if used is None else used
    out: list[str] = []
    for ln in code.split("\n"):
        if m := _C_LINE.match(ln):
            used.add(m[2])
            text = _comment_text(comments, m[2], lang)
            out += [m[1] + ("# " + t if t else "#") for t in text.split("\n")]
        elif m := _C_TRAIL.match(ln):
            used.add(m[3])
            text = _comment_text(comments, m[3], lang)
            if "\n" in text:
                raise ValueError(f"comment {m[3]!r} ({lang}): a comment after code is one line")
            out.append(f"{m[1]}{m[2]}# {text}")
        else:
            ln = _localize_docstrings(ln, lang, comments, used)
            if "@c:" in ln:
                raise ValueError(f"malformed comment marker ('# @c:<key>' or a docstring '@c:<key>'): {ln!r}")
            out.append(ln)
    return "\n".join(out)


def messages_code(table: dict, lang_key: str = "LANG") -> str:
    """The code cell of the printed messages (both languages) and ``say``."""
    lines = [
        "# Printed messages of this notebook in English and Chinese (tutorial/source/text.toml)",
        "MSG = {",
    ]
    for key, t in table.items():
        for lg in ("en", "zh"):
            if lg not in t:
                raise KeyError(f"text.toml has no {lg!r} text for message {key!r}")
            if '"' in t[lg] or "\\" in t[lg] or "\n" in t[lg]:
                raise ValueError(f"message {key!r} ({lg}): no double quotes, backslashes or newlines")
        lines += [f'    "{key}": {{', f'        "en": "{t["en"]}",', f'        "zh": "{t["zh"]}",', "    },"]
    lines += [
        "}",
        "",
        "",
        "def say(key, **values):",
        '    """Print message ``key`` in the language of this notebook."""',
        f"    print(MSG[key][{lang_key}].format(**values))",
    ]
    return "\n".join(lines)


def notebook(lang: str, source: str | None = None, text: dict | None = None) -> dict:
    """The notebook of ``lang`` (``en`` / ``zh``) as a JSON-ready dict."""
    src = SOURCE.read_text() if source is None else source
    tab = tomllib.loads(TEXT.read_text()) if text is None else text
    comments = tab.get("comment", {})
    used_comments: set[str] = set()
    nb_cells = []
    for i, (kind, body) in enumerate(cells(src)):
        if kind == "messages":
            kind, body = "code", messages_code(tab.get("print", {}))
        elif kind == "code":
            body = localize(body.replace(LANG_TOKEN, f'"{lang}"'), lang, comments, used_comments)
        if kind == "markdown":
            if body not in tab or lang not in tab[body]:
                raise KeyError(f"text.toml has no {lang!r} text for cell {body!r}")
            nb_cells.append(
                {
                    "cell_type": "markdown",
                    "id": f"{body}",
                    "metadata": {},
                    "source": _lines(tab[body][lang].strip()),
                }
            )
        else:
            nb_cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": f"code-{i:02d}",
                    "metadata": {},
                    "outputs": [],
                    "source": _lines(body),
                }
            )
    unused = sorted(set(tab) - {b for k, b in cells(src) if k == "markdown"} - {"print", "comment"})
    if unused:
        raise KeyError(f"text.toml keys without a cell: {unused}")
    unused = sorted(set(comments) - used_comments)
    if unused:
        raise KeyError(f"text.toml [comment] keys without a marker in tutorial.py: {unused}")
    return {
        "cells": nb_cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    stale = []
    for lang, path in LANGUAGES.items():
        text = render(notebook(lang))
        if "--check" in argv:
            if not path.is_file() or path.read_text() != text:
                stale.append(path.name)
        else:
            path.write_text(text)
    if stale:
        print(f"out of date (run python tutorial/build_notebooks.py): {stale}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
