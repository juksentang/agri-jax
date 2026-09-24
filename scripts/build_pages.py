#!/usr/bin/env python3
"""Build the GitHub Pages site from the shell-less showcase sources.

The showcase pages in docs/showcase/ are written without the
<!doctype>/<html>/<head>/<body> shell, because the claude.ai artifact host
adds one.  This script wraps them into complete standalone documents with a
reset that matches that host, adds a language switch, and writes

    docs/showcase/index.html     (zh, source)  ->  docs/index.html
    docs/showcase/en/index.html  (en, source)  ->  docs/en/index.html

Usage:  python scripts/build_pages.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# (source, output, html lang, href to the other language, which label is current)
PAGES = [
    (DOCS / "showcase" / "index.html", DOCS / "index.html", "zh-CN", "en/", "zh"),
    (DOCS / "showcase" / "en" / "index.html", DOCS / "en" / "index.html", "en", "../", "en"),
]

# Reset matching the artifact host; the page's own <style> follows and overrides it.
RESET = (
    "body{margin:0;font:14px/1.5 system-ui,-apple-system,\"Segoe UI\",Roboto,sans-serif;background:#faf9f5}"
    "img{max-width:100%;height:auto}"
    "[hidden]{display:none!important}"
    ".langsw{max-width:1680px;margin:0 auto;padding:6px clamp(0px,2.2vw,32px) 0;display:flex;justify-content:flex-end;"
    "gap:8px;font-size:12px;line-height:18px;color:var(--muted,#56625D)}"
    ".langsw a{color:inherit}.langsw b{color:var(--ink,#16201C);font-weight:600}"
)


def lang_switch(other_href: str, current: str) -> str:
    zh = "<b>中文</b>" if current == "zh" else f'<a href="{other_href}" hreflang="zh-CN" lang="zh-CN">中文</a>'
    en = "<b>English</b>" if current == "en" else f'<a href="{other_href}" hreflang="en" lang="en">English</a>'
    return f'<div class="langsw" role="navigation" aria-label="Language">{zh}<span aria-hidden="true">/</span>{en}</div>\n'


def build(src: Path, out: Path, lang: str, other_href: str, current: str) -> None:
    text = src.read_text(encoding="utf-8")
    if re.search(r"<!doctype|<html[\s>]|<body[\s>]", text, re.I):
        sys.exit(f"{src}: expected a shell-less page, found a document shell")
    m = re.match(r"\s*(<title>.*?</title>)\s*(.*?</style>)\s*", text, re.S)
    if not m:
        sys.exit(f"{src}: expected <title>, then links and one <style> block at the top")
    title, head_rest, body = m.group(1), m.group(2), text[m.end():]
    html = (
        "<!doctype html>\n"
        f'<html lang="{lang}">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
        f"{title}\n"
        f"<style>{RESET}</style>\n"
        f"{head_rest}\n"
        "</head>\n<body>\n"
        f"{lang_switch(other_href, current)}"
        f"{body.rstrip()}\n"
        "</body>\n</html>\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}  ({len(html):,} bytes)")


def main() -> None:
    for page in PAGES:
        build(*page)


if __name__ == "__main__":
    main()
