"""A very small SVG toolkit for the showcase figures.

The figures are drawn as plain SVG (no plotting library) so that they can be themed by the page that
shows them. Every colour is a CSS variable with a fallback:

* inlined into a page that defines ``--panel --ink --muted --rule --teal --rust --violet`` (the showcase
  does), the figure follows the page, including its light/dark switch;
* used as ``<img src="...svg">`` or opened on its own, the fallbacks apply and ``prefers-color-scheme``
  selects the light or the dark set.

Palette (same roles as the showcase page): teal = Agri-JAX, rust = the Fortran reference model
(DSSAT-CSM / RZWQM2), ink and muted for every label (text never wears a series colour), hairline rules.
All figures are 1000 x 640 viewBox units, the aspect of the page's figure panel; text sizes are the
page's 15 / 17 / 20 / 26 units.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

W, H = 1000, 640

_SANS = '"Noto Sans SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif'
_MONO = '"IBM Plex Mono",Menlo,Consolas,monospace'

# light and dark values of the page palette (fallbacks when the page does not define the variables)
_LIGHT = dict(
    bg="#E6E9E3",
    ink="#16201C",
    mut="#56625D",
    rule="#C8CEC7",
    neu="#B9C0B7",
    teal="#0F8A66",
    rust="#C2591A",
    violet="#6B4FBB",
)
_DARK = dict(
    bg="#1A201D",
    ink="#E8EBE5",
    mut="#97A29C",
    rule="#2E3632",
    neu="#48504C",
    teal="#2FA582",
    rust="#D96A28",
    violet="#8E78D8",
)
_VAR = dict(
    bg="panel", ink="ink", mut="muted", rule="rule", neu="neutral", teal="teal", rust="rust", violet="violet"
)


def _vars(palette: dict[str, str]) -> str:
    return ";".join(f"--f-{k}:var(--{_VAR[k]},{v})" for k, v in palette.items())


# ``.ajfig.ajfig`` doubles the class so that these rules win over a host page's ``.scene svg text`` rules.
CSS = f"""
svg.ajfig.ajfig{{{_vars(_LIGHT)};font-family:{_SANS}}}
@media (prefers-color-scheme:dark){{svg.ajfig.ajfig{{{_vars(_DARK)}}}}}
.ajfig.ajfig text{{font-family:{_SANS};fill:var(--f-mut);font-size:15px;white-space:pre}}
.ajfig.ajfig text.k{{fill:var(--f-ink)}}
.ajfig.ajfig text.m{{fill:var(--f-mut)}}
.ajfig.ajfig text.n{{font-family:{_MONO};font-variant-numeric:tabular-nums}}
.ajfig.ajfig text.b{{font-weight:700}}
.ajfig.ajfig text.w{{font-weight:500}}
.ajfig.ajfig text.h{{paint-order:stroke;stroke:var(--f-bg);stroke-width:4px;stroke-linejoin:round}}
.ajfig.ajfig text.s17{{font-size:17px}}
.ajfig.ajfig text.s20{{font-size:20px}}
.ajfig.ajfig text.s26{{font-size:26px}}
.ajfig.ajfig .bg{{fill:var(--f-bg)}}
.ajfig.ajfig .grid{{stroke:var(--f-rule);stroke-width:1;fill:none}}
.ajfig.ajfig .axis{{stroke:var(--f-mut);stroke-width:1;fill:none}}
.ajfig.ajfig .ink{{stroke:var(--f-ink);stroke-width:1;fill:none}}
.ajfig.ajfig .ink2{{stroke:var(--f-ink);stroke-width:2;fill:none;stroke-linejoin:round}}
.ajfig.ajfig .fteal{{fill:var(--f-teal)}}
.ajfig.ajfig .frust{{fill:var(--f-rust)}}
.ajfig.ajfig .fneu{{fill:var(--f-neu)}}
.ajfig.ajfig .fink{{fill:var(--f-ink)}}
.ajfig.ajfig .steal{{stroke:var(--f-teal);fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}
.ajfig.ajfig .srust{{stroke:var(--f-rust);fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}
.ajfig.ajfig .ring{{stroke:var(--f-bg);stroke-width:2}}
.ajfig.ajfig .w4{{stroke-width:4}}
.ajfig.ajfig .w15{{stroke-width:1.5}}
.ajfig.ajfig .soft{{opacity:.55}}
"""


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(s: str, size: float = 15, mono: bool = False) -> float:
    """Rough rendered width of ``s`` [viewBox units]: CJK glyphs 1 em, mono 0.6 em, sans 0.5 em."""
    w = 0.0
    for ch in s:
        if ord(ch) >= 0x2E80:
            w += 1.0
        elif mono:
            w += 0.6
        else:
            w += 0.5 if not ch.isupper() else 0.64
    return w * size


def wrap(s: str, width: float, size: float = 15, mono: bool = False) -> list[str]:
    """Greedy line breaking of ``s`` to ``width`` [units] by the :func:`tw` estimate (at spaces; CJK anywhere)."""
    toks = re.findall(r"[\u2e80-\uffff]|[^\s\u2e80-\uffff]+|\s+", s)
    lines, cur = [], ""
    for tk in toks:
        if tk.isspace() and not cur:
            continue
        if cur and tw(cur + tk, size, mono) > width and not tk.isspace() and tk not in "，。；：、）！？)":
            lines.append(cur.rstrip())
            cur = tk
        else:
            cur += tk
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


def paragraph(
    x: float, y: float, s: str, width: float, size: int = 15, lead: float = 22, cls: str = "m"
) -> tuple[list[str], float]:
    """Wrapped text lines starting at baseline ``y``; returns (svg elements, y after the last line)."""
    out = []
    for ln in wrap(s, width, size):
        out.append(txt(x, y, ln, size, cls))
        y += lead
    return out, y


def check_fit(y_next_baseline: float, lead: float = 22.0) -> None:
    """Fail loudly if the last text line (the next one would sit at ``y_next_baseline``) leaves the panel."""
    last = y_next_baseline - lead
    if last > H - 12:
        raise ValueError(f"text runs off the figure: last baseline {last:.0f} > {H - 12}")


def grouped(v: float) -> str:
    """Thousands with a thin space: 11859.3 -> '11 859.3'."""
    a, _, b = f"{v:.1f}".partition(".")
    a = f"{int(a):,}".replace(",", "\u2009")
    return f"{a}.{b}" if b != "0" else a


def txt(
    x: float, y: float, s: str, size: int = 15, cls: str = "", anchor: str = "start", rot: float = 0
) -> str:
    """One text line. ``cls``: k ink, m muted (default), n mono numerals, b bold, w medium."""
    c = f"s{size}" if size != 15 else ""
    c = " ".join(p for p in (c, cls) if p)
    a = "" if anchor == "start" else f' text-anchor="{anchor}"'
    r = f' transform="rotate({rot:.1f} {x:.1f} {y:.1f})"' if rot else ""
    k = f' class="{c}"' if c else ""
    return f'<text x="{x:.1f}" y="{y:.1f}"{a}{r}{k}>{esc(s)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, cls: str = "grid") -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"/>'


def rect(x: float, y: float, w: float, h: float, cls: str, rx: float = 0) -> str:
    r = f' rx="{rx:g}"' if rx else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"{r} class="{cls}"/>'


def path(d: str, cls: str) -> str:
    return f'<path d="{d}" class="{cls}"/>'


def circle(cx: float, cy: float, r: float, cls: str, ring: bool = True) -> str:
    k = cls + " ring" if ring else cls
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:g}" class="{k}"/>'


def polyline(xs, ys, cls: str) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))
    return f'<polyline points="{pts}" class="{cls}"/>'


def hbar(x0: float, x1: float, y: float, h: float, cls: str, r: float = 4) -> str:
    """Horizontal bar from the baseline ``x0``: square at the baseline, ``r`` px rounded at the data end."""
    x1 = max(x1, x0 + 1.0)
    r = min(r, (x1 - x0) / 2, h / 2)
    d = (
        f"M{x0:.1f},{y:.1f}H{x1 - r:.1f}A{r:.1f},{r:.1f} 0 0 1 {x1:.1f},{y + r:.1f}V{y + h - r:.1f}"
        f"A{r:.1f},{r:.1f} 0 0 1 {x1 - r:.1f},{y + h:.1f}H{x0:.1f}Z"
    )
    return path(d, cls)


def vbar(x: float, y0: float, y1: float, w: float, cls: str, r: float = 4) -> str:
    """Vertical bar from the baseline ``y0`` (bottom) up to ``y1``: rounded at the top."""
    y1 = min(y1, y0 - 1.0)
    r = min(r, (y0 - y1) / 2, w / 2)
    d = (
        f"M{x:.1f},{y0:.1f}V{y1 + r:.1f}A{r:.1f},{r:.1f} 0 0 1 {x + r:.1f},{y1:.1f}H{x + w - r:.1f}"
        f"A{r:.1f},{r:.1f} 0 0 1 {x + w:.1f},{y1 + r:.1f}V{y0:.1f}Z"
    )
    return path(d, cls)


class Scale:
    """Linear or log10 map from data to pixels."""

    def __init__(self, d0: float, d1: float, r0: float, r1: float, log: bool = False):
        self.log, self.r0, self.r1 = log, r0, r1
        self.d0, self.d1 = (math.log10(d0), math.log10(d1)) if log else (d0, d1)

    def __call__(self, v: float) -> float:
        t = math.log10(v) if self.log else v
        return self.r0 + (t - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


def step_path(xs, ys) -> str:
    """SVG path of a step function that is right-continuous: rises at every x (an empirical CDF)."""
    d = f"M{xs[0]:.1f},{ys[0]:.1f}"
    for i in range(1, len(xs)):
        d += f"H{xs[i]:.1f}V{ys[i]:.1f}"
    return d


def nice_ceil(v: float) -> float:
    """Smallest of 1, 2, 5 times a power of ten that is at least ``v`` (v > 0)."""
    e = math.floor(math.log10(v))
    for m in (1, 2, 5, 10):
        if m * 10**e >= v * (1 - 1e-12):
            return m * 10**e
    return 10 ** (e + 1)


def fmt_sci(v: float, digits: int = 1) -> str:
    """``4.0e-5`` style, exponent without padding."""
    m, e = f"{v:.{digits}e}".split("e")
    return f"{m}e{int(e)}"


def fmt_s(v: float) -> str:
    """A duration in seconds: 2 decimals below 1 s, 1 decimal below 100 s, whole seconds above."""
    if v < 1:
        return f"{v:.2f} s"
    if v < 100:
        return f"{v:.1f} s"
    return f"{v:.0f} s"


def svg_doc(name: str, lang: str, title: str, desc: str, body: list[str], extra_css: str = "") -> str:
    """Wrap the drawing: a background panel, title and description for assistive technology."""
    tid, did = f"{name}-{lang}-t", f"{name}-{lang}-d"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="ajfig ajfig" viewBox="0 0 {W} {H}" role="img" '
        f'aria-labelledby="{tid} {did}">\n<title id="{tid}">{esc(title)}</title>\n'
        f'<desc id="{did}">{esc(desc)}</desc>\n<style>{CSS}{extra_css}</style>\n'
        f'<rect width="{W}" height="{H}" rx="2" class="bg"/>\n' + "\n".join(body) + "\n</svg>\n"
    )


def data_dir(arg: str | None) -> Path:
    """``--data-dir`` if given, else ``$AGRI_JAX_DATA``, else ``~/agri_jax_data``."""
    return Path(arg or os.environ.get("AGRI_JAX_DATA", "~/agri_jax_data")).expanduser()


def fig_dir(arg: str | None) -> Path:
    """``--out`` if given, else ``docs/showcase/fig`` of this checkout."""
    p = Path(arg) if arg else Path(__file__).resolve().parents[2] / "docs" / "showcase" / "fig"
    p.mkdir(parents=True, exist_ok=True)
    return p


def add_common_args(ap) -> None:
    ap.add_argument("--data-dir", help="result directory (default: $AGRI_JAX_DATA or ~/agri_jax_data)")
    ap.add_argument("--out", help="output directory (default: docs/showcase/fig)")
    ap.add_argument("--lang", nargs="+", default=["en", "zh"], choices=["en", "zh"])
