"""Command line of the conformance kit.

``python -m agrijax.testing.conformance [--key GLOB] [--package DIST] [--no-builtin] [--list]
[-- pytest args]``.

Runs the generic conformance test under pytest for the selected cases; ``--list`` prints the
cases and their origin without running anything. Arguments after ``--`` go to pytest.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    ap = argparse.ArgumentParser(prog="python -m agrijax.testing.conformance", description=__doc__)
    ap.add_argument("--key", default="*", help="glob over registry keys, e.g. 'pet/*'")
    ap.add_argument("--package", default=None, help="only the cases of this distribution")
    ap.add_argument("--no-builtin", action="store_true", help="only the entry points' cases")
    ap.add_argument("--list", action="store_true", help="list the selected cases and exit")
    ns = ap.parse_args(argv)
    if ns.list:
        from .discover import discover, select

        for c in select(discover(builtin=not ns.no_builtin), key=ns.key, package=ns.package):
            print(f"{c.key}  ({c.origin})")
        return 0
    import pytest

    args = [
        "--pyargs",
        "agrijax.testing.conformance.test_conformance",
        "-p",
        "agrijax.testing.conformance.pytest_plugin",
        "--agrijax-key",
        ns.key,
    ]
    if ns.package:
        args += ["--agrijax-package", ns.package]
    if ns.no_builtin:
        args.append("--agrijax-no-builtin")
    return int(pytest.main([*args, *extra]))


if __name__ == "__main__":
    sys.exit(main())
