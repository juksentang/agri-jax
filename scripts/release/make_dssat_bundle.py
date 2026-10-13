"""Make the DSSAT-CSM reference bundle Agri-JAX downloads to compare with DSSAT (a GitHub release asset).

    python scripts/release/make_dssat_bundle.py --engine <dssat_engine root> --out <directory outside the repo>

The engine root is a DSSAT-CSM v4.8.6.0 source tree with its build (``source/build486/bin/dscsm048``,
``source/Data``: the layout the validation runs use). The bundle ``dssat-4.8.6-build486-linux-x86_64.tar.gz``
holds, under ``dssat-4.8.6-build486/``:

* ``bin/dscsm048`` -- the build486 binary itself (statically linked; unmodified);
* ``bin/`` -- DSSAT's ``Data`` directory of the same source tree (``*.CDE``, ``DSCSM048.CTR``,
  ``DSSATPRO.*``, ``MODEL.ERR``, ``Genotype``, ``StandardData``, ``Pest``; the ``Help``,
  ``BatchFiles`` and ``Default`` directories are left out);
* ``LICENSE`` -- DSSAT-CSM's BSD-3-Clause licence (``source/license.txt``);
* ``PROVENANCE.txt`` -- version, source commit, compiler and flags, linking, SHA-256 of ``dscsm048``.

The archive is deterministic (sorted entries, fixed times and owners, gzip without a timestamp): the
same inputs give the same bytes and SHA-256, printed at the end (``agrijax.io.examples.REFERENCE_BUNDLE``).
Do not commit the bundle.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import re
import subprocess
import tarfile
from pathlib import Path

NAME = "dssat-4.8.6-build486"
ASSET = f"{NAME}-linux-x86_64.tar.gz"
SKIP_DIRS = {"Help", "BatchFiles", "Default"}
SKIP_FILES = {"DSSATPRO.L48.in"}
#: fixed modification time of every entry (2026-09-23 00:00 UTC, the build486 date)
MTIME = 1790121600


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache(cache: Path, key: str) -> str:
    for ln in cache.read_text(errors="replace").splitlines():
        if ln.startswith(key + ":"):
            return ln.split("=", 1)[1].strip()
    return ""


def provenance(eng: Path, exe: Path) -> str:
    src, bdir = eng / "source", eng / "source" / "build486"
    cache = bdir / "CMakeCache.txt"
    try:
        commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        tag = subprocess.run(["git", "-C", str(src), "describe", "--tags", "--exact-match"], capture_output=True, text=True, check=False).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit, tag = "", ""
    log = (bdir / "cmake.log").read_text(errors="replace") if (bdir / "cmake.log").is_file() else ""
    m = re.search(r"COMMIT: ([0-9a-f]{40})", log)
    commit = commit or (m.group(1) if m else "unknown")
    comp = re.search(rb"GCC: \(([^)]*)\) ([0-9.]+)", exe.read_bytes())
    compiler = f"GNU Fortran {comp.group(2).decode()} ({comp.group(1).decode()})" if comp else "unknown"
    lines = [
        "DSSAT-CSM reference bundle for Agri-JAX",
        "",
        "Program        DSSAT Cropping System Model (DSSAT-CSM) v4.8.6.0, dscsm048",
        f"Source         https://github.com/DSSAT/dssat-csm-os tag {tag or 'v4.8.6.0'} commit {commit}",
        "Build          build486, the build all Agri-JAX validation runs use, unmodified",
        f"Compiler       {compiler}",
        f"Build type     {_cache(cache, 'CMAKE_BUILD_TYPE')}",
        f"Fortran flags  {_cache(cache, 'CMAKE_Fortran_FLAGS')} {_cache(cache, 'CMAKE_Fortran_FLAGS_RELEASE')}".rstrip(),
        f"Linker flags   {_cache(cache, 'CMAKE_EXE_LINKER_FLAGS')}  (statically linked)",
        f"Target         Linux x86-64",
        f"SHA-256        {_sha(exe)}  bin/dscsm048",
        "Data           bin/: DSSAT's Data directory of the same source tree (DSSATPRO.L48 as CMake",
        "               configures it), without Help, BatchFiles and Default",
        "Licence        BSD-3-Clause, Copyright (c) the DSSAT Foundation (LICENSE)",
        "",
        "Agri-JAX uses this program only as the reference it is compared with; Agri-JAX itself is an",
        "independent implementation from published equations and runs without it.",
    ]
    return "\n".join(lines) + "\n"


def _add(tf: tarfile.TarFile, arc: str, data: bytes | None, mode: int) -> None:
    ti = tarfile.TarInfo(arc)
    ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname, ti.mode = MTIME, 0, 0, "", "", mode
    if data is None:
        ti.type = tarfile.DIRTYPE
        tf.addfile(ti)
    else:
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))


def make(eng: Path, out: Path) -> Path:
    exe = eng / "source" / "build486" / "bin" / "dscsm048"
    data = eng / "source" / "Data"
    lic = eng / "source" / "license.txt"
    for p in (exe, data, lic):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    entries: list[tuple[str, bytes | None, int]] = [(NAME, None, 0o755), (f"{NAME}/bin", None, 0o755)]
    entries.append((f"{NAME}/bin/dscsm048", exe.read_bytes(), 0o755))
    for p in sorted(data.rglob("*")):
        rel = p.relative_to(data)
        if rel.parts[0] in SKIP_DIRS or p.name in SKIP_FILES:
            continue
        arc = f"{NAME}/bin/{rel.as_posix()}"
        entries.append((arc, None, 0o755) if p.is_dir() else (arc, p.read_bytes(), 0o644))
    entries.append((f"{NAME}/LICENSE", lic.read_bytes(), 0o644))
    entries.append((f"{NAME}/PROVENANCE.txt", provenance(eng, exe).encode(), 0o644))
    out.mkdir(parents=True, exist_ok=True)
    dest = out / ASSET
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for arc, blob, mode in sorted(entries, key=lambda e: e[0]):
            _add(tf, arc, blob, mode)
    with open(dest, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=9) as gz:
        gz.write(buf.getvalue())
    (out / "PROVENANCE.txt").write_text(provenance(eng, exe))
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if a.out.resolve().is_relative_to(repo):
        raise SystemExit("write the bundle outside the repository (it must not be committed)")
    dest = make(a.engine.expanduser(), a.out.expanduser())
    print(f"{dest}\n  size   {dest.stat().st_size} bytes\n  sha256 {_sha(dest)}")


if __name__ == "__main__":
    main()
