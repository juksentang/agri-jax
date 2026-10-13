"""DSSAT's public data and the DSSAT-CSM reference program: download, cache, verify.

Agri-JAX re-implements the DSSAT-CSM v4.8.6.0 maize model from its published equations and reads
DSSAT's own experiment files; it needs no DSSAT program to run. Two kinds of files come from DSSAT
(BSD-3-Clause, (c) the DSSAT Foundation), none distributed with Agri-JAX:

* **DSSAT's data** (:func:`fetch_dssat_data`): the example experiment UFGA8201 (Gainesville,
  Florida, 1982: FileX, observed A / T files), the Gainesville weather of 1978-1987 and
  ``SOIL.SOL`` from ``DSSAT/dssat-csm-data`` at a fixed commit (:data:`EXAMPLE_FILES`), and the maize
  genotype files (``MZCER048.CUL`` / ``.ECO`` / ``.SPE``) and standard data (``CO2048.WDA``,
  ``RESCH048.SDA``) of ``DSSAT/dssat-csm-os`` v4.8.6.0 (:data:`DATA_FILES`, taken from the release's
  source archive: the files as a checkout of the tag holds them). Every file is checked against its
  SHA-256. Layout: ``<root>/example_data/{Maize,Soil,Weather}``, ``<root>/Data/{Genotype,StandardData}``.
* **The reference program** ``dscsm048`` (:func:`install_reference`), only to compare with: the
  prebuilt build of DSSAT-CSM v4.8.6.0 that Agri-JAX is validated against (:data:`REFERENCE_BUNDLE`,
  a GitHub release asset of Agri-JAX: the statically linked binary, DSSAT's ``Data`` directory, its
  licence and a provenance note), or, as a fallback, built from the ``dssat-csm-os`` v4.8.6.0 source
  with CMake and gfortran (:func:`build_dscsm048`).

Everything is cached under :func:`cache_dir` (``AGRI_JAX_CACHE``, default ``~/.cache/agrijax``).
``AGRI_JAX_DSSAT`` pointing to an engine root that already has ``dscsm048`` (and, for the data, its
``example_data``) is used as it is, with nothing downloaded.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DATA_FILES",
    "DSSAT_CSM_DATA_COMMIT",
    "DSSAT_CSM_OS",
    "EXAMPLE_FILES",
    "REFERENCE_BUNDLE",
    "ChecksumError",
    "ReferenceUnavailableError",
    "ReleaseAsset",
    "SourceArchive",
    "build_dscsm048",
    "cache_dir",
    "download",
    "fetch_dssat_data",
    "fetch_dssat_source",
    "fetch_examples",
    "install_reference",
]


@dataclass(frozen=True)
class SourceArchive:
    """A GitHub repository at a fixed commit, downloaded as one ``.tar.gz`` archive."""

    repo: str
    tag: str
    commit: str
    sha256: str

    @property
    def url(self) -> str:
        return f"https://codeload.github.com/{self.repo}/tar.gz/{self.commit}"


@dataclass(frozen=True)
class ReleaseAsset:
    """A file attached to a GitHub release."""

    repo: str
    tag: str
    name: str
    sha256: str

    @property
    def url(self) -> str:
        return f"https://github.com/{self.repo}/releases/download/{self.tag}/{self.name}"


#: DSSAT-CSM v4.8.6.0, the version Agri-JAX is validated against (the tag's commit; the archive's
#: SHA-256 measured 2026-09-30; its source tree is the one the validation build486 was built from)
DSSAT_CSM_OS = SourceArchive(
    repo="DSSAT/dssat-csm-os",
    tag="v4.8.6.0",
    commit="d6556cca9f926c40eddffd9adcf77e970db7511b",
    sha256="4affb18358ba9af06f4ca8d30f4eaf34861912cca8a73443c8c20f1bf06b022b",
)
#: ``DSSAT/dssat-csm-data`` commit of the example files (the example data of the validation runs)
DSSAT_CSM_DATA_COMMIT = "1c28443e5443c446861af9aa7942357d4c6b1ec5"
_DATA_URL = "https://raw.githubusercontent.com/DSSAT/dssat-csm-data/{commit}/{path}"
#: example files (path in ``dssat-csm-data`` -> SHA-256): UFGA8201 and what its runs read
EXAMPLE_FILES: dict[str, str] = {
    "Maize/UFGA8201.MZX": "b049688224bc561b3257ac54dddaf3544a900f55b5c2e8ec2207d49f27fd0a5e",
    "Maize/UFGA8201.MZA": "e0c938e6e5abc4b037dbf411b0f36a735c7cb0e5bcde197a66916007e720b7ab",
    "Maize/UFGA8201.MZT": "3335dc9316d8ce88612dc74d0cdc44713941147b21252796c0838d7437e49ad8",
    "Soil/SOIL.SOL": "da439c9ba6c65b664a26f15d6d32c0f25cb1126962ce7099bd16af4cabb31e87",
    "Weather/UFGA7801.WTH": "8d78d31c64cee019552924e951846c497bf63a12eb9a880f2da34b3ea9bd050f",
    "Weather/UFGA7901.WTH": "b9e0ca56f9d3b4399874b16d06b735b6a28894c8e6b2305763448ce391e7df96",
    "Weather/UFGA8001.WTH": "008b76a235613e7f6d64b045216849b459cb36e5ba2b973ddba62ea54f7f5353",
    "Weather/UFGA8101.WTH": "4f043d2c7426e10d5862507c3af25255fb574aabce89658c10a980d9b814619d",
    "Weather/UFGA8201.WTH": "61430bbee16cb8b3d6bf8e610bb9db309f34b28911de0f8f80925706110d80ff",
    "Weather/UFGA8301.WTH": "979df295e5b07a62616316e53364176561d238bfc85167132eb269907e53d9f1",
    "Weather/UFGA8401.WTH": "1e3a6144915889954a4bc8a29e7265bf4dabeb03ffec83e3961c52292f3c6c3c",
    "Weather/UFGA8501.WTH": "67e6a3da245374759ff5e2bdeafe9b23581c939ba4f131130f0aabf30d745943",
    "Weather/UFGA8601.WTH": "c86f47ea3ba1edcf6a78a4ace656ddb013dfc522ae3550e9b15d589b28f1582d",
    "Weather/UFGA8701.WTH": "87a4d8cfc002051fbd08057ef7416c8407aa1d31925006f6a195a021221b73ed",
}
#: DSSAT data files the maize day reads (path in ``dssat-csm-os`` -> SHA-256 of the file in the
#: v4.8.6.0 archive, CRLF line ends as a checkout holds them)
DATA_FILES: dict[str, str] = {
    "Data/Genotype/MZCER048.CUL": "3ce6005d07bbf09c2e08f67bb021957ebb16c81d73226b5ed307420f0a59ecae",
    "Data/Genotype/MZCER048.ECO": "d0144abddf9931e7e83421c6b3302fc52f60aaecd4cbfc9488dd823903a38beb",
    "Data/Genotype/MZCER048.SPE": "a70464cf21829afd4af5f04a0ee7eaa1a716f6f701a050dfdaf5c1f860321dae",
    "Data/StandardData/CO2048.WDA": "00f8de515a580fe67b2a28343bd5e8542be9a7bd3b44ffe7b133f8e19713a6b4",
    "Data/StandardData/RESCH048.SDA": "d11a563e3df4c67ec21fa0ac6c2109f4c93dc2a53c6e57a595aa53669b9a1ccf",
}
#: the prebuilt reference program: ``dscsm048`` of DSSAT-CSM v4.8.6.0 (build486, the binary of the
#: validation runs, statically linked) with DSSAT's ``Data`` directory, a GitHub release asset of
#: Agri-JAX (made by ``scripts/release/make_dssat_bundle.py``)
REFERENCE_BUNDLE = ReleaseAsset(
    repo="juksentang/agri-jax",
    tag="dssat-4.8.6-build486",
    name="dssat-4.8.6-build486-linux-x86_64.tar.gz",
    sha256="4b7d399f3fde3eb964e6d2c6d6e79d9ab120b9c7a33c9d8b0ae4d9b2a0f139e1",
)
#: seconds before a download gives up
TIMEOUT_S = 120
#: bytes read per chunk while hashing
_CHUNK = 1 << 20


class ChecksumError(OSError):
    """A downloaded or cached file does not have the expected SHA-256."""


class ReferenceUnavailableError(OSError):
    """The DSSAT reference program could not be fetched (no network, or neither the prebuilt bundle
    nor the DSSAT source could be downloaded). Only the comparisons with DSSAT need it."""


#: the files that mark a directory as one Agri-JAX unpacked (and may therefore replace)
_BUNDLE_MARK, _SOURCE_MARK = "PROVENANCE.txt", ".agrijax_source"


def _check_replaceable(d: Path, mark: str) -> None:
    """Refuse to delete ``d`` unless it is absent, empty or holds ``mark`` (made by Agri-JAX)."""
    if d.is_dir() and any(d.iterdir()) and not (d / mark).is_file():
        raise FileExistsError(
            f"{d} exists and was not made by Agri-JAX (no {mark}); it is left as it is: "
            "choose a new or empty directory"
        )
    if d.exists() and not d.is_dir():
        raise FileExistsError(f"{d} exists and is not a directory")


def cache_dir() -> Path:
    """The download cache: ``AGRI_JAX_CACHE`` or ``~/.cache/agrijax``."""
    return Path(os.environ.get("AGRI_JAX_CACHE") or Path.home() / ".cache" / "agrijax").expanduser()


def sha256_of(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _urlopen(url: str, timeout: float) -> Any:
    return urllib.request.urlopen(url, timeout=timeout)


def download(url: str, dest: str | os.PathLike[str], sha256: str, *, timeout: float = TIMEOUT_S) -> Path:
    """``url`` saved to ``dest`` if its SHA-256 is ``sha256`` (else :class:`ChecksumError`, nothing
    kept). A ``dest`` already there with that hash is reused without a download."""
    d = Path(dest)
    if d.is_file() and sha256_of(d) == sha256:
        return d
    d.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".dl_", dir=d.parent)
    h = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as out, _urlopen(url, timeout) as r:
            while chunk := r.read(_CHUNK):
                h.update(chunk)
                out.write(chunk)
        got = h.hexdigest()
        if got != sha256:
            raise ChecksumError(f"{url}: SHA-256 {got}, expected {sha256} (download not kept)")
        os.replace(tmp, d)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return d


def _extract(tgz: Path, dest: Path, members: Callable[[str], bool] | None = None) -> Path:
    """Unpack ``tgz`` (the entries whose archive path passes ``members``) into a temporary directory
    next to ``dest``; returns the archive's single top directory there."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".x_", dir=dest.parent))
    with tarfile.open(tgz, "r:gz") as tf:
        sel = [m for m in tf.getmembers() if members is None or members(m.name)]
        if hasattr(tarfile, "data_filter"):
            tf.extractall(tmp, members=sel, filter="data")
        else:  # pragma: no cover (Python < 3.11.4)
            tf.extractall(tmp, members=sel)
    tops = [p for p in tmp.iterdir() if p.is_dir()]
    if len(tops) != 1:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError(f"{tgz.name}: expected one top directory, found {[p.name for p in tops]}")
    return tops[0]


def _source_archive(archive: SourceArchive | None = None) -> Path:
    archive = archive or DSSAT_CSM_OS
    return download(
        archive.url, cache_dir() / "downloads" / f"dssat-csm-os-{archive.commit}.tar.gz", archive.sha256
    )


def fetch_dssat_source(root: str | os.PathLike[str], archive: SourceArchive | None = None) -> Path:
    """The engine source unpacked into ``<root>/source`` (reused when already there); returns it."""
    archive = archive or DSSAT_CSM_OS
    r = Path(root)
    src = r / "source"
    stamp = src / ".agrijax_source"
    if stamp.is_file() and stamp.read_text().strip() == archive.commit:
        return src
    _check_replaceable(src, _SOURCE_MARK)
    top = _extract(_source_archive(archive), src)
    try:
        if src.exists():
            shutil.rmtree(src)
        top.rename(src)
        stamp.write_text(archive.commit + "\n")
    finally:
        shutil.rmtree(top.parent, ignore_errors=True)
    return src


def _run(argv: list[str], cwd: Path, log: Path) -> None:
    with open(log, "a") as f:
        f.write("$ " + " ".join(argv) + "\n")
        f.flush()
        rc = subprocess.run(argv, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=False).returncode
    if rc != 0:
        tail = "".join(log.read_text(errors="replace").splitlines(keepends=True)[-30:])
        raise RuntimeError(f"{argv[0]} failed (exit {rc}); log {log}, tail:\n{tail}")


def build_dscsm048(root: str | os.PathLike[str], *, jobs: int | None = None) -> tuple[Path, float]:
    """Build ``dscsm048`` from ``<root>/source`` with CMake (release build, gfortran) into
    ``<root>/source/build486/bin``; returns ``(executable, seconds)`` (0 s when it was built before)."""
    src = Path(root) / "source"
    bdir = src / "build486"
    exe = bdir / "bin" / "dscsm048"
    if exe.is_file():
        return exe, 0.0
    missing = [t for t in ("cmake", "gfortran") if shutil.which(t) is None]
    if missing:
        raise RuntimeError(
            f"building dscsm048 needs {' and '.join(missing)} (Debian / Ubuntu / Colab: "
            "`apt-get install -y gfortran cmake`)"
        )
    bdir.mkdir(parents=True, exist_ok=True)
    log = bdir / "agrijax_build.log"
    n = int(jobs) if jobs else max(1, os.cpu_count() or 1)
    t0 = time.perf_counter()
    _run(["cmake", "-S", str(src), "-B", str(bdir), "-DCMAKE_BUILD_TYPE=RELEASE"], bdir, log)
    _run(["cmake", "--build", str(bdir), "-j", str(n)], bdir, log)
    if not exe.is_file():
        raise RuntimeError(f"the build finished without {exe}; log {log}")
    return exe, time.perf_counter() - t0


def _place(src: Path, dest: Path, sha: str) -> None:
    if not (dest.is_file() and sha256_of(dest) == sha):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)


def fetch_examples(
    root: str | os.PathLike[str],
    files: Mapping[str, str] | None = None,
    *,
    commit: str = DSSAT_CSM_DATA_COMMIT,
) -> Path:
    """The example files (default :data:`EXAMPLE_FILES`) in ``<root>/example_data``, each checked
    against its SHA-256 (downloads are kept under :func:`cache_dir`); returns that directory."""
    ex = Path(root) / "example_data"
    store = cache_dir() / "downloads" / f"dssat-csm-data-{commit}"
    for rel, sha in (EXAMPLE_FILES if files is None else files).items():
        got = download(_DATA_URL.format(commit=commit, path=rel), store / rel, sha)
        _place(got, ex / rel, sha)
    return ex


def _fetch_data_files(root: Path, files: Mapping[str, str], archive: SourceArchive | None = None) -> None:
    """``files`` (``Data/...`` paths of ``dssat-csm-os``) from its archive into ``root``."""
    archive = archive or DSSAT_CSM_OS
    todo = {r: s for r, s in files.items() if not ((root / r).is_file() and sha256_of(root / r) == s)}
    if not todo:
        return
    top = _extract(_source_archive(archive), root / "Data", lambda name: name.split("/", 1)[-1] in todo)
    try:
        for rel, sha in todo.items():
            got = top / rel
            if not got.is_file() or sha256_of(got) != sha:
                raise ChecksumError(f"{archive.repo} {archive.tag}: {rel} missing or with another SHA-256")
            _place(got, root / rel, sha)
    finally:
        shutil.rmtree(top.parent, ignore_errors=True)


def _check_examples(engine: Path, files: Mapping[str, str]) -> list[str]:
    """The example files missing under ``engine/example_data`` or with another hash."""
    bad = []
    for rel, sha in files.items():
        p = engine / "example_data" / rel
        if not p.is_file():
            bad.append(f"{rel} (missing)")
        elif sha256_of(p) != sha:
            bad.append(f"{rel} (different SHA-256)")
    return bad


def _engine_data(root: Path) -> Path | None:
    """The data directory (``Genotype``, ``StandardData``) of a DSSAT root, if any."""
    for d in (root / "source" / "Data", root / "Data", root / "bin"):
        if (d / "Genotype").is_dir():
            return d
    return None


def fetch_dssat_data(root: str | os.PathLike[str] | None = None, *, verbose: bool = False) -> Path:
    """The root of DSSAT's public data (``example_data`` and ``Data/Genotype``, ``Data/StandardData``).

    With ``AGRI_JAX_DSSAT`` set to an engine root that holds the example experiments (the validation
    clusters' mirror): that root, nothing downloaded (a warning names example files missing or
    different). Otherwise ``root`` (default ``<cache_dir()>/dssat-data``): :data:`EXAMPLE_FILES`
    from ``dssat-csm-data`` and :data:`DATA_FILES` from the ``dssat-csm-os`` v4.8.6.0 archive,
    downloaded once and checked against their SHA-256."""
    env = os.environ.get("AGRI_JAX_DSSAT")
    if root is None and env:
        eng = Path(env).expanduser()
        if (eng / "example_data" / "Maize").is_dir() and _engine_data(eng) is not None:
            bad = _check_examples(eng, EXAMPLE_FILES)
            if bad:
                warnings.warn(f"example files under {eng}/example_data: {bad}", stacklevel=2)
            return eng
    r = Path(root).expanduser() if root is not None else cache_dir() / "dssat-data"
    fetch_examples(r, EXAMPLE_FILES)
    _fetch_data_files(r, DATA_FILES)
    if verbose:
        print(
            f"DSSAT data (DSSAT/dssat-csm-data {DSSAT_CSM_DATA_COMMIT[:10]}, DSSAT/dssat-csm-os "
            f"{DSSAT_CSM_OS.tag}; BSD-3): {r} ({len(EXAMPLE_FILES) + len(DATA_FILES)} files, SHA-256 checked)"
        )
    return r


def _use(engine: Path) -> None:
    """Make ``engine`` the session's default DSSAT engine (``AGRI_JAX_DSSAT`` and
    :data:`agrijax.port.run_fortran.DSSAT_ENGINE`, read by the calls that default to it)."""
    from agrijax.port import run_fortran

    os.environ["AGRI_JAX_DSSAT"] = str(engine)
    run_fortran.DSSAT_ENGINE = engine


def unpack_bundle(tgz: str | os.PathLike[str], root: str | os.PathLike[str]) -> Path:
    """Unpack a reference bundle (``scripts/release/make_dssat_bundle.py``) into ``root``; returns it."""
    r = Path(root)
    _check_replaceable(r, _BUNDLE_MARK)
    top = _extract(Path(tgz), r)
    try:
        if r.exists():
            shutil.rmtree(r)
        top.rename(r)
    finally:
        shutil.rmtree(top.parent, ignore_errors=True)
    return r


def install_reference(
    root: str | os.PathLike[str] | None = None,
    *,
    build: bool = False,
    jobs: int | None = None,
    verbose: bool = True,
    printer: Callable[[str], None] = print,
    asset: ReleaseAsset | None = None,
) -> Path:
    """The root of the DSSAT-CSM v4.8.6.0 reference program ``dscsm048`` (made the session's default,
    ``AGRI_JAX_DSSAT``), to compare Agri-JAX with DSSAT; Agri-JAX itself does not need it.

    * ``AGRI_JAX_DSSAT`` set to an engine root that holds ``dscsm048``: that root, nothing downloaded;
    * otherwise the prebuilt bundle :data:`REFERENCE_BUNDLE` (the validated build486, statically
      linked, with DSSAT's ``Data`` directory; SHA-256 checked) unpacked into ``root`` (default
      ``<cache_dir()>/dssat-4.8.6-build486``), once;
    * with ``build=True``, or when the bundle cannot be downloaded: the ``dssat-csm-os`` v4.8.6.0 source
      built with CMake and gfortran (a few minutes; needs ``gfortran`` and ``cmake``).

    An existing ``root`` that Agri-JAX did not make is never replaced (:class:`FileExistsError`); when
    nothing can be downloaded, :class:`ReferenceUnavailableError` says why."""
    from agrijax.port.run_fortran import dscsm_paths

    asset = asset or REFERENCE_BUNDLE
    say = printer if verbose else (lambda _m: None)
    env = os.environ.get("AGRI_JAX_DSSAT")
    if root is None and env and dscsm_paths(Path(env).expanduser())[0].is_file():
        eng = Path(env).expanduser()
        say(f"DSSAT-CSM reference program: {eng} (AGRI_JAX_DSSAT; nothing downloaded)")
        _use(eng)
        return eng
    failed = ""
    if not build:
        eng = Path(root).expanduser() if root is not None else cache_dir() / asset.tag
        if not dscsm_paths(eng)[0].is_file():
            _check_replaceable(eng, _BUNDLE_MARK)
            try:
                unpack_bundle(download(asset.url, cache_dir() / "downloads" / asset.name, asset.sha256), eng)
            except FileExistsError:
                raise
            except OSError as e:
                failed = f"the prebuilt bundle ({asset.url}): {e}; "
                warnings.warn(
                    f"the reference bundle could not be downloaded ({e}); building it from source",
                    stacklevel=2,
                )
                build = True
        if not build:
            say(f"DSSAT-CSM v4.8.6.0 reference program (prebuilt build486, {asset.tag}): {eng}")
            _use(eng)
            return eng
    eng = Path(root).expanduser() if root is not None else cache_dir() / f"dssat-{DSSAT_CSM_OS.tag}-src"
    say(f"DSSAT-CSM {DSSAT_CSM_OS.tag} source ({DSSAT_CSM_OS.repo} {DSSAT_CSM_OS.commit[:10]}), building ...")
    try:
        fetch_dssat_source(eng)
    except FileExistsError:
        raise
    except OSError as e:
        raise ReferenceUnavailableError(
            f"could not get the DSSAT reference program dscsm048: {failed}the DSSAT source "
            f"({DSSAT_CSM_OS.url}): {e}. Check the network connection, or set AGRI_JAX_DSSAT to a "
            "DSSAT-CSM v4.8.6.0 root that holds dscsm048. Only the comparisons with DSSAT need it; "
            "Agri-JAX's own runs do not."
        ) from e
    exe, secs = build_dscsm048(eng, jobs=jobs)
    say(f"dscsm048: {exe}" + (f" (built in {secs:.0f} s)" if secs else " (cached build)"))
    _use(eng)
    return eng
