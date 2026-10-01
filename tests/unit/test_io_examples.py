"""``agrijax.io.examples`` without the network: downloads, hash checks, the cache, DSSAT's data from its
repositories, the reference program (an engine given by ``AGRI_JAX_DSSAT``; the prebuilt bundle; the
build from source as the fallback) with the network and the compilers mocked."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

import agrijax.io.examples as ex
from agrijax.port import run_fortran


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@pytest.fixture
def served(monkeypatch):
    """``{url: bytes}`` served by a mocked ``urlopen``; records the URLs asked for."""
    pages: dict[str, bytes] = {}
    asked: list[str] = []

    def fake(url, timeout):
        asked.append(url)
        if url not in pages:
            raise OSError(f"no network in the unit tier ({url})")
        return _Resp(pages[url])

    monkeypatch.setattr(ex, "_urlopen", fake)
    return pages, asked


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGRI_JAX_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("AGRI_JAX_DSSAT", "")  # restored after the test (install_reference sets it)
    monkeypatch.setattr(run_fortran, "DSSAT_ENGINE", run_fortran.DSSAT_ENGINE)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_download_checks_the_hash_and_reuses_the_file(tmp_path, served):
    pages, asked = served
    pages["https://x/a"] = b"hello"
    p = ex.download("https://x/a", tmp_path / "d" / "a.txt", _sha(b"hello"))
    assert p.read_bytes() == b"hello"
    ex.download("https://x/a", p, _sha(b"hello"))
    assert asked == ["https://x/a"]  # the verified file is reused
    with pytest.raises(ex.ChecksumError, match="SHA-256"):
        ex.download("https://x/a", tmp_path / "d" / "b.txt", _sha(b"other"))
    assert not (tmp_path / "d" / "b.txt").exists()
    assert [q.name for q in (tmp_path / "d").iterdir()] == ["a.txt"]  # no partial file left
    # a cached file with another hash is downloaded again
    p.write_bytes(b"tampered")
    ex.download("https://x/a", p, _sha(b"hello"))
    assert p.read_bytes() == b"hello" and len(asked) == 3


def test_example_files_come_from_the_pinned_commit(tmp_path, served):
    pages, asked = served
    files = {"Maize/X.MZX": _sha(b"x"), "Soil/S.SOL": _sha(b"s")}
    for rel, b in (("Maize/X.MZX", b"x"), ("Soil/S.SOL", b"s")):
        pages[f"https://raw.githubusercontent.com/DSSAT/dssat-csm-data/{ex.DSSAT_CSM_DATA_COMMIT}/{rel}"] = b
    d = ex.fetch_examples(tmp_path / "eng", files)
    assert (d / "Maize" / "X.MZX").read_bytes() == b"x" and (d / "Soil" / "S.SOL").read_bytes() == b"s"
    assert all(ex.DSSAT_CSM_DATA_COMMIT in u for u in asked)
    assert set(ex.EXAMPLE_FILES) >= {"Maize/UFGA8201.MZX", "Maize/UFGA8201.MZT", "Soil/SOIL.SOL"}
    assert all(len(h) == 64 for h in ex.EXAMPLE_FILES.values())


def _source_tgz(commit: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in (("CMakeLists.txt", b"project(x)\n"), ("Data/DSCSM048.CTR", b"ctr\n")):
            info = tarfile.TarInfo(f"dssat-csm-os-{commit}/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_source_archive_is_unpacked_once(tmp_path, served):
    pages, asked = served
    arch = ex.SourceArchive("DSSAT/dssat-csm-os", "v0", "abc123", "")
    tgz = _source_tgz(arch.commit)
    arch = ex.SourceArchive(arch.repo, arch.tag, arch.commit, _sha(tgz))
    pages[arch.url] = tgz
    assert arch.url == "https://codeload.github.com/DSSAT/dssat-csm-os/tar.gz/abc123"
    src = ex.fetch_dssat_source(tmp_path / "eng", arch)
    assert (src / "Data" / "DSCSM048.CTR").read_text() == "ctr\n"
    ex.fetch_dssat_source(tmp_path / "eng", arch)
    assert asked == [arch.url]
    bad = ex.SourceArchive(arch.repo, arch.tag, "def456", _sha(b"nope"))
    pages[bad.url] = _source_tgz("def456")
    with pytest.raises(ex.ChecksumError):
        ex.fetch_dssat_source(tmp_path / "eng2", bad)


def test_the_pinned_engine_is_v4860():
    assert ex.DSSAT_CSM_OS.tag == "v4.8.6.0"
    assert ex.DSSAT_CSM_OS.commit.startswith("d6556cca") and len(ex.DSSAT_CSM_OS.sha256) == 64


def test_build_needs_the_compilers(tmp_path, monkeypatch):
    (tmp_path / "source").mkdir()
    monkeypatch.setattr(ex.shutil, "which", lambda t: None)
    with pytest.raises(RuntimeError, match="gfortran"):
        ex.build_dscsm048(tmp_path)


def test_build_runs_cmake_and_is_cached(tmp_path, monkeypatch):
    (tmp_path / "source").mkdir()
    calls = []

    def fake_run(argv, cwd, log):
        calls.append(argv)
        if "--build" in argv:
            exe = tmp_path / "source" / "build486" / "bin" / "dscsm048"
            exe.parent.mkdir(parents=True)
            exe.write_text("")

    monkeypatch.setattr(ex.shutil, "which", lambda t: f"/usr/bin/{t}")
    monkeypatch.setattr(ex, "_run", fake_run)
    exe, secs = ex.build_dscsm048(tmp_path, jobs=2)
    assert exe.is_file() and secs >= 0.0
    assert calls[0][:2] == ["cmake", "-S"] and "-DCMAKE_BUILD_TYPE=RELEASE" in calls[0]
    assert calls[1][-2:] == ["-j", "2"]
    assert ex.build_dscsm048(tmp_path) == (exe, 0.0) and len(calls) == 2


def _fake_engine(root: Path, files: dict[str, bytes]) -> Path:
    exe = root / "source" / "build486" / "bin" / "dscsm048"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    (root / "source" / "Data" / "Genotype").mkdir(parents=True)
    for rel, b in files.items():
        p = root / "example_data" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    return root


def _tgz(top: str, files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_reference_from_agri_jax_dssat_downloads_nothing(tmp_path, monkeypatch, served):
    _, asked = served
    eng = _fake_engine(tmp_path / "mirror", {})
    monkeypatch.setenv("AGRI_JAX_DSSAT", str(eng))
    msgs: list[str] = []
    assert ex.install_reference(printer=msgs.append) == eng
    assert asked == [] and "nothing downloaded" in msgs[0] and run_fortran.DSSAT_ENGINE == eng


def test_reference_bundle_is_downloaded_checked_and_cached(tmp_path, served):
    pages, asked = served
    blob = _tgz("dssat-x", {"bin/dscsm048": b"ELF", "bin/Genotype/MZCER048.CUL": b"cul", "LICENSE": b"BSD"})
    asset = ex.ReleaseAsset("juksentang/agri-jax", "dssat-x", "dssat-x.tar.gz", _sha(blob))
    pages[asset.url] = blob
    assert asset.url == "https://github.com/juksentang/agri-jax/releases/download/dssat-x/dssat-x.tar.gz"
    eng = ex.install_reference(asset=asset, verbose=False)
    assert eng == tmp_path / "cache" / "dssat-x"
    assert run_fortran.dscsm_paths(eng) == (eng / "bin" / "dscsm048", eng / "bin")
    assert (eng / "LICENSE").read_bytes() == b"BSD"
    import os

    assert os.environ["AGRI_JAX_DSSAT"] == str(eng) and run_fortran.DSSAT_ENGINE == eng
    ex.install_reference(asset=asset, verbose=False)
    assert asked == [asset.url]  # unpacked once
    # the pinned bundle: the validated build486 on the project's release
    b = ex.REFERENCE_BUNDLE
    assert (b.repo, b.tag, b.name) == (
        "juksentang/agri-jax",
        "dssat-4.8.6-build486",
        "dssat-4.8.6-build486-linux-x86_64.tar.gz",
    )
    assert len(b.sha256) == 64 and b.sha256 != "0" * 64


def test_reference_falls_back_to_building_from_source(tmp_path, monkeypatch, served):
    asset = ex.ReleaseAsset("juksentang/agri-jax", "dssat-x", "dssat-x.tar.gz", "0" * 64)  # not served
    done = []
    monkeypatch.setattr(ex, "fetch_dssat_source", lambda root: done.append("source"))

    def build(root, jobs=None):
        done.append("build")
        return _fake_engine(Path(root), {}) / "source" / "build486" / "bin" / "dscsm048", 1.0

    monkeypatch.setattr(ex, "build_dscsm048", build)
    with pytest.warns(UserWarning, match="building it from source"):
        eng = ex.install_reference(asset=asset, verbose=False)
    assert eng == tmp_path / "cache" / "dssat-v4.8.6.0-src" and done == ["source", "build"]
    assert run_fortran.dscsm_paths(eng)[0].is_file()
    done.clear()
    eng2 = ex.install_reference(tmp_path / "b", build=True, verbose=False)
    assert eng2 == tmp_path / "b" and done == ["source", "build"]


def test_dssat_data_from_dssat_s_repositories(tmp_path, monkeypatch, served):
    pages, asked = served
    monkeypatch.setattr(ex, "EXAMPLE_FILES", {"Maize/X.MZX": _sha(b"x")})
    pages[
        f"https://raw.githubusercontent.com/DSSAT/dssat-csm-data/{ex.DSSAT_CSM_DATA_COMMIT}/Maize/X.MZX"
    ] = b"x"
    cul = b"*MAIZE CULTIVARS\r\n"
    blob = _tgz(
        "dssat-csm-os-abc", {"Data/Genotype/MZCER048.CUL": cul, "Data/Other.txt": b"o", "README.md": b"r"}
    )
    arch = ex.SourceArchive("DSSAT/dssat-csm-os", "v0", "abc", _sha(blob))
    pages[arch.url] = blob
    monkeypatch.setattr(ex, "DSSAT_CSM_OS", arch)
    monkeypatch.setattr(ex, "DATA_FILES", {"Data/Genotype/MZCER048.CUL": _sha(cul)})
    root = ex.fetch_dssat_data()
    assert root == tmp_path / "cache" / "dssat-data"
    assert (root / "example_data" / "Maize" / "X.MZX").read_bytes() == b"x"
    assert (root / "Data" / "Genotype" / "MZCER048.CUL").read_bytes() == cul
    assert not (root / "Data" / "Other.txt").exists()
    n = len(asked)
    ex.fetch_dssat_data()
    assert len(asked) == n  # cached and checked, nothing downloaded again
    monkeypatch.setattr(ex, "DATA_FILES", {"Data/Genotype/MZCER048.CUL": _sha(b"other")})
    with pytest.raises(ex.ChecksumError, match=r"MZCER048\.CUL"):
        ex.fetch_dssat_data(tmp_path / "d2")
    # the pinned data files: the maize genotype and the standard data the day reads
    assert set(ex.DATA_FILES) >= {"Data/Genotype/MZCER048.CUL"}


def test_pinned_data_files():
    assert set(ex.DATA_FILES) == {
        "Data/Genotype/MZCER048.CUL",
        "Data/Genotype/MZCER048.ECO",
        "Data/Genotype/MZCER048.SPE",
        "Data/StandardData/CO2048.WDA",
        "Data/StandardData/RESCH048.SDA",
    }
    assert all(len(h) == 64 for h in ex.DATA_FILES.values())


def test_dssat_data_of_a_mirror_downloads_nothing(tmp_path, monkeypatch, served):
    _, asked = served
    files = {"Maize/X.MZX": b"x"}
    monkeypatch.setattr(ex, "EXAMPLE_FILES", {k: _sha(v) for k, v in files.items()})
    eng = _fake_engine(tmp_path / "mirror", files)
    monkeypatch.setenv("AGRI_JAX_DSSAT", str(eng))
    assert ex.fetch_dssat_data() == eng and asked == []
    (eng / "example_data" / "Maize" / "X.MZX").write_bytes(b"changed")
    with pytest.warns(UserWarning, match="different SHA-256"):
        ex.fetch_dssat_data()


def test_a_directory_not_made_by_agri_jax_is_never_replaced(tmp_path, monkeypatch, served):
    """``install_reference(root)`` / ``fetch_dssat_source(root)`` on a user's existing directory
    refuse instead of deleting it (review finding: the unpack replaced ``root`` wholesale)."""
    pages, _ = served
    blob = _tgz("dssat-x", {"bin/dscsm048": b"ELF", "PROVENANCE.txt": b"p"})
    asset = ex.ReleaseAsset("juksentang/agri-jax", "dssat-x", "dssat-x.tar.gz", _sha(blob))
    pages[asset.url] = blob
    mine = tmp_path / "my_dssat"
    mine.mkdir()
    (mine / "notes.txt").write_text("keep me")
    with pytest.raises(FileExistsError, match="not made by Agri-JAX"):
        ex.install_reference(mine, asset=asset, verbose=False)
    assert (mine / "notes.txt").read_text() == "keep me"
    with pytest.raises(FileExistsError, match="not made by Agri-JAX"):
        ex.unpack_bundle(ex.download(asset.url, tmp_path / "b.tgz", asset.sha256), mine)
    assert (mine / "notes.txt").read_text() == "keep me"
    # an empty directory, or one a bundle was unpacked into before, is (re)filled
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run_fortran.dscsm_paths(ex.install_reference(empty, asset=asset, verbose=False))[0].is_file()
    (empty / "bin" / "dscsm048").unlink()
    assert run_fortran.dscsm_paths(ex.install_reference(empty, asset=asset, verbose=False))[0].is_file()
    # the source tree: only one Agri-JAX unpacked is replaced
    (mine / "source").mkdir()
    (mine / "source" / "main.for").write_text("keep me too")
    with pytest.raises(FileExistsError, match="not made by Agri-JAX"):
        ex.fetch_dssat_source(mine)
    assert (mine / "source" / "main.for").read_text() == "keep me too"


def test_reference_without_network_says_why(tmp_path, served):
    """Neither the bundle nor the source can be downloaded: one readable error naming both, not a
    bare ``URLError`` after a "building" message."""
    asset = ex.ReleaseAsset("juksentang/agri-jax", "dssat-x", "dssat-x.tar.gz", "0" * 64)  # not served
    with pytest.warns(UserWarning, match="building it from source"):
        with pytest.raises(ex.ReferenceUnavailableError) as e:
            ex.install_reference(asset=asset, verbose=False)
    msg = str(e.value)
    assert asset.url in msg and ex.DSSAT_CSM_OS.url in msg and "AGRI_JAX_DSSAT" in msg
    assert "own runs do not" in msg
    with pytest.raises(ex.ReferenceUnavailableError, match="network"):
        ex.install_reference(tmp_path / "src", build=True, verbose=False)
