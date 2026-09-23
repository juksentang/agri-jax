"""Placeholder so the tier collects; real differential tests arrive with the first ported subroutine."""

from pathlib import Path


def test_dumps_dir_exists(dumps_dir: Path) -> None:
    assert dumps_dir.is_dir()
