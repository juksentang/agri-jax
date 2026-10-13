"""Placeholder so the tier collects; the CA-TPA day-by-day comparison arrives with the first model."""

from pathlib import Path


def test_catpa_scenario_present(catpa_scenario: Path) -> None:
    assert (catpa_scenario / "IPNAMES.DAT").is_file()
