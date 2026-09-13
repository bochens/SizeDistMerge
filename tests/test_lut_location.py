"""The repository owns one LUT directory; installed packages retain the tables."""

from pathlib import Path

import pytest

from sizedistmerge import resources


@pytest.mark.parametrize("kind", ["pops", "uhsas", "pcasp"])
def test_checkout_uses_top_level_lut(kind):
    root = Path(__file__).resolve().parents[1]
    assert resources.lut_path(kind) == root / "lut" / resources._LUT_NAMES[kind]
    assert not (root / "src/sizedistmerge/data/lut").exists()


def test_installed_lookup_uses_packaged_lut(tmp_path, monkeypatch):
    installed = tmp_path / "site-packages/sizedistmerge"
    table = installed / "lut" / resources._LUT_NAMES["pops"]
    table.mkdir(parents=True)
    monkeypatch.setattr(resources, "__file__", str(installed / "resources.py"))
    monkeypatch.setattr(resources.resources, "files", lambda package: installed / "lut")
    assert resources.lut_path("pops") == table
