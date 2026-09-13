"""Keep published notebooks portable and safe to run as examples."""

import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_flat_source_package_mapping():
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
    mapping = configuration["tool"]["setuptools"]["package-dir"]
    assert mapping["sizedistmerge"] == "src"
    assert mapping["sizedistmerge.lut"] == "lut"
    assert (ROOT / "src/__init__.py").is_file()
    assert not (ROOT / "src/sizedistmerge").exists()


def test_only_clean_example_notebooks_are_public():
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(paths) == 4
    for path in paths:
        assert path.name.endswith("_example.ipynb")
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            assert "/Users/" not in source
            assert "/Volumes/" not in source
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert not cell["outputs"]
                compile(source, str(path), "exec")


def test_expensive_example_steps_are_opt_in():
    build = (ROOT / "notebooks/build_optical_luts_example.ipynb").read_text()
    campaign = (ROOT / "notebooks/arcsix_production_example.ipynb").read_text()
    assert "RUN_BUILD = False" in build
    assert "RUN_MERGE = False" in campaign
    assert "RUN_QC = False" in campaign
