import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_import_sizedistmerge_from_outside_repo(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    code = """
import sys
import sizedistmerge as sdm
assert "miepython" not in sys.modules
assert sdm.edges_from_mids_geometric([10, 20, 40]).shape == (4,)
assert isinstance(sdm.da_to_dv(100.0, rho_p=1000.0), float)
assert callable(sdm.calculate_dry_diameter)
assert callable(sdm.kappa_from_growth_factor)
assert "miepython" not in sys.modules
assert callable(sdm.POPSGeom)
assert "miepython" in sys.modules
assert callable(sdm.merge_sizedists_tikhonov_consensus)
assert callable(sdm.number_to_surface_area_spectrum)
assert not hasattr(sdm, "twomey_inversion")
assert sdm.lut_path("uhsas").name == "uhsas_sigma_col_1054nm.zarr"
assert sdm.lut_path("pops").name == "pops_sigma_col_405nm.zarr"
assert "sizedistmerge/data/lut" in sdm.lut_path("uhsas").as_posix()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_count_conserving_rebin_preserves_total_counts():
    sys.path.insert(0, str(SRC))
    import sizedistmerge as sdm

    old_edges = np.array([10.0, 100.0, 1000.0])
    new_edges = np.array([10.0, 31.6227766017, 100.0, 316.227766017, 1000.0])
    old_y = np.array([5.0, 10.0])

    new_y = sdm.rebin_dndlog_by_edges_overlap(old_edges, new_edges, old_y)

    old_counts = np.sum(old_y * np.diff(np.log10(old_edges)))
    new_counts = np.nansum(new_y * np.diff(np.log10(new_edges)))
    assert np.isclose(new_counts, old_counts)


def test_validation_errors_are_clear_for_bad_grids():
    sys.path.insert(0, str(SRC))
    import sizedistmerge as sdm

    try:
        sdm.edges_from_mids_geometric([20.0, 10.0])
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-increasing mids")


def test_package_has_no_twomey_module():
    assert not (SRC / "sizedistmerge" / "twomey.py").exists()
    assert not (SRC / "sizedistmerge" / "twomey_inversion.py").exists()
