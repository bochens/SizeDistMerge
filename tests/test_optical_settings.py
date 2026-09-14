"""Settings files must reproduce the optics, not just look like the drawings."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from sizedistmerge import optical_diameter as optics
from sizedistmerge import optical_geometry, optical_lut

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name,settings_type,factory', [
    ('pops', optical_geometry.POPSGeom, optical_geometry.pops_optical_setup),
    ('uhsas', optical_geometry.UHSASGeom, optical_geometry.uhsas_optical_setup),
    ('pcasp', optical_geometry.PCASPGeom, optical_geometry.pcasp_optical_setup),
])
def test_toml_reproduces_legacy_geometry_and_cross_sections(name, settings_type, factory):
    configured = optical_geometry.load_optical_setup(name)
    legacy = factory(settings_type())
    assert configured.to_dict() == legacy.to_dict()
    assert factory().to_dict() == legacy.to_dict()
    diameters = np.geomspace(60., 6000., 25)
    for refractive_index in (1.3, 1.615 + .001j, 1.8 + .1j):
        before = optics.setup_csca(diameters, refractive_index, legacy)
        after = optics.setup_csca(diameters, refractive_index, configured)
        for channel in before:
            np.testing.assert_array_equal(before[channel], after[channel])


def test_custom_settings_reject_misspellings_and_invalid_polarization(tmp_path):
    text = (ROOT/'opc_setups/pops.toml').read_text()
    custom = tmp_path/'custom.toml'
    custom.write_text(text.replace('half_angle_deg', 'half_angel_deg'))
    with pytest.raises(ValueError, match='Unknown CollectionCone'):
        optical_geometry.load_optical_setup(custom)
    custom.write_text(text.replace('polarization = [1.0, 0.0, 0.0]', 'polarization = [0.0, 0.0, 1.0]'))
    with pytest.raises(ValueError, match='perpendicular'):
        optical_geometry.load_optical_setup(custom)
    custom.write_text(text.replace('half_angle_deg = 52.0', 'half_angle_deg = 40.0'))
    assert optical_geometry.load_optical_setup(custom).channels[0].collect[0].half_angle_deg == 40.


def test_public_exports_come_from_their_own_modules():
    import sizedistmerge
    assert sizedistmerge.SigmaLUT is optical_lut.SigmaLUT
    assert sizedistmerge.load_optical_setup is optical_geometry.load_optical_setup
    assert sizedistmerge.setup_csca is optics.setup_csca
    for removed in ('pops_csca', 'uhsas_csca', 'pcasp_csca', 'pops_geometry_cache',
                    'uhsas_geometry_cache', 'pops_csca_parallel', 'uhsas_csca_parallel'):
        assert not hasattr(optics, removed)
        assert not hasattr(sizedistmerge, removed)


def test_installed_style_settings_and_import_order(tmp_path):
    # Copy only source and settings, not the large LUT data. This simulates
    # an installation with no checkout/pyproject.toml beside the package.
    package = tmp_path/'sizedistmerge'
    shutil.copytree(ROOT/'src', package, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'opc_setups', package/'opc_setups')
    environment = dict(os.environ, PYTHONPATH=str(tmp_path))
    code = '''
from sizedistmerge.optical_lut import SigmaLUT
from sizedistmerge.optical_geometry import load_optical_setup
from sizedistmerge.optical_diameter import setup_csca
for name in ('pops', 'uhsas', 'pcasp'):
    setup = load_optical_setup(name)
    assert setup_csca([100.], 1.5, setup)
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path,
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
