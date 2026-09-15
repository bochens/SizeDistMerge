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


@pytest.mark.parametrize('name', ['pops', 'uhsas', 'pcasp'])
def test_toml_setup_roundtrip_and_default_channel(name):
    setup = optical_geometry.load_optical_setup(name)
    assert setup.response_channel in [channel.name for channel in setup.channels]
    assert setup.reference and setup.name
    restored = optical_geometry.OpticalSetup.from_dict(setup.to_dict())
    assert restored == setup
    for ri in (1.3, 1.615+.001j, 1.8+.1j):
        expected = optics.setup_csca([100, 1000, 5000], ri, setup)
        actual = optics.setup_csca([100, 1000, 5000], ri, restored)
        for channel in expected:
            np.testing.assert_array_equal(expected[channel], actual[channel])


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
                    'uhsas_geometry_cache', 'pops_csca_parallel', 'uhsas_csca_parallel',
                    'POPSGeom', 'UHSASGeom', 'PCASPGeom', 'pops_optical_setup',
                    'uhsas_optical_setup', 'pcasp_optical_setup', 'RI_POPS_SRC',
                    'RI_UHSAS_SRC', 'build_pops_sigma_lut', 'build_uhsas_sigma_lut',
                    'build_pcasp_sigma_lut', 'build_sigma_lut'):
        assert not hasattr(optics, removed)
        assert not hasattr(sizedistmerge, removed)


def test_missing_saved_setup_does_not_fall_back_to_current_toml():
    with pytest.raises(ValueError, match='lacks its full optical_setup'):
        optical_geometry.optical_setup_from_lut_metadata({
            'kernel': 'POPS', 'optical_model_version': optical_geometry.OPTICAL_MODEL_VERSION,
            'wavelength_nm': 405.})


def test_new_toml_name_requires_no_python_preset(tmp_path):
    custom = tmp_path/'another_opc.toml'
    custom.write_text((ROOT/'opc_setups/pops.toml').read_text().replace('name = "POPS"','name = "Another OPC"'))
    setup = optical_geometry.load_optical_setup(custom)
    assert setup.name == 'Another OPC'
    assert setup.response_channel == 'Collection'


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
for name in ('pops', 'uhsas', 'pcasp', 'grimm_11d_unpolarized_assumed'):
    setup = load_optical_setup(name)
    assert setup_csca([100.], 1.5, setup)
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path,
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_grimm_toml_reproduces_original_notebook_setup():
    # This is the numerical setup used by the original provisional LUT build.
    # It is a preservation check, not verification of the real 11-D optics.
    previous = optical_geometry.OpticalSetup(
        wavelength_nm=655.,
        beams=(
            optical_geometry.IncidentBeam((0, 0, 1), (1, 0, 0), .5),
            optical_geometry.IncidentBeam((0, 0, 1), (0, 1, 0), .5),
        ),
        channels=(optical_geometry.CollectionChannel('Photodiode', (
            optical_geometry.CollectionCone((0, 1, 0), 60.),
            optical_geometry.CollectionCone((0, -1, 0), 9.),
        )),),
        aerosol_direction=(-1, 0, 0), angular_step_deg=.125,
    )
    configured = optical_geometry.load_optical_setup('grimm_11d_unpolarized_assumed')
    from dataclasses import replace
    assert replace(configured, name='', reference='', notes='', response_channel=None) == previous
    from_path = optical_geometry.load_optical_setup(ROOT/'opc_setups/grimm_11d_unpolarized_assumed.toml')
    assert from_path == configured
    diameters = np.geomspace(200., 40000., 20)
    for refractive_index in (1.3, 1.59+.001j, 1.8+.1j):
        before = optics.setup_csca(diameters, refractive_index, previous)['Photodiode']
        after = optics.setup_csca(diameters, refractive_index, configured)['Photodiode']
        np.testing.assert_array_equal(before, after)
