"""PCASP checks against a separate dimensional Mie integral, not a LUT."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
import pytest
import zarr
from sizedistmerge import optical_geometry, optical_lut

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sizedistmerge import optical_diameter as od
import miepython as mie


def dimensional_band_integral(diameter, ri, lo, hi):
    """Full azimuth: pi*(|S1|^2+|S2|^2)/k^2 integrated in cos(theta).

    Uses unnormalized Wiscombe amplitudes and a Gauss rule, independent of
    the implementation's qsca normalization and trapezoidal theta grid.
    """
    nodes, weights = leggauss(192)
    a, b = np.cos(np.deg2rad(hi)), np.cos(np.deg2rad(lo))
    mu = (a+b)/2 + (b-a)/2*nodes
    s1, s2 = mie.S1_S2(ri, np.pi*diameter/632.8, mu, norm="wiscombe")
    wave_number_um = 2*np.pi/.6328
    return np.pi/wave_number_um**2*(b-a)/2*np.dot(weights, abs(s1)**2+abs(s2)**2)


def test_pcasp_angles_and_paper_weight():
    setup = optical_geometry.load_optical_setup("pcasp")
    theta = np.deg2rad([20., 40., 75., 130., 160.])
    azimuth = np.linspace(0, 2*np.pi, 17)
    outgoing = setup.channels[0].collect[0].directions(theta[:, None], azimuth)
    direct = setup.channels[0].accepts(outgoing)
    reverse = setup.channels[0].accepts(-outgoing)
    assert np.all(direct == np.array([False, True, True, False, False])[:, None])
    assert np.all(reverse == np.array([False, False, True, True, False])[:, None])
    # Paper uses a weight of 1 in each shoulder and 2 in the overlap.
    assert np.all(direct.astype(int)+reverse == np.array([0, 1, 2, 1, 0])[:, None])
    assert [b.irradiance_fraction for b in setup.beams] == [.5, .5]


@pytest.mark.parametrize("diameter", [70., 100., 500., 1000., 3000., 5000.])
@pytest.mark.parametrize("ri", [1.3+0j, 1.58+0j, 1.8+.1j])
def test_independent_dimensional_mie(diameter, ri):
    expected = .5*(dimensional_band_integral(diameter, ri, 35., 120.)
                   + dimensional_band_integral(diameter, ri, 60., 145.))
    assert od.setup_csca([diameter], ri, optical_geometry.load_optical_setup("pcasp"))["Collection"][0] == pytest.approx(expected, rel=2e-4)


def test_coaxial_band_solid_angle_and_endpoint_values():
    setup = optical_geometry.load_optical_setup("pcasp")
    expected = 2*np.pi*(np.cos(np.deg2rad(35))-np.cos(np.deg2rad(120)))
    for c in od.setup_geometry_cache(setup)["Collection"]:
        actual = np.trapezoid(c.dphi*np.sin(c.theta_rad), c.theta_rad)
        assert actual == pytest.approx(expected, rel=2e-6)
        # No artificial half-height sample at either rim.
        assert np.all(c.dphi == 2*np.pi)
        assert np.all(c.perp_phi == np.pi)
        assert np.all(c.parallel_phi == np.pi)


def test_polarization_and_beam_normalization():
    setup = optical_geometry.load_optical_setup("pcasp")
    rotated = replace(setup, beams=tuple(replace(b, polarization=(0, 1, 0)) for b in setup.beams))
    d = [100., 1000., 5000.]
    assert np.array_equal(od.setup_csca(d, 1.58, setup)["Collection"],
                          od.setup_csca(d, 1.58, rotated)["Collection"])
    for ratio in [0., .999, 1., .4]:
        beams = (replace(setup.beams[0],irradiance_fraction=1/(1+ratio)),)
        if ratio:
            beams += (replace(setup.beams[1],irradiance_fraction=ratio/(1+ratio)),)
        actual = od.setup_csca(d,1.58,replace(setup,beams=beams))["Collection"]
        expected = [(dimensional_band_integral(x, 1.58, 35, 120)
                     +ratio*dimensional_band_integral(x, 1.58, 60, 145))/(1+ratio) for x in d]
        assert np.allclose(actual, expected, rtol=2e-4)


def test_convergence():
    d = np.geomspace(70, 5000, 61)
    coarse = od.setup_csca(d, 1.8+.001j, optical_geometry.load_optical_setup("pcasp"))["Collection"]
    fine = od.setup_csca(d, 1.8+.001j, replace(optical_geometry.load_optical_setup("pcasp"), angular_step_deg=.125))["Collection"]
    assert np.max(abs(coarse/fine-1)) < 2e-4


@pytest.mark.parametrize('value', [-1., 181., float('nan')])
def test_invalid_geometry(value):
    with pytest.raises(ValueError):
        optical_geometry.CollectionCone((0,0,1),value)


def test_pcasp_lut_metadata_and_shared_calculation(tmp_path):
    path = tmp_path/"pcasp.zarr"
    optical_lut.build_setup_sigma_lut(str(path), optical_geometry.load_optical_setup("pcasp"), D_range=(100., 2000., 5),
                            n_range=(1.5, 1.6, .1), k_values=(0., .001), jobs_per_k=1)
    root = zarr.open_group(path, mode="r")
    assert root.attrs["instrument"] == "PCASP"
    assert sum(beam.irradiance_fraction for beam in optical_geometry.optical_setup_from_lut_metadata(root.attrs).beams) == 1
    assert root.attrs["build_complete"] is True
    setup = optical_geometry.optical_setup_from_lut_metadata(root.attrs)
    d = root["coords/D_nm"][:]
    assert setup == optical_geometry.load_optical_setup("pcasp")
    assert np.allclose(root["sigma_col"][:, 0, 0], od.setup_csca(d, 1.5, setup)["Collection"], rtol=1e-6)
    with pytest.raises(FileExistsError):
        optical_lut.build_setup_sigma_lut(str(path), optical_geometry.load_optical_setup("pcasp"))


def test_public_api():
    import sizedistmerge as sdm
    assert sdm.build_setup_sigma_lut is optical_lut.build_setup_sigma_lut
    assert sdm.load_optical_setup is optical_geometry.load_optical_setup
    for removed in ('PCASPGeom','PCASP_WAVELENGTH_NM','pcasp_optical_setup','build_pcasp_sigma_lut'):
        assert not hasattr(sdm,removed)


def test_packaged_pcasp_lut():
    from sizedistmerge import lut_path

    path = lut_path(" PCASP ")
    assert path.name == "pcasp_sigma_col_632p8nm.zarr"
    lut = optical_lut.SigmaLUT(path)
    root = zarr.open_group(path, mode="r")
    assert root.attrs["instrument"] == "PCASP"
    assert root.attrs["build_complete"] is True
    assert lut.SIG.shape == (1000, 1001, 200)
    setup = optical_geometry.optical_setup_from_lut_metadata(root.attrs)
    configured = optical_geometry.load_optical_setup("pcasp")
    assert setup == configured
    indices = [0, 250, 500, 750, 999]
    d = lut.Dg[indices]
    ri = complex(lut.ng[560], lut.kg[9])
    expected = od.setup_csca(d, ri, setup)["Collection"]
    np.testing.assert_allclose(lut.SIG[indices, 560, 9], expected, rtol=1e-6, atol=0.)


def test_saved_mieconscat_reference():
    """Original MieConScat 1.1.8 / Wiscombe solver, not miepython.

    Source archive SHA256:
    b90749e5c8445d897ef1490eac723673904a85bb98213cd5b7d06a10552106d1
    Values are means of the 35-120 and 60-145 degree integrals, in um^2.
    The original comparison notebook is retained locally, not shipped as an example.
    """
    d = [60.0, 189.73665961010286, 600.0000000000003, 1897.3665961010286, 6000.0]
    cases = [
        ((1.3+0j), [1.190209960593204e-06, 0.0010150896101969627, 0.08962920769725846, 1.1335345186923464, 5.918199334375292]),
        ((1.58+0j), [3.847637178487575e-06, 0.0037287636938788194, 0.286882294576591, 1.2764560949999753, 12.525902204492544]),
        ((1.58+0.001j), [3.847597329157092e-06, 0.003726520214402105, 0.28479638611009656, 1.239563772149312, 11.129503286470822]),
        ((1.8+0.8j), [1.301686000827302e-05, 0.00959844060408238, 0.07731472990744512, 0.38525801633827983, 3.426582339116197]),
    ]
    for ri, expected in cases:
        np.testing.assert_allclose(od.setup_csca(d, ri, optical_geometry.load_optical_setup("pcasp"))["Collection"], expected, rtol=1e-4, atol=0.)
