"""Physical checks of the optical integral, independent of the production LUTs."""

from dataclasses import replace
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sizedistmerge import optical_diameter as od
import miepython as mie


def detector_coordinate_integral(diameter, ri, wavelength, outer, inner=0):
    """Independent integral in detector coordinates, not theta/dphi weights.

    Laser z; electric field x; collection axis y. Uniform solid angle is
    d(cos(gamma))*d(psi), where gamma is measured from the detector axis.
    Project the fixed electric field onto each actual scattering plane.
    """
    nodes, wg = leggauss(96)
    lo, hi = np.cos(np.deg2rad(outer)), np.cos(np.deg2rad(inner))
    cy = 0.5 * ((hi - lo) * nodes + hi + lo)
    wg = 0.5 * (hi - lo) * wg
    psi = 2 * np.pi * (np.arange(384) + 0.5) / 384
    transverse = np.sqrt(1 - cy**2)[:, None]
    nz = transverse * np.cos(psi)
    nx = transverse * np.sin(psi)
    s1, s2 = mie.S1_S2(ri, np.pi * diameter / wavelength, nz.ravel(), norm="qsca")
    # y/sqrt(x*x+y*y) is the perpendicular projection of the incident E.
    perp = cy[:, None]**2 / (1 - nz**2)
    para = nx**2 / (1 - nz**2)
    intensity = abs(s1.reshape(nz.shape))**2 * perp + abs(s2.reshape(nz.shape))**2 * para
    return np.pi * (diameter * 0.5e-3)**2 * np.sum(wg[:, None] * intensity) * 2*np.pi/384


@pytest.mark.parametrize("outer,inner", [(52., 0.), (57., 14.8)])
def test_isotropic_solid_angle_matches_exact_cone(outer, inner):
    c = od._side_collection_cache(outer, inner, 0.25)
    actual = np.trapezoid(c.dphi * np.sin(c.theta_rad), c.theta_rad)
    expected = 2*np.pi*(np.cos(np.deg2rad(inner)) - np.cos(np.deg2rad(outer)))
    assert actual == pytest.approx(expected, rel=3e-4)
    assert np.allclose(c.perp_phi + c.parallel_phi, c.dphi)
    assert np.all(c.perp_phi >= 0) and np.all(c.parallel_phi >= 0)


def test_uhsas_central_opening_is_114_degrees_not_180():
    width, _, _ = od._cone_azimuth_weights(np.array([np.pi/2]), 57.)
    assert np.rad2deg(width[0]) == pytest.approx(114.)
    c = od.uhsas_geometry_cache(od.UHSASGeom())
    center = np.argmin(abs(c.theta_rad - np.pi/2))
    assert np.rad2deg(c.dphi[center]) == pytest.approx(114. - 29.6)


@pytest.mark.parametrize("kind", ["pops", "uhsas"])
@pytest.mark.parametrize("ri", [1.3+0j, 1.615+0.001j, 1.8+0.1j])
@pytest.mark.parametrize("diameter", [90., 500., 2000.])
def test_mie_integral_against_independent_detector_coordinates(kind, ri, diameter):
    if kind == "pops":
        actual = od.pops_csca([diameter], ri, 405., geom=od.POPSGeom())[0]
        expected = detector_coordinate_integral(diameter, ri, 405., 52.)
    else:
        actual = od.uhsas_csca([diameter], ri, 1054., geom=od.UHSASGeom())[0]
        expected = detector_coordinate_integral(diameter, ri, 1054., 57., 14.8)
    # 0.1% numerical agreement; this is NOT an instrument-calibration tolerance.
    assert actual == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("outer,inner", [(52., 0.), (57., 14.8)])
def test_rayleigh_polarized_collection_fraction(outer, inner):
    d, wl, m = 1., 405., 1.52+0j
    c = od._side_collection_cache(outer, inner, 0.125)
    actual = od._collected_cross_section(d, m, wl, c)
    qsca = mie.efficiencies(m, d, wl)[1]
    total = np.pi * (d * 0.5e-3)**2 * qsca
    co, ci = np.cos(np.deg2rad(outer)), np.cos(np.deg2rad(inner))
    # Dipole radiation: 1-(E dot direction)^2, integrated over the annular cone.
    fraction = (3*(ci-co) + ci**3-co**3) / 8
    assert actual / total == pytest.approx(fraction, rel=2e-4)


def test_pops_direct_path_is_explicit_and_uses_its_own_distance():
    geom = od.POPSGeom()
    assert od.pops_geometry_cache(geom).direct is None
    with pytest.raises(ValueError, match="pmt_aperture_distance_mm"):
        od.pops_geometry_cache(replace(geom, pmt_aperture_d_mm=5.))
    # Synthetic geometry only: 20 mm is NOT a claimed POPS dimension.
    configured = replace(geom, pmt_aperture_d_mm=5., pmt_aperture_distance_mm=20.)
    c = od.pops_geometry_cache(configured)
    alpha = np.rad2deg(np.arctan(2.5/20.))
    expected = detector_coordinate_integral(200., 1.615+0.001j, 405., alpha)
    diff = (od.pops_csca([200.], 1.615+0.001j, 405., geom=configured)
            - od.pops_csca([200.], 1.615+0.001j, 405., geom=geom))[0]
    assert diff == pytest.approx(expected, rel=2e-3)
    assert c.direct is not None


@pytest.mark.parametrize("kind", ["pops", "uhsas"])
def test_angular_resolution_convergence(kind):
    geom = od.POPSGeom() if kind == "pops" else od.UHSASGeom()
    fn, wl = (od.pops_csca, 405.) if kind == "pops" else (od.uhsas_csca, 1054.)
    d = np.array([60., 200., 1000., 3000., 6000.])
    coarse = fn(d, 1.8+0.001j, wl, geom=geom)
    fine = fn(d, 1.8+0.001j, wl, geom=replace(geom, ring_step_deg=0.125))
    assert np.allclose(coarse, fine, rtol=2e-3, atol=0)


def test_numpy_and_numba_and_parallel_paths_agree(monkeypatch):
    d = [100., 500., 1000.]
    for fn, parallel, geom, wl in [(od.pops_csca, od.pops_csca_parallel, od.POPSGeom(), 405.),
                                    (od.uhsas_csca, od.uhsas_csca_parallel, od.UHSASGeom(), 1054.)]:
        accelerated = fn(d, 1.6+.001j, wl, geom=geom)
        threaded = parallel(d, 1.6+.001j, wl, geom=geom, n_jobs=2)
        with monkeypatch.context() as mp:
            mp.setattr(od, "_HAVE_NUMBA", False)
            plain = fn(d, 1.6+.001j, wl, geom=geom)
        assert np.allclose(accelerated, plain, rtol=1e-12, atol=0)
        assert np.array_equal(accelerated, threaded)


def test_wavelength_size_scaling():
    a = od.pops_csca([200.], 1.52+0j, 405., geom=od.POPSGeom())
    b = od.pops_csca([400.], 1.52+0j, 810., geom=od.POPSGeom())
    assert b[0] == pytest.approx(4*a[0])


@pytest.mark.parametrize("increasing", [True, False])
def test_forward_and_inverse_use_identical_curve_with_plateaus(increasing):
    d = np.geomspace(50., 3000., 101)
    sigma = d**2 * np.exp(0.7*np.sin(10*np.log(d)))
    if not increasing:
        sigma = 1/sigma
    f, inv = od.make_monotone_sigma_interpolator(d, sigma, response_bins=50,
                                               increasing=increasing)
    q = np.geomspace(100., 2000., 121)
    assert np.allclose(inv(f(q)), q, rtol=2e-12, atol=0)
    assert np.isnan(inv(0.))
    assert np.isnan(inv(np.inf))
    assert np.isnan(inv(1e100))


def test_same_ri_conversion_is_identity_even_for_oscillatory_curve():
    class ToyLUT:
        Dg = np.geomspace(30., 6000., 1000)

        def sigma_curve(self, d, n, k):
            return d**2 * np.exp(.6*np.sin(12*np.log(d))) * (n+k)

    lut = ToyLUT()
    d = np.geomspace(80., 3000., 101)
    out = od.convert_do_lut(d, 1.615+.001j, 1.615+.001j, lut, response_bins=100)
    assert np.allclose(out, d, rtol=2e-12, atol=0)


@pytest.mark.parametrize("kind", ["pops", "uhsas"])
def test_small_lut_is_versioned_matches_kernel_and_cannot_be_overwritten(tmp_path, kind):
    path = tmp_path / f"{kind}.zarr"
    geom, fn, wl = ((od.POPSGeom(), od.pops_csca, 405.) if kind == "pops"
                     else (od.UHSASGeom(), od.uhsas_csca, 1054.))
    args = dict(D_range=(100., 1000., 6), n_range=(1.5, 1.6, .1),
                k_values=(0., .001), chunks=(6, 2, 1), jobs_per_k=1)
    od.build_sigma_lut(str(path), kind, wl, geom, **args)
    lut = od.SigmaLUT(str(path))
    assert np.allclose(lut.sigma_curve(lut.Dg, 1.5, 0.),
                       fn(lut.Dg, 1.5+0j, wl, geom=geom), rtol=1e-6, atol=0)
    assert od.sigma_query_zarr(str(path), 100., 1.5, 0.) == pytest.approx(lut.SIG[0, 0, 0])
    with pytest.raises(FileExistsError):
        od.build_sigma_lut(str(path), kind, wl, geom, **args)
    root = zarr.open(str(path), mode="r+")
    assert root.attrs["optical_model_version"] == od.OPTICAL_MODEL_VERSION
    assert root.attrs["build_complete"] is True
    # Removing version simulates a historical table without changing any data.
    del root.attrs["optical_model_version"]
    with pytest.raises(ValueError, match="Rebuild"):
        od.SigmaLUT(str(path))
    with pytest.raises(ValueError, match="Rebuild"):
        od.sigma_query_zarr(str(path), 100., 1.5, 0.)
    with pytest.warns(UserWarning, match="legacy"):
        old = od.SigmaLUT(str(path), allow_legacy=True)
    assert np.array_equal(old.SIG, lut.SIG)
    root.attrs["build_complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        od.SigmaLUT(str(path), allow_legacy=True)
