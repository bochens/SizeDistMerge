"""Shared geometry: orientation, masks, polarization and saved provenance."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sizedistmerge import optical_diameter as od
from sizedistmerge.optical_geometry import channel_azimuth_weights


def test_cone_solid_angle_boundary_and_validation():
    cone = od.CollectionCone.from_solid_angle((1, 2, 3), 1.7)
    assert cone.solid_angle_sr == pytest.approx(1.7)
    rim = cone.boundary(np.linspace(0, 2*np.pi, 101))
    assert np.allclose(np.linalg.norm(rim, axis=1), 1)
    assert np.allclose(rim @ cone.axis, np.cos(np.deg2rad(cone.half_angle_deg)))
    with pytest.raises(ValueError):
        od.CollectionCone((0, 0, 0), 30)
    with pytest.raises(ValueError):
        od.IncidentBeam(polarization=(0, 0, 1))
    with pytest.raises(ValueError):
        od.OpticalSetup(405, (od.IncidentBeam(irradiance_fraction=.5),),
                        (od.CollectionChannel("a", (cone,)),))


def test_union_exclusions_and_seam_match_direction_membership():
    beam = od.IncidentBeam()
    channel = od.CollectionChannel('detector',
        (od.CollectionCone((1, 1, 1), 55), od.CollectionCone((-1, 1, 0), 65)),
        (od.CollectionCone((0, 1, .1), 20), od.CollectionCone((.2, 1, .1), 25)))
    phi = (np.arange(40000)+.5)*2*np.pi/40000
    for theta in np.deg2rad([20, 60, 90, 115, 150]):
        vectors = (np.cos(theta)*np.array(beam.direction) + np.sin(theta)
                   * (np.cos(phi)[:, None]*beam.transverse
                      + np.sin(phi)[:, None]*np.array(beam.polarization)))
        accepted = channel.accepts(vectors)
        actual = np.array(channel_azimuth_weights(np.array([theta]), beam, channel))[:, 0]
        expected = np.array([accepted.sum(), np.sum(accepted*np.cos(phi)**2),
                             np.sum(accepted*np.sin(phi)**2)])*2*np.pi/len(phi)
        assert np.allclose(actual, expected, atol=4e-4, rtol=0)


@pytest.mark.parametrize('axis', [(0, 0, 1), (1, 0, 0), (1, 2, 3), (0, 0, -1)])
def test_arbitrary_direction_dipole_limit(axis):
    cone = od.CollectionCone(axis, 35)
    setup = od.OpticalSetup(405, (od.IncidentBeam(),),
                            (od.CollectionChannel('detector', (cone,)),), angular_step_deg=.125)
    actual = od.setup_csca([1.], 1.52, setup)['detector'][0]
    total = np.pi*(.5e-3)**2*od.mie.efficiencies(1.52, 1., 405.)[1]
    a2 = np.dot(cone.axis, setup.beams[0].polarization)**2
    c = np.cos(np.deg2rad(cone.half_angle_deg))
    fraction = .75*((1+a2)/2*(1-c) - (3*a2-1)/6*(1-c**3))
    assert actual/total == pytest.approx(fraction, rel=8e-4)


def test_tilted_cone_against_independent_detector_coordinate_quadrature():
    cone = od.CollectionCone((.3, .7, .8), 43.)
    setup = od.OpticalSetup(700., (od.IncidentBeam(),),
                           (od.CollectionChannel('tilted', (cone,)),), angular_step_deg=.125)
    x, w = leggauss(100)
    lo = np.cos(np.deg2rad(cone.half_angle_deg))
    mu = (1-lo)*x/2+(1+lo)/2
    w *= (1-lo)/2
    az = (np.arange(384)+.5)*2*np.pi/384
    directions = cone.directions(np.arccos(mu)[:, None], az[None, :])
    dcs = od.directional_cross_section(500., 1.6+.01j, setup, directions)
    expected = np.sum(w[:, None]*dcs)*2*np.pi/384
    actual = od.setup_csca([500.], 1.6+.01j, setup)['tilted'][0]
    assert actual == pytest.approx(expected, rel=5e-4)


def test_rotating_all_inputs_together_preserves_result():
    cone = od.CollectionCone((.3, .7, .8), 43.)
    setup = od.OpticalSetup(700., (od.IncidentBeam(),), (od.CollectionChannel('a', (cone,)),))
    # A proper rotation, not a change of the angle between E and the collector.
    q, _ = np.linalg.qr(np.array([[1., 2., 3.], [-1., .2, 4.], [2., 3., 1.]]))
    beam = setup.beams[0]
    rotated = replace(setup, beams=(od.IncidentBeam(q @ beam.direction, q @ beam.polarization),),
                      channels=(od.CollectionChannel('a', (od.CollectionCone(q @ cone.axis, 43.),)),))
    assert np.allclose(od.setup_csca([100., 500., 1000.], 1.6, setup)['a'],
                       od.setup_csca([100., 500., 1000.], 1.6, rotated)['a'], rtol=1e-12)


def test_uhsas_channels_and_total_irradiance_no_factor_two():
    setup = od.uhsas_optical_setup()
    result = od.setup_csca([100., 500., 1000.], 1.52, setup)
    original = od.uhsas_csca([100., 500., 1000.], 1.52, 1054., geom=od.UHSASGeom())
    assert list(result) == ['Collection 1', 'Collection 2']
    for value in result.values():
        assert np.array_equal(value, original)


def test_full_sphere_directional_integral_is_total_scattering():
    setup = od.uhsas_optical_setup()
    mu, weights = leggauss(100)
    phi = (np.arange(120)+.5)*2*np.pi/120
    dirs = np.stack(np.broadcast_arrays(
        np.sqrt(1-mu[:, None]**2)*np.cos(phi), np.sqrt(1-mu[:, None]**2)*np.sin(phi),
        mu[:, None]), axis=-1)
    actual = np.sum(od.directional_cross_section(500., 1.52, setup, dirs)*weights[:, None])*2*np.pi/120
    expected = np.pi*.25**2*od.mie.efficiencies(1.52, 500., 1054.)[1]
    assert actual == pytest.approx(expected, rel=1e-11)


def test_custom_lut_stores_full_setup_and_selected_channel(tmp_path):
    setup = od.OpticalSetup(650, (od.IncidentBeam(),), (od.CollectionChannel(
        'tilted', (od.CollectionCone((0, 1, 1), 40),), (od.CollectionCone((0, 1, 1), 10),)),))
    assert od.OpticalSetup.from_dict(json.loads(json.dumps(setup.to_dict()))) == setup
    path = tmp_path/'custom.zarr'
    od.build_setup_sigma_lut(str(path), setup, channel='tilted',
        D_range=(100, 600, 3), n_range=(1.5, 1.6, .1), k_values=(0., .001), chunks=(3, 2, 1), jobs_per_k=1)
    root = zarr.open(str(path), mode='r')
    assert root.attrs['response_channels'] == ['tilted']
    assert od.optical_setup_from_lut_metadata(root.attrs) == setup
    lut = od.SigmaLUT(str(path))
    assert np.allclose(lut.SIG[:, 0, 0], od.setup_csca(lut.Dg, 1.5, setup)['tilted'], rtol=1e-7)
    with pytest.raises(ValueError, match='select exactly one'):
        od.build_setup_sigma_lut(str(tmp_path/'bad.zarr'), setup, channel='absent')
    assert not (tmp_path/'bad.zarr').exists()


# Values captured before the shared-interface edit (checkpoint 960c1ee).
# They protect production defaults independently of the new setup factories.
PRE_INTERFACE_VALUES = [{"name":"pops","D":[60,100,300,1000,3000,5000],"ri":[1.3,0],"sigma":[0.000002890416255571536,0.00005768060667035498,0.0068684533987805295,0.08225418799890577,0.49016005818499386,1.3916039533230757]},{"name":"pops","D":[60,100,300,1000,3000,5000],"ri":[1.615,0.001],"sigma":[0.000010621938680515722,0.00022853338676618215,0.019697650387839112,0.09916915682655313,0.7553849985239482,2.027041908487524]},{"name":"pops","D":[60,100,300,1000,3000,5000],"ri":[1.8,0.1],"sigma":[0.00001654160586805507,0.00036124230023324236,0.016202155335005464,0.034092339746772574,0.22521039366917792,0.6057805762493121]},{"name":"uhsas","D":[60,100,300,1000,3000,5000],"ri":[1.3,0],"sigma":[6.748510400976342e-8,0.0000014372442912587205,0.0009157667187144968,0.10444388115139802,1.0427615484437947,2.6815746265487834]},{"name":"uhsas","D":[60,100,300,1000,3000,5000],"ri":[1.615,0.001],"sigma":[2.3701173361593797e-7,0.000005120946736809653,0.0037219694581012036,0.27538025055085563,1.3444917312492024,3.287463282336221]},{"name":"uhsas","D":[60,100,300,1000,3000,5000],"ri":[1.8,0.1],"sigma":[3.6267376044275517e-7,0.000007883995307992212,0.005859004841796273,0.13240479014591122,0.35102111901919014,0.7543707041550171]}]


def test_preset_values_are_bitwise_unchanged_from_pre_interface_snapshot():
    for row in PRE_INTERFACE_VALUES:
        ri = complex(*row['ri'])
        if row['name'] == 'pops':
            actual = od.pops_csca(row['D'], ri, 405., geom=od.POPSGeom())
            shared = od.setup_csca(row['D'], ri, od.pops_optical_setup())['Collection']
        else:
            actual = od.uhsas_csca(row['D'], ri, 1054., geom=od.UHSASGeom())
            shared = od.setup_csca(row['D'], ri, od.uhsas_optical_setup())['Collection 1']
        assert np.array_equal(actual, row['sigma'])
        assert np.array_equal(shared, row['sigma'])
