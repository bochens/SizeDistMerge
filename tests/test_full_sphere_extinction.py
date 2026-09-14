"""Check the production angular integrator against a separate Mie-series calculation.

Reference equations: Bohren & Huffman, pp. 103 and 128; also reproduced at
https://miepython.readthedocs.io/en/latest/07_algorithm.html . The reference
below uses SciPy Bessel functions, not miepython amplitudes or efficiencies.
It is restricted to real refractive indices and moderate size parameters.
"""
import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn

from sizedistmerge.optical_geometry import OpticalSetup, IncidentBeam, CollectionChannel, CollectionCone
from sizedistmerge.optical_diameter import setup_csca


def bessel_series_efficiencies(size_parameter, refractive_index=1.5):
    """Return Qext and Qsca from the direct spherical-Bessel Mie series."""
    x = float(size_parameter)
    if not 0.1 <= x <= 1000 or not np.isreal(refractive_index) or refractive_index <= 1:
        raise ValueError('Reference supports real m>1 and 0.1<=x<=1000')
    # Retain terms beyond the usual x + 4*x^(1/3) + 2 cutoff.
    order = np.arange(1, int(np.ceil(x + 4*x**(1/3) + 12)) + 1)
    inside_x = refractive_index * x
    psi = x * spherical_jn(order, x)
    psi_derivative = spherical_jn(order, x) + x*spherical_jn(order, x, derivative=True)
    inside_psi = inside_x * spherical_jn(order, inside_x)
    inside_derivative = (spherical_jn(order, inside_x)
                         + inside_x*spherical_jn(order, inside_x, derivative=True))
    outgoing = psi + 1j*x*spherical_yn(order, x)
    outgoing_derivative = psi_derivative + 1j*(spherical_yn(order, x)
                                               + x*spherical_yn(order, x, derivative=True))
    electric = (refractive_index*inside_psi*psi_derivative - psi*inside_derivative) / (
        refractive_index*inside_psi*outgoing_derivative - outgoing*inside_derivative)
    magnetic = (inside_psi*psi_derivative - refractive_index*psi*inside_derivative) / (
        inside_psi*outgoing_derivative - refractive_index*outgoing*inside_derivative)
    multiplicity = 2*order + 1
    extinction = 2/x**2 * np.sum(multiplicity * (electric + magnetic).real)
    scattering = 2/x**2 * np.sum(multiplicity * (abs(electric)**2 + abs(magnetic)**2))
    return float(extinction), float(scattering)


def integrated_full_sphere_efficiency(size_parameters, refractive_index=1.5, step_deg=.025):
    """Use the current scattering code, then divide by particle projected area."""
    wavelength_nm = 550.
    diameters_nm = np.atleast_1d(size_parameters) * wavelength_nm / np.pi
    setup = OpticalSetup(wavelength_nm, (IncidentBeam(),),
        (CollectionChannel('Full sphere', (CollectionCone((0, 0, 1), 180.),)),),
        angular_step_deg=step_deg)
    cross_sections = setup_csca(diameters_nm, refractive_index, setup)['Full sphere']
    projected_area_um2 = np.pi * (diameters_nm * .5e-3)**2
    return cross_sections / projected_area_um2


@pytest.mark.parametrize('size_parameter', [0.1, 1., 5., 20., 50., 100., 200.])
def test_current_integral_matches_nonabsorbing_extinction(size_parameter):
    extinction, scattering = bessel_series_efficiencies(size_parameter)
    assert extinction == pytest.approx(scattering, rel=2e-13)
    integrated = integrated_full_sphere_efficiency(size_parameter)[0]
    assert integrated == pytest.approx(extinction, rel=4e-4)


def test_published_nonabsorbing_benchmark():
    # Prahl's algorithm notebook: m=1.55, diameter=1.05 um, wavelength=.6328 um.
    size_parameter = 2*np.pi*.525/.6328
    extinction, _ = bessel_series_efficiencies(size_parameter, 1.55)
    assert extinction == pytest.approx(3.1054255, abs=5e-8)
    assert integrated_full_sphere_efficiency(size_parameter, 1.55)[0] == pytest.approx(3.1054255, rel=1e-5)


def test_refining_angles_reduces_large_sphere_error():
    reference, _ = bessel_series_efficiencies(200.)
    errors = [abs(integrated_full_sphere_efficiency(200., step_deg=step)[0] - reference)
              for step in (.25, .05, .025)]
    assert errors[2] < errors[1] < errors[0]


def test_large_size_efficiency_is_near_two():
    # The approach is oscillatory, not a monotone or exactly-two condition.
    values = integrated_full_sphere_efficiency([150., 175., 200.])
    assert np.all(abs(values - 2.) < .15)
