"""Humidity ratios must change optical size without changing particle number."""

import numpy as np
import pytest
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline

from sizedistmerge import kappa_kohler as humidity


DIAMETERS_NM = np.geomspace(30., 1000., 61)
SPECTRUM = 100. * np.exp(-0.5 * (np.log(DIAMETERS_NM / 150.) / 0.5)**2)


def test_growth_receives_metres_and_returns_nm(monkeypatch):
    def grow(rh, diameter_m, kappa):
        np.testing.assert_allclose(diameter_m, DIAMETERS_NM * 1e-9)
        assert rh == 0.8
        return diameter_m * 1.5

    monkeypatch.setattr(humidity, 'calculate_wet_diameter', grow)
    wet_nm, rh = humidity._humidified_diameters_nm(DIAMETERS_NM, 80., 0.3)
    np.testing.assert_allclose(wet_nm, DIAMETERS_NM * 1.5)
    assert rh == 0.8


@pytest.mark.parametrize('sulfate', [False, True])
def test_real_mie_matches_explicit_dry_coordinate_integral(sulfate):
    mie = humidity._miepython_module()
    wet_nm = humidity.calculate_wet_diameter(0.8, DIAMETERS_NM * 1e-9, 0.3) * 1e9
    if sulfate:
        index_curve = CubicSpline(
            [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [1.453, 1.44, 1.428, 1.417, 1.403, 1.379, 1.335])
        dry_index, wet_index = complex(index_curve(0.3)), complex(index_curve(0.8))
        result = humidity.calculate_humidification_factor_ammonium_sulfate(
            DIAMETERS_NM, SPECTRUM, 0.8, 0.3, 550.)
    else:
        dry_fraction = (DIAMETERS_NM / wet_nm)**3
        dry_index = 1.5 - 0.01j
        wet_index = 1.5 * dry_fraction + 1.33 * (1 - dry_fraction) - 0.01j
        result = humidity.calculate_humidification_factor(
            DIAMETERS_NM, SPECTRUM, 0.8, 0.3, 550., 1.5, 0.01)

    # Compute each particle's cross-section separately, then integrate using
    # the original spectrum coordinate. This is independent of the helper path.
    wet_efficiencies = np.array([
        mie.efficiencies(complex(np.broadcast_to(wet_index, wet_nm.shape)[i]), diameter, 550.)[:3]
        for i, diameter in enumerate(wet_nm)])
    dry_efficiencies = np.array([
        mie.efficiencies(dry_index, diameter, 550.)[:3] for diameter in DIAMETERS_NM])
    wet_total = simpson(wet_efficiencies * (np.pi / 4 * wet_nm**2 * SPECTRUM)[:, None],
                        x=np.log10(DIAMETERS_NM), axis=0)
    dry_total = simpson(dry_efficiencies * (np.pi / 4 * DIAMETERS_NM**2 * SPECTRUM)[:, None],
                        x=np.log10(DIAMETERS_NM), axis=0)
    np.testing.assert_allclose(result, wet_total / dry_total, rtol=1e-12)
    assert np.all(np.isfinite(result))


def test_size_dependent_growth_preserves_number(monkeypatch):
    # Artificial efficiencies give every size the same cross-section. Changing
    # the size coordinate must therefore leave all bulk coefficients unchanged.
    class ConstantCrossSection:
        @staticmethod
        def efficiencies_mx(index, size_parameter):
            efficiency = 1. / size_parameter**2
            return efficiency, efficiency, efficiency, np.zeros_like(efficiency)

    monkeypatch.setattr(humidity, '_miepython_module', lambda: ConstantCrossSection)
    wet_nm = DIAMETERS_NM * (1 + 0.4 * np.log(DIAMETERS_NM / DIAMETERS_NM[0]))
    ratios = humidity._humidity_optical_ratios(
        DIAMETERS_NM, wet_nm, SPECTRUM, 1.5, 1.4, 550.)
    np.testing.assert_allclose(ratios, 1., rtol=1e-14)


@pytest.mark.parametrize('sulfate', [False, True])
def test_percent_and_fraction_rh_agree(sulfate):
    function = (humidity.calculate_humidification_factor_ammonium_sulfate if sulfate
                else humidity.calculate_humidification_factor)
    args = () if sulfate else (1.5, 0.)
    np.testing.assert_allclose(
        function(DIAMETERS_NM, SPECTRUM, 80., 0.3, 550., *args),
        function(DIAMETERS_NM, SPECTRUM, 0.8, 0.3, 550., *args), rtol=1e-14)


def test_no_growth_gives_unit_ratios():
    result = humidity.calculate_humidification_factor(
        DIAMETERS_NM, SPECTRUM, 0.8, 0., 550., 1.5, 0.)
    np.testing.assert_allclose(result, 1., rtol=1e-14)


def test_empty_distribution_has_no_enhancement():
    with pytest.raises(ValueError, match='undefined'):
        humidity.calculate_humidification_factor(
            DIAMETERS_NM, np.zeros_like(SPECTRUM), 0.8, 0.3, 550., 1.5, 0.)


@pytest.mark.parametrize('rh', [np.nan, 0., 1., 100.])
def test_invalid_humidity_is_rejected(rh):
    with pytest.raises(ValueError, match='rh must'):
        humidity.calculate_humidification_factor(
            DIAMETERS_NM, SPECTRUM, rh, 0.3, 550., 1.5, 0.)
