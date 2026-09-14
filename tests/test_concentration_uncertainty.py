import numpy as np
import pytest

from sizedistmerge.combine import merge_sizedists_bin_totals
from sizedistmerge.uncertainty import (
    UncertaintyUnavailable, blocked_mean_factor,
    bin_total_log_sensitivity, concentration_uncertainty,
)


def test_blocks_estimate_covariance_of_mean_with_cross_bin_covariance():
    x = np.random.default_rng(3).uniform(1, 4, (60, 6))
    for size in (1, 5, 10):
        F, mean, count, occupied = blocked_mean_factor(x, block_samples=size)
        expected = np.cov(x.reshape(60//size, size, 6).mean(1), rowvar=False)/(60//size)
        np.testing.assert_allclose(F.T@F, expected, atol=1e-15)
        np.testing.assert_allclose(mean, x.mean(0))
        assert np.all(count == 60) and np.all(occupied == 60//size)


def test_missing_samples_are_not_zero_and_true_zeros_remain():
    x = np.arange(120, dtype=float).reshape(60, 2)
    x[::3, 0] = np.nan
    x[1, 0] = 0.
    F, mean, counts, _ = blocked_mean_factor(x)
    np.testing.assert_allclose(mean, np.nanmean(x, 0))
    scores = np.where(np.isfinite(x), x-mean, 0)/counts
    np.testing.assert_allclose(F, np.sqrt(12/11)*scores.reshape(12, 5, 2).sum(1))
    x[:50, 1] = np.nan
    with pytest.raises(UncertaintyUnavailable):
        blocked_mean_factor(x)


def test_sensitivity_with_adjacent_zero_and_fixed_output_grid():
    edges = np.geomspace(10, 1000, 21)
    native = np.geomspace(12, 900, 11)
    numbers = np.geomspace(80, 1, 10)
    numbers[-1] = 0
    width = np.diff(np.log10(native))
    series = [dict(name='APS', edges=native, number=numbers,
                   zero_endpoint='last', zero_number_scale=numbers[-2]/width[-2]*width)]
    y, d = merge_sizedists_bin_totals(edges, series, lam=3e-6)
    K, inputs, _ = bin_total_log_sensitivity(series, d, edges,
                                            zero_scale_sources={('APS', 9): ('APS', 8)})
    np.testing.assert_allclose(K@np.ones(len(inputs)), 1, atol=2e-7)
    # Explicit refits are a derivative check only, not the uncertainty algorithm.
    direction = np.cos(np.arange(9))
    curves = []
    for sign in (-1, 1):
        num = numbers.copy()
        num[:9] *= 10**(sign*.001*direction)
        s = [dict(series[0], number=num, zero_number_scale=num[-2]/width[-2]*width)]
        fit, _ = merge_sizedists_bin_totals(edges, s, lam=3e-6)
        curves.append(np.log10(fit))
    keep = np.isfinite(y)
    np.testing.assert_allclose(((curves[1]-curves[0])/.002)[keep], (K@direction)[keep], atol=.025)


def test_shared_fluctuation_is_not_divided_by_number_of_instruments():
    time_factor = np.tile([.8, 1.2, .9, 1.1], 15)
    samples = time_factor[:, None]*np.array([10, 20])[None, :]
    result = concentration_uncertainty(np.array([100., np.nan]),
                                       np.array([[.5, .5], [.5, .5]]), samples)
    one = concentration_uncertainty(np.array([100.]), np.ones((1, 1)), samples[:, :1])
    np.testing.assert_allclose(result['log10_standard_uncertainty'][0], one['log10_standard_uncertainty'][0])
    u = result['log10_standard_uncertainty'][0]
    assert result['lower_1sigma'][0] == pytest.approx(100*10**-u)
    assert result['upper_1sigma'][0] == pytest.approx(100*10**u)
    assert np.isnan(result['lower_1sigma'][1])


def test_icartt_appends_standard_errors_without_changing_existing_columns(tmp_path):
    from netCDF4 import Dataset
    from campaign_merge_production.arcsix_merge_production import write_icartt_from_netcdf
    path = tmp_path / 'test.nc'
    with Dataset(path, 'w') as ds:
        ds.base_time_iso = '2024-05-28T00:00:00'
        ds.createDimension('chunk', 2)
        ds.createDimension('bin', 3)
        ds.createDimension('edge', 4)
        ds.createVariable('fine_edges_nm', 'f8', ('edge',))[:] = [10, 20, 40, 80]
        for name in ('time_start_since_base_s', 'time_end_since_base_s',
                     'optimization_best_cost', 'warning_high_cost',
                     'warning_merged_gt10_diff_from_cpc', 'retrieved_uhsas_n_fit',
                     'retrieved_aps_density'):
            ds.createVariable(name, 'f8', ('chunk',))[:] = [0, 60]
        ds.createVariable('merged_dNdlogDp', 'f8', ('chunk', 'bin'))[:] = [[1, 2, 3], [4, 5, 6]]
    old = write_icartt_from_netcdf(path, tmp_path/'old.ict')
    with Dataset(path, 'a') as ds:
        for name, value in (('merged_log10_standard_uncertainty', .1),
                            ('merged_dNdlogDp_standard_uncertainty', .2)):
            v = ds.createVariable(name, 'f8', ('chunk', 'bin'), fill_value=np.nan)
            v[:] = [[value]*3, [np.nan]*3]
    new = write_icartt_from_netcdf(path, tmp_path/'new.ict')

    def read(p):
        lines = p.read_text().splitlines()
        nh = int(lines[0].split(',')[0])
        return [s.strip() for s in lines[nh-1].split(',')], np.loadtxt(p, skiprows=nh, delimiter=',')

    old_cols, old_data = read(old)
    cols, data = read(new)
    assert cols[:len(old_cols)] == old_cols
    np.testing.assert_array_equal(data[:, :len(old_cols)], old_data)
    assert [n for n in cols if n.startswith('DNLOG_')] == [n for n in old_cols if n.startswith('DNLOG_')]
    assert cols[len(old_cols):] == ['SD_LOG10_001', 'SD_LOG10_002', 'SD_LOG10_003',
                                   'SD_DNLOG_001', 'SD_DNLOG_002', 'SD_DNLOG_003']
    np.testing.assert_allclose(data[0, -6:], [.1]*3+[.2]*3)
    np.testing.assert_array_equal(data[1, -6:], [-9999.]*6)
    assert '20%' not in new.read_text()
