"""Concentration-only uncertainty for the native-bin, log-space combination.

This estimates a temporal sampling contribution to uncertainty of the mean,
not a full instrument error budget. Alignment, bin selection, weights and
regularization are held fixed. No bootstrap or alignment refit is performed.
"""

import numpy as np
from scipy.linalg import eigvalsh, solve

from .combine import bin_overlap_matrix, second_diff_nonuniform


class UncertaintyUnavailable(ValueError):
    """The fit or temporal samples do not support this local uncertainty estimate."""


def blocked_mean_factor(samples, *, block_samples=5, min_samples=20, min_blocks=4):
    """Estimate uncertainty of the mean from changes between time blocks.

    Each row is one sampling time; each column is the number concentration in
    one instrument bin. All instruments must use the same, evenly spaced clock.
    Keep measured zeros and use NaN for missing data. Five rows form a
    five-second block only when the input is sampled once per second.

    We group nearby samples because they may fluctuate together. Different
    blocks are assumed independent; grouping alone does not prove that they are.
    Return the covariance factor, means, sample counts and occupied-block counts.
    factor.T @ factor gives the covariance of the means: both their individual
    variances and how they vary together. With complete data this equals the
    covariance of block means divided by the number of blocks.
    """
    sample_numbers = np.asarray(samples, float)
    if (sample_numbers.ndim != 2 or sample_numbers.shape[1] == 0
            or not isinstance(block_samples, int)
            or block_samples < 1 or sample_numbers.shape[0] % block_samples):
        raise ValueError('Need a time-by-bin array and an integer number of equal blocks')
    block_count = sample_numbers.shape[0] // block_samples
    if block_count < 2 or min_samples < 2 or min_blocks < 2:
        raise ValueError('At least two samples and blocks are required')
    if (np.any(np.isinf(sample_numbers))
            or np.any(sample_numbers[np.isfinite(sample_numbers)] < 0)):
        raise ValueError('Samples must be nonnegative bin numbers or NaN')
    valid = np.isfinite(sample_numbers)
    counts = valid.sum(axis=0)
    occupied = valid.reshape(block_count, block_samples, -1).any(axis=1).sum(axis=0)
    if np.any(counts < min_samples) or np.any(occupied < min_blocks):
        raise UncertaintyUnavailable(
            f'Need >= {min_samples} samples in >= {min_blocks} blocks in every retained positive bin')
    means = np.nansum(sample_numbers, axis=0) / counts
    if np.any(means <= 0):
        raise UncertaintyUnavailable('Log propagation requires positive native-bin means')
    # Divide by each bin's observed sample count, not always by 60. A missing
    # second contributes nothing to the mean; it is not a measured zero.
    score = np.where(valid, sample_numbers - means, 0.) / counts
    # Keep each block's contribution for every bin. This preserves shared rises
    # and falls across instruments instead of assuming their errors cancel out.
    # factor.T @ factor gives the covariance of the means without storing it here.
    factor = np.sqrt(block_count / (block_count - 1)) * score.reshape(
        block_count, block_samples, -1
    ).sum(axis=1)
    return factor, means, counts, occupied


def bin_total_log_sensitivity(series, diagnostics, output_edges_nm, *, zero_scale_sources=None):
    """Estimate how small input concentration changes would move the merged fit.

    Return a sensitivity matrix, its input-bin labels, and a condition number
    that indicates how close the calculation is to being singular. The matrix
    relates changes in log10 input-bin totals to changes in log10 output heights.
    It includes the measurements and the smoothing used in the original fit.

    Zero totals cannot be logged. If a retained zero's fitting scale depends on
    a neighbouring positive bin, supply {zero_bin_key: positive_bin_key} so that
    dependence is included. Otherwise its scale stays fixed. Each key is
    (instrument_name, native_bin_index); other fit settings remain unchanged.
    """
    fit = diagnostics
    by_name = {instrument['name']: instrument for instrument in series}
    if len(by_name) != len(series):
        raise ValueError('Instrument names must be unique')
    keys = [tuple(key) for key in fit['used_bins']]
    # Find how much of each model bin lies inside each measured bin. Multiplying
    # these log-diameter widths by model heights gives the predicted bin totals.
    overlap = np.array([
        bin_overlap_matrix(by_name[name]['edges'][i:i+2], fit['model_edges_nm'])[0]
        for name, i in keys
    ])
    model_height = np.asarray(fit['model_dNdlogDp'], float)
    number = np.asarray(fit['measured_number'], float)
    predicted_number = overlap @ model_height
    if np.any(~np.isfinite(predicted_number)) or np.any(predicted_number <= 0):
        raise UncertaintyUnavailable('Invalid fitted native-bin prediction')
    # How much of a measured bin's predicted total comes from each model bin?
    # A contribution of 0.3 means a small 1% increase in that model height
    # would increase this predicted total by about 0.3%.
    contribution = overlap * model_height[None, :] / predicted_number[:, None]
    positive = number > 0
    inputs = [keys[i] for i in np.flatnonzero(positive)]
    lookup = {key: j for j, key in enumerate(inputs)}
    zero_scale_sources = zero_scale_sources or {}
    if any(zero_key not in keys or zero_key in lookup or positive_key not in lookup
           for zero_key, positive_key in zero_scale_sources.items()):
        raise ValueError('Zero-scale dependencies must link a retained zero to a retained positive bin')
    mids_log = np.log10(np.sqrt(fit['model_edges_nm'][:-1] * fit['model_edges_nm'][1:]))
    curvature, integration_weights = second_diff_nonuniform(mids_log)
    # How strongly does the fit resist a change in its shape? The Hessian
    # describes how the cost curves around the fitted solution. Start with
    # smoothing, then add the constraints supplied by the measurements.
    hessian = curvature.T @ (float(fit['lambda']) * integration_weights[:, None] * curvature)
    # Each column will describe how changing one input pushes the fit away
    # from its current minimum. The solve below balances that push against
    # the resistance described by the Hessian above.
    rhs = np.zeros((len(model_height), len(inputs)))
    ln10 = np.log(10.)
    for i, key in enumerate(keys):
        weight = fit['bin_weight'][i]
        if positive[i]:
            residual = np.log10(predicted_number[i] / number[i])
            gradient = contribution[i]
            # The fitted prediction need not equal the measurement exactly.
            # Keep the term involving that mismatch when calculating cost curvature.
            hessian += weight * (
                np.outer(gradient, gradient)
                + residual * ln10 * (np.diag(gradient) - np.outer(gradient, gradient))
            )
            rhs[:, lookup[key]] += weight * gradient
        else:
            # For a measured zero, use the same linear error as the merge.
            # If its scale comes from a positive neighbour, changes to that
            # neighbour must also affect the uncertainty calculation below.
            residual = predicted_number[i] / (fit['zero_number_scale'][i] * ln10)
            gradient = ln10 * residual * contribution[i]
            hessian += weight * (
                np.outer(gradient, gradient) + residual * ln10 * np.diag(gradient)
            )
            if key in zero_scale_sources:
                rhs[:, lookup[zero_scale_sources[key]]] += 2 * ln10 * weight * residual * gradient
    eigenvalues = eigvalsh(hessian)
    if eigenvalues[0] <= 0:
        raise UncertaintyUnavailable('Non-positive fit curvature; local propagation is not supported')
    # Estimate how each merged bin would change if an input concentration
    # changed slightly, without rerunning the fit.
    # Rows are model bins; columns are measured input bins. A value of 0.3
    # means a small 1% input increase gives about a 0.3% output increase.
    # We use this to propagate concentration uncertainty to the merged result,
    # keeping fitted diameters, weights, and smoothing fixed.
    sensitivity = solve(hessian, rhs, assume_a='pos')
    edges = np.asarray(output_edges_nm, float)
    matches = np.flatnonzero(np.isclose(fit['model_edges_nm'], edges[0], rtol=1e-12, atol=0))
    if matches.size != 1:
        raise ValueError('Output grid must be contained in the fitted model grid')
    left = int(matches[0])
    np.testing.assert_allclose(fit['model_edges_nm'][left:left+len(edges)], edges, rtol=1e-12)
    # The fit may include extra bins outside the requested output range so that
    # it can fit whole measured bins. Remove those extra rows before returning.
    return sensitivity[left:left+len(edges)-1], inputs, float(eigenvalues[-1] / eigenvalues[0])


def concentration_uncertainty(merged, sensitivity, samples, *, block_samples=5,
                              min_samples=20, min_blocks=4):
    """Propagate native-bin mean variability to +/- ONE log standard deviation.

    Bounds are y*10**(-u_log10) and y*10**(+u_log10), NOT y +/- u_log10.
    They are asymmetric on a linear concentration axis and are not claimed
    confidence intervals. Also return the first-order linear standard error.
    To plot two log standard deviations, use y * 10**(+/- 2*u_log10);
    do not multiply the one-deviation bounds themselves by two.
    The small block-by-output factor retains cross-bin covariance without
    saving a full output covariance matrix. Unreported output bins remain NaN.
    """
    factor, means, counts, occupied = blocked_mean_factor(
        samples, block_samples=block_samples, min_samples=min_samples, min_blocks=min_blocks)
    merged_height = np.asarray(merged, float)
    sensitivity = np.asarray(sensitivity, float)
    if sensitivity.shape != (merged_height.size, means.size) or np.any(~np.isfinite(sensitivity)):
        raise ValueError('Sensitivity dimensions/values do not match output and input bins')
    # The sensitivity describes relative changes, so first express the input
    # errors in log10 units by dividing by mean * ln(10). Then pass them through
    # the sensitivity matrix. Keeping the block contributions together preserves
    # shared fluctuations; this is equivalent to J @ input_covariance @ J.T.
    log_factor = (factor / (means * np.log(10.))) @ sensitivity.T
    log_factor[:, ~np.isfinite(merged_height)] = np.nan
    log_standard_uncertainty = np.sqrt(np.sum(log_factor**2, axis=0))
    with np.errstate(over='ignore', under='ignore'):
        lower = merged_height * 10.**(-log_standard_uncertainty)
        upper = merged_height * 10.**log_standard_uncertainty
    reported = np.isfinite(merged_height)
    if np.any(~np.isfinite(upper[reported])) or np.any(lower[reported] <= 0):
        raise UncertaintyUnavailable('Propagated log bounds exceed numerical range')
    return dict(log10_standard_uncertainty=log_standard_uncertainty, lower_1sigma=lower, upper_1sigma=upper,
                standard_uncertainty=merged_height*np.log(10.)*log_standard_uncertainty, log10_covariance_factor=log_factor,
                input_mean=means, input_sample_count=counts, input_block_count=occupied)
