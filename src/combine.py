# Distribution combination: legacy interpolated fits and native-bin-total fits.
# The current production path is merge_sizedists_bin_totals near the end of
# this module. The interpolation functions remain for older callers/replays.
# Their alpha arrays and lambda values have different meanings; do not swap
# the two paths merely because both use Tikhonov (curvature) smoothing.
# ----------------------------------------------------------------------
# Global merge of particle size distributions using Tikhonov smoothing
# on a nonuniform log-diameter grid, with nonnegative bounded LS solve.
#
# Public API:
#   - make_grid_from_series(series_list, n_points="auto", bounds=None, require_positive_y=False)
#   - sigma_from_bands(y_lo, y_hi)
#   - fractional_sigma(y, frac)
#   - merge_sizedists_tikhonov(diam_grid_nm, series_list, lam=1e-6, eps=1e-12, nonneg=True)
#   - compute_data_weights(diam_grid_nm, series_list, eps=1e-12)
#
# Each series in `series_list` is a dict:
#   {"x": array_nm, "y": array, "sigma": array_or_None, "alpha": scalar_or_grid_array}
# alpha is an absolute per-instrument weight, not a pairwise weight. Arrays are
# defined on the solver's diam_grid_nm, not on the instrument's native bins.
#
# Example (pseudocode):
#   series = [
#       {"x": x_FIMS, "y": y_FIMS, "sigma": sigma_FIMS, "alpha": 1.0},
#       {"x": x_UHS,  "y": y_UHS,  "sigma": sigma_UHS,  "alpha": 0.5},
#       {"x": x_APS,  "y": y_APS,  "sigma": sigma_APS,  "alpha": 1.0},
#   ]
#   Dg = make_grid_from_series(series, n_points="auto")
#   merged, wsum, diag = merge_sizedists_tikhonov(Dg, series, lam=5e-7)
# ----------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from scipy.optimize import lsq_linear, least_squares
from scipy.special import logsumexp


# ----------------------------- Interpolation ----------------------------- #

def _source_mask(x, y, positive_only, preserve_zero_endpoint):
    """Keep positive samples and the adjacent zero just outside their size range."""
    if preserve_zero_endpoint not in (None, "first", "last"):
        raise ValueError("preserve_zero_endpoint must be None, 'first', or 'last'")
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if not positive_only:
        return valid
    keep = valid & (y > 0)
    if preserve_zero_endpoint is not None:
        # Work in diameter order without dropping missing/negative y values:
        # only the immediate neighbour may be retained, never bridge over one.
        indices = np.flatnonzero(np.isfinite(x) & (x > 0))
        indices = indices[np.argsort(x[indices], kind="stable")]
        positive = np.flatnonzero(keep[indices])
        if positive.size:
            neighbour = positive[0] - 1 if preserve_zero_endpoint == "first" else positive[-1] + 1
            if 0 <= neighbour < indices.size:
                index = indices[neighbour]
                if valid[index] and y[index] == 0:
                    keep[index] = True
    return keep


def log_interp(
    x_src_nm: np.ndarray,
    y_src: np.ndarray,
    x_dst_nm: np.ndarray,
    *,
    positive_only: bool = False,
    preserve_zero_endpoint: Optional[str] = None,
) -> np.ndarray:
    """
    Interpolate y(x) from x_src_nm onto x_dst_nm using log10(x) as the axis.
    Values outside the source span in log-space are set to NaN.
    With positive_only=True, omit nonpositive source values BEFORE interpolation.
    Interpolate across omitted interior samples; do not extrapolate past the
    remaining samples. preserve_zero_endpoint='first' or 'last' retains only an
    exact zero immediately before the first positive sample or after the last
    positive sample, respectively, as an interpolation anchor. Earlier/later
    consecutive zeros are omitted.
    Negative and missing values are never retained by this exception.

    Parameters
    ----------
    x_src_nm : (n_src,) array_like
        Source diameters [nm], must be > 0 where valid.
    y_src : (n_src,) array_like
        Source values aligned with x_src_nm.
    x_dst_nm : (n_dst,) array_like
        Destination diameters [nm], must be > 0.

    Returns
    -------
    y_interp : (n_dst,) ndarray
        Interpolated values, NaN outside the native log-span of x_src_nm.
    """
    x_src_nm = np.asarray(x_src_nm, float)
    y_src = np.asarray(y_src, float)
    x_dst_nm = np.asarray(x_dst_nm, float)
    if x_src_nm.ndim != 1 or y_src.ndim != 1 or x_src_nm.size != y_src.size:
        raise ValueError("x_src_nm and y_src must be 1D arrays with the same length")
    if x_dst_nm.ndim != 1:
        raise ValueError("x_dst_nm must be a 1D array")
    if np.any(~np.isfinite(x_dst_nm)) or np.any(x_dst_nm <= 0):
        raise ValueError("x_dst_nm must be finite and > 0")

    valid_src = _source_mask(x_src_nm, y_src, positive_only, preserve_zero_endpoint)
    if not np.any(valid_src):
        return np.full_like(x_dst_nm, np.nan, dtype=float)

    log_x_src = np.log10(x_src_nm[valid_src])
    y_src_ok = y_src[valid_src]
    order = np.argsort(log_x_src)
    log_x_src = log_x_src[order]
    y_src_ok = y_src_ok[order]
    log_x_src, inv, counts = np.unique(log_x_src, return_inverse=True, return_counts=True)
    if log_x_src.size != y_src_ok.size:
        y_src_ok = np.bincount(inv, weights=y_src_ok) / counts
    if log_x_src.size < 2:
        return np.full_like(x_dst_nm, np.nan, dtype=float)
    log_x_dst = np.log10(x_dst_nm)

    y_interp = np.interp(log_x_dst, log_x_src, y_src_ok, left=np.nan, right=np.nan)

    # Mask extrapolation beyond source span
    log_min, log_max = np.min(log_x_src), np.max(log_x_src)
    outside = (log_x_dst < log_min) | (log_x_dst > log_max)
    y_interp[outside] = np.nan
    return y_interp


# ------------------ Second derivative on nonuniform log grid ------------------ #

def second_diff_nonuniform(log_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a discrete second-derivative operator L on a nonuniform grid in log-space,
    and quadrature weights so sum w_c * (L y)^2 ≈ ∫ (y'')^2 d(log D).

    Parameters
    ----------
    log_grid : (n,) array_like
        Strictly increasing grid in log space, log(D).

    Returns
    -------
    L : ((n-2), n) ndarray
        Operator approximating y'' at interior nodes:
        (L y)[i-1] ≈ y''(log_grid[i]) for i=1..n-2 (1-based interior).
    quad_w : (n-2,) ndarray
        Quadrature weights ~ local spacing for grid-invariant smoothing.
    """
    log_grid = np.asarray(log_grid, float)
    n_grid = log_grid.size
    if n_grid < 3:
        raise ValueError("second_diff_nonuniform: need at least 3 grid points.")
    if np.any(~np.isfinite(log_grid)) or not np.all(np.diff(log_grid) > 0):
        raise ValueError("second_diff_nonuniform: log_grid must be finite and strictly increasing.")

    curvature_operator = np.zeros((n_grid - 2, n_grid), float)
    integration_weights = np.zeros(n_grid - 2, float)

    # Interior nodes i = 1..n-2 (0-based indexing)
    for i in range(1, n_grid - 1):
        h_left = log_grid[i] - log_grid[i - 1]
        h_right = log_grid[i + 1] - log_grid[i]

        # Nonuniform 3-point second derivative coefficients
        coef_left = 2.0 / (h_left * (h_left + h_right))
        coef_mid = -2.0 / (h_left * h_right)
        coef_right = 2.0 / (h_right * (h_left + h_right))

        row = i - 1
        curvature_operator[row, i - 1] = coef_left
        curvature_operator[row, i] = coef_mid
        curvature_operator[row, i + 1] = coef_right

        integration_weights[row] = 0.5 * (h_left + h_right)  # local cell size for ∫(y'')^2 dt
    return curvature_operator, integration_weights


# ----------------------------- Grid utilities ----------------------------- #

def make_grid_from_series(
    series_list: List[Dict[str, Any]],
    n_points: Union[int, str] = "auto",
    bounds: Optional[Tuple[float, float]] = None,
    require_positive_y: bool = False,
    min_points: int = 3,
) -> np.ndarray:
    """
    Build a common log-spaced diameter grid covering the union of valid spans.

    Parameters
    ----------
    series_list : list of dict
        Each dict must have "x" and "y". Arrays must be same length per series.
    n_points : int or "auto", default "auto"
        If "auto", use total count of valid x across all series. Else use given int.
    bounds : (Dmin, Dmax) or None
        If provided, clip the union span to these bounds (both > 0).
    require_positive_y : bool, default False
        If True, mask out y <= 0 when forming the union span.
    min_points : int, default 3
        Minimum number of valid points required to form a grid.

    Returns
    -------
    diam_grid_nm : ndarray
        Geometric (log) spaced grid from union(min x) to union(max x).

    Raises
    ------
    ValueError if fewer than `min_points` valid points after masking/clipping.
    """
    all_x = []
    if not series_list:
        raise ValueError("make_grid_from_series: series_list must not be empty.")
    for s in series_list:
        x = np.asarray(s["x"], float)
        y = np.asarray(s["y"], float)
        if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
            raise ValueError("make_grid_from_series: each series x and y must be 1D arrays with the same length.")
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        if require_positive_y:
            mask &= (y > 0)
        if np.any(mask):
            all_x.append(x[mask])

    if len(all_x) == 0:
        raise ValueError("make_grid_from_series: no valid x across series.")

    x_concat = np.concatenate(all_x)
    D_min = np.min(x_concat)
    D_max = np.max(x_concat)

    if bounds is not None:
        Dmin_clip, Dmax_clip = bounds
        if (Dmin_clip is not None) and (Dmin_clip > 0):
            D_min = max(D_min, Dmin_clip)
        if (Dmax_clip is not None) and (Dmax_clip > 0):
            D_max = min(D_max, Dmax_clip)
        if not (D_max > D_min):
            raise ValueError("make_grid_from_series: invalid bounds or no overlap after clipping.")

    if n_points == "auto":
        n_valid_total = int(np.sum([arr.size for arr in all_x]))
        n_pts = max(min_points, n_valid_total)
    else:
        n_pts = int(n_points)
        if n_pts < min_points:
            n_pts = min_points

    diam_grid_nm = np.geomspace(D_min, D_max, n_pts)
    return diam_grid_nm


# -------------------------- Uncertainty conveniences -------------------------- #

def sigma_from_bands(y_lo: np.ndarray, y_hi: np.ndarray) -> np.ndarray:
    """
    Return half the distance between lower and upper bounds.

    This is one standard deviation only if the supplied bounds already mean
    +/- one standard deviation in linear units. It does not infer a confidence
    level or convert a two-standard-deviation band into a one-deviation band.
    """
    y_lo = np.asarray(y_lo, float)
    y_hi = np.asarray(y_hi, float)
    return 0.5 * (y_hi - y_lo)


def fractional_sigma(y: np.ndarray, frac: float) -> np.ndarray:
    """
    Apply a caller-chosen relative uncertainty: sigma = |y| * frac.

    For example, frac=0.2 gives 20% of each height. This assigns uncertainty;
    it does not estimate it from measurements.
    """
    y = np.asarray(y, float)
    return np.abs(y) * float(frac)


# ------------------------------- Data weighting ------------------------------- #

def smooth_weight_profile(
    diam_grid_nm: np.ndarray,
    *,
    start_nm: float,
    end_nm: float,
    start_weight: float,
    end_weight: float,
) -> np.ndarray:
    """Return one instrument's smooth, absolute weight on a diameter grid.

    The weight is constant below start_nm and above end_nm. Between those
    limits it follows 3*t**2 - 2*t**3, where t is fractional log-diameter.
    Both the weight and its slope are continuous at the endpoints. The
    transition is monotonic and has no overshoot.

    Use explicit, physically justified diameter limits in the same coordinate
    as diam_grid_nm. Do not move the limits whenever a minute has a zero bin.
    The function knows nothing about instrument names or pairs. Supply its
    result as a series' alpha to either merge solver or compute_data_weights.
    Weights are not normalized here: changing their total also changes the
    strength of the data relative to the smoothness penalty.
    """
    grid = np.asarray(diam_grid_nm, float)
    if grid.ndim != 1 or not grid.size or np.any(~np.isfinite(grid)) or np.any(grid <= 0):
        raise ValueError("diam_grid_nm must be a non-empty, finite, positive 1D array")
    bounds = np.asarray([start_nm, end_nm], float)
    if np.any(~np.isfinite(bounds)) or not (0 < start_nm < end_nm):
        raise ValueError("require finite 0 < start_nm < end_nm")
    weights = np.asarray([start_weight, end_weight], float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("profile weights must be finite and nonnegative")
    t = np.clip((np.log(grid) - np.log(start_nm)) / (np.log(end_nm) - np.log(start_nm)), 0., 1.)
    blend = t * t * (3. - 2. * t)
    return start_weight + (end_weight - start_weight) * blend


def _weight_on_grid(alpha, grid: np.ndarray) -> np.ndarray:
    """Accept a constant or an explicit weight at every solver-grid node."""
    weight = np.asarray(alpha, float)
    if weight.ndim == 0:
        weight = np.full(grid.shape, float(weight))
    elif weight.shape != grid.shape:
        raise ValueError("alpha must be a scalar or a 1D array matching diam_grid_nm")
    if np.any(~np.isfinite(weight)) or np.any(weight < 0):
        raise ValueError("alpha weights must be finite and nonnegative")
    return weight


def compute_data_weights(
    diam_grid_nm: np.ndarray,
    series_list: List[Dict[str, Any]],
    eps: float = 1e-12,
) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray]]:
    """
    Compute per-instrument data weights on `diam_grid_nm`.

    For each series:
      - Interpolate y and sigma onto the grid in log-space.
      - Form inverse variance weights 1/(sigma^2+eps).
      - Multiply by the instrument alpha.
      - Zero out where either y or sigma is NaN.

    Parameters
    ----------
    diam_grid_nm : (n,) array_like
        Target diameter grid.
    series_list : list of dict
        Each dict has "x", "y", and optionally "sigma", "alpha".
    eps : float
        Variance floor to avoid infinite weights.

    Returns
    -------
    weights_per_series : list of (n,) ndarrays
        Effective weights per instrument on the grid.
    weight_sum : (n,) ndarray
        Sum of weights across instruments.
    weights_normalized : list of (n,) ndarrays
        Per-instrument weights normalized by the sum at each node.
        If total weight is zero at a node, normalized weights are zero there.
    """
    diam_grid_nm = np.asarray(diam_grid_nm, float)
    n = diam_grid_nm.size
    if diam_grid_nm.ndim != 1 or n == 0:
        raise ValueError("diam_grid_nm must be a non-empty 1D array")
    if np.any(~np.isfinite(diam_grid_nm)) or np.any(diam_grid_nm <= 0):
        raise ValueError("diam_grid_nm must be finite and > 0")
    if not np.all(np.diff(diam_grid_nm) > 0):
        raise ValueError("diam_grid_nm must be strictly increasing")

    weights_per_series = []
    for s in series_list:
        x_nm = s["x"]
        y_vals = s["y"]
        sigma_vals = s.get("sigma", None)
        alpha = _weight_on_grid(s.get("alpha", 1.0), diam_grid_nm)

        y_on_grid = log_interp(x_nm, y_vals, diam_grid_nm)

        if sigma_vals is None:
            inv_var_on_grid = np.ones_like(diam_grid_nm)
        else:
            sigma_on_grid = log_interp(x_nm, sigma_vals, diam_grid_nm)
            inv_var_on_grid = 1.0 / (np.square(sigma_on_grid) + eps)

        valid = np.isfinite(y_on_grid) & np.isfinite(inv_var_on_grid)
        eff_weight = np.zeros(n, float)
        eff_weight[valid] = alpha[valid] * inv_var_on_grid[valid]
        weights_per_series.append(eff_weight)

    weight_sum = np.sum(weights_per_series, axis=0)
    tiny = 1e-30
    weights_normalized = [w / np.maximum(weight_sum, tiny) for w in weights_per_series]
    return weights_per_series, weight_sum, weights_normalized


# ------------------------------ Core merge solver ------------------------------ #

def merge_sizedists_tikhonov(
    diam_grid_nm: np.ndarray,
    series_list: List[Dict[str, Any]],
    lam: float = 1e-6,
    eps: float = 1e-12,
    nonneg: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Legacy merge: interpolate instrument curves onto a common diameter grid,
    then fit their heights with a penalty for bending between neighboring points.
    This is Tikhonov smoothing. It does not fit measured bin totals directly;
    use merge_sizedists_bin_totals for that calculation.

    Minimizes (in stacked least-squares form):
        || D^{1/2} (M - ybar) ||_2^2 + || sqrt(lam) * Wc^{1/2} L M ||_2^2
    subject to M >= 0 if `nonneg` is True.
    """

    diam_grid_nm = np.asarray(diam_grid_nm, float)
    n_grid = diam_grid_nm.size
    if diam_grid_nm.ndim != 1 or n_grid < 3:
        raise ValueError("diam_grid_nm must be a 1D array with at least 3 points")
    if np.any(~np.isfinite(diam_grid_nm)) or np.any(diam_grid_nm <= 0):
        raise ValueError("diam_grid_nm must be finite and > 0")
    if not np.all(np.diff(diam_grid_nm) > 0):
        raise ValueError("diam_grid_nm must be strictly increasing")
    if not series_list:
        raise ValueError("series_list must not be empty")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and >= 0")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and > 0")
    log_grid = np.log10(diam_grid_nm)

    # Accumulate inverse-variance weighted sums on the grid
    weighted_sum_y = np.zeros(n_grid, float)   # Σ_i (α_i * w_ij * y_ij)
    weight_sum     = np.zeros(n_grid, float)   # Σ_i (α_i * w_ij)

    for s in series_list:
        x_nm       = np.asarray(s["x"], float)
        y_vals     = np.asarray(s["y"], float)
        sigma_vals = s.get("sigma", None)
        alpha      = _weight_on_grid(s.get("alpha", 1.0), diam_grid_nm)

        y_on_grid = log_interp(x_nm, y_vals, diam_grid_nm)

        if sigma_vals is None:
            inv_var_on_grid = np.ones_like(diam_grid_nm)
        else:
            sigma_on_grid   = log_interp(x_nm, np.asarray(sigma_vals, float), diam_grid_nm)
            inv_var_on_grid = 1.0 / (np.square(sigma_on_grid) + eps)

        valid = np.isfinite(y_on_grid) & np.isfinite(inv_var_on_grid)
        eff_weight = np.zeros(n_grid, float)
        eff_weight[valid] = alpha[valid] * inv_var_on_grid[valid]

        weighted_sum_y += eff_weight * np.nan_to_num(y_on_grid, nan=0.0)
        weight_sum     += eff_weight

    # Where do we actually have *any* data?
    data_mask = weight_sum > 0
    if data_mask.sum() < 3:
        raise ValueError("merge_sizedists_tikhonov: insufficient supported grid points (<3).")

    tiny = 1e-30

    # Data term on the FULL grid
    ybar = np.zeros_like(weighted_sum_y)
    sqrt_weight_data = np.zeros_like(weight_sum)

    # A zero here is only a placeholder. Its weight is also zero, so a gap
    # does not tell the fit that concentration should be zero. The smoothness
    # penalty connects the fit across those unmeasured points.
    ybar[data_mask] = weighted_sum_y[data_mask] / np.maximum(weight_sum[data_mask], tiny)
    sqrt_weight_data[data_mask] = np.sqrt(np.maximum(weight_sum[data_mask], tiny))

    A_data_matrix = np.diag(sqrt_weight_data)       # (n, n)
    rhs_data      = sqrt_weight_data * ybar         # (n,)

    # Smoothness term on FULL nonuniform log grid
    L_smooth, quad_w = second_diff_nonuniform(log_grid)
    A_smooth_matrix  = np.sqrt(lam) * (np.sqrt(quad_w)[:, None] * L_smooth)
    rhs_smooth       = np.zeros(L_smooth.shape[0], float)

    # Solve both requirements together: stay close to measurements, while
    # avoiding excessive bending. lam controls the strength of the second part.
    A_stacked   = np.vstack([A_data_matrix, A_smooth_matrix])
    rhs_stacked = np.concatenate([rhs_data, rhs_smooth])

    # Solve bounded LS on the *full* grid
    lower_bound = 0.0 if nonneg else -np.inf
    upper_bound = np.inf
    result = lsq_linear(A_stacked, rhs_stacked,
                        bounds=(lower_bound, upper_bound),
                        method="trf")
    merged_full = result.x  # length n_grid

    # Clip to data span: NaN outside [first, last] node with any data
    merged_vals = merged_full.copy()
    first = int(np.argmax(data_mask))
    last  = int(len(data_mask) - 1 - np.argmax(data_mask[::-1]))
    merged_vals[:first]  = np.nan
    merged_vals[last+1:] = np.nan

    diagnostics = {
        "support_mask": data_mask,          # where there is any data
        "ybar_supported": ybar[data_mask],  # just for reference
        "sqrt_weight_data": sqrt_weight_data[data_mask],
        "L_smooth": L_smooth,
        "quad_w": quad_w,
        "solver": result,
    }

    return merged_vals, weight_sum, diagnostics


def merge_sizedists_tikhonov_consensus(
    diam_grid_nm: np.ndarray,
    series_list: List[Dict[str, Any]],
    *,
    lam: float = 1e-6,
    eps: float = 1e-12,
    nonneg: bool = True,
    min_overlap: int = 3,
    c: float = 2.5,
    use_consensus: bool = True,
    eps_scale: float = 1e-12,
    data_space: str = "linear",   # "linear" or "log10"
    ignore_nonpositive_source: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Like merge_sizedists_tikhonov, but with a per-grid-node "consensus/voting" reweighting.

    Choice of fitted height:
    - data_space="linear": do everything in linear y
    - data_space="log10": do everything in log10(y), then convert merged back to linear y

    In log10 mode:
    - y <= 0 is treated as missing (ignored): it becomes NaN in solver-space and gets zero weight.
    - if sigma is provided in linear y units, it is converted to sigma_log10 via:
          sigma_log10 = sigma / (y * ln(10))
      Only where y > 0 and sigma > 0; otherwise ignored.
    - nonneg bound is ignored in log10 mode (since 10**Z is always positive).

    alpha may be a nonnegative scalar or an array on diam_grid_nm. Its value
    multiplies the separate agreement and optional uncertainty weights; it is
    not a guaranteed final share. A zero-weight series does not vote.

    Set use_consensus=False to use only the base weights (including any
    supplied uncertainty weights), with no agreement adjustment. The grid,
    interpolation and Tikhonov smoothing are identical; c is unused.

    Set ignore_nonpositive_source=True to omit nonpositive native samples
    before interpolation, avoiding artificial dips around zero source bins.
    Interior gaps are interpolated across; endpoints are not extrapolated.
    Each series may set preserve_zero_endpoint='first' or 'last' to keep an
    zero immediately outside the positive-data range during interpolation. In log10 mode,
    a resulting exact zero still has no fitting weight; no positive floor is added.
    The default False preserves the historical calculation for replay.
    """
    if data_space not in ("linear", "log10"):
        raise ValueError(f"data_space must be 'linear' or 'log10', got: {data_space!r}")

    diam_grid_nm = np.asarray(diam_grid_nm, float)
    n_grid = diam_grid_nm.size
    if diam_grid_nm.ndim != 1 or n_grid < 3:
        raise ValueError("diam_grid_nm must be a 1D array with at least 3 points")
    if np.any(~np.isfinite(diam_grid_nm)) or np.any(diam_grid_nm <= 0):
        raise ValueError("diam_grid_nm must be finite and > 0")
    if not np.all(np.diff(diam_grid_nm) > 0):
        raise ValueError("diam_grid_nm must be strictly increasing")
    if not series_list:
        raise ValueError("series_list must not be empty")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and >= 0")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and > 0")
    if min_overlap < 1:
        raise ValueError("min_overlap must be >= 1")
    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be finite and > 0")
    if not np.isfinite(eps_scale) or eps_scale <= 0:
        raise ValueError("eps_scale must be finite and > 0")
    log_grid = np.log10(diam_grid_nm)

    n_series = len(series_list)

    # ---- interpolate all series onto grid ----
    Y_lin = np.full((n_series, n_grid), np.nan, float)  # always store linear interpolated y
    Y_use = np.full((n_series, n_grid), np.nan, float)  # solver-space y (linear or log10)
    W_base = np.zeros((n_series, n_grid), float)        # inverse-variance weights in solver-space

    ln10 = np.log(10.0)

    for i, s in enumerate(series_list):
        x_nm       = np.asarray(s["x"], float)
        y_vals     = np.asarray(s["y"], float)
        sigma_vals = s.get("sigma", None)
        alpha      = _weight_on_grid(s.get("alpha", 1.0), diam_grid_nm)

        y_on_grid = log_interp(x_nm, y_vals, diam_grid_nm,
                               positive_only=ignore_nonpositive_source,
                               preserve_zero_endpoint=s.get('preserve_zero_endpoint'))
        Y_lin[i, :] = y_on_grid
        if sigma_vals is not None and ignore_nonpositive_source:
            sigma_vals = np.asarray(sigma_vals, float)
            if sigma_vals.shape != y_vals.shape:
                raise ValueError("sigma must have the same shape as source y")
            keep = _source_mask(x_nm, y_vals, True, s.get('preserve_zero_endpoint'))
            sigma_vals = np.where(keep, sigma_vals, np.nan)

        if data_space == "linear":
            # ---- fit in linear y ----
            Y_use[i, :] = y_on_grid

            if sigma_vals is None:
                valid = np.isfinite(y_on_grid)
                W_base[i, valid] = alpha[valid]
            else:
                sigma_on_grid = log_interp(x_nm, np.asarray(sigma_vals, float), diam_grid_nm)
                inv_var = 1.0 / (np.square(sigma_on_grid) + eps)
                valid = np.isfinite(y_on_grid) & np.isfinite(inv_var)
                W_base[i, valid] = alpha[valid] * inv_var[valid]

        else:
            # ---- fit in log10(y); ignore y<=0 by treating as missing ----
            pos = np.isfinite(y_on_grid) & (y_on_grid > 0)

            ylog = np.full(n_grid, np.nan, float)
            ylog[pos] = np.log10(y_on_grid[pos])
            Y_use[i, :] = ylog

            if sigma_vals is None:
                valid = np.isfinite(ylog)
                W_base[i, valid] = alpha[valid]
            else:
                sigma_on_grid = log_interp(x_nm, np.asarray(sigma_vals, float), diam_grid_nm)

                # only use where y>0 and sigma>0 and finite
                ok = pos & np.isfinite(sigma_on_grid) & (sigma_on_grid > 0)

                sigma_log = np.full(n_grid, np.nan, float)
                sigma_log[ok] = sigma_on_grid[ok] / (y_on_grid[ok] * ln10)

                inv_var = 1.0 / (np.square(sigma_log) + eps)
                valid = np.isfinite(ylog) & np.isfinite(inv_var)
                W_base[i, valid] = alpha[valid] * inv_var[valid]

    # ---- consensus weights per node ----
    W_cons = np.ones((n_series, n_grid), float)

    for j in range(n_grid):
        yj = Y_use[:, j]
        valid = np.isfinite(yj) & (W_base[:, j] > 0)
        m = int(np.sum(valid))
        if not use_consensus or m < min_overlap:
            # Disabled or insufficient overlap: base weights only.
            W_cons[:, j] = np.where(valid, 1.0, 0.0)
            continue

        vals = yj[valid]
        med = np.median(vals)

        mad = np.median(np.abs(vals - med))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale < eps_scale:
            scale = eps_scale

        z = (yj - med) / scale
        w = np.exp(-0.5 * (z / c) ** 2)
        w[~valid] = 0.0
        W_cons[:, j] = w

    # ---- effective weights ----
    W_eff = W_base * W_cons

    # ---- build ybar + weight_sum in solver space ----
    weighted_sum_y = np.nansum(W_eff * np.nan_to_num(Y_use, nan=0.0), axis=0)
    weight_sum     = np.sum(W_eff, axis=0)

    data_mask = weight_sum > 0
    if data_mask.sum() < 3:
        raise ValueError("merge_sizedists_tikhonov_consensus: insufficient supported grid points (<3).")

    tiny = 1e-30
    ybar = np.zeros(n_grid, float)
    sqrt_weight_data = np.zeros(n_grid, float)

    ybar[data_mask] = weighted_sum_y[data_mask] / np.maximum(weight_sum[data_mask], tiny)
    sqrt_weight_data[data_mask] = np.sqrt(np.maximum(weight_sum[data_mask], tiny))

    A_data_matrix = np.diag(sqrt_weight_data)
    rhs_data      = sqrt_weight_data * ybar

    # ---- smoothness term on solved variable (linear y or log10(y)) ----
    L_smooth, quad_w = second_diff_nonuniform(log_grid)
    A_smooth_matrix  = np.sqrt(lam) * (np.sqrt(quad_w)[:, None] * L_smooth)
    rhs_smooth       = np.zeros(L_smooth.shape[0], float)

    A_stacked   = np.vstack([A_data_matrix, A_smooth_matrix])
    rhs_stacked = np.concatenate([rhs_data, rhs_smooth])

    # bounds:
    if data_space == "linear":
        lower_bound = 0.0 if nonneg else -np.inf
        upper_bound = np.inf
    else:
        lower_bound = -np.inf
        upper_bound = np.inf

    result = lsq_linear(
        A_stacked, rhs_stacked,
        bounds=(lower_bound, upper_bound),
        method="trf"
    )
    merged_full = result.x  # solver space

    # convert back to linear y if needed
    if data_space == "log10":
        merged_full_lin = np.power(10.0, merged_full)
    else:
        merged_full_lin = merged_full

    # clip to supported span
    merged_vals = merged_full_lin.copy()
    first = int(np.argmax(data_mask))
    last  = int(len(data_mask) - 1 - np.argmax(data_mask[::-1]))
    merged_vals[:first]  = np.nan
    merged_vals[last+1:] = np.nan

    diagnostics = {
        "support_mask": data_mask,
        "data_space": data_space,
        "ignore_nonpositive_source": ignore_nonpositive_source,
        "W_base": W_base,
        "W_consensus": W_cons,
        "use_consensus": bool(use_consensus),
        "W_effective": W_eff,
        "y_on_grid_linear": Y_lin,
        "y_on_grid_solver": Y_use,
        "ybar_supported_solver": ybar[data_mask],
        "sqrt_weight_data": sqrt_weight_data[data_mask],
        "L_smooth": L_smooth,
        "quad_w": quad_w,
        "solver": result,
    }

    return merged_vals, weight_sum, diagnostics


def bin_overlap_matrix(measured_edges_nm, model_edges_nm):
    """Log-diameter widths shared by each measured bin and each model bin.

    Multiplying this matrix by model dN/dlog10D gives measured-bin number
    predictions. This integrates a piecewise-constant model; it does not
    interpolate the measurements or assign additional independent observations.
    """
    arrays = [np.asarray(bin_edges, float) for bin_edges in (measured_edges_nm, model_edges_nm)]
    for bin_edges in arrays:
        if (bin_edges.ndim != 1 or bin_edges.size < 2 or np.any(~np.isfinite(bin_edges))
                or np.any(bin_edges <= 0) or np.any(np.diff(bin_edges) <= 0)):
            raise ValueError('Bin edges must be finite, positive, and strictly increasing')
    measured_log_edges, model_log_edges = (np.log10(bin_edges) for bin_edges in arrays)
    # For each measured/model bin pair, find the width they share in log diameter.
    # Non-overlapping bins contribute zero. Width times model height gives the
    # predicted number concentration in that shared part of the measured bin.
    return np.maximum(
        0.,
        np.minimum(measured_log_edges[1:, None], model_log_edges[None, 1:])
        - np.maximum(measured_log_edges[:-1, None], model_log_edges[None, :-1]),
    )


def _bin_total_system(log_height, log_overlap, log_number, sqrt_weight, smooth, zero_number_scale=None):
    """Log residuals for positives; optional scaled linear residuals for zeros.

    A positive entry in zero_number_scale identifies a measured zero. Its
    residual is predicted_number / (scale * ln(10)), NOT log(predicted/epsilon).
    The ln(10) gives the same local sensitivity as a positive log residual at
    predicted_number=scale. Scale sets a fitting weight, not a detection limit
    or a claimed measurement uncertainty. Zero entries retain the original
    positive-bin calculation exactly.
    """
    # Predict measured-bin totals by adding the contributions from model bins.
    # Do the sum in log space because the optimizer may try very large heights.
    # logsumexp uses natural logs, so convert our log10 heights with ln(10).
    # Zero overlap is stored as -inf and contributes nothing to the sum.
    terms = log_overlap + np.log(10.)*log_height[None, :]
    log_pred = logsumexp(terms, axis=1)
    data_error = log_pred/np.log(10.)-log_number
    # Tell the optimizer how changing each model height changes the fitting error.
    # These derivatives let it choose a direction without trying every change.
    data_derivative = np.exp(terms-log_pred[:, None])
    if zero_number_scale is not None:
        is_zero = zero_number_scale > 0
        log_scale = np.log(zero_number_scale[is_zero])
        data_error[is_zero] = np.exp(log_pred[is_zero]-log_scale)/np.log(10.)
        data_derivative[is_zero] = np.exp(terms[is_zero]-log_scale[:, None])
    residual = np.r_[sqrt_weight*data_error, smooth@log_height]
    jacobian = np.vstack([sqrt_weight[:, None]*data_derivative, smooth])
    return residual, jacobian, log_pred/np.log(10.)


def native_bin_consensus_multipliers(series_list, c=2.0):
    """Agreement multipliers averaged over each native bin, without interpolation.

    On intervals formed by the union of native edges, compare positive bin
    average heights from at least three instruments in log10 space. Apply
    exp(-0.5*((log_height-median)/(1.4826*MAD*c))**2), then average over each
    native bin by log-diameter width. MAD is median absolute deviation from
    the median. Larger c weakens suppression. Intervals with fewer than three
    positive instruments keep multiplier one. A 1e-12 floor prevents removing
    observations or changing output support. No consensus smoothing is used.
    """
    if not np.isfinite(c) or c <= 0 or not series_list:
        raise ValueError('Need positive finite c and at least one native series')
    names = [series.get('name', str(i)) for i, series in enumerate(series_list)]
    if len(names) != len(set(names)):
        raise ValueError('Consensus requires unique instrument names')
    for series in series_list:
        native_edges = np.asarray(series['edges'], float)
        bin_overlap_matrix(native_edges, native_edges)
        if np.asarray(series['number']).shape != (len(native_edges)-1,):
            raise ValueError('One number is required per native bin')
    # Split only for comparing agreement. These intervals do not become new
    # observations or adjustable model bins in the final fit.
    edges = np.unique(np.concatenate([series['edges'] for series in series_list]))
    interval_midpoints = np.sqrt(edges[:-1]*edges[1:])
    interval_widths = np.diff(np.log10(edges))
    values = np.full((len(series_list), len(interval_midpoints)), np.nan)
    indices = []
    for instrument_index, series in enumerate(series_list):
        native_edges = np.asarray(series['edges'], float)
        native_indices = np.searchsorted(native_edges, interval_midpoints, side='right')-1
        indices.append(native_indices)
        valid = (native_indices >= 0) & (native_indices < len(native_edges)-1)
        height = np.asarray(series['number'], float)/np.diff(np.log10(native_edges))
        values[instrument_index, valid] = height[native_indices[valid]]
    weights = np.ones_like(values)
    for j in range(len(interval_midpoints)):
        valid = np.isfinite(values[:, j]) & (values[:, j] > 0)
        if valid.sum() < 3:
            continue
        logs = np.log10(values[valid, j])
        median = np.median(logs)
        scale = max(1.4826*np.median(abs(logs-median)), 1e-12)
        weights[valid, j] = np.maximum(np.exp(-.5*((logs-median)/(scale*c))**2), 1e-12)
    # Average the agreement factors back to ONE multiplier per native bin.
    # Weight by log width so a tiny overlap does not represent the entire bin.
    result = []
    for instrument_index, series in enumerate(series_list):
        native_indices = indices[instrument_index]
        native_bin_count = len(series['number'])
        valid = (native_indices >= 0) & (native_indices < native_bin_count)
        total = np.bincount(
            native_indices[valid],
            weights=interval_widths[valid] * weights[instrument_index, valid],
            minlength=native_bin_count,
        )
        result.append(total/np.diff(np.log10(series['edges'])))
    return result


def merge_sizedists_bin_totals(output_edges_nm, series_list, *, lam, max_nfev=300):
    """Fit native-bin number totals on fixed output bins, without input interpolation.

    Each series supplies ``edges`` (converted diameter edges), ``number``
    (measured number per bin, cm-3), optional ``name`` and ``alpha``. Alpha is
    a nonnegative scalar or one value per NATIVE bin, not per output-grid node.
    Diameter conversion remains a separate step that preserves these totals.

    Fit log10(model dN/dlog10D), comparing log10(predicted/measured bin number).
    Each squared data residual has weight alpha*Delta(log10 D). The bin-width
    factor limits dependence on native channel count, but is not an uncertainty
    model. Penalize curvature of log10(dN/dlog10D) versus log10(D), integrated
    over diameter using second_diff_nonuniform. Because the previous method
    sums unscaled grid-point residuals, its numerical lambda is NOT equivalent.

    By default zero/negative/nonfinite number bins are omitted. A series may
    opt into zero_endpoint='first' or 'last' to retain ONLY the zero immediately
    before its first positive bin or after its last positive bin. Never bridge
    missing/negative bins or retain further consecutive zeros. This option
    requires zero_number_scale (positive scalar or one value per native bin).
    The retained zero's error is predicted_number/(scale*ln(10)), a scaled
    linear error; the measurement remains exactly zero. A transparent local
    scale is the neighboring positive mean height times the zero bin's log
    width. It is a weighting choice, NOT a counting-uncertainty model. No floor,
    interpolation anchors, forced zero output, or Poisson likelihood is used.
    Entire retained bins intersecting the requested range are fitted, including parts
    outside it; auxiliary model bins cover those parts. Output edges NEVER
    change. Keep the fitted height in every output bin overlapping the outer
    retained-data range, including at most one partly covered bin at either end.
    Do not dilute heights by coverage. Entirely outside bins are NaN, and a bin
    that only touches the measurement boundary is outside. The fitted height
    extends across a retained partial bin; its unmeasured portion is an estimate.
    Coverage fractions are diagnostic only, not concentration weights.
    Internal unsupported gaps are inferred by smoothness and marked in diagnostics.
    This function does not calculate consensus or update alignment. Callers can
    include previously calculated consensus multipliers in the supplied alpha.

    One adjustable log-height is fitted per model bin.
    """
    edges = np.asarray(output_edges_nm, float)
    bin_overlap_matrix(edges, edges)  # Validate once before using logarithms.
    if edges.size < 4 or not series_list:
        raise ValueError('Need at least three output bins and one input series')
    if not np.isfinite(lam) or lam < 0 or max_nfev < 1:
        raise ValueError('lambda must be nonnegative and max_nfev positive')
    # Keep the order of rows and used identical: uncertainty propagation and
    # saved diagnostics use (instrument name, native index) to identify each row.
    rows = []
    used = []
    omitted = []
    for series_index, item in enumerate(series_list):
        native_edges = np.asarray(item['edges'], float)
        bin_overlap_matrix(native_edges, edges)
        measured_numbers = np.asarray(item['number'], float)
        if measured_numbers.ndim != 1 or measured_numbers.size != native_edges.size-1:
            raise ValueError('One measured number is required per native bin')
        native_weights = np.asarray(item.get('alpha', 1.), float)
        if native_weights.ndim == 0:
            native_weights = np.full_like(measured_numbers, float(native_weights))
        if (native_weights.shape != measured_numbers.shape
                or np.any(~np.isfinite(native_weights)) or np.any(native_weights < 0)):
            raise ValueError('alpha must be nonnegative, scalar or one value per native bin')
        keep = (
            np.isfinite(measured_numbers) & (measured_numbers > 0)
            & (native_weights > 0)
            & (native_edges[:-1] < edges[-1]) & (native_edges[1:] > edges[0])
        )
        endpoint = item.get('zero_endpoint')
        zero_scale = np.zeros_like(measured_numbers)
        if endpoint not in (None, 'first', 'last'):
            raise ValueError("zero_endpoint must be None, 'first', or 'last'")
        if endpoint is not None:
            scale = np.asarray(item.get('zero_number_scale', np.nan), float)
            if scale.ndim == 0:
                scale = np.full_like(measured_numbers, float(scale))
            if scale.shape != measured_numbers.shape or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError('zero_number_scale must be positive, scalar or one value per native bin')
            positives = np.flatnonzero(np.isfinite(measured_numbers) & (measured_numbers > 0))
            if positives.size:
                i = positives[0]-1 if endpoint == 'first' else positives[-1]+1
                if (0 <= i < measured_numbers.size and measured_numbers[i] == 0 and native_weights[i] > 0
                        and native_edges[i] < edges[-1] and native_edges[i+1] > edges[0]):
                    keep[i] = True
                    zero_scale[i] = scale[i]
        name = item.get('name', str(series_index))
        omitted.append({'name': name, 'total_bins': int(measured_numbers.size), 'used_bins': int(keep.sum()),
                        'zero_bins': int((measured_numbers == 0).sum()),
                        'retained_zero_bins': int((zero_scale > 0).sum())})
        for i in np.flatnonzero(keep):
            rows.append((native_edges[i], native_edges[i+1], measured_numbers[i],
                         native_weights[i], zero_scale[i]))
            used.append((name,int(i)))
    if not rows:
        raise ValueError('No positive measured-bin totals in the requested diameter range')
    lower,upper,number,alpha,zero_scale = np.asarray(rows).T
    if not np.any(number > 0):
        raise ValueError('Need positive measured-bin totals to accompany retained zeros')
    # A measured bin may extend beyond the requested output range. Add temporary
    # model bins so we still fit its whole measured total. These extra bins are
    # removed from the returned result; the user's output edges do not change.
    log_edges = np.log10(edges)
    step = np.median(np.diff(log_edges))
    left = max(0,int(np.ceil((log_edges[0]-np.log10(lower.min()))/step)))
    right = max(0,int(np.ceil((np.log10(upper.max())-log_edges[-1])/step)))
    model_edges = np.r_[10**(log_edges[0]-step*np.arange(left,0,-1)),edges,
                        10**(log_edges[-1]+step*np.arange(1,right+1))]
    # One row per retained measurement, one column per model height. Multiplying
    # overlap_matrix @ model_height predicts whole measured-bin totals.
    overlap_matrix = np.maximum(0., np.minimum(np.log10(upper)[:,None],np.log10(model_edges[1:])[None,:])
                   - np.maximum(np.log10(lower)[:,None],np.log10(model_edges[:-1])[None,:]))
    widths = np.log10(upper/lower)
    np.testing.assert_allclose(overlap_matrix.sum(axis=1),widths,rtol=1e-10,atol=1e-12)
    with np.errstate(divide='ignore'):
        log_overlap = np.log(overlap_matrix)
    sqrt_weight = np.sqrt(alpha*widths)
    mids_log = (np.log10(model_edges[:-1])+np.log10(model_edges[1:]))/2
    # Penalize bends in the fitted log distribution. Account for bin spacing:
    # the same change over a narrow diameter interval is a sharper bend.
    curvature_operator, integration_weights = second_diff_nonuniform(mids_log)
    smooth = np.sqrt(lam*integration_weights)[:,None]*curvature_operator
    # Start from overlapping native-bin averages, not interpolated observations.
    start_weight = overlap_matrix.T*alpha[None,:]
    denominator = start_weight.sum(axis=1)
    # The optimizer needs a positive starting height even near a measured zero.
    # Use the zero-bin scale for that starting guess only; its target stays zero.
    initial_log_height = np.log10(np.where(number > 0, number, zero_scale)/widths)
    initial_log_heights = np.full(model_edges.size-1,np.median(initial_log_height))
    supported = denominator > 0
    initial_log_heights[supported] = (start_weight@initial_log_height)[supported]/denominator[supported]
    log_number = np.log10(np.where(number > 0, number, 1.))  # Zero rows are overridden.
    def residual(log_height):
        return _bin_total_system(log_height, log_overlap, log_number, sqrt_weight, smooth, zero_scale)[0]

    def jacobian(log_height):
        return _bin_total_system(log_height, log_overlap, log_number, sqrt_weight, smooth, zero_scale)[1]

    result = least_squares(residual,initial_log_heights,jac=jacobian,max_nfev=max_nfev,
                           ftol=1e-9,xtol=1e-9,gtol=1e-9,method='trf')
    if not result.success:
        raise RuntimeError(f'Bin-total fit did not converge: {result.message}')
    fitted_log_y = result.x
    model_height = 10**fitted_log_y
    if not np.all(np.isfinite(model_height) & (model_height > 0)):
        raise RuntimeError('Bin-total fit returned nonfinite or nonpositive model values')
    # Keep the user's fixed output edges and the fitted height. Permit only
    # the one boundary-straddling bin at each end; never dilute its height
    # or extend the curve into a further bin with no measurement overlap.
    covered_width = np.maximum(0., np.minimum(log_edges[1:], np.log10(upper.max()))
                               - np.maximum(log_edges[:-1], np.log10(lower.min())))
    coverage_fraction = np.clip(covered_width / np.diff(log_edges), 0., 1.)
    overlaps_range = (edges[:-1] < upper.max()) & (edges[1:] > lower.min())
    y = model_height[left:left+edges.size-1].copy()
    y[~overlaps_range] = np.nan
    coverage = np.isclose(coverage_fraction, 1., rtol=0., atol=1e-12)
    predicted = overlap_matrix@model_height
    log_error = np.full_like(number, np.nan)
    positive = number > 0
    log_error[positive] = np.log10(predicted[positive]/number[positive])
    data_error = log_error.copy()
    data_error[~positive] = predicted[~positive]/(zero_scale[~positive]*np.log(10.))
    diagnostics = {'method':'native-bin-totals-log10-v1-experimental', 'lambda':float(lam),
        'parameter_count':int(result.x.size),
        'measured_number':number,'predicted_number':predicted,'used_bins':used,'omitted':omitted,
        'bin_weight':alpha*widths,'log_residual':log_error,
        'zero_number_scale':zero_scale,'data_residual':data_error,
        'zero_policy':'adjacent-explicit-scaled-linear' if np.any(~positive) else 'omit-all',
        'data_loss':float(np.sum(alpha*widths*data_error**2)),
        'roughness':float(np.sum(integration_weights*(curvature_operator@fitted_log_y)**2)),
        'model_edges_nm':model_edges,'model_dNdlogDp':model_height,
        'output_full_range_coverage':coverage,
        'output_coverage_fraction':coverage_fraction,
        'output_direct_support':supported[left:left+edges.size-1],
        'nfev':int(result.nfev),'optimality':float(result.optimality),
        'solver_message':str(result.message)}
    return y, diagnostics


# ------------------------------ Module metadata ------------------------------ #


__all__ = [
    "log_interp",
    "second_diff_nonuniform",
    "make_grid_from_series",
    "sigma_from_bands",
    "fractional_sigma",
    "compute_data_weights",
    "smooth_weight_profile",
    "merge_sizedists_tikhonov",
    "merge_sizedists_tikhonov_consensus",
    "bin_overlap_matrix",
    "merge_sizedists_bin_totals",
    "native_bin_consensus_multipliers",
]
