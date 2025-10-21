# sizedist_combine.py
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
#   {"x": array_nm, "y": array, "sigma": array_or_None, "alpha": float_optional}
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
from scipy.optimize import lsq_linear


# ----------------------------- Interpolation ----------------------------- #

def log_interp(
    x_src_nm: np.ndarray,
    y_src: np.ndarray,
    x_dst_nm: np.ndarray,
) -> np.ndarray:
    """
    Interpolate y(x) from x_src_nm onto x_dst_nm using log10(x) as the axis.
    Values outside the source span in log-space are set to NaN.

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

    valid_src = np.isfinite(x_src_nm) & np.isfinite(y_src) & (x_src_nm > 0)
    if not np.any(valid_src):
        return np.full_like(x_dst_nm, np.nan, dtype=float)

    log_x_src = np.log10(x_src_nm[valid_src])
    y_src_ok = y_src[valid_src]
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

    L = np.zeros((n_grid - 2, n_grid), float)
    quad_w = np.zeros(n_grid - 2, float)

    # Interior nodes i = 1..n-2 (0-based indexing)
    for i in range(1, n_grid - 1):
        h_left = log_grid[i] - log_grid[i - 1]
        h_right = log_grid[i + 1] - log_grid[i]

        # Nonuniform 3-point second derivative coefficients
        coef_left = 2.0 / (h_left * (h_left + h_right))
        coef_mid = -2.0 / (h_left * h_right)
        coef_right = 2.0 / (h_right * (h_left + h_right))

        row = i - 1
        L[row, i - 1] = coef_left
        L[row, i] = coef_mid
        L[row, i + 1] = coef_right

        quad_w[row] = 0.5 * (h_left + h_right)  # local cell size for ∫(y'')^2 dt
    return L, quad_w


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
    for s in series_list:
        x = np.asarray(s["x"], float)
        y = np.asarray(s["y"], float)
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
    Convert lower/upper bands to a 1-sigma proxy: sigma = 0.5 * (y_hi - y_lo).
    """
    y_lo = np.asarray(y_lo, float)
    y_hi = np.asarray(y_hi, float)
    return 0.5 * (y_hi - y_lo)


def fractional_sigma(y: np.ndarray, frac: float) -> np.ndarray:
    """
    Fractional uncertainty proxy: sigma = |y| * frac.
    """
    y = np.asarray(y, float)
    return np.abs(y) * float(frac)


# ------------------------------- Data weighting ------------------------------- #

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

    weights_per_series = []
    for s in series_list:
        x_nm = s["x"]
        y_vals = s["y"]
        sigma_vals = s.get("sigma", None)
        alpha = float(s.get("alpha", 1.0))

        y_on_grid = log_interp(x_nm, y_vals, diam_grid_nm)

        if sigma_vals is None:
            inv_var_on_grid = np.ones_like(diam_grid_nm)
        else:
            sigma_on_grid = log_interp(x_nm, sigma_vals, diam_grid_nm)
            inv_var_on_grid = 1.0 / (np.square(sigma_on_grid) + eps)

        valid = np.isfinite(y_on_grid) & np.isfinite(inv_var_on_grid)
        eff_weight = np.zeros(n, float)
        eff_weight[valid] = alpha * inv_var_on_grid[valid]
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
    Merge multiple size distributions on a common grid with a Tikhonov curvature
    penalty in log-diameter.

    Minimizes (in stacked least-squares form):
        || D^{1/2} (M - ybar) ||_2^2 + || sqrt(lam) * Wc^{1/2} L M ||_2^2
    subject to M >= 0 if `nonneg` is True.

    Parameters
    ----------
    diam_grid_nm : (n,) array_like
        Diameter grid [nm].
    series_list : list of dict
        Each dict has "x", "y", and optionally "sigma", "alpha".
    lam : float
        Smoothness strength λ >= 0.
    eps : float
        Variance floor to avoid infinite weights.
    nonneg : bool
        If True, enforce M >= 0 via bounded least squares.

    Returns
    -------
    merged_vals : (n,) ndarray
        Merged spectrum on `diam_grid_nm`. Unsupported nodes are NaN.
    weight_sum : (n,) ndarray
        Total data weight per node, Σ_i α_i /(σ_ij^2+eps).
    diagnostics : dict
        Useful intermediate arrays and solver info:
          - "support_mask": bool mask where weight_sum > 0
          - "ybar_supported": per-node weighted mean on support
          - "sqrt_weight_data": sqrt of per-node weight on support
          - "L_smooth": second-derivative operator on support
          - "quad_w": quadrature weights on support
          - "solver": SciPy result object (status, nit, cost, etc.)
    """
    diam_grid_nm = np.asarray(diam_grid_nm, float)
    n_grid = diam_grid_nm.size
    log_grid = np.log(diam_grid_nm)

    # Accumulate inverse-variance weighted sums on the grid
    weighted_sum_y = np.zeros(n_grid, float)   # Σ_i (α_i * w_ij * y_ij)
    weight_sum = np.zeros(n_grid, float)       # Σ_i (α_i * w_ij)

    for s in series_list:
        x_nm = s["x"]
        y_vals = s["y"]
        sigma_vals = s.get("sigma", None)
        alpha = float(s.get("alpha", 1.0))

        y_on_grid = log_interp(x_nm, y_vals, diam_grid_nm)

        if sigma_vals is None:
            inv_var_on_grid = np.ones_like(diam_grid_nm)
        else:
            sigma_on_grid = log_interp(x_nm, sigma_vals, diam_grid_nm)
            inv_var_on_grid = 1.0 / (np.square(sigma_on_grid) + eps)

        valid = np.isfinite(y_on_grid) & np.isfinite(inv_var_on_grid)
        eff_weight = np.zeros(n_grid, float)
        eff_weight[valid] = alpha * inv_var_on_grid[valid]

        weighted_sum_y += eff_weight * np.nan_to_num(y_on_grid, nan=0.0)
        weight_sum += eff_weight

    # Restrict to nodes with at least one contributing instrument
    support_mask = weight_sum > 0
    if support_mask.sum() < 3:
        raise ValueError("merge_sizedists_tikhonov: insufficient supported grid points (<3).")

    grid_supported = diam_grid_nm[support_mask]
    log_grid_supported = log_grid[support_mask]
    weight_sum_supported = weight_sum[support_mask]
    weighted_sum_y_supported = weighted_sum_y[support_mask]

    tiny = 1e-30  # numerical floor

    # Data term: D^{1/2}(M - ybar) ≈ 0
    ybar_supported = weighted_sum_y_supported / np.maximum(weight_sum_supported, tiny)
    sqrt_weight_data = np.sqrt(np.maximum(weight_sum_supported, tiny))
    A_data_matrix = np.diag(sqrt_weight_data)               # shape: (ns, ns)
    rhs_data = sqrt_weight_data * ybar_supported            # shape: (ns,)

    # Smoothness term on nonuniform log grid: sqrt(lam)*Wc^{1/2}*L*M ≈ 0
    L_smooth, quad_w = second_diff_nonuniform(log_grid_supported)
    A_smooth_matrix = np.sqrt(lam) * (np.sqrt(quad_w)[:, None] * L_smooth)
    rhs_smooth = np.zeros(L_smooth.shape[0], float)

    # Stack data and smooth blocks
    A_stacked = np.vstack([A_data_matrix, A_smooth_matrix])
    rhs_stacked = np.concatenate([rhs_data, rhs_smooth])

    # Solve bounded LS: min ||A M - b||_2 subject to bounds
    lower_bound = 0.0 if nonneg else -np.inf
    upper_bound = np.inf
    result = lsq_linear(A_stacked, rhs_stacked, bounds=(lower_bound, upper_bound), method="trf")
    merged_supported = result.x

    # Scatter back to full grid; unsupported nodes -> NaN
    merged_vals = np.full_like(diam_grid_nm, np.nan, dtype=float)
    merged_vals[support_mask] = merged_supported

    diagnostics = {
        "support_mask": support_mask,
        "ybar_supported": ybar_supported,
        "sqrt_weight_data": sqrt_weight_data,
        "L_smooth": L_smooth,
        "quad_w": quad_w,
        "solver": result,
    }
    return merged_vals, weight_sum, diagnostics


# ------------------------------ Module metadata ------------------------------ #

__all__ = [
    "log_interp",
    "second_diff_nonuniform",
    "make_grid_from_series",
    "sigma_from_bands",
    "fractional_sigma",
    "compute_data_weights",
    "merge_sizedists_tikhonov",
]