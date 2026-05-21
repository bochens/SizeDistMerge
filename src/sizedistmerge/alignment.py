"""Alignment and optimization helpers for overlapping size distributions."""

from __future__ import annotations
from typing import List, Dict, Tuple, Callable
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution

from .utils import remap_dndlog_by_edges, mids_from_edges
from .optical_diameter import SigmaLUT, convert_do_lut


__all__ = [
    "mse_overlap_sizedist",
    "objective_opc_vs_ref",
    "optimize_refractive_index_for_opc",
    "temporal_parameter_penalty",
    "objective_joint_named_temporal",
    "objective_multi_custom",
    "optimize_multi_custom",
]


def _clean(x, y):
    """
    Remove non-finite and non-positive entries. Returns arrays of same length subset.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[m], y[m]


def _has_overlap(xa: np.ndarray, xb: np.ndarray) -> bool:
    xa = np.asarray(xa, float)
    xb = np.asarray(xb, float)
    xa = xa[np.isfinite(xa) & (xa > 0)]
    xb = xb[np.isfinite(xb) & (xb > 0)]
    if xa.size == 0 or xb.size == 0: 
        return False
    lo = max(np.min(xa), np.min(xb))
    hi = min(np.max(xa), np.max(xb))
    return np.isfinite(lo) and np.isfinite(hi) and lo < hi and lo > 0.0


def mse_overlap_sizedist(x1, y1, x2, y2, *, moment="N", space="linear"):
    """
    Compare two size distributions over their overlapping diameter range.

    Parameters
    ----------
    x1, y1 : arrays
        Mid diameters and spectrum 1 (same units as x2, y2).
    x2, y2 : arrays
        Mid diameters and spectrum 2.
    moment : {"N","S","V"}
        Weight by number (1), surface (π D^2), or volume ((π/6) D^3). D in μm.
    space : {"linear","log"}
        Compute MSE in chosen space. Log requires positive values.

    Returns
    -------
    float
        Mean squared error over a common log-spaced grid.
    """
    x1 = np.asarray(x1, float)
    y1 = np.asarray(y1, float)
    x2 = np.asarray(x2, float)
    y2 = np.asarray(y2, float)

    lo = max(np.min(x1), np.min(x2))
    hi = min(np.max(x1), np.max(x2))
    if lo >= hi:
        return np.inf

    npts = int(max(128, min(1024, x1.size, x2.size)))
    L = np.linspace(np.log10(lo), np.log10(hi), npts); D = 10.0**L

    y1g = np.interp(L, np.log10(x1), y1)
    y2g = np.interp(L, np.log10(x2), y2)

    D_um = D / 1000.0
    if moment == "S":
        w = np.pi * (D_um ** 2)
    elif moment == "V":
        w = (np.pi / 6.0) * (D_um ** 3)
    else:
        w = 1.0

    Y1 = y1g * w
    Y2 = y2g * w

    if space == "log":
        m = (Y1 > 0) & (Y2 > 0) & np.isfinite(Y1) & np.isfinite(Y2)
        if not np.any(m):
            return np.inf
        return float(mean_squared_error(np.log10(Y1[m]), np.log10(Y2[m])))
    elif space == "linear":
        m = np.isfinite(Y1) & np.isfinite(Y2)
        if not np.any(m):
            return np.inf
        return float(mean_squared_error(Y1[m], Y2[m]))
    raise ValueError("space must be 'linear' or 'log'")



def objective_opc_vs_ref(
    nk,
    D_ref,                 # reference mids [nm]
    y_ref,                 # reference dN/dlog10D on D_ref
    edges_DRV,             # OPC edges [nm], length = M+1
    y_DRV,                 # OPC dN/dlog10D on edges_DRV bins, length = M
    m_src,                 # complex refractive index used by OPC binning (source)
    lut: SigmaLUT,
    *,
    response_bins=50,
    moment="N",
    space="linear",
):
    """
    Objective for optimizer. Remap OPC bin EDGES from m_src to trial m_dst = n+ik,
    then remap dN/dlog10D by conserving bin counts, and compare to reference.

    Returns the MSE between reference (D_ref, N_ref) and remapped OPC (Dm, Nm).
    """
    n, k = float(nk[0]), float(nk[1])

    # 1) Map EDGE array via σ(D; m): edges' -> edges''
    edges_mapped = convert_do_lut(
        Do_nm=edges_DRV,
        ri_src=m_src,
        ri_dst=complex(n, k),
        lut=lut,
        response_bins=response_bins,
    )

    # 2) Remap spectrum from old edges -> new edges (number-conserving)
    Nm = remap_dndlog_by_edges(edges_DRV, edges_mapped, y_DRV)

    # 3) Compare on mids of mapped edges
    Dm = mids_from_edges(edges_mapped)

    return mse_overlap_sizedist(D_ref, y_ref, Dm, Nm, moment=moment, space=space)


def optimize_refractive_index_for_opc(
    mids_ref,              # reference mids [nm]
    y_ref,                 # reference distribution on mids_ref
    edges_DRV,             # OPC edges [nm], length = M+1
    y_DRV,                 # OPC dN/dlog10D on edges_DRV bins, length = M
    ri_src,
    lut: SigmaLUT,
    *,
    response_bins=50,
    moment="N",
    space="linear",
    bounds=((1.3, 1.8), (0.0, 0.1)),
    maxiter=200,
    tol=1e-6,
    seed=123,
):
    """
    Fit complex refractive index (n, k) that best matches an OPC spectrum (given as
    bin edges + dN/dlog10D) remapped via σ(D; m) to a reference spectrum.

    Returns
    -------
    n_best, k_best, best_cost, result, history
    """
    mids_ref, y_ref = _clean(mids_ref, y_ref)
    y_DRV = np.asarray(y_DRV, dtype=float)

    history = {"total": []}

    def obj(nk):
        return objective_opc_vs_ref(
            nk,                         # nk is a 2d vector, same as bounds
            mids_ref,
            y_ref,
            edges_DRV,
            y_DRV,
            ri_src,
            lut,
            response_bins=response_bins,
            moment=moment,
            space=space,
        )

    def _cb(xk, _convergence):
        history["total"].append(float(obj(xk)))
        return False

    result = differential_evolution(
        obj,
        bounds=bounds,
        strategy="best1bin",
        maxiter=maxiter,
        tol=tol,
        recombination=0.7,
        polish=True,
        seed=seed,
        callback=_cb,
    )
    n_best = float(result.x[0])
    k_best = float(result.x[1])
    return n_best, k_best, float(result.fun), result, history



def _build_param_slices(bounds_list: List[List[Tuple[float, float]]]) -> List[slice]:
    """
    Given per-instrument bounds, return slices that index each instrument's
    parameter vector inside the single flattened optimizer vector `x_vec`.

    Parameters
    ----------
    bounds_list : list of list of (float, float)
        For each instrument i, bounds_list[i] is a list of (lo, hi) tuples,
        one for each parameter of that instrument. Example:
            [
              [(1.3, 1.8), (0.0, 0.1)],       # instrument 0 has 2 params
              [(600.0, 2000.0), (0.8, 1.4)],  # instrument 1 has 2 params
              [(0.9, 1.2)]                    # instrument 2 has 1 param
            ]

    Returns
    -------
    slices : list of slice
        A list of Python slices, one per instrument, so that:
            theta_i = x_vec[slices[i]]
        selects the sub-vector for instrument i.
    """
    slices = []
    start = 0
    for bnds in bounds_list:
        p = len(bnds)                  # number of parameters for this instrument
        slices.append(slice(start, start + p))
        start += p                     # advance start by the number of params
    return slices


def _multi_custom_data_cost_and_flag(
    x_vec: np.ndarray,
    ref_mids: np.ndarray, ref_y: np.ndarray,   # fixed reference curve (no remap)
    instruments: List[Dict[str, object]],      # per-instrument configs (see below)
    param_slices: List[slice],                  # slices into x_vec for each instrument
    *,
    moment: str = "N",
    space: str = "linear",
    pair_weights: List[Tuple[int, int, float]] | None = None,  # optional list of (i, j, w)
):
    # --- Remap all instruments for this x_vec ---
    # For each instrument i:
    #   - extract its parameter subvector theta_i
    #   - compute new edges with remap_fn(edges, theta_i, **kwargs)
    #   - number-conserving remap of y onto new edges
    #   - store (mids, y_remapped) for later comparisons
    curves = []  # list of (mids, y_remapped) in the same order as `instruments`
    for i, cfg in enumerate(instruments):
        s = param_slices[i]
        theta = np.asarray(x_vec[s], float)             # instrument i parameters
        edges_src = np.asarray(cfg["edges"], float)     # original edges
        y_src = np.asarray(cfg["y"], float)             # original spectrum on those edges
        remap_fn: Callable = cfg["remap_fn"]            # user-supplied remapper
        kwargs = dict(cfg.get("kwargs", {}))            # extra args for remap_fn

        # NEW edges via user remap function (instrument-specific physics lives inside)
        edges_dst = np.asarray(remap_fn(edges_src, theta, **kwargs), float)

        # Number-conserving remap of the spectrum from old bins → new bins
        y_dst = remap_dndlog_by_edges(edges_src, edges_dst, y_src)

        # Midpoints of the new edges, used for overlap/MSE comparisons
        mids = mids_from_edges(edges_dst)
        curves.append((mids, y_dst))

    # --- Accumulate cost vs reference (overlap-aware, weight-normalized) ---
    cost_sum = 0.0
    weight_sum = 0.0

    for i, cfg in enumerate(instruments):
        w_ref = float(cfg.get("w_ref", 1.0))
        if w_ref == 0.0:
            continue

        xm, ym = curves[i]
        # Only score if there is actual overlap with the reference
        if _has_overlap(ref_mids, xm):
            mse = mse_overlap_sizedist(ref_mids, ref_y, xm, ym, moment=moment, space=space)
            cost_sum   += w_ref * mse
            weight_sum += w_ref

    # --- Optional cross-instrument pairwise costs (also overlap-aware) ---
    if pair_weights:
        for (i, j, w) in pair_weights:
            w = float(w)
            if w == 0.0:
                continue
            if 0 <= i < len(curves) and 0 <= j < len(curves):
                xi, yi = curves[i]
                xj, yj = curves[j]
                if _has_overlap(xi, xj):
                    mse = mse_overlap_sizedist(xi, yi, xj, yj, moment=moment, space=space)
                    cost_sum   += w * mse
                    weight_sum += w

    # Average by total weight of the comparisons that actually had overlap.
    # If NOTHING overlapped, keep the old neutral behavior.
    if weight_sum > 0.0:
        return float(cost_sum / weight_sum), True
    return 0.0, False


def objective_multi_custom(
    x_vec: np.ndarray,
    ref_mids: np.ndarray, ref_y: np.ndarray,   # fixed reference curve (no remap)
    instruments: List[Dict[str, object]],      # per-instrument configs (see below)
    param_slices: List[slice],                  # slices into x_vec for each instrument
    *,
    moment: str = "N",
    space: str = "linear",
    pair_weights: List[Tuple[int, int, float]] | None = None,  # optional list of (i, j, w)
) -> float:
    """
    Compute a weighted sum of mismatch (MSE) terms between remapped instruments
    and a reference curve, plus optional cross-instrument pairwise terms.

    Cost function:
        cost = sum_i  w_ref_i * MSE(remap(i), ref)
             + sum_{(i,j) in pairs} w_ij * MSE(remap(i), remap(j))

    Instrument/reference and pair terms are skipped when their spectra do not
    overlap. If no real comparison contributes, this keeps the old neutral
    behavior and returns 0.0.
    """
    cost, _has_comparison = _multi_custom_data_cost_and_flag(
        x_vec,
        ref_mids,
        ref_y,
        instruments,
        param_slices,
        moment=moment,
        space=space,
        pair_weights=pair_weights,
    )
    return cost


def temporal_parameter_penalty(
    x_vec: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Penalize parameter jumps relative to the previous successful chunk.

    The penalty is an unnormalized weighted sum:
        sum_i weights_i * (x_i - target_i)^2

    Non-positive weights disable a parameter. Targets with positive weights
    must be finite so bad temporal state cannot silently enter the optimizer.
    """
    x_vec = np.asarray(x_vec, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if x_vec.ndim != 1 or target.ndim != 1 or weights.ndim != 1:
        raise ValueError("x_vec, target, and weights must be one-dimensional")
    if x_vec.size != target.size or x_vec.size != weights.size:
        raise ValueError("temporal target and weights must match optimizer parameter length")
    if not np.all(np.isfinite(weights)):
        raise ValueError("temporal weights must be finite")
    if np.any(weights < 0.0):
        raise ValueError("temporal weights must be non-negative")

    active = weights > 0.0
    if not np.any(active):
        return 0.0
    if not np.all(np.isfinite(target[active])):
        raise ValueError("temporal target must be finite where temporal weight is positive")
    if not np.all(np.isfinite(x_vec[active])):
        return np.inf

    delta = x_vec[active] - target[active]
    return float(np.sum(weights[active] * delta * delta))


def _remap_one_instrument(edges, y, remap_fn: Callable, theta, kwargs):
    edges_src = np.asarray(edges, dtype=float)
    y_src = np.asarray(y, dtype=float)
    edges_dst = np.asarray(remap_fn(edges_src, np.asarray(theta, dtype=float), **dict(kwargs or {})), dtype=float)
    y_dst = remap_dndlog_by_edges(edges_src, edges_dst, y_src)
    return mids_from_edges(edges_dst), y_dst


def objective_joint_named_temporal(
    params,
    *,
    ref_mids,
    ref_y,
    uhsas_edges,
    uhsas_y,
    pops_edges,
    pops_y,
    aps_edges,
    aps_y,
    uhsas_remap_fn,
    uhsas_kwargs,
    pops_remap_fn,
    pops_kwargs,
    aps_remap_fn,
    aps_kwargs,
    moment: str = "V",
    space: str = "linear",
    w_uhsas: float = 1.0,
    w_pops: float = 1.0,
    w_aps: float = 1.0,
    pair_w: float = 0.0,
    temporal_w: float = 0.0,
    temporal_w_uh: float = 0.0,
    temporal_w_po: float = 0.0,
    temporal_w_rho: float = 0.0,
    prev_params=None,
    smooth_rho: bool = True,
):
    """
    Named UHSAS/POPS/APS objective with old no-overlap behavior.

    Instrument/reference terms and pair terms are skipped when there is no
    overlap. If no real comparison contributes, the objective returns 0.0 and
    does not apply temporal penalties.
    """
    params = np.asarray(params, dtype=float)
    if params.shape != (3,):
        raise ValueError("params must be shape (3,) = [n_uhsas, n_pops, rho_aps]")
    n_uh, n_po, rho = float(params[0]), float(params[1]), float(params[2])

    xm_uh, ym_uh = _remap_one_instrument(
        uhsas_edges, uhsas_y, uhsas_remap_fn, np.array([n_uh], float), uhsas_kwargs
    )
    xm_po, ym_po = _remap_one_instrument(
        pops_edges, pops_y, pops_remap_fn, np.array([n_po], float), pops_kwargs
    )
    xm_ap, ym_ap = _remap_one_instrument(
        aps_edges, aps_y, aps_remap_fn, np.array([rho], float), aps_kwargs
    )

    cost_sum = 0.0
    w_sum = 0.0

    for w, xm, ym in (
        (float(w_uhsas), xm_uh, ym_uh),
        (float(w_pops), xm_po, ym_po),
        (float(w_aps), xm_ap, ym_ap),
    ):
        if w == 0.0 or not _has_overlap(ref_mids, xm):
            continue
        mse = mse_overlap_sizedist(ref_mids, ref_y, xm, ym, moment=moment, space=space)
        if np.isfinite(mse):
            cost_sum += w * mse
            w_sum += w

    base = (cost_sum / w_sum) if w_sum > 0.0 else 0.0
    any_comparison = w_sum > 0.0

    if pair_w != 0.0:
        pair_cost = 0.0
        pair_count = 0

        for xa, ya, xb, yb in (
            (xm_uh, ym_uh, xm_ap, ym_ap),
            (xm_po, ym_po, xm_ap, ym_ap),
        ):
            if not _has_overlap(xa, xb):
                continue
            mse = mse_overlap_sizedist(xa, ya, xb, yb, moment=moment, space=space)
            if np.isfinite(mse):
                pair_cost += mse
                pair_count += 1

        if pair_count > 0:
            base = float(base) + float(pair_w) * (pair_cost / pair_count)
            any_comparison = True

    if (not any_comparison) or (not np.isfinite(base)):
        return 0.0

    if prev_params is not None:
        prev = np.asarray(prev_params, dtype=float)
        if prev.shape != (3,):
            raise ValueError("prev_params must be shape (3,) = [n_uh, n_po, rho]")

        d_uh = n_uh - float(prev[0])
        d_po = n_po - float(prev[1])
        d_rho = rho - float(prev[2])

        tw = float(temporal_w)
        if tw != 0.0:
            base += tw * (d_uh * d_uh + d_po * d_po)
            if smooth_rho:
                base += tw * (d_rho * d_rho)

        t_uh = float(temporal_w_uh)
        t_po = float(temporal_w_po)
        t_rh = float(temporal_w_rho)
        if t_uh != 0.0:
            base += t_uh * (d_uh * d_uh)
        if t_po != 0.0:
            base += t_po * (d_po * d_po)
        if smooth_rho and t_rh != 0.0:
            base += t_rh * (d_rho * d_rho)

    return float(base)


def optimize_multi_custom(
    ref_mids: np.ndarray, ref_y: np.ndarray,
    instruments: List[Dict[str, object]],
    bounds_list: List[List[Tuple[float, float]]],  # per-instrument list of (lo, hi) bounds
    *,
    moment: str = "N",
    space: str = "linear",
    pair_weights: List[Tuple[int, int, float]] | None = None,
    temporal_target: np.ndarray | None = None,
    temporal_weights: np.ndarray | None = None,
    maxiter: int = 200,
    tol: float = 1e-6,
    seed: int = 123,
):
    """
    Generic optimizer over heterogeneous instruments.
    You provide:
      - a single fixed reference curve (ref_mids, ref_y),
      - a list of instrument configs (edges, y, remap_fn, kwargs, w_ref),
      - per-instrument parameter bounds.

    The parameter vector for the optimizer is the concatenation of each
    instrument's parameters (in the same order as `instruments`), where the
    number of parameters for instrument i is len(bounds_list[i]).

    Parameters
    ----------
    ref_mids, ref_y : arrays
        Reference mids and sizedist values (NOT remapped).
    instruments : list of dict
        Each dict describes one instrument to be remapped. Required keys:
          "edges"    : np.ndarray of bin edges (length M_i+1)
          "y"        : np.ndarray of sizedist on those edges (length M_i)
          "remap_fn" : Callable(edges, theta, **kwargs) -> new_edges
          "kwargs"   : dict of fixed keyword args passed to remap_fn
          "w_ref"    : float weight vs reference (default 1.0)
        NOTE: We do not assume anything about the physics; remap_fn can do
              RI-based LUT mapping, aerodynamic conversions, etc.
    bounds_list : list of list of (lo, hi)
        For each instrument i, a list of (lo, hi) tuples, one per parameter of
        theta_i. The product over i defines the full hyper-rectangle the
        optimizer searches. Example:
            bounds_list = [
                [(1.3, 1.8), (0.0, 0.1)],        # theta_0 has 2 params
                [(600.0, 2000.0), (0.8, 1.4)]    # theta_1 has 2 params
            ]
        These become SciPy differential_evolution `bounds`.

    moment : {"N","S","V"}, optional (default "N")
        Weighting used inside the MSE overlap comparison.
    space : {"linear","log"}, optional (default "linear")
        Whether to compute MSE in linear or log10 space (log requires positives).
    pair_weights : list[tuple(int,int,float)] | None
        Optional list of (i, j, w) to penalize mismatch between *remapped*
        instruments i and j in addition to reference comparisons.
        Use i, j indices into `instruments` (0-based).
    temporal_target, temporal_weights : arrays or None
        Optional parameter-space regularization against the previous
        successful chunk. Arrays must match the flattened optimizer parameter
        vector. The temporal term added to the objective is
        sum(weights * (x - target)^2).
    maxiter, tol, seed : optimizer controls
        Passed directly to SciPy differential_evolution.

    Returns
    -------
    best_thetas : list[np.ndarray]
        The optimized parameter vectors theta_i for each instrument i, in the
        same order as provided in `instruments`. Each theta_i has length equal
        to len(bounds_list[i]).
    best_cost : float
        Final objective value at the optimizer's best solution.
    result : scipy.optimize.OptimizeResult
        Full result object from differential_evolution (for diagnostics).
    history : dict
        Contains "total": list of objective values recorded via callback after
        each population update (useful for plotting convergence).

    Notes
    -----
    - This function does not attempt to "know" instrument types. Whatever
      physics is required to map edges must be handled inside the user-supplied
      `remap_fn` and its kwargs.
    - Count conservation is always enforced via `remap_dndlog_by_edges`.
    - Overlap for MSE is computed on a geometric grid spanning the common range.
    """
    if len(instruments) != len(bounds_list):
        raise ValueError("instruments and bounds_list must have the same length")
    if not instruments:
        raise ValueError("at least one instrument is required")

    # --- Flatten bounds for SciPy and build param slices to unpack per instrument ---
    # flat_bounds is a simple list of (lo, hi) tuples for each scalar parameter.
    # param_slices[i] selects theta_i from a flat x_vec as x_vec[param_slices[i]].
    flat_bounds: List[Tuple[float, float]] = [b for inst in bounds_list for b in inst]
    if not flat_bounds:
        raise ValueError("bounds_list must contain at least one parameter bound")
    param_slices = _build_param_slices(bounds_list)

    if (temporal_target is None) != (temporal_weights is None):
        raise ValueError("temporal_target and temporal_weights must be provided together")
    if temporal_target is not None:
        temporal_target = np.asarray(temporal_target, dtype=float)
        temporal_weights = np.asarray(temporal_weights, dtype=float)
        if temporal_target.shape != (len(flat_bounds),) or temporal_weights.shape != (len(flat_bounds),):
            raise ValueError("temporal_target and temporal_weights must match flattened bounds length")
        _ = temporal_parameter_penalty(
            np.asarray([lo for lo, _hi in flat_bounds], dtype=float),
            temporal_target,
            temporal_weights,
        )

    history = {"total": [], "data": [], "temporal": []}

    # Closure passed to SciPy: maps x_vec -> scalar cost
    def data_obj_with_flag(x_vec):
        return _multi_custom_data_cost_and_flag(
            x_vec, ref_mids, ref_y, instruments, param_slices,
            moment=moment, space=space, pair_weights=pair_weights,
        )

    def data_obj(x_vec):
        data_cost, _has_comparison = data_obj_with_flag(x_vec)
        return data_cost

    def temporal_obj(x_vec):
        if temporal_target is None or temporal_weights is None:
            return 0.0
        return temporal_parameter_penalty(x_vec, temporal_target, temporal_weights)

    def obj(x_vec):
        data_cost, has_comparison = data_obj_with_flag(x_vec)
        temporal_cost = temporal_obj(x_vec) if has_comparison and np.isfinite(data_cost) else 0.0
        return float(data_cost + temporal_cost)

    # Callback to record convergence history (optional but handy)
    def _cb(xk, _conv):
        data_cost, has_comparison = data_obj_with_flag(xk)
        temporal_cost = temporal_obj(xk) if has_comparison and np.isfinite(data_cost) else 0.0
        history["data"].append(float(data_cost))
        history["temporal"].append(float(temporal_cost))
        history["total"].append(float(data_cost + temporal_cost))
        return False  # returning True would stop the optimizer early

    # --- Run differential evolution over the flat parameter vector ---
    res = differential_evolution(
        obj,
        bounds=flat_bounds,
        strategy="best1bin",
        maxiter=maxiter,
        tol=tol,
        recombination=0.7,
        polish=True,
        seed=seed,
        callback=_cb,
    )

    # --- Unpack best solution back into per-instrument theta vectors ---
    best_thetas: List[np.ndarray] = []
    for s in param_slices:
        best_thetas.append(np.asarray(res.x[s], float).copy())

    best_data_cost, best_has_comparison = data_obj_with_flag(res.x)
    best_temporal_cost = (
        temporal_obj(res.x)
        if best_has_comparison and np.isfinite(best_data_cost)
        else 0.0
    )
    history["best_data_cost"] = float(best_data_cost)
    history["best_temporal_cost"] = float(best_temporal_cost)
    history["best_total_cost"] = float(best_data_cost + best_temporal_cost)

    return best_thetas, float(res.fun), res, history
