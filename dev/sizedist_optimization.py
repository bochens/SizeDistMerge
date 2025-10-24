# sizedist_optimization.py
# Minimal, generic optimizer for refractive index (n, k) using an OPC σ-LUT remap.
from __future__ import annotations
from typing import List, Dict, Tuple, Callable
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution

from sizedist_utils import remap_dndlog_by_edges, mids_from_edges  # number-conserving remap + mids
from optical_diameter_core import SigmaLUT, convert_do_lut  # remaps EDGES via σ(D; m)
from diameter_conversion_core import da_to_dv  # APS-style diameter conversion


__all__ = [
    "_clean",
    "clip_with_edges",
    "mse_overlap_sizedist",
    "objective_opc_vs_ref",
    "optimize_refractive_index_for_opc",
]


def _clean(x, y):
    """
    Remove non-finite and non-positive entries. Returns arrays of same length subset.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[m], y[m]


def clip_with_edges(mids, y, Dlo, Dhi, xmin=None, xmax=None):
    """
    Clip a spectrum and its bin edges to [xmin, xmax] in the mid-point space.
    Pass None to skip a bound.
    """
    mids = np.asarray(mids, dtype=float)
    y    = np.asarray(y, dtype=float)
    Dlo  = np.asarray(Dlo, dtype=float)
    Dhi  = np.asarray(Dhi, dtype=float)

    m = np.ones_like(mids, dtype=bool)
    if xmin is not None:
        m &= mids >= float(xmin)
    if xmax is not None:
        m &= mids <= float(xmax)
    return mids[m], y[m], Dlo[m], Dhi[m]

def _has_overlap(xa: np.ndarray, xb: np.ndarray) -> bool:
    xa = np.asarray(xa, float); xb = np.asarray(xb, float)
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

    npts = int(max(32, min(128, x1.size, x2.size)))
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
    else:
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

    This function is agnostic to instrument type. Each instrument provides:
      - "edges": np.ndarray, original bin edges [nm], length M_i+1
      - "y":     np.ndarray, original dN/dlog10D on those bins, length M_i
      - "remap_fn": Callable(edges, theta, **kwargs) -> new_edges
            Your function that maps OLD edges -> NEW edges given a parameter
            vector theta (any length) and any fixed kwargs you need (e.g., LUTs).
      - "kwargs": dict, fixed keyword arguments passed to remap_fn
      - "w_ref": float, weight for instrument-vs-reference term (default 1.0)

    Parameters
    ----------
    x_vec : np.ndarray
        Flattened optimizer parameter vector (concatenation of all instruments'
        parameter vectors). Use `param_slices[i]` to extract theta_i for instrument i.
    ref_mids, ref_y : arrays
        The reference sizedist midpoints and values (already in the moment you want).
        The reference is NOT remapped here.
    instruments : list of dict
        See definition above. Each dict fully defines how to remap one instrument.
    param_slices : list of slice
        Output of _build_param_slices(bounds_list). Each slice selects theta_i.
    moment : {"N", "S", "V"}
        Which weighting to use inside MSE:
          N → number       (weight = 1)
          S → surface      (weight ~ D^2)
          V → volume       (weight ~ D^3)
        D is in micrometers internally for the weighting.
    space : {"linear", "log"}
        Compute MSE either on linear values or on log10(values) when positive.
    pair_weights : list[tuple(int, int, float)] | None
        Optional cross-instrument terms. Each tuple (i, j, w) adds:
            w * MSE(remap(i), remap(j))
        Use indices into `instruments` (0-based).

    Returns
    -------
    float
        Total cost for the optimizer to minimize.
    """
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
    # If NOTHING overlapped, return 0.0 (neutral: neither reward nor penalize).
    if weight_sum > 0.0:
        return float(cost_sum / weight_sum)
    else:
        return 0.0


def optimize_multi_custom(
    ref_mids: np.ndarray, ref_y: np.ndarray,
    instruments: List[Dict[str, object]],
    bounds_list: List[List[Tuple[float, float]]],  # per-instrument list of (lo, hi) bounds
    *,
    moment: str = "N",
    space: str = "linear",
    pair_weights: List[Tuple[int, int, float]] | None = None,
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
    # --- Flatten bounds for SciPy and build param slices to unpack per instrument ---
    # flat_bounds is a simple list of (lo, hi) tuples for each scalar parameter.
    # param_slices[i] selects theta_i from a flat x_vec as x_vec[param_slices[i]].
    flat_bounds: List[Tuple[float, float]] = [b for inst in bounds_list for b in inst]
    param_slices = _build_param_slices(bounds_list)

    history = {"total": []}

    # Closure passed to SciPy: maps x_vec -> scalar cost
    def obj(x_vec):
        return objective_multi_custom(
            x_vec, ref_mids, ref_y, instruments, param_slices,
            moment=moment, space=space, pair_weights=pair_weights,
        )

    # Callback to record convergence history (optional but handy)
    def _cb(xk, _conv):
        history["total"].append(float(obj(xk)))
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

    return best_thetas, float(res.fun), res, history