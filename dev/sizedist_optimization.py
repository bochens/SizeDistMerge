# sizedist_optimization.py
# Minimal, generic optimizer for refractive index (n, k) using an OPC σ-LUT remap.
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution

# Requires your LUT utilities to be available on PYTHONPATH
from optical_diameter_core import SigmaLUT, convert_do_lut  # remaps EDGES via σ(D; m)
from sizedist_utils import remap_dndlog_by_edges, mids_from_edges  # number-conserving remap + mids

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


def mse_overlap_sizedist(x1, y1, x2, y2, *, metric="N", space="linear"):
    """
    Compare two size distributions over their overlapping diameter range.

    Parameters
    ----------
    x1, y1 : arrays
        Mid diameters and spectrum 1 (same units as x2, y2).
    x2, y2 : arrays
        Mid diameters and spectrum 2.
    metric : {"N","S","V"}
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
    D = np.geomspace(lo, hi, npts)

    y1g = np.interp(D, x1, y1)
    y2g = np.interp(D, x2, y2)

    D_um = D / 1000.0
    if metric == "S":
        w = np.pi * (D_um ** 2)
    elif metric == "V":
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
    metric="N",
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

    return mse_overlap_sizedist(D_ref, y_ref, Dm, Nm, metric=metric, space=space)


def optimize_refractive_index_for_opc(
    x_ref,                 # reference mids [nm]
    y_ref,                 # reference dN/dlog10D on x_ref
    edges_DRV,             # OPC edges [nm], length = M+1
    y_DRV,                 # OPC dN/dlog10D on edges_DRV bins, length = M
    m_src,
    lut: SigmaLUT,
    *,
    response_bins=50,
    metric="N",
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
    x_ref, y_ref = _clean(x_ref, y_ref)
    y_DRV = np.asarray(y_DRV, dtype=float)

    history = {"total": []}

    def obj(nk):
        return objective_opc_vs_ref(
            nk,                         # nk is a 2d vector, same as bounds
            x_ref,
            y_ref,
            edges_DRV,
            y_DRV,
            m_src,
            lut,
            response_bins=response_bins,
            metric=metric,
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