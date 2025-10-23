# sizedist_optimization.py
# Minimal, generic optimizer for refractive index (n, k) using an OPC σ-LUT remap.

from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution

# Requires your LUT utilities to be available on PYTHONPATH
from dev.opc_response_calculation import remap_bins_lut, SigmaLUT  # SigmaLUT is imported for convenience


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
    Compare two dN/dlogD sizedists over their overlapping diameter range.

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

    xu1, yu1 = x1, y1
    xu2, yu2 = x2, y2

    lo = max(np.min(xu1), np.min(xu2))
    hi = min(np.max(xu1), np.max(xu2))
    if lo >= hi:
        return np.inf

    npts = int(max(32, min(128, xu1.size, xu2.size)))
    D = np.geomspace(lo, hi, npts)

    y1g = np.interp(D, xu1, yu1)
    y2g = np.interp(D, xu2, yu2)

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
    D_ref,
    N_ref,
    Dlo_DRV,
    Dhi_DRV,
    N_DRV,
    m_src,
    lut: SigmaLUT,
    *,
    n_bins=100,
    metric="N",
    space="linear",
):
    """
    Objective for optimizer. Remap OPC spectrum from m_src to trial m_dst = n+ik
    then compare to reference spectrum over the overlap.
    """
    n, k = float(nk[0]), float(nk[1])
    Dm, Nm, _ = remap_bins_lut(
        D_lower_nm=Dlo_DRV,
        D_upper_nm=Dhi_DRV,
        dNdlog10D=N_DRV,
        m_src=m_src,
        m_dst=complex(n, k),
        lut=lut,
        n_bins=n_bins,
    )
    return mse_overlap_sizedist(D_ref, N_ref, Dm, Nm, metric=metric, space=space)


def optimize_refractive_index_for_opc(
    x_ref,
    y_ref,
    Dlo_DRV,
    Dhi_DRV,
    y_DRV,
    m_src,
    lut: SigmaLUT,
    *,
    n_bins=100,
    metric="N",
    space="linear",
    bounds=((1.3, 1.8), (0.0, 0.1)),
    maxiter=200,
    tol=1e-6,
    seed=123,
):
    """
    Fit complex refractive index (n, k) that best matches a remapped OPC spectrum
    to a reference spectrum.

    Returns
    -------
    n_best, k_best, best_cost, result, history
    """
    x_ref, y_ref = _clean(x_ref, y_ref)
    y_DRV = np.asarray(y_DRV, dtype=float)

    history = {"total": []}

    def obj(nk):
        return objective_opc_vs_ref(
            nk,
            x_ref,
            y_ref,
            Dlo_DRV,
            Dhi_DRV,
            y_DRV,
            m_src,
            lut,
            n_bins=n_bins,
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