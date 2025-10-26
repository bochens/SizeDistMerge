# src/sizedist_utils.py
# super-minimal helpers for aerosol size distributions (mids-first, no OOP, no validation)

from __future__ import annotations
from typing import Optional
import numpy as np

# ───────────────────────── geometry ─────────────────────────

def edges_from_mids_geometric(bin_mid_nm: np.ndarray) -> np.ndarray:
    """
    Infer bin edges from geometric centers; extrapolate end edges by neighbor ratios.
    Returns array of length N+1. If N < 2, returns [].
    """
    c = np.asarray(bin_mid_nm, float)
    if c.ndim != 1 or c.size < 2:
        return np.asarray([], dtype=float)
    inner = np.sqrt(c[:-1] * c[1:])
    r0, rN = c[1] / c[0], c[-1] / c[-2]
    lower0 = c[0] / np.sqrt(r0)
    upperN = c[-1] * np.sqrt(rN)
    return np.concatenate([[lower0], inner, [upperN]])

def mids_from_edges(edges_nm: np.ndarray) -> np.ndarray:
    """Geometric midpoints from edges. If len<2, returns []."""
    e = np.asarray(edges_nm, float)
    if e.ndim != 1 or e.size < 2:
        return np.asarray([], dtype=float)
    return np.sqrt(e[:-1] * e[1:])

def delta_log10_from_edges(edges_nm: np.ndarray) -> np.ndarray:
    """Δlog10(Dp) for each bin from edges. If len<2, returns []."""
    e = np.asarray(edges_nm, float)
    if e.ndim != 1 or e.size < 2:
        return np.asarray([], dtype=float)
    return np.log10(e[1:]) - np.log10(e[:-1])

# ───────────────────────── converters ───────────────────────

def dsdlog_from_dndlog(bin_mid_nm: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """
    arr -> dS/dlogDp via π * (D_um)^2.
    Use this SAME function on your uncertainty array to propagate ±.
    """
    D_um = np.asarray(bin_mid_nm, float) * 1e-3
    return np.asarray(arr, float) * (np.pi * D_um**2)

def dvdlog_from_dndlog(bin_mid_nm: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """
    arr -> dV/dlogDp via (π/6) * (D_um)^3.
    Use this SAME function on your uncertainty array to propagate ±.
    """
    D_um = np.asarray(bin_mid_nm, float) * 1e-3
    return np.asarray(arr, float) * ((np.pi / 6.0) * D_um**3)

# ─────────────────────── optional extras ────────────────────
# only if you want counts <-> spectrum. If you don't, ignore these.

def counts_from_dndlog(
    dndlogdp: np.ndarray,
    *,
    bin_mid_nm: Optional[np.ndarray] = None,
    edges_nm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    dN per bin = (dN/dlogDp) * Δlog10(Dp).
    Provide edges_nm directly, or just bin_mid_nm and we'll infer edges.
    If neither is usable, returns [].
    """
    if edges_nm is None and bin_mid_nm is None:
        return np.asarray([], dtype=float)
    e = edges_nm if edges_nm is not None else edges_from_mids_geometric(bin_mid_nm)
    dlog = delta_log10_from_edges(e)
    return np.asarray(dndlogdp, float) * dlog

def dndlog_from_counts(counts: np.ndarray, *, edges_nm: np.ndarray) -> np.ndarray:
    """
    dN/dlogDp = counts / Δlog10(Dp) using provided edges.
    If edges invalid, returns [].
    """
    dlog = delta_log10_from_edges(edges_nm)
    if dlog.size == 0:
        return np.asarray([], dtype=float)
    return np.asarray(counts, float) / dlog


def remap_dndlog_by_edges(old_edges_nm, new_edges_nm, dndlogdp):
    """
    Remap dN/dlog10Dp from old bin edges -> new bin edges by conserving bin counts.
    """
    old_edges_nm = np.asarray(old_edges_nm, float)
    new_edges_nm = np.asarray(new_edges_nm, float)
    dndlogdp = np.asarray(dndlogdp, float)

    assert old_edges_nm.size == new_edges_nm.size == dndlogdp.size + 1, \
        "Edges must be length N+1 and dndlogdp length N."

    dlog_old = np.diff(np.log10(old_edges_nm))
    dlog_new = np.diff(np.log10(new_edges_nm))
    factor = dlog_old / dlog_new

    dndlogdp_new = dndlogdp * factor

    return dndlogdp_new


def select_between(m, e, y, s=None, xmin=None, xmax=None):
    import numpy as np

    m = np.asarray(m, dtype=float)
    e = np.asarray(e, dtype=float)
    y = np.asarray(y, dtype=float)
    s_arr = None if s is None else np.asarray(s, dtype=float)

    N = e.size - 1  # number of bins

    # Build keep mask using edges (bin is fully inside the clip interval)
    if xmin is None:
        left_ok = np.ones(N, dtype=bool)
    else:
        left_ok = (e[:-1] >= float(xmin))

    if xmax is None:
        right_ok = np.ones(N, dtype=bool)
    else:
        right_ok = (e[1:] <= float(xmax))

    keep = left_ok & right_ok  # shape (N,)

    if not np.any(keep):
        empty = np.array([], dtype=float)
        return empty, empty, empty, (None if s is None else empty)

    # Indices of the contiguous kept block (it will be contiguous for a simple interval)
    i0 = int(np.argmax(keep))           # first kept bin
    i1 = int(np.where(keep)[0][-1])     # last kept bin

    m_out = m[keep]
    y_out = y[keep]
    s_out = None if s is None else s_arr[keep]
    e_out = e[i0 : i1 + 2]              # crop edges to bound the kept bins

    return m_out, e_out, y_out, s_out