"""Reusable aerosol size-distribution helpers."""

from __future__ import annotations
from typing import Optional
import numpy as np

def _as_1d_float(name: str, values, *, allow_nan: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    invalid = np.isinf(arr) if allow_nan else ~np.isfinite(arr)
    if np.any(invalid):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_positive_increasing(name: str, values, *, min_size: int) -> np.ndarray:
    arr = _as_1d_float(name, values)
    if arr.size < min_size:
        return arr
    if np.any(arr <= 0):
        raise ValueError(f"{name} must be > 0")
    if not np.all(np.diff(arr) > 0):
        raise ValueError(f"{name} must be strictly increasing")
    return arr


def _validate_spectrum_lengths(edges_nm: np.ndarray, dndlogdp: np.ndarray) -> None:
    if dndlogdp.ndim != 1:
        raise ValueError("dndlogdp must be a 1D array")
    if dndlogdp.size != edges_nm.size - 1:
        raise ValueError("dndlogdp length must be one less than edges length")


def edges_from_mids_geometric(bin_mid_nm: np.ndarray) -> np.ndarray:
    """Infer geometric bin edges from bin midpoints."""
    c = _validate_positive_increasing("bin_mid_nm", bin_mid_nm, min_size=2)
    if c.size < 2:
        return np.asarray([], dtype=float)
    inner = np.sqrt(c[:-1] * c[1:])
    r0, rN = c[1] / c[0], c[-1] / c[-2]
    lower0 = c[0] / np.sqrt(r0)
    upperN = c[-1] * np.sqrt(rN)
    return np.concatenate([[lower0], inner, [upperN]])


def mids_from_edges(edges_nm: np.ndarray) -> np.ndarray:
    """Return geometric bin midpoints from bin edges."""
    e = _validate_positive_increasing("edges_nm", edges_nm, min_size=2)
    if e.size < 2:
        return np.asarray([], dtype=float)
    return np.sqrt(e[:-1] * e[1:])


def delta_log10_from_edges(edges_nm: np.ndarray) -> np.ndarray:
    """Return log10 bin widths from bin edges."""
    e = _validate_positive_increasing("edges_nm", edges_nm, min_size=2)
    if e.size < 2:
        return np.asarray([], dtype=float)
    return np.log10(e[1:]) - np.log10(e[:-1])


def dsdlog_from_dndlog(bin_mid_nm: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """Convert ``dN/dlogDp`` to ``dS/dlogDp`` using diameter in micrometers."""
    D_um = _as_1d_float("bin_mid_nm", bin_mid_nm) * 1e-3
    values = _as_1d_float("arr", arr, allow_nan=True)
    if values.size != D_um.size:
        raise ValueError("arr length must match bin_mid_nm length")
    return values * (np.pi * D_um**2)


def dvdlog_from_dndlog(bin_mid_nm: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """Convert ``dN/dlogDp`` to ``dV/dlogDp`` using diameter in micrometers."""
    D_um = _as_1d_float("bin_mid_nm", bin_mid_nm) * 1e-3
    values = _as_1d_float("arr", arr, allow_nan=True)
    if values.size != D_um.size:
        raise ValueError("arr length must match bin_mid_nm length")
    return values * ((np.pi / 6.0) * D_um**3)


def counts_from_dndlog(
    dndlogdp: np.ndarray,
    *,
    bin_mid_nm: Optional[np.ndarray] = None,
    edges_nm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert ``dN/dlogDp`` to per-bin counts."""
    if edges_nm is None and bin_mid_nm is None:
        return np.asarray([], dtype=float)
    e = edges_nm if edges_nm is not None else edges_from_mids_geometric(bin_mid_nm)
    dlog = delta_log10_from_edges(e)
    values = _as_1d_float("dndlogdp", dndlogdp, allow_nan=True)
    if values.size != dlog.size:
        raise ValueError("dndlogdp length must match number of bins")
    return values * dlog


def dndlog_from_counts(counts: np.ndarray, *, edges_nm: np.ndarray) -> np.ndarray:
    """Convert per-bin counts to ``dN/dlogDp``."""
    dlog = delta_log10_from_edges(edges_nm)
    if dlog.size == 0:
        return np.asarray([], dtype=float)
    counts = _as_1d_float("counts", counts, allow_nan=True)
    if counts.size != dlog.size:
        raise ValueError("counts length must match number of bins")
    return counts / dlog


def remap_dndlog_by_edges(old_edges_nm, new_edges_nm, dndlogdp):
    """Remap onto same-length edge arrays while conserving counts in each bin."""
    old_edges_nm = _validate_positive_increasing("old_edges_nm", old_edges_nm, min_size=2)
    new_edges_nm = _validate_positive_increasing("new_edges_nm", new_edges_nm, min_size=2)
    dndlogdp = _as_1d_float("dndlogdp", dndlogdp, allow_nan=True)
    _validate_spectrum_lengths(old_edges_nm, dndlogdp)
    if new_edges_nm.size != old_edges_nm.size:
        raise ValueError("remap_dndlog_by_edges requires old and new edge arrays with the same length")

    dlog_old = np.diff(np.log10(old_edges_nm))
    dlog_new = np.diff(np.log10(new_edges_nm))
    return dndlogdp * (dlog_old / dlog_new)


def rebin_dndlog_by_edges_overlap(old_edges_nm, new_edges_nm, dndlogdp, *, min_coverage=0.999):
    """Rebin by log-space bin overlap, conserving counts where bins overlap."""
    old_edges_nm = _validate_positive_increasing("old_edges_nm", old_edges_nm, min_size=2)
    new_edges_nm = _validate_positive_increasing("new_edges_nm", new_edges_nm, min_size=2)
    dndlogdp = _as_1d_float("dndlogdp", dndlogdp, allow_nan=True)
    _validate_spectrum_lengths(old_edges_nm, dndlogdp)
    min_coverage = float(min_coverage)
    if not np.isfinite(min_coverage) or min_coverage < 0.0 or min_coverage > 1.0:
        raise ValueError("min_coverage must be finite and between 0 and 1")

    dlog_old = np.diff(np.log10(old_edges_nm))
    counts_old = dndlogdp * dlog_old

    n_old = len(dndlogdp)
    n_new = len(new_edges_nm) - 1
    counts_new = np.zeros(n_new, float)
    covered_logw = np.zeros(n_new, float)

    i_old = 0
    for j in range(n_new):
        a0 = new_edges_nm[j]
        a1 = new_edges_nm[j + 1]
        while i_old < n_old:
            b0 = old_edges_nm[i_old]
            b1 = old_edges_nm[i_old + 1]

            if b1 <= a0:
                i_old += 1
                continue
            if b0 >= a1:
                break

            lo = max(a0, b0)
            hi = min(a1, b1)
            log_lo = np.log10(lo)
            log_hi = np.log10(hi)

            overlap_logw = log_hi - log_lo
            if overlap_logw > 0:
                frac = overlap_logw / (np.log10(b1) - np.log10(b0))
                frac = max(0.0, min(1.0, frac))
                counts_new[j] += counts_old[i_old] * frac
                covered_logw[j] += overlap_logw

            if b1 <= a1:
                i_old += 1
            else:
                break

    dlog_new = np.diff(np.log10(new_edges_nm))
    dndlogdp_new = counts_new / dlog_new

    coverage_frac = covered_logw / dlog_new
    dndlogdp_new[coverage_frac < min_coverage] = np.nan

    return dndlogdp_new


remap_dndlog_by_edges_any = rebin_dndlog_by_edges_overlap


def select_between(m, e, y, s=None, xmin=None, xmax=None):
    """Select bins whose full edge interval is within ``xmin`` and ``xmax``."""
    m = _as_1d_float("m", m)
    e = _validate_positive_increasing("e", e, min_size=2)
    y = _as_1d_float("y", y, allow_nan=True)
    s_arr = None if s is None else np.asarray(s, dtype=float)

    N = e.size - 1
    if m.size != N or y.size != N:
        raise ValueError("m and y must have length len(e) - 1")
    if s_arr is not None:
        if s_arr.ndim != 1 or s_arr.size != N or np.any(np.isinf(s_arr)):
            raise ValueError("s must be a 1D array with length len(e) - 1 and no infinite values")

    if xmin is None:
        left_ok = np.ones(N, dtype=bool)
    else:
        left_ok = (e[:-1] >= float(xmin))

    if xmax is None:
        right_ok = np.ones(N, dtype=bool)
    else:
        right_ok = (e[1:] <= float(xmax))

    keep = left_ok & right_ok

    if not np.any(keep):
        empty = np.array([], dtype=float)
        return empty, empty, empty, (None if s is None else empty)

    i0 = int(np.argmax(keep))
    i1 = int(np.where(keep)[0][-1])

    m_out = m[keep]
    y_out = y[keep]
    s_out = None if s is None else s_arr[keep]
    e_out = e[i0 : i1 + 2]

    return m_out, e_out, y_out, s_out


__all__ = [
    "edges_from_mids_geometric",
    "mids_from_edges",
    "delta_log10_from_edges",
    "dsdlog_from_dndlog",
    "dvdlog_from_dndlog",
    "counts_from_dndlog",
    "dndlog_from_counts",
    "remap_dndlog_by_edges",
    "rebin_dndlog_by_edges_overlap",
    "remap_dndlog_by_edges_any",
    "select_between",
]
