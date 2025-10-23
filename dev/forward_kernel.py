#dev/forward_kernel.py

from __future__ import annotations
import numpy as np

def K_tophat(fine_edges_log10D: np.ndarray, inst_edges_log10D: np.ndarray) -> np.ndarray:
    """
    Construct a number-conserving top-hat kernel K.

    Parameters
    ----------
    fine_edges_log10D : array-like
        Edges of fine grid in log10(Dp [nm]).
    inst_edges_log10D : array-like
        Edges of instrument bins in log10(Dp [nm]).

    Returns
    -------
    K : np.ndarray
        Shape (M_bins, N_fine_cells). Each row j integrates f(D)
        over the instrument bin j by summing f[i]*Δlog10D_overlap.
    """
    fine_edges = np.asarray(fine_edges_log10D, float)
    inst_edges = np.asarray(inst_edges_log10D, float)
    n_fine = fine_edges.size - 1
    n_bins = inst_edges.size - 1
    K = np.zeros((n_bins, n_fine), float)

    # Helper for 1D interval overlap
    def overlap_1d(a0, a1, b0, b1):
        lo = max(a0, b0)
        hi = min(a1, b1)
        return max(0.0, hi - lo)

    for j in range(n_bins):
        jb0, jb1 = inst_edges[j], inst_edges[j+1]
        for i in range(n_fine):
            ib0, ib1 = fine_edges[i], fine_edges[i+1]
            K[j, i] = overlap_1d(ib0, ib1, jb0, jb1)
    return K


def 