# dev/forward_kernel.py
from __future__ import annotations

import numpy as np

# NOTE:
# - Kernels here are NUMBER-conserving when multiplied by a spectrum in dN/dlog10D.
# - Given fine-grid edges E_f (log10 D) and instrument edges E_I (log10 D),
#   K[j,i] = length of overlap in log10 D between fine bin i and instrument bin j.
#   Then  N_inst[j] = sum_i K[j,i] * (dN/dlog10D)[i].
#
# Keep it dead simple per Bo's request: no try/except, no optional fallbacks.


def K_tophat(fine_edges_log10D: np.ndarray, inst_edges_log10D: np.ndarray) -> np.ndarray:
    """
    Construct a number-conserving top-hat kernel K in log10(D).

    Parameters
    ----------
    fine_edges_log10D : (N_f+1,) array
        Edges of *fine* grid in log10(Dp [nm]).
    inst_edges_log10D : (N_I+1,) array
        Edges of *instrument* bins in log10(Dp [nm]).

    Returns
    -------
    K : (N_I, N_f) ndarray
        Overlap-length matrix in log10(D). If y is dN/dlog10D on the fine grid,
        instrument-bin numbers are K @ y.
    """
    fine_edges = np.asarray(fine_edges_log10D, float)
    inst_edges = np.asarray(inst_edges_log10D, float)

    n_fine = fine_edges.size - 1
    n_bins = inst_edges.size - 1
    if n_fine <= 0 or n_bins <= 0:
        return np.zeros((0, 0), dtype=float)

    K = np.zeros((n_bins, n_fine), dtype=float)

    def overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
        lo = a0 if a0 > b0 else b0
        hi = a1 if a1 < b1 else b1
        d = hi - lo
        return d if d > 0.0 else 0.0

    for j in range(n_bins):
        jb0, jb1 = inst_edges[j], inst_edges[j + 1]
        for i in range(n_fine):
            ib0, ib1 = fine_edges[i], fine_edges[i + 1]
            K[j, i] = overlap_1d(ib0, ib1, jb0, jb1)

    return K


def K_opc_from_lut(
    fine_edges_nm: np.ndarray,
    Do_edges_nm: np.ndarray,
    ri_src,
    ri_dst,
    lut,
    *,
    response_bins: int = 50,
    eps: float = 1e-12,
):
    """
    Forward kernel for an OPC defined in optical *diameter* (Do) space, using the
    existing σ(D; m) LUT machinery to map instrument Do-bin edges from calibration
    refractive index -> particle refractive index.

    We reuse `convert_do_lut` from optical_diameter_core to map the instrument Do
    edges (defined at ri_src) to *physical Dp* edges for the target particle RI
    (ri_dst). Then we build a plain tophat kernel in log10(Dp) using those mapped
    edges.

    Parameters
    ----------
    fine_edges_nm : (N_f+1,) array
        Edges of your fine grid in *physical* diameter (nm).
    Do_edges_nm : (N_I+1,) array
        Instrument optical-diameter (Do) bin edges (nm), at calibration RI (ri_src).
    ri_src : complex or float
        Complex (n + i k) or real n for the calibration RI used to define Do_edges_nm.
    ri_dst : complex or float
        Complex (n + i k) or real n for the *particle* refractive index to forward model.
    lut : SigmaLUT
        Your prebuilt σ(D; m) LUT object from optical_diameter_core.
    response_bins : int
        Passed through to convert_do_lut (controls internal response discretization).
    eps : float
        Numerical epsilon passed through to convert_do_lut.

    Returns
    -------
    K : (N_I, N_f) ndarray
        Number-conserving kernel in log10(Dp).
    mapped_edges_nm : (N_I+1,) array
        Instrument bin edges mapped into *physical Dp* space for ri_dst.
    """
    from optical_diameter_core import convert_do_lut  # local import to avoid cycles

    fine_edges_nm = np.asarray(fine_edges_nm, float)
    Do_edges_nm = np.asarray(Do_edges_nm, float)

    if fine_edges_nm.ndim != 1 or Do_edges_nm.ndim != 1:
        raise ValueError("edges must be 1D arrays")
    if fine_edges_nm.size < 2 or Do_edges_nm.size < 2:
        return np.zeros((0, 0), float), np.asarray([], float)

    # Map the instrument optical edges to physical Dp edges for the target RI
    mapped_edges_nm = convert_do_lut(
        Do_nm=Do_edges_nm,
        ri_src=ri_src,
        ri_dst=ri_dst,
        lut=lut,
        response_bins=response_bins,
        eps=eps,
    )

    # Build the number-conserving tophat kernel in log space of *physical* diameter
    K = K_tophat(
        fine_edges_log10D=np.log10(fine_edges_nm),
        inst_edges_log10D=np.log10(mapped_edges_nm),
    )
    return K, mapped_edges_nm


def apply_kernel_counts(K: np.ndarray, dndlog_fine: np.ndarray) -> np.ndarray:
    """
    Convenience helper: instrument-bin numbers from a fine-grid spectrum.

    Parameters
    ----------
    K : (N_I, N_f) ndarray
        From K_tophat or K_opc_from_lut.
    dndlog_fine : (N_f,) array
        Spectrum dN/dlog10D on the fine grid associated with K.

    Returns
    -------
    N_inst : (N_I,) array
        Number per instrument bin (same units as integrating dN/dlog10D over Δlog10D).
    """
    return np.asarray(K, float) @ np.asarray(dndlog_fine, float)