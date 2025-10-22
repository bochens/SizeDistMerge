# dev/AerosolSizedist.py
# Simple, readable, and minimally branched.
# - Exactly two tiny math helpers (_infer_edges_from_centers, _delta_log)
# - One public pair of converters (dsdlog_from_dndlog, dvdlog_from_dndlog)
# - Constructor does minimal branching, then ONE final check does all validation
# - Two factory methods (Option B) for clarity
# - Thin object wrappers to compute/store dS/dlogDp and dV/dlogDp

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np


# -------------------- enums --------------------

class size_kind(str, Enum):
    OPTICAL = "optical"
    MOBILITY = "mobility"
    AERODYNAMIC = "aerodynamic"
    VOLUME = "volume equivalent"


class data_state(str, Enum):
    NATIVE = "native"
    CONVERTED = "converted"
    MERGED = "merged"
    SYNTHETIC = "synthetic"


# -------------------- tiny math helpers --------------------

def _infer_edges_from_centers(bin_mid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Infer bin edges from geometric centers; extrapolate end edges by geometric ratio."""
    c = np.asarray(bin_mid, float)
    if c.ndim != 1 or c.size < 2:
        raise ValueError("bin_mid must be 1-D with length >= 2.")
    inner = np.sqrt(c[:-1] * c[1:])
    r0, rN = c[1] / c[0], c[-1] / c[-2]
    lower0 = c[0] / np.sqrt(r0)
    upperN = c[-1] * np.sqrt(rN)
    edges = np.concatenate([[lower0], inner, [upperN]])
    return edges[:-1], edges[1:]


def _delta_log(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """ΔlogDp = ln(upper) - ln(lower) (any D units; ratio cancels)."""
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    return np.log(upper) - np.log(lower)


# -------------------- PUBLIC converters (not tied to class) --------------------

def dsdlog_from_dndlog(bin_mid: np.ndarray, dndlogdp: np.ndarray) -> np.ndarray:
    """dS/dlogDp (µm²-weighted) from dN/dlogDp and bin_mid in nm."""
    D_um = np.asarray(bin_mid, float) * 1e-3
    S = np.pi * (D_um ** 2)
    return np.asarray(dndlogdp, float) * S


def dvdlog_from_dndlog(bin_mid: np.ndarray, dndlogdp: np.ndarray) -> np.ndarray:
    """dV/dlogDp (µm³-weighted) from dN/dlogDp and bin_mid in nm."""
    D_um = np.asarray(bin_mid, float) * 1e-3
    V = (np.pi / 6.0) * (D_um ** 3)
    return np.asarray(dndlogdp, float) * V


# -------------------- class --------------------

@dataclass(init=False)
class AerosolSizedist:
    # Bin geometry (nm)
    bin_lower: np.ndarray
    bin_upper: np.ndarray
    bin_mid:   np.ndarray

    # Spectra
    bin_nconc: np.ndarray                  # counts per bin (dN)
    dndlogdp:  np.ndarray                  # number distribution (dN/dlogDp)
    dsdlogdp:  Optional[np.ndarray] = None # optional
    dvdlogdp:  Optional[np.ndarray] = None # optional

    # Flags
    size_kind:  size_kind  = size_kind.VOLUME
    data_state: data_state = data_state.NATIVE
    instrument: Optional[str] = None

    # --------- constructor (minimal branching; normalize only) ---------
    # Normalize geometry:
    #   - If bin_mid only: infer edges from centers.
    #   - If bin_mid + both edges: take provided edges now; consistency is checked later.
    #   - If edges only: derive centers as sqrt(lower*upper).
    # Normalize spectra:
    #   - If counts only: compute dN/dlogDp via division by Δlog.
    #   - If dN/dlogDp only: compute counts via multiplication by Δlog.
    #   - If both: store both; consistency is checked later.
    # Then assign and run a single centralized validator.
    def __init__(self, *,
        # geometry: either (bin_mid) OR (bin_lower & bin_upper)
        bin_mid: Optional[np.ndarray] = None,
        bin_lower: Optional[np.ndarray] = None,
        bin_upper: Optional[np.ndarray] = None,
        # spectra: either (bin_nconc) OR (dndlogdp) OR both
        bin_nconc: Optional[np.ndarray] = None,
        dndlogdp: Optional[np.ndarray] = None,
        # flags
        size_kind: size_kind = size_kind.VOLUME,
        data_state: data_state = data_state.NATIVE,
        instrument: Optional[str] = None,
        # tolerances (used in the single final check)
        tol_rel: float = 1e-8,
        tol_abs: float = 1e-12,
    ) -> None:
        import warnings

        # --- geometry normalization (no validation here) ---
        warn_msg = None
        if bin_mid is not None:                                         # centers provided
            mid = np.asarray(bin_mid, float)
            if (bin_lower is None) and (bin_upper is None):             # no edges -> infer edges
                lower, upper = _infer_edges_from_centers(mid)
            elif (bin_lower is None) != (bin_upper is None):            # one edge -> override by inference
                warn_msg = (
                    "Received only one of bin_lower/bin_upper with bin_mid; "
                    "inferring BOTH edges from centers and overriding the provided one."
                )
                lower, upper = _infer_edges_from_centers(mid)
            else:                                                       # both edges provided -> assign
                lower = np.asarray(bin_lower, float)
                upper = np.asarray(bin_upper, float)
        else:                                                           # centers not provided -> calculate mid
            lower = None if bin_lower is None else np.asarray(bin_lower, float)
            upper = None if bin_upper is None else np.asarray(bin_upper, float)
            mid   = None if (lower is None or upper is None) else np.sqrt(lower * upper)

        # --- spectra normalization (no validation here) ---
        dlog = _delta_log(lower, upper) if (lower is not None and upper is not None) else None

        if (bin_nconc is None) and (dndlogdp is None):                  # both nconc and dndlopgdp absent
            counts = None
            dnd    = None
        elif dndlogdp is None:                                          # counts only
            counts = np.asarray(bin_nconc, float)
            dnd    = None if dlog is None else counts / dlog
        elif bin_nconc is None:                                         # dN/dlogDp only
            dnd    = np.asarray(dndlogdp, float)
            counts = None if dlog is None else dnd * dlog
        else:                                                           # both provided
            counts = np.asarray(bin_nconc, float)
            dnd    = np.asarray(dndlogdp, float)

        # --- assign (post-normalization, pre-validation) ---
        self.bin_lower = lower
        self.bin_upper = upper
        self.bin_mid   = mid
        self.bin_nconc = counts
        self.dndlogdp  = dnd
        self.size_kind = size_kind
        self.data_state = data_state
        self.instrument = instrument

        # --- single final validation; then populate surface/volume spectra ---
        self._final_check(tol_rel=tol_rel, tol_abs=tol_abs, warn_msg=warn_msg)
        self.set_dsdlogdp()
        self.set_dvdlogdp()

    # -------------------- single, centralized validator --------------------
    # Validates everything exactly once:
    #   - geometry presence/shape/monotonicity
    #   - center/edge geometric-mean consistency
    #   - ΔlogDp > 0
    #   - spectra presence/shape
    #   - counts ≈ dN/dlogDp * ΔlogDp (if both supplied)
    def _final_check(self, *, tol_rel: float, tol_abs: float, warn_msg: Optional[str]) -> None:
        import warnings

        if warn_msg:
            warnings.warn(warn_msg)

        # geometry presence
        if (self.bin_lower is None) or (self.bin_upper is None) or (self.bin_mid is None):
            raise ValueError("Supply bin_mid, or both bin_lower and bin_upper (geometry incomplete).")

        # geometry shapes
        n_lower, n_upper, n_mid = map(len, (self.bin_lower, self.bin_upper, self.bin_mid))
        if not (n_lower == n_upper == n_mid):
            raise ValueError("bin_lower, bin_upper, bin_mid must have the SAME length.")

        # geometry monotonicity and center/edge consistency
        if not np.all(self.bin_upper > self.bin_lower):
            raise ValueError("Each bin must satisfy bin_upper[i] > bin_lower[i].")
        mid_from_edges = np.sqrt(self.bin_lower * self.bin_upper)
        if not np.allclose(mid_from_edges, self.bin_mid, rtol=tol_rel, atol=tol_abs):
            raise ValueError("bin_mid is not consistent with bin_lower/bin_upper (geometric mean mismatch).")

        # Δlog must be strictly positive
        dlog = _delta_log(self.bin_lower, self.bin_upper)
        if not np.all(dlog > 0):
            raise ValueError("Non-positive ΔlogDp detected (check bin edge ordering/values).")

        # spectra presence
        if (self.bin_nconc is None) and (self.dndlogdp is None):
            raise ValueError("Provide bin_nconc OR dndlogdp (spectra missing).")

        # spectra shapes
        nb = n_mid
        for name, arr in (("bin_nconc", self.bin_nconc), ("dndlogdp", self.dndlogdp),
                          ("dsdlogdp", self.dsdlogdp), ("dvdlogdp", self.dvdlogdp)):
            if arr is not None and len(arr) != nb:
                raise ValueError(f"{name} must have length {nb}.")

        # spectra consistency when both provided
        if (self.bin_nconc is not None) and (self.dndlogdp is not None):
            if not np.allclose(self.dndlogdp * dlog, self.bin_nconc, rtol=tol_rel, atol=tol_abs):
                raise ValueError("bin_nconc ≠ dndlogdp * ΔlogDp (mismatch).")
            

    # -------------------- basics --------------------
    def __repr__(self) -> str:
        n = len(self.bin_mid) if self.bin_mid is not None else 0
        return f"<AerosolSizedist {self.instrument or ''} ({self.size_kind.value}, {self.data_state.value}) bins={n}>"

    # -------------------- class wrappers over PUBLIC helpers --------------------
    def dsdlog_from_dndlog(self) -> np.ndarray:
        """Return dS/dlogDp computed from this object's dN/dlogDp."""
        return dsdlog_from_dndlog(self.bin_mid, self.dndlogdp)

    def dvdlog_from_dndlog(self) -> np.ndarray:
        """Return dV/dlogDp computed from this object's dN/dlogDp."""
        return dvdlog_from_dndlog(self.bin_mid, self.dndlogdp)

    def set_dsdlogdp(self) -> None:
        """Compute & store dS/dlogDp in self.dsdlogdp."""
        self.dsdlogdp = self.dsdlog_from_dndlog()

    def set_dvdlogdp(self) -> None:
        """Compute & store dV/dlogDp in self.dvdlogdp."""
        self.dvdlogdp = self.dvdlog_from_dndlog()