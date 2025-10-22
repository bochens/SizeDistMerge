# dev/OpcSizedist.py
# Minimal subclasses for OPC-style size distributions.
# - OpcSizedist: adds wavelength and refractive index (n, k) to AerosolSizedist
# - PopsSizedist: fixed to λ=405 nm and instrument="POPS"
# - UhsasSizedist: fixed to λ=1054 nm and instrument="UHSAS"
#
# No LUT hooks here (kept intentionally minimal). You can wire remapping later.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .AerosolSizedist import (
    AerosolSizedist,
    size_kind,
    data_state,
)

@dataclass(init=False)
class OpcSizedist(AerosolSizedist):
    wavelength_nm: float   # laser wavelength in nm
    ri_n: float            # real part of refractive index
    ri_k: float            # imaginary part of refractive index

    def __init__(
        self,
        *,
        # geometry (either centers OR edges)
        bin_mid: Optional[np.ndarray] = None,
        bin_lower: Optional[np.ndarray] = None,
        bin_upper: Optional[np.ndarray] = None,
        # spectra (either counts OR dN/dlogDp, or both)
        bin_nconc: Optional[np.ndarray] = None,
        dndlogdp: Optional[np.ndarray] = None,
        # OPC params
        wavelength_nm: float,
        ri_n: float,
        ri_k: float,
        # flags
        instrument: Optional[str] = None,
        state: data_state = data_state.NATIVE,
        # tolerances
        tol_rel: float = 1e-8,
        tol_abs: float = 1e-12,
    ) -> None:
        self.wavelength_nm = float(wavelength_nm)
        self.ri_n = float(ri_n)
        self.ri_k = float(ri_k)

        super().__init__(
            bin_mid=bin_mid,
            bin_lower=bin_lower,
            bin_upper=bin_upper,
            bin_nconc=bin_nconc,
            dndlogdp=dndlogdp,
            size_kind=size_kind.OPTICAL,
            data_state=state,
            instrument=instrument or "OPC",
            tol_rel=tol_rel,
            tol_abs=tol_abs,
        )

    def set_refractive_index(self, n: float, k: float) -> None:
        self.ri_n = float(n)
        self.ri_k = float(k)

    def __repr__(self) -> str:
        nbin = len(self.bin_mid) if self.bin_mid is not None else 0
        return (
            f"<OpcSizedist {self.instrument or ''} "
            f"(λ={self.wavelength_nm:.1f} nm, n={self.ri_n:.3f}, k={self.ri_k:.3f}; "
            f"{self.size_kind.value}, {self.data_state.value}) bins={nbin}>"
        )
    

# OpcSizedist children

POPS_WAVELENGTH_NM = 405.0
POPS_DEFAULT_RI_N  = 1.615 # PSL Calibration. 1.615+0.001i Subject to change
POPS_DEFAULT_RI_K  = 0.001 

@dataclass(init=False)
class PopsSizedist(OpcSizedist):
    """POPS-specific subclass: fixed λ=405 nm; instrument defaults to 'POPS'."""
    def __init__(
        self,
        *,
        bin_mid: Optional[np.ndarray] = None,
        bin_lower: Optional[np.ndarray] = None,
        bin_upper: Optional[np.ndarray] = None,
        bin_nconc: Optional[np.ndarray] = None,
        dndlogdp: Optional[np.ndarray] = None,
        ri_n: float,
        ri_k: float,
        instrument: Optional[str] = "POPS",
        state: data_state = data_state.NATIVE,
        tol_rel: float = 1e-8,
        tol_abs: float = 1e-12,
    ) -> None:
        super().__init__(
            bin_mid=bin_mid,
            bin_lower=bin_lower,
            bin_upper=bin_upper,
            bin_nconc=bin_nconc,
            dndlogdp=dndlogdp,
            wavelength_nm=POPS_WAVELENGTH_NM,
            ri_n=ri_n,
            ri_k=ri_k,
            instrument=instrument,
            state=state,
            tol_rel=tol_rel,
            tol_abs=tol_abs,
        )

UHSAS_WAVELENGTH_NM = 1054.0
UHSAS_DEFAULT_RI_N  = 1.52 # PSL Calibration. 1.615+0.001i Subject to change
UHSAS_DEFAULT_RI_K  = 0.0

@dataclass(init=False)
class UhsasSizedist(OpcSizedist):
    """UHSAS-specific subclass: fixed λ=1054 nm; instrument defaults to 'UHSAS'."""
    def __init__(
        self,
        *,
        bin_mid: Optional[np.ndarray] = None,
        bin_lower: Optional[np.ndarray] = None,
        bin_upper: Optional[np.ndarray] = None,
        bin_nconc: Optional[np.ndarray] = None,
        dndlogdp: Optional[np.ndarray] = None,
        ri_n: float,
        ri_k: float,
        instrument: Optional[str] = "UHSAS",
        state: data_state = data_state.NATIVE,
        tol_rel: float = 1e-8,
        tol_abs: float = 1e-12,
    ) -> None:
        super().__init__(
            bin_mid=bin_mid,
            bin_lower=bin_lower,
            bin_upper=bin_upper,
            bin_nconc=bin_nconc,
            dndlogdp=dndlogdp,
            wavelength_nm=UHSAS_WAVELENGTH_NM,
            ri_n=ri_n,
            ri_k=ri_k,
            instrument=instrument,
            state=state,
            tol_rel=tol_rel,
            tol_abs=tol_abs,
        )