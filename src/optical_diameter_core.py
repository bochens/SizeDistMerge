from __future__ import annotations
import os
os.environ.setdefault("MIEPYTHON_USE_JIT", "1")

from dataclasses import dataclass
from typing import Union
import time
import numpy as np
import miepython as mie
import zarr
from joblib import Parallel, delayed
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator

# -------------------------------
# Constants and centralized defaults
# -------------------------------

POPS_WAVELENGTH_NM   = 405.0
UHSAS_WAVELENGTH_NM  = 1054.0

DEFAULT_D_RANGE  = (60.0, 5000.0, 360)    # (min, max, num points)
DEFAULT_N_RANGE  = (1.30, 1.80, 0.001)    # (min, max, step)
DEFAULT_K_VALUES = (0.0, 0.001, 0.01, 0.1)
DEFAULT_CHUNKS   = (128, 64, 1)           # (D, n, k) for Zarr v3

RI_UHSAS_SRC=complex(1.52, 0.00)
RI_POPS_SRC =complex(1.615, 0.001)
# -------------------------------
# Geometries
# -------------------------------

@dataclass(frozen=True)
class POPSGeom:
    ring_theta_min_deg: float = 38.0
    ring_theta_max_deg: float = 143.0
    ring_step_deg:      float = 1.0
    mirror_diameter_mm: float = 25.0
    distance_to_mirror_mm: float = 14.3
    pmt_aperture_d_mm:  float = 5.0
    pmt_center_deg:     float = 90.0


@dataclass(frozen=True)
class UHSASGeom:
    # Angular bands you asked for
    big_theta_min_deg: float = 33.0
    big_theta_max_deg: float = 147.0
    small_theta_min_deg: float = 75.2
    small_theta_max_deg: float = 104.8
    ring_step_deg:      float = 1.0

    # Plane distance from interaction region (manual: 8 mm)
    aperture_distance_mm: float = 8.0

    # Effective disk diameters are set by these half-angles via D = 2 L tan(angle)
    big_outer_halfangle_deg:   float = 57.0     # ≈ manual ±57°
    inner_stop_halfangle_deg:  float = 14.8     # manual ±14.8°

# -------------------------------
# Fast helpers (Numba optional)
# -------------------------------
try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

if _HAVE_NUMBA:
    @njit(cache=True, fastmath=True)
    def _trapz_numba(y, x):
        s = 0.0
        for i in range(x.size - 1):
            dx = x[i+1] - x[i]
            s += 0.5 * (y[i] + y[i+1]) * dx
        return s

    @njit(cache=True, fastmath=True)
    def _trapz_weighted_numba(y, w, x):
        s = 0.0
        for i in range(x.size - 1):
            dx = x[i+1] - x[i]
            s += 0.5 * (y[i]*w[i] + y[i+1]*w[i+1]) * dx
        return s
else:
    _trapz_numba = None
    _trapz_weighted_numba = None

# -------------------------------
# Geometry caches
# -------------------------------

class _POPSCache:
    __slots__ = ("mu", "dphi", "theta_ring_rad", "i_pmt", "omega_pmt", "ring_idx")
    def __init__(self, mu, dphi, theta_ring_rad, i_pmt, omega_pmt):
        self.mu = mu                                # cos(θ) for ring + PMT (last entry)
        self.dphi = dphi                            # Δφ_disk(θ) for ring points
        self.theta_ring_rad = theta_ring_rad        # θ grid for ring (radians)
        self.i_pmt = i_pmt                          # index of PMT (90°) in mu
        self.omega_pmt = omega_pmt                  # Ω_PMT
        self.ring_idx = np.arange(theta_ring_rad.size, dtype=int)  # select ring in (ring+PMT)

def pops_geometry_cache(geom: POPSGeom) -> _POPSCache:
    # --- your POPS geometry math kept exactly the same ---
    ring_deg = np.arange(geom.ring_theta_min_deg, geom.ring_theta_max_deg,
                         geom.ring_step_deg, dtype=float)
    theta_eval_deg = np.r_[ring_deg, geom.pmt_center_deg]
    mu = np.cos(np.deg2rad(theta_eval_deg))

    R = 0.5 * geom.mirror_diameter_mm
    L = float(geom.distance_to_mirror_mm)
    x_span = np.linspace(-R, R, ring_deg.size)
    chord_norm = 2.0 * np.sqrt(np.maximum(0.0, R**2 - x_span**2)) / L
    dphi = 2.0 * np.arcsin(np.clip(0.5 * chord_norm, 0.0, 1.0))

    theta_ring_rad = np.deg2rad(ring_deg)
    i_pmt = ring_deg.size
    omega_pmt = np.pi * (0.5 * geom.pmt_aperture_d_mm)**2 / (L**2)
    return _POPSCache(mu, dphi, theta_ring_rad, i_pmt, omega_pmt)


class _UHSASCache:
    __slots__ = ("th_big","mu_big","dphi_big","th_small","mu_small","dphi_small")
    def __init__(self, th_big, mu_big, dphi_big, th_small, mu_small, dphi_small):
        self.th_big   = th_big
        self.mu_big   = mu_big
        self.dphi_big = dphi_big
        self.th_small   = th_small
        self.mu_small   = mu_small
        self.dphi_small = dphi_small

def _disk_dphi_from_D_at_L(d_mm: float, L_mm: float, n_pts: int) -> np.ndarray:
    """
    POPS-style azimuth width Δφ(θ) made from a circular disk of diameter d at distance L.
    Same construction as POPS: sample chord along the disk and map 1:1 onto θ-grid points.
    Note: Δφ from this construction depends only on disk geometry, not explicitly on θ;
    we keep the same indexing pattern for consistency with POPS.
    """
    R = 0.5 * float(d_mm)
    L = float(L_mm)
    x = np.linspace(-R, R, int(n_pts))
    chord_over_L = 2.0 * np.sqrt(np.maximum(0.0, R*R - x*x)) / L
    return 2.0 * np.arcsin(np.clip(0.5 * chord_over_L, 0.0, 1.0))

def uhsas_geometry_cache(geom: UHSASGeom) -> _UHSASCache:
    # θ grids
    big_deg   = np.arange(geom.big_theta_min_deg,   geom.big_theta_max_deg,   geom.ring_step_deg, dtype=float)
    small_deg = np.arange(geom.small_theta_min_deg, geom.small_theta_max_deg, geom.ring_step_deg, dtype=float)
    th_big, th_small = np.deg2rad(big_deg), np.deg2rad(small_deg)
    mu_big, mu_small = np.cos(th_big), np.cos(th_small)

    # Effective disk diameters from the given half-angles at the same plane distance L
    L = float(geom.aperture_distance_mm)
    D_big_mm   = 2.0 * L * np.tan(np.deg2rad(geom.big_outer_halfangle_deg))
    D_small_mm = 2.0 * L * np.tan(np.deg2rad(geom.inner_stop_halfangle_deg))

    # POPS-style Δφ arrays (same indexing pattern as POPS)
    dphi_big   = _disk_dphi_from_D_at_L(D_big_mm,   L, big_deg.size)
    dphi_small = _disk_dphi_from_D_at_L(D_small_mm, L, small_deg.size)

    return _UHSASCache(th_big, mu_big, dphi_big, th_small, mu_small, dphi_small)

# -------------------------------
# σ_col kernels (perpendicular pol.), cached
# -------------------------------

def pops_csca(
    D_nm,
    m_particle,
    wavelength_nm: float,
    *,
    geom: POPSGeom,
    _cache: _POPSCache | None = None,
):
    """
    POPS σ_col(D, m) [µm²], perpendicular.
    σ = ∫ (dσ/dΩ)(θ)·Δφ_disk(θ) dθ  +  (dσ/dΩ)(90°)·Ω_PMT
    Geometry (Δφ, μ, θ-grid) pulled from cache.
    """
    D_nm = np.atleast_1d(D_nm).astype(float)
    a_um = 0.5 * D_nm * 1e-3
    c = _cache or pops_geometry_cache(geom)

    out = np.empty_like(D_nm, float)
    for i, D in enumerate(D_nm):
        xsize = 2.0 * np.pi * (0.5 * D / wavelength_nm)
        PM = mie.phase_matrix(m_particle, xsize, c.mu, norm="qsca")
        P11, P12 = PM[0,0,:], PM[0,1,:]
        Pdet = 0.5*(P11 - P12)  # perpendicular
        dcs = np.pi * (a_um[i]**2) * Pdet

        if _HAVE_NUMBA:
            mirror_term = _trapz_weighted_numba(dcs[c.ring_idx], c.dphi, c.theta_ring_rad)  # type: ignore[arg-type]
            pmt_term    = dcs[c.i_pmt] * c.omega_pmt
        else:
            mirror_term = np.trapz(dcs[c.ring_idx] * c.dphi, c.theta_ring_rad)
            pmt_term    = dcs[c.i_pmt] * c.omega_pmt

        out[i] = mirror_term + pmt_term
    return out


def pops_csca_parallel(
    D_nm,
    m_particle,
    wavelength_nm: float,
    *,
    geom: POPSGeom,
    _cache: _POPSCache | None = None,
    n_jobs: int = -1,
    backend: str = "threads",
):
    D_nm = np.atleast_1d(D_nm).astype(float)
    c = _cache or pops_geometry_cache(geom)
    def _one(d):
        return pops_csca([d], m_particle, wavelength_nm, geom=geom, _cache=c)[0]
    vals = Parallel(n_jobs=n_jobs, prefer=backend)(delayed(_one)(float(d)) for d in D_nm)
    return np.asarray(vals, dtype=float)


def uhsas_csca(
    D_nm,
    m_particle,
    wavelength_nm: float,
    *,
    geom: UHSASGeom,
    _cache: _UHSASCache | None = None,
):
    """
    UHSAS σ_col(D, m) [µm²], perpendicular.
    POPS-style azimuth for BOTH rings (BIG 33–147°, SMALL 75.2–104.8°),
    then σ = ∫_BIG (dσ/dΩ)·Δφ_big(θ) dθ  −  ∫_SMALL (dσ/dΩ)·Δφ_small(θ) dθ
    Geometry (θ, μ, Δφ) pulled from cache.
    """
    D_nm = np.atleast_1d(D_nm).astype(float)
    a_um = 0.5 * D_nm * 1e-3
    c = _cache or uhsas_geometry_cache(geom)

    out = np.empty_like(D_nm, float)
    for i, D in enumerate(D_nm):
        xsize = 2.0 * np.pi * (0.5 * D / wavelength_nm)

        # BIG ring
        PMb = mie.phase_matrix(m_particle, xsize, c.mu_big, norm="qsca")
        Pdet_b = 0.5*(PMb[0,0,:] - PMb[0,1,:])
        dcs_b = np.pi * (a_um[i]**2) * Pdet_b
        if _HAVE_NUMBA:
            sig_big = _trapz_weighted_numba(dcs_b, c.dphi_big, c.th_big)  # type: ignore[arg-type]
        else:
            sig_big = np.trapz(dcs_b * c.dphi_big, c.th_big)

        # SMALL exclusion ring
        PMs = mie.phase_matrix(m_particle, xsize, c.mu_small, norm="qsca")
        Pdet_s = 0.5*(PMs[0,0,:] - PMs[0,1,:])
        dcs_s = np.pi * (a_um[i]**2) * Pdet_s
        if _HAVE_NUMBA:
            sig_small = _trapz_weighted_numba(dcs_s, c.dphi_small, c.th_small)  # type: ignore[arg-type]
        else:
            sig_small = np.trapz(dcs_s * c.dphi_small, c.th_small)

        out[i] = sig_big - sig_small
    return out


def uhsas_csca_parallel(
    D_nm,
    m_particle,
    wavelength_nm: float,
    *,
    geom: UHSASGeom,
    _cache: _UHSASCache | None = None,
    n_jobs: int = -1,
    backend: str = "threads",
):
    D_nm = np.atleast_1d(D_nm).astype(float)
    c = _cache or uhsas_geometry_cache(geom)
    def _one(d):
        return uhsas_csca([d], m_particle, wavelength_nm, geom=geom, _cache=c)[0]
    vals = Parallel(n_jobs=n_jobs, prefer=backend)(delayed(_one)(float(d)) for d in D_nm)
    return np.asarray(vals, dtype=float)

# -------------------------------
# LUT builder: reuse a single geometry cache for speed
# -------------------------------

def build_sigma_lut(
    zpath: str,
    kernel: str,
    wavelength_nm: float,
    geom,
    *,
    D_range = DEFAULT_D_RANGE,
    n_range = DEFAULT_N_RANGE,
    k_values = DEFAULT_K_VALUES,
    chunks = DEFAULT_CHUNKS,
    jobs_per_k: int = -1,
    parallel_backend: str = "threads",
):
    kern = kernel.lower()
    if kern not in ("pops", "uhsas"):
        raise ValueError("kernel must be 'pops' or 'uhsas'.")

    D_min, D_max, D_pts   = D_range
    n_min, n_max, n_step  = n_range

    Dg = np.geomspace(float(D_min), float(D_max), int(D_pts)).astype(float)
    ng = np.arange(float(n_min), float(n_max) + 1e-12, float(n_step), dtype=float)
    kg = np.asarray(k_values, dtype=float)

    root = zarr.open(zpath, mode="w")
    coords = root.create_group("coords")
    coords.create_array("D_nm", data=Dg)
    coords.create_array("n",    data=ng)
    coords.create_array("k",    data=kg)

    SIG = root.create_array(
        "sigma_col",
        shape=(Dg.size, ng.size, kg.size),
        dtype="f4",
        chunks=chunks,
    )

    # ---- build geometry once ----
    if kern == "pops":
        cache = pops_geometry_cache(geom)
        def _curve_for_n(n_val, k_val):
            m = complex(float(n_val), float(k_val))
            return pops_csca(Dg, m, wavelength_nm, geom=geom, _cache=cache).astype(np.float32)
        kernel_name = "POPS"
    else:
        cache = uhsas_geometry_cache(geom)
        def _curve_for_n(n_val, k_val):
            m = complex(float(n_val), float(k_val))
            return uhsas_csca(Dg, m, wavelength_nm, geom=geom, _cache=cache).astype(np.float32)
        kernel_name = "UHSAS"

    block_n = chunks[1]
    total_k = kg.size
    for ik, k in enumerate(kg):
        t_k = time.perf_counter()
        print(f"[{kernel_name}] [k {ik+1}/{total_k}] k = {k:g}", flush=True)
        j0 = 0
        while j0 < ng.size:
            t_blk = time.perf_counter()
            j1 = min(j0 + block_n, ng.size)
            cols = Parallel(n_jobs=jobs_per_k, prefer=parallel_backend, verbose=0)(
                delayed(_curve_for_n)(float(nv), float(k)) for nv in ng[j0:j1]
            )
            SIG[:, j0:j1, ik] = np.stack(cols, axis=1)
            print(f"  wrote n[{j0}:{j1}) in {time.perf_counter()-t_blk:.2f}s", flush=True)
            j0 = j1
        print(f"done k={k:g} (elapsed {time.perf_counter()-t_k:.2f}s)", flush=True)

    # metadata (UHSAS: include effective disk diameters)
    attrs = {
        "description": f"{kernel_name} collected scattering cross-section LUT (qsca norm, perpendicular pol.)",
        "units_sigma_col": "um^2",
        "D_range_nm": [float(D_min), float(D_max)],
        "n_range": [float(n_min), float(n_max), float(n_step)],
        "k_values": kg.tolist(),
        "wavelength_nm": float(wavelength_nm),
        "polarization": "perpendicular",
        "kernel": kernel_name,
    }
    if kern == "pops":
        attrs.update({
            "ring_theta_min_deg": float(geom.ring_theta_min_deg),
            "ring_theta_max_deg": float(geom.ring_theta_max_deg),
            "ring_step_deg": float(geom.ring_step_deg),
            "mirror_diameter_mm": float(geom.mirror_diameter_mm),
            "distance_to_mirror_mm": float(geom.distance_to_mirror_mm),
            "pmt_aperture_d_mm": float(geom.pmt_aperture_d_mm),
            "pmt_center_deg": float(geom.pmt_center_deg),
        })
    else:
        L = float(geom.aperture_distance_mm)
        eff_big_d_mm   = 2.0 * L * np.tan(np.deg2rad(geom.big_outer_halfangle_deg))
        eff_small_d_mm = 2.0 * L * np.tan(np.deg2rad(geom.inner_stop_halfangle_deg))
        attrs.update({
            "big_theta_min_deg": float(geom.big_theta_min_deg),
            "big_theta_max_deg": float(geom.big_theta_max_deg),
            "small_theta_min_deg": float(geom.small_theta_min_deg),
            "small_theta_max_deg": float(geom.small_theta_max_deg),
            "ring_step_deg": float(geom.ring_step_deg),
            "aperture_distance_mm": L,
            "big_outer_halfangle_deg": float(geom.big_outer_halfangle_deg),
            "inner_stop_halfangle_deg": float(geom.inner_stop_halfangle_deg),
            "eff_big_disk_d_mm": float(eff_big_d_mm),
            "eff_small_disk_d_mm": float(eff_small_d_mm),
        })
    root.attrs.update(attrs)

    return dict(zpath=zpath, D_grid_nm=Dg, n_grid=ng, k_grid=kg)


def build_pops_sigma_lut(
    zpath: str, geom: POPSGeom, *,
    D_range=DEFAULT_D_RANGE, n_range=DEFAULT_N_RANGE, k_values=DEFAULT_K_VALUES,
    chunks=DEFAULT_CHUNKS, jobs_per_k=-1, parallel_backend="threads",
    wavelength_nm: float = POPS_WAVELENGTH_NM
):
    return build_sigma_lut(zpath, "pops", wavelength_nm, geom,
                           D_range=D_range, n_range=n_range, k_values=k_values,
                           chunks=chunks, jobs_per_k=jobs_per_k, parallel_backend=parallel_backend)


def build_uhsas_sigma_lut(
    zpath: str, geom: UHSASGeom, *,
    D_range=DEFAULT_D_RANGE, n_range=DEFAULT_N_RANGE, k_values=DEFAULT_K_VALUES,
    chunks=DEFAULT_CHUNKS, jobs_per_k=-1, parallel_backend="threads",
    wavelength_nm: float = UHSAS_WAVELENGTH_NM
):
    return build_sigma_lut(zpath, "uhsas", wavelength_nm, geom,
                           D_range=D_range, n_range=n_range, k_values=k_values,
                           chunks=chunks, jobs_per_k=jobs_per_k, parallel_backend=parallel_backend)

# -------------------------------
# Trilinear query (generic + RAM class)
# -------------------------------

def sigma_query_zarr(zpath: str, D_nm: float, n: float, k: float) -> float:
    """
    Trilinear interpolation via SciPy RegularGridInterpolator (values clamped to grid).
    """
    z = zarr.open(zpath, mode="r")
    Dg = z["coords/D_nm"][:].astype(float)
    ng = z["coords/n"][:].astype(float)
    kg = z["coords/k"][:].astype(float)
    SIG = z["sigma_col"][:].astype(float)

    interp = RegularGridInterpolator((Dg, ng, kg), SIG, method="linear",
                                     bounds_error=False, fill_value=None)

    Dq = float(np.clip(D_nm, Dg[0], Dg[-1]))
    nq = float(np.clip(n,    ng[0], ng[-1]))
    kq = float(np.clip(k,    kg[0], kg[-1]))
    return float(interp([[Dq, nq, kq]])[0])


class SigmaLUT:
    """Load σ_col(D,n,k) into RAM once. Fast trilinear queries."""
    def __init__(self, zpath: str):
        z = zarr.open(zpath, mode="r")
        self.zpath = zpath
        self.Dg  = z["coords/D_nm"][:].astype(float)
        self.ng  = z["coords/n"][:].astype(float)
        self.kg  = z["coords/k"][:].astype(float)
        self.SIG = z["sigma_col"][:].astype(float)
        self.kernel = str(z.attrs.get("kernel", ""))
        self.wavelength_nm = float(z.attrs.get("wavelength_nm", np.nan))
        self.polarization  = str(z.attrs.get("polarization", ""))

        self._interp = RegularGridInterpolator(
            (self.Dg, self.ng, self.kg), self.SIG,
            method="linear", bounds_error=False, fill_value=None
        )

    def _tri_single(self, D_nm: float, n: float, k: float) -> float:
        Dq = float(np.clip(D_nm, self.Dg[0], self.Dg[-1]))
        nq = float(np.clip(n,    self.ng[0], self.ng[-1]))
        kq = float(np.clip(k,    self.kg[0], self.kg[-1]))
        return float(self._interp([[Dq, nq, kq]])[0])

    def sigma_curve(self, D_vec_nm, n: float, k: float) -> np.ndarray:
        D_vec_nm = np.asarray(D_vec_nm, float)
        Dq = np.clip(D_vec_nm, self.Dg[0], self.Dg[-1])
        nq = float(np.clip(n,   self.ng[0], self.ng[-1]))
        kq = float(np.clip(k,   self.kg[0], self.kg[-1]))
        pts = np.column_stack([Dq, np.full_like(Dq, nq), np.full_like(Dq, kq)])
        return self._interp(pts).astype(float)

# -------------------------------
# Monotone σ(D) + inverse
# -------------------------------

def make_monotone_sigma_interpolator(
    D_nm, sigma_col, *, increasing=True, sample_weight=None,
    response_bins=None,
):
    """
    Build monotone σ(D) and its inverse D(σ) with optional binning + isotonic regression.
    """
    D = np.asarray(D_nm, float)
    S = np.asarray(sigma_col, float)
    if np.any(D <= 0) or np.any(S <= 0):
        raise ValueError("Need D>0 and σ>0 (uses log–log).")

    order = np.argsort(D)
    x = np.log(D[order])
    y = np.log(S[order])

    if response_bins is not None and response_bins > 1:
        response_bins = int(response_bins)
        edges = np.linspace(x.min(), x.max(), response_bins + 1)
        xb, yb, wb = [], [], []
        for i in range(response_bins):
            if i < response_bins-1:
                mask = (x >= edges[i]) & (x < edges[i+1])
            else:
                mask = (x >= edges[i]) & (x <= edges[i+1])
            if not np.any(mask):
                continue
            xb.append(x[mask].mean())
            yb.append(np.median(y[mask]))
            wb.append(int(mask.sum()))
        if len(xb) < 2:
            raise ValueError("Too few non-empty bins.")
        x = np.asarray(xb); y = np.asarray(yb)
        sample_weight = np.asarray(wb, float)

    iso = IsotonicRegression(increasing=bool(increasing), out_of_bounds="clip")
    if sample_weight is None:
        yhat = iso.fit_transform(x, y)
    else:
        iso.fit(x, y, sample_weight=sample_weight)
        yhat = iso.predict(x)

    f_ll = PchipInterpolator(x, yhat, extrapolate=False)

    vals, inv, counts = np.unique(yhat, return_inverse=True, return_counts=True)
    if vals.size < 2:
        raise ValueError("Isotonic fit collapsed to a constant.")
    x_avg = np.bincount(inv, weights=x) / counts
    finv_ll = PchipInterpolator(vals, x_avg, extrapolate=False)

    def f_sigma(Dq):
        Dq = np.asarray(Dq, float)
        return np.exp(f_ll(np.log(Dq)))

    def g_diam(sig):
        sig = np.asarray(sig, float)
        return np.exp(finv_ll(np.log(sig)))

    return f_sigma, g_diam


# -------------------------------
# Size-distribution remap (change refractive index)
# -------------------------------

def convert_do_lut(
    Do_nm,
    ri_src, ri_dst, lut: "SigmaLUT",
    *, response_bins=50, eps=1e-12
):
    """
    Map OPC bins from ri_src -> ri_dst via σ(D;m). Conserve number exactly:
        N_i' = N_i  for each original bin i
    Then compute dN/dlog10D' on the new edges.

    Args
    ----
    Do_nm : optical diameter in nm
    ri_src: refractive index used for Do_nm
    ri_dst: destination refractive index associated with the return
    lut: LUT object

    Returns:
        Dp_centers_nm, dNdlog10Dp_new, Dp_edges_nm
    """

    # build monotone σ maps on LUT grid
    Dg = np.asarray(lut.Dg, float)
    ns, ks = float(np.real(ri_src)), float(np.imag(ri_src))
    nd, kd = float(np.real(ri_dst)), float(np.imag(ri_dst))
    sigma_src = lut.sigma_curve(Dg, ns, ks)
    sigma_dst = lut.sigma_curve(Dg, nd, kd)

    f_src_sigma, _    = make_monotone_sigma_interpolator(Dg, sigma_src, response_bins=response_bins, increasing=True)
    _, D_of_sigma_dst = make_monotone_sigma_interpolator(Dg, sigma_dst, response_bins=response_bins, increasing=True)

    # map edges: D -> σ (at ri_src)
    sigma_edges = f_src_sigma(Do_nm)

    # invert: σ -> D' (at m_dst)
    Do_nm_new = D_of_sigma_dst(sigma_edges)

    return Do_nm_new
    

# -------------------------------
# Public API
# -------------------------------
__all__ = [
    # Geom + caches
    "POPSGeom", "UHSASGeom",
    "pops_geometry_cache", "uhsas_geometry_cache",
    # Kernels
    "pops_csca", "pops_csca_parallel",
    "uhsas_csca", "uhsas_csca_parallel",
    # LUT build
    "build_sigma_lut",
    "build_pops_sigma_lut",
    "build_uhsas_sigma_lut",
    # Query
    "SigmaLUT",
    "sigma_query_zarr",
    # Monotone + inverse
    "make_monotone_sigma_interpolator",
    # Remap
    "remap_bins_lut",
    # Constants
    "POPS_WAVELENGTH_NM",
    "UHSAS_WAVELENGTH_NM",
]