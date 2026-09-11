from __future__ import annotations
import os
import tempfile
import warnings


def _configure_miepython_jit() -> None:
    os.environ.setdefault("MIEPYTHON_USE_JIT", "1")
    os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.gettempdir())


_configure_miepython_jit()

from dataclasses import dataclass
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

# Old disk-width LUTs must not silently enter new calculations.
OPTICAL_MODEL_VERSION = "solid-angle-polarized-cones-v1"
# -------------------------------
# Geometries
# -------------------------------

@dataclass(frozen=True)
class POPSGeom:
    """POPS mirror cone from Gao et al. (2016), Fig. 1: 38--142 degrees.

    Its circular rim defines a 52 degree cone about the collection axis.
    Mirror curvature is already represented by that measured half-angle;
    the vertex distance and diameter must not be used as a flat-disk opening.
    Mirror-only is the default (as in Liu et al., 2021, Appendix A).
    Optional direct collection requires its own particle-to-aperture distance.
    """
    ring_theta_min_deg: float = 38.0
    ring_theta_max_deg: float = 142.0
    ring_step_deg:      float = 0.25
    mirror_diameter_mm: float = 25.0
    distance_to_mirror_mm: float = 14.3
    pmt_aperture_d_mm:  float = 0.0
    pmt_center_deg:     float = 90.0
    mirror_halfangle_deg: float = 52.0
    pmt_aperture_distance_mm: float | None = None


@dataclass(frozen=True)
class UHSASGeom:
    """One UHSAS collection arm: 14.8--57 degree annular cone.

    Howell et al. (2021), Fig. 1 and Appendix A. Cross-sections are per arm,
    per total incident irradiance, without detector gain. The opposite arm
    is identical for a sphere. Counterpropagating incoherent beams give the
    same integral because this acceptance is symmetric under theta -> pi-theta.
    """
    big_theta_min_deg: float = 33.0
    big_theta_max_deg: float = 147.0
    small_theta_min_deg: float = 75.2
    small_theta_max_deg: float = 104.8
    ring_step_deg:      float = 0.25

    # Plane distance from interaction region (manual: 8 mm)
    aperture_distance_mm: float = 8.0

    # These half-angles define acceptance; distance alone adds no weighting.
    big_outer_halfangle_deg:   float = 57.0     # ≈ manual ±57°
    inner_stop_halfangle_deg:  float = 14.8     # manual ±14.8°

# -------------------------------
# Fast helpers (Numba optional)
# -------------------------------
try:
    from numba import njit
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

else:
    _trapz_numba = None

# -------------------------------
# Geometry caches
# -------------------------------

@dataclass(frozen=True)
class _SideCollectionCache:
    theta_rad: np.ndarray
    mu: np.ndarray
    dphi: np.ndarray
    perp_phi: np.ndarray
    parallel_phi: np.ndarray


@dataclass(frozen=True)
class _POPSCache:
    mirror: _SideCollectionCache
    direct: _SideCollectionCache | None


def _cone_azimuth_weights(theta_rad, halfangle_deg):
    """Azimuth integrals for a circular cone at theta=90, phi=0.

    phi is measured from the central scattering plane; the laser electric
    field is perpendicular to that plane. The cone condition is
    sin(theta)*cos(phi) >= cos(alpha). Its half-width is therefore
    beta = acos(cos(alpha)/sin(theta)), not asin of a sampled disk chord.
    Integrate cos(phi)^2 and sin(phi)^2 for the two polarization components.
    The sin(theta) solid-angle factor is applied separately in the kernel.
    """
    if not np.isfinite(halfangle_deg) or not 0 <= halfangle_deg < 90:
        raise ValueError("collection half-angle must be finite and in [0, 90) degrees")
    theta = np.asarray(theta_rad, dtype=float)
    beta = np.zeros_like(theta)
    if halfangle_deg > 0:
        ca = np.cos(np.deg2rad(halfangle_deg))
        inside = np.sin(theta) > ca
        beta[inside] = np.arccos(np.clip(ca / np.sin(theta[inside]), 0.0, 1.0))
    sine_term = 0.5 * np.sin(2.0 * beta)
    return 2.0 * beta, beta + sine_term, beta - sine_term


def _side_collection_cache(outer_deg, inner_deg, step_deg):
    if not np.isfinite(step_deg) or step_deg <= 0:
        raise ValueError("ring_step_deg must be finite and > 0")
    if not (np.isfinite(outer_deg) and np.isfinite(inner_deg)
            and 0 <= inner_deg < outer_deg < 90):
        raise ValueError("require 0 <= inner half-angle < outer half-angle < 90 degrees")
    # Exact physical boundaries and centre do not change with grid resolution.
    lo, hi = 90.0 - outer_deg, 90.0 + outer_deg
    # A narrow direct aperture still needs enough samples at its curved edges.
    count = max(256, int(np.ceil((hi - lo) / step_deg))) + 1
    deg = np.unique(np.r_[np.linspace(lo, hi, count),
                          90.0 - inner_deg, 90.0, 90.0 + inner_deg])
    th = np.deg2rad(deg)
    outer = _cone_azimuth_weights(th, outer_deg)
    inner = _cone_azimuth_weights(th, inner_deg)
    weights = [np.maximum(a - b, 0.0) for a, b in zip(outer, inner)]
    return _SideCollectionCache(th, np.cos(th), *weights)


def _check_theta_coverage(low, high, halfangle, label):
    if not (np.isfinite(low) and np.isfinite(high)
            and 0 <= low <= 90 - halfangle and 90 + halfangle <= high <= 180):
        raise ValueError(f"{label} theta limits must cover 90 +/- its collection half-angle")


def pops_geometry_cache(geom: POPSGeom) -> _POPSCache:
    _check_theta_coverage(geom.ring_theta_min_deg, geom.ring_theta_max_deg,
                          geom.mirror_halfangle_deg, "POPS mirror")
    mirror = _side_collection_cache(geom.mirror_halfangle_deg, 0.0, geom.ring_step_deg)
    if not np.isfinite(geom.pmt_aperture_d_mm) or geom.pmt_aperture_d_mm < 0:
        raise ValueError("pmt_aperture_d_mm must be finite and >= 0")
    direct = None
    if geom.pmt_aperture_d_mm > 0:
        distance = geom.pmt_aperture_distance_mm
        if distance is None or not np.isfinite(distance) or distance <= 0:
            raise ValueError("Direct POPS collection requires pmt_aperture_distance_mm; "
                             "do not use the mirror distance. Set pmt_aperture_d_mm=0 "
                             "for an explicit mirror-only model.")
        if geom.pmt_center_deg != 90.0:
            raise ValueError("Direct POPS collection currently supports only pmt_center_deg=90")
        alpha = np.rad2deg(np.arctan(0.5 * geom.pmt_aperture_d_mm / distance))
        direct = _side_collection_cache(alpha, 0.0, geom.ring_step_deg)
    return _POPSCache(mirror, direct)


def uhsas_geometry_cache(geom: UHSASGeom) -> _SideCollectionCache:
    _check_theta_coverage(geom.big_theta_min_deg, geom.big_theta_max_deg,
                          geom.big_outer_halfangle_deg, "UHSAS outer")
    _check_theta_coverage(geom.small_theta_min_deg, geom.small_theta_max_deg,
                          geom.inner_stop_halfangle_deg, "UHSAS exclusion")
    return _side_collection_cache(geom.big_outer_halfangle_deg,
                                  geom.inner_stop_halfangle_deg, geom.ring_step_deg)


def _collected_cross_section(D_nm, m_particle, wavelength_nm, cache):
    """One polarized side-collection path, in um^2.

    miepython qsca normalization supplies intensity per steradian, including
    Qsca. P11-P12=|S1|^2 and P11+P12=|S2|^2: no extra factor of one half.
    """
    x = np.pi * D_nm / wavelength_nm
    PM = mie.phase_matrix(m_particle, x, cache.mu, norm="qsca")
    perpendicular = PM[0, 0, :] - PM[0, 1, :]
    parallel = PM[0, 0, :] + PM[0, 1, :]
    phi_integral = perpendicular * cache.perp_phi + parallel * cache.parallel_phi
    integrand = phi_integral * np.sin(cache.theta_rad)
    integral = (_trapz_numba(integrand, cache.theta_rad) if _HAVE_NUMBA
                else np.trapezoid(integrand, cache.theta_rad))
    radius_um = 0.5 * D_nm * 1e-3
    return np.pi * radius_um**2 * integral


# -------------------------------
# Collected cross-sections (fixed linear polarization, azimuth resolved)
# -------------------------------

def pops_csca(
    D_nm,
    m_particle,
    wavelength_nm: float,
    *,
    geom: POPSGeom,
    _cache: _POPSCache | None = None,
):
    """POPS collected cross-section [um^2] for linearly polarized light.

    Integrate over the mirror cone and, only when explicitly configured,
    the direct PMT cone. Both use sin(theta) dtheta dphi.
    """
    D_nm = np.atleast_1d(D_nm).astype(float)
    if D_nm.ndim != 1 or np.any(~np.isfinite(D_nm)) or np.any(D_nm <= 0):
        raise ValueError("D_nm must be 1D, finite and > 0")
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be finite and > 0")
    c = _cache or pops_geometry_cache(geom)
    out = np.empty_like(D_nm)
    for i, D in enumerate(D_nm):
        out[i] = _collected_cross_section(D, m_particle, wavelength_nm, c.mirror)
        if c.direct is not None:
            out[i] += _collected_cross_section(D, m_particle, wavelength_nm, c.direct)
    return out


def pops_csca_parallel(  # parallel calculation. As in parallel computing, not polarization.
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
    _cache: _SideCollectionCache | None = None,
):
    """UHSAS collected cross-section [um^2], per collection arm.

    Integrate polarized intensity over the 14.8--57 degree annular cone,
    with the full sin(theta) dtheta dphi solid-angle measure.
    """
    D_nm = np.atleast_1d(D_nm).astype(float)
    if D_nm.ndim != 1 or np.any(~np.isfinite(D_nm)) or np.any(D_nm <= 0):
        raise ValueError("D_nm must be 1D, finite and > 0")
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be finite and > 0")
    c = _cache or uhsas_geometry_cache(geom)
    return np.asarray([_collected_cross_section(D, m_particle, wavelength_nm, c)
                       for D in D_nm], dtype=float)


def uhsas_csca_parallel(
    D_nm,
    m_particle,
    wavelength_nm: float,
    *,
    geom: UHSASGeom,
    _cache: _SideCollectionCache | None = None,
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
    if not (np.isfinite(D_min) and np.isfinite(D_max) and float(D_min) > 0 and float(D_max) > float(D_min)):
        raise ValueError("D_range must have finite D_min > 0 and D_max > D_min")
    if int(D_pts) < 2:
        raise ValueError("D_range point count must be >= 2")
    if not (np.isfinite(n_min) and np.isfinite(n_max) and np.isfinite(n_step) and float(n_step) > 0):
        raise ValueError("n_range must be finite with positive step")
    if float(n_max) < float(n_min):
        raise ValueError("n_range maximum must be >= minimum")
    if np.any(~np.isfinite(np.asarray(k_values, dtype=float))):
        raise ValueError("k_values must be finite")
    if int(chunks[0]) <= 0 or int(chunks[1]) <= 0 or int(chunks[2]) <= 0:
        raise ValueError("chunks must contain positive integers")

    Dg = np.geomspace(float(D_min), float(D_max), int(D_pts)).astype(float)
    ng = np.arange(float(n_min), float(n_max) + 1e-12, float(n_step), dtype=float)
    kg = np.asarray(k_values, dtype=float)

    # Validate geometry before touching disk, and never overwrite an existing LUT.
    if kern == "pops":
        cache = pops_geometry_cache(geom)
    else:
        cache = uhsas_geometry_cache(geom)
    if os.path.lexists(zpath):
        raise FileExistsError(f"LUT destination already exists: {zpath}; choose a new directory")
    root = zarr.open_group(zpath, mode="w-")
    root.attrs.update({"optical_model_version": OPTICAL_MODEL_VERSION,
                       "build_complete": False})
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
        def _curve_for_n(n_val, k_val):
            m = complex(float(n_val), float(k_val))
            return pops_csca(Dg, m, wavelength_nm, geom=geom, _cache=cache).astype(np.float32)
        kernel_name = "POPS"
    else:
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
        "description": f"{kernel_name} collected scattering cross-section (polarized solid-angle integral)",
        "optical_model_version": OPTICAL_MODEL_VERSION,
        "build_complete": True,
        "solid_angle_measure": "sin(theta) dtheta dphi",
        "normalization": "miepython qsca; P11-P12 and P11+P12; no extra 0.5",
        "collection_arms": 1,
        "units_sigma_col": "um^2",
        "D_range_nm": [float(D_min), float(D_max)],
        "n_range": [float(n_min), float(n_max), float(n_step)],
        "k_values": kg.tolist(),
        "wavelength_nm": float(wavelength_nm),
        "polarization": "linear E perpendicular to laser and central collection axis; azimuth resolved",
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
            "mirror_halfangle_deg": float(geom.mirror_halfangle_deg),
            "pmt_aperture_distance_mm": geom.pmt_aperture_distance_mm,
            "direct_collection": cache.direct is not None,
            "geometry_reference": "Gao et al. 2016 Fig. 1; mirror-only default per Liu et al. 2021 Appendix A",
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
            "geometry_reference": "Howell et al. 2021 Fig. 1 and Appendix A",
            "irradiance_basis": "total incident irradiance; symmetric counterpropagating beams",
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

def _check_lut_model(root, *, allow_legacy=False):
    version = root.attrs.get("optical_model_version")
    if root.attrs.get("build_complete") is False:
        raise ValueError("LUT build is incomplete; do not use it for diameter conversion.")
    if version == OPTICAL_MODEL_VERSION:
        if root.attrs.get("build_complete") is not True:
            raise ValueError("Corrected LUT has no completed-build marker.")
        return
    if version is None and allow_legacy:
        warnings.warn("Reading a legacy optical LUT for comparison only; it uses the "
                      "uncorrected angular integration.", UserWarning, stacklevel=3)
        return
    raise ValueError(
        f"LUT optical model {version!r} is not {OPTICAL_MODEL_VERSION!r}. "
        "Rebuild into a NEW directory with the corrected optical code. "
        "For deliberate legacy comparisons only, pass allow_legacy=True."
    )


def sigma_query_zarr(zpath: str, D_nm: float, n: float, k: float, *,
                     allow_legacy=False) -> float:
    """
    Trilinear interpolation via SciPy RegularGridInterpolator (values clamped to grid).
    """
    if not all(np.isfinite(v) for v in (D_nm, n, k)):
        raise ValueError("D_nm, n, and k must be finite")
    if D_nm <= 0:
        raise ValueError("D_nm must be > 0")
    z = zarr.open(zpath, mode="r")
    _check_lut_model(z, allow_legacy=allow_legacy)
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
    """Load a completed, current-model LUT for fast trilinear queries.

    Historical tables require allow_legacy=True explicitly and emit a warning.
    That option is for comparisons, not for corrected production.
    """
    def __init__(self, zpath: str, *, allow_legacy=False):
        z = zarr.open(zpath, mode="r")
        _check_lut_model(z, allow_legacy=allow_legacy)
        self.zpath = zpath
        self.Dg  = z["coords/D_nm"][:].astype(float)
        self.ng  = z["coords/n"][:].astype(float)
        self.kg  = z["coords/k"][:].astype(float)
        self.SIG = z["sigma_col"][:].astype(float)
        self.kernel = str(z.attrs.get("kernel", ""))
        self.wavelength_nm = float(z.attrs.get("wavelength_nm", np.nan))
        self.polarization  = str(z.attrs.get("polarization", ""))
        if self.SIG.shape != (self.Dg.size, self.ng.size, self.kg.size):
            raise ValueError("sigma_col shape does not match D/n/k coordinate lengths")
        for label, grid in (("D_nm", self.Dg), ("n", self.ng), ("k", self.kg)):
            if np.any(~np.isfinite(grid)):
                raise ValueError(f"{label} coordinate contains non-finite values")
            if grid.size < 2 or not np.all(np.diff(grid) > 0):
                raise ValueError(f"{label} coordinate must be strictly increasing with at least 2 points")
        if np.any(~np.isfinite(self.SIG)):
            raise ValueError("sigma_col contains non-finite values")

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
    if D.ndim != 1 or S.ndim != 1 or D.size != S.size:
        raise ValueError("D_nm and sigma_col must be 1D arrays with the same length")
    if D.size < 2:
        raise ValueError("Need at least two D/sigma points")
    if np.any(~np.isfinite(D)) or np.any(~np.isfinite(S)) or np.any(D <= 0) or np.any(S <= 0):
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

    vals, inv, counts = np.unique(yhat, return_inverse=True, return_counts=True)
    if vals.size < 2:
        raise ValueError("Isotonic fit collapsed to a constant.")
    x_avg = np.bincount(inv, weights=x) / counts
    # The old forward curve kept plateaus, while the inverse replaced each
    # plateau with a mean diameter. Those were different curves. Use the same
    # representative knots for BOTH directions, then invert the forward curve
    # itself. Collapsing equal responses remains an explicit sizing approximation
    # in the Mie-oscillation region, not a resolution of its physical ambiguity.
    if increasing:
        f_ll = PchipInterpolator(x_avg, vals, extrapolate=False)
    else:
        f_ll = PchipInterpolator(x_avg[::-1], vals[::-1], extrapolate=False)

    def f_sigma(Dq):
        Dq = np.asarray(Dq, float)
        return np.exp(f_ll(np.log(Dq)))

    def g_diam(sig):
        sig = np.asarray(sig, float)
        result = np.full(sig.shape, np.nan)
        flat = sig.ravel()
        valid = np.isfinite(flat) & (flat > 0)
        target = np.full(flat.shape, np.nan)
        target[valid] = np.log(flat[valid])
        # Accommodate only floating-point roundoff at the endpoint responses.
        tol = 8 * np.finfo(float).eps * max(1.0, np.max(np.abs(vals)))
        valid &= (target >= vals[0] - tol) & (target <= vals[-1] + tol)
        if not np.any(valid):
            return result
        q = np.clip(target[valid], vals[0], vals[-1])
        j = np.clip(np.searchsorted(vals, q, side="right") - 1, 0, vals.size - 2)
        low = np.minimum(x_avg[j], x_avg[j + 1])
        high = np.maximum(x_avg[j], x_avg[j + 1])
        # Bracketed, vectorized bisection of the actual forward cubic. Forty-eight
        # halvings make log-diameter error negligible relative to LUT resolution.
        for _ in range(48):
            mid = 0.5 * (low + high)
            go_right = (f_ll(mid) < q) if increasing else (f_ll(mid) > q)
            low = np.where(go_right, mid, low)
            high = np.where(go_right, high, mid)
        result.ravel()[valid] = np.exp(0.5 * (low + high))
        return result

    return f_sigma, g_diam


# -------------------------------
# Size-distribution remap (change refractive index)
# -------------------------------

def convert_do_lut(
    Do_nm,
    ri_src, ri_dst, lut: "SigmaLUT",
    *, response_bins=50, eps=1e-12, source_sigma_fn=None
):
    """
    Map a diameter axis (typically OPC bin edges) from ri_src -> ri_dst using the LUT.

    This function ONLY maps the x-axis (D or edges). It does NOT remap any y-values.
    Use remap_dndlog_by_edges(old_edges, new_edges, y_old) to conservatively rebin
    the size distribution onto the mapped edges.

    Parameters
    ----------
    Do_nm : array-like
        Diameter grid to map (usually bin edges) [nm], must be strictly increasing and > 0.
    ri_src, ri_dst : complex
        Source/destination refractive indices for the optical mapping.
    lut : SigmaLUT
        Provides sigma_curve(D, n, k).
    response_bins : int
        Binning used inside monotone sigma(D) construction (isotonic regression).
    eps : float
        Small positive threshold for sanity checks (used only for validation).
    source_sigma_fn : callable, optional
        Precomputed source-response function. This avoids rebuilding the fixed
        source RI curve inside repeated optimization calls.

    Returns
    -------
    Do_nm_new : ndarray
        Mapped diameter grid [nm], same shape as Do_nm.
    """
    Do_nm = np.asarray(Do_nm, float)
    if Do_nm.ndim != 1:
        raise ValueError("Do_nm must be 1D")
    if np.any(Do_nm <= 0) or not np.all(np.diff(Do_nm) > 0):
        raise ValueError("Do_nm must be strictly increasing and > 0")

    # build monotone σ maps on LUT grid
    Dg = np.asarray(lut.Dg, float)
    ns, ks = float(np.real(ri_src)), float(np.imag(ri_src))
    nd, kd = float(np.real(ri_dst)), float(np.imag(ri_dst))

    sigma_dst = lut.sigma_curve(Dg, nd, kd)

    if source_sigma_fn is None:
        sigma_src = lut.sigma_curve(Dg, ns, ks)
        f_src_sigma, _ = make_monotone_sigma_interpolator(
            Dg, sigma_src, response_bins=response_bins, increasing=True
        )
    else:
        f_src_sigma = source_sigma_fn

    _, D_of_sigma_dst = make_monotone_sigma_interpolator(
        Dg, sigma_dst, response_bins=response_bins, increasing=True
    )

    # map D -> σ (at ri_src)
    sigma_edges = f_src_sigma(Do_nm)
    if not np.all(np.isfinite(sigma_edges)) or np.any(sigma_edges <= eps):
        raise ValueError("Non-finite or non-positive σ encountered; check LUT and monotone fit.")

    # invert σ -> D' (at ri_dst)
    Do_nm_new = D_of_sigma_dst(sigma_edges)
    Do_nm_new = np.asarray(Do_nm_new, float)

    if Do_nm_new.shape != Do_nm.shape:
        raise ValueError("Mapped diameter grid shape mismatch.")
    if not np.all(np.isfinite(Do_nm_new)) or np.any(Do_nm_new <= 0):
        raise ValueError("Mapped diameters contain non-finite or non-positive values.")
    if not np.all(np.diff(Do_nm_new) > 0):
        raise ValueError("Mapped diameters are not strictly increasing (check response_bins / LUT).")

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
    "convert_do_lut",
    # Constants
    "POPS_WAVELENGTH_NM",
    "UHSAS_WAVELENGTH_NM",
    "RI_UHSAS_SRC",
    "RI_POPS_SRC",
    "OPTICAL_MODEL_VERSION",
]
