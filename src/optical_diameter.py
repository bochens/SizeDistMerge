"""Optical calculations, ordered from the main operations to their helpers.

Merging: convert_do_lut -> LUT response -> monotone curve -> equal-signal diameter.
LUT building: setup_csca -> geometry weights -> Mie intensities -> cross-section.

Human-readable instrument settings live in opc_setups/*.toml. Optical geometry
and validation are in optical_geometry; LUT storage/building is in optical_lut.
Old imports from this module remain supported.
"""
from __future__ import annotations
import os
import tempfile
from dataclasses import dataclass
import numpy as np
from joblib import Parallel, delayed
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator

def _configure_miepython_jit() -> None:
    os.environ.setdefault("MIEPYTHON_USE_JIT", "1")
    os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.gettempdir())


_configure_miepython_jit()
import miepython as mie

from .optical_geometry import (
    CollectionCone, CollectionChannel, IncidentBeam, OpticalSetup,
    channel_azimuth_weights, load_optical_setup,
    POPSGeom, UHSASGeom, PCASPGeom,
    pops_optical_setup, uhsas_optical_setup, pcasp_optical_setup,
    las_uhsas_proxy_setup, optical_setup_from_lut_metadata,
    POPS_WAVELENGTH_NM, UHSAS_WAVELENGTH_NM, PCASP_WAVELENGTH_NM,
    RI_POPS_SRC, RI_UHSAS_SRC, OPTICAL_MODEL_VERSION,
)

from .optical_lut import (
    SigmaLUT, sigma_query_zarr, build_sigma_lut, build_setup_sigma_lut,
    build_pcasp_sigma_lut, build_pops_sigma_lut, build_uhsas_sigma_lut,
    DEFAULT_D_RANGE, DEFAULT_N_RANGE, DEFAULT_K_VALUES, DEFAULT_CHUNKS,
)

# Main calculations: read these first.

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
        # Strictly increasing source and destination responses preserve edge
        # order mathematically. Still reject finite-precision edge collapse here,
        # before another routine divides concentration by a zero bin width.
        raise ValueError("Mapped diameters are not strictly increasing (check response_bins / LUT).")

    return Do_nm_new


def setup_csca(D_nm, m_particle, setup: OpticalSetup, *, _cache=None):
    """Collected scattering cross-sections [um^2], returned by channel name.

    Uses the same OpticalSetup as a drawing. Separate detectors are never
    silently summed. Beam intensities are added with their specified fractions
    of total irradiance; coherent standing-wave interference is not modeled.
    """
    diameters = np.atleast_1d(D_nm).astype(float)
    if diameters.ndim != 1 or np.any(~np.isfinite(diameters)) or np.any(diameters <= 0):
        raise ValueError("D_nm must be 1D, finite and positive")
    caches = setup_geometry_cache(setup) if _cache is None else _cache
    result = {}
    evaluated = []
    for channel in setup.channels:
        values = np.zeros_like(diameters)
        for beam, cache in zip(setup.beams, caches[channel.name], strict=True):
            # Reuse identical integrals, not just similar-looking cones. For
            # symmetric UHSAS optics this avoids recalculating four equal paths.
            spectrum = None
            for previous_cache, previous_spectrum in evaluated:
                if _same_scattering_geometry(cache, previous_cache):
                    spectrum = previous_spectrum
                    break
            if spectrum is None:
                spectrum = np.array([
                    _collected_cross_section(d, m_particle, setup.wavelength_nm, cache)
                    for d in diameters])
                evaluated.append((cache, spectrum))
            values += beam.irradiance_fraction * spectrum
        result[channel.name] = values
    return result


def _collected_cross_section(D_nm, m_particle, wavelength_nm, cache):
    """One polarized collection path, in um^2.

    miepython qsca normalization supplies intensity per steradian, including
    Qsca. P11-P12 and P11+P12 are the perpendicular and parallel squared
    amplitudes in that normalization, not the unnormalized textbook amplitudes.
    They already describe intensity: do not square them again or halve them.
    """
    size_parameter = np.pi * D_nm / wavelength_nm
    phase_matrix = mie.phase_matrix(m_particle, size_parameter, cache.mu, norm="qsca")
    perpendicular = phase_matrix[0, 0, :] - phase_matrix[0, 1, :]
    parallel = phase_matrix[0, 0, :] + phase_matrix[0, 1, :]
    # The cache already integrates the squared polarization projections over
    # accepted azimuths. Only sin(theta) dtheta remains of the solid angle.
    phi_integral = perpendicular * cache.perp_phi + parallel * cache.parallel_phi
    integrand = phi_integral * np.sin(cache.theta_rad)
    integral = (_trapz_numba(integrand, cache.theta_rad) if _HAVE_NUMBA
                else np.trapezoid(integrand, cache.theta_rad))
    radius_um = 0.5 * D_nm * 1e-3
    # The angular integral is still normalized to the particle's projected
    # area. Multiply by pi*r^2 to get a cross-section in um^2. The alternative
    # textbook prefactor (wavelength/2pi)^2 belongs to unnormalized amplitudes;
    # applying it here as well would count the normalization twice.
    return np.pi * radius_um**2 * integral


def directional_cross_section(D_nm, m_particle, setup: OpticalSetup, directions):
    """Polarized differential scattering cross-section [um^2/sr].

    Evaluates outgoing unit directions before applying any collection mask.
    Suitable for a phase-pattern drawing or an independent angular integral.
    """
    directions = np.asarray(directions, dtype=float)
    if (directions.shape[-1:] != (3,) or not np.all(np.isfinite(directions))
            or not np.allclose(np.linalg.norm(directions, axis=-1), 1., atol=1e-12, rtol=0)):
        raise ValueError("directions must be finite unit 3-vectors")
    if not np.isfinite(D_nm) or D_nm <= 0:
        raise ValueError("D_nm must be finite and positive")
    flat = directions.reshape(-1, 3)
    total = np.zeros(len(flat))
    for beam in setup.beams:
        cos_theta = np.clip(flat @ beam.direction, -1., 1.)
        phase_matrix = mie.phase_matrix(m_particle, np.pi*D_nm/setup.wavelength_nm, cos_theta, norm="qsca")
        # Each outgoing ray and the beam define their own scattering plane.
        # Find the fraction of electric-field intensity parallel to that plane;
        # the rest is perpendicular. Using only the central plane would give
        # incorrect polarization weights for rays elsewhere in the aperture.
        # At exactly forward/backward scattering the two intensities are equal.
        parallel = np.divide((flat @ beam.polarization)**2, 1-cos_theta**2,
                             out=np.zeros_like(cos_theta), where=(1-cos_theta**2) > 1e-14)
        parallel = np.clip(parallel, 0., 1.)
        total += beam.irradiance_fraction*((phase_matrix[0, 0]-phase_matrix[0, 1])*(1-parallel)
                                          + (phase_matrix[0, 0]+phase_matrix[0, 1])*parallel)
    return (np.pi*(D_nm*.5e-3)**2*total).reshape(directions.shape[:-1])


# Response smoothing and inversion.

def make_monotone_sigma_interpolator(
    D_nm, sigma_col, *, increasing=True, sample_weight=None,
    response_bins=None,
):
    """
    Build a monotone response and its inverse on one shared log-log curve.

    Diameters are in nm and cross-sections in um^2. Optional response_bins
    groups LUT samples into equally spaced log-diameter intervals; these are
    smoothing intervals, not the instrument's measurement bins. Grouped fits
    use sample counts as weights, replacing any supplied sample_weight.

    Isotonic regression removes decreases (or increases when increasing=False).
    Equal fitted signals are replaced by one mean log-diameter. PCHIP, a
    shape-preserving piecewise cubic, connects those retained points. The inverse
    searches that SAME cubic rather than fitting another cubic with swapped axes.
    Neither returned function extrapolates beyond its retained endpoint knots.
    """
    diameters = np.asarray(D_nm, float)
    cross_sections = np.asarray(sigma_col, float)
    if diameters.ndim != 1 or cross_sections.ndim != 1 or diameters.size != cross_sections.size:
        raise ValueError("D_nm and sigma_col must be 1D arrays with the same length")
    if diameters.size < 2:
        raise ValueError("Need at least two D/sigma points")
    if (np.any(~np.isfinite(diameters)) or np.any(~np.isfinite(cross_sections))
            or np.any(diameters <= 0) or np.any(cross_sections <= 0)):
        raise ValueError("Need D>0 and σ>0 (uses log–log).")

    order = np.argsort(diameters)
    # Natural logs are used internally; changing log base consistently would
    # describe the same curve. Output remains physical diameter/cross-section.
    x = np.log(diameters[order])
    y = np.log(cross_sections[order])

    if response_bins is not None and response_bins > 1:
        response_bins = int(response_bins)
        edges = np.linspace(x.min(), x.max(), response_bins + 1)
        representative_log_diameters = []
        representative_log_signals = []
        representative_counts = []
        for i in range(response_bins):
            if i < response_bins-1:
                mask = (x >= edges[i]) & (x < edges[i+1])
            else:
                mask = (x >= edges[i]) & (x <= edges[i+1])
            if not np.any(mask):
                continue
            representative_log_diameters.append(x[mask].mean())
            representative_log_signals.append(np.median(y[mask]))
            representative_counts.append(int(mask.sum()))
        if len(representative_log_diameters) < 2:
            raise ValueError("Too few non-empty bins.")
        x = np.asarray(representative_log_diameters)
        y = np.asarray(representative_log_signals)
        sample_weight = np.asarray(representative_counts, float)

    isotonic_fit = IsotonicRegression(increasing=bool(increasing), out_of_bounds="clip")
    if sample_weight is None:
        fitted_log_signals = isotonic_fit.fit_transform(x, y)
    else:
        isotonic_fit.fit(x, y, sample_weight=sample_weight)
        fitted_log_signals = isotonic_fit.predict(x)

    distinct_log_signals, plateau_indices, plateau_counts = np.unique(
        fitted_log_signals, return_inverse=True, return_counts=True
    )
    if distinct_log_signals.size < 2:
        raise ValueError("Isotonic fit collapsed to a constant.")
    # Average representative positions within each plateau without weighting
    # them again by original sample count. This is a distinct step from isotonic
    # regression's weighted adjustment of the representative signal values.
    retained_log_diameters = np.bincount(plateau_indices, weights=x) / plateau_counts
    # The old forward curve kept plateaus, while the inverse replaced each
    # plateau with a mean diameter. Those were different curves. Use the same
    # representative knots for BOTH directions, then invert the forward curve
    # itself. Collapsing equal responses remains an explicit sizing approximation
    # in the Mie-oscillation region, not a resolution of its physical ambiguity.
    if increasing:
        log_response = PchipInterpolator(retained_log_diameters, distinct_log_signals, extrapolate=False)
    else:
        log_response = PchipInterpolator(retained_log_diameters[::-1], distinct_log_signals[::-1], extrapolate=False)

    def f_sigma(Dq):
        # Preserve the callable's existing keyword while using a readable local name.
        query_diameters = np.asarray(Dq, float)
        return np.exp(log_response(np.log(query_diameters)))

    def g_diam(sig):
        query_signals = np.asarray(sig, float)
        result = np.full(query_signals.shape, np.nan)
        flat = query_signals.ravel()
        valid = np.isfinite(flat) & (flat > 0)
        target = np.full(flat.shape, np.nan)
        target[valid] = np.log(flat[valid])
        # Accommodate only floating-point roundoff at the endpoint responses.
        tol = 8 * np.finfo(float).eps * max(1.0, np.max(np.abs(distinct_log_signals)))
        valid &= (target >= distinct_log_signals[0] - tol) & (target <= distinct_log_signals[-1] + tol)
        if not np.any(valid):
            return result
        target_log_signals = np.clip(target[valid], distinct_log_signals[0], distinct_log_signals[-1])
        interval_index = np.clip(
            np.searchsorted(distinct_log_signals, target_log_signals, side="right") - 1,
            0, distinct_log_signals.size - 2,
        )
        low = np.minimum(retained_log_diameters[interval_index],
                         retained_log_diameters[interval_index + 1])
        high = np.maximum(retained_log_diameters[interval_index],
                          retained_log_diameters[interval_index + 1])
        # Bracketed, vectorized bisection of the actual forward cubic. Forty-eight
        # halvings make log-diameter error negligible relative to LUT resolution.
        for _ in range(48):
            mid = 0.5 * (low + high)
            go_right = ((log_response(mid) < target_log_signals) if increasing
                        else (log_response(mid) > target_log_signals))
            low = np.where(go_right, mid, low)
            high = np.where(go_right, high, mid)
        result.ravel()[valid] = np.exp(0.5 * (low + high))
        return result

    return f_sigma, g_diam


# Reusable geometry integration weights.

def setup_geometry_cache(setup: OpticalSetup):
    """Calculate which directions reach each detector, once for each beam.

    These weights depend on the optical geometry and polarization, not on
    particle size or refractive index. Reuse them throughout a LUT build,
    but rebuild them if the setup changes.
    """
    return {channel.name: tuple(channel_geometry_cache(beam, channel, setup.angular_step_deg)
                               for beam in setup.beams) for channel in setup.channels}


def channel_geometry_cache(beam, channel, step_deg):
    """Exact azimuth integration for arbitrary cone directions and exclusions.

    The original side-cone numerical grid is retained for that special case,
    preserving the existing POPS/UHSAS values and production cost exactly.
    """
    if not np.isfinite(step_deg) or step_deg <= 0:
        raise ValueError("angular step must be finite and positive")
    if len(channel.collect) == 1 and len(channel.exclude) <= 1:
        cone = channel.collect[0]
        inner = channel.exclude[0].half_angle_deg if channel.exclude else 0.
        concentric = not channel.exclude or np.allclose(channel.exclude[0].axis, cone.axis, atol=1e-14, rtol=0)
        along_beam = np.dot(cone.axis, beam.direction)
        if concentric and 0 <= inner < cone.half_angle_deg and abs(abs(along_beam)-1) < 1e-14:
            # For a coaxial band, acceptance is a step at each rim. Integrate
            # only the accepted interval, with its one-sided endpoint values;
            # do not smear either step across an uncollected angular cell.
            lo, hi = inner, cone.half_angle_deg
            if along_beam < 0:
                lo, hi = 180-hi, 180-lo
            count = max(256, int(np.ceil((hi-lo)/step_deg))) + 1
            theta = np.deg2rad(np.linspace(lo, hi, count))
            return _SideCollectionCache(theta, np.cos(theta),
                                        np.full(count, 2*np.pi),
                                        np.full(count, np.pi), np.full(count, np.pi))
        if (concentric and 0 <= inner < cone.half_angle_deg < 90
                and abs(np.dot(cone.axis, beam.direction)) < 1e-14
                and abs(np.dot(cone.axis, beam.polarization)) < 1e-14):
            return _side_collection_cache(cone.half_angle_deg, inner, step_deg)
    limits, events = [], []
    for cone in channel.collect + channel.exclude:
        center = np.rad2deg(np.arccos(np.clip(np.dot(cone.axis, beam.direction), -1, 1)))
        lo, hi = max(0., center-cone.half_angle_deg), min(180., center+cone.half_angle_deg)
        events.extend((lo, center, hi))
        if cone in channel.collect:
            limits.append((lo, hi))
    lo, hi = min(v[0] for v in limits), max(v[1] for v in limits)
    if hi <= lo:
        raise ValueError("collection must have nonzero angular extent")
    count = max(256, int(np.ceil((hi-lo)/step_deg))) + 1
    deg = np.unique(np.r_[np.linspace(lo, hi, count),
                          [v for v in events if lo <= v <= hi]])
    theta = np.deg2rad(deg)
    return _SideCollectionCache(theta, np.cos(theta),
                                *channel_azimuth_weights(theta, beam, channel))


def _same_scattering_geometry(first, second):
    """Whether two paths supply identical inputs to the scattering integral."""
    if first is second:
        return True
    fields = ("theta_rad", "mu", "perp_phi", "parallel_phi")
    return all(np.array_equal(getattr(first, name), getattr(second, name)) for name in fields)


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

# Compatibility wrappers for existing callers.

def pops_geometry_cache(geom: POPSGeom) -> _POPSCache:
    """Legacy cache view; new calculations use setup_geometry_cache directly."""
    setup = pops_optical_setup(geom)
    caches = [channel_geometry_cache(setup.beams[0], ch, setup.angular_step_deg)
              for ch in setup.channels]
    return _POPSCache(caches[0], caches[1] if len(caches) > 1 else None)


def uhsas_geometry_cache(geom: UHSASGeom) -> _SideCollectionCache:
    """Legacy single-arm cache view, retained for callers inspecting its arrays."""
    setup = uhsas_optical_setup(geom)
    return channel_geometry_cache(setup.beams[0], setup.channels[0], setup.angular_step_deg)


def pops_csca(
    D_nm,
    m_particle,
    wavelength_nm: float,
    *,
    geom: POPSGeom,
    _cache=None,
):
    """POPS collected cross-section [um^2] for linearly polarized light.

    Uses the general setup integrator. Preserve the existing single-array
    output: mirror collection plus the optional explicitly configured direct
    path. Old _POPSCache inputs are accepted for caller compatibility.
    """
    setup = pops_optical_setup(geom, wavelength_nm=wavelength_nm)
    if isinstance(_cache, _POPSCache):
        paths = (_cache.mirror,) if _cache.direct is None else (_cache.mirror, _cache.direct)
        _cache = {channel.name: (path,) for channel, path in zip(setup.channels, paths, strict=True)}
    result = setup_csca(D_nm, m_particle, setup, _cache=_cache)
    return np.sum(list(result.values()), axis=0)


def pops_csca_parallel(  # parallel calculation. As in parallel computing, not polarization.
    D_nm,
    m_particle,
    wavelength_nm: float,
    *,
    geom: POPSGeom,
    _cache=None,
    n_jobs: int = -1,
    backend: str = "threads",
):
    D_nm = np.atleast_1d(D_nm).astype(float)
    c = (_cache if _cache is not None else
         setup_geometry_cache(pops_optical_setup(geom, wavelength_nm=wavelength_nm)))
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
    _cache=None,
):
    """UHSAS collected cross-section [um^2], per collection arm.

    Uses the general setup integrator with both opposing beams. Return only
    Collection 1, not the sum of the two detectors. A legacy single-arm cache
    remains valid for the identical beam-relative geometries of this preset.
    """
    setup = uhsas_optical_setup(geom, wavelength_nm=wavelength_nm)
    if isinstance(_cache, _SideCollectionCache):
        _cache = {channel.name: (_cache,) * len(setup.beams) for channel in setup.channels}
    return setup_csca(D_nm, m_particle, setup, _cache=_cache)["Collection 1"]


def uhsas_csca_parallel(
    D_nm,
    m_particle,
    wavelength_nm: float,
    *,
    geom: UHSASGeom,
    _cache=None,
    n_jobs: int = -1,
    backend: str = "threads",
):
    D_nm = np.atleast_1d(D_nm).astype(float)
    c = (_cache if _cache is not None else
         setup_geometry_cache(uhsas_optical_setup(geom, wavelength_nm=wavelength_nm)))
    def _one(d):
        return uhsas_csca([d], m_particle, wavelength_nm, geom=geom, _cache=c)[0]
    vals = Parallel(n_jobs=n_jobs, prefer=backend)(delayed(_one)(float(d)) for d in D_nm)
    return np.asarray(vals, dtype=float)


def pcasp_csca(D_nm, m_particle, wavelength_nm=PCASP_WAVELENGTH_NM, *,
               geom: PCASPGeom | None = None, _cache=None):
    """PCASP collected cross-section [um^2] per total incident irradiance.

    With the default equal beams, multiply by two only when comparing with
    Rosenberg's Table-1 weighting (which uses outgoing-beam irradiance).
    Use the unscaled result for both sides of a diameter conversion.
    """
    setup = pcasp_optical_setup(geom, wavelength_nm=wavelength_nm)
    return setup_csca(D_nm, m_particle, setup, _cache=_cache)["Collection"]


__all__ = [
    "load_optical_setup",
    # Geom + caches
    "POPSGeom", "UHSASGeom", "PCASPGeom",
    "CollectionCone", "CollectionChannel", "IncidentBeam", "OpticalSetup",
    "pops_optical_setup", "uhsas_optical_setup", "pcasp_optical_setup", "optical_setup_from_lut_metadata",
    "las_uhsas_proxy_setup",
    "setup_geometry_cache", "channel_geometry_cache", "setup_csca",
    "directional_cross_section", "build_setup_sigma_lut",
    "pops_geometry_cache", "uhsas_geometry_cache",
    # Kernels
    "pops_csca", "pops_csca_parallel",
    "uhsas_csca", "uhsas_csca_parallel",
    "pcasp_csca",
    # LUT build
    "build_sigma_lut",
    "build_pops_sigma_lut",
    "build_uhsas_sigma_lut",
    "build_pcasp_sigma_lut",
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
    "PCASP_WAVELENGTH_NM",
    "RI_UHSAS_SRC",
    "RI_POPS_SRC",
    "OPTICAL_MODEL_VERSION",
]
