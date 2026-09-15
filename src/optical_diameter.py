"""Optical calculations, ordered from the main operations to their helpers.

Merging: convert_do_lut -> LUT response -> monotone curve -> equal-signal diameter.
LUT building: setup_csca -> geometry weights -> Mie intensities -> cross-section.

Human-readable instrument settings live in opc_setups/*.toml. Optical geometry
and validation are in optical_geometry; LUT storage/building is in optical_lut.
Import settings from optical_geometry and table readers/builders from optical_lut.
"""
from __future__ import annotations
import os
import tempfile
from dataclasses import dataclass
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator

def _configure_miepython_jit() -> None:
    """Allow Mie calculations to use compiled code, unless the user opted out."""
    # miepython reads these settings when it is imported, so configure it first.
    # setdefault leaves an explicit user setting alone. Numba is the optional
    # compiler that speeds up repeated numerical calculations.
    os.environ.setdefault("MIEPYTHON_USE_JIT", "1")
    os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.gettempdir())


_configure_miepython_jit()
import miepython as mie

from .optical_geometry import OpticalSetup, channel_azimuth_weights
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optical_lut import SigmaLUT

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

    # Read both responses at the same LUT diameters. sigma_curve interpolates
    # in refractive index as needed; it does not run a new Mie calculation.
    Dg = np.asarray(lut.Dg, float)
    ns, ks = float(np.real(ri_src)), float(np.imag(ri_src))
    nd, kd = float(np.real(ri_dst)), float(np.imag(ri_dst))

    sigma_dst = lut.sigma_curve(Dg, nd, kd)

    if source_sigma_fn is None:
        sigma_src = lut.sigma_curve(Dg, ns, ks)
        # We need only the forward function for the calibration material:
        # what signal would its reported diameter produce?
        f_src_sigma, _ = make_monotone_sigma_interpolator(
            Dg, sigma_src, response_bins=response_bins, increasing=True
        )
    else:
        # During alignment, the calibration material stays the same. The caller
        # can build its response once rather than repeat this smoothing each try.
        f_src_sigma = source_sigma_fn

    # For the assumed material we need the opposite direction: which diameter
    # produces that signal? This function searches the same smoothed curve.
    _, D_of_sigma_dst = make_monotone_sigma_interpolator(
        Dg, sigma_dst, response_bins=response_bins, increasing=True
    )

    # Convert every input bin edge to a signal under the calibration RI.
    sigma_edges = f_src_sigma(Do_nm)
    if not np.all(np.isfinite(sigma_edges)) or np.any(sigma_edges <= eps):
        raise ValueError("Non-finite or non-positive σ encountered; check LUT and monotone fit.")

    # Keep those signals unchanged and find their diameters under the new RI.
    # No concentration is changed here; the caller preserves each bin's total
    # when it adjusts the height for the new bin width.
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
    # A cache holds the accepted angles and polarization weights, not Mie
    # results. It can be reused for other sizes and refractive indices, but
    # must have been built from this same setup, including its angular spacing.
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
                # Each call integrates the light from one particle size into
                # this detector for this beam. The returned area is in um².
                spectrum = np.array([
                    _collected_cross_section(d, m_particle, setup.wavelength_nm, cache)
                    for d in diameters])
                evaluated.append((cache, spectrum))
            # These fractions refer to total laser irradiance. For two equal
            # beams each contributes half, so we do not add a factor of two.
            values += beam.irradiance_fraction * spectrum
        # Keep detectors separate. The LUT builder or analysis must choose the
        # detector it needs, rather than silently adding all collection arms.
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
    # Evaluate scattering at cos(theta), the angular coordinate expected by
    # miepython. With norm="qsca", the full-sphere integral of P11 is Qsca;
    # this is why the final area factor below is the particle's projected area.
    phase_matrix = mie.phase_matrix(m_particle, size_parameter, cache.mu, norm="qsca")
    perpendicular = phase_matrix[0, 0, :] - phase_matrix[0, 1, :]
    parallel = phase_matrix[0, 0, :] + phase_matrix[0, 1, :]
    # The cache already integrates the squared polarization projections over
    # accepted azimuths. Only sin(theta) dtheta remains of the solid angle.
    phi_integral = perpendicular * cache.perp_phi + parallel * cache.parallel_phi
    integrand = phi_integral * np.sin(cache.theta_rad)
    # The trapezoidal rule adds the area under the sampled angular curve.
    # Both branches use the same rule; Numba only accelerates the loop.
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
        # The dot product of unit directions is cos(theta). Clipping removes
        # possible roundoff outside [-1, 1], not any physical scattering angles.
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
    # Unlike setup_csca, this does not integrate over accepted angles. It is
    # area per steradian, with the same array layout as the supplied directions.
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
        # Reduce densely sampled oscillations to one representative per log
        # interval. Empty intervals supply no information and are skipped.
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
    elif sample_weight is not None:
        # Each supplied weight belongs to the diameter/signal in the same row.
        # Keep those rows together when sorting; otherwise a different point
        # can receive the strongest weight.
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape != diameters.shape:
            raise ValueError("sample_weight must be 1D with one value per diameter")
        sample_weight = sample_weight[order]

    isotonic_fit = IsotonicRegression(increasing=bool(increasing), out_of_bounds="clip")
    # Isotonic regression moves the representative signals as little as possible
    # in weighted squared-error terms, while forbidding decreases. It can leave
    # equal adjacent signals; the next step handles those plateaus.
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
        # PCHIP connects these knots without introducing a reversal between
        # them. Extrapolation is disabled because no response is known outside
        # the retained diameter range.
        log_response = PchipInterpolator(retained_log_diameters, distinct_log_signals, extrapolate=False)
    else:
        log_response = PchipInterpolator(retained_log_diameters[::-1], distinct_log_signals[::-1], extrapolate=False)

    def f_sigma(Dq):
        """Evaluate cross-section at a diameter, returning to physical units."""
        query_diameters = np.asarray(Dq, float)
        return np.exp(log_response(np.log(query_diameters)))

    def g_diam(sig):
        """Find the diameter giving this signal on the forward curve above."""
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
        # Sorted signal knots tell us which two diameters enclose each answer.
        # We can then search just that interval instead of the whole LUT range.
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
    # Each detector has one set of weights per incident beam. A reversed beam
    # has a different definition of forward scattering, even for the same lens.
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
            # This side-facing circular aperture has a simple exact azimuth
            # formula. Keep its established theta grid as well as its weights.
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
    # Add cone edges and centres to the regular grid so geometry changes do
    # not fall unnoticed between samples. The azimuth helper combines accepted
    # cones and subtracts exclusions before computing polarization weights.
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
    """Angular samples and weights for one detector/beam pair.

    theta_rad is the scattering angle and mu is its cosine. dphi is the total
    accepted azimuth width. perp_phi and parallel_phi include the squared
    electric-field projections integrated over that width. They add to dphi.
    The name reflects the original side-facing implementation; the same arrays
    also describe tilted and beam-aligned collection regions.
    """
    theta_rad: np.ndarray
    mu: np.ndarray
    dphi: np.ndarray
    perp_phi: np.ndarray
    parallel_phi: np.ndarray


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
    """Build angular weights for a side-facing cone with an optional opening."""
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
    # Remove the inner opening from the outer cone. Roundoff can otherwise
    # leave a tiny negative accepted width at a boundary; light cannot have
    # a negative collection weight.
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

__all__ = [
    "convert_do_lut", "setup_csca", "directional_cross_section",
    "make_monotone_sigma_interpolator", "setup_geometry_cache",
    "channel_geometry_cache",
]
