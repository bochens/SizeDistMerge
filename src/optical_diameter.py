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
from .optical_geometry import (
    CollectionCone, CollectionChannel, IncidentBeam, OpticalSetup,
    channel_azimuth_weights,
)

# -------------------------------
# Constants and centralized defaults
# -------------------------------

POPS_WAVELENGTH_NM   = 405.0
UHSAS_WAVELENGTH_NM  = 1054.0
PCASP_WAVELENGTH_NM  = 632.8

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


@dataclass(frozen=True)
class PCASPGeom:
    """Nominal PCASP acceptance from Rosenberg et al. (2012), Table 1.

    A full-azimuth band about the outgoing laser spans 35--120 degrees.
    The same physical collector therefore spans 60--145 degrees relative
    to the returning beam. This is not a side-facing circular cone.
    Equal incoherent beam irradiances reproduce the paper's approximation;
    cross-sections here are divided by their total incident irradiance.
    """
    theta_min_deg: float = 35.0
    theta_max_deg: float = 120.0
    ring_step_deg: float = 0.25
    reflected_beam_ratio: float = 1.0  # returning/outgoing irradiance at particle

    def __post_init__(self):
        if not (np.isfinite(self.theta_min_deg) and np.isfinite(self.theta_max_deg)
                and 0 <= self.theta_min_deg < self.theta_max_deg <= 180):
            raise ValueError("PCASP requires 0 <= theta_min_deg < theta_max_deg <= 180")
        if not np.isfinite(self.ring_step_deg) or self.ring_step_deg <= 0:
            raise ValueError("PCASP ring_step_deg must be finite and positive")
        if not np.isfinite(self.reflected_beam_ratio) or self.reflected_beam_ratio < 0:
            raise ValueError("PCASP reflected_beam_ratio must be finite and nonnegative")

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


def pops_optical_setup(geom: POPSGeom | None = None, *, wavelength_nm=POPS_WAVELENGTH_NM):
    """One source for the POPS calculation and drawing; mirror-only by default."""
    geom = geom or POPSGeom()
    _check_theta_coverage(geom.ring_theta_min_deg, geom.ring_theta_max_deg,
                          geom.mirror_halfangle_deg, "POPS mirror")
    channels = [CollectionChannel("Collection", (CollectionCone((0, 1, 0), geom.mirror_halfangle_deg),))]
    if not np.isfinite(geom.pmt_aperture_d_mm) or geom.pmt_aperture_d_mm < 0:
        raise ValueError("pmt_aperture_d_mm must be finite and >= 0")
    if geom.pmt_aperture_d_mm > 0:
        distance = geom.pmt_aperture_distance_mm
        if distance is None or not np.isfinite(distance) or distance <= 0:
            raise ValueError("Direct POPS collection requires pmt_aperture_distance_mm; "
                             "do not use the mirror distance. Set pmt_aperture_d_mm=0 "
                             "for an explicit mirror-only model.")
        if geom.pmt_center_deg != 90.0:
            raise ValueError("Direct POPS collection currently supports only pmt_center_deg=90")
        alpha = np.rad2deg(np.arctan(0.5 * geom.pmt_aperture_d_mm / distance))
        # Preserve the legacy side-on direct-path approximation. This optional
        # path does not establish an independently measured detector position.
        channels.append(CollectionChannel("Direct (side-on approximation)",
                                          (CollectionCone((0, 1, 0), alpha),)))
    return OpticalSetup(wavelength_nm, (IncidentBeam(),), tuple(channels),
                        angular_step_deg=geom.ring_step_deg)


def uhsas_optical_setup(geom: UHSASGeom | None = None, *, wavelength_nm=UHSAS_WAVELENGTH_NM):
    """Opposing UHSAS collectors, kept as separate outputs, and two cavity beams.

    The equal incoherent beam fractions sum to one: normalization is to total
    incident irradiance. Both integrals equal the existing single-beam result
    for these symmetric side-facing openings; this does not add a factor two.
    """
    geom = geom or UHSASGeom()
    _check_theta_coverage(geom.big_theta_min_deg, geom.big_theta_max_deg,
                          geom.big_outer_halfangle_deg, "UHSAS outer")
    _check_theta_coverage(geom.small_theta_min_deg, geom.small_theta_max_deg,
                          geom.inner_stop_halfangle_deg, "UHSAS exclusion")
    channels = tuple(CollectionChannel(
        name, (CollectionCone(axis, geom.big_outer_halfangle_deg),),
        (CollectionCone(axis, geom.inner_stop_halfangle_deg),))
        for name, axis in (("Collection 1", (0, 1, 0)), ("Collection 2", (0, -1, 0))))
    beams = (IncidentBeam(irradiance_fraction=.5),
             IncidentBeam(direction=(0, 0, -1), irradiance_fraction=.5))
    return OpticalSetup(wavelength_nm, beams, channels, angular_step_deg=geom.ring_step_deg)


def pcasp_optical_setup(geom: PCASPGeom | None = None, *, wavelength_nm=PCASP_WAVELENGTH_NM):
    """One collector and two incoherent beams, with the same setup for plots.

    A cone about +z, minus its inner cone, makes the published angular band.
    Rotational symmetry makes the integrated result independent of the
    chosen transverse polarization direction. The full aperture is a nominal
    model, not a measured transmission map for an individual instrument.
    """
    geom = geom or PCASPGeom()
    ratio = geom.reflected_beam_ratio
    beams = (IncidentBeam(irradiance_fraction=1/(1+ratio)),)
    if ratio > 0:
        beams += (IncidentBeam(direction=(0, 0, -1), irradiance_fraction=ratio/(1+ratio)),)
    channel = CollectionChannel(
        "Collection", (CollectionCone((0, 0, 1), geom.theta_max_deg),),
        (CollectionCone((0, 0, 1), geom.theta_min_deg),))
    return OpticalSetup(wavelength_nm, beams, (channel,), angular_step_deg=geom.ring_step_deg)


def las_uhsas_proxy_setup(*, polarization: str, geom: UHSASGeom | None = None):
    """Experimental LAS 3340-family model at 633 nm using UHSAS openings.

    The LAS manual confirms opposing side collectors and a 633 nm cavity,
    but does not specify these aperture angles. Moore et al. (2021), section
    2.2, likewise use UHSAS angles for their LAS calculation. Our circular
    annular cones are an explicit 3-D hypothesis, NOT a reproduction of their
    angle-only calculation or an independently verified LAS geometry.

    Select ``unpolarized``, ``perpendicular`` or ``parallel`` explicitly; the
    latter two refer to the central plane containing beam and collector axes.
    Unpolarized light is the incoherent equal-intensity sum of orthogonal
    polarizations. Opposing detectors remain separate, normalized to total
    incident irradiance, without gain or reflection/transmission losses.
    """
    states = {"unpolarized": ((1., 0., 0.), (0., 1., 0.)),
              "perpendicular": ((1., 0., 0.),),
              "parallel": ((0., 1., 0.),)}
    if polarization not in states:
        raise ValueError("polarization must be unpolarized, perpendicular or parallel")
    base = uhsas_optical_setup(geom, wavelength_nm=633.)
    vectors = states[polarization]
    beams = tuple(IncidentBeam(direction=beam.direction, polarization=vector,
                               irradiance_fraction=beam.irradiance_fraction/len(vectors))
                  for beam in base.beams for vector in vectors)
    return OpticalSetup(base.wavelength_nm, beams, base.channels,
                        aerosol_direction=base.aerosol_direction,
                        angular_step_deg=base.angular_step_deg)


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


def setup_geometry_cache(setup: OpticalSetup):
    """Calculate which directions reach each detector, once for each beam.

    These weights depend on the optical geometry and polarization, not on
    particle size or refractive index. Reuse them throughout a LUT build,
    but rebuild them if the setup changes.
    """
    return {channel.name: tuple(channel_geometry_cache(beam, channel, setup.angular_step_deg)
                               for beam in setup.beams) for channel in setup.channels}


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


def _same_scattering_geometry(first, second):
    """Whether two paths supply identical inputs to the scattering integral."""
    if first is second:
        return True
    fields = ("theta_rad", "mu", "perp_phi", "parallel_phi")
    return all(np.array_equal(getattr(first, name), getattr(second, name)) for name in fields)


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


def pcasp_csca(D_nm, m_particle, wavelength_nm=PCASP_WAVELENGTH_NM, *,
               geom: PCASPGeom | None = None, _cache=None):
    """PCASP collected cross-section [um^2] per total incident irradiance.

    With the default equal beams, multiply by two only when comparing with
    Rosenberg's Table-1 weighting (which uses outgoing-beam irradiance).
    Use the unscaled result for both sides of a diameter conversion.
    """
    setup = pcasp_optical_setup(geom, wavelength_nm=wavelength_nm)
    return setup_csca(D_nm, m_particle, setup, _cache=_cache)["Collection"]


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


def optical_setup_from_lut_metadata(attrs):
    """Read a saved setup, or reconstruct only the known version-1 presets.

    Old LUTs did not store full vectors. Their versioned model fixes those
    vectors; require its metadata rather than guessing a generic geometry.
    """
    if "optical_setup" in attrs:
        setup = OpticalSetup.from_dict(attrs["optical_setup"])
        if setup.wavelength_nm != attrs["wavelength_nm"]:
            raise ValueError("LUT wavelength and saved optical setup disagree")
        return setup
    if attrs.get("optical_model_version") != OPTICAL_MODEL_VERSION:
        raise ValueError("LUT has no supported, unambiguous optical setup")
    kernel = attrs.get("kernel", "").lower()
    cls, factory = {"pops": (POPSGeom, pops_optical_setup),
                    "uhsas": (UHSASGeom, uhsas_optical_setup)}.get(kernel, (None, None))
    if cls is None:
        raise ValueError("custom LUT must store its full optical_setup")
    required = ("mirror_halfangle_deg", "ring_step_deg", "pmt_aperture_d_mm") if kernel == "pops" else (
        "big_outer_halfangle_deg", "inner_stop_halfangle_deg", "ring_step_deg")
    if not all(k in attrs for k in required):
        raise ValueError("LUT is missing required geometry metadata")
    values = {key: attrs[key] for key in cls.__dataclass_fields__ if key in attrs}
    return factory(cls(**values), wavelength_nm=attrs["wavelength_nm"])


# -------------------------------
# Collected cross-sections (fixed linear polarization, azimuth resolved)
# -------------------------------

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
    response_channel: str | None = None,
):
    """Build a new lookup table of unsmoothed scattering cross-sections.

    The stored axes are diameter [nm], real refractive index n, and imaginary
    index k. Each n/k pair supplies one complete diameter-response curve.
    Smoothing is done when using the table, not here. An interrupted build
    keeps build_complete=False so the reader will reject it.
    """
    kern = kernel.lower()
    if kern not in ("pops", "uhsas", "custom"):
        raise ValueError("kernel must be 'pops', 'uhsas' or 'custom'.")
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be finite and positive")
    if kern != "custom" and response_channel is not None:
        raise ValueError("use build_setup_sigma_lut to select a response channel")

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
    if ng.size < 2 or kg.ndim != 1 or kg.size < 2 or not np.all(np.diff(kg) > 0):
        raise ValueError("LUT interpolation requires at least two increasing n and k coordinates")

    # Validate geometry before touching disk, and never overwrite an existing LUT.
    if kern == "pops":
        setup = pops_optical_setup(geom, wavelength_nm=wavelength_nm)
        response_channels = [c.name for c in setup.channels]
    elif kern == "uhsas":
        setup = uhsas_optical_setup(geom, wavelength_nm=wavelength_nm)
        response_channels = [setup.channels[0].name]
    else:
        if not isinstance(geom, OpticalSetup) or geom.wavelength_nm != wavelength_nm:
            raise ValueError("custom geometry must be an OpticalSetup with matching wavelength")
        setup = geom
        selected = [c for c in setup.channels if c.name == response_channel]
        if len(selected) != 1:
            raise ValueError("select exactly one named response_channel for the LUT")
        response_channels = [response_channel]
    selected = tuple(c for c in setup.channels if c.name in response_channels)
    calculation_setup = OpticalSetup(setup.wavelength_nm, setup.beams, selected,
                                     setup.aerosol_direction, setup.angular_step_deg)
    cache = setup_geometry_cache(calculation_setup)
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

    # All instruments use the same integrator and precomputed setup cache.
    # UHSAS/custom select one detector; POPS retains its optional path sum.
    kernel_name = kern.upper()

    def _curve_for_n(n_val, k_val):
        m = complex(float(n_val), float(k_val))
        result = setup_csca(Dg, m, calculation_setup, _cache=cache)
        return np.sum(list(result.values()), axis=0).astype(np.float32)

    # Workers calculate separate refractive-index curves. The parent process
    # writes each completed group, so workers do not write the same array chunks.
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

    # Save the exact setup with the results so later plots and calculations
    # can recover its directions, openings, exclusions and beam fractions.
    # Mark complete only after all refractive-index curves have been written.
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
        "optical_setup": setup.to_dict(),
        "response_channels": response_channels,
        "irradiance_basis": "total incident irradiance; incoherent beam intensity fractions sum to one",
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
            "direct_collection": len(response_channels) > 1,
            "geometry_reference": "Gao et al. 2016 Fig. 1; mirror-only default per Liu et al. 2021 Appendix A",
        })
    elif kern == "uhsas":
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
    else:
        attrs["polarization"] = "linear polarization vectors specified in optical_setup"
    root.attrs.update(attrs)

    return dict(zpath=zpath, D_grid_nm=Dg, n_grid=ng, k_grid=kg)


def build_setup_sigma_lut(zpath, setup: OpticalSetup, *, channel, **kwargs):
    """Build a LUT for one explicitly selected detector in the shared setup."""
    return build_sigma_lut(zpath, "custom", setup.wavelength_nm, setup,
                           response_channel=channel, **kwargs)


def build_pcasp_sigma_lut(zpath, geom: PCASPGeom | None = None, *,
                         wavelength_nm=PCASP_WAVELENGTH_NM, **kwargs):
    """Build a new PCASP LUT through the shared geometry integrator.

    This does not assume a campaign calibration refractive index. The saved
    kernel is CUSTOM because it uses the same general integrator as a custom
    setup; the instrument and literature provenance are recorded separately.
    Existing destinations are rejected by the common builder.
    """
    geom = geom or PCASPGeom()
    setup = pcasp_optical_setup(geom, wavelength_nm=wavelength_nm)
    result = build_setup_sigma_lut(zpath, setup, channel="Collection", **kwargs)
    root = zarr.open_group(zpath, mode="a")
    root.attrs.update({
        "instrument": "PCASP",
        "description": "PCASP nominal collected scattering cross-section",
        "geometry_reference": "Rosenberg et al. 2012 Table 1; doi:10.5194/amt-5-1147-2012",
        "geometry_scope": "nominal full-azimuth acceptance; no measured aperture transmission or detector gain",
        "outgoing_beam_collection_deg": [geom.theta_min_deg, geom.theta_max_deg],
        "returning_beam_collection_deg": [180-geom.theta_max_deg, 180-geom.theta_min_deg],
        "reflected_beam_ratio": geom.reflected_beam_ratio,
        "outgoing_irradiance_basis_multiplier": 1+geom.reflected_beam_ratio,
    })
    return result


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
    Read the LUT and interpolate between neighboring D, n and k coordinates.

    Out-of-range queries use the nearest boundary, not extrapolation. This
    function loads the whole table on each call; reuse SigmaLUT for many queries.
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
    """Keep a completed LUT in memory for repeated D, n and k interpolation.

    Historical tables require allow_legacy=True explicitly and emit a warning.
    That option is for comparisons, not for corrected production.
    Queries outside the stored range are clipped to its boundaries. Clipping
    is not a physically justified extension of the response beyond the table.
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
        # Strictly increasing source and destination responses preserve edge
        # order mathematically. Still reject finite-precision edge collapse here,
        # before another routine divides concentration by a zero bin width.
        raise ValueError("Mapped diameters are not strictly increasing (check response_bins / LUT).")

    return Do_nm_new


# -------------------------------
# Public API
# -------------------------------
__all__ = [
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
