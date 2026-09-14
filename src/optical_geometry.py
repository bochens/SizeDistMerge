"""Shared optical inputs for scattering calculations and geometry drawings.

Directions are unit vectors in one common Cartesian frame. A cone is a set
of outgoing light directions, not a physical mirror surface. Each detector
accepts the union of its collection cones minus the union of its exclusions.
Different detectors remain separate; overlapping cones within one detector
are counted only once. No lens throughput or detector gain is assumed here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from importlib import resources
from pathlib import Path
import tomllib
import numpy as np


def load_optical_setup(name_or_path):
    """Load a built-in OPC name or a custom TOML file as a checked setup.

    Use the returned object for both drawing and calculation. Loading does
    not alter a saved LUT; a LUT continues to carry its own original setup.
    Unknown fields are rejected so a misspelled setting cannot be ignored.
    """
    requested = Path(name_or_path)
    if requested.suffix == '.toml':
        with requested.open('rb') as stream:
            settings = tomllib.load(stream)
    else:
        name = str(name_or_path).lower()
        if name not in ('pops', 'uhsas', 'pcasp'):
            raise ValueError('Choose pops, uhsas, pcasp, or a path ending in .toml')
        checkout_file = Path(__file__).resolve().parents[1] / 'opc_setups' / f'{name}.toml'
        if (Path(__file__).resolve().parents[1] / 'pyproject.toml').is_file():
            source = checkout_file
        else:
            source = resources.files('sizedistmerge.opc_setups').joinpath(f'{name}.toml')
        with source.open('rb') as stream:
            settings = tomllib.load(stream)
    if settings.pop('schema_version', None) != 1:
        raise ValueError('Optical setup requires schema_version = 1')
    for key in ('name', 'reference', 'notes'):
        settings.pop(key, None)

    def reject_unknown(values, cls):
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'Unknown {cls.__name__} settings: {sorted(unknown)}')

    reject_unknown(settings, OpticalSetup)
    for beam in settings.get('beams', []):
        reject_unknown(beam, IncidentBeam)
    for channel in settings.get('channels', []):
        reject_unknown(channel, CollectionChannel)
        for cone in channel.get('collect', []) + channel.get('exclude', []):
            reject_unknown(cone, CollectionCone)
    return OpticalSetup.from_dict(settings)


def _unit(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)) or np.linalg.norm(value) == 0:
        raise ValueError(f"{name} must be a finite nonzero 3-vector")
    length = np.linalg.norm(value)
    # Preserve already-normalized saved vectors on a metadata round-trip.
    if abs(length - 1.) > 4*np.finfo(float).eps:
        value = value / length
    return tuple(float(x) for x in value)


@dataclass(frozen=True)
class CollectionCone:
    """A circular opening centered on ``axis``, with half-angle in degrees.

    Use ``from_solid_angle`` to specify its area on the unit sphere instead,
    in steradians (the full sphere is 4*pi). Exclusions use this same class.
    """
    axis: tuple[float, float, float]
    half_angle_deg: float

    def __post_init__(self):
        object.__setattr__(self, "axis", _unit(self.axis, "cone axis"))
        angle = float(self.half_angle_deg)
        if not np.isfinite(angle) or not 0 <= angle <= 180:
            raise ValueError("half_angle_deg must be between 0 and 180")
        object.__setattr__(self, "half_angle_deg", angle)

    @classmethod
    def from_solid_angle(cls, axis, solid_angle_sr):
        if not np.isfinite(solid_angle_sr) or not 0 <= solid_angle_sr <= 4*np.pi:
            raise ValueError("solid_angle_sr must be between 0 and 4*pi")
        return cls(axis, np.rad2deg(np.arccos(1 - solid_angle_sr/(2*np.pi))))

    @property
    def solid_angle_sr(self):
        return float(2*np.pi*(1 - np.cos(np.deg2rad(self.half_angle_deg))))

    def contains(self, directions):
        """Membership of unit direction vectors, with the final dimension 3."""
        directions = np.asarray(directions, dtype=float)
        if directions.shape[-1:] != (3,):
            raise ValueError("directions must end in dimension 3")
        if self.half_angle_deg == 0:
            return np.zeros(directions.shape[:-1], dtype=bool)
        return directions @ np.asarray(self.axis) >= np.cos(np.deg2rad(self.half_angle_deg)) - 1e-14

    def directions(self, angle_rad, azimuth_rad):
        """Unit vectors at a polar angle and azimuth around this cone's axis.

        This is also the plotting interface. At ``half_angle_deg`` it traces
        the exact same boundary used by ``contains`` and by the integrator.
        """
        axis = np.asarray(self.axis)
        # Choose a direction clearly different from the cone axis, then remove
        # its component along that axis. This gives a stable starting direction
        # for drawing the rim, even when the cone points along x, y or z.
        seed = np.eye(3)[np.argmin(np.abs(axis))]
        transverse_basis = seed - np.dot(seed, axis)*axis
        transverse_basis /= np.linalg.norm(transverse_basis)
        azimuth_basis = np.cross(axis, transverse_basis)
        angle, phi = np.broadcast_arrays(angle_rad, azimuth_rad)
        return (np.cos(angle)[..., None]*axis + np.sin(angle)[..., None]
                * (np.cos(phi)[..., None]*transverse_basis
                   + np.sin(phi)[..., None]*azimuth_basis))

    def boundary(self, azimuth_rad):
        return self.directions(np.deg2rad(self.half_angle_deg), azimuth_rad)


@dataclass(frozen=True)
class CollectionChannel:
    """One detector's accepted directions, without detector amplification.

    A ray is accepted if it lies in any collection cone and no exclusion cone.
    Overlapping collection cones do not count the same ray twice.
    """
    name: str
    collect: tuple[CollectionCone, ...]
    exclude: tuple[CollectionCone, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "collect", tuple(self.collect))
        object.__setattr__(self, "exclude", tuple(self.exclude))
        if not self.name or not self.collect:
            raise ValueError("a channel needs a name and at least one collection cone")
        if not all(isinstance(cone, CollectionCone) for cone in self.collect + self.exclude):
            raise TypeError("collect and exclude must contain CollectionCone objects")

    def accepts(self, directions):
        mask = np.logical_or.reduce([cone.contains(directions) for cone in self.collect])
        for cone in self.exclude:
            mask &= ~cone.contains(directions)
        return mask


@dataclass(frozen=True)
class IncidentBeam:
    """A linearly polarized beam, with a fraction of total incident irradiance.

    Multiple beams are added as intensities, not as coherent electric fields;
    this does not model a standing-wave interference pattern.
    """
    direction: tuple[float, float, float] = (0., 0., 1.)
    polarization: tuple[float, float, float] = (1., 0., 0.)
    irradiance_fraction: float = 1.

    def __post_init__(self):
        object.__setattr__(self, "direction", _unit(self.direction, "beam direction"))
        object.__setattr__(self, "polarization", _unit(self.polarization, "polarization"))
        if abs(np.dot(self.direction, self.polarization)) > 1e-12:
            raise ValueError("polarization must be perpendicular to the beam")
        if not np.isfinite(self.irradiance_fraction) or self.irradiance_fraction <= 0:
            raise ValueError("irradiance_fraction must be finite and positive")

    @property
    def transverse(self):
        # With beam=z and E=x this is y. Thus phi=0 is y, increasing toward x;
        # this is intentionally not the usual azimuth measured from x toward y.
        return np.cross(self.direction, self.polarization)


@dataclass(frozen=True)
class OpticalSetup:
    """All physical inputs shared by calculation and drawing.

    ``aerosol_direction`` indicates downstream flow; it is drawing metadata,
    not an extra weighting in the scattering integral. Cross-sections are
    normalized by total irradiance and are returned separately by channel.
    """
    wavelength_nm: float
    beams: tuple[IncidentBeam, ...]
    channels: tuple[CollectionChannel, ...]
    aerosol_direction: tuple[float, float, float] = (-1., 0., 0.)
    angular_step_deg: float = .25

    def __post_init__(self):
        object.__setattr__(self, "beams", tuple(self.beams))
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "aerosol_direction", _unit(self.aerosol_direction, "aerosol direction"))
        if not np.isfinite(self.wavelength_nm) or self.wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be finite and positive")
        if not np.isfinite(self.angular_step_deg) or self.angular_step_deg <= 0:
            raise ValueError("angular_step_deg must be finite and positive")
        if not self.beams or not self.channels:
            raise ValueError("a setup needs at least one beam and one channel")
        if not np.isclose(sum(beam.irradiance_fraction for beam in self.beams), 1., rtol=0, atol=1e-12):
            raise ValueError("beam irradiance fractions must sum to 1")
        names = [channel.name for channel in self.channels]
        if len(set(names)) != len(names):
            raise ValueError("channel names must be unique")

    def to_dict(self):
        """JSON-compatible physical setup for reproducible LUT metadata."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        # Rebuild the beams and cones from saved LUT metadata. Their constructors
        # also check the directions and angle limits, just as for a new setup.
        data = dict(data)
        data['beams'] = tuple(IncidentBeam(**beam) for beam in data['beams'])
        data['channels'] = tuple(CollectionChannel(
            channel['name'], tuple(CollectionCone(**cone) for cone in channel['collect']),
            tuple(CollectionCone(**cone) for cone in channel.get('exclude', ())))
            for channel in data['channels'])
        return cls(**data)


def _merge_intervals(intervals):
    merged = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return merged


def _cap_intervals(theta, beam, cone):
    """Exact accepted azimuth intervals at one scattering angle theta.

    At a fixed angle from the beam, rotating a ray traces a circle. Find the
    portions of that circle inside the cone. Returned endpoints are radians.
    In the formula below, k is the beam direction, E its polarization, and
    u=k cross E: ray = k*cos(theta) + sin(theta)*(u*cos(phi)+E*sin(phi)).
    """
    axis = np.asarray(cone.axis)
    if cone.half_angle_deg == 0:
        return []
    if cone.half_angle_deg == 180:
        return [(0., 2*np.pi)]
    # At fixed theta, the scattered ray traces a circle about the beam.
    # The cone condition becomes amplitude*cos(phi-center) >= threshold.
    transverse_projection = np.dot(axis, beam.transverse)
    polarization_projection = np.dot(axis, beam.polarization)
    amplitude = np.sin(theta)*np.hypot(transverse_projection, polarization_projection)
    threshold = np.cos(np.deg2rad(cone.half_angle_deg)) - np.cos(theta)*np.dot(axis, beam.direction)
    if amplitude < 1e-15:
        # At a pole, or for a beam-centred cone, azimuth cannot change membership.
        return [(0., 2*np.pi)] if threshold <= 0 else []
    threshold_ratio = threshold/amplitude
    if threshold_ratio >= 1:
        return []
    if threshold_ratio <= -1:
        return [(0., 2*np.pi)]
    center = np.arctan2(polarization_projection, transverse_projection) % (2*np.pi)
    halfwidth = np.arccos(threshold_ratio)
    lo, hi = center-halfwidth, center+halfwidth
    # An opening that crosses phi=0 is two intervals in [0, 2*pi], not a
    # negative-width interval. Resolve this before unions and exclusions.
    if lo < 0:
        return [(0., hi), (lo+2*np.pi, 2*np.pi)]
    if hi > 2*np.pi:
        return [(0., hi-2*np.pi), (lo, 2*np.pi)]
    return [(lo, hi)]


def channel_azimuth_weights(theta_rad, beam, channel):
    """Integrate 1, cos(phi)^2 and sin(phi)^2 over the accepted azimuths.

    Cone unions and exclusions are resolved before integration: an overlapping
    opening is not counted twice and an excluded overlap is not subtracted twice.
    The caller supplies the remaining sin(theta)*dtheta solid-angle measure.
    """
    theta = np.asarray(theta_rad, dtype=float)
    weights = np.zeros((3, theta.size))
    for i, angle in enumerate(theta.ravel()):
        intervals = _merge_intervals([interval for cone in channel.collect for interval in _cap_intervals(angle, beam, cone)])
        excluded = _merge_intervals([interval for cone in channel.exclude for interval in _cap_intervals(angle, beam, cone)])
        for cut_lo, cut_hi in excluded:
            pieces = []
            for lo, hi in intervals:
                if cut_hi <= lo or cut_lo >= hi:
                    pieces.append((lo, hi))
                else:
                    if lo < cut_lo:
                        pieces.append((lo, cut_lo))
                    if hi > cut_hi:
                        pieces.append((cut_hi, hi))
            intervals = pieces
        for lo, hi in intervals:
            width = hi-lo
            # Integral cos(phi)^2 dphi = phi/2 + sin(2*phi)/4.
            # The sine-squared integral is the remaining width, so the two
            # polarization weights add back to the accepted azimuthal width.
            cosine = width/2 + (np.sin(2*hi)-np.sin(2*lo))/4
            weights[:, i] += width, cosine, width-cosine
    return tuple(np.maximum(weight.reshape(theta.shape), 0.) for weight in weights)


# Instrument presets; explicit legacy settings remain supported.
POPS_WAVELENGTH_NM   = 405.0
UHSAS_WAVELENGTH_NM  = 1054.0
PCASP_WAVELENGTH_NM  = 632.8


RI_UHSAS_SRC=complex(1.52, 0.00)
RI_POPS_SRC =complex(1.615, 0.001)

# Old disk-width LUTs must not silently enter new calculations.
OPTICAL_MODEL_VERSION = "solid-angle-polarized-cones-v1"

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


def _check_theta_coverage(low, high, halfangle, label):
    if not (np.isfinite(low) and np.isfinite(high)
            and 0 <= low <= 90 - halfangle and 90 + halfangle <= high <= 180):
        raise ValueError(f"{label} theta limits must cover 90 +/- its collection half-angle")


def pops_optical_setup(geom: POPSGeom | None = None, *, wavelength_nm=POPS_WAVELENGTH_NM):
    """One source for the POPS calculation and drawing; mirror-only by default."""
    if geom is None:
        return replace(load_optical_setup("pops"), wavelength_nm=float(wavelength_nm))
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
    if geom is None:
        return replace(load_optical_setup("uhsas"), wavelength_nm=float(wavelength_nm))
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
    if geom is None:
        return replace(load_optical_setup("pcasp"), wavelength_nm=float(wavelength_nm))
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
