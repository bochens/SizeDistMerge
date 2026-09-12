"""Shared optical inputs for scattering calculations and geometry drawings.

Directions are unit vectors in one common Cartesian frame. A cone is a set
of outgoing light directions, not a physical mirror surface. Each detector
accepts the union of its collection cones minus the union of its exclusions.
Different detectors remain separate; overlapping cones within one detector
are counted only once. No lens throughput or detector gain is assumed here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


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
        seed = np.eye(3)[np.argmin(np.abs(axis))]
        u = seed - np.dot(seed, axis)*axis
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        angle, phi = np.broadcast_arrays(angle_rad, azimuth_rad)
        return (np.cos(angle)[..., None]*axis + np.sin(angle)[..., None]
                * (np.cos(phi)[..., None]*u + np.sin(phi)[..., None]*v))

    def boundary(self, azimuth_rad):
        return self.directions(np.deg2rad(self.half_angle_deg), azimuth_rad)


@dataclass(frozen=True)
class CollectionChannel:
    """One detector output: union(collect) minus union(exclude), without gain."""
    name: str
    collect: tuple[CollectionCone, ...]
    exclude: tuple[CollectionCone, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "collect", tuple(self.collect))
        object.__setattr__(self, "exclude", tuple(self.exclude))
        if not self.name or not self.collect:
            raise ValueError("a channel needs a name and at least one collection cone")
        if not all(isinstance(c, CollectionCone) for c in self.collect + self.exclude):
            raise TypeError("collect and exclude must contain CollectionCone objects")

    def accepts(self, directions):
        mask = np.logical_or.reduce([c.contains(directions) for c in self.collect])
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
        if not np.isclose(sum(b.irradiance_fraction for b in self.beams), 1., rtol=0, atol=1e-12):
            raise ValueError("beam irradiance fractions must sum to 1")
        names = [c.name for c in self.channels]
        if len(set(names)) != len(names):
            raise ValueError("channel names must be unique")

    def to_dict(self):
        """JSON-compatible physical setup for reproducible LUT metadata."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data['beams'] = tuple(IncidentBeam(**b) for b in data['beams'])
        data['channels'] = tuple(CollectionChannel(
            c['name'], tuple(CollectionCone(**v) for v in c['collect']),
            tuple(CollectionCone(**v) for v in c.get('exclude', ()))) for c in data['channels'])
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

    Outgoing direction = k*cos(theta) + sin(theta)*(u*cos(phi)+E*sin(phi)),
    with u=k cross E. Dotting with the cone axis gives A*cos(phi-delta)>=B.
    """
    axis = np.asarray(cone.axis)
    if cone.half_angle_deg == 0:
        return []
    if cone.half_angle_deg == 180:
        return [(0., 2*np.pi)]
    a, b = np.dot(axis, beam.transverse), np.dot(axis, beam.polarization)
    amplitude = np.sin(theta)*np.hypot(a, b)
    threshold = np.cos(np.deg2rad(cone.half_angle_deg)) - np.cos(theta)*np.dot(axis, beam.direction)
    if amplitude < 1e-15:
        return [(0., 2*np.pi)] if threshold <= 0 else []
    q = threshold/amplitude
    if q >= 1:
        return []
    if q <= -1:
        return [(0., 2*np.pi)]
    center = np.arctan2(b, a) % (2*np.pi)
    halfwidth = np.arccos(q)
    lo, hi = center-halfwidth, center+halfwidth
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
        intervals = _merge_intervals([p for c in channel.collect for p in _cap_intervals(angle, beam, c)])
        excluded = _merge_intervals([p for c in channel.exclude for p in _cap_intervals(angle, beam, c)])
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
            cosine = width/2 + (np.sin(2*hi)-np.sin(2*lo))/4
            weights[:, i] += width, cosine, width-cosine
    return tuple(np.maximum(v.reshape(theta.shape), 0.) for v in weights)
