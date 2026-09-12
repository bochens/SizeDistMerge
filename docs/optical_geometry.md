# One setup for optical calculations and drawings

`OpticalSetup` is the physical input. Plotting colors, opacity and camera
position are separate presentation settings and never alter the integral.
The local paper-figure notebook contains `draw_geometry(ax, setup, ...)`.
That manuscript notebook and its outputs are maintained separately, not
distributed as part of this package. The drawing notes below describe its
use of the public setup interface; the renderer is not a package API.

```python
from sizedistmerge.optical_geometry import (
    CollectionCone, CollectionChannel, IncidentBeam, OpticalSetup,
)
from sizedistmerge.optical_diameter import setup_csca, build_setup_sigma_lut

# Example only: not a specification of a real instrument.
setup = OpticalSetup(
    wavelength_nm=650.,
    beams=(IncidentBeam(direction=(0, 0, 1), polarization=(1, 0, 0)),),
    channels=(CollectionChannel(
        name="Side detector",
        collect=(CollectionCone(axis=(0, 1, 1), half_angle_deg=40.),),
        exclude=(CollectionCone(axis=(0, 1, 1), half_angle_deg=10.),),
    ),),
    aerosol_direction=(-1, 0, 0),
    angular_step_deg=.25,
)

sigma = setup_csca([100., 500., 1000.], 1.52+0j, setup)
# sigma["Side detector"] is collected cross-section in square micrometers.

# In the paper notebook, use precisely the same setup:
# draw_geometry(ax, setup, laser_color="#b76b70", phase_ri=1.52)

# Optional: choose a NEW destination; existing LUTs are never overwritten.
# build_setup_sigma_lut("new_detector.zarr", setup, channel="Side detector")
```

## Directions and openings

All directions use the same Cartesian x/y/z frame, with the particle at the
origin. Vectors are normalized; polarization must be perpendicular to its
beam. The cone axis specifies where the opening points, not just its width.

Use `CollectionCone.from_solid_angle(axis=(0, 1, 0), solid_angle_sr=1.)`
to give the cone's solid angle instead of its half-angle. Solid angle is
area on a unit sphere, measured in steradians; the full sphere is `4*pi`.
For one cone, `Omega = 2*pi*(1-cos(half_angle))`.

Each detector accepts the union of its collection cones, minus the union of
its excluded cones. Overlapping openings are counted once. Exclusions may
point in different directions; only their overlap with accepted directions
is removed. The sum of individual cone areas is not generally the accepted
area when cones overlap or are excluded.

Different detector channels remain separate. The caller must explicitly
choose a channel for a LUT; drawing two collectors does not double a signal.
Multiple beams are combined using their fractions of **total irradiance**,
which must sum to one. This is an incoherent intensity sum, not a model of
standing-wave interference. Lens transmission, detector gain and polarization
analyzers after scattering are not included in this interface.

## How the integral uses the setup

PCASP is available through `pcasp_optical_setup()`. It uses one full-azimuth
band about the outgoing laser (35-120 degrees), defined with the same cone
and exclusion interface. The returning beam sees that same opening at
60-145 degrees. See [PCASP inputs and assumptions](pcasp_optics.md).
The paper-figure notebook provides a visible example passed unchanged to
both the calculation and the 3D renderer.

For each scattering angle, the code solves each cone's dot-product condition
to find accepted azimuth intervals. It combines the intervals, applies the
exclusions, and integrates the two linear-polarization components over them.
The remaining polar-angle integral includes `sin(theta) dtheta`.
The E field is projected onto each scattering plane; it is not assumed
perpendicular to every collected direction's plane.

The original side-cone grid remains in use for the POPS/UHSAS special cases.
The refactor is checked against 36 cross-sections captured before the edit.
For new or very narrow/off-axis geometries, check angular-resolution
convergence by reducing `angular_step_deg`; a generic geometry interface
does not validate an instrument's physical dimensions.

## Presets, LUT metadata and the figure

`pops_optical_setup()` describes the approved mirror-only POPS model.
The optional legacy POPS direct path retains its explicitly labeled side-on
approximation, not a newly verified aperture location; it is off by default.
Use a custom setup with measured directions for a different direct path.
`uhsas_optical_setup()` describes two opposing collection channels, each
with its own central excluded cone, and two equal counterpropagating beams.
The old `pops_csca` and `uhsas_csca` interfaces remain compatible.

New LUTs store the complete serialized setup and selected response channels.
`optical_setup_from_lut_metadata()` restores it for plotting. It also adapts
the known version-1 POPS/UHSAS LUT metadata; it rejects unknown old geometries
rather than inferring their missing directions. Existing production LUTs
are not changed by the interface refactor.

The paper notebook renders one native 3D scene with PyVista/VTK, including
curved collectors, rays, flow, the polarization symbol, and the phase curve.
Its mesh uses the setup's cone directions and acceptance masks. Concentric
rims are exact; other cuts are sampled on mesh cells for display only.
No complete background sphere is drawn. Front surfaces have opacity 0.50,
and rear surfaces opacity 0.12. Grid lines exist only on collection surfaces.
All collection channels share one display color and one legend swatch;
their physical acceptance regions remain separate in the calculation.
Secondary collectors use one-quarter of the primary display opacity for
surfaces, grids and outlines. This visual de-emphasis does not change their
contribution to the calculated response.
The labeled beam-collection reference plane has opacity 0.06, with no
sphere-shaped cutout. A scattering plane is defined by the incident and
observed directions: the displayed plane uses the central collection
direction, not every accepted ray. Its smaller corner label lies in that
plane. An unlabeled second plane contains the beam and electric-field
direction at opacity 0.035. Right-angle corner markers are not drawn.
Only the cone boundary used by each marked angle is drawn alongside the
collection axis. The phase curve matches the laser color; the compact
polarization wave sits upstream, away from the central phase curve.
Depth peeling, which sorts transparent surfaces by their actual 3D depth,
must be active. This rendering never feeds back into the scattering integral.
The 1000 nm
phase-pattern overlay uses `directional_cross_section`, before the detector
mask, for the primary incident beam alone, including UHSAS. Its polar
radius is logarithmic in relative intensity: center at 1e-3 of peak and
maximum radius at peak, without concentric guides. Clipping below that
floor affects the drawing only, not scattering. The response calculation
retains every beam in the original setup.

The current drawing supports one laser axis, including counterpropagating
beams with the same polarization axis, and arbitrary collection cones.
It rejects additional laser axes rather than silently omitting them.

PyVista/VTK are optional figure-only dependencies, not production dependencies.
This figure was rendered with PyVista 0.49.0 and VTK 9.6.2 in the isolated
`.cache/paper-3d-venv` environment. The notebook includes setup instructions.
PDF/SVG contain the high-resolution 3D scene and on-plane lettering as an
image; response curves, legend, angle values and headings remain vector.
