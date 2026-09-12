# Optical collection model and LUT migration

The optical kernels now integrate over solid angle and account for the fixed
laser polarization across each collection opening. This is a numerical and
geometrical correction, not a new instrument calibration. Existing POPS/UHSAS
packaged LUTs have now been replaced by completed, versioned tables. Historical
production outputs are unchanged; campaign-wide effects on retrieved refractive
indices and merged distributions have not yet been established.

## What is calculated

`pops_csca` and `uhsas_csca` return collected scattering cross-section in square
micrometers, for a homogeneous sphere in air. They do not include laser power,
mirror reflectivity, detector gain, particle-position variability, or
instrument-specific calibration adjustments. Transmission is uniform inside
each modeled opening and zero outside it.

Let the laser travel along z and its electric field point along x. A collection
axis lies along y. Theta is the scattering angle from z. Phi is measured around
z from the central scattering plane containing y and z. A circular collection
cone with half-angle alpha accepts a direction when

```text
sin(theta) cos(phi) >= cos(alpha).
```

At a given theta the allowed azimuth is -beta to +beta, with

```text
beta(theta) = acos(cos(alpha) / sin(theta))
```

inside the cone, and zero outside. This gives 114 degrees of full azimuthal
width at theta=90 degrees for a 57 degree cone. The former disk-chord calculation
instead clipped this width to 180 degrees.

With miepython's `norm="qsca"`, define `Pperp = P11-P12` and
`Pparallel = P11+P12`. These already include scattering efficiency. The detected
intensity varies with phi because a fixed laser polarization cannot be
perpendicular to every scattering plane. The two azimuth integrals are

```text
Wperp     = integral[-beta,beta] cos(phi)^2 dphi = beta + sin(2 beta)/2
Wparallel = integral[-beta,beta] sin(phi)^2 dphi = beta - sin(2 beta)/2.
```

The code evaluates

```text
sigma_col = pi * radius_um^2
            * integral (Pperp*Wperp + Pparallel*Wparallel) sin(theta) dtheta.
```

The sine factor converts angle increments to solid angle. It is not already
part of the phase-matrix values or the azimuthal width. There is no additional
factor of one half. For an annulus, subtract the inner cone's two weights from
the outer cone's weights on the same theta grid.

## Instrument choices

- **POPS:** a 52 degree mirror cone centered at 90 degrees to the laser, using
  the published 38--142 degree limits. The mirror's curved shape is represented
  by those acceptance limits, not a flat disk placed at its vertex distance.
  The default is mirror-only. This follows the path included in Liu et al.'s
  Appendix A; it does not claim to reproduce that appendix's polarization
  approximation term for term. Direct PMT collection can be enabled only by
  supplying a nonzero `pmt_aperture_d_mm` and an explicit
  `pmt_aperture_distance_mm`. The mirror distance is never substituted.
- **UHSAS:** one collection arm with outer half-angle 57 degrees and central
  exclusion half-angle 14.8 degrees. The opposite arm is identical for a
  homogeneous sphere and changes only the overall scale. Cross-sections are
  per arm and per total incident irradiance. The two incoherent laser
  propagation directions give the same integral for this symmetric opening.
  Direct collection through the central opening is omitted as in Howell et al.

The mirror-diameter and vertex-distance fields retained in geometry objects are descriptive
dimensions; the collection half-angles determine the mirror acceptance. Theta
limit fields must cover those physical openings. The angular grid includes
exact opening boundaries, uses a maximum step of 0.25 degrees by default,
and has at least 256 intervals for a narrow optional direct aperture. Changing
resolution no longer changes the physical acceptance angles.

Sources:

- [Gao et al. (2016), optical layout and calibration](https://doi.org/10.1080/02786826.2015.1131809), Fig. 1 and pp. 89, 93--94.
- [Liu et al. (2021)](https://doi.org/10.5194/amt-14-6101-2021), Appendix A and Fig. A1.
- [Howell et al. (2021)](https://doi.org/10.5194/amt-14-7381-2021), Fig. 1 and Appendices A--B.
- [miepython normalization documentation](https://miepython.readthedocs.io/en/stable/03a_normalization.html).

## Diameter conversion

The response curve is still grouped in log diameter and fitted to increase
monotonically. Equal-response plateaus are represented by their mean log
diameter. That is a sizing approximation in the oscillatory Mie region, not a
unique physical solution to the ambiguity.

Previously the forward interpolation retained every plateau knot but the
inverse interpolation used different, collapsed knots. They were not inverses,
so even an unchanged refractive index could shift diameters. Both directions
now use the same collapsed-knot cubic. The inverse is found by bracketed
bisection of that actual forward curve, without extrapolation. The 100 response
groups selected for R2 are distinct from the angular integration resolution.
Number-conserving changes to distribution bin widths are unchanged.

## Rebuilding and reading LUTs

New tables have `optical_model_version="solid-angle-polarized-cones-v1"` and
`build_complete=True` only after calculation finishes. Existing destination
paths are rejected rather than overwritten. `SigmaLUT` and `sigma_query_zarr`
reject historical, unknown-version, or incomplete tables by default.

For a small local check (NOT a production-resolution LUT):

```python
from sizedistmerge import optical_diameter as od

od.build_pops_sigma_lut(
    "pops_mirror_only_check.zarr", od.POPSGeom(),
    D_range=(100., 3000., 100), n_range=(1.3, 1.8, 0.05),
    k_values=(0., 0.001), jobs_per_k=1,
)
lut = od.SigmaLUT("pops_mirror_only_check.zarr")
```

For deliberately inspecting historical tables:

```python
from sizedistmerge import SigmaLUT

old_lut = SigmaLUT("path/to/preserved/old_pops.zarr", allow_legacy=True)
```

That explicit option emits a warning. It does not correct the old table or
restore the historical inverse algorithm. Exact historical reproduction needs
the historical code as well. Production must use newly built and reviewed
tables at explicit paths. Do not resume a run made with old LUTs into new output.

The packaged tables contain 1000 logarithmically spaced diameters (60--6000 nm
for POPS; 30--6000 nm for UHSAS), 1001 real refractive indices from 1.30 to 1.80,
and 32 imaginary-index values from 0 to 0.8, including an exact 0.001 entry.
The angular step is at most 0.25 degrees. Rebuild them with
`notebooks/rebuild_optical_luts_solid_angle.ipynb`; the instrument detection
range is narrower than the calculation grid.

## Verification boundary

`tests/test_optical_geometry.py` checks exact circular-cone solid angles,
the Rayleigh polarized limit, a second Mie integral expressed in detector
coordinates, angular resolution, NumPy/Numba and parallel agreement,
forward/inverse consistency, and LUT build/version safeguards. These are
checks of the stated idealized model, not validation against measured
calibration data. Full production LUT rebuilds and campaign comparisons are
separate steps.

## Published curve comparisons

`scripts/plot_corrected_optical_response.py` plots the packaged tables using
Gao et al. (2016), Fig. 4 for POPS (1.615+0.001i and 1.45), and Fig. 2 of
[ARM's UHSAS artifact report](https://www.arm.gov/publications/programdocs/doe-sc-arm-tr-304.pdf)
for UHSAS (1.58, 1.50 and 1.40). That report is referenced by Section 7.5 of
the [UHSAS handbook](https://www.arm.gov/publications/tech_reports/handbooks/uhsas_handbook.pdf).
Plots retain calculated cross-section units rather than assigning an unknown
detector gain. The calculation grid is not the instrument detection range.

These are not exact reproductions of published calibration curves. In the
ARM report's approximate equal-signal example, a 570 nm particle at n=1.58
corresponds to 650 nm at n=1.50 or 830 nm at n=1.40. The present raw LUT gives
about 616 and 699 nm, respectively. This difference in inferred diameter
cannot be removed by a common signal scale factor. The report does not specify
the complete angular/polarization calculation needed to identify its cause.
Visual similarity alone must not be presented as quantitative validation.

A separate plot uses Howell et al.'s four weakly/nonabsorbing calibration
materials. Their aggregate-soot calculations are not represented by a
homogeneous-sphere LUT and are not reproduced here.
