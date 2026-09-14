# PCASP optical model

This is a nominal optical model for PCASP, not a calibration of a particular
aircraft instrument. It does not change POPS/UHSAS LUTs or ARCSIX production.

## Source and geometry

[Rosenberg et al. (2012)](https://doi.org/10.5194/amt-5-1147-2012),
Section 1.2 and Table 1, describe a 632.8 nm laser and a parabolic collector:

| Beam | Accepted scattering angles | Default fraction of total incident light |
| --- | --- | --- |
| Outgoing | 35-120 degrees | 0.5 |
| Returning | 60-145 degrees | 0.5 |

The paper treats collection as rotationally symmetric about the laser. These
are **two views of one physical collector**, not two detectors. In the shared
`OpticalSetup`, that collector is a 120-degree cone about the outgoing beam
minus the concentric 35-degree cone. Reversing the beam produces the second
angular range automatically. It is not a side-facing POPS/UHSAS cone.

The [DMT PCASP-100X manual, DOC-0228 Rev C (2017)](https://dropletmeasure.wpenginepowered.com/wp-content/uploads/2020/02/DOC-0228-Rev-C-PCASP-100X-Manual.pdf),
Section 2.1 and Figure 2, show the parabolic mirror, returning beam, focusing
optics, and detector. Section 7 specifies PSL calibration using an index of
1.58. This does **not** establish the calibration of every deployed PCASP;
the source index remains an explicit input when converting campaign data.

## Calculation and units

`setup_csca` with the PCASP setup, and `build_pcasp_sigma_lut`, use the shared polarized
Mie calculation for homogeneous spheres in air. They integrate the scattered
power per unit solid angle over the collector. Units are square micrometers.
Full-azimuth collection makes the integral independent of the chosen
transverse polarization direction.

The default beams have equal irradiance and are added incoherently (their
intensities, not their electric fields, are added). The paper describes a
crystal oscillator that prevents interference and reflects 99.9% of the
outgoing light, then uses equal weights in Table 1. The default follows that
table. `PCASPGeom(reflected_beam_ratio=.999)` permits the stated ratio instead.

Our cross-section is `collected power / total incident irradiance`, consistent
with the shared optical interface. Thus the equivalent angular weights are
0.5 over 35-60 degrees, 1 over 60-120 degrees, and 0.5 over 120-145 degrees.
Rosenberg's table uses 1, 2, and 1, relative to the outgoing beam. Multiply our
cross-section by **2** to compare with that convention under equal beams.
This is a known normalization change, not an empirical curve adjustment.
It cancels in size conversion if source and target curves use the same basis.

## Checks and limitations

- The independent reference is the authors' MieConScat 1.1.8 source, using
  Wiscombe's original Fortran MIEV0 solver and Rosenberg's `scatteringcs`
  wrapper. The two published angular integrals are averaged to put both
  calculations on the total-irradiance basis. Across 1,687 cases (241
  diameters from 60 to 6000 nm and seven explicit refractive indices), the
  maximum relative difference was **0.00760%**, the median **0.000208%**,
  and the 95th percentile **0.00180%**. The comparison includes absorbing
  and non-absorbing spheres. The reference uses 501 angular samples per
  interval and single-precision amplitudes; this model uses a 0.25-degree
  grid and double-precision amplitudes.
- A separate calculation integrates unnormalized Mie amplitudes in
  `cos(theta)` using Gaussian quadrature. It checks absolute units and the
  angular integral independently of the implementation's normalized phase
  matrix and trapezoidal grid.
- Coaxial bands are integrated only within their accepted interval. Their
  sharp rims are not smeared into adjacent uncollected angles.
- The existing POPS/UHSAS numerical paths remain unchanged, checked against
  their pre-interface saved values.
- Rosenberg Figure 1 provides a useful response-curve comparison, but is not
  measured detector calibration. Its caption lists material references rather
  than each numerical refractive index. Do not claim an exact reproduction
  of every material curve without those inputs.
- The model omits measured aperture transmission, mirror/lens losses, gain
  stages, particle-position effects, and instrument-specific optical offsets.
  A measured calibration remains necessary for absolute detector signals.
- A monotone curve, used for diameter conversion, is a chosen simplification
  of the oscillatory Mie response, not additional instrument information.

## Reproduce the comparison and build a table

The completed nominal table is packaged at
`lut/pcasp_sigma_col_632p8nm.zarr`. Open it with:

```python
from sizedistmerge import SigmaLUT, lut_path

pcasp_lut = SigmaLUT(lut_path("pcasp"))
```

It was built on 2026-09-12 from commit
`1331884e0b3097efe880dc8c29d08bf133274033` with the settings below.
All 32,032,000 cross-sections are finite and positive. Sampled stored values
agree with direct calculations to within 5.2e-8 relative error; this checks
table storage, not agreement with measured instrument calibration.
The table metadata records the geometry, units, irradiance convention, and
completion marker. The original build output is retained separately.

The original reference-comparison notebook and its run records are retained
locally. The checked reference values and source-archive hash remain in
`tests/test_pcasp_optics.py`.

The public [LUT build example](../notebooks/build_optical_luts_example.ipynb)
uses the shared setup interface for PCASP, POPS and UHSAS. Builds are disabled
by default and write to a separate directory. The extended diameter grid is a
calculation range, not a claim about the instrument's measurement range.
Response groups used for monotone conversion are separate from LUT diameter
grid points. The example does not replace existing tables or launch a campaign.

For a direct calculation without a table:

```python
from sizedistmerge.optical_diameter import pcasp_optical_setup, setup_csca

setup = pcasp_optical_setup()
sigma_um2 = setup_csca([100., 500., 1000., 3000.], 1.58+0j, setup)["Collection"]
```

The local paper-figure notebook passes this same setup to calculation and
drawing. That manuscript notebook and its generated figures are maintained
separately and are not distributed with this code change.
