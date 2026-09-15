# Packaged optical lookup tables

The four tables below were rebuilt on 15 September 2026 using the shared
solid-angle/polarization integrator and the instrument TOML files. Each
contains 1000 diameter values, 1001 real refractive-index values, and 200
imaginary refractive-index values. The array order is diameter, real index,
imaginary index, and the stored scattering cross-section is in square
micrometres (float32).

| Instrument/model | File under `lut/` | Diameter grid (nm) | Angular step |
| --- | --- | --- | --- |
| POPS | `pops_sigma_col_405nm.zarr` | 60–6000 | 0.25° |
| UHSAS | `uhsas_sigma_col_1054nm.zarr` | 30–6000 | 0.25° |
| PCASP | `pcasp_sigma_col_632p8nm.zarr` | 60–6000 | 0.25° |
| GRIMM, provisional unpolarized model | `grimm_11d_unpolarized_assumed_655nm.zarr` | 200–40000 | 0.125° |

Diameter grids are logarithmically spaced calculation ranges, not claims
about instrument detection ranges. The real index spans 1.30–1.80 in steps
of 0.0005. The imaginary-index grid is exactly
`np.r_[0., np.geomspace(1e-4, 0.8, 199)]`. Unlike the earlier 32-value grid,
it does not include a specially inserted value of 0.001. The diameter and
real-index arrays are bit-for-bit unchanged from the preceding tables.

## Model and calibration boundaries

Every table stores its complete `optical_setup`, selected `response_channels`,
wavelength, units, normalization and optical-model version in its root Zarr
metadata. `SigmaLUT` reconstructs that saved setup, not whatever a TOML file
contains later. Calibration refractive indices are explicit inputs in the
user's production notebook; they are not universal instrument defaults.

POPS uses mirror-only collection. The UHSAS table represents Collection 1,
normalized to total incident irradiance, including both modeled beam
directions; the two detectors are not silently added. GRIMM retains the
1.109-derived assumed geometry and unpolarized illumination. Neither its
geometry nor its polarization is verified specifically for 11-D. These
assumptions are stated in both the TOML and the table metadata.

Our scattering tables are calculated with this repository's integrator,
not copied from another LUT dataset. Related work includes
[Formenti and Di Biagio (2026)](https://doi.org/10.5194/ar-2026-20);
that reference does not by itself validate the provisional 11-D setup.

## Reproduction and verification

Use `notebooks/build_optical_luts_example.ipynb`; builds are disabled until
explicitly enabled and write to a separate directory. Publication builds
used six workers and frozen copies of the code and TOMLs. POPS/UHSAS used
code based on commit `ae7a80b1f`; PCASP/GRIMM preparation was recorded at
`100481e98`. Machine-specific logs and backups remain local. The numerical
checks below were completed before replacing the active tables.

| Table | Direct calculation checks | Maximum relative storage error | Entire k=0 slice unchanged |
| --- | ---: | ---: | --- |
| POPS | 75 | 5.415 × 10⁻⁸ | Yes |
| UHSAS | 75 | 4.972 × 10⁻⁸ | Yes |
| PCASP | 75 | 5.422 × 10⁻⁸ | Yes |
| GRIMM | 75 | 5.798 × 10⁻⁸ | Yes |

Every stored value was checked to be finite and positive. Grid arrays,
saved optical setups, response channels, and completion markers were also
checked. The 75 sampled points per table cover five diameters, three real
indices and five imaginary indices. Comparing against the same integrator
checks storage/build consistency, not independent physical accuracy.
Separate optical tests cover solid angle, polarization, Rayleigh behavior,
and full-sphere scattering/extinction. Old tables were preserved locally.

These tables do not update an already completed campaign merge. Historical
R2 products retain their frozen 32-k LUTs and associated processing settings.
