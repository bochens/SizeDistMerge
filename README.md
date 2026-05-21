<a href="https://www.atmos.colostate.edu/" target="_blank">
  <img src="assets/CSU-Rams-Head-Symbol-357.jpg" align="right" height="90" alt="Colorado State University Atmospheric Science">
</a>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-%3E%3D3.12-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-in%20development-orange.svg)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17459971.svg)](https://doi.org/10.5281/zenodo.17459971)

# SizeDistMerge

SizeDistMerge is a Python toolkit for merging aerosol size distributions from
multiple sizing instruments. It provides reusable tools for particle-size bin
geometry, conservative spectrum remapping, aerodynamic-to-volume diameter
conversion, OPC optical response lookup tables, overlap-based alignment, and
smooth merged distributions on a common diameter grid.

The installable package is imported as:

```python
import sizedistmerge as sdm
```

ARCSIX campaign production code is kept outside the installable package in
`arcsix_production/`.

If you use this code in your research, please cite:
<https://zenodo.org/records/17459970>

SizeDistMerge is developed and maintained by the
[Kreidenweis Research Group](https://chem.atmos.colostate.edu/) at
[Colorado State University](https://www.atmos.colostate.edu/).

## Install

Install from the repository root:

```bash
python -m pip install -e .
```

The package requires Python `>=3.12`.

Optional `numba` support can be installed with:

```bash
python -m pip install -e ".[numba]"
```

## Quick Start

```python
import numpy as np
import sizedistmerge as sdm

mids_nm = np.array([10.0, 20.0, 40.0])
edges_nm = sdm.edges_from_mids_geometric(mids_nm)

dndlog = np.array([100.0, 80.0, 30.0])
counts = sdm.counts_from_dndlog(dndlog, edges_nm=edges_nm)
dndlog_again = sdm.dndlog_from_counts(counts, edges_nm=edges_nm)

dv_nm = sdm.da_to_dv(1000.0, rho_p=1000.0)
uhsas_lut = sdm.SigmaLUT(str(sdm.lut_path("uhsas")))
```

Most reusable functions are available directly as `sdm.function(...)`. The
submodules can also be imported when that is clearer:

```python
from sizedistmerge import optical_diameter, combine
```

## API Guide

### Size Distribution Utilities

Implementation: `src/sizedistmerge/utils.py`

Use these functions to describe log-spaced bins, convert between
`dN/dlogDp` and per-bin counts, and conservatively move spectra between grids.

- `edges_from_mids_geometric()` - Infers geometric bin edges from bin
  midpoints.
- `mids_from_edges()` - Returns geometric bin midpoints from bin edges.
- `delta_log10_from_edges()` - Returns each bin width in log10 diameter space.
- `select_between()` - Selects bins whose full edge interval is within a
  requested diameter range.
- `counts_from_dndlog()` - Converts `dN/dlogDp` values to per-bin counts.
- `dndlog_from_counts()` - Converts per-bin counts back to `dN/dlogDp`.
- `dsdlog_from_dndlog()` - Converts number spectra to surface-area spectra.
- `dvdlog_from_dndlog()` - Converts number spectra to volume spectra.
- `remap_dndlog_by_edges()` - Remaps a spectrum between same-length edge grids
  while conserving counts in each bin.
- `remap_dndlog_by_edges_any()` - Rebins a spectrum onto a different edge grid
  by log-space overlap while conserving counts where bins overlap.
- `rebin_dndlog_by_edges_overlap()` - Rebins by log-space overlap and can mark
  bins invalid when coverage is below a requested threshold.

### Diameter Conversion

Implementation: `src/sizedistmerge/diameter_conversion.py`

- `mean_free_path()` - Computes gas mean free path from pressure and
  temperature.
- `cunningham()` - Computes the Cunningham slip correction for particles in air.
- `da_to_dv()` - Converts aerodynamic diameter to volume-equivalent diameter,
  for example when APS bins are remapped using density and shape factor.

### Optical Diameter And LUTs

Implementation: `src/sizedistmerge/optical_diameter.py`

This module computes OPC optical response: particle diameter to scattering
cross-section integrated over the collection geometry. POPS and UHSAS
geometries are implemented. It also builds and reads lookup tables of
`sigma(D; m)` as a function of particle diameter and complex refractive index.

The main remapping function is `convert_do_lut()`. It maps an optical diameter
axis, usually bin edges, from a source refractive index to a target refractive
index using a precomputed LUT. `make_monotone_sigma_interpolator()` enforces a
one-to-one response curve so the mapping is invertible.

- `POPSGeom` - Stores the POPS optical collection geometry used in scattering
  calculations.
- `UHSASGeom` - Stores the UHSAS optical collection geometry used in scattering
  calculations.
- `pops_geometry_cache()` - Precomputes reusable POPS geometry terms for
  repeated response calculations.
- `uhsas_geometry_cache()` - Precomputes reusable UHSAS geometry terms for
  repeated response calculations.
- `pops_csca()` - Computes POPS collection-angle scattering response for a
  particle size and refractive index.
- `pops_csca_parallel()` - Computes POPS scattering response over many
  diameters using parallel workers.
- `uhsas_csca()` - Computes UHSAS collection-angle scattering response for a
  particle size and refractive index.
- `uhsas_csca_parallel()` - Computes UHSAS scattering response over many
  diameters using parallel workers.
- `build_sigma_lut()` - Builds a `sigma(D; m)` lookup table for a supplied OPC
  response function.
- `build_pops_sigma_lut()` - Builds a POPS response lookup table at the package
  POPS wavelength.
- `build_uhsas_sigma_lut()` - Builds a UHSAS response lookup table at the
  package UHSAS wavelength.
- `SigmaLUT` - Opens a stored Zarr LUT and provides interpolated sigma curves.
- `sigma_query_zarr()` - Queries one LUT response value for diameter and
  refractive index.
- `make_monotone_sigma_interpolator()` - Builds monotone `sigma(D)` and inverse
  `D(sigma)` mappings for optical diameter conversion.
- `convert_do_lut()` - Maps an OPC diameter grid from one refractive index to
  another using a `SigmaLUT`.
- `POPS_WAVELENGTH_NM` - Gives the POPS wavelength used for packaged response
  calculations.
- `UHSAS_WAVELENGTH_NM` - Gives the UHSAS wavelength used for packaged response
  calculations.
- `RI_UHSAS_SRC` - Gives the package default source refractive index for UHSAS
  bin remapping.
- `RI_POPS_SRC` - Gives the package default source refractive index for POPS
  bin remapping.

Packaged LUTs are available through `sdm.lut_path("pops")` and
`sdm.lut_path("uhsas")`.

### Alignment And Retrieved Parameters

Implementation: `src/sizedistmerge/alignment.py`

Use these functions to compare overlapping size-distribution regions and
retrieve alignment parameters such as refractive index or APS density.

- `mse_overlap_sizedist()` - Compares two spectra over their shared diameter
  range using mean squared error.
- `objective_opc_vs_ref()` - Evaluates mismatch after remapping an OPC spectrum
  to a trial refractive index against a reference spectrum.
- `optimize_refractive_index_for_opc()` - Fits the complex refractive index
  that best aligns an OPC spectrum with a reference spectrum.
- `objective_multi_custom()` - Evaluates a weighted multi-instrument mismatch
  after applying each instrument remapping function.
- `optimize_multi_custom()` - Optimizes heterogeneous instrument parameters
  such as OPC refractive index and APS density together.
- `objective_joint_named_temporal()` - Evaluates the named UHSAS/POPS/APS
  alignment objective with optional temporal regularization.
- `temporal_parameter_penalty()` - Penalizes jumps from a previous parameter
  estimate when temporal smoothing is requested.

### Merging Onto A Common Grid

Implementation: `src/sizedistmerge/combine.py`

Use these functions to reconstruct a smooth aerosol size distribution from
multiple instruments on a common diameter grid.

- `log_interp()` - Interpolates spectra in log-diameter space.
- `second_diff_nonuniform()` - Builds a second-difference curvature operator
  for a nonuniform log-diameter grid.
- `make_grid_from_series()` - Creates a common diameter grid from one or more
  instrument grids.
- `sigma_from_bands()` - Estimates uncertainty from lower and upper uncertainty
  bands.
- `fractional_sigma()` - Creates a fractional uncertainty array from a spectrum.
- `compute_data_weights()` - Converts uncertainty estimates into data weights
  for the merge solver.
- `merge_sizedists_tikhonov()` - Solves a smooth Tikhonov-regularized merged
  distribution on a common grid.
- `merge_sizedists_tikhonov_consensus()` - Adds consensus reweighting to the
  Tikhonov merge so instrument agreement can influence the final solution.

### ICARTT And Instrument Tables

Implementation: `src/sizedistmerge/ict_utils.py`

These readers and table helpers are reusable outside ARCSIX production. They
handle ICARTT files, instrument size-bin metadata, spectra columns, time grids,
and flag segments.

- `read_ict()` - Reads one ICARTT file or a directory of ICARTT files using
  header metadata.
- `read_ict_file()` - Reads one ICARTT file and filters it by optional time
  bounds.
- `read_ict_dir()` - Reads all matching ICARTT files in a directory and
  concatenates them.
- `pick_ict_files()` - Selects instrument files that cover requested dates or
  time bounds.
- `parse_bound()` - Parses a start or end bound as a timezone-naive timestamp.
- `as_timezone_naive_timestamp()` - Converts a datetime-like value to a
  timezone-naive timestamp without shifting wall-clock time.
- `timezone_naive_index()` - Drops timezone metadata from a `DatetimeIndex`
  without shifting wall-clock time.
- `drop_timezone_from_index()` - Returns a DataFrame copy with a timezone-naive
  datetime index when needed.
- `read_aps()` - Reads APS ICARTT data and labels aerodynamic size bins.
- `read_pops()` - Reads POPS ICARTT data and labels optical size bins.
- `read_uhsas()` - Reads UHSAS ICARTT data and labels optical size bins.
- `read_fims()` - Reads FIMS ICARTT data and labels mobility-diameter size
  bins.
- `read_nmass()` - Reads NMASS condensation-nuclei channel data.
- `read_inlet_flag()` - Reads inlet flag data for filtering or diagnostics.
- `read_microphysical()` - Reads LARGE microphysical data such as CPC totals.
- `label_size_bins()` - Relabels spectrum columns using bin metadata from the
  ICARTT header.
- `get_spectra()` - Extracts time-resolved spectral columns without averaging.
- `mean_spectrum()` - Computes arithmetic mean spectra and standard deviations
  for bin-labeled columns.
- `number_to_surface_area_spectrum()` - Converts number spectra to surface-area
  spectra.
- `check_common_grid()` - Checks whether multiple frames share the same time
  index.
- `align_to_common_grid()` - Aligns frames onto a shared time grid with optional
  interpolation.
- `filter_by_spectra_presence()` - Keeps times with enough instruments
  reporting valid spectral data.
- `find_flag_column()` - Finds the likely flag column in a flag DataFrame.
- `flag_segments()` - Converts a time-indexed flag series into contiguous flag
  segments.
- `flag_fractions()` - Computes the fraction of total segment duration spent in
  each flag value.

### Hygroscopicity

Implementation: `src/sizedistmerge/kappa_kohler.py`

This module contains kappa-Kohler and hygroscopic growth utilities.

- `kappa_petter_and_Kreidenweis_2010_EQ10()` - Calculates kappa from critical
  diameter and critical supersaturation using Petters and Kreidenweis Eq. 10.
- `Sc_petter_and_Kreidenweis_2010_EQ10()` - Calculates critical
  supersaturation from dry diameter and kappa.
- `Dd_petter_and_Kreidenweis_2010_EQ10()` - Calculates critical dry diameter
  from kappa and critical supersaturation.
- `S_petter_and_Kreidenweis_2010_EQ6()` - Calculates equilibrium saturation
  ratio from wet diameter, dry diameter, and kappa.
- `find_peak_S_D_binary_search()` - Finds the peak saturation ratio for one or
  more dry diameters.
- `calculate_critical_diameter_interpolated()` - Estimates activation critical
  diameter from aerosol size distributions and CCN concentrations.
- `calculate_kappa_fitting()` - Fits kappa from critical diameter and critical
  supersaturation.
- `calculate_kappa()` - Solves kappa from critical diameter and critical
  supersaturation.
- `calculate_critical_diameter()` - Finds the smallest dry diameter that
  activates for a given kappa and supersaturation.
- `calculate_wet_diameter()` - Computes wet diameter at a requested relative
  humidity.
- `calculate_dry_diameter()` - Inverts the growth calculation to recover dry
  diameter from wet diameter and relative humidity.
- `calculate_humidification_factor()` - Calculates optical humidification
  factor from dry spectrum, kappa, wavelength, and refractive index.
- `calculate_humidification_factor_ammonium_sulfate()` - Calculates
  humidification factor using ammonium-sulfate optical assumptions.
- `calculate_coefficients()` - Calculates optical coefficients from size
  distribution, refractive index, and wavelength.
- `kappa_from_growth_factor()` - Solves kappa from growth factor and relative
  humidity.
- `growth_factor_from_kappa()` - Computes growth factor from kappa and relative
  humidity.
- `plot_Sc_Dd_base()` - Plots the critical supersaturation and dry-diameter
  reference relationship.

### Plotting

Implementation: `src/sizedistmerge/plot.py`

- `plot_size_distributions()` - Plots mean spectra with optional one-standard
  deviation bands as regular lines.
- `plot_size_distributions_steps()` - Plots mean spectra with uncertainty bands
  as edge-aligned step distributions.

## ARCSIX Production Workflow

ARCSIX production is provided as a campaign-specific module outside the
installable library API. Run it from the repository root:

```python
import arcsix_production.arcsix_merge_production as mp

mp.run_arcsix_merge_for_periods(...)
mp.run_post_merge_product_qc(...)
mp.convert_qc_netcdf_to_icartt(...)
```

The production module handles ARCSIX directory conventions, the
FIMS/UHSAS/POPS/APS production combination, inlet/cloud filtering, batch logs,
diagnostic plots, NetCDF writing, post-merge QC, and ICARTT conversion.

Important production call points:

- `load_arcsix_merge_frames_for_day()` - Loads one ARCSIX day for period
  discovery using the production reader settings.
- `split_frames()` - Splits loaded instrument frames into fixed-duration
  chunks.
- `periods_from_split_frames()` - Converts split chunks into explicit
  `(start, end)` periods.
- `periods_from_frames()` - Builds fixed-width merge periods directly from
  loaded instrument frames.
- `overlap_seconds_all_instruments()` - Counts same-grid seconds where every
  requested instrument has data.
- `plot_period_totals()` - Plots time-series totals for a merge period and
  optional diagnostic overlays.
- `edges_from_meta_or_mids()` - Gets bin edges from metadata or infers them
  from bin midpoints.
- `mean_spectrum_with_edges()` - Computes a mean spectrum with matching bin
  edges and uncertainty.
- `filter_chunk_by_inlet_flag()` - Removes inlet-flagged samples from each
  instrument chunk.
- `make_filtered_specs()` - Averages filtered instrument chunks into spectra
  and plotting metadata.
- `chunk_is_incloud()` - Flags whether a merge period intersects inlet-flagged
  cloud time.
- `run_joint_optimization()` - Optimizes available OPC and APS parameters
  against the FIMS reference spectrum.
- `make_tikhonov_merged_spec()` - Builds a Tikhonov-regularized merged spectrum
  for one period.
- `make_consensus_merged_spec()` - Builds a consensus-weighted Tikhonov merged
  spectrum for one period.
- `plot_sizedist_all()` - Plots number and volume size distributions for raw,
  aligned, and merged spectra.
- `write_day_netcdf()` - Writes one day of merged spectra and diagnostics to a
  NetCDF product.
- `run_arcsix_merge_for_periods()` - Runs the ARCSIX merge workflow for a list
  of explicit time periods.
- `find_merged_netcdf_files()` - Finds raw daily merge NetCDF files under a
  production output directory.
- `integrate_dndlog_gt_cutoff()` - Integrates merged `dN/dlogDp` above a
  requested particle diameter cutoff.
- `compute_robust_linear_bounds()` - Computes robust residual bounds using
  median and MAD statistics.
- `linear_resid_and_flag()` - Computes merged-minus-CPC residuals and warning
  flags outside supplied bounds.
- `gather_post_merge_qc()` - Collects chunk-level QC quantities from raw merge
  NetCDF files.
- `run_post_merge_product_qc()` - Writes QC plots, a QC table, and QC-flagged
  NetCDF files.
- `write_icartt_from_netcdf()` - Writes one ICARTT file from one QC-flagged
  NetCDF file.
- `convert_qc_netcdf_to_icartt()` - Converts all QC-flagged daily NetCDF files
  in a directory to ICARTT files.

## Notebooks

Current package examples are in `notebooks/`:

- `size_distribution_core_demo.ipynb` - data-free demo for core bin and
  remapping utilities.
- `optics_lut_demo.ipynb` - data-free demo for packaged POPS/UHSAS LUT usage.
- `arcsix_merge_1min_5min_package.ipynb` - ARCSIX 1-minute and 5-minute
  production recipe using the packaged library and campaign production module.

## Repository Structure

- `src/sizedistmerge/` - installable library package.
- `src/sizedistmerge/data/lut/` - packaged POPS and UHSAS lookup tables.
- `arcsix_production/` - ARCSIX-specific production workflows and product
  writers.
- `notebooks/` - runnable examples for package utilities and ARCSIX recipes.
- `tests/` - package import, data, architecture, and behavior tests.

## Packaged LUT Data

The package includes POPS and UHSAS LUTs under `src/sizedistmerge/data/lut/`:

- `pops_sigma_col_405nm.zarr`
- `uhsas_sigma_col_1054nm.zarr`

Use:

```python
pops_lut_path = sdm.lut_path("pops")
uhsas_lut_path = sdm.lut_path("uhsas")
```

## License

This project is distributed under the MIT license. See `LICENSE.txt`.
