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

- Bin geometry: `edges_from_mids_geometric()`, `mids_from_edges()`,
  `delta_log10_from_edges()`, `select_between()`.
- Spectrum units and moments: `counts_from_dndlog()`, `dndlog_from_counts()`,
  `dsdlog_from_dndlog()`, `dvdlog_from_dndlog()`.
- Count-conserving remapping: `remap_dndlog_by_edges()`,
  `remap_dndlog_by_edges_any()`, `rebin_dndlog_by_edges_overlap()`.

### Diameter Conversion

Implementation: `src/sizedistmerge/diameter_conversion.py`

Use `da_to_dv()` to convert aerodynamic diameter to volume-equivalent
diameter, for example when APS bins need to be remapped using particle density
and dynamic shape factor. Supporting helpers are `mean_free_path()` and
`cunningham()`.

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

- Instrument geometry: `POPSGeom`, `UHSASGeom`, `pops_geometry_cache()`,
  `uhsas_geometry_cache()`.
- Scattering response: `pops_csca()`, `pops_csca_parallel()`, `uhsas_csca()`,
  `uhsas_csca_parallel()`.
- LUT build/query: `build_sigma_lut()`, `build_pops_sigma_lut()`,
  `build_uhsas_sigma_lut()`, `SigmaLUT`, `sigma_query_zarr()`.
- LUT remapping: `make_monotone_sigma_interpolator()`, `convert_do_lut()`.
- Constants: `POPS_WAVELENGTH_NM`, `UHSAS_WAVELENGTH_NM`, `RI_UHSAS_SRC`,
  `RI_POPS_SRC`.

Packaged LUTs are available through `sdm.lut_path("pops")` and
`sdm.lut_path("uhsas")`.

### Alignment And Retrieved Parameters

Implementation: `src/sizedistmerge/alignment.py`

Use these functions to compare overlapping size-distribution regions and
retrieve alignment parameters such as refractive index or APS density.

- Overlap comparison: `mse_overlap_sizedist()`.
- Single-OPC fitting: `objective_opc_vs_ref()`,
  `optimize_refractive_index_for_opc()`.
- Multi-instrument fitting: `objective_multi_custom()`,
  `optimize_multi_custom()`.
- Temporal fitting helpers: `objective_joint_named_temporal()`,
  `temporal_parameter_penalty()`.

### Merging Onto A Common Grid

Implementation: `src/sizedistmerge/combine.py`

Use these functions to reconstruct a smooth aerosol size distribution from
multiple instruments on a common diameter grid.

- Grid and interpolation: `make_grid_from_series()`, `log_interp()`,
  `second_diff_nonuniform()`.
- Uncertainty and weights: `sigma_from_bands()`, `fractional_sigma()`,
  `compute_data_weights()`.
- Merge solvers: `merge_sizedists_tikhonov()`,
  `merge_sizedists_tikhonov_consensus()`.

### ICARTT And Instrument Tables

Implementation: `src/sizedistmerge/ict_utils.py`

These readers and table helpers are reusable outside ARCSIX production. They
handle ICARTT files, instrument size-bin metadata, spectra columns, time grids,
and flag segments.

- Generic ICARTT reads: `read_ict()`, `read_ict_file()`, `read_ict_dir()`,
  `pick_ict_files()`, `parse_bound()`.
- Time handling: `as_timezone_naive_timestamp()`, `timezone_naive_index()`,
  `drop_timezone_from_index()`.
- Instrument reads: `read_aps()`, `read_pops()`, `read_uhsas()`,
  `read_fims()`, `read_nmass()`, `read_inlet_flag()`, `read_microphysical()`.
- Spectra extraction: `label_size_bins()`, `get_spectra()`,
  `mean_spectrum()`, `number_to_surface_area_spectrum()`.
- Time/grid filtering: `check_common_grid()`, `align_to_common_grid()`,
  `filter_by_spectra_presence()`.
- Flag helpers: `find_flag_column()`, `flag_segments()`, `flag_fractions()`.

### Hygroscopicity

Implementation: `src/sizedistmerge/kappa_kohler.py`

This module contains kappa-Kohler and hygroscopic growth utilities.

- Petters and Kreidenweis equations: `kappa_petter_and_Kreidenweis_2010_EQ10()`,
  `Sc_petter_and_Kreidenweis_2010_EQ10()`,
  `Dd_petter_and_Kreidenweis_2010_EQ10()`,
  `S_petter_and_Kreidenweis_2010_EQ6()`.
- Critical size and kappa retrieval: `find_peak_S_D_binary_search()`,
  `calculate_critical_diameter_interpolated()`, `calculate_kappa_fitting()`,
  `calculate_kappa()`, `calculate_critical_diameter()`.
- Growth and optics helpers: `calculate_wet_diameter()`,
  `calculate_dry_diameter()`, `calculate_humidification_factor()`,
  `calculate_humidification_factor_ammonium_sulfate()`,
  `calculate_coefficients()`, `kappa_from_growth_factor()`,
  `growth_factor_from_kappa()`.
- Plot helper: `plot_Sc_Dd_base()`.

### Plotting

Implementation: `src/sizedistmerge/plot.py`

- `plot_size_distributions()`
- `plot_size_distributions_steps()`

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

- Period construction: `load_arcsix_merge_frames_for_day()`, `split_frames()`,
  `periods_from_split_frames()`, `periods_from_frames()`,
  `overlap_seconds_all_instruments()`.
- Chunk preparation: `plot_period_totals()`, `edges_from_meta_or_mids()`,
  `mean_spectrum_with_edges()`, `filter_chunk_by_inlet_flag()`,
  `make_filtered_specs()`, `chunk_is_incloud()`.
- Alignment and merge production: `run_joint_optimization()`,
  `make_tikhonov_merged_spec()`, `make_consensus_merged_spec()`,
  `plot_sizedist_all()`, `write_day_netcdf()`,
  `run_arcsix_merge_for_periods()`.
- Product QC and ICARTT: `find_merged_netcdf_files()`,
  `integrate_dndlog_gt_cutoff()`, `compute_robust_linear_bounds()`,
  `linear_resid_and_flag()`, `gather_post_merge_qc()`,
  `run_post_merge_product_qc()`, `write_icartt_from_netcdf()`,
  `convert_qc_netcdf_to_icartt()`.

## Notebooks

Current package examples are in `notebooks/`:

- `size_distribution_core_demo.ipynb` - data-free demo for core bin and
  remapping utilities.
- `optics_lut_demo.ipynb` - data-free demo for packaged POPS/UHSAS LUT usage.
- `arcsix_merge_1min_5min_package.ipynb` - ARCSIX 1-minute and 5-minute
  production recipe using the packaged library and campaign production module.

## Package Layout

- `src/sizedistmerge/utils.py` - bin geometry, `dN/dlogDp` conversions, and
  count-conserving remapping.
- `src/sizedistmerge/diameter_conversion.py` - aerodynamic-to-volume diameter
  conversion.
- `src/sizedistmerge/optical_diameter.py` - POPS/UHSAS optical response,
  LUT builders, and optical diameter remapping.
- `src/sizedistmerge/alignment.py` - overlap objectives, RI/density
  optimization, and temporal regularization helpers.
- `src/sizedistmerge/combine.py` - Tikhonov and consensus merge routines.
- `src/sizedistmerge/ict_utils.py` - ICARTT readers, instrument table helpers,
  spectra extraction, and shared time handling.
- `src/sizedistmerge/kappa_kohler.py` - kappa-Kohler and hygroscopic growth
  utilities.
- `src/sizedistmerge/plot.py` - reusable size-distribution plotting.
- `src/sizedistmerge/resources.py` - package-data lookup helpers such as
  `sdm.lut_path(...)`.
- `arcsix_production/arcsix_merge_production.py` - ARCSIX-specific production,
  product QC, and ICARTT conversion.

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
