<!-- Top banner + right-aligned CSU logo (same pattern as your TAMU example) -->
<a href="https://www.atmos.colostate.edu/" target="_blank"> <img src="assets/CSU-Rams-Head-Symbol-357.jpg" align="right" height="90" alt="Colorado State University Atmospheric Science"> </a>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3137/)
[![Status](https://img.shields.io/badge/status-in%20development-orange.svg)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17459971.svg)](https://doi.org/10.5281/zenodo.17459971)

# SizeDistMerge
Python toolkit for merging aerosol size distributions measured by different aerosol sizing instruments.

If you use this code in your research, please **cite it using the DOI: https://zenodo.org/records/17459970**.

Code currently developed and maintained by [Kreidenweis Research Group](https://chem.atmos.colostate.edu/) at [Colorado State University](https://www.atmos.colostate.edu/). Collaboration welcome—issues and pull requests appreciated.

## Install

SizeDistMerge is now structured as an installable Python package.

```bash
python -m pip install -e .
```

The package requires Python `>=3.12`. The optional `numba` dependency can be
installed with:

```bash
python -m pip install -e ".[numba]"
```

## Basic Usage

Reusable library functions are available from the top-level package:

```python
import sizedistmerge as sdm

edges = sdm.edges_from_mids_geometric([10, 20, 40])
dv = sdm.da_to_dv(1000.0, rho_p=1000.0)
lut = sdm.SigmaLUT(str(sdm.lut_path("uhsas")))
```

The top-level API is intended for reusable size-distribution, diameter,
optical/LUT, alignment, and merge utilities. Campaign production workflows are
kept under explicit ARCSIX imports.

## Function And Module Guide

Most reusable functions are exported at the top level, so users can call
functions such as `sdm.convert_do_lut(...)` after
`import sizedistmerge as sdm`. The module paths below show where the
implementation lives.

### Map From The Old README

The old README described the project by loose `src/*.py` files. Those files are
now either packaged under `src/sizedistmerge/` or archived under
`legacy_src_storage/`.

- Old `optical_diameter_core.py` is now
  `src/sizedistmerge/optical_diameter.py`. The old README's core functions are
  still present as `sdm.make_monotone_sigma_interpolator()` and
  `sdm.convert_do_lut()`. The same module also contains `sdm.POPSGeom`,
  `sdm.UHSASGeom`, `sdm.SigmaLUT`, `sdm.pops_csca()`, `sdm.uhsas_csca()`,
  and the POPS/UHSAS LUT builders.
- Old `diameter_conversion_core.py` is now
  `src/sizedistmerge/diameter_conversion.py`. The old README's APS remapping
  function is still `sdm.da_to_dv()`, with `sdm.mean_free_path()` and
  `sdm.cunningham()` as the supporting gas/drag corrections.
- Old `sizedist_utils.py` is now `src/sizedistmerge/utils.py`. The important
  helpers are the bin geometry functions, `dN/dlogDp` count conversions, and
  count-conserving remapping functions listed below.
- Old `sizedist_alignment.py` is now `src/sizedistmerge/alignment.py`. The
  equivalent public functions are `sdm.mse_overlap_sizedist()`,
  `sdm.objective_opc_vs_ref()`, `sdm.optimize_refractive_index_for_opc()`,
  `sdm.objective_multi_custom()`, `sdm.optimize_multi_custom()`, and
  `sdm.objective_joint_named_temporal()`.
- Old `sizedist_combine.py` is now `src/sizedistmerge/combine.py`. The main
  merge solvers are `sdm.merge_sizedists_tikhonov()` and
  `sdm.merge_sizedists_tikhonov_consensus()`, with shared-grid and weighting
  helpers listed below.
- Old `ict_utils.py` is now `src/sizedistmerge/ict_utils.py`. The ICARTT
  readers, instrument readers, spectra extraction, time handling, common-grid
  alignment, and flag helpers are still reusable package code.
- Old `sizedist_plot.py` is now `src/sizedistmerge/plot.py`, with
  `sdm.plot_size_distributions()` and `sdm.plot_size_distributions_steps()`.
- Old `kappa_kohler_theory.py` is now `src/sizedistmerge/kappa_kohler.py`.
  The Petters and Kreidenweis equations, kappa retrieval helpers, wet/dry
  diameter solvers, and growth-factor conversions are still exported.
- Old `merge_production.py` is now
  `arcsix_production/arcsix_merge_production.py`. It is intentionally loose
  ARCSIX production code, not installable library API. Old v1/v2-style entry
  points were consolidated into `run_arcsix_merge_for_periods()` plus
  arguments such as `instruments=...` and `apply_alignment=...`.
- Old `forward_kernel.py` and `twomey_inversion.py` are archived in
  `legacy_src_storage/`. They were experimental Twomey/testing code and are
  intentionally not part of the current public package API.

Old ARCSIX production names were also consolidated:

- `load_aufi_oneday()`, `load_aufi_oneday_v2()`, and `load_af_oneday()` map to
  `load_arcsix_merge_frames_for_day()` for normal use. The lower-level readers
  are `read_arcsix_merge_instruments_for_day()` and
  `read_arcsix_aps_fims_for_day()`.
- `make_filtered_specs_v2()` maps to `make_filtered_specs()`.
- `run_joint_optimization_v2()` maps to `run_joint_optimization()`.
- `make_consensus_merged_spec_v2()` maps to `make_consensus_merged_spec()`.
- `write_day_netcdf_v2()` maps to `write_day_netcdf()`.
- `run_arcsix_merge_for_periods_fims_aps_only()` maps to
  `run_arcsix_merge_for_periods(..., instruments=("FIMS", "APS"),
  apply_alignment=False)`.

The archived Twomey/testing functions `K_tophat()`, `K_opc_from_lut()`,
`apply_kernel_counts()`, and `twomey_inversion()` do not have current public
package equivalents.

### Size Distribution Utilities

`src/sizedistmerge/utils.py` contains the core helpers for aerosol size
distributions:

- Bin geometry: `edges_from_mids_geometric()`, `mids_from_edges()`,
  `delta_log10_from_edges()`, `select_between()`.
- Spectrum units and moments: `counts_from_dndlog()`, `dndlog_from_counts()`,
  `dsdlog_from_dndlog()`, `dvdlog_from_dndlog()`.
- Count-conserving remapping: `remap_dndlog_by_edges()`,
  `remap_dndlog_by_edges_any()`, `rebin_dndlog_by_edges_overlap()`.

### Optical Diameter And LUTs

`src/sizedistmerge/optical_diameter.py` contains the OPC optical response
tools. It computes the relationship between particle diameter and scattering
signal using Mie theory, integrated over the instrument collection geometry.
POPS and UHSAS geometries are implemented.

The same module also builds and reads lookup tables of scattering response
`sigma(D; m)` as a function of particle diameter and refractive index.
`make_monotone_sigma_interpolator()` enforces a one-to-one response curve so
diameter bin edges can be remapped between refractive indices. The main remap
function is `convert_do_lut()`, which uses precomputed LUTs for speed.

Important call points:

- Instrument geometry: `POPSGeom`, `UHSASGeom`, `pops_geometry_cache()`,
  `uhsas_geometry_cache()`.
- Scattering response: `pops_csca()`, `pops_csca_parallel()`, `uhsas_csca()`,
  `uhsas_csca_parallel()`.
- LUT build/query: `build_sigma_lut()`, `build_pops_sigma_lut()`,
  `build_uhsas_sigma_lut()`, `SigmaLUT`, `sigma_query_zarr()`.
- LUT remapping: `make_monotone_sigma_interpolator()`, `convert_do_lut()`.
- Constants: `POPS_WAVELENGTH_NM`, `UHSAS_WAVELENGTH_NM`, `RI_UHSAS_SRC`,
  `RI_POPS_SRC`.

Packaged LUT paths are provided by `sdm.lut_path("pops")` and
`sdm.lut_path("uhsas")`.

### Diameter Conversion

`src/sizedistmerge/diameter_conversion.py` contains `da_to_dv()`, which
converts aerodynamic diameter to volume-equivalent diameter. This is used when
APS bins need to be remapped based on particle density and dynamic shape
factor. Supporting physics helpers are `mean_free_path()` and `cunningham()`.

### Alignment And Retrieved Parameters

`src/sizedistmerge/alignment.py` contains overlap-cost and optimization
routines for aligning size distributions. These functions remap bins, compare
overlapping regions with mean squared error, and retrieve instrument-dependent
parameters such as refractive index and APS density.

Important call points:

- Overlap comparison: `mse_overlap_sizedist()`.
- Single-OPC fitting: `objective_opc_vs_ref()`,
  `optimize_refractive_index_for_opc()`.
- Multi-instrument fitting: `objective_multi_custom()`,
  `optimize_multi_custom()`.
- Named temporal fitting for UHSAS/POPS/APS: `objective_joint_named_temporal()`,
  `temporal_parameter_penalty()`.

### Merging Onto A Common Grid

`src/sizedistmerge/combine.py` reconstructs a smooth merged aerosol size
distribution from multiple instruments on a shared diameter grid. It contains
the Tikhonov and consensus merge helpers.

Important call points:

- Grid and interpolation: `make_grid_from_series()`, `log_interp()`,
  `second_diff_nonuniform()`.
- Uncertainty/weight helpers: `sigma_from_bands()`, `fractional_sigma()`,
  `compute_data_weights()`.
- Merge solvers: `merge_sizedists_tikhonov()`,
  `merge_sizedists_tikhonov_consensus()`.

### ICARTT And Instrument Tables

`src/sizedistmerge/ict_utils.py` reads ICARTT files and instrument tables,
extracts size-bin metadata, handles spectra columns, aligns time grids, and
provides shared time helpers. POPS, UHSAS, APS, FIMS, NMASS, inlet flag, and
microphysical readers live here because those readers are reusable outside
ARCSIX production.

Important call points:

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

`src/sizedistmerge/kappa_kohler.py` contains kappa-Kohler and hygroscopic
growth utilities.

Important call points:

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

`src/sizedistmerge/plot.py` contains reusable plotting helpers for aerosol size
distributions: `plot_size_distributions()` and
`plot_size_distributions_steps()`.

### Package Data

`src/sizedistmerge/resources.py` contains `lut_path()`, which returns packaged
POPS or UHSAS LUT paths. Use `sdm.lut_path("pops")` or
`sdm.lut_path("uhsas")` instead of hard-coded relative paths.

### ARCSIX Production Code

`arcsix_production/arcsix_merge_production.py` is loose campaign code, not part
of the installable package. It contains the ARCSIX-specific production workflow:
directory conventions, the FIMS/UHSAS/POPS/APS production combination,
inlet/cloud filtering, batch logs, diagnostic plots, NetCDF writing,
post-merge QC, and ICARTT conversion.

Important call points:

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

## ARCSIX Production Workflow

ARCSIX batch production, post-merge QC, and ICARTT conversion are kept as
loose campaign code outside the installable package:

```python
from pathlib import Path
import importlib.util
import sys

production_path = Path("arcsix_production/arcsix_merge_production.py").resolve()
spec = importlib.util.spec_from_file_location("arcsix_merge_production", production_path)
mp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mp
spec.loader.exec_module(mp)

mp.run_arcsix_merge_for_periods(...)
mp.run_post_merge_product_qc(...)
mp.convert_qc_netcdf_to_icartt(...)
```

The current package notebook
`notebooks/arcsix_merge_1min_5min_package.ipynb` shows the 5-minute and
1-minute ARCSIX recipes using the packaged API. The notebook keeps run-specific
settings visible, while reusable mechanics live in `.py` modules.

Small data-free API demos are in:

- `notebooks/size_distribution_core_demo.ipynb`
- `notebooks/optics_lut_demo.ipynb`

## Package Layout

The import package is under `src/sizedistmerge/`:

- `utils.py` - size-bin geometry, `dN/dlogDp` conversions, and count-conserving remapping.
- `diameter_conversion.py` - aerodynamic-to-volume-equivalent diameter conversion.
- `kappa_kohler.py` - kappa-Kohler and hygroscopic-growth utilities.
- `optical_diameter.py` - POPS/UHSAS Mie-response geometry, LUT builders, and optical diameter remapping.
- `alignment.py` - MSE overlap objectives, RI/density optimization, and temporal regularization helpers.
- `combine.py` - Tikhonov and consensus merge routines on common grids.
- `ict_utils.py` - ICARTT readers, instrument table helpers, and shared time handling.
- `plot.py` - size-distribution plotting helpers.
- `resources.py` - package-data lookup helpers such as `sdm.lut_path(...)`.

Old flat scripts are stored separately under `legacy_src_storage/` for
reference. Old exploratory notebooks are kept locally under
`legacy_notebooks/` and are intentionally ignored by git. New code should
import reusable utilities from `sizedistmerge`. ARCSIX campaign production
lives outside the package in `arcsix_production/arcsix_merge_production.py`.

## Packaged LUT Data

The POPS and UHSAS LUTs are packaged inside the wheel under
`src/sizedistmerge/data/lut/`:

- `pops_sigma_col_405nm.zarr`
- `uhsas_sigma_col_1054nm.zarr`

Use `sdm.lut_path("pops")` or `sdm.lut_path("uhsas")` instead of hard-coded
relative paths.
