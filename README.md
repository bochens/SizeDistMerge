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

## Important Functions

Most users should start with these call points. The package exposes them at the
top level as `sdm.function(...)`.

### Size Distribution Utilities

Implementation: `src/sizedistmerge/utils.py`

- `edges_from_mids_geometric()` - Infers geometric bin edges from bin
  midpoints.
- `counts_from_dndlog()` and `dndlog_from_counts()` - Convert between
  `dN/dlogDp` and per-bin counts.
- `remap_dndlog_by_edges_any()` - Rebins a spectrum onto a new edge grid while
  conserving particle counts.
- `dsdlog_from_dndlog()` and `dvdlog_from_dndlog()` - Convert number spectra to
  surface-area or volume spectra.

### Diameter Conversion

Implementation: `src/sizedistmerge/diameter_conversion.py`

- `da_to_dv()` - Converts aerodynamic diameter to volume-equivalent diameter,
  for example when APS bins are remapped using particle density and dynamic
  shape factor.

### Optical Diameter And LUTs

Implementation: `src/sizedistmerge/optical_diameter.py`

- `SigmaLUT` - Opens a stored POPS or UHSAS response lookup table.
- `lut_path()` - Returns the packaged POPS or UHSAS LUT path.
- `convert_do_lut()` - Maps an OPC diameter grid from one refractive index to
  another using a `SigmaLUT`.
- `make_monotone_sigma_interpolator()` - Builds the monotone optical-response
  mapping used by `convert_do_lut()`.
- `POPSGeom` and `UHSASGeom` - Define the instrument collection geometries used
  for response calculations.
- `build_pops_sigma_lut()` and `build_uhsas_sigma_lut()` - Rebuild POPS or
  UHSAS response LUTs when the optical-response grid needs to be regenerated.

Packaged LUTs are available through `sdm.lut_path("pops")` and
`sdm.lut_path("uhsas")`.

### Alignment And Retrieved Parameters

Implementation: `src/sizedistmerge/alignment.py`

- `mse_overlap_sizedist()` - Compares two spectra over their shared diameter
  range.
- `optimize_refractive_index_for_opc()` - Fits the refractive index that best
  aligns an OPC spectrum with a reference spectrum.
- `optimize_multi_custom()` - Optimizes multiple instrument parameters together,
  such as OPC refractive index and APS density.
- `objective_joint_named_temporal()` - Evaluates the UHSAS/POPS/APS alignment
  objective with optional temporal regularization.

### Merging Onto A Common Grid

Implementation: `src/sizedistmerge/combine.py`

- `make_grid_from_series()` - Builds a common diameter grid from one or more
  instrument grids.
- `compute_data_weights()` - Converts uncertainty estimates into weights for
  the merge solvers.
- `merge_sizedists_tikhonov()` - Solves a smooth Tikhonov-regularized merged
  distribution on a common grid.
- `merge_sizedists_tikhonov_consensus()` - Adds consensus reweighting to the
  Tikhonov merge so instrument agreement can influence the final solution.

### ICARTT And Instrument Tables

Implementation: `src/sizedistmerge/ict_utils.py`

- `read_ict()` - Reads one ICARTT file or a directory of ICARTT files using
  header metadata.
- `read_aps()`, `read_pops()`, `read_uhsas()`, and `read_fims()` - Read common
  aerosol instrument tables with size-bin labels.
- `get_spectra()` and `mean_spectrum()` - Extract time-resolved spectra or
  average spectra from bin-labeled columns.
- `align_to_common_grid()` - Aligns multiple frames onto a shared time grid.
- `filter_by_spectra_presence()` - Keeps only times with enough instruments
  reporting valid spectra.

### Hygroscopicity

Implementation: `src/sizedistmerge/kappa_kohler.py`

- `calculate_kappa()` - Solves kappa from critical diameter and critical
  supersaturation.
- `calculate_critical_diameter()` - Finds the dry diameter that activates for a
  given kappa and supersaturation.
- `calculate_wet_diameter()` and `calculate_dry_diameter()` - Convert between
  wet and dry particle diameter at a specified relative humidity.
- `kappa_from_growth_factor()` and `growth_factor_from_kappa()` - Convert
  between hygroscopic growth factor and kappa.
- `calculate_humidification_factor()` - Computes an optical humidification
  factor from dry spectrum, kappa, wavelength, and refractive index.

### Plotting

Implementation: `src/sizedistmerge/plot.py`

- `plot_size_distributions()` - Plots mean spectra with optional uncertainty
  bands as regular lines.
- `plot_size_distributions_steps()` - Plots mean spectra and uncertainty bands
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

Main production entry points:

- `run_arcsix_merge_for_periods()` - Runs the ARCSIX merge workflow for a list
  of explicit time periods.
- `load_arcsix_merge_frames_for_day()` and `periods_from_frames()` - Build
  period inputs from daily instrument data.
- `run_joint_optimization()` - Optimizes available OPC and APS parameters
  against the FIMS reference spectrum.
- `make_consensus_merged_spec()` - Builds the consensus-weighted merged
  spectrum for one period.
- `write_day_netcdf()` - Writes one day of merged spectra and diagnostics to a
  NetCDF product.
- `run_post_merge_product_qc()` - Writes QC plots, a QC table, and QC-flagged
  NetCDF files.
- `convert_qc_netcdf_to_icartt()` - Converts QC-flagged daily NetCDF files to
  ICARTT files.

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
