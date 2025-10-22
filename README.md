<!-- Top banner + right-aligned CSU logo (same pattern as your TAMU example) -->
<a href="https://www.atmos.colostate.edu/" target="_blank"> <img src="assets/CSU-Rams-Head-Symbol-357.jpg" align="right" height="90" alt="Colorado State University Atmospheric Science"> </a>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3137/)
[![Status](https://img.shields.io/badge/status-in%20development-orange.svg)]()

# SizeDistMerge
Python toolkit for merging aerosol size distributions measured by different aerosol sizing instruments.

## Repo Structure

### `dev/`

- **opc_profiles.py**  
Utilities for remapping bin diameter of optical particle counters (opc) based on changing refractive index assumption. Calculate, store, and read look up table (LUT) for diameter-scattering crosss-seciton relations.

- **sizedist_optimization.py**  
Optimization routines that uses the bin-remapping functions to align aerosol size distribution and simutaneously retrieve aerosol properties (refractive index, density, and etc.) by minimizing the mean square error (mse) of overlapping regions of two or more size distribution.

- **sizedist_combine.py**  
Routines to reconstruct a smooth aerosol size distribution based on multiple size distributions onto a common grid.x

### OPC mie scattering LUT

- **pops_sigma_col_405nm.zarr**  
Calculated scattering cross-sections for POPS at **405 nm**, indexed by particle diameter and real and imaginary part of refractive index.

- **uhsas_sigma_col_1054nm.zarr**  
Similar dataset for UHSAS at **1054 nm**.

