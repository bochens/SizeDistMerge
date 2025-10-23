<!-- Top banner + right-aligned CSU logo (same pattern as your TAMU example) -->
<a href="https://www.atmos.colostate.edu/" target="_blank"> <img src="assets/CSU-Rams-Head-Symbol-357.jpg" align="right" height="90" alt="Colorado State University Atmospheric Science"> </a>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3137/)
[![Status](https://img.shields.io/badge/status-in%20development-orange.svg)]()

# SizeDistMerge
Python toolkit for merging aerosol size distributions measured by different aerosol sizing instruments.

## Repo Structure

### `dev/`

- **optical_diameter_core.py**
Utilities for computing the **optical response function** — the relationship between particle diameter and collected scattering amplitude — for various Optical Particle Counters (OPCs).  
Currently, **POPS** and **UHSAS** geometries are implemented using **Mie scattering**.

This module also provides tools for generating and interfacing with **lookup tables (LUTs)** of σ(D; m), which describe the scattering cross-section as a function of particle diameter and refractive index.  

The function `make_monotone_sigma_interpolator()` enforces monotonicity in the OPC response curve, allowing a one-to-one mapping between diameters at different refractive indices.  
This enables **remapping** (i.e., converting bin edges between refractive indices) while conserving particle counts.

The key function is **`convert_do_lut()`**, which:
- Takes an array of diameters (typically bin edges for rebinning `dN/dlogDp` data),
- Applies the OPC response function for a **source refractive index** (e.g., calibration conditions),
- And remaps to a **target refractive index**, producing a new set of equivalent bin edges that reflect how particle sizing shifts under different optical properties.

- **sizedist_optimization.py**  
Optimization routines that uses the bin-remapping functions to align aerosol size distribution and simutaneously retrieve aerosol properties (refractive index, density, and etc.) by minimizing the mean square error (mse) of overlapping regions of two or more size distribution.

- **sizedist_combine.py**  
Routines to reconstruct a smooth aerosol size distribution based on multiple size distributions onto a common grid.x

### OPC mie scattering LUT

- **pops_sigma_col_405nm.zarr**  
Calculated scattering cross-sections for POPS at **405 nm**, indexed by particle diameter and real and imaginary part of refractive index.

- **uhsas_sigma_col_1054nm.zarr**  
Similar dataset for UHSAS at **1054 nm**.

