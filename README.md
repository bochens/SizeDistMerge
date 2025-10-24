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
  Contains functions for computing the **optical response function** — the relationship between particle diameter and the scattering amplitude (the scattering cross-section integrated over the opc collection angle) — for various Optical Particle Counters (OPCs). Currently, **POPS** and **UHSAS** geometries are implemented using **Mie scattering**.  

  This module also provides tools for generating and interfacing with **lookup tables (LUTs)** of σ(D; m) as a function of particle diameter and refractive index.  

  The function `make_monotone_sigma_interpolator()` enforces monotonicity in the OPC response curve, allowing a one-to-one mapping between diameters at different refractive indices. This enables **remapping** (i.e., converting bin edges between refractive indices) while conserving particle counts.

  The key function is **`convert_do_lut()`**, which:
  - Takes an array of optical diameters (typically bin edges for rebinning `dN/dlogDp` data),
  - Applies the OPC response function for a **source refractive index**,
  - And remaps to a **target refractive index**, producing a new set of equivalent bin edges that reflect how particle sizing shifts under different optical properties.
  - Use pre-computed LUT for speed.

- **diameter-conversion-core.py** 
  Contains `da_to_dv()` to convert aerodynamic diameter (da) to volume equivalent diameter, and is currently used for remapping APS instrument bins based on density and Chi.

  Will add more functions to convert among mobility diameter (db), da and dv.

- **sizedist_optimization.py**  
  Optimization routines that use the bin-remapping functions to align aerosol size distributions and simultaneously retrieve aerosol properties (e.g., **refractive index**, **density**, etc.)  
  by minimizing the **mean squared error (MSE)** over overlapping regions of two or more instruments' size distributions.

- **sizedist_combine.py**  
  Routines to reconstruct a **smooth aerosol size distribution** from multiple instruments onto a common diameter grid.

- **sizedist_utils.py**
  Contains useful helper functions for dealing with aerosol size distribution


### OPC mie scattering LUT

- **pops_sigma_col_405nm.zarr**  
  Calculated scattering cross-sections for POPS at **405 nm**, indexed by particle diameter and real and imaginary part of refractive index.

- **uhsas_sigma_col_1054nm.zarr**  
  Similar dataset for UHSAS at **1054 nm**.

