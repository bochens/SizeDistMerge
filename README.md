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
