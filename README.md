# SizeDistMerge

SizeDistMerge is a Python framework for aligning and combining aerosol size
distributions from electrical mobility analyzers, optical particle counters,
and aerodynamic particle sizers. It provides optical lookup tables (LUTs),
diameter conversion, particle-number-preserving bin resizing, overlap fitting,
and weighted combination with Tikhonov smoothing.

## Repository layout

```text
src/                  Python library files directly in this folder
lut/                  POPS, UHSAS and PCASP optical tables
notebooks/            reusable examples without saved outputs
campaign_merge_production/  campaign-specific processing and QC helpers
tests/                automated library checks
docs/                 optical model and geometry documentation
local/                ignored personal research work, not published
```

The installation configuration maps `src/` to the Python import name
`sizedistmerge`; there is no nested source folder. Top-level `lut/` is the only
source of tables. Installations include those same tables as package resources.
Use `lut_path()` rather than assuming an installed file location.

## Install

With Python 3.12 or newer, run from the repository root:

```bash
python -m pip install -e .
```

Use the environment selected by your notebook kernel. Re-run this command when
updating an editable installation from the old nested source layout. Adding
`src/` to `PYTHONPATH` is not a substitute for installation. Optional acceleration
is available with `python -m pip install -e '.[numba]'`.

## Examples

- [Size distributions](notebooks/size_distribution_example.ipynb): bin counts,
  concentration-preserving resizing and aerodynamic diameter conversion.
- [Optical LUT use](notebooks/optical_lut_example.ipynb): response curves and
  diameter conversion between refractive indices.
- [Build optical LUTs](notebooks/build_optical_luts_example.ipynb): shared geometry
  definitions and new tables for POPS, UHSAS and PCASP.
- [ARCSIX processing](notebooks/arcsix_production_example.ipynb): configurable
  inputs, one-minute periods, merging and post-processing QC.

LUT builds and campaign processing are disabled by default. These are examples,
not archived R1/R2 production records. Full campaign runs, manuscript plots and
exploratory comparisons are retained in ignored `local/` directories.

## Library use

```python
import numpy as np
import sizedistmerge as sdm

edges = np.array([20., 40., 80., 160.])
distribution = np.array([1200., 800., 250.])
counts = sdm.counts_from_dndlog(distribution, edges_nm=edges)
new_edges = edges * 1.15
new_distribution = sdm.remap_dndlog_by_edges(edges, new_edges, distribution)
print(sdm.lut_path("pops"))
```

`src/` separates bin utilities, diameter conversion, optical geometry and LUTs,
alignment, combination, input/output, hygroscopic growth and plotting into
individual modules. ARCSIX-specific processing stays outside the core library.

## Optical assumptions

The tables use corrected solid-angle integration for homogeneous spheres.
They are model calculations, not individual instrument calibrations. The loader
checks the model version and build-completion marker. Read the
[optical model](docs/optical_model.md), [geometry interface](docs/optical_geometry.md),
and [PCASP assumptions](docs/pcasp_optics.md) before applying them.
Check the actual calibration refractive index and do not mix LUT versions when
resuming a campaign run.

## Tests and citation

With pytest installed in the same environment, run `python -m pytest tests`.
These checks assess numerical behavior and stated model assumptions, not
agreement with every instrument calibration.

Please cite the [SizeDistMerge software record](https://zenodo.org/records/17459970)
and the method references appropriate to your application. SizeDistMerge is
developed by the Kreidenweis Research Group at Colorado State University and
distributed under the [MIT license](LICENSE.txt).
