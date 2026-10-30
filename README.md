# SizeDistMerge

SizeDistMerge is a Python package for aligning and merging aerosol particle size
distributions measured by electrical mobility analyzers, optical particle
counters, and aerodynamic particle sizers. It combines diameter conversion,
fitting over overlapping size ranges, and weighted merging to produce a size
distribution on a common diameter grid.

The package provides:

- Optical diameter conversion using instrument-specific scattering lookup
  tables (LUTs) and specified refractive indices.
- Aerodynamic-to-volume-equivalent diameter conversion using particle density
  and shape factor.
- Bin-edge conversion and rebinning that preserve particle number concentration.
- Alignment of distributions by fitting their overlapping size ranges.
- Merging of bin number concentrations with adjustable weights and a smoothness
  penalty that limits rapid changes in the fitted distribution.

## Installation

SizeDistMerge requires **Python 3.12 or newer**. Install from GitHub:

```bash
git clone https://github.com/bochens/SizeDistMerge.git
cd SizeDistMerge
python -m pip install .
```

Python dependencies are installed automatically. For optional Numba acceleration,
use `python -m pip install '.[numba]'` from the repository directory.

To use the example notebooks, install SizeDistMerge into the Python environment
selected by your Jupyter kernel.

## Quick start

### Fit an optical distribution to a reference

This synthetic example uses the POPS lookup table to fit the real part `n` of
the particle refractive index `n + ik`, with the imaginary part `k` fixed at zero.
The reference is constructed with `n = 1.45`. Diameters are in nanometers, and
distribution heights are `dN/dlog10(D)` in cm⁻³.

```python
import numpy as np
import sizedistmerge as sdm

lut = sdm.SigmaLUT(str(sdm.lut_path("pops")))
calibration_ri = 1.615 + 0.001j
optical_edges_nm = np.array([150., 200., 300., 500., 800.])
optical_dndlog = np.array([100., 60., 20., 5.])

# Construct a reference with a known refractive index.
reference_edges_nm = sdm.convert_do_lut(
    optical_edges_nm, calibration_ri, 1.45 + 0j, lut, response_bins=100
)
reference_dndlog = sdm.remap_dndlog_by_edges(
    optical_edges_nm, reference_edges_nm, optical_dndlog
)

n_fit, k_fit, cost, result, history = sdm.optimize_refractive_index_for_opc(
    sdm.mids_from_edges(reference_edges_nm), reference_dndlog,
    optical_edges_nm, optical_dndlog, calibration_ri, lut,
    bounds=((1.35, 1.60), (0., 0.)),
    response_bins=100, moment="N", space="log", seed=123,
)
aligned_edges_nm = sdm.convert_do_lut(
    optical_edges_nm, calibration_ri, complex(n_fit, k_fit), lut,
    response_bins=100,
)
print(f"Fitted refractive index: {n_fit:.3f} + {k_fit:.3f}i")
print("Optimizer converged:", result.success)
```

The fitted real refractive index should be approximately `1.45`. The optimizer
converts the optical bin edges at each trial refractive index, preserves each
bin's number concentration, and minimizes squared differences between the
logarithms of the distribution heights over their overlapping diameter range.
`aligned_edges_nm` gives the fitted diameter edges for subsequent merging.
For measured data, supply the reference spectrum, optical bin edges and
distribution, calibration refractive index, and physically appropriate bounds.
The fitted index depends on the reference and optical-model assumptions.

### Merge aligned distributions

This example merges two synthetic distributions whose bin edges are already
expressed in the same diameter coordinate. Diameters are in nanometers, and
`number` contains the number concentration in each bin in cm⁻³.

```python
import numpy as np
import sizedistmerge as sdm

distributions = [
    {
        "name": "instrument_a",
        "edges": np.array([20., 40., 80., 160.]),
        "number": np.array([300., 200., 75.]),
    },
    {
        "name": "instrument_b",
        "edges": np.array([80., 160., 320., 640.]),
        "number": np.array([80., 30., 10.]),
    },
]
output_edges_nm = np.array([20., 40., 80., 160., 320., 640.])

merged, diagnostics = sdm.merge_sizedists_bin_totals(
    output_edges_nm, distributions, lam=1e-3
)
bin_concentration = sdm.counts_from_dndlog(merged, edges_nm=output_edges_nm)
print("Merged bin concentrations (cm^-3):", bin_concentration)
```

`merged` contains `dN/dlog10(D)`, the number concentration per unit base-10
logarithm of particle diameter; `N` is number concentration and `D` is diameter.
`diagnostics` includes fitted input-bin concentrations, residuals, and diameter
coverage. The parameter `lam` controls the smoothness penalty; select it for the
size resolution and measurement agreement required by your application.

## Examples and documentation

The example notebooks cover the main workflows:

- [Size distributions](notebooks/size_distribution_example.ipynb): bin number
  concentrations, changes to bin edges, and aerodynamic diameter conversion.
- [Optical diameter conversion](notebooks/optical_lut_example.ipynb): lookup-table
  queries, response curves, and conversion between refractive indices.
- [Building optical tables](notebooks/build_optical_luts_example.ipynb):
  instrument geometry and scattering calculations.
- [ARCSIX campaign processing](notebooks/arcsix_production_example.ipynb): input
  configuration, one-minute averaging, alignment, merging, and quality control.

For LUT construction and campaign processing, configure the input paths and
enable the corresponding steps in the notebook.

Further documentation describes the [optical model](docs/optical_model.md),
[instrument configuration](docs/optical_code_guide.md),
[collection geometry](docs/optical_geometry.md),
[PCASP assumptions](docs/pcasp_optics.md), and
[concentration uncertainty](docs/concentration_uncertainty.md).

## Scientific assumptions

The optical model calculates collected scattering cross-sections for homogeneous
spherical particles using the specified wavelength, polarization, and collection
geometry. Optical diameter conversion requires the instrument's calibration
refractive index and an assumed particle refractive index. Aerodynamic diameter
conversion requires particle density and shape factor, which describes the
effect of particle shape on drag.

Bundled optical tables cover the Portable Optical Particle Spectrometer (POPS),
Ultra-High Sensitivity Aerosol Spectrometer (UHSAS), and Passive Cavity Aerosol
Spectrometer Probe (PCASP). Use `sdm.lut_path("pops")`, `sdm.lut_path("uhsas")`, or
`sdm.lut_path("pcasp")` to locate a table. Instrument settings are provided in
[`opc_setups/`](opc_setups/).

For measured distributions, apply instrument-specific quality control and
diameter conversion before merging. Evaluate the fitted agreement, diameter
coverage, and sensitivity to model assumptions when interpreting the result.

## Contributing and testing

Report bugs and propose improvements through
[GitHub Issues](https://github.com/bochens/SizeDistMerge/issues).
Bug reports should include a minimal example, the Python and package versions,
and the expected and observed behavior. Contributions can be submitted as pull
requests with a description of the change and relevant checks.

For development, run these commands from the repository directory:

```bash
python -m pip install -e .
python -m pip install pytest
python -m pytest tests
```

## Citation

If you use SizeDistMerge in research, cite the
[software record on Zenodo](https://zenodo.org/records/17459970) and report the
software version used. Cite the method and instrument references relevant to
your analysis; optical-model references are listed in the linked documentation.

## License

SizeDistMerge is developed by the Kreidenweis Research Group at Colorado State
University and distributed under the [MIT license](LICENSE.txt).
