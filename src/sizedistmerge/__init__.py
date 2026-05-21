"""Public package API for SizeDistMerge.

The package exposes reusable functions directly as ``sizedistmerge.function``
without importing every heavy dependency during ``import sizedistmerge``.
"""

from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"

_MODULE_EXPORTS = {
    "utils": (
        "edges_from_mids_geometric",
        "mids_from_edges",
        "delta_log10_from_edges",
        "dsdlog_from_dndlog",
        "dvdlog_from_dndlog",
        "counts_from_dndlog",
        "dndlog_from_counts",
        "remap_dndlog_by_edges",
        "rebin_dndlog_by_edges_overlap",
        "remap_dndlog_by_edges_any",
        "select_between",
    ),
    "diameter_conversion": (
        "mean_free_path",
        "cunningham",
        "da_to_dv",
    ),
    "kappa_kohler": (
        "kappa_petter_and_Kreidenweis_2010_EQ10",
        "Sc_petter_and_Kreidenweis_2010_EQ10",
        "Dd_petter_and_Kreidenweis_2010_EQ10",
        "S_petter_and_Kreidenweis_2010_EQ6",
        "find_peak_S_D_binary_search",
        "calculate_critical_diameter_interpolated",
        "calculate_kappa_fitting",
        "calculate_kappa",
        "calculate_critical_diameter",
        "plot_Sc_Dd_base",
        "calculate_wet_diameter",
        "calculate_dry_diameter",
        "calculate_humidification_factor",
        "calculate_humidification_factor_ammonium_sulfate",
        "calculate_coefficients",
        "kappa_from_growth_factor",
        "growth_factor_from_kappa",
    ),
    "optical_diameter": (
        "POPSGeom",
        "UHSASGeom",
        "pops_geometry_cache",
        "uhsas_geometry_cache",
        "pops_csca",
        "pops_csca_parallel",
        "uhsas_csca",
        "uhsas_csca_parallel",
        "build_sigma_lut",
        "build_pops_sigma_lut",
        "build_uhsas_sigma_lut",
        "SigmaLUT",
        "sigma_query_zarr",
        "make_monotone_sigma_interpolator",
        "convert_do_lut",
        "POPS_WAVELENGTH_NM",
        "UHSAS_WAVELENGTH_NM",
        "RI_UHSAS_SRC",
        "RI_POPS_SRC",
    ),
    "alignment": (
        "mse_overlap_sizedist",
        "objective_opc_vs_ref",
        "optimize_refractive_index_for_opc",
        "temporal_parameter_penalty",
        "objective_joint_named_temporal",
        "objective_multi_custom",
        "optimize_multi_custom",
    ),
    "combine": (
        "log_interp",
        "second_diff_nonuniform",
        "make_grid_from_series",
        "sigma_from_bands",
        "fractional_sigma",
        "compute_data_weights",
        "merge_sizedists_tikhonov",
        "merge_sizedists_tikhonov_consensus",
    ),
    "ict_utils": (
        "read_ict",
        "read_ict_file",
        "read_ict_dir",
        "pick_ict_files",
        "parse_bound",
        "read_aps",
        "read_pops",
        "read_uhsas",
        "read_fims",
        "read_nmass",
        "read_inlet_flag",
        "read_microphysical",
        "label_size_bins",
        "get_spectra",
        "mean_spectrum",
        "number_to_surface_area_spectrum",
        "check_common_grid",
        "align_to_common_grid",
        "filter_by_spectra_presence",
        "find_flag_column",
        "flag_segments",
        "flag_fractions",
        "check_meta",
    ),
    "plot": (
        "plot_size_distributions",
        "plot_size_distributions_steps",
    ),
    "resources": (
        "lut_path",
    ),
}

_SUBMODULES = tuple(_MODULE_EXPORTS) + (
    "arcsix_merge_production",
)
_EXPORT_TO_MODULE = {
    export: module_name
    for module_name, exports in _MODULE_EXPORTS.items()
    for export in exports
}

__all__ = sorted(
    {
        "__version__",
        *_SUBMODULES,
        *_EXPORT_TO_MODULE,
    }
)


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is not None:
        module = import_module(f".{module_name}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
