"""ARCSIX merge production and product-stage utilities."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
from datetime import timedelta
from netCDF4 import Dataset, stringtochar
from datetime import datetime
import traceback

from .ict_utils import (read_aps, read_nmass, read_pops, read_uhsas, read_fims, read_inlet_flag, read_microphysical, 
    check_common_grid, filter_by_spectra_presence, mean_spectrum, get_spectra)
from .utils import (edges_from_mids_geometric, dvdlog_from_dndlog,  remap_dndlog_by_edges_any,
    dsdlog_from_dndlog, mids_from_edges, remap_dndlog_by_edges, select_between, delta_log10_from_edges)
from .alignment import optimize_multi_custom
from .optical_diameter import SigmaLUT, convert_do_lut, RI_UHSAS_SRC, RI_POPS_SRC
from .diameter_conversion import da_to_dv
from .combine import make_grid_from_series, merge_sizedists_tikhonov, merge_sizedists_tikhonov_consensus
from .plot import plot_size_distributions, plot_size_distributions_steps
from .resources import lut_path

__all__ = [
    "DEFAULT_TEMPORAL_PRIOR_PARAMS",
    "as_timezone_naive_timestamp",
    "drop_timezone_from_index",
    "load_aufi_oneday",
    "load_af_oneday",
    "load_arcsix_merge_frames_for_day",
    "split_frames",
    "periods_from_split_frames",
    "periods_from_frames",
    "overlap_seconds_all_instruments",
    "plot_period_totals",
    "edges_from_meta_or_mids",
    "log_bin_table_horizontal",
    "mean_spectrum_with_edges",
    "filter_chunk_by_inlet_flag",
    "make_filtered_specs",
    "min_nonzero",
    "plot_history",
    "run_joint_optimization",
    "make_tikhonov_merged_spec",
    "make_consensus_merged_spec",
    "plot_sizedist_all",
    "chunk_is_incloud",
    "write_day_netcdf",
    "run_arcsix_merge_for_periods",
    "PostMergeQCResult",
    "find_merged_netcdf_files",
    "integrate_dndlog_gt_cutoff",
    "compute_robust_linear_bounds",
    "linear_resid_and_flag",
    "gather_post_merge_qc",
    "run_post_merge_product_qc",
    "write_icartt_from_netcdf",
    "convert_qc_netcdf_to_icartt",
]


DEFAULT_TEMPORAL_PRIOR_PARAMS = (1.52, 1.615, 1000.0)


@dataclass(frozen=True)
class PostMergeQCResult:
    """Paths and thresholds produced by post-merge QC."""

    qc_table_path: Path
    qc_plot_dir: Path
    qc_netcdf_dir: Path
    qc_netcdf_paths: tuple[Path, ...]
    r_low_warn: float
    r_high_warn: float
    r_med: float
    sigma_r: float
    total_chunks: int
    kept_chunks: int
    dropped_extreme_chunks: int


def find_merged_netcdf_files(base_dir: str | Path) -> list[Path]:
    """Return raw per-day merge NetCDFs under ``base_dir``.

    Product folders created by this module are intentionally excluded so a
    rerun does not recursively QC its own output.
    """

    base = Path(base_dir)
    excluded = {"qc_flagged_nc", "icartt_from_qc_flagged_nc", "qc_plots"}
    files = []
    for path in sorted(base.glob("**/*_sizedist_merged.nc")):
        if excluded.intersection(path.parts):
            continue
        files.append(path)
    return files


def _read_var_as_array(ds: Dataset, name: str) -> np.ndarray:
    if name not in ds.variables:
        raise RuntimeError(f"NetCDF missing required variable: {name}")
    var = ds.variables[name]
    arr = np.array(var[:], dtype=float)
    fill_value = getattr(var, "_FillValue", None)
    if fill_value is not None:
        arr[arr == fill_value] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr


def _parse_base_time(ds: Dataset) -> datetime:
    if "base_time_iso" not in ds.ncattrs():
        raise RuntimeError("NetCDF missing global attribute base_time_iso")
    base_iso = str(ds.getncattr("base_time_iso")).strip()
    if base_iso.endswith("Z"):
        base_iso = base_iso[:-1]
    if base_iso.endswith("+00:00") or base_iso.endswith("-00:00"):
        base_iso = base_iso[:-6]
    base_dt = datetime.fromisoformat(base_iso)
    if base_dt.tzinfo is not None:
        base_dt = base_dt.replace(tzinfo=None)
    return base_dt


def _ensure_naive_datetime_index(idx: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex for CPC time axis.")
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx


def _optional_1d(ds: Dataset, name: str, n_chunks: int) -> np.ndarray:
    if name not in ds.variables:
        return np.full(n_chunks, np.nan, dtype=float)
    arr = _read_var_as_array(ds, name)
    if arr.ndim != 1 or arr.size != n_chunks:
        raise RuntimeError(f"{name} must be 1D with length {n_chunks}, got shape={arr.shape}")
    return arr


def _gt_cutoff_weights_from_edges(fine_edges_nm: np.ndarray, cutoff_nm: float) -> np.ndarray:
    edges = np.asarray(fine_edges_nm, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("fine_edges_nm must be a 1D array with length >= 2")
    if not np.all(np.isfinite(edges)):
        raise ValueError("fine_edges_nm contains non-finite values")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("fine_edges_nm must be strictly increasing")
    cutoff_nm = float(cutoff_nm)
    if not np.isfinite(cutoff_nm) or cutoff_nm <= 0:
        raise ValueError("cutoff_nm must be a positive finite number")

    nbins = edges.size - 1
    weights = np.zeros(nbins, dtype=float)
    if cutoff_nm <= edges[0]:
        weights[:] = 1.0
        return weights
    if cutoff_nm >= edges[-1]:
        return weights

    log_edges = np.log10(edges)
    idx = np.searchsorted(edges, cutoff_nm, side="right") - 1
    if idx + 1 < nbins:
        weights[idx + 1 :] = 1.0

    lo = log_edges[idx]
    hi = log_edges[idx + 1]
    lc = np.log10(cutoff_nm)
    weights[idx] = float(np.clip((hi - lc) / (hi - lo), 0.0, 1.0))
    return weights


def integrate_dndlog_gt_cutoff(
    dndlogdp: np.ndarray,
    fine_edges_nm: np.ndarray,
    cutoff_nm: float = 10.0,
) -> np.ndarray:
    """Integrate ``dN/dlog10Dp`` over ``Dp > cutoff_nm`` for each chunk."""

    spectra = np.asarray(dndlogdp, dtype=float)
    if spectra.ndim != 2:
        raise ValueError("dndlogdp must be 2D with shape (chunk, fine_bin)")

    edges = np.asarray(fine_edges_nm, dtype=float)
    dlog10 = np.diff(np.log10(edges))
    if spectra.shape[1] != dlog10.size:
        raise RuntimeError(
            f"Shape mismatch: dndlogdp has {spectra.shape[1]} bins but "
            f"fine_edges_nm implies {dlog10.size} bins"
        )

    weights = _gt_cutoff_weights_from_edges(edges, cutoff_nm)
    return np.nansum(spectra * (dlog10[None, :] * weights[None, :]), axis=1)


def compute_robust_linear_bounds(
    merged: np.ndarray,
    cpc: np.ndarray,
    *,
    k_sigma: float = 10.0,
    min_points: int = 50,
) -> tuple[float, float, float, float]:
    """Return robust bounds for ``merged - cpc`` using median +/- k*MAD."""

    y = np.asarray(merged, dtype=float)
    x = np.asarray(cpc, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.any(ok):
        raise RuntimeError("No valid merged/CPC points for robust linear bounds")

    residual = y[ok] - x[ok]
    min_points = int(min_points)
    if residual.size < min_points:
        raise RuntimeError(
            f"Not enough valid merged/CPC points for robust bounds: "
            f"{residual.size} < {min_points}"
        )

    r_med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - r_med)))
    sigma = float(1.4826 * mad)
    if not np.isfinite(sigma) or sigma <= 0:
        raise RuntimeError(f"Invalid robust sigma: sigma={sigma} (mad={mad})")

    k_sigma = float(k_sigma)
    return (
        float(r_med - k_sigma * sigma),
        float(r_med + k_sigma * sigma),
        r_med,
        sigma,
    )


def linear_resid_and_flag(
    merged: np.ndarray,
    cpc: np.ndarray,
    *,
    r_low: float,
    r_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``merged - cpc`` and a 0/1 warning mask outside supplied bounds."""

    y = np.asarray(merged, dtype=float)
    x = np.asarray(cpc, dtype=float)
    residual = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    residual[ok] = y[ok] - x[ok]
    flag = ok & ((residual < float(r_low)) | (residual > float(r_high)))
    return residual, flag


def _cpc_statistic(values: pd.Series, statistic: str) -> float:
    if statistic == "median":
        return float(values.median())
    if statistic == "mean":
        return float(values.mean())
    raise ValueError("cpc_statistic must be 'median' or 'mean'")


def _read_cpc_series(micro_dir: Path, date_str: str, cpc_column: str) -> pd.Series:
    micro = read_microphysical(micro_dir, start=date_str, end=None, prefix="ARCSIX")
    if cpc_column not in micro:
        raise RuntimeError(f"Microphysical data for {date_str} missing {cpc_column!r}")
    cpc_series = pd.to_numeric(micro[cpc_column], errors="coerce")
    cpc_series.index = _ensure_naive_datetime_index(cpc_series.index)
    return cpc_series


def _chunk_cpc_values(
    cpc_series: pd.Series,
    base_dt: datetime,
    time_start_s: np.ndarray,
    time_end_s: np.ndarray,
    statistic: str,
) -> np.ndarray:
    values = np.full(time_start_s.size, np.nan, dtype=float)
    for idx in range(time_start_s.size):
        if not (np.isfinite(time_start_s[idx]) and np.isfinite(time_end_s[idx])):
            continue
        t0 = base_dt + timedelta(seconds=float(time_start_s[idx]))
        t1 = base_dt + timedelta(seconds=float(time_end_s[idx]))
        chunk = cpc_series.loc[t0:t1]
        if chunk.size:
            values[idx] = _cpc_statistic(chunk, statistic)
    return values


def gather_post_merge_qc(
    base_dir: str | Path,
    data_dir: str | Path,
    *,
    cutoff_nm: float = 10.0,
    cpc_column: str = "CNgt10nm",
    cpc_statistic: str = "median",
    micro_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Collect per-chunk QC quantities from raw merge NetCDF files."""

    base = Path(base_dir)
    micro_path = Path(micro_dir) if micro_dir is not None else Path(data_dir) / "LARGE-MICROPHYSICAL"
    nc_files = find_merged_netcdf_files(base)
    if not nc_files:
        raise FileNotFoundError(f"No raw *_sizedist_merged.nc files found under {base}")

    records: list[dict[str, object]] = []
    micro_cache: dict[str, pd.Series] = {}

    for nc_path in nc_files:
        date_str = nc_path.stem.split("_")[0]
        if date_str not in micro_cache:
            micro_cache[date_str] = _read_cpc_series(micro_path, date_str, cpc_column)
        cpc_series = micro_cache[date_str]

        with Dataset(nc_path, mode="r") as ds:
            cost = _read_var_as_array(ds, "optimization_best_cost")
            merged = _read_var_as_array(ds, "merged_dNdlogDp")
            edges = _read_var_as_array(ds, "fine_edges_nm")
            time_start_s = _read_var_as_array(ds, "time_start_since_base_s")
            time_end_s = _read_var_as_array(ds, "time_end_since_base_s")
            base_dt = _parse_base_time(ds)

            if cost.ndim != 1:
                raise RuntimeError(f"{nc_path}: optimization_best_cost must be 1D")
            n_chunks = int(cost.size)
            if merged.ndim != 2 or merged.shape[0] != n_chunks:
                raise RuntimeError(f"{nc_path}: merged_dNdlogDp chunk dimension mismatch")
            if time_start_s.size != n_chunks or time_end_s.size != n_chunks:
                raise RuntimeError(f"{nc_path}: time_start/end chunk length mismatch")

            uhsas_n = _optional_1d(ds, "retrieved_uhsas_n_fit", n_chunks)
            pops_n = _optional_1d(ds, "retrieved_pops_n_fit", n_chunks)
            rho = _optional_1d(ds, "retrieved_aps_density", n_chunks)
            merged_total = integrate_dndlog_gt_cutoff(merged, edges, cutoff_nm)
            cpc_values = _chunk_cpc_values(
                cpc_series,
                base_dt,
                time_start_s,
                time_end_s,
                cpc_statistic,
            )

        for idx in range(n_chunks):
            t0 = (
                base_dt + timedelta(seconds=float(time_start_s[idx]))
                if np.isfinite(time_start_s[idx])
                else pd.NaT
            )
            t1 = (
                base_dt + timedelta(seconds=float(time_end_s[idx]))
                if np.isfinite(time_end_s[idx])
                else pd.NaT
            )
            records.append(
                {
                    "source_nc": str(nc_path),
                    "date": date_str,
                    "chunk": idx,
                    "time_start": t0,
                    "time_end": t1,
                    "optimization_best_cost": float(cost[idx]) if np.isfinite(cost[idx]) else np.nan,
                    "retrieved_uhsas_n_fit": float(uhsas_n[idx]) if np.isfinite(uhsas_n[idx]) else np.nan,
                    "retrieved_pops_n_fit": float(pops_n[idx]) if np.isfinite(pops_n[idx]) else np.nan,
                    "retrieved_aps_density": float(rho[idx]) if np.isfinite(rho[idx]) else np.nan,
                    f"CPC_{cpc_statistic}_{cpc_column}": (
                        float(cpc_values[idx]) if np.isfinite(cpc_values[idx]) else np.nan
                    ),
                    f"MERGED_total_gt{int(cutoff_nm)}nm": (
                        float(merged_total[idx]) if np.isfinite(merged_total[idx]) else np.nan
                    ),
                }
            )

    return pd.DataFrame.from_records(records)


def _plot_cost_hist(cost: np.ndarray, out_png: Path, high_cost_thresh: float) -> None:
    finite = np.asarray(cost, dtype=float)
    finite = finite[np.isfinite(finite)]
    finite = finite[(finite >= 0.0) & (finite <= 1.0)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(finite, bins=np.linspace(0.0, 1.0, 51), edgecolor="black", alpha=0.8)
    ax.axvline(float(high_cost_thresh), color="orange", lw=2.0)
    ax.set_xlabel("optimization_best_cost")
    ax.set_ylabel("Number of chunks")
    ax.set_title(f"warning_high_cost: cost > {high_cost_thresh:g}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_simple_hist(values: np.ndarray, out_png: Path, xlabel: str, title: str) -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    fig, ax = plt.subplots(figsize=(7, 4))
    if finite.size:
        ax.hist(finite, bins=50, edgecolor="black", alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of chunks")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_merged_vs_cpc(
    cpc: np.ndarray,
    merged: np.ndarray,
    flag: np.ndarray,
    *,
    r_low: float,
    r_high: float,
    out_png: Path,
    plot_lo: float,
    plot_hi: float,
    k_sigma_warn: float,
    cpc_label: str,
) -> None:
    x = np.asarray(cpc, dtype=float)
    y = np.asarray(merged, dtype=float)
    bad = np.asarray(flag, dtype=bool)
    ok = np.isfinite(x) & np.isfinite(y)
    bad &= ok
    good = ok & ~bad

    fig, ax = plt.subplots(figsize=(6, 6))
    if np.any(good):
        ax.scatter(x[good], y[good], s=16, alpha=0.6, label="not flagged")
    if np.any(bad):
        ax.scatter(
            x[bad],
            y[bad],
            s=28,
            alpha=0.9,
            marker="x",
            label="warning_merged_gt10_diff_from_cpc=1",
        )
    xx = np.linspace(float(plot_lo), float(plot_hi), 200)
    ax.plot(xx, xx, "k--", lw=1.5, label="1:1")
    ax.plot(xx, xx + float(r_low), color="orange", lw=1.3, label="warning bounds")
    ax.plot(xx, xx + float(r_high), color="orange", lw=1.3)
    ax.set_xlim(float(plot_lo), float(plot_hi))
    ax.set_ylim(float(plot_lo), float(plot_hi))
    ax.set_xlabel(f"{cpc_label} (#/cm3)")
    ax.set_ylabel("Merged total >10 nm (#/cm3)")
    ax.set_title(
        f"warning_merged_gt10_diff_from_cpc: robust linear residual outlier "
        f"(K={k_sigma_warn:g})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _add_flag_var(
    dst_ds: Dataset,
    name: str,
    dims: tuple[str, ...],
    data: np.ndarray,
    *,
    long_name: str,
    comment: str,
) -> None:
    if name in dst_ds.variables:
        raise RuntimeError(f"Destination file already has variable {name}")
    var = dst_ds.createVariable(name, "i1", dims)
    var[:] = np.asarray(data, dtype=np.int8)
    var.long_name = long_name
    var.units = "1"
    var.flag_values = np.array([0, 1], dtype=np.int8)
    var.flag_meanings = "ok warning"
    var.comment = comment


def _copy_netcdf_subset_by_dim(
    src: Dataset,
    dst_path: Path,
    *,
    subset_dim: str,
    keep_idx: np.ndarray,
) -> Dataset:
    if dst_path.exists():
        dst_path.unlink()

    dst = Dataset(dst_path, mode="w", format="NETCDF4")
    for attr in src.ncattrs():
        dst.setncattr(attr, src.getncattr(attr))

    keep_count = int(keep_idx.size)
    for dim_name, dim in src.dimensions.items():
        if dim_name == subset_dim:
            dst.createDimension(dim_name, keep_count)
        else:
            dst.createDimension(dim_name, None if dim.isunlimited() else len(dim))

    for var_name, src_var in src.variables.items():
        fill_value = getattr(src_var, "_FillValue", None)
        if fill_value is None:
            dst_var = dst.createVariable(var_name, src_var.dtype, src_var.dimensions)
        else:
            dst_var = dst.createVariable(
                var_name,
                src_var.dtype,
                src_var.dimensions,
                fill_value=fill_value,
            )
        for attr in src_var.ncattrs():
            if attr != "_FillValue":
                dst_var.setncattr(attr, src_var.getncattr(attr))

        data = src_var[:]
        if subset_dim in src_var.dimensions:
            axis = src_var.dimensions.index(subset_dim)
            data = np.take(data, keep_idx, axis=axis)
        dst_var[:] = data

    return dst


def _write_qc_flagged_nc_files(
    base_dir: Path,
    out_dir: Path,
    *,
    data_dir: Path,
    cutoff_nm: float,
    cpc_column: str,
    cpc_statistic: str,
    high_cost_thresh: float,
    k_sigma_warn: float,
    k_sigma_drop: float,
    r_low_warn: float,
    r_high_warn: float,
    r_med: float,
    sigma_r: float,
    micro_dir: Path | None,
) -> tuple[tuple[Path, ...], int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    nc_files = find_merged_netcdf_files(base_dir)
    micro_path = micro_dir if micro_dir is not None else data_dir / "LARGE-MICROPHYSICAL"
    micro_cache: dict[str, pd.Series] = {}
    written: list[Path] = []
    total_dropped = 0
    total_kept = 0

    for src_path in nc_files:
        dst_path = out_dir / src_path.name
        date_str = src_path.stem.split("_")[0]
        if date_str not in micro_cache:
            micro_cache[date_str] = _read_cpc_series(micro_path, date_str, cpc_column)
        cpc_series = micro_cache[date_str]

        with Dataset(src_path, mode="r") as src:
            cost = _read_var_as_array(src, "optimization_best_cost")
            if cost.ndim != 1:
                raise RuntimeError(f"{src_path}: optimization_best_cost must be 1D")
            chunk_dim = src.variables["optimization_best_cost"].dimensions[0]
            n_chunks = int(cost.size)
            merged = _read_var_as_array(src, "merged_dNdlogDp")
            edges = _read_var_as_array(src, "fine_edges_nm")
            time_start_s = _read_var_as_array(src, "time_start_since_base_s")
            time_end_s = _read_var_as_array(src, "time_end_since_base_s")
            base_dt = _parse_base_time(src)

            merged_total = integrate_dndlog_gt_cutoff(merged, edges, cutoff_nm)
            cpc_values = _chunk_cpc_values(
                cpc_series,
                base_dt,
                time_start_s,
                time_end_s,
                cpc_statistic,
            )
            residual, warn_flag = linear_resid_and_flag(
                merged_total,
                cpc_values,
                r_low=r_low_warn,
                r_high=r_high_warn,
            )
            high_cost = np.isfinite(cost) & (cost > float(high_cost_thresh))

            ok = np.isfinite(residual)
            drop_mask = ok & (
                (residual < r_med - float(k_sigma_drop) * sigma_r)
                | (residual > r_med + float(k_sigma_drop) * sigma_r)
            )
            keep_idx = np.flatnonzero(~drop_mask).astype(int)
            if keep_idx.size <= 0:
                raise RuntimeError(
                    f"{src_path}: all chunks were dropped by extreme CPC filter; "
                    "refusing to write empty QC NetCDF"
                )

            total_dropped += int(np.sum(drop_mask))
            total_kept += int(keep_idx.size)
            dst = _copy_netcdf_subset_by_dim(
                src,
                dst_path,
                subset_dim=chunk_dim,
                keep_idx=keep_idx,
            )
            _add_flag_var(
                dst,
                "warning_high_cost",
                (chunk_dim,),
                high_cost[keep_idx],
                long_name="QC warning: optimization cost exceeds threshold",
                comment=f"1 if optimization_best_cost > {high_cost_thresh:g}, else 0",
            )
            _add_flag_var(
                dst,
                "warning_merged_gt10_diff_from_cpc",
                (chunk_dim,),
                warn_flag[keep_idx],
                long_name=(
                    f"QC warning: merged total(>{cutoff_nm:g} nm) vs CPC is a "
                    "linear residual outlier"
                ),
                comment=(
                    f"r = merged_total_gt{cutoff_nm:g}nm - CPC_{cpc_statistic}. "
                    f"Warning bounds: median +/- {k_sigma_warn:g}*(1.4826*MAD), "
                    f"r_med={r_med}, sigma_r={sigma_r}, "
                    f"r_low_warn={r_low_warn}, r_high_warn={r_high_warn}. "
                    f"Extreme chunks were removed with K_DROP={k_sigma_drop:g}."
                ),
            )
            dst.close()
            written.append(dst_path)

    return tuple(written), total_kept, total_dropped


def run_post_merge_product_qc(
    base_dir: str | Path,
    data_dir: str | Path,
    *,
    qc_dir: str | Path | None = None,
    qc_netcdf_dir: str | Path | None = None,
    micro_dir: str | Path | None = None,
    high_cost_thresh: float = 0.2,
    cutoff_nm: float = 10.0,
    k_sigma_warn: float = 10.0,
    k_sigma_drop: float = 20.0,
    min_points_for_robust: int = 50,
    cpc_column: str = "CNgt10nm",
    cpc_statistic: str = "median",
    plot_lo: float = 0.0,
    plot_hi: float = 8000.0,
) -> PostMergeQCResult:
    """Write post-merge QC plots, CSV, and QC-flagged NetCDF copies."""

    base = Path(base_dir)
    data_path = Path(data_dir)
    qc_plot_dir = Path(qc_dir) if qc_dir is not None else base / "qc_plots"
    qc_nc_dir = Path(qc_netcdf_dir) if qc_netcdf_dir is not None else base / "qc_flagged_nc"
    qc_plot_dir.mkdir(parents=True, exist_ok=True)

    table = gather_post_merge_qc(
        base,
        data_path,
        cutoff_nm=cutoff_nm,
        cpc_column=cpc_column,
        cpc_statistic=cpc_statistic,
        micro_dir=micro_dir,
    )
    cpc_col = f"CPC_{cpc_statistic}_{cpc_column}"
    merged_col = f"MERGED_total_gt{int(cutoff_nm)}nm"

    r_low, r_high, r_med, sigma_r = compute_robust_linear_bounds(
        table[merged_col].to_numpy(float),
        table[cpc_col].to_numpy(float),
        k_sigma=k_sigma_warn,
        min_points=min_points_for_robust,
    )
    residual, merged_warn = linear_resid_and_flag(
        table[merged_col].to_numpy(float),
        table[cpc_col].to_numpy(float),
        r_low=r_low,
        r_high=r_high,
    )
    high_cost = np.isfinite(table["optimization_best_cost"].to_numpy(float)) & (
        table["optimization_best_cost"].to_numpy(float) > float(high_cost_thresh)
    )

    table = table.copy()
    table["linear_residual_merged_minus_cpc"] = residual
    table["warning_high_cost"] = high_cost.astype(int)
    table["warning_merged_gt10_diff_from_cpc"] = merged_warn.astype(int)

    _plot_cost_hist(
        table["optimization_best_cost"].to_numpy(float),
        qc_plot_dir / "hist_optimization_best_cost_with_thresh.png",
        high_cost_thresh,
    )
    _plot_simple_hist(
        table["retrieved_uhsas_n_fit"].to_numpy(float),
        qc_plot_dir / "hist_uhsas_n.png",
        "retrieved_uhsas_n_fit",
        "UHSAS n distribution",
    )
    _plot_simple_hist(
        table["retrieved_pops_n_fit"].to_numpy(float),
        qc_plot_dir / "hist_pops_n.png",
        "retrieved_pops_n_fit",
        "POPS n distribution",
    )
    _plot_simple_hist(
        table["retrieved_aps_density"].to_numpy(float),
        qc_plot_dir / "hist_aps_rho.png",
        "retrieved_aps_density (kg m-3)",
        "APS density distribution",
    )
    _plot_merged_vs_cpc(
        table[cpc_col].to_numpy(float),
        table[merged_col].to_numpy(float),
        merged_warn,
        r_low=r_low,
        r_high=r_high,
        out_png=qc_plot_dir / "scatter_merged_gt10nm_vs_cpc_flagged.png",
        plot_lo=plot_lo,
        plot_hi=plot_hi,
        k_sigma_warn=k_sigma_warn,
        cpc_label=f"CPC {cpc_statistic} {cpc_column}",
    )

    table_path = qc_plot_dir / "per_chunk_qc_with_flags.csv"
    table.to_csv(table_path, index=False)

    qc_paths, kept, dropped = _write_qc_flagged_nc_files(
        base,
        qc_nc_dir,
        data_dir=data_path,
        cutoff_nm=cutoff_nm,
        cpc_column=cpc_column,
        cpc_statistic=cpc_statistic,
        high_cost_thresh=high_cost_thresh,
        k_sigma_warn=k_sigma_warn,
        k_sigma_drop=k_sigma_drop,
        r_low_warn=r_low,
        r_high_warn=r_high,
        r_med=r_med,
        sigma_r=sigma_r,
        micro_dir=Path(micro_dir) if micro_dir is not None else None,
    )

    return PostMergeQCResult(
        qc_table_path=table_path,
        qc_plot_dir=qc_plot_dir,
        qc_netcdf_dir=qc_nc_dir,
        qc_netcdf_paths=qc_paths,
        r_low_warn=r_low,
        r_high_warn=r_high,
        r_med=r_med,
        sigma_r=sigma_r,
        total_chunks=int(len(table)),
        kept_chunks=int(kept),
        dropped_extreme_chunks=int(dropped),
    )


def _read_1d(ds: Dataset, name: str) -> np.ndarray:
    arr = _read_var_as_array(ds, name)
    if arr.ndim != 1:
        raise RuntimeError(f"Expected {name} to be 1D, got shape={arr.shape}")
    return arr


def _read_2d(ds: Dataset, name: str) -> np.ndarray:
    arr = _read_var_as_array(ds, name)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected {name} to be 2D, got shape={arr.shape}")
    return arr


def _sec_of_day(dt: datetime) -> float:
    midnight = datetime(dt.year, dt.month, dt.day)
    return float((dt - midnight).total_seconds())


def _to_icartt_number(value, *, fill: float, float_fmt: str) -> str:
    try:
        number = float(value)
    except Exception:
        return str(fill)
    if not np.isfinite(number):
        return str(fill)
    return float_fmt.format(number)


def _edges_and_bin_labels(
    fine_edges_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    edges = np.asarray(fine_edges_nm, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise RuntimeError(f"fine_edges_nm must be 1D with len>=2, got shape={edges.shape}")
    if not np.all(np.isfinite(edges)):
        raise RuntimeError("fine_edges_nm contains non-finite values")
    if not np.all(np.diff(edges) > 0):
        raise RuntimeError("fine_edges_nm must be strictly increasing")

    centers = np.sqrt(edges[:-1] * edges[1:])
    labels = [f"DNLOG_{idx + 1:03d}" for idx in range(centers.size)]
    stdnames = [f"Aerosol_NumberSizeDistribution_Merged_{label}" for label in labels]
    return edges, centers, labels, stdnames


def _build_normal_comments(
    *,
    var_list_line: str,
    other_comments_lines: list[str],
    pi_contact_info: str,
    platform: str,
    location: str,
    associated_data: str,
    instrument_info: str,
    data_info: str,
    uncertainty: str,
    ulod_flag: str,
    ulod_value: str,
    llod_flag: str,
    llod_value: str,
    dm_contact_info: str,
    project_info: str,
    stipulations_on_use: str,
    revision: str,
    revision_comment: str,
) -> list[str]:
    lines = [
        f"PI_CONTACT_INFO: {pi_contact_info}",
        f"PLATFORM: {platform}",
        f"LOCATION: {location}",
        f"ASSOCIATED_DATA: {associated_data}",
        f"INSTRUMENT_INFO: {instrument_info}",
        f"DATA_INFO: {data_info}",
        f"UNCERTAINTY: {uncertainty}",
        f"ULOD_FLAG: {ulod_flag}",
        f"ULOD_VALUE: {ulod_value}",
        f"LLOD_FLAG: {llod_flag}",
        f"LLOD_VALUE: {llod_value}",
        f"DM_CONTACT_INFO: {dm_contact_info}",
        f"PROJECT_INFO: {project_info}",
        f"STIPULATIONS_ON_USE: {stipulations_on_use}",
    ]
    if other_comments_lines:
        lines.append("OTHER_COMMENTS: " + other_comments_lines[0].lstrip())
        lines.extend("  " + line.lstrip() for line in other_comments_lines[1:])
    else:
        lines.append("OTHER_COMMENTS: N/A")
    lines.extend([f"REVISION: {revision}", f"{revision}: {revision_comment}", var_list_line])
    return lines


def _build_icartt_header_lines(
    *,
    nheader: int,
    ffi: int,
    icartt_version: str,
    pi_name: str,
    pi_affiliation: str,
    data_source_desc: str,
    mission_name: str,
    date_data: datetime,
    date_file: datetime,
    indep_def_line: str,
    ndep: int,
    dep_scale_line: str,
    dep_miss_line: str,
    dep_def_lines: list[str],
    special_comments: list[str],
    normal_comments: list[str],
) -> list[str]:
    lines = [
        f"{nheader}, {ffi}, {icartt_version}",
        pi_name,
        pi_affiliation,
        data_source_desc,
        mission_name,
        "1, 1",
        (
            f"{date_data.year:04d}, {date_data.month:02d}, {date_data.day:02d}, "
            f"{date_file.year:04d}, {date_file.month:02d}, {date_file.day:02d}"
        ),
        "0",
        indep_def_line,
        str(int(ndep)),
        dep_scale_line,
        dep_miss_line,
        *dep_def_lines,
        str(len(special_comments)),
        *special_comments,
        str(len(normal_comments)),
        *normal_comments,
    ]
    if len(lines) != int(nheader):
        raise RuntimeError(f"Internal ICARTT header count mismatch: {len(lines)} != {nheader}")
    return lines


def write_icartt_from_netcdf(
    nc_path: str | Path,
    out_path: str | Path,
    *,
    product_name: str = "ARCSIX-MERGED-SIZEDIST",
    revision: str = "R1",
    data_source_desc: str | None = None,
    mission_name: str = "ARCSIX 2024",
    pi_name: str = "Perkins, Russell",
    pi_affiliation: str = "Colorado State University",
    pi_contact_info: str = (
        "Russell.Perkins@colostate.edu, Colorado State University, "
        "3915 W. Laporte Ave. Fort Collins, CO 80521, USA"
    ),
    platform: str = "NASA P-3B",
    location: str = "N/A",
    associated_data: str = "N/A",
    instrument_info: str | None = None,
    data_info: str | None = None,
    uncertainty: str = (
        "Estimated relative uncertainty is 20% for DNLOG_### variables only. "
        "QC variables and retrieved parameters are diagnostics and have no assigned uncertainty."
    ),
    dm_contact_info: str = "Russell Perkins, Colorado State University, Russell.Perkins@colostate.edu",
    project_info: str = (
        "Arctic Radiation-Cloud-Aerosol-Surface Interaction EXperiment (ARCSIX): "
        "May 2024-August 2024"
    ),
    stipulations_on_use: str | None = None,
    revision_comment: str | None = None,
    fill: float = -9999.0,
    float_fmt: str = "{:.6g}",
    ffi: int = 1001,
    icartt_version: str = "V02_2016",
    ulod_flag: str = "-7777",
    ulod_value: str = "N/A",
    llod_flag: str = "-8888",
    llod_value: str = "N/A",
) -> Path:
    """Write one ICARTT file from one QC-flagged merge NetCDF."""

    nc_path = Path(nc_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if data_source_desc is None:
        data_source_desc = "FIMS-UHSAS-POPS-APS merged aerosol size distribution, NASA P-3B aircraft"
    if instrument_info is None:
        instrument_info = (
            "Merged size distribution product derived from FIMS, UHSAS, POPS, APS. "
            "Includes retrieved effective UHSAS/POPS refractive index and APS density used in the merge "
            "(experimental diagnostics)."
        )
    if data_info is None:
        data_info = (
            "Merged aerosol number size distribution reported as dN/dlog10(Dp) (#/cm3) "
            "on the fine_edges_nm bin grid. Retrieved parameters are diagnostic outputs from the merge."
        )
    if stipulations_on_use is None:
        stipulations_on_use = (
            "Retrieved parameters are FOR EXPERIMENTAL/DIAGNOSTIC USE ONLY. "
            "Please consult the PI before using these retrieved parameters in publications. "
            "All other variables may be used following the dataset's standard citation and acknowledgement guidance."
        )
    if revision_comment is None:
        revision_comment = (
            "Merged aerosol number size distribution product from FIMS, UHSAS, POPS, and APS. "
            "Retrieved-parameter columns are experimental/diagnostic only."
        )

    with Dataset(nc_path, mode="r") as ds:
        base_dt = _parse_base_time(ds)
        t0_s = _read_1d(ds, "time_start_since_base_s")
        t1_s = _read_1d(ds, "time_end_since_base_s")
        if t0_s.size != t1_s.size:
            raise RuntimeError(f"{nc_path.name}: time_start/end size mismatch")

        merged = _read_2d(ds, "merged_dNdlogDp")
        if merged.shape[0] != t0_s.size:
            raise RuntimeError(f"{nc_path.name}: merged_dNdlogDp chunk mismatch")

        fine_edges_nm = _read_1d(ds, "fine_edges_nm")
        edges_nm, centers_nm, bin_labels, bin_stdnames = _edges_and_bin_labels(fine_edges_nm)

        one_d_vars = {
            "optimization_best_cost": _read_1d(ds, "optimization_best_cost"),
            "warning_high_cost": _read_1d(ds, "warning_high_cost"),
            "warning_merged_gt10_diff_from_cpc": _read_1d(
                ds,
                "warning_merged_gt10_diff_from_cpc",
            ),
            "retrieved_uhsas_n_fit": _read_1d(ds, "retrieved_uhsas_n_fit"),
            "retrieved_aps_density": _read_1d(ds, "retrieved_aps_density"),
        }
        if "retrieved_pops_n_fit" in ds.variables:
            one_d_vars["retrieved_pops_n_fit"] = _read_1d(ds, "retrieved_pops_n_fit")

        for name, values in one_d_vars.items():
            if values.size != t0_s.size:
                raise RuntimeError(f"{nc_path.name}: {name} length mismatch vs chunks")

        start_dt = [base_dt + timedelta(seconds=float(s)) for s in t0_s]
        stop_dt = [base_dt + timedelta(seconds=float(s)) for s in t1_s]
        time_start = np.array([_sec_of_day(dt) for dt in start_dt], dtype=float)
        time_stop = np.array([_sec_of_day(dt) for dt in stop_dt], dtype=float)
        time_mid = 0.5 * (time_start + time_stop)

        date_data = datetime(base_dt.year, base_dt.month, base_dt.day)
        now = datetime.now()
        date_file = datetime(now.year, now.month, now.day)

        dep_names = [
            "Time_Stop",
            "Time_Mid",
            "optimization_best_cost",
            "warning_high_cost",
            "warning_merged_gt10_diff_from_cpc",
            "retrieved_uhsas_n_fit",
        ]
        if "retrieved_pops_n_fit" in one_d_vars:
            dep_names.append("retrieved_pops_n_fit")
        dep_names.append("retrieved_aps_density")
        dep_names.extend(bin_labels)

        dep_def_lines = [
            "Time_Stop, seconds, Time_Stop, Seconds from 00:00 on measurement date at stop of chunk",
            "Time_Mid, seconds, Time_Mid, Seconds from 00:00 on measurement date at mid-point of chunk",
            (
                "optimization_best_cost, none, DataQuality_OptimizationCost_Merged_BestCost, "
                "Optimization best cost (unitless)"
            ),
            (
                "warning_high_cost, none, DataQuality_WarningFlag_Merged_HighCost, "
                "0/1 QC warning: optimization_best_cost exceeds threshold"
            ),
            (
                "warning_merged_gt10_diff_from_cpc, none, "
                "DataQuality_WarningFlag_Merged_Gt10nmVsCPC, "
                "0/1 QC warning: merged-vs-CPC linear residual outlier"
            ),
            (
                "retrieved_uhsas_n_fit, none, Aerosol_RefractiveIndex_Insitu_Merged_UHSASnFit, "
                "Effective real refractive index for UHSAS used in merge optimization "
                "(EXPERIMENTAL/DIAGNOSTIC)"
            ),
        ]
        if "retrieved_pops_n_fit" in one_d_vars:
            dep_def_lines.append(
                "retrieved_pops_n_fit, none, Aerosol_RefractiveIndex_Insitu_Merged_POPSnFit, "
                "Effective real refractive index for POPS used in merge optimization "
                "(EXPERIMENTAL/DIAGNOSTIC)"
            )
        dep_def_lines.append(
            "retrieved_aps_density, kg m-3, Aerosol_ParticleDensity_Insitu_Merged_APSDensityFit, "
            "Effective particle density for APS used in merge optimization (EXPERIMENTAL/DIAGNOSTIC)"
        )
        for idx, label in enumerate(bin_labels):
            dep_def_lines.append(
                f"{label}, #/cm3, {bin_stdnames[idx]}, "
                f"Merged dN/dlog10(Dp) for bin {idx + 1:03d}"
            )

        if len(dep_names) != len(dep_def_lines):
            raise RuntimeError("Internal ICARTT variable definition length mismatch")

        dep_scale_line = ", ".join(["1"] * len(dep_names))
        dep_miss_line = ", ".join([str(fill)] * len(dep_names))
        edges_one_line = "FINE_EDGES_NM: " + ", ".join(float_fmt.format(v) for v in edges_nm)
        centers_one_line = "FINE_CENTERS_NM: " + ", ".join(float_fmt.format(v) for v in centers_nm)
        other_comments = [
            "Retrieved parameters are EXPERIMENTAL/DIAGNOSTIC ONLY.",
            f"base_time_iso: {ds.getncattr('base_time_iso')}",
            edges_one_line,
            centers_one_line,
        ]
        var_list_line = ", ".join(["Time_Start"] + dep_names)
        normal_comments = _build_normal_comments(
            var_list_line=var_list_line,
            other_comments_lines=other_comments,
            pi_contact_info=pi_contact_info,
            platform=platform,
            location=location,
            associated_data=associated_data,
            instrument_info=instrument_info,
            data_info=data_info,
            uncertainty=uncertainty,
            ulod_flag=ulod_flag,
            ulod_value=ulod_value,
            llod_flag=llod_flag,
            llod_value=llod_value,
            dm_contact_info=dm_contact_info,
            project_info=project_info,
            stipulations_on_use=stipulations_on_use,
            revision=revision,
            revision_comment=revision_comment,
        )
        special_comments: list[str] = []
        nheader = 14 + len(dep_names) + len(special_comments) + len(normal_comments)
        header = _build_icartt_header_lines(
            nheader=nheader,
            ffi=ffi,
            icartt_version=icartt_version,
            pi_name=pi_name,
            pi_affiliation=pi_affiliation,
            data_source_desc=data_source_desc,
            mission_name=mission_name,
            date_data=date_data,
            date_file=date_file,
            indep_def_line="Time_Start, seconds, Time_Start, Seconds from 00:00 on measurement date at start of chunk",
            ndep=len(dep_names),
            dep_scale_line=dep_scale_line,
            dep_miss_line=dep_miss_line,
            dep_def_lines=dep_def_lines,
            special_comments=special_comments,
            normal_comments=normal_comments,
        )

        output = np.full((t0_s.size, 1 + len(dep_names)), np.nan, dtype=float)
        output[:, 0] = time_start
        col = 1
        output[:, col] = time_stop
        col += 1
        output[:, col] = time_mid
        col += 1
        for name in [
            "optimization_best_cost",
            "warning_high_cost",
            "warning_merged_gt10_diff_from_cpc",
            "retrieved_uhsas_n_fit",
        ]:
            output[:, col] = one_d_vars[name]
            col += 1
        if "retrieved_pops_n_fit" in one_d_vars:
            output[:, col] = one_d_vars["retrieved_pops_n_fit"]
            col += 1
        output[:, col] = one_d_vars["retrieved_aps_density"]
        col += 1
        output[:, col:] = merged
        if col + merged.shape[1] != output.shape[1]:
            raise RuntimeError("Internal ICARTT column packing mismatch")

    with out_path.open("w", newline="\n") as handle:
        for line in header:
            handle.write(line + "\n")
        for row in output:
            handle.write(
                ", ".join(_to_icartt_number(v, fill=fill, float_fmt=float_fmt) for v in row)
                + "\n"
            )

    return out_path


def convert_qc_netcdf_to_icartt(
    input_dir: str | Path,
    out_dir: str | Path,
    *,
    product_name: str = "ARCSIX-MERGED-SIZEDIST",
    revision: str = "R1",
    data_source_desc: str | None = None,
    revision_comment: str | None = None,
    **writer_kwargs,
) -> tuple[Path, ...]:
    """Convert all QC-flagged daily NetCDF files in ``input_dir`` to ICARTT."""

    input_path = Path(input_dir)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    nc_files = sorted(input_path.glob("*_sizedist_merged.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No *_sizedist_merged.nc files found in {input_path}")

    written: list[Path] = []
    for nc_file in nc_files:
        date_token = nc_file.name.split("_")[0].replace("-", "")
        out_name = f"{product_name}_P3B_{date_token}_{revision}.ict"
        out_file = output_path / out_name
        written.append(
            write_icartt_from_netcdf(
                nc_file,
                out_file,
                product_name=product_name,
                revision=revision,
                data_source_desc=data_source_desc,
                revision_comment=revision_comment,
                **writer_kwargs,
            )
        )
    return tuple(written)



def _as_timezone_naive_timestamp(value):
    """Parse a datetime-like value and drop timezone metadata without shifting clock time."""
    ts = pd.to_datetime(value)
    if isinstance(ts, pd.Timestamp) and ts.tz is not None:
        return ts.tz_localize(None)
    return ts


def as_timezone_naive_timestamp(value):
    """Parse a datetime-like value and drop timezone metadata without shifting clock time."""
    return _as_timezone_naive_timestamp(value)


def _drop_timezone_from_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a tz-naive DatetimeIndex, preserving displayed clock time."""
    if df is None or df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        out = df.copy()
        out.index = df.index.tz_localize(None)
        return out
    return df


def drop_timezone_from_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with a tz-naive index, preserving displayed clock time."""
    return _drop_timezone_from_index(df)


def _nonnegative_float(value, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _temporal_regularization_enabled(
    temporal_w_uh: float,
    temporal_w_po: float,
    temporal_w_rho: float,
    smooth_rho: bool,
) -> bool:
    temporal_w_uh = _nonnegative_float(temporal_w_uh, "temporal_w_uh")
    temporal_w_po = _nonnegative_float(temporal_w_po, "temporal_w_po")
    temporal_w_rho = _nonnegative_float(temporal_w_rho, "temporal_w_rho")
    return (
        temporal_w_uh > 0.0
        or temporal_w_po > 0.0
        or (bool(smooth_rho) and temporal_w_rho > 0.0)
    )


def _resolve_temporal_prior(prev_params):
    if prev_params is None:
        raise ValueError(
            "Temporal regularization requires explicit prev_params/temporal_prior_params; "
            "pass DEFAULT_TEMPORAL_PRIOR_PARAMS explicitly if that lab prior is intended."
        )
    arr = np.asarray(prev_params, dtype=float)
    if arr.shape != (3,):
        raise ValueError("prev_params must be [uhsas_n, pops_n, aps_density]")
    return arr


def _validate_consensus_data_space(value) -> str:
    value = str(value)
    if value not in ("linear", "log10"):
        raise ValueError(f"consensus_data_space must be 'linear' or 'log10', got: {value!r}")
    return value


def _validate_output_edges(output_edges):
    if output_edges is None:
        return None
    arr = np.asarray(output_edges, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("output_edges must be a 1D array with at least two edges")
    if not np.all(np.isfinite(arr)):
        raise ValueError("output_edges must contain only finite values")
    if np.any(arr <= 0.0):
        raise ValueError("output_edges must be positive")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError("output_edges must be strictly increasing")
    return arr


def _normalize_merge_instruments(instruments, *, include_pops: bool = True) -> tuple[str, ...]:
    if instruments is None:
        return ("FIMS", "UHSAS", "POPS", "APS") if include_pops else ("FIMS", "UHSAS", "APS")

    allowed = {"FIMS", "UHSAS", "POPS", "APS"}
    order = ("FIMS", "UHSAS", "POPS", "APS")
    selected = []
    for item in instruments:
        name = str(item).upper()
        if name not in allowed:
            raise ValueError(f"unsupported instrument {item!r}; expected one of {sorted(allowed)}")
        if name not in selected:
            selected.append(name)

    selected_set = set(selected)
    supported_sets = [
        {"FIMS", "APS"},
        {"FIMS", "UHSAS", "APS"},
        {"FIMS", "UHSAS", "POPS", "APS"},
    ]
    if selected_set not in supported_sets:
        raise ValueError(
            "supported ARCSIX instrument sets are "
            "('FIMS', 'APS'), ('FIMS', 'UHSAS', 'APS'), and "
            "('FIMS', 'UHSAS', 'POPS', 'APS')"
        )

    return tuple(name for name in order if name in selected_set)


def _validate_apply_alignment(apply_alignment, instruments: tuple[str, ...]) -> bool:
    apply_alignment = bool(apply_alignment)
    if not apply_alignment and instruments != ("FIMS", "APS"):
        raise ValueError("apply_alignment=False is only supported for instruments=('FIMS', 'APS')")
    return apply_alignment


def _build_temporal_arrays(
    instrument_names,
    *,
    prev_params,
    temporal_w_uh: float,
    temporal_w_po: float,
    temporal_w_rho: float,
    smooth_rho: bool,
):
    temporal_w_uh = _nonnegative_float(temporal_w_uh, "temporal_w_uh")
    temporal_w_po = _nonnegative_float(temporal_w_po, "temporal_w_po")
    temporal_w_rho = _nonnegative_float(temporal_w_rho, "temporal_w_rho")
    if not _temporal_regularization_enabled(
        temporal_w_uh,
        temporal_w_po,
        temporal_w_rho,
        smooth_rho,
    ):
        return None, None
    prev = _resolve_temporal_prior(prev_params)

    targets = []
    weights = []
    for name in instrument_names:
        if name == "UHSAS":
            targets.append(prev[0])
            weights.append(temporal_w_uh)
        elif name == "POPS":
            targets.append(prev[1])
            weights.append(temporal_w_po)
        elif name == "APS":
            targets.append(prev[2])
            weights.append(temporal_w_rho if smooth_rho else 0.0)
        else:
            raise ValueError(f"unknown temporal instrument {name!r}")

    return np.asarray(targets, dtype=float), np.asarray(weights, dtype=float)


def load_aufi_oneday(a_date, aps_dir, uhsas_dir, fims_dir, pops_dir=None):
    """
    Read ARCSIX APS, FIMS, and optionally UHSAS/POPS for one day.

    All available instruments are aligned to the FIMS time grid, FIMS QC=2 rows
    are removed, and rows are kept only when every loaded instrument has a
    valid spectrum.
    """
    frames = {
        "APS": read_aps(aps_dir, start=a_date, end=None, prefix="ARCSIX"),
        "FIMS": read_fims(fims_dir, start=a_date, end=None, prefix="ARCSIX"),
    }
    if uhsas_dir is not None:
        frames["UHSAS"] = read_uhsas(uhsas_dir, start=a_date, end=None, prefix="ARCSIX")
    if pops_dir is not None:
        frames["POPS"] = read_pops(pops_dir, start=a_date, end=None, prefix="ARCSIX")

    frames = {k: v for k, v in frames.items() if v is not None and not v.empty}
    if "FIMS" not in frames:
        return {}

    _ = check_common_grid(frames, ref_key="FIMS", round_to=None)

    fims_qc = pd.to_numeric(
        frames["FIMS"].get("QC_Flag", pd.Series(index=frames["FIMS"].index)),
        errors="coerce",
    )
    extra = {"FIMS": fims_qc.ne(2)}
    filtered, _keep = filter_by_spectra_presence(
        frames,
        col_prefix="dNdlogDp",
        min_instruments=None,
        extra_masks=extra,
        treat_nonpositive_as_nan=False,
    )

    return filtered

def load_af_oneday(a_date, aps_dir, fims_dir):
    """
    Read raw APS + FIMS for the day, line them up on the FIMS time grid,
    then drop any times where not all spectral instruments have valid data
    (FIMS also must pass QC != 2). This version does NOT read UHSAS.
    """
    aps  = read_aps (aps_dir,  start=a_date, end=None, prefix="ARCSIX")
    fims = read_fims(fims_dir, start=a_date, end=None, prefix="ARCSIX")

    frames = {"APS": aps, "FIMS": fims}
    _ = check_common_grid(frames, ref_key="FIMS", round_to=None)

    fims_qc = pd.to_numeric(fims.get("QC_Flag", pd.Series(index=fims.index)), errors="coerce")
    extra = {"FIMS": fims_qc.ne(2)}  # bad = 2

    filtered, _keep = filter_by_spectra_presence(
        frames,
        col_prefix="dNdlogDp",
        min_instruments=None,
        extra_masks=extra,
        treat_nonpositive_as_nan=False,
    )

    return filtered


def load_arcsix_merge_frames_for_day(
    data_dir,
    date,
    *,
    instruments=None,
    include_pops: bool = True,
    fims_lag: float = 10,
) -> dict[str, pd.DataFrame]:
    """
    Load one ARCSIX day for period discovery using the same frame filtering as
    the merge runner.

    Returned indices are timezone-naive wall-clock times. FIMS is shifted
    earlier by ``fims_lag`` seconds, matching the production merge path.
    """
    data_dir = Path(data_dir)
    merge_instruments = _normalize_merge_instruments(instruments, include_pops=include_pops)

    aps_dir = data_dir / "LARGE-APS"
    uhsas_dir = data_dir / "PUTLS-UHSAS" if "UHSAS" in merge_instruments else None
    pops_dir = data_dir / "PUTLS-POPS" if "POPS" in merge_instruments else None
    fims_dir = data_dir / "FIMS"

    frames = load_aufi_oneday(date, aps_dir, uhsas_dir, fims_dir, pops_dir)
    frames = {name: _drop_timezone_from_index(df) for name, df in frames.items()}

    if "FIMS" in frames and frames["FIMS"] is not None and not frames["FIMS"].empty:
        frames["FIMS"] = frames["FIMS"].copy()
        frames["FIMS"].index = frames["FIMS"].index - pd.Timedelta(seconds=float(fims_lag))

    return frames


def split_frames(frames, seconds):
    # find global start/end without using set()
    starts = [df.index[0] for df in frames.values() if len(df)]
    ends   = [df.index[-1] for df in frames.values() if len(df)]
    if not starts or not ends:
        return []

    t0 = min(starts)
    t_last = max(ends)

    out = []
    step = timedelta(seconds=seconds)
    t = t0
    while t < t_last:
        t2 = t + step
        # slice each df in its own (already sorted) order
        out.append({
            name: df.loc[t:t2 - pd.Timedelta(microseconds=1)]
            for name, df in frames.items()
        })
        t = t2
    return out


def periods_from_split_frames(chunks) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Convert ``split_frames`` chunks into explicit ``(start, end)`` periods.

    This matches the old notebooks' behavior: for each chunk, use the minimum
    first timestamp and maximum last timestamp across non-empty instrument
    frames.
    """
    periods = []
    for chunk in chunks:
        times = [
            t
            for df in chunk.values()
            if df is not None and len(df)
            for t in (df.index[0], df.index[-1])
        ]
        if not times:
            continue
        periods.append(
            (
                _as_timezone_naive_timestamp(min(times)),
                _as_timezone_naive_timestamp(max(times)),
            )
        )
    return periods


def periods_from_frames(frames, seconds) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split frames into fixed-width chunks, then return explicit periods."""
    return periods_from_split_frames(split_frames(frames, seconds))


def overlap_seconds_all_instruments(
    chunk: dict[str, pd.DataFrame],
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    instruments: tuple[str, ...] = ("APS", "UHSAS", "FIMS", "POPS"),
    freq: str = "1s",
) -> int:
    """Count same-grid seconds with at least one sample from every instrument."""
    if t_start is None or t_end is None:
        return 0
    if t_end <= t_start:
        return 0

    grid = pd.date_range(t_start, t_end, freq=freq, inclusive="left")
    if len(grid) == 0:
        return 0

    all_mask = np.ones(len(grid), dtype=bool)
    for name in instruments:
        df = chunk.get(name)
        if df is None or df.empty:
            return 0

        dfi = df.loc[t_start:t_end]
        if dfi.empty:
            return 0

        pres = dfi.resample(freq).size() > 0
        pres = pres.reindex(grid, fill_value=False)
        all_mask &= pres.to_numpy()
        if not all_mask.any():
            return 0

    return int(all_mask.sum())


def plot_period_totals(
    frames,
    *,
    min_diam_nm=None,
    max_diam_nm=None,
    gauss_win=10,
    gauss_std=3,
    max_gap_factor=3.0,
    title=None,
    inlet_flag: pd.DataFrame | None = None,
    inlet_col: str = "InletFlag_LARGE",
    cpc_total: pd.Series | None = None,
    t_start: pd.Timestamp | None = None,
    t_end: pd.Timestamp | None = None,

):
    MIN_DIAM_NM = min_diam_nm or {}
    MAX_DIAM_NM = max_diam_nm or {}

    totals = {}
    smoothed_totals = {}
    order = [k for k in ("POPS", "UHSAS", "FIMS", "APS") if k in frames]

    def _plot_no_bridge(ax, s: pd.Series, *, label=None, **kwargs):
        if s is None or s.empty:
            return
        t = s.index.to_numpy()
        y = s.to_numpy()
        if len(t) <= 1:
            ax.plot(t, y, label=label, **kwargs)
            return
        dt = np.diff(t).astype("timedelta64[s]").astype(float)
        med = np.nanmedian(dt) if np.isfinite(dt).any() else np.nan
        thr = max_gap_factor * med if np.isfinite(med) and med > 0 else np.inf
        split_idx = np.where(dt > thr)[0].tolist()
        first = True
        start = 0
        for cut in split_idx + [len(y) - 1]:
            seg = slice(start, cut + 1)
            if cut + 1 - start >= 2:
                ax.plot(t[seg], y[seg], label=(label if first else "_nolegend_"), **kwargs)
                first = False
            start = cut + 1

    # --- build totals from frames ---
    for name in order:
        df = frames.get(name)
        if df is None or df.empty:
            continue

        mids_nm, spec_wide = get_spectra(df, col_prefix="dNdlogDp", long=False)
        if spec_wide.empty or mids_nm.size == 0:
            continue

        dmin = MIN_DIAM_NM.get(name)
        dmax = MAX_DIAM_NM.get(name)
        if (dmin is not None) or (dmax is not None):
            lo_ok = mids_nm >= (float(dmin) if dmin is not None else -np.inf)
            hi_ok = mids_nm <= (float(dmax) if dmax is not None else np.inf)
            mask = lo_ok & hi_ok
            if not np.any(mask):
                continue
            i0 = int(np.argmax(mask))
            i1 = int(len(mids_nm) - np.argmax(mask[::-1]) - 1)
            mids_nm = mids_nm[i0:i1 + 1]
            spec_wide = spec_wide.iloc[:, i0:i1 + 1]
        else:
            i0 = 0
            i1 = len(mids_nm) - 1

        meta = (getattr(df, "attrs", {}) or {}).get("bin_meta") or {}
        lo = np.asarray(meta.get("lower_nm", []), float)
        up = np.asarray(meta.get("upper_nm", []), float)
        if lo.size and up.size and lo.size == up.size:
            full_edges = np.r_[lo, up[-1]]
            edges_nm = full_edges[i0:i1 + 2]
        else:
            edges_nm = edges_from_mids_geometric(mids_nm)
        if edges_nm.size != mids_nm.size + 1:
            edges_nm = edges_from_mids_geometric(mids_nm)

        dlog10 = delta_log10_from_edges(edges_nm)
        A = np.where(np.isfinite(spec_wide.to_numpy(float)), spec_wide.to_numpy(float), 0.0)
        N = (A * dlog10[None, :]).sum(axis=1)

        s_raw = pd.Series(N, index=spec_wide.index, name=name)
        s_gauss = (
            s_raw.rolling(gauss_win, win_type="gaussian", min_periods=1, center=True)
                 .mean(std=gauss_std)
        )

        totals[name] = s_raw
        smoothed_totals[name] = s_gauss

    # --- decide plotting window: use t_start/t_end if provided, else data range ---
    all_times = []
    for df in frames.values():
        if len(df):
            all_times.append(df.index[0])
            all_times.append(df.index[-1])

    if t_start is not None and t_end is not None:
        t0_plot = t_start
        t1_plot = t_end
    elif all_times:
        t0_plot = min(all_times)
        t1_plot = max(all_times)
    else:
        t0_plot = None
        t1_plot = None

    # --- fixed axis rectangle ---
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_axes([0.15, 0.12, 0.70, 0.78])
    ax2 = ax.twinx()
    ax2.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_alpha(0)

    # --- shade inlet ---
    if inlet_flag is not None and len(inlet_flag):
        all_times = []
        for df in frames.values():
            if len(df):
                all_times.append(df.index[0])
                all_times.append(df.index[-1])
        if all_times:
            t0 = min(all_times); t1 = max(all_times)
            sub = inlet_flag.loc[t0:t1]
            if inlet_col in sub.columns:
                sflag = sub[inlet_col].fillna(0).astype(float)
                on = (sflag != 0).astype(int)
                chg = on.diff().fillna(on.iloc[0]).ne(0)
                starts = sub.index[chg & (on == 1)]
                ends   = sub.index[chg.shift(-1, fill_value=True) & (on == 1)].tolist()
                if len(ends) < len(starts):
                    ends.append(t1)
                for st, en in zip(starts, ends):
                    ax.axvspan(st, en, color="0.9", alpha=0.6, zorder=-100, linewidth=0)

    _fixed = {"APS": "tab:blue", "UHSAS": "tab:green", "FIMS": "tab:red", "POPS": "tab:orange"}
    tab = list(plt.cm.tab10.colors)
    color_map = {k: (_fixed[k] if k in _fixed else tab[i % len(tab)]) for i, k in enumerate(order)}

    left_vals = []
    right_vals = []

    # --- frames data ---
    for k in order:
        s_raw = totals.get(k)
        s_gauss = smoothed_totals.get(k)
        if s_raw is None:
            continue
        c = color_map[k]
        if k == "APS":
            _plot_no_bridge(ax2, s_raw,  color=c, alpha=0.25, linewidth=2)
            _plot_no_bridge(ax2, s_gauss, color=c, alpha=1.0,  linewidth=2, label=k)
            right_vals += [s_raw.min(), s_raw.max(), s_gauss.min(), s_gauss.max()]
        else:
            _plot_no_bridge(ax, s_raw,  color=c, alpha=0.20, linewidth=2)
            _plot_no_bridge(ax, s_gauss, color=c, alpha=1.0,  linewidth=2, label=k)
            left_vals += [s_raw.min(), s_raw.max(), s_gauss.min(), s_gauss.max()]

    # --- CPC (smoothed same way) ---
    if cpc_total is not None and not cpc_total.empty:
        cpc_gauss = (
            cpc_total.rolling(gauss_win, win_type="gaussian", min_periods=1, center=True)
                     .mean(std=gauss_std)
        )
        _plot_no_bridge(ax, cpc_total, label="CPC total", color="black", alpha=0.25, linewidth=2)
        _plot_no_bridge(ax, cpc_gauss, label="CPC total", color="black", linewidth=2)
        left_vals += [cpc_gauss.min(), cpc_gauss.max()]

    ax.set_xlabel("Time", labelpad=6)
    ax.set_ylabel(r"Total number conc.  (# cm$^{-3}$)", labelpad=12)
    ax2.set_ylabel(r"Total number conc.  (# cm$^{-3}$) — APS", labelpad=14)
    ax.set_title(title or "Total Number Concentration (period)", pad=10)

    # --- force x-limits to match chunk if we know them ---
    if (t0_plot is not None) and (t1_plot is not None):
        ax.set_xlim(t0_plot, t1_plot)
        ax2.set_xlim(t0_plot, t1_plot)

    if left_vals:
        y0 = min(left_vals); y1 = max(left_vals)
        if y0 == y1:
            y0 -= 1.0; y1 += 1.0
        ax.set_ylim(y0, y1)
        ax.set_yticks(np.linspace(y0, y1, 5))

    if right_vals:
        y0r = min(right_vals); y1r = max(right_vals)
        if y0r == y1r:
            y0r -= 0.1; y1r += 0.1
        ax2.set_ylim(y0r, y1r)
        ax2.set_yticks(np.linspace(y0r, y1r, 5))

    ax.yaxis.set_label_coords(-0.13, 0.5)
    ax2.yaxis.set_label_coords(1.13, 0.5)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    uniq = {}
    for h, l in list(zip(h1, l1)) + list(zip(h2, l2)):
        uniq[l] = h
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", fontsize=9)

    return fig, (ax, ax2)


def edges_from_meta_or_mids(df: pd.DataFrame, mids_nm: np.ndarray) -> np.ndarray:
    meta = (df.attrs.get("bin_meta") or {})
    lo = meta.get("lower_nm")
    up = meta.get("upper_nm")
    if isinstance(lo, (list, tuple)) and isinstance(up, (list, tuple)) and len(lo) == len(up) and len(lo) > 0:
        return np.r_[np.asarray(lo, float), float(up[-1])]
    return edges_from_mids_geometric(mids_nm)


def log_bin_table_horizontal(log_path: Path | str, name: str, mids: np.ndarray, n_vals: np.ndarray) -> None:
    with open(log_path, "a") as f:
        f.write(f"\t\t[{name}] Bin summary (horizontal)\n")
        idx_str = [f"{i:d}" for i in range(len(mids))]
        mid_str = [f"{m:.3f}" for m in mids]
        n_str   = [f"{int(n)}" for n in n_vals]

        def row(label, arr):
            f.write(f"\t\t{label:>8} | " + " ".join(f"{s:>8}" for s in arr) + "\n")

        row("idx", idx_str)
        row("mid_nm", mid_str)
        row("n", n_str)


def mean_spectrum_with_edges(
    df: pd.DataFrame,
    label: str,
    *,
    col_prefix: str = "dNdlogDp",
    ddof: int = 1,
):
    out = mean_spectrum(df, label, col_prefix=col_prefix, ddof=ddof)
    if out is None:
        return None
    mids, mean_vals, sigma_vals, label_out, n_vals = out
    edges = edges_from_meta_or_mids(df, mids)
    return mids, edges, mean_vals, sigma_vals, label_out, n_vals

def _inlet_flag_to_bool(inlet_flag_sub: pd.DataFrame | pd.Series) -> pd.Series:
    """
    new
    True = flagged (bad); False = good.
    Works for inlet_flag being a Series or a DataFrame with one/multiple columns.
    """
    if isinstance(inlet_flag_sub, pd.DataFrame):
        return (inlet_flag_sub.fillna(0) != 0).any(axis=1)
    return (inlet_flag_sub.fillna(0) != 0)


def filter_chunk_by_inlet_flag(
    chunk: dict[str, pd.DataFrame],
    inlet_flag: pd.DataFrame | pd.Series,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    pad_s: int,
) -> dict[str, pd.DataFrame]:
    """
    new
    Drop instrument rows whose timestamps are within ±pad_s of ANY inlet-flagged timestamp.
    Filtering is applied per 5-min chunk window.
    """
    if inlet_flag is None or len(inlet_flag) == 0:
        raise RuntimeError("inlet_flag is empty; cannot filter by inlet flagged time.")

    pad = pd.Timedelta(seconds=pad_s)

    # Look at inlet_flag only for this chunk (with padding)
    sub = inlet_flag.loc[(t_start - pad):(t_end + pad)]
    if sub.empty:
        return chunk

    bad = _inlet_flag_to_bool(sub)
    bad_times = bad[bad].index
    if len(bad_times) == 0:
        return chunk

    bad_df = pd.DataFrame({"bad_time": bad_times}).sort_values("bad_time")

    out: dict[str, pd.DataFrame] = {}
    for name, df in chunk.items():
        if df is None or df.empty:
            out[name] = df
            continue
        if not df.index.is_monotonic_increasing:
            raise ValueError(f"{name} index is not sorted; cannot filter reliably.")

        sample = pd.DataFrame({"t": df.index})
        hit = (
            pd.merge_asof(
                sample,
                bad_df,
                left_on="t",
                right_on="bad_time",
                direction="nearest",
                tolerance=pad,
            )["bad_time"]
            .notna()
            .to_numpy()
        )
        out[name] = df.loc[~hit].copy()

    return out

def make_filtered_specs(
    frames: dict[str, pd.DataFrame],
    log_path: Path | str,
    line_kwargs=None,
    fill_kwargs=None,
    *,
    order: tuple[str, ...] = ("FIMS", "UHSAS", "POPS", "APS"),
):
    """Average loaded instrument spectra and attach plotting styles."""
    specs = {}
    bin_counts = {}
    
    for name in order:
        if name in frames and not frames[name].empty:
            out = mean_spectrum_with_edges(frames[name], name)
            if out is not None:
                m, e, y, s, _, n = out
                log_bin_table_horizontal(log_path, name, m, n)
                specs[name] = (m, e, y, s)
                bin_counts[name] = n

    if line_kwargs is None:
        line_kwargs = {
            "_default": {"linewidth": 1.5, "color": "k"},
            "FIMS":  {"color": "tab:red",               "ls": "dashed"},
            "UHSAS": {"color": "tab:green", "alpha": 0.6, "ls": "dashed"},
            "POPS":  {"color": "tab:orange", "alpha": 0.6, "ls": "dashed"},
            "APS":   {"color": "tab:blue",  "alpha": 0.6, "ls": "dashed"},
        }

    if fill_kwargs is None:
        fill_kwargs = {
            "_default": {"alpha": 0.1, "color": "k"},
            "FIMS":  {"alpha": 0.1, "color": "tab:red"},
            "UHSAS": False,
            "POPS":  False,
            "APS":   False,
        }

    return specs, line_kwargs, fill_kwargs, bin_counts


def min_nonzero(a):
    a = np.asarray(a, dtype=float)
    nz = a[np.isfinite(a) & (a > 0)]
    return float(nz.min()) if nz.size else 0.0


def _resolve_lut_path(lut_dir: str | Path | None, kind: str, filename: str) -> Path:
    if lut_dir is None:
        path = lut_path(kind)
    else:
        base = Path(lut_dir)
        path = base if base.name == filename else base / filename
    if not path.exists():
        raise FileNotFoundError(f"{kind} LUT not found at {path}")
    return path


def _load_uhsas_lut(lut_dir: str | Path | None):
    path = _resolve_lut_path(lut_dir, "uhsas", "uhsas_sigma_col_1054nm.zarr")
    return SigmaLUT(str(path))

def _load_pops_lut(lut_dir: str | Path | None):
    path = _resolve_lut_path(lut_dir, "pops", "pops_sigma_col_405nm.zarr")
    return SigmaLUT(str(path))

def _uhsas_remap_fn(edges, theta, *, lut, ri_src, k=0, response_bins=50):
    n = float(theta[0])
    return convert_do_lut(
        Do_nm=edges,
        ri_src=ri_src,
        ri_dst=complex(n, k),
        lut=lut,
        response_bins=response_bins,
    )

def _pops_remap_fn(edges, theta, *, lut, ri_src, k=0, response_bins=50):
    """Mirror of _uhsas_remap_fn for POPS."""
    n = float(theta[0])
    return convert_do_lut(
        Do_nm=edges, ri_src=ri_src, ri_dst=complex(n, k),
        lut=lut, response_bins=response_bins,
    )


def _aps_remap_fn(edges, theta, *, chi_t=1.0, rho0=1000.0, pres_hPa=1013.25, temp_C=20.0):
    rho_p = float(theta[0])
    return da_to_dv(
        edges,
        rho_p=rho_p,
        chi_t=float(chi_t),
        rho0=float(rho0),
        pres_hPa=float(pres_hPa),
        temp_C=float(temp_C),
    )


def plot_history(hist):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    iters = np.arange(1, len(hist["total"]) + 1)
    ax.plot(iters, hist["total"], label="total", linewidth=2, color="k")
    if "data" in hist and len(hist["data"]) == len(hist["total"]):
        ax.plot(iters, hist["data"], label="data", linewidth=1.4, color="tab:blue")
    if "temporal" in hist and len(hist["temporal"]) == len(hist["total"]):
        temporal = np.asarray(hist["temporal"], dtype=float)
        if np.any(temporal != 0.0):
            ax.plot(iters, temporal, label="temporal", linewidth=1.4, color="tab:orange")
    ax.set_xlabel("DE iteration")
    ax.set_ylabel("Cost")
    ax.set_title("Optimization history")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def run_joint_uhsas_aps_opt_from_specs(
    specs: dict, line_kwargs: dict, fill_kwargs: dict,
    *,
    moment: str = "V", space: str = "linear",
    pair_w: float = 1.0, uhsas_bounds=((1.3, 1.8),),
    aps_bounds=((950.0, 2000.0),), uhsas_xmin=200,
    uhsas_xmax=None, fims_xmin=None,
    fims_xmax=500, lut_dir=None,
):
    return run_joint_optimization(
        specs,
        line_kwargs,
        fill_kwargs,
        moment=moment,
        space=space,
        pair_w=pair_w,
        uhsas_bounds=uhsas_bounds,
        aps_bounds=aps_bounds,
        uhsas_xmin=uhsas_xmin,
        uhsas_xmax=uhsas_xmax,
        fims_xmin=fims_xmin,
        fims_xmax=fims_xmax,
        lut_dir=lut_dir,
    )


def run_joint_optimization(
    specs: dict, line_kwargs: dict, fill_kwargs: dict,
    *,
    moment: str = "V", space: str = "linear",
    pair_w: float = 1.0, uhsas_bounds=((1.3, 1.8),),
    pops_bounds=None,
    aps_bounds=((950.0, 2000.0),), uhsas_xmin=200,
    uhsas_xmax=None, fims_xmin=None,
    fims_xmax=400, pops_xmin=None, pops_xmax=None,
    lut_dir=None,
    pops_ri_src=None,
    w_uhsas: float = 1.0,
    w_pops: float = 1.0,
    w_aps: float = 1.0,
    temporal_w_uh: float = 0.0,
    temporal_w_po: float = 0.0,
    temporal_w_rho: float = 0.0,
    prev_params=None,
    smooth_rho: bool = True,
):
    """
    Jointly optimize loaded OPC/APS instruments against FIMS.

    FIMS is the reference. UHSAS, POPS, and APS are included when present in
    ``specs``. The returned ``opt_res`` includes fitted labels so callers do
    not have to reconstruct display names from rounded parameters.
    """
    if "FIMS" not in specs:
        raise ValueError("FIMS spectrum is required for joint optimization.")

    w_uhsas = _nonnegative_float(w_uhsas, "w_uhsas")
    w_pops = _nonnegative_float(w_pops, "w_pops")
    w_aps = _nonnegative_float(w_aps, "w_aps")

    if pops_bounds is None:
        pops_bounds = uhsas_bounds
    if pops_ri_src is None:
        # Legacy ARCSIX production used the UHSAS source RI for POPS LUT remapping.
        pops_ri_src = RI_UHSAS_SRC

    m_FIMS, e_FIMS, y_FIMS, s_FIMS = specs["FIMS"]
    m_fims_sel, e_fims_sel, y_fims_sel, s_fims_sel = select_between(
        m_FIMS, e_FIMS, y_FIMS, s_FIMS, xmin=fims_xmin, xmax=fims_xmax
    )

    specs["FIMS_applied"] = (m_fims_sel, e_fims_sel, y_fims_sel, s_fims_sel)
    line_kwargs["FIMS_applied"] = {"color": "tab:red", "linewidth": 2.0}
    fill_kwargs["FIMS_applied"] = False

    instruments = []
    bounds_list = []
    selected = {}
    instrument_names = []

    if "UHSAS" in specs:
        m_UHSAS, e_UHSAS, y_UHSAS, s_UHSAS = specs["UHSAS"]
        _, e_sel, y_sel, s_sel = select_between(
            m_UHSAS, e_UHSAS, y_UHSAS, s_UHSAS, xmin=uhsas_xmin, xmax=uhsas_xmax
        )
        lut_uhsas = _load_uhsas_lut(lut_dir)
        selected["UHSAS"] = (e_sel, y_sel, s_sel, lut_uhsas)
        instruments.append(
            {
                "edges": e_sel,
                "y": y_sel,
                "w_ref": w_uhsas,
                "remap_fn": _uhsas_remap_fn,
                "kwargs": {"lut": lut_uhsas, "ri_src": RI_UHSAS_SRC, "response_bins": 50},
            }
        )
        bounds_list.append(list(uhsas_bounds))
        instrument_names.append("UHSAS")

    if "POPS" in specs:
        m_POPS, e_POPS, y_POPS, s_POPS = specs["POPS"]
        _, e_sel, y_sel, s_sel = select_between(
            m_POPS, e_POPS, y_POPS, s_POPS, xmin=pops_xmin, xmax=pops_xmax
        )
        lut_pops = _load_pops_lut(lut_dir)
        selected["POPS"] = (e_sel, y_sel, s_sel, lut_pops)
        instruments.append(
            {
                "edges": e_sel,
                "y": y_sel,
                "w_ref": w_pops,
                "remap_fn": _pops_remap_fn,
                "kwargs": {"lut": lut_pops, "ri_src": pops_ri_src, "response_bins": 50},
            }
        )
        bounds_list.append(list(pops_bounds))
        instrument_names.append("POPS")

    if "APS" in specs:
        _m_APS, e_APS, y_APS, s_APS = specs["APS"]
        selected["APS"] = (e_APS, y_APS, s_APS, None)
        instruments.append(
            {
                "edges": e_APS,
                "y": y_APS,
                "w_ref": w_aps,
                "remap_fn": _aps_remap_fn,
                "kwargs": {"chi_t": 1.0, "rho0": 1000.0, "pres_hPa": 1013.25, "temp_C": 20.0},
            }
        )
        bounds_list.append(list(aps_bounds))
        instrument_names.append("APS")

    if not instruments:
        raise ValueError("At least one of UHSAS, POPS, or APS is required for joint optimization.")

    pair_weights = []
    if pair_w != 0 and "APS" in instrument_names:
        aps_idx = instrument_names.index("APS")
        for inst_idx, name in enumerate(instrument_names):
            if name in ("UHSAS", "POPS"):
                pair_weights.append((inst_idx, aps_idx, pair_w))

    temporal_target, temporal_weights = _build_temporal_arrays(
        instrument_names,
        prev_params=prev_params,
        temporal_w_uh=temporal_w_uh,
        temporal_w_po=temporal_w_po,
        temporal_w_rho=temporal_w_rho,
        smooth_rho=smooth_rho,
    )

    best_thetas, best_cost, _res, hist = optimize_multi_custom(
        ref_mids=m_fims_sel,
        ref_y=y_fims_sel,
        instruments=instruments,
        bounds_list=bounds_list,
        moment=moment,
        space=space,
        pair_weights=pair_weights if pair_weights else None,
        temporal_target=temporal_target,
        temporal_weights=temporal_weights,
        maxiter=200,
        tol=1e-6,
        seed=123,
    )

    opt_res = {
        "n_fit": np.nan,
        "n_pops_fit": np.nan,
        "rho_fit": np.nan,
        "best_cost": best_cost,
        "data_cost": float(hist.get("best_data_cost", best_cost)),
        "temporal_cost": float(hist.get("best_temporal_cost", 0.0)),
        "temporal_target": temporal_target,
        "temporal_weights": temporal_weights,
        "pops_ri_src": pops_ri_src,
        "hist": hist,
        "fit_labels": {},
    }
    style = {
        "UHSAS": "tab:green",
        "POPS": "tab:orange",
        "APS": "tab:blue",
    }

    for name, theta in zip(instrument_names, best_thetas):
        color = style[name]
        edges, y, s, lut = selected[name]
        value = float(theta[0])

        if name == "UHSAS":
            fit_edges = _uhsas_remap_fn(
                edges, [value], lut=lut, ri_src=RI_UHSAS_SRC, response_bins=120
            )
            label = f"UHSAS fit (n={value:.3f})"
            opt_res["n_fit"] = value
        elif name == "POPS":
            fit_edges = _pops_remap_fn(
                edges, [value], lut=lut, ri_src=pops_ri_src, response_bins=120
            )
            label = f"POPS fit (n={value:.3f})"
            opt_res["n_pops_fit"] = value
        else:
            fit_edges = _aps_remap_fn(
                edges,
                [value],
                chi_t=1.0,
                rho0=1000.0,
                pres_hPa=1013.25,
                temp_C=20.0,
            )
            label = f"APS fit (ρ={value*0.001:.3f} g/cm$^3$)"
            opt_res["rho_fit"] = value

        specs[label] = (
            mids_from_edges(fit_edges),
            fit_edges,
            remap_dndlog_by_edges(edges, fit_edges, y),
            remap_dndlog_by_edges(edges, fit_edges, s),
        )
        line_kwargs[label] = {"color": color, "linewidth": 2.0}
        fill_kwargs[label] = {"alpha": 0.1, "color": color}
        opt_res["fit_labels"][name] = label

    return specs, line_kwargs, fill_kwargs, opt_res


def make_tikhonov_merged_spec(
    *,
    e_fims_sel,
    y_fims_sel,
    e_aps_fit=None,
    y_aps_fit=None,
    e_uhsas_fit=None,
    y_uhsas_fit=None,
    lam: float = 1e-5,
    n_points: int = 200,
    alpha_fims: float = 1.0,
    alpha_uhsas: float = 1.0,
    alpha_aps: float = 1.0,
):
    """
    Tikhonov merge onto a common grid.

    Supports:
      - 2-instrument merge: FIMS + APS (set e_uhsas_fit/y_uhsas_fit=None or alpha_uhsas=0)
      - 3-instrument merge: FIMS + UHSAS + APS

    Notes:
      - If UHSAS edges/values are None OR alpha_uhsas == 0, UHSAS is excluded.
      - This removes the need to "fake" UHSAS by passing APS twice.
    """
    if e_aps_fit is None or y_aps_fit is None:
        raise ValueError("APS fitted spectrum is required for Tikhonov merge.")

    # mids from edges
    m_FIMS = mids_from_edges(e_fims_sel)
    m_APS  = mids_from_edges(e_aps_fit)

    # Build series list (always include FIMS and APS)
    series = [
        {"x": m_FIMS, "y": y_fims_sel, "alpha": float(alpha_fims)},
        {"x": m_APS,  "y": y_aps_fit,  "alpha": float(alpha_aps)},
    ]

    # Optionally include UHSAS
    use_uhsas = (
        (e_uhsas_fit is not None)
        and (y_uhsas_fit is not None)
        and (float(alpha_uhsas) != 0.0)
    )
    if use_uhsas:
        m_UHSASf = mids_from_edges(e_uhsas_fit)
        series.insert(1, {"x": m_UHSASf, "y": y_uhsas_fit, "alpha": float(alpha_uhsas)})

    # grid
    Dg = make_grid_from_series(series, n_points=n_points)

    # tikhonov merge
    y_merged, wsum, diag = merge_sizedists_tikhonov(
        Dg,
        series,
        lam=lam,
        eps=1e-12,
        nonneg=True,
    )

    Dg_edge = edges_from_mids_geometric(Dg)

    specs_label = "Tikhonov merged"
    specs_out = {
        specs_label: (Dg, Dg_edge, y_merged, np.full_like(Dg, np.nan)),
    }
    line_kwargs_out = {
        specs_label: {"color": "gray", "linewidth": 2.2},
    }
    fill_kwargs_out = {
        specs_label: False,
    }

    return specs_out, line_kwargs_out, fill_kwargs_out, diag


def make_consensus_merged_spec(
    *,
    e_fims_sel,
    y_fims_sel,
    e_uhsas_fit=None,
    y_uhsas_fit=None,
    e_pops_fit=None,
    y_pops_fit=None,
    e_aps_fit,
    y_aps_fit,
    lam: float = 1e-4,
    n_points: int = 200,
    alpha_fims: float = 1.0,
    alpha_uhsas: float = 1.0,
    alpha_pops: float = 1.0,
    alpha_aps: float = 1.0,
    c_punish: float = 1.0,
    data_space: str = "linear",
):
    """
    Consensus Tikhonov merge for FIMS, APS, and optional UHSAS/POPS fits.
    """
    if e_aps_fit is None or y_aps_fit is None:
        raise ValueError("APS fitted spectrum is required for consensus merge.")

    series = [
        {"x": mids_from_edges(e_fims_sel), "y": y_fims_sel, "alpha": float(alpha_fims)},
    ]
    label_parts = ["F"]

    if e_uhsas_fit is not None and y_uhsas_fit is not None and float(alpha_uhsas) != 0.0:
        series.append(
            {"x": mids_from_edges(e_uhsas_fit), "y": y_uhsas_fit, "alpha": float(alpha_uhsas)}
        )
        label_parts.append("U")

    if e_pops_fit is not None and y_pops_fit is not None and float(alpha_pops) != 0.0:
        series.append(
            {"x": mids_from_edges(e_pops_fit), "y": y_pops_fit, "alpha": float(alpha_pops)}
        )
        label_parts.append("P")

    series.append({"x": mids_from_edges(e_aps_fit), "y": y_aps_fit, "alpha": float(alpha_aps)})
    label_parts.append("A")

    Dg = make_grid_from_series(series, n_points=n_points)

    y_merged, wsum, diag = merge_sizedists_tikhonov_consensus(
        Dg,
        series,
        lam=lam,
        eps=1e-12,
        nonneg=True,
        c=c_punish,
        data_space=data_space,
    )

    Dg_edge = edges_from_mids_geometric(Dg)

    specs_label = f"Merged ({'+'.join(label_parts)})"
    specs_out = {
        specs_label: (Dg, Dg_edge, y_merged, np.full_like(Dg, np.nan)),
    }
    line_kwargs_out = {
        specs_label: {"color": "black", "linewidth": 2.5, "zorder": 10},
    }
    fill_kwargs_out = {
        specs_label: False,
    }

    return specs_out, line_kwargs_out, fill_kwargs_out, diag


def plot_sizedist_all(
    *,
    specs,
    merged_spec,
    line_kwargs,
    merged_line_kwargs,
    fill_kwargs,
    merged_fill_kwargs,
    inlet_flag,
    d_str: str,
):
    # -------- helper to get max y from line specs only --------
    def _max_from_specs(sp):
        ys = []
        for _, (_, _, vals, _) in sp.items():
            if vals is not None:
                ys.append(np.nanmax(vals))
        return float(np.nanmax(ys)) if ys else np.nan

    #print(inlet_flag)
    # ================= dN/dlogDp =================
    fig_N, (ax_N, axf_N), _ = plot_size_distributions_steps(
        specs=specs,
        inlet_flag=inlet_flag,
        yscale="log",
        xlim=(10, 1e4),
        line_kwargs=line_kwargs,
        fill_kwargs=fill_kwargs,
        show_flag_strip=True,
        moment="N",
    )

    # y-limit from LINES ONLY
    y_max_N = max(_max_from_specs(specs), _max_from_specs(merged_spec))
    if np.isfinite(y_max_N) and y_max_N > 0:
        ax_N.set_ylim(1e-4, y_max_N * 2)
    ax_N.set_title(d_str)

    # ================= dV/dlogDp =================
    specs_V = {
        lab: (mids, edges,
              dvdlog_from_dndlog(mids, vals),
              dvdlog_from_dndlog(mids, sigma))
        for lab, (mids, edges, vals, sigma) in specs.items()
    }
    merged_spec_V = {
        lab: (mids, edges,
              dvdlog_from_dndlog(mids, vals),
              dvdlog_from_dndlog(mids, sigma))
        for lab, (mids, edges, vals, sigma) in merged_spec.items()
    }

    fig_V, (ax_V, axf_V), _ = plot_size_distributions_steps(
        specs=specs_V,
        inlet_flag=inlet_flag,
        yscale="linear",
        xlim=(10, 1e4),
        line_kwargs=line_kwargs,
        fill_kwargs=fill_kwargs,
        show_flag_strip=True,
        moment="V",
    )

    y_max_V = max(_max_from_specs(specs_V), _max_from_specs(merged_spec_V))
    if np.isfinite(y_max_V) and y_max_V > 0:
        ax_V.set_ylim(0, y_max_V * 1.15)
    ax_V.set_title(d_str)

    return (fig_N, ax_N), (fig_V, ax_V), (specs_V, merged_spec_V)


def chunk_is_incloud(inlet_flag: pd.DataFrame | pd.Series,
                     t_start,
                     t_end,
                     tol_s: int = 10) -> int:
    if inlet_flag is None or len(inlet_flag) == 0:
        return 0
    if t_start is None or t_end is None:
        return 0

    pad = pd.Timedelta(seconds=tol_s)
    sub = inlet_flag.loc[(t_start - pad):(t_end + pad)]

    if sub.empty:
        return 0

    # handle Series vs DataFrame
    if isinstance(sub, pd.DataFrame):
        hit = (sub.fillna(0) != 0).any(axis=1).any()
    else:  # Series
        hit = (sub.fillna(0) != 0).any()

    return 1 if hit else 0


def write_day_netcdf(
    day_dir: Path,
    a_date: str,
    *,
    day_fine_edges: np.ndarray,
    day_fims_algn: list[np.ndarray],
    day_uhsas_algn: list[np.ndarray],
    day_aps_algn: list[np.ndarray],
    day_fine_vals: list[np.ndarray],
    day_times_start: list,
    day_times_end: list,
    day_incloud_flag: list[int],
    day_n_fit: list[float],
    day_rho_fit: list[float],
    day_best_cost: list[float],
    day_pops_algn: list[np.ndarray] | None = None,
    day_n_pops_fit: list[float] | None = None,
    orig_APS_edges: np.ndarray | None = None,
    orig_UHSAS_edges: np.ndarray | None = None,
    orig_POPS_edges: np.ndarray | None = None,
    orig_FIMS_edges: np.ndarray | None = None,
):
    nc_path = day_dir / f"{a_date}_sizedist_merged.nc"

    n_chunk = len(day_fine_vals)
    n_fine  = len(day_fine_edges) - 1 if day_fine_edges is not None else 0
    has_pops = day_pops_algn is not None
    if has_pops and day_n_pops_fit is None:
        day_n_pops_fit = [np.nan] * n_chunk

    t_start_strs = [str(t) if (t is not None and pd.notna(t)) else "" for t in day_times_start]
    t_end_strs   = [str(t) if (t is not None and pd.notna(t)) else "" for t in day_times_end]

    base_str = next((s for s in t_start_strs if s != ""), "")
    if base_str:
        base_dt = pd.to_datetime(base_str)
        base_iso = base_dt.isoformat()
    else:
        base_dt = None
        base_iso = ""

    start_since = []
    end_since   = []
    for s_str, e_str in zip(t_start_strs, t_end_strs):
        if base_dt is not None and s_str != "":
            s_dt = pd.to_datetime(s_str)
            start_since.append((s_dt - base_dt).total_seconds())
        else:
            start_since.append(np.nan)
        if base_dt is not None and e_str != "":
            e_dt = pd.to_datetime(e_str)
            end_since.append((e_dt - base_dt).total_seconds())
        else:
            end_since.append(np.nan)

    with Dataset(nc_path, "w") as nc:
        nc.createDimension("chunk", n_chunk)
        nc.createDimension("fine_bin", n_fine)
        nc.createDimension("fine_edge", n_fine + 1)

        if orig_APS_edges is not None:
            nc.createDimension("aps_edge", len(orig_APS_edges))
        if orig_UHSAS_edges is not None:
            nc.createDimension("uhsas_edge", len(orig_UHSAS_edges))
        if orig_POPS_edges is not None:
            nc.createDimension("pops_edge", len(orig_POPS_edges))
        if orig_FIMS_edges is not None:
            nc.createDimension("fims_edge", len(orig_FIMS_edges))

        # string attrs (same as before)
        nc.time_start_list = t_start_strs
        nc.time_end_list   = t_end_strs
        if base_iso:
            nc.base_time_iso = base_iso

        # write base_time_iso also as a char variable
        if base_iso:
            strlen = len(base_iso)
            nc.createDimension("base_time_strlen", strlen)
            v_base = nc.createVariable("base_time_iso", "S1", ("base_time_strlen",))
            v_base[:] = stringtochar(np.array([base_iso], dtype=f"S{strlen}"))

        v_t0 = nc.createVariable("time_start_since_base_s", "f8", ("chunk",))
        v_t1 = nc.createVariable("time_end_since_base_s",   "f8", ("chunk",))
        v_t0[:] = np.asarray(start_since, float)
        v_t1[:] = np.asarray(end_since, float)
        v_t0.long_name = "chunk start time since base_time_iso"
        v_t0.units = "s"
        v_t1.long_name = "chunk end time since base_time_iso"
        v_t1.units = "s"

        v_fe = nc.createVariable("fine_edges_nm", "f8", ("fine_edge",))
        v_fe[:] = day_fine_edges
        v_fe.long_name = "Common fine diameter bin edges"
        v_fe.units = "nm"

        v_fims = nc.createVariable("fims_aligned_dNdlogDp", "f8", ("chunk", "fine_bin"))
        v_uhsa = nc.createVariable("uhsas_aligned_dNdlogDp", "f8", ("chunk", "fine_bin"))
        if has_pops:
            v_pops = nc.createVariable("pops_aligned_dNdlogDp", "f8", ("chunk", "fine_bin"))
        v_aps  = nc.createVariable("aps_aligned_dNdlogDp",   "f8", ("chunk", "fine_bin"))
        v_mrg  = nc.createVariable("merged_dNdlogDp",        "f8", ("chunk", "fine_bin"))

        v_fims[:, :] = np.asarray(day_fims_algn)
        v_uhsa[:, :] = np.asarray(day_uhsas_algn)
        if has_pops:
            v_pops[:, :] = np.asarray(day_pops_algn)
        v_aps[:, :]  = np.asarray(day_aps_algn)
        v_mrg[:, :]  = np.asarray(day_fine_vals)

        v_fims.long_name = "FIMS dN/dlog10Dp rebinned to common fine edges"
        v_fims.units = "#/cm3"
        v_uhsa.long_name = "UHSAS dN/dlog10Dp (fitted) rebinned to common fine edges"
        v_uhsa.units = "#/cm3"
        if has_pops:
            v_pops.long_name = "POPS dN/dlog10Dp (fitted) rebinned to common fine edges"
            v_pops.units = "#/cm3"
        v_aps.long_name = "APS dN/dlog10Dp (density-corrected) rebinned to common fine edges"
        v_aps.units = "#/cm3"
        v_mrg.long_name = "Merged dN/dlog10Dp on common fine edges"
        v_mrg.units = "#/cm3"

        v_inc = nc.createVariable("inlet_incloud_flag", "i4", ("chunk",))
        v_inc[:] = np.asarray(day_incloud_flag, int)
        v_inc.long_name = "1 = this chunk is within ±10 s of inlet in-cloud flag; 0 = otherwise"

        v_n = nc.createVariable("retrieved_uhsas_n_fit", "f8", ("chunk",))
        v_n[:] = np.asarray(day_n_fit, float)
        v_n.long_name = "Retrieved UHSAS real refractive index (n) from joint optimization"
        v_n.units = "1"

        if has_pops:
            v_np = nc.createVariable("retrieved_pops_n_fit", "f8", ("chunk",))
            v_np[:] = np.asarray(day_n_pops_fit, float)
            v_np.long_name = "Retrieved POPS real refractive index (n) from joint optimization"
            v_np.units = "1"

        v_rho = nc.createVariable("retrieved_aps_density", "f8", ("chunk",))
        v_rho[:] = np.asarray(day_rho_fit, float)
        v_rho.long_name = "Retrieved APS particle density from joint optimization"
        v_rho.units = "kg m-3"

        v_cost = nc.createVariable("optimization_best_cost", "f8", ("chunk",))
        v_cost[:] = np.asarray(day_best_cost, float)
        v_cost.long_name = "Final optimization cost for this chunk"
        v_cost.units = "1"

        if orig_APS_edges is not None:
            v = nc.createVariable("aps_edges_nm", "f8", ("aps_edge",))
            v[:] = orig_APS_edges
            v.long_name = "Original APS diameter bin edges"
            v.units = "nm"
        if orig_UHSAS_edges is not None:
            v = nc.createVariable("uhsas_edges_nm", "f8", ("uhsas_edge",))
            v[:] = orig_UHSAS_edges
            v.long_name = "Original UHSAS diameter bin edges"
            v.units = "nm"
        if orig_POPS_edges is not None:
            v = nc.createVariable("pops_edges_nm", "f8", ("pops_edge",))
            v[:] = orig_POPS_edges
            v.long_name = "Original POPS diameter bin edges"
            v.units = "nm"
        if orig_FIMS_edges is not None:
            v = nc.createVariable("fims_edges_nm", "f8", ("fims_edge",))
            v[:] = orig_FIMS_edges
            v.long_name = "Original FIMS diameter bin edges"
            v.units = "nm"

        instrument_desc = "FIMS-UHSAS-POPS-APS" if has_pops else "FIMS-UHSAS-APS"
        nc.description = f"ARCSIX size distribution merge using {instrument_desc}."
        nc.date_merged = a_date
        nc.source = "arcsix_merge_production.write_day_netcdf"
        nc.conventions = "CF-1.8 (partial)"

    return nc_path


def run_arcsix_merge_for_periods(
    time_periods,
    data_dir,
    output_dir,
    *,
    fims_lag=10,
    incloud_pad_s=10,
    min_samples_per_inst=50,
    min_overlap_s=None,
    overlap_freq="1s",
    moment="V",
    space="linear",
    pair_w=1.0,
    bounds_uhsas=((1.3, 1.8),),
    bounds_aps=((950.0, 2000.0),),
    uhsas_xmin=200,
    uhsas_xmax=None,
    fims_xmin=None,
    fims_xmax=500,
    pops_xmin=None,
    pops_xmax=None,
    lut_dir=None,
    pops_ri_src=None,
    fine_bin=200,
    include_pops=True,
    instruments=None,
    apply_alignment=True,
    w_uhsas=1.0,
    w_pops=1.0,
    w_aps=1.0,
    temporal_w_uh=0.0,
    temporal_w_po=0.0,
    temporal_w_rho=0.0,
    temporal_prior_params=None,
    smooth_rho=True,
    uhsas_combine_weight=0.5,
    pops_combine_weight=0.2,
    aps_combine_weight=1.0,
    smoothness_lam=1e-4,
    consensus_c=1.0,
    consensus_data_space="linear",
    output_edges=None,
):
    """
    Run ARCSIX aerosol size distribution merge for a list of specified time periods.

    Parameters
    ----------
    time_periods : iterable of (str, str) or (datetime-like, datetime-like)
        Each entry is (start_time, end_time), e.g.
            [("2024-05-28 10:00", "2024-05-28 10:30"), ...]
        Treated as timezone-naive wall-clock datetimes. If timezone metadata is
        present, it is stripped without shifting the clock time.

    data_dir : str or Path
        Base ARCSIX_P3B data directory containing LARGE-APS, PUTLS-UHSAS,
        FIMS, and optionally PUTLS-POPS.
    output_dir : str or Path
        Output directory for batch NetCDF + plots + logs.
    """

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_instruments = _normalize_merge_instruments(instruments, include_pops=include_pops)
    apply_alignment = _validate_apply_alignment(apply_alignment, merge_instruments)
    include_pops = "POPS" in merge_instruments
    if include_pops and pops_ri_src is None:
        # Match legacy production notebooks; callers may pass RI_POPS_SRC explicitly.
        pops_ri_src = RI_UHSAS_SRC

    if not apply_alignment:
        if _temporal_regularization_enabled(
            temporal_w_uh,
            temporal_w_po,
            temporal_w_rho,
            smooth_rho,
        ):
            raise ValueError("temporal regularization requires joint alignment; direct FIMS+APS merge has no fitted parameters")
        return _run_arcsix_merge_for_periods_direct(
            time_periods,
            data_dir,
            output_dir,
            instruments=merge_instruments,
            fims_lag=fims_lag,
            incloud_pad_s=incloud_pad_s,
            min_samples_per_inst=min_samples_per_inst,
            min_overlap_s=min_overlap_s,
            overlap_freq=overlap_freq,
            fims_xmin=fims_xmin,
            fims_xmax=fims_xmax,
            fine_bin=fine_bin,
            smoothness_lam=smoothness_lam,
            output_edges=output_edges,
            alpha_fims=1.0,
            aps_combine_weight=aps_combine_weight,
        )

    # instrument subdirs
    aps_dir   = data_dir / "LARGE-APS"
    uhsas_dir = data_dir / "PUTLS-UHSAS" if "UHSAS" in merge_instruments else None
    pops_dir  = data_dir / "PUTLS-POPS" if include_pops else None
    fims_dir  = data_dir / "FIMS"
    inlet_dir = data_dir / "LARGE-InletFlag"
    micro_dir = data_dir / "LARGE-MICROPHYSICAL"

    if lut_dir is not None:
        lut_dir = Path(lut_dir)
    consensus_data_space = _validate_consensus_data_space(consensus_data_space)
    output_edges_arr = _validate_output_edges(output_edges)
    min_overlap_s = (
        None
        if min_overlap_s is None
        else _nonnegative_float(min_overlap_s, "min_overlap_s")
    )
    overlap_freq = str(overlap_freq)

    temporal_enabled = _temporal_regularization_enabled(
        temporal_w_uh,
        temporal_w_po,
        temporal_w_rho,
        smooth_rho,
    )
    temporal_prior = (
        _resolve_temporal_prior(temporal_prior_params)
        if temporal_enabled
        else None
    )

    log_file  = output_dir / "output_log.txt"
    error_log = output_dir / "error_log.txt"
    failed_chunks = []

    # ---------- parse periods, force tz-naive wall-clock datetimes ----------
    _periods = []
    for idx, (s_raw, e_raw) in enumerate(time_periods):
        s_ts = _as_timezone_naive_timestamp(s_raw)
        e_ts = _as_timezone_naive_timestamp(e_raw)

        if e_ts <= s_ts:
            raise ValueError(f"Period {idx}: end <= start ({s_ts} >= {e_ts})")

        date_str = s_ts.strftime("%Y-%m-%d")
        _periods.append(
            {
                "idx": idx,
                "start": s_ts,
                "end": e_ts,
                "date": date_str,
            }
        )

    # Unique dates we actually need to process
    dates = sorted({p["date"] for p in _periods})

    # Log header (unchanged from batch style)
    with log_file.open("a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write("ARCSIX Aerosol Size Distribution Merge Log\n")
        f.write(f"Generated: {ts} by Bo Chen\n\n")
        mode_desc = "+".join(merge_instruments)
        f.write(f"MODE: specified time periods ({mode_desc}, no 5-min daily splitting)\n")
        f.write("TIME_PERIODS:\n")
        for p in _periods:
            f.write(f"  [{p['idx']:03d}] {p['start']} -> {p['end']}\n")
        f.write("\n")

    for a_date in dates:
        periods_for_date = [p for p in _periods if p["date"] == a_date]
        if not periods_for_date:
            continue

        with log_file.open("a") as f:
            f.write(f"merging {a_date} for {len(periods_for_date)} specified period(s)\n")
            f.write("------------------------ SETTINGS ------------------------\n")
            f.write(f"DATA_DIR: {data_dir}\n")
            f.write(f"OUTPUT_DIR: {output_dir}\n")
            f.write(f"INSTRUMENTS: {merge_instruments}\n")
            f.write(f"APPLY_ALIGNMENT: {apply_alignment}\n")
            f.write(f"FIMS_LAG: {fims_lag}  # shift FIMS <FIMS_LAG> seconds earlier\n")
            f.write(f"INCLOUD_PAD_S: {incloud_pad_s}  # seconds around inlet_flag to mark a chunk in-cloud\n")
            f.write(f"MIN_SAMPLES_PER_INST: {min_samples_per_inst}  # skip chunks with fewer than this\n")
            f.write(f"MIN_OVERLAP_S: {min_overlap_s}\n")
            f.write(f"OVERLAP_FREQ: {overlap_freq}\n")
            f.write(f"MOMENT: {moment}  # moment used in joint optimization (N/S/V)\n")
            f.write(f"SPACE: {space}  # cost space (linear/log)\n")
            f.write(f"PAIR_W: {pair_w}  # cross-consistency weight OPCs-APS\n")
            f.write(f"W_UHSAS: {w_uhsas}\n")
            if include_pops:
                f.write(f"W_POPS: {w_pops}\n")
            f.write(f"W_APS: {w_aps}\n")
            f.write(f"BOUNDS_UHSAS: {bounds_uhsas}  # real refractive index lower/upper\n")
            f.write(f"BOUNDS_APS: {bounds_aps}  # density lower/upper (kg m-3)\n")
            f.write(f"UHSAS_XMIN: {uhsas_xmin}\n")
            f.write(f"UHSAS_XMAX: {uhsas_xmax}\n")
            if include_pops:
                f.write(f"POPS_XMIN: {pops_xmin}\n")
                f.write(f"POPS_XMAX: {pops_xmax}\n")
                f.write(f"POPS_RI_SRC: {pops_ri_src}\n")
            f.write(f"FIMS_XMIN: {fims_xmin}\n")
            f.write(f"FIMS_XMAX: {fims_xmax}\n")
            f.write(f"LUT_DIR: {lut_dir}\n")
            f.write(f"FINE_BIN: {fine_bin}  # number of fine bins for Tikhonov merged spec\n")
            f.write(f"USES_POPS: {include_pops}\n")
            f.write(f"UHSAS_COMBINE_WEIGHT: {uhsas_combine_weight}\n")
            if include_pops:
                f.write(f"POPS_COMBINE_WEIGHT: {pops_combine_weight}\n")
            f.write(f"APS_COMBINE_WEIGHT: {aps_combine_weight}\n")
            f.write(f"SMOOTHNESS_LAM: {smoothness_lam}\n")
            f.write(f"CONSENSUS_C: {consensus_c}\n")
            f.write(f"CONSENSUS_DATA_SPACE: {consensus_data_space}\n")
            output_edges_desc = (
                f"{output_edges_arr.size - 1} bins from {output_edges_arr[0]} to {output_edges_arr[-1]} nm"
                if output_edges_arr is not None
                else "native consensus grid"
            )
            f.write(f"OUTPUT_EDGES: {output_edges_desc}\n")
            f.write(f"TEMPORAL_ENABLED: {temporal_enabled}\n")
            f.write(f"TEMPORAL_W_UH: {temporal_w_uh}\n")
            if include_pops:
                f.write(f"TEMPORAL_W_PO: {temporal_w_po}\n")
            f.write(f"TEMPORAL_W_RHO: {temporal_w_rho}\n")
            f.write(f"SMOOTH_RHO: {smooth_rho}\n")
            if temporal_prior is not None:
                f.write(f"TEMPORAL_PRIOR_PARAMS: {temporal_prior.tolist()}\n")
            f.write("----------------------------------------------------------\n\n")

        day_dir = output_dir / a_date
        day_dir.mkdir(parents=True, exist_ok=True)
        totals_dir = day_dir / "time_series"
        opt_dir    = day_dir / "loss_curve"
        plots_dir  = day_dir / "merge_plots"
        totals_dir.mkdir(exist_ok=True)
        opt_dir.mkdir(exist_ok=True)
        plots_dir.mkdir(exist_ok=True)

        common_edges     = output_edges_arr.copy() if output_edges_arr is not None else None
        day_fims_algn    = []
        day_uhsas_algn   = []
        day_pops_algn    = []
        day_aps_algn     = []
        day_merged       = []
        day_times_start  = []
        day_times_end    = []
        day_incloud      = []
        orig_APS_edges   = None
        orig_UHSAS_edges = None
        orig_POPS_edges  = None
        orig_FIMS_edges  = None
        day_n_fit        = []
        day_n_pops_fit   = []
        day_rho_fit      = []
        day_best_cost    = []
        prev_params      = temporal_prior.copy() if temporal_prior is not None else None

        # --------- load frames & drop timezone metadata without clock conversion ---------
        filtered_frames = load_aufi_oneday(a_date, aps_dir, uhsas_dir, fims_dir, pops_dir)

        for name, df in list(filtered_frames.items()):
            filtered_frames[name] = _drop_timezone_from_index(df)

        # shift FIMS by -fims_lag seconds (indices already tz-naive now)
        if "FIMS" in filtered_frames and not filtered_frames["FIMS"].empty:
            filtered_frames["FIMS"] = filtered_frames["FIMS"].copy()
            filtered_frames["FIMS"].index = (
                filtered_frames["FIMS"].index - pd.Timedelta(seconds=fims_lag)
            )

        inlet_flag = read_inlet_flag(inlet_dir, start=a_date, end=None, prefix="ARCSIX")
        inlet_flag = _drop_timezone_from_index(inlet_flag)

        micro = read_microphysical(micro_dir, start=a_date, end=None, prefix="ARCSIX")
        micro = _drop_timezone_from_index(micro)

        cpc_total = pd.to_numeric(micro.get("CNgt10nm"), errors="coerce")

        # Loop over explicitly specified periods for this date
        for i, period in enumerate(periods_for_date):
            t_start = period["start"]
            t_end   = period["end"]

            try:
                with log_file.open("a") as f:
                    f.write(f"\tsizedist {i:03d} (period_idx={period['idx']:03d}): {t_start} -> {t_end}\n")

                # Build chunk by slicing each instrument in the period
                a_chunk = {}
                for name, df in filtered_frames.items():
                    if df is None or df.empty:
                        continue
                    sub = df.loc[t_start:t_end]
                    if not sub.empty:
                        a_chunk[name] = sub

                # empty window: no data from any instrument
                if not a_chunk:
                    with log_file.open("a") as f:
                        f.write(f"\t[SKIP] chunk {i:03d} empty window (no data in any instrument)\n")
                    continue

                inlet_chunk     = inlet_flag.loc[t_start:t_end] if not inlet_flag.empty else inlet_flag.iloc[0:0]
                cpc_total_chunk = cpc_total.loc[t_start:t_end]  if not cpc_total.empty else cpc_total.iloc[0:0]

                # 0) in-cloud flag for this chunk
                inc_flag = chunk_is_incloud(inlet_flag, t_start, t_end, tol_s=incloud_pad_s)

                # 1) time series plot
                fig1, _ = plot_period_totals(
                    a_chunk,
                    title=f"{a_date} sizedist {i:03d}",
                    inlet_flag=inlet_flag,
                    gauss_win=10,
                    gauss_std=2,
                    cpc_total=cpc_total_chunk,
                    t_start=t_start,            # <--- NEW
                    t_end=t_end,                # <--- NEW
                )

                fig1.savefig(totals_dir / f"sizedist_{i:03d}_totals.png", dpi=150)
                plt.close(fig1)
                
                ##################################################################
                # 1.5) build a version of a_chunk with inlet-flagged data removed
                pad = pd.Timedelta(seconds=incloud_pad_s)
                inlet_for_filter = inlet_flag.loc[(t_start - pad):(t_end + pad)]
                if isinstance(inlet_for_filter, pd.DataFrame):
                    sflag = (inlet_for_filter.fillna(0) != 0).any(axis=1)
                else:
                    sflag = inlet_for_filter.fillna(0) != 0

                # duration of this specified window
                dur_s = (t_end - t_start).total_seconds()

                with log_file.open("a") as f:
                    f.write(
                        f"\t[INLET_FILTER] chunk {i:03d}: "
                        f"{len(inlet_for_filter)} inlet_flag samples within +/-{incloud_pad_s}s, "
                        f"{int(sflag.sum())} flagged, "
                        f"duration={dur_s:.1f} s\n"
                    )

                filtered_all = filter_chunk_by_inlet_flag(
                    a_chunk,
                    inlet_flag,
                    t_start,
                    t_end,
                    pad_s=incloud_pad_s,
                )
                filtered_chunk = {}

                for name, df in a_chunk.items():
                    if df is None or df.empty:
                        continue

                    n_before = len(df)
                    df2 = filtered_all.get(name)
                    if df2 is None:
                        df2 = df.iloc[0:0]
                    n_after = len(df2)

                    with log_file.open("a") as f:
                        f.write(
                            f"\t    {name}: before={n_before}, "
                            f"after={n_after}, "
                            f"removed={n_before - n_after}\n"
                        )

                    if not df2.empty:
                        filtered_chunk[name] = df2

                # if everything was nuked by inlet filter, skip this chunk
                if not filtered_chunk:
                    with log_file.open("a") as f:
                        f.write(f"\t[SKIP] chunk {i:03d} all data removed by inlet flag\n")
                    continue

                if min_overlap_s is not None:
                    ov_s = overlap_seconds_all_instruments(
                        filtered_chunk,
                        t_start,
                        t_end,
                        instruments=merge_instruments,
                        freq=overlap_freq,
                    )
                    if ov_s < min_overlap_s:
                        with log_file.open("a") as f:
                            f.write(
                                f"\t[SKIP] chunk {i:03d} insufficient overlap: "
                                f"{ov_s:d}s < {int(min_overlap_s)}s (freq={overlap_freq})\n"
                            )
                        continue

                # always use the filtered chunk for mean spectra
                chunk_for_specs = filtered_chunk
                ##################################################################
                
                # 2) mean specs
                specs, line_kwargs, fill_kwargs, bin_counts = make_filtered_specs(
                    chunk_for_specs,
                    log_file,
                )
                
                required_instruments = merge_instruments
                if any(name not in specs for name in required_instruments):
                    status = ", ".join(f"{name}={name in specs}" for name in required_instruments)
                    with log_file.open("a") as f:
                        f.write(
                            f"\t[SKIP] chunk {i:03d} missing instrument(s): "
                            f"{status}\n"
                        )
                    continue

                # gate on NON-ZERO AVERAGE per instrument
                low_data_reason = None
                for _name in required_instruments:
                    if _name in bin_counts and len(bin_counts[_name]) > 0:
                        arr = np.asarray(bin_counts[_name], int)
                        nz = arr[arr > 0]
                        nz_avg = float(nz.mean()) if nz.size > 0 else 0.0
                        if nz_avg < min_samples_per_inst:
                            low_data_reason = f"{_name} nonzero_avg={nz_avg:.1f} < {min_samples_per_inst}"
                            break
                if low_data_reason is not None:
                    with log_file.open("a") as f:
                        f.write(f"\t[SKIP] chunk {i:03d} low data: {low_data_reason}\n")
                    continue

                # save original edges once
                if orig_APS_edges is None:
                    orig_APS_edges = specs["APS"][1]
                if "UHSAS" in specs and orig_UHSAS_edges is None:
                    orig_UHSAS_edges = specs["UHSAS"][1]
                if include_pops and orig_POPS_edges is None:
                    orig_POPS_edges = specs["POPS"][1]
                if orig_FIMS_edges is None:
                    orig_FIMS_edges = specs["FIMS"][1]

                # 3) optimization
                specs_opt, line_kwargs_opt, fill_kwargs_opt, opt_res = run_joint_optimization(
                    specs,
                    line_kwargs,
                    fill_kwargs,
                    moment=moment,
                    space=space,
                    pair_w=pair_w,
                    uhsas_bounds=list(bounds_uhsas),
                    aps_bounds=list(bounds_aps),
                    uhsas_xmin=uhsas_xmin,
                    uhsas_xmax=uhsas_xmax,
                    fims_xmin=fims_xmin,
                    fims_xmax=fims_xmax,
                    pops_xmin=pops_xmin,
                    pops_xmax=pops_xmax,
                    lut_dir=lut_dir,
                    pops_ri_src=pops_ri_src,
                    w_uhsas=w_uhsas,
                    w_pops=w_pops,
                    w_aps=w_aps,
                    temporal_w_uh=temporal_w_uh,
                    temporal_w_po=temporal_w_po,
                    temporal_w_rho=temporal_w_rho,
                    prev_params=prev_params,
                    smooth_rho=smooth_rho,
                )
                if temporal_enabled:
                    prev_params = np.asarray(
                        [opt_res["n_fit"], opt_res["n_pops_fit"], opt_res["rho_fit"]],
                        dtype=float,
                    )

                # 4) loss curve
                fig_h, ax_h = plot_history(opt_res["hist"])
                ax_h.set_title(f"opt hist {a_date} {i:03d}")
                fig_h.savefig(opt_dir / f"sizedist_{i:03d}_opt_hist.png", dpi=150)
                plt.close(fig_h)

                # 5) log
                with log_file.open("a") as f:
                    fit_parts = []
                    if "UHSAS" in opt_res["fit_labels"]:
                        fit_parts.append(f"UHSAS n_fit = {opt_res['n_fit']:.4f}")
                    if "POPS" in opt_res["fit_labels"]:
                        fit_parts.append(f"POPS n_fit = {opt_res['n_pops_fit']:.4f}")
                    if "APS" in opt_res["fit_labels"]:
                        fit_parts.append(f"APS rho_fit = {opt_res['rho_fit']:.1f} kg/m^3")
                    f.write(
                        "\t\t"
                        f"{', '.join(fit_parts)}, "
                        f"cost = {opt_res['best_cost']:.6g}, "
                        f"data_cost = {opt_res['data_cost']:.6g}, "
                        f"temporal_cost = {opt_res['temporal_cost']:.6g}\n\n"
                    )

                uh_label = opt_res["fit_labels"].get("UHSAS")
                po_label = opt_res["fit_labels"].get("POPS")
                aps_label = opt_res["fit_labels"]["APS"]

                # 6) consensus merged
                uhsas_merge_kwargs = (
                    {
                        "e_uhsas_fit": specs_opt[uh_label][1],
                        "y_uhsas_fit": specs_opt[uh_label][2],
                        "alpha_uhsas": uhsas_combine_weight,
                    }
                    if uh_label is not None
                    else {}
                )
                pops_merge_kwargs = (
                    {
                        "e_pops_fit": specs_opt[po_label][1],
                        "y_pops_fit": specs_opt[po_label][2],
                        "alpha_pops": pops_combine_weight,
                    }
                    if include_pops
                    else {}
                )
                tik_specs, tik_lines, tik_fills, tik_diag = make_consensus_merged_spec(
                    e_fims_sel=specs_opt["FIMS_applied"][1],
                    y_fims_sel=specs_opt["FIMS_applied"][2],
                    e_aps_fit=specs_opt[aps_label][1],
                    y_aps_fit=specs_opt[aps_label][2],
                    lam=smoothness_lam,
                    n_points=fine_bin,
                    alpha_aps=aps_combine_weight,
                    c_punish=consensus_c,
                    data_space=consensus_data_space,
                    **uhsas_merge_kwargs,
                    **pops_merge_kwargs,
                )

                specs_opt.update(tik_specs)
                line_kwargs_opt.update(tik_lines)
                fill_kwargs_opt.update(tik_fills)

                # 7) plots
                (figN, axN), (figV, axV), (specs_V, merged_spec_V) = plot_sizedist_all(
                    specs=specs_opt,
                    merged_spec=tik_specs,
                    line_kwargs=line_kwargs_opt,
                    merged_line_kwargs=tik_lines,
                    fill_kwargs=fill_kwargs_opt,
                    merged_fill_kwargs=tik_fills,
                    inlet_flag=inlet_chunk,
                    d_str=a_date,
                )
                figN.savefig(plots_dir / f"{a_date}_chunk{i:03d}_dNdlogDp.png", dpi=200)
                plt.close(figN)
                figV.savefig(plots_dir / f"{a_date}_chunk{i:03d}_dVdlogDp.png", dpi=200)
                plt.close(figV)

                # 8) collect + rebin
                tik_name  = next(iter(tik_specs.keys()))
                tik_edges = tik_specs[tik_name][1]
                tik_vals  = tik_specs[tik_name][2]

                fims_mids, fims_edges, fims_vals, _ = specs_opt["FIMS_applied"]
                if uh_label is not None:
                    uh_mids, uh_edges, uh_vals, _ = specs_opt[uh_label]
                if include_pops:
                    po_mids, po_edges, po_vals, _ = specs_opt[po_label]
                aps_mids, aps_edges, aps_vals, _ = specs_opt[aps_label]

                if common_edges is None:
                    common_edges = tik_edges

                fims_on_common   = remap_dndlog_by_edges_any(fims_edges,  common_edges, fims_vals)
                if uh_label is not None:
                    uhsas_on_common = remap_dndlog_by_edges_any(uh_edges, common_edges, uh_vals)
                else:
                    uhsas_on_common = np.full_like(fims_on_common, np.nan)
                if include_pops:
                    pops_on_common = remap_dndlog_by_edges_any(po_edges, common_edges, po_vals)
                aps_on_common    = remap_dndlog_by_edges_any(aps_edges,   common_edges, aps_vals)
                merged_on_common = remap_dndlog_by_edges_any(tik_edges,   common_edges, tik_vals)

                day_fims_algn.append(fims_on_common)
                day_uhsas_algn.append(uhsas_on_common)
                if include_pops:
                    day_pops_algn.append(pops_on_common)
                day_aps_algn.append(aps_on_common)
                day_merged.append(merged_on_common)
                day_times_start.append(t_start)
                day_times_end.append(t_end)
                day_incloud.append(inc_flag)
                day_n_fit.append(opt_res["n_fit"])
                if include_pops:
                    day_n_pops_fit.append(opt_res["n_pops_fit"])
                day_rho_fit.append(opt_res["rho_fit"])
                day_best_cost.append(opt_res["best_cost"])

            except Exception as e:
                failed_chunks.append((a_date, i, period["idx"], type(e).__name__, str(e)))
                err_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with error_log.open("a") as ef:
                    ef.write(f"[{err_ts}] ERROR on date {a_date} chunk {i:03d} (period_idx={period['idx']:03d})\n")
                    ef.write(f"t_start={t_start}, t_end={t_end}\n")
                    ef.write(f"{type(e).__name__}: {e}\n")
                    ef.write(traceback.format_exc())
                    ef.write("\n")
                with log_file.open("a") as f:
                    f.write(f"\t[ERROR] chunk {i:03d} failed, see error_log.txt\n")

        # write per-day
        if common_edges is not None and len(day_merged) > 0:
            write_day_netcdf(
                day_dir,
                a_date,
                day_fine_edges=np.asarray(common_edges, float),
                day_fims_algn=np.asarray(day_fims_algn, float),
                day_uhsas_algn=np.asarray(day_uhsas_algn, float),
                day_aps_algn=np.asarray(day_aps_algn, float),
                day_fine_vals=np.asarray(day_merged, float),
                day_times_start=day_times_start,
                day_times_end=day_times_end,
                day_incloud_flag=np.asarray(day_incloud, int),
                day_n_fit=np.asarray(day_n_fit, float),
                day_rho_fit=np.asarray(day_rho_fit, float),
                day_best_cost=np.asarray(day_best_cost, float),
                day_pops_algn=np.asarray(day_pops_algn, float) if include_pops else None,
                day_n_pops_fit=np.asarray(day_n_pops_fit, float) if include_pops else None,
                orig_APS_edges=np.asarray(orig_APS_edges, float) if orig_APS_edges is not None else None,
                orig_UHSAS_edges=np.asarray(orig_UHSAS_edges, float) if orig_UHSAS_edges is not None else None,
                orig_POPS_edges=np.asarray(orig_POPS_edges, float) if include_pops and orig_POPS_edges is not None else None,
                orig_FIMS_edges=np.asarray(orig_FIMS_edges, float) if orig_FIMS_edges is not None else None,
            )

    if failed_chunks:
        preview = "; ".join(
            f"{date} chunk {chunk:03d} ({err_type}: {msg})"
            for date, chunk, _period_idx, err_type, msg in failed_chunks[:3]
        )
        if len(failed_chunks) > 3:
            preview += f"; ... {len(failed_chunks) - 3} more"
        raise RuntimeError(
            f"{len(failed_chunks)} ARCSIX merge chunk(s) failed. "
            f"First failure(s): {preview}. See {error_log}."
        )


def _run_arcsix_merge_for_periods_direct(
    time_periods,
    data_dir,
    output_dir,
    *,
    instruments=("FIMS", "APS"),
    fims_lag=10,
    incloud_pad_s=10,
    min_samples_per_inst=50,
    min_overlap_s=None,
    overlap_freq="1s",
    fims_xmin=None,
    fims_xmax=None,
    fine_bin=200,
    smoothness_lam=1e-4,
    output_edges=None,
    alpha_fims=1.0,
    aps_combine_weight=1.0,
):
    """
    Run ARCSIX aerosol size distribution merge for a list of specified time periods,
    using ONLY FIMS and APS, with NO UHSAS reading and NO alignment optimization.
    This directly runs the final Tikhonov merge on FIMS + APS mean spectra.

    Parameters
    ----------
    time_periods : iterable of (str, str) or (datetime-like, datetime-like)
        Each entry is (start_time, end_time), e.g.
            [("2024-05-28 10:00", "2024-05-28 10:30"), ...]
        Treated as timezone-naive wall-clock datetimes. If timezone metadata is
        present, it is stripped without shifting the clock time.

    data_dir : str or Path
        Base ARCSIX_P3B data directory containing LARGE-APS and FIMS, etc.
    output_dir : str or Path
        Output directory for batch NetCDF + plots + logs.
    """

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_edges_arr = _validate_output_edges(output_edges)
    min_overlap_s = (
        None
        if min_overlap_s is None
        else _nonnegative_float(min_overlap_s, "min_overlap_s")
    )
    overlap_freq = str(overlap_freq)

    # instrument subdirs (NO UHSAS here)
    aps_dir   = data_dir / "LARGE-APS"
    fims_dir  = data_dir / "FIMS"
    inlet_dir = data_dir / "LARGE-InletFlag"
    micro_dir = data_dir / "LARGE-MICROPHYSICAL"

    log_file  = output_dir / "output_log.txt"
    error_log = output_dir / "error_log.txt"
    failed_chunks = []

    # ---------- parse periods, force tz-naive wall-clock datetimes ----------
    _periods = []
    for idx, (s_raw, e_raw) in enumerate(time_periods):
        s_ts = _as_timezone_naive_timestamp(s_raw)
        e_ts = _as_timezone_naive_timestamp(e_raw)

        if e_ts <= s_ts:
            raise ValueError(f"Period {idx}: end <= start ({s_ts} >= {e_ts})")

        date_str = s_ts.strftime("%Y-%m-%d")
        _periods.append(
            {
                "idx": idx,
                "start": s_ts,
                "end": e_ts,
                "date": date_str,
            }
        )

    # Unique dates we actually need to process
    dates = sorted({p["date"] for p in _periods})

    # Log header
    with log_file.open("a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write("ARCSIX Aerosol Size Distribution Merge Log\n")
        f.write(f"Generated: {ts} by Bo Chen\n\n")
        f.write(f"MODE: specified time periods ({'+'.join(instruments)}, direct no-alignment merge)\n")
        f.write("TIME_PERIODS:\n")
        for p in _periods:
            f.write(f"  [{p['idx']:03d}] {p['start']} -> {p['end']}\n")
        f.write("\n")

    for a_date in dates:
        periods_for_date = [p for p in _periods if p["date"] == a_date]
        if not periods_for_date:
            continue

        with log_file.open("a") as f:
            f.write(f"merging {a_date} for {len(periods_for_date)} specified period(s)\n")
            f.write("------------------------ SETTINGS ------------------------\n")
            f.write(f"DATA_DIR: {data_dir}\n")
            f.write(f"OUTPUT_DIR: {output_dir}\n")
            f.write(f"INSTRUMENTS: {instruments}\n")
            f.write("ALIGNMENT_STRATEGY: direct\n")
            f.write(f"FIMS_LAG: {fims_lag}  # shift FIMS <FIMS_LAG> seconds earlier\n")
            f.write(f"INCLOUD_PAD_S: {incloud_pad_s}  # seconds around inlet_flag to mark a chunk in-cloud\n")
            f.write(f"MIN_SAMPLES_PER_INST: {min_samples_per_inst}  # skip chunks with fewer than this\n")
            f.write(f"MIN_OVERLAP_S: {min_overlap_s}\n")
            f.write(f"OVERLAP_FREQ: {overlap_freq}\n")
            f.write(f"FIMS_XMIN: {fims_xmin}\n")
            f.write(f"FIMS_XMAX: {fims_xmax}\n")
            f.write(f"FINE_BIN: {fine_bin}  # number of fine bins for Tikhonov merged spec\n")
            output_edges_desc = (
                f"{output_edges_arr.size - 1} bins from {output_edges_arr[0]} to {output_edges_arr[-1]} nm"
                if output_edges_arr is not None
                else "native Tikhonov grid"
            )
            f.write(f"OUTPUT_EDGES: {output_edges_desc}\n")
            f.write(f"ALPHA_FIMS: {alpha_fims}\n")
            f.write(f"ALPHA_APS: {aps_combine_weight}\n")
            f.write(f"SMOOTHNESS_LAM: {smoothness_lam}\n")
            f.write("----------------------------------------------------------\n\n")

        day_dir = output_dir / a_date
        day_dir.mkdir(parents=True, exist_ok=True)
        totals_dir = day_dir / "time_series"
        plots_dir  = day_dir / "merge_plots"
        totals_dir.mkdir(exist_ok=True)
        plots_dir.mkdir(exist_ok=True)

        common_edges     = output_edges_arr.copy() if output_edges_arr is not None else None
        day_fims_algn    = []
        day_uhsas_algn   = []  # will be filled with NaN
        day_aps_algn     = []
        day_merged       = []
        day_times_start  = []
        day_times_end    = []
        day_incloud      = []
        orig_APS_edges   = None
        orig_UHSAS_edges = None  # always None here
        orig_FIMS_edges  = None
        day_n_fit        = []
        day_rho_fit      = []
        day_best_cost    = []

        # --------- load frames & drop timezone metadata without clock conversion ---------
        # APS + FIMS only
        filtered_frames = load_af_oneday(a_date, aps_dir, fims_dir)

        for name, df in list(filtered_frames.items()):
            filtered_frames[name] = _drop_timezone_from_index(df)

        # shift FIMS by -fims_lag seconds (indices already tz-naive now)
        if "FIMS" in filtered_frames and not filtered_frames["FIMS"].empty:
            filtered_frames["FIMS"] = filtered_frames["FIMS"].copy()
            filtered_frames["FIMS"].index = (
                filtered_frames["FIMS"].index - pd.Timedelta(seconds=fims_lag)
            )

        inlet_flag = read_inlet_flag(inlet_dir, start=a_date, end=None, prefix="ARCSIX")
        inlet_flag = _drop_timezone_from_index(inlet_flag)

        micro = read_microphysical(micro_dir, start=a_date, end=None, prefix="ARCSIX")
        micro = _drop_timezone_from_index(micro)

        cpc_total = pd.to_numeric(micro.get("CNgt10nm"), errors="coerce")

        # Loop over explicitly specified periods for this date
        for i, period in enumerate(periods_for_date):
            t_start = period["start"]
            t_end   = period["end"]

            try:
                with log_file.open("a") as f:
                    f.write(f"\tsizedist {i:03d} (period_idx={period['idx']:03d}): {t_start} -> {t_end}\n")

                # Build chunk by slicing each instrument in the period (APS + FIMS only)
                a_chunk = {}
                for name, df in filtered_frames.items():
                    if df is None or df.empty:
                        continue
                    sub = df.loc[t_start:t_end]
                    if not sub.empty:
                        a_chunk[name] = sub

                # empty window: no data from any instrument
                if not a_chunk:
                    with log_file.open("a") as f:
                        f.write(f"\t[SKIP] chunk {i:03d} empty window (no data in APS/FIMS)\n")
                    continue

                inlet_chunk     = inlet_flag.loc[t_start:t_end] if not inlet_flag.empty else inlet_flag.iloc[0:0]
                cpc_total_chunk = cpc_total.loc[t_start:t_end]  if not cpc_total.empty else cpc_total.iloc[0:0]

                # 0) in-cloud flag for this chunk
                inc_flag = chunk_is_incloud(inlet_flag, t_start, t_end, tol_s=incloud_pad_s)

                # 1) time series plot
                fig1, _ = plot_period_totals(
                    a_chunk,
                    title=f"{a_date} sizedist {i:03d}",
                    inlet_flag=inlet_flag,
                    gauss_win=10,
                    gauss_std=2,
                    cpc_total=cpc_total_chunk,
                    t_start=t_start,
                    t_end=t_end,
                )

                fig1.savefig(totals_dir / f"sizedist_{i:03d}_totals.png", dpi=150)
                plt.close(fig1)

                ##################################################################
                # 1.5) build a version of a_chunk with inlet-flagged data removed
                pad = pd.Timedelta(seconds=incloud_pad_s)
                inlet_for_filter = inlet_flag.loc[(t_start - pad):(t_end + pad)]
                if isinstance(inlet_for_filter, pd.DataFrame):
                    sflag = (inlet_for_filter.fillna(0) != 0).any(axis=1)
                else:
                    sflag = inlet_for_filter.fillna(0) != 0

                # duration of this specified window
                dur_s = (t_end - t_start).total_seconds()

                with log_file.open("a") as f:
                    f.write(
                        f"\t[INLET_FILTER] chunk {i:03d}: "
                        f"{len(inlet_for_filter)} inlet_flag samples within +/-{incloud_pad_s}s, "
                        f"{int(sflag.sum())} flagged, "
                        f"duration={dur_s:.1f} s\n"
                    )

                filtered_all = filter_chunk_by_inlet_flag(
                    a_chunk,
                    inlet_flag,
                    t_start,
                    t_end,
                    pad_s=incloud_pad_s,
                )
                filtered_chunk = {}

                for name, df in a_chunk.items():
                    if df is None or df.empty:
                        continue

                    n_before = len(df)
                    df2 = filtered_all.get(name)
                    if df2 is None:
                        df2 = df.iloc[0:0]
                    n_after = len(df2)

                    with log_file.open("a") as f:
                        f.write(
                            f"\t    {name}: before={n_before}, "
                            f"after={n_after}, "
                            f"removed={n_before - n_after}\n"
                        )

                    if not df2.empty:
                        filtered_chunk[name] = df2

                # if everything was nuked by inlet filter, skip this chunk
                if not filtered_chunk:
                    with log_file.open("a") as f:
                        f.write(f"\t[SKIP] chunk {i:03d} all data removed by inlet flag\n")
                    continue

                if min_overlap_s is not None:
                    ov_s = overlap_seconds_all_instruments(
                        filtered_chunk,
                        t_start,
                        t_end,
                        instruments=("APS", "FIMS"),
                        freq=overlap_freq,
                    )
                    if ov_s < min_overlap_s:
                        with log_file.open("a") as f:
                            f.write(
                                f"\t[SKIP] chunk {i:03d} insufficient overlap: "
                                f"{ov_s:d}s < {int(min_overlap_s)}s (freq={overlap_freq})\n"
                            )
                        continue

                # always use the filtered chunk for mean spectra
                chunk_for_specs = filtered_chunk
                ##################################################################

                # 2) mean specs (APS + FIMS only, UHSAS frame is absent)
                specs, line_kwargs, fill_kwargs, bin_counts = make_filtered_specs(
                    chunk_for_specs,
                    log_file,
                    order=("FIMS", "APS"),
                )

                # require APS + FIMS only
                if ("APS" not in specs) or ("FIMS" not in specs):
                    with log_file.open("a") as f:
                        f.write(
                            f"\t[SKIP] chunk {i:03d} missing instrument(s): "
                            f"APS={'APS' in specs}, FIMS={'FIMS' in specs}\n"
                        )
                    continue

                # gate on NON-ZERO AVERAGE per instrument (APS/FIMS)
                low_data_reason = None
                for _name in ("APS", "FIMS"):
                    if _name in bin_counts and len(bin_counts[_name]) > 0:
                        arr = np.asarray(bin_counts[_name], int)
                        nz = arr[arr > 0]
                        nz_avg = float(nz.mean()) if nz.size > 0 else 0.0
                        if nz_avg < min_samples_per_inst:
                            low_data_reason = f"{_name} nonzero_avg={nz_avg:.1f} < {min_samples_per_inst}"
                            break
                if low_data_reason is not None:
                    with log_file.open("a") as f:
                        f.write(f"\t[SKIP] chunk {i:03d} low data: {low_data_reason}\n")
                    continue

                # save original edges once
                if orig_APS_edges is None:
                    orig_APS_edges = specs["APS"][1]
                if orig_FIMS_edges is None:
                    orig_FIMS_edges = specs["FIMS"][1]

                # 3) direct Tikhonov merged (FIMS + APS only, NO alignment)
                m_FIMS, e_FIMS, y_FIMS, s_FIMS = specs["FIMS"]
                m_APS,  e_APS,  y_APS,  s_APS  = specs["APS"]

                # apply optional FIMS range
                if (fims_xmin is not None) or (fims_xmax is not None):
                    m_FIMS_sel, e_FIMS_sel, y_FIMS_sel, s_FIMS_sel = select_between(
                        m_FIMS, e_FIMS, y_FIMS, s_FIMS,
                        xmin=fims_xmin,
                        xmax=fims_xmax,
                    )
                else:
                    m_FIMS_sel, e_FIMS_sel, y_FIMS_sel, s_FIMS_sel = m_FIMS, e_FIMS, y_FIMS, s_FIMS

                tik_specs, tik_lines, tik_fills, tik_diag = make_tikhonov_merged_spec(
                    e_fims_sel=e_FIMS_sel,
                    y_fims_sel=y_FIMS_sel,
                    e_aps_fit=e_APS,
                    y_aps_fit=y_APS,
                    # no UHSAS in this mode
                    e_uhsas_fit=None,
                    y_uhsas_fit=None,
                    lam=smoothness_lam,
                    n_points=fine_bin,
                    alpha_fims=alpha_fims,
                    alpha_aps=aps_combine_weight,
                )

                specs_opt       = dict(specs)
                line_kwargs_opt = dict(line_kwargs)
                fill_kwargs_opt = dict(fill_kwargs)

                specs_opt.update(tik_specs)
                line_kwargs_opt.update(tik_lines)
                fill_kwargs_opt.update(tik_fills)

                # 4) plots
                (figN, axN), (figV, axV), (specs_V, merged_spec_V) = plot_sizedist_all(
                    specs=specs_opt,
                    merged_spec=tik_specs,
                    line_kwargs=line_kwargs_opt,
                    merged_line_kwargs=tik_lines,
                    fill_kwargs=fill_kwargs_opt,
                    merged_fill_kwargs=tik_fills,
                    inlet_flag=inlet_chunk,
                    d_str=a_date,
                )
                figN.savefig(plots_dir / f"{a_date}_chunk{i:03d}_dNdlogDp.png", dpi=200)
                plt.close(figN)
                figV.savefig(plots_dir / f"{a_date}_chunk{i:03d}_dVdlogDp.png", dpi=200)
                plt.close(figV)

                # 5) collect + rebin onto common fine grid
                tik_name  = next(iter(tik_specs.keys()))
                tik_edges = tik_specs[tik_name][1]
                tik_vals  = tik_specs[tik_name][2]

                fims_edges = e_FIMS_sel
                aps_edges  = e_APS

                if common_edges is None:
                    common_edges = tik_edges

                fims_on_common   = remap_dndlog_by_edges_any(fims_edges,  common_edges, y_FIMS_sel)
                aps_on_common    = remap_dndlog_by_edges_any(aps_edges,   common_edges, y_APS)
                merged_on_common = remap_dndlog_by_edges_any(tik_edges,   common_edges, tik_vals)

                # UHSAS aligned is just NaN placeholder
                uhsas_on_common  = np.full_like(fims_on_common, np.nan)

                day_fims_algn.append(fims_on_common)
                day_uhsas_algn.append(uhsas_on_common)
                day_aps_algn.append(aps_on_common)
                day_merged.append(merged_on_common)
                day_times_start.append(t_start)
                day_times_end.append(t_end)
                day_incloud.append(inc_flag)

                # No alignment -> NaNs for fit parameters
                day_n_fit.append(np.nan)
                day_rho_fit.append(np.nan)
                day_best_cost.append(np.nan)

            except Exception as e:
                failed_chunks.append((a_date, i, period["idx"], type(e).__name__, str(e)))
                err_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with error_log.open("a") as ef:
                    ef.write(f"[{err_ts}] ERROR on date {a_date} chunk {i:03d} (period_idx={period['idx']:03d})\n")
                    ef.write(f"t_start={t_start}, t_end={t_end}\n")
                    ef.write(f"{type(e).__name__}: {e}\n")
                    ef.write(traceback.format_exc())
                    ef.write("\n")
                with log_file.open("a") as f:
                    f.write(f"\t[ERROR] chunk {i:03d} failed, see error_log.txt\n")

        # write per-day
        if common_edges is not None and len(day_merged) > 0:
            write_day_netcdf(
                day_dir,
                a_date,
                day_fine_edges=np.asarray(common_edges, float),
                day_fims_algn=np.asarray(day_fims_algn, float),
                day_uhsas_algn=np.asarray(day_uhsas_algn, float),
                day_aps_algn=np.asarray(day_aps_algn, float),
                day_fine_vals=np.asarray(day_merged, float),
                day_times_start=day_times_start,
                day_times_end=day_times_end,
                day_incloud_flag=np.asarray(day_incloud, int),
                day_n_fit=np.asarray(day_n_fit, float),
                day_rho_fit=np.asarray(day_rho_fit, float),
                day_best_cost=np.asarray(day_best_cost, float),
                orig_APS_edges=np.asarray(orig_APS_edges, float) if orig_APS_edges is not None else None,
                orig_UHSAS_edges=None,
                orig_FIMS_edges=np.asarray(orig_FIMS_edges, float) if orig_FIMS_edges is not None else None,
            )

    if failed_chunks:
        preview = "; ".join(
            f"{date} chunk {chunk:03d} ({err_type}: {msg})"
            for date, chunk, _period_idx, err_type, msg in failed_chunks[:3]
        )
        if len(failed_chunks) > 3:
            preview += f"; ... {len(failed_chunks) - 3} more"
        raise RuntimeError(
            f"{len(failed_chunks)} ARCSIX direct merge chunk(s) failed. "
            f"First failure(s): {preview}. See {error_log}."
        )
