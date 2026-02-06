import sys
from pathlib import Path
BASE_DIR = Path.cwd().parent
sys.path.append(str(BASE_DIR / "src"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
from datetime import timedelta
from pathlib import Path
from netCDF4 import Dataset, stringtochar
from datetime import datetime, timezone
import traceback

from ict_utils import (read_aps, read_nmass, read_pops, read_uhsas, read_fims, read_inlet_flag, read_microphysical, 
    check_common_grid, filter_by_spectra_presence, mean_spectrum, get_spectra)
from sizedist_utils import (edges_from_mids_geometric, dvdlog_from_dndlog,  remap_dndlog_by_edges_any,
    dsdlog_from_dndlog, mids_from_edges, remap_dndlog_by_edges, select_between, delta_log10_from_edges)
from sizedist_alignment import optimize_multi_custom
from optical_diameter_core import SigmaLUT, convert_do_lut, RI_UHSAS_SRC
from diameter_conversion_core import da_to_dv
from sizedist_combine import make_grid_from_series, merge_sizedists_tikhonov
from sizedist_plot import plot_size_distributions, plot_size_distributions_steps


def load_aufi_oneday(a_date, aps_dir, uhsas_dir, fims_dir):
    # read raw APS/UHSAS/FIMS for the day, line them up on FIMS time grid, 
    # then drop any times where not all spectral instruments have valid data
    # (FIMS also must pass QC!=2)
    aps        = read_aps       (aps_dir,   start=a_date, end=None, prefix="ARCSIX")
    uhsas      = read_uhsas     (uhsas_dir, start=a_date, end=None, prefix="ARCSIX")
    fims       = read_fims      (fims_dir,  start=a_date, end=None, prefix="ARCSIX")

    frames = {"APS": aps, "UHSAS": uhsas, "FIMS": fims}
    _ = check_common_grid(frames, ref_key="FIMS", round_to=None)

    fims_qc = pd.to_numeric(fims.get("QC_Flag", pd.Series(index=fims.index)), errors="coerce")
    extra = {"FIMS": fims_qc.ne(2)}  # bad=2
    filtered, _keep = filter_by_spectra_presence(
        frames, col_prefix="dNdlogDp",
        min_instruments=None,
        extra_masks=extra, treat_nonpositive_as_nan=False
    )

    return filtered

def load_aufi_oneday_v2(a_date, aps_dir, uhsas_dir, fims_dir, pops_dir):
    """
    Enhanced loader: Aligns APS, UHSAS, FIMS, and POPS on the FIMS time grid
    and filters for concurrent valid spectra. 
    Matches the logic of 'load_aufi_oneday' exactly.
    """
    from ict_utils import (read_aps, read_uhsas, read_fims, read_pops, 
                           check_common_grid, filter_by_spectra_presence)

    # 1. Read raw instruments with 'ARCSIX' prefix (Matches V1)
    aps   = read_aps(aps_dir,     start=a_date, end=None, prefix="ARCSIX")
    uhsas = read_uhsas(uhsas_dir, start=a_date, end=None, prefix="ARCSIX")
    fims  = read_fims(fims_dir,   start=a_date, end=None, prefix="ARCSIX")
    pops  = read_pops(pops_dir,   start=a_date, end=None, prefix="ARCSIX")

    frames = {"APS": aps, "UHSAS": uhsas, "FIMS": fims, "POPS": pops}
    
    # Drop any that are completely missing for the day to avoid alignment errors
    frames = {k: v for k, v in frames.items() if (v is not None and not v.empty)}
    
    if "FIMS" not in frames:
        return {} # Need the reference grid

    # 2. ALIGN everything on FIMS time grid (Matches V1)
    # This ensures every row in every dataframe has the exact same timestamp index.
    _ = check_common_grid(frames, ref_key="FIMS", round_to=None)

    # 3. Apply FIMS QC logic (Matches V1)
    fims_qc = pd.to_numeric(frames["FIMS"].get("QC_Flag", pd.Series(index=frames["FIMS"].index)), errors="coerce")
    extra = {"FIMS": fims_qc.ne(2)}  # QC=2 is bad
    
    # 4. Filter for concurrent valid data (Matches V1)
    # min_instruments=None forces it to only keep rows where ALL instruments in 'frames' have spectra
    filtered, _keep = filter_by_spectra_presence(
        frames, col_prefix="dNdlogDp",
        min_instruments=None,
        extra_masks=extra, treat_nonpositive_as_nan=False
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

    ax.set_xlabel("Time (UTC)", labelpad=6)
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
    aps: pd.DataFrame,
    uhsas: pd.DataFrame,
    fims: pd.DataFrame,
    log_path: Path | str,
    line_kwargs=None,
    fill_kwargs=None,
):
    specs = {}
    bin_counts = {}  # NEW

    # APS
    if "APS" in frames and not frames["APS"].empty:
        out_aps = mean_spectrum_with_edges(frames["APS"], "APS")
        if out_aps is not None:
            m_APS, e_APS, y_APS, s_APS, _, n_APS = out_aps
            log_bin_table_horizontal(log_path, "APS", m_APS, n_APS)
            specs["APS"] = (m_APS, e_APS, y_APS, s_APS)
            bin_counts["APS"] = n_APS

    # UHSAS
    if "UHSAS" in frames and not frames["UHSAS"].empty:
        out_uh = mean_spectrum_with_edges(frames["UHSAS"], "UHSAS")
        if out_uh is not None:
            m_UHSAS, e_UHSAS, y_UHSAS, s_UHSAS, _, n_UHSAS = out_uh
            log_bin_table_horizontal(log_path, "UHSAS", m_UHSAS, n_UHSAS)
            specs["UHSAS"] = (m_UHSAS, e_UHSAS, y_UHSAS, s_UHSAS)
            bin_counts["UHSAS"] = n_UHSAS

    # FIMS
    if "FIMS" in frames and not frames["FIMS"].empty:
        out_f = mean_spectrum_with_edges(frames["FIMS"], "FIMS")
        if out_f is not None:
            m_FIMS, e_FIMS, y_FIMS, s_FIMS, _, n_FIMS = out_f
            log_bin_table_horizontal(log_path, "FIMS", m_FIMS, n_FIMS)
            specs["FIMS"] = (m_FIMS, e_FIMS, y_FIMS, s_FIMS)
            bin_counts["FIMS"] = n_FIMS

    if line_kwargs is None:
        line_kwargs = {
            "_default": {"linewidth": 1.5, "color": "k"},
            "APS":   {"color": "tab:blue",  "alpha": 0.6, "ls": "dashed"},
            "UHSAS": {"color": "tab:green", "alpha": 0.6, "ls": "dashed"},
            "FIMS":  {"color": "tab:red",               "ls": "dashed"},
        }

    if fill_kwargs is None:
        fill_kwargs = {
            "_default": {"alpha": 0.1, "color": "k"},
            "APS":   False,
            "UHSAS": False,
            "FIMS":  {"alpha": 0.1, "color": "tab:red"},
        }

    return specs, line_kwargs, fill_kwargs, bin_counts


def make_filtered_specs_v2(
    frames: dict[str, pd.DataFrame],
    aps: pd.DataFrame,      # Added to match your call
    uhsas: pd.DataFrame,    # Added to match your call
    fims: pd.DataFrame,     # Added to match your call
    pops: pd.DataFrame,     # Added to match your call
    log_path: Path | str,
    line_kwargs=None,
    fill_kwargs=None,
):
    """Averages all instruments (including POPS) and prepares plot styles."""
    specs = {}
    bin_counts = {}

    # Standard order for consistent logging/plotting
    order = ["FIMS", "UHSAS", "POPS", "APS"]
    
    for name in order:
        # We use 'frames' (the a_chunk dict) because it's cleaner,
        # but the separate DFs are now accepted so the script won't crash.
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
    a = np.asarray(a)
    nz = a[a > 0]
    return nz.min() if nz.size else 0


def _load_uhsas_lut(lut_dir: str | Path | None):
    if lut_dir is None:
        base = Path.cwd().parent
        lut_path = base / "lut" / "uhsas_sigma_col_1054nm.zarr"
    else:
        lut_path = Path(lut_dir) / "uhsas_sigma_col_1054nm.zarr"
    return SigmaLUT(str(lut_path))

def _load_pops_lut(lut_dir: str | Path | None):
    if lut_dir is None:
        base = Path.cwd().parent
        lut_path = base / "lut" / "pops_sigma_col_405nm.zarr"
    else:
        lut_path = Path(lut_dir) / "pops_sigma_col_405nm.zarr"
    return SigmaLUT(str(lut_path))

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


def _select_ranges(
    m_UHSAS, e_UHSAS, y_UHSAS, s_UHSAS,
    m_FIMS, e_FIMS, y_FIMS, s_FIMS,
    *, uhsas_xmin, uhsas_xmax, fims_xmin, fims_xmax,
):
    m_uhsas_sel, e_uhsas_sel, y_uhsas_sel, s_uhsas_sel = select_between(
        m_UHSAS, e_UHSAS, y_UHSAS, s_UHSAS, xmin=uhsas_xmin, xmax=uhsas_xmax
    )
    m_fims_sel, e_fims_sel, y_fims_sel, s_fims_sel = select_between(
        m_FIMS, e_FIMS, y_FIMS, s_FIMS, xmin=fims_xmin, xmax=fims_xmax
    )
    return (
        m_uhsas_sel, e_uhsas_sel, y_uhsas_sel, s_uhsas_sel,
        m_fims_sel, e_fims_sel, y_fims_sel, s_fims_sel,
    )


def _build_instruments(e_uhsas_sel, y_uhsas_sel, e_APS, y_APS, lut_uhsas):
    return [
        {
            "edges":   e_uhsas_sel,
            "y":       y_uhsas_sel,
            "remap_fn": _uhsas_remap_fn,
            "kwargs":  {"lut": lut_uhsas, "ri_src": RI_UHSAS_SRC, "response_bins": 50},
            "w_ref":   1.0,
        },
        {
            "edges":   e_APS,
            "y":       y_APS,
            "remap_fn": _aps_remap_fn,
            "kwargs":  {"chi_t": 1.0, "rho0": 1000.0, "pres_hPa": 1013.25, "temp_C": 20.0},
            "w_ref":   1.0,
        },
    ]


def plot_history(hist):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    iters = np.arange(1, len(hist["total"]) + 1)
    ax.plot(iters, hist["total"], label="total", linewidth=2, color="k")
    ax.set_xlabel("DE iteration")
    ax.set_ylabel("MSE")
    ax.set_title("UHSAS + APS optimization history")
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
    # ---- unpack directly (no helper) ----
    m_APS,   e_APS,   y_APS,   s_APS   = specs["APS"]
    m_UHSAS, e_UHSAS, y_UHSAS, s_UHSAS = specs["UHSAS"]
    m_FIMS,  e_FIMS,  y_FIMS,  s_FIMS  = specs["FIMS"]

    lut_uhsas = _load_uhsas_lut(lut_dir)

    (
        m_uhsas_sel, e_uhsas_sel, y_uhsas_sel, s_uhsas_sel,
        m_fims_sel,  e_fims_sel,  y_fims_sel,  s_fims_sel,
    ) = _select_ranges(
        m_UHSAS, e_UHSAS, y_UHSAS, s_UHSAS,
        m_FIMS,  e_FIMS,  y_FIMS,  s_FIMS,
        uhsas_xmin=uhsas_xmin, uhsas_xmax=uhsas_xmax,
        fims_xmin=fims_xmin,   fims_xmax=fims_xmax,
    )

    # keep a copy of the actually used FIMS
    specs["FIMS_applied"] = (m_fims_sel, e_fims_sel, y_fims_sel, s_fims_sel)
    line_kwargs["FIMS_applied"] = {"color": "tab:red", "linewidth": 2.0}
    fill_kwargs["FIMS_applied"] = False

    instruments = _build_instruments(e_uhsas_sel, y_uhsas_sel, e_APS, y_APS, lut_uhsas)
    bounds_list = [list(uhsas_bounds), list(aps_bounds)]
    pair_weights = [(0, 1, pair_w)] if pair_w != 0 else None

    # ---- run DE ----
    best_thetas, best_cost, res, hist = optimize_multi_custom(
        ref_mids=m_fims_sel,
        ref_y=y_fims_sel,
        instruments=instruments,
        bounds_list=bounds_list,
        moment=moment,
        space=space,
        pair_weights=pair_weights,
        maxiter=200,
        tol=1e-6,
        seed=123,
    )

    theta_uhsas, theta_aps = best_thetas
    n_fit   = float(theta_uhsas[0])
    rho_fit = float(theta_aps[0])

    # ---- apply fitted params ----
    e_UHSAS_fit = _uhsas_remap_fn(
        e_uhsas_sel, [n_fit],
        lut=lut_uhsas, ri_src=RI_UHSAS_SRC, response_bins=120,
    )
    m_UHSAS_fit = mids_from_edges(e_UHSAS_fit)
    y_UHSAS_fit = remap_dndlog_by_edges(e_uhsas_sel, e_UHSAS_fit, y_uhsas_sel)
    s_UHSAS_fit = remap_dndlog_by_edges(e_uhsas_sel, e_UHSAS_fit, s_uhsas_sel)

    e_APS_fit = _aps_remap_fn(
        e_APS, [rho_fit],
        chi_t=1.0, rho0=1000.0, pres_hPa=1013.25, temp_C=20.0,
    )
    m_APS_fit = mids_from_edges(e_APS_fit)
    y_APS_fit = remap_dndlog_by_edges(e_APS, e_APS_fit, y_APS)
    s_APS_fit = remap_dndlog_by_edges(e_APS, e_APS_fit, s_APS)

    # ---- append to specs for later plotting ----
    uh_label = f"UHSAS fit (n={n_fit:.3f})"
    specs[uh_label] = (m_UHSAS_fit, e_UHSAS_fit, y_UHSAS_fit, s_UHSAS_fit)
    line_kwargs[uh_label] = {"color": "tab:green", "linewidth": 2.0}
    fill_kwargs[uh_label] = {"alpha": 0.10, "color": "tab:green"}

    aps_label = f"APS fit (ρ={rho_fit*0.001:.3f} g/cm$^3$)"
    specs[aps_label] = (m_APS_fit, e_APS_fit, y_APS_fit, s_APS_fit)
    line_kwargs[aps_label] = {"color": "tab:blue", "linewidth": 2.0}
    fill_kwargs[aps_label] = {"alpha": 0.10, "color": "tab:blue"}

    # return hist only; you plot outside
    return specs, line_kwargs, fill_kwargs, \
    { "n_fit": n_fit, "rho_fit": rho_fit, "best_cost": best_cost, "hist": hist,}


def run_joint_optimization_v2(
    specs: dict, line_kwargs: dict, fill_kwargs: dict,
    *,
    moment: str = "V", space: str = "linear",
    pair_w: float = 1.0, uhsas_bounds=((1.3, 1.8),),
    aps_bounds=((950.0, 2000.0),), uhsas_xmin=200,
    uhsas_xmax=None, fims_xmin=None,
    fims_xmax=400, pops_xmin=None, pops_xmax=None,
    lut_dir=None
):
    """
    4-Instrument Version: Jointly optimizes UHSAS, POPS, and APS parameters relative to FIMS.
    Follows the exact manual structure of run_joint_uhsas_aps_opt_from_specs.
    """
    from sizedist_utils import mids_from_edges, remap_dndlog_by_edges, select_between

    # 1. Unpack all instruments directly
    m_APS,   e_APS,   y_APS,   s_APS   = specs["APS"]
    m_UHSAS, e_UHSAS, y_UHSAS, s_UHSAS = specs["UHSAS"]
    m_POPS,  e_POPS,  y_POPS,  s_POPS  = specs["POPS"]
    m_FIMS,  e_FIMS,  y_FIMS,  s_FIMS  = specs["FIMS"]

    # 2. Load LUTs
    lut_uhsas = _load_uhsas_lut(lut_dir)
    lut_pops  = _load_pops_lut(lut_dir)

    # 3. Select fitting ranges
    (
        m_uhsas_sel, e_uhsas_sel, y_uhsas_sel, s_uhsas_sel,
        m_fims_sel,  e_fims_sel,  y_fims_sel,  s_fims_sel,
    ) = _select_ranges(
        m_UHSAS, e_UHSAS, y_UHSAS, s_UHSAS,
        m_FIMS,  e_FIMS,  y_FIMS,  s_FIMS,
        uhsas_xmin=uhsas_xmin, uhsas_xmax=uhsas_xmax,
        fims_xmin=fims_xmin,   fims_xmax=fims_xmax,
    )
    
    # Manually select POPS range using the same utility
    _, e_pops_sel, y_pops_sel, s_pops_sel = select_between(
        m_POPS, e_POPS, y_POPS, s_POPS, xmin=pops_xmin, xmax=pops_xmax
    )

    # 4. Save copy of applied FIMS for plotting
    specs["FIMS_applied"] = (m_fims_sel, e_fims_sel, y_fims_sel, s_fims_sel)
    line_kwargs["FIMS_applied"] = {"color": "tab:red", "linewidth": 2.0}
    fill_kwargs["FIMS_applied"] = False

    # 5. Build manual instrument list
    instruments = [
        {
            "edges":   e_uhsas_sel, "y": y_uhsas_sel, "w_ref": 1.0,
            "remap_fn": _uhsas_remap_fn,
            "kwargs":  {"lut": lut_uhsas, "ri_src": RI_UHSAS_SRC, "response_bins": 50},
        },
        {
            "edges":   e_pops_sel, "y": y_pops_sel, "w_ref": 1.0,
            "remap_fn": _pops_remap_fn,
            "kwargs":  {"lut": lut_pops, "ri_src": RI_UHSAS_SRC, "response_bins": 50},
        },
        {
            "edges":   e_APS, "y": y_APS, "w_ref": 1.0,
            "remap_fn": _aps_remap_fn,
            "kwargs":  {"chi_t": 1.0, "rho0": 1000.0, "pres_hPa": 1013.25, "temp_C": 20.0},
        },
    ]
    
    # 6. Set bounds and pair weights (UHSAS-APS and POPS-APS)
    bounds_list = [list(uhsas_bounds), list(uhsas_bounds), list(aps_bounds)]
    pair_weights = []
    if pair_w != 0:
        pair_weights.append((0, 2, pair_w)) # UHSAS vs APS
        pair_weights.append((1, 2, pair_w)) # POPS vs APS

    # 7. Run DE optimization
    best_thetas, best_cost, res, hist = optimize_multi_custom(
        ref_mids=m_fims_sel,
        ref_y=y_fims_sel,
        instruments=instruments,
        bounds_list=bounds_list,
        moment=moment,
        space=space,
        pair_weights=pair_weights if pair_weights else None,
        maxiter=200,
        tol=1e-6,
        seed=123,
    )

    theta_uhsas, theta_pops, theta_aps = best_thetas
    n_fit_uh = float(theta_uhsas[0])
    n_fit_po = float(theta_pops[0])
    rho_fit  = float(theta_aps[0])

    # 8. Apply fitted params and update specs
    # UHSAS Fit
    e_uh_fit = _uhsas_remap_fn(e_uhsas_sel, [n_fit_uh], lut=lut_uhsas, ri_src=RI_UHSAS_SRC, response_bins=120)
    uh_label = f"UHSAS fit (n={n_fit_uh:.3f})"
    specs[uh_label] = (mids_from_edges(e_uh_fit), e_uh_fit, 
                       remap_dndlog_by_edges(e_uhsas_sel, e_uh_fit, y_uhsas_sel),
                       remap_dndlog_by_edges(e_uhsas_sel, e_uh_fit, s_uhsas_sel))
    line_kwargs[uh_label] = {"color": "tab:green", "linewidth": 2.0}
    fill_kwargs[uh_label] = {"alpha": 0.1, "color": "tab:green"}

    # POPS Fit
    e_po_fit = _pops_remap_fn(e_pops_sel, [n_fit_po], lut=lut_pops, ri_src=RI_UHSAS_SRC, response_bins=120)
    po_label = f"POPS fit (n={n_fit_po:.3f})"
    specs[po_label] = (mids_from_edges(e_po_fit), e_po_fit,
                       remap_dndlog_by_edges(e_pops_sel, e_po_fit, y_pops_sel),
                       remap_dndlog_by_edges(e_pops_sel, e_po_fit, s_pops_sel))
    line_kwargs[po_label] = {"color": "tab:orange", "linewidth": 2.0}
    fill_kwargs[po_label] = {"alpha": 0.1, "color": "tab:orange"}

    # APS Fit
    e_aps_fit = _aps_remap_fn(e_APS, [rho_fit], chi_t=1.0, rho0=1000.0)
    aps_label = f"APS fit (ρ={rho_fit*0.001:.3f} g/cm$^3$)"
    specs[aps_label] = (mids_from_edges(e_aps_fit), e_aps_fit,
                        remap_dndlog_by_edges(e_APS, e_aps_fit, y_APS),
                        remap_dndlog_by_edges(e_APS, e_aps_fit, s_APS))
    line_kwargs[aps_label] = {"color": "tab:blue", "linewidth": 2.0}
    fill_kwargs[aps_label] = {"alpha": 0.1, "color": "tab:blue"}

    # 9. Return history and parameters for logging
    return specs, line_kwargs, fill_kwargs, {
        "n_fit": n_fit_uh,
        "n_pops_fit": n_fit_po,
        "rho_fit": rho_fit,
        "best_cost": best_cost,
        "hist": hist,
    }


def make_tikhonov_merged_spec(
    *,
    e_fims_sel,
    y_fims_sel,
    e_aps_fit,
    y_aps_fit,
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


def make_consensus_merged_spec_v2(
    *,
    e_fims_sel,
    y_fims_sel,
    e_uhsas_fit,
    y_uhsas_fit,
    e_pops_fit,
    y_pops_fit,
    e_aps_fit,
    y_aps_fit,
    lam: float = 1e-4,
    n_points: int = 200,
    alpha_fims: float = 1.0,
    alpha_uhsas: float = 1.0,
    alpha_pops: float = 1.0,
    alpha_aps: float = 1.0,
    c_punish: float = 1.0,
    data_space: str = "linear",  # Added to match your low-level function
):
    """
    Consensus Merger v2: Now correctly calls merge_sizedists_tikhonov_consensus
    to utilize the voting/consensus logic and the 'c' penalty parameter.
    """
    from sizedist_utils import mids_from_edges, edges_from_mids_geometric
    # Import the CORRECT consensus function
    from sizedist_combine import make_grid_from_series, merge_sizedists_tikhonov_consensus

    # 1. Get midpoints for the inversion
    m_fims  = mids_from_edges(e_fims_sel)
    m_uhsas = mids_from_edges(e_uhsas_fit)
    m_pops  = mids_from_edges(e_pops_fit)
    m_aps   = mids_from_edges(e_aps_fit)

    # 2. Build the series list
    series = [
        {"x": m_fims,  "y": y_fims_sel,  "alpha": float(alpha_fims)},
        {"x": m_uhsas, "y": y_uhsas_fit, "alpha": float(alpha_uhsas)},
        {"x": m_pops,  "y": y_pops_fit,  "alpha": float(alpha_pops)},
        {"x": m_aps,   "y": y_aps_fit,   "alpha": float(alpha_aps)},
    ]

    # 3. Create the high-resolution common grid (Dg)
    Dg = make_grid_from_series(series, n_points=n_points)

    # 4. Perform CONSENSUS Tikhonov merging
    # THIS is where c_punish (passed as 'c') actually gets used for voting
    y_merged, wsum, diag = merge_sizedists_tikhonov_consensus(
        Dg,
        series,
        lam=lam,
        eps=1e-12,
        nonneg=True,
        c=c_punish,          # <--- Logic: w = exp(-0.5 * (z / c) ** 2)
        data_space=data_space
    )

    # 5. Generate edges for the new merged grid
    Dg_edge = edges_from_mids_geometric(Dg)

    # 6. Format output for 'plot_sizedist_all'
    specs_label = "Merged (F+U+P+A)"
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
    orig_APS_edges: np.ndarray | None = None,
    orig_UHSAS_edges: np.ndarray | None = None,
    orig_FIMS_edges: np.ndarray | None = None,
):
    nc_path = day_dir / f"{a_date}_sizedist_merged.nc"

    n_chunk = len(day_fine_vals)
    n_fine  = len(day_fine_edges) - 1 if day_fine_edges is not None else 0

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
        v_aps  = nc.createVariable("aps_aligned_dNdlogDp",   "f8", ("chunk", "fine_bin"))
        v_mrg  = nc.createVariable("merged_dNdlogDp",        "f8", ("chunk", "fine_bin"))

        v_fims[:, :] = np.asarray(day_fims_algn)
        v_uhsa[:, :] = np.asarray(day_uhsas_algn)
        v_aps[:, :]  = np.asarray(day_aps_algn)
        v_mrg[:, :]  = np.asarray(day_fine_vals)

        v_fims.long_name = "FIMS dN/dlog10Dp rebinned to common fine edges"
        v_fims.units = "#/cm3"
        v_uhsa.long_name = "UHSAS dN/dlog10Dp (fitted) rebinned to common fine edges"
        v_uhsa.units = "#/cm3"
        v_aps.long_name = "APS dN/dlog10Dp (density-corrected) rebinned to common fine edges"
        v_aps.units = "#/cm3"
        v_mrg.long_name = "Tikhonov-merged dN/dlog10Dp on common fine edges"
        v_mrg.units = "#/cm3"

        v_inc = nc.createVariable("inlet_incloud_flag", "i4", ("chunk",))
        v_inc[:] = np.asarray(day_incloud_flag, int)
        v_inc.long_name = "1 = this chunk is within ±10 s of inlet in-cloud flag; 0 = otherwise"

        v_n = nc.createVariable("retrieved_uhsas_n_fit", "f8", ("chunk",))
        v_n[:] = np.asarray(day_n_fit, float)
        v_n.long_name = "Retrieved UHSAS real refractive index (n) from joint optimization"
        v_n.units = "1"

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
        if orig_FIMS_edges is not None:
            v = nc.createVariable("fims_edges_nm", "f8", ("fims_edge",))
            v[:] = orig_FIMS_edges
            v.long_name = "Original FIMS diameter bin edges"
            v.units = "nm"

        nc.description = "ARCSIX size distribution merge using FIMS-UHSAS-APS."
        nc.date_merged = a_date
        nc.source = "merge_production.write_day_netcdf"
        nc.conventions = "CF-1.8 (partial)"

    return nc_path


def write_day_netcdf_v2(
    day_dir: Path,
    a_date: str,
    *,
    day_fine_edges: np.ndarray,
    day_fims_algn: list[np.ndarray],
    day_uhsas_algn: list[np.ndarray],
    day_pops_algn: list[np.ndarray],   # Added POPS
    day_aps_algn: list[np.ndarray],
    day_fine_vals: list[np.ndarray],
    day_times_start: list,
    day_times_end: list,
    day_incloud_flag: list[int],
    day_n_fit: list[float],            # For UHSAS
    day_n_pops_fit: list[float],       # Added POPS fit
    day_rho_fit: list[float],
    day_best_cost: list[float],
    orig_APS_edges: np.ndarray | None = None,
    orig_UHSAS_edges: np.ndarray | None = None,
    orig_POPS_edges: np.ndarray | None = None, # Added POPS edges
    orig_FIMS_edges: np.ndarray | None = None,
):
    nc_path = day_dir / f"{a_date}_sizedist_merged_v2.nc"

    n_chunk = len(day_fine_vals)
    n_fine  = len(day_fine_edges) - 1 if day_fine_edges is not None else 0

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

        nc.time_start_list = t_start_strs
        nc.time_end_list   = t_end_strs
        if base_iso:
            nc.base_time_iso = base_iso

        if base_iso:
            strlen = len(base_iso)
            nc.createDimension("base_time_strlen", strlen)
            v_base = nc.createVariable("base_time_iso", "S1", ("base_time_strlen",))
            v_base[:] = stringtochar(np.array([base_iso], dtype=f"S{strlen}"))

        v_t0 = nc.createVariable("time_start_since_base_s", "f8", ("chunk",))
        v_t1 = nc.createVariable("time_end_since_base_s",   "f8", ("chunk",))
        v_t0[:] = np.asarray(start_since, float)
        v_t1[:] = np.asarray(end_since, float)
        v_t0.units = "s"
        v_t1.units = "s"

        v_fe = nc.createVariable("fine_edges_nm", "f8", ("fine_edge",))
        v_fe[:] = day_fine_edges
        v_fe.long_name = "Common fine diameter bin edges"
        v_fe.units = "nm"

        v_fims = nc.createVariable("fims_aligned_dNdlogDp", "f8", ("chunk", "fine_bin"))
        v_uhsa = nc.createVariable("uhsas_aligned_dNdlogDp", "f8", ("chunk", "fine_bin"))
        v_pops = nc.createVariable("pops_aligned_dNdlogDp",  "f8", ("chunk", "fine_bin"))
        v_aps  = nc.createVariable("aps_aligned_dNdlogDp",   "f8", ("chunk", "fine_bin"))
        v_mrg  = nc.createVariable("merged_dNdlogDp",        "f8", ("chunk", "fine_bin"))

        v_fims[:, :] = np.asarray(day_fims_algn)
        v_uhsa[:, :] = np.asarray(day_uhsas_algn)
        v_pops[:, :] = np.asarray(day_pops_algn)
        v_aps[:, :]  = np.asarray(day_aps_algn)
        v_mrg[:, :]  = np.asarray(day_fine_vals)

        v_fims.long_name = "FIMS dN/dlog10Dp rebinned to common fine edges"
        v_fims.units = "#/cm3"
        v_uhsa.long_name = "UHSAS dN/dlog10Dp (fitted) rebinned to common fine edges"
        v_uhsa.units = "#/cm3"
        v_pops.long_name = "POPS dN/dlog10Dp (fitted) rebinned to common fine edges"
        v_pops.units = "#/cm3"
        v_aps.long_name = "APS dN/dlog10Dp (density-corrected) rebinned to common fine edges"
        v_aps.units = "#/cm3"
        v_mrg.long_name = "Tikhonov-Consensus merged dN/dlog10Dp on common fine edges"
        v_mrg.units = "#/cm3"

        v_inc = nc.createVariable("inlet_incloud_flag", "i4", ("chunk",))
        v_inc[:] = np.asarray(day_incloud_flag, int)

        v_n = nc.createVariable("retrieved_uhsas_n_fit", "f8", ("chunk",))
        v_n[:] = np.asarray(day_n_fit, float)
        v_n.long_name = "Retrieved UHSAS real refractive index (n) from joint optimization"
        v_n.units = "1"

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

        if orig_APS_edges is not None:
            nc.createVariable("aps_edges_nm", "f8", ("aps_edge",))[:] = orig_APS_edges
        if orig_UHSAS_edges is not None:
            nc.createVariable("uhsas_edges_nm", "f8", ("uhsas_edge",))[:] = orig_UHSAS_edges
        if orig_POPS_edges is not None:
            nc.createVariable("pops_edges_nm", "f8", ("pops_edge",))[:] = orig_POPS_edges
        if orig_FIMS_edges is not None:
            nc.createVariable("fims_edges_nm", "f8", ("fims_edge",))[:] = orig_FIMS_edges

        nc.description = "ARCSIX size distribution merge using FIMS-UHSAS-POPS-APS."
        nc.date_merged = a_date
        nc.source = "merge_production.write_day_netcdf_v2"
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
    moment="V",
    space="linear",
    pair_w=1.0,
    bounds_uhsas=((1.3, 1.8),),
    bounds_aps=((950.0, 2000.0),),
    uhsas_xmin=200,
    uhsas_xmax=None,
    fims_xmin=None,
    fims_xmax=500,
    lut_dir=None,
    fine_bin=200,
    uhsas_combine_weight=0.5,
):
    """
    Run ARCSIX aerosol size distribution merge for a list of specified time periods.

    Parameters
    ----------
    time_periods : iterable of (str, str) or (datetime-like, datetime-like)
        Each entry is (start_time, end_time), e.g.
            [("2024-05-28 10:00", "2024-05-28 10:30"), ...]
        Interpreted as UTC, handled as tz-naive (no timezone conversions).

    data_dir : str or Path
        Base ARCSIX_P3B data directory containing LARGE-APS, PUTLS-UHSAS, FIMS, etc.
    output_dir : str or Path
        Output directory for batch NetCDF + plots + logs.
    """

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # instrument subdirs
    aps_dir   = data_dir / "LARGE-APS"
    uhsas_dir = data_dir / "PUTLS-UHSAS"
    fims_dir  = data_dir / "FIMS"
    inlet_dir = data_dir / "LARGE-InletFlag"
    micro_dir = data_dir / "LARGE-MICROPHYSICAL"

    if lut_dir is None:
        lut_dir = BASE_DIR / "lut"
    else:
        lut_dir = Path(lut_dir)

    log_file  = output_dir / "output_log.txt"
    error_log = output_dir / "error_log.txt"

    # ---------- parse periods, force tz-naive (UTC interpreted but no tz info) ----------
    _periods = []
    for idx, (s_raw, e_raw) in enumerate(time_periods):
        s_ts = pd.to_datetime(s_raw)
        e_ts = pd.to_datetime(e_raw)

        # drop timezone if present -> tz-naive
        if isinstance(s_ts, pd.Timestamp) and s_ts.tz is not None:
            s_ts = s_ts.tz_convert(None)
        if isinstance(e_ts, pd.Timestamp) and e_ts.tz is not None:
            e_ts = e_ts.tz_convert(None)

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
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
        f.write("ARCSIX Aerosol Size Distribution Merge Log\n")
        f.write(f"Generated: {ts} by Bo Chen\n\n")
        f.write("MODE: specified time periods (no 5-min daily splitting)\n")
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
            f.write(f"FIMS_LAG: {fims_lag}  # shift FIMS <FIMS_LAG> seconds earlier\n")
            f.write(f"INCLOUD_PAD_S: {incloud_pad_s}  # seconds around inlet_flag to mark a chunk in-cloud\n")
            f.write(f"MIN_SAMPLES_PER_INST: {min_samples_per_inst}  # skip chunks with fewer than this\n")
            f.write(f"MOMENT: {moment}  # moment used in joint optimization (N/S/V)\n")
            f.write(f"SPACE: {space}  # cost space (linear/log)\n")
            f.write(f"PAIR_W: {pair_w}  # cross-consistency weight UHSAS–APS\n")
            f.write(f"BOUNDS_UHSAS: {bounds_uhsas}  # real refractive index lower/upper\n")
            f.write(f"BOUNDS_APS: {bounds_aps}  # density lower/upper (kg m-3)\n")
            f.write(f"UHSAS_XMIN: {uhsas_xmin}\n")
            f.write(f"UHSAS_XMAX: {uhsas_xmax}\n")
            f.write(f"FIMS_XMIN: {fims_xmin}\n")
            f.write(f"FIMS_XMAX: {fims_xmax}\n")
            f.write(f"LUT_DIR: {lut_dir}\n")
            f.write(f"FINE_BIN: {fine_bin}  # number of fine bins for Tikhonov merged spec\n")
            f.write(f"UHSAS_COMBINE_WEIGHT: {uhsas_combine_weight}\n")
            f.write("----------------------------------------------------------\n\n")

        day_dir = output_dir / a_date
        day_dir.mkdir(parents=True, exist_ok=True)
        totals_dir = day_dir / "time_series"
        opt_dir    = day_dir / "loss_curve"
        plots_dir  = day_dir / "merge_plots"
        totals_dir.mkdir(exist_ok=True)
        opt_dir.mkdir(exist_ok=True)
        plots_dir.mkdir(exist_ok=True)

        common_edges     = None
        day_fims_algn    = []
        day_uhsas_algn   = []
        day_aps_algn     = []
        day_merged       = []
        day_times_start  = []
        day_times_end    = []
        day_incloud      = []
        orig_APS_edges   = None
        orig_UHSAS_edges = None
        orig_FIMS_edges  = None
        day_n_fit        = []
        day_rho_fit      = []
        day_best_cost    = []

        # --------- load frames & drop timezone from indices (tz-naive like batch) ---------
        filtered_frames = load_aufi_oneday(a_date, aps_dir, uhsas_dir, fims_dir)

        for name, df in list(filtered_frames.items()):
            if df is None or df.empty:
                continue
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df2 = df.copy()
                df2.index = df.index.tz_convert(None)
                filtered_frames[name] = df2

        # shift FIMS by -fims_lag seconds (indices already tz-naive now)
        if "FIMS" in filtered_frames and not filtered_frames["FIMS"].empty:
            filtered_frames["FIMS"] = filtered_frames["FIMS"].copy()
            filtered_frames["FIMS"].index = (
                filtered_frames["FIMS"].index - pd.Timedelta(seconds=fims_lag)
            )

        inlet_flag = read_inlet_flag(inlet_dir, start=a_date, end=None, prefix="ARCSIX")
        if isinstance(inlet_flag.index, pd.DatetimeIndex) and inlet_flag.index.tz is not None:
            inlet_flag = inlet_flag.copy()
            inlet_flag.index = inlet_flag.index.tz_convert(None)

        micro = read_microphysical(micro_dir, start=a_date, end=None, prefix="ARCSIX")
        if isinstance(micro.index, pd.DatetimeIndex) and micro.index.tz is not None:
            micro = micro.copy()
            micro.index = micro.index.tz_convert(None)

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
                chunk_for_specs = a_chunk

                # treat anything non-zero as "bad" and drop those times
                sflag = pd.to_numeric(inlet_chunk["InletFlag_LARGE"], errors="coerce").fillna(0)
                good_idx = sflag.index[sflag == 0]

                # duration of this specified window
                dur_s = (t_end - t_start).total_seconds()

                with log_file.open("a") as f:
                    f.write(
                        f"\t[INLET_FILTER] chunk {i:03d}: "
                        f"{len(inlet_chunk)} inlet_flag samples, "
                        f"{(sflag != 0).sum()} flagged, "
                        f"{len(good_idx)} unflagged, "
                        f"duration={dur_s:.1f} s\n"
                    )

                filtered_chunk = {}

                for name, df in a_chunk.items():
                    if df is None or df.empty:
                        continue

                    n_before = len(df)
                    df2 = df.loc[df.index.intersection(good_idx)]
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

                # always use the filtered chunk for mean spectra
                chunk_for_specs = filtered_chunk
                ##################################################################
                
                # 2) mean specs
                specs, line_kwargs, fill_kwargs, bin_counts = make_filtered_specs(
                    chunk_for_specs,
                    chunk_for_specs.get("APS",   pd.DataFrame()),
                    chunk_for_specs.get("UHSAS", pd.DataFrame()),
                    chunk_for_specs.get("FIMS",  pd.DataFrame()),
                    log_file,
                )
                
                # require all 3 instruments
                if ("APS" not in specs) or ("UHSAS" not in specs) or ("FIMS" not in specs):
                    with log_file.open("a") as f:
                        f.write(
                            f"\t[SKIP] chunk {i:03d} missing instrument(s): "
                            f"APS={'APS' in specs}, UHSAS={'UHSAS' in specs}, FIMS={'FIMS' in specs}\n"
                        )
                    continue

                # gate on NON-ZERO AVERAGE per instrument
                low_data_reason = None
                for _name in ("APS", "UHSAS", "FIMS"):
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
                if orig_UHSAS_edges is None:
                    orig_UHSAS_edges = specs["UHSAS"][1]
                if orig_FIMS_edges is None:
                    orig_FIMS_edges = specs["FIMS"][1]

                # 3) optimization
                specs_opt, line_kwargs_opt, fill_kwargs_opt, opt_res = run_joint_uhsas_aps_opt_from_specs(
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
                    lut_dir=lut_dir,
                )

                # 4) loss curve
                fig_h, ax_h = plot_history(opt_res["hist"])
                ax_h.set_title(f"opt hist {a_date} {i:03d}")
                fig_h.savefig(opt_dir / f"sizedist_{i:03d}_opt_hist.png", dpi=150)
                plt.close(fig_h)

                # 5) log
                with log_file.open("a") as f:
                    f.write(
                        f"\t\tUHSAS n_fit = {opt_res['n_fit']:.4f}, "
                        f"APS rho_fit = {opt_res['rho_fit']:.1f} kg/m^3, "
                        f"cost = {opt_res['best_cost']:.6g}\n\n"
                    )

                # 6) Tikhonov merged
                tik_specs, tik_lines, tik_fills, tik_diag = make_tikhonov_merged_spec(
                    e_fims_sel=specs_opt["FIMS_applied"][1],
                    y_fims_sel=specs_opt["FIMS_applied"][2],
                    e_uhsas_fit=specs_opt[f"UHSAS fit (n={opt_res['n_fit']:.3f})"][1],
                    y_uhsas_fit=specs_opt[f"UHSAS fit (n={opt_res['n_fit']:.3f})"][2],
                    e_aps_fit=specs_opt[f"APS fit (ρ={opt_res['rho_fit']*0.001:.3f} g/cm$^3$)"][1],
                    y_aps_fit=specs_opt[f"APS fit (ρ={opt_res['rho_fit']*0.001:.3f} g/cm$^3$)"][2],
                    lam=1e-4,
                    n_points=fine_bin,
                    alpha_uhsas=uhsas_combine_weight,
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
                uh_name   = f"UHSAS fit (n={opt_res['n_fit']:.3f})"
                uh_mids, uh_edges, uh_vals, _ = specs_opt[uh_name]
                aps_name  = f"APS fit (ρ={opt_res['rho_fit']*0.001:.3f} g/cm$^3$)"
                aps_mids, aps_edges, aps_vals, _ = specs_opt[aps_name]

                if common_edges is None:
                    common_edges = tik_edges

                fims_on_common   = remap_dndlog_by_edges_any(fims_edges,  common_edges, fims_vals)
                uhsas_on_common  = remap_dndlog_by_edges_any(uh_edges,    common_edges, uh_vals)
                aps_on_common    = remap_dndlog_by_edges_any(aps_edges,   common_edges, aps_vals)
                merged_on_common = remap_dndlog_by_edges_any(tik_edges,   common_edges, tik_vals)

                day_fims_algn.append(fims_on_common)
                day_uhsas_algn.append(uhsas_on_common)
                day_aps_algn.append(aps_on_common)
                day_merged.append(merged_on_common)
                day_times_start.append(t_start)
                day_times_end.append(t_end)
                day_incloud.append(inc_flag)
                day_n_fit.append(opt_res["n_fit"])
                day_rho_fit.append(opt_res["rho_fit"])
                day_best_cost.append(opt_res["best_cost"])

            except Exception as e:
                err_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
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
                orig_UHSAS_edges=np.asarray(orig_UHSAS_edges, float) if orig_UHSAS_edges is not None else None,
                orig_FIMS_edges=np.asarray(orig_FIMS_edges, float) if orig_FIMS_edges is not None else None,
            )


def run_arcsix_merge_for_periods_fims_aps_only(
    time_periods,
    data_dir,
    output_dir,
    *,
    fims_lag=10,
    incloud_pad_s=10,
    min_samples_per_inst=50,
    fims_xmin=None,
    fims_xmax=None,
    fine_bin=200,
    smoothness_lam = 1E-4,
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
        Interpreted as UTC, handled as tz-naive (no timezone conversions).

    data_dir : str or Path
        Base ARCSIX_P3B data directory containing LARGE-APS and FIMS, etc.
    output_dir : str or Path
        Output directory for batch NetCDF + plots + logs.
    """

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # instrument subdirs (NO UHSAS here)
    aps_dir   = data_dir / "LARGE-APS"
    fims_dir  = data_dir / "FIMS"
    inlet_dir = data_dir / "LARGE-InletFlag"
    micro_dir = data_dir / "LARGE-MICROPHYSICAL"

    log_file  = output_dir / "output_log.txt"
    error_log = output_dir / "error_log.txt"

    # ---------- parse periods, force tz-naive (UTC interpreted but no tz info) ----------
    _periods = []
    for idx, (s_raw, e_raw) in enumerate(time_periods):
        s_ts = pd.to_datetime(s_raw)
        e_ts = pd.to_datetime(e_raw)

        # drop timezone if present -> tz-naive
        if isinstance(s_ts, pd.Timestamp) and s_ts.tz is not None:
            s_ts = s_ts.tz_convert(None)
        if isinstance(e_ts, pd.Timestamp) and e_ts.tz is not None:
            e_ts = e_ts.tz_convert(None)

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
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
        f.write("ARCSIX Aerosol Size Distribution Merge Log\n")
        f.write(f"Generated: {ts} by Bo Chen\n\n")
        f.write("MODE: specified time periods (FIMS + APS only, no UHSAS, no alignment)\n")
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
            f.write(f"FIMS_LAG: {fims_lag}  # shift FIMS <FIMS_LAG> seconds earlier\n")
            f.write(f"INCLOUD_PAD_S: {incloud_pad_s}  # seconds around inlet_flag to mark a chunk in-cloud\n")
            f.write(f"MIN_SAMPLES_PER_INST: {min_samples_per_inst}  # skip chunks with fewer than this\n")
            f.write(f"FIMS_XMIN: {fims_xmin}\n")
            f.write(f"FIMS_XMAX: {fims_xmax}\n")
            f.write(f"FINE_BIN: {fine_bin}  # number of fine bins for Tikhonov merged spec\n")
            f.write("----------------------------------------------------------\n\n")

        day_dir = output_dir / a_date
        day_dir.mkdir(parents=True, exist_ok=True)
        totals_dir = day_dir / "time_series"
        plots_dir  = day_dir / "merge_plots"
        totals_dir.mkdir(exist_ok=True)
        plots_dir.mkdir(exist_ok=True)

        common_edges     = None
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

        # --------- load frames & drop timezone from indices (tz-naive like batch) ---------
        # APS + FIMS only
        filtered_frames = load_af_oneday(a_date, aps_dir, fims_dir)

        for name, df in list(filtered_frames.items()):
            if df is None or df.empty:
                continue
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df2 = df.copy()
                df2.index = df.index.tz_convert(None)
                filtered_frames[name] = df2

        # shift FIMS by -fims_lag seconds (indices already tz-naive now)
        if "FIMS" in filtered_frames and not filtered_frames["FIMS"].empty:
            filtered_frames["FIMS"] = filtered_frames["FIMS"].copy()
            filtered_frames["FIMS"].index = (
                filtered_frames["FIMS"].index - pd.Timedelta(seconds=fims_lag)
            )

        inlet_flag = read_inlet_flag(inlet_dir, start=a_date, end=None, prefix="ARCSIX")
        if isinstance(inlet_flag.index, pd.DatetimeIndex) and inlet_flag.index.tz is not None:
            inlet_flag = inlet_flag.copy()
            inlet_flag.index = inlet_flag.index.tz_convert(None)

        micro = read_microphysical(micro_dir, start=a_date, end=None, prefix="ARCSIX")
        if isinstance(micro.index, pd.DatetimeIndex) and micro.index.tz is not None:
            micro = micro.copy()
            micro.index = micro.index.tz_convert(None)

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
                chunk_for_specs = a_chunk

                # treat anything non-zero as "bad" and drop those times
                sflag = pd.to_numeric(inlet_chunk["InletFlag_LARGE"], errors="coerce").fillna(0)
                good_idx = sflag.index[sflag == 0]

                # duration of this specified window
                dur_s = (t_end - t_start).total_seconds()

                with log_file.open("a") as f:
                    f.write(
                        f"\t[INLET_FILTER] chunk {i:03d}: "
                        f"{len(inlet_chunk)} inlet_flag samples, "
                        f"{(sflag != 0).sum()} flagged, "
                        f"{len(good_idx)} unflagged, "
                        f"duration={dur_s:.1f} s\n"
                    )

                filtered_chunk = {}

                for name, df in a_chunk.items():
                    if df is None or df.empty:
                        continue

                    n_before = len(df)
                    df2 = df.loc[df.index.intersection(good_idx)]
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

                # always use the filtered chunk for mean spectra
                chunk_for_specs = filtered_chunk
                ##################################################################

                # 2) mean specs (APS + FIMS only, UHSAS frame is absent)
                specs, line_kwargs, fill_kwargs, bin_counts = make_filtered_specs(
                    chunk_for_specs,
                    chunk_for_specs.get("APS",   pd.DataFrame()),
                    pd.DataFrame(),  # NO UHSAS
                    chunk_for_specs.get("FIMS",  pd.DataFrame()),
                    log_file,
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
                    alpha_fims=1.0,
                    alpha_aps=1.0,
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

                fims_edges = e_FIMS
                aps_edges  = e_APS

                if common_edges is None:
                    common_edges = tik_edges

                fims_on_common   = remap_dndlog_by_edges_any(fims_edges,  common_edges, y_FIMS)
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
                err_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
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