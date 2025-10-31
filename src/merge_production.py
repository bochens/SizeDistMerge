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
from netCDF4 import Dataset

from ict_utils import (read_aps, read_nmass, read_pops, read_uhsas, read_fims, read_inlet_flag,
    check_common_grid, filter_by_spectra_presence, mean_spectrum, get_spectra)
from sizedist_utils import (edges_from_mids_geometric, dvdlog_from_dndlog, 
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


def log_bin_table_horizontal(log_path, name, mids, n_vals):
    with open(log_path, "a") as f:
        f.write(f"\n[{name}] Bin summary (horizontal)\n")
        idx_str = [f"{i:d}" for i in range(len(mids))]
        mid_str = [f"{m:.3f}" for m in mids]
        n_str   = [f"{int(n)}" for n in n_vals]

        def row(label, arr):
            f.write(f"{label:>8} | " + " ".join(f"{s:>8}" for s in arr) + "\n")

        row("idx", idx_str)
        row("mid_nm", mid_str)
        row("n", n_str)


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

    # APS
    if "APS" in frames and not frames["APS"].empty:
        out_aps = mean_spectrum_with_edges(frames["APS"], "APS")
        if out_aps is not None:
            m_APS, e_APS, y_APS, s_APS, _, n_APS = out_aps
            log_bin_table_horizontal(log_path, "APS", m_APS, n_APS)
            specs["APS"] = (m_APS, e_APS, y_APS, s_APS)

    # UHSAS
    if "UHSAS" in frames and not frames["UHSAS"].empty:
        out_uh = mean_spectrum_with_edges(frames["UHSAS"], "UHSAS")
        if out_uh is not None:
            m_UHSAS, e_UHSAS, y_UHSAS, s_UHSAS, _, n_UHSAS = out_uh
            log_bin_table_horizontal(log_path, "UHSAS", m_UHSAS, n_UHSAS)
            specs["UHSAS"] = (m_UHSAS, e_UHSAS, y_UHSAS, s_UHSAS)

    # FIMS
    if "FIMS" in frames and not frames["FIMS"].empty:
        out_f = mean_spectrum_with_edges(frames["FIMS"], "FIMS")
        if out_f is not None:
            m_FIMS, e_FIMS, y_FIMS, s_FIMS, _, n_FIMS = out_f
            log_bin_table_horizontal(log_path, "FIMS", m_FIMS, n_FIMS)
            specs["FIMS"] = (m_FIMS, e_FIMS, y_FIMS, s_FIMS)

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

    return specs, line_kwargs, fill_kwargs


def _load_uhsas_lut(lut_dir: str | Path | None):
    if lut_dir is None:
        base = Path.cwd().parent
        lut_path = base / "lut" / "uhsas_sigma_col_1054nm.zarr"
    else:
        lut_path = Path(lut_dir) / "uhsas_sigma_col_1054nm.zarr"
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


def make_tikhonov_merged_spec(
    *,
    e_fims_sel, y_fims_sel,
    e_uhsas_fit, y_uhsas_fit,
    e_aps_fit,   y_aps_fit,
    lam: float = 1e-5, n_points = 200
):
    # mids from edges
    m_FIMS   = mids_from_edges(e_fims_sel)
    m_UHSASf = mids_from_edges(e_uhsas_fit)
    m_APSf   = mids_from_edges(e_aps_fit)

    series = [
        {"x": m_FIMS,   "y": y_fims_sel,   "alpha": 1.0},
        {"x": m_UHSASf, "y": y_uhsas_fit,  "alpha": 1.0},
        {"x": m_APSf,   "y": y_aps_fit,    "alpha": 1.0},
    ]

    Dg = make_grid_from_series(series, n_points=200)
    y_merged, wsum, diag = merge_sizedists_tikhonov(
        Dg, series, lam=lam, eps=1e-12, nonneg=True
    )

    Dg_edge = edges_from_mids_geometric(Dg)

    specs_label = "Tikhonov merged"
    specs_out = {
        specs_label: (Dg, Dg_edge, y_merged, np.full_like(Dg, np.nan))
    }
    line_kwargs_out = {
        specs_label: {"color": "gray", "linewidth": 2.2}
    }
    fill_kwargs_out = {
        specs_label: False
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
    orig_APS_edges: np.ndarray | None = None,
    orig_UHSAS_edges: np.ndarray | None = None,
    orig_FIMS_edges: np.ndarray | None = None,
):
    """
    Save per-day merged sizedist to NetCDF.
    All aligned/merged arrays must already be on the SAME fine edges.
    """
    nc_path = day_dir / f"{a_date}_sizedist_merged.nc"

    n_chunk = len(day_fine_vals)
    n_fine  = len(day_fine_edges) - 1 if day_fine_edges is not None else 0

    # strings for time
    t_start_strs = [str(t) if (t is not None and pd.notna(t)) else "" for t in day_times_start]
    t_end_strs   = [str(t) if (t is not None and pd.notna(t)) else "" for t in day_times_end]

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

        # store times as attrs (simpler than ragged strings)
        nc.time_start_list = t_start_strs
        nc.time_end_list   = t_end_strs

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

        # inlet/in-cloud flag per chunk
        v_inc = nc.createVariable("inlet_incloud_flag", "i4", ("chunk",))
        v_inc[:] = np.asarray(day_incloud_flag, int)
        v_inc.long_name = "1 = this chunk is within ±10 s of inlet in-cloud flag; 0 = otherwise"

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