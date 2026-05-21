from __future__ import annotations

import numpy as np
import pandas as pd
from .ict_utils import (
    find_flag_column, flag_segments, flag_fractions,
)

def _get_kwargs_for(label, user_dict, fallback: dict):
    out = dict(fallback or {})
    if user_dict:
        if "_default" in user_dict and isinstance(user_dict["_default"], dict):
            out.update(user_dict["_default"])
        if label in user_dict and isinstance(user_dict[label], dict):
            out.update(user_dict[label])
    return out

# --- NEW: metric → ylabel mapper ---
def _moment_ylabel(moment: str) -> str:
    # Accept either symbols or indices
    m = str(moment).upper()
    if m in ("N", "0"): return r"$\mathrm{d}N/\mathrm{d}\log D_p$  (# cm$^{-3}$)"
    if m in ("S", "2"): return r"$\mathrm{d}S/\mathrm{d}\log D_p$  ($\mu\mathrm{m}^2\,\mathrm{cm}^{-3}$)"
    if m in ("V", "3"): return r"$\mathrm{d}V/\mathrm{d}\log D_p$  ($\mu\mathrm{m}^3\,\mathrm{cm}^{-3}$)"
    return ""


def _validate_spec_tuple(label: str, spec):
    try:
        mids, edges, vals, sigma = spec
    except Exception as exc:
        raise ValueError(f"{label!r} spec must be (mids, edges, vals, sigma)") from exc

    mids = np.asarray(mids, dtype=float)
    edges = np.asarray(edges, dtype=float)
    vals = np.asarray(vals, dtype=float)
    sigma = None if sigma is None else np.asarray(sigma, dtype=float)

    if mids.ndim != 1 or edges.ndim != 1 or vals.ndim != 1:
        raise ValueError(f"{label!r} mids, edges, and vals must be 1D arrays")
    if mids.size != vals.size or edges.size != vals.size + 1:
        raise ValueError(f"{label!r} must have len(mids)==len(vals) and len(edges)==len(vals)+1")
    if np.any(~np.isfinite(mids)) or np.any(~np.isfinite(edges)) or np.any(mids <= 0) or np.any(edges <= 0):
        raise ValueError(f"{label!r} mids and edges must be finite and > 0")
    if not np.all(np.diff(edges) > 0):
        raise ValueError(f"{label!r} edges must be strictly increasing")
    if np.any(np.isinf(vals)):
        raise ValueError(f"{label!r} vals must not contain infinite values")
    if sigma is not None:
        if sigma.ndim != 1 or sigma.size != vals.size or np.any(np.isinf(sigma)):
            raise ValueError(f"{label!r} sigma must be None or a 1D array matching vals without infinities")
    return mids, edges, vals, sigma


def plot_size_distributions(
    specs: dict,                   # {"any_label": (mids, edges, mean, sigma), ...}
    inlet_flag: pd.DataFrame,
    yscale: str = "log",
    xlim: tuple | None = (10, 1e4),
    ylim: tuple | None = None,     # None means autoscale
    line_kwargs: dict | None = None,   # {"_default": {...}, "APS": {..., "label": "APS nice"}}
    fill_kwargs: dict | None = None,   # {"_default": {...}, "APS": {...}} or {"APS": False}
    show_flag_strip: bool = True,
    *,
    legend: bool = True,
    legend_loc: str = "best",
    legend_labels: dict | None = None,  # {"APS": "APS (raw)", "APS_rho_900": "APS (ρ=900)"}
    legend_order: list[str] | None = None,
    moment: str | None = None,          # optional: choose ylabel via moment ("N","S","V")
    ylabel: str | None = None,          # overrides moment if provided
    # NEW: optionally reuse existing figure/axes (ax required if fig is given)
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    axf: plt.Axes | None = None,
    figsize = (8, 6),
):
    """
    Plot mean ±1σ size distributions as regular lines.
    If `fig`/`ax` are provided, the plot draws onto them (and `axf` for the flag strip).
    If `show_flag_strip=True` but `axf` is None while `ax` is provided, the flag strip
    is disabled for this call (so we don't create extra axes unexpectedly).
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    # --- figure/axes handling ---
    reuse_axes = (ax is not None)
    if reuse_axes:
        # If caller gave an ax but no axf, don't create a flag strip
        if show_flag_strip and axf is None:
            show_flag_strip = False
        if fig is None:
            fig = ax.figure
    else:
        if show_flag_strip:
            fig, (ax, axf) = plt.subplots(
                2, 1, figsize=figsize,
                gridspec_kw={"height_ratios":[4,0.15]}, sharex=False
            )
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            axf = None

    ax.set_xscale("log")
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # ylabel selection
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    elif moment is not None:
        ax.set_ylabel(_moment_ylabel(moment))
    else:
        ax.set_ylabel(_moment_ylabel("N"))

    ax.set_xlabel(r"$D_p$ (nm)")
    if not reuse_axes:
        ax.set_title("Time-averaged size distributions — ±1σ")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)

    handles = {}

    for key, spec in specs.items():
        mids, _edges, vals, sigma = _validate_spec_tuple(key, spec)
        lk = _get_kwargs_for(key, line_kwargs, {"linewidth": 1.5})
        display_label = (lk.get("label")) or ((legend_labels or {}).get(key)) or key
        lk_no_label = {k: v for k, v in lk.items() if k != "label"}

        h, = ax.plot(mids, vals, label=display_label, **lk_no_label)
        handles[key] = h

        fk = None if fill_kwargs is None else fill_kwargs.get(key, fill_kwargs.get("_default", None))
        if fk is not False and sigma is not None:
            ylo = vals - sigma
            yhi = vals + sigma
            if isinstance(fk, dict):
                ax.fill_between(mids, ylo, yhi, **fk)
            else:
                ax.fill_between(mids, ylo, yhi, alpha=0.18, color=h.get_color())

    if legend and handles:
        if legend_order:
            hs = [handles[k] for k in legend_order if k in handles]
            ls = [h.get_label() for h in hs]
            ax.legend(hs, ls, loc=legend_loc)
        else:
            ax.legend(loc=legend_loc)

    # flag strip (only if we have an axf to draw on)
    if show_flag_strip and axf is not None:
        if inlet_flag is None:
            raise ValueError("inlet_flag is required when show_flag_strip=True")
        flag_col = find_flag_column(inlet_flag)
        segments = flag_segments(inlet_flag[flag_col])
        fractions = flag_fractions(segments)
        if len(segments) > 0:
            frac_txt = " | ".join(f"Flag {k}: {v*100:.0f}%" for k, v in sorted(fractions.items()))
            ax.text(0.01, 0.98, f"InletFlag time fraction — {frac_txt}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9)

            cmap = {0:"C0",1:"C1",2:"C2",3:"C3"}
            for t0, t1, v in segments:
                axf.axvspan(t0, t1, alpha=0.25, color=cmap.get(v, "0.7"), ec=None)
            axf.set_xlim(segments[0][0], segments[-1][1])
            axf.set_ylabel("Inlet\nFlag", rotation=0, labelpad=25, va="center")
            axf.set_yticks([])
            axf.grid(False)
            axf.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            axf.xaxis.set_minor_locator(mdates.MinuteLocator(interval=2))
            axf.set_xlabel("Time")

    plt.tight_layout()
    return fig, (ax, axf), handles


__all__ = [
    "plot_size_distributions",
    "plot_size_distributions_steps",
]


def plot_size_distributions_steps(
    specs: dict,                   # {"any_label": (mids, edges, mean, sigma), ...}
    inlet_flag: pd.DataFrame,
    yscale: str = "log",
    xlim: tuple | None = (10, 1e4),
    ylim: tuple | None = None,
    line_kwargs: dict | None = None,   # {"_default": {...}, "APS": {..., "label": "..."}}
    fill_kwargs: dict | None = None,   # {"_default": {...}, "APS": {...}} or {"APS": False}
    show_flag_strip: bool = True,
    *,
    legend: bool = True,
    legend_loc: str = "best",
    legend_labels: dict | None = None,
    legend_order: list[str] | None = None,
    moment: str | None = None,          # choose ylabel via moment ("N","S","V")
    ylabel: str | None = None,          # overrides moment if provided
    # NEW: optionally reuse existing figure/axes
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    axf: plt.Axes | None = None,
    figsize = (8, 6),
):
    """
    Plot mean ±1σ size distributions as step histograms with edge-aligned uncertainty.
    If `fig`/`ax` are provided, draw onto them (and `axf` for the flag strip).
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    # --- figure/axes handling ---
    reuse_axes = (ax is not None)
    if reuse_axes:
        if show_flag_strip and axf is None:
            show_flag_strip = False
        if fig is None:
            fig = ax.figure
    else:
        if show_flag_strip:
            fig, (ax, axf) = plt.subplots(
                2, 1, figsize=figsize,
                gridspec_kw={"height_ratios":[4,0.15]}, sharex=False
            )
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            axf = None

    ax.set_xscale("log")
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # ylabel selection
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    elif moment is not None:
        ax.set_ylabel(_moment_ylabel(moment))
    else:
        ax.set_ylabel(_moment_ylabel("N"))

    ax.set_xlabel(r"$D_p$ (nm)")
    if not reuse_axes:
        ax.set_title("Time-averaged size distributions — step ±1σ")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)

    handles = {}

    for key, spec in specs.items():
        mids, edges, vals, sigma = _validate_spec_tuple(key, spec)
        lk = _get_kwargs_for(key, line_kwargs, {"linewidth": 2.0})
        display_label = (lk.get("label")) or ((legend_labels or {}).get(key)) or key
        color = lk.get("color", None)
        lk_no_label = {k: v for k, v in lk.items() if k not in {"label", "color"}}

        h = ax.stairs(
            vals, edges,
            fill=False,
            baseline=None,
            color=color,
            label=display_label,
            **lk_no_label
        )
        handles[key] = h

        fk = None if fill_kwargs is None else fill_kwargs.get(key, fill_kwargs.get("_default", None))
        if fk is not False and sigma is not None:
            ylo = vals - sigma
            yhi = vals + sigma
            ylo_e = np.r_[ylo, ylo[-1]]
            yhi_e = np.r_[yhi, yhi[-1]]

            if isinstance(fk, dict):
                ax.fill_between(edges, ylo_e, yhi_e, step='post', **fk)
            else:
                fill_color = color if color is not None else h.get_edgecolor()
                ax.fill_between(edges, ylo_e, yhi_e, step='post', alpha=0.18, color=fill_color)

    if legend and handles:
        if legend_order:
            hs = [handles[k] for k in legend_order if k in handles]
            ls = [h.get_label() for h in hs]
            ax.legend(hs, ls, loc=legend_loc)
        else:
            ax.legend(loc=legend_loc)

    if show_flag_strip and axf is not None:
        if inlet_flag is None:
            raise ValueError("inlet_flag is required when show_flag_strip=True")
        flag_col = find_flag_column(inlet_flag)
        segments = flag_segments(inlet_flag[flag_col])
        fractions = flag_fractions(segments)
        if len(segments) > 0:
            frac_txt = " | ".join(f"Flag {k}: {v*100:.0f}%" for k, v in sorted(fractions.items()))
            ax.text(0.01, 0.98, f"InletFlag time fraction — {frac_txt}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9)

            cmap = {0:"C0",1:"C1",2:"C2",3:"C3"}
            for t0, t1, v in segments:
                axf.axvspan(t0, t1, alpha=0.25, color=cmap.get(v, "0.7"), ec=None)
            axf.set_xlim(segments[0][0], segments[-1][1])
            axf.set_ylabel("Inlet\nFlag", rotation=0, labelpad=25, va="center")
            axf.set_yticks([])
            axf.grid(False)
            axf.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            axf.xaxis.set_minor_locator(mdates.MinuteLocator(interval=2))
            axf.set_xlabel("Time")

    plt.tight_layout()
    return fig, (ax, axf), handles
