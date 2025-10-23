import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ict_utils import (
    find_flag_column, flag_segments, flag_fractions,
)

def _get_kwargs_for(label, user_dict, fallback: dict):
    """
    Resolve kwargs for a given label with optional '_default' in user_dict.
    Precedence: user_dict[label] > user_dict['_default'] > fallback
    """
    out = dict(fallback or {})
    if user_dict:
        if "_default" in user_dict and isinstance(user_dict["_default"], dict):
            out.update(user_dict["_default"])
        if label in user_dict and isinstance(user_dict[label], dict):
            out.update(user_dict[label])
    return out


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
    legend_order: list[str] | None = None
):
    """
    Plot mean ±1σ size distributions as regular lines.

    Styling is fully controlled by `line_kwargs` and `fill_kwargs`.
    Legend labels can be overridden via:
      - line_kwargs[label]["label"]
      - legend_labels[label]
    Order can be set with legend_order=[...].
    """
    if show_flag_strip:
        fig, (ax, axf) = plt.subplots(
            2, 1, figsize=(8, 6),
            gridspec_kw={"height_ratios":[4,0.15]}, sharex=False
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        axf = None

    ax.set_xscale("log")
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel(r"$D_p$ (nm)")
    ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}\log D_p$  (# cm$^{-3}$)")
    ax.set_title("Time-averaged size distributions — ±1σ")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)

    handles = {}

    for key, (mids, edges, vals, sigma) in specs.items():
        lk = _get_kwargs_for(key, line_kwargs, {"linewidth": 1.5})

        # Resolve display label
        display_label = (
            (lk.get("label"))
            or ((legend_labels or {}).get(key))
            or key
        )
        # Don't pass "label" twice if user included it
        lk_no_label = {k: v for k, v in lk.items() if k != "label"}

        h, = ax.plot(mids, vals, label=display_label, **lk_no_label)
        handles[key] = h

        # Fill handling
        fk = None if fill_kwargs is None else fill_kwargs.get(key, fill_kwargs.get("_default", None))
        if fk is not False:
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

    flag_col = find_flag_column(inlet_flag)
    segments = flag_segments(inlet_flag[flag_col])
    fractions = flag_fractions(segments)
    if len(segments) > 0:
        frac_txt = " | ".join(f"Flag {k}: {v*100:.0f}%" for k, v in sorted(fractions.items()))
        ax.text(0.01, 0.98, f"InletFlag time fraction — {frac_txt}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9)

    if show_flag_strip and len(segments) > 0:
        cmap = {0:"C0",1:"C1",2:"C2",3:"C3"}
        for t0, t1, v in segments:
            axf.axvspan(t0, t1, alpha=0.25, color=cmap.get(v, "0.7"), ec=None)
        axf.set_xlim(segments[0][0], segments[-1][1])
        axf.set_ylabel("Inlet\nFlag", rotation=0, labelpad=25, va="center")
        axf.set_yticks([])
        axf.grid(False)
        axf.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        axf.xaxis.set_minor_locator(mdates.MinuteLocator(interval=2))
        axf.set_xlabel("Time (UTC)")

    plt.tight_layout()
    return fig, (ax, axf), handles


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
    legend_order: list[str] | None = None
):
    """
    Plot mean ±1σ size distributions as step histograms with edge-aligned
    uncertainty shading. Legend labels/order fully controllable.
    """
    if show_flag_strip:
        fig, (ax, axf) = plt.subplots(
            2, 1, figsize=(8, 6),
            gridspec_kw={"height_ratios":[4,0.15]}, sharex=False
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        axf = None

    ax.set_xscale("log")
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel(r"$D_p$ (nm)")
    ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}\log D_p$  (# cm$^{-3}$)")
    ax.set_title("Time-averaged size distributions — step ±1σ")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)

    handles = {}

    for key, (mids, edges, vals, sigma) in specs.items():
        lk = _get_kwargs_for(key, line_kwargs, {"linewidth": 2.0})
        display_label = (
            (lk.get("label"))
            or ((legend_labels or {}).get(key))
            or key
        )
        color = lk.get("color", None)

        # don't pass "label" twice
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
        if fk is not False:
            ylo = vals - sigma
            yhi = vals + sigma
            ylo_e = np.r_[ylo, ylo[-1]]
            yhi_e = np.r_[yhi, yhi[-1]]

            if isinstance(fk, dict):
                ax.fill_between(edges, ylo_e, yhi_e, step='post', **fk)
            else:
                ax.fill_between(edges, ylo_e, yhi_e, step='post', alpha=0.18, color=h.get_color())

    if legend and handles:
        if legend_order:
            hs = [handles[k] for k in legend_order if k in handles]
            ls = [h.get_label() for h in hs]
            ax.legend(hs, ls, loc=legend_loc)
        else:
            ax.legend(loc=legend_loc)

    flag_col = find_flag_column(inlet_flag)
    segments = flag_segments(inlet_flag[flag_col])
    fractions = flag_fractions(segments)
    if len(segments) > 0:
        frac_txt = " | ".join(f"Flag {k}: {v*100:.0f}%" for k, v in sorted(fractions.items()))
        ax.text(0.01, 0.98, f"InletFlag time fraction — {frac_txt}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9)

    if show_flag_strip and len(segments) > 0:
        cmap = {0:"C0",1:"C1",2:"C2",3:"C3"}
        for t0, t1, v in segments:
            axf.axvspan(t0, t1, alpha=0.25, color=cmap.get(v, "0.7"), ec=None)
        axf.set_xlim(segments[0][0], segments[-1][1])
        axf.set_ylabel("Inlet\nFlag", rotation=0, labelpad=25, va="center")
        axf.set_yticks([])
        axf.grid(False)
        axf.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        axf.xaxis.set_minor_locator(mdates.MinuteLocator(interval=2))
        axf.set_xlabel("Time (UTC)")

    plt.tight_layout()
    return fig, (ax, axf), handles