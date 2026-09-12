"""Plot corrected POPS/UHSAS LUT curves using published comparison RIs.

Run with the Research Python environment. This reads the completed new LUTs
only; it does not modify the LUTs or run aerosol-distribution production.
"""

from pathlib import Path
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, ScalarFormatter
import numpy as np
from scipy.optimize import brentq

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from sizedistmerge.optical_diameter import (
    SigmaLUT,
    make_monotone_sigma_interpolator,
)

LUT_DIR = PROJECT / "src" / "sizedistmerge" / "data" / "lut"
OUTPUT_DIR = PROJECT / "outputs" / "corrected_optical_response"
RESPONSE_BINS = 100  # Same response-curve grouping as the R2 notebook.
# Approximate the framing of Gao et al. (2016), Fig. 4 (POPS), and
# ARM TR-304 (2024), Fig. 2, linked by the UHSAS handbook. Their signal y axes cannot
# be copied numerically: these limits retain our cross-section units.
# This changes only the view; calculations still use each full LUT.
DISPLAY_LIMITS = {
    "POPS": ((130, 2500), (2e-3, 0.8)),
    "UHSAS": ((300, 1000), (2e-3, 0.4)),
}
CURVE_LINEWIDTH = 2.5

CASES = (
    ("POPS", "pops_sigma_col_405nm.zarr", 405,
     ((complex(1.615, 0.001), "1.615 + 0.001i"),
      (complex(1.45, 0), "1.45"))),
    ("UHSAS", "uhsas_sigma_col_1054nm.zarr", 1054,
     ((complex(1.58, 0), "1.58"),
      (complex(1.50, 0), "1.50"),
      (complex(1.40, 0), "1.40"))),
)


def literature_comparison():
    """Use published RIs, without fitting gains or digitizing observations.

    ARM TR-304 Fig. 2 uses 1.58, 1.50, and 1.40. Howell (2021), Table 1
    and Fig. 3 give the four weakly/nonabsorbing calibration materials below.
    Absorption is passed as positive k, the convention accepted by SigmaLUT.
    Howell's aggregate soot curves are not homogeneous-sphere Mie curves;
    they are deliberately excluded, not approximated by an out-of-grid RI.
    """
    lut = SigmaLUT(str(LUT_DIR / "uhsas_sigma_col_1054nm.zarr"))
    sets = (
        ((1.58+0j, "1.58 (PSL)", "black"),
         (1.50+0j, "1.50", "#1976d2"), (1.40+0j, "1.40", "#d84315")),
        ((1.572+0j, "PSL: 1.572", "0.45"),
         (1.5314+0j, "NaCl: 1.5314", "blue"),
         (1.468+0j, r"Na$_2$SO$_4$: 1.468", "magenta"),
         (complex(1.426, 1.36e-6), r"H$_2$SO$_4$: 1.426 + 1.36×10$^{-6}$i", "red")),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 4.05), layout="constrained")
    for ax, curves, label, limits in zip(axes, sets, ("(a)", "(b)"),
                                       ((300, 1000), (50, 2000))):
        d = np.geomspace(*limits, 1000)
        for ri, text, color in curves:
            y = lut.sigma_curve(d, ri.real, ri.imag)
            ax.loglog(d, y, color=color, lw=2.2, label=text)
        ax.set(xlim=limits, xlabel="Diameter (nm)",
               ylabel=r"Collected cross-section (µm$^2$)")
        ax.set_box_aspect(1)
        ax.text(0, 1.025, label, transform=ax.transAxes, fontweight="bold")
        ax.grid(which="both", alpha=.22, linestyle=":")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylim(2e-3, .4)
    axes[1].set_ylim(1e-8, 2)
    axes[0].xaxis.set_major_locator(FixedLocator([300, 400, 500, 600, 800, 1000]))
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"uhsas_literature_indices.{ext}", dpi=200)
    plt.close(fig)
    # The ARM report gives approximate equal-response diameters. Test those
    # directly on raw LUT curves, independently of the monotone size converter.
    response = float(lut.sigma_curve(np.array([570.]), 1.58, 0)[0])
    checks = []
    for n, published in ((1.5, 650.), (1.4, 830.)):
        diameter = brentq(lambda d: float(lut.sigma_curve(np.array([d]), n, 0)[0])-response,
                          570., 1000.)
        checks.append(dict(n=n, arm_approximate_nm=published, new_lut_nm=diameter,
                           difference_percent=100*(diameter/published-1)))
    (OUTPUT_DIR / "uhsas_literature_check.json").write_text(json.dumps({
        "ARM_report": "https://www.arm.gov/publications/programdocs/doe-sc-arm-tr-304.pdf",
        "Howell": "https://doi.org/10.5194/amt-14-7381-2021",
        "basis": "raw LUT, signal equal to 570 nm sphere with n=1.58; no gain adjustment",
        "comparisons": checks,
        "limitation": "Published diameters are approximate. This is not validation against raw calibration data."
    }, indent=2)+"\n")
    print("ARM equal-response check:", checks)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 1.0,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })

    combined, combined_axes = plt.subplots(1, 2, figsize=(7.9, 4.0))
    combined.subplots_adjust(left=0.10, right=0.99, bottom=0.16, top=0.94,
                             wspace=0.24)
    for (instrument, filename, wavelength, ri_cases), combined_ax, panel_label in zip(
        CASES, combined_axes, ("(a)", "(b)")
    ):
        lut = SigmaLUT(str(LUT_DIR / filename))
        assert lut.wavelength_nm == wavelength
        diameters = lut.Dg
        display_diameters = np.geomspace(diameters[0], diameters[-1], 4000)
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.14, top=0.94)
        smoothed = []

        # The raw curves use all stored diameter points; interpolation between
        # RI grid points is handled by the same reader as diameter conversion.
        for (ri, label), raw_color, fit_color in zip(
            ri_cases, ("#1f77b4", "#ff7f0e", "#333333"),
            ("#1f77b4", "#ff7f0e", "#333333")
        ):
            assert lut.ng[0] <= ri.real <= lut.ng[-1]
            assert lut.kg[0] <= ri.imag <= lut.kg[-1]
            raw = lut.sigma_curve(diameters, ri.real, ri.imag)
            assert np.all(np.isfinite(raw) & (raw > 0))
            forward, _ = make_monotone_sigma_interpolator(
                diameters, raw, response_bins=RESPONSE_BINS
            )
            fit = forward(display_diameters)
            # Grouping/plateau collapse narrows the usable fitted range. Do
            # not extrapolate or connect across points outside that range.
            valid = np.isfinite(fit) & (fit > 0)
            assert valid.sum() > 2
            assert np.all(np.diff(fit[valid]) >= 0)
            for target in (ax, combined_ax):
                target.loglog(diameters, raw, color=raw_color, lw=CURVE_LINEWIDTH,
                              linestyle="-",
                              label=f"m = {label} (LUT)")
            smoothed.append((fit, fit_color, label))
            print(f"{instrument}, m={ri}: {len(raw)} LUT points; "
                  f"fitted range {display_diameters[valid][0]:.1f}–"
                  f"{display_diameters[valid][-1]:.1f} nm")

        for fit, color, label in smoothed:
            for target in (ax, combined_ax):
                target.loglog(display_diameters, fit, color=color, lw=CURVE_LINEWIDTH,
                              linestyle="--",
                              label=f"m = {label} (smoothed)")

        for target in (ax, combined_ax):
            target.text(0, 1.025, panel_label, transform=target.transAxes,
                        ha="left", va="bottom", fontsize=10, fontweight="bold")
            target.set_xlabel("Diameter, D (nm)")
            target.set_ylabel(r"Collected scattering cross-section, $\sigma_{\mathrm{col}}$ (µm²)")
            target.set_box_aspect(1)
            target.set_xlim(*DISPLAY_LIMITS[instrument][0])
            target.set_ylim(*DISPLAY_LIMITS[instrument][1])
            if instrument == "POPS":
                target.xaxis.set_major_locator(FixedLocator([200, 300, 400, 600, 1000, 2000]))
                target.xaxis.set_major_formatter(ScalarFormatter())
            else:
                target.xaxis.set_major_locator(FixedLocator([300, 400, 500, 600, 800, 1000]))
                target.xaxis.set_major_formatter(ScalarFormatter())
            target.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
            target.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=100))
            target.grid(which="major", color="0.76", linestyle=":", linewidth=0.7)
            target.grid(which="minor", color="0.87", linestyle=":", linewidth=0.5)
            target.legend(loc="upper left", framealpha=0.95,
                          fontsize=7.5 if target is combined_ax else 8)
        for extension in ("png", "pdf"):
            output = OUTPUT_DIR / f"{instrument.lower()}_response.{extension}"
            fig.savefig(output, dpi=190)
            print(output)
        plt.close(fig)
        del lut

    for extension in ("png", "pdf"):
        output = OUTPUT_DIR / f"pops_uhsas_response.{extension}"
        combined.savefig(output, dpi=190, bbox_inches="tight", pad_inches=0.10)
        print(output)
    plt.close(combined)
    literature_comparison()


if __name__ == "__main__":
    main()
