"""LUT building and storage; response smoothing lives in optical_diameter."""
from __future__ import annotations
import os
import time
import warnings
import numpy as np
import zarr
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator
from .optical_geometry import (
    OpticalSetup,
    POPSGeom, UHSASGeom, PCASPGeom,
    pops_optical_setup, uhsas_optical_setup, pcasp_optical_setup,
    POPS_WAVELENGTH_NM, UHSAS_WAVELENGTH_NM, PCASP_WAVELENGTH_NM,
    OPTICAL_MODEL_VERSION,
)

DEFAULT_D_RANGE  = (60.0, 5000.0, 360)    # (min, max, num points)
DEFAULT_N_RANGE  = (1.30, 1.80, 0.001)    # (min, max, step)
DEFAULT_K_VALUES = (0.0, 0.001, 0.01, 0.1)
DEFAULT_CHUNKS   = (128, 64, 1)           # (D, n, k) for Zarr v3

def build_sigma_lut(
    zpath: str,
    kernel: str,
    wavelength_nm: float,
    geom,
    *,
    D_range = DEFAULT_D_RANGE,
    n_range = DEFAULT_N_RANGE,
    k_values = DEFAULT_K_VALUES,
    chunks = DEFAULT_CHUNKS,
    jobs_per_k: int = -1,
    parallel_backend: str = "threads",
    response_channel: str | None = None,
):
    """Build a new lookup table of unsmoothed scattering cross-sections.

    The stored axes are diameter [nm], real refractive index n, and imaginary
    index k. Each n/k pair supplies one complete diameter-response curve.
    Smoothing is done when using the table, not here. An interrupted build
    keeps build_complete=False so the reader will reject it.
    """
    # Import at call time so the storage and calculation modules stay independent.
    from .optical_diameter import setup_geometry_cache, setup_csca

    kern = kernel.lower()
    if kern not in ("pops", "uhsas", "custom"):
        raise ValueError("kernel must be 'pops', 'uhsas' or 'custom'.")
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be finite and positive")
    if kern != "custom" and response_channel is not None:
        raise ValueError("use build_setup_sigma_lut to select a response channel")

    D_min, D_max, D_pts   = D_range
    n_min, n_max, n_step  = n_range
    if not (np.isfinite(D_min) and np.isfinite(D_max) and float(D_min) > 0 and float(D_max) > float(D_min)):
        raise ValueError("D_range must have finite D_min > 0 and D_max > D_min")
    if int(D_pts) < 2:
        raise ValueError("D_range point count must be >= 2")
    if not (np.isfinite(n_min) and np.isfinite(n_max) and np.isfinite(n_step) and float(n_step) > 0):
        raise ValueError("n_range must be finite with positive step")
    if float(n_max) < float(n_min):
        raise ValueError("n_range maximum must be >= minimum")
    if np.any(~np.isfinite(np.asarray(k_values, dtype=float))):
        raise ValueError("k_values must be finite")
    if int(chunks[0]) <= 0 or int(chunks[1]) <= 0 or int(chunks[2]) <= 0:
        raise ValueError("chunks must contain positive integers")

    Dg = np.geomspace(float(D_min), float(D_max), int(D_pts)).astype(float)
    ng = np.arange(float(n_min), float(n_max) + 1e-12, float(n_step), dtype=float)
    kg = np.asarray(k_values, dtype=float)
    if ng.size < 2 or kg.ndim != 1 or kg.size < 2 or not np.all(np.diff(kg) > 0):
        raise ValueError("LUT interpolation requires at least two increasing n and k coordinates")

    # Validate geometry before touching disk, and never overwrite an existing LUT.
    if kern == "pops":
        setup = pops_optical_setup(geom, wavelength_nm=wavelength_nm)
        response_channels = [c.name for c in setup.channels]
    elif kern == "uhsas":
        setup = uhsas_optical_setup(geom, wavelength_nm=wavelength_nm)
        response_channels = [setup.channels[0].name]
    else:
        if not isinstance(geom, OpticalSetup) or geom.wavelength_nm != wavelength_nm:
            raise ValueError("custom geometry must be an OpticalSetup with matching wavelength")
        setup = geom
        selected = [c for c in setup.channels if c.name == response_channel]
        if len(selected) != 1:
            raise ValueError("select exactly one named response_channel for the LUT")
        response_channels = [response_channel]
    selected = tuple(c for c in setup.channels if c.name in response_channels)
    calculation_setup = OpticalSetup(setup.wavelength_nm, setup.beams, selected,
                                     setup.aerosol_direction, setup.angular_step_deg)
    cache = setup_geometry_cache(calculation_setup)
    if os.path.lexists(zpath):
        raise FileExistsError(f"LUT destination already exists: {zpath}; choose a new directory")
    root = zarr.open_group(zpath, mode="w-")
    root.attrs.update({"optical_model_version": OPTICAL_MODEL_VERSION,
                       "build_complete": False})
    coords = root.create_group("coords")
    coords.create_array("D_nm", data=Dg)
    coords.create_array("n",    data=ng)
    coords.create_array("k",    data=kg)

    SIG = root.create_array(
        "sigma_col",
        shape=(Dg.size, ng.size, kg.size),
        dtype="f4",
        chunks=chunks,
    )

    # All instruments use the same integrator and precomputed setup cache.
    # UHSAS/custom select one detector; POPS retains its optional path sum.
    kernel_name = kern.upper()

    def _curve_for_n(n_val, k_val):
        m = complex(float(n_val), float(k_val))
        result = setup_csca(Dg, m, calculation_setup, _cache=cache)
        return np.sum(list(result.values()), axis=0).astype(np.float32)

    # Workers calculate separate refractive-index curves. The parent process
    # writes each completed group, so workers do not write the same array chunks.
    block_n = chunks[1]
    total_k = kg.size
    for ik, k in enumerate(kg):
        t_k = time.perf_counter()
        print(f"[{kernel_name}] [k {ik+1}/{total_k}] k = {k:g}", flush=True)
        j0 = 0
        while j0 < ng.size:
            t_blk = time.perf_counter()
            j1 = min(j0 + block_n, ng.size)
            cols = Parallel(n_jobs=jobs_per_k, prefer=parallel_backend, verbose=0)(
                delayed(_curve_for_n)(float(nv), float(k)) for nv in ng[j0:j1]
            )
            SIG[:, j0:j1, ik] = np.stack(cols, axis=1)
            print(f"  wrote n[{j0}:{j1}) in {time.perf_counter()-t_blk:.2f}s", flush=True)
            j0 = j1
        print(f"done k={k:g} (elapsed {time.perf_counter()-t_k:.2f}s)", flush=True)

    # Save the exact setup with the results so later plots and calculations
    # can recover its directions, openings, exclusions and beam fractions.
    # Mark complete only after all refractive-index curves have been written.
    attrs = {
        "description": f"{kernel_name} collected scattering cross-section (polarized solid-angle integral)",
        "optical_model_version": OPTICAL_MODEL_VERSION,
        "build_complete": True,
        "solid_angle_measure": "sin(theta) dtheta dphi",
        "normalization": "miepython qsca; P11-P12 and P11+P12; no extra 0.5",
        "collection_arms": 1,
        "units_sigma_col": "um^2",
        "D_range_nm": [float(D_min), float(D_max)],
        "n_range": [float(n_min), float(n_max), float(n_step)],
        "k_values": kg.tolist(),
        "wavelength_nm": float(wavelength_nm),
        "polarization": "linear E perpendicular to laser and central collection axis; azimuth resolved",
        "kernel": kernel_name,
        "optical_setup": setup.to_dict(),
        "response_channels": response_channels,
        "irradiance_basis": "total incident irradiance; incoherent beam intensity fractions sum to one",
    }
    if kern == "pops":
        attrs.update({
            "ring_theta_min_deg": float(geom.ring_theta_min_deg),
            "ring_theta_max_deg": float(geom.ring_theta_max_deg),
            "ring_step_deg": float(geom.ring_step_deg),
            "mirror_diameter_mm": float(geom.mirror_diameter_mm),
            "distance_to_mirror_mm": float(geom.distance_to_mirror_mm),
            "pmt_aperture_d_mm": float(geom.pmt_aperture_d_mm),
            "pmt_center_deg": float(geom.pmt_center_deg),
            "mirror_halfangle_deg": float(geom.mirror_halfangle_deg),
            "pmt_aperture_distance_mm": geom.pmt_aperture_distance_mm,
            "direct_collection": len(response_channels) > 1,
            "geometry_reference": "Gao et al. 2016 Fig. 1; mirror-only default per Liu et al. 2021 Appendix A",
        })
    elif kern == "uhsas":
        L = float(geom.aperture_distance_mm)
        eff_big_d_mm   = 2.0 * L * np.tan(np.deg2rad(geom.big_outer_halfangle_deg))
        eff_small_d_mm = 2.0 * L * np.tan(np.deg2rad(geom.inner_stop_halfangle_deg))
        attrs.update({
            "big_theta_min_deg": float(geom.big_theta_min_deg),
            "big_theta_max_deg": float(geom.big_theta_max_deg),
            "small_theta_min_deg": float(geom.small_theta_min_deg),
            "small_theta_max_deg": float(geom.small_theta_max_deg),
            "ring_step_deg": float(geom.ring_step_deg),
            "aperture_distance_mm": L,
            "big_outer_halfangle_deg": float(geom.big_outer_halfangle_deg),
            "inner_stop_halfangle_deg": float(geom.inner_stop_halfangle_deg),
            "eff_big_disk_d_mm": float(eff_big_d_mm),
            "eff_small_disk_d_mm": float(eff_small_d_mm),
            "geometry_reference": "Howell et al. 2021 Fig. 1 and Appendix A",
            "irradiance_basis": "total incident irradiance; symmetric counterpropagating beams",
        })
    else:
        attrs["polarization"] = "linear polarization vectors specified in optical_setup"
    root.attrs.update(attrs)

    return dict(zpath=zpath, D_grid_nm=Dg, n_grid=ng, k_grid=kg)


def build_setup_sigma_lut(zpath, setup: OpticalSetup, *, channel, **kwargs):
    """Build a LUT for one explicitly selected detector in the shared setup."""
    return build_sigma_lut(zpath, "custom", setup.wavelength_nm, setup,
                           response_channel=channel, **kwargs)


def build_pcasp_sigma_lut(zpath, geom: PCASPGeom | None = None, *,
                         wavelength_nm=PCASP_WAVELENGTH_NM, **kwargs):
    """Build a new PCASP LUT through the shared geometry integrator.

    This does not assume a campaign calibration refractive index. The saved
    kernel is CUSTOM because it uses the same general integrator as a custom
    setup; the instrument and literature provenance are recorded separately.
    Existing destinations are rejected by the common builder.
    """
    geom = geom or PCASPGeom()
    setup = pcasp_optical_setup(geom, wavelength_nm=wavelength_nm)
    result = build_setup_sigma_lut(zpath, setup, channel="Collection", **kwargs)
    root = zarr.open_group(zpath, mode="a")
    root.attrs.update({
        "instrument": "PCASP",
        "description": "PCASP nominal collected scattering cross-section",
        "geometry_reference": "Rosenberg et al. 2012 Table 1; doi:10.5194/amt-5-1147-2012",
        "geometry_scope": "nominal full-azimuth acceptance; no measured aperture transmission or detector gain",
        "outgoing_beam_collection_deg": [geom.theta_min_deg, geom.theta_max_deg],
        "returning_beam_collection_deg": [180-geom.theta_max_deg, 180-geom.theta_min_deg],
        "reflected_beam_ratio": geom.reflected_beam_ratio,
        "outgoing_irradiance_basis_multiplier": 1+geom.reflected_beam_ratio,
    })
    return result


def build_pops_sigma_lut(
    zpath: str, geom: POPSGeom, *,
    D_range=DEFAULT_D_RANGE, n_range=DEFAULT_N_RANGE, k_values=DEFAULT_K_VALUES,
    chunks=DEFAULT_CHUNKS, jobs_per_k=-1, parallel_backend="threads",
    wavelength_nm: float = POPS_WAVELENGTH_NM
):
    return build_sigma_lut(zpath, "pops", wavelength_nm, geom,
                           D_range=D_range, n_range=n_range, k_values=k_values,
                           chunks=chunks, jobs_per_k=jobs_per_k, parallel_backend=parallel_backend)


def build_uhsas_sigma_lut(
    zpath: str, geom: UHSASGeom, *,
    D_range=DEFAULT_D_RANGE, n_range=DEFAULT_N_RANGE, k_values=DEFAULT_K_VALUES,
    chunks=DEFAULT_CHUNKS, jobs_per_k=-1, parallel_backend="threads",
    wavelength_nm: float = UHSAS_WAVELENGTH_NM
):
    return build_sigma_lut(zpath, "uhsas", wavelength_nm, geom,
                           D_range=D_range, n_range=n_range, k_values=k_values,
                           chunks=chunks, jobs_per_k=jobs_per_k, parallel_backend=parallel_backend)


def _check_lut_model(root, *, allow_legacy=False):
    version = root.attrs.get("optical_model_version")
    if root.attrs.get("build_complete") is False:
        raise ValueError("LUT build is incomplete; do not use it for diameter conversion.")
    if version == OPTICAL_MODEL_VERSION:
        if root.attrs.get("build_complete") is not True:
            raise ValueError("Corrected LUT has no completed-build marker.")
        return
    if version is None and allow_legacy:
        warnings.warn("Reading a legacy optical LUT for comparison only; it uses the "
                      "uncorrected angular integration.", UserWarning, stacklevel=3)
        return
    raise ValueError(
        f"LUT optical model {version!r} is not {OPTICAL_MODEL_VERSION!r}. "
        "Rebuild into a NEW directory with the corrected optical code. "
        "For deliberate legacy comparisons only, pass allow_legacy=True."
    )


def sigma_query_zarr(zpath: str, D_nm: float, n: float, k: float, *,
                     allow_legacy=False) -> float:
    """
    Read the LUT and interpolate between neighboring D, n and k coordinates.

    Out-of-range queries use the nearest boundary, not extrapolation. This
    function loads the whole table on each call; reuse SigmaLUT for many queries.
    """
    if not all(np.isfinite(v) for v in (D_nm, n, k)):
        raise ValueError("D_nm, n, and k must be finite")
    if D_nm <= 0:
        raise ValueError("D_nm must be > 0")
    z = zarr.open(zpath, mode="r")
    _check_lut_model(z, allow_legacy=allow_legacy)
    Dg = z["coords/D_nm"][:].astype(float)
    ng = z["coords/n"][:].astype(float)
    kg = z["coords/k"][:].astype(float)
    SIG = z["sigma_col"][:].astype(float)

    interp = RegularGridInterpolator((Dg, ng, kg), SIG, method="linear",
                                     bounds_error=False, fill_value=None)

    Dq = float(np.clip(D_nm, Dg[0], Dg[-1]))
    nq = float(np.clip(n,    ng[0], ng[-1]))
    kq = float(np.clip(k,    kg[0], kg[-1]))
    return float(interp([[Dq, nq, kq]])[0])


class SigmaLUT:
    """Keep a completed LUT in memory for repeated D, n and k interpolation.

    Historical tables require allow_legacy=True explicitly and emit a warning.
    That option is for comparisons, not for corrected production.
    Queries outside the stored range are clipped to its boundaries. Clipping
    is not a physically justified extension of the response beyond the table.
    """
    def __init__(self, zpath: str, *, allow_legacy=False):
        z = zarr.open(zpath, mode="r")
        _check_lut_model(z, allow_legacy=allow_legacy)
        self.zpath = zpath
        self.Dg  = z["coords/D_nm"][:].astype(float)
        self.ng  = z["coords/n"][:].astype(float)
        self.kg  = z["coords/k"][:].astype(float)
        self.SIG = z["sigma_col"][:].astype(float)
        self.kernel = str(z.attrs.get("kernel", ""))
        self.wavelength_nm = float(z.attrs.get("wavelength_nm", np.nan))
        self.polarization  = str(z.attrs.get("polarization", ""))
        if self.SIG.shape != (self.Dg.size, self.ng.size, self.kg.size):
            raise ValueError("sigma_col shape does not match D/n/k coordinate lengths")
        for label, grid in (("D_nm", self.Dg), ("n", self.ng), ("k", self.kg)):
            if np.any(~np.isfinite(grid)):
                raise ValueError(f"{label} coordinate contains non-finite values")
            if grid.size < 2 or not np.all(np.diff(grid) > 0):
                raise ValueError(f"{label} coordinate must be strictly increasing with at least 2 points")
        if np.any(~np.isfinite(self.SIG)):
            raise ValueError("sigma_col contains non-finite values")

        self._interp = RegularGridInterpolator(
            (self.Dg, self.ng, self.kg), self.SIG,
            method="linear", bounds_error=False, fill_value=None
        )

    def _tri_single(self, D_nm: float, n: float, k: float) -> float:
        Dq = float(np.clip(D_nm, self.Dg[0], self.Dg[-1]))
        nq = float(np.clip(n,    self.ng[0], self.ng[-1]))
        kq = float(np.clip(k,    self.kg[0], self.kg[-1]))
        return float(self._interp([[Dq, nq, kq]])[0])

    def sigma_curve(self, D_vec_nm, n: float, k: float) -> np.ndarray:
        D_vec_nm = np.asarray(D_vec_nm, float)
        Dq = np.clip(D_vec_nm, self.Dg[0], self.Dg[-1])
        nq = float(np.clip(n,   self.ng[0], self.ng[-1]))
        kq = float(np.clip(k,   self.kg[0], self.kg[-1]))
        pts = np.column_stack([Dq, np.full_like(Dq, nq), np.full_like(Dq, kq)])
        return self._interp(pts).astype(float)
