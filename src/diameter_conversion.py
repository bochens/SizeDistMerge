import numpy as np
from scipy.optimize import root_scalar


def mean_free_path(pres_Pa, temp_K):
    """Estimate air molecular mean free path [m] from pressure [Pa] and temperature [K]."""
    pres_Pa = np.asarray(pres_Pa, dtype=float)
    temp_K = np.asarray(temp_K, dtype=float)
    if np.any(~np.isfinite(pres_Pa)) or np.any(pres_Pa <= 0):
        raise ValueError("pres_Pa must be finite and > 0")
    if np.any(~np.isfinite(temp_K)) or np.any(temp_K <= 0):
        raise ValueError("temp_K must be finite and > 0")

    boltzmann_constant = 1.38e-23       # J/K
    molecular_diameter_m  = 3.7e-10        # m (approx. for air)
    mean_free_path_m = boltzmann_constant * temp_K / (np.sqrt(2.0) * np.pi * molecular_diameter_m**2 * pres_Pa)
    return mean_free_path_m

def cunningham(diam_nm, pres_hPa, temp_C):
    """
    Cunningham slip correction for particles in air (Davies 1945 form).

    Parameters
    ----------
    diam_nm : float or array-like
        Particle diameter in nanometers (nm).
    pres_hPa : float or array-like
        Pressure in hectopascals (hPa).
    temp_C : float or array-like
        Temperature in degrees Celsius (°C).

    Returns
    -------
    slip_correction : ndarray
        Cunningham slip correction factor (dimensionless).
        The mean free path is used internally but is not returned.
    """
    diam_m = np.asarray(diam_nm, dtype=float) * 1e-9
    pres_Pa = np.asarray(pres_hPa, dtype=float) * 100.0
    temp_K  = np.asarray(temp_C,  dtype=float) + 273.15
    if np.any(~np.isfinite(diam_m)) or np.any(diam_m <= 0):
        raise ValueError("diam_nm must be finite and > 0")
    if np.any(~np.isfinite(pres_Pa)) or np.any(pres_Pa <= 0):
        raise ValueError("pres_hPa must be finite and > 0")
    if np.any(~np.isfinite(temp_K)) or np.any(temp_K <= 0):
        raise ValueError("temp_C must convert to finite absolute temperature > 0 K")

    # Davies (1945) coefficients for air
    a1, a2, a3 = 1.257, 0.4, 0.55

    # Mean free path uses the hard-sphere estimate above, in metres.
    mean_free_path_m = mean_free_path(pres_Pa, temp_K)
    
    # Small particles do not experience the drag predicted by a continuous
    # fluid model. This factor corrects that drag; it is not a diameter ratio.
    slip_correction = 1.0 + 2.0 * (mean_free_path_m / diam_m) * (a1 + a2 * np.exp(-a3 * diam_m / mean_free_path_m))
    return slip_correction

def da_to_dv(
    da_nm,               # diameter(s) or bin edges [nm]; this function maps no concentrations.
    rho_p,               # particle density [kg/m^3]
    chi_t=1.0,           # transition-corrected dynamic shape factor [-]
    rho0=1000.0,         # reference density (water) [kg/m^3]
    pres_hPa=1013.25,    # pressure for slip [hPa]
    temp_C=20.0,         # temperature for slip [°C]
    xtol=1e-12, rtol=1e-10, maxiter=200
):
    """
    Convert aerodynamic diameter(s) Da [nm] -> volume-equivalent diameter(s) Dv [nm].
    Accepts scalar or array-like da_nm and returns matching shape (scalar in, scalar out).

    Solve rho0 * Da^2 * Cc(Da) = (rho_p/chi_t) * Dv^2 * Cc(Dv).
    Cc is the dimensionless slip correction. rho0 is the fixed reference
    density, not the fitted particle density rho_p. Concentrations must be
    resized separately, preserving the number in each mapped bin.
    """
    da_nm_arr = np.asarray(da_nm, dtype=float)
    if np.any(~np.isfinite(da_nm_arr)) or np.any(da_nm_arr <= 0):
        raise ValueError("da_nm must be finite and > 0")
    for name, value in (("rho_p", rho_p), ("chi_t", chi_t), ("rho0", rho0)):
        value_arr = np.asarray(value, dtype=float)
        if np.any(~np.isfinite(value_arr)) or np.any(value_arr <= 0):
            raise ValueError(f"{name} must be finite and > 0")

    rho_p = float(rho_p)
    chi_t = float(chi_t)
    rho0 = float(rho0)
    dv_out = np.empty_like(da_nm_arr, dtype=float)

    # Older implementations returned (slip factor, mean free path). Accept
    # that form as well, although the current function returns only the factor.
    def slip_factor(d_nm):
        out = cunningham(d_nm, pres_hPa, temp_C)
        return out[0] if isinstance(out, tuple) else out

    # Iterate elementwise because root_scalar is scalar-only
    diameter_entries = np.ndenumerate(da_nm_arr)
    for index, da in diameter_entries:
        da_m = da * 1e-9
        aerodynamic_slip = slip_factor(da)
        aerodynamic_drag_term = rho0 * da_m**2 * aerodynamic_slip           # SI-consistent

        def drag_difference(dv_nm):
            dv_m = dv_nm * 1e-9
            volume_slip = slip_factor(dv_nm)
            volume_drag_term = (rho_p / chi_t) * dv_m**2 * volume_slip
            return aerodynamic_drag_term - volume_drag_term

        # Search for the diameter that makes the two drag terms equal.
        # The trial interval is in nm; each drag calculation converts to metres.
        # If these bounds do not enclose a solution, the solver raises an error.
        lo = da / 1e3
        hi = da * 1e3

        root_result = root_scalar(drag_difference, bracket=(lo, hi), method="brentq",
                          xtol=xtol, rtol=rtol, maxiter=maxiter)
        if not root_result.converged:
            raise RuntimeError(f"dv solve did not converge at index {index}")
        dv_out[index] = root_result.root

    # Return a scalar for a scalar input; otherwise preserve the diameter array shape.
    return dv_out if np.ndim(da_nm) != 0 else float(dv_out)


__all__ = [
    "mean_free_path",
    "cunningham",
    "da_to_dv",
]
