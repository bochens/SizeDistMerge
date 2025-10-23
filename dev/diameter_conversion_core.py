import numpy as np
from scipy.optimize import root_scalar



def mean_free_path(pres_Pa, temp_K):
    kB = 1.38e-23       # J/K
    d  = 3.7e-10        # m (approx. for air)
    mfp = kB * temp_K / (np.sqrt(2.0) * np.pi * d**2 * pres_Pa)
    return mfp

def cunningham(diam_nm, pres_hPa, temp_C):
    """
    Cunningham slip correction for particles in air (Davies 1945 form).

    Parameters
    ----------
    diam_nm : float or array-like
        Particle diameter in micrometers (nm).
    pres_hPa : float or array-like
        Pressure in hectopascals (hPa).
    temp_C : float or array-like
        Temperature in degrees Celsius (°C).

    Returns
    -------
    ccorr : ndarray
        Cunningham slip correction factor (dimensionless).
    mfp : ndarray
        Gas mean free path in meters (m), hard-sphere estimate.
    """
    # convert inputs to SI
    diam_m = np.asarray(diam_nm, dtype=float) * 1e-9
    pres_Pa = np.asarray(pres_hPa, dtype=float) * 100.0
    temp_K  = np.asarray(temp_C,  dtype=float) + 273.15

    # Davies (1945) coefficients for air
    a1, a2, a3 = 1.257, 0.4, 0.55

    # mean free path (hard-sphere) using molecular diameter d
    mfp = mean_free_path(pres_Pa, temp_K)
    
    # Cunningham slip correction
    ccorr = 1.0 + 2.0 * (mfp / diam_m) * (a1 + a2 * np.exp(-a3 * diam_m / mfp))
    return ccorr

def da_to_dv(
    da_nm,               # diameter(s). When dndlogdp is passed, this must be BIN EDGES [nm].
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
    """
    da_nm_arr = np.asarray(da_nm, dtype=float)
    dv_out = np.empty_like(da_nm_arr, dtype=float)

    # Helper: get only the Cunningham factor, regardless of whether cunningham returns (ccorr, mfp) or ccorr
    def _Cc(d_nm):
        out = cunningham(d_nm, pres_hPa, temp_C)
        return out[0] if isinstance(out, tuple) else out

    # Iterate elementwise because root_scalar is scalar-only
    it = np.ndenumerate(da_nm_arr)
    for idx, da in it:
        da_m = da * 1e-9
        C_da = _Cc(da)                          # use ccorr only
        term1 = rho0 * da_m**2 * C_da           # SI-consistent

        def f(dv_nm):
            dv_m = dv_nm * 1e-9
            C_dv = _Cc(dv_nm)
            term2 = (rho_p / chi_t) * dv_m**2 * C_dv
            return term1 - term2

        # Wide but safe bracket (0.001x .. 1000x Da) in nm
        lo = da / 1e3
        hi = da * 1e3

        sol = root_scalar(f, bracket=(lo, hi), method="brentq",
                          xtol=xtol, rtol=rtol, maxiter=maxiter)
        if not sol.converged:
            raise RuntimeError(f"dv solve did not converge at index {idx}")
        dv_out[idx] = sol.root

    # Match input shape/type when no spectra passed
    return dv_out if np.ndim(da_nm) != 0 else float(dv_out)

