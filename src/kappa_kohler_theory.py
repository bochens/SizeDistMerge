import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy
import sys
import miepython
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar
import scipy.interpolate
import scipy.integrate
from scipy.optimize import brentq

_water_density = 997048 #g/m3
_molar_mass_water = 18.01528

def _to_rh_frac(RH):
    """Accept RH as 0-1 or 0-100; return 0-1."""
    RH = np.asarray(RH, float)
    return np.where(RH > 1.5, RH / 100.0, RH)

def kappa_petter_and_Kreidenweis_2010_EQ10(critical_diameter, critical_saturation, surface_tension = 0.072 #J/m2
                                                                                 , Mw = _molar_mass_water           #g/mol
                                                                                 , T = 298.15              #K
                                                                                 , density = _water_density        #g/m3
                                                                                 , R=8.3145):              #J/(mol K)
    A = (4 * surface_tension * Mw) / (R * T * density)
    kappa = (4 * A**3) / (27 * critical_diameter**3 * np.square(np.log(critical_saturation)))

    return kappa

def Sc_petter_and_Kreidenweis_2010_EQ10(critical_diameter, kappa, surface_tension = 0.072 #J/m2
                                              , Mw = _molar_mass_water           #g/mol
                                              , T = 298.15              #K
                                              , density = _water_density        #g/m3
                                              , R=8.3145):              #J/(mol K)
    
    A = (4 * surface_tension * Mw) / (R * T * density)
    Sc = np.exp(np.sqrt((4 * A**3)/(27 * critical_diameter**3 * kappa)))
    return Sc

def Dd_petter_and_Kreidenweis_2010_EQ10(kappa, critical_saturation, surface_tension = 0.072 #J/m2
                                              , Mw = _molar_mass_water           #g/mol
                                              , T = 298.15              #K
                                              , density = _water_density        #g/m3
                                              , R=8.3145):              #J/(mol K)
    
    A = (4 * surface_tension * Mw) / (R * T * density)
    critical_diameter = np.power((4 * A**3) / (27 * kappa * np.square(np.log(critical_saturation))), 1/3)
    return critical_diameter

def S_petter_and_Kreidenweis_2010_EQ6(wet_diameter, dry_diameter, kappa, surface_tension = 0.072 #J/m2
                                                                       , Mw = _molar_mass_water           #g/mol
                                                                       , T = 298.15              #K
                                                                       , density = _water_density        #g/m3
                                                                       , R=8.3145):              #J/(mol K)
    
    B = (4 * surface_tension * Mw) / (R * T * density * wet_diameter)
    S = (wet_diameter**3 - dry_diameter**3) / (wet_diameter**3 - (1-kappa) * dry_diameter**3) * np.exp(B)
    return S

def find_peak_S_D_binary_search(Dd_input, kappa, surface_tension=0.072, Mw=_molar_mass_water, T=298.15, density=_water_density, R=8.3145, tol=1e-12):
    """
    Find the peak of the S(D) function using a binary search approach for a single dry diameter or an array of dry diameters.
    
    Parameters:
    Dd_input (float or np.array): Single dry diameter or array of dry diameters in meters
    kappa (float): Hygroscopicity parameter
    surface_tension (float): Surface tension in J/m^2
    Mw (float): Molar mass of water in g/mol
    T (float): Temperature in Kelvin
    density (float): Density of water in g/m^3
    R (float): Universal gas constant in J/(mol*K)
    tol (float): Tolerance for the convergence of the binary search

    Returns:
    If Dd_input is a single value: Returns two single values (peak diameter and peak S(D)).
    If Dd_input is an array: Returns two arrays (array of peak diameters and array of peak S(D) values).
    """

    # Check if the input is a single value or an array
    if isinstance(Dd_input, (float, np.float64)):
        Dd_array = [Dd_input]  # Convert to a list for consistency
        single_value_input = True
    else:
        Dd_array = Dd_input
        single_value_input = False

    peak_diameters = []
    peak_S_D_values = []

    for Dd in Dd_array:
        left = Dd
        right = Dd * 100

        while right - left > tol:
            mid = (left + right) / 2
            mid_left = (left + mid) / 2
            mid_right = (mid + right) / 2

            if S_petter_and_Kreidenweis_2010_EQ6(mid_left, Dd, kappa, surface_tension, Mw, T, density, R) < S_petter_and_Kreidenweis_2010_EQ6(mid, Dd, kappa, surface_tension, Mw, T, density, R):
                left = mid_left
            elif S_petter_and_Kreidenweis_2010_EQ6(mid_right, Dd, kappa, surface_tension, Mw, T, density, R) > S_petter_and_Kreidenweis_2010_EQ6(mid, Dd, kappa, surface_tension, Mw, T, density, R):
                left = mid
            else:
                right = mid_right

        peak_D = (left + right) / 2
        peak_S_D = S_petter_and_Kreidenweis_2010_EQ6(peak_D, Dd, kappa, surface_tension, Mw, T, density, R)

        peak_diameters.append(peak_D)
        peak_S_D_values.append(peak_S_D)

    if single_value_input:
        return peak_diameters[0], peak_S_D_values[0]
    else:
        return np.array(peak_diameters), np.array(peak_S_D_values)

def calculate_critical_diameter_interpolated(aerosol_sizes, aerosol_size_dist, super_saturations, ccn_concentrations, tolerance=1E-6):
    critical_diameters = []

    # Remove NaN values from the aerosol size distribution
    non_nan_indices = np.where(~np.isnan(aerosol_size_dist))
    aerosol_sizes = aerosol_sizes[non_nan_indices]
    aerosol_size_dist = aerosol_size_dist[non_nan_indices]

    # Create an interpolation of the aerosol size distribution
    original_interpolation = scipy.interpolate.interp1d(
        aerosol_sizes, aerosol_size_dist, kind='linear', bounds_error=False, fill_value="extrapolate"
    )

    #print('Calculating critical dry diameter: Debug Information')

    for ss_i, an_ss in enumerate(super_saturations):
        a_ccnc = ccn_concentrations[ss_i]

        lower_bound = min(aerosol_sizes)
        upper_bound = max(aerosol_sizes)
        crit_diameter = None

        max_iterations = 1000  # To prevent infinite loops
        iteration_count = 0

        while upper_bound - lower_bound > tolerance and iteration_count < max_iterations:
            iteration_count += 1
            # Compute the midpoint in log scale
            mid_point = 10**((np.log10(lower_bound) + np.log10(upper_bound)) / 2)

            # Create a fine grid for integration
            fine_grid = np.linspace(mid_point, aerosol_sizes[-1], 2000)
            interpolated_dist = original_interpolation(fine_grid)

            # Handle invalid interpolation values
            interpolated_dist = np.nan_to_num(interpolated_dist, nan=0.0, posinf=0.0, neginf=0.0)

            if len(interpolated_dist) > 0 and len(fine_grid) > 0:
                # Perform numerical integration
                integral = scipy.integrate.simpson(y=interpolated_dist, x=np.log10(fine_grid))

                # Update bounds based on the integral value
                if integral < a_ccnc:
                    upper_bound = mid_point
                else:
                    lower_bound = mid_point
            else:
                # Interpolation failed, stop the loop
                #print(f"Interpolation failed at iteration {iteration_count}. Breaking out of loop.")
                break

        # Check for convergence
        if iteration_count < max_iterations and upper_bound - lower_bound <= tolerance:
            crit_diameter = 10**((np.log10(lower_bound) + np.log10(upper_bound)) / 2)
        else:
            #print(f"Reached max iterations or failed to converge for SS={an_ss:.4f}. Returning None.")
            crit_diameter = None

        critical_diameters.append(crit_diameter)

        # Debugging information
        if crit_diameter is not None:
            #print(f"Super Saturation: {an_ss:.4f}, CCN Concentration: {a_ccnc:.4f}, "
            #      f"Critical Diameter: {crit_diameter:.4f}, Iterations: {iteration_count}")
            pass
        else:
            #print(f"Super Saturation: {an_ss:.4f}, CCN Concentration: {a_ccnc:.4f}, "
            #      f"Critical Diameter: None, Iterations: {iteration_count}")
            pass

    critical_diameters = np.array(critical_diameters, dtype=object)  # Use dtype=object to accommodate None

    return critical_diameters

def calculate_kappa_fitting(Dc, Sc):
    '''
    Sc is the critical super saturation
    Dc is the critical diameter
    '''
    def func_to_fit(Dd, kappa):
        _, peak_S_D = find_peak_S_D_binary_search(Dd, kappa)
        return peak_S_D

    kappa_list = []
    for i in range(len(Sc)):
        popt, pcov = curve_fit(lambda Dd, kappa: func_to_fit(Dd, kappa), Dc[i], Sc[i], p0=0.1, bounds=(0, 2))
        kappa_list.append(popt[0])
    return kappa_list

def calculate_kappa(Dc, Sc, x0=0.001, x1=2.0, max_expand=5):
    kappa_list = []
    for i in range(len(Sc)):
        def f(k): 
            _, peak_S_D = find_peak_S_D_binary_search(Dc[i], k)
            return peak_S_D - Sc[i]

        a, b = x0, x1
        fa, fb = f(a), f(b)
        # expand b until f(a) and f(b) have opposite sign
        for _ in range(max_expand):
            if fa*fb < 0:
                break
            b *= 2
            fb = f(b)
        else:
            raise RuntimeError(f"Couldn't bracket root around {x0}–{x1}")

        sol = root_scalar(f, bracket=[a, b], method='brentq',
                          xtol=1e-8, rtol=1e-8, maxiter=1000)
        if not sol.converged:
            raise RuntimeError(f"Root solve failed at index {i}")
        kappa_list.append(sol.root)

    return kappa_list

def calculate_critical_diameter(kappa_list, Sc): #
    '''
    Finding the smallest particle diameter with a certain kappa that will activate under an Sc.
    Given a list of kappa values and the corresponding critical supersaturations (Sc),
    this function returns the calculated critical diameters (Dc).
    '''
    Dc_list = []
    for i in range(len(Sc)):
        def func_to_solve(Dd):
            _, peak_S_D = find_peak_S_D_binary_search(Dd, kappa_list[i])
            return peak_S_D - Sc[i]
        
        # Use a numerical solver to find the root (Dc) where func_to_solve equals zero
        Dc_solution = root_scalar(func_to_solve, bracket=[1e-9, 1e-5], method='brentq')  # Adjust bracket range as needed
        Dc_list.append(Dc_solution.root)
        
    return Dc_list

def plot_Sc_Dd_base(fig = None, axis = None, figsize=(5,5)):
    
    if (fig is None) or (axis is None):
        fig, axis = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    ddry = np.logspace(1, 3, 20)  # 0.01 um to 1 um

    Dw_k_1, Sc_k_1         = find_peak_S_D_binary_search(ddry*1E-9, 1)
    Dw_k_01, Sc_k_01       = find_peak_S_D_binary_search(ddry*1E-9, 0.1)
    Dw_k_001, Sc_k_001     = find_peak_S_D_binary_search(ddry*1E-9, 0.01)
    Dw_k_0001, Sc_k_0001   = find_peak_S_D_binary_search(ddry*1E-9, 0.001)
    Dw_k_00001, Sc_k_00001 = find_peak_S_D_binary_search(ddry*1E-9, 0.0001)
    Dw_k_0, Sc_k_0         = find_peak_S_D_binary_search(ddry*1E-9, 0)

    axis.plot(ddry*1E-3, (Sc_k_1-1)*100, c = 'k', ls = 'solid', alpha = 0.75)
    axis.plot(ddry*1E-3, (Sc_k_01-1)*100, c = 'k', ls = 'dashed', alpha = 0.75)
    axis.plot(ddry*1E-3, (Sc_k_001-1)*100, c = 'k', ls = 'dashdot', alpha = 0.75)
    axis.plot(ddry*1E-3, (Sc_k_0001-1)*100, c = 'k', ls = 'dotted', alpha = 0.75)
    axis.plot(ddry*1E-3, (Sc_k_0-1)*100, c = 'k', ls = 'solid', lw = 3, alpha = 0.75)

    axis.set_xlabel('Dry Diameter (µm)')
    axis.set_xscale('log')

    axis.set_ylabel('Critical Supersaturation (%)')
    axis.set_yscale('log')
    axis.text(0.016, 1.9, r'$\kappa=1$' , ha='center')
    axis.text(0.034, 1.9, r'$0.1$' , ha='right')
    axis.text(0.064, 1.9, r'$0.01$' , ha='right')
    axis.text(0.093, 1.9, r'$0.001$' , ha='center')

    axis.text(0.3, 0.7, r'$\kappa=0$', ha='center', rotation=-56, fontweight='bold')

    axis.set_ylim(0.08,1.8)
    axis.set_xlim(0.01,1)

    return fig, axis

def calculate_wet_diameter(
    RH,
    dry_diameter,
    kappa,
    *,
    # NEW: define what "dry_diameter" means (reference RH)
    dry_rh=0.0,              # 0-1 or 0-100. If 0, dry_diameter is true dry (RH→0).
    surface_tension=0.072,   # J/m^2
    Mw=18.01528,             # g/mol
    T=298.15,                # K
    density=_water_density,          # g/m^3
    R=8.3145,                # J/(mol K)
    max_factor=200.0,
    eps_factor=1e-10,
    xtol=1e-12,
    rtol=1e-10,
    maxiter=200,
):
    """
    Return wet diameter at RH, given a *reference* diameter `dry_diameter` defined at `dry_rh`.

    If dry_rh == 0:
        interpret `dry_diameter` as true dry Dd (RH→0), and solve Eq.6 directly.
    Else:
        interpret `dry_diameter` as diameter at RH = dry_rh (a reference RH),
        first invert to true dry D0, then grow to RH.

    Supports broadcasting among RH and dry_diameter.
    """
    RHf = _to_rh_frac(RH)
    RHref = float(_to_rh_frac(dry_rh))
    Dref = np.asarray(dry_diameter, float)
    kappa = float(kappa)

    if kappa < 0:
        raise ValueError("kappa must be >= 0")
    if not np.isfinite(RHref):
        raise ValueError("dry_rh must be finite")
    if RHref < 0 or RHref >= 1:
        raise ValueError("dry_rh must be in [0,1) (or 0-100%)")

    RHb, Drefb = np.broadcast_arrays(RHf, Dref)

    if np.any(~np.isfinite(RHb)) or np.any(~np.isfinite(Drefb)):
        raise ValueError("RH and dry_diameter must be finite")
    if np.any(Drefb <= 0):
        raise ValueError("dry_diameter must be > 0")
    if np.any(RHb <= 0) or np.any(RHb >= 1):
        raise ValueError("RH must be in (0,1) (or 0-100%)")

    def S_eq6(Dw, Dd_scalar):
        # Petters & Kreidenweis Eq.6 as you already have
        return S_petter_and_Kreidenweis_2010_EQ6(
            Dw, Dd_scalar, kappa,
            surface_tension=surface_tension, Mw=Mw, T=T, density=density, R=R
        )

    def _solve_wet_from_true_dry(RH_target, Dd0):
        # solve f(Dw)=S(Dw,Dd0)-RH_target = 0 for Dw>=Dd0
        def f(Dw):
            return float(S_eq6(Dw, Dd0) - RH_target)

        a = Dd0 * (1.0 + eps_factor)
        b = Dd0 * max_factor

        fa = f(a)
        fb = f(b)

        expand = 0
        while fa * fb > 0 and expand < 20:
            b *= 2.0
            fb = f(b)
            expand += 1

        if fa * fb > 0:
            raise RuntimeError(
                f"Could not bracket wet-diameter root. "
                f"Dd0={Dd0:.3e} m, RH={RH_target:.4f}, kappa={kappa:.4g}, "
                f"f(a)={fa:.3e}, f(b)={fb:.3e}, b={b:.3e}"
            )
        return float(brentq(f, a, b, xtol=xtol, rtol=rtol, maxiter=maxiter))

    def _solve_true_dry_from_ref(RH_ref, Dref_val):
        # find Dd0 such that wet(Dd0,RH_ref)=Dref_val
        # i.e., g(Dd0)=Dw(RH_ref;Dd0)-Dref_val = 0
        def g(Dd0):
            Dw = _solve_wet_from_true_dry(RH_ref, Dd0)
            return Dw - Dref_val

        # bracket Dd0: must be <= Dref_val (for RH_ref>0), but allow equality
        # pick a very small lower bound relative to Dref_val
        a = Dref_val / max_factor
        b = Dref_val  # should be above/boundary

        ga = g(a)
        gb = g(b)

        expand = 0
        while ga * gb > 0 and expand < 30:
            a /= 2.0
            ga = g(a)
            expand += 1

        if ga * gb > 0:
            raise RuntimeError(
                f"Could not bracket true-dry solve. "
                f"Dref={Dref_val:.3e} m at RH_ref={RH_ref:.4f}, kappa={kappa:.4g}, "
                f"g(a)={ga:.3e}, g(b)={gb:.3e}, a={a:.3e}, b={b:.3e}"
            )

        return float(brentq(g, a, b, xtol=xtol, rtol=rtol, maxiter=maxiter))

    out = np.empty_like(Drefb, dtype=float)

    it = np.nditer(Drefb, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        Dref_i = float(Drefb[idx])
        RH_i = float(RHb[idx])

        if RHref == 0.0:
            # Dref is true dry
            out[idx] = _solve_wet_from_true_dry(RH_i, Dref_i)
        else:
            # Dref is diameter at RHref -> invert to true dry then grow to RH_i
            Dd0 = _solve_true_dry_from_ref(RHref, Dref_i)
            out[idx] = _solve_wet_from_true_dry(RH_i, Dd0)

        it.iternext()

    return out

def calculate_dry_diameter(
    RH,
    wet_diameter,
    kappa,
    *,
    # NEW: define what "dry" means (reference RH for the returned diameter)
    dry_rh=0.0,              # 0-1 or 0-100. If 0, return true dry D0 (RH→0).
    surface_tension=0.072,   # J/m^2
    Mw=18.01528,             # g/mol
    T=298.15,                # K
    density=_water_density,          # g/m^3
    R=8.3145,                # J/(mol K)
    max_factor=200.0,
    eps_factor=1e-10,
    xtol=1e-12,
    rtol=1e-10,
    maxiter=200,
):
    """
    Invert Eq.6: given wet diameter at RH, return diameter at reference RH = dry_rh.

    - If dry_rh == 0: returns true dry diameter D0.
    - If dry_rh > 0: returns D_ref such that wet(D0,RH_ref)=D_ref, where D0 is the
      true dry inferred from (RH, wet_diameter).

    Supports broadcasting among RH and wet_diameter.
    """
    RHf = _to_rh_frac(RH)
    RHref = float(_to_rh_frac(dry_rh))
    Dw_obs = np.asarray(wet_diameter, float)
    kappa = float(kappa)

    if kappa < 0:
        raise ValueError("kappa must be >= 0")
    if not np.isfinite(RHref):
        raise ValueError("dry_rh must be finite")
    if RHref < 0 or RHref >= 1:
        raise ValueError("dry_rh must be in [0,1) (or 0-100%)")

    RHb, Dwb = np.broadcast_arrays(RHf, Dw_obs)

    if np.any(~np.isfinite(RHb)) or np.any(~np.isfinite(Dwb)):
        raise ValueError("RH and wet_diameter must be finite")
    if np.any(Dwb <= 0):
        raise ValueError("wet_diameter must be > 0")
    if np.any(RHb <= 0) or np.any(RHb >= 1):
        raise ValueError("RH must be in (0,1) (or 0-100%)")

    def S_eq6(Dw, Dd_scalar):
        return S_petter_and_Kreidenweis_2010_EQ6(
            Dw, Dd_scalar, kappa,
            surface_tension=surface_tension, Mw=Mw, T=T, density=density, R=R
        )

    def _solve_true_dry_from_wet(RH_obs, Dw_val):
        # solve h(Dd0)=S(Dw_val,Dd0)-RH_obs=0 for Dd0 <= Dw_val
        def h(Dd0):
            return float(S_eq6(Dw_val, Dd0) - RH_obs)

        # bracket Dd0: (tiny) ... (Dw_val*(1-eps))
        b = Dw_val * (1.0 - 1e-12)
        a = b / max_factor

        ha = h(a)
        hb = h(b)

        expand = 0
        while ha * hb > 0 and expand < 30:
            a /= 2.0
            ha = h(a)
            expand += 1

        if ha * hb > 0:
            raise RuntimeError(
                f"Could not bracket dry-diameter root. "
                f"Dw={Dw_val:.3e} m at RH={RH_obs:.4f}, kappa={kappa:.4g}, "
                f"h(a)={ha:.3e}, h(b)={hb:.3e}, a={a:.3e}, b={b:.3e}"
            )

        return float(brentq(h, a, b, xtol=xtol, rtol=rtol, maxiter=maxiter))

    def _solve_wet_from_true_dry(RH_target, Dd0):
        def f(Dw):
            return float(S_eq6(Dw, Dd0) - RH_target)

        a = Dd0 * (1.0 + eps_factor)
        b = Dd0 * max_factor

        fa = f(a)
        fb = f(b)

        expand = 0
        while fa * fb > 0 and expand < 20:
            b *= 2.0
            fb = f(b)
            expand += 1

        if fa * fb > 0:
            raise RuntimeError(
                f"Could not bracket wet-diameter root. "
                f"Dd0={Dd0:.3e} m, RH={RH_target:.4f}, kappa={kappa:.4g}, "
                f"f(a)={fa:.3e}, f(b)={fb:.3e}, b={b:.3e}"
            )
        return float(brentq(f, a, b, xtol=xtol, rtol=rtol, maxiter=maxiter))

    out = np.empty_like(Dwb, dtype=float)

    it = np.nditer(Dwb, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        Dw_i = float(Dwb[idx])
        RH_i = float(RHb[idx])

        # Step 1: infer true dry diameter
        Dd0 = _solve_true_dry_from_wet(RH_i, Dw_i)

        if RHref == 0.0:
            out[idx] = Dd0
        else:
            # Step 2: compute diameter at RHref (reference "dry")
            out[idx] = _solve_wet_from_true_dry(RHref, Dd0)

        it.iternext()

    return out

def calculate_humidification_factor(dry_sizes, dndlogdps, rh, kappa, wavelength, dry_ri_n, dry_ri_k, water_ri_n = 1.33):
    '''
        dry_sizes : an array in nanometers
        dndlogdps : an array
        rh        : a scalar
        kappa     : a scalar
        Wavelength: in nanometers
    '''

    wet_sizes      = calculate_wet_diameter(rh, dry_sizes, kappa)
    dry_v_raitos   = dry_sizes**3 / wet_sizes**3
    water_v_ratios = 1 - dry_v_raitos
    wet_ri_ns      = dry_ri_n * dry_v_raitos + water_ri_n * water_v_ratios
    #wet_ri_ns      = dry_ri_n
    wet_ris    = wet_ri_ns - dry_ri_k * 1j
    dry_ri     = dry_ri_n  - dry_ri_k * 1j

    wet_extinction_coeff, wet_scattering_coeff, wet_bckscatter_coeff = calculate_coefficients(wet_sizes, dndlogdps, wet_ris, wavelength)
    dry_extinction_coeff, dry_scattering_coeff, dry_bckscatter_coeff = calculate_coefficients(dry_sizes, dndlogdps, dry_ri , wavelength)

    ext_humidificaiton_factor = wet_extinction_coeff/dry_extinction_coeff
    sca_humidificaiton_factor = wet_scattering_coeff/dry_scattering_coeff
    bck_humidificaiton_factor = wet_bckscatter_coeff/dry_bckscatter_coeff

    if rh == 1.0:
        print(ext_humidificaiton_factor)

    return ext_humidificaiton_factor, sca_humidificaiton_factor, bck_humidificaiton_factor

def calculate_humidification_factor_ammonium_sulfate(dry_sizes, dndlogdps, rh, kappa, wavelength):
    '''
        Cotterell er al. 2017
        
        dry_sizes : an array
        dndlogdps : an array
        rh        : a scalar
        kappa     : a scalar
        Wavelength: in nanometers
    '''

    as_rh     = np.array([0.4  , 0.5 , 0.6  , 0.7  , 0.8  , 0.9  , 1.0  ])
    as_realri = np.array([1.453, 1.44, 1.428, 1.417, 1.403, 1.379, 1.335])
    cs = CubicSpline(as_rh, as_realri, bc_type='not-a-knot')

    wet_sizes = calculate_wet_diameter(rh, dry_sizes, kappa)
    dry_ri_n = cs(0.3)
    dry_ri_k = 0

    wet_ri_ns = cs(rh)
    
    wet_ris    = wet_ri_ns - dry_ri_k * 1j
    dry_ri     = dry_ri_n  - dry_ri_k * 1j

    wet_extinction_coeff, wet_scattering_coeff, wet_bckscatter_coeff = calculate_coefficients(wet_sizes, dndlogdps, wet_ris, wavelength)
    dry_extinction_coeff, dry_scattering_coeff, dry_bckscatter_coeff = calculate_coefficients(dry_sizes, dndlogdps, dry_ri , wavelength)


    ext_humidificaiton_factor = wet_extinction_coeff/dry_extinction_coeff
    sca_humidificaiton_factor = wet_scattering_coeff/dry_scattering_coeff
    bck_humidificaiton_factor = wet_bckscatter_coeff/dry_bckscatter_coeff


    return ext_humidificaiton_factor, sca_humidificaiton_factor, bck_humidificaiton_factor

def calculate_coefficients(sizes, dndlogdps, ri, wavelength):
    '''
        Calculate extinction coefficient, scattering coefficient, or backscatter coefficient
    '''

    size_parameters = np.pi * sizes/wavelength
    qexts, qscas, qbcks, gs = miepython.mie(ri, size_parameters)


    exts_times_dndlogdp = np.pi * (sizes**2)/4 * qexts * dndlogdps
    scas_times_dndlogdp = np.pi * (sizes**2)/4 * qscas * dndlogdps
    bcks_times_dndlogdp = np.pi * (sizes**2)/4 * qbcks * dndlogdps

    non_nan_index = np.where(~np.isnan(dndlogdps))[0]

    extinction_coeff = scipy.integrate.simpson(y = exts_times_dndlogdp[non_nan_index], x = np.log10(sizes[non_nan_index]))
    scattering_coeff = scipy.integrate.simpson(y = scas_times_dndlogdp[non_nan_index], x = np.log10(sizes[non_nan_index]))
    bckscatter_coeff = scipy.integrate.simpson(y = bcks_times_dndlogdp[non_nan_index], x = np.log10(sizes[non_nan_index]))

    return extinction_coeff, scattering_coeff, bckscatter_coeff

def kappa_from_growth_factor(
    GF,
    RH,
    *,
    Dd=None,                 # optional dry diameter (meters) for Kelvin term
    surface_tension=0.072,   # J/m^2
    Mw=18.01528,             # g/mol (keep consistent with your code)
    T=298.15,                # K
    density=_water_density,          # g/m^3 (water density in your units)
    R=8.3145,                # J/(mol K)
):
    """
    Solve Petters & Kreidenweis Eq.6 (subsaturated) for kappa given growth factor GF and RH.

    Definitions:
      GF = Dw / Dd
      RH is ambient relative humidity w.r.t. liquid water, expressed as S (0-1) or % (0-100)

    Assumptions:
      - Solution is at equilibrium (subsaturated).
      - If you provide Dd (meters), Kelvin term exp(A/Dw) is included.
      - If Dd is None, Kelvin term is neglected (A=0), which is usually fine for accumulation mode.

    Returns:
      kappa (same broadcasted shape as GF and RH)

    Notes:
      Rearranged from Eq.6 with Dw = GF * Dd and S = RH:
        RH = [(GF^3 - 1) / (GF^3 - (1-kappa))] * exp(A/(Dd*GF))
      Let X = RH / exp(A/(Dd*GF))
      Then kappa = (GF^3 - 1)/X - GF^3 + 1
    """
    GF = np.asarray(GF, float)
    RHf = _to_rh_frac(RH)

    if np.any(~np.isfinite(GF)) or np.any(~np.isfinite(RHf)):
        raise ValueError("GF and RH must be finite")
    if np.any(GF <= 1.0):
        raise ValueError("GF must be > 1 for subsaturated growth-factor inversion")
    if np.any(RHf <= 0.0) or np.any(RHf >= 1.0):
        raise ValueError("RH must be in (0,1) (or 0-100%)")

    # Kelvin term factor: exp(A / Dw) where Dw = Dd * GF
    if Dd is None:
        kelvin = 1.0
    else:
        Dd = np.asarray(Dd, float)
        if np.any(~np.isfinite(Dd)) or np.any(Dd <= 0.0):
            raise ValueError("Dd must be finite and > 0 (meters)")
        # A (meters) consistent with your Eq.6 implementation:
        A = (4.0 * surface_tension * Mw) / (R * T * density)
        # Broadcast to match
        RHf, GF, Dd = np.broadcast_arrays(RHf, GF, Dd)
        kelvin = np.exp(A / (Dd * GF))

    # If Dd is None, we still want broadcasting between RH and GF
    RHf, GF = np.broadcast_arrays(RHf, GF)

    X = RHf / kelvin
    GF3 = GF**3

    # kappa from algebra
    kappa = (GF3 - 1.0) / X - GF3 + 1.0

    # sanity: kappa should be >= 0 in most cases; don't silently clip
    if np.any(kappa < -1e-6):
        bad = np.nanmin(kappa)
        raise RuntimeError(f"Computed kappa has negative values (min={bad:.3g}). Check RH/GF/Dd and units.")
    return kappa


def growth_factor_from_kappa(
    kappa,
    RH,
    *,
    Dd=None,                 # optional dry diameter (meters) for Kelvin term
    surface_tension=0.072,   # J/m^2
    Mw=18.01528,             # g/mol
    T=298.15,                # K
    density=_water_density,          # g/m^3
    R=8.3145,                # J/(mol K)
):
    """
    Forward (no root-solve): compute GF from kappa and RH neglecting Kelvin if Dd is None.
    With Kelvin term, need a root solve; so for Dd!=None we raise.

    Without Kelvin:
      RH = (GF^3 - 1) / (GF^3 - (1-kappa))
      => GF^3 = (RH*(1-kappa) - 1) / (RH - 1)

    Returns:
      GF (broadcasted)
    """
    RHf = _to_rh_frac(RH)
    kappa = np.asarray(kappa, float)

    if Dd is not None:
        raise NotImplementedError("With Kelvin term (Dd provided), GF requires a root solve; use your calculate_wet_diameter.")

    if np.any(~np.isfinite(RHf)) or np.any(~np.isfinite(kappa)):
        raise ValueError("RH and kappa must be finite")
    if np.any(RHf <= 0.0) or np.any(RHf >= 1.0):
        raise ValueError("RH must be in (0,1) (or 0-100%)")
    if np.any(kappa < 0.0):
        raise ValueError("kappa must be >= 0")

    RHf, kappa = np.broadcast_arrays(RHf, kappa)

    num = RHf * (1.0 - kappa) - 1.0
    den = RHf - 1.0
    GF3 = num / den

    if np.any(GF3 <= 1.0):
        mn = np.nanmin(GF3)
        raise RuntimeError(f"Computed GF^3 <= 1 (min={mn:.3g}). Check RH/kappa.")
    return GF3 ** (1.0 / 3.0)