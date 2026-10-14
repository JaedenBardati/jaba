import warnings

import numba
import numpy as np
import scipy as sp

from ..utils import constants as c
from ..utils import units as u
from ..utils.resources import load_resource
from ..utils.grid import CartesianGrid, _cartesian_grid_query, numba_threads


### --- Recombination Lines --- ###

transition_wavelength_cgs = {
    'HI 2-3': 656.3e-7,           #H alpha
    'HI 2-4': 486.1e-7,           #H beta
    'HeII 3-4': 468.6e-7,
    'HeII 3-5': 320.3e-7,
    'HeII 4-5': 1012.4e-7,
    'HeII 4-6': 656.0e-7,
}

transition_frequency_cgs = {
    k: c.c.cgs.value/v for k, v in transition_wavelength_cgs.items()
}

Pequignot_etal_1991_params = {
    'HI 2-3': (2.274, -0.659, 1.939, 0.574),           #H alpha
    'HI 2-4': (0.503, -0.515, 1.496, 0.698),           #H beta
    'HeII 3-4': (1.549, -0.693, 2.884, 0.609),
    'HeII 3-5': (0.479, -0.524, 2.688, 0.711),
    'HeII 4-5': (1.038, -0.730, 4.044, 0.653),
    'HeII 4-6': (0.361, -0.562, 3.845, 0.760),
}

def effective_recombination_coefficients_cgs(T, z=1, transition='HI 2-3', source='HopkinsPersonalCommunication'):
    # Returns the effective recombination coefficients for recombination lines.
    # Taken from Pequignot et al. (1991)
    #   T          = temperature of free electrons (K)
    #   z          = ionic charge (1) -- Default: Neutral (z=1)
    #   transition = the transition that produces the line -- Default: H alpha (transition='HI 2-3')
    # Returns in cm^3/2 (cgs) units.
    if source == 'Pequignot_etal_1991':
        a, b, c, d = Pequignot_etal_1991_params[transition]
        t = 1e-4*T/z**2
        return 1e-13*z*(a*t**b)/(1+c*t**d)
    if source == 'HopkinsPersonalCommunication':
        T4 = T/1e4
        return 2e-14 * (3.5 + np.log(1+15.8/T4)) / (0.25*T4**0.5 + 0.04*T4 + 0.0034*T4**1.5)
    else:
        raise NotImplementedError('Only Pequignot et al 1991 is supported for now.')

def recombination_emissivity_cgs(T, n_e, n_i=None, z=1, transition='HI 2-3', source='HopkinsPersonalCommunication'):
    # Returns emissivity of entered recombination lines.
    # Taken from equation ...
    #   T          = temperature of free electrons (K)
    #   n_e        = free electron number density (cm^-3)
    #   n_i        = ion number density (cm^-3) - Default: Neutral plasma (n_i=n_e)
    #   z          = ionic charge (1) - Default: Hydrogen (z=1)
    #   transition = the transition that produces the line -- Default: H alpha (transition='HI 2-3')
    # Returns in erg/s/cm^3/sr (cgs) units. Multiply by the emitting surface for luminosity.
    if n_i is None:
        n_i = n_e
    alpha_eff = effective_recombination_coefficients_cgs(T, transition=transition, source=source)
    E_trans = (c.h.cgs.value * c.c.cgs.value)/transition_wavelength_cgs[transition]
    return alpha_eff * n_e * n_i * E_trans


### --- Line broadening --- ###

def fwhm_from_sigma(sigma):
    return 2.3548200450309493*sigma

def sigma_from_fwhm(fwhm):
    return 0.42466090014400953*fwhm

def voight_profile_cgs(x, x0, spec_var='frequency',
                       fwhm_G=None, sigma_v=None, temp=None, part_mass=None, 
                       fwhm_L=None, gamma_ul=None):
    # Generalized Voight profile for convolution of Gaussian and Lorentzian profiles (e.g. equation 6.37 of Draine ISM text).
    # ENTER ALL (mandatory): 
    #   x          = frequencies, wavelengths or energies to get profile at (Hz, cm, or erg)
    #   x0         = frequency, wavelength or energy to center line around (Hz, cm, or erg)
    #   spec_var   = spectral variable ('frequency', 'wavelength', or 'energy')  - returns phi_nu, phi_lambda or phi_E
    # ENTER ANY (optional): Gaussian parameter
    #   fwhm_G     = full width half max of Gaussian profile (Hz, cm, or erg - same units as entered above)
    #   sigma_v    = standard deviation a Maxwellian velocity profile (km/s)
    #   temp       = temperature of the gas in a Maxwellian distribution (K)
    #   IF ENTERED temp MUST ALSO ENTER:
    #     part_mass  = mass of the particles in a Maxwellian distribution (g)
    # ENTER ONE (optional): Lorentzian parameter
    #   fwhm_L     = full width half max of Lorentzian profile (Hz, cm, or erg - same units as entered above)
    #   gamma_ul   = Sum of Einstein coefficients (1) - see chapter 6 of Draine ISM 
    # MUST ENTER AT LEAST ONE PARAMETER.
    # QUICK GUIDE:
    #   FOR INTRINSIC BROADENING (intrinsic/quantum broadening): Define fwhm_L or gamma_ul
    #   FOR TURBULENT BROADENING (Maxwell/velocity broadening):  Enter sigma_v (e.g. rms velocity)
    #   FOR THERMAL BROADENING   (Maxwell broadening):           Enter temp
    assert spec_var in ['frequency', 'wavelength', 'energy'], "Must enter a valid spectral variable ('frequency', 'wavelength', or 'energy')."
    assert (temp is None) or ((temp is not None) and (part_mass is not None)), 'Must enter a particle mass if you entered a temperature.'
    assert (gamma_ul is None) or (fwhm_L is None), 'Must only enter one of fwhm_L or gamma_ul.'

    _x = x - x0
    if spec_var == 'frequency':
        nu = x
        nu0 = x0
    if spec_var == 'wavelength':
        nu = c.c.cgs.value/x
        nu0 = c.c.cgs.value/x0
    if spec_var == 'energy':
        nu = x/c.h.cgs.value
        nu0 = x0/c.h.cgs.value

    g = nu / c.c.cgs # non-relativistic for now
    
    sigma_G = 0 if fwhm_G is None else sigma_from_fwhm(fwhm_G)
    if sigma_v is not None:
        sigma_G = np.sqrt(sigma_G**2 + (sigma_v * g)**2)
    if temp is not None:
        sigma_G = np.sqrt(sigma_G**2 + np.abs(c.k_B.cgs.value*temp/part_mass)*(g)**2)

    if fwhm_L is None:
        fwhm_L = 0
    if gamma_ul is not None:
        fwhm_L = gamma_ul/(2*np.pi)
    
    return sp.special.voigt_profile(_x, sigma_G, fwhm_L)



### --- Dust --- ###

def kappa_dust_MRN_cgs(nu, dust_to_gas=0.01, a_min=0.005e-4, a_max=0.25e-4, rho_grain=3.0):
    # Returns approximate MRN dust absorption and scattering opacities (cm^2/g).
    #   nu          = frequency (Hz)
    #   dust_to_gas = dust-to-gas mass ratio (1) - Default: 0.01
    #   a_min/max   = minimum/maximum grain radius (cm) - Defaults: 0.005/0.25 micron
    #   rho_grain   = grain material density (g/cm^3) - Default: 3
    #   Uses dn/da proportional to a^-3.5, Q_abs=min(x, 1), Q_sca=min(x^4, 1). 
    #   Returns values per gram of gas, use dust_to_gas=1 for values per gram of dust. 
    #   This is a simple Rayleigh approximation, without material-dependent Mie features.

    # split the size integral at x=1 and normalize by the mrn grain mass
    transition = c.c.cgs.value/(2*np.pi*nu)
    split = np.clip(transition, a_min, a_max)
    geometric = 2*(split**-0.5 - a_max**-0.5)
    normalization = 3*dust_to_gas/(8*rho_grain*(np.sqrt(a_max) - np.sqrt(a_min)))
    
    # calculate the absorption and scattering efficiencies at the transition grain size
    Q_abs = np.minimum(2*np.pi*split*nu/c.c.cgs.value, 1)
    Q_sca = np.minimum((2*np.pi*split*nu/c.c.cgs.value)**4, 1)

    # integrate Q_abs and Q_sca over the mrn grain-size distribution
    kappa_abs = normalization*(Q_abs*(1 - (a_min/split)**0.5)/(0.5*np.sqrt(split)) + geometric)
    kappa_sca = normalization*(Q_sca*(1 - (a_min/split)**3.5)/(3.5*np.sqrt(split)) + geometric)
    return kappa_abs, kappa_sca

def alpha_dust_absorption_cgs(rho, nu, dust_to_gas=0.01, a_min=0.005e-4, a_max=0.25e-4, rho_grain=3.0):
    # Returns the dust absorption coefficient (cm^-1), alpha_abs=rho*kappa_abs.
    #   rho = gas mass density (g/cm^3); remaining parameters as above.
    return rho*kappa_dust_MRN_cgs(nu, dust_to_gas, a_min, a_max, rho_grain)[0]

def alpha_dust_scattering_cgs(rho, nu, dust_to_gas=0.01, a_min=0.005e-4, a_max=0.25e-4, rho_grain=3.0):
    # Returns the total dust scattering coefficient (cm^-1), alpha_sca=rho*kappa_sca.
    #   rho = gas mass density (g/cm^3); remaining parameters as above.
    return rho*kappa_dust_MRN_cgs(nu, dust_to_gas, a_min, a_max, rho_grain)[1]

def dust_emissivity_MRN_cgs(T, rho, nu, dust_to_gas=0.01, a_min=0.005e-4, a_max=0.25e-4, rho_grain=3.0):
    # Returns thermal MRN dust emissivity j_nu=alpha_abs*B_nu in erg/s/cm^3/sr/Hz.
    #   T           = dust temperature (K), assumed common to all grain sizes
    #   rho         = gas mass density (g/cm^3)
    #   nu          = frequency (Hz)
    alpha_abs = alpha_dust_absorption_cgs(rho, nu, dust_to_gas, a_min, a_max, rho_grain)
    return alpha_abs*planck_function_nu(nu*u.Hz, T*u.K).value


### --- Thomson/Compton scattering opacities --- ###

def alpha_thomson_scattering(n_e):
    # Returns the extinction coefficient (alpha_nu=n*sigma). Independent of frequency.
    #   n_e  = free electron number density (cm^-3)
    return c.sigma_T.cgs.value * n_e

def Klein_Nishina_correction_factor(photon_energy):
    # Accurate to ~0.01%, sets to 1 for lambda >~ 25 nm.
    eps = np.abs((photon_energy.astype(np.float64)/(c.m_e*c.c**2)).to(u.dimensionless_unscaled).value)
    regular_thomson_range = eps < 1e-4
    return regular_thomson_range + ~regular_thomson_range*(3.0/4.0 * ((1+eps)/eps**3 * (2*eps*(1+eps)/(1+2*eps) - np.log(1+2*eps)) + np.log(1+2*eps)/(2*eps) - (1+3*eps)/(1+2*eps)**2))


### --- Free-free/bremsstrahlung opacities --- ###

def alpha_thermal_bremsstrahlung_cgs(T, n_e, nu, Z=1, n_i=None, g_ff=1):
    # Returns free-free extinction coefficient (alpha_nu=n*sigma) as a function of frequency and without any Rayleigh-Jeans assumptions.
    # Taken from equation 5.18a of Rybicki & Lightman.
    #   T    = temperature of free electrons (K)
    #   n_e  = free electron number density (cm^-3)
    #   nu   = frequency (Hz)
    #   Z    = ionic charge (1) - Default: Hydrogen (Z=1)
    #   n_i  = ion number density (cm^-3) - Default: Neutral plasma (n_i=n_e)
    #   g_ff = Gaunt factor (1) - Default: Classical and non-relativistic (g_ff=1)
    if n_i is None:
        n_i = n_e
    coeff = 4 * c.e.gauss.value**6/(3*c.m_e.cgs.value * c.h.cgs.value * c.c.cgs.value) * np.sqrt(2*np.pi/(3*c.k_B.cgs.value*c.m_e.cgs.value))
    return coeff * T**-0.5 * Z**2 * n_e * n_i * nu**-3 * (1 - np.exp(-c.h.cgs.value*nu/(c.k_B.cgs.value*T))) * g_ff


### --- Other opacity functions --- ###

def effective_extinction_optical_depth(tau_abs, tau_sca, d=3):
    # Returns the effective optical depth given a scattering and absorption depth.
    #   tau_abs  = absorption optical depth (1)
    #   tau_sca  = scattering optical depth (1)
    #   d        = dimensionality of the system (1, 2, or 3) - Default: 3D
    return np.sqrt(d*tau_abs*(tau_abs + tau_sca))



### --- SEDs --- ###
@u.quantity_input
def accretion_disk_spectrum_lam(lam: u.micron, L_AGN=1*u.erg*u.s**-1, alpha_ox=None, _nowarning=False):
    # Returns a simple power-law AGN accretion disk-only spectrum as used 
    # in Bardati et al. (2026) based on a fitting from Shen et al. (2020).
    # Accurately ranges from ~1e-4 microns to ~10 microns. Note that you need 
    # to add a torus component for full AGN SEDs. Spectrum is L_lamda. L_AGN 
    # is integrated luminosity and L_2500A is the specific luminosity at 2500A.
    if np.any(lam < 0.999*u.AA):
        if not _nowarning:
            warnings.warn('This accretion disk spectrum is likely only accurate > ~1e-4 microns, but you have entered value(s) outside this range.')
    if alpha_ox is None:
        if np.isclose(L_AGN.to('erg/s').value, 1):
            alpha_ox = -1.45
            if not _nowarning:
                warnings.warn('No alpha_ox or luminosity entered. Defaulting to alpha_ox=-1.5, as used in appendix A of Bardati+26.')
        else:
            L_2500A = 0.1*L_AGN*(2500*u.AA)/c.c # L_2500A is approximately 10% of total luminosity for typical AGN SEDs (approximate, but should really iterate to get this)
            beta = 0.721 # pm 0.011 from Steffen+06 as shown in Shen+20
            C = 4.531 # pm 0.688 from Steffen+06 as shown in Shen+20
            A = 0.384*(1-beta)
            Cp = 0.384*C
            alpha_ox = -A*np.log10(L_2500A.to('erg s**-1 Hz**-1').value) + Cp
            if not _nowarning:
                warnings.warn('No alpha_ox but total luminosity was entered. Setting alpha_ox={:.3f}, by using 2500A-alpha_ox relation from Steffen+06.'.format(alpha_ox))
                warnings.warn('Note that using this relation is very rough at the moment and really requires 1) a prescription for varying gamma (now gamma=-1.9 is hardcoded) and 2) a more careful integration to get L_2500A from the input L_AGN, rather than just assuming L_2500A=0.1*L_AGN.')

    gamma = 1.9
    alpha_x = 1 - gamma
    alpha_ux = 2.61*alpha_ox + 1.38  # actually depends on gamma
    lam_unit = lam.unit
    lam = lam.to(u.micron).value
    spectrum = np.ones_like(lam)
    _wavs = [1e-4, 5e-3, 0.05, 0.1, 1]
    _powers = [-alpha_x-2, -alpha_ux-2, -0.25, -1.5, -4]
    _pp = 0
    for _w, _p in zip(_wavs, _powers):
        spectrum *= (lam >= _w)*(lam/_w)**(_p - _pp) + (lam < _w)
        _pp = _p
    normalization = np.trapezoid(spectrum, lam)

    return L_AGN * spectrum / (normalization * lam_unit)

def stellar_population_spectrum_lam(lam: u.micron, model='BC03_miles_chabrier',
                                    age=1*u.Gyr, Z=0.02, L_star=None, initial_mass=None, cache=False):
    # Returns a stellar population spectrum for a given model.
    #   lam    = wavelength (microns)
    #   model        = BC03 resource name; downloaded on first use via get_resource_path
    #   age          = population age(s) as a time quantity - Default: 1 Gyr
    #   Z            = metal mass fraction(s) Z, not Z/Z_sun - Default: 0.02
    #   L_star       = optional luminosity integrated over the full native table wavelength range
    #   initial_mass = initial stellar mass as a mass quantity - Default: 1 M_sun if L_star is omitted
    #   cache        = whether to cache the loaded resources - Default: False
    # Returns L_lambda in erg/s per input wavelength unit for a single-age, unattenuated population.
    # Interpolates flux linearly in age, Z and wavelength; extrapolation is not allowed.
    # Enter only one normalization. L_star normalization is independent of the requested wavelength grid.
    # Age, metallicity and normalization broadcast together; output shape is population_shape + lam.shape.
    # enable cache for faster repeated calls, but you will have to manually clear the resource cache when you are done (utils.resources.clear_resource_cache)

    # validate the requested population and normalization before loading resources
    wavelengths = np.asarray(lam.to_value(u.AA))
    age_yr = np.asarray(age.to_value(u.yr))
    Z = np.asarray(Z)
    if np.any(~np.isfinite(age_yr)) or np.any(age_yr < 0):
        raise ValueError('age must contain finite, nonnegative time quantities.')
    if np.any(~np.isfinite(Z)) or np.any(Z <= 0):
        raise ValueError('Z must contain finite, positive metal mass fractions.')
    if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise ValueError('Wavelengths must be finite and positive.')
    if L_star is not None and initial_mass is not None:
        raise ValueError('Enter only one of L_star and initial_mass.')
    scale = L_star.to_value(u.erg/u.s) if L_star is not None else (1 if initial_mass is None else initial_mass.to_value(u.Msun))
    if np.any(~np.isfinite(scale)) or np.any(np.asarray(scale) < 0):
        raise ValueError('The luminosity or initial mass must contain finite, nonnegative values.')
    age_yr, Z, scale = np.broadcast_arrays(age_yr, Z, scale)

    # interpolate the population on the native wavelength grid without extrapolation
    metals, ages, native_lam, spectra = load_resource('stellar_population_models', model, cache=cache)
    if np.any((age_yr < ages[0]) | (age_yr > ages[-1])) or np.any((Z < metals[0]) | (Z > metals[-1])):
        raise ValueError(f'BC03 requires age in [{ages[0]}, {ages[-1]}] yr and Z in [{metals[0]}, {metals[-1]}].')
    if np.any(wavelengths < native_lam[0]) or np.any(wavelengths > native_lam[-1]):
        raise ValueError(f'BC03 wavelengths must be within [{native_lam[0]}, {native_lam[-1]}] Angstrom.')
    points = np.stack((Z, age_yr), axis=-1).reshape(-1, 2)
    spectrum = sp.interpolate.RegularGridInterpolator((metals, ages), spectra)(points).reshape(age_yr.shape + (native_lam.size,))

    # scale the native luminosity per initial solar mass or normalize over the complete table
    if L_star is not None:
        normalization = sp.integrate.trapezoid(spectrum, native_lam)
        if np.any(normalization <= 0):
            raise ValueError('Cannot normalize a BC03 spectrum with zero integrated luminosity.')
        spectrum = spectrum * (scale / normalization)[..., None]
    else:
        spectrum = spectrum * scale[..., None] * u.Lsun.to(u.erg/u.s)
    sampled = sp.interpolate.interp1d(native_lam, spectrum, axis=-1, assume_sorted=True, bounds_error=True)(wavelengths)
    return (sampled * u.erg/u.s/u.AA).to(u.erg/u.s/lam.unit)


def planck_function_lam(lam: u.micron, T: u.K):
    # Returns the Planck function B_lambda(T).
    T = T.to(u.K).value
    lam = lam.to(u.cm).value
    B_lam = (2*c.h.cgs.value*c.c.cgs.value**2)/lam**5 * 1/(np.exp(c.h.cgs.value*c.c.cgs.value/(lam*c.k_B.cgs.value*T)) - 1)
    return B_lam * u.erg / (u.s * u.cm**2 * u.sr * u.cm)

def planck_function_nu(nu: u.Hz, T: u.K):
    # Returns the Planck function B_nu(T).
    T = T.to(u.K).value
    nu = nu.to(u.Hz).value
    B_nu = (2*c.h.cgs.value*nu**3)/c.c.cgs.value**2 * 1/(np.exp(c.h.cgs.value*nu/(c.k_B.cgs.value*T)) - 1)
    return B_nu * u.erg / (u.s * u.cm**2 * u.sr * u.Hz)



### --- GIZMO opacities --- ###

def GIZMO_FREEFREE_kappa(snap, gamma=5.0/3.0, idx=0):
    # kappa in the GIZMO RT_FREEFREE band, as caculated by the code
    U_TO_TEMP_UNITS = c.m_p/c.k_B
    T_eff = (0.59*(gamma-1.0)*snap[idx, 'InternalEnergy'] * U_TO_TEMP_UNITS).to(u.K).value
    kappa_abs = (0.25 + 1e30*snap[idx, 'Density'].to(u.g/u.cm**3).value*pow(T_eff, -3.5))
    return kappa_abs * (u.cm**2/u.g)







### --- Radiative Transfer --- ###






### --- Aliases --- ###
alpha_free_free_cgs = alpha_thermal_bremsstrahlung_cgs
alpha_ff_cgs = alpha_thermal_bremsstrahlung_cgs
alpha_electron_scattering = alpha_thomson_scattering
alpha_es_cgs = alpha_thomson_scattering
alpha_T_cgs = alpha_thomson_scattering
tau_eff = effective_extinction_optical_depth
blackbody_lam = planck_function_lam
blackbody_nu = planck_function_nu

# Backward-compatible aliases used by existing notebooks - should be removed eventually
h_cgs = c.h.cgs.value
c_cgs = c.c.cgs.value




### --- Photometric bandpasses --- ###

BANDPASS_FILTERS = tuple([f'Johnson.{band}' for band in 'UBVRI'] +
                        [f'SDSS.{band}' for band in 'ugriz'] + [f'RGB.{band}' for band in 'RGB'])


def load_bandpass(name, cache=True):
    """Return wavelength, transmission, detector convention and provenance for a filter.

    Johnson.UBVRI use SVO Generic/Johnson profiles, including Johnson R/I rather than
    Cousins R/I. SDSS.ugriz use SVO SLOAN/SDSS full-system responses. Standard profiles
    download lazily through the bandpass_filters resource registry and are cached.
    RGB.R/G/B are built-in Gaussian test responses centered at 610/545/455 nm, with
    sigma 45/35/25 nm and support 380--780 nm. They approximate visible color channels;
    they are not calibrated human cone sensitivities, CIE color matching or sRGB filters.
    Returned cached arrays should be treated as read-only.
    """
    names = {key.lower(): key for key in BANDPASS_FILTERS}
    if str(name).lower() not in names: raise ValueError(f'Unknown bandpass {name!r}; choose from {BANDPASS_FILTERS}.')
    name = names[str(name).lower()]
    if not name.startswith('RGB.'):
        return dict(load_resource('bandpass_filters', name, cache=cache), name=name)
    center, sigma = {'R': (610., 45.), 'G': (545., 35.), 'B': (455., 25.)}[name[-1]]
    wavelength = np.linspace(380., 780., 401)*u.nm
    transmission = np.exp(-0.5*((wavelength.value-center)/sigma)**2)
    transmission[[0, -1]] = 0.0
    return dict(name=name, wavelength=wavelength, transmission=transmission, detector='energy',
                description='Approximate visible test channel; not a colorimetric standard.', source='analytic Gaussian')


def bandpass_response(lam, bandpass, cache=True):
    """Sample a named or custom bandpass dictionary; response is zero outside its table."""
    profile = load_bandpass(bandpass, cache=cache) if isinstance(bandpass, str) else bandpass
    wavelength = u.Quantity(lam, u.AA).to_value(profile['wavelength'].unit)
    return np.interp(wavelength, profile['wavelength'].value, profile['transmission'], left=0.0, right=0.0)


def integrate_bandpass(lam, spectrum, bandpass=None, axis=-1, normalize=False, weighting='energy',
                       wavelength_edges=None, allow_partial=False, cache=True):
    """Integrate an f_lambda spectrum or datacube through a bandpass along its spectral axis.

    lam is a one-dimensional wavelength grid (bare numbers mean Angstrom). spectrum
    is a spectral density, preferably with units, sampled at lam. bandpass is a name
    from BANDPASS_FILTERS, a dictionary with wavelength/transmission arrays, or None
    to integrate the complete input range with unit response. Output keeps other axes.
    Energy weighting returns integral(f_lambda*T*d_lambda). Photon weighting inserts
    lambda/(h*c) and returns a photon rate for an energy spectrum. 'native' selects the
    filter's recorded detector convention. normalize=True instead returns a weighted
    mean f_lambda with the input spectrum's units for either weighting convention.
    No magnitude zero points, redshift transformations or display stretches are applied.

    Sampled spectra and responses are piecewise linear; their product is integrated
    on their combined knots without allocating an interpolated copy of the datacube.
    Supply wavelength_edges for bin-averaged spectra such as raytrace_line_profile:
    these are held constant within each bin, preserving their integrated luminosity.
    The input must span the complete filter table unless allow_partial=True, which
    integrates only the supplied spectral interval (e.g. a line's contribution to a
    broad band). Fine filter sampling cannot recover unresolved spectral features.
    """

    # validate the spectral grid and its optional bin boundaries
    lam = u.Quantity(lam) if isinstance(lam, u.Quantity) else u.Quantity(lam, u.AA)
    wave = np.asarray(lam.value, dtype=float)
    spectrum = u.Quantity(np.asanyarray(spectrum), copy=False)
    if wave.ndim != 1 or wave.size < (2 if wavelength_edges is None else 1) or np.any(~np.isfinite(wave)) or np.any(wave <= 0) or np.any(np.diff(wave) <= 0):
        raise ValueError('Use a finite, positive, increasing wavelength grid.')
    if spectrum.ndim < 1 or not -spectrum.ndim <= axis < spectrum.ndim or spectrum.shape[axis] != wave.size:
        raise ValueError('The spectral axis must match the wavelength grid.')
    edges = None if wavelength_edges is None else u.Quantity(wavelength_edges, lam.unit).to_value(lam.unit)
    if edges is not None and (edges.shape != (wave.size+1,) or np.any(~np.isfinite(edges)) or edges[0] <= 0 or
            np.any(np.diff(edges) <= 0) or np.any(wave < edges[:-1]) or np.any(wave > edges[1:])):
        raise ValueError('Wavelength edges must bracket every spectral bin and increase strictly.')
    samples = wave if edges is None else edges
    profile = dict(wavelength=samples[[0, -1]]*lam.unit, transmission=np.ones(2), detector='energy') if bandpass is None else (
        load_bandpass(bandpass, cache=cache) if isinstance(bandpass, str) else bandpass)
    filter_wave = u.Quantity(profile['wavelength']).to_value(lam.unit)
    transmission = np.asarray(profile['transmission'], dtype=float)
    if filter_wave.ndim != 1 or filter_wave.size < 2 or np.any(~np.isfinite(filter_wave)) or np.any(filter_wave <= 0) or np.any(np.diff(filter_wave) <= 0):
        raise ValueError('Filter wavelengths must be finite, positive and increasing.')
    if transmission.shape != filter_wave.shape or np.any(~np.isfinite(transmission)) or np.any(transmission < 0) or not np.any(transmission > 0):
        raise ValueError('Filter transmission must be finite, nonnegative and nonzero.')
    margin = 8*np.finfo(float).eps
    if not allow_partial and (samples[0] > filter_wave[0]*(1+margin) or samples[-1] < filter_wave[-1]*(1-margin)):
        raise ValueError('The spectrum does not cover the full bandpass; use allow_partial=True for a partial contribution.')
    weighting = profile.get('detector', 'energy') if weighting == 'native' else weighting
    if weighting not in ('energy', 'photon'): raise ValueError("weighting must be 'energy', 'photon' or 'native'.")

    # project exact interval quadrature onto the original spectral samples
    knots = np.unique(np.r_[samples, filter_wave[(filter_wave > samples[0]) & (filter_wave < samples[-1])]])
    half_width = 0.5*np.diff(knots)
    query = (0.5*(knots[:-1]+knots[1:])[:, None] + half_width[:, None]*np.array([-1., 1.])/np.sqrt(3)).ravel()
    quadrature = np.repeat(half_width, 2)*np.interp(query, filter_wave, transmission, left=0.0, right=0.0)
    if weighting == 'photon': quadrature *= (query*lam.unit/(c.h*c.c)).to_value(1/u.erg)
    indices = np.clip(np.searchsorted(samples, query, side='right')-1, 0, wave.size-(2 if edges is None else 1))
    if edges is None:
        fraction = (query-wave[indices])/(wave[indices+1]-wave[indices])
        weights = np.bincount(indices, weights=quadrature*(1-fraction), minlength=wave.size)
        weights += np.bincount(indices+1, weights=quadrature*fraction, minlength=wave.size)
    else:
        weights = np.bincount(indices, weights=quadrature, minlength=wave.size)
    output_unit = spectrum.unit*lam.unit*(u.photon/u.erg if weighting == 'photon' else 1)
    if normalize:
        if weights.sum() <= 0: raise ValueError('Cannot normalize a bandpass with zero overlap.')
        weights /= weights.sum(); output_unit = spectrum.unit
    return np.einsum('...i,i->...', np.moveaxis(spectrum.value, axis, -1), weights)*output_unit


### --- Particle radiative transfer --- ###

@numba.njit(cache=True, error_model='numpy')
def _trace_cartesian_ray(ray, x0, displacement, lengths, axes, jnu, alpha, factors, field_indices,
                          step_fraction, neighbours_count, shared_alpha, reverse, empty_row,
                          boundary, intensity, tau, errors):
    """Fused transfer for one ray, with reusable spectral buffers."""
    n_freq, n_cases = factors.shape
    dimension = x0.shape[1]
    point = np.empty((1, dimension))
    emission = np.empty(n_freq)
    absorption = np.empty(1 if shared_alpha else n_freq)
    column = np.zeros(1 if shared_alpha else n_freq)
    weights = np.empty(neighbours_count)
    rows = np.empty(neighbours_count, dtype=np.int64)
    for channel in range(n_freq):
        for case in range(n_cases):
            intensity[ray, channel, case] = 0. if reverse else boundary[ray, channel, case]
    parameter = 0.
    while parameter < 1. and lengths[ray] > 0:
        for dim in range(dimension): point[0, dim] = x0[ray, dim]+parameter*displacement[ray, dim]
        distances, indices = _cartesian_grid_query(point, axes, neighbours_count)
        h = distances[0, -1]
        if not np.isfinite(h) or h <= 0:
            errors[ray] = 1
            break
        next_parameter = min(parameter+step_fraction*h/lengths[ray], 1.)
        if next_parameter <= parameter:
            errors[ray] = 2
            break
        length = (next_parameter-parameter)*lengths[ray]
        invh = 1/h
        norm = (3/np.pi)*invh*invh*invh if dimension == 3 else (15/(7*np.pi))*invh*invh if dimension == 2 else .5*invh
        total_weight = 0.
        empty = empty_row >= 0
        for neighbour in range(neighbours_count):
            q = distances[0, neighbour]*invh
            value = (8/3)-16*(1-q)*q*q if q < .5 else (16/3)*(1-q)*(1-q)*(1-q) if q < 1 else 0.
            weights[neighbour] = value*norm
            total_weight += weights[neighbour]
            index = indices[0, neighbour]
            rows[neighbour] = field_indices[index] if field_indices.size else index
            if rows[neighbour] != empty_row and weights[neighbour] != 0: empty = False
        parameter = next_parameter
        if empty and total_weight > 0: continue

        # interpolate only occupied samples into reusable per-ray spectral buffers
        emission.fill(0.); absorption.fill(0.)
        for neighbour in range(neighbours_count):
            weight = weights[neighbour]/total_weight
            if weight == 0: continue
            index = rows[neighbour]
            if shared_alpha: absorption[0] += weight*alpha[index, 0]
            for channel in range(n_freq):
                emission[channel] += weight*jnu[index, channel]
                if not shared_alpha: absorption[channel] += weight*alpha[index, channel]

        # advance all opacity cases using one cumulative column per opacity channel
        for channel in range(n_freq):
            opacity_channel = 0 if shared_alpha else channel
            increment = absorption[opacity_channel]*length
            for case in range(n_cases):
                factor = factors[channel, case]
                dtau = increment*factor
                segment = 1. if dtau == 0 else -np.expm1(-dtau)/dtau
                if reverse:
                    transmission = 1. if factor == 0 else np.exp(-column[opacity_channel]*factor)
                    intensity[ray, channel, case] += emission[channel]*transmission*length*segment
                else:
                    transmission = 1. if dtau == 0 else np.exp(-dtau)
                    intensity[ray, channel, case] = intensity[ray, channel, case]*transmission+emission[channel]*length*segment
            if not shared_alpha: column[channel] += increment
        if shared_alpha: column[0] += absorption[0]*length

    # apply the incident boundary and form optical depths only after finishing the ray
    for channel in range(n_freq):
        for case in range(n_cases):
            depth = column[0 if shared_alpha else channel]*factors[channel, case]
            if tau.size: tau[ray, channel, case] = depth
            if reverse and boundary[ray, channel, case] != 0:
                intensity[ray, channel, case] += boundary[ray, channel, case]*np.exp(-depth)


@numba.njit(cache=True, parallel=False, error_model='numpy')
def _trace_cartesian_serial(x0, displacement, lengths, axes, jnu, alpha, factors, field_indices,
                          step_fraction, neighbours_count, shared_alpha, reverse, empty_row,
                          boundary, intensity, tau, errors):
    for ray in range(len(x0)):
        _trace_cartesian_ray(ray, x0, displacement, lengths, axes, jnu, alpha, factors, field_indices, step_fraction, neighbours_count, shared_alpha, reverse, empty_row, boundary, intensity, tau, errors)


@numba.njit(cache=True, parallel=True, error_model='numpy')
def _trace_cartesian_parallel(x0, displacement, lengths, axes, jnu, alpha, factors, field_indices,
                          step_fraction, neighbours_count, shared_alpha, reverse, empty_row,
                          boundary, intensity, tau, errors):
    for ray in numba.prange(len(x0)):
        _trace_cartesian_ray(ray, x0, displacement, lengths, axes, jnu, alpha, factors, field_indices, step_fraction, neighbours_count, shared_alpha, reverse, empty_row, boundary, intensity, tau, errors)

def raytrace_particles(x0, x1, jnu, alpha, pos=None, tree=None, step_fraction=0.25, I0=0.0,
                       approx=True, N_NGB=32, workers=-1, query_chunk_size=250_000,
                       ret_tau=False, log=False, eta_update_seconds=5.0,
                       grid_to_particle=None, max_grid_step=1/32, path_rtol=1e-4,
                       reverse=False, ret_particle_contributions=False, pixel_area=1.0,
                       opacity_factors=None, frequency_chunk_size=4, field_indices=None, backend='auto'):
    # Integrates dI_nu/ds = jnu - alpha*I_nu from x0 to x1 using mfm_value_at_pos.
    #   x0/x1            = starting/end points in the chosen coordinates, shaped (..., D)
    #   jnu              = particle emissivity per solid angle, shaped (N_particles,) or (N_particles, N_freq)
    #   alpha            = particle absorption coefficient kappa*rho, with the same formats (one channel may broadcast)
    #   pos/tree         = physical Cartesian particle positions (N_particles, D) or their KD-tree, in matching particle order
    #   step_fraction    = step length divided by local h (default 0.25); the last step is clipped to x1
    #   I0               = intensity at x0; scalar or broadcastable to the returned intensity shape
    #   approx/N_NGB     = smoothing options for mfm_value_at_pos; require at least N_NGB particles (2*N_NGB if not approx)
    #   workers          = number of KD-tree query workers (-1 for all)
    #   query_chunk_size = maximum query points per interpolation call (at least two for the iterative smoothing solver)
    #   ret_tau          = also return the integrated optical depths; log enables time-based progress reports
    #   grid_to_particle = optional inverse coordinate map: (N, D) grid coordinates -> (N, D) physical Cartesian positions
    #   max_grid_step    = maximum fraction of a transformed path per step, to resolve curvature (default 1/32)
    #   path_rtol        = relative tolerance between one-chord and two-chord physical path lengths (default 1e-4)
    # Traces q(t)=x0+t*(x1-x0), 0<=t<=1; grid_to_particle(q(t)) may curve in physical space.
    # Use unwrapped angles to select the intended angular path; no automatic periodic wrapping is applied.
    # pos/tree and the map output use a common physical length unit, alpha its inverse, and jnu intensity per unit length.
    # No Jacobian rescales jnu or alpha: both remain physical coefficients; only the path length is transformed.
    # Returns intensities shaped x0.shape[:-1], with a trailing N_freq axis for two-dimensional coefficient inputs.
    # At each step, query h, jnu, and alpha at the physical position; advance at most step_fraction*h in physical arc length.
    # Here h is the kernel support radius returned by mfm_value_at_pos, not half that radius.
    # Coefficients are held constant over each segment (first-order spatial sampling); reduce step_fraction for convergence.
    # Uses the exact constant-coefficient segment solution, including its transparent limit.
    # No scattering source, Doppler shifts, geometric dilution of intensity, or automatic clipping to a particle boundary.
    # The interpolator extrapolates outside the particle cloud; choose x0/x1 to delimit the integration domain.

    # reverse=True accumulates foreground-attenuated emission from x1 toward x0.
    # ret_particle_contributions implies reverse=True and appends an (N_particles, N_channels)
    # array of area-weighted emergent contributions, excluding I0. Scalar fields omit the channel axis.
    # Return order is intensity, optional tau, optional particle_contributions; pixel_area broadcasts over rays.

    # opacity_factors optionally multiplies alpha by an (N_freq, N_cases) array, with frequency broadcasting.
    # This shares emissivities and spatial opacity fields across cases without duplicating particle arrays.
    # Outputs then have trailing (N_freq, N_cases) axes, including particle_contributions.
    # frequency_chunk_size bounds spectral temporaries (default 4; None uses all channels).
    # Every chunk shares the same spatial queries and steps.
    # A CartesianGrid may replace tree for fields on a Cartesian product of coordinate axes.
    # backend='auto' uses fused Numba transfer for CartesianGrid with approx=True,
    # no coordinate map and no particle contributions; other calls use the NumPy path.
    # backend='numpy' forces the reference path; 'numba' requires those supported settings.
    # Numba preserves sampling and precision, parallelizes rays using workers, and uses
    # query_chunk_size as the number of complete rays per batch. frequency_chunk_size
    # applies only to NumPy; Numba needs only one spectral buffer per active thread.
    # field_indices optionally maps each tree point to a coefficient row, allowing empty grid
    # cells to share one zero row. Geometry and weights still use the complete tree. Returned
    # particle contributions then group by coefficient row, with shape (N_rows, ...).

    from ..utils.grid import mfm_value_at_pos

    # flatten the endpoint grid and prepare a shared physical particle tree
    x0, x1 = np.asarray(x0, dtype=np.float64), np.asarray(x1, dtype=np.float64)
    assert x0.ndim >= 2 and x1.shape == x0.shape, 'x0 and x1 must both have shape (..., D).'
    ray_shape = x0.shape[:-1]
    x0, x1 = x0.reshape(-1, x0.shape[-1]), x1.reshape(-1, x1.shape[-1])
    assert np.isfinite(step_fraction) and 0 < step_fraction <= 1, 'step_fraction must be in (0, 1].'
    assert query_chunk_size > 0, 'query_chunk_size must be a positive integer.'
    assert 0 < max_grid_step <= 1 and path_rtol > 0, 'Require max_grid_step in (0, 1] and path_rtol > 0.'
    if tree is None:
        from scipy.spatial import cKDTree
        tree = cKDTree(pos)
    assert x0.shape[1] == tree.m, 'Ray and particle coordinates must have the same dimension.'
    assert 2 <= N_NGB <= tree.n // (1 if approx else 2), 'Too few particles for the requested N_NGB.'
    reverse = reverse or ret_particle_contributions
    if reverse: x0, x1 = x1, x0
    displacement = x1 - x0
    length = np.linalg.norm(displacement, axis=1)

    # broadcast scalar particle fields across frequency channels
    jnu, alpha = np.asarray(jnu, dtype=np.float64), np.asarray(alpha, dtype=np.float64)
    assert jnu.ndim in (1, 2) and alpha.ndim in (1, 2), 'jnu and alpha must be particle arrays.'
    n_fields = jnu.shape[0]
    assert alpha.shape[0] == n_fields, 'jnu and alpha must have the same number of rows.'
    if field_indices is None:
        assert n_fields == tree.n, 'jnu and alpha must match the particle tree.'
    else:
        field_indices = np.asarray(field_indices)
        assert field_indices.shape == (tree.n,) and np.issubdtype(field_indices.dtype, np.integer), 'field_indices must contain one integer per tree point.'
        assert field_indices.min() >= 0 and field_indices.max() < n_fields, 'field_indices must index coefficient rows.'
    spectral = jnu.ndim == 2 or alpha.ndim == 2
    jnu = jnu[:, None] if jnu.ndim == 1 else jnu
    alpha = alpha[:, None] if alpha.ndim == 1 else alpha
    factors = np.ones((1, 1)) if opacity_factors is None else np.asarray(opacity_factors, dtype=float)
    assert factors.ndim == 2 and factors.shape[1] > 0, 'opacity_factors must have shape (N_freq, N_cases).'
    n_freq = np.broadcast_shapes(jnu.shape[1:], alpha.shape[1:], factors.shape[:1])[0]
    n_rays, n_cases = x0.shape[0], factors.shape[1]
    shared_alpha = alpha.shape[1] == 1
    jnu, alpha = np.broadcast_to(jnu, (n_fields, n_freq)), np.broadcast_to(alpha, (n_fields, n_freq))
    factors = np.broadcast_to(factors, (n_freq, n_cases))
    frequency_chunk_size = n_freq if frequency_chunk_size is None else frequency_chunk_size
    assert int(frequency_chunk_size) == frequency_chunk_size and frequency_chunk_size > 0, 'frequency_chunk_size must be a positive integer.'
    frequency_chunk_size = int(frequency_chunk_size)

    # select compiled Cartesian transfer without changing the query or step algorithm
    if backend not in ('auto', 'numpy', 'numba'): raise ValueError("backend must be 'auto', 'numpy' or 'numba'.")
    supported = isinstance(tree, CartesianGrid) and approx and grid_to_particle is None and not ret_particle_contributions
    if backend == 'numba' and not supported:
        raise ValueError('The numba backend requires CartesianGrid, approx=True, no coordinate map and no particle contributions.')
    output_shape = ray_shape + ((n_freq, n_cases) if opacity_factors is not None else (n_freq,) if spectral else ())
    if supported and backend != 'numpy':
        threads = numba.get_num_threads() if workers == -1 else workers
        if int(threads) != threads or not 1 <= threads <= numba.config.NUMBA_NUM_THREADS:
            raise ValueError('workers must be -1 or a positive integer within the configured Numba thread limit.')
        boundary = np.broadcast_to(np.asarray(I0, dtype=np.float64), output_shape).reshape(n_rays, n_freq, n_cases)
        intensity = np.empty((n_rays, n_freq, n_cases))
        tau = np.empty_like(intensity) if ret_tau else np.empty((0, 0, 0))
        mapping = np.empty(0, dtype=np.int64) if field_indices is None else field_indices
        empty_row = 0 if mapping.size and np.all(jnu[0] == 0) and np.all(alpha[0] == 0) else -1
        kernel = _trace_cartesian_parallel if threads > 1 else _trace_cartesian_serial
        if log:
            import time
            started = last_report = time.perf_counter()
            print(f'Cartesian transfer: {n_rays:,} rays on {threads} threads (first call may compile).', flush=True)
        with numba_threads(int(threads)):
            for first in range(0, n_rays, query_chunk_size):
                last = min(first+query_chunk_size, n_rays)
                errors = np.zeros(last-first, dtype=np.uint8)
                kernel(x0[first:last], displacement[first:last], length[first:last], tree.axes,
                       jnu, alpha, factors, mapping, step_fraction, N_NGB, shared_alpha, reverse, empty_row,
                       boundary[first:last], intensity[first:last], tau[first:last] if ret_tau else tau, errors)
                if np.any(errors == 1): raise ValueError('Local smoothing lengths must be finite and positive.')
                if np.any(errors == 2): raise ValueError('The local smoothing length is too small to advance in floating-point precision.')
                if log and time.perf_counter()-last_report >= eta_update_seconds:
                    now = time.perf_counter()
                    print(f'Cartesian transfer {last/n_rays:.1%} / ETA {(now-started)*(n_rays-last)/last/60:.2f} min', flush=True)
                    last_report = now
        if log: print(f'Cartesian transfer finished in {time.perf_counter()-started:.2f} seconds.', flush=True)
        intensity = intensity.reshape(output_shape)
        return (intensity, tau.reshape(output_shape)) if ret_tau else intensity

    # initialize the final rays using their original scalar or spectral output format
    output_shape = ray_shape + ((n_freq, n_cases) if opacity_factors is not None else (n_freq,) if spectral else ())
    intensity = np.broadcast_to(np.asarray(I0, dtype=np.float64), output_shape).copy().reshape(n_rays, n_freq, n_cases)
    tau = np.zeros_like(intensity)
    boundary_intensity = intensity.copy() if reverse else None
    if reverse: intensity.fill(0.0)
    if ret_particle_contributions:
        area = np.broadcast_to(np.asarray(pixel_area, dtype=float), ray_shape).ravel()
        assert np.all(np.isfinite(area)) and np.all(area > 0), 'Pixel areas must be finite and positive.'
        particle_contributions = np.zeros((n_fields, n_freq, n_cases))

    # bound the sample allocation while leaving zero-length rays unchanged
    parameter = np.zeros(n_rays)
    active = np.flatnonzero(np.any(displacement != 0, axis=1))
    if log:
        import time
        start_time = last_report = time.perf_counter()
        completed = 0.0
        total = active.size

    while active.size:

        # advance unfinished rays independently in bounded query chunks
        for ray_start in range(0, active.size, query_chunk_size):
            rays = active[ray_start:ray_start + query_chunk_size]

            # query the local support radius and reuse its neighbors for both coefficients
            query_pos = x0[rays] + parameter[rays, None]*displacement[rays]
            if grid_to_particle is not None:
                query_pos = np.asarray(grid_to_particle(query_pos), dtype=np.float64)
                assert query_pos.shape == (rays.size, tree.m), 'grid_to_particle must preserve the coordinate array shape.'
            n_queries = query_pos.shape[0]
            if not approx and n_queries == 1:
                query_pos = np.repeat(query_pos, 2, axis=0)
            h, neighbours, W = mfm_value_at_pos(query_pos, tree=tree, ret_h=True, ret_neighbours=True,
                                               approx=approx, N_NGB=N_NGB, workers=workers)
            h, neighbours, W = h[:n_queries], neighbours[:n_queries], W[:n_queries]
            if np.any(~np.isfinite(h)) or np.any(h <= 0):
                raise ValueError('Local smoothing lengths must be finite and positive.')

            # set the physical step directly for Cartesian coordinates
            target_step = step_fraction*h
            if grid_to_particle is None:
                next_parameter = np.minimum(parameter[rays] + target_step/length[rays], 1.0)
                step_length = (next_parameter - parameter[rays])*length[rays]
            else:

                # estimate the local coordinate-to-distance scale without requiring an analytic Jacobian
                probe_step = np.minimum(1e-5, 1.0 - parameter[rays])
                probe = x0[rays] + (parameter[rays] + probe_step)[:, None]*displacement[rays]
                physical_start = query_pos[:n_queries]
                physical_probe = np.asarray(grid_to_particle(probe), dtype=np.float64)
                speed = np.linalg.norm(physical_probe - physical_start, axis=1)/probe_step
                parameter_step = np.full(rays.size, max_grid_step)
                np.divide(target_step, speed, out=parameter_step, where=speed > 0)
                np.minimum(parameter_step, max_grid_step, out=parameter_step)
                np.minimum(parameter_step, 1.0 - parameter[rays], out=parameter_step)
                step_length = np.empty(rays.size)
                next_parameter = np.empty(rays.size)
                pending = np.arange(rays.size)

                # shorten trial segments until their physical length and curvature are resolved
                while pending.size:
                    selected = rays[pending]
                    end_parameter = np.minimum(parameter[selected] + parameter_step[pending], 1.0)
                    if np.any(end_parameter <= parameter[selected]):
                        raise ValueError('The transformed path step is too small to advance in floating-point precision.')
                    midpoint = 0.5*(parameter[selected] + end_parameter)
                    physical_mid = np.asarray(grid_to_particle(x0[selected] + midpoint[:, None]*displacement[selected]), dtype=np.float64)
                    physical_end = np.asarray(grid_to_particle(x0[selected] + end_parameter[:, None]*displacement[selected]), dtype=np.float64)
                    chord = np.linalg.norm(physical_end - physical_start[pending], axis=1)
                    arc = np.linalg.norm(physical_mid - physical_start[pending], axis=1) + np.linalg.norm(physical_end - physical_mid, axis=1)
                    if np.any(~np.isfinite(arc)):
                        raise ValueError('grid_to_particle must return finite positions along each ray.')
                    refine = (arc > target_step[pending]*(1 + 1e-10)) | (arc - chord > path_rtol*arc)
                    step_length[pending], next_parameter[pending] = arc, end_parameter
                    pending = pending[refine]
                    parameter_step[pending] *= 0.5

            # interpolate the physical coefficients with the shared neighbor weights
            if np.any(next_parameter <= parameter[rays]):
                raise ValueError('The local smoothing length is too small to advance in floating-point precision.')
            step_length = step_length[:, None, None]
            W /= W.sum(axis=1, keepdims=True)
            if field_indices is not None: neighbours = field_indices[neighbours]
            if shared_alpha: alpha_shared = np.einsum('ij,ij->i', alpha[neighbours, 0], W)[:, None]
            for start in range(0, n_freq, frequency_chunk_size):
                stop = min(start+frequency_chunk_size, n_freq)
                channels = slice(start, stop)
                j_sample = np.einsum('ijf,ij->if', jnu[neighbours, channels], W)
                alpha_sample = alpha_shared if shared_alpha else np.einsum('ijf,ij->if', alpha[neighbours, channels], W)

                # apply the segment solution with a stable zero-opacity limit
                dtau = alpha_sample[..., None]*factors[channels]*step_length
                emission_factor = np.ones_like(dtau)
                np.divide(-np.expm1(-dtau), dtau, out=emission_factor, where=dtau != 0)
                if reverse:
                    escaped_length = np.exp(-tau[rays, channels])*step_length*emission_factor
                    intensity[rays, channels] += j_sample[..., None]*escaped_length
                    if ret_particle_contributions:
                        for channel in range(start, stop):
                            contributions = (W*jnu[neighbours, channel])[..., None]*(escaped_length[:, channel-start]*area[rays, None])[:, None, :]
                            np.add.at(particle_contributions[:, channel], neighbours.ravel(), contributions.reshape(-1, n_cases))
                else:
                    intensity[rays, channels] = intensity[rays, channels]*np.exp(-dtau) + j_sample[..., None]*step_length*emission_factor
                tau[rays, channels] += dtau
            del neighbours, W, query_pos

            # report progress after completed interpolation chunks
            if log:
                completed += np.sum(next_parameter - parameter[rays])
                now = time.perf_counter()
                if now - last_report >= eta_update_seconds:
                    eta = (now - start_time)*(total - completed)/completed
                    print(f'\rRay tracing {completed/total:.1%} / ETA {eta/60:.2f} min', end='', flush=True)
                    last_report = now
            parameter[rays] = next_parameter
        active = active[parameter[active] < 1.0]

    # restore the scalar or spectral result shape
    if log:
        print(f'\rRay tracing 100.0% / finished in {time.perf_counter() - start_time:.2f} seconds.')
    if reverse: intensity += boundary_intensity*np.exp(-tau)
    intensity, tau = intensity.reshape(output_shape), tau.reshape(output_shape)
    result = [intensity]
    if ret_tau: result.append(tau)
    if ret_particle_contributions: result.append(particle_contributions.reshape((n_fields,)+output_shape[len(ray_shape):]))
    return tuple(result) if len(result) > 1 else intensity


def raytrace_line_profile(x0, x1, pixel_area, emissivity, alpha, wavelength0,
                          pos=None, tree=None, velocity=None, temperature=None, particle_mass=c.m_p + c.m_e,
                          velocity_dispersion=0*u.km/u.s, case_names=None,
                          spectral_width=3000.0, n_spectral=100, velocity_center=0.0, systemic_velocity=0.0,
                          step_fraction=0.25, approx=True, N_NGB=32, workers=-1,
                          query_chunk_size=4096, particle_chunk_size=4096, log=False, eta_update_seconds=5.0,
                          thermal_broadening=True, doppler_shift=True, relativistic=True):
    """Trace a line toward +z using explicit particle arrays and grey opacity channels.

    Geometry, angle-integrated emissivity and opacity use cm, erg/s/cm^3 and cm^-1;
    quantities are accepted. alpha is (N_particles,) or (N_particles, N_cases).
    wavelength0 is a wavelength quantity (plain numbers mean Angstrom).
    velocity is (N_particles, 3) in km/s; temperature is in K; particle_mass is in g.
    velocity_dispersion is an additional rest-frame Gaussian sigma in km/s, scalar or per particle.
    Thermal and extra dispersions add in quadrature. Relativistic bulk shifts include
    transverse motion; emission beaming, natural broadening and line self-opacity are omitted.
    Broadening follows grey transport, with exact Gaussian bin integrals and no wing renormalization.
    Returns velocity/wavelength grids and a profiles dictionary keyed by case_names, each
    containing dL_dv, dL_dlambda, line_luminosity, luminosity_in_window, window_fraction,
    line-integrated intensity and tau. Luminosities are directional isotropic equivalents.
    The observer direction is shared by all rays. Memory is particles times opacity cases,
    with spectral temporaries bounded by particle_chunk_size; no snapshot is required.
    """

    # reuse the existing particle interpolator and thermal gaussian integrals
    import time
    from scipy.special import ndtr
    from scipy.spatial import cKDTree

    # convert geometry and particle coefficients without duplicating already-cgs arrays
    x0 = np.asarray(u.Quantity(x0, u.cm, copy=False).to_value(u.cm), dtype=float)
    x1 = np.asarray(u.Quantity(x1, u.cm, copy=False).to_value(u.cm), dtype=float)
    ray_shape = x0.shape[:-1]
    assert x0.shape == x1.shape and x0.shape[-1] == 3, 'Ray endpoints must have matching (..., 3) shapes.'
    area = np.broadcast_to(u.Quantity(pixel_area, u.cm**2).to_value(u.cm**2), ray_shape).ravel()
    x0, x1 = x0.reshape(-1, 3), x1.reshape(-1, 3)
    assert np.all(np.isfinite(x0)) and np.all(np.isfinite(x1)), 'Ray endpoints must be finite.'
    assert np.all(x0[:, :2] == x1[:, :2]) and np.all(x1[:, 2] > x0[:, 2]), 'Rays must point along +z.'
    assert np.all(np.isfinite(area)) and np.all(area > 0), 'Pixel areas must be finite and positive.'
    emissivity = u.Quantity(emissivity, u.erg/u.s/u.cm**3, copy=False).to_value(u.erg/u.s/u.cm**3)
    alpha = np.asarray(u.Quantity(alpha, u.cm**-1, copy=False).to_value(u.cm**-1))
    if alpha.ndim == 1: alpha = alpha[:, None]
    if tree is None: tree = cKDTree(u.Quantity(pos, u.cm, copy=False).to_value(u.cm))
    assert tree.m == 3, 'The particle tree must be three-dimensional.'
    assert emissivity.shape == (tree.n,) and alpha.ndim == 2 and alpha.shape[0] == tree.n, 'Coefficients must match the particle tree.'
    assert np.all(np.isfinite(emissivity)) and np.all(emissivity >= 0) and np.all(np.isfinite(alpha)) and np.all(alpha >= 0)
    cases = tuple(case_names) if case_names is not None else tuple(f'case_{i}' for i in range(alpha.shape[1]))
    assert len(cases) == alpha.shape[1] and len(set(cases)) == len(cases), 'Use one unique name per opacity channel.'
    n_cases = len(cases)
    if doppler_shift:
        velocity = u.Quantity(velocity, u.km/u.s).to_value(u.km/u.s)
        assert velocity.shape == (tree.n, 3), 'velocity must have shape (N_particles, 3).'
    if thermal_broadening:
        temperature = np.broadcast_to(u.Quantity(temperature, u.K).to_value(u.K), (tree.n,))
    dispersion = np.broadcast_to(u.Quantity(velocity_dispersion, u.km/u.s).to_value(u.km/u.s), (tree.n,))
    assert np.all(np.isfinite(dispersion)) and np.all(dispersion >= 0), 'Velocity dispersions must be finite and nonnegative.'
    particle_mass = u.Quantity(particle_mass, u.g).to_value(u.g)
    assert np.ndim(particle_mass) == 0 and np.isfinite(particle_mass) and particle_mass > 0
    assert 0 < step_fraction <= 1 and N_NGB >= 2, 'Use 0 < step_fraction <= 1 and N_NGB >= 2.'
    assert tree.n >= (N_NGB if approx else 2*N_NGB), 'Not enough particles for the requested neighbor count.'
    assert query_chunk_size >= 1 and particle_chunk_size >= 1 and eta_update_seconds > 0, 'Chunk sizes and update interval must be positive.'

    # define an optical-velocity grid with exactly integrated thermal probabilities per bin
    velocity_unit = u.km/u.s
    spectral_width = u.Quantity(spectral_width, velocity_unit).to_value(velocity_unit)
    velocity_center = u.Quantity(velocity_center, velocity_unit).to_value(velocity_unit)
    systemic_velocity = u.Quantity(systemic_velocity, velocity_unit).to_value(velocity_unit)
    assert np.isfinite(spectral_width) and spectral_width > 0 and int(n_spectral) == n_spectral and n_spectral >= 1
    assert np.isfinite(velocity_center) and np.isfinite(systemic_velocity)
    speed_of_light = c.c.to_value(velocity_unit)
    if doppler_shift and relativistic and abs(systemic_velocity) >= speed_of_light:
        raise ValueError('Systemic velocity must have magnitude below the speed of light.')
    velocity_edges = np.linspace(velocity_center - spectral_width/2, velocity_center + spectral_width/2, int(n_spectral) + 1)
    velocity_centers = 0.5*(velocity_edges[:-1] + velocity_edges[1:])
    wavelength0 = u.Quantity(wavelength0, u.AA).to(u.AA)
    wavelength_edges = wavelength0*(1 + velocity_edges/c.c.to_value(velocity_unit))
    assert np.all(wavelength_edges > 0*u.AA), 'The optical-velocity grid must give positive wavelengths.'
    wavelength = 0.5*(wavelength_edges[:-1] + wavelength_edges[1:])

    # accumulate particle escape contributions once for all spectral bins and all cases
    start_time = last_report = time.perf_counter()
    intensity, tau, particle_luminosity = raytrace_particles(
        x0, x1, emissivity/(4*np.pi), alpha, tree=tree, step_fraction=step_fraction,
        approx=approx, N_NGB=N_NGB, workers=workers, query_chunk_size=query_chunk_size,
        ret_tau=True, ret_particle_contributions=True, pixel_area=area, log=log,
        eta_update_seconds=eta_update_seconds)
    particle_luminosity *= 4*np.pi
    ray_luminosity = intensity*(4*np.pi*area[:, None])

    # broaden particle contributions individually rather than averaging velocities or temperatures first
    spectrum = np.zeros((velocity_centers.size, n_cases))
    thermal_factor = (c.k_B/(particle_mass*u.g)).to_value(velocity_unit**2/u.K)
    for particle_start in range(0, tree.n, particle_chunk_size):
        stop = min(particle_start + particle_chunk_size, tree.n)
        luminosity = particle_luminosity[particle_start:stop]
        emitting = np.any(luminosity > 0, axis=1)
        if not np.any(emitting): continue
        luminosity = luminosity[emitting]

        # shift line centers classically or with transverse motion and the relativistic frame boost
        centers = np.zeros(luminosity.shape[0])
        wavelength_factor = np.ones(centers.size)
        if doppler_shift:
            particle_velocity = velocity[particle_start:stop][emitting]
            if np.any(~np.isfinite(particle_velocity)): raise ValueError('Emitting particles must have finite velocities.')
            if relativistic:
                beta = particle_velocity/speed_of_light
                beta_squared = np.einsum('ij,ij->i', beta, beta)
                if np.any(beta_squared >= 1): raise ValueError('Emitting particle velocities must be subluminal.')
                systemic_beta = systemic_velocity/speed_of_light
                log_factor = -0.5*np.log1p(-beta_squared) + np.log1p(-beta[:, 2]) + np.arctanh(systemic_beta)
                wavelength_factor = np.exp(log_factor)
                centers = speed_of_light*np.expm1(log_factor)
            else:
                centers = systemic_velocity - particle_velocity[:, 2]

        # retain independent thermal broadening and transform its wavelength scale with the bulk shift
        sigma = dispersion[particle_start:stop][emitting]*wavelength_factor
        if thermal_broadening:
            particle_temperature = temperature[particle_start:stop][emitting]
            if np.any(~np.isfinite(particle_temperature)) or np.any(particle_temperature < 0):
                raise ValueError('Thermally broadened emitting particles must have finite, nonnegative temperatures.')
            sigma = np.sqrt(thermal_factor*particle_temperature + dispersion[particle_start:stop][emitting]**2)*wavelength_factor
        cold = sigma == 0

        # integrate gaussian tails stably, skipping them entirely when all lines are unbroadened
        probability = np.zeros((centers.size, velocity_centers.size))
        if np.any(~cold):
            scaled_edges = (velocity_edges[None, :] - centers[:, None])/np.where(cold, 1.0, sigma)[:, None]
            lower, upper = scaled_edges[:, :-1], scaled_edges[:, 1:]
            right_tail = lower > 0
            cdf_lower = ndtr(np.where(right_tail, -upper, lower))
            cdf_upper = ndtr(np.where(right_tail, -lower, upper))
            probability = np.maximum(cdf_upper - cdf_lower, 0.0)
        if np.any(cold):
            probability[cold] = 0.0
            cold_rows = np.flatnonzero(cold)
            bins = np.searchsorted(velocity_edges, centers[cold], side='right') - 1
            bins[centers[cold] == velocity_edges[-1]] = velocity_centers.size - 1
            inside = (bins >= 0) & (bins < velocity_centers.size)
            probability[cold_rows[inside], bins[inside]] = 1.0
        spectrum += probability.T @ luminosity
        if log and time.perf_counter() - last_report >= eta_update_seconds:
            print(f'Line spectral accumulation {100*stop/tree.n:.1f}%', flush=True)
            last_report = time.perf_counter()

    # return unit-bearing spectra, integrated luminosities and matching line-integrated images
    result = {'velocity': velocity_centers*velocity_unit, 'velocity_edges': velocity_edges*velocity_unit,
              'wavelength': wavelength, 'wavelength_edges': wavelength_edges, 'wavelength0': wavelength0,
              'profiles': {}}
    for case, name in enumerate(cases):
        total = ray_luminosity[:, case].sum()
        in_window = spectrum[:, case].sum()
        result['profiles'][name] = {
            'dL_dv': spectrum[:, case]*u.erg/u.s/np.diff(result['velocity_edges']),
            'dL_dlambda': spectrum[:, case]*u.erg/u.s/np.diff(wavelength_edges),
            'line_luminosity': total*u.erg/u.s, 'luminosity_in_window': in_window*u.erg/u.s,
            'window_fraction': in_window/total if total > 0 else np.nan,
            'intensity': (ray_luminosity[:, case]/(4*np.pi*area)).reshape(ray_shape)*u.erg/u.s/u.cm**2/u.sr,
            'tau': tau[:, case].reshape(ray_shape)}
    if log: print(f'Line profiles finished in {time.perf_counter() - start_time:.2f} s.', flush=True)
    return result
