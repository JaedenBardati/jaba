import warnings

import numpy as np
import scipy as sp

from ..utils import constants as c
from ..utils import units as u


### --- Recombination Lines --- ###

transition_wavelength_cgs = {
    'HI 2-3': 656.3e-7,           #H alpha
    'HI 2-4': 486.1e-7,           #H beta
    'HeII 3-4': 468.6e-7,
    'HeII 3-5': 320.3e-7,
    'HeII 4-5': 1012.4e-7,
    'HeII 4-6': 656.0e-7,
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


### --- Absorption opacities --- ###

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



### --- Scattering opacities --- ###

def alpha_thomson_scattering(n_e):
    # Returns the extinction coefficient (alpha_nu=n*sigma). Independent of frequency.
    #   n_e  = free electron number density (cm^-3)
    return c.sigma_T.cgs.value * n_e

def effective_extinction_optical_depth(tau_abs, tau_sca):
    # Returns the effective optical depth given a scattering and absorption depth.
    #   tau_abs  = absorption optical depth (1)
    #   tau_sca  = scattering optical depth (1)
    return np.sqrt(tau_abs*(tau_abs + tau_sca))

def Klein_Nishina_correction_factor(photon_energy):
    # Accurate to ~0.01%, sets to 1 for lambda >~ 25 nm.
    eps = np.abs((photon_energy.astype(np.float64)/(c.m_e*c.c**2)).to(u.dimensionless_unscaled).value)
    regular_thomson_range = eps < 1e-4
    return regular_thomson_range + ~regular_thomson_range*(3.0/4.0 * ((1+eps)/eps**3 * (2*eps*(1+eps)/(1+2*eps) - np.log(1+2*eps)) + np.log(1+2*eps)/(2*eps) - (1+3*eps)/(1+2*eps)**2))


### --- GIZMO opacities --- ###

def GIZMO_FREEFREE_kappa(snap, gamma=5.0/3.0, idx=0):
    # kappa in the GIZMO RT_FREEFREE band, as caculated by the code
    U_TO_TEMP_UNITS = c.m_p/c.k_B
    T_eff = (0.59*(gamma-1.0)*snap[idx, 'InternalEnergy'] * U_TO_TEMP_UNITS).to(u.K).value
    kappa_abs = (0.25 + 1e30*snap[idx, 'Density'].to(u.g/u.cm**3).value*pow(T_eff, -3.5))
    return kappa_abs * (u.cm**2/u.g)



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


def planck_function_lam(lam: u.micron, T: u.K):
    # Returns the Planck function B_lambda(T).
    T = T.to(u.K).value
    lam = lam.to(u.cm).value
    B_lam = (2*c.h.cgs.value*c.c.cgs.value**2)/lam**5 * 1/(np.exp(c.h.cgs.value*c.c.cgs.value/(lam*c.k_B.cgs.value*T)) - 1)
    return B_lam * u.erg / (u.s * u.cm**2 * u.sr * u.cm)




### --- Aliases --- ###
alpha_free_free_cgs = alpha_thermal_bremsstrahlung_cgs
alpha_ff_cgs = alpha_thermal_bremsstrahlung_cgs
alpha_electron_scattering = alpha_thomson_scattering
alpha_es_cgs = alpha_thomson_scattering
alpha_T_cgs = alpha_thomson_scattering
tau_eff = effective_extinction_optical_depth

# Backward-compatible aliases used by existing notebooks - should be removed eventually
h_cgs = c.h.cgs.value
c_cgs = c.c.cgs.value