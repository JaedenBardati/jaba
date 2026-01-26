import numpy as np
import scipy as sp

import ..units as u
import ..constants as c


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

def effective_recombination_coefficients_cgs(T, z=1, transition='HI 2-3', source='Pequignot_etal_1991'):
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
    else:
        raise NotImplementedError('Only Pequignot et al 1991 is supported for now.')

def recombination_emissivity_cgs(T, n_e, n_i=None, z=1, transition='HI 2-3'):
    # Returns emissivity of entered recombination lines.
    # Taken from equation ...
    #   T          = temperature of free electrons (K)
    #   n_e        = free electron number density (cm^-3)
    #   n_i        = ion number density (cm^-3) - Default: Neutral plasma (n_i=n_e)
    #   z          = ionic charge (1) - Default: Hydrogen (z=1)
    #   transition = the transition that produces the line -- Default: H alpha (transition='HI 2-3')
    # Returns in erg/s/cm^3 (cgs) units. Multiply by the emitting surface for luminosity.
    if n_i is None:
        n_i = n_e
    alpha_eff = effective_recombination_coefficients_cgs(T, transition=transition, )
    E_trans = c.h.cgs * c.c.cgs/transition_wavelength_cgs[transition]
    return 4*np.pi * alpha_eff * n_e * n_i * E_trans


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
        nu = c.c.cgs/x
        nu0 = c.c.cgs/x0
    if spec_var == 'energy':
        nu = x/c.h.cgs
        nu0 = x0/c.h.cgs

    g = nu / c.c.cgs # non-relativistic for now
    
    sigma_G = 0 if fwhm_G is None else sigma_from_fwhm(fwhm_G)
    if sigma_v is not None:
        sigma_G = np.sqrt(sigma_G**2 + (sigma_v * g)**2)
    if temp is not None:
        sigma_G = np.sqrt(sigma_G**2 + np.abs(c.k_B.cgs*temp/part_mass)*(g)**2)

    if fwhm_L is None:
        fwhm_L = 0
    if gamma_ul is not None:
        fwhm_L = gamma_ul/(2*np.pi)
    
    return sp.special.voigt_profile(_x, sigma_G, fwhm_L)


### --- Extinction opacity --- ###

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
    coeff = 4 * c.e.gauss**6/(3*c.m_e.cgs * c.h.cgs * c.c.cgs) * np.sqrt(2*np.pi/(3*c.k_B.cgs*c.m_e.cgs))
    return coeff * T**-0.5 * Z**2 * n_e * n_i * nu**-3 * (1 - np.exp(-c.h.cgs*nu/(c.k_B.cgs*T))) * g_ff

def alpha_thomson_scattering(n_e):
    # Returns the extinction coefficient (alpha_nu=n*sigma). Independent of frequency.
    #   n_e  = free electron number density (cm^-3)
    return c.sigma_T.cgs * n_e

def effective_extinction_optical_depth(tau_abs, tau_sca):
    # Returns the effective optical depth given a scattering and absorption depth.
    #   tau_abs  = absorption optical depth (1)
    #   tau_sca  = scattering optical depth (1)
    return np.sqrt(tau_abs*(tau_abs + tau_sca))


### --- Aliases --- ###
alpha_free_free_cgs = alpha_thermal_bremsstrahlung_cgs
alpha_ff_cgs = alpha_thermal_bremsstrahlung_cgs
alpha_electron_scattering = alpha_thomson_scattering
alpha_es_cgs = alpha_thomson_scattering
alpha_T_cgs = alpha_thomson_scattering
tau_eff = effective_extinction_optical_depth




