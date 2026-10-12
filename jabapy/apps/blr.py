import numpy as np
from scipy.special import expit

from ..physics import radiation as rad
from ..utils import units as u
from ..utils import constants as c


def get_Halpha_alpha_emissivity(snap, source='Pequignot_etal_1991', g_ff=1.0, d=3.0, dust=True): # for gizmo
    """Return Halpha emissivity and absorption, scattering and effective opacities.

    dust=True adds MRN dust opacities at Halpha before computing the effective opacity.
    Use DustToGasRatio_Local when available, otherwise half the metal mass fraction.
    If Dust_Temperature is available, multiply by the notebook's smooth survival factor
    1/(1+exp((T_dust-1500 K)/75 K)); without it assume the dust survives. The resulting
    dust-to-gas ratio is clipped to [0, 1]. dust=False returns gas-only coefficients.
    Dust does not change the intrinsic recombination emissivity.
    """
    metals = snap[0, 'Metallicity']
    Z = metals[:, 0]
    Y = metals[:, 1]
    X = 1 - Y - Z
    n_H = (snap.dens0/c.m_p).to('cm**-3')*X
    n_e = n_H * snap[0, 'ElectronAbundance']
    nu_Halpha = 4.5668e14  # Halpha frequency in Hz
    
    emissivity = rad.recombination_emissivity_cgs(snap.temp0.to('K').value, n_e.to('cm**-3').value, transition='HI 2-3', source=source) * u.erg / u.s / u.cm**3
    alpha_abs = rad.alpha_ff_cgs(T=snap.temp0.to('K').value, n_e=n_e.to('cm**-3').value, nu=nu_Halpha, g_ff=g_ff) / u.cm
    alpha_sca = rad.alpha_T_cgs(n_e.to('cm**-3').value) / u.cm

    # add surviving dust absorption and scattering before combining the opacities
    if dust:
        ratio = snap[0, 'DustToGasRatio_Local'] if (0, 'DustToGasRatio_Local') in snap else 0.5*Z
        if (0, 'Dust_Temperature') in snap:
            ratio = ratio*expit((1500-snap[0, 'Dust_Temperature'].to_value(u.K))/75)
        rho_dust = snap.dens0.to_value(u.g/u.cm**3)*np.clip(np.asarray(ratio), 0, 1)
        k_abs, k_sca = rad.kappa_dust_MRN_cgs(nu_Halpha, dust_to_gas=1.0)
        alpha_abs += rho_dust*k_abs/u.cm
        alpha_sca += rho_dust*k_sca/u.cm
    alpha_eff = rad.effective_extinction_optical_depth(tau_abs=alpha_abs.to_value('cm**-1'), tau_sca=alpha_sca.to_value('cm**-1'), d=d) / u.cm
    return emissivity, alpha_abs, alpha_sca, alpha_eff


def get_Halpha_line_profile(snap, x0, x1, pixel_area, emissivity=None, alpha_abs=None, alpha_eff=None,
                           spectral_width=3000.0, n_spectral=100, velocity_center=0.0, systemic_velocity=0.0,
                           tree=None, step_fraction=0.25, approx=True, N_NGB=32, workers=-1,
                           query_chunk_size=4096, particle_chunk_size=4096, log=False, eta_update_seconds=5.0,
                           thermal_broadening=True, doppler_shift=True, relativistic=True,
                           source='Pequignot_etal_1991', g_ff=1.0, d=3.0, dust=True):
    """Build GIZMO Halpha coefficients when omitted and trace the +z line and images.

    Supplied coefficients override the snapshot-derived coefficients, including dust.
    Each None is filled independently by
    get_Halpha_alpha_emissivity(snap, source=source, g_ff=g_ff, d=d, dust=dust); explicit zeros
    are preserved. emissivity is angle-integrated volume emissivity in erg/s/cm^3,
    not particle luminosity. Emergent luminosities are returned in profiles.
    Omit all three coefficients for the snapshot-derived gas+dust calculation. Set
    dust=False for gas-only defaults; supplied opacities are never modified by this flag.
    See radiation.raytrace_line_profile for transfer semantics.
    """

    # derive only missing coefficients and preserve supplied gas or dust contributions
    if emissivity is None or alpha_abs is None or alpha_eff is None:
        derived = get_Halpha_alpha_emissivity(snap, source=source, g_ff=g_ff, d=d, dust=dust)
        if emissivity is None: emissivity = derived[0]
        if alpha_abs is None: alpha_abs = derived[1]
        if alpha_eff is None: alpha_eff = derived[3]
    alpha_abs = u.Quantity(alpha_abs, u.cm**-1).to_value(u.cm**-1)
    alpha_eff = u.Quantity(alpha_eff, u.cm**-1).to_value(u.cm**-1)
    alpha = np.column_stack((np.zeros_like(alpha_abs), alpha_abs, alpha_eff))/u.cm
    return rad.raytrace_line_profile(
        x0, x1, pixel_area, emissivity, alpha, (c.c/(4.5668e14/u.s)).to(u.AA),
        pos=snap.pos0 if tree is None else None, tree=tree,
        velocity=snap.vel0 if doppler_shift else None, temperature=snap.temp0 if thermal_broadening else None,
        case_names=('pure_emission', 'absorption', 'effective'), spectral_width=spectral_width,
        n_spectral=n_spectral, velocity_center=velocity_center, systemic_velocity=systemic_velocity,
        step_fraction=step_fraction, approx=approx, N_NGB=N_NGB, workers=workers,
        query_chunk_size=query_chunk_size, particle_chunk_size=particle_chunk_size,
        log=log, eta_update_seconds=eta_update_seconds, thermal_broadening=thermal_broadening,
        doppler_shift=doppler_shift, relativistic=relativistic
    )
