import numpy as np

from ..analysis import radiation as rad
from ..utils import units as u
from ..utils import constants as c


def get_Halpha_alpha_emissivity(snap, source='Pequignot_etal_1991', g_ff=1.0): # for gizmo
    metals = snap[0, 'Metallicity']
    Z = metals[:, 0]
    Y = metals[:, 1]
    X = 1 - Y - Z
    n_H = (snap.dens0/c.m_p).to('cm**-3')*X
    n_e = n_H * snap[0, 'ElectronAbundance']
    nu_Halpha = 4.5668e14  # Halpha frequency in Hz
    
    emissivity = rad.recombination_emissivity_cgs(snap.temp0.to('K').value, n_e.to('cm**-3').value, transition='HI 2-3', source=source) * u.erg / u.s / u.cm**3 / u.sr
    alpha_abs = rad.alpha_ff_cgs(T=snap.temp0.to('K').value, n_e=n_e.to('cm**-3').value, nu=nu_Halpha, g_ff=g_ff) / u.cm
    alpha_sca = rad.alpha_T_cgs(n_e.to('cm**-3').value) / u.cm
    alpha_eff = np.sqrt(alpha_abs * (alpha_sca + alpha_abs))
    return emissivity, alpha_abs, alpha_sca, alpha_eff