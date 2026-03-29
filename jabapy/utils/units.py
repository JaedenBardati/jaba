from astropy.units import * 

# alias for dimensionless unit
def_unit('1', dimensionless_unscaled)
add_enabled_units(['1'])

### --- CGS Units --- ###
# c_cgs       = 2.99792458e10      # Speed of light in vaccum (cm s^-1)                               c.c.cgs 
# h_cgs       = 6.62607554e-27     # Planck's constant (erg s)                                        c.h.cgs 
# hbar_cgs    = 1.0545726663e-27   # Planck's reduced constant (erg s)                                c.hbar.cgs 
# G_cgs       = 6.6725985e-8       # Gravitational constant (cm^3 g^-1 s^-2)                          c.G.cgs 
# q_e_cgs     = 4.8032068e-10      # Electric charge (1 esu = 1 cm^1.5 g^0.5 s^-1)                    c.e.gauss              
# m_e_cgs     = 9.109389754e-28    # Mass of electron (g)                                             c.m_e.cgs
# m_p_cgs     = 1.67262311e-24     # Mass of proton (g)                                               c.m_p.cgs                                               
# m_n_cgs     = 1.67492861e-24     # Mass of neutron (g)                                              c.m_n.cgs
# N_a_cgs     = 6.022136736e23     # Avagadro's number (1)                                            c.N_A.cgs
# k_B_cgs     = 1.38065812e-16     # Boltzmann constant (erg K^-1)                                    c.k_B.cgs
# sig_SB_cgs  = 5.6705119e-5       # Stefan-Boltzmann constant (erg cm^-2 K^-4 s^-1)                  c.sigma_sb.cgs              

# AU_cgs      = 1.496e13           # Astronomical unit (cm)
# pc_cgs      = 3.086e18           # Parsec (cm)
# ly_cgs      = 9.463e17           # Light-year (cm)
# Msun_cgs    = 1.99e33            # Solar mass (g)
# Rsun_cgs    = 6.96e10            # Solar radius (cm)
# Lsun_cgs    = 3.9e33             # Solar luminosity (erg s^-1)
# Tsun_cgs    = 5.780e3            # Solar effective temperature (K)
# M_earth_cgs = 5.976e27           # Earth mass (g)
# R_earth_cgs = 6.378e8            # Earth radius (cm)

# sigma_T_cgs  = (8.0*np.pi/3.0)*(q_e_cgs**2/(m_e_cgs*c_cgs**2))**2   # Thomson cross-section (cm^2)
