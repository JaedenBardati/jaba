import numpy as np
from scipy.integrate import quad

import ..units as u
import ..constants as c


def comoving_distance(z, z0=0.0, h=c.DEFAULT_h, Ode0=c.DEFAULT_Ode0, Om0=c.DEFAULT_Om0, Or0=c.DEFAULT_Or0, Ok0=c.DEFAULT_Ok0, tol=1e-3):
    '''Returns the comoving distance at the given redshift z in Mpc. This integrates to find answer, so do not call too many times.'''
    d_H = 2997.92*u.Mpc/h
    E = lambda z: 1/np.sqrt(Ode0 + Om0*(1+z)**3 + Or0*(1+z)**4 + Ok0*(1+z)**2)
    result, error = quad(E, z0, z)
    if error > tol*result:
        raise Exception("Error ({}) is too high compared to the inferred distance")
    return d_H*result

def transverse_comoving_distance(z, z0=0.0, h=c.DEFAULT_h, Ode0=c.DEFAULT_Ode0, Om0=c.DEFAULT_Om0, Or0=c.DEFAULT_Or0, Ok0=c.DEFAULT_Ok0):
    '''Returns the transverse comoving distance at the given redshift z in Mpc. This integrates to find answer, so do not call too many times.'''
    d_C = comoving_distance(z, z0=z0, h=h, Ode0=Ode0, Om0=Om0, Or0=Or0, Ok0=Ok0)
    if Ok0 == 0.0:
        return d_C
    d_H = 2997.92*u.Mpc/h # in Mpc
    root_Ok0 = np.sqrt(np.abs(Ok0))
    if Ok0 > 0.0:
        curve_factor = d_H/root_Ok0 * np.sinh(root_Ok0 * (d_C/d_H).to(u.dimensionless_unscaled))
    else:
        curve_factor = d_H/root_Ok0 * np.sin(root_Ok0 * (d_C/d_H).to(u.dimensionless_unscaled))
    return curve_factor*d_C

def luminosity_distance(z, z0=0.0, h=c.DEFAULT_h, Ode0=c.DEFAULT_Ode0, Om0=c.DEFAULT_Om0, Or0=c.DEFAULT_Or0, Ok0=c.DEFAULT_Ok0):
    '''Returns the luminosity distance at the given redshift z in Mpc. This integrates to find answer, so do not call too many times.'''
    d_M = transverse_comoving_distance(z, z0=z0, h=h, Ode0=Ode0, Om0=Om0, Or0=Or0, Ok0=Ok0)
    return d_M*(1+z)

def angular_diameter_distance(z, z0=0.0, h=c.DEFAULT_h, Ode0=c.DEFAULT_Ode0, Om0=c.DEFAULT_Om0, Or0=c.DEFAULT_Or0, Ok0=c.DEFAULT_Ok0):
    '''Returns the angular diameter distance at the given redshift z in Mpc. This integrates to find answer, so do not call too many times.'''
    d_M = transverse_comoving_distance(z, z0=z0, h=h, Ode0=Ode0, Om0=Om0, Or0=Or0, Ok0=Ok0)
    return d_M/(1+z)

def angular_scale(z, z0=0.0, h=c.DEFAULT_h, Ode0=c.DEFAULT_Ode0, Om0=c.DEFAULT_Om0, Or0=c.DEFAULT_Or0, Ok0=c.DEFAULT_Ok0):
    '''Returns the angular scale at the given redshift z in kpc/arcsec. This integrates to find answer, so do not call too many times.'''
    d_A = angular_diameter_distance(z, z0=z0, h=h, Ode0=Ode0, Om0=Om0, Or0=Or0, Ok0=Ok0)
    return (d_A/(206265*u.arcsec)).to('kpc/arcsec')

# aliases 
dC = comoving_distance
dM = transverse_comoving_distance
dL = luminosity_distance
dA = angular_diameter_distance