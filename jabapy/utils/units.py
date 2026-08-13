from fractions import Fraction
import re

from astropy.units import * 
from astropy.constants import c

# custom units
ld = def_unit('ld', 1.0 * day * c)
t_H = def_unit('t_H', 14.5 * Gyr, format={'latex': r'$t_\text{H}$'}) # ~ 1/H_0
G_cgs = def_unit('G_cgs', (g ** Fraction(1, 2)) * (cm ** Fraction(-1, 2)) * (s ** -1), format={'latex': r'$G_\text{cgs}$'})
add_enabled_units([
    ld,
    t_H,
    G_cgs,
])

# custom unit aliases
m_p = M_p
Msol = M_sun
solMass = M_sun
light_day = ld
hubble_time = t_H
set_enabled_aliases({
    '1': dimensionless_unscaled,
    'm_p': M_p, 
    'Msol': Msol, 
    'solMass': Msol,
    'light_day': ld,
    'hubble_time': t_H,
})


#### unit checks and conversion convience functions ####
# THERE ARE FOUR TYPES OF INPUTS TO CONSIDER: UNITLESS, UNIT-STR, UNIT-OBJ, QUANTITY.
#
# TERMINOLOGY: UNIT-STR is the string representation of a unit, UNIT-OBJ is the actual unit object (e.g. from astropy),
#              UNIT-ONLY means it is a unit, but has no value. QUANTITY means it has both a unit and a value. UNIT-LIKE means either of the above.
#              UNIT is the unit of a quantity, and VALUE is the unitless part of a quantity.
#
# UNITLESS does not have .unit or .value or .to       - and rightfully does not, but may want default (assumed) unit functionality
# UNIT-STR does not have .unit or .value or .to       - should have full functionality
# UNIT-OBJ does not have .unit or .value, but has .to - should have full functionality
# QUANTITY has .unit, .value, and .to                 - and rightfully does
# I implement all these functionalities in get_unit, get_value and to_unit below. 
# 
def is_unit_str(x):  # e.g. 'Msun/pc**3' or 'm/s'
    if not isinstance(x, str):
        return False
    try:
        Unit(x)
        return True
    except:
        return False
    
def is_unit_obj(x):  # e.g. u.pc or u.Msun/u.pc**3
    return isinstance(x, UnitBase)

def is_quantity(x):  # e.g. 1.0*u.Msun or [1.0, 2.0]*u.pc, but NOT u.Msun or 'Msun/pc**3'
    return isinstance(x, Quantity) 

def is_unit_only(x): # either a unit object or a unit string
    return is_unit_obj(x) or is_unit_str(x)

def is_unit_like(x): # has any form of unit information (all above cases)
    return is_quantity(x) or is_unit_obj(x) or is_unit_str(x)

def is_unitless(x): # has no unit information (e.g. 1.0 or [1.0, 2.0])
    return not is_quantity(x) and not is_unit_obj(x) and not is_unit_str(x)


def get_unit(x, default_unit=None):
    """
    Returns the unit of a quantity, or the unit itself if it's already a unit. 
    If unitless, returns default unit if provided, otherwise errors.
    Always returns a unit object.
    """
    if isinstance(x, Quantity):
        return x.unit
    if isinstance(x, UnitBase):
        return x
    if is_unit_str(x):
        return Unit(x)
    #unitless
    if default_unit is not None:
        return Unit(default_unit)
    raise ValueError("Input must be a quantity or a unit.")

def get_value(x, default_unit=None):
    """
    Returns the unitless value of a quantity, errors otherwise. 
    If unitless, returns value if a default unit is provided, otherwise errors.
    Always returns a unitless array or scalar.
    """
    if isinstance(x, Quantity):
        return x.value
    if isinstance(x, UnitBase) or is_unit_str(x):
        return 1.0
    #unitless
    if default_unit is not None:
        return x
    raise ValueError("Input must be a quantity or a unit.")

def to_unit(x, unit, default_unit=None):
    """
    Converts a quantity to a specified unit, or returns the unit itself if it's already a unit. 
    If unitless, returns default if provided, errors otherwise.
    Always returns a quantity.
    """
    if isinstance(x, Quantity):
        return x.to(unit)
    if isinstance(x, UnitBase):
        return (1.0 * x).to(unit)
    if is_unit_str(x):
        return Unit(x).to(unit)
    # unitless
    if default_unit is not None:
        return (x*Unit(default_unit)).to(unit)
    raise ValueError("Input must be a quantity or a unit.")


# above functions always return a 
# if unitless, then if:
#   default = None -> errors 
#   default = u.pc -> assumes input is in units of pc, returns quantity in units of pc
#   default = 1 -> assumes input is unitless 

def get_value_in_unit(x, unit, default_unit=None):
    """
    Returns the unitless value of a quantity in a specified unit.
    If unitless, assumes default unit if provided, errors otherwise.
    Always returns a unitless array or scalar.
    """
    if isinstance(x, Quantity):
        return (x.to(unit)).value
    if isinstance(x, UnitBase):
        return (x.to(unit)).value
    if is_unit_str(x):
        return (Unit(x).to(unit)).value
    #unitless
    if default_unit is not None:
        if default_unit == unit:
            return x
        return x * (Unit(default_unit).to(unit)).value
    raise ValueError("Input must be a quantity or a unit.")



#### from astropy to other unit systems ####
def to_pynbody(x, _debug=False):
    import pynbody
    if isinstance(x, Quantity):
        unit_str = x.unit.to_string()
        value = x.value
    elif isinstance(x, UnitBase):
        unit_str = x.to_string()
        value = None
    elif is_unit_str(x):
        unit_str = Unit(x).to_string()
        value = None
    else:
        raise ValueError("Input must be a unit or have a unit attribute.")
        
    if unit_str == '' or unit_str == '1' or unit_str == 'dimensionless' or unit_str == 'dimensionless_unscaled': # handle unitless case
        return value*pynbody.units.Unit('1') if value is not None else pynbody.units.Unit('1')

    # expand this as needed (also remember from_pynbody if relevant)
    replacements = {
        'Msun': 'Msol',
        'solMass': 'Msol',
        'M_p': 'm_p',
        'Angstrom': (0.1, 'nm'),
        # ...
    }

    # convert to pynbody format (e.g. m4 -> m**4 and m/s -> m s**-1)
    new_unit_str = ''
    prefactor = 1.0
    after_slash = False # assume everything after the first slash will flip exponent, ignore repeated slashes (e.g. m/s/s will be treated as m/s^2)
    ps = re.findall(r'[A-Za-z]+(?:_[A-Za-z]+)*|\d+|[()/]', unit_str)
    if _debug:
        print(f'Original unit string: "{unit_str}"')
        print(f'Parsed components: {ps}')
    i = 0
    while i < len(ps):
        p = ps[i]
        if '/' in p: # flip exponent after first slash
            if _debug:
                print(f'Encountered "/", setting after_slash to True.')
            after_slash = True
            i += 1
            continue
        if '(' in p or ')' in p: # ignore parentheses entirely
            i += 1
            continue
        if p.isnumeric() and p != '1': # should have handled exponents earlier
            raise ValueError(f"Unexpected numeric component '{p}' in unit string. Exponents should be handled as part of the preceding unit, e.g. 'm2' instead of 'm 2'.")
        if p == '1': # ignore unitless part
            i += 1
            continue

        # otherwise, its likely a unit
        nextp = ps[i+1] if i < len(ps)-1 else None # get the next component to check for exponent
        if nextp is not None and nextp.isnumeric():
            future_power = int(nextp)
            i+=1 # skip next component since it's an exponent (NOTE you should not use i after this part of code)
        else:
            future_power = 1
        future_power = -future_power if after_slash else future_power

        if p in replacements: # replace unit name if needed
            replacement = replacements[p]
            if isinstance(replacement, str):
                p = replacement
            else:
                prefactor *= replacement[0]**future_power
                p = replacement[1]
        
        if future_power != 1:
            p += f'**{future_power}'

        p = ' ' + p # preappend space
        
        if _debug:
            print(f'prefactor: {prefactor}, unit_str: "{new_unit_str}", p: "{p}"')
        
        new_unit_str += p
        i += 1
    unit_str = new_unit_str
    if prefactor != 1.0: # preappend any constant factor
        unit_str = f'{prefactor} ' + unit_str
    unit_str = unit_str.strip()

    if _debug:
        print(f'Final unit string: "{unit_str}"')

    new_qty = value*pynbody.units.Unit(unit_str) if value is not None else pynbody.units.Unit(unit_str)
    return new_qty


def from_pynbody(x):
    import pynbody
    if isinstance(x, str):
        unit_str = Unit(x).to_string()
        value = None
    elif isinstance(x, pynbody.units.UnitBase):
        unit_str = str(x)
        value = None
    elif isinstance(x, pynbody.array.SimArray):
        unit_str = str(x.units)
        value = x.value
    else:
        raise ValueError("Input must be a pynbody Unit or SimArray.")
    
    # expand this as needed (also remember to_pynbody)
    replacements = {
        'Msol': 'Msun',
        'm_p': 'M_p',
        # ...
    }
    for old, new in replacements.items():
        if old in unit_str:
            unit_str = unit_str.replace(old, new)

    return value*Unit(unit_str) if value is not None else Unit(unit_str)


def to_yt(x):
    import yt
    if is_unit_only(x):
        unit_str = x.to_string()
        value = None
    elif is_quantity(x):
        unit_str = x.unit.to_string()
        value = x.value
    else:
        raise ValueError("Input must be a unit or have a unit attribute.")
    
    # replace this as needed (also remember from_yt)
    # ...

    return value*yt.units.Unit(unit_str) if value is not None else yt.units.Unit(unit_str)


def from_yt(x):
    import yt
    if isinstance(x, yt.units.Unit):
        unit_str = str(x)
        value = None
    elif isinstance(x, yt.units.ytArray):
        unit_str = str(x.units)
        value = x.value
    else:
        raise ValueError("Input must be a yt Unit or ytArray.")
    
    # replace this as needed (also remember to_yt)
    # ...

    new_qty = value*Unit(unit_str) if value is not None else Unit(unit_str)
    return new_qty



# Nice units to use for e.g. quick plotting/estimates
def to_nice_units(x, default_unit=None):
    """
    Returns the value in a "nice" unit for the given quantity x. 
    This is just something convenient for e.g.  quick plotting or estimates.
    Requires unit to be defined.
    """
    x_unit = get_unit(x, default_unit=default_unit)
    x_val = get_value(x, default_unit=default_unit)
    x = abs(x_val) * x_unit

    if x_unit.is_equivalent(cm):
        if x < 1 * m:
            # assume it's a wavelength
            if x >= 1 * mm:
                return x.to(mm)
            elif x >= 1 * um:
                return x.to(um)
            return x.to(AA)
        else:
            # assume it's a distance
            if x < 0.01 * AU:
                return x.to(cm)
            if x < 0.01 * AU:
                return x.to(cm)
            if x < 1 * ld:
                return x.to(AU)
            if x < 0.1*pc:
                return x.to(ld)
            if x < 100*pc:
                return x.to(pc)
            if x < 100*kpc:
                return x.to(kpc)
            if x < 100*Mpc:
                return x.to(Mpc)
            return x.to(Gpc)
    if x_unit.is_equivalent(g):
        if x > 1e-10 * Msol:
            return x.to(Msol)
        if x > 1e-10 * g:
            return x.to(g)
        return x.to(m_p)
    if x_unit.is_equivalent(s):
        if 1e-1 < x < 1e1 * ns:
            return x.to(ns)
        if x < 1 * hour:
            return x.to(s)
        if x < 1 * day:
            return x.to(hour)
        if x < 1 * yr:
            return x.to(day)
        if x < 1 * Myr:
            return x.to(yr)
        if x < 1 * Gyr:
            return x.to(Myr)
        if x < 10 * Gyr:
            return x.to(Gyr)
        return x.to(t_H)
    
    return x  # if no nice unit found, return original quantity



#### GIZMO code units, as set in allvars.h in GIZMO code ####
GRAVITY_G_CGS = 6.672e-8
SOLAR_MASS_CGS = 1.989e33
SOLAR_LUM_CGS = 3.826e33
SOLAR_RADIUS_CGS = 6.957e10
BOLTZMANN_CGS = 1.38066e-16
C_LIGHT_CGS = 2.9979e10
PROTONMASS_CGS = 1.6726e-24
ELECTRONMASS_CGS = 9.10953e-28
THOMPSON_CX_CGS = 6.65245e-25
ELECTRONCHARGE_CGS = 4.8032e-10
SECONDS_PER_YEAR = 3.155e7
HUBBLE_H100_CGS = 3.2407789e-18
ELECTRONVOLT_IN_ERGS = 1.60217733e-12
HABING_FLUX_CGS = 1.6e-3
DRAINE_FLUX_CGS = 1.7 * HABING_FLUX_CGS


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

