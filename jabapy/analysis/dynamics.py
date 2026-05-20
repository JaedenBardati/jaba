import numpy as np

from ..utils import units as u


u.quantity_input(pos=u.kpc, vel=u.km/u.s, mass=u.Msun)
def total_angular_momentum(mass, pos, vel):
    """Calculate total angular momentum vector given a list of particles/cells."""
    angmom = (mass.value[:, np.newaxis] * np.cross(pos.value, vel.value)).sum(axis=0)
    angmom = angmom * mass.unit * pos.unit * vel.unit
    return angmom

