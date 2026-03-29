
import os, warnings

from .io import hdf5
from .utils import units as u

import numpy as np # pyright: ignore[reportMissingImports]

########################################################################################################################################################################
#####################                                                    LOAD IN GIZMO                                                             #####################
########################################################################################################################################################################


class HDF5_Snapshot:
    """
    A general HDF5 snapshot file loader. 

    General structure is:
        Group: Header
            -> Attributes: Simulation/Snapshot Metadata
        Group: Particle Type
            -> Dataset: Property Array
            -> Dataset: Property Array 2
            -> ...
        Group: Particle Type 2
            -> Dataset: Property Array
            -> Dataset: Property Array 2
            -> ...
        ...

    Notes:
      Grabs each dataset individually, rather than all at once. 
      Stores datasets and attributes for future reference in a private class variable.
      Ignores any groups not named the Header name below or in the supported particle types. 
      Ignores any attributes not in the header group. 
      Ignores any datasets not in the particle type groups.
    """
    _HEADER_NAME = 'Header'
    _SUPPORTED_PARTICLES_TYPES = ['PartType0','PartType1','PartType2','PartType3','PartType4','PartType5',]
    _KNOWN_GROUPS = [_HEADER_NAME,] + _SUPPORTED_PARTICLES_TYPES

    def __init__(self, filepath):
        self.filepath = filepath
        self.name = os.path.splitext(os.path.basename(filepath))[0]

    @property
    def groups(self):
        if not hasattr(self, '_groups'):
            self._groups = hdf5.get_group_names(self.filepath)
            for g in self._groups:
                if g not in self._KNOWN_GROUPS:
                    warnings.warn('Unknown group "{}" found. Please update quickchecksim to support it. Ignoring for now...'.format(g))
        return self._groups

    @property
    def metadata(self):
        if not hasattr(self, '_attributes'):
            self._attributes = hdf5.get_attributes(self.filepath, self._HEADER_NAME)
        return self._attributes
    
    @property
    def particle_types(self):
        if not hasattr(self, '_particles_types'):
            self._particles_types = [g for g in self.groups if g in self._SUPPORTED_PARTICLES_TYPES]
        return self._particles_types

    def _dataset_names(self, particle_type):
        if particle_type not in self.particle_types:
            raise KeyError('Particle type "{}" not recognized. Use "particle_types" attribute for a list.'.format(particle_type))
        if not hasattr(self, '_dataset_names_' + particle_type):
            _dataset_names = hdf5.get_dataset_names(self.filepath, particle_type)
            setattr(self, '_dataset_names_' + particle_type, _dataset_names)
        return getattr(self, '_dataset_names_' + particle_type)

    def _get_dataset(self, particle_type, dataset_name, nosave=False):
        _attr = '_dataset_' + particle_type + '_' + dataset_name
        if nosave or not hasattr(self, _attr):
            if particle_type not in self.particle_types:
                raise KeyError('Particle type "{}" not recognized. Use "particle_types" attribute for a list.'.format(particle_type))
            if dataset_name not in self._dataset_names(particle_type):
                raise KeyError('Data set {} for particle type "{}" not recognized.'.format(dataset_name, particle_type))
            dataset_path = particle_type + '/' + dataset_name
            data = hdf5.get_dataset(self.filepath, dataset_path)
            if nosave:
                return data
            setattr(self, _attr, data)
        return getattr(self, _attr)

    def _resolve_particle_type(self, key):
        # get particle type from index
        if isinstance(key, int):
            if key < len(self._SUPPORTED_PARTICLES_TYPES):
                key = self._SUPPORTED_PARTICLES_TYPES[key]
            else:
                raise KeyError('Unrecognized key index "{}". Must be an index to the list of supported particle types (here, {})'.format(self._SUPPORTED_PARTICLES_TYPES))
        return key

    def __getitem__(self, key):
        key = self._resolve_particle_type(key)
        if key in self.particle_types:
            return self._dataset_names(key)
        elif isinstance(key, tuple) and len(key) == 2:
            if isinstance(key[0], int):
                key = (self._resolve_particle_type(key[0]), key[1])
            return self._get_dataset(key[0], key[1])
        else:
            raise KeyError('Unrecognized key "{}". Must be a 2-tuple where the first instance is a particle type and the second is a dataset entry. Alternatively, you can retrieve the dataset names by just passing the particle type alone'.format(key))
        
    


class GIZMO_Snapshot(HDF5_Snapshot):
    """
    Class for GIZMO specific snapshots. Uses metadata to infer code units to CGS conversions.
    Uses GIZMO standard formatting: http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html
    YOU MUST DOUBLE CHECK THESE UNITS! Add new particle data names to _DATASET_UNIT_DICT.
    """

    FORCE_RESET_UNITS = False # use this on runtime to force recalculation of units

    _DATASET_UNIT_DICT = {  # add a 
        'Acceleration': ('acceleration', u.cm/u.s**2),
        'Coordinates': ('length', u.cm),
        'Density': ('density', u.g/u.cm**3),
        'DensityGradient': ('density_grad', u.g/u.cm**4),
        'DustToGasRatio_Local': (None, u.dimensionless_unscaled),
        'Dust_Temperature': (None, u.K),
        'EddingtonTensor': (None, u.dimensionless_unscaled),
        'ElectronAbundance': (None, u.dimensionless_unscaled),
        'HydroAcceleration': ('acceleration', u.cm/u.s**2),
        'HII': (None, u.dimensionless_unscaled),
        'IRBand_Radiation_Temperature': (None, u.K),
        'InternalEnergy': (None, (u.cm/u.s)**2),
        'MagneticField': ('magnetic_field', u.G),
        'Masses': ('mass', u.g),
        'Metallicity': (None, u.dimensionless_unscaled),
        'MolecularMassFraction': (None, u.dimensionless_unscaled),
        'NeutralHydrogenAbundance': (None, u.dimensionless_unscaled),
        'ParticleChildIDsNumber': (None, u.dimensionless_unscaled),
        'ParticleIDGenerationNumber': (None, u.dimensionless_unscaled),
        'ParticleIDs': (None, u.dimensionless_unscaled),
        'PhotonEnergy': ('energy', (u.g*u.cm/u.s)**2),
        'PhotonFluxDensity': ('flux_density', u.cm**-2*u.s**-1),
        'PhotonOpacity': ('opacity', u.cm**2/u.g),
        'Potential': ('internal_energy', (u.cm/u.s)**2),
        'Pressure': ('pressure', (u.cm/u.s)**2/u.cm**3),
        'RadiativeAcceleration': ('acceleration', u.cm/u.s**2),
        'SmoothingLength': ('length', u.cm),
        'SoundSpeed': ('velocity', u.cm/u.s),
        'StarFormationRate': (None, u.Msun/u.yr),  # fixed output units of Msun/yr
        'StellarFormationTime': ('time', u.s),
        'Temperature': (None, u.K),
        'Velocities': ('velocity', u.cm/u.s),
        'VelocityGradient': ('velocity_grad', u.s**-1),
    }

    @property
    def cosmological(self):
        """Note if cosmological, time is a scale factor not physical time."""
        if not hasattr(self, '_cosmological'):
            if 'ComovingIntegrationOn' not in self.metadata:
                warnings.warn('Cannot determine if it is a cosmological simulation from metadata - assuming non-cosmological.') 
                self._cosmological = False
            else:
                self._cosmological = bool(int(self.metadata['ComovingIntegrationOn']))
        return self._cosmological
    
    @property
    def hubble_param(self):
        if not hasattr(self, '_HubbleParam'):
            if 'HubbleParam' not in self.metadata:
                warnings.warn('Cannot determine Hubble parameter from metadata - assuming h = 1.')
                self._HubbleParam = 1
            else:
                self._HubbleParam = float(self.metadata['HubbleParam'])
        return self._HubbleParam
    
    @property
    def redshift(self):
        if not hasattr(self, '_redshift'):
            if 'Redshift' not in self.metadata:
                warnings.warn('Cannot determine redshift from metadata - assuming z = 0.')
                self._redshift = 0
            else:
                self._redshift = float(self.metadata['Redshift'])
        return self._redshift

    @property
    def scale_factor(self):
        if not hasattr(self, '_scale_factor'):
            self._scale_factor = 1/(1+self.redshift)
        return self._scale_factor
    
    @property
    def box_size(self):
        if not hasattr(self, '_BoxSize'):
            if 'BoxSize' not in self.metadata:              
                warnings.warn('Cannot determine redshift from metadata - assuming box = 10000 code units.')
                self._BoxSize = 10000.0
            else:
                self._BoxSize = float(self.metadata['BoxSize']) * self.unit_system['length'] * u.cm
        return self._BoxSize

    @property
    def omega_matter(self):
        if not hasattr(self, '_omega_matter'):
            if 'Omega_Matter' not in self.metadata:
                if self.cosmological:
                    warnings.warn('Cannot determine omega matter from metadata - assuming 0.27.')
                    self._omega_matter = 0.27
                else:
                    warnings.warn('Cannot determine omega matter from metadata - assuming 0.')
                    self._omega_matter = 0.0
            else:
                self._omega_matter = float(self.metadata['Omega_Matter'])
        return self._omega_matter
    
    @property
    def omega_lambda(self):
        if not hasattr(self, '_omega_lambda'):
            if 'Omega_Lambda' not in self.metadata:
                if self.cosmological:
                    warnings.warn('Cannot determine omega lambda from metadata - assuming 0.73.')
                    self._omega_lambda = 0.27
                else:
                    warnings.warn('Cannot determine omega lambda from metadata - assuming 0.')
                    self._omega_lambda = 0.0
            else:
                self._omega_lambda = float(self.metadata['Omega_Lambda'])
        return self._omega_lambda

    @property
    def omega_baryon(self):
        if not hasattr(self, '_omega_baryon'):
            if 'Omega_Baryon' not in self.metadata:
                if self.cosmological:
                    warnings.warn('Cannot determine omega baryon from metadata - assuming 0.044.')
                    self._omega_baryon = 0.044
                else:
                    warnings.warn('Cannot determine omega baryon from metadata - assuming 0.')
                    self._omega_baryon = 0.0
            else:
                self._omega_baryon = float(self.metadata['Omega_Baryon'])
        return self._omega_baryon

    @property
    def omega_radiation(self):
        if not hasattr(self, '_omega_radiation'):
            if 'OmegaRadiation' not in self.metadata:
                warnings.warn('Cannot determine omega baryon from metadata - assuming 0.')
                self._omega_radiation = 0.0
            else:
                self._omega_radiation = float(self.metadata['OmegaRadiation'])
        return self._omega_radiation

    def scale_factor_to_time(self, scale_factor):
        raise NotImplementedError('Need to add cosmological integration to program.')

    @property
    def unit_system(self):
        if not hasattr(self, '_cgs_unit_system') or self.FORCE_RESET_UNITS:
            MASS_cgs = None
            for m in ['UnitMass_In_CGS', 'UnitMass_in_g']:
                if m in self.metadata:
                    if MASS_cgs is not None:
                        raise Exception('Ambiguous mass units in metadata.')
                    MASS_cgs = self.metadata[m]
            LENGTH_cgs = None
            for l in ['UnitLength_In_CGS', 'UnitLength_in_cm']:
                if l in self.metadata:
                    if LENGTH_cgs is not None:
                        raise Exception('Ambiguous length units in metadata.')
                    LENGTH_cgs = self.metadata[l]
            VELOCITY_cgs = None
            for v in ['UnitVelocity_In_CGS', 'UnitVelocity_in_cm_per_s']:
                if v in self.metadata:
                    if VELOCITY_cgs is not None:
                        raise Exception('Ambiguous velocity units in metadata.')
                    VELOCITY_cgs = self.metadata[v]
            if MASS_cgs is None or LENGTH_cgs is None or VELOCITY_cgs is None:
                raise Exception('Could not find unit system in metadata.')
            
            MAG_cgs = None
            for v in []: # NOT 'Internal_UnitB_In_Gauss', 'UnitMagneticField_in_gauss' --> must assume snapshots return magnetic field in gauss
                if v in self.metadata:
                    if MAG_cgs is not None:
                        raise Exception('Ambiguous magnetic field units in metadata.')
                    MAG_cgs = self.metadata[v]

            cosmo = self.cosmological
            h = self.hubble_param
            MASS_code = MASS_cgs/h
            LENGTH_code = LENGTH_cgs/h
            VELOCITY_code = VELOCITY_cgs
            INTERNAL_ENERGY_code = VELOCITY_code**2
            DENSITY_code = MASS_code/(LENGTH_code**3)
            MAGNETIC_FIELD_code = MAG_cgs
            DIVERGENCE_DAMPING_FIELD_code = MAGNETIC_FIELD_code*VELOCITY_code if MAGNETIC_FIELD_code is not None else None
            
            if not cosmo:
                TIME_code = LENGTH_code/VELOCITY_code
                self._cgs_unit_system = {
                    'mass': MASS_code, # mass factor to convert from code units to g
                    'length': LENGTH_code, # length factor to convert from code units to cm
                    'velocity': VELOCITY_code, # velocity factor to convert from code units to cm/s
                    'time': TIME_code,
                    'internal_energy': INTERNAL_ENERGY_code,
                    'density': DENSITY_code,
                    'magnetic_field': MAGNETIC_FIELD_code, # magnetic field factor to convert from code units to Gauss (COULD BE NONE)
                    'div_damping_field': DIVERGENCE_DAMPING_FIELD_code, # (COULD BE NONE)
                }
            else:
                # time unit is a scale factor
                a_scale = self.scale_factor
                MASS_physical = MASS_code
                LENGTH_physical = LENGTH_code * a_scale
                VELOCITY_physical = VELOCITY_code * np.sqrt(a_scale)
                INTERNAL_ENERGY_physical = INTERNAL_ENERGY_code
                DENSITY_physical = DENSITY_code / a_scale**3
                MAGNETIC_FIELD_physical = MAGNETIC_FIELD_code
                DIVERGENCE_DAMPING_FIELD_physical = DIVERGENCE_DAMPING_FIELD_code
                self._cgs_unit_system = {
                    'mass': MASS_physical, # mass factor to convert from code units to g
                    'length': LENGTH_physical, # length factor to convert from code units to cm
                    'velocity': VELOCITY_physical, # velocity factor to convert from code units to cm/s
                    'time': None,   # None indicates that it is a scale factor
                    'internal_energy': INTERNAL_ENERGY_physical,
                    'density': DENSITY_physical,
                    'magnetic_field': MAGNETIC_FIELD_physical, # magnetic field factor to convert from code units to Gauss (COULD BE NONE)
                    'div_damping_field': DIVERGENCE_DAMPING_FIELD_physical, # (COULD BE NONE)
                }

            # derived units (I think these are correct? (but double check if used)
            self._cgs_unit_system['acceleration'] = self._cgs_unit_system['velocity']**2/self._cgs_unit_system['length']
            self._cgs_unit_system['density_grad'] = self._cgs_unit_system['density']/self._cgs_unit_system['length']
            self._cgs_unit_system['flux_density'] = self._cgs_unit_system['velocity']/self._cgs_unit_system['length']**3
            self._cgs_unit_system['opacity'] = self._cgs_unit_system['length']*self._cgs_unit_system['length']/self._cgs_unit_system['mass']
            self._cgs_unit_system['pressure'] = self._cgs_unit_system['internal_energy']/(self._cgs_unit_system['length']**3)
            self._cgs_unit_system['velocity_grad'] = self._cgs_unit_system['velocity']/self._cgs_unit_system['length']
            self._cgs_unit_system['energy'] = self._cgs_unit_system['mass']*self._cgs_unit_system['internal_energy']
            
        return self._cgs_unit_system

    def _get_dataset(self, particle_type, dataset_name, nosave=False):
        _attr = '_dataset_' + particle_type + '_' + dataset_name
        if nosave or not hasattr(self, _attr):
            raw_data = super()._get_dataset(particle_type, dataset_name, nosave=True).astype(np.float64)
            if dataset_name not in self._DATASET_UNIT_DICT.keys():
                warnings.warn('Dataset "{}" units not recognized. Please adapt GizmoDataset._DATASET_UNIT_DICT to include its units. For now, assuming it is unitless...'.format(dataset_name))
                return raw_data * u.dimensionless_unscaled
            else:
                factor_key, _units = self._DATASET_UNIT_DICT[dataset_name]
                factor = self.unit_system[factor_key] if factor_key is not None else 1
                if factor is None:
                    if factor_key == 'time':
                        assert self.cosmological, 'Time factor should only be None when it is a cosmological simulation.'
                        raw_data = self.scale_factor_to_time(raw_data)
                    else:
                        warnings.warn('Unit system conversion factor for "{}" is None. This may be due to an unsupported format use - please update code accordingly or verify file.'.format(factor_key))
                    factor = 1
                unit_data = raw_data * factor * _units
                if nosave:
                    return unit_data
                setattr(self, _attr, unit_data)
        return getattr(self, _attr)




def _add_convenience_properties(cls):
    for name, (group, dset, unit) in cls._CONVENIENCE_ATTRS.items():
        @property
        def prop(self, group=group, dset=dset, unit=unit):
            _attr = f"_convenience_dataset_{group}_{dset}"
            if not hasattr(self, _attr):
                setattr(self, _attr, self._get_dataset(group, dset, nosave=True).to(unit))
            return getattr(self, _attr)
        setattr(cls, name, prop)
    return cls


@_add_convenience_properties
class GIZMO_Snapshot_ConvenientWrapper(GIZMO_Snapshot):
    """
    Convenience Wrapper Class around GIZMO snapshot to quickly access common particle data. 
    """
    _CONVENIENCE_ATTRS = {
        'pos0': ('PartType0', 'Coordinates', 'pc'),
        'pos1': ('PartType1', 'Coordinates', 'pc'),
        'pos2': ('PartType2', 'Coordinates', 'pc'),
        'pos3': ('PartType3', 'Coordinates', 'pc'),
        'pos4': ('PartType4', 'Coordinates', 'pc'),
        'pos5': ('PartType5', 'Coordinates', 'pc'),
        'vel0': ('PartType0', 'Velocities', 'km/s'),
        'vel1': ('PartType1', 'Velocities', 'km/s'),
        'vel2': ('PartType2', 'Velocities', 'km/s'),
        'vel3': ('PartType3', 'Velocities', 'km/s'),
        'vel4': ('PartType4', 'Velocities', 'km/s'),
        'vel5': ('PartType5', 'Velocities', 'km/s'),
        'mass0': ('PartType0', 'Masses', 'Msun'),
        'mass1': ('PartType1', 'Masses', 'Msun'),
        'mass2': ('PartType2', 'Masses', 'Msun'),
        'mass3': ('PartType3', 'Masses', 'Msun'),
        'mass4': ('PartType4', 'Masses', 'Msun'),
        'mass5': ('PartType5', 'Masses', 'Msun'),
        'dens0': ('PartType0', 'Density', 'M_p/cm**3'),
        'dens1': ('PartType1', 'Density', 'M_p/cm**3'),
        'dens2': ('PartType2', 'Density', 'M_p/cm**3'),
        'dens3': ('PartType3', 'Density', 'M_p/cm**3'),
        'dens4': ('PartType4', 'Density', 'M_p/cm**3'),
        'dens5': ('PartType5', 'Density', 'M_p/cm**3'),
        'temp0': ('PartType0', 'Temperature', 'K'),
        'temp1': ('PartType1', 'Temperature', 'K'),
        'temp2': ('PartType2', 'Temperature', 'K'),
        'temp3': ('PartType3', 'Temperature', 'K'),
        'temp4': ('PartType4', 'Temperature', 'K'),
        'temp5': ('PartType5', 'Temperature', 'K'),
        'acc0': ('PartType0', 'Acceleration', 'km/(s Myr)'),
        'acc1': ('PartType1', 'Acceleration', 'km/(s Myr)'),
        'acc2': ('PartType2', 'Acceleration', 'km/(s Myr)'),
        'acc3': ('PartType3', 'Acceleration', 'km/(s Myr)'),
        'acc4': ('PartType4', 'Acceleration', 'km/(s Myr)'),
        'acc5': ('PartType5', 'Acceleration', 'km/(s Myr)'),
        'pot0': ('PartType0', 'Potential', 'erg/g'),
        'pot1': ('PartType1', 'Potential', 'erg/g'),
        'pot2': ('PartType2', 'Potential', 'erg/g'),
        'pot3': ('PartType3', 'Potential', 'erg/g'),
        'pot4': ('PartType4', 'Potential', 'erg/g'),
        'pot5': ('PartType5', 'Potential', 'erg/g'),
        'smooth0': ('PartType0', 'SmoothingLength', 'pc')
    }


def load_gizmo(filepath, debugging=False):
    """Loads GIZMO into a Snapshot for use."""
    if debugging:
        print("--- HDF5 File Dump ---")
        hdf5.explore(filepath)
    snap = GIZMO_Snapshot_ConvenientWrapper(filepath)
    return snap

