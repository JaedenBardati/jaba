
import os, warnings, inspect, gc

import numpy as np # pyright: ignore[reportMissingImports]

from .io import hdf5
from .utils import units as u
from .utils import coordinates as coord
from .utils import grid
from .utils import visual as jv 
from .utils import datastructures
from .analysis import dynamics as dyn

###
### Structure for Snapshot Class Construction:
### AbstractSnapshot (define required values) -> LoaderSnapshot (unitless for a particular load method, not unique to any code) -> BasicCodeSnapshot (+units and code conventions) -> FullCodeSnapshot (+transforms, images, simple calculations and other convience methods/properties) 
###                ^ abstract_snapshot should be as simple as possible, but define simple get/set qtys required by convience stuff later but can be rewritten/amended by particular code adaptation
### 
### EVENTUALLY TODO: 
###  - Separately make a Simulation object composed of a series of snapshot objects that inherit from FullCodeSnapshots, adding on time-based convience functions
###  - Generalize loaders using some jaba.io.load()-like function & combine FullCodeSnapshot convience maker stuff and LoaderSnapshot into AbstractSnapshot so that the inheritance is simply AbstractSnapshot -> FullCodeSnapshot (which is where you define units and code conventions, along with any loader dependent quantities if relevant)
### 
### 
###

### TO DO:
    # - add some quick visualization stuff to the convenience snapshot classes
    # - need to handle u.to_unit/u.get_unit/u.get_value better.
    # - generalize (particle_type, dataset_name) pairs to just key?? (i.e. snap['PartType0', 'Density'] and snap[0, 'Density'] behaviour) + standardize code 
    # - combine into abstract class before and ensure generality
    # - add derived datasets
    # - expand convenience features
    # - add transforming to different coordinate systems (e.g. cylindrical, spherical, etc.) -> maybe just add derived datasets upon request and recalculate on transform?
    # - add a way of creating a blank/custom snapshot
    # - add more different codes/sims (e.g. TNG, EAGLE, etc.), rather than just GIZMO and convience attrs
    # - add separate simulation class for time-based functions
    # - add SKIRT analysis code 
    # - add particle masking/view?
###

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
    
    @property
    def available_datasets(self):
        if not hasattr(self, '_available_datasets'):
            self._available_datasets = set((p, d) for p in self.particle_types for d in self._dataset_names(p))
        return self._available_datasets

    @staticmethod
    def _get_dataset_path(particle_type, dataset_name):
        return particle_type + '/' + dataset_name

    @staticmethod
    def _get_dataset_attr_name(particle_type, dataset_name):
        return '_dataset_' + particle_type + '_' + dataset_name

    def _get_dataset(self, particle_type, dataset_name, nosave=False):
        _attr = self._get_dataset_attr_name(particle_type, dataset_name)
        if nosave or not hasattr(self, _attr):
            if particle_type not in self.particle_types:
                raise KeyError('Particle type "{}" not recognized. Use "particle_types" attribute for a list.'.format(particle_type))
            if dataset_name not in self._dataset_names(particle_type):
                raise KeyError('Data set {} for particle type "{}" not recognized.'.format(dataset_name, particle_type))
            dataset_path = self._get_dataset_path(particle_type, dataset_name)
            data = hdf5.get_dataset(self.filepath, dataset_path)
            if nosave:
                return data
            setattr(self, _attr, data)
        return getattr(self, _attr)

    @staticmethod
    def _resolve_particle_type_name(key):
        # get particle type name version
        if isinstance(key, int):
            if key < len(HDF5_Snapshot._SUPPORTED_PARTICLES_TYPES):
                key = HDF5_Snapshot._SUPPORTED_PARTICLES_TYPES[key]
            else:
                raise KeyError('Unrecognized particle type key "{}". Must be an integer or in the list of supported particle types: {}'.format(HDF5_Snapshot._SUPPORTED_PARTICLES_TYPES))
        return key

    @staticmethod
    def _resolve_particle_type_number(key):
        # get particle type number version
        if isinstance(key, int):
            if 0 <= key and key < len(HDF5_Snapshot._SUPPORTED_PARTICLES_TYPES):
                return key
            raise KeyError('Unrecognized particle type key "{}". Must be an integer in the range [0, {})'.format(key, len(HDF5_Snapshot._SUPPORTED_PARTICLES_TYPES)))
        if key in HDF5_Snapshot._SUPPORTED_PARTICLES_TYPES:
            return int(str(key)[-1]) #specfic O(1) solution to GIZMO/GADGET only, but in general can be: HDF5_Snapshot._SUPPORTED_PARTICLES_TYPES.index(key), though this is O(n)
        
        raise KeyError('Unrecognized particle type key "{}". Must be an integer or in the list of supported particle types: {}'.format(HDF5_Snapshot._SUPPORTED_PARTICLES_TYPES))

    def _resolve_dataset_key(self, particle_type, dataset_name):
        key = self._resolve_particle_type_name(key)
        if key in self.particle_types:
            return self._dataset_names(key)
        elif isinstance(key, tuple) and len(key) == 2:
            if isinstance(key[0], int):
                key = (self._resolve_particle_type_name(key[0]), key[1])
            return self._get_dataset(key[0], key[1])
        else:
            raise KeyError('Unrecognized key "{}". Must be a 2-tuple where the first instance is a particle type and the second is a dataset entry. Alternatively, you can retrieve the dataset names by just passing the particle type alone'.format(key))

    def __getitem__(self, key):
        key = self._resolve_particle_type_name(key)
        if key in self.particle_types:
            return self._dataset_names(key)
        elif isinstance(key, tuple) and len(key) == 2:
            if isinstance(key[0], int):
                key = (self._resolve_particle_type_name(key[0]), key[1])
            return self._get_dataset(key[0], key[1])
        elif isinstance(key, str):
            available_datasets = []  # get available particle types if dataset is specified - TODO: put this in its own function and check that there are no reduncancies with available_datasets, _dataset_names, etc.
            for parttype in self.particle_types:
                if key in self._dataset_names(parttype):
                    available_datasets.append((parttype, key))
            if available_datasets:
                return available_datasets
        raise KeyError('Unrecognized key "{}". Must be a 2-tuple where the first instance is a particle type and the second is a dataset entry. Alternatively, you can retrieve the dataset names by just passing the particle type alone'.format(key))
        
    def __contains__(self, key):
        # key can be a particle type / index, dataset name, or a 2-tuple of (particle type / index, dataset name)
        try:
            self[key]
            return True
        except KeyError:
            return False

    def _write_dataset(self, particle_type, dataset_name, data, overwrite=False):
        """Writes a dataset to the snapshot file. If overwrite is False, will not overwrite existing datasets."""
        _attr = self._get_dataset_attr_name(particle_type, dataset_name)
        hdf5._write_dataset(self.filepath, 
                            self._get_dataset_path(particle_type, dataset_name), 
                            data, 
                            overwrite=overwrite
        )
        setattr(self, _attr, data)

    def _write_data(self, data_dict, overwrite=False):
        """Writes multiple datasets to the snapshot file from a dictionary of (particle_type, dataset_name): data pairs. If overwrite is False, will not overwrite existing datasets.
        This is more efficient than writing each dataset individually, as it only opens the file once."""
        hdf5._write_data(self.filepath, data_dict, overwrite=overwrite)
        for (particle_type, dataset_name), data in data_dict.items():
            setattr(self, self._get_dataset_attr_name(particle_type, dataset_name), data)

    def _delete_dataset(self, particle_type, dataset_name):
        """Deletes a dataset from the snapshot file and from the class instance."""
        _attr = self._get_dataset_attr_name(particle_type, dataset_name)
        hdf5._delete_dataset(self.filepath, self._get_dataset_path(particle_type, dataset_name))
        if hasattr(self, _attr):
            delattr(self, _attr)

    def _write_metadata(self, metadata_dict, overwrite=False):
        """Writes metadata to the snapshot file. If overwrite is False, will not overwrite existing metadata."""
        self.metadata # call to ensure original metadata is loaded
        hdf5._write_metadata(self.filepath, {self._HEADER_NAME: metadata_dict}, overwrite=overwrite)
        for key, value in metadata_dict.items():
            self._attributes[key] = value

    def _rename_dataset(self, particle_type, old_dataset_name, new_dataset_name):
        """Renames a dataset in the snapshot file and updates the class instance."""
        _old_attr = self._get_dataset_attr_name(particle_type, old_dataset_name)
        _new_attr = self._get_dataset_attr_name(particle_type, new_dataset_name)
        hdf5._rename_dataset(self.filepath, 
                            self._get_dataset_path(particle_type, old_dataset_name), 
                            self._get_dataset_path(particle_type, new_dataset_name)
        )
        if hasattr(self, _old_attr):
            data = getattr(self, _old_attr)
            setattr(self, _new_attr, data)
            delattr(self, _old_attr)

    def _rename_metadata(self, old_key, new_key):
        """Renames a metadata key in the snapshot file and updates the class instance."""
        self.metadata # call to ensure original metadata is loaded
        hdf5._rename_metadata(self.filepath, self._HEADER_NAME, old_key, new_key)
        if old_key in self._attributes:
            value = self._attributes[old_key]
            del self._attributes[old_key]
            self._attributes[new_key] = value



class Basic_GIZMO_Snapshot(HDF5_Snapshot):
    """
    Class for GIZMO specific snapshots. Uses metadata to infer code units to CGS conversions.
    Uses GIZMO standard formatting: http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html
    YOU MUST DOUBLE CHECK THESE UNITS! Add new particle data names to _DATASET_UNITS. 
    """

    _IS_COMOVING_UNIT = 'is_comoving'
    _TIME_UNIT_THAT_COULD_BE_A_SCALE_FACTOR = 'time'
    _DATASET_UNITS = { 
        # powers of each composite unit (mass, length, velocity) from the default (physical) unit system and denote if it is comoving (which will factor scale factor in to dimensional analysis).
        # OR 'time' denoting that, when the simulation is cosmological, the dataset is a scale factor,
        # OR a fixed conversion factor returning jaba units, 
        # OR None if no units. 
        'Masses': {'mass': 1, _IS_COMOVING_UNIT: False},
        'Velocities': {'velocity': 1, _IS_COMOVING_UNIT: True},
        'Coordinates': {'length': 1, _IS_COMOVING_UNIT: True},
        'Density': {'mass': 1, 'length': -3, _IS_COMOVING_UNIT: True},
        'SmoothingLength': {'length': 1, _IS_COMOVING_UNIT: True},
        'KernelMaxRadius': {'length': 1, _IS_COMOVING_UNIT: True}, # this is the new "smoothing length" (for MFM/MFV?) in the latest version! 
        'Pressure': {'mass': 1, 'length': -3, 'velocity': 2},  # the rest of the units here need to be checked if they are comoving or not - I assume they are all output as physical units
        'Acceleration': {'mass': 0, 'length': -1, 'velocity': 2},
        'InternalEnergy': {'velocity': 2, _IS_COMOVING_UNIT: False},
        'Potential': {'velocity': 2},
        'Temperature': u.K,
        'PhotonEnergy': {'mass': 1, 'velocity': 2},
        'DensityGradient': {'mass': 1, 'length': -4},
        'VelocityGradient': {'velocity': 1, 'length': -1},
        'SoundSpeed': {'velocity': 1},
        'RadiativeAcceleration': {'mass': 0, 'length': -1, 'velocity': 2},
        'HydroAcceleration': {'mass': 0, 'length': -1, 'velocity': 2},
        'PhotonOpacity': {'length': 2, 'mass': -1},
        'PhotonFluxDensity': {'mass': 1, 'velocity': 3, 'length': -3}, # need to check this one
        'BH_Mass': {'mass': 1},
        'BH_Mdot': {'mass': 1,  'length': -1, 'velocity': 1},
        'BH_Mass_AlphaDisk': {'mass': 1},
        'BH_AccretionLength': {'length': 1},
        'BH_Specific_AngMom': {'mass': 0, 'length': 1, 'velocity': 1},
        'MagneticField': u.G,
        'Dust_Temperature': u.K,
        'IRBand_Radiation_Temperature': u.K,
        'StarFormationRate': u.Msun/u.yr,
        'StarLuminosity_Solar': u.Lsun,
        'StellarFormationTime': _TIME_UNIT_THAT_COULD_BE_A_SCALE_FACTOR,
        'Metallicity': None,
        'ElectronAbundance': None,
        'NeutralHydrogenAbundance': None,
        'MolecularMassFraction': None,
        'HII': None,
        'EddingtonTensor': None,
        'ParticleChildIDsNumber': None,
        'ParticleIDGenerationNumber': None,
        'ParticleIDs': None,
        'DustToGasRatio_Local': None,
        'OStarNumber': None,
        'BH_NProgs': None,
    }
    _INT64_DATATYPES = {'ParticleIDs', 'ParticleChildIDsNumber', 'ParticleIDGenerationNumber'} # read as long long for e.g., particle IDs

    @property
    def Cosmological(self):
        """Note if cosmological, time is a scale factor not physical time."""
        if not hasattr(self, '_Cosmological'):
            if 'ComovingIntegrationOn' not in self.metadata:
                warnings.warn('Cannot determine if it is a cosmological simulation from metadata. Assuming non-cosmological.') 
                self._Cosmological = False
            else:
                self._Cosmological = bool(int(self.metadata['ComovingIntegrationOn']))
        return self._Cosmological
    cosmological = Cosmological
    is_cosmological = Cosmological

    @property
    def HubbleParam(self):
        if not hasattr(self, '_HubbleParam'):
            if 'HubbleParam' not in self.metadata:
                if self.Cosmological:
                    warnings.warn('Cannot determine Hubble parameter from metadata. Assuming h = 0.7 since it is cosmological.')
                    self._HubbleParam = 0.7
                else:
                    warnings.warn('Cannot determine Hubble parameter from metadata. Assuming h = 1.0 since it is not cosmological.')
                    self._HubbleParam = 1.0
            else:
                self._HubbleParam = float(self.metadata['HubbleParam'])
        return self._HubbleParam
    HubbleParameter = HubbleParam
    hubble_param = HubbleParam
    hubble_parameter = HubbleParam
    h = HubbleParam

    @property
    def Redshift(self):
        if not hasattr(self, '_Redshift'):
            if 'Redshift' not in self.metadata:
                warnings.warn('Cannot determine redshift from metadata. Assuming z = 0.')
                self._Redshift = 0
            else:
                self._Redshift = float(self.metadata['Redshift'])
        return self._Redshift
    redshift = Redshift
    z = Redshift  # note: this may cause conflict in the future with some other value

    @property
    def ScaleFactor(self):
        if not hasattr(self, '_ScaleFactor'):
            self._ScaleFactor = 1/(1+self.redshift)
        return self._ScaleFactor
    scale_factor = ScaleFactor
    a_scale = ScaleFactor
    a = ScaleFactor # note: this may cause conflict in the future with some other value
    
    @property
    def BoxSize(self):
        if not hasattr(self, '_BoxSize'):
            if 'BoxSize' not in self.metadata:              
                warnings.warn('Cannot determine box size from metadata. Returning zero.')
                self._BoxSize = 0
            else:
                self._BoxSize = float(self.metadata['BoxSize']) * self.UnitSystem['length'] * u.cm
        return self._BoxSize
    box_size = BoxSize

    @property
    def OmegaMatter(self):
        if not hasattr(self, '_OmegaMatter'):
            if 'Omega_Matter' not in self.metadata:
                if self.cosmological:
                    warnings.warn('Cannot determine omega matter from metadata. Assuming 0.27.')
                    self._OmegaMatter = 0.27
                else:
                    warnings.warn('Cannot determine omega matter from metadata. Assuming 0.')
                    self._OmegaMatter = 0.0
            else:
                self._OmegaMatter = float(self.metadata['Omega_Matter'])
        return self._OmegaMatter
    omega_matter = OmegaMatter
    Omega_Matter = OmegaMatter
    
    @property
    def OmegaLambda(self):
        if not hasattr(self, '_OmegaLambda'):
            if 'Omega_Lambda' not in self.metadata:
                if self.cosmological:
                    warnings.warn('Cannot determine omega lambda from metadata. Assuming 0.73.')
                    self._OmegaLambda = 0.73
                else:
                    warnings.warn('Cannot determine omega lambda from metadata. Assuming 0.')
                    self._OmegaLambda = 0.0
            else:
                self._OmegaLambda = float(self.metadata['Omega_Lambda'])
        return self._OmegaLambda
    omega_lambda = OmegaLambda
    Omega_Lambda = OmegaLambda

    @property
    def OmegaBaryon(self):
        if not hasattr(self, '_OmegaBaryon'):
            if 'Omega_Baryon' not in self.metadata:
                if self.cosmological:
                    warnings.warn('Cannot determine omega baryon from metadata. Assuming 0.044.')
                    self._OmegaBaryon = 0.044
                else:
                    warnings.warn('Cannot determine omega baryon from metadata. Assuming 0.')
                    self._OmegaBaryon = 0.0
            else:
                self._OmegaBaryon = float(self.metadata['Omega_Baryon'])
        return self._OmegaBaryon
    omega_baryon = OmegaBaryon
    Omega_Baryon = OmegaBaryon

    @property
    def OmegaRadiation(self):
        if not hasattr(self, '_OmegaRadiation'):
            if 'OmegaRadiation' not in self.metadata:
                warnings.warn('Cannot determine omega radiation from metadata. Assuming 0.')
                self._OmegaRadiation = 0.0
            else:
                self._OmegaRadiation = float(self.metadata['OmegaRadiation'])
        return self._OmegaRadiation
    omega_radiation = OmegaRadiation
    Omega_Radiation = OmegaRadiation

    def scale_factor_to_time(self, scale_factor):  # move to cosmology module when added
        raise NotImplementedError('Need to add cosmological integration to program.')

    @property
    def UnitSystem_In_CGS(self):
        if not hasattr(self, '_UnitSystem_In_CGS'):  # to force the recalulation of units, delete _UnitSystem attribute
            UnitMass_in_g = None
            for m in ['UnitMass_In_CGS', 'UnitMass_in_g']:
                if m in self.metadata:
                    if UnitMass_in_g is not None:
                        raise Exception('Ambiguous mass units in metadata.')
                    UnitMass_in_g = self.metadata[m]
            UnitVelocity_in_cm_per_s = None
            for l in ['UnitLength_In_CGS', 'UnitLength_in_cm']:
                if l in self.metadata:
                    if UnitVelocity_in_cm_per_s is not None:
                        raise Exception('Ambiguous length units in metadata.')
                    UnitVelocity_in_cm_per_s = self.metadata[l]
            UnitVelocity_in_cgs = None
            for v in ['UnitVelocity_In_CGS', 'UnitVelocity_in_cm_per_s']:
                if v in self.metadata:
                    if UnitVelocity_in_cgs is not None:
                        raise Exception('Ambiguous velocity units in metadata.')
                    UnitVelocity_in_cgs = self.metadata[v]
            if UnitMass_in_g is None or UnitVelocity_in_cm_per_s is None or UnitVelocity_in_cgs is None:
                raise Exception('Could not find unit system in metadata.')

            # adjust for hubble parameter factors by GIZMO to get code units
            h = self.HubbleParam
            self._UNIT_MASS_IN_CGS = UnitMass_in_g/h
            self._UNIT_VELOCITY_IN_CGS = UnitVelocity_in_cgs
            self._UNIT_LENGTH_IN_CGS = UnitVelocity_in_cm_per_s/h

            # derive other code units as in allvars.h in GIZMO code for easy use
            self._UNIT_TIME_IN_CGS = self._UNIT_LENGTH_IN_CGS / self._UNIT_VELOCITY_IN_CGS
            self._UNIT_ENERGY_IN_CGS = self._UNIT_MASS_IN_CGS * self._UNIT_VELOCITY_IN_CGS * self._UNIT_VELOCITY_IN_CGS
            self._UNIT_PRESSURE_IN_CGS = self._UNIT_ENERGY_IN_CGS/(self._UNIT_LENGTH_IN_CGS*self._UNIT_LENGTH_IN_CGS*self._UNIT_LENGTH_IN_CGS)
            self._UNIT_DENSITY_IN_CGS = self._UNIT_MASS_IN_CGS/(self._UNIT_LENGTH_IN_CGS*self._UNIT_LENGTH_IN_CGS*self._UNIT_LENGTH_IN_CGS)
            self._UNIT_SPECEGY_IN_CGS = self._UNIT_PRESSURE_IN_CGS / self._UNIT_DENSITY_IN_CGS  # specific energy
            self._UNIT_SURFDEN_IN_CGS = self._UNIT_DENSITY_IN_CGS * self._UNIT_LENGTH_IN_CGS
            self._UNIT_FLUX_IN_CGS = self._UNIT_PRESSURE_IN_CGS * self._UNIT_VELOCITY_IN_CGS
            self._UNIT_LUM_IN_CGS = self._UNIT_ENERGY_IN_CGS / self._UNIT_TIME_IN_CGS
            #self._UNIT_B_IN_GAUSS = np.sqrt(4.0*np.pi*self._UNIT_PRESSURE_IN_CGS)  # this is CODE units NOT what is output, always output in Gauss.

            self._UNIT_MASS_IN_SOLAR = self._UNIT_MASS_IN_CGS / u.SOLAR_MASS_CGS
            self._UNIT_DENSITY_IN_NHCGS = self._UNIT_DENSITY_IN_CGS / u.PROTONMASS_CGS
            self._UNIT_TIME_IN_YR = self._UNIT_TIME_IN_CGS / u.SECONDS_PER_YEAR
            self._UNIT_TIME_IN_MYR = self._UNIT_TIME_IN_CGS / (u.SECONDS_PER_YEAR*1.0e6)
            self._UNIT_TIME_IN_GYR = self._UNIT_TIME_IN_CGS / (u.SECONDS_PER_YEAR*1.0e9)
            self._UNIT_LENGTH_IN_SOLAR = self._UNIT_LENGTH_IN_CGS / u.SOLAR_RADIUS_CGS
            self._UNIT_LENGTH_IN_AU = self._UNIT_LENGTH_IN_CGS / 1.496e13
            self._UNIT_LENGTH_IN_PC = self._UNIT_LENGTH_IN_CGS / 3.085678e18
            self._UNIT_LENGTH_IN_KPC = self._UNIT_LENGTH_IN_CGS / 3.085678e21
            self._UNIT_PRESSURE_IN_EV = self._UNIT_PRESSURE_IN_CGS / u.ELECTRONVOLT_IN_ERGS
            self._UNIT_VEL_IN_KMS = self._UNIT_VELOCITY_IN_CGS / 1.0e5
            self._UNIT_LUM_IN_SOLAR = self._UNIT_LUM_IN_CGS / u.SOLAR_LUM_CGS
            self._UNIT_FLUX_IN_HABING = self._UNIT_FLUX_IN_CGS / u.HABING_FLUX_CGS
            self._UNIT_EGY_DENSITY_IN_HABING = self._UNIT_PRESSURE_IN_CGS / (u.HABING_FLUX_CGS / u.C_LIGHT_CGS)

            self._U_TO_TEMP_UNITS = (u.PROTONMASS_CGS / u.BOLTZMANN_CGS) * (self._UNIT_ENERGY_IN_CGS/self._UNIT_MASS_IN_CGS)
            self._C_LIGHT_CODE = u.C_LIGHT_CGS/self._UNIT_VELOCITY_IN_CGS
            
            # package this together in a general unit system dictionary for general conversion from code units to physical units using jaba unit system.
            self._UnitSystem_In_CGS = {  # TODO: make this public and something required for convience functions, then write unit system conversion functions
                'mass': self._UNIT_MASS_IN_CGS*u.g, # mass factor to convert from code units to g
                'length': self._UNIT_LENGTH_IN_CGS*u.cm, # length factor to convert from code units to cm
                'velocity': self._UNIT_VELOCITY_IN_CGS*u.cm/u.s, # velocity factor to convert from code units to cm/s
            }
            if self.cosmological:
                # convert to physical units
                raise NotImplementedError('Need to add cosmological integration to program.')
                # a_scale = self.ScaleFactor
                # MASS_physical = MASS_code
                # LENGTH_physical = LENGTH_code * a_scale
                # VELOCITY_physical = VELOCITY_code * np.sqrt(a_scale)
                # INTERNAL_ENERGY_physical = INTERNAL_ENERGY_code
                # DENSITY_physical = DENSITY_code / a_scale**3
                self._UnitSystem_In_CGS['length'] *= self.ScaleFactor
                self._UnitSystem_In_CGS['velocity'] *= np.sqrt(self.ScaleFactor) 
                # also note that time unit is really a scale factor now

        return self._UnitSystem_In_CGS
    unit_system_in_cgs = UnitSystem_In_CGS
    UnitSystem = UnitSystem_In_CGS
    unit_system = UnitSystem_In_CGS

    def _get_dataset(self, particle_type, dataset_name, nosave=False):
        _attr = self._get_dataset_attr_name(particle_type, dataset_name)
        if nosave or not hasattr(self, _attr):
            dtype = np.float64 if dataset_name not in self._INT64_DATATYPES else np.int64
            raw_data = super()._get_dataset(particle_type, dataset_name, nosave=True).astype(dtype) 
            if dataset_name not in self._DATASET_UNITS.keys():
                warnings.warn('Dataset "{}" does not have defined units. Please update code if necessary - assuming its unitless.'.format(dataset_name))
                # If you get this warning, you need to add info on the dataset to _DATASET_UNITS to get units. 
                return raw_data * u.dimensionless_unscaled
            
            _unit_metadata = self._DATASET_UNITS[dataset_name]
            if _unit_metadata is None:
                unit_data = raw_data * u.dimensionless_unscaled
            elif _unit_metadata == self._TIME_UNIT_THAT_COULD_BE_A_SCALE_FACTOR:
                if self.Cosmological:
                    unit_data = self.scale_factor_to_time(raw_data)
                else:
                    unit_data = raw_data * self.UnitSystem['length']/self.UnitSystem['velocity']
            elif isinstance(_unit_metadata, dict):
                factor = 1.0
                convert_comoving_to_physical = self.Cosmological and self._IS_COMOVING_UNIT in _unit_metadata and _unit_metadata[self._IS_COMOVING_UNIT]
                for unit_system_key in _unit_metadata.keys():
                    if unit_system_key != self._IS_COMOVING_UNIT:
                        factor *= self.UnitSystem_In_CGS[unit_system_key] ** _unit_metadata[unit_system_key]
                        if convert_comoving_to_physical:
                            if unit_system_key == 'length':
                                factor *= self.ScaleFactor ** _unit_metadata[unit_system_key]
                            elif unit_system_key == 'velocity':
                                factor *= self.ScaleFactor ** (0.5 * _unit_metadata[unit_system_key])
                unit_data = raw_data * factor
            else:
                unit_data = raw_data * _unit_metadata
            if nosave:
                return unit_data
            setattr(self, _attr, unit_data)
        return getattr(self, _attr)








class StandardDataset(): 
    """ 
    A class that represents a specific dataset with standard behaviour (e.g., snap.mass, snap.position, snap.velocity). 
    Access particle type specific datasets via indexing (e.g., snap.mass[0] or snap.mass['PartType0'] for PartType0).
    """
    def __init__(self, name, snapshot):
        self.name = name                # e.g., 'mass', 'position', 'velocity', etc.
        self._snapshot = snapshot       # parent snapshot object itself
        self._dataset_names = dict()    # name of the dataset that it corresponds to in the snapshot file, determined on first access for each particle type.

    def _get_dataset_info_for_particle_type(self, particle_type):
        parttype_name = self._snapshot._resolve_particle_type_name(particle_type)
        dataset_name = self._dataset_names.get(parttype_name, None)
        if dataset_name is None:
            # find the dataset name for this standard dataset and particle type
            for possible_dset in self._snapshot._STANDARD_DSET_POSSIBLE_DATASET_NAMES[self.name]:
                if possible_dset in self._snapshot._dataset_names(parttype_name):
                    if dataset_name is not None:
                        raise Exception(f"There are multiple possible datasets that {self.name} could refer to and I'm not sure which it is. Possible datasets are {dataset_name} or {possible_dset}.")
                    dataset_name = possible_dset
            if dataset_name is None:
                raise AttributeError(f'I cannot find the dataset that corresponds to "{self.name}". It might not be in the snapshot, or you may need to add support for the name recognition.')  # if you get this error: adjust add relevant dataset to _STANDARD_DSET_POSSIBLE_DATASET_NAMES
            self._dataset_names[parttype_name] = dataset_name
        return (parttype_name, dataset_name)

    def __getitem__(self, particle_type):
        """ Get the dataset for the specified particle type. """
        parttype_name, dataset_name = self._get_dataset_info_for_particle_type(particle_type)
        return self._snapshot[parttype_name, dataset_name]

    def __setitem__(self, particle_type, value):
        """ Set the dataset for the specified particle type. """
        parttype_name, dataset_name = self._get_dataset_info_for_particle_type(particle_type)

        if (parttype_name, dataset_name) in self.loaded_datasets:
            # if loaded, get unit from the loaded dataset
            dset_unit = u.get_unit(self._snapshot[parttype_name, dataset_name], default_unit=u.dimensionless_unscaled)
        elif dataset_name in self._snapshot._DATASET_UNITS.keys():
            # if not, guess its cgs in the appropriate dimensions if its in the dataset units dictionary 
            dset_unit = 1
            for subunit, subunittype in zip((u.cm,u.g,u.cm/u.s), ('mass','length','velocity')):
                if subunittype in self._snapshot._DATASET_UNITS[dataset_name].keys():
                    dset_unit *= subunit ** self._snapshot._DATASET_UNITS[dataset_name][subunittype]
            dset_unit = u.get_unit(dset_unit, default_unit=u.dimensionless_unscaled)
        else:
            # otherwise, assume its unitless if no units are defined in that entry
            warnings.warn('Dataset "{}" does not have defined units. Please update code if necessary - assuming its unitless.'.format(dataset_name)) # If you get this warning, you need to add info on the dataset to _DATASET_UNITS to get units. 
            dset_unit = u.dimensionless_unscaled
        
        # assume the unit of the value passed is the same as the dataset unit if not specified
        value_unit = u.get_unit(value, default_unit=dset_unit)
        
        # check the dataset unit and the value unit are compatible, otherwise raise an error
        if not value_unit.is_equivalent(dset_unit):
            raise ValueError('Value has units "{}" which are not compatible with the dataset units "{}".'.format(value_unit, dset_unit))
        
        # update the dataset in memory  # TODO: add a switch to toggle between setting the dataset in the snapshot file or just in memory. - also make sure the method below is standardized in something like _set_dataset()
        setattr(self, self._snapshot._get_dataset_attr_name(parttype_name, dataset_name), value) 

    def __delitem__(self, particle_type):
        """ Delete the dataset for the specified particle type. """
        parttype_name, dataset_name = self._get_dataset_info_for_particle_type(particle_type)

        if (parttype_name, dataset_name) in self.loaded_datasets:
            # TODO: add a switch to toggle between deleting the dataset in the snapshot file or just in memory. - also make sure the method below is standardized in something like _delete_dataset()
            delattr(self, self._get_dataset_attr_name(parttype_name, dataset_name))
        else:
            raise AttributeError('Dataset {} for particle type {} has not yet been loaded. Please load it before attempting to delete it.'.format(dataset_name, parttype_name))

    def __set__(self, instance, value):
        raise NotImplementedError("Currently, you cannot set a standard dataset property without specifying its particle type.") # TODO: think about what makes sense here - maybe allow setting the dataset for all particle types at once? only if they are the same size? 

    def __del__(self):
        pass
        #raise NotImplementedError("Currently, you cannot delete a standard dataset property without specifying its particle type.") # TODO: think about what makes sense here - maybe allow deleting the dataset for all particle types at once?

    def __repr__(self):
        return f"<Standard dataset '{self.name}' in snapshot '{self._snapshot.name}'>"




def _add_convenience_properties(cls):
    #assert isinstance(cls, Snapshot), "Convenience snapshot constructor must be a class that inherits from Snapshot." # TODO
    assert hasattr(cls, "_STANDARD_DSET_POSSIBLE_DATASET_NAMES"), "When constructing a convenience snapshot, you need to specify the standard datasets."
    assert hasattr(cls, "_TRANSFORMATION_BEHAVIORS"), "When constructing a convenience snapshot, you need to specify the transformation behaviour."
    #assert hasattr(cls, "_STANDARD_DSET_ALIASES"), "When constructing a convenience snapshot, you need to specify any standard dataset aliases." # TODO
    
    def __init__(self, *args, **kwargs):
        super(cls, self).__init__(*args, **kwargs)
        self.loaded_datasets = set()  # track which datasets have been loaded
        self._loaded_standard_datasets = dict() # track which standard dataset have been called and what loaded dataset they correspond to
        #self.derived_datasets = set()  # TODO: track which datasets have been derived from loaded datasets -> then add some switch that prioritizes memory vs speed?

        self.absolute_centers = {'position': None, 'velocity': None}  # absolute centers of the current position and velocity in terms of the original snapshot orientation
        self.transformation_matrix = None  # transformation matrix of the current position in terms of the original snapshot orientation
        self._inv_transformation_matrix = None
    setattr(cls, '__init__', __init__)

    ## Add properties to handle loading datasets via shortcut attributes
    for _name, _ in cls._STANDARD_DSET_POSSIBLE_DATASET_NAMES.items():
        # support snap.pos[0] format
        @property
        def prop(self, _name=_name):
            if _name not in self._loaded_standard_datasets:
                self._loaded_standard_datasets[_name] = StandardDataset(_name, self) # make new object and add to loaded standard datasets dictionary
            return self._loaded_standard_datasets[_name]  # return the object
        @prop.setter
        def prop(self, value, _name=_name):
            getattr(self, _name).__set__(value) # make sure the standard dataset object is created and loaded + pass on to __set__ to handle
        @prop.deleter
        def prop(self, _name=_name):
            getattr(self, _name).__del__() # make sure the standard dataset object is created and loaded + pass on to __del__ to handle
            del self._loaded_standard_datasets[_name] # remove from the loaded standard datasets dictionary
        setattr(cls, _name, prop)
        
        # also support old snap.pos0 format (deprecated)
        for _parttype in cls._SUPPORTED_PARTICLES_TYPES:
            @property
            def prop2(self, _name=_name, _parttype=_parttype):
                return getattr(self, _name).__getitem__(_parttype) # make sure the standard dataset object is created and loaded + pass on to __getitem__ to handle
            @prop2.setter
            def prop2(self, value, _name=_name, _parttype=_parttype):
                getattr(self, _name).__setitem__(_parttype, value)  # make sure the standard dataset object is created and loaded + pass on to __setitem__ to handle
            @prop2.deleter
            def prop2(self, _name=_name, _parttype=_parttype):
                getattr(self, _name).__delitem__(_parttype)  # make sure the standard dataset object is created and loaded + pass on to __delitem__ to handle
            setattr(cls, _name + str(cls._resolve_particle_type_number(_parttype)), prop2)

    ## Add method to handle general spatial transformations and effect on other tensor-like datasets

    def transform(self, center=None, vcenter=None, z=None, y=None, x=None, zdir=None, ydir=None, xdir=None, absolute=True, in_radians=False, strictness=2, verbose=False):
        """
        Center around position and velocity, then rotate the snapshot to a given orientation. 
        Center and rotation are given in terms of the current snapshot orientation, unless absolute=True, in which case they are given in terms of the original snapshot orientation.
        This will only be applied upon next loading of each transformed dataset.
        """
        if center is None and vcenter is None and z is None and y is None and x is None and zdir is None and ydir is None and xdir is None:
            raise ValueError("At least one transformation parameter must be specified.")
        if center is not None and not u.is_unit_like(center): # TODO: eventually also convert unit-strs and assume default position unit if no units
            raise ValueError("Position center must have units.")
        if vcenter is not None and not u.is_unit_like(vcenter): # TODO: eventually also convert unit-strs and assume default velocity unit if no units
            raise ValueError("Velocity center must have units.")

        if absolute:
            self.reset_transform()  # reset to original snapshot orientation before applying new absolute transformation

        # transform the center to coordinates in the original snapshot orientation, setting new absolute center (to original coords) and relative center (to current coords)
        relative_centers = {}
        for _centers_like, _center in (('position', center), ('velocity', vcenter)): # supported centers-like values
            relative_centers[_centers_like] = coord.transform_general_tensor(_center, T=self._inv_transformation_matrix, transforms_like='vector') if _center is not None else None
            self.absolute_centers[_centers_like] = (
                ( (relative_centers[_centers_like] if relative_centers[_centers_like] is not None else 0) + 
                  (self.absolute_centers[_centers_like] if self.absolute_centers[_centers_like] is not None else 0) ) 
                if relative_centers[_centers_like] is not None or self.absolute_centers[_centers_like] is not None else None
            )

        # get transformation matrix
        _relative_T = coord.general_rotation_matrix(z=z, y=y, x=x, zdir=zdir, ydir=ydir, xdir=xdir) # cannot be None anymore
        if np.allclose(_relative_T - np.eye(3), 0): # if no transformation
            _relative_T = None
            self.transformation_matrix = self.transformation_matrix
        else:
            if self.transformation_matrix is None:
                self.transformation_matrix = _relative_T
            else:
                self.transformation_matrix = _relative_T @ self.transformation_matrix
            self._inv_transformation_matrix = np.linalg.inv(self.transformation_matrix)

        # for each tensor-like dataset, center and transform if the dataset has already been loaded, otherwise do on-the-fly transformation when it is loaded
        for _dset, (_transforms_like, _centers_like, *_) in self._TRANSFORMATION_BEHAVIORS.items():
            _transforms_like = _transforms_like if _transforms_like is not None else 'scalar'
            if _transforms_like != 'scalar':
                for (_group, _dset2) in self.loaded_datasets: # TODO: this is fine, but it would be better if there was a way of looking at all entries of only one part with e.g. multidimensional dict
                    if _dset2 != _dset:
                        continue
                    if verbose:
                        print(f"Transforming loaded dataset {_dset} like a {_transforms_like}...")
                    _attr = self._get_dataset_attr_name(_group, _dset)
                    _data = coord.transform_general_tensor(
                        getattr(self, _attr), 
                        center=relative_centers[_centers_like], 
                        T=_relative_T, 
                        transforms_like=_transforms_like
                    )
                    setattr(self, _attr, _data)
        if verbose:
            print('Other datasets will be transformed on the fly when loaded.')    

        return self
    setattr(cls, 'transform', transform)

    ## Adjust the get dataset function to convert units and apply transformations when loading
    def _get_dataset(self, particle_type, dataset_name):
        if (particle_type, dataset_name) not in self.loaded_datasets:
            # get dataset in original units and orientation
            unit_data = super(cls, self)._get_dataset(particle_type, dataset_name)
            
            # only transform/convert units if there is a standard unit and transformation behavior defined for this dataset
            if dataset_name in self._TRANSFORMATION_BEHAVIORS or (particle_type, dataset_name) in self._TRANSFORMATION_BEHAVIORS: 
                _transforms_like, _centers_like, *_ = self._TRANSFORMATION_BEHAVIORS[dataset_name] if dataset_name in self._TRANSFORMATION_BEHAVIORS else self._TRANSFORMATION_BEHAVIORS[(particle_type, dataset_name)]

                # convert to preferred units  # TODO: add default unit support too?
                # if unit_data.unit != _unit:
                #     unit_data = unit_data.to(_unit)
                #     setattr(self, self._get_dataset_attr_name(particle_type, dataset_name), unit_data)
                
                # transform to new frame if necessary
                unit_data = coord.transform_general_tensor(
                    unit_data,
                    #center=u.to_unit(self.absolute_centers[_centers_like], _unit, _unit) if _centers_like is not None and self.absolute_centers[_centers_like] is not None else None, # TODO: add unit support too?
                    center=self.absolute_centers[_centers_like] if _centers_like is not None else None,
                    T=self.transformation_matrix,
                    transforms_like=_transforms_like,
                    reverse_order=False
                )
                setattr(self, self._get_dataset_attr_name(particle_type, dataset_name), unit_data)
            else:
                #warnings.warn('Dataset "{}" for particle type "{}" does not have a defined transformation behavior. Assuming its a scalar.'.format(dataset_name, particle_type))
                # If you get this warning, you *may* need to update _TRANSFORMATION_BEHAVIORS for the appropriate transformation behavior. 
                setattr(self, self._get_dataset_attr_name(particle_type, dataset_name), unit_data)

            # add to list of loaded datasets, transformations will be applied on the fly from now on
            self.loaded_datasets.add((particle_type, dataset_name))
        return getattr(self, self._get_dataset_attr_name(particle_type, dataset_name))
    setattr(cls, '_get_dataset', _get_dataset)


    ## Add some extra convenience methods for specific transformation operations

    def absolute_transform(self, *args, **kwargs):
        self.reset_transform()
        return self.transform(*args, **kwargs)
    setattr(cls, 'absolute_transform', absolute_transform)

    # translation in position space
    setattr(cls, 'center_pos', lambda self, center, verbose=False: self.transform(center=center, verbose=verbose))
    setattr(cls, 'center_x', lambda self, x, verbose=False: self.transform(center=[u.get_value(x, default_unit=self._POSITION_UNIT), 0.0, 0.0]*u.get_unit(x, default_unit=self._POSITION_UNIT), verbose=verbose))
    setattr(cls, 'center_y', lambda self, y, verbose=False: self.transform(center=[0.0, u.get_value(y, default_unit=self._POSITION_UNIT), 0.0]*u.get_unit(y, default_unit=self._POSITION_UNIT), verbose=verbose))
    setattr(cls, 'center_z', lambda self, z, verbose=False: self.transform(center=[0.0, 0.0, u.get_value(z, default_unit=self._POSITION_UNIT)]*u.get_unit(z, default_unit=self._POSITION_UNIT), verbose=verbose))
    setattr(cls, 'shift', lambda self, shift, verbose=False: self.transform(center=-np.asarray(u.get_value(shift, default_unit=self._POSITION_UNIT))*u.get_unit(shift, default_unit=self._POSITION_UNIT), verbose=verbose))
    setattr(cls, 'shift_x', lambda self, distance, verbose=False: self.transform(center=[-u.get_value(distance, default_unit=self._POSITION_UNIT), 0.0, 0.0]*u.get_unit(distance, default_unit=self._POSITION_UNIT), verbose=verbose))
    setattr(cls, 'shift_y', lambda self, distance, verbose=False: self.transform(center=[0.0, -u.get_value(distance, default_unit=self._POSITION_UNIT), 0.0]*u.get_unit(distance, default_unit=self._POSITION_UNIT), verbose=verbose))
    setattr(cls, 'shift_z', lambda self, distance, verbose=False: self.transform(center=[0.0, 0.0, -u.get_value(distance, default_unit=self._POSITION_UNIT)]*u.get_unit(distance, default_unit=self._POSITION_UNIT), verbose=verbose))

    # translation in velocity space
    setattr(cls, 'center_vel', lambda self, vcenter, verbose=False: self.transform(vcenter=vcenter, verbose=verbose))
    setattr(cls, 'center_vx', lambda self, vx, verbose=False: self.transform(vcenter=[u.get_value(vx, default_unit=self._VELOCITY_UNIT), 0.0, 0.0]*u.get_unit(vx, default_unit=self._VELOCITY_UNIT), verbose=verbose))
    setattr(cls, 'center_vy', lambda self, vy, verbose=False: self.transform(vcenter=[0.0, u.get_value(vy, default_unit=self._VELOCITY_UNIT), 0.0]*u.get_unit(vy, default_unit=self._VELOCITY_UNIT), verbose=verbose))
    setattr(cls, 'center_vz', lambda self, vz, verbose=False: self.transform(vcenter=[0.0, 0.0, u.get_value(vz, default_unit=self._VELOCITY_UNIT)]*u.get_unit(vz, default_unit=self._VELOCITY_UNIT), verbose=verbose))
    setattr(cls, 'boost', lambda self, boost, verbose=False: self.transform(vcenter=-np.asarray(u.get_value(boost, default_unit=self._VELOCITY_UNIT))*u.get_unit(boost, default_unit=self._VELOCITY_UNIT), verbose=verbose))
    setattr(cls, 'boost_vx', lambda self, vx, verbose=False: self.transform(vcenter=[-u.get_value(vx, default_unit=self._VELOCITY_UNIT), 0.0, 0.0]*u.get_unit(vx, default_unit=self._VELOCITY_UNIT), verbose=verbose))
    setattr(cls, 'boost_vy', lambda self, vy, verbose=False: self.transform(vcenter=[0.0, -u.get_value(vy, default_unit=self._VELOCITY_UNIT), 0.0]*u.get_unit(vy, default_unit=self._VELOCITY_UNIT), verbose=verbose))
    setattr(cls, 'boost_vz', lambda self, vz, verbose=False: self.transform(vcenter=[0.0, 0.0, -u.get_value(vz, default_unit=self._VELOCITY_UNIT)]*u.get_unit(vz, default_unit=self._VELOCITY_UNIT), verbose=verbose))

    # rotation align axes to given directions
    setattr(cls, 'align', lambda self, zdir=None, xdir=None, ydir=None, verbose=False: self.transform(zdir=zdir, xdir=xdir, ydir=ydir, verbose=verbose))
    setattr(cls, 'align_z', lambda self, zdir, verbose=False: self.transform(zdir=zdir, verbose=verbose))
    setattr(cls, 'align_y', lambda self, ydir, verbose=False: self.transform(ydir=ydir, verbose=verbose))
    setattr(cls, 'align_x', lambda self, xdir, verbose=False: self.transform(xdir=xdir, verbose=verbose))

    # rotation by angle about current axis
    setattr(cls, 'rotate', lambda self, z=None, y=None, x=None, in_radians=False, verbose=False, strictness=1: self.transform(z=z, y=y, x=x, in_radians=in_radians, verbose=verbose, strictness=strictness))
    setattr(cls, 'rotate_z', lambda self, angle, in_radians=False, verbose=False, strictness=1: self.transform(z=angle, in_radians=in_radians, verbose=verbose, strictness=strictness))
    setattr(cls, 'rotate_y', lambda self, angle, in_radians=False, verbose=False, strictness=1: self.transform(y=angle, in_radians=in_radians, verbose=verbose, strictness=strictness))
    setattr(cls, 'rotate_x', lambda self, angle, in_radians=False, verbose=False, strictness=1: self.transform(x=angle, in_radians=in_radians, verbose=verbose, strictness=strictness))
    
    # rotation by angle as seen from given viewing direction/location
    # this can get complicated for yaw and pitch since the rotation is dependent on the location and orientation of the viewer
    # for now just assume viewing from z direction, such that roll is always just rotation about z axis (cannot be said for yaw/pitch, commented code below roll is wrong...)
    setattr(cls, 'roll', lambda self, angle, in_radians=False, verbose=False, strictness=1: self.transform(z=angle, in_radians=in_radians, verbose=verbose, strictness=strictness))
    # setattr(cls, 'pitch', lambda self, angle, in_radians=False, verbose=False, strictness=1: self.transform(y=-angle, in_radians=in_radians, verbose=verbose, strictness=strictness)) # note yaw is inverted y direction (follows wikipedia definition)
    # setattr(cls, 'yaw', lambda self, angle, in_radians=False, verbose=False, strictness=1: self.transform(x=angle, in_radians=in_radians, verbose=verbose, strictness=strictness))

    ## Other transformation methods

    # center on a single particle
    def center_on(self, particle_type, idx=0, verbose=False):
        center = getattr(self, 'pos')[particle_type][idx]
        vcenter = getattr(self, 'vel')[particle_type][idx]
        if verbose:
            print(f"Centering on particle of type {particle_type} at index {idx} with position {center} and velocity {vcenter}...")
        return self.transform(center=center, vcenter=vcenter, verbose=verbose)
    setattr(cls, 'center_on', center_on)

    def faceon(self, radius=None, particle_type=0):
        """Align the snapshot's coordinate system to the angular momentum vector of a given particle type."""
        angmom = self.total_angular_momentum(particle_type, radius=radius)
        return self.transform(zdir=angmom)
    setattr(cls, 'faceon', faceon)

    def edgeon(self, radius=None, particle_type=0):
        """Align the snapshot's coordinate system to the angular momentum vector of a given particle type."""
        angmom = self.total_angular_momentum(particle_type, radius=radius)
        return self.transform(ydir=angmom)
    setattr(cls, 'edgeon', edgeon)


    ## Reset the snapshot transformation and reloading data states

    def reset_transform(self):
        """Reset all transformations and centers to original snapshot orientation and position."""
        for particle_type, dataset_name in self.loaded_datasets:
            if dataset_name in self._TRANSFORMATION_BEHAVIORS or (particle_type, dataset_name) in self._TRANSFORMATION_BEHAVIORS: 
                _transforms_like, _centers_like, *_ = self._TRANSFORMATION_BEHAVIORS[dataset_name] if dataset_name in self._TRANSFORMATION_BEHAVIORS else self._TRANSFORMATION_BEHAVIORS[(particle_type, dataset_name)]
                _transforms_like = _transforms_like if _transforms_like is not None else 'scalar' # if none, assume scalar (no transformation)
                if _transforms_like != 'scalar':
                    _attr = self._get_dataset_attr_name(particle_type, dataset_name)
                    _data = coord.transform_general_tensor(
                        getattr(self, _attr), 
                        center=-self.absolute_centers[_centers_like] if _centers_like is not None and self.absolute_centers[_centers_like] is not None else None,
                        T=self._inv_transformation_matrix, 
                        transforms_like=_transforms_like,
                        reverse_order=True,
                    )
                    setattr(self, _attr, _data)
        self.absolute_centers = {'position': None, 'velocity': None}
        self.transformation_matrix = None
        self._inv_transformation_matrix = None
    setattr(cls, 'reset_transform', reset_transform)
    
    def load_standard_data(self, standard_datasets=None):
        """Load standard datasets that are needed for most analyses and transformations, e.g. position, velocity, mass, density and smoothing length."""
        if standard_datasets is None:
            standard_datasets = self._STANDARD_DSET_POSSIBLE_DATASET_NAMES.keys()
        for parttype in self.particle_types:
            for dataset_name in standard_datasets:
                try: # TODO: should probably check available datasets instead
                    getattr(self, dataset_name)[parttype]
                except KeyError:
                    pass
    setattr(cls, 'load_standard_data', load_standard_data)

    def reload_loaded_data(self):
        """Keeps the same transformation state, but reloads from file and recalculates everything."""
        pass # TODO

    def load_all_available_data(self):
        """Load all datasets for all particle types."""
        pass # TODO 


    ## Commonly-used derived value calculations

    def total_angular_momentum(self, particle_type, radius=None): # TODO: need to generalize parttype, datatype somehow?
        """Calculate total angular momentum vector for a given particle type."""
        pos = getattr(self, 'pos')[particle_type]
        vel = getattr(self, 'vel')[particle_type]
        mass = getattr(self, 'mass')[particle_type]
        if radius is not None:  # TODO: generalize with some sort of "view" function to mask datasets
            radius = u.to_unit(radius, pos.unit, default_unit=pos.unit)
            rcut = np.linalg.norm(pos, axis=1) < radius # np.linalg.norm works with units
            pos = pos[rcut]
            vel = vel[rcut]
            mass = mass[rcut]
        return dyn.total_angular_momentum(mass, pos, vel)
    setattr(cls, 'total_angular_momentum', total_angular_momentum)

    ## TODO: particle masking?


    ## Convert to another snapshot format for use with other analysis tools

    def to_pynbody(self, gas_parttype=0, star_parttype=4, dm_parttype=1, verbose=False):
        """Convert all loaded, recognized (convenience) datasets to a pynbody snapshot for use with pynbody's analysis and visualization tools. Add support as needed."""
        gas_parttype = self._resolve_particle_type_number(gas_parttype)
        star_parttype = self._resolve_particle_type_number(star_parttype)
        dm_parttype = self._resolve_particle_type_number(dm_parttype)
        assert gas_parttype != star_parttype != dm_parttype, "Gas, star, and dark matter particle types must be different for conversion to pynbody snapshot."
        import pynbody

        # quick check for pos to determine length (always required for particle data)
        Ngas, Nstar, Ndm = 0, 0, 0
        for std_dset_name, std_dset_obj in self._loaded_standard_datasets.items():
            for parttype in self.particle_types:
                if parttype not in [gas_parttype, star_parttype, dm_parttype]:
                    continue
                if Ngas == 0 and parttype == gas_parttype:
                    Ngas = std_dset_obj[parttype].shape[0]
                if Nstar == 0 and parttype == star_parttype:
                    Nstar = std_dset_obj[parttype].shape[0]
                if Ndm == 0 and parttype == dm_parttype:
                    Ndm = std_dset_obj[parttype].shape[0]
            if Ngas > 0 and Nstar > 0 and Ndm > 0:
                break
        if Ngas == 0 and Nstar == 0 and Ndm == 0:
            raise Exception('No dataset found for any specified particle type. Please load datasets you want to convert. See snap.loaded_datasets for reference.')

        # make base
        s = pynbody.new(gas=Ngas, star=Nstar, dm=Ndm)

        # add datasets
        std_dset_name_to_pynbody_name = {
            'pos': 'pos', 
            'vel': 'vel', 
            'mass': 'mass', 
            'dens': 'rho', 
            'temp': 'temp', 
            'smooth': 'smooth', # add more as needed
        }
        for std_dset_name, std_dset_obj in self._loaded_standard_datasets.items():
            pynbody_name = std_dset_name_to_pynbody_name.get(std_dset_name, None)
            if pynbody_name is None:
                continue # if no support for this dataset, skip it
            for parttype in self.particle_types:
                if parttype not in [gas_parttype, star_parttype, dm_parttype]:
                    continue
                dset = getattr(self, std_dset_name)[parttype]
                value, pynbody_unit = u.get_value(dset, default_unit=1), u.to_pynbody(u.get_unit(dset, default_unit=1))
                if parttype == gas_parttype:
                    _type = 'gas'
                    s.gas[pynbody_name] = value
                    s.gas[pynbody_name].units = pynbody_unit
                elif parttype == star_parttype:
                    _type = 'star'
                    s.star[pynbody_name] = value
                    s.star[pynbody_name].units = pynbody_unit
                elif parttype == dm_parttype:       
                    _type = 'dm'
                    s.dm[pynbody_name] = value
                    s.dm[pynbody_name].units = pynbody_unit
                if verbose:
                    print(f"Added dataset {std_dset_name} attribute as {pynbody_name} dataset to pynbody snapshot for parttype {_type}.")
        return s
    setattr(cls, 'to_pynbody', to_pynbody)

    def to_yt(self, verbose=False):
        """Convert all loaded datasets to a yt dataset for use with yt's analysis and visualization tools. Add support as needed."""
        import yt
        data = {key: u.to_yt(getattr(self, self._get_dataset_attr_name(*key))) for key in self.loaded_datasets}
        ds = yt.load_uniform_grid(
            data, 
            (64, 64, 64),  # dims -> TODO MAYBE: should infer from data (make function for this), should also add length_unit and bbox
            nprocs=1    # generalize?
        )
        return ds
    setattr(cls, 'to_yt', to_yt)


    ## Visualization methods
    def quicklook(self, qty=None, parttype=None, Lbox=None, dim=1000, dir='z', logscale=True, show=True, out=None):
        """Quickly visualize a projection of the snapshot quantity you want."""
        if dir not in ['x', 'y', 'z']:
            raise ValueError('dir must be x, y, or z.')
        dirs = [0, 1, 2]
        dirs.remove('xyz'.index(dir)) # convert to axis index
        if qty is None:
            qty = 'mass'
        if parttype is None:
            parttype = 0
        if Lbox is None:
            # try to estimate a reasonable box size
            est_cen = self.pos[parttype].mean(axis=0)[np.newaxis, :]
            est_r2 = ((self.pos[parttype] - est_cen)**2).sum(axis=1)
            r_mean = np.sqrt(est_r2).mean()
            r2_mean = est_r2.mean()
            r_std = np.sqrt(r2_mean - r_mean**2)
            Lbox = r_mean + r_std
        qty_arr = self[(parttype, qty)] if (parttype, qty) in self else getattr(self, qty)[parttype]
        Lbox = u.to_nice_units(Lbox, default_unit=self.pos[parttype].unit) # estimate a good unit for the image
        pos = self.pos[parttype].to(Lbox.unit) # TODO: allow for the sustained unit conversion of datasets (through e.g. self.pos.to()) - though, likely don't use here.
        maxs = u.get_value(np.ones(2) * u.to_unit(Lbox, pos.unit, pos.unit)/2.0)
        mins = -maxs
        g = grid.bin_particles_direct(
            pos.value[:, dirs], 
            qty_arr.value, 
            mins, 
            maxs, 
            (dim, dim)
        )
        if logscale:
            g = np.log10(g)
        if out is not None:
            show=False
        jv.imshow(g.T, 
                  origin='lower', 
                  extent=(mins[0], maxs[0], mins[1], maxs[1]), 
                  clf_before=True,
                  aspect='equal', 
                  xlabel='X Position ({})'.format(pos.unit), 
                  ylabel='Y Position ({})'.format(pos.unit), 
                  title='parttype {} {} projection along {} direction'.format(parttype, qty, dir), 
                  colorbar_label='{} {}'.format(qty, '(' + qty_arr.unit + ')' if qty in self.available_datasets and qty_arr.unit != u.dimensionless_unscaled else ''),
                  show=show,
                  out=out,
        )
    setattr(cls, 'quicklook', quicklook)

    def projection_plot(self, qty='mass', Lbox=None, dim=1000, dir='z'):
        """More detailed visualization of the snapshot using matplotlib. Add support for more visualizations as needed."""
        pass # TODO

    # Basic lazy-loaded tree structure for position if needed
    def tree(self, particle_type=0, dim=None, leafsize=16, fast_build=False): # fast build will cost more in query time, use only if you need to build the tree more than you query it
        """A lazy-loaded tree structure overlayed on the position arrays of a given particle type."""
        _name = '_tree_' + str(self._resolve_particle_type_number(particle_type)) + '_' + str(dim)
        if not hasattr(self, _name):
            # make custom class that inherits from cKDTree but adds units
            from scipy.spatial import cKDTree
            class cKDTree_wUnits(cKDTree):
                def __init__(self, data, leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None):
                    super().__init__(data.value, leafsize=leafsize, compact_nodes=compact_nodes, copy_data=copy_data, balanced_tree=balanced_tree, boxsize=boxsize)
                    self.data_unit = data.unit

                def count_neighbors(self, other, r, *args, **kwargs):
                    if not isinstance(other, cKDTree_wUnits):
                        raise ValueError("Other must be an instance of cKDTree_wUnits.")
                    if other.data_unit != self.data_unit:
                        raise ValueError("Other must have the same data unit as self.") # in principle, could convert units, but ill do that if i need it
                    r_value = u.get_value_in_unit(r, self.data_unit, default_unit=self.data_unit)
                    return super().count_neighbors(other, r_value, *args, **kwargs)

                def query(self, x, *args, **kwargs):
                    x_value = u.get_value_in_unit(x, self.data_unit, default_unit=self.data_unit)
                    ret = super().query(x_value, *args, **kwargs)
                    return (ret[0] * self.data_unit, *ret[1:])
                
                def query_ball_point(self, x, r, *args, **kwargs):
                    x_value = u.get_value_in_unit(x, self.data_unit, default_unit=self.data_unit)
                    r_value = u.get_value_in_unit(r, self.data_unit, default_unit=self.data_unit)
                    return super().query_ball_point(x_value, r_value, *args, **kwargs)
                
                def query_ball_tree(self, other, r, *args, **kwargs):
                    if not isinstance(other, cKDTree_wUnits):
                        raise ValueError("Other must be an instance of cKDTree_wUnits.")
                    if other.data_unit != self.data_unit:
                        raise ValueError("Other must have the same data unit as self.") # in principle, could convert units, but ill do that if i need it
                    r_value = u.get_value_in_unit(r, self.data_unit, default_unit=self.data_unit)
                    return super().query_ball_tree(other, r_value, *args, **kwargs)
                
                def sparse_distance_matrix(self, other, r, *args, **kwargs):
                    if not isinstance(other, cKDTree_wUnits):
                        raise ValueError("Other must be an instance of cKDTree_wUnits.")
                    if other.data_unit != self.data_unit:
                        raise ValueError("Other must have the same data unit as self.") # in principle, could convert units, but ill do that if i need it
                    r_value = u.get_value_in_unit(r, self.data_unit, default_unit=self.data_unit)
                    return super().sparse_distance_matrix(other, r_value, *args, **kwargs) * self.data_unit
                
                def query_pairs(self, r, *args, **kwargs):
                    r_value = u.get_value_in_unit(r, self.data_unit, default_unit=self.data_unit)
                    return super().query_pairs(r_value, *args, **kwargs)
            
            if dim is None:
                setattr(self, _name, 
                        cKDTree_wUnits(getattr(self, 'pos' + str(self._resolve_particle_type_number(particle_type))), 
                                    leafsize=leafsize, compact_nodes=(not fast_build), balanced_tree=(not fast_build), copy_data=False))
            elif isinstance(dim, str) and dim in ['x', 'y', 'z']:
                dim_idx = 'xyz'.index(dim)
            elif isinstance(dim, int):
                setattr(self, _name, 
                        cKDTree_wUnits(getattr(self, 'pos' + str(self._resolve_particle_type_number(particle_type)))[:, dim], 
                                    leafsize=leafsize, compact_nodes=(not fast_build), balanced_tree=(not fast_build), copy_data=False))
            else:
                raise ValueError("dim must be None, an integer index, or one of 'x', 'y', or 'z'.")
        return getattr(self, _name)
    setattr(cls, 'tree', tree)

    return cls







@_add_convenience_properties
class GIZMO_Snapshot(Basic_GIZMO_Snapshot):
    """
    Convenience Wrapper Class around GIZMO snapshot to quickly access common particle data. 
    """
    _TRANSFORMATION_BEHAVIORS = { # TODO: combine with units eventually? plus add unit behaviour?
        # FORMAT:
        #  X : (transforms_like, centers_like)
        # with 
        #   1) X = dataset_name
        #   2) X = (particle_type, dataset_name) 
        # If format 1 is used, all particle types will be treated the same way. Format 2 will overwrite format 1 for a specific particle type.
        # If not specified, the default behavior is to treat the dataset as a scalar and not to center it.
        # Passing None in any value sets it to its default behaviour.
        'Velocities': ('vector', 'velocity'),
        'Acceleration': ('vector', None),
        'MagneticField': ('vector', None),
    }

    _STANDARD_DSET_POSSIBLE_DATASET_NAMES = { 
        # when one of these is called for the first time, it will go through all possible versions of these and choose the first one that exists in 
        # the snapshot and use that one moving forward. An error will be raised if more or less than 1 of the names exist in the snapshot.
        'pos': ('Coordinates',),
        'vel': ('Velocities',),
        'acc': ('Acceleration',),
        'mass': ('Masses',),
        'dens': ('Density',),
        'temp': ('Temperature',),
        'pot': ('Potential',),
        'smooth': ('SmoothingLength','KernelMaxRadius'),
        'mag': ('MagneticField',),
        'metal': ('Metallicity',),
        'fe': ('ElectronAbundance',),
    }
    # _STANDARD_DSET_ALIASES = { # will be converted to lowercase first # TODO ?
    #     'position': 'pos',
    #     'velocity': 'vel',
    #     'acceleration': 'acc',
    #     'hsml': 'smooth',
    #     'density': 'dens',
    #     'temperature': 'temp',
    #     'potential': 'pot',
    #     'electron_abundance': 'fe',
    #     'metallicity': 'metal',
    #     'magnetic_field': 'mag',
    #     'b_field': 'mag',
    # }





def load_gizmo(filepath, debugging=False):
    """Loads GIZMO into a Snapshot for use."""
    if debugging:
        print("--- HDF5 File Dump ---")
        hdf5.explore(filepath)
    snap = GIZMO_Snapshot(filepath)
    return snap


def load(filepath, verbose=False):
    """General load function that attempts to determine the appropriate snapshot/simulation class to use based on the file format and metadata."""
    # for now, just assume it's always GIZMO, but in the future we can add more logic here to determine the appropriate class
    if verbose:
        print(f"Loading GIZMO snapshot from file {filepath} ...")
    return load_gizmo(filepath, debugging=verbose)


