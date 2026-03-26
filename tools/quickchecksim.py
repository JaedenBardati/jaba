#!/usr/bin/python3
"""
Runs a quick check on simulations to determine if they are reasonable to continue running.
  Also has a variety of convience functions for follow-up interative analysis.
  Goal is to be an all-in-one package with minimal to no uncommon packages. 
  Requires (brackets are optional): numpy, matplotlib, astropy, (h5py), (pandas)
  Tested on python 3.7
Jaeden Bardati 2025+

Basic class/function structure:
  FILETYPE LOADER (e.g. HDF5_Snapshot) --> SNAPSHOT TYPE CONVENIENCE FUNCTION (e.g. GIZMO_Snapshot) --> STANDARD SNAPSHOT FORM (e.g. Standardized_GIZMO_Snapshot)
  QUICK CHECK ANALYSIS <-- GENERAL ANALYSIS + GENERAL PLOTTING

Currently implemented checks:
  - Mass density radial plot
  - Particle number radial plot
  - Simple maps of mass density
  - ...

TO DO: 
 - Integrate into jaba package
 - Add more checks and more plotting functions  
"""
if __name__ == "__main__":
    print('loading packages...')

## Builtin packages
import glob, sys, time, warnings
import os.path

## Required packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')                    # non-interactive mode default
from astropy import units
from astropy import constants

## Warning about optional packages
# import importlib.util
# if not importlib.util.find_spec('h5py'):
#     warnings.warn('h5py package not found - please install it in your python environment if you want to use GIZMO loading functionality.')
# if not importlib.util.find_spec('pandas'):
#     warnings.warn('pandas package not found - please install it in your python environment if you want to use CSV loading functionality.')


########################################################################################################################################################################
#####################                                                       FILE ARGUMENTS                                                         #####################
########################################################################################################################################################################
### Based on https://gist.github.com/JaedenBardati/81c4543b84a49584ea09bf529fbdf29c

"""
A simple file argument handler which checks format.

Example usage: 
                Call in x.py 					                        -->			Return 			                <--		File Argument Call
    get_file_arguments(str, int, float, fill_empties_with_none=True) 	--> ("abc", 1, 3.14) 				        <-- "python x.py abc 1 3.14" was called.
    get_file_arguments(str, int, float, fill_empties_with_none=True) 	--> ("abc", None, None)				        <-- "python x.py abc" was called.
    get_file_arguments(i=str, o=str, require_options=True) 			    --> {i="in", o="out"}			         	<-- "python x.py i in -o out" was called.
    get_file_arguments(i=str, o=str, require_options=True) 			    --> {i=("in", "in2"), o=("out", "out2")}	<-- "python x.py i in in1 -o out out2" was called.
    get_file_arguments(int, i=str, o=str) 					            --> (1), {i="in"}				            <-- "python x.py 1 -i in" was called.
"""

#import sys

class _ArgumentForm:
    """Class that handles the form for a given argument, or arguments with repeated form."""
    
    class IncorrectArgumentForm(Exception):
        pass
    
    def __init__(self, *args, name=None):
        """
        Enter at least one of the following unordered arguments: 
            > type: The type of the argument (e.g. int). None means all types are allowed. Identified by the type "type".
            > num: The number of arguments (e.g. 3). None means any number of arguments are allowed. Identified by the type "int".
        Also, an optional "name" keyword parameter is used to identify the argument form from others.
        """
        if len(args) <= 0: raise TypeError('There must be at least one format argument.')
        if len(args) > 2: raise TypeError('There must be at most two format arguments.')
        self.name = None if name is None else str(name)  # name must be a string (or something that can easily be turned into one)!

        self.type = None	# Type determines the type of the argument. None means all types are allowed.
        self.num = None		# Integer determines the number of arguments this represents. None means any number of arguments are allowed. 
        for arg in args:
            if type(arg) is type:
                if self.type is None:
                    self.type = arg
                else:
                    raise TypeError('There must be at most one "type" type argument.')
            elif type(arg) is int:
                if self.num is None:
                    self.num = arg
                else:
                    raise TypeError('There must be at most one "int" type argument.')
            else:
                raise TypeError('There must be only "type" or "int" type arguments entered.')
        if self.type is None and self.num is None:
            raise TypeError('There must be at least one type or integer format argument.')
            
    def check_form(self, *args, fill_empties_with_none=False):
        """Checks if the entered argument(s) is in the correct form."""
        args = list(args)
        num_empties = 0
        if self.num is not None:
            # check number of arguments
            if len(args) < self.num:
                if fill_empties_with_none:
                    num_empties = self.num - len(args)
                else:
                    raise _ArgumentForm.IncorrectArgumentForm('Not enough arguments for option "{}". Requires {} argument(s) and instead {} were entered.'.format(self.name, self.num, len(args)))
            elif len(args) > self.num:
                raise _ArgumentForm.IncorrectArgumentForm('Too many arguments for option "{}". Requires {} argument(s) and instead {} were entered.'.format(self.name, self.num, len(args)))

        # check if each argument has the right type
        for i in range(len(args)):
            try:
                args[i] = self.type(args[i])
            except:
                raise _ArgumentForm.IncorrectArgumentForm('Wrong type entered in argument "{}". It must have type: {}.'.format(args[i], self.type))
        
        result = tuple(list(args) + [None]*num_empties)
        return result[0] if len(result) == 1 else result  # add any empties as None at the end


class _FullArgumentForm:
    """Class that handles "full" argument forms potentially consisting of multiple types and of option flags."""
    def __init__(self, *form):
        """The arguments must have type _ArgumentForm. Option _ArgumentForms are identified by their name property. """
        try:
            for f in form:
                if type(f) is not _ArgumentForm:
                    raise TypeError
        except TypeError:
            raise TypeError("Form must have elements of type _ArgumentForm.")

        option_names = [f.name for f in form if f.name is not None]
        if len(set(option_names)) != len(option_names): 
            raise TypeError('There must not be any repeated options in the form definition.')
        if None in option_names and option_names.index(None) == list(reversed(option_names)).index(None):
            raise TypeError('All non-options must be at the beginning of form.')

        for f in form:
            if f.num != 1 and f.name == None: 
                raise NotImplementedError('At the moment, multiple argument _ArgumentForms are only supported for options.')

        self.form = form

    def check_form(self, *args, fill_empties_with_none=False, require_options=False):
        """Checks if the entered argument(s) is in the correct form."""
        # split args into main forms and option forms
        form_option_names = [f.name for f in self.form if f.name is not None]  # options are identified by their name property
        arg_option_names = []
        option_indices = [-1]  # "option" at -1 ==> begins at 0
        for i, arg in enumerate(args):  # for each argument
            if str(arg) in form_option_names and arg is not None:  # is this an option flag?
                option_indices.append(i)  # if so, append where it is in the arugment
                arg_option_names.append(str(arg))  # and save the option name that it is also
        option_indices.append(len(args))
        
        if len(set(arg_option_names)) != len(arg_option_names):
            raise _ArgumentForm.IncorrectArgumentForm('There must not be more than one of the same option flag.')

        if require_options and len(form_option_names) != len(arg_option_names):
            raise _ArgumentForm.IncorrectArgumentForm('Not all option flags are present, but they are required to.')
        
        split_args = [args[option_indices[i]+1:option_indices[i+1]] for i in range(len(option_indices) - 1)]

        # first split (always non-options)
        first_args = split_args[0]
        first_form = [f for f in self.form if f.name is None]
        first_form_num = sum([f.num for f in first_form])  # assumes that f.num != None

        num_empties = 0
        if len(first_args) < first_form_num: # check number of arguments
            if fill_empties_with_none:
                num_empties = first_form_num - len(first_args)
            else:
                raise _ArgumentForm.IncorrectArgumentForm("Not enough required arguments. Requires {} argument(s) and instead {} were entered.".format(first_form_num, len(first_args)))
        elif len(first_args) > first_form_num:
            raise _ArgumentForm.IncorrectArgumentForm("Too many required arguments. Requires {} argument(s) and instead {} were entered.".format(first_form_num, len(first_args)))

        return_tuple = []
        for arg, argform in zip(first_args, first_form):
            return_tuple.append(argform.check_form(arg))
        return_tuple = tuple(return_tuple + [None]*num_empties)  # add any empties as None at the end

        # do all the options now
        return_dict = {}
        for option_name, split_args in zip(arg_option_names, split_args[1:]):
            split_form = [f for f in self.form if f.name == option_name]
            if len(split_args) != 0:
                arg = split_args
                assert len(split_form) == 1  # this is just assumed right below
                return_dict[option_name] = split_form[0].check_form(*arg, fill_empties_with_none=fill_empties_with_none)
            else:
                if fill_empties_with_none or split_form[0].num is None:
                    return_dict[option_name] = None
                else:
                    raise _ArgumentForm.IncorrectArgumentForm("Not enough arguments for option {}. Requires {} argument(s) and instead {} were entered.".format(option_name, split_form[0].num, len(split_args)))

        if len(return_dict) == 0:
            return return_tuple
        elif len(return_tuple) == 0:
            return return_dict
        else:
            return return_tuple, return_dict


class _FileArguments:
    """Class that handles getting and checking the arguments to a python file."""

    def __init__(self, form=None, fill_empties_with_none=False, require_options=False,  check=False, args=None):
        """Initializes the contraints on the arguments. Form must have type _FullArgumentForm."""
        if form is not None and type(form) is not _FullArgumentForm:
            raise TypeError("Form must be None or have type _FullArgumentForm.")
        if args is not None:
            try:
                list(args)
            except TypeError:
                raise TypeError('Manually entering arguments requires that they are iterable.')

        self.form = form
        self.fill_empties_with_none = fill_empties_with_none
        self.require_options = require_options

        self.args = args
        self._args_checked = False

        if check: 
            self.get_args()
    
    def _get_args(self):
        """Gets the arguments without checking them"""
        if self.args is None:
            self.args = sys.argv[1:]
        return self.args

    def _check_args(self):
        """Checks the arguments for if they are compatible with the desired contraints"""
        assert self.args is not None, 'The arguments must be defined before they are checked.'
        if not self._args_checked:
            if self.form is not None:
                self.args = self.form.check_form(*self.args, fill_empties_with_none=self.fill_empties_with_none, require_options=self.require_options)
        self._args_checked = True
        
    def get_args(self):
        """Gets the arguments and checks if they are of the right form."""
        self._get_args()
        self._check_args()
        return self.args


def get_file_arguments(*form, fill_empties_with_none=False, require_options=False, option_prefix='-', **options_form):
    """Shortens getting the file arguments. This is the main function to call."""
    # construct the form objects
    form = list(form)
    for i, f in enumerate(form):
        try:
            (*f,)
            raise NotImplementedError('Currently no support for fancy formatting in non-options form.')
        except TypeError: # if not iterable
            f = (f,1)
        form[i] = _ArgumentForm(*f)
    for k, v in options_form.items():
        try:
            (*v,)
        except TypeError: # if not iterable
            v = (v,)
        options_form[k] = _ArgumentForm(*v, name=option_prefix+k) # overwrite previous one
        
    form = list(form) + list(options_form.values())
    fullform = _FullArgumentForm(*form)

    # find, check and return arguments
    return _FileArguments(fullform, fill_empties_with_none=fill_empties_with_none, require_options=require_options, check=True).args



########################################################################################################################################################################
#####################                                                      CODE TIMER                                                              #####################
########################################################################################################################################################################
### Based on https://gist.github.com/JaedenBardati/e953033508000f637a4121982429a56e

"""
A simple timer package that implements timed debugging. 
The objective was to make something simple and easy to implement in existing code quickly.
Basic Usage:
  log_timing("Doing something important")        # begins the timer and prints the string entered.
  ## insert code to time...
  
  log_timing("Doing something else important")   # ends the timer, prints the time, and starts the next timer with the string entered
  ## insert code to time...
  
  ## More timers ...
  
  log_timing()                                   # ends the timer and prints the time since the last call
See tests for more examples.
"""

#import time


class Timer():
  """Times stuff."""
  
  def __init__(self, start_now=None, start_text=None, end_text=None, logger=None):
    if start_now is None: start_now = False
    if start_text is None: start_text = "Starting timer . . ."
    if end_text is None: end_text = "took {:0.2f} seconds.\n"
    if logger is None: logger = print
    
    self.start_text = start_text
    self.end_text = end_text
    self.logger = logger
    self._start_time = None
    
    if start_now: self.start()
    
  def _get_time(self):
    return time.time()
  
  def elasped_time(self):
    return self._get_time() - self._start_time
  
  def start(self):
    if self.logger: self.logger(self.start_text)
    self._start_time = self._get_time()
  
  def _stop(self):
    elapsed_time = self.elasped_time()
    if self.logger: self.logger(self.end_text.format(elapsed_time))
  
  def stop(self):
    self._stop()
    self._start_time = None
  
  def update(self, start_now=True, start_text=None, end_text=None, logger=None):
    self._stop()
    self.__init__(start_now  = self.start_now  if start_now  is None else start_now, 
                  start_text = self.start_text if start_text is None else start_text, 
                  end_text   = self.end_text   if end_text   is None else end_text, 
                  logger     = self.logger     if logger     is None else logger)
  


def create_global_timer(start_text=None, log_it=True, start_now=True, **kwargs):
  if log_it:
    global _TIMER
    _TIMER = Timer(start_now=start_now, start_text=start_text, **kwargs)


def update_global_timer(start_text=None, log_it=True, **kwargs):
  if log_it:
    global _TIMER
    _TIMER.update(start_text=start_text, **kwargs)


def stop_global_timer(log_it=True, **kwargs):
  if log_it:
    if "_TIMER" in globals():
      global _TIMER
      _TIMER.stop(**kwargs)
      del globals()["_TIMER"]


def globally_time(function, *args, **kwargs):
  if function == 0:
    create_global_timer(*args, **kwargs)
  elif function == 1:
    update_global_timer(*args, **kwargs)
  elif function == 2:
    stop_global_timer(*args, **kwargs)
  else:
    raise Exception("You need to enter 0, 1, or 2 for function.")


def log_timing(start_text=None, log_it=True, **kwargs):
  if start_text is not None:
    if "_TIMER" not in globals():
      create_global_timer(start_text=start_text, log_it=log_it, **kwargs)
    else:
      update_global_timer(start_text=start_text, log_it=log_it, **kwargs)
  else:
    stop_global_timer(log_it=log_it, **kwargs)
  


########################################################################################################################################################################
#####################                                               GENERAL HDF5 LOADING                                                           #####################
########################################################################################################################################################################
### Based on https://gist.github.com/JaedenBardati/58d492e0427793e41d9da289865ef327

"""
A simple HDF5 loader.

Examples/Tests:

explore_h5('blackhole_mergers.hdf5')
explore_h5('snap_042')
get_dataset_h5('blackhole_mergers.hdf5', 'details/mass')
get_dataset_h5('snap_042', 'PartType5/Coordinates')
"""

#import os.path
#import glob

#import h5py
#import numpy as np

class OutputH5List(list):
    pass

def _handle_file_path_input_h5(func):
    # This decorator handles handles any multi-file file path
    # cases (pass name without extension for multi-file support).
    EXTS_SUPPORTED = ('.hdf5', '.h5')
    def _wrapper(file_path, *args, **kwargs):
        file_path_name, file_path_ext = os.path.splitext(file_path)
        if file_path_ext:
            # if there is an extension, handle single-file
            # just return the function result on that file
            return func(file_path, *args, _idx=None, **kwargs)
        # otherwise, check if there is a single file with the name that
        # was passed and a recognized extension 
        file_path_singlefile_list = [file_path_name + ext for ext in EXTS_SUPPORTED]
        isfile_singlefile_list = [os.path.isfile(fp) for fp in file_path_singlefile_list]
        issinglefile = any(isfile_singlefile_list)
        # also check for the multi-file format (by looking for any file in that form)
        file_path_ext_pair_multifile_list = [(x, ext) for ext in EXTS_SUPPORTED for x in glob.glob(file_path_name + '.[0-9]*' + ext)]
        isfile_multifile_list = [os.path.isfile(fp[0]) for fp in file_path_ext_pair_multifile_list]
        ismultifile = len(isfile_multifile_list) != 0
        if not (issinglefile or ismultifile):
            raise FileNotFoundError('No file exists named "{}" that fits the supported format.'.format(file_path))
        if not ismultifile and sum(isfile_singlefile_list) == 1:
            # if there is only a single file with an hdf5 extension, just use that.
            _file_path = file_path_singlefile_list[isfile_singlefile_list.index(True)]
            return func(_file_path, *args, _idx=None, **kwargs)
        if ismultifile and not issinglefile and all(fp[1] == file_path_ext_pair_multifile_list[0][1] for fp in file_path_ext_pair_multifile_list[1:]):
            # if there are no single file formats and not different extensions, handle multiple files
            # loop over all files and return list of function results
            indices = [int(fp[:-len(ext)].split('.')[-1]) for fp, ext in file_path_ext_pair_multifile_list]
            outs = []
            for idx, (fp, _) in sorted(zip(indices, file_path_ext_pair_multifile_list)):
                outs.append(func(fp, *args, _idx=idx, **kwargs))
            return OutputH5List(outs)
        else:
            raise ValueError("Ambiguous file path: multiple hdf5 files found with file name '{}' without all having the standard multi-file structure.".format(file_path))
    return _wrapper

def _only_open_first_h5_file(func):
    @_handle_file_path_input_h5
    def _wrapper(*args, _idx=None, **kwargs):
        if _idx is None or _idx == 0:
            return func(*args, **kwargs)
        return None
    return _wrapper

def _combine_output_multi_h5_file_arrays(func):
    def _wrapper(*args, **kwargs):
        outs = _handle_file_path_input_h5(func)(*args, **kwargs)
        if isinstance(outs, OutputH5List):
            return np.concatenate(outs, axis=0)
        else:
            return outs
    return _wrapper

@_only_open_first_h5_file
def explore_h5(file_path):
    import h5py
    with h5py.File(file_path, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")
                for k in f[name].attrs.keys():
                    print(f"  Attribute: {k} = {f[name].attrs[k]}")
        print("All Groups: ", list(f.keys()))
        f.visititems(print_structure)
        pass

@_combine_output_multi_h5_file_arrays
def get_dataset_h5(file_path, dataset_path, _idx=None, dtype=np.float64):
    import h5py
    with h5py.File(file_path, 'r') as f:
        data = np.array(f[dataset_path], dtype=dtype)
    return data

@_only_open_first_h5_file
def get_attributes_h5(file_path, group):
    import h5py
    with h5py.File(file_path, 'r') as f:
        attrs = dict(f[group].attrs)
    return attrs

@_only_open_first_h5_file
def get_group_names_h5(file_path):
    import h5py
    with h5py.File(file_path, 'r') as f:
        groups = list(f.keys())
    return groups

@_only_open_first_h5_file
def get_dataset_names_h5(file_path, group):
    import h5py
    with h5py.File(file_path, 'r') as f:
        dataset_names = list(f[group].keys())
    return dataset_names



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
            self._groups = get_group_names_h5(self.filepath)
            for g in self._groups:
                if g not in self._KNOWN_GROUPS:
                    warnings.warn('Unknown group "{}" found. Please update quickchecksim to support it. Ignoring for now...'.format(g))
        return self._groups

    @property
    def metadata(self):
        if not hasattr(self, '_attributes'):
            self._attributes = get_attributes_h5(self.filepath, self._HEADER_NAME)
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
            _dataset_names = get_dataset_names_h5(self.filepath, particle_type)
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
            data = get_dataset_h5(self.filepath, dataset_path)
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
        'Acceleration': ('acceleration', units.cm/units.s**2),
        'Coordinates': ('length', units.cm),
        'Density': ('density', units.g/units.cm**3),
        'DensityGradient': ('density_grad', units.g/units.cm**4),
        'DustToGasRatio_Local': (None, units.dimensionless_unscaled),
        'Dust_Temperature': (None, units.K),
        'EddingtonTensor': (None, units.dimensionless_unscaled),
        'ElectronAbundance': (None, units.dimensionless_unscaled),
        'HydroAcceleration': ('acceleration', units.cm/units.s**2),
        'HII': (None, units.dimensionless_unscaled),
        'IRBand_Radiation_Temperature': (None, units.K),
        'InternalEnergy': (None, (units.cm/units.s)**2),
        'MagneticField': ('magnetic_field', units.G),
        'Masses': ('mass', units.g),
        'Metallicity': (None, units.dimensionless_unscaled),
        'MolecularMassFraction': (None, units.dimensionless_unscaled),
        'NeutralHydrogenAbundance': (None, units.dimensionless_unscaled),
        'ParticleChildIDsNumber': (None, units.dimensionless_unscaled),
        'ParticleIDGenerationNumber': (None, units.dimensionless_unscaled),
        'ParticleIDs': (None, units.dimensionless_unscaled),
        'PhotonEnergy': ('energy', (units.g*units.cm/units.s)**2),
        'PhotonFluxDensity': ('flux_density', units.cm**-2*units.s**-1),
        'PhotonOpacity': ('opacity', units.cm**2/units.g),
        'Potential': ('internal_energy', (units.cm/units.s)**2),
        'Pressure': ('pressure', (units.cm/units.s)**2/units.cm**3),
        'RadiativeAcceleration': ('acceleration', units.cm/units.s**2),
        'SmoothingLength': ('length', units.cm),
        'SoundSpeed': ('velocity', units.cm/units.s),
        'StarFormationRate': (None, units.Msun/units.yr),  # fixed output units of Msun/yr
        'StellarFormationTime': ('time', units.s),
        'Temperature': (None, units.K),
        'Velocities': ('velocity', units.cm/units.s),
        'VelocityGradient': ('velocity_grad', units.s**-1),
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
                self._BoxSize = float(self.metadata['BoxSize']) * self.unit_system['length'] * units.cm
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
            raw_data = super()._get_dataset(particle_type, dataset_name, nosave=True)
            if dataset_name not in self._DATASET_UNIT_DICT.keys():
                warnings.warn('Dataset "{}" units not recognized. Please adapt GizmoDataset._DATASET_UNIT_DICT to include its units. For now, assuming it is unitless...'.format(dataset_name))
                return raw_data * units.dimensionless_unscaled
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
        explore_h5(filepath)
    snap = GIZMO_Snapshot_ConvenientWrapper(filepath)
    return snap


########################################################################################################################################################################
#####################                                            LOAD IN SKIRT ASCII FORMAT                                                        #####################
########################################################################################################################################################################
### Taken from https://github.com/JaedenBardati/skirt-datacube

def load_dat_file(filename, chunksize=None):
    """Function that loads a .dat file in the format of SKIRT input/output."""
    import pandas as pd
    
    # get header
    header = {}
    firstNonCommentRowIndex = None
    with open(filename) as file:
        for i, line in enumerate(file):
            l = line.strip()
            if l[0] == '#':
                l = l[1:].lstrip()
                if l[:6].lower() == 'column':
                    l = l[6:].lstrip()
                    split_l = l.split(':')
                    assert len(split_l) == 2 # otherwise, unfamiliar form!
                    icol = int(split_l[0]) # error here means we have the form: # column %s, where %s is not an integer
                    l = split_l[1].lstrip() # this should be the column name
                    header[icol] = l
            else:
                firstNonCommentRowIndex = i
                break
    assert firstNonCommentRowIndex is not None # otherwise the entire file is just comments

    # set up column names
    if firstNonCommentRowIndex == 0:
        columns = None
    else:
        columns = [None for i in range(max(header.keys()))]
        for k, v in header.items(): columns[k-1] = v
        assert None not in columns # otherwise, missing column 
    
    # get data
    df = pd.read_csv(filename, delim_whitespace=True, skiprows=firstNonCommentRowIndex, header=None, names=columns, chunksize=chunksize)
    
    return df


# ...

########################################################################################################################################################################
#####################                                     ANALYSIS AND PLOTTING FUNCTIONS                                                          #####################
########################################################################################################################################################################


def get1Dmean(x, qty, weight=None, nbins=100, xmin=None, xmax=None, xlog=False, sum_instead=False, cumsum_instead=False, reverse_x=False):  
    """
    Plots a general 1d plot. Not super efficient.
    """
    assert not sum_instead or weight is None, 'Weighted sum not supported.'
    assert not cumsum_instead or weight is None, 'Weighted cummulative sum not supported.'
    assert not sum_instead or not cumsum_instead, 'Cannot have both sum_instead=True and cumsum_instead=True'

    x = np.asarray(x)
    qty = np.asarray(qty)
    if weight is not None:
        weight = np.asarray(weight)

    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)

    if xlog:
        xbins = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
    else:
        xbins = np.linspace(xmin, xmax, nbins+1)

    if reverse_x:
        xbins = xbins[::-1]

    qty_bins = np.zeros(nbins)
    if cumsum_instead:
        cumsum = 0.0
    for i in range(nbins):
        if i == 0:
            mask = np.logical_and(xbins[i] <= x, x <= xbins[i+1])
        else:
            mask = np.logical_and(xbins[i] < x, x <= xbins[i+1])

        if np.any(mask):
            if weight is None:
                if sum_instead:
                    qty_bins[i] = np.sum(qty[mask])
                elif cumsum_instead:
                    binsum = np.sum(qty[mask])
                    qty_bins[i] = cumsum+binsum
                    cumsum += binsum
                else:
                    qty_bins[i] = np.mean(qty[mask])
            else:
                qty_bins[i] = np.average(qty[mask], weights=weight[mask])
        else:
            qty_bins[i] = np.nan

    x_mid = 0.5*(xbins[1:] + xbins[:-1])
    return x_mid, qty_bins


def basic_figure_wrapper(plotting_function):
    """General wrapper that encapulates most of the repeated parts when plotting."""
    def wrapper_function(*args, 
                         fig=None, ax=None, figsize=(6.4, 4.8), 
                         xlog=False, ylog=False, xlabel=None, ylabel=None, title=None,
                         xmin=None, xmax=None, ymin=None, ymax=None,
                         show_legend=False, legend_frameon=False, legend_loc=0, 
                         tight_layout=True, clf_before=False, clf_after=None, out=None, show=False, dpi=300, 
                         fontsize=10, axes_linewidth=1.25, legend_fontsize=None,
                         major_ticks_on=None, minor_ticks_on=None, tick_direction='in',
                         major_ticksize=8, minor_ticksize=4, major_tickwidth=1.5, minor_tickwidth=1.5,
                         xtick_bottom=True, xtick_top=True, ytick_left=True, ytick_right=True, 
                         **kwargs):
        if (fig is None) != (ax is None):
            raise ValueError('If fig or ax is inputted, the other should be also.')
        if clf_before:
            plt.clf()
        
        with matplotlib.rc_context({
            'xtick.direction': tick_direction,
            'ytick.direction': tick_direction,
            'font.size': fontsize,
            'axes.linewidth': axes_linewidth,
            'xtick.major.size': major_ticksize,
            'ytick.major.size': major_ticksize,
            'xtick.minor.size': minor_ticksize,
            'ytick.minor.size': minor_ticksize,
            'xtick.major.width': major_tickwidth,
            'ytick.major.width': major_tickwidth,
            'xtick.minor.width': minor_tickwidth,
            'ytick.minor.width': minor_tickwidth,
            'xtick.bottom': xtick_bottom,
            'xtick.top': xtick_top,
            'ytick.left': ytick_left,
            'ytick.right': ytick_right,
        }):
            if fig is None and ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            
            plotting_function(*args, ax=ax, **kwargs)
            
            if title is not None:
                ax.set_title(title)
            if minor_ticks_on is not None:
                if minor_ticks_on:
                    ax.minorticks_on()
                else:
                    ax.minorticks_off()
            if major_ticks_on is not None:
                if major_ticks_on:
                    ax.majorticks_on()
                else:
                    ax.majorticks_off()
            if xmin is not None or xmax is not None:
                ax.set_xlim([xmin, xmax])
            if ymin is not None or ymax is not None:
                ax.set_xlim([ymin, ymax])
            if xlog:
                ax.set_xscale('log')
            if ylog:
                ax.set_yscale('log')
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            if tight_layout:
                fig.tight_layout()
            if show_legend:
                ax.legend(frameon=legend_frameon, loc=legend_loc, fontsize=legend_fontsize if legend_fontsize is not None else fontsize)
            if out is not None:
                fig.savefig(out, dpi=dpi)
                if clf_after is None:
                    clf_after = True
            if show:
                plt.show()
                if clf_after is None:
                    clf_after = True
            if clf_after is None:
                clf_after = False
            if clf_after:
                plt.clf()
            return fig, ax
        
    return wrapper_function

@basic_figure_wrapper
def plot1Dline(x, y, label=None, color=None, ls='-', lw=None, ax=None):
    """General 1D plotting function for lines. Use a tuple for y to plot multiple lines."""
    if not isinstance(y, tuple):
        y = (y,)
    if not isinstance(label, tuple):
        label = (label,)
    if not isinstance(color, tuple):
        color = (color,)
    if not isinstance(ls, tuple):
        ls = (ls,)
    if not isinstance(lw, tuple):
        lw = (lw,)
    for i in range(len(y)):
        ax.plot(x, y[i], label=label[i], color=color[i], ls=ls[i], lw=lw[i])


def plot1Dmean(x, qty, weights=None, labels=None, linestyles='-', nbins=100, xlog=False, ylog=False, xlabel=None, ylabel=None, out=None, sum_instead=False, cumsum_instead=False, reverse_x=False, _fig=None, _ax=None):
    # TODO DELETE
    if not isinstance(weights, tuple):
        weights = (weights,)
    if not isinstance(labels, tuple):
        labels = (labels,)*len(weights)
    if not isinstance(linestyles, tuple):
        linestyles = (linestyles,)*len(weights)
    
    fig, ax, _out, _show_legend = _fig, _ax, None, False
    for i, w in enumerate(weights):
        _x, _y = get1Dmean(x, qty, weight=w, nbins=nbins, xlog=xlog, sum_instead=sum_instead)
        if i == len(weights) - 1 and out is not None:
            _out = out
            _show_legend = True if any(l is not None for l in labels) else False
        fig, ax = plot1Dline(_x, _y, fig=fig, ax=ax, label=labels[i], ls=linestyles[i], xlog=xlog, ylog=ylog, xlabel=xlabel, ylabel=ylabel, show_legend=_show_legend, out=_out)
    return fig, ax

########################################################################################################################################################################
#####################                                              QUICK CHECK SIMULATION                                                          #####################
########################################################################################################################################################################

def quick_check(filepath, output_dir=None, debugging=False, center_around_BH=True):
    """ THIS IS THE MAIN FUNCTION TO CHANGE..."""
    if output_dir is None:
        output_dir='.'
    output_dir+='/'

    # load snapshot/simulation
    log_timing(f"loading snapshot at {filepath} ...")
    snap = load_gizmo(filepath, debugging=True) # should likely be debugging=debugging
    
    # pre-load required data
    snap.pos0, snap.dens0, snap.mass0,

    # center around particle
    if center_around_BH:
       bh_parttype=3
       bh_index=0
       cpos = snap['PartType%d'%bh_parttype, 'Coordinates'][bh_index][np.newaxis, :]
       cvel = snap['PartType%d'%bh_parttype, 'Velocities'][bh_index][np.newaxis, :]

    # check data
    log_timing(f"running check plots ...")
    r = np.sqrt(np.sum((snap.pos0 - cpos)**2, axis=1))  # spherical radius
    plot1Dmean(r.to('pc'), snap.dens0.to('g/cm**3'), weights=(None, snap.mass0, snap.mass0/snap.dens0), 
               labels=('mass-weighted', 'volume-weighted', 'particle mean'), 
               linestyles=('-', '-', '--'),
               xlabel=r'Spherical radius $r$ [pc]', ylabel=r'Mass density $\rho$ [g/cm$^3$]', 
               xlog=True, ylog=True, out=output_dir+'dens_{}.pdf'.format(snap.name))  # mass density plot
    plot1Dmean(r.to('pc'), np.ones(len(snap.dens0)), weights=(None,), 
               labels=('particle',), 
               linestyles=('-',),
               xlabel=r'Spherical radius $r$ [pc]', ylabel=r'Particle number', 
               xlog=True, ylog=True, out=output_dir+'number_{}.pdf'.format(snap.name), sum_instead=True)  # particle number plot
    
    plt.clf()
    plt.hist(np.array(snap.mass0.to('Msun')), bins=50)
    plt.xlabel(r'Particle mass ($M_\odot$)')
    plt.savefig(output_dir+'hist_mass_{}.pdf'.format(snap.name))
    plt.clf()
    plt.hist(np.array(r.to('pc')), bins=50)
    plt.xlabel(r'Particle radius (pc)')
    plt.savefig(output_dir+'hist_radius_{}.pdf'.format(snap.name))
    plt.clf()
    plt.hist(np.array(snap.dens0.to('g/cm**3')), bins=50)
    plt.xlabel(r'Particle density (g/cm$^3$)')
    plt.savefig(output_dir+'hist_dens_{}.pdf'.format(snap.name))
    plt.clf()
    plt.hist(np.log10(np.array(snap.mass0.to('Msun'))), bins=50)
    plt.xlabel(r'log particle mass ($M_\odot$)')
    plt.savefig(output_dir+'hist_logmass_{}.pdf'.format(snap.name))
    plt.clf()
    plt.hist(np.log10(np.array(r.to('pc'))), bins=50)
    plt.xlabel(r'log particle radius (pc)')
    plt.savefig(output_dir+'hist_logradius_{}.pdf'.format(snap.name))
    plt.clf()
    plt.hist(np.log10(np.array(snap.dens0.to('g/cm**3'))), bins=50)
    plt.xlabel(r'log particle density (g/cm$^3$)')
    plt.savefig(output_dir+'hist_logdens_{}.pdf'.format(snap.name))
    plt.clf()

    # resolution plots
    _x1, _y1 = get1Dmean(r.to('pc'), snap.mass0.to('Msun'), nbins=100, xlog=True, cumsum_instead=True) # Mencl
    _x2, _y2 = get1Dmean(r.to('pc'), snap.mass0.to('Msun'), nbins=100, xlog=True) # delta m
    fig, ax = plot1Dline(_x1, _y1, label=r'$M_\mathrm{encl}$', ls='--', color='black')
    plot1Dline(_x2, _y2, fig=fig, ax=ax, label=r'$\delta m$', ls='-', color='black', xlog=True, ylog=True, xlabel='spherical radius (pc)', ylabel=r'Mass resolution ($M_\odot$)', show_legend=True, out=output_dir+'mass_resolution_{}.pdf'.format(snap.name))
    _x3, _y3 = get1Dmean(r.to('pc'), np.array((snap.mass0/snap.dens0).to('pc**3'))**(1/3.), nbins=100, xlog=True) # delta r 
    plot1Dline(_x3, _y3, label=r'$\delta r$', ls='-', color='black', xlog=True, ylog=True, xlabel='spherical radius (pc)', ylabel=r'Spatial resolution $\delta x$ (pc)', out=output_dir+'spatial_resolution_{}.pdf'.format(snap.name))


    

    # Pmag = np.sum(snap['PartType0','MagneticField'].to('G')**2, axis=1)/(8*np.pi)
    # Pmag_rbins, Pmag_bins = get1Dmean(r.to('pc'), Pmag, xlog=True)
    # plot1Dline(Pmag_rbins, (Pmag_bins, ), label=('$P_\mathrm{mag}$',), color=('blue',), xlog=True, ylog=True, out='Pmag_{}.pdf'.format(snap.name))
    
    # Pth = ((5./3.-1.)*snap.dens0*snap['PartType0','InternalEnergy']/constants.c**2).to('g cm**-3')
    # Pth_rbins, Pth_bins = get1Dmean(r.to('pc'), Pth, xlog=True)
    # plot1Dline(Pth_rbins, (Pth_bins, ), label=('$P_\mathrm{th}$',), color=('red',), xlog=True, ylog=True, out='Pth_{}.pdf'.format(snap.name))

    # Prad = ((4./3.-1.) * np.sum(snap['PartType0','PhotonEnergy'],axis=1) / (snap.mass0/snap.dens0)).to('g cm**-3')
    # Prad_rbins, Prad_bins = get1Dmean(r.to('pc'), Prad, xlog=True)
    # plot1Dline(Prad_rbins, (Prad_bins, ), label=('$P_\mathrm{th}$',), color=('green',), xlog=True, ylog=True, out='Prad_{}.pdf'.format(snap.name))
    
    print('black holes:')
    blackholetype='PartType3' #'PartType5'
    massname='Masses' # 'BH_Mass'
    print('  masses (Msun)  : {}'.format((snap[blackholetype,massname]).to('Msun'))) # *snap.metadata['UnitMass_In_CGS']*units.g if BH_Mass ? 
    #print('  mdots (Msun/yr): {}'.format(snap[blackholetype,'BH_Mdot']))
    print('  radius (pc)    : {}'.format(np.sqrt(np.sum(snap[blackholetype,'Coordinates'].to('pc')**2, axis=1))))
    print('  speed (km/s)   : {}'.format(np.sqrt(np.sum(snap[blackholetype,'Velocities'].to('km/s')**2, axis=1))))

    # ...

    # temp: use pynbody for maps
    import pynbody
    
    s = pynbody.new(gas=len(snap.mass0))
    s.gas['pos'] = np.array(snap.pos0.to('pc'), dtype=np.float64)
    s.gas['mass'] = np.array(snap.mass0.to('Msun'), dtype=np.float64)
    s.gas['smooth'] = np.array(snap.smooth0.to('pc'), dtype=np.float64)
    s.gas['vel'] = np.array(snap.vel0.to('km/s'), dtype=np.float64)
    s.gas['temp'] = np.array(snap.temp0.to('K'), dtype=np.float64)

    s.physical_units()
    s['pos'].units = 'pc'
    s['mass'].units = 'Msol'
    s['smooth'].units = 'pc'
    s['vel'].units = 'km s**-1'
    s['temp'].units = 'K'

    r = np.sum(np.array(s['pos'].in_units('pc'), dtype=np.float64)**2, axis=1)
    max_r_pc = np.nanmax(r)
    min_r_pc = np.nanmin(r)
    ooms = max(int(np.log10(max_r_pc/min_r_pc)), 1)
    for oom in range(ooms):   # todo make this better...
        r_pc = max_r_pc/10**oom
        extent = (-r_pc, r_pc, -r_pc, r_pc)
        _map = pynbody.plot.sph.image(s.gas, width=r_pc, units="m_p cm**-2", noplot=True, resolution=500, threaded=False)#, restrict_depth=True)
        plt.title('gas')
        plt.imshow(np.log10(_map), extent=extent, origin='lower')
        plt.colorbar(label=r'log gas column density $\int \rho dz$ [$m_p cm^{-2}$]')
        plt.xlabel('x [pc]')
        plt.ylabel('y [pc]')
        plt.savefig(output_dir+'map_dens_{}_oom{}.pdf'.format(snap.name, oom))
        plt.clf()

        _map2 = pynbody.plot.sph.image(s.gas, qty='temp', width=r_pc, units="K", noplot=True, resolution=500, threaded=False)#, restrict_depth=True)
        plt.title('gas')
        plt.imshow(np.log10(_map2), extent=extent, origin='lower')
        plt.colorbar(label=r'log mean gas temperature $T$ [$K$]')
        plt.xlabel('x [pc]')
        plt.ylabel('y [pc]')
        plt.savefig(output_dir+'map_temp_{}_oom{}.pdf'.format(snap.name, oom))
        plt.clf()

        s.gas['Pmag'] = np.sum(snap['PartType0', 'MagneticField']**2, axis=1)/(8*np.pi)
        s.gas['Pmag'].units = 'K' # only to escape Gauss/unitless issue for plotting purposes
        _map3 = pynbody.plot.sph.image(s.gas, qty='Pmag', width=r_pc, units="K", noplot=True, resolution=500, threaded=False)#, restrict_depth=True)
        plt.title('gas')
        plt.imshow(np.log10(_map3), extent=extent, origin='lower')
        plt.colorbar(label=r'log mean magnetic field pressure $P_\mathrm{mag}$ [dyn/cm$^2$]')
        plt.xlabel('x [pc]')
        plt.ylabel('y [pc]')
        plt.savefig(output_dir+'map_Pmag_{}_oom{}.pdf'.format(snap.name, oom))
        plt.clf()

        s.gas['plasma beta'] = np.array(snap.dens0.to('g cm**-3')/(2*1.67262192e-24)*constants.k_B.to('erg K**-1')*snap.temp0.to('K')/s.gas['Pmag'], dtype=np.float64)
        s.gas['plasma beta'].units = 'K' # only to escape unitless issue for plotting purposes
        _map4 = pynbody.plot.sph.image(s.gas, qty='plasma beta', width=r_pc, units='K', noplot=True, resolution=500, threaded=False)#, restrict_depth=True)
        plt.title('gas')
        plt.imshow(np.log10(_map4), extent=extent, origin='lower')
        plt.colorbar(label=r'log mean gas $\beta_\mathrm{plasma}$')
        plt.xlabel('x [pc]')
        plt.ylabel('y [pc]')
        plt.savefig(output_dir+'map_plasmabeta_{}_oom{}.pdf'.format(snap.name, oom))
        plt.clf()

    log_timing()


def main():
    log_timing(f"getting filearguments...")
    filepath, analysis_dir, debug_flag = get_file_arguments(str, str, int, fill_empties_with_none=True)
    assert filepath is not None, 'Need to enter a filepath.'
    if analysis_dir is None:
        print('no analysis output directory specified, defaulting to output in local folder..')
        analysis_dir='.'
    print('debug flag {} -> {}'.format(debug_flag, bool(debug_flag)))
    quick_check(filepath, output_dir=analysis_dir, debugging=bool(debug_flag))

if __name__ == '__main__':
    main()

