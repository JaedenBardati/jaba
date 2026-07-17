import os.path
import glob

import h5py
import numpy as np

EXTS_SUPPORTED = ('.hdf5', '.h5', '.hdf')



## ----------------------------------- ##
## -------------- Input -------------- ##
## ----------------------------------- ##


class _MultipleHDF5OutputList(list): # internal wrapper indicating that this is a list of outputs from multiple hdf5 files.
    pass

def _handle_file_path_input(func):
    # This decorator handles handles any multi-file file path
    # cases (pass name without extension for multi-file support).
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
            return _MultipleHDF5OutputList(outs)
        else:
            raise ValueError("Ambiguous file path: multiple hdf5 files found with file name '{}' without all having the standard multi-file structure.".format(file_path))
    return _wrapper

def _only_open_first_file(func):
    @_handle_file_path_input
    def _wrapper(*args, _idx=None, **kwargs):
        if _idx is None or _idx == 0:
            return func(*args, **kwargs)
        return None
    return _wrapper

def _combine_output_multi_file_arrays(func):
    def _wrapper(*args, **kwargs):
        outs = _handle_file_path_input(func)(*args, **kwargs)
        if isinstance(outs, _MultipleHDF5OutputList):
            return np.concatenate(outs, axis=0)
        else:
            return outs
    return _wrapper

def _combine_output_multi_file_dict_of_arrays(func):
    def _wrapper(*args, **kwargs):
        outs = _handle_file_path_input(func)(*args, **kwargs)
        if isinstance(outs, _MultipleHDF5OutputList):
            combined_dict = {}
            for out in outs:
                for key, value in out.items():
                    if key in combined_dict:
                        combined_dict[key].append(value)
                        continue
                    combined_dict[key] = [value,]
            for key, value_list in combined_dict.items():
                combined_dict[key] = np.concatenate(value_list, axis=0)
        else:
            return outs
    return _wrapper

@_only_open_first_file
def _test_open(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            pass
    except:
        return 0
    return 1




@_only_open_first_file
def get_group_names(file_path):
    with h5py.File(file_path, 'r') as f:
        groups = list(f.keys())
    return groups

@_only_open_first_file
def get_attributes(file_path, group):
    with h5py.File(file_path, 'r') as f:
        attrs = dict(f[group].attrs)
    return attrs

@_only_open_first_file
def get_dataset_names(file_path, group):
    with h5py.File(file_path, 'r') as f:
        dataset_names = list(f[group].keys())
    return dataset_names


@_only_open_first_file
def explore(file_path):
    '''Prints the file structure and basic attributes of the data.'''
    with h5py.File(file_path, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")
                for k in f[name].attrs.keys():
                    print(f"  Attribute: {k} = {f[name].attrs[k]}")
        f.visititems(print_structure)
        print("f keys: ", f.keys())
        pass

@_only_open_first_file
def get_metadata(file_path):
    '''Returns all metadata as a dictionary (path) of dictionaries (metadata attribute name). Only looks at the first file if there are many.'''
    all_data = {}
    def collect_datasets(name, obj):
        if isinstance(obj, h5py.Group):
            all_data[name] = dict(f[name].attrs)
    with h5py.File(file_path, 'r') as f:
        f.visititems(collect_datasets)
    return all_data

@_combine_output_multi_file_arrays
def get_dataset(file_path, dataset_path, keepViewObject=False, _idx=None):
    '''Returns the desired dataset as a numpy array (or HDF5View object if enabled).'''
    with h5py.File(file_path, 'r') as f:
        data = np.array(f[dataset_path]) if not keepViewObject else f[dataset_path]
    return data

@_combine_output_multi_file_dict_of_arrays
def get_data(file_path, _idx=None):
    '''Returns all data as a dictionary. Not recommended if your data is large or in multiple files.'''
    all_data = {}
    def collect_datasets(name, obj):
        if isinstance(obj, h5py.Dataset):
            all_data[name] = obj[()]
    with h5py.File(file_path, 'r') as f:
        f.visititems(collect_datasets)
    return all_data


# examples/tests
#explore('blackhole_mergers.hdf5')
#explore('snap_042')
#get_dataset('blackhole_mergers.hdf5', 'details/mass')
#get_dataset('snap_042', 'PartType5/Coordinates')



## ----------------------------------- ##
## -------------- Output ------------- ##
## ----------------------------------- ##

def _write_dataset(file_path, dataset_path, data, overwrite=False):
    '''Targetted write to a given dataset in a given file. If overwrite is False, will not overwrite existing datasets. Creates a new file if it does not exist.'''
    with h5py.File(file_path, 'a') as f:
        if dataset_path in f:
            if overwrite:
                del f[dataset_path]
            else:
                raise ValueError(f"Dataset {dataset_path} already exists in {file_path}. Set overwrite=True to overwrite.")
        f.create_dataset(dataset_path, data=data)

def _write_data(file_path, data_dict, overwrite=False):
    '''Writes all data in a dictionary to a given file. If overwrite is False, will not overwrite existing datasets. Creates a new file if it does not exist.'''
    with h5py.File(file_path, 'a') as f:
        for dataset_path, data in data_dict.items():
            if dataset_path in f:
                if overwrite:
                    del f[dataset_path]
                else:
                    raise ValueError(f"Dataset {dataset_path} already exists in {file_path}. Set overwrite=True to overwrite.")
            f.create_dataset(dataset_path, data=data)

def _delete_dataset(file_path, dataset_path):
    '''Deletes a given dataset in a given file.'''
    with h5py.File(file_path, 'a') as f:
        if dataset_path in f:
            del f[dataset_path]
        else:
            raise ValueError(f"Dataset {dataset_path} does not exist in {file_path}.")

def _write_metadata(file_path, metadata_dict, overwrite=False):
    '''Writes all metadata in a dictionary to a given file. If overwrite is False, will not overwrite existing metadata. Creates a new file if it does not exist.'''
    with h5py.File(file_path, 'a') as f:
        for group_path, attrs in metadata_dict.items():
            if group_path not in f:
                f.create_group(group_path)
            for attr_name, attr_value in attrs.items():
                if attr_name in f[group_path].attrs:
                    if overwrite:
                        del f[group_path].attrs[attr_name]
                    else:
                        raise ValueError(f"Attribute {attr_name} already exists in {group_path} of {file_path}. Set overwrite=True to overwrite.")
                f[group_path].attrs[attr_name] = attr_value

def _rename_dataset(file_path, old_dataset_path, new_dataset_path):
    '''Renames a given dataset in a given file.'''
    with h5py.File(file_path, 'a') as f:
        if old_dataset_path not in f:
            raise ValueError(f"Dataset {old_dataset_path} does not exist in {file_path}.")
        f.move(old_dataset_path, new_dataset_path)

def _rename_metadata(file_path, group_path, old_attr_name, new_attr_name):
    '''Renames a given attribute in a given group of a given file.'''
    with h5py.File(file_path, 'a') as f:
        if group_path not in f:
            raise ValueError(f"Group {group_path} does not exist in {file_path}.")
        if old_attr_name not in f[group_path].attrs:
            raise ValueError(f"Attribute {old_attr_name} does not exist in {group_path} of {file_path}.")
        value = f[group_path].attrs[old_attr_name]
        del f[group_path].attrs[old_attr_name]
        f[group_path].attrs[new_attr_name] = value
