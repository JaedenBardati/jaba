import os.path
import glob

import h5py
import numpy as np

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

@_combine_output_multi_h5_file_arrays
def get_dataset_h5(file_path, dataset_path, _idx=None):
    with h5py.File(file_path, 'r') as f:
        data = np.array(f[dataset_path])
    return data


# examples/tests
#explore_h5('blackhole_mergers.hdf5')
#explore_h5('snap_042')
#get_dataset_h5('blackhole_mergers.hdf5', 'details/mass')
#get_dataset_h5('snap_042', 'PartType5/Coordinates')
