import os

import .hdf5 as hdf5
import .fits as fits
_FILETYPE_MODULES = {'hdf5':hdf5, 'fits':fits,}    # STRICTLY LOWERCASE KEYS, order by decreasing strictness of file type rules

"""
Requirements for a new filetype:
 1) ** EXTS_SUPPORTED list containing a list of all supported extensions - check that this does not conflict with other extensions
 2) ** _test_open(file_path) function that returns 0 if the file failed to open and 1 if the file successfully opened (i.e. fits the form of the file type)
 3) ** explore(file_path) function that displays as summary of the file structure for the user
 4) * get_metadata(file_path, **kwargs) function that returns the metadata in no particular format but returns None if no metadata.
 5) * get_data(file_path, **kwargs) function that returns all the data in no particular format (note this only really works for small files)
 6) (optional) get_dataset(file_path, *args, **kwargs) function that returns the dataset(s) in no particular format
"""

def guess_filetype(file_path, _force_filetype=None):
    '''This function tries to decide what your data is by looking at it. It should be fairly general, so long as you define a _test_open function and EXTS_SUPPORTED list for each filetype.'''
    if _force_filetype is None:
        if not os.path.exists(file_path):
            raise FileNotFoundError('File at path "{}" not found.'.format(file_path))
        
        # assume based on extensions
        root, ext = os.path.splitext(file_path)
        for filetype in _FILETYPE_MODULES.values():
            if ext in filetype.EXTS_SUPPORTED:
                return filetype.__name__
        
        # assume type if it sucessfully loads
        for filetype in _FILETYPE_MODULES.values():
            if ext in filetype._test_open():
                return filetype.__name__
        
        raise NotImplementedError('Loading this filetype is not supported. Please add support.') 
    
    if _force_filetype.lower() not in _FILETYPE_MODULES.keys():
        raise NotImplementedError('Loading this filetype does not seem to be supported. Supported filetypes: {}'.format(list(_FILETYPE_MODULES.keys())))
    
    return _force_filetype


def explore(file_path, filetype=None):
    """Gives you an overview of your data. If the filetype is not specified, it will try to guess it."""
    filetype = guess_filetype(file_path, _force_filetype=filetype)
    return filetype.explore(file_path)


def get_metadata(file_path, filetype=None, **kwargs):
    """Returns the metadata associated with your file. If the filetype is not specified, it will try to guess it. If there is no metadata, None will be returned."""
    filetype = guess_filetype(file_path, _force_filetype=filetype)
    return filetype.get_metadata(file_path, **kwargs)
    

def get_data(file_path, filetype=None, **kwargs):
    """Returns the data associated with your file. If the filetype is not specified, it will try to guess it."""
    filetype = guess_filetype(file_path, _force_filetype=filetype)
    return filetype.get_data(file_path, **kwargs)
    