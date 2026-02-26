import numpy as np
from astropy.io import fits

EXTS_SUPPORTED = ('.fits', '.fit', '.fts')

def _test_open(file_path):
    try:
        with fits.open(file_path) as hdul:
            pass
    except:
        return 0
    return 1

def explore(file_path):
    '''Prints the file structure and basic attributes of the data.'''
    with fits.open(file_path) as hdul:
        for i, hdu in enumerate(hdul):
            print(f"Shape of image data {i}: {hdu.data.shape}")
            print(f'Image data {i} header:')
            print(repr(hdu.header))
            print()

def get_metadata(file_path):
    '''Returns all metadata as a list of dictonaries.'''
    metadata = []
    with fits.open(file_path) as hdul:
        for i, hdu in enumerate(hdul):
            metadata.append(hdu.header.__dict__)
    return metadata

def get_dataset(file_path, index=0):
    '''Returns the desired dataset as a numpy array.'''
    with fits.open(file_path) as hdul:
        data = np.array(hdul[index].data)
    return data

def get_data(file_path):
    '''Returns all data as a list of numpy arrays. Not recommended if your data is very large.'''
    all_data = []
    with fits.open(file_path) as hdul:
        for index in range(len(hdul)):
            all_data.append(np.array(hdul[index].data))
    return all_data


# examples/tests
#explore('skirtrun_temp_1_1_T_xy.fits')
#get_dataset('skirtrun_temp_1_1_T_xy.fits')
