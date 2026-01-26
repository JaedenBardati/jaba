import numpy as np
from astropy.io import fits

def explore_fits(file_path):
    with fits.open(file_path) as hdul:
        for i, hdu in enumerate(hdul):
            print(f"Shape of image data {i}: {hdu.data.shape}")
            print(f'Image data {i} header:')
            print(repr(hdu.header))
            print()

def get_dataset_fits(file_path, index=0):
    with fits.open(file_path) as hdul:
        image_data = np.array(hdul[index].data)
    return image_data

# examples/tests
#explore_fits('skirtrun_temp_1_1_T_xy.fits')
#get_dataset_fits('skirtrun_temp_1_1_T_xy.fits')
