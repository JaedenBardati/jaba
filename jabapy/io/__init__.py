from . import general

__all__ = ['general', 'fits', 'hdf5',]

# common uses
guess_filetype = general.guess_filetype
explore = general.explore
get_metadata = general.get_metadata
get_dataset = general.get_dataset
