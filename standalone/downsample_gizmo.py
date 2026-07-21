"""Downsample particles in a GIZMO HDF5 snapshot.
Jaeden Bardati 2026 (jbardati@caltech.edu)

Usage:
    python downsample_gizmo.py factor original_file.hdf5 downsampled_file.hdf5 [particle_type_bitmask]

By default only downsamples gas (PartType0).     
"""

import sys

import h5py
import numpy as np


_METADATA_GROUP = 'Header'
_PARTICLE_GROUPS = ("PartType0","PartType1","PartType2","PartType3","PartType4","PartType5")

_EXTENSIVE_DATASETS = {"Masses", "SmoothingLength",'KernelMaxRadius', "PhotonEnergy"} # add more if needed (I'd always appreciate a pull request if you find more datasets to be generally scaled)
_LENGTH_EXTENSIVE_DATASETS = {"SmoothingLength",'KernelMaxRadius'}  # extensive dsets that scale with length, not volume (as is otherwise assumed), namely scale with factor^(1/3) instead of factor


def _copy_snapshot(original_filename, new_filename, blacklist=None, whitelist=None):
    """Copy the Header and GIZMO particle datasets to new filename."""
    with h5py.File(original_filename, "r") as original, h5py.File(new_filename, "a") as new:
        original_header = original[_METADATA_GROUP]
        new_header = new.require_group(_METADATA_GROUP)
        for name, value in original_header.attrs.items():
            new_header.attrs[name] = value

        for particle_type in _PARTICLE_GROUPS:
            if particle_type not in original:
                continue
            original_group = original[particle_type]
            new_group = new.require_group(particle_type)
            for dataset_name in original_group:
                if blacklist is not None and dataset_name in blacklist:
                    continue
                if whitelist is not None and dataset_name not in whitelist:
                    continue
                if dataset_name in new_group:
                    del new_group[dataset_name]
                original_group.copy(dataset_name, new_group, name=dataset_name)


def downsample(
    original_filename,
    downsampled_filename,
    factor=32,
    ptype_bitmask=1, # bit mask of particle types to downsample, ie. 1 = just gas and 19 = parttype 0, 1, and 4
    seed=0,
    log=True,
):
    """Randomly downsample selected particle types and rescale extensive data."""
    if factor < 1:
        raise ValueError("factor must be at least 1, sampling more than the original is not supported.") # doing this involves spliting particles/cells, which is very non-trivial in MFM/MFV
    if original_filename == downsampled_filename:
        raise ValueError("The output file must differ from the input file.")

    _copy_snapshot(original_filename, downsampled_filename, blacklist=None, whitelist=list())

    rng = np.random.default_rng(seed)
    with h5py.File(original_filename, "r") as snapshot, h5py.File(downsampled_filename, "a") as new:
        header = snapshot[_METADATA_GROUP]
        total = np.array(header.attrs["NumPart_Total"], copy=True)
        this_file = np.array(header.attrs["NumPart_ThisFile"], copy=True)
        if not np.array_equal(total, this_file):
            raise ValueError("Multi-file snapshots are not currently supported.") # again, super happy if you want to add this feature and make a pull request!

        for i, particle_type in enumerate(_PARTICLE_GROUPS):
            if particle_type not in snapshot:
                continue

            group = snapshot[particle_type]
            if (ptype_bitmask & (1 << i)) == 0:
                for dataset_name in group: # copy over all datasets for this particle type without downsampling
                    new_group = new.require_group(particle_type)
                    if dataset_name in new_group:
                        del new_group[dataset_name]
                    group.copy(dataset_name, new_group, name=dataset_name)
                continue

            if "Coordinates" not in group:
                raise KeyError(f"{particle_type}/Coordinates is required for downsampling")
            particle_count, dimension = group["Coordinates"].shape
            new_count = particle_count // factor
            indices = np.sort(rng.choice(particle_count, size=new_count, replace=False)) # h5py requires fancy-index arrays to be strictly increasing

            if log:
                print(f"Downsampling {particle_type} from {particle_count} to {new_count} particles...")

            dataset_names = list(group.keys())
            if log:
                largest_dataset_name_size = max(len(str(name)) for name in dataset_names)
            for dataset_name in dataset_names:
                if log:
                    print(' > treating {} as {} variable...'.format(dataset_name.ljust(largest_dataset_name_size), 
                          'length-extensive' if dataset_name in _LENGTH_EXTENSIVE_DATASETS else 'extensive' if dataset_name in _EXTENSIVE_DATASETS else 'intensive'))
                dataset = group[dataset_name]
                if not isinstance(dataset, h5py.Dataset) or dataset.shape[:1] != (particle_count,):
                    continue
                data = dataset[indices]
                if dataset_name in _EXTENSIVE_DATASETS:
                    exponent = 1.0 / dimension if dataset_name in _LENGTH_EXTENSIVE_DATASETS else 1.0
                    data *= factor ** exponent
                attributes = dict(dataset.attrs)
                if dataset_name in new[particle_type]:
                    del new[particle_type][dataset_name]
                new_dataset = new[particle_type].create_dataset(dataset_name, data=data)
                for name, value in attributes.items():
                    new_dataset.attrs[name] = value

            particle_index = int(particle_type[-1])
            total[particle_index] = new_count
            this_file[particle_index] = new_count

        new[_METADATA_GROUP].attrs["NumPart_Total"] = total
        new[_METADATA_GROUP].attrs["NumPart_ThisFile"] = this_file

        if log:
            for name in ("NumPart_Total", "NumPart_ThisFile"):
                print(f"{name}: {[int(x) for x in np.array(header.attrs[name], dtype=int)]} -> {[int(x) for x in np.array(new[_METADATA_GROUP].attrs[name], dtype=int)]}")

    if log:
        print('done.')


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        raise SystemExit(
            "Usage: python downsample_gizmo.py FACTOR INPUT.hdf5 OUTPUT.hdf5"
        )
    downsampling_factor = int(sys.argv[1])
    input_filename = sys.argv[2]
    output_filename = sys.argv[3]
    particle_type_bitmask = int(sys.argv[4]) if len(sys.argv) == 5 else 1
    downsample(input_filename, output_filename, factor=downsampling_factor, ptype_bitmask=particle_type_bitmask)
