"""Downsample particles in a GIZMO HDF5 snapshot.
Jaeden Bardati 2026 (jbardati@caltech.edu)

Usage:
    python rescale_disk.py mdot_factor original_file.hdf5 rescaled_file.hdf5

By default only affects gas (PartType0).   
"""

import sys

import h5py
import numpy as np


_METADATA_GROUP = 'Header'
_PARTICLE_GROUPS = ("PartType0","PartType1","PartType2","PartType3","PartType4","PartType5")

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


def rescale_disk(
    original_filename,
    rescaled_filename,
    mdot_factor=0.3,
    log=True,
    zero_time=True,
):
    """Randomly downsample selected particle types and rescale extensive data."""
    if original_filename == rescaled_filename:
        raise ValueError("The output file must differ from the input file.")
    _copy_snapshot(original_filename, rescaled_filename, blacklist=None, whitelist=list())

    with h5py.File(original_filename, "r") as snapshot, h5py.File(rescaled_filename, "a") as new:
        header = snapshot[_METADATA_GROUP]
        total = np.array(header.attrs["NumPart_Total"], copy=True)
        this_file = np.array(header.attrs["NumPart_ThisFile"], copy=True)
        Minimum_Mass_For_Cell_Merge = header.attrs["Minimum_Mass_For_Cell_Merge"]
        Maximum_Mass_For_Cell_Split = header.attrs["Maximum_Mass_For_Cell_Split"]
        if not np.array_equal(total, this_file):
            raise ValueError("Multi-file snapshots are not currently supported.")

        for i, particle_type in enumerate(_PARTICLE_GROUPS):
            if particle_type not in snapshot:
                continue

            group = snapshot[particle_type]
            if particle_type != "PartType0":
                for dataset_name in group: # copy over all datasets for this particle type without rescaling
                    new_group = new.require_group(particle_type)
                    if dataset_name in new_group:
                        del new_group[dataset_name]
                    group.copy(dataset_name, new_group, name=dataset_name)
                continue

            particle_count, dimension = group["Coordinates"].shape
            for dataset_name in group.keys():
                dataset = group[dataset_name]
                data = dataset[()]
                
                # assume keplerian disk dominated by black hole potential - preserve magnetic field v_A/v_c, thermal Toomre Q, and radiative efficiency
                if dataset_name == 'Masses':
                    data *= mdot_factor
                elif dataset_name == 'Density':
                    data *= mdot_factor
                elif dataset_name == 'DensityGradient':
                    data *= mdot_factor
                elif dataset_name == 'MagneticField':
                    data *= np.sqrt(mdot_factor)
                elif dataset_name == 'InternalEnergy':
                    data *= mdot_factor ** 2
                elif dataset_name == 'Temperature':
                    data *= mdot_factor ** 2
                elif dataset_name == 'SoundSpeed':
                    data *= mdot_factor
                elif dataset_name == 'PhotonOpacity':
                    data *= mdot_factor
                elif dataset_name == 'Pressure':
                    data *= mdot_factor
                elif dataset_name == 'PhotonEnergy':
                    data *= mdot_factor
                elif dataset_name == 'PhotonFluxDensity':
                    data *= mdot_factor

                attributes = dict(dataset.attrs)
                if dataset_name in new[particle_type]:
                    del new[particle_type][dataset_name]
                new_dataset = new[particle_type].create_dataset(dataset_name, data=data)
                for name, value in attributes.items():
                    new_dataset.attrs[name] = value
        new[_METADATA_GROUP].attrs["Minimum_Mass_For_Cell_Merge"] = Minimum_Mass_For_Cell_Merge * mdot_factor
        new[_METADATA_GROUP].attrs["Maximum_Mass_For_Cell_Split"] = Maximum_Mass_For_Cell_Split * mdot_factor
        if zero_time:
            new[_METADATA_GROUP].attrs["Time"] = 0.0

    if log:
        print('done.')


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: python rescale_disk.py FACTOR INPUT.hdf5 OUTPUT.hdf5"
        )
    mdot_factor = float(sys.argv[1])
    input_filename = sys.argv[2]
    output_filename = sys.argv[3]
    rescale_disk(input_filename, output_filename, mdot_factor=mdot_factor)
