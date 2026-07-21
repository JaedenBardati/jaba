"""Convert a legacy (pre-2025, C) GIZMO snapshot format to the modern (post-2026, C++/Kokkos) format.
Jaeden Bardati 2026 (jbardati@caltech.edu)

Usage:
    python convert_gizmo_legacy_to_cpp.py snapshot.hdf5 [converted.hdf5]

The command converts the supplied file in place. Supplying a second path creates or updates that file instead.
"""

import sys

import h5py


_METADATA_GROUP = 'Header'
_PARTICLE_GROUPS = ("PartType0","PartType1","PartType2","PartType3","PartType4","PartType5")

_METADATA_RENAMES = { # add more changes here if needed (I'd always appreciate a pull request if you find more changes that need to be made!)
    "SeedBlackHoleMass": "SeedSinkMass",
    "SeedAlphaDiskMass": "SeedReservoirMass",
    "BlackHoleAccretionFactor": "SinkAccretionFactor",
    "BlackHoleEddingtonFactor": "SinkEddingtonFactor",
    "BlackHoleFeedbackFactor": "SinkFeedbackFactor",
    "BlackHoleMaxAccretionRadius": "SinkMaxAccretionRadius",
    "BlackHoleNgbFactor": "SinkNgbFactor",
    "BlackHoleRadiativeEfficiency": "SinkRadiativeEfficiency",
    "BAL_f_accretion": "Sink_accreted_fraction",
    "BAL_f_launch_v": "Sink_outflow_jetlaunchvelscaling",
    "BAL_internal_temperature": "Sink_outflow_temperature",
    "BAL_v_outflow": "Sink_outflow_velocity",
}

_DATASET_RENAMES = { # add more changes here if needed
    ("PartType0", "SmoothingLength"): ("PartType0", "KernelMaxRadius"),
}


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


def convert_legacy_snapshot_to_new_format(
    original_filename, new_filename=None, blacklist=None, whitelist=None, log=True,
):
    """Convert a legacy GIZMO HDF5 snapshot, in place unless given an output path."""
    converted_filename = original_filename if new_filename is None else new_filename
    if new_filename is not None and new_filename != original_filename:
        _copy_snapshot(original_filename, new_filename, blacklist, whitelist)

    with h5py.File(converted_filename, "r+") as snapshot:
        header = snapshot[_METADATA_GROUP]
        for old_name, new_name in _METADATA_RENAMES.items():
            if old_name in header.attrs:
                if log:
                    print(f"Renaming metadata '{old_name}' to '{new_name}'...")
                value = header.attrs[old_name]
                del header.attrs[old_name]
                header.attrs[new_name] = value

        for (old_group, old_name), (new_group, new_name) in _DATASET_RENAMES.items():
            if old_group in snapshot and old_name in snapshot[old_group]:
                if log:
                    print(f"Renaming dataset '{old_group}/{old_name}' to '{new_group}/{new_name}'...")
                if new_group in snapshot and new_name in snapshot[new_group]:
                    del snapshot[new_group][new_name]
                snapshot.move(f"{old_group}/{old_name}", f"{new_group}/{new_name}")
    if log:
        print('done.')


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            "Usage: python convert_gizmo_legacy_to_cpp.py INPUT.hdf5 [OUTPUT.hdf5]"
        )
    convert_legacy_snapshot_to_new_format(*sys.argv[1:])
