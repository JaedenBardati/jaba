"""
Various GIZMO initial data functions.
"""
import numpy as np

from ..snapshot import HDF5_Snapshot
from ..io import hdf5



def copy(original_filename, new_filename, blacklist=None, whitelist=None, overwrite=False):
    """
    Simple copy of a snapshot. 
    """
    snap = HDF5_Snapshot(original_filename)
    newsnap = HDF5_Snapshot(new_filename)

    # copy metadata
    newsnap._write_metadata(snap.metadata, overwrite=overwrite)

    # copy datasets, but follow blacklist and whitelist rules
    for ptype in snap.particle_types:
        for dataset in snap._dataset_names(ptype):
            if blacklist is not None and dataset in blacklist:
                continue
            if whitelist is not None and dataset not in whitelist:
                continue
            data = snap[ptype, dataset]
            newsnap._write_dataset(ptype, dataset, data, overwrite=overwrite)
    
    return newsnap



def _handle_copy(original_filename, new_filename, blacklist=None, whitelist=None):
    # Helper function to handle copying/overwriting a snapshot file.
    snap = HDF5_Snapshot(original_filename)
    if new_filename is not None:
        print(f"copying {original_filename} to {new_filename}...")
        newsnap = copy(original_filename, new_filename, blacklist=blacklist, whitelist=whitelist, overwrite=True)
    else:
        print(f"changing {original_filename} in place...")
        newsnap = snap
    return snap, newsnap




def downsample(original_filename, downsampled_filename=None, factor=[32,], ptypes=['PartType0',], blacklist=None, whitelist=None, seed=0, log=True):
    """
    Simple downsample of a snapshot by a given factor (approximately). By default only downsamples gas. 
    The units of the downsampled snapshot will be the same as the original snapshot (no unit conversion or coordinate transformation).
    If downsampled_filename is None, will overwrite the original file and not make a copy.
    Note that this is functionally equivalent to the standalone script downsample_gizmo.py
    """
    extensive_vars={'Masses','SmoothingLength','KernelMaxRadius','PhotonEnergy',}
    length_extensive_vars={'SmoothingLength','KernelMaxRadius'} # extensive vars that scale with length, not volume
    position_var='Coordinates'
    npart_metavars={'NumPart_ThisFile','NumPart_Total',} 

    if seed is not None:
        np.random.seed(seed)
    if isinstance(factor, int):
        factor = [factor,]*len(ptypes)
    
    snap, downsampled_snap = _handle_copy(original_filename, downsampled_filename, blacklist=blacklist, whitelist=['',])

    newtotal = snap.metadata['NumPart_Total']
    newsizes = snap.metadata['NumPart_ThisFile']
    assert all(newtotal[j] == newsizes[j] for j in range(len(newsizes))), "This downsampling code does not support multi-file snapshots yet." 
    for i, ptype in enumerate(snap._SUPPORTED_PARTICLES_TYPES):
        if ptype in snap.particle_types: # make sure that it exists in the original snapshot
            if ptype in ptypes: # if it is a particle type to downsample 
                if log:
                    print(f"downsampling {ptype} by a factor of {factor[i]}...")

                pos_data = snap[ptype, position_var]
                dim = pos_data.shape[1] # infer dimension of the simulation from position 
                length = pos_data.shape[0] # infer length of the dataset from position 
                newsize = length//factor[i]
                indices = np.random.choice(length, size=newsize, replace=False)
                for dataset in snap._dataset_names(ptype):
                    data = snap[ptype, dataset]
                    downsampled_data = data[indices, ...] if data.ndim > 1 else data[indices]
                    if dataset in extensive_vars:
                        p = 1.0/dim if dataset in length_extensive_vars else 1
                        downsampled_data *= factor[i]**p  # scale extensive properties by the downsampling factor
                    
                    downsampled_snap._write_dataset(ptype, dataset, downsampled_data, overwrite=True)
                newtotal[i] = newsize
                newsizes[i] = newsize
            else:
                if log:
                    print(f"keeping {ptype} unchanged...")
                    for dataset in snap._dataset_names(ptype):
                        if blacklist is not None and dataset in blacklist:
                            continue
                        if whitelist is not None and dataset not in whitelist:
                            continue
                        data = snap[ptype, dataset]
                        downsampled_snap._write_dataset(ptype, dataset, data, overwrite=True)

    # hardcode change npart for each particle type (only for gizmo/gadget snapshots)
    downsampled_snap._write_metadata(
        {f'NumPart_ThisFile': newsizes, 
         f'NumPart_Total': newtotal,
         }, overwrite=True)
    

def convert_legacy_snapshot_to_new_format(original_filename, new_filename=None, blacklist=None, whitelist=None):
    """
    Converts a legacy (2025, c) GIZMO snapshot to the new (2026, c++) GIZMO version format. 
    If new_filename is None, will overwrite the original file and not make a copy.
    Note that this is functionally equivalent to the standalone script convert_gizmo_legacy_to_cpp.py
    """
    snap, newsnap = _handle_copy(original_filename, new_filename, blacklist=blacklist, whitelist=whitelist)

    # rename metadata
    rename_map_metadata = {
        'SeedBlackHoleMass': 'SeedSinkMass',
        'SeedAlphaDiskMass': 'SeedReservoirMass',
        'BlackHoleAccretionFactor': 'SinkAccretionFactor',
        'BlackHoleEddingtonFactor': 'SinkEddingtonFactor',
        'BlackHoleFeedbackFactor': 'SinkFeedbackFactor',
        'BlackHoleMaxAccretionRadius': 'SinkMaxAccretionRadius',
        'BlackHoleNgbFactor': 'SinkNgbFactor',
        'BlackHoleRadiativeEfficiency': 'SinkRadiativeEfficiency',
        'BAL_f_accretion': 'Sink_accreted_fraction',
        'BAL_f_launch_v': 'Sink_outflow_jetlaunchvelscaling',
        'BAL_internal_temperature': 'Sink_outflow_temperature',
        'BAL_v_outflow': 'Sink_outflow_velocity',
    }
    original_metadata = dict(snap.metadata)
    for old_key, new_key in rename_map_metadata.items():
        if old_key in original_metadata:
            newsnap._rename_metadata(old_key, new_key)

    # rename datasets
    newsnap._rename_dataset('PartType0', 'SmoothingLength', 'KernelMaxRadius')

    return newsnap