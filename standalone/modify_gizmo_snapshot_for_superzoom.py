import h5py
import numpy as np
import warnings

def remake_snap_for_zoom_in(
    snap_dir           = './',
    snap_name          = 'run_wBHgrowth_resim_1BHcen1out_snapshot_125_zoom1ic',
    special_type       = 'PartType5',  # type of particle to center around
    special_id         = None,         # id of particle to center around if None, will use the most massive particle of the given type
    remove_other_BHs   = False,        # set to True if you want to remove the other special particles (BHs)
    move_BH_type5_to_3 = True,         # set to True if you want to convert the special particle from type 5 (with BH_mass) to 3 (with Masses)
    new_unit_l         = 1.4959787e13,  # AU
    new_unit_m         = 1.989e33,      # Msun
    new_unit_v         = 1.e7,          # 100 km/s
    r_cut_phys_kpc     = np.inf,
    set_time_to_zero   = True,  # zero out the time or convert it? will set to zero no matter what if cosmological sim  
    modify_snap        = False, # set to False to debug
    ):
    """
    YOU SHOULD BACKUP THE FILE BEFOREHAND!! assumes b unit is gauss
    Usage:
    import modify_gizmo_snapshot_for_superzoom as sz
    sz.remake_snap_for_zoom_in(modify_snap=True)
    """
    if move_BH_type5_to_3:
        assert special_type == 'PartType5', 'need special_type="PartType5" if move_BH_type5_to_3 is on.'
    
    # WRITES OVER THE READ IN SNAPSHOT -- MAKE COPY BEFORE!
    snapshot=snap_dir+snap_name+'.hdf5'
    with h5py.File(snapshot,'r+') as F:
        ## load data for our special particle
        P0=F[special_type]
        if special_id is None: # convert from id to index
            special_index = np.argmax(P0['Masses']) # by default choose most massive one
        else:
            assert special_id in P0['ParticleIDs'], 'special_id not found in the particle list'
            special_index = np.where(P0['ParticleIDs'] == special_id)[0,0]
        del special_id
        cen0=np.array(P0['Coordinates'][special_index], dtype=np.float64)
        print('special particle location =', cen0)
        vel0=np.array(P0['Velocities'][special_index], dtype=np.float64)
        acc0=np.array(P0['Acceleration'][special_index], dtype=np.float64)
        if 'Potential' in list(P0.keys()):
            phi0=np.array(P0['Potential'][special_index], dtype=np.float64)
        else:
            phi0 = 0.0
        
        header = F['Header']
        comoving = bool(int(header.attrs['ComovingIntegrationOn']))
        print(('' if comoving else 'non-') + 'cosmological sim input')
        if(comoving): 
            ascale = header.attrs['Time']
            h = header.attrs['HubbleParam']
        else:
            ascale = 1.
            h = 1.
        old_unit_l = header.attrs['UnitLength_In_CGS'] # includes little h factor[s]
        old_unit_v = header.attrs['UnitVelocity_In_CGS']
        old_unit_m = header.attrs['UnitMass_In_CGS']
        r_cut = r_cut_phys_kpc * h # convert to units of kpc/h
        
        unit_conv_l = old_unit_l/new_unit_l
        unit_conv_v = old_unit_v/new_unit_v
        unit_conv_m = old_unit_m/new_unit_m
        unit_conv_e = unit_conv_m * unit_conv_v*unit_conv_v
        unit_conv_t = unit_conv_l/unit_conv_v

        npart_new = np.array([0,0,0,0,0,0],'int32')
        for key_P in list(F.keys()):
            if(key_P != 'Header'):
                P = F[key_P]
                if('Type0' in key_P): j=0
                if('Type1' in key_P): j=1
                if('Type2' in key_P): j=2
                if('Type3' in key_P): j=3
                if('Type4' in key_P): j=4
                if('Type5' in key_P): j=5
                print('particle type', key_P)
                if list(P.keys()) == []:
                    print('> skipping because this type has no datasets')
                    continue
                Pc = np.array(P['Coordinates'], dtype=np.float64)  - np.array(P0['Coordinates'][0], dtype=np.float64)
                r = ascale * np.sqrt(np.sum(Pc*Pc,axis=1))
                print(r.shape)
                ok = np.where(r < r_cut)[0]
                ok_size = (r.take(ok,axis=0)).size
                npart_new[j] = ok_size
                for q in list(P.keys()):
                    if(modify_snap):
                        P_tmp = np.array(P[q], dtype=np.float64).take(ok,axis=0)
                        del P[q]

                        if('Coordinates'==q): 
                            P_tmp = P_tmp - cen0
                            P_tmp *= ascale*unit_conv_l
                        if('Velocities'==q): 
                            P_tmp = P_tmp - vel0
                            P_tmp *= np.sqrt(ascale)*unit_conv_v
                        if((q=='Acceleration') or (q=='HydroAcceleration') or (q=='RadiativeAcceleration')): 
                            P_tmp = P_tmp - acc0
                            P_tmp *= unit_conv_v / unit_conv_t; 
                        if(q=='Potential'): 
                            P_tmp = P_tmp - phi0
                            P_tmp *= unit_conv_m/(ascale*unit_conv_l); 
                        if('Masses'==q or 'BH_Mass'==q or 'BH_Mass_AlphaDisk'==q or 'SinkInitialMass'==q or 'ZAMS_Mass'==q or 'Mass_D'==q): 
                            P_tmp *= unit_conv_m
                        if('SmoothingLength'==q or 'BH_AccretionLength'==q or 'SinkRadius'==q): 
                            P_tmp *= ascale*unit_conv_l
                        if('PhotonEnergy'==q): 
                            P_tmp *= unit_conv_e
                        if('PhotonFluxDensity'==q): 
                            P_tmp *= unit_conv_e / (unit_conv_t*(unit_conv_l**2))
                        if('InternalEnergy'==q): 
                            P_tmp *= unit_conv_e/unit_conv_m
                        if('Density'==q): 
                            P_tmp *= unit_conv_m/(ascale*unit_conv_l)**3
                        if('Pressure'==q): 
                            P_tmp *= unit_conv_e/(ascale*unit_conv_l)**3
                        if('StellarFormationTime'==q or 'ProtoStellarAge'==q): 
                            if(comoving): 
                                ascale_formed = P_tmp[:]
                                ascale_now = header.attrs['Time']
                                H0_inv_s = 3.086e17 / header.attrs['HubbleParam']
                                O_m = 0.3089; O_l = 0.6911; 
                                da_half = 0.5*np.abs(ascale_now - ascale_formed)
                                a = ascale_formed; aEzInv_f = 1./(a * np.sqrt(O_m/(a**3) + O_l))
                                a = ascale_now; aEzInv_n = 1./(a * np.sqrt(O_m/(a**3) + O_l))
                                dt = H0_inv_s * da_half * (aEzInv_f + aEzInv_n)
                                dt_newunits = -dt / (new_unit_l/new_unit_v)
                                P_tmp = dt_newunits
                            else:
                                if set_time_to_zero:
                                    t0 = header.attrs['Time']
                                    P_tmp = P_tmp - t0
                                P_tmp *= unit_conv_t
                        if('CosmicRayEnergy'==q): 
                            P_tmp *= unit_conv_e
                        if('DensityGradient'==q): 
                            P_tmp *= unit_conv_m/(unit_conv_l)**4
                        if(q=='VelocityGradient'): 
                            P_tmp *= unit_conv_v/unit_conv_l
                        if(q=='PhotonOpacity'): 
                            P_tmp *= unit_conv_m/(unit_conv_l*unit_conv_l)

                        if('BH_Mdot'==q):
                            P_tmp *= unit_conv_m/unit_conv_t
                        if('BH_Specific_AngMom'==q):
                            P_tmp *= unit_conv_l*unit_conv_v
                        if('SoundSpeed'==q): 
                            P_tmp *= unit_conv_v 
                        # if(q=='Metallicity'):
                        #     if(snapshot_number == 0):
                        #         if(j==0):
                        #             nmetals_forlater = P_tmp.shape[1]
                        #     else:
                        #         nmetals = P_tmp.shape[1]
                        #         if(nmetals < nmetals_forlater):
                        #             print(' -- not as many metals, zeroing the missing fields -- ',nmetals,nmetals_forlater)
                        #             Z_tmp = np.zeros((P_tmp.shape[0],nmetals_forlater))
                        #             Z_tmp[:,0:nmetals] = P_tmp
                        #             P_tmp = Z_tmp                                
                        #         if(nmetals > nmetals_forlater):
                        #             print(' -- too many metals, only keeping the important fields -- ',nmetals,nmetals_forlater)
                        #             Z_tmp = np.zeros((P_tmp.shape[0],nmetals_forlater))
                        #             Z_tmp = P_tmp[:,0:nmetals_forlater]
                        #             P_tmp = Z_tmp

                        if key_P == special_type and remove_other_BHs:
                            # remove all BHs other than special index
                            P_tmp = P_tmp[[special_index,], ...]
                            npart_new[j] = 1

                        P.create_dataset(q,data=P_tmp)

        if move_BH_type5_to_3:
            # move BHs from type 5 --> type 3
            special_type_out = 'PartType3'
            bh_mass = 'BH_Mass'

            if npart_new[3] != 0:
                warnings.warn("There is already a type 3 particle in the file, this will be overwritten.")
            npart_new[3], npart_new[5] = npart_new[5], 0 #npart_new[3]

            if(modify_snap):
                if special_type_out in list(F.keys()): # clear type3 if exists
                    print('clearing out', special_type_out)
                    P = F[special_type_out] # type 3 group
                    for q in list(P.keys()):
                        del P[q]
                else: # make type3 if does not exist
                    F.create_group(special_type_out)

                P = F[special_type_out] # type 3 group

                assert bh_mass in list(P0.keys()), 'There is no "{}" dataset in group "{}"'.format(bh_mass, special_type)

                # transfer/make each desired type 3 dataset (or set to zero if does not exist).
                print('copying relevant datasets from', special_type, 'to', special_type_out)
                for q in ['Acceleration', 'Coordinates', 'Masses', 'Potential', 'Velocities', 'ParticleIDs', 'ParticleIDGenerationNumber', 'ParticleChildIDsNumber']: 
                    if(q == 'Masses'): # exception for Masses which should be taken to be the BH mass
                        P0_tmp = np.array(P0[bh_mass], dtype=np.float64)
                    elif q in list(P0.keys()): # transfer from type 5
                        P0_tmp = np.array(P0[q], dtype=np.float64)  # type 5 element
                    else:  # default to zero with warning
                        P0_tmp = np.zeros(npart_new[3], dtype=np.float64)
                        warnings.warn('Set type 3 dataset "{}" to zero (size {}) by default (not found in type 5)'.format(q, npart_new[3]))
                    P.create_dataset(q, data=P0_tmp)
                
                # clear and delete parttype 5
                print('clearing out and deleting', special_type)
                for q in list(P0.keys()):
                    del P0[q]
                del F[special_type]

        print('npart_new=', npart_new)
        if(modify_snap):
            converted_time = header.attrs['Time']*unit_conv_l/unit_conv_v
            if set_time_to_zero or comoving:
                print('in new units, time was', converted_time, 'but setting it to zero now')
                header.attrs.modify('Time',0.) # zero out time
            else:
                print('in new units, time is', converted_time)
                print('^ set this to the TimeBegin in params.txt for gizmo')
                header.attrs.modify('Time',converted_time)
            for key_H in ['NumPart_ThisFile','NumPart_Total']:
                header.attrs.modify(key_H,npart_new)
            header.attrs.modify('UnitLength_In_CGS',new_unit_l)
            header.attrs.modify('UnitVelocity_In_CGS',new_unit_v)
            header.attrs.modify('UnitMass_In_CGS',new_unit_m)
    


if __name__ == "__main__":
    import sys
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            "Usage: python modify_gizmo_snapshot_for_superzoom.py SNAPSHOT_NAME [DEBUG]"
        )
    snap_name = str(sys.argv[1])
    debug = int(sys.argv[2]) if len(sys.argv) == 3 else True
    if debug:
        print("DEBUG MODE: no changes will be made to the snapshot file.")
    else:
        print("Snapshot file changes will be made. Make sure you have a backup of the original file.")
    remake_snap_for_zoom_in(snap_name=snap_name, modify_snap=not debug)

