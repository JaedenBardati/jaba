#!/usr/bin/python3
"""
Runs a quick check on simulations to determine if they are reasonable to continue running.
  Also has a variety of convience functions for follow-up interactive analysis.
  Requires my "jaba" package (https://github.com/JaedenBardati/jaba/tree/main)
Jaeden Bardati 2025+
"""
if __name__ == "__main__":
    print('loading packages...')

## Builtin packages
import glob, sys, time, warnings
import os.path

## jaba
import jaba
from jaba.utils.filearguments import get_file_arguments 
from jaba.utils.timing import log_timing
from jaba.utils import units as u
from jaba.utils import constants as c
from jaba.utils import visual as jv

# other
import numpy as np


########################################################################################################################################################################
#####################                                            LOAD IN SKIRT ASCII FORMAT                                                        #####################
########################################################################################################################################################################
### Taken from https://github.com/JaedenBardati/skirt-datacube
# TODO put in jaba

def load_dat_file(filename, chunksize=None):
    """Function that loads a .dat file in the format of SKIRT input/output."""
    import pandas as pd
    
    # get header
    header = {}
    firstNonCommentRowIndex = None
    with open(filename) as file:
        for i, line in enumerate(file):
            l = line.strip()
            if l[0] == '#':
                l = l[1:].lstrip()
                if l[:6].lower() == 'column':
                    l = l[6:].lstrip()
                    split_l = l.split(':')
                    assert len(split_l) == 2 # otherwise, unfamiliar form!
                    icol = int(split_l[0]) # error here means we have the form: # column %s, where %s is not an integer
                    l = split_l[1].lstrip() # this should be the column name
                    header[icol] = l
            else:
                firstNonCommentRowIndex = i
                break
    assert firstNonCommentRowIndex is not None # otherwise the entire file is just comments

    # set up column names
    if firstNonCommentRowIndex == 0:
        columns = None
    else:
        columns = [None for i in range(max(header.keys()))]
        for k, v in header.items(): columns[k-1] = v
        assert None not in columns # otherwise, missing column 
    
    # get data
    df = pd.read_csv(filename, delim_whitespace=True, skiprows=firstNonCommentRowIndex, header=None, names=columns, chunksize=chunksize)
    
    return df


# ...

########################################################################################################################################################################
#####################                                     ANALYSIS AND PLOTTING FUNCTIONS                                                          #####################
########################################################################################################################################################################
# TODO put in jaba

from jaba.utils.grid import bin_particles_direct, bin_particles_maxmin, bin_particles_percentiles

def get_profile(x, qty, weight=None, kind='mean', nbins=100, xlog=False, xmin=None, xmax=None, p_volume=None, force_dx=False): # TODO put force_dx as option below instead (split deriv in two)
    """
    Gets a general 1d profile from particle data. 

    Consider quantity q_i, where i is the particle index to total N.
    Specify kind to determine the type of profile to compute, and weight to use a weighted profile.

    Kinds of profiles:
      A) mean    : (weighted) mean <q_i> in each bin - e.g., intensive-like variables like mean density rho - units are [q_i]. 
      B) median  : (weighted) median of quantity in each bin, as mean above
      
      C) deriv   : sum of q_i in each bin divided by the bin width - e.g., extensive-like variables like mass as dM/dlogR or dM/dr - units are [q_i]/dex (if log binned and specified) or [q_i]/length (otherwise).
      D) cumsum  : cumulative sum of q_i in each bin starting from r=0 - e.g. extensive-like variables like mass as enclosed mass M(<r) - units are [q_i].
      E) revsum  : reverse cumulative sum of quantity in each bin, as cumsum above

      F) std     : (weighted) standard deviation of quantity in each bin
      G) rms     : (weighted) root mean square of quantity in each bin
      H) q1      : (weighted) first quartile of quantity in each bin
      I) q3      : (weighted) third quartile of quantity in each bin
      J) max     : maximum of quantity in each bin
      K) min     : minimum of quantity in each bin

      You can also enter a string containing any order of the above option capital letter indexes and it will return a tuple of the corresponding profiles in the order specified. 
      You may want to use kind='AF' for (x_mid, mean, std), or kind='EJK' for (x_mid, median, q1, q3).
    """ 
    # TODO: add support for median, q1, q3, etc.
    # TODO: extend to more than just 1D
    # TODO: combine most of this into a couple of numba-compiled functions with only one pass of particles
    # TODO: add multiple qtys at once and support for M > nthreads for high potential performance gains
    kind = kind.replace('mean','A').replace('median','B').replace('deriv','C').replace('cumsum','D').replace('revsum','E').replace('std','F').replace('rms','G').replace('q1','H').replace('q3','I').replace('max','J').replace('min','K')
    kinds = set(); kinds_order = [c for c in kind if 'A' <= c <= 'K' and not (c in kinds or kinds.add(c))]

    # prepare input
    x = np.asarray(x)
    qty = np.asarray(qty)
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)
    if weight is not None:
        weight = np.asarray(weight)
        assert weight.shape == qty.shape, 'The weight array must have the same shape as qty.'
    else:
        weight = np.ones_like(qty, order='C')
    assert len(x.shape) == 1, 'Invalid x shape. Must be 1D array.'
    if xlog:
        x = np.log10(x)
        if xmin is not None:
            xmin = np.log10(xmin)
        if xmax is not None:
            xmax = np.log10(xmax)
    if p_volume is None:
        p_volume = 75.0 # use q1, q3 by default
    mins = (xmin,)
    maxs = (xmax,)
    dims = (nbins,)
    _x = x[:, np.newaxis]

    # gather up what needs to be computed
    idx = 0
    qtys = list()
    qty_names_to_idx = dict()
    if set(['C','D','E']) & kinds: # unweighted qty
        qtys.append(qty)
        qty_names_to_idx['unweighted qty'] = idx
        idx += 1
    if set(['A','F']) & kinds:  #weighted qty
        qtys.append(weight*qty)
        qty_names_to_idx['weighted qty'] = idx
        idx += 1
    if set(['A','F','G']) & kinds: # weight
        qtys.append(weight)
        qty_names_to_idx['weight'] = idx
        idx += 1
    if set(['F','G']) & kinds: # second order weighted qty
        qtys.append(weight*qty**2)
        qty_names_to_idx['weighted qty^2'] = idx
        idx += 1

    # if relevant, sum binned quantities in some order together
    if set(['A','C','D','E','F','G']) & kinds:
        qty_bins = bin_particles_direct(_x, *qtys, mins=mins, maxs=maxs, dims=dims)
        qty_bins = np.atleast_2d(qty_bins)

    # if relevant, get max/min of each bin
    if 'J' in kinds or 'K' in kinds:
        max_grid, min_grid = bin_particles_maxmin(_x, qty, mins=mins, maxs=maxs, dims=dims)

    #if relevant, get midpoint/q1/q3 of each bin
    if 'A' in kinds or 'B' in kinds or 'H' in kinds or 'I' in kinds:
        q1, median, q3 = bin_particles_percentiles(_x, qty, weight=weight, mins=mins, maxs=maxs, dims=dims, p_volume=p_volume) # for now, no weighting

    # get xbin edges and midpoints
    x_bins, = [np.arange(d + 1) * ((mx - mn) / d) + mn for mn, mx, d in zip(mins, maxs, dims)]
    x_mid = 0.5*(x_bins[1:]+x_bins[:-1]) 
    if xlog:
        x_mid = 10**x_mid
    
    # precompute some shared quantities
    if 'A' in kinds or 'F' in kinds:
        mean = qty_bins[qty_names_to_idx['weighted qty']] / qty_bins[qty_names_to_idx['weight']]
    if 'F' in kinds or 'G' in kinds:
        mean_sq = qty_bins[qty_names_to_idx['weighted qty^2']] / qty_bins[qty_names_to_idx['weight']]

    # structure return
    ret = [x_mid,]
    for kind in kinds_order:
        if kind == 'A': # mean
            ret.append(mean)
        elif kind == 'B': # median
            ret.append(median)
        elif kind == 'C': # deriv
            dx = np.diff(10**x_bins) if (xlog and force_dx) else np.diff(x_bins)
            ret.append(qty_bins[qty_names_to_idx['unweighted qty']] / dx)
        elif kind == 'D': # cumsum
            ret.append(np.cumsum(qty_bins[qty_names_to_idx['unweighted qty']]))
        elif kind == 'E': # revsum
            ret.append(np.cumsum(qty_bins[qty_names_to_idx['unweighted qty']][::-1])[::-1])
        elif kind == 'F': # std
            ret.append(np.sqrt(mean_sq - mean**2))
        elif kind == 'G': # rms
            ret.append(np.sqrt(mean_sq))
        elif kind == 'H': # q1
            ret.append(q1)
        elif kind == 'I': # q3
            ret.append(q3)
        elif kind == 'J': # max
            ret.append(max_grid)
        elif kind == 'K': # min
            ret.append(min_grid)
    return ret


# deprecated
# def plot1Dmean(x, qty, weights=None, labels=None, linestyles='-', kind='A', nbins=100, xlog=False, ylog=False, xlabel=None, ylabel=None, out=None, show=False, **kwargs):
#     """
#     Plots a general 1d profile from particle data. 
#     """
#     if not isinstance(weights, tuple):
#         weights = (weights,)
#     if not isinstance(labels, tuple):
#         labels = (labels,)*len(weights)
#     if not isinstance(linestyles, tuple):
#         linestyles = (linestyles,)*len(weights)

#     _out, _show, _show_legend = None, False, False
#     for i, w in enumerate(weights):
#         if i == len(weights) - 1 and (out is not None or show): # if last iteration (and print is happening)
#             _out = out
#             _show = show
#             _show_legend = True if any(l is not None for l in labels) else False

#         ret = get_profile(x, qty, weight=w, nbins=nbins, xlog=xlog, kind=kind)
#         if  len(kind) == 3:
#             raise NotImplementedError("JAEDEN BLOCKING THIS UNTIL TESTING.")
#             _x, _y, _mx, _mn = ret
#             jv.plot(_x, _y, label=labels[i], ls=linestyles[i], xlog=xlog, ylog=ylog, xlabel=xlabel, ylabel=ylabel, **kwargs)
#         elif len(kind) == 1:
#             _x, _y = ret
#             jv.plot(_x, _y, label=labels[i], ls=linestyles[i], xlog=xlog, ylog=ylog, xlabel=xlabel, ylabel=ylabel, **kwargs)
#         else:
#             assert False, f"Unsupported kind ({kind}) specified in plot1Dmean."


########################################################################################################################################################################
#####################                                              QUICK CHECK SIMULATION                                                          #####################
########################################################################################################################################################################

def quick_check(filepath, output_dir=None, debugging=False, center_around_BH=True,
        # top level flags:
        print_black_hole_info=True,
        dump_particle_histograms=False,
        radial_profiles=True,
        general_maps=False,
        BLR_analysis=True,
        ):
    """ THIS IS THE MAIN FUNCTION TO CHANGE... """
    if output_dir is None:
        output_dir='.'
    output_dir+='/'

    # set plotting style
    with jv.style_context(None):  # e.g. 'ApJ', 1
        jv.plt.rcParams['axes.prop_cycle'] = jv.plt.cycler(color=jv.plt.cm.Set1.colors) # TODO add to style_context

        # load snapshot/simulation
        log_timing(f"loading snapshot at {filepath} ...")
        snap = jaba.load(filepath, verbose=True)
        
        ### black holes ###
        if print_black_hole_info:
            print('black hole info:')
            blackholetype = 'PartType3' #'PartType5'
            massname = 'Masses' # 'BH_Mass'
            print('  masses (Msun)  : {}'.format((snap[blackholetype,massname]).to('Msun'))) # *snap.metadata['UnitMass_In_CGS']*u.g if BH_Mass ? 
            #print('  mdots (Msun/yr): {}'.format(snap[blackholetype,'BH_Mdot'].to('Msun/yr')))
            print('  radius (pc)    : {}'.format(np.sqrt(np.sum(snap[blackholetype,'Coordinates'].to('pc')**2, axis=1))))
            print('  speed (km/s)   : {}'.format(np.sqrt(np.sum(snap[blackholetype,'Velocities'].to('km/s')**2, axis=1))))

        # center around particle
        if center_around_BH:
            bh_parttype=3
            bh_index=0
            cpos = snap.pos[bh_parttype][bh_index][np.newaxis, :].to('pc')
            cvel = snap.vel[bh_parttype][bh_index][np.newaxis, :].to('km/s')
            rsink = (snap.metadata['Fixed_ForceSoftening_Keplerian_Kernel_Extent'][bh_parttype]*snap.metadata['UnitLength_In_CGS']*u.cm).to('pc')
            snap.center_on(bh_parttype, bh_index)
            snap.faceon(10**np.mean(np.log10(snap.r(bh_index).to('pc').value))*u.pc)
            print(f"centered around BH particle {bh_index} of type {bh_parttype} at {cpos} with velocity {cvel} and sink radius {rsink}...")
        else:
            cpos = np.zeros((1,3))*u.pc
            cvel = np.zeros((1,3))*u.km/u.s
            rsink = 0.0*u.pc

        # assign meaning to particle types
        snap.particle_type_meanings = { # temp, should be inferred
            'PartType0':'gas', 
            'PartType1':'dm',  # high res dm
            'PartType2':'lo_res_dm', 
            'PartType3':'bh', 
            'PartType4':'star', # ssp
            'PartType5':'sink_stars'
        }
        particle_type_meaning_strs_dict = { # put somehow in snapshot? first is subscript and second is full name
            'gas': ('gas', 'Gas'),
            'dm': ('DM', 'Dark Matter'),
            'hi_res_dm': ('DM, high-res', 'High Resolution Dark Matter'),
            'lo_res_dm': ('DM, low-res', 'Low Resolution Dark Matter'),
            'bh': ('BH', 'Black Hole'),
            'star': ('stars', 'Stars'),
            'ssp': ('stars, populations', 'Stellar Populations'),
            'sink_stars': ('stars, resolved', 'Resolved Stars')
        }
        particle_types = [pt for pt in snap.particle_types if (snap._resolve_particle_type_number(pt) != bh_parttype if center_around_BH else True)]
        particle_type_meaning_subscript_strs = [particle_type_meaning_strs_dict[snap._resolve_particle_type_meaning(pt)][0] for pt in particle_types]
        particle_type_meaning_full_strs = [particle_type_meaning_strs_dict[snap._resolve_particle_type_meaning(pt)][1] for pt in particle_types]

        ### particle histograms ###
        num_hist_bins = 50
        if dump_particle_histograms:
            log_timing('making particle histograms...')
            hist_dir = output_dir + 'hist/'
            os.makedirs(hist_dir, exist_ok=True)
            for parttype in snap.particle_types:
                ptnum = snap._resolve_particle_type_number(parttype)
                if not center_around_BH or ptnum != bh_parttype:
                    parttype_subdir = hist_dir + f'p{ptnum}/'
                    os.makedirs(parttype_subdir, exist_ok=True)
                    for dataset in snap[parttype]:
                        dsname = str(dataset).lower()
                        print(' > type {} dataset {}'.format(ptnum, dataset))
                        with np.errstate(invalid='ignore', divide='ignore'):
                            a = np.nan_to_num(snap[parttype, dataset].cgs.value, nan=np.nan, posinf=np.nan, neginf=np.nan)
                            loga = np.nan_to_num(np.log10(np.abs(a)), nan=np.nan, posinf=np.nan, neginf=np.nan)
                            if not np.all(np.isnan(a)):
                                jv.hist(a, bins=num_hist_bins, xlabel='{} for type {} (CGS)'.format(dsname, ptnum),
                                        out=parttype_subdir+'lin_{}_p{}_{}.png'.format(dsname, ptnum, snap.name))
                            else:
                                print('   >> all values are NaN, skipping linear hist...')
                            if not np.all(np.isnan(loga)):
                                jv.hist(loga, bins=num_hist_bins, xlabel='log {} for type {} (CGS)'.format(dsname, ptnum),
                                        out=parttype_subdir+'log_{}_p{}_{}.png'.format(dsname, ptnum, snap.name))
                            else:
                                print('   >> all values are NaN, skipping log hist...')
                    # extra histograms for derived datasets # TODO: just add these as derived datasets and iterate on those instead
                    jv.hist(snap.r(0).cgs.value, bins=num_hist_bins, xlabel='{} for type {} (CGS)'.format('rsph', ptnum),
                            out=parttype_subdir+'lin_{}_p{}_{}.png'.format('rsph', ptnum, snap.name))
                    jv.hist(snap.R(0).cgs.value, bins=num_hist_bins, xlabel='{} for type {} (CGS)'.format('rcyl', ptnum),
                            out=parttype_subdir+'lin_{}_p{}_{}.png'.format('rcyl', ptnum, snap.name))
                    jv.hist(np.log10(snap.r(0).cgs.value), bins=num_hist_bins, xlabel='log {} for type {} (CGS)'.format('rsph', ptnum),
                            out=parttype_subdir+'log_{}_p{}_{}.png'.format('rsph', ptnum, snap.name))
                    jv.hist(np.log10(snap.R(0).cgs.value), bins=num_hist_bins, xlabel='log {} for type {} (CGS)'.format('rcyl', ptnum),
                            out=parttype_subdir+'log_{}_p{}_{}.png'.format('rcyl', ptnum, snap.name))

        ### radial profiles ###
        num_radial_bins = 150
        if radial_profiles:
            # quick settings for this section
            xmin = rsink.to('pc').value
            xmax = np.max(snap.r(0).to('pc').value)*0.033  # fraction of max radius to use as outer edge of radial profile (must be set manually) 
            force_regular_log_major_ticks=True
            force_minor_ticks=True

            # get started ..
            log_timing('making radial profiles...')
            profile_dir = output_dir + 'profiles/'
            os.makedirs(profile_dir, exist_ok=True)
            def get_profile_with_defaults(*args, **kwargs):
                kwargs.setdefault('nbins', num_radial_bins)
                kwargs.setdefault('xlog', True)
                kwargs.setdefault('xmin', xmin)
                kwargs.setdefault('xmax', xmax)
                return get_profile(*args, **kwargs)

            ### resolution plots ###
            # gas mass resolution
            _a = get_profile_with_defaults(snap.r(0).to('pc'), snap.mass0.to('Msun'), kind='D') # Mencl
            _b = get_profile_with_defaults(snap.r(0).to('pc'), snap.mass0.to('Msun'), kind='A') # delta m
            jv.loglog(*zip(_a, _b), label=(r'$M_\mathrm{encl}$', r'$\delta m$'), ls=('--', '-'), color='black', out=profile_dir+'mass_resolution_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Mass resolution $\delta m$ ($M_\odot$)', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks)

            # gas spatial resolution
            _a = get_profile_with_defaults(snap.r(0).to('pc'), np.array((snap.mass0/snap.dens0).to('pc**3'))**(1/3.), kind='A') # delta r 
            jv.loglog(*_a, ls='-', color='black', out=profile_dir+'spatial_resolution_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Spatial resolution $\delta r$ (pc)', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks)

            # distribution of particle number
            _s = [get_profile_with_defaults(snap.r(pt).to('pc'), np.ones(len(snap.mass[pt])), kind='C') for pt in particle_types] # dN/dlogR for each particle type
            jv.loglog(*zip(*_s), label=tuple(particle_type_meaning_full_strs), ls='-', color=None, out=profile_dir+'number_distribution_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Particle number distribution ($\mathrm{d}N/\mathrm{d}\log r$)', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks)

            ### dynamics plots ### (e.g. fig 7 of FIF1)
            # gas density profile
            _bbins, _by, _bymin, _bymax = get_profile_with_defaults(snap.r(0).to('pc'), snap.dens0.to('m_p/cm**3').value, weight=snap.mass0.to('Msun')/snap.dens0.to('Msun/pc**3'), kind='AHI', p_volume=90.0) # volume-weighted mean and 90 percentile volume
            _cbins, _cy, _cymin, _cymax = get_profile_with_defaults(snap.r(0).to('pc'), snap.dens0.to('m_p/cm**3').value, weight=snap.mass0.to('Msun'), kind='AHI', p_volume=90.0) # mass-weighted mean and 90 percentile volume
            jv.plot((_bbins, _cbins), (_by, _cy), label=('volume-weighted', 'mass-weighted'), ls=('-', '--'))
            jv.fill_between((_bbins, _cbins), (_bymin, _cymin), (_bymax, _cymax), color=('C0', 'C1'), alpha=(0.3, 0.3), out=profile_dir+'dens_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Mass density $\rho$ ($m_p$/cm$^3$)', loglog=True, xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks)

            # surface density (gas, stars, dark matter, etc.)
            def surface_density_profile(R, mass):
                x, y = get_profile_with_defaults(R, mass, kind='C') # dM/dlogR
                return x, y / (2 * np.pi * x**2 * np.log(10.0)) # Sigma = 1/(2pi R^2) * dM/dlnR
            _s = [surface_density_profile(snap.R(pt).to('pc'), snap.mass[pt].to('Msun')) for pt in particle_types]
            jv.loglog(*zip(*_s), label=tuple(particle_type_meaning_full_strs), ls='-', color=None, out=profile_dir+'surface_density_{}.png'.format(snap.name), xlabel='Cylindrical radius $R$ (pc)', ylabel=r'Surface density $\Sigma$ ($M_\odot$/pc$^2$)', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks)

            # circular velocity profile (v_circ = sqrt(GM(<r)/r)), with contributions from gas, stars, BH, DM
            _s = [get_profile_with_defaults(snap.r(pt).to('pc'), snap.mass[pt].to('Msun'), kind='D') for pt in particle_types] # M(<r) for each particle type
            _s = [(x, np.sqrt(c.G * y / (x * u.pc))) for x, y in _s]
            _s_names = list(particle_type_meaning_full_strs)
            if center_around_BH:
                _s.insert(0, (_s[0][0], np.sqrt(c.G * snap.mass['bh'].to('Msun') / (_s[0][0] * u.pc))))
                _s_names.insert(0, 'Black Hole')
            _s.insert(0, (_s[0][0], np.sum([y for x, y in _s], axis=0)))
            _s_names.insert(0, 'Total')
            jv.loglog(*zip(*_s), label=tuple(n for n in _s_names), ls='-', color=None, out=profile_dir+'vcirc_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Circular velocity $v_\mathrm{circ}$ (km/s)', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks)
            
            # accretion rate 
            _vr = np.einsum('ij,ij->i', snap.vel0.to('km/s').value, (snap.pos0 / snap.r(0)[:, np.newaxis]).to(1).value)  # radial velocity
            _in, _out = _vr < 0, _vr >= 0
            Mdot_in = get_profile_with_defaults(snap.r(0)[_in].to('pc'), snap.mass0[_in].to('Msun') * np.abs(_vr[_in]), kind='C', force_dx=True) # dM/dR
            Mdot_out = get_profile_with_defaults(snap.r(0)[_out].to('pc'), snap.mass0[_out].to('Msun') * np.abs(_vr[_out]), kind='C', force_dx=True) # dM/dR
            jv.loglog(*zip(Mdot_in, Mdot_out), label=('Inflow', 'Outflow'), ls=('-', '--'), color=('C0', 'C1'), out=profile_dir+'mdot_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Accretion rate $\dot{M}$ ($M_\odot$/yr)', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks)

            # aspect ratio (fig 5 of FIF zoom-out draft)
            _rbin, _zrms = get_profile_with_defaults(snap.r(0).to('pc'), snap.pos0[:, 2].to('pc'), weight=snap.dens0, kind='rms')
            jv.loglog(_rbin, _zrms/_rbin, ls='-', color='black', out=profile_dir+'aspect_ratio_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Aspect ratio $H/R$', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks, ymin=1e-3, ymax=2)


            ### thermochemistry plots ### (e.g. fig 10 of FIF1)
            # temperature profile (gas, radiation, dust)
            _a = get_profile_with_defaults(snap.r(0).to('pc'), snap.temp0.to('K'), kind='A') # gas temperature
            _b = get_profile_with_defaults(snap.r(0).to('pc'), snap['PartType0','IRBand_Radiation_Temperature'].to('K'), kind='A') # radiation temperature
            _c = get_profile_with_defaults(snap.r(0).to('pc'), snap['PartType0','Dust_Temperature'].to('K'), kind='A') # dust temperature
            jv.loglog(*zip(_a, _b, _c), label=('Gas', 'Radiation', 'Dust'), ls=('-', '--', '-.'), color=None, out=profile_dir+'temp_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Temperature $T$ (K)', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks)


            # species fraction profile (free e, HII, HI, H2, metals)
            Z = snap.metals0[:, 0] # mass fraction of all metals
            Y = snap.metals0[:, 1] # mass fraction of helium
            X = (1-Y-Z) # mass fraction of hydrogen
            fe = snap['PartType0','ElectronAbundance']*X
            _a = get_profile_with_defaults(snap.r(0).to('pc'), fe, kind='A') # free electron fraction
            _b = get_profile_with_defaults(snap.r(0).to('pc'), snap['PartType0','HII'], kind='A') # HII fraction
            _c = get_profile_with_defaults(snap.r(0).to('pc'), snap['PartType0','NeutralHydrogenAbundance'], kind='A') # HI fraction
            _d = get_profile_with_defaults(snap.r(0).to('pc'), snap['PartType0','MolecularMassFraction'], kind='A') # H2 fraction
            _e = get_profile_with_defaults(snap.r(0).to('pc'), Z, kind='A') # metals fraction
            jv.loglog(*zip(_a, _b, _c, _d, _e), label=('Free Electrons', 'Ionized H (HII)', 'Atomic H (HI)', 'Molecular H (H2)', 'Metallicity (Z)'), ls=('-', '--', '-.', ':', (0, (6, 1))), color=None, out=profile_dir+'species_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Species mass fraction', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks, ymin=1e-5, ymax=1.4)

            # magnetic field profile (B mag rms, B_r, B_theta, B_phi)
            B = snap['PartType0','MagneticField'].to('G_cgs').value
            weight = snap.mass0.value
            _a = get_profile_with_defaults(snap.r(0).to('pc'), np.linalg.norm(B, axis=1), weight=weight, kind='rms') # rms of Bmag
            _b = get_profile_with_defaults(snap.r(0).to('pc'), (B[:, 0]*snap.x(0) + B[:, 1]*snap.y(0) + B[:, 2]*snap.z(0)) / snap.r(0), weight=weight, kind='rms') # rms of B_r
            _c = get_profile_with_defaults(snap.r(0).to('pc'), (snap.z(0)*(B[:, 0]*snap.x(0) + B[:, 1]*snap.y(0)) - B[:, 2]*snap.R(0)**2)/(snap.r(0)*snap.R(0)), weight=weight, kind='rms') # rms of B_theta
            _d = get_profile_with_defaults(snap.r(0).to('pc'), (B[:, 1]*snap.x(0) + B[:, 0]*snap.y(0))/snap.R(0), weight=weight, kind='rms') # rms of B_phi
            jv.loglog(*zip(_a, _b, _c, _d), label=(r'$\langle|B|\rangle^{1/2}$', 'Radial', 'Polodial', 'Toroidal'), ls=('-', '--', '-.', ':'), color=None, out=profile_dir+'Bfield_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Magnetic field strength $B$ (Gauss)', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks)

            # pressure/stress profile (kinetic, magnetic, thermal, radiation)
            Pkin = ((snap.dens0 * np.linalg.norm(snap.vel0.cgs, axis=1)**2)).to('g cm**-1 s**-2') 
            Pmag = np.sum(snap['PartType0','MagneticField'].to('G_cgs').cgs**2, axis=1)/(8*np.pi)
            Pth = ((5./3.-1.)*snap.dens0*snap['PartType0','InternalEnergy']).to('g cm**-1 s**-2')
            Prad = ((1./3.) * np.sum(snap['PartType0','PhotonEnergy'],axis=1) * snap.dens0/snap.mass0).to('g cm**-1 s**-2') # assume isotropic for now
            _a = get_profile_with_defaults(snap.r(0).to('pc'), Pkin.to('g cm**-1 s**-2').value, kind='A') # kinetic/ram pressure
            _b = get_profile_with_defaults(snap.r(0).to('pc'), Pmag.to('g cm**-1 s**-2').value, kind='A') # magnetic pressure
            _c = get_profile_with_defaults(snap.r(0).to('pc'), Pth.to('g cm**-1 s**-2').value, kind='A') # thermal pressure
            _d = get_profile_with_defaults(snap.r(0).to('pc'), Prad.to('g cm**-1 s**-2').value, kind='A') # radiation pressure
            jv.loglog(*zip(_a, _b, _c, _d), label=(r'$P_\mathrm{ram}$', r'$P_\mathrm{mag}$', r'$P_\mathrm{th}$', r'$P_\mathrm{rad}$'), ls=('-', '--', '-.', ':'), color=None)
            _a = get_profile_with_defaults(snap.r(0).to('pc'), ((snap.dens0 * snap.vel0[:, 2].cgs**2)).to('g cm**-1 s**-2').value, kind='A') # vertical ram pressure
            _bbins, _by = get_profile_with_defaults(snap.r(0).to('pc'), np.linalg.norm(snap.vel0.cgs, axis=1)**2, kind='mean') 
            _cbins, _cy = get_profile_with_defaults(snap.r(0).to('pc'), snap.dens0, kind='mean')
            _b = (_bbins, _cy * _by) # turbulent pressure
            # assert center_around_BH, "next line assumes bh centering"
            # _c = get_profile_with_defaults(snap.r(0).to('pc'), snap.dens0*c.G*snap.mass[bh_parttype][bh_index]/(snap.r(0).to('pc')**2), kind='A') # radial hydrostatic pressure
            # _c = (_c[0], np.concatenate(([0.0], np.cumsum((_c[1][:-1] + _c[1][1:]) / 2.0 * np.diff(_c[0])))))
            # _cbins, _cgrid = jaba.utils.grid.bin_particles_direct(
            #     np.log10([snap.R(0).to('cm').value, np.abs(snap.z(0).to('cm').value)]).T, 
            #     np.column_stack([
            #         snap.mass0.to('g').value * np.abs(snap.z(0).to('cm').value)/(snap.r(0).to('cm').value)**1.5,
            #         np.ones_like(snap.mass0.value)
            #     ]),
            #     mins=[np.log10(rsink.to('cm').value), 0.0],
            #     dims=[num_radial_bins, num_radial_bins],
            #     ret_bins=True,
            # ) 
            # _Rmid = (_cbins[0][:-1] + _cbins[0][1:])*0.5
            # _zmid = (_cbins[1][:-1] + _cbins[1][1:])*0.5
            # _rmid = (_Rmid**2 + _zmid**2)**0.5
            # _cy = c.G * snap.mass[bh_parttype][bh_index]/(2*np.pi*_Rmid*np.diff(_cbins[0])) * np.sum(_cgrid[0, ...]/_cgrid[1, ...], axis=1) # temp should be cumsum   # vertical hydrostatic pressure
            # _c = (_rmid * u.cm.to(u.pc), _cy)
            # print(np.shape(_c[0]), np.shape(_c[1])) #to finish debugging...
            jv.loglog(*zip(_b, _a), label=(r'$P_\mathrm{turb}$', r'$P_\mathrm{ram,z}$'), ls=('-', (0, (6, 1))), color=None) # , (0, (5, 1, 3, 1))
            jv.close(out=profile_dir+'pressure_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Pressure $P$ (g cm$^{-1}$ s$^{-2}$)', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks, loglog=True)


        ### general maps ###
        if general_maps:
            log_timing('making general maps...')
            maps_dir = output_dir + 'profiles/'
            os.makedirs(maps_dir, exist_ok=True)
            # temp: use pynbody for maps (TODO: port over jaba implementation)
            import pynbody
            import matplotlib.pyplot as plt
            with warnings.catch_warnings(): # for now, suppress pynbody warnings about log10 non-positive values
                warnings.simplefilter("ignore", category=RuntimeWarning)
            
                s = pynbody.new(gas=len(snap.mass0))
                s.gas['pos'] = np.array(snap.pos0.to('pc'), dtype=np.float64)
                s.gas['mass'] = np.array(snap.mass0.to('Msun'), dtype=np.float64)
                s.gas['smooth'] = np.array(snap.smooth0.to('pc'), dtype=np.float64)
                s.gas['vel'] = np.array(snap.vel0.to('km/s'), dtype=np.float64)
                s.gas['temp'] = np.array(snap.temp0.to('K'), dtype=np.float64)

                s.physical_units()
                s['pos'].units = 'pc'
                s['mass'].units = 'Msol'
                s['smooth'].units = 'pc'
                s['vel'].units = 'km s**-1'
                s['temp'].units = 'K'

                incls = [0, 90]#[0, 30, 60, 90, 120, 150, 180]
                for inclination in incls:
                    with s.rotate_x(inclination):
                        print('> doing inclination {}...'.format(inclination))
                        os.makedirs(maps_dir + 'inc{}/'.format(inclination), exist_ok=True)

                        r = np.sum(np.array(s['pos'].in_units('pc'), dtype=np.float64)**2, axis=1)
                        max_r_pc = np.nanmax(r)
                        min_r_pc = np.nanmin(r)
                        ooms = max(int(np.log10(max_r_pc/min_r_pc)), 1)
                        for oom in range(ooms):   # todo make this better...
                            r_pc = max_r_pc/10**oom
                            extent = (-r_pc, r_pc, -r_pc, r_pc)
                            _map = pynbody.plot.sph.image(s.gas, width=r_pc, units="m_p cm**-2", noplot=True, resolution=500, threaded=False)#, restrict_depth=True)
                            plt.title('gas')
                            plt.imshow(np.log10(_map), extent=extent, origin='lower')
                            plt.colorbar(label=r'log gas column density $\int \rho dz$ [$m_p cm^{-2}$]')
                            plt.xlabel('x [pc]')
                            plt.ylabel('y [pc]')
                            plt.savefig(maps_dir +'inc{}/map_dens_{}_oom{}_inc{}.png'.format(inclination, snap.name, oom, inclination))
                            plt.clf()

                            _map2 = pynbody.plot.sph.image(s.gas, qty='temp', width=r_pc, units="K", noplot=True, resolution=500, threaded=False)#, restrict_depth=True)
                            plt.title('gas')
                            plt.imshow(np.log10(_map2), extent=extent, origin='lower')
                            plt.colorbar(label=r'log mean gas temperature $T$ [$K$]')
                            plt.xlabel('x [pc]')
                            plt.ylabel('y [pc]')
                            plt.savefig(maps_dir +'inc{}/map_temp_{}_oom{}_inc{}.png'.format(inclination, snap.name, oom, inclination))
                            plt.clf()

                            s.gas['Pmag'] = np.sum(snap['PartType0', 'MagneticField']**2, axis=1)/(8*np.pi)
                            s.gas['Pmag'].units = 'K' # only to escape Gauss/unitless issue for plotting purposes
                            _map3 = pynbody.plot.sph.image(s.gas, qty='Pmag', width=r_pc, units="K", noplot=True, resolution=500, threaded=False)#, restrict_depth=True)
                            plt.title('gas')
                            plt.imshow(np.log10(_map3), extent=extent, origin='lower')
                            plt.colorbar(label=r'log mean magnetic field pressure $P_\mathrm{mag}$ [dyn/cm$^2$]')
                            plt.xlabel('x [pc]')
                            plt.ylabel('y [pc]')
                            plt.savefig(maps_dir +'inc{}/map_Pmag_{}_oom{}_inc{}.png'.format(inclination, snap.name, oom, inclination))
                            plt.clf()

                            s.gas['plasma beta'] = np.array(snap.dens0.to('g cm**-3')/(2*1.67262192e-24)*c.k_B.to('erg K**-1')*snap.temp0.to('K'), dtype=np.float64)/np.array(s.gas['Pmag'], dtype=np.float64)
                            s.gas['plasma beta'].units = 'K' # only to escape unitless issue for plotting purposes
                            _map4 = pynbody.plot.sph.image(s.gas, qty='plasma beta', width=r_pc, units='K', noplot=True, resolution=500, threaded=False)#, restrict_depth=True)
                            plt.title('gas')
                            plt.imshow(np.log10(_map4), extent=extent, origin='lower')
                            plt.colorbar(label=r'log mean gas $\beta_\mathrm{plasma}$')
                            plt.xlabel('x [pc]')
                            plt.ylabel('y [pc]')
                            plt.savefig(maps_dir +'inc{}/map_plasmabeta_{}_oom{}_inc{}.png'.format(inclination, snap.name, oom, inclination))
                            plt.clf()


        ### broad line region ###
        if BLR_analysis:
            from jaba.apps import blr
            import matplotlib.pyplot as plt
            import pynbody

            try:
                cpos
                cvel
            except NameError:
                bh_parttype = 3
                cpos = snap['PartType%d' % bh_parttype, 'Coordinates'][0][np.newaxis, :].to('pc')
                cvel = snap['PartType%d' % bh_parttype, 'Velocities'][0][np.newaxis, :].to('km/s')

            emissivity, alpha_abs, alpha_sca, alpha_eff = blr.get_Halpha_alpha_emissivity(snap)
            luminosity = (emissivity * 4*np.pi*u.sr * snap.mass0/snap.dens0).to('erg/s')
            volume = np.array((snap.mass0/snap.dens0).to('pc**3'), dtype=np.float64)
            rsink_pc = rsink.to('pc').value if 'rsink' in locals() else 0.0

            
            blr_dir = output_dir + 'blr/'
            os.makedirs(blr_dir, exist_ok=True)
    
            ################# radial profiles #################
            # quick settings for this section
            xmin = rsink.to('pc').value
            xmax = np.max(snap.r(0).to('pc').value)*0.99  # fraction of max radius to use as outer edge of radial profile (must be set manually) 
            force_regular_log_major_ticks=True
            force_minor_ticks=True
            
            log_timing('making BLR radial profiles...')
            blr_profile_dir = blr_dir + 'profiles/'
            os.makedirs(blr_profile_dir, exist_ok=True)
            def get_profile_with_defaults(*args, **kwargs):
                kwargs.setdefault('nbins', num_radial_bins)
                kwargs.setdefault('xlog', True)
                kwargs.setdefault('xmin', xmin)
                kwargs.setdefault('xmax', xmax)
                return get_profile(*args, **kwargs)

            # distribution of particle number
            _s = [get_profile_with_defaults(snap.r(pt).to('pc'), np.ones(len(snap.mass[pt])), kind='C') for pt in particle_types] # dN/dlogR for each particle type
            _meanL, _stdL = np.mean(luminosity), np.std(luminosity)
            _gtr_mean, _gtr_mean3 = luminosity > _meanL, luminosity > _meanL + 3*_stdL
            _s.append(get_profile_with_defaults(snap.r(0)[_gtr_mean].to('pc'), np.ones(len(snap.mass[0][_gtr_mean])), kind='C'))
            _s.append(get_profile_with_defaults(snap.r(0)[_gtr_mean3].to('pc'), np.ones(len(snap.mass[0][_gtr_mean3])), kind='C'))
            jv.loglog(*zip(*_s), label=tuple(particle_type_meaning_full_strs + [r'BLR ($L_\text{BLR} > \langle L\rangle$)', r'BLR ($L_\text{BLR} > \langle L\rangle + 3\sigma_L$)']), ls='-', color=None, out=blr_profile_dir+'number_distribution_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Particle number distribution ($\mathrm{d}N/\mathrm{d}\log r$)', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks)
            print('LBLR total = {}'.format(luminosity.sum()), 'LBLR mean = {}'.format(luminosity[_gtr_mean].sum()), 'LBLR 3sigma = {}'.format(luminosity[_gtr_mean3].sum()))

            ### dynamics plots ### (e.g. fig 7 of FIF1)
            # gas density profile
            _abins, _ay, _aymin, _aymax = get_profile_with_defaults(snap.r(0)[_gtr_mean].to('pc'), snap.dens0[_gtr_mean].to('m_p/cm**3').value, weight=luminosity[_gtr_mean], kind='AHI', p_volume=90.0)
            _a3bins, _a3y, _a3ymin, _a3ymax = get_profile_with_defaults(snap.r(0)[_gtr_mean3].to('pc'), snap.dens0[_gtr_mean3].to('m_p/cm**3').value, weight=luminosity[_gtr_mean3], kind='AHI', p_volume=90.0)
            _bbins, _by, _bymin, _bymax = get_profile_with_defaults(snap.r(0).to('pc'), snap.dens0.to('m_p/cm**3').value, weight=snap.mass0.to('Msun')/snap.dens0.to('Msun/pc**3'), kind='AHI', p_volume=90.0) # volume-weighted mean and 90 percentile volume
            _cbins, _cy, _cymin, _cymax = get_profile_with_defaults(snap.r(0).to('pc'), snap.dens0.to('m_p/cm**3').value, weight=snap.mass0.to('Msun'), kind='AHI', p_volume=90.0) # mass-weighted mean and 90 percentile volume
            jv.plot((_abins, _a3bins, _bbins, _cbins), (_ay, _a3y, _by, _cy), label=(r'BLR ($L_\text{BLR} > \langle L\rangle$)', r'BLR ($L_\text{BLR} > \langle L\rangle + 3\sigma_L$)', r'volume-weighted', r'mass-weighted'), ls=('-', '--', '-.', ':'))
            jv.fill_between((_abins, _a3bins, _bbins, _cbins), (_aymin, _a3ymin, _bymin, _cymin), (_aymax, _a3ymax, _bymax, _cymax), color=('C0', 'C1', 'C2', 'C3'), alpha=0.3, out=blr_profile_dir+'dens_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Mass density $\rho$ ($m_p$/cm$^3$)', loglog=True, xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks)

            # surface density (gas, stars, dark matter, etc.)
            def surface_density_profile(R, mass, **kwargs):
                x, y = get_profile_with_defaults(R, mass, kind='C', **kwargs) # dM/dlogR
                return x, y / (2 * np.pi * x**2 * np.log(10.0)) # Sigma = 1/(2pi R^2) * dM/dlnR
            _s = [surface_density_profile(snap.R(pt).to('pc'), snap.mass[pt].to('Msun'), nbins=20 if pt == "PartType4" else num_radial_bins) for pt in particle_types]
            _s.append(surface_density_profile(snap.R(0)[_gtr_mean].to('pc'), snap.mass[0][_gtr_mean].to('Msun')))
            _s.append(surface_density_profile(snap.R(0)[_gtr_mean3].to('pc'), snap.mass[0][_gtr_mean3].to('Msun')))
            jv.loglog(*zip(*_s), label=tuple(particle_type_meaning_full_strs + [r'BLR ($L_\text{BLR} > \langle L\rangle$)', r'BLR ($L_\text{BLR} > \langle L\rangle + 3\sigma_L$)']), ls='-', color=None, out=blr_profile_dir+'surface_density_{}.png'.format(snap.name), xlabel='Cylindrical radius $R$ (pc)', ylabel=r'Surface density $\Sigma$ ($M_\odot$/pc$^2$)', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks)
            
            # accretion rate 
            _vr = np.einsum('ij,ij->i', snap.vel0.to('km/s').value, (snap.pos0 / snap.r(0)[:, np.newaxis]).to(1).value)  # radial velocity
            _in, _out = _vr < 0, _vr >= 0
            Mdot_in = get_profile_with_defaults(snap.r(0)[_in].to('pc'), snap.mass0[_in].to('Msun') * np.abs(_vr[_in]), kind='C', force_dx=True) # dM/dR
            Mdot_out = get_profile_with_defaults(snap.r(0)[_out].to('pc'), snap.mass0[_out].to('Msun') * np.abs(_vr[_out]), kind='C', force_dx=True) # dM/dR
            Mdot_in2 = get_profile_with_defaults(snap.r(0)[_gtr_mean & _in].to('pc'), snap.mass0[_gtr_mean & _in].to('Msun') * np.abs(_vr[_gtr_mean & _in]), kind='C', force_dx=True) # dM/dR
            Mdot_out2 = get_profile_with_defaults(snap.r(0)[_gtr_mean & _out].to('pc'), snap.mass0[_gtr_mean & _out].to('Msun') * np.abs(_vr[_gtr_mean & _out]), kind='C', force_dx=True) # dM/dR
            Mdot_in3 = get_profile_with_defaults(snap.r(0)[_gtr_mean3 & _in].to('pc'), snap.mass0[_gtr_mean3 & _in].to('Msun') * np.abs(_vr[_gtr_mean3 & _in]), kind='C', force_dx=True) # dM/dR
            Mdot_out3 = get_profile_with_defaults(snap.r(0)[_gtr_mean3 & _out].to('pc'), snap.mass0[_gtr_mean3 & _out].to('Msun') * np.abs(_vr[_gtr_mean3 & _out]), kind='C', force_dx=True) # dM/dR
            jv.loglog(*zip(Mdot_in, Mdot_out, Mdot_in2, Mdot_out2, Mdot_in3, Mdot_out3), label=('Inflow', 'Outflow', r'Inflow BLR ($L_\text{BLR} > \langle L\rangle$)', r'Outflow BLR ($L_\text{BLR} > \langle L\rangle$)', r'Inflow BLR ($L_\text{BLR} > \langle L\rangle + 3\sigma_L$)', r'Outflow BLR ($L_\text{BLR} > \langle L\rangle + 3\sigma_L$)'), ls=('-', '--', '-', '--', '-', '--'), color=('C0', 'C0', 'C1', 'C1', 'C2', 'C2'), out=blr_profile_dir+'mdot_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Accretion rate $\dot{M}$ ($M_\odot$/yr)', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks)

            # aspect ratio (fig 5 of FIF zoom-out draft) --> TODO: use something like <z> +/- sigma_z instead of rms, since they aren't necessarily symmetric about z=0 
            _rbin, _zrms = get_profile_with_defaults(snap.r(0).to('pc'), snap.pos0[:, 2].to('pc'), weight=snap.dens0, kind='rms')
            _rbin2, _zmean2, _zrms2 = get_profile_with_defaults(snap.r(0)[_gtr_mean].to('pc'), snap.pos0[:, 2][_gtr_mean].to('pc'), weight=luminosity[_gtr_mean], kind='mean, rms')
            _rbin3, _zmean3, _zrms3 = get_profile_with_defaults(snap.r(0)[_gtr_mean3].to('pc'), snap.pos0[:, 2][_gtr_mean3].to('pc'), weight=luminosity[_gtr_mean3], kind='mean, rms')
            jv.loglog((_rbin2, _rbin3), (_zrms2/_rbin2, _zrms3/_rbin3), label=(r'mean BLR ($L_\text{BLR} > \langle L\rangle$)', r'mean BLR ($L_\text{BLR} > \langle L\rangle + 3\sigma_L$)'), ls='--')
            jv.loglog((_rbin, _rbin2, _rbin3), (_zrms/_rbin, _zrms2/_rbin2, _zrms3/_rbin3), label=('Gas', r'BLR ($L_\text{BLR} > \langle L\rangle$)', r'BLR ($L_\text{BLR} > \langle L\rangle + 3\sigma_L$)'),  ls='-', out=blr_profile_dir+'aspect_ratio_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Aspect ratio $H/R$', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks, ymin=1e-3, ymax=2)

            ### thermochemistry plots ### (e.g. fig 10 of FIF1)
            # temperature profile (gas, radiation, dust)
            _a = get_profile_with_defaults(snap.r(0).to('pc'), snap.temp0.to('K'), kind='A') # gas temperature
            _b = get_profile_with_defaults(snap.r(0).to('pc'), snap['PartType0','IRBand_Radiation_Temperature'].to('K'), kind='A') # radiation temperature
            _c = get_profile_with_defaults(snap.r(0).to('pc'), snap['PartType0','Dust_Temperature'].to('K'), kind='A') # dust temperature
            _a2 = get_profile_with_defaults(snap.r(0)[_gtr_mean].to('pc'), snap.temp0[_gtr_mean].to('K'), kind='A') # gas temperature
            _b2 = get_profile_with_defaults(snap.r(0)[_gtr_mean].to('pc'), snap['PartType0','IRBand_Radiation_Temperature'][_gtr_mean].to('K'), kind='A') # radiation temperature
            _c2 = get_profile_with_defaults(snap.r(0)[_gtr_mean].to('pc'), snap['PartType0','Dust_Temperature'][_gtr_mean].to('K'), kind='A') # dust temperature
            jv.loglog(*zip(_a, _b, _c, _a2, _b2, _c2), label=('Gas', 'Radiation', 'Dust', r'BLR Gas ($L_\text{BLR} > \langle L\rangle$)', r'BLR Radiation ($L_\text{BLR} > \langle L\rangle$)', r'BLR Dust ($L_\text{BLR} > \langle L\rangle$)'), ls=('-', '--', '-.', '-', '--', '-.'), color=None, out=blr_profile_dir+'temp_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Temperature $T$ (K)', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks)

            # species fraction profile (free e, HII, HI, H2, metals)
            Z = snap.metals0[:, 0] # mass fraction of all metals
            Y = snap.metals0[:, 1] # mass fraction of helium
            X = (1-Y-Z) # mass fraction of hydrogen
            fe = snap['PartType0','ElectronAbundance']*X
            _a = get_profile_with_defaults(snap.r(0).to('pc'), fe, weight=snap.mass0, kind='A') # free electron fraction
            _b = get_profile_with_defaults(snap.r(0).to('pc'), snap['PartType0','HII'], weight=snap.mass0, kind='A') # HII fraction
            _c = get_profile_with_defaults(snap.r(0).to('pc'), snap['PartType0','NeutralHydrogenAbundance'], weight=snap.mass0, kind='A') # HI fraction
            _d = get_profile_with_defaults(snap.r(0).to('pc'), snap['PartType0','MolecularMassFraction'], weight=snap.mass0, kind='A') # H2 fraction
            _e = get_profile_with_defaults(snap.r(0).to('pc'), Z, weight=snap.mass0, kind='A') # metals fraction
            _a2 = get_profile_with_defaults(snap.r(0).to('pc'), fe, weight=luminosity, kind='A') # free electron fraction
            _b2 = get_profile_with_defaults(snap.r(0).to('pc'), snap['PartType0','HII'], weight=luminosity, kind='A') # HII fraction
            _c2 = get_profile_with_defaults(snap.r(0).to('pc'), snap['PartType0','NeutralHydrogenAbundance'], weight=luminosity, kind='A') # HI fraction
            _d2 = get_profile_with_defaults(snap.r(0).to('pc'), snap['PartType0','MolecularMassFraction'], weight=luminosity, kind='A') # H2 fraction
            _e2 = get_profile_with_defaults(snap.r(0).to('pc'), Z, weight=luminosity, kind='A') # metals fraction
            jv.loglog(*zip(_a, _b, _c, _d, _e, _a2, _b2, _c2, _d2, _e2), label=('Free Electrons', 'Ionized H (HII)', 'Atomic H (HI)', 'Molecular H (H2)', 'Metallicity (Z)', r'BLR Free Electrons ($L_\text{BLR} > \langle L\rangle$)', r'BLR Ionized H (HII) ($L_\text{BLR} > \langle L\rangle$)', r'BLR Atomic H (HI) ($L_\text{BLR} > \langle L\rangle$)', r'BLR Molecular H (H2) ($L_\text{BLR} > \langle L\rangle$)', r'BLR Metallicity (Z) ($L_\text{BLR} > \langle L\rangle$)'), ls=('-', '--', '-.', ':', (0, (6, 1)), '-', '--', '-.', ':', (0, (6, 1))), color=None, out=blr_profile_dir+'species_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Species mass fraction', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks, ymin=1e-5, ymax=1.4)

            # magnetic field profile (Bmag, Br, Btheta, Bphi)
            B = snap['PartType0','MagneticField'].to('G_cgs').value
            weight = snap.mass0.value
            _a = get_profile_with_defaults(snap.r(0).to('pc'), np.linalg.norm(B, axis=1), weight=weight, kind='rms') # rms of Bmag
            _b = get_profile_with_defaults(snap.r(0).to('pc'), (B[:, 0]*snap.x(0) + B[:, 1]*snap.y(0) + B[:, 2]*snap.z(0)) / snap.r(0), weight=weight, kind='rms') # rms of B_r
            _c = get_profile_with_defaults(snap.r(0).to('pc'), (snap.z(0)*(B[:, 0]*snap.x(0) + B[:, 1]*snap.y(0)) - B[:, 2]*snap.R(0)**2)/(snap.r(0)*snap.R(0)), weight=weight, kind='rms') # rms of B_theta
            _d = get_profile_with_defaults(snap.r(0).to('pc'), (B[:, 1]*snap.x(0) + B[:, 0]*snap.y(0))/snap.R(0), weight=weight, kind='rms') # rms of B_phi
            weight = luminosity.value
            _a2 = get_profile_with_defaults(snap.r(0).to('pc'), np.linalg.norm(B, axis=1), weight=weight, kind='rms') # rms of Bmag
            _b2 = get_profile_with_defaults(snap.r(0).to('pc'), (B[:, 0]*snap.x(0) + B[:, 1]*snap.y(0) + B[:, 2]*snap.z(0)) / snap.r(0), weight=weight, kind='rms') # rms of B_r
            _c2 = get_profile_with_defaults(snap.r(0).to('pc'), (snap.z(0)*(B[:, 0]*snap.x(0) + B[:, 1]*snap.y(0)) - B[:, 2]*snap.R(0)**2)/(snap.r(0)*snap.R(0)), weight=weight, kind='rms') # rms of B_theta
            _d2 = get_profile_with_defaults(snap.r(0).to('pc'), (B[:, 1]*snap.x(0) + B[:, 0]*snap.y(0))/snap.R(0), weight=weight, kind='rms') # rms of B_phi
            jv.loglog(*zip(_a, _b, _c, _d, _a2, _b2, _c2, _d2), label=(r'$\langle|B|\rangle^{1/2}$', 'Radial', 'Polodial', 'Toroidal', r'BLR $\langle|B|\rangle^{1/2}$', r'BLR Radial', r'BLR Polodial', r'BLR Toroidal'), ls=('-', '--', '-.', ':', '-', '--', '-.', ':'), color=None, out=blr_profile_dir+'Bfield_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Magnetic field strength $B$ (Gauss)', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks)

            # pressure profile (Pkin, Pmag, Pth, Prad)
            Pkin = ((snap.dens0 * np.linalg.norm(snap.vel0.cgs, axis=1)**2)).to('g cm**-1 s**-2') 
            Pmag = np.sum(snap['PartType0','MagneticField'].to('G_cgs').cgs**2, axis=1)/(8*np.pi)
            Pth = ((5./3.-1.)*snap.dens0*snap['PartType0','InternalEnergy']).to('g cm**-1 s**-2')
            Prad = ((1./3.) * np.sum(snap['PartType0','PhotonEnergy'],axis=1) * snap.dens0/snap.mass0).to('g cm**-1 s**-2') # assume isotropic for now
            _a = get_profile_with_defaults(snap.r(0).to('pc'), Pkin.to('g cm**-1 s**-2').value, kind='A') # kinetic/ram pressure
            _b = get_profile_with_defaults(snap.r(0).to('pc'), Pmag.to('g cm**-1 s**-2').value, kind='A') # magnetic pressure
            _c = get_profile_with_defaults(snap.r(0).to('pc'), Pth.to('g cm**-1 s**-2').value, kind='A') # thermal pressure
            _d = get_profile_with_defaults(snap.r(0).to('pc'), Prad.to('g cm**-1 s**-2').value, kind='A') # radiation pressure
            jv.loglog(*zip(_a, _b, _c, _d), label=(r'$P_\mathrm{ram}$', r'$P_\mathrm{mag}$', r'$P_\mathrm{th}$', r'$P_\mathrm{rad}$'), ls=('-', '--', '-.', ':'), color=None)
            _a = get_profile_with_defaults(snap.r(0).to('pc'), ((snap.dens0 * snap.vel0[:, 2].cgs**2)).to('g cm**-1 s**-2').value, kind='A') # vertical ram pressure
            _bbins, _by = get_profile_with_defaults(snap.r(0).to('pc'), np.linalg.norm(snap.vel0.cgs, axis=1)**2, kind='mean') 
            _cbins, _cy = get_profile_with_defaults(snap.r(0).to('pc'), snap.dens0, kind='mean')
            _b = (_bbins, _cy * _by) # turbulent pressure
            jv.loglog(*zip(_b, _a), label=(r'$P_\mathrm{turb}$', r'$P_\mathrm{ram,z}$'), ls=('-', (0, (6, 1))), color=None) # , (0, (5, 1, 3, 1))

            Pkin = ((snap.dens0 * np.linalg.norm(snap.vel0.cgs, axis=1)**2)).to('g cm**-1 s**-2') 
            Pmag = np.sum(snap['PartType0','MagneticField'].to('G_cgs').cgs**2, axis=1)/(8*np.pi)
            Pth = ((5./3.-1.)*snap.dens0*snap['PartType0','InternalEnergy']).to('g cm**-1 s**-2')
            Prad = ((1./3.) * np.sum(snap['PartType0','PhotonEnergy'],axis=1) * snap.dens0/snap.mass0).to('g cm**-1 s**-2') # assume isotropic for now
            _a = get_profile_with_defaults(snap.r(0).to('pc'), Pkin.to('g cm**-1 s**-2').value, weight=luminosity, kind='A') # kinetic/ram pressure
            _b = get_profile_with_defaults(snap.r(0).to('pc'), Pmag.to('g cm**-1 s**-2').value, weight=luminosity, kind='A') # magnetic pressure
            _c = get_profile_with_defaults(snap.r(0).to('pc'), Pth.to('g cm**-1 s**-2').value, weight=luminosity, kind='A') # thermal pressure
            _d = get_profile_with_defaults(snap.r(0).to('pc'), Prad.to('g cm**-1 s**-2').value, weight=luminosity, kind='A') # radiation pressure
            jv.loglog(*zip(_a, _b, _c, _d), label=(r'BLR $P_\mathrm{ram}$', r'BLR $P_\mathrm{mag}$', r'BLR $P_\mathrm{th}$', r'BLR $P_\mathrm{rad}$'), ls=('-', '--', '-.', ':'), color=None)
            _a = get_profile_with_defaults(snap.r(0).to('pc'), ((snap.dens0 * snap.vel0[:, 2].cgs**2)).to('g cm**-1 s**-2').value, weight=luminosity, kind='A') # vertical ram pressure
            _bbins, _by = get_profile_with_defaults(snap.r(0).to('pc'), np.linalg.norm(snap.vel0.cgs, axis=1)**2, weight=luminosity, kind='mean') 
            _cbins, _cy = get_profile_with_defaults(snap.r(0).to('pc'), snap.dens0, weight=luminosity, kind='mean')
            _b = (_bbins, _cy * _by) # turbulent pressure
            jv.loglog(*zip(_b, _a), label=(r'BLR $P_\mathrm{turb}$', r'BLR $P_\mathrm{ram,z}$'), ls=('-', (0, (6, 1))), color=None) # , (0, (5, 1, 3, 1))
            jv.close(out=blr_profile_dir+'pressure_{}.png'.format(snap.name), xlabel='Spherical radius $r$ (pc)', ylabel=r'Pressure $P$ (g cm$^{-1}$ s$^{-2}$)', xmin=xmin, xmax=xmax, force_regular_log_major_ticks=force_regular_log_major_ticks, force_minor_ticks=force_minor_ticks, loglog=True)




            ################# maps #################
            log_timing('making BLR maps...')

            s = pynbody.new(gas=len(snap.mass0))
            s.gas['pos'] = np.array((snap.pos0 - cpos).to('pc').value, dtype=np.float64)
            s.gas['vel'] = np.array((snap.vel0 - cvel).to('km/s').value, dtype=np.float64)
            s.gas['lum'] = np.array(luminosity.to('erg/s').value, dtype=np.float64)
            s.gas['alpha_abs'] = np.array(alpha_abs.to('pc**-1').value, dtype=np.float64)
            s.gas['alpha_sca'] = np.array(alpha_sca.to('pc**-1').value, dtype=np.float64)
            s.gas['alpha_eff'] = np.array(alpha_eff.to('pc**-1').value, dtype=np.float64)
            s.gas['pos'].units = 'pc'
            s.gas['vel'].units = 'km s**-1'
            s.gas['lum'].units = 'erg s**-1'
            s.gas['alpha_abs'].units = 'pc**-1'
            s.gas['alpha_sca'].units = 'pc**-1'
            s.gas['alpha_eff'].units = 'pc**-1'

            incls = [0, 15, 30, 45]
            tau_ngrid = 128
            nbins = 150
            flux_factor = (3.086e18)**-2

            def _lookup_tau(tau_grid, pos, mins, dx):
                idx = np.floor((pos - mins) / dx).astype(np.int64)
                m = np.all((idx >= 0) & (idx < tau_grid.shape[0]), axis=1)
                tau = np.zeros(pos.shape[0], dtype=np.float64)
                tau[m] = tau_grid[idx[m, 0], idx[m, 1], idx[m, 2]]
                return tau

            r0 = np.sqrt(np.sum((snap.pos0 - cpos)**2, axis=1)).to('pc')
            rmax = np.nanmax(r0.value)
            rmin = np.nanmin(r0.value[r0.value > 0]) if np.any(r0.value > 0) else rsink_pc
            tau_scale_min = max(rsink_pc, rmin, 1e-4)
            tau_scale_max = max(rmax, tau_scale_min)
            tau_scales = 10.0 ** np.arange(np.floor(np.log10(tau_scale_min)), np.ceil(np.log10(tau_scale_max)) + 1)
            rbins, lint = get_profile(r0, luminosity.to('erg/s').value, kind='C', nbins=nbins, xlog=True, xmin=max(rsink_pc, rmin), xmax=rmax)
            jv.plot(rbins, lint, label=('intrinsic',), xlog=True, ylog=True, xlabel='spherical radius (pc)', ylabel=r'$dL/d\log r$ (erg/s)', show_legend=True, out=blr_dir + 'radial_intrinsic.png')

            line_intr_bins = None
            line_esc_bins = None
            line_intr = list()
            line_esc = list()
            radial_esc = list()
            tau_bins = None
            tau_eff_radial = list()
            tau_abs_radial = list()
            tau_sca_radial = list()
            vlim = 5000.0 #np.nanmax(np.abs(np.array(snap.vel0.to('km/s'), dtype=np.float64)))

            for inclination in incls:
                with s.rotate_x(inclination):
                    pos = np.array(s.gas['pos'].in_units('pc'), dtype=np.float64)
                    vel = np.array(s.gas['vel'].in_units('km s**-1'), dtype=np.float64)
                    lum = np.array(s.gas['lum'], dtype=np.float64)
                    aabs = np.array(s.gas['alpha_abs'], dtype=np.float64)
                    asca = np.array(s.gas['alpha_sca'], dtype=np.float64)
                    aeff = np.array(s.gas['alpha_eff'], dtype=np.float64)
                    max_r_pc = np.nanmax(np.sqrt(np.sum(pos**2, axis=1)))
                    min_r_pc = np.nanmin(np.sqrt(np.sum(pos**2, axis=1)))

                    scale_results = list()
                    for oom, width in enumerate(tau_scales):
                        dims2 = (512, 512)
                        mins3 = np.array([-width, -width, -width], dtype=np.float64)
                        maxs3 = -mins3
                        dx3 = (maxs3 - mins3) / tau_ngrid
                        cell_vol = dx3[0] * dx3[1] * dx3[2]
                        lum3 = bin_particles_direct(pos, lum * volume / cell_vol, mins=mins3, maxs=maxs3, dims=(tau_ngrid, tau_ngrid, tau_ngrid))
                        aabs3 = bin_particles_direct(pos, aabs * volume / cell_vol, mins=mins3, maxs=maxs3, dims=(tau_ngrid, tau_ngrid, tau_ngrid))
                        asca3 = bin_particles_direct(pos, asca * volume / cell_vol, mins=mins3, maxs=maxs3, dims=(tau_ngrid, tau_ngrid, tau_ngrid))
                        aeff3 = bin_particles_direct(pos, aeff * volume / cell_vol, mins=mins3, maxs=maxs3, dims=(tau_ngrid, tau_ngrid, tau_ngrid))
                        tau_abs3 = np.cumsum(aabs3[:, :, ::-1], axis=2)[:, :, ::-1] * dx3[2]
                        tau_sca3 = np.cumsum(asca3[:, :, ::-1], axis=2)[:, :, ::-1] * dx3[2]
                        tau3 = np.cumsum(aeff3[:, :, ::-1], axis=2)[:, :, ::-1] * dx3[2]
                        lum3_esc = lum3 * np.exp(-tau3)
                        lum2 = np.sum(lum3, axis=2) * dx3[2] * flux_factor
                        lum2_esc = np.sum(lum3_esc, axis=2) * dx3[2] * flux_factor
                        tau2 = tau3[:, :, 0]
                        scale_results.append((width, mins3, dx3, tau3, tau_abs3, tau_sca3))

                        extent = (-width, width, -width, width)
                        os.makedirs(blr_dir + 'inc{}/'.format(inclination), exist_ok=True)

                        plt.clf()
                        plt.imshow(np.log10(lum2).T, extent=extent, origin='lower', cmap='plasma')
                        plt.colorbar(label=r'log $F_{\mathrm{BLR}}$ [erg s$^{-1}$ cm$^{-2}$]')
                        plt.xlabel('x [pc]')
                        plt.ylabel('y [pc]')
                        plt.title('Intrinsic BLR emission')
                        plt.savefig(blr_dir + 'inc{}/intrinsic_oom{}_inc{}.png'.format(inclination, oom, inclination))

                        plt.clf()
                        plt.imshow(np.log10(lum2_esc).T, extent=extent, origin='lower', cmap='plasma')
                        plt.colorbar(label=r'log $F_{\mathrm{BLR}}$ [erg s$^{-1}$ cm$^{-2}$]')
                        plt.xlabel('x [pc]')
                        plt.ylabel('y [pc]')
                        plt.title('Extincted BLR emission')
                        plt.savefig(blr_dir + 'inc{}/extincted_oom{}_inc{}.png'.format(inclination, oom, inclination))

                        plt.clf()
                        plt.imshow(np.log10(tau2).T, extent=extent, origin='lower', cmap='viridis')
                        plt.colorbar(label=r'log $\tau_{\mathrm{eff}}$')
                        plt.xlabel('x [pc]')
                        plt.ylabel('y [pc]')
                        plt.title('Extincted optical depth')
                        plt.savefig(blr_dir + 'inc{}/tau_eff_oom{}_inc{}.png'.format(inclination, oom, inclination))

                    tau_eff_part = np.zeros(pos.shape[0], dtype=np.float64)
                    tau_abs_part = np.zeros(pos.shape[0], dtype=np.float64)
                    tau_sca_part = np.zeros(pos.shape[0], dtype=np.float64)
                    rvals = r0.value
                    prev_scale = 0.0
                    for width, mins3, dx3, tau3, tau_abs3, tau_sca3 in scale_results:
                        mask = (rvals > prev_scale) & (rvals <= width)
                        if np.any(mask):
                            tau_eff_part[mask] = _lookup_tau(tau3, pos[mask], mins3, dx3)
                            tau_abs_part[mask] = _lookup_tau(tau_abs3, pos[mask], mins3, dx3)
                            tau_sca_part[mask] = _lookup_tau(tau_sca3, pos[mask], mins3, dx3)
                        prev_scale = width
                    if prev_scale < rvals.max():
                        mask = rvals > prev_scale
                        if np.any(mask):
                            width, mins3, dx3, tau3, tau_abs3, tau_sca3 = scale_results[-1]
                            tau_eff_part[mask] = _lookup_tau(tau3, pos[mask], mins3, dx3)
                            tau_abs_part[mask] = _lookup_tau(tau_abs3, pos[mask], mins3, dx3)
                            tau_sca_part[mask] = _lookup_tau(tau_sca3, pos[mask], mins3, dx3)
                    lum_esc = lum * np.exp(-tau_eff_part)

                    rbins_i, lint_i = get_profile(r0, lum, nbins=nbins, kind='C', xlog=True, xmin=max(rsink_pc, rmin), xmax=rmax)
                    _, lext_i = get_profile(r0, lum_esc, nbins=nbins, kind='C', xlog=True, xmin=max(rsink_pc, rmin), xmax=rmax)
                    vbins_i, lvel_i = get_profile(vel[:, 2], lum, nbins=nbins, kind='A', xmin=-vlim, xmax=vlim)
                    _, lvel_ei = get_profile(vel[:, 2], lum_esc, nbins=nbins, kind='A', xmin=-vlim, xmax=vlim)
                    rbins_t, tau_eff_i = get_profile(r0, tau_eff_part, nbins=nbins, kind='C', xlog=True, xmin=max(rsink_pc, rmin), xmax=rmax)
                    _, tau_abs_i = get_profile(r0, tau_abs_part, nbins=nbins, kind='C', xlog=True, xmin=max(rsink_pc, rmin), xmax=rmax)
                    _, tau_sca_i = get_profile(r0, tau_sca_part, nbins=nbins, kind='C', xlog=True, xmin=max(rsink_pc, rmin), xmax=rmax)

                    if line_intr_bins is None:
                        line_intr_bins = vbins_i
                        line_esc_bins = vbins_i
                    if tau_bins is None:
                        tau_bins = rbins_t
                    line_intr.append(lvel_i)
                    line_esc.append(lvel_ei)
                    radial_esc.append(lext_i)
                    tau_eff_radial.append(tau_eff_i)
                    tau_abs_radial.append(tau_abs_i)
                    tau_sca_radial.append(tau_sca_i)

            plt.clf()
            jv.plot(rbins_i, tuple(radial_esc), label=tuple('inc {} deg'.format(i) for i in incls), ls=('-', '--', '-.', ':'), xlog=True, ylog=True, xlabel='spherical radius (pc)', ylabel=r'$dL/d\log r$ (erg/s)', show_legend=True, clf_before=True, out=blr_dir + 'radial_extincted.png')
            plt.clf()
            jv.plot(line_intr_bins, tuple(line_intr), label=tuple('inc {} deg'.format(i) for i in incls), ls=('-', '--', '-.', ':'), xlabel=r'$v_{\mathrm{los}}$ (km/s)', ylabel=r'$dL/dv$ (erg s$^{-1}$ km$^{-1}$ s)', show_legend=True, clf_before=True, out=blr_dir + 'line_intrinsic.png')
            plt.clf()
            jv.plot(line_esc_bins, tuple(line_esc), label=tuple('inc {} deg'.format(i) for i in incls), ls=('-', '--', '-.', ':'), xlabel=r'$v_{\mathrm{los}}$ (km/s)', ylabel=r'$dL/dv$ (erg s$^{-1}$ km$^{-1}$ s)', show_legend=True, clf_before=True, out=blr_dir + 'line_extincted.png')
            plt.clf()
            jv.plot(tau_bins, tuple(tau_eff_radial), label=tuple('inc {} deg'.format(i) for i in incls), ls=('-', '--', '-.', ':'), xlog=True, ylog=True, xlabel='spherical radius (pc)', ylabel=r'$\tau_{\mathrm{eff}}$', show_legend=True, clf_before=True, out=blr_dir + 'tau_eff_radial.png')
            plt.clf()
            jv.plot(tau_bins, tuple(tau_abs_radial), label=tuple('inc {} deg'.format(i) for i in incls), ls=('-', '--', '-.', ':'), xlog=True, ylog=True, xlabel='spherical radius (pc)', ylabel=r'$\tau_{\mathrm{abs}}$', show_legend=True, clf_before=True, out=blr_dir + 'tau_abs_radial.png')
            plt.clf()
            jv.plot(tau_bins, tuple(tau_sca_radial), label=tuple('inc {} deg'.format(i) for i in incls), ls=('-', '--', '-.', ':'), xlog=True, ylog=True, xlabel='spherical radius (pc)', ylabel=r'$\tau_{\mathrm{es}}$', show_legend=True, clf_before=True, out=blr_dir + 'tau_es_radial.png')

        log_timing()


def main():
    log_timing(f"getting filearguments...")
    filepath, analysis_dir, debug_flag = get_file_arguments(str, str, int, fill_empties_with_none=True)
    
    assert filepath is not None, 'Need to enter a filepath.'
    if analysis_dir is None:
        print('no analysis output directory specified, defaulting to output in local folder..')
        analysis_dir='.'
    print('debug flag {} -> {}'.format(debug_flag, bool(debug_flag)))
    quick_check(filepath, output_dir=analysis_dir, debugging=bool(debug_flag))

if __name__ == '__main__':
    main()

