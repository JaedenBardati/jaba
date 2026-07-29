#!/usr/bin/python3
"""
Runs a quick check on simulations to determine if they are reasonable to continue running.
  Also has a variety of convience functions for follow-up interactive analysis.
  Requires my "jaba" package (https://github.com/JaedenBardati/jaba/tree/main)
Jaeden Bardati 2025+

Basic class/function structure:
  FILETYPE LOADER (e.g. HDF5_Snapshot) --> SNAPSHOT TYPE CONVENIENCE FUNCTION (e.g. GIZMO_Snapshot) --> STANDARD SNAPSHOT FORM (e.g. Standardized_GIZMO_Snapshot)
  QUICK CHECK ANALYSIS <-- GENERAL ANALYSIS + GENERAL PLOTTING

Currently implemented checks:
  - Mass density radial plot
  - Particle number radial plot
  - Simple maps of mass density
  - ...

TO DO: 
 - Integrate into jaba package
 - Add more checks and more plotting functions  
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


def get1Dmean(x, qty, weight=None, nbins=100, xmin=None, xmax=None, xlog=False, sum_instead=False, cumsum_instead=False, reverse_x=False):  
    """
    Plots a general 1d plot. Not super efficient.
    """
    assert not sum_instead or weight is None, 'Weighted sum not supported.'
    assert not cumsum_instead or weight is None, 'Weighted cummulative sum not supported.'
    assert not sum_instead or not cumsum_instead, 'Cannot have both sum_instead=True and cumsum_instead=True'

    x = np.asarray(x)
    qty = np.asarray(qty)
    if weight is not None:
        weight = np.asarray(weight)

    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)

    if xlog:
        xbins = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
    else:
        xbins = np.linspace(xmin, xmax, nbins+1)

    if reverse_x:
        xbins = xbins[::-1]

    qty_bins = np.zeros(nbins)
    if cumsum_instead:
        cumsum = 0.0
    for i in range(nbins):
        if i == 0:
            mask = np.logical_and(xbins[i] <= x, x <= xbins[i+1])
        else:
            mask = np.logical_and(xbins[i] < x, x <= xbins[i+1])

        if np.any(mask):
            if weight is None:
                if sum_instead:
                    qty_bins[i] = np.sum(qty[mask])
                elif cumsum_instead:
                    binsum = np.sum(qty[mask])
                    qty_bins[i] = cumsum+binsum
                    cumsum += binsum
                else:
                    qty_bins[i] = np.mean(qty[mask])
            else:
                qty_bins[i] = np.average(qty[mask], weights=weight[mask])
        else:
            qty_bins[i] = np.nan

    x_mid = 0.5*(xbins[1:] + xbins[:-1])
    return x_mid, qty_bins



def plot1Dmean(x, qty, weights=None, labels=None, linestyles='-', nbins=100, xlog=False, ylog=False, xlabel=None, ylabel=None, out=None, show=False, sum_instead=False, cumsum_instead=False, reverse_x=False):
    # TODO DELETE
    if not isinstance(weights, tuple):
        weights = (weights,)
    if not isinstance(labels, tuple):
        labels = (labels,)*len(weights)
    if not isinstance(linestyles, tuple):
        linestyles = (linestyles,)*len(weights)
    
    _out, _show, _show_legend = None, False, False
    for i, w in enumerate(weights):
        _x, _y = get1Dmean(x, qty, weight=w, nbins=nbins, xlog=xlog, sum_instead=sum_instead)
        if i == len(weights) - 1 and (out is not None or show): # if last iteration (and print is happening)
            _out = out
            _show = show
            _show_legend = True if any(l is not None for l in labels) else False
        jv.plot(_x, _y, label=labels[i], ls=linestyles[i], xlog=xlog, ylog=ylog, xlabel=xlabel, ylabel=ylabel, show_legend=_show_legend, out=_out, show=_show)

########################################################################################################################################################################
#####################                                              QUICK CHECK SIMULATION                                                          #####################
########################################################################################################################################################################

def quick_check(filepath, output_dir=None, debugging=False, center_around_BH=True):
    """ THIS IS THE MAIN FUNCTION TO CHANGE..."""
    if output_dir is None:
        output_dir='.'
    output_dir+='/'

    # load snapshot/simulation
    log_timing(f"loading snapshot at {filepath} ...")
    snap = jaba.load(filepath) # should likely be debugging=debugging
    
    # pre-load required data
    snap.pos0, snap.dens0, snap.mass0,

    # center around particle
    if center_around_BH:
       bh_parttype=3
       bh_index=0
       cpos = snap['PartType%d'%bh_parttype, 'Coordinates'][bh_index][np.newaxis, :]
       cvel = snap['PartType%d'%bh_parttype, 'Velocities'][bh_index][np.newaxis, :]

    # check data
    log_timing(f"running check plots ...")
    r = np.sqrt(np.sum((snap.pos0 - cpos)**2, axis=1))  # spherical radius

    jv.use_paper(None)
    #jv.use_paper('ApJ', 'singlecolumn')
    plot1Dmean(r.to('pc'), snap.dens0.to('g/cm**3'), weights=(None, snap.mass0, snap.mass0/snap.dens0), 
               labels=('mass-weighted', 'volume-weighted', 'particle mean'), 
               linestyles=('-', '-', '--'),
               xlabel=r'Spherical radius $r$ [pc]', ylabel=r'Mass density $\rho$ [g/cm$^3$]', 
               xlog=True, ylog=True, out=output_dir+'dens_{}.png'.format(snap.name))  # mass density plot
    plot1Dmean(r.to('pc'), np.ones(len(snap.dens0)), weights=(None,), 
               labels=('particle',), 
               linestyles=('-',),
               xlabel=r'Spherical radius $r$ [pc]', ylabel=r'Particle number', 
               xlog=True, ylog=True, out=output_dir+'number_{}.png'.format(snap.name), sum_instead=True)  # particle number plot
    
    import matplotlib.pyplot as plt
    plt.clf()
    plt.hist(np.array(snap.mass0.to('Msun')), bins=50)
    plt.xlabel(r'Particle mass ($M_\odot$)')
    plt.savefig(output_dir+'hist_mass_{}.png'.format(snap.name))
    plt.clf()
    plt.hist(np.array(r.to('pc')), bins=50)
    plt.xlabel(r'Particle radius (pc)')
    plt.savefig(output_dir+'hist_radius_{}.png'.format(snap.name))
    plt.clf()
    plt.hist(np.array(snap.dens0.to('g/cm**3')), bins=50)
    plt.xlabel(r'Particle density (g/cm$^3$)')
    plt.savefig(output_dir+'hist_dens_{}.png'.format(snap.name))
    plt.clf()
    plt.hist(np.log10(np.array(snap.mass0.to('Msun'))), bins=50)
    plt.xlabel(r'log particle mass ($M_\odot$)')
    plt.savefig(output_dir+'hist_logmass_{}.png'.format(snap.name))
    plt.clf()
    plt.hist(np.log10(np.array(r.to('pc'))), bins=50)
    plt.xlabel(r'log particle radius (pc)')
    plt.savefig(output_dir+'hist_logradius_{}.png'.format(snap.name))
    plt.clf()
    plt.hist(np.log10(np.array(snap.dens0.to('g/cm**3'))), bins=50)
    plt.xlabel(r'log particle density (g/cm$^3$)')
    plt.savefig(output_dir+'hist_logdens_{}.png'.format(snap.name))
    plt.clf()

    # resolution plots
    _x1, _y1 = get1Dmean(r.to('pc'), snap.mass0.to('Msun'), nbins=100, xlog=True, cumsum_instead=True) # Mencl
    _x2, _y2 = get1Dmean(r.to('pc'), snap.mass0.to('Msun'), nbins=100, xlog=True) # delta m
    jv.plot(_x1, _y1, label=r'$M_\mathrm{encl}$', ls='--', color='black')
    jv.plot(_x2, _y2, label=r'$\delta m$', ls='-', color='black', xlog=True, ylog=True, xlabel='spherical radius (pc)', ylabel=r'Mass resolution ($M_\odot$)', show_legend=True, out=output_dir+'mass_resolution_{}.png'.format(snap.name))
    
    _x3, _y3 = get1Dmean(r.to('pc'), np.array((snap.mass0/snap.dens0).to('pc**3'))**(1/3.), nbins=100, xlog=True) # delta r 
    jv.plot(_x3, _y3, label=r'$\delta r$', ls='-', color='black', xlog=True, ylog=True, xlabel='spherical radius (pc)', ylabel=r'Spatial resolution $\delta x$ (pc)', out=output_dir+'spatial_resolution_{}.png'.format(snap.name))

    # Pmag = np.sum(snap['PartType0','MagneticField'].to('G')**2, axis=1)/(8*np.pi)
    # Pmag_rbins, Pmag_bins = get1Dmean(r.to('pc'), Pmag, xlog=True)
    # jv.plot(Pmag_rbins, (Pmag_bins, ), label=('$P_\mathrm{mag}$',), color=('blue',), xlog=True, ylog=True, out='Pmag_{}.png'.format(snap.name))
    
    # Pth = ((5./3.-1.)*snap.dens0*snap['PartType0','InternalEnergy']/c.c**2).to('g cm**-3')
    # Pth_rbins, Pth_bins = get1Dmean(r.to('pc'), Pth, xlog=True)
    # jv.plot(Pth_rbins, (Pth_bins, ), label=('$P_\mathrm{th}$',), color=('red',), xlog=True, ylog=True, out='Pth_{}.png'.format(snap.name))

    # Prad = ((4./3.-1.) * np.sum(snap['PartType0','PhotonEnergy'],axis=1) / (snap.mass0/snap.dens0)).to('g cm**-3')
    # Prad_rbins, Prad_bins = get1Dmean(r.to('pc'), Prad, xlog=True)
    # jv.plot(Prad_rbins, (Prad_bins, ), label=('$P_\mathrm{th}$',), color=('green',), xlog=True, ylog=True, out='Prad_{}.png'.format(snap.name))
    
    print('black holes:')
    blackholetype='PartType3' #'PartType5'
    massname='Masses' # 'BH_Mass'
    print('  masses (Msun)  : {}'.format((snap[blackholetype,massname]).to('Msun'))) # *snap.metadata['UnitMass_In_CGS']*u.g if BH_Mass ? 
    #print('  mdots (Msun/yr): {}'.format(snap[blackholetype,'BH_Mdot'].to('Msun/yr')))
    print('  radius (pc)    : {}'.format(np.sqrt(np.sum(snap[blackholetype,'Coordinates'].to('pc')**2, axis=1))))
    print('  speed (km/s)   : {}'.format(np.sqrt(np.sum(snap[blackholetype,'Velocities'].to('km/s')**2, axis=1))))

    # ...

    # temp: use pynbody for maps (TODO: port over jaba implementation)
    import pynbody
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
                os.makedirs(output_dir + 'inc{}/'.format(inclination), exist_ok=True)

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
                    plt.savefig(output_dir +'inc{}/map_dens_{}_oom{}_inc{}.png'.format(inclination, snap.name, oom, inclination))
                    plt.clf()

                    _map2 = pynbody.plot.sph.image(s.gas, qty='temp', width=r_pc, units="K", noplot=True, resolution=500, threaded=False)#, restrict_depth=True)
                    plt.title('gas')
                    plt.imshow(np.log10(_map2), extent=extent, origin='lower')
                    plt.colorbar(label=r'log mean gas temperature $T$ [$K$]')
                    plt.xlabel('x [pc]')
                    plt.ylabel('y [pc]')
                    plt.savefig(output_dir +'inc{}/map_temp_{}_oom{}_inc{}.png'.format(inclination, snap.name, oom, inclination))
                    plt.clf()

                    s.gas['Pmag'] = np.sum(snap['PartType0', 'MagneticField']**2, axis=1)/(8*np.pi)
                    s.gas['Pmag'].units = 'K' # only to escape Gauss/unitless issue for plotting purposes
                    _map3 = pynbody.plot.sph.image(s.gas, qty='Pmag', width=r_pc, units="K", noplot=True, resolution=500, threaded=False)#, restrict_depth=True)
                    plt.title('gas')
                    plt.imshow(np.log10(_map3), extent=extent, origin='lower')
                    plt.colorbar(label=r'log mean magnetic field pressure $P_\mathrm{mag}$ [dyn/cm$^2$]')
                    plt.xlabel('x [pc]')
                    plt.ylabel('y [pc]')
                    plt.savefig(output_dir +'inc{}/map_Pmag_{}_oom{}_inc{}.png'.format(inclination, snap.name, oom, inclination))
                    plt.clf()

                    s.gas['plasma beta'] = np.array(snap.dens0.to('g cm**-3')/(2*1.67262192e-24)*c.k_B.to('erg K**-1')*snap.temp0.to('K'), dtype=np.float64)/np.array(s.gas['Pmag'], dtype=np.float64)
                    s.gas['plasma beta'].units = 'K' # only to escape unitless issue for plotting purposes
                    _map4 = pynbody.plot.sph.image(s.gas, qty='plasma beta', width=r_pc, units='K', noplot=True, resolution=500, threaded=False)#, restrict_depth=True)
                    plt.title('gas')
                    plt.imshow(np.log10(_map4), extent=extent, origin='lower')
                    plt.colorbar(label=r'log mean gas $\beta_\mathrm{plasma}$')
                    plt.xlabel('x [pc]')
                    plt.ylabel('y [pc]')
                    plt.savefig(output_dir +'inc{}/map_plasmabeta_{}_oom{}_inc{}.png'.format(inclination, snap.name, oom, inclination))
                    plt.clf()

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

