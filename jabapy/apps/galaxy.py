import gzip
import re
import shutil
import tempfile
from pathlib import Path
import time

import numpy as np
import scipy as sp

from ..physics import radiation as rad
from ..utils import units as u
from ..utils import constants as c
from ..utils.resources import get_resource_path, load_resource
from ..utils.grid import CartesianGrid, bin_particles_direct


def _BC03_current_mass_fraction(model, metals, ages):
    """Read living-star plus remnant fractions from the SSP .4color tables."""

    # convert the companion mass tables onto the spectral age grid
    directory = get_resource_path('stellar_population_models', model, '')
    paths = sorted(directory.rglob('*.4color.gz')) or sorted(directory.rglob('*.4color'))
    fractions = []
    for path in paths:
        with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
            text = stream.read()
        metal = float(re.search(r'\bZ\s*=\s*([0-9.Ee+-]+)', text).group(1))
        data = np.loadtxt(text.splitlines())
        fraction = np.interp(ages, 10**data[:, 0], data[:, 10], left=1.0)
        fractions.append((metal, fraction))
    fractions.sort(key=lambda row: row[0])
    if len(fractions) != len(metals) or not np.allclose([row[0] for row in fractions], metals):
        raise ValueError('BC03 .4color mass tables must match the spectral metallicity grid.')
    return sp.interpolate.RegularGridInterpolator((metals, ages), np.stack([row[1] for row in fractions]))


def galaxy_image(snap=None, bands=('RGB.R', 'RGB.G', 'RGB.B'), component='effective',
                 percentile=99.5, stretch=0.03, ax=None, title=None, continuum=None,
                 cache=True, **continuum_kwargs):
    """Render a galaxy RGB image in one call, using the notebook's shared asinh stretch.

    Pass a snapshot and any galaxy_continuum options (npix, grid_shape, Lbox, inner_radius,
    workers, log, etc.). Defaults here use 128 linear wavelengths from 0.28 to 1.25 micron
    and omit optical-depth cubes. Explicit lam or wavelength_range overrides that range.
    If npix is supplied without grid_shape, the grid uses npix cells along each axis.
    Alternatively pass continuum=the result of galaxy_continuum or raytrace_galaxy_continuum
    to recolor a cube without another calculation; do not also supply snapshot/RT options.

    bands assigns three radiation.load_bandpass names to display red, green and blue.
    The default RGB filters are approximate visible test responses, not calibrated human
    vision or sRGB. Johnson I/V/B and SDSS i/r/g are alternative false-color assignments.
    component selects pure_emission, absorption or effective attenuation. The common
    asinh stretch preserves channel ratios before clipping to [0, 1]; percentile sets
    the positive mean-brightness reference and stretch sets its linear-to-log transition.
    Empty images remain black. Filter coverage must fit inside the continuum wavelengths.

    Returns rgb (y, x, 3), unstretched band-averaged channels with I_lambda units per micron,
    extent in kpc, bands, component, scale, figure, ax and the Matplotlib image artist.
    An optional ax receives the plot; the figure is kept open for notebook display/saving.
    Full spectral cubes are not retained in the result. No snapshot data are modified.
    """

    # validate display settings before preparing or tracing the galaxy
    if isinstance(bands, str) or len(bands) != 3: raise ValueError('Choose three filter names in display red, green, blue order.')
    if component not in ('pure_emission', 'absorption', 'effective'): raise ValueError('Unknown continuum component.')
    if not np.isfinite(percentile) or not 0 < percentile <= 100: raise ValueError('percentile must lie in (0, 100].')
    if not np.isfinite(stretch) or stretch <= 0: raise ValueError('stretch must be finite and positive.')
    if continuum is not None and (snap is not None or continuum_kwargs): raise ValueError('Use either a snapshot with RT options or an existing continuum.')
    if continuum is None and snap is None: raise ValueError('Provide a snapshot or an existing continuum.')
    profiles = [rad.load_bandpass(name, cache=cache) for name in bands]

    # reuse the complete continuum workflow with optical sampling for a visual image
    if continuum is None:
        if continuum_kwargs.get('lam') is None and continuum_kwargs.get('wavelength_range') is None:
            continuum_kwargs['wavelength_range'] = [0.28, 1.25]*u.micron
        if 'npix' in continuum_kwargs: continuum_kwargs.setdefault('grid_shape', continuum_kwargs['npix'])
        continuum_kwargs.setdefault('n_wavelength', 128)
        continuum_kwargs.setdefault('wavelength_spacing', 'linear')
        continuum_kwargs.setdefault('return_optical_depth', False)
        continuum = galaxy_continuum(snap, cache=cache, **continuum_kwargs)

    # integrate the chosen filters and apply the notebook's common brightness stretch
    unit = u.erg/u.s/u.cm**2/u.sr/u.micron
    channels = np.stack([rad.integrate_bandpass(continuum['wavelength'], continuum['datacubes'][component],
                        profile, normalize=True).to_value(unit) for profile in profiles], axis=-1)
    if channels.ndim != 3 or np.any(~np.isfinite(channels)): raise ValueError('The continuum must produce finite 2D band images.')
    brightness = np.mean(channels, axis=-1)
    positive = brightness > 0
    scale = np.percentile(brightness[positive], percentile) if np.any(positive) else 0.
    rgb = np.zeros_like(channels)
    if scale > 0:
        stretched = np.arcsinh(brightness/(stretch*scale))/np.arcsinh(1/stretch)
        np.divide(channels*stretched[..., None], brightness[..., None], out=rgb, where=positive[..., None])
        np.clip(rgb, 0, 1, out=rgb)

    # plot through the shared visual helper and keep the figure available to the caller
    from ..utils import visual as jv
    extent = continuum['extent'].to(u.kpc)
    if title is None: title = 'Approximate visible RGB' if tuple(bands) == ('RGB.R', 'RGB.G', 'RGB.B') else ' / '.join(bands)+' (false color)'
    image = jv.imshow(rgb, ax=ax, origin='lower', extent=extent.value, aspect='equal',
                      xlabel='x [kpc]', ylabel='y [kpc]', title=title, show_colorbar=False,
                      show=False, clf_after=False, close_after=False)
    return dict(rgb=rgb, channels=channels*unit, extent=extent, bands=tuple(bands), component=component,
                scale=scale*unit, figure=image.figure, ax=image.axes, image=image)


def galaxy_continuum(snap, lam=None, Lbox=10*u.kpc, depth=None, center=None, npix=64, grid_shape=64,
                     model='BC03_miles_chabrier', star_type=4, dust_to_metal=0.5,
                     dust_temperature=None, dust_sublimation_temperature=1500*u.K,
                     stellar_mass_is_initial=False, clip_population=True,
                     chunk_size=250000, frequency_chunk_size=4, step_fraction=0.25, N_NGB=8,
                     workers=-1, query_chunk_size=4096, log=False, cache=True,
                     wavelength_range=None, n_wavelength=24, wavelength_spacing='log',
                     storage='auto', storage_dir=None, max_grid_bytes=4*1024**3, inner_radius=0*u.kpc, backend='auto', return_optical_depth=True):
    """Trace intrinsic stellar populations and thermal MRN dust through a GIZMO galaxy toward +z.

    Lbox is the full physical image width; depth defaults to Lbox. center is relative to
    the snapshot's current transformed frame (default zero). Only particles inside this
    box contribute. inner_radius excludes gas and stars at radii smaller than this value
    from center (bare numbers mean kpc; default zero). This cuts particles before gridding;
    interpolation can smooth the edge of the excluded region.
    Data come from the snapshot accessors in its current units and frame.
    Stellar luminosity, dust mass and thermal emission are deposited conservatively into
    Cartesian cells, then raytrace_particles interpolates those cell-centered coefficients.
    grid_shape controls this spatial approximation; increasing it tests convergence.
    Wavelengths default to 24 logarithmic samples from 0.1 to 1000 micron, in the rest frame.
    Set wavelength_range=(lower, upper), n_wavelength and wavelength_spacing='log'/'linear'
    to generate a grid; bare limits mean micron. Alternatively provide explicit lam samples.
    lam and wavelength_range are mutually exclusive.
    The spectrum contains stars plus dust, with no AGN, nebular emission or scattering source.
    DustToGasRatio_Local is used if available, otherwise dust_to_metal times gas metallicity.
    Dust_Temperature supplies the emission temperature unless a scalar override is given;
    dust above dust_sublimation_temperature is excluded. No radiative-equilibrium iteration
    is performed, so the saved temperatures need not balance the absorbed stellar power.
    GIZMO stellar masses are treated as current living-star plus remnant masses and converted
    using BC03 .4color fractions; stellar_mass_is_initial=True disables that conversion.
    Ages and metallicities outside the library are clipped if clip_population=True and
    their counts are returned. Nonpositive formation times are excluded as wind particles.
    Three cases are traced: pure_emission, absorption and effective (sqrt(abs*(abs+sca))).
    Returns wavelength, extent, pixel_area, particle counts and datacubes[name] containing
    I_lambda with shape (y, x, wavelength) for each opacity case. optical_depth[name] has
    the same shape. No wavelength integration is applied; use radiation.integrate_bandpass
    to form images or integrate the cube over pixels to obtain a directional luminosity.
    Snapshot arrays are not modified; particle and spectral temporaries are chunked.
    The snapshot may cache full input fields in RAM; chunk_size does not bound that cache.
    storage, storage_dir and max_grid_bytes select coefficient storage as described in
    prepare_galaxy_continuum; frequency_chunk_size also bounds preparation temporaries.
    For repeated renders, call prepare_galaxy_continuum once and reuse its result with
    raytrace_galaxy_continuum to change npix or integration settings without rereading data.
    backend='auto' uses parallel compiled transfer; 'numpy' selects the reference tracer.
    return_optical_depth=False omits optical-depth cubes to reduce output memory.
    """

    if int(npix) != npix or npix < 1: raise ValueError('npix must be a positive integer.')
    prepared = prepare_galaxy_continuum(snap, lam=lam, Lbox=Lbox, depth=depth, center=center, grid_shape=grid_shape,
        model=model, star_type=star_type, dust_to_metal=dust_to_metal, dust_temperature=dust_temperature,
        dust_sublimation_temperature=dust_sublimation_temperature, stellar_mass_is_initial=stellar_mass_is_initial,
        clip_population=clip_population, chunk_size=chunk_size, log=log, cache=cache,
        wavelength_range=wavelength_range, n_wavelength=n_wavelength, wavelength_spacing=wavelength_spacing,
        frequency_chunk_size=frequency_chunk_size, storage=storage, storage_dir=storage_dir, max_grid_bytes=max_grid_bytes,
        inner_radius=inner_radius)
    return raytrace_galaxy_continuum(prepared, npix=npix, frequency_chunk_size=frequency_chunk_size,
        step_fraction=step_fraction, N_NGB=N_NGB, workers=workers, query_chunk_size=query_chunk_size, log=log, backend=backend, return_optical_depth=return_optical_depth)


def prepare_galaxy_continuum(snap, lam=None, Lbox=10*u.kpc, depth=None, center=None, grid_shape=64,
                            model='BC03_miles_chabrier', star_type=4, dust_to_metal=0.5,
                            dust_temperature=None, dust_sublimation_temperature=1500*u.K,
                            stellar_mass_is_initial=False, clip_population=True, chunk_size=250000,
                            log=False, cache=True, wavelength_range=None, n_wavelength=24, wavelength_spacing='log',
                            frequency_chunk_size=4, storage='auto', storage_dir=None, max_grid_bytes=4*1024**3, inner_radius=0*u.kpc):
    """Read a GIZMO snapshot and prepare reusable stellar and dust coefficients.

    Uses the physics, units and spatial deposition described in galaxy_continuum.
    Reads snap.pos, snap.mass, snap.metals and snapshot-specific fields through snap,
    honoring its cached values and transformed frame. The snapshot caches full input
    fields; chunk_size bounds processing temporaries, not the snapshot cache.
    Returns a dictionary for raytrace_galaxy_continuum, retaining the emissivity grid
    and dust density in the selected storage. No global grid cache is kept; release the dictionary
    when finished. cache controls the BC03 resource cache only. Reprepare after changing
    the snapshot/frame, box, inner_radius, grid_shape, wavelengths, stellar model or dust parameters.
    inner_radius excludes gas and stars with distance < inner_radius from center before
    occupancy counting and deposition. Bare radii mean kpc; zero disables the cutoff.
    Particles on the boundary are retained; interpolation can smooth the cutoff edge.
    lam or wavelength_range/n_wavelength/wavelength_spacing select the rest-frame spectrum.
    Particle chunks accumulate in place through bin_particles_direct; no full-size
    temporary deposition grids are allocated. frequency_chunk_size bounds particle spectra.
    storage='auto' uses dense coefficients if they fit max_grid_bytes (default 4 GiB),
    otherwise 'compact' stores only occupied cells after an extra coordinate pass. Empty
    cells still participate in interpolation with exactly zero coefficients. 'dense' and
    'compact' select either layout explicitly. Compact storage adds a full-grid integer
    field_indices array mapping Cartesian cells to coefficient rows; row zero is empty.
    max_grid_bytes limits coefficient and index arrays, not total process RAM or outputs;
    increase it deliberately or use None to disable the check. No precision is reduced.
    storage_dir optionally puts coefficients in new .npy memory maps in a unique subdirectory
    (returned as storage_path), never overwriting files. Index maps remain in RAM. Allow
    sufficient disk space; mapped pages use reclaimable RAM and disk access can be slow.
    Files persist after the dictionary is released; remove them manually when no longer needed.
    """

    # validate the physical image and wavelength grids
    if lam is not None and wavelength_range is not None:
        raise ValueError('Supply explicit lam samples or wavelength_range, not both.')
    if lam is None:
        limits = u.Quantity([0.1, 1000] if wavelength_range is None else wavelength_range, u.micron).to_value(u.micron)
        if limits.shape != (2,) or np.any(~np.isfinite(limits)) or not 0 < limits[0] < limits[1]:
            raise ValueError('wavelength_range must contain two finite, positive, increasing limits.')
        if int(n_wavelength) != n_wavelength or n_wavelength < 2 or wavelength_spacing not in ('log', 'linear'):
            raise ValueError("Use n_wavelength >= 2 and wavelength_spacing='log' or 'linear'.")
        sample = np.geomspace if wavelength_spacing == 'log' else np.linspace
        lam = sample(*limits, int(n_wavelength))*u.micron
    else:
        lam = u.Quantity(lam, u.micron)
    wave = np.asarray(lam.to_value(u.AA))
    if wave.ndim != 1 or wave.size < 2 or np.any(~np.isfinite(wave)) or np.any(np.diff(wave) <= 0) or wave[0] <= 0:
        raise ValueError('lam must be an increasing one-dimensional wavelength grid with at least two samples.')
    Lbox = u.Quantity(Lbox, u.kpc).to_value(u.cm)
    depth = Lbox if depth is None else u.Quantity(depth, u.kpc).to_value(u.cm)
    center = np.zeros(3) if center is None else u.Quantity(center, u.kpc).to_value(u.cm)
    inner_radius = u.Quantity(inner_radius, u.kpc).to_value(u.cm)
    if np.ndim(inner_radius) != 0 or not np.isfinite(inner_radius) or inner_radius < 0:
        raise ValueError('inner_radius must be a finite, nonnegative scalar length.')
    dims = np.broadcast_to(np.asarray(grid_shape, dtype=int), (3,))
    if Lbox <= 0 or depth <= 0 or np.any(dims < 2):
        raise ValueError('Use positive image dimensions and at least two grid cells per axis.')
    if chunk_size < 1 or dust_to_metal < 0:
        raise ValueError('Chunk sizes must be positive and dust_to_metal nonnegative.')
    widths = np.array([Lbox, Lbox, depth]); lower = center-widths/2; upper = center+widths/2
    cell_width = widths/dims; volume = np.prod(cell_width); n_cells = int(np.prod(dims))
    frequency_chunk_size = wave.size if frequency_chunk_size is None else frequency_chunk_size
    if int(frequency_chunk_size) != frequency_chunk_size or frequency_chunk_size < 1:
        raise ValueError('frequency_chunk_size must be a positive integer or None.')
    frequency_chunk_size = int(frequency_chunk_size)
    if storage not in ('auto', 'dense', 'compact') or (max_grid_bytes is not None and max_grid_bytes <= 0):
        raise ValueError("Use storage='auto', 'dense' or 'compact' and a positive max_grid_bytes or None.")
    dense_bytes = n_cells*(wave.size+1)*8
    if storage == 'auto': storage = 'compact' if max_grid_bytes is not None and dense_bytes > max_grid_bytes else 'dense'
    if log: print(f'Galaxy dense grid: {dense_bytes/1024**3:.2f} GiB; using {storage} storage.', flush=True)
    field_indices = None; n_fields = n_cells; index_bytes = 0
    counts = dict(gas=0, stars=0, population_clipped=0, wind_excluded=0)

    # count occupied cells before allocating their spectra
    if storage == 'compact':
        index_dtype = np.int32 if n_cells < np.iinfo(np.int32).max else np.int64
        index_bytes = n_cells*np.dtype(index_dtype).itemsize
        if max_grid_bytes is not None and index_bytes+n_cells > max_grid_bytes:
            raise MemoryError('Compact occupancy and index arrays exceed max_grid_bytes; reduce grid_shape or increase the budget.')
        occupied = np.zeros(n_cells, dtype=bool)
        last_report = time.perf_counter()
        for part in (0, star_type):
            positions = snap.pos[part]
            for start in range(0, positions.shape[0], chunk_size):
                pos = positions[start:start+chunk_size].to_value(u.cm)
                inside = np.all((pos >= lower) & (pos < upper), axis=1)
                if inner_radius > 0: inside &= np.linalg.norm(pos-center, axis=1) >= inner_radius
                bins = ((pos[inside]-lower)*(dims/(upper-lower))).astype(np.int64)
                bins = bins[np.all(bins < dims, axis=1)]
                occupied[np.ravel_multi_index(bins.T, tuple(dims))] = True
                if log and time.perf_counter()-last_report >= 5:
                    fraction = min(start+chunk_size, positions.shape[0])/positions.shape[0]
                    print(f'Galaxy occupancy: type {part}, {fraction:.1%} read.', flush=True)
                    last_report = time.perf_counter()
        n_fields = int(np.count_nonzero(occupied))+1
    coefficient_bytes = n_fields*(wave.size+1)*8
    required_bytes = index_bytes + (coefficient_bytes if storage_dir is None else 0)
    if max_grid_bytes is not None and required_bytes > max_grid_bytes:
        raise MemoryError(f'Galaxy {storage} grids require {required_bytes/1024**3:.2f} GiB, exceeding '
            f'max_grid_bytes={max_grid_bytes/1024**3:.2f} GiB. Reduce grid_shape or wavelengths, '
            'increase the budget if RAM permits, or supply storage_dir for disk-backed coefficients.')
    if storage == 'compact':
        field_indices = np.zeros(n_cells, dtype=index_dtype)
        offset = 1
        for start in range(0, n_cells, 1000000):
            selected = np.flatnonzero(occupied[start:start+1000000])+start
            field_indices[selected] = np.arange(offset, offset+selected.size, dtype=index_dtype)
            offset += selected.size
        del occupied

    # allocate coefficients in ram or new disk-backed files
    storage_path = None
    if storage_dir is None:
        dust_density = np.zeros(n_fields); emission = np.zeros((n_fields, wave.size))
    else:
        directory = Path(storage_dir).expanduser(); directory.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(directory).free < coefficient_bytes+1024**2:
            raise OSError(f'Insufficient disk space for {coefficient_bytes/1024**3:.2f} GiB of galaxy coefficients.')
        storage_path = Path(tempfile.mkdtemp(prefix='galaxy-continuum-', dir=directory))
        dust_density = np.lib.format.open_memmap(storage_path/'dust_density.npy', mode='w+', dtype=np.float64, shape=(n_fields,))
        emission = np.lib.format.open_memmap(storage_path/'emission.npy', mode='w+', dtype=np.float64, shape=(n_fields, wave.size))
    bin_dims = (n_fields,) if field_indices is not None else tuple(dims)
    dust_grid = dust_density.reshape(bin_dims); emission_grid = emission.T.reshape((wave.size,)+bin_dims)
    bin_kwargs = dict(mins=[0] if field_indices is not None else lower,
                      maxs=[n_fields] if field_indices is not None else upper, dims=bin_dims, nthreads=1)
    if log: print(f'Galaxy coefficients: {n_fields:,} rows, {coefficient_bytes/1024**3:.2f} GiB; '
                  f'index map: {index_bytes/1024**3:.2f} GiB.', flush=True)

    # prepare compact stellar spectra and the current-to-initial mass conversion
    metals, ages, native_lam, spectra = load_resource('stellar_population_models', model, cache=cache)
    if wave[0] < native_lam[0] or wave[-1] > native_lam[-1]:
        raise ValueError('Wavelengths exceed the BC03 table range.')
    sampled = sp.interpolate.interp1d(native_lam, spectra, axis=-1, assume_sorted=True)(wave)
    spectral_chunks = [(start, min(start+frequency_chunk_size, wave.size)) for start in range(0, wave.size, frequency_chunk_size)]
    stellar_spectra = [sp.interpolate.RegularGridInterpolator((metals, ages), sampled[..., start:stop])
                       for start, stop in spectral_chunks]
    mass_fraction = None if stellar_mass_is_initial else _BC03_current_mass_fraction(model, metals, ages)
    nu = c.c.cgs.value/(wave*u.AA).to_value(u.cm)
    k_abs, k_sca = rad.kappa_dust_MRN_cgs(nu, dust_to_gas=1.0)
    wavelength_jacobian = c.c.cgs.value/(wave*u.AA).to_value(u.cm)**2 * u.AA.to(u.cm)

    # use physical formation times supplied by the snapshot
    if snap.Cosmological:
        now = snap.scale_factor_to_time(snap.ScaleFactor).to_value(u.yr)
    else:
        units = snap.UnitSystem_In_CGS
        now = (float(snap.metadata['Time'])*units['length']/units['velocity']).to_value(u.yr)
    last_report = time.perf_counter()
    for part in (0, star_type):
        positions = snap.pos[part]; total = positions.shape[0]
        for start in range(0, total, chunk_size):
            stop = min(start+chunk_size, total)
            pos = positions[start:stop].to_value(u.cm)
            inside = np.all((pos >= lower) & (pos < upper), axis=1)
            if inner_radius > 0: inside &= np.linalg.norm(pos-center, axis=1) >= inner_radius
            if not np.any(inside): continue
            selected = pos[inside]
            masses = snap.mass[part][start:stop][inside].to_value(u.g)
            metallicity = snap.metals[part][start:stop][inside].to_value(u.dimensionless_unscaled)
            if metallicity.ndim == 2: metallicity = metallicity[:, 0]
            if np.any(~np.isfinite(masses)) or np.any(masses < 0) or np.any(~np.isfinite(metallicity)):
                raise ValueError('Selected particle masses and metallicities must be finite, with nonnegative mass.')

            # deposit gas dust mass and temperature-dependent thermal luminosity per wavelength
            if part == 0:
                counts['gas'] += selected.shape[0]
                ratio = snap[part, 'DustToGasRatio_Local'][start:stop][inside].to_value(u.dimensionless_unscaled) if (part, 'DustToGasRatio_Local') in snap else dust_to_metal*np.maximum(metallicity, 0)
                temp = snap[part, 'Dust_Temperature'][start:stop][inside].to_value(u.K) if dust_temperature is None else np.full(selected.shape[0], u.Quantity(dust_temperature, u.K).to_value(u.K))
                if np.any(~np.isfinite(temp)) or np.any(temp < 0) or np.any(~np.isfinite(ratio)) or np.any(ratio < 0):
                    raise ValueError('Dust temperatures and dust-to-gas ratios must be finite and nonnegative.')
                mass = masses*ratio*(temp <= u.Quantity(dust_sublimation_temperature, u.K).to_value(u.K))

            # deposit stellar luminosities after converting formation times and retained masses
            else:
                formed = snap[part, 'StellarFormationTime'][start:stop][inside].to_value(u.yr)
                valid = np.isfinite(formed) & (formed > 0)
                counts['wind_excluded'] += int(np.sum(~valid))
                selected, formed, masses, metallicity = selected[valid], formed[valid], masses[valid], metallicity[valid]
                age = now-formed
                outside = (age < ages[0]) | (age > ages[-1]) | (metallicity < metals[0]) | (metallicity > metals[-1])
                if not clip_population and np.any(outside): raise ValueError('Stellar populations exceed the BC03 age or metallicity bounds.')
                counts['population_clipped'] += int(np.sum(outside)); counts['stars'] += selected.shape[0]
                points = np.column_stack((np.clip(metallicity, metals[0], metals[-1]), np.clip(age, ages[0], ages[-1])))
                mass = masses/u.Msun.to(u.g)
                if mass_fraction is not None: mass /= mass_fraction(points)

            # map occupied cells and deposit bounded spectral chunks directly in density units
            bin_pos = selected
            if field_indices is not None:
                bins = ((selected-lower)*(dims/(upper-lower))).astype(np.int64)
                valid_bins = np.all(bins < dims, axis=1)
                bin_pos = np.full((selected.shape[0], 1), n_fields, dtype=float)
                bin_pos[valid_bins, 0] = field_indices[np.ravel_multi_index(bins[valid_bins].T, tuple(dims))]+0.5
            if part == 0: bin_particles_direct(bin_pos, mass/volume, out=dust_grid, **bin_kwargs)
            for (first, last), stellar_spectrum in zip(spectral_chunks, stellar_spectra):
                if part == 0:
                    luminosity = np.empty((selected.shape[0], last-first))
                    for k in range(first, last):
                        x = np.full(temp.shape, np.inf)
                        np.divide(c.h.cgs.value*nu[k], c.k_B.cgs.value*temp, out=x, where=temp > 0)
                        Bnu = 2*c.h.cgs.value*nu[k]**3/c.c.cgs.value**2*np.exp(-x)/(-np.expm1(-x))
                        luminosity[:, k-first] = mass*k_abs[k]*Bnu*wavelength_jacobian[k]/volume
                else:
                    luminosity = stellar_spectrum(points)
                    luminosity *= (mass*u.Lsun.to(u.erg/u.s)/(4*np.pi*volume))[:, None]
                out = emission_grid[first] if last-first == 1 else emission_grid[first:last]
                bin_particles_direct(bin_pos, luminosity, out=out, **bin_kwargs)
                del luminosity
            if storage_path is not None:
                emission.flush(); dust_density.flush()
            if log and time.perf_counter()-last_report >= 5:
                print(f'Galaxy deposition: type {part}, {100*stop/total:.1f}% read; selected {counts["gas"]:,} gas and {counts["stars"]:,} stars.', flush=True)
                last_report = time.perf_counter()
    if counts['stars'] == 0: raise ValueError('No stars selected; center the snapshot or set center explicitly.')

    return dict(wavelength=lam, emission=emission, dust_density=dust_density, field_indices=field_indices,
                storage=storage, storage_path=storage_path, coefficient_bytes=coefficient_bytes, k_abs=k_abs, k_sca=k_sca,
                lower=lower, upper=upper, cell_width=cell_width, Lbox=Lbox, center=center,
                grid_shape=tuple(dims), particle_counts=counts)


def raytrace_galaxy_continuum(prepared, npix=64, frequency_chunk_size=4, step_fraction=0.25,
                             N_NGB=8, workers=-1, query_chunk_size=4096, log=False, backend='auto', return_optical_depth=True):
    """Render a prepared galaxy without reading its snapshot or rebuilding its spectra.

    prepared is the dictionary returned by prepare_galaxy_continuum. npix and integration
    settings may change between calls; the dictionary is not modified. All wavelengths
    and three opacity cases share spatial queries and steps. Returns the same datacubes
    and metadata as galaxy_continuum. backend='auto' uses fused Numba
    transfer, with workers controlling parallel rays (-1 uses the current Numba thread
    count). query_chunk_size bounds complete rays per batch. Empty compact-grid samples
    skip spectral work. The first call includes compilation. backend='numpy' selects
    the original reference tracer; frequency_chunk_size only affects that backend.
    return_optical_depth=False omits optical_depth from the result and, with the compiled
    backend, avoids allocating those cubes. Wavelength-unit conversion is done in place.
    """
    if int(npix) != npix or npix < 1: raise ValueError('npix must be a positive integer.')
    npix = int(npix)
    lam, dims, Lbox = prepared['wavelength'], prepared['grid_shape'], prepared['Lbox']
    lower, upper, center, cell_width = (prepared[key] for key in ('lower', 'upper', 'center', 'cell_width'))
    counts = prepared['particle_counts'].copy()

    # trace all opacity cases on a common grid with bounded spectral allocations
    axes = [lower[i]+(np.arange(dims[i])+0.5)*cell_width[i] for i in range(3)]
    grid = CartesianGrid(axes)
    edges = np.linspace(-Lbox/2, Lbox/2, npix+1)
    centers = (edges[:-1]+edges[1:])/2
    xx, yy = np.meshgrid(centers+center[0], centers+center[1], indexing='xy')
    x0 = np.stack((xx, yy, np.full_like(xx, lower[2])), axis=-1)
    x1 = x0.copy(); x1[..., 2] = upper[2]
    k_abs, k_sca = prepared['k_abs'], prepared['k_sca']
    factors = np.stack((np.zeros_like(k_abs), k_abs, np.sqrt(k_abs*(k_abs+k_sca))), axis=-1)
    traced = rad.raytrace_particles(x0, x1, prepared['emission'], prepared['dust_density'],
        tree=grid, opacity_factors=factors, frequency_chunk_size=frequency_chunk_size, N_NGB=N_NGB,
        step_fraction=step_fraction, query_chunk_size=query_chunk_size, workers=workers, ret_tau=return_optical_depth,
        log=log, reverse=True, field_indices=prepared.get('field_indices'), backend=backend)
    intensity = traced[0] if return_optical_depth else traced
    output_unit = u.erg/u.s/u.cm**2/u.sr/lam.unit
    intensity *= (u.erg/u.s/u.cm**2/u.sr/u.AA).to(output_unit)
    intensity = u.Quantity(intensity, output_unit, copy=False)
    pixel_area = (Lbox/npix)**2
    names = ('pure_emission', 'absorption', 'effective')
    result = dict(wavelength=lam, extent=np.array([-Lbox/2,Lbox/2,-Lbox/2,Lbox/2])*u.cm,
                pixel_area=pixel_area*u.cm**2, particle_counts=counts, grid_shape=tuple(dims),
                datacubes={name: intensity[..., i] for i, name in enumerate(names)})
    if return_optical_depth: result['optical_depth'] = {name: traced[1][..., i] for i, name in enumerate(names)}
    return result
