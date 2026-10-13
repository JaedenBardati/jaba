import shutil
import tempfile
from pathlib import Path
import time

import numpy as np
import scipy as sp

from ..physics import radiation as rad
from ..utils import units as u
from ..utils import constants as c
from ..utils import resources
from ..utils.grid import CartesianGrid, bin_particles_direct, bin_particles_kernel


def galaxy_image(snap=None, bands=('RGB.R', 'RGB.G', 'RGB.B'), component='effective',
                 percentile=99.5, stretch=0.03, ax=None, title=None, datacube=None,
                 cache=True, plot=True, **datacube_kwargs):
    """Render a galaxy RGB image in one call, using the notebook's shared asinh stretch.

    Pass a snapshot and any galaxy_datacube options (npix, grid_shape, Lbox, inner_radius,
    workers, log, etc.). By default each filter supplies 21 evenly spaced wavelengths
    across its full table plus its transmission peak; their sorted union is traced.
    RGB.R/G/B give 24 samples over 380--780 nm. This is a continuum sampling heuristic;
    use explicit lam for spectral features or convergence checks. Explicit wavelength_range
    overrides filter sampling; n_wavelength or wavelength_spacing alone requests a regular
    grid spanning all selected filters. Optical-depth cubes are omitted by default.
    If npix is supplied without grid_shape, the grid uses npix cells along each axis.
    supersampling=2 averages a 2 by 2 ray pattern per pixel before the RGB stretch;
    use supersampling=1 for the original single-ray sampling. Output size remains npix.
    inner_radius defaults to 0.05*Lbox/npix (5% of a pixel width); pass zero to disable it.
    Alternatively pass datacube=the result of galaxy_datacube or raytrace_galaxy_datacube
    to recolor a cube without another calculation; do not also supply snapshot/RT options.

    bands assigns three radiation.load_bandpass names to display red, green and blue.
    The default RGB filters are approximate visible test responses, not calibrated human
    vision or sRGB. Johnson I/V/B and SDSS i/r/g are alternative false-color assignments.
    component selects pure_emission, absorption or effective attenuation. The common
    asinh stretch preserves channel ratios before clipping to [0, 1]; percentile sets
    the positive mean-brightness reference and stretch sets its linear-to-log transition.
    Empty images remain black. Filter coverage must fit inside the datacube wavelengths.

    Returns rgb (y, x, 3), unstretched band-averaged channels with I_lambda units per micron,
    extent in kpc, bands, component, scale, figure, ax and the Matplotlib image artist.
    plot=True displays the figure, optionally on ax; the returned figure can be saved.
    Without ax, plotting reuses the current axes or creates them if none exist.
    plot=False returns image data without plotting; figure, ax and image are None.
    A filename or Path saves the plot without displaying it and closes the saved figure
    (including a supplied ax's figure); the returned figure object remains usable.
    Full spectral cubes are not retained in the result. No snapshot data are modified.
    """

    # validate display settings before preparing or tracing the galaxy
    if not isinstance(plot, (bool, str, Path)): raise TypeError('plot must be True, False, or a filename/Path.')
    if isinstance(bands, str) or len(bands) != 3: raise ValueError('Choose three filter names in display red, green, blue order.')
    if component not in ('pure_emission', 'absorption', 'effective'): raise ValueError('Unknown datacube component.')
    if not np.isfinite(percentile) or not 0 < percentile <= 100: raise ValueError('percentile must lie in (0, 100].')
    if not np.isfinite(stretch) or stretch <= 0: raise ValueError('stretch must be finite and positive.')
    if datacube is not None and (snap is not None or datacube_kwargs): raise ValueError('Use either a snapshot with RT options or an existing datacube.')
    if datacube is None and snap is None: raise ValueError('Provide a snapshot or an existing datacube.')
    profiles = [rad.load_bandpass(name, cache=cache) for name in bands]

    # reuse the complete datacube workflow with optical sampling for a visual image
    if datacube is None:
        if datacube_kwargs.get('lam') is None and datacube_kwargs.get('wavelength_range') is None:
            waves = [profile['wavelength'].to_value(u.nm) for profile in profiles]
            if 'n_wavelength' in datacube_kwargs or 'wavelength_spacing' in datacube_kwargs:
                datacube_kwargs['wavelength_range'] = [min(wave[0] for wave in waves), max(wave[-1] for wave in waves)]*u.nm
            else:
                samples = [np.r_[np.linspace(wave[0], wave[-1], 21), wave[np.argmax(profile['transmission'])]]
                           for wave, profile in zip(waves, profiles)]
                datacube_kwargs['lam'] = np.unique(np.concatenate(samples))*u.nm
        if 'npix' in datacube_kwargs: datacube_kwargs.setdefault('grid_shape', datacube_kwargs['npix'])
        if 'inner_radius' not in datacube_kwargs:
            datacube_kwargs['inner_radius'] = 0.05*u.Quantity(datacube_kwargs.get('Lbox', 10), u.kpc)/datacube_kwargs.get('npix', 64)
        datacube_kwargs.setdefault('n_wavelength', 128)
        datacube_kwargs.setdefault('wavelength_spacing', 'linear')
        datacube_kwargs.setdefault('return_optical_depth', False)
        datacube = galaxy_datacube(snap, cache=cache, **datacube_kwargs)

    # integrate the chosen filters and apply the notebook's common brightness stretch
    unit = u.erg/u.s/u.cm**2/u.sr/u.micron
    channels = np.stack([rad.integrate_bandpass(datacube['wavelength'], datacube['datacubes'][component],
                        profile, normalize=True).to_value(unit) for profile in profiles], axis=-1)
    if channels.ndim != 3 or np.any(~np.isfinite(channels)): raise ValueError('The datacube must produce finite 2D band images.')
    brightness = np.mean(channels, axis=-1)
    positive = brightness > 0
    scale = np.percentile(brightness[positive], percentile) if np.any(positive) else 0.
    rgb = np.zeros_like(channels)
    if scale > 0:
        stretched = np.arcsinh(brightness/(stretch*scale))/np.arcsinh(1/stretch)
        np.divide(channels*stretched[..., None], brightness[..., None], out=rgb, where=positive[..., None])
        np.clip(rgb, 0, 1, out=rgb)

    # return physical band images and display data without creating a figure when disabled
    extent = datacube['extent'].to(u.kpc)
    result = dict(rgb=rgb, channels=channels*unit, extent=extent, bands=tuple(bands), component=component,
                  scale=scale*unit, figure=None, ax=None, image=None)
    if plot is False: return result

    # reuse the visual helper's saving options and suppress interactive display during export
    from ..utils import visual as jv
    output = Path(plot) if isinstance(plot, (str, Path)) else None
    if title is None: title = 'Approximate visible RGB' if tuple(bands) == ('RGB.R', 'RGB.G', 'RGB.B') else ' / '.join(bands)+' (false color)'
    image = jv.imshow(rgb, ax=ax, origin='lower', extent=extent.value, aspect='equal',
                      xlabel='x [kpc]', ylabel='y [kpc]', title=title, show_colorbar=False,
                      out=output, show=output is None,
                      clf_after=False, close_after=output is not None)
    result.update(figure=image.figure, ax=image.axes, image=image)
    return result


def _galaxy_wavelengths(lam, wavelength_range, n_wavelength, wavelength_spacing):
    """Validate a shared rest-frame wavelength grid for preparation and batching."""
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
    return lam


def galaxy_datacube(snap, lam=None, Lbox=10*u.kpc, depth=None, center=None, npix=64, grid_shape=64,
                     model='BC03_miles_chabrier', star_type=4, dust_to_metal=0.5,
                     dust_temperature=None, dust_sublimation_temperature=1500*u.K,
                     stellar_mass_is_initial=False, clip_population=True,
                     chunk_size=250000, frequency_chunk_size=4, step_fraction=0.25, N_NGB=8,
                     workers=-1, query_chunk_size=4096, log=False, cache=True,
                     wavelength_range=None, n_wavelength=24, wavelength_spacing='log',
                     storage='auto', storage_dir=None, max_grid_bytes=4*1024**3, inner_radius=0*u.kpc, backend='auto', return_optical_depth=True, supersampling=2, dust_deposition='kernel', wavelength_batch_size='auto'):
    """Trace intrinsic stellar populations and thermal MRN dust through a GIZMO galaxy toward +z.

    Lbox is the full physical image width; depth defaults to Lbox. center is relative to
    the snapshot's current transformed frame (default zero). Only particles inside this
    box contribute. inner_radius excludes gas and stars at radii smaller than this value
    from center (bare numbers mean kpc; default zero). This cuts particles before gridding;
    interpolation can smooth the edge of the excluded region.
    Data come from the snapshot accessors in its current units and frame.
    Stellar luminosity is binned directly. By default dust_deposition='kernel' spreads
    gas dust mass and thermal emission with the compact cubic kernel using snap.smooth[0]
    as its support radius. Cell-center weights are normalized to conserve each selected
    particle's mass and luminosity. Unresolved kernels with no supported cell centers
    use the containing cell. 'direct' restores single-cell dust deposition.
    Particle centers determine box/inner-radius selection; kernels truncated by the box
    are renormalized inside it and can extend into the excluded inner region.
    raytrace_particles then interpolates the cell-centered coefficients.
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
    prepare_galaxy_datacube; frequency_chunk_size also bounds preparation temporaries.
    wavelength_batch_size='auto' limits live coefficients to max_grid_bytes by preparing,
    tracing and releasing wavelength batches when a dense full spectrum would exceed it.
    This preserves the spectrum and spatial deposition, but repeats particle processing
    and ray queries. Final image cubes, snapshot caches and work arrays are additional RAM.
    An integer >= 2 sets the batch size; None disables batching. Explicit storage_dir
    keeps automatic batching disabled so disk-backed full spectra remain available.
    If even two dense channels do not fit, preparation tries the normal compact layout
    and enforces its budget. Explicit prepare_galaxy_datacube still retains all channels.
    For repeated renders, call prepare_galaxy_datacube once and reuse its result with
    raytrace_galaxy_datacube to change npix or integration settings without rereading data.
    backend='auto' uses parallel compiled transfer; 'numpy' selects the reference tracer.
    return_optical_depth=False omits optical-depth cubes to reduce output memory.
    supersampling is a positive integer (default 2), tracing its square in uniformly
    spaced subpixel rays and averaging linear intensities. npix and pixel_area describe
    the final pixels. Returned optical depths are mean ray depths for diagnostics,
    not effective depths computed from mean transmission. Use 1 for single-ray sampling.
    """

    if int(npix) != npix or npix < 1: raise ValueError('npix must be a positive integer.')
    if isinstance(supersampling, bool) or not isinstance(supersampling, (int, np.integer)) or supersampling < 1:
        raise ValueError('supersampling must be a positive integer.')

    # bound live coefficient grids by tracing independent wavelength batches
    lam = _galaxy_wavelengths(lam, wavelength_range, n_wavelength, wavelength_spacing)
    if wavelength_batch_size != 'auto' and wavelength_batch_size is not None:
        if isinstance(wavelength_batch_size, bool) or not isinstance(wavelength_batch_size, (int, np.integer)) or wavelength_batch_size < 2:
            raise ValueError('wavelength_batch_size must be auto, None or an integer >= 2.')
    batch_size = lam.size
    if wavelength_batch_size == 'auto' and storage_dir is None and max_grid_bytes is not None and max_grid_bytes > 0:
        cells = int(np.prod(np.broadcast_to(np.asarray(grid_shape, dtype=int), (3,))))
        if cells > 0:
            index_bytes = cells*(4 if cells < np.iinfo(np.int32).max else 8) if storage == 'compact' else 0
            rows = cells+1 if storage == 'compact' else cells
            capacity = int((max_grid_bytes-index_bytes)//(8*rows))-1
            if capacity >= 2: batch_size = min(batch_size, capacity)
    elif wavelength_batch_size is not None and wavelength_batch_size != 'auto':
        batch_size = min(batch_size, wavelength_batch_size)
    result = None
    for first in range(0, lam.size, batch_size):
        last = min(first+batch_size, lam.size)
        # overlap the final singleton to retain the preparation API's two-sample minimum
        begin = first if last-first > 1 else first-1
        batch_lam = lam[begin:last]
        if log and batch_size < lam.size:
            print(f'Galaxy wavelength batch: {first+1}–{last}/{lam.size}.', flush=True)
        prepared = prepare_galaxy_datacube(snap, lam=batch_lam, Lbox=Lbox, depth=depth, center=center, grid_shape=grid_shape,
            model=model, star_type=star_type, dust_to_metal=dust_to_metal, dust_temperature=dust_temperature,
            dust_sublimation_temperature=dust_sublimation_temperature, stellar_mass_is_initial=stellar_mass_is_initial,
            clip_population=clip_population, chunk_size=chunk_size, log=log, cache=cache,
            wavelength_range=None, n_wavelength=n_wavelength, wavelength_spacing=wavelength_spacing,
            frequency_chunk_size=frequency_chunk_size, storage=storage, storage_dir=storage_dir, max_grid_bytes=max_grid_bytes,
            inner_radius=inner_radius, dust_deposition=dust_deposition)
        rendered = raytrace_galaxy_datacube(prepared, npix=npix, frequency_chunk_size=frequency_chunk_size,
            step_fraction=step_fraction, N_NGB=N_NGB, workers=workers, query_chunk_size=query_chunk_size, log=log,
            backend=backend, return_optical_depth=return_optical_depth, supersampling=supersampling)
        del prepared
        if batch_size == lam.size: return rendered

        # allocate final image-sized outputs once and release each batch before preparing the next
        if result is None:
            result = {key: value for key, value in rendered.items() if key not in ('datacubes', 'optical_depth')}
            result['wavelength'] = lam
            for key in ('datacubes', 'optical_depth'):
                if key in rendered:
                    result[key] = {}
                    for name, values in rendered[key].items():
                        output = np.empty(values.shape[:-1]+(lam.size,))
                        result[key][name] = u.Quantity(output, values.unit, copy=False) if isinstance(values, u.Quantity) else output
            del values
        for key in ('datacubes', 'optical_depth'):
            if key in rendered:
                for name in rendered[key]: result[key][name][..., first:last] = rendered[key][name][..., first-begin:]
        del rendered
    return result


def prepare_galaxy_datacube(snap, lam=None, Lbox=10*u.kpc, depth=None, center=None, grid_shape=64,
                            model='BC03_miles_chabrier', star_type=4, dust_to_metal=0.5,
                            dust_temperature=None, dust_sublimation_temperature=1500*u.K,
                            stellar_mass_is_initial=False, clip_population=True, chunk_size=250000,
                            log=False, cache=True, wavelength_range=None, n_wavelength=24, wavelength_spacing='log',
                            frequency_chunk_size=4, storage='auto', storage_dir=None, max_grid_bytes=4*1024**3, inner_radius=0*u.kpc, dust_deposition='kernel'):
    """Read a GIZMO snapshot and prepare reusable stellar and dust coefficients.

    Uses the physics, units and spatial deposition described in galaxy_datacube.
    Reads snap.pos, snap.mass, snap.metals and snapshot-specific fields through snap,
    honoring its cached values and transformed frame. The snapshot caches full input
    fields; chunk_size bounds processing temporaries, not the snapshot cache.
    Returns a dictionary for raytrace_galaxy_datacube, retaining the emissivity grid
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
    dust_deposition='kernel' (default) requires gas smoothing lengths and expands compact
    occupancy to include the complete deposited support. This can increase coefficient
    storage and preparation time; the same memory budget checks and disk storage apply.
    'direct' reproduces the former single-cell gas deposition. Stars remain directly binned.
    """

    # validate the physical image and wavelength grids
    lam = _galaxy_wavelengths(lam, wavelength_range, n_wavelength, wavelength_spacing)
    wave = np.asarray(lam.to_value(u.AA))
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
    if dust_deposition not in ('kernel', 'direct'): raise ValueError("dust_deposition must be 'kernel' or 'direct'.")
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
                if part == 0 and dust_deposition == 'kernel':
                    h = snap.smooth[part][start:start+chunk_size][inside].to_value(u.cm)
                    metallicity = snap.metals[part][start:start+chunk_size][inside].to_value(u.dimensionless_unscaled)
                    if metallicity.ndim == 2: metallicity = metallicity[:, 0]
                    ratio = snap[part, 'DustToGasRatio_Local'][start:start+chunk_size][inside].to_value(u.dimensionless_unscaled) if (part, 'DustToGasRatio_Local') in snap else dust_to_metal*np.maximum(metallicity, 0)
                    temp = snap[part, 'Dust_Temperature'][start:start+chunk_size][inside].to_value(u.K) if dust_temperature is None else np.full(h.shape, u.Quantity(dust_temperature, u.K).to_value(u.K))
                    active = (snap.mass[part][start:start+chunk_size][inside].to_value(u.g)*ratio > 0) & (temp <= u.Quantity(dust_sublimation_temperature, u.K).to_value(u.K))
                    bin_particles_kernel(pos[inside][active], h=h[active], mins=lower, maxs=upper, dims=dims, occupancy=occupied)
                else: occupied[np.ravel_multi_index(bins.T, tuple(dims))] = True
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
    metals, ages, native_lam, spectra = resources.load_resource('stellar_population_models', model, cache=cache)
    if wave[0] < native_lam[0] or wave[-1] > native_lam[-1]:
        raise ValueError('Wavelengths exceed the BC03 table range.')
    sampled = sp.interpolate.interp1d(native_lam, spectra, axis=-1, assume_sorted=True)(wave)
    spectral_chunks = [(start, min(start+frequency_chunk_size, wave.size)) for start in range(0, wave.size, frequency_chunk_size)]
    stellar_spectra = [sp.interpolate.RegularGridInterpolator((metals, ages), sampled[..., start:stop])
                       for start, stop in spectral_chunks]
    mass_fraction = None if stellar_mass_is_initial else resources._BC03_current_mass_fraction(model, metals, ages)
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
                if dust_deposition == 'kernel': h = snap.smooth[part][start:stop][inside].to_value(u.cm)

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
            kernel_gas = part == 0 and dust_deposition == 'kernel'
            kernel_kwargs = dict(h=h, mins=lower, maxs=upper, dims=dims, field_indices=field_indices) if kernel_gas else {}
            if part == 0:
                if kernel_gas: bin_particles_kernel(selected, mass/volume, out=dust_grid, **kernel_kwargs)
                else: bin_particles_direct(bin_pos, mass/volume, out=dust_grid, **bin_kwargs)
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
                if kernel_gas: bin_particles_kernel(selected, luminosity, out=out, **kernel_kwargs)
                else: bin_particles_direct(bin_pos, luminosity, out=out, **bin_kwargs)
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
                grid_shape=tuple(dims), particle_counts=counts, dust_deposition=dust_deposition)


def raytrace_galaxy_datacube(prepared, npix=64, frequency_chunk_size=4, step_fraction=0.25,
                             N_NGB=8, workers=-1, query_chunk_size=4096, log=False, backend='auto', return_optical_depth=True, supersampling=2):
    """Render a prepared galaxy without reading its snapshot or rebuilding its spectra.

    prepared is the dictionary returned by prepare_galaxy_datacube. npix and integration
    settings may change between calls; the dictionary is not modified. All wavelengths
    and three opacity cases share spatial queries and steps. Returns the same datacubes
    and metadata as galaxy_datacube. backend='auto' uses fused Numba
    transfer, with workers controlling parallel rays (-1 uses the current Numba thread
    count). query_chunk_size bounds complete rays per batch. Empty compact-grid samples
    skip spectral work. The first call includes compilation. backend='numpy' selects
    the original reference tracer; frequency_chunk_size only affects that backend.
    return_optical_depth=False omits optical_depth from the result and, with the compiled
    backend, avoids allocating those cubes. Wavelength-unit conversion is done in place.
    supersampling=2 averages four uniformly spaced rays per final pixel in linear
    intensity; 1 reproduces pixel-center sampling. Subpixel offsets are traced one at
    a time, retaining the accumulated output and one sample cube, not a full enlarged
    cube. Optical depths are averaged separately as diagnostics, not used to attenuate
    the averaged intensity. Runtime scales approximately with supersampling squared.
    """
    if int(npix) != npix or npix < 1: raise ValueError('npix must be a positive integer.')
    if isinstance(supersampling, bool) or not isinstance(supersampling, (int, np.integer)) or supersampling < 1:
        raise ValueError('supersampling must be a positive integer.')
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

    # accumulate linear subpixel intensities without constructing an enlarged spectral cube
    offsets = ((np.arange(supersampling)+0.5)/supersampling-0.5)*(Lbox/npix)
    intensity = None; optical_depth = None
    for iy, dy in enumerate(offsets):
        for ix, dx in enumerate(offsets):
            x0[..., 0] = xx+dx; x0[..., 1] = yy+dy
            x1[..., :2] = x0[..., :2]
            if log: print(f'Galaxy subpixel rays: {iy*supersampling+ix+1}/{supersampling**2}', flush=True)
            traced = rad.raytrace_particles(x0, x1, prepared['emission'], prepared['dust_density'],
                tree=grid, opacity_factors=factors, frequency_chunk_size=frequency_chunk_size, N_NGB=N_NGB,
                step_fraction=step_fraction, query_chunk_size=query_chunk_size, workers=workers, ret_tau=return_optical_depth,
                log=log, reverse=True, field_indices=prepared.get('field_indices'), backend=backend)
            if intensity is None:
                intensity = traced[0] if return_optical_depth else traced
                if return_optical_depth: optical_depth = traced[1]
            else:
                intensity += traced[0] if return_optical_depth else traced
                if return_optical_depth: optical_depth += traced[1]
            del traced
    intensity /= supersampling**2
    if return_optical_depth: optical_depth /= supersampling**2

    # retain final pixel geometry and convert the averaged spectrum to the requested units
    output_unit = u.erg/u.s/u.cm**2/u.sr/lam.unit
    intensity *= (u.erg/u.s/u.cm**2/u.sr/u.AA).to(output_unit)
    intensity = u.Quantity(intensity, output_unit, copy=False)
    pixel_area = (Lbox/npix)**2
    names = ('pure_emission', 'absorption', 'effective')
    result = dict(wavelength=lam, extent=np.array([-Lbox/2,Lbox/2,-Lbox/2,Lbox/2])*u.cm,
                pixel_area=pixel_area*u.cm**2, particle_counts=counts, grid_shape=tuple(dims), supersampling=int(supersampling),
                datacubes={name: intensity[..., i] for i, name in enumerate(names)})
    if return_optical_depth: result['optical_depth'] = {name: optical_depth[..., i] for i, name in enumerate(names)}
    return result
