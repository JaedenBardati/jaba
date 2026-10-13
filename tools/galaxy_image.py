#!/usr/bin/env python3
"""Load, center and render a galaxy to an exact-size PNG and a floating-point RGB array."""
import argparse
import hashlib
import json
import os
import tempfile
import fcntl
from pathlib import Path


def _publish(path, writer):
    """Publish a fully flushed new file without replacing an existing result."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.partial-', delete=False) as file:
        temporary = Path(file.name)
        writer(file)
        file.flush()
        os.fsync(file.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _batch_checkpoint(directory, first, last, rendered=None):
    """Store completed image-sized batches without pickled Python objects."""
    import numpy as np
    from jaba.utils import units as u
    path = directory/f'batch-{first:05d}-{last:05d}.npz'
    if rendered is None:
        if not path.exists(): return None
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            result = metadata['plain']
            if 'grid_shape' in result: result['grid_shape'] = tuple(result['grid_shape'])
            for key, unit in metadata['quantities'].items(): result[key] = saved[key]*u.Unit(unit)
            for group, entries in metadata['groups'].items():
                result[group] = {name: saved[group+'_'+name]*u.Unit(unit) if unit else saved[group+'_'+name].copy()
                                 for name, unit in entries.items()}
        return result

    # retain units and geometry alongside the numerical arrays
    arrays = {}; metadata = dict(plain={}, quantities={}, groups={})
    for key, value in rendered.items():
        if key in ('datacubes', 'optical_depth'):
            metadata['groups'][key] = {}
            for name, data in value.items():
                arrays[key+'_'+name] = np.asarray(data)
                metadata['groups'][key][name] = str(data.unit) if isinstance(data, u.Quantity) else None
        elif isinstance(value, u.Quantity):
            arrays[key] = value.value
            metadata['quantities'][key] = str(value.unit)
        else: metadata['plain'][key] = value
    arrays['metadata'] = json.dumps(metadata, default=lambda value: value.item())
    _publish(path, lambda file: np.savez(file, **arrays))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', type=Path, help='GIZMO snapshot filename')
    parser.add_argument('--box-size', type=float, default=20., help='full image width in kpc')
    parser.add_argument('--npix', type=int, default=128, help='output width and height in pixels')
    parser.add_argument('--grid-shape', type=int, help='cells per axis; defaults to min(npix, 512)')
    parser.add_argument('--frequency-chunk-size', type=int, help='particle spectral chunk; defaults to min(24, budget-based capacity)')
    parser.add_argument('--max-grid-gib', type=float, default=4., help='coefficient budget in GiB; total RAM use is higher')
    parser.add_argument('--output', type=Path, help='output prefix; defaults to ./<snapshot stem>_rgb')
    parser.add_argument('--center-type', type=int, default=3, help='particle type passed to center_on')
    parser.add_argument('--faceon-radius-kpc', type=float, help='optionally rotate face-on using this radius in kpc')
    parser.add_argument('--component', choices=('effective', 'absorption', 'pure_emission'), default='effective')
    parser.add_argument('--step-fraction', type=float, default=.25)
    parser.add_argument('--supersampling', type=int, default=1)
    parser.add_argument('--stellar-deposition', choices=('direct', 'kernel'), default='direct')
    parser.add_argument('--stellar-ngb', type=int, default=32)
    parser.add_argument('--dust-deposition', choices=('direct', 'kernel'), default='kernel')
    parser.add_argument('--wavelengths-nm', type=float, nargs='+', help='increasing wavelength samples; otherwise infer from RGB filters')
    parser.add_argument('--workers', type=int, default=-1)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--restart-dir', type=Path, help='checkpoint directory; defaults to <output>.restart')
    parser.add_argument('--resume', action='store_true', help='resume matching checkpoints and finish missing outputs')
    parser.add_argument('--wavelength-batch-size', type=int, default=3, help='checkpoint after this many wavelengths; minimum 2')
    args = parser.parse_args(argv)

    # validate settings and refuse to overwrite existing outputs before loading the snapshot
    import numpy as np
    grid_shape = min(args.npix, 512) if args.grid_shape is None else args.grid_shape
    if args.npix < 1 or grid_shape < 2: parser.error('npix must be positive and grid-shape must be at least 2.')
    if any(not np.isfinite(value) or value <= 0 for value in (args.box_size, args.max_grid_gib, args.step_fraction)):
        parser.error('box-size, max-grid-gib and step-fraction must be finite and positive.')
    if args.supersampling < 1 or args.stellar_ngb < 1: parser.error('supersampling and stellar-ngb must be positive.')
    if args.faceon_radius_kpc is not None and (not np.isfinite(args.faceon_radius_kpc) or args.faceon_radius_kpc <= 0):
        parser.error('faceon-radius-kpc must be finite and positive.')
    budget = int(args.max_grid_gib*1024**3)
    capacity = max(1, budget//(grid_shape**3*8)-1)
    frequency_chunk_size = min(24, capacity) if args.frequency_chunk_size is None else args.frequency_chunk_size
    if frequency_chunk_size < 1: parser.error('frequency-chunk-size must be positive.')
    prefix = args.output.expanduser() if args.output is not None else Path(args.filename.stem+'_rgb')
    png, raw = Path(str(prefix)+'.png'), Path(str(prefix)+'.npy')
    if not args.resume and (png.exists() or raw.exists()): parser.error('Output already exists; choose a new --output prefix or use --resume.')
    if args.wavelengths_nm is not None:
        wave = np.asarray(args.wavelengths_nm)
        if wave.size < 2 or np.any(~np.isfinite(wave)) or np.any(wave <= 0) or np.any(np.diff(wave) <= 0):
            parser.error('wavelengths-nm must contain at least two finite, positive, increasing values.')

    # validate checkpoint identity and keep a process lock that releases automatically on failure
    import jaba
    from jaba.apps import galaxy
    from jaba.utils import units as u
    from matplotlib.image import imsave
    if args.wavelength_batch_size < 2: parser.error('wavelength-batch-size must be at least 2.')
    directory = args.restart_dir.expanduser() if args.restart_dir is not None else Path(str(prefix)+'.restart')
    if args.resume and not directory.is_dir(): parser.error('No restart directory found.')
    if not args.resume and directory.exists(): parser.error('Restart directory exists; use --resume or choose a new output prefix.')
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory/'run.lock').open('a')
    with lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: parser.error('Another process is using this restart directory.')
        filename = args.filename.expanduser().resolve()
        stat = filename.stat() if filename.exists() else None
        settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                    if key not in ('resume', 'quiet', 'restart_dir', 'output', 'filename')}
        settings.update(filename=str(filename), output=str(prefix.resolve()),
                        snapshot=None if stat is None else [stat.st_size, stat.st_mtime_ns],
                        code=hashlib.sha256(Path(galaxy.__file__).read_bytes()+Path(__file__).read_bytes()+
                            Path(galaxy.rad.__file__).read_bytes()+Path(jaba.utils.grid.__file__).read_bytes()).hexdigest())
        manifest = directory/'settings.json'
        if args.resume:
            if not manifest.exists() or json.loads(manifest.read_text()) != settings:
                parser.error('Restart settings, snapshot or code differ; use a new output/restart directory.')
        else: _publish(manifest, lambda file: file.write(json.dumps(settings, sort_keys=True).encode()))

        # reuse a completed rgb image or render only unfinished wavelength batches
        rgb_file = directory/'rgb.npy'
        if rgb_file.exists():
            rgb = np.load(rgb_file, allow_pickle=False)
        else:
            if not args.quiet:
                print(f'Image: {args.npix} x {args.npix}; grid: {grid_shape} cubed; spectral chunk: {frequency_chunk_size}', flush=True)
            snap = jaba.load(str(filename))
            snap.reset_transform()
            snap.center_on(args.center_type)
            if args.faceon_radius_kpc is not None: snap.faceon(args.faceon_radius_kpc*u.kpc)
            spectral = {} if args.wavelengths_nm is None else dict(lam=wave*u.nm)
            image = galaxy.galaxy_image(snap, Lbox=args.box_size*u.kpc, npix=args.npix, grid_shape=grid_shape,
                frequency_chunk_size=frequency_chunk_size, max_grid_bytes=budget,
                wavelength_batch_size=min(args.wavelength_batch_size, max(2, capacity)),
                _batch_checkpoint=lambda first, last, result=None: _batch_checkpoint(directory, first, last, result),
                component=args.component, step_fraction=args.step_fraction, supersampling=args.supersampling,
                stellar_deposition=args.stellar_deposition, stellar_ngb=args.stellar_ngb,
                dust_deposition=args.dust_deposition, workers=args.workers, log=not args.quiet, plot=False, **spectral)
            rgb = image['rgb']
            _publish(rgb_file, lambda file: np.save(file, rgb, allow_pickle=False))

        # publish exact-size pixels and unquantized rgb without replacing completed outputs
        prefix.parent.mkdir(parents=True, exist_ok=True)
        if not raw.exists(): _publish(raw, lambda file: np.save(file, rgb, allow_pickle=False))
        if not png.exists(): _publish(png, lambda file: imsave(file, rgb, format='png', origin='lower', dpi=100))
        if not args.quiet: print(f'Saved {png} and {raw}; restart files retained in {directory}', flush=True)
    return png, raw


if __name__ == '__main__':
    main()
