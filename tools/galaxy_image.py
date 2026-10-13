#!/usr/bin/env python3
"""Load, center and render a galaxy to an exact-size PNG and a floating-point RGB array."""
import argparse
from pathlib import Path


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
    if png.exists() or raw.exists(): parser.error('Output already exists; choose a new --output prefix.')
    if args.wavelengths_nm is not None:
        wave = np.asarray(args.wavelengths_nm)
        if wave.size < 2 or np.any(~np.isfinite(wave)) or np.any(wave <= 0) or np.any(np.diff(wave) <= 0):
            parser.error('wavelengths-nm must contain at least two finite, positive, increasing values.')

    # center in the snapshot frame and render without a plotting figure
    import jaba
    from jaba.apps import galaxy
    from jaba.utils import units as u
    from matplotlib.image import imsave
    if not args.quiet:
        print(f'Image: {args.npix} x {args.npix}; grid: {grid_shape} cubed; spectral chunk: {frequency_chunk_size}', flush=True)
    snap = jaba.load(str(args.filename.expanduser()))
    snap.reset_transform()
    snap.center_on(args.center_type)
    if args.faceon_radius_kpc is not None: snap.faceon(args.faceon_radius_kpc*u.kpc)
    spectral = {} if args.wavelengths_nm is None else dict(lam=wave*u.nm)
    image = galaxy.galaxy_image(snap, Lbox=args.box_size*u.kpc, npix=args.npix, grid_shape=grid_shape,
        frequency_chunk_size=frequency_chunk_size, max_grid_bytes=budget, wavelength_batch_size='auto',
        component=args.component, step_fraction=args.step_fraction, supersampling=args.supersampling,
        stellar_deposition=args.stellar_deposition, stellar_ngb=args.stellar_ngb,
        dust_deposition=args.dust_deposition, workers=args.workers, log=not args.quiet, plot=False, **spectral)

    # write pixels directly without axes or resizing; preserve unquantized rgb with row zero at the bottom
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with raw.open('xb') as file: np.save(file, image['rgb'], allow_pickle=False)
    with png.open('xb') as file: imsave(file, image['rgb'], format='png', origin='lower', dpi=100)
    if not args.quiet: print(f'Saved {png} and {raw}', flush=True)
    return png, raw


if __name__ == '__main__':
    main()
