import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from jabapy.apps import galaxy
from jabapy.snapshot import GIZMO_Snapshot
from astropy.cosmology import LambdaCDM
from jabapy.utils import resources
from jabapy.utils import units as u


def write_header(file):
    header = file.create_group('Header')
    header.attrs.update(Time=1e9, ComovingIntegrationOn=0, HubbleParam=1.,
                        UnitMass_In_CGS=u.Msun.to(u.g), UnitLength_In_CGS=u.kpc.to(u.cm),
                        UnitVelocity_In_CGS=(u.kpc/u.yr).to(u.cm/u.s))


class TestGalaxyDatacube(unittest.TestCase):
    def test_stellar_kernel_conservation_and_compact_support(self):
        options = {key: value for key, value in self.kw.items() if key not in ('npix', 'N_NGB', 'workers')}
        options.update(dust_to_metal=0, grid_shape=8)
        direct = galaxy.prepare_galaxy_datacube(self.snap, **options)
        dense = galaxy.prepare_galaxy_datacube(self.snap, stellar_deposition='kernel', **options)
        compact = galaxy.prepare_galaxy_datacube(self.snap, stellar_deposition='kernel', storage='compact', **options)
        np.testing.assert_allclose(dense['emission'].sum(axis=0), direct['emission'].sum(axis=0), rtol=1e-14)
        self.assertGreater(np.count_nonzero(dense['emission']), np.count_nonzero(direct['emission']))
        np.testing.assert_allclose(compact['emission'][compact['field_indices']], dense['emission'], rtol=1e-14)
        self.assertEqual(dense['stellar_deposition'], 'kernel')
        self.assertEqual(dense['stellar_ngb'], 32)

    def test_stellar_neighbour_radii_exclude_self_and_wind(self):
        class Stars:
            pos = {4: np.column_stack((np.arange(5.), np.zeros((5, 2))))*u.cm}
            def __getitem__(self, key): return np.array([1., 0., 1., 1., 1.])*u.yr
        snap = Stars()
        np.testing.assert_allclose(galaxy._stellar_kernel_radii(snap, 4, 1, 2, False), [2., 0., 1., 1., 1.])
        np.testing.assert_allclose(galaxy._stellar_kernel_radii(snap, 4, 32, 2, False), [4., 0., 2., 3., 4.])
        for ngb in (0, -1, True, 1.5):
            with self.assertRaises(ValueError): galaxy._stellar_kernel_radii(snap, 4, ngb, 2, False)

    def test_stellar_radii_reused_across_wavelength_batches(self):
        options = dict(self.kw, stellar_deposition='kernel', wavelength_batch_size=2)
        with patch.object(galaxy, '_stellar_kernel_radii', wraps=galaxy._stellar_kernel_radii) as radii:
            batched = galaxy.galaxy_datacube(self.snap, **options)
            self.assertEqual(radii.call_count, 1)
        full = galaxy.galaxy_datacube(self.snap, **dict(options, wavelength_batch_size=None))
        for name in full['datacubes']:
            np.testing.assert_allclose(batched['datacubes'][name], full['datacubes'][name], rtol=1e-12)

    def test_wavelength_batches_match_full_kernel_transfer(self):

        # compare all transfer cases, both backends, supersampling and an odd final batch
        options = dict(self.kw, lam=np.geomspace(.1, 1000, 5)*u.micron,
                       dust_deposition='kernel', supersampling=2)
        for backend in ('numpy', 'numba'):
            for storage in ('dense', 'compact'):
                with self.subTest(backend=backend, storage=storage):
                    full = galaxy.galaxy_datacube(self.snap, backend=backend, storage=storage,
                                                 wavelength_batch_size=None, **options)
                    batched = galaxy.galaxy_datacube(self.snap, backend=backend, storage=storage,
                                                    wavelength_batch_size=2, **options)
                    self.assertEqual(full['particle_counts'], batched['particle_counts'])
                    np.testing.assert_array_equal(full['wavelength'], batched['wavelength'])
                    for key in ('datacubes', 'optical_depth'):
                        for name in full[key]:
                            np.testing.assert_allclose(full[key][name], batched[key][name], rtol=1e-12, atol=0)

    def test_automatic_batches_respect_budget_and_release_grids(self):
        prepare = galaxy.prepare_galaxy_datacube
        previous = []; sizes = []

        # ensure the next allocation cannot overlap the previous coefficient grids
        def checked_prepare(*args, **kwargs):
            self.assertTrue(all(ref() is None for ref in previous))
            prepared = prepare(*args, **kwargs)
            self.assertLessEqual(prepared['coefficient_bytes'], 3*4**3*8)
            sizes.append(prepared['wavelength'].size)
            previous[:] = [weakref.ref(prepared[key]) for key in ('emission', 'dust_density')]
            return prepared
        options = dict(self.kw, dust_deposition='kernel', return_optical_depth=False)
        del options['lam']
        with patch.object(galaxy, 'prepare_galaxy_datacube', side_effect=checked_prepare):
            result = galaxy.galaxy_datacube(self.snap, n_wavelength=5, max_grid_bytes=3*4**3*8, **options)
        self.assertEqual(sizes, [2, 2, 2])
        self.assertTrue(all(ref() is None for ref in previous))
        self.assertNotIn('optical_depth', result)
        self.assertEqual(result['datacubes']['effective'].shape, (2, 2, 5))

    def test_invalid_wavelength_batch_size(self):
        for size in (0, 1, True, 2.5, 'invalid'):
            with self.subTest(size=size), self.assertRaises(ValueError):
                galaxy.galaxy_datacube(self.snap, wavelength_batch_size=size, **self.kw)

    def setUp(self):
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        path = Path(directory.name)/'snapshot.hdf5'
        with h5py.File(path, 'w') as file:
            write_header(file)
            for part in (0, 4):
                group = file.create_group(f'PartType{part}')
                group['Coordinates'] = [[-.2, 0, 0], [.2, 0, 0]]
                group['Masses'] = [1., 1.]
                group['Metallicity'] = [[.02], [.02]]
                if part == 0:
                    group['Dust_Temperature'] = [20., 30.]
                    group['SmoothingLength'] = [.45, .35]
                else: group['StellarFormationTime'] = [8e8, 9e8]
        self.snap = GIZMO_Snapshot(path)
        table = (np.array([.01, .03]), np.array([0., 1e9]), np.array([900., 1e5, 1.1e7]), np.full((2, 2, 3), 2.))
        source = patch.object(resources, 'load_resource', return_value=table); source.start(); self.addCleanup(source.stop)
        fraction = patch.object(resources, '_BC03_current_mass_fraction', return_value=lambda pts: np.full(pts.shape[0], .5))
        fraction.start(); self.addCleanup(fraction.stop)
        self.kw = dict(lam=[.1, 10, 1000]*u.micron, Lbox=1*u.kpc, grid_shape=4, npix=2, N_NGB=8, workers=1, dust_deposition='direct')

    def test_one_call_visual_image(self):
        import matplotlib.pyplot as plt
        self.addCleanup(plt.close, 'all')
        options = {key: value for key, value in self.kw.items() if key != 'lam'}
        image = galaxy.galaxy_image(self.snap, **options)
        self.assertEqual(image['rgb'].shape, (2, 2, 3))
        self.assertTrue(np.all(np.isfinite(image['rgb'])))
        self.assertTrue(np.any(image['rgb'] > 0))
        self.assertTrue(np.all((image['rgb'] >= 0) & (image['rgb'] <= 1)))
        np.testing.assert_allclose(image['extent'].to_value(u.kpc), [-.5, .5, -.5, .5])

    def test_datacube_shapes_attenuation_and_chunking(self):
        one = galaxy.galaxy_datacube(self.snap, frequency_chunk_size=1, chunk_size=1, **self.kw)
        all_channels = galaxy.galaxy_datacube(self.snap, frequency_chunk_size=3, chunk_size=10, **self.kw)
        self.assertEqual(one['particle_counts']['stars'], 2)
        self.assertEqual(one['particle_counts']['gas'], 2)
        for name in one['datacubes']:
            self.assertEqual(one['datacubes'][name].shape, (2, 2, 3))
            np.testing.assert_allclose(one['datacubes'][name], all_channels['datacubes'][name])
        thin, absorption, effective = [one['datacubes'][key].value for key in ('pure_emission', 'absorption', 'effective')]
        self.assertTrue(np.all(thin >= absorption))
        self.assertTrue(np.all(absorption >= effective))

    def test_prepared_grid_reuse_without_snapshot_or_resources(self):
        preparation = {key: value for key, value in self.kw.items() if key not in ('npix', 'N_NGB', 'workers')}
        prepared = galaxy.prepare_galaxy_datacube(self.snap, **preparation)
        emission, density = prepared['emission'].copy(), prepared['dust_density'].copy()
        expected = galaxy.galaxy_datacube(self.snap, **self.kw)
        with patch.object(h5py, 'File', side_effect=AssertionError('Snapshot read during rendering')), \
             patch.object(resources, 'load_resource', side_effect=AssertionError('Resource read during rendering')):
            actual = galaxy.raytrace_galaxy_datacube(prepared, npix=2, workers=1)
            higher = galaxy.raytrace_galaxy_datacube(prepared, npix=3, step_fraction=.15, workers=1)
        np.testing.assert_array_equal(prepared['emission'], emission)
        np.testing.assert_array_equal(prepared['dust_density'], density)
        self.assertEqual(higher['datacubes']['absorption'].shape, (3, 3, 3))
        for name in actual['datacubes']:
            for field in ('datacubes', 'optical_depth'):
                np.testing.assert_allclose(actual[field][name], expected[field][name])

    def test_supersampling_matches_fine_pixel_intensity_average(self):

        # compare subpixel tracing with independently rendered fine pixels on both backends
        preparation = {key: value for key, value in self.kw.items() if key not in ('npix', 'N_NGB', 'workers')}
        for backend, storage, factor in (('numpy', 'dense', 2), ('numba', 'compact', 3)):
            with self.subTest(backend=backend, storage=storage, factor=factor):
                prepared = galaxy.prepare_galaxy_datacube(self.snap, storage=storage, **preparation)
                coarse = galaxy.raytrace_galaxy_datacube(prepared, npix=2, supersampling=factor, backend=backend, workers=1)
                fine = galaxy.raytrace_galaxy_datacube(prepared, npix=2*factor, supersampling=1, backend=backend, workers=1)
                for field in ('datacubes', 'optical_depth'):
                    for name in coarse[field]:
                        expected = fine[field][name].reshape(2, factor, 2, factor, 3).mean(axis=(1, 3))
                        np.testing.assert_allclose(coarse[field][name], expected, rtol=1e-10, atol=0)
                self.assertEqual(coarse['supersampling'], factor)
                np.testing.assert_allclose(coarse['pixel_area'], factor**2*fine['pixel_area'])
                np.testing.assert_array_equal(coarse['extent'], fine['extent'])
                no_tau = galaxy.raytrace_galaxy_datacube(prepared, npix=2, supersampling=factor,
                    backend=backend, workers=1, return_optical_depth=False)
                self.assertNotIn('optical_depth', no_tau)
                for name in coarse['datacubes']:
                    np.testing.assert_allclose(no_tau['datacubes'][name], coarse['datacubes'][name])

    def test_supersampling_defaults_validation_and_single_ray(self):

        # preserve pixel-center behavior when disabled and reject invalid factors before loading
        preparation = {key: value for key, value in self.kw.items() if key not in ('npix', 'N_NGB', 'workers')}
        prepared = galaxy.prepare_galaxy_datacube(self.snap, **preparation)
        with patch.object(galaxy.rad, 'raytrace_particles', wraps=galaxy.rad.raytrace_particles) as trace:
            single = galaxy.raytrace_galaxy_datacube(prepared, npix=2, supersampling=1, workers=1)
            self.assertEqual(trace.call_count, 1)
            starts = trace.call_args.args[0]
            np.testing.assert_allclose(starts[..., 0], np.array([[-.25, .25], [-.25, .25]])*u.kpc.to(u.cm))
            np.testing.assert_allclose(starts[..., 1], np.array([[-.25, -.25], [.25, .25]])*u.kpc.to(u.cm))
            trace.reset_mock()
            default = galaxy.raytrace_galaxy_datacube(prepared, npix=2, workers=1, supersampling=2)
            self.assertEqual(trace.call_count, 4)
        self.assertEqual(default['supersampling'], 2)
        self.assertEqual(single['supersampling'], 1)
        for factor in (0, -1, 1.5, True, None):
            with self.subTest(factor=factor), patch.object(galaxy, 'prepare_galaxy_datacube') as prepare:
                with self.assertRaises(ValueError): galaxy.galaxy_datacube(self.snap, supersampling=factor)
                prepare.assert_not_called()
                with self.assertRaises(ValueError): galaxy.raytrace_galaxy_datacube(prepared, supersampling=factor)

    def test_default_kernel_spreads_dust_and_thermal_emission(self):

        # conserve dust mass and thermal power while expanding support consistently in every layout
        kwargs = {key: value for key, value in self.kw.items() if key not in ('npix', 'N_NGB', 'workers', 'dust_deposition')}
        direct = galaxy.prepare_galaxy_datacube(self.snap, dust_deposition='direct', **kwargs)
        dense = galaxy.prepare_galaxy_datacube(self.snap, storage='dense', **kwargs)
        compact = galaxy.prepare_galaxy_datacube(self.snap, storage='compact', **kwargs)
        self.assertEqual(dense['dust_deposition'], 'kernel')
        self.assertGreater(np.count_nonzero(dense['dust_density']), np.count_nonzero(direct['dust_density']))
        np.testing.assert_allclose(dense['dust_density'].sum(), direct['dust_density'].sum(), rtol=1e-13)
        np.testing.assert_allclose(dense['emission'].sum(axis=0), direct['emission'].sum(axis=0), rtol=1e-13)
        for key in ('dust_density', 'emission'):
            np.testing.assert_allclose(compact[key][compact['field_indices']], dense[key], rtol=1e-13)
            np.testing.assert_array_equal(compact[key][0], 0)
        with tempfile.TemporaryDirectory() as directory:
            disk = galaxy.prepare_galaxy_datacube(self.snap, storage='compact', storage_dir=directory, **kwargs)
            for key in ('dust_density', 'emission'):
                np.testing.assert_allclose(disk[key], compact[key])
        with self.assertRaises(ValueError): galaxy.prepare_galaxy_datacube(self.snap, dust_deposition='invalid', **kwargs)

    def test_deposition_conservation_boundaries_and_exclusions(self):
        positions = np.array([[-.5, -.5, -.5], [-.1, 0, 0], [-.1, 0, 0], [.5, 0, 0], [-.5001, 0, 0], [.2, 0, 0]])
        path = self.snap.filepath.with_name('boundaries.hdf5')
        with h5py.File(path, 'w') as file:
            write_header(file)
            for part in (0, 4):
                group = file.create_group(f'PartType{part}')
                group['Coordinates'] = positions; group['Masses'] = np.arange(1., 7.)
                group['Metallicity'] = np.full((6, 1), .02)
                if part == 0: group['Dust_Temperature'] = [0., 20., 2000., 30., 30., 30.]
                else: group['StellarFormationTime'] = [8e8, 9e8, 0., 8e8, 8e8, -1.]
        self.snap = GIZMO_Snapshot(path)
        kw = {key: value for key, value in self.kw.items() if key not in ('npix', 'N_NGB', 'workers')}
        histogram = lambda pos, weights: np.histogramdd(pos, bins=(4, 4, 4), range=[(-.5, .5)]*3, weights=weights)[0].ravel()
        volume = (.25*u.kpc).to_value(u.cm)**3
        dust_positions = positions[[0, 1, 5]]; dust_masses = np.array([.01, .02, .06])*u.Msun.to(u.g)
        expected_density = histogram(dust_positions, dust_masses)/volume
        stellar = histogram(positions[:2], [1., 2.])*4*u.Lsun.to(u.erg/u.s)/(4*np.pi)
        for chunk, storage in [(chunk, storage) for chunk in (1, 3, 100) for storage in ('dense', 'compact')]:
            prepared = galaxy.prepare_galaxy_datacube(self.snap, chunk_size=chunk, storage=storage, **kw)
            if prepared['field_indices'] is not None:
                for key in ('emission', 'dust_density'): prepared[key] = prepared[key][prepared['field_indices']]
            self.assertEqual(prepared['particle_counts'], dict(gas=4, stars=2, population_clipped=0, wind_excluded=2))
            np.testing.assert_allclose(prepared['dust_density'], expected_density, rtol=1e-13, atol=0)
            for k, lam in enumerate(kw['lam']):
                with np.errstate(over='ignore'):
                    brightness = galaxy.rad.planck_function_lam(lam, [20., 30.]*u.K).to_value(u.erg/u.s/u.cm**2/u.sr/u.AA)
                thermal = histogram(dust_positions, dust_masses*prepared['k_abs'][k]*np.r_[0., brightness])
                np.testing.assert_allclose(prepared['emission'][:, k], (stellar+thermal)/volume, rtol=1e-13, atol=0)

    def test_wavelength_range_and_datacube_units(self):
        kw = {key: value for key, value in self.kw.items() if key != 'lam'}
        for spacing, sample in [('linear', np.linspace), ('log', np.geomspace)]:
            result = galaxy.galaxy_datacube(self.snap, wavelength_range=[.3, 1.2]*u.micron,
                                             n_wavelength=9, wavelength_spacing=spacing, **kw)
            np.testing.assert_allclose(result['wavelength'].to_value(u.micron), sample(.3, 1.2, 9))
            self.assertEqual(set(result['datacubes']), {'pure_emission', 'absorption', 'effective'})
            for name, cube in result['datacubes'].items():
                self.assertEqual(cube.shape, (2, 2, 9))
                self.assertTrue(cube.unit.is_equivalent(u.erg/u.s/u.cm**2/u.sr/u.micron))
                self.assertEqual(result['optical_depth'][name].shape, cube.shape)
                band = galaxy.rad.integrate_bandpass(result['wavelength'], cube)
                expected = np.trapezoid(cube.value, result['wavelength'].value, axis=-1)*cube.unit*result['wavelength'].unit
                np.testing.assert_allclose(band, expected, rtol=1e-13)
        with self.assertRaises(ValueError): galaxy.galaxy_datacube(self.snap, wavelength_range=[1., .3], **kw)
        with self.assertRaises(ValueError): galaxy.galaxy_datacube(self.snap, n_wavelength=1, **kw)
        with self.assertRaises(ValueError): galaxy.galaxy_datacube(self.snap, wavelength_range=[.3, 1.2], **self.kw)

    def test_compact_storage_matches_dense_geometry_and_spectra(self):
        kw = {key: value for key, value in self.kw.items() if key not in ('npix', 'N_NGB', 'workers')}
        for dims in (4, (4, 5, 6)):
            dense = galaxy.prepare_galaxy_datacube(self.snap, **dict(kw, grid_shape=dims), storage='dense')
            compact = galaxy.prepare_galaxy_datacube(self.snap, **dict(kw, grid_shape=dims), storage='compact',
                                                      frequency_chunk_size=1, chunk_size=1)
            self.assertLess(compact['emission'].size, dense['emission'].size)
            for key in ('emission', 'dust_density'):
                np.testing.assert_allclose(compact[key][compact['field_indices']], dense[key], rtol=1e-13, atol=0)
                np.testing.assert_array_equal(compact[key][0], 0)
            for neighbors in (4, 8):
                a = galaxy.raytrace_galaxy_datacube(dense, npix=3, N_NGB=neighbors)
                b = galaxy.raytrace_galaxy_datacube(compact, npix=3, N_NGB=neighbors)
                for field in ('datacubes', 'optical_depth'):
                    for name in a[field]: np.testing.assert_allclose(a[field][name], b[field][name], rtol=1e-13, atol=0)

    def test_auto_storage_budget_and_early_dense_rejection(self):
        kw = {key: value for key, value in self.kw.items() if key not in ('npix', 'N_NGB', 'workers')}
        prepared = galaxy.prepare_galaxy_datacube(self.snap, max_grid_bytes=400, **kw)
        self.assertEqual(prepared['storage'], 'compact')
        self.assertLessEqual(prepared['emission'].nbytes+prepared['dust_density'].nbytes+prepared['field_indices'].nbytes, 400)
        with patch.object(h5py, 'File', side_effect=AssertionError('Snapshot opened before budget check')), \
             patch.object(resources, 'load_resource', side_effect=AssertionError('Resources loaded before budget check')):
            with self.assertRaisesRegex(MemoryError, 'exceeding'):
                galaxy.prepare_galaxy_datacube(self.snap, storage='dense', max_grid_bytes=400, **kw)
            with self.assertRaisesRegex(MemoryError, 'occupancy'):
                galaxy.prepare_galaxy_datacube(self.snap, max_grid_bytes=100, **kw)
        with self.assertRaisesRegex(MemoryError, 'exceeding'):
            galaxy.prepare_galaxy_datacube(self.snap, max_grid_bytes=330, **kw)
        unlimited = galaxy.prepare_galaxy_datacube(self.snap, max_grid_bytes=None, **kw)
        self.assertEqual(unlimited['storage'], 'dense')

    def test_disk_storage_is_unique_and_preserves_coefficients(self):
        kw = {key: value for key, value in self.kw.items() if key not in ('npix', 'N_NGB', 'workers')}
        with tempfile.TemporaryDirectory() as directory:
            for storage in ('dense', 'compact'):
                expected = galaxy.prepare_galaxy_datacube(self.snap, storage=storage, **kw)
                disk = galaxy.prepare_galaxy_datacube(self.snap, storage=storage, storage_dir=directory, **kw)
                other = galaxy.prepare_galaxy_datacube(self.snap, storage=storage, storage_dir=directory, **kw)
                self.assertNotEqual(disk['storage_path'], other['storage_path'])
                for key in ('emission', 'dust_density'):
                    self.assertIsInstance(disk[key], np.memmap)
                    saved = np.load(disk['storage_path']/(key+'.npy'), mmap_mode='r')
                    np.testing.assert_array_equal(saved, expected[key])
                a = galaxy.raytrace_galaxy_datacube(expected, npix=2)
                b = galaxy.raytrace_galaxy_datacube(disk, npix=2)
                for field in ('datacubes', 'optical_depth'):
                    for name in a[field]: np.testing.assert_array_equal(a[field][name], b[field][name])
            with patch.object(galaxy.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(OSError, 'disk space'):
                    galaxy.prepare_galaxy_datacube(self.snap, storage_dir=directory, **kw)

    def test_preparation_bounds_particle_spectra(self):
        original = galaxy.bin_particles_direct
        widths = []
        def deposit(pos, values, **kwargs):
            widths.append(1 if values.ndim == 1 else values.shape[1])
            return original(pos, values, **kwargs)
        with patch.object(galaxy, 'bin_particles_direct', side_effect=deposit):
            result = galaxy.galaxy_datacube(self.snap, frequency_chunk_size=2, **self.kw)
        self.assertEqual(max(widths), 2)
        expected = galaxy.galaxy_datacube(self.snap, frequency_chunk_size=None, **self.kw)
        for name in expected['datacubes']:
            np.testing.assert_allclose(result['datacubes'][name], expected['datacubes'][name], rtol=1e-13, atol=0)

    def test_mapped_particle_contributions(self):
        grid = galaxy.CartesianGrid([np.arange(4.)]*3)
        mapping = np.zeros(grid.n, dtype=int); mapping[[5, 10]] = 1; mapping[40] = 2
        emission = np.array([[0., 0.], [1., 2.], [3., 4.]])
        opacity = np.array([0., .1, .2])
        rays = np.array([[.83, 1.17, .13], [2.21, 1.72, .21]])
        for approx in (True, False):
            kwargs = dict(tree=grid, N_NGB=8, approx=approx, ret_tau=True, ret_particle_contributions=True)
            dense = galaxy.rad.raytrace_particles(rays, rays+[.1, -.1, 2.5], emission[mapping], opacity[mapping], **kwargs)
            compact = galaxy.rad.raytrace_particles(rays, rays+[.1, -.1, 2.5], emission, opacity, field_indices=mapping, **kwargs)
            for a, b in zip(dense[:2], compact[:2]): np.testing.assert_allclose(a, b, rtol=1e-13, atol=0)
            summed = np.zeros_like(compact[2]); np.add.at(summed, mapping, dense[2])
            np.testing.assert_allclose(summed, compact[2], rtol=1e-13, atol=0)

    def test_preparation_uses_cached_snapshot_fields_and_frame(self):
        kw = {key: value for key, value in self.kw.items() if key not in ('npix', 'N_NGB', 'workers')}
        for part in (0, 4):
            self.snap.pos[part]; self.snap.mass[part]; self.snap.metals[part]
        self.snap[0, 'Dust_Temperature']; self.snap[4, 'StellarFormationTime']
        self.snap.mass[4][:] *= 2
        self.snap.pos[0][:] += [1, .07, .11]*u.kpc
        self.snap.pos[4][:] += [1, .07, .11]*u.kpc
        self.snap.transform(center=[1, 0, 0]*u.kpc, z=90*u.deg)
        positions = self.snap.pos[4].copy(); masses = self.snap.mass[4].copy()
        self.snap.filepath = Path('/nonexistent/snapshot.hdf5')
        with patch.object(h5py, 'File', side_effect=AssertionError('Bypassed cached snapshot accessors')):
            dense = galaxy.prepare_galaxy_datacube(self.snap, dust_to_metal=0, **kw)
            compact = galaxy.prepare_galaxy_datacube(self.snap, dust_to_metal=0, storage='compact', **kw)
        for key in ('emission', 'dust_density'):
            np.testing.assert_allclose(compact[key][compact['field_indices']], dense[key], rtol=1e-13, atol=0)
        expected = np.histogramdd(positions.to_value(u.kpc), bins=(4, 4, 4), range=[(-.5, .5)]*3,
                                  weights=[2., 2.])[0].ravel()*4*u.Lsun.to(u.erg/u.s)/(4*np.pi*(.25*u.kpc).to_value(u.cm)**3)
        np.testing.assert_allclose(dense['emission'], np.broadcast_to(expected[:, None], dense['emission'].shape), rtol=1e-13, atol=0)
        np.testing.assert_array_equal(self.snap.pos[4], positions)
        np.testing.assert_array_equal(self.snap.mass[4], masses)

    def test_cosmological_formation_times_through_snapshot(self):
        with h5py.File(self.snap.filepath, 'r+') as file:
            file['Header'].attrs.update(ComovingIntegrationOn=1, HubbleParam=.7, Redshift=1.,
                                       Omega_Matter=.3, Omega_Lambda=.7)
            file['PartType4/StellarFormationTime'][:] = [.49, 0.]
        cosmology = LambdaCDM(H0=70, Om0=.3, Ode0=.7, Tcmb0=0)
        birth = self.snap[4, 'StellarFormationTime'].to_value(u.yr)
        np.testing.assert_allclose(birth[0], cosmology.age(1/.49-1).to_value(u.yr))
        self.assertEqual(birth[1], 0)
        converted = self.snap.scale_factor_to_time([.5, 0., -1., np.nan, -np.inf]).to_value(u.Gyr)
        np.testing.assert_allclose(converted[:3], [cosmology.age(1).value, 0., -1.])
        self.assertTrue(np.all(np.isnan(converted[3:])))
        np.testing.assert_allclose(self.snap.scale_factor_to_time(.5), cosmology.age(1))
        kw = {key: value for key, value in self.kw.items() if key not in ('npix', 'N_NGB', 'workers')}
        for storage in ('dense', 'compact'):
            prepared = galaxy.prepare_galaxy_datacube(self.snap, storage=storage, clip_population=False, **kw)
            self.assertEqual(prepared['particle_counts']['stars'], 1)
            self.assertEqual(prepared['particle_counts']['wind_excluded'], 1)
            self.assertEqual(prepared['particle_counts']['population_clipped'], 0)

    def test_inner_radius_selection_and_compact_occupancy(self):
        kw = {key: value for key, value in self.kw.items() if key not in ('npix', 'N_NGB', 'workers')}
        center = [.2, 0, 0]*u.kpc
        baseline = galaxy.prepare_galaxy_datacube(self.snap, center=center, **kw)
        retained = np.ravel_multi_index((0, 2, 2), (4, 4, 4))
        for storage in ('dense', 'compact'):
            for radius, chunk in ((.4*u.kpc, 1), (400*u.pc, 10), (.4, 1)):
                prepared = galaxy.prepare_galaxy_datacube(self.snap, center=center, inner_radius=radius,
                                                          storage=storage, chunk_size=chunk, **kw)
                self.assertEqual(prepared['particle_counts']['gas'], 1)
                self.assertEqual(prepared['particle_counts']['stars'], 1)
                if storage == 'compact': self.assertEqual(prepared['emission'].shape[0], 2)
                for key in ('emission', 'dust_density'):
                    actual = prepared[key] if storage == 'dense' else prepared[key][prepared['field_indices']]
                    expected = np.zeros_like(baseline[key]); expected[retained] = baseline[key][retained]
                    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0)
        result = galaxy.galaxy_datacube(self.snap, center=center, inner_radius=400*u.pc, **self.kw)
        self.assertEqual(result['particle_counts']['stars'], 1)
        with self.assertRaisesRegex(ValueError, 'No stars selected'):
            galaxy.prepare_galaxy_datacube(self.snap, inner_radius=1*u.kpc, **kw)

    def test_inner_radius_zero_and_validation(self):
        kw = {key: value for key, value in self.kw.items() if key not in ('npix', 'N_NGB', 'workers')}
        for storage in ('dense', 'compact'):
            default = galaxy.prepare_galaxy_datacube(self.snap, storage=storage, **kw)
            zero = galaxy.prepare_galaxy_datacube(self.snap, storage=storage, inner_radius=0*u.pc, **kw)
            for key in ('emission', 'dust_density'): np.testing.assert_array_equal(default[key], zero[key])
        for radius in (-1*u.pc, np.nan*u.kpc, np.inf*u.kpc, [0, 1]*u.pc):
            with self.assertRaisesRegex(ValueError, 'inner_radius'):
                galaxy.prepare_galaxy_datacube(self.snap, inner_radius=radius, **kw)

    def test_initial_mass_conversion_and_zero_dust(self):
        current = galaxy.galaxy_datacube(self.snap, dust_to_metal=0, **self.kw)
        initial = galaxy.galaxy_datacube(self.snap, dust_to_metal=0, stellar_mass_is_initial=True, **self.kw)
        for name in current['datacubes']:
            np.testing.assert_allclose(current['datacubes'][name], 2*initial['datacubes'][name])
            np.testing.assert_allclose(current['optical_depth'][name], 0)


if __name__ == '__main__':
    unittest.main()
