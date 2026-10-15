import gzip
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import astropy.units as u
from scipy.integrate import trapezoid

from jabapy.physics import radiation
from jabapy.utils import resources


class TestStellarPopulationSpectrum(unittest.TestCase):

    def setUp(self):

        # write small binary tables with known spectra in both byte orders
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.ages = np.array([0., 1e6, 1e9])
        self.wavelengths = np.array([100., 1000., 10000.])
        self.spectra = np.arange(1., 19.).reshape(2, 3, 3)
        for i, (metallicity, endian) in enumerate([(0.01, '<'), (0.03, '>')]):
            def record(payload):
                marker = np.array(len(payload), dtype=endian+'i4').tobytes()
                return marker + payload + marker
            def vector(values):
                return np.array(len(values), dtype=endian+'i4').tobytes() + np.asarray(values, dtype=endian+'f4').tobytes()
            data = record(vector(self.ages) + f' Z={metallicity} '.encode()) + record(vector(self.wavelengths))
            data += b''.join(record(vector(row) + b'extra') for row in self.spectra[i])
            with gzip.open(self.directory / f'model_{i}.ised.gz', 'wb') as stream:
                stream.write(data)

        # direct resource loading to the fixture and isolate the in-memory cache
        loader = patch.object(resources, 'get_resource_path', return_value=self.directory)
        loader.start()
        self.addCleanup(loader.stop)
        resources.clear_resource_cache()
        self.addCleanup(resources.clear_resource_cache)

    def test_interpolation_and_initial_mass(self):
        result = radiation.stellar_population_spectrum_lam(self.wavelengths*u.AA, age=0.5*u.Myr,
                                                           Z=0.02, initial_mass=2*u.Msun)
        expected = self.spectra[:, :2].mean(axis=(0, 1)) * 2*u.Lsun.to(u.erg/u.s)
        np.testing.assert_allclose(result.to_value(u.erg/u.s/u.AA), expected)

    def test_luminosity_and_sampling(self):
        lam = self.wavelengths*u.AA
        luminosity = 1e42*u.erg/u.s
        full = radiation.stellar_population_spectrum_lam(lam, L_star=luminosity, cache=True)
        np.testing.assert_allclose(trapezoid(full.value, self.wavelengths), luminosity.value)
        scalar = radiation.stellar_population_spectrum_lam(lam[1].to(u.micron), L_star=luminosity)
        np.testing.assert_allclose(scalar.to_value(full.unit), full.value[1])
        grid = radiation.stellar_population_spectrum_lam(lam[[2, 0, 1, 1]].reshape(2, 2), L_star=luminosity)
        np.testing.assert_allclose(grid.value, full.value[[2, 0, 1, 1]].reshape(2, 2))
        self.assertEqual(resources._load_resource_cached.cache_info().misses, 1)

    def test_grid_nodes_and_zero_mass(self):
        for i, metallicity in enumerate([0.01, 0.03]):
            for j, age in enumerate(self.ages):
                result = radiation.stellar_population_spectrum_lam(self.wavelengths*u.AA, age=age*u.yr, Z=metallicity)
                np.testing.assert_allclose(result.to_value(u.Lsun/u.AA), self.spectra[i, j])
        result = radiation.stellar_population_spectrum_lam(self.wavelengths*u.AA, initial_mass=0*u.Msun)
        self.assertTrue(np.all(result.value == 0))

    def test_population_broadcasting(self):
        lam = np.array([[100., 550.], [10000., 1000.]])*u.AA
        ages = np.array([[0.], [5e5], [1e9]])*u.yr
        metallicity = np.array([0.01, 0.02])
        masses = np.array([2., 3.])*u.Msun
        result = radiation.stellar_population_spectrum_lam(lam, age=ages, Z=metallicity, initial_mass=masses)
        self.assertEqual(result.shape, (3, 2, 2, 2))
        for i in range(3):
            for j in range(2):
                expected = radiation.stellar_population_spectrum_lam(lam, age=ages[i, 0], Z=metallicity[j], initial_mass=masses[j])
                np.testing.assert_allclose(result[i, j].value, expected.value)

    def test_array_luminosity_normalization(self):
        lam = self.wavelengths*u.AA
        luminosities = np.array([0., 1e40, 2e40])*u.erg/u.s
        result = radiation.stellar_population_spectrum_lam(lam, age=[0., 0.5, 1000.]*u.Myr,
                                                          Z=[0.01, 0.02, 0.03], L_star=luminosities)
        self.assertEqual(result.shape, (3, 3))
        np.testing.assert_allclose(trapezoid(result.value, self.wavelengths, axis=-1), luminosities.value)
        scalar = radiation.stellar_population_spectrum_lam(lam[1], age=[0., 0.5, 1000.]*u.Myr,
                                                          Z=[0.01, 0.02, 0.03], L_star=luminosities)
        self.assertEqual(scalar.shape, (3,))
        np.testing.assert_allclose(scalar.value, result.value[:, 1])

    def test_normalization_only_array(self):
        result = radiation.stellar_population_spectrum_lam(1000*u.AA, initial_mass=[0., 1., 2.]*u.Msun)
        expected = radiation.stellar_population_spectrum_lam(1000*u.AA)
        np.testing.assert_allclose(result.value, np.array([0., 1., 2.])*expected.value)

    def test_invalid_population_arrays(self):
        cases = [{'age': [1., np.nan]*u.Myr}, {'age': [1., 2000.]*u.Myr},
                 {'Z': [0.02, 0.04]}, {'Z': [0.02, -0.01]},
                 {'initial_mass': [1., -1.]*u.Msun}, {'L_star': [1., np.inf]*u.erg/u.s},
                 {'age': [1., 2.]*u.Myr, 'Z': [0.01, 0.02, 0.03]}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                radiation.stellar_population_spectrum_lam(1000*u.AA, **kwargs)

    def test_invalid_parameters(self):
        cases = [{'age': -1*u.yr}, {'age': 2*u.Gyr}, {'Z': 0.001}, {'Z': np.nan},
                 {'initial_mass': -1*u.Msun}, {'L_star': np.inf*u.erg/u.s},
                 {'L_star': 1*u.erg/u.s, 'initial_mass': 1*u.Msun}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                radiation.stellar_population_spectrum_lam(1000*u.AA, **kwargs)

    def test_invalid_wavelengths(self):
        for lam in [0*u.AA, 10*u.AA, 20000*u.AA, np.nan*u.AA]:
            with self.subTest(lam=lam), self.assertRaises(ValueError):
                radiation.stellar_population_spectrum_lam(lam)

    def test_corrupt_record(self):
        with gzip.open(self.directory / 'model_0.ised.gz', 'wb') as stream:
            stream.write(b'broken')
        with self.assertRaises(ValueError):
            radiation.stellar_population_spectrum_lam(1000*u.AA)


class TestResourceLoading(unittest.TestCase):

    def setUp(self):
        resources.clear_resource_cache()
        self.addCleanup(resources.clear_resource_cache)

    def test_cache_bypass_and_clear(self):
        calls = []
        def loader(directory):
            calls.append(directory)
            return {'call': len(calls)}
        registry = {'example': {'table': {'loader': loader}}}
        with patch.object(resources, 'RESOURCE_REGISTRY', registry), patch.object(resources, 'get_resource_path', return_value=Path('/example')):
            first = resources.load_resource('example', 'table')
            self.assertIs(resources.load_resource('example', 'table'), first)
            self.assertEqual(resources.load_resource('example', 'table', cache=False), {'call': 2})
            self.assertIs(resources.load_resource('example', 'table'), first)
            resources.clear_resource_cache()
            self.assertEqual(resources.load_resource('example', 'table'), {'call': 3})
        self.assertEqual(calls, [Path('/example')]*3)

    def test_resource_without_loader(self):
        registry = {'example': {'file': {}}}
        with patch.object(resources, 'RESOURCE_REGISTRY', registry), patch.object(resources, 'get_resource_path', return_value=Path('/example')):
            self.assertEqual(resources.load_resource('example', 'file'), Path('/example'))

    def test_invalid_resource(self):
        with self.assertRaises(ValueError):
            resources.load_resource('missing', 'missing')
        with self.assertRaises(ValueError):
            resources.load_resource('stellar_population_models', 'missing')

    def test_bc03_registry_loaders(self):
        entries = resources.RESOURCE_REGISTRY['stellar_population_models']
        self.assertEqual(len(entries), 12)
        self.assertTrue(all(info['loader'] is resources._load_BC03_spectra for info in entries.values()))


if __name__ == '__main__':
    unittest.main()
