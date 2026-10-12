import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from jabapy.apps import galaxy
from jabapy.utils import units as u


class TestGalaxyImage(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, 'all')
        self.lam = np.linspace(.28, 1.25, 128)*u.micron
        self.unit = u.erg/u.s/u.cm**2/u.sr/u.micron
        spatial = np.array([[0., .001, .01], [.02, .05, 1.]])
        cube = spatial[..., None]*(1+self.lam.value)*self.unit
        self.datacube = dict(wavelength=self.lam, extent=np.array([-5., 5., -5., 5.])*u.kpc,
                              datacubes={'effective': cube, 'absorption': 2*cube, 'pure_emission': 3*cube})

    def test_notebook_stretch_and_existing_axes(self):
        fig, ax = plt.subplots()
        before = self.datacube['datacubes']['effective'].copy()
        with patch.object(galaxy, 'galaxy_datacube', side_effect=AssertionError('Unexpected raytrace')):
            result = galaxy.galaxy_image(datacube=self.datacube, ax=ax)
        channels = result['channels'].to_value(self.unit)
        brightness = np.mean(channels, axis=-1)
        scale = np.percentile(brightness[brightness > 0], 99.5)
        stretched = np.arcsinh(brightness/(.03*scale))/np.arcsinh(1/.03)
        expected = np.zeros_like(channels)
        np.divide(channels*stretched[..., None], brightness[..., None], out=expected, where=brightness[..., None] > 0)
        np.testing.assert_allclose(result['rgb'], np.clip(expected, 0, 1))
        np.testing.assert_array_equal(self.datacube['datacubes']['effective'], before)
        np.testing.assert_array_equal(result['image'].get_array(), result['rgb'])
        np.testing.assert_array_equal(result['image'].get_extent(), [-5., 5., -5., 5.])
        self.assertIs(result['figure'], fig); self.assertIs(result['ax'], ax)
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(result['image'].origin, 'lower')
        self.assertNotIn('datacubes', result)

    def test_filters_components_and_wavelength_units(self):
        first = galaxy.galaxy_image(datacube=self.datacube)
        other = dict(self.datacube, wavelength=self.lam.to(u.AA),
                     datacubes={key: cube.to(self.unit*u.micron/u.AA) for key, cube in self.datacube['datacubes'].items()})
        second = galaxy.galaxy_image(datacube=other, component='absorption', bands=('RGB.B', 'RGB.G', 'RGB.R'))
        np.testing.assert_allclose(second['channels'], 2*first['channels'][..., ::-1])
        np.testing.assert_allclose(second['rgb'], first['rgb'][..., ::-1])

    def test_empty_image_and_invalid_inputs(self):
        self.datacube['datacubes']['effective'] *= 0
        result = galaxy.galaxy_image(datacube=self.datacube)
        np.testing.assert_array_equal(result['rgb'], np.zeros((2, 3, 3)))
        self.assertEqual(result['scale'].value, 0.)
        with patch.object(galaxy, 'galaxy_datacube') as trace:
            for kwargs in ({'stretch': 0}, {'percentile': 101}, {'percentile': np.nan}, {'bands': ['RGB.R']},
                           {'bands': ('missing',)*3}, {'component': 'missing'}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): galaxy.galaxy_image(object(), **kwargs)
            trace.assert_not_called()
        with self.assertRaises(ValueError): galaxy.galaxy_image()
        with self.assertRaises(ValueError): galaxy.galaxy_image(object(), datacube=self.datacube)
        with self.assertRaises(ValueError): galaxy.galaxy_image(datacube=self.datacube, npix=8)

    def test_one_call_defaults_and_explicit_spectral_grid(self):
        snap = object()
        with patch.object(galaxy, 'galaxy_datacube', return_value=self.datacube) as trace:
            galaxy.galaxy_image(snap, npix=512, grid_shape=128, inner_radius=.1, log=True)
            self.assertIs(trace.call_args.args[0], snap)
            options = trace.call_args.kwargs
            self.assertEqual(options['npix'], 512); self.assertEqual(options['grid_shape'], 128)
            self.assertEqual(options['inner_radius'], .1); self.assertTrue(options['log'])
            self.assertEqual(options['n_wavelength'], 128); self.assertEqual(options['wavelength_spacing'], 'linear')
            self.assertFalse(options['return_optical_depth'])
            np.testing.assert_allclose(options['wavelength_range'].to_value(u.micron), [.28, 1.25])
            galaxy.galaxy_image(snap, npix=256)
            self.assertEqual(trace.call_args.kwargs['grid_shape'], 256)
            galaxy.galaxy_image(snap, lam=self.lam, return_optical_depth=True, cache=False)
            self.assertIs(trace.call_args.kwargs['lam'], self.lam)
            self.assertNotIn('grid_shape', trace.call_args.kwargs)
            self.assertNotIn('wavelength_range', trace.call_args.kwargs)
            self.assertFalse(trace.call_args.kwargs['cache'])
            galaxy.galaxy_image(snap, wavelength_range=[.1, 2.], n_wavelength=200, wavelength_spacing='log')
            self.assertEqual(trace.call_args.kwargs['wavelength_range'], [.1, 2.])
            self.assertEqual(trace.call_args.kwargs['n_wavelength'], 200)
            self.assertEqual(trace.call_args.kwargs['wavelength_spacing'], 'log')
