import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy import units as u, constants as c

from jabapy.physics import radiation as rad
from jabapy.utils import resources


class TestBandpasses(unittest.TestCase):
    def setUp(self):
        self.lam = np.array([3500., 4000., 5000., 6500.])*u.AA
        self.unit = u.erg/u.s/u.cm**2/u.AA
        self.profile = dict(wavelength=[4000., 4500., 5000.]*u.AA, transmission=np.array([0., 1., 0.]), detector='photon')

    def test_constant_spectrum_units_and_normalization(self):
        spectrum = np.full(4, 3.)*self.unit
        expected = 1500*u.erg/u.s/u.cm**2
        actual = rad.integrate_bandpass(self.lam, spectrum, self.profile)
        np.testing.assert_allclose(actual.to_value(expected.unit), expected.value)
        converted = rad.integrate_bandpass(self.lam.to(u.nm), spectrum.to(self.unit*u.AA/u.nm), self.profile)
        np.testing.assert_allclose(converted.to_value(expected.unit), expected.value)
        mean = rad.integrate_bandpass(self.lam, spectrum, self.profile, normalize=True)
        np.testing.assert_allclose(mean.to_value(self.unit), 3.)
        photon = rad.integrate_bandpass(self.lam, spectrum, self.profile, weighting='native')
        expected_photon = (expected*4500*u.AA/(c.h*c.c)*u.photon).to(u.photon/u.s/u.cm**2)
        np.testing.assert_allclose(photon.to_value(expected_photon.unit), expected_photon.value)
        mean = rad.integrate_bandpass(self.lam, spectrum, self.profile, weighting='photon', normalize=True)
        np.testing.assert_allclose(mean.to_value(self.unit), 3.)

    def test_linear_spectrum_and_cube_axes(self):
        spectrum = self.lam.value/4500*self.unit
        actual = rad.integrate_bandpass(self.lam, spectrum, self.profile)
        np.testing.assert_allclose(actual.to_value(u.erg/u.s/u.cm**2), 500., rtol=1e-13)
        cube = np.arange(1, 7).reshape(2, 3, 1)*np.ones(4)*self.unit
        expected = np.arange(1, 7).reshape(2, 3)*500
        for axis in (0, 1, 2):
            image = rad.integrate_bandpass(self.lam, np.moveaxis(cube, -1, axis), self.profile, axis=axis)
            np.testing.assert_allclose(image.to_value(u.erg/u.s/u.cm**2), expected)

    def test_bin_averaged_line_and_partial_coverage(self):
        edges = [4000., 4100., 4300.]*u.AA
        lam = .5*(edges[:-1]+edges[1:]); spectrum = [2., 3.]*u.erg/u.s/u.AA
        result = rad.integrate_bandpass(lam, spectrum, wavelength_edges=edges)
        np.testing.assert_allclose(result.to_value(u.erg/u.s), 800.)
        broad = dict(wavelength=[3000., 6000.]*u.AA, transmission=[.5, .5])
        with self.assertRaises(ValueError): rad.integrate_bandpass(lam, spectrum, broad, wavelength_edges=edges)
        partial = rad.integrate_bandpass(lam, spectrum, broad, wavelength_edges=edges, allow_partial=True)
        np.testing.assert_allclose(partial.to_value(u.erg/u.s), 400.)
        single = rad.integrate_bandpass([4050.]*u.AA, [2.]*u.erg/u.s/u.AA, wavelength_edges=edges[:2])
        np.testing.assert_allclose(single.to_value(u.erg/u.s), 200.)

    def test_line_edges_coincident_with_converted_filter_knots(self):
        edges = np.linspace(6530., 6590., 101)*u.AA
        centers = .5*(edges[:-1]+edges[1:])
        binned = rad.integrate_bandpass(centers, np.ones(100)*self.unit, 'RGB.R', wavelength_edges=edges, allow_partial=True)
        sampled = rad.integrate_bandpass(edges, np.ones(101)*self.unit, 'RGB.R', allow_partial=True)
        np.testing.assert_allclose(binned.to_value(sampled.unit), sampled.value, rtol=1e-13)
        bare = rad.integrate_bandpass([1., 2., 3.], [4., 4., 4.])
        np.testing.assert_allclose(bare.to_value(u.AA), 8.)

    def test_rgb_and_standard_filter_names(self):
        self.assertEqual(len(rad.BANDPASS_FILTERS), 13)
        for name in ('RGB.R', 'RGB.G', 'RGB.B'):
            profile = rad.load_bandpass(name)
            self.assertEqual(profile['detector'], 'energy')
            np.testing.assert_allclose(rad.bandpass_response([100., 1000.]*u.nm, profile), 0.)
            self.assertGreater(np.max(profile['transmission']), .99)
        with patch.object(rad, 'load_resource', return_value=self.profile) as load:
            profile = rad.load_bandpass('sdss.R', cache=False)
            load.assert_called_once_with('bandpass_filters', 'SDSS.r', cache=False)
            self.assertEqual(profile['name'], 'SDSS.r')
        with self.assertRaises(ValueError): rad.load_bandpass('missing')

    def test_invalid_inputs_and_no_overlap(self):
        spectrum = np.ones(4)*self.unit
        for lam in (self.lam[::-1], self.lam[:1], [1., np.nan, 3., 4.]*u.AA):
            with self.assertRaises(ValueError): rad.integrate_bandpass(lam, spectrum, self.profile)
        with self.assertRaises(ValueError): rad.integrate_bandpass(self.lam, spectrum, self.profile, weighting='invalid')
        with self.assertRaises(ValueError): rad.integrate_bandpass(self.lam, spectrum[:3], self.profile)
        with self.assertRaises(ValueError): rad.integrate_bandpass(self.lam, spectrum, dict(wavelength=self.lam, transmission=[0., -1., 1., 0.]))
        far = dict(wavelength=[8000., 9000.]*u.AA, transmission=[1., 1.])
        np.testing.assert_allclose(rad.integrate_bandpass(self.lam, spectrum, far, allow_partial=True).value, 0.)
        with self.assertRaises(ValueError): rad.integrate_bandpass(self.lam, spectrum, far, allow_partial=True, normalize=True)


class TestFilterResources(unittest.TestCase):
    def test_registry_download_and_votable_loader(self):
        registry = resources.RESOURCE_REGISTRY['bandpass_filters']
        self.assertEqual(len(registry), 10)
        self.assertTrue(all(info['loader'] is resources._load_bandpass for info in registry.values()))
        xml = '''<?xml version="1.0"?><VOTABLE version="1.1"><RESOURCE><TABLE>
<PARAM name="WavelengthUnit" datatype="char" arraysize="*" value="Angstrom"/>
<PARAM name="DetectorType" datatype="char" arraysize="*" value="1"/>
<PARAM name="Description" datatype="char" arraysize="*" value="test"/>
<PARAM name="filterID" datatype="char" arraysize="*" value="SLOAN/SDSS.r"/>
<FIELD name="Wavelength" datatype="double" unit="Angstrom"/><FIELD name="Transmission" datatype="double"/>
<DATA><TABLEDATA><TR><TD>5000</TD><TD>0</TD></TR><TR><TD>6000</TD><TD>0.5</TD></TR><TR><TD>7000</TD><TD>0</TD></TR></TABLEDATA></DATA>
</TABLE></RESOURCE></VOTABLE>'''
        resources.clear_resource_cache()
        self.addCleanup(resources.clear_resource_cache)
        with tempfile.TemporaryDirectory() as directory, patch.object(resources, 'CACHE_DIR', Path(directory)):
            def download(url, path): Path(path).write_text(xml)
            with patch.object(resources.urllib.request, 'urlretrieve', side_effect=download) as fetch:
                result = resources.load_resource('bandpass_filters', 'SDSS.r')
                self.assertIs(resources.load_resource('bandpass_filters', 'SDSS.r'), result)
                fetch.assert_called_once()
                self.assertTrue((Path(directory)/'bandpass_filters/SDSS.r/profile.xml').is_file())
                np.testing.assert_allclose(result['wavelength'].to_value(u.AA), [5000., 6000., 7000.])
                np.testing.assert_allclose(result['transmission'], [0., .5, 0.])
                self.assertEqual(result['detector'], 'photon')


if __name__ == '__main__':
    unittest.main()
