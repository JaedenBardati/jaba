import unittest
from unittest.mock import patch

import numpy as np
from astropy.cosmology import LambdaCDM

from jabapy.physics import cosmology as cosmo
from jabapy.snapshot import GIZMO_Snapshot
from jabapy.utils import units as u


class TestScaleFactorTime(unittest.TestCase):
    def test_scalar_array_and_formation_flags(self):
        cosmology = LambdaCDM(H0=70, Om0=.3, Ode0=.7, Tcmb0=0)
        np.testing.assert_allclose(cosmo.scale_factor_to_time(.5, h=.7, Om0=.3, Ode0=.7), cosmology.age(1))
        values = np.array([[.25, .5], [0., -1.], [np.nan, np.inf], [-np.inf, 1.]])
        result = cosmo.scale_factor_to_time(values, h=.7, Om0=.3, Ode0=.7).to_value(u.Gyr)
        self.assertEqual(result.shape, values.shape)
        np.testing.assert_allclose(result[0], cosmology.age([3., 1.]).value)
        np.testing.assert_array_equal(result[1], [0., -1.])
        self.assertTrue(np.all(np.isnan(result[2])))
        self.assertTrue(np.isnan(result[3, 0]))
        np.testing.assert_allclose(result[3, 1], cosmology.age(0).value)
        self.assertEqual(cosmo.scale_factor_to_time(np.empty((0, 2))).shape, (0, 2))
        np.testing.assert_array_equal(cosmo.scale_factor_to_time([0., -1.]).value, [0., -1.])

    def test_radiation_and_explicit_curvature(self):
        for params in (dict(h=.7, Om0=.3, Ode0=.6, Or0=.01, Ok0=.09),
                       dict(h=.7, Om0=.3, Ode0=.6, Or0=0., Ok0=0.)):
            result = cosmo.scale_factor_to_time([.5, .5, 1.], **params)
            expected = [cosmo.time_since_bigbang(z, **params).to_value(u.Gyr) for z in (1., 1., 0.)]
            np.testing.assert_allclose(result.to_value(u.Gyr), expected, rtol=1e-13)

    def test_existing_snapshot_with_legacy_method(self):
        class ExistingSnapshot(GIZMO_Snapshot):
            def scale_factor_to_time(self, scale_factor):
                return cosmo.scale_factor_to_time(scale_factor, h=self.HubbleParam, Ode0=self.OmegaLambda,
                                                  Om0=self.OmegaMatter, Or0=self.OmegaRadiation, Ok0=0.)
        snap = ExistingSnapshot('/unused/snapshot.hdf5')
        snap._attributes = dict(HubbleParam=.7, Omega_Matter=.3, Omega_Lambda=.7, OmegaRadiation=0.)
        with patch.object(cosmo, 'scale_factor_to_time'):
            del cosmo.scale_factor_to_time
            with self.assertRaisesRegex(AttributeError, 'scale_factor_to_time'):
                snap.scale_factor_to_time(.5)
        np.testing.assert_allclose(snap.scale_factor_to_time(.5), LambdaCDM(H0=70, Om0=.3, Ode0=.7, Tcmb0=0).age(1))


if __name__ == '__main__':
    unittest.main()
