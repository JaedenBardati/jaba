import unittest
import numpy as np
import numba
from jabapy.physics import radiation as rad
from jabapy.apps import galaxy
from jabapy.utils import units as u
from jabapy.utils.grid import CartesianGrid


class TestFastTransfer(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(130)
        self.tree = CartesianGrid((np.linspace(-2, 2, 7), np.linspace(-1.5, 1.5, 6), np.linspace(-3, 3, 8)))
        self.x0 = np.column_stack((self.rng.uniform(-2, 2, 9), self.rng.uniform(-1.5, 1.5, 9), np.full(9, -3.)))
        self.x1 = self.x0.copy(); self.x1[:, 2] = 3.
        self.x1[0] = self.x0[0]
        self.j = self.rng.random((self.tree.n, 5))
        self.alpha = self.rng.random(self.tree.n)
        self.factors = np.column_stack((np.zeros(5), np.linspace(.2, 2, 5), np.linspace(.5, 4, 5)))

    def compare(self, **kwargs):
        args = dict(tree=self.tree, N_NGB=8, ret_tau=True, opacity_factors=self.factors,
                    query_chunk_size=4, step_fraction=.27, **kwargs)
        reference = rad.raytrace_particles(self.x0, self.x1, self.j, self.alpha, backend='numpy', workers=1, **args)
        for workers in (1, min(2, numba.config.NUMBA_NUM_THREADS)):
            old = numba.get_num_threads()
            actual = rad.raytrace_particles(self.x0, self.x1, self.j, self.alpha, backend='numba', workers=workers, **args)
            self.assertEqual(numba.get_num_threads(), old)
            for result, expected in zip(actual, reference):
                self.assertTrue(np.all(np.isfinite(result)))
                np.testing.assert_allclose(result, expected, rtol=3e-13, atol=1e-14)

    def test_dense_forward_reverse_and_frequency_opacity(self):
        for reverse in (False, True):
            self.compare(reverse=reverse, I0=np.linspace(.2, 1, 15).reshape(5, 3))
        self.alpha = self.rng.random((self.tree.n, 5))
        self.compare(reverse=True, I0=.3)
        self.compare(reverse=False, I0=.3)

    def test_sparse_empty_emission_and_absorption(self):
        mapping = np.zeros(self.tree.n, dtype=np.int32)
        mapping[::7] = 1; mapping[3::13] = 2; mapping[5::17] = 3
        self.j = np.array([[0.]*5, [1.]*5, [0.]*5, [2.]*5])
        self.alpha = np.array([0., 0., .6, .2])
        self.compare(field_indices=mapping, reverse=True, I0=.7)
        self.compare(field_indices=mapping, reverse=False, I0=.7)
        mapping.fill(0)
        self.compare(field_indices=mapping, reverse=True, I0=.7)
        self.j[0] = .5
        self.compare(field_indices=mapping, reverse=True, I0=.7)

    def test_scalar_broadcast_and_no_optical_depth(self):
        for ndim in (1, 2, 3):
            tree = CartesianGrid(tuple(np.linspace(-1, 1, 5) for _ in range(ndim)))
            x0 = np.full((2, ndim), -.8); x1 = np.full((2, ndim), .8)
            for alpha in (np.zeros(tree.n), np.full(tree.n, 1e-14), np.full(tree.n, 1e5)):
                args = dict(tree=tree, N_NGB=3, I0=.7, reverse=True, workers=1)
                expected = rad.raytrace_particles(x0, x1, np.ones(tree.n), alpha, backend='numpy', **args)
                actual = rad.raytrace_particles(x0, x1, np.ones(tree.n), alpha, backend='numba', **args)
                self.assertEqual(actual.shape, (2,))
                np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=1e-14)

    def test_backend_validation_and_fallback(self):
        kw = dict(tree=self.tree, N_NGB=8, workers=1)
        for backend in ('invalid',):
            with self.assertRaises(ValueError):
                rad.raytrace_particles(self.x0, self.x1, self.j, self.alpha, backend=backend, **kw)
        with self.assertRaisesRegex(ValueError, 'requires'):
            rad.raytrace_particles(self.x0, self.x1, self.j, self.alpha, backend='numba', approx=False, **kw)
        for workers in (0, -2):
            with self.assertRaisesRegex(ValueError, 'workers'):
                rad.raytrace_particles(self.x0, self.x1, self.j, self.alpha, backend='numba', **dict(kw, workers=workers))
        a = rad.raytrace_particles(self.x0, self.x1, self.j, self.alpha, ret_particle_contributions=True, **kw)
        b = rad.raytrace_particles(self.x0, self.x1, self.j, self.alpha, ret_particle_contributions=True, backend='numpy', **kw)
        for actual, expected in zip(a, b): np.testing.assert_array_equal(actual, expected)

    def test_galaxy_backend_units_and_optional_optical_depth(self):
        n = 6
        prepared = dict(wavelength=np.linspace(.3, 1., 5)*u.micron, grid_shape=(n,)*3, Lbox=2.,
                        lower=np.full(3, -1.), upper=np.ones(3), center=np.zeros(3), cell_width=np.full(3, 2/n),
                        particle_counts={}, emission=self.rng.random((n**3, 5)), dust_density=self.rng.random(n**3),
                        k_abs=np.linspace(.1, 1, 5), k_sca=np.linspace(.2, 2, 5))
        emission = prepared['emission'].copy()
        reference = galaxy.raytrace_galaxy_datacube(prepared, npix=5, backend='numpy', workers=1)
        for chunk in (1, 7, 4096):
            for ret_tau in (True, False):
                result = galaxy.raytrace_galaxy_datacube(prepared, npix=5, workers=min(2, numba.config.NUMBA_NUM_THREADS), query_chunk_size=chunk,
                                                         return_optical_depth=ret_tau)
                self.assertEqual('optical_depth' in result, ret_tau)
                for name in reference['datacubes']:
                    self.assertEqual(result['datacubes'][name].unit, reference['datacubes'][name].unit)
                    np.testing.assert_allclose(result['datacubes'][name], reference['datacubes'][name], rtol=3e-13)
                    if ret_tau: np.testing.assert_allclose(result['optical_depth'][name], reference['optical_depth'][name], rtol=3e-13)
        np.testing.assert_array_equal(prepared['emission'], emission)

    def test_physical_length_scales(self):
        self.tree = CartesianGrid(tuple(axis*1e21 for axis in self.tree.axes))
        self.x0 *= 1e21; self.x1 *= 1e21
        self.j /= 1e21; self.alpha /= 1e21
        self.compare(reverse=True, I0=.7)
        self.compare(reverse=False, I0=.7)
