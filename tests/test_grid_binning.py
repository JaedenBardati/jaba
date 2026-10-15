import unittest

import numba
import numpy as np

from jabapy.utils.grid import bin_particles_direct


class TestDirectBinning(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(52)
        self.dims = (3, 4, 5)
        self.pos = rng.uniform(0, 1, (120, 3))
        self.qty = rng.uniform(0, 1, (120, 2))
        self.kw = dict(mins=np.zeros(3), maxs=np.ones(3), dims=self.dims)
        indices = np.ravel_multi_index(np.floor(self.pos*np.array(self.dims)).astype(int).T, self.dims)
        self.expected = np.stack([np.bincount(indices, weights=q, minlength=60).reshape(self.dims) for q in self.qty.T])

    def test_default_and_strided_chunk_accumulation(self):
        expected = self.expected
        np.testing.assert_allclose(bin_particles_direct(self.pos, self.qty, **self.kw), expected)
        storage = np.full((60, 3), .5)
        out = storage[:, :2].T.reshape((2,) + self.dims)
        for start in range(0, len(self.pos), 17):
            returned = bin_particles_direct(self.pos[start:start+17], self.qty[start:start+17], out=out, **self.kw)
            self.assertIs(returned, out)
        np.testing.assert_allclose(out, .5+expected, rtol=1e-14)
        np.testing.assert_array_equal(storage[:, 2], .5)
        bin_particles_direct(np.empty((0, 3)), np.empty((0, 2)), out=out, **self.kw)
        np.testing.assert_allclose(out, .5+expected, rtol=1e-14)

    def test_scalar_output_and_bin_edges(self):
        storage = np.zeros((60, 2)); out = storage[:, 0].reshape(self.dims)
        bins, result = bin_particles_direct(self.pos, self.qty[:, 0], out=out, ret_bins=True, **self.kw)
        self.assertIs(result, out)
        np.testing.assert_allclose(out, self.expected[0])
        for edges, dim in zip(bins, self.dims): np.testing.assert_allclose(edges, np.linspace(0, 1, dim+1))

    def test_parallel_accumulation(self):
        if numba.config.NUMBA_NUM_THREADS < 2: self.skipTest('Two Numba threads required.')
        out = np.ones((2,) + self.dims)
        bin_particles_direct(self.pos, self.qty, out=out, nthreads=2, **self.kw)
        np.testing.assert_allclose(out, 1+self.expected, rtol=1e-14)

    def test_invalid_output(self):
        for out in (np.zeros((2, 60)), np.zeros((2,) + self.dims, dtype=np.float32)):
            with self.assertRaises(AssertionError): bin_particles_direct(self.pos, self.qty, out=out, **self.kw)
        out = np.zeros((2,) + self.dims); out.flags.writeable = False
        with self.assertRaises(AssertionError): bin_particles_direct(self.pos, self.qty, out=out, **self.kw)
        out = np.zeros((2, 4, 3, 5)).transpose(0, 2, 1, 3)
        with self.assertRaises(AttributeError): bin_particles_direct(self.pos, self.qty, out=out, **self.kw)


if __name__ == '__main__':
    unittest.main()
