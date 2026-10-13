import unittest
import numpy as np
from jabapy.utils.grid import bin_particles_kernel


class TestKernelDeposition(unittest.TestCase):
    def test_conservation_symmetry_boundary_and_unresolved(self):

        # preserve each quantity even at box boundaries and below the cell resolution
        for pos, h in (([0, 0, 0], .8), ([-.99, .2, .1], .8), ([.12, .23, .34], 1e-8), ([0, 0, 0], 0)):
            with self.subTest(pos=pos, h=h):
                grid = bin_particles_kernel([pos], [[2., 5.]], h=[h], mins=[-1]*3, maxs=[1]*3, dims=[8]*3)
                np.testing.assert_allclose(grid.sum(axis=(1, 2, 3)), [2, 5], rtol=1e-14)
                self.assertTrue(np.all(np.isfinite(grid)))
                self.assertTrue(np.all(grid >= 0))
                if h == .8 and pos == [0, 0, 0]:
                    self.assertGreater(np.count_nonzero(grid[0]), 8)
                    np.testing.assert_allclose(grid, grid[:, ::-1, :, :])
                    np.testing.assert_allclose(grid, grid[:, :, ::-1, :])

    def test_compact_occupancy_and_strided_accumulation(self):

        # occupancy must include the full support and agree with dense multichannel deposition
        pos = np.array([[0, 0, 0], [.6, -.4, .1]])
        kwargs = dict(h=[.7, .3], mins=[-1]*3, maxs=[1]*3, dims=[8]*3)
        occupied = np.zeros(512, dtype=bool)
        bin_particles_kernel(pos, occupancy=occupied, **kwargs)
        mapping = np.zeros(512, dtype=np.int32)
        mapping[occupied] = np.arange(1, occupied.sum()+1)
        values = np.array([[2., 5.], [3., 7.]])
        dense = bin_particles_kernel(pos, values, **kwargs)
        compact = np.zeros((occupied.sum()+1, 2)).T
        bin_particles_kernel(pos, values, out=compact, field_indices=mapping, **kwargs)
        np.testing.assert_allclose(compact[:, mapping], dense.reshape(2, -1))
        np.testing.assert_array_equal(compact[:, 0], 0)
        bin_particles_kernel(pos, values, out=compact, field_indices=mapping, **kwargs)
        np.testing.assert_allclose(compact[:, mapping], 2*dense.reshape(2, -1))
        self.assertTrue(np.all(occupied[dense[0].reshape(-1)>0]))

    def test_invalid_radii_and_outside_particles(self):
        kwargs = dict(mins=[-1]*3, maxs=[1]*3, dims=[4]*3)
        for h in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError): bin_particles_kernel([[0, 0, 0]], [1], h=[h], **kwargs)
        grid = bin_particles_kernel([[2, 0, 0]], [1], h=[2], **kwargs)
        self.assertEqual(grid.sum(), 0)
