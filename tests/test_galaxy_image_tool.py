import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from matplotlib.image import imread
import jabapy
from jabapy.apps import galaxy

spec = importlib.util.spec_from_file_location('galaxy_image_tool', Path(__file__).resolve().parents[1]/'tools'/'galaxy_image.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class TestGalaxyImageTool(unittest.TestCase):
    def test_output_pixels_array_and_transform_order(self):
        rgb = np.array([[[1., 0., 0.], [.1, .2, .3]], [[0., 1., 0.], [0., 0., 1.]]])
        snap = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.object(jabapy, 'load', return_value=snap), patch.object(galaxy, 'galaxy_image', return_value={'rgb': rgb}) as render:
            png, raw = tool.main(['snapshot.hdf5', '--npix', '2', '--output', str(Path(directory)/'image'), '--quiet'])
            np.testing.assert_array_equal(np.load(raw), rgb)
            pixels = imread(png)
            self.assertEqual(pixels.shape[:2], (2, 2))
            np.testing.assert_allclose(pixels[..., :3], rgb[::-1], atol=1/255)
            self.assertEqual([call[0] for call in snap.mock_calls], ['reset_transform', 'center_on'])
            snap.center_on.assert_called_once_with(3)
            self.assertFalse(render.call_args.kwargs['plot'])
            with self.assertRaises(SystemExit): tool.main(['snapshot.hdf5', '--output', str(Path(directory)/'image'), '--quiet'])
            np.testing.assert_array_equal(np.load(raw), rgb)

    def test_grid_and_frequency_defaults_and_overrides(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(jabapy, 'load'), patch.object(galaxy, 'galaxy_image', return_value={'rgb': np.zeros((2, 2, 3))}) as render:
            for pixels, grid, frequency in ((128, 128, 24), (512, 512, 3), (1920, 512, 3)):
                tool.main(['snapshot.hdf5', '--npix', str(pixels), '--output', str(Path(directory)/str(pixels)), '--quiet'])
                self.assertEqual(render.call_args.kwargs['grid_shape'], grid)
                self.assertEqual(render.call_args.kwargs['frequency_chunk_size'], frequency)
                self.assertEqual(render.call_args.kwargs['npix'], pixels)
            tool.main(['snapshot.hdf5', '--grid-shape', '64', '--frequency-chunk-size', '8', '--stellar-deposition', 'kernel',
                       '--wavelengths-nm', '380', '500', '780', '--output', str(Path(directory)/'custom'), '--quiet'])
            self.assertEqual(render.call_args.kwargs['grid_shape'], 64)
            self.assertEqual(render.call_args.kwargs['frequency_chunk_size'], 8)
            self.assertEqual(render.call_args.kwargs['stellar_deposition'], 'kernel')
            np.testing.assert_array_equal(render.call_args.kwargs['lam'].value, [380, 500, 780])


if __name__ == '__main__':
    unittest.main()
