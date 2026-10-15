import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from matplotlib.image import imread
import jabapy
from jabapy.apps import galaxy
from jabapy.utils import units as u

spec = importlib.util.spec_from_file_location('galaxy_image_tool', Path(__file__).resolve().parents[1]/'tools'/'galaxy_image.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class TestGalaxyImageTool(unittest.TestCase):
    def test_batch_checkpoint_roundtrip_and_no_overwrite(self):
        batch = dict(wavelength=np.array([400., 500.])*u.nm, extent=np.arange(4.)*u.kpc,
                     pixel_area=2.*u.cm**2, grid_shape=(np.int64(4),)*3,
                     particle_counts={'stars': 2}, supersampling=1,
                     datacubes={'effective': np.ones((2, 2, 2))*u.erg},
                     optical_depth={'effective': np.ones((2, 2, 2))}, line_wavelength=656.3*u.nm,
                     line_intensity={'effective': np.full((2, 2), 3.)*u.erg/u.s/u.cm**2/u.sr},
                     line_optical_depth={'effective': np.ones((2, 2))})
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.assertIsNone(tool._batch_checkpoint(directory, 0, 2))
            tool._batch_checkpoint(directory, 0, 2, batch)
            loaded = tool._batch_checkpoint(directory, 0, 2)
            np.testing.assert_array_equal(loaded['datacubes']['effective'], batch['datacubes']['effective'])
            self.assertEqual(loaded['grid_shape'], batch['grid_shape'])
            self.assertEqual(loaded['pixel_area'], batch['pixel_area'])
            np.testing.assert_array_equal(loaded['line_intensity']['effective'], batch['line_intensity']['effective'])
            self.assertEqual(loaded['line_wavelength'], batch['line_wavelength'])
            with self.assertRaises(FileExistsError): tool._batch_checkpoint(directory, 0, 2, batch)

    def test_resume_after_png_failure_and_reject_changed_settings(self):
        rgb = np.full((2, 2, 3), .25)
        with tempfile.TemporaryDirectory() as directory, patch.object(jabapy, 'load'), patch.object(galaxy, 'galaxy_image', return_value={'rgb': rgb}) as render:
            args = ['snapshot.hdf5', '--npix', '2', '--output', str(Path(directory)/'image'), '--quiet']
            with patch('matplotlib.image.imsave', side_effect=RuntimeError('interrupted export')):
                with self.assertRaises(RuntimeError): tool.main(args)
            restart = Path(directory)/'image.restart'
            self.assertTrue((restart/'rgb.npy').exists())
            with self.assertRaises(SystemExit): tool.main(args+['--resume', '--box-size', '30'])
            render.reset_mock()
            png, raw = tool.main(args+['--resume'])
            render.assert_not_called()
            self.assertTrue(png.exists())
            np.testing.assert_array_equal(np.load(raw), rgb)
            self.assertFalse(restart.exists())

    def test_keep_restart(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(jabapy, 'load'), patch.object(galaxy, 'galaxy_image', return_value={'rgb': np.zeros((2, 2, 3))}):
            args = ['snapshot.hdf5', '--npix', '2', '--output', str(Path(directory)/'image'), '--quiet', '--keep-restart']
            tool.main(args)
            restart = Path(directory)/'image.restart'
            self.assertTrue((restart/'rgb.npy').exists())
            manifest = restart/'settings.json'
            saved = json.loads(manifest.read_text())
            original_code = saved['code']
            saved['code'] = 'incompatible'
            manifest.write_text(json.dumps(saved))
            (restart/'notes.txt').write_text('preserve unrelated files')
            with self.assertRaises(SystemExit): tool.main(args[:-1]+['--resume'])
            saved['code'] = original_code
            manifest.write_text(json.dumps(saved))
            tool.main(args[:-1]+['--resume'])
            self.assertEqual((restart/'notes.txt').read_text(), 'preserve unrelated files')
            self.assertFalse((restart/'settings.json').exists())

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
                       '--gas-emission', 'halpha', '--bands', 'RGB.R', 'RGB.B', 'RGB.G',
                       '--wavelengths-nm', '380', '500', '780', '--output', str(Path(directory)/'custom'), '--quiet'])
            self.assertEqual(render.call_args.kwargs['grid_shape'], 64)
            self.assertEqual(render.call_args.kwargs['frequency_chunk_size'], 8)
            self.assertEqual(render.call_args.kwargs['stellar_deposition'], 'kernel')
            self.assertEqual(render.call_args.kwargs['gas_emission'], 'halpha')
            self.assertEqual(render.call_args.kwargs['bands'], ('RGB.R', 'RGB.B', 'RGB.G'))
            np.testing.assert_array_equal(render.call_args.kwargs['lam'].value, [380, 500, 780])


if __name__ == '__main__':
    unittest.main()
